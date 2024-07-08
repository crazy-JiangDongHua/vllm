import contextlib
import time
from enum import IntEnum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union
from itertools import chain, accumulate

import numpy as np
import torch
import torch.nn as nn

from vllm.attention import (AttentionMetadata, AttentionMetadataPerStage,
                            get_attn_backend)
from vllm.config import (DeviceConfig, LoRAConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, TensorizerConfig,
                         VisionLanguageConfig)
from vllm.distributed import (broadcast_tensor_dict, 
                              with_pynccl_for_all_reduce,
                              batch_send_tensor_dict,
                              recv_tensor_dict, 
                              get_atten_tensor_model_parallel_group_ranks_list,
                              get_tensor_model_parallel_rank,
                              get_atten_data_model_parallel_rank,
                              get_atten_tensor_model_parallel_group)
from vllm.distributed.device_communicators import (custom_all_reduce,
                                                   pynccl_utils)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (MultiModalData, SamplerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.utils import (CudaMemoryProfiler, async_tensor_h2d, is_hip,
                        is_pin_memory_available, make_tensor_with_pad,
                        maybe_expand_dim)

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
# 取值 512 是因为之前是 256，然后假定 dp size 是 2
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 512.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 10)
]


class PreparePromptMetadata(NamedTuple):
    input_tokens: List[int]
    input_positions_list: List[List[int]]
    attn_metadata_list: List[Optional[AttentionMetadataPerStage]]
    prompt_lens: List[int]
    subquery_lens: List[int]
    lora_index_mapping: List[int]
    lora_prompt_mapping: List[int]
    lora_requests: Set[LoRARequest]
    multi_modal_input: Optional[torch.Tensor]
    slot_mapping_list: List[List[int]]
    cumsum_batchsize_list: List[int]
    post_sort_indexes: List[int]
    reorder_seq_group_metadata_list: List[SequenceGroupMetadata]    

    @classmethod
    def empty(cls, dp_size=1):
        return PreparePromptMetadata(
            input_tokens=[],
            input_positions_list=[[] for _ in range(dp_size)],
            attn_metadata_list=[None for _ in range(dp_size)],
            prompt_lens=[],
            subquery_lens=[],
            lora_index_mapping=[],
            lora_prompt_mapping=[],
            lora_requests=set(),
            multi_modal_input=None,
            slot_mapping_list=[[] for _ in range(dp_size)],
            cumsum_batchsize_list=[],
            post_sort_indexes=[],
            reorder_seq_group_metadata_list=[]
        )


class PrepareDecodeMetadata(NamedTuple):
    input_tokens: List[int]
    input_positions_list: List[List[int]]
    attn_metadata_list: List[Optional[AttentionMetadata]]
    lora_index_mapping: List[int]
    lora_prompt_mapping: List[int]
    lora_requests: Set[LoRARequest]
    slot_mapping_list: List[List[int]]
    cumsum_batchsize_list: List[int]
    post_sort_indexes: List[int]
    reorder_seq_group_metadata_list: List[SequenceGroupMetadata]  

    @classmethod
    def empty(cls, dp_size=1):
        return PrepareDecodeMetadata(
            input_tokens=[],
            input_positions_list=[[] for _ in range(dp_size)],
            attn_metadata_list=[None for _ in range(dp_size)],
            lora_index_mapping=[],
            lora_prompt_mapping=[],
            lora_requests=set(),
            slot_mapping_list=[[] for _ in range(dp_size)],
            cumsum_batchsize_list=[],
            post_sort_indexes=[],
            reorder_seq_group_metadata_list=[]
        )


# How batches are constructed.
class BatchType(IntEnum):
    # Every batch is prefill.
    PREFILL = 0
    # Every batch is decode.
    DECODE = 1
    # Batch is a mixture of prefill and decode.
    MIXED = 2


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        tensorizer_config: Optional[TensorizerConfig] = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config
        self.tensorizer_config = tensorizer_config
        self.is_driver_worker = is_driver_worker

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device

        self.model = None
        self.block_size = None  # Set after initial profiling.
        self.lora_manager = None

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None else 0)
        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = None  # Set after initial profiling.
        self.pin_memory = is_pin_memory_available()
        self.kv_cache_dtype = kv_cache_dtype
        self.vision_language_config = vision_language_config

        self.attn_backend = get_attn_backend(
            self.model_config.dtype if model_config is not None else None)

    def load_model(self) -> None:
        with CudaMemoryProfiler() as m:
            self.model = get_model(
                self.model_config,
                self.device_config,
                lora_config=self.lora_config,
                vision_language_config=self.vision_language_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                tensorizer_config=self.tensorizer_config,
            )

        self.model_memory_usage = m.consumed_memory
        logger.info(f"Loading model weights took "
                    f"{self.model_memory_usage / float(2**30):.4f} GB")

        if self.lora_config:
            assert hasattr(self.model, "supported_lora_modules"
                           ) and self.model.supported_lora_modules, (
                               "Model does not support LoRA")
            assert hasattr(
                self.model,
                "embedding_modules"), "Model does not have embedding_modules"
            assert hasattr(self.model, "embedding_padding_modules"
                           ), "Model does not have embedding_padding_modules"
            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens, self.vocab_size,
                self.lora_config, self.device, self.model.embedding_modules,
                self.model.embedding_padding_modules)
            self.model = self.lora_manager.create_lora_manager(self.model)

        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently scaled KV cache is only enabled on ROCm
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                else:
                    raise RuntimeError("Using FP8 KV cache and scaling "
                                       "factors provided but model "
                                       f"{self.model.__class__} does not "
                                       "support loading scaling factors.")
            else:
                logger.warn("Using FP8 KV cache but no scaling factors "
                            "provided. Defaulting to scaling factors of 1.0. "
                            "This may lead to less accurate results!")
        elif self.model_config.quantization_param_path is not None:
            logger.warn("KV cache scaling factors provided, "
                        "but the KV cache data type is not FP8. "
                        "KV cache scaling factors will not be used.")

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        dp_size = self.parallel_config.atten_data_parallel_size
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE) // dp_size, self.get_max_block_per_batch()),
            dtype=np.int32)

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_context_len_to_capture + block_size - 1) // block_size

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> PreparePromptMetadata:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        prompt_lens: List[int] = []
        context_lens: List[int] = []
        subquery_lens: List[int] = []
        prefix_block_tables: List[List[int]] = []
        multi_modal_input_list: List[torch.Tensor] = []

        # 包含全部的 input tokens ，但是按照 dp dispatch 重新排序
        # 这里直接用 input_tokens 替代
        sub_input_tokens: List[int] = []
        # 包含 dp dispatch 以后，当前 rank 需要处理的 token 的 position
        # 这里直接用 input_positions 代替
        sub_input_positions: List[int] = []
        # 每个 seq group 之前被分配到的 dp rank
        dp_res_list: List[int] = []
        # 因为 dp 所以每个 rank 分到的数据量不同，所以 all gather 的时候
        # 需要知道每个 rank 要通信的 tensor 的大小, 因为只有 batchsize 维度
        # 不同，为了方便编程，返回了 cumsum 后的 batchsize list
        cumsum_batchsize_list: List[int] = []
        # 经过 dp 以后，input token 保持 dp 顺序，按照 post_sort_indexes 索引可以回到原来的顺序
        # 这个是最后 seq group 的顺序，因为 dp 也是打乱了 seq group 的顺序，不会打乱 seq group
        # 里面的 seq 顺序
        post_sort_indexes:List[int] = []

        # for sample, 将 prepare_prompt 的参数全部根据 dp dispatch 的结果重新排序
        # 注意都是 seqgroup 的顺序，prefill 中一个 seq group 只有一个 seq，所以
        # subquery_len 和 prompt_lens 虽然存的都是 seq 的长度，但也是 seq group 的长度
        reorder_subquery_lens:List[int] = []
        reorder_prompt_lens:List[int] = []
        reorder_seq_group_metadata_list:List[SequenceGroupMetadata] = []

        dp_size = self.parallel_config.atten_data_parallel_size
        if len(seq_group_metadata_list) == 0:
            return PreparePromptMetadata.empty(dp_size=dp_size)

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt

            dp_res_list.append(seq_group_metadata.dp_res)

            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            computed_block_nums = seq_group_metadata.computed_block_nums
            if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None
                             or computed_block_nums == [])):
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching "
                    "now.")

            token_chunk_size = seq_group_metadata.token_chunk_size
            seq_data = seq_group_metadata.seq_data[seq_id]
            computed_len = seq_data.get_num_computed_tokens()
            # We should use get_len here because in case of preemption
            # it contains output tokens.
            prefill_end = min(seq_data.get_len(),
                              computed_len + token_chunk_size)
            prompt_tokens = seq_data.get_token_ids()[computed_len:prefill_end]
            prompt_len = prefill_end
            prompt_lens.append(prompt_len)

            # NOTE: This only works for oooooooxxx style attention.
            if computed_block_nums is not None and len(
                    computed_block_nums) > 0 and self.sliding_window is None:
                # Prefix is not supported with sliding_window
                computed_len = len(computed_block_nums) * self.block_size
                prompt_tokens = prompt_tokens[computed_len:]
                prefix_block_tables.append(computed_block_nums)
            elif self.scheduler_config.chunked_prefill_enabled:
                if seq_group_metadata.block_tables is not None:
                    # Prefill has chunked before.
                    block_table = seq_group_metadata.block_tables[seq_id]
                    prefix_block_tables.append(block_table)
                else:
                    # The first prefill.
                    prefix_block_tables.append([])
            else:
                prefix_block_tables.append([])
                # Right now, prefill start is always 0. However, this
                # assumption can be changed once chunked prefill is introduced.
                assert computed_len == 0

            # actual prompt lens
            context_lens.append(computed_len)
            subquery_lens.append(prompt_len - computed_len)

            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(computed_len, prefill_end)))
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * (prompt_len - computed_len)
            lora_prompt_mapping.extend(
                [lora_id] *
                (prompt_len - computed_len
                 if seq_group_metadata.sampling_params.prompt_logprobs else 1))

            if seq_group_metadata.multi_modal_data:
                multi_modal_input_list.append(
                    seq_group_metadata.multi_modal_data.data)

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                assert computed_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention")
                start_idx = max(0, prompt_len - self.sliding_window)

            tmp_slot_mapping = []
            for i in range(computed_len, prefill_end):
                if i < start_idx:
                    tmp_slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                tmp_slot_mapping.append(slot)
            slot_mapping.append(tmp_slot_mapping)

        use_dp = dp_size > 1
        if use_dp:
            # 假定没有 lora, 没有 multi model input
            assert len(lora_requests)==0 and len(multi_modal_input_list)==0, \
                "currently mix dp and tp in attention, not support lora or multi_modal_input"
            
            # 假定没有 chunked prefill, subquery_lens is same as prompt_lens
            # 而且 prefill 的时候，每个 seqgroup 肯定只有一个 seq
            all_sub_seq_group_indexes = self.dp_dispatch_algo(subquery_lens, dp_res_list)
            
            # 修改 dp_res 的值，这个好像只有主进程有 seq group 这个变量
            if self.is_driver_worker:
                for dp_res, sub_seq_group_indexes in enumerate(all_sub_seq_group_indexes):
                    for i in sub_seq_group_indexes:
                        # 在 profile run 的时候，不会生成 seqgroup，所以不用保存
                        if seq_group_metadata_list[i].seq_group is not None:
                            seq_group_metadata_list[i].seq_group.dp_res = dp_res

            # 从 all_sub_seq_group_indexes 生成变量
            input_positions_list = [[input_positions[i] for i in sub_seq_group_indexes]
                                    for sub_seq_group_indexes in all_sub_seq_group_indexes]
            slot_mapping_list = [[slot_mapping[i] for i in sub_seq_group_indexes]
                                 for sub_seq_group_indexes in all_sub_seq_group_indexes]
            subquery_lens_list = [[subquery_lens[i] for i in sub_seq_group_indexes]
                                  for sub_seq_group_indexes in all_sub_seq_group_indexes]
            prompt_lens_list = [[prompt_lens[i] for i in sub_seq_group_indexes]
                                for sub_seq_group_indexes in all_sub_seq_group_indexes]
            context_lens_list = [[context_lens[i] for i in sub_seq_group_indexes]
                                 for sub_seq_group_indexes in all_sub_seq_group_indexes]
            prefix_block_tables_list = [[prefix_block_tables[i] for i in sub_seq_group_indexes]
                                        for sub_seq_group_indexes in all_sub_seq_group_indexes]
            
            batchsize_list = [sum([subquery_lens[i] for i in sub_seq_group_indexes]) \
                                   for sub_seq_group_indexes in all_sub_seq_group_indexes]
            cumsum_batchsize_list = list(accumulate([0] + batchsize_list))

            # 包含全部的 input tokens ，但是按照 dp dispatch 重新排序
            all_sub_seq_group_indexes = list(chain.from_iterable(all_sub_seq_group_indexes))
            input_tokens = [input_tokens[i] for i in all_sub_seq_group_indexes]

            # for sample use
            reorder_subquery_lens = [subquery_lens[i] for i in all_sub_seq_group_indexes]
            reorder_prompt_lens = [prompt_lens[i] for i in all_sub_seq_group_indexes]
            reorder_seq_group_metadata_list = [seq_group_metadata_list[i] for i in all_sub_seq_group_indexes]

            # 模型运行完以后，还需要按照原来的顺序把结果给拼回来
            post_sort_indexes = [-1] * len(all_sub_seq_group_indexes)
            for u, v in enumerate(all_sub_seq_group_indexes):
                post_sort_indexes[v] = u
            
            # lora 和 multi input 的列表都没有做 dp 的拆分，因为还没有支持
        else:
            # 没有使用 dp 的时候，也还是打包成列表的形式，可以复用之后的处理
            input_positions_list = [input_positions]
            slot_mapping_list = [slot_mapping]
            subquery_lens_list = [subquery_lens]
            prompt_lens_list = [prompt_lens]
            context_lens_list = [context_lens]
            prefix_block_tables_list = [prefix_block_tables]

        # 将 var 从 List[List[int]] 转为 List[int]
        input_tokens = list(chain.from_iterable(input_tokens))
        input_positions_list = [list(chain.from_iterable(input_positions))
                                for input_positions in input_positions_list]
        slot_mapping_list = [list(chain.from_iterable(slot_mapping))
                             for slot_mapping in slot_mapping_list]

        max_subquery_len_list = [max(subquery_lens) if subquery_lens else 0
                                 for subquery_lens in subquery_lens_list]
        max_prompt_len_list = [max(prompt_lens) if prompt_lens else 0
                               for prompt_lens in prompt_lens_list]
        # assert max_subquery_len > 0

        context_lens_tensor_list = [torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=self.device)
                                    for context_lens in context_lens_list]

        # without process, maybe bug
        if multi_modal_input_list:
            assert self.vision_language_config, (
                "Multi-modal inputs are only supported by "
                "vision language models.")
            multi_modal_input = torch.cat(multi_modal_input_list,
                                          dim=0).to(self.device)
        else:
            multi_modal_input = None

        # Prepare prefix block tables
        max_prompt_block_table_len_list = \
            [max(len(t) for t in prefix_block_tables) \
                if prefix_block_tables else 0 \
             for prefix_block_tables in prefix_block_tables_list]
        block_tables_list = [make_tensor_with_pad(
                                prefix_block_tables,
                                max_len=max_prompt_block_table_len,
                                pad=0,
                                dtype=torch.int,
                                device=self.device
                                )
            for max_prompt_block_table_len in max_prompt_block_table_len_list]

        # Query length can be shorter than key (i.e., prompt) when prefill
        # is chunked or prefix cached.
        subquery_lens_tensor_list = [torch.tensor(subquery_lens,
                                            dtype=torch.long,
                                            device=self.device)
                                     for subquery_lens in subquery_lens_list]
        subquery_start_loc_list = [torch.zeros(subquery_lens_tensor.shape[0] + 1,
                                         dtype=torch.int32,
                                         device=self.device)
                                   for subquery_lens_tensor in subquery_lens_tensor_list]

        prompt_lens_tensor_list = [torch.tensor(prompt_lens,
                                          dtype=torch.long,
                                          device=self.device)
                                   for prompt_lens in prompt_lens_list]
        seq_start_loc_list = [torch.zeros(prompt_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)
                              for prompt_lens_tensor in prompt_lens_tensor_list]

        for subquery_lens_tensor, subquery_start_loc in \
            zip(subquery_lens_tensor_list, subquery_start_loc_list):
            torch.cumsum(subquery_lens_tensor,
                        dim=0,
                        dtype=subquery_start_loc.dtype,
                        out=subquery_start_loc[1:])

        for prompt_lens_tensor, seq_start_loc in \
            zip(prompt_lens_tensor_list, seq_start_loc_list):
            torch.cumsum(prompt_lens_tensor,
                        dim=0,
                        dtype=seq_start_loc.dtype,
                        out=seq_start_loc[1:])

        # 这部分的数据都是 dp 以后的结果
        attn_metadata_list = []
        for dp_rank in range(dp_size):
            attn_metadata_list.append(
                self.attn_backend.make_metadata(
                    is_prompt=True,
                    prompt_lens=prompt_lens_list[dp_rank],
                    prompt_lens_tensor=prompt_lens_tensor_list[dp_rank],
                    max_subquery_len=max_subquery_len_list[dp_rank],
                    max_context_len=None,
                    max_prompt_len=max_prompt_len_list[dp_rank],
                    subquery_start_loc=subquery_start_loc_list[dp_rank],
                    seq_start_loc=seq_start_loc_list[dp_rank],
                    context_lens=context_lens_tensor_list[dp_rank],
                    block_tables=block_tables_list[dp_rank],
                    use_cuda_graph=False,
                )
            )
            

        # 这部分只有 prompt_lens 和 subquery_lens 是 dp 之前的结果
        # 为了给 sampler 提供原来的数据
        return PreparePromptMetadata(
            input_tokens=input_tokens,
            input_positions_list=input_positions_list,
            attn_metadata_list=attn_metadata_list,
            prompt_lens=prompt_lens if not use_dp else reorder_prompt_lens,
            subquery_lens=subquery_lens if not use_dp else reorder_subquery_lens,
            lora_index_mapping=lora_index_mapping,
            lora_prompt_mapping=lora_prompt_mapping,
            lora_requests=lora_requests,
            multi_modal_input=multi_modal_input,
            slot_mapping_list=slot_mapping_list,
            cumsum_batchsize_list=cumsum_batchsize_list,
            post_sort_indexes=post_sort_indexes,
            reorder_seq_group_metadata_list=reorder_seq_group_metadata_list
        )

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> PrepareDecodeMetadata:
        dp_size = self.parallel_config.atten_data_parallel_size

        input_tokens: List[int] = []
        input_positions_list: List[List[int]] = [[] for _ in range(dp_size)]
        slot_mapping_list: List[List[int]] = [[] for _ in range(dp_size)]
        context_lens_list: List[List[int]] = [[] for _ in range(dp_size)]
        block_tables_list: List[List[List[int]]] = [[] for _ in range(dp_size)]
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        # 包含全部的 input tokens ，但是按照 dp dispatch 重新排序，
        # 这里直接用 input_tokens 替代了
        sub_input_tokens: List[int] = []
        # 包含 dp dispatch 以后，当前 rank 需要处理的 token 的 position
        # 这里直接就用 input_positions 替代了
        sub_input_positions: List[int] = []
        # 因为 dp 所以每个 rank 分到的数据量不同，所以 all gather 的时候
        # 需要知道每个 rank 要通信的 tensor 的大小, 因为只有 batchsize 维度
        # 不同，为了方便编程，返回了 cumsum 后的 batchsize list
        cumsum_batchsize_list: List[int] = []
        # 经过 dp 以后，input token 保持 dp 顺序，按照 post_sort_indexes 索引可以回到原来的顺序
        post_sort_indexes:List[int] = []

        # for sample, 将 prepare_prompt 的参数全部根据 dp dispatch 的结果重新排序
        # 注意都是 seqgroup 的顺序
        reorder_seq_group_metadata_list:List[SequenceGroupMetadata] = []

        if len(seq_group_metadata_list) == 0:
            return PrepareDecodeMetadata.empty(dp_size=dp_size)

        use_dp = dp_size > 1

        # 因为在 prefill 的时候完成过 dp dispatch，然后 prompt 的 kv 已经存在对应 rank 的 
        # kv cache 上，所以在 decode 阶段就不用再做 dp dispatch 了，直接用之前的结果
        all_sub_seq_indexes:List[List[int]] = [[] for _ in range(dp_size)]
        all_sub_seq_group_indexes:List[List[int]] = [[] for _ in range(dp_size)]

        seq_cnt = 0
        max_context_len = 0
        for seq_group_cnt, seq_group_metadata in enumerate(seq_group_metadata_list):
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            # without process, maybe bugs
            lora_id = seq_group_metadata.lora_int_id
            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            dp_res = seq_group_metadata.dp_res
            # 当 dp size == 1 时，dp_res == -1，语意是正确的
            all_sub_seq_group_indexes[dp_res].append(seq_group_cnt)

            # 在 python3.7 以后，dict 的 keys 始终保持插入顺序
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                all_sub_seq_indexes[dp_res].append(seq_cnt)

                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                context_len = seq_len if self.sliding_window is None else min(
                        seq_len, self.sliding_window)
                # 这是所有 rank 中最大的 max_context_len
                max_context_len = max(max_context_len, context_len)

                # without process, maybe bug
                lora_index_mapping.append(lora_id)
                lora_prompt_mapping.append(lora_id)
                
                position = seq_len - 1
                input_positions_list[dp_res].append(position)
                context_lens_list[dp_res].append(context_len)
                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping_list[dp_res].append(slot)
                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                            self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables_list[dp_res].append(block_table)

                seq_cnt += 1

        # vLLM uses cuda graph only for decoding requests.
        # See `capture_model` API for more details.
        # For decoding requests, batch_size == input_tokens.
        batchsize_list = [len(sub_seq_indexes) for sub_seq_indexes in all_sub_seq_indexes]
        # 当 dp size == 1 时，这就是 batch size
        max_batch_size = max(batchsize_list)

        use_captured_graph = (
            not self.model_config.enforce_eager
            and max_batch_size <= _BATCH_SIZES_TO_CAPTURE[-1] // dp_size
            and max_context_len <= self.max_context_len_to_capture)
            
        print(f"use dp: {use_dp}, use_captured_graph: {use_captured_graph}")

        if use_dp:
            # 假定没有 lora, 没有 multi model input
            assert len(lora_requests) == 0, "currently mix dp and tp in attention, not support lora or multi_modal_input"
            
            # 这里暂时是 List[List[int]] 类型，方便 padding
            input_tokens:List[List[int]] = [[input_tokens[i] for i in sub_seq_indexes]
                            for sub_seq_indexes in all_sub_seq_indexes]
            
            # 一些 dp 专用的变量
            cumsum_batchsize_list = list(accumulate([0] + batchsize_list))    
            all_sub_seq_group_indexes = list(chain.from_iterable(all_sub_seq_group_indexes))
            post_sort_indexes = [-1] * len(all_sub_seq_group_indexes)
            for u, v in enumerate(all_sub_seq_group_indexes):
                post_sort_indexes[v] = u
            reorder_seq_group_metadata_list = [seq_group_metadata_list[i] for i in all_sub_seq_group_indexes]
        else:
            # 为了共用之用的处理
            input_tokens:List[List[int]] = [input_tokens]

        if use_captured_graph:
            graph_batch_size = _get_graph_batch_size(max_batch_size)
            assert graph_batch_size >= max_batch_size
            for dp_rank, sub_seq_indexes in enumerate(all_sub_seq_indexes):
                for _ in range(graph_batch_size - len(sub_seq_indexes)):
                    sub_seq_indexes.append(-1)
                    input_tokens[dp_rank].append(0)
                    input_positions_list[dp_rank].append(0)
                    slot_mapping_list[dp_rank].append(_PAD_SLOT_ID)
                    context_lens_list[dp_rank].append(1)
                    block_tables_list[dp_rank].append([])
                    # bug when use dp and use lora
                    lora_index_mapping.append(0)
                batchsize_list[dp_rank] = graph_batch_size
        
        input_tokens = list(chain.from_iterable(input_tokens))

        # 这是当前 rank 中最大的 max_context_len
        max_context_len_list = [max(context_lens) if context_lens else 0 
                                for context_lens in context_lens_list]
        context_lens_list = [torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)
                             for context_lens in context_lens_list]

        if use_captured_graph:
            # When using cuda-graph all these tensors should be
            # padded.
            l_sum = 0
            for dp_rank in range(dp_size):
                l = context_lens_list[dp_rank].shape[0]
                assert l == len(input_positions_list[dp_rank])
                assert l == len(slot_mapping_list[dp_rank])
                l_sum += l
            # context_lens is per rank, but input_tokens is all rank
            assert l_sum == len(input_tokens)

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            for i, (batch_size, block_tables) in enumerate(zip(batchsize_list, block_tables_list)):
                input_block_tables = self.graph_block_tables[:batch_size]
                for j, block_table in enumerate(block_tables):
                    if block_table:
                        input_block_tables[j, :len(block_table)] = block_table
                block_tables_list[i] = torch.tensor(input_block_tables, device=self.device)
        else:
            max_block_table_len_list = [
                max(len(block_table) for block_table in block_tables) if block_tables else 0
                for block_tables in block_tables_list
            ]
            block_tables_list = [
                make_tensor_with_pad(
                    block_tables,
                    max_len=max_block_table_len,
                    pad=0,
                    dtype=torch.int,
                    device=self.device,
                )
                for block_tables, max_block_table_len in \
                    zip(block_tables_list, max_block_table_len_list)
            ]

        attn_metadata_list = []
        for dp_rank in range(dp_size):
            attn_metadata_list.append(
                self.attn_backend.make_metadata(
                    is_prompt=False,
                    prompt_lens=None,
                    prompt_lens_tensor=None,
                    max_subquery_len=None,
                    max_context_len=max_context_len_list[dp_rank],
                    max_prompt_len=None,
                    subquery_start_loc=None,
                    seq_start_loc=None,
                    context_lens=context_lens_list[dp_rank],
                    block_tables=block_tables_list[dp_rank],
                    use_cuda_graph=use_captured_graph,
                )
            )
        return PrepareDecodeMetadata(
            input_tokens=input_tokens,
            input_positions_list=input_positions_list,
            attn_metadata_list=attn_metadata_list,
            lora_index_mapping=lora_index_mapping,
            lora_prompt_mapping=lora_prompt_mapping,
            lora_requests=lora_requests,
            slot_mapping_list=slot_mapping_list,
            cumsum_batchsize_list=cumsum_batchsize_list,
            post_sort_indexes=post_sort_indexes,
            reorder_seq_group_metadata_list=reorder_seq_group_metadata_list,
        )

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
        subquery_lens: Optional[List[int]],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        generators: List[torch.Generator] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0
        categorized_sampled_token_indices_start_idx = 0

        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                assert subquery_lens is not None
                subquery_len = subquery_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += subquery_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append([
                        categorized_sample_indices_start_idx,
                        categorized_sampled_token_indices_start_idx
                    ])
                categorized_sample_indices_start_idx += 1
                categorized_sampled_token_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + subquery_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              subquery_len - 1)
                selected_token_start_idx += subquery_len

                if sampling_params.seed is not None:
                    seq_group_metadata.state.generator = torch.Generator(
                        device=self.device).manual_seed(sampling_params.seed)
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        zip(
                            range(
                                categorized_sample_indices_start_idx,
                                categorized_sample_indices_start_idx +
                                num_seqs),
                            range(
                                categorized_sampled_token_indices_start_idx,
                                categorized_sampled_token_indices_start_idx +
                                num_seqs)))
                categorized_sample_indices_start_idx += num_seqs
                categorized_sampled_token_indices_start_idx += num_seqs

            if sampling_params.seed is not None:
                generators.append(seq_group_metadata.state.generator)

        selected_token_indices = async_tensor_h2d(selected_token_indices,
                                                  dtype=torch.long,
                                                  target_device=self.device,
                                                  pin_memory=self.pin_memory)

        categorized_sample_indices = {
            t: maybe_expand_dim(
                async_tensor_h2d(seq_ids,
                                 dtype=torch.int,
                                 target_device=self.device,
                                 pin_memory=self.pin_memory), 2, 2)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            generators=generators,
            perform_sampling=self.is_driver_worker
        )
        return sampling_metadata

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Set[int], LoRAMapping, torch.Tensor]:
        dp_size = self.parallel_config.atten_data_parallel_size
        use_dp = dp_size > 1

        if self.is_driver_worker:
            prefill_reqs = []
            decode_reqs = []
            for seq_group_meta in seq_group_metadata_list:
                if seq_group_meta.is_prompt:
                    prefill_reqs.append(seq_group_meta)
                else:
                    decode_reqs.append(seq_group_meta)

            # Prepare input tensors.
            (
                input_tokens, # all dp rank, if use dp, means input tokens all rank but reorder
                input_positions_list, # means input postions per dp rank 
                prefill_attn_metadata_list, # means metadata per dp rank
                prompt_lens, #  all dp rank, if use dp, have been reorder for sample
                subquery_lens, #  all dp rank, if use dp, have been reorder for sample
                lora_index_mapping, # without process, maybe bug
                lora_prompt_mapping, # without process, maybe bug
                lora_requests, # without process, maybe bug
                multi_modal_input, # without process, maybe bug
                slot_mapping_list, # means slot mapping per dp rank
                cumsum_batchsize_list,
                post_sort_indexes,
                reorder_prefill_reqs
            ) = self._prepare_prompt(prefill_reqs)
            (
                decode_input_tokens, # all dp rank, if use dp, means input tokens all rank but reorder
                decode_input_positions_list, # means input postions per dp rank 
                decode_attn_metadata_list, # means metadata per dp rank
                decode_lora_index_mapping, # without process, maybe bug
                decode_lora_prompt_mapping, # without process, maybe bug
                decode_lora_requests, # without process, maybe bug
                decode_slot_mapping_list, # means slot mapping per dp rank
                decode_cumsum_batchsize_list,
                decode_post_sort_indexes,
                reorder_decode_reqs
            ) = self._prepare_decode(decode_reqs)
            
            if use_dp:
                assert not self.scheduler_config.chunked_prefill_enabled, \
                    "current mix dp and tp, not support chunked prefill"
            sampling_metadata = self._prepare_sample(
                seq_group_metadata_list \
                if not use_dp else \
                # 假定没有 chunk prefill
                reorder_prefill_reqs + reorder_decode_reqs,
                prompt_lens,
                subquery_lens)

            if not self.scheduler_config.chunked_prefill_enabled:
                assert (len(prefill_reqs) and len(decode_reqs)) == 0

            num_prefills_list = [len(prefill_attn_metadata.prompt_lens) \
                                 if prefill_attn_metadata is not None else 0
                                 for prefill_attn_metadata in prefill_attn_metadata_list]
            num_prefill_tokens_list = [len(input_positions) for input_positions in input_positions_list]
            num_decode_tokens_list = [len(decode_input_positions) for decode_input_positions in decode_input_positions_list]

            # Coalesce tensors. Note that attn_metadata is currently not
            # coalesced for simplicity.
            input_tokens.extend(decode_input_tokens)
            for input_positions, decode_input_positions in zip(input_positions_list, decode_input_positions_list):
                input_positions.extend(decode_input_positions)
            for slot_mapping, decode_slot_mapping in zip(slot_mapping_list, decode_slot_mapping_list):
                slot_mapping.extend(decode_slot_mapping)
            
            # TODO(jiangdonghua)：prefill 里面没处理 lora 的逻辑，可能有 bug
            lora_index_mapping.extend(decode_lora_index_mapping)
            lora_prompt_mapping.extend(decode_lora_prompt_mapping)
            lora_requests.update(decode_lora_requests)

            input_tokens = torch.tensor(input_tokens,
                                        dtype=torch.long,
                                        device=self.device)
            input_positions_list = [torch.tensor(input_positions,
                                           dtype=torch.long,
                                           device=self.device)
                                    for input_positions in input_positions_list]
            slot_mapping_list = [torch.tensor(slot_mapping,
                                        dtype=torch.long,
                                        device=self.device)
                                 for slot_mapping in slot_mapping_list]   

            if self.lora_config:
                lora_mapping = LoRAMapping(
                    lora_index_mapping,
                    lora_prompt_mapping,
                )
            else:
                lora_mapping = None
            
            # 这里不能用 num_prefill_tokens==0 或者 num_decode_tokens==0 判断
            # 因为有 dp 时，存在某个 rank 没有分到 seq 的情况
            if decode_cumsum_batchsize_list:
                cumsum_batchsize_list = decode_cumsum_batchsize_list
                post_sort_indexes = decode_post_sort_indexes
            
            metadata_dict_list = []
            decode_metadata_dict_list = []
            for dp_rank in range(dp_size):
                # send the metadata.
                # If batch contains both prefill and decode, it sends 2 
                # If it only contains 1 type, it triggers a single send.
                if (prefill_attn_metadata_list[dp_rank] is not None
                        and decode_attn_metadata_list[dp_rank] is not None):
                    batch_type = BatchType.MIXED
                elif prefill_attn_metadata_list[dp_rank] is not None:
                    batch_type = BatchType.PREFILL
                else:
                    batch_type = BatchType.DECODE
                
                metadata_dict = {
                    "input_tokens": input_tokens,
                    "input_positions": input_positions_list[dp_rank],
                    "selected_token_indices":
                    sampling_metadata.selected_token_indices,
                    "lora_requests": lora_requests,
                    "lora_mapping": lora_mapping,
                    "multi_modal_input": multi_modal_input,
                    "num_prefill_tokens": num_prefill_tokens_list[dp_rank],
                    "num_decode_tokens": num_decode_tokens_list[dp_rank],
                    "slot_mapping": slot_mapping_list[dp_rank],
                    "num_prefills": num_prefills_list[dp_rank],
                    "batch_type": batch_type,
                    "cumsum_batchsize_list": cumsum_batchsize_list,
                    "post_sort_indexes": post_sort_indexes
                }
                if prefill_attn_metadata_list[dp_rank] is not None:
                    metadata_dict.update(prefill_attn_metadata_list[dp_rank].asdict_zerocopy())
                else:
                    metadata_dict.update(decode_attn_metadata_list[dp_rank].asdict_zerocopy())
                metadata_dict_list.append(metadata_dict)
                
                if batch_type == BatchType.MIXED:
                    assert decode_attn_metadata_list[dp_rank] is not None
                    decode_metadata_dict = decode_attn_metadata_list[dp_rank].asdict_zerocopy()
                    decode_metadata_dict_list.append(decode_metadata_dict)

            # driver worker p2p send metadata 给每个 atten tp 组里面 rank
            atten_tp_group_ranks_list: List[List[int]] = \
                get_atten_tensor_model_parallel_group_ranks_list()

            batch_send_tensor_dict(
                tensor_dict_list=metadata_dict_list, 
                dst_group_list=atten_tp_group_ranks_list
            )
            # 这里可能存在 bug ，现在不会执行所以没管了
            if batch_type == BatchType.MIXED:
                batch_send_tensor_dict(
                    tensor_dict_list=decode_metadata_dict_list, 
                    dst_group_list=atten_tp_group_ranks_list
                )
            
            # driver worker get self metadata
            driver_rank = 0
            prefill_attn_metadata = prefill_attn_metadata_list[driver_rank]
            decode_attn_metadata = decode_attn_metadata_list[driver_rank]
            input_positions = input_positions_list[driver_rank]
            num_prefill_tokens = num_prefill_tokens_list[driver_rank]
            num_decode_tokens = num_decode_tokens_list[driver_rank]
            slot_mapping = slot_mapping_list[driver_rank]
            num_prefills = num_prefills_list[driver_rank]
        else:
            metadata_dict = recv_tensor_dict(src=0)
            
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            slot_mapping = metadata_dict.pop("slot_mapping")
            num_prefills = metadata_dict.pop("num_prefills")
            selected_token_indices = metadata_dict.pop(
                "selected_token_indices")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            multi_modal_input = metadata_dict.pop("multi_modal_input")
            num_prefill_tokens = metadata_dict.pop("num_prefill_tokens")
            num_decode_tokens = metadata_dict.pop("num_decode_tokens")
            batch_type = metadata_dict.pop("batch_type")
            cumsum_batchsize_list = metadata_dict.pop("cumsum_batchsize_list")
            post_sort_indexes = metadata_dict.pop("post_sort_indexes")

            # Create an attention metadata.
            prefill_attn_metadata = None
            decode_attn_metadata = None
            if batch_type == BatchType.PREFILL or batch_type == BatchType.MIXED:
                prefill_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            else:
                decode_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                seq_data=None,
                prompt_lens=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                generators=None,
                perform_sampling=False,
            )

            # if it is a mixed batch, decode attn_metadata is broadcasted
            # separately.
            if batch_type == BatchType.MIXED:
                decode_metadata_dict = recv_tensor_dict(src=0)
                decode_attn_metadata = self.attn_backend.make_metadata(
                    **decode_metadata_dict)

        attn_metadata = AttentionMetadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=prefill_attn_metadata,
            decode_metadata=decode_attn_metadata,
            kv_cache_dtype=self.kv_cache_dtype,
        )

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, lora_requests, lora_mapping,
                multi_modal_input, cumsum_batchsize_list, post_sort_indexes)

    def dp_dispatch_algo(
        self, 
        seq_group_lens: List[int],
        dp_locs: List[int],
    ) -> Tuple[List[int], List[List[int]]]:
        '''
        seq_group_lens: 
            the length of all seq group, 
            for prefill, it means the prompt length of the only seq in seq group
            for decode, it means the context length sum of seq in seq group
        dp_locs:
            the result after dp dipatch before, -1 means seq which have not been dispathed
            >=0 means which dp rank seq have been dispatched
        output:
            the seq index to be processed by each dp node

        Since we are assume there is no chunk prefill, so dp dispatch will not be performed 
        in the decode seq group, so the seq_group_lens are all prompt length, the element in
        dp_locs are all -1。
        '''
        # TODO(jiangdonghua): optimal this func
        dp_size = self.parallel_config.atten_data_parallel_size
        assert dp_size == 2, \
            f"currently only support atten dp size be equal 2, but get {dp_size}"

        assert all([loc == -1 for loc in dp_locs]), \
            f"expect the element in dp_locs are all -1, but get {dp_locs}"
        
        boundary = 32 * 1024

        # When there are no very long or short seq, divide equally
        # For example, during simulation runtime to estimate available kv cache size
        if all(l < boundary for l in seq_group_lens) or all(l >= boundary for l in seq_group_lens):
            # 想让 driver work 多干一点
            mid = (len(seq_group_lens) + 1) // 2
            return [list(range(0, mid)), list(range(mid, len(seq_group_lens)))]
            
        
        # stupid dispatch
        rank0_res, rank1_res = [], []
        for i, l in enumerate(seq_group_lens):
            if l >= boundary:
                rank1_res.append(i)
            if l < boundary:
                rank0_res.append(i)
        
        return [rank0_res, rank1_res]


    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_input, cumsum_batchsize_list,
         post_sort_indexes
        ) = self.prepare_input_tensors(seq_group_metadata_list)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            logger.info(f"use cuda graph, pad batch is {graph_batch_size}")
            model_executable = self.graph_runners[graph_batch_size]
        else:
            logger.info("not use cuda graph")
            model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
            "cumsum_batchsize_list": cumsum_batchsize_list,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})

        with torch.cuda.profiler.profile():
            import nvtx
            with nvtx.annotate("execute model"):
                hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        with torch.cuda.profiler.profile():
            import nvtx
            with nvtx.annotate("Compute logits."):
                logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not sampling_metadata.perform_sampling:
            return None

        # Sample the next token.
        with torch.cuda.profiler.profile():
            import nvtx
            with nvtx.annotate("Sample"):
                output:Optional[SamplerOutput] = self.model.sample(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )

        # dp 划分的时候可能会打乱原有的句子顺序，这里需要按照原有的顺序装回去，
        if output and post_sort_indexes:
            output = output.reorder_by_indexes(post_sort_indexes)
        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests = []
        dummy_lora_requests_per_seq = []
        if self.lora_config:
            for idx in range(self.lora_config.max_loras):
                lora_id = idx + 1
                dummy_lora_request = LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_local_path="/not/a/real/path",
                )
                self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                 rank=LORA_WARMUP_RANK)
                dummy_lora_requests.append(dummy_lora_request)
            dummy_lora_requests_per_seq = [
                dummy_lora_requests[idx % len(dummy_lora_requests)]
                for idx in range(max_num_seqs)
            ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        if self.vision_language_config:
            max_num_seqs = min(
                max_num_seqs,
                int(max_num_batched_tokens /
                    self.vision_language_config.image_feature_size))
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data, fake_multi_modal_input = _prepare_fake_inputs(
                seq_len, self.vision_language_config)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=fake_multi_modal_input,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return

    def remove_all_loras(self) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_all_loras()

    def set_active_loras(self, lora_requests: List[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_loras(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_loras()

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[torch.Tensor]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        # NOTE(woosuk): This is a hack to ensure that the NCCL backend is never
        # deleted before the CUDA graphs.
        self.pynccl_backend = pynccl_utils.get_nccl_backend()

        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        from vllm.distributed import get_atten_data_model_parallel_rank
        dp_size = self.parallel_config.atten_data_parallel_size
        rank = get_tensor_model_parallel_rank()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        max_batch_size_per_rank = max_batch_size // dp_size
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size_per_rank, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size_per_rank, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        context_lens = torch.ones(max_batch_size_per_rank, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs * dp_size)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        # NOTE(woosuk): There are 3 backends for all-reduce: custom all-reduce
        # kernel, pynccl, and PyTorch NCCL. When using CUDA graph, we use
        # either custom all-reduce kernel or pynccl. When not using CUDA
        # graph, we use either custom all-reduce kernel or PyTorch NCCL.
        # We always prioritize using custom all-reduce kernel but fall back
        # to PyTorch or pynccl if it is disabled or not supported.
        with custom_all_reduce.capture():
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            for batch_size in reversed(batch_size_capture_list):
                if batch_size < dp_size:
                    # 对于这种情况，因为存在 padding ，所以之后更大的 batchsize 已经囊括了
                    # 例如 batchsize=1 ，dp_size=2 ，padding 以后 batchsize=2
                    continue
                assert batch_size % dp_size == 0, \
                    f"batch_size({batch_size}) must be divisible by dp_size({dp_size})"

                batch_size_per_rank =  batch_size // dp_size
                cumsum_batchsize_list = [batch_size_per_rank*i for i in range(dp_size+1)]
                
                # Create dummy attn_metadata.
                decode_metadata = self.attn_backend.make_metadata(
                    is_prompt=False,
                    prompt_lens=None,
                    prompt_lens_tensor=None,
                    max_subquery_len=None,
                    max_context_len=self.max_context_len_to_capture,
                    max_prompt_len=None,
                    subquery_start_loc=None,
                    seq_start_loc=None,
                    context_lens=context_lens[:batch_size_per_rank],
                    block_tables=block_tables[:batch_size_per_rank],
                    use_cuda_graph=True,
                )
                attn_metadata = AttentionMetadata(
                    num_prefills=0,
                    num_prefill_tokens=0,
                    num_decode_tokens=batch_size_per_rank,
                    slot_mapping=slot_mapping[:batch_size_per_rank],
                    prefill_metadata=None,
                    decode_metadata=decode_metadata,
                    kv_cache_dtype=self.kv_cache_dtype,
                )
                
                # 这部分没有测试，可能会有问题
                if self.lora_config:
                    lora_mapping = LoRAMapping(
                        [0] * batch_size_per_rank,
                        [0] * batch_size_per_rank,
                    )
                    self.set_active_loras(set(), lora_mapping)

                graph_runner = CUDAGraphRunner(self.model)
                graph_runner.capture(
                    input_tokens[:batch_size], # 只有 input token 是 all rank 的
                    input_positions[:batch_size_per_rank],
                    kv_caches,
                    attn_metadata,
                    cumsum_batchsize_list,
                    memory_pool=self.graph_memory_pool,
                )
                self.graph_memory_pool = graph_runner.graph.pool()
                self.graph_runners[batch_size] = graph_runner

                print(f"rank {rank}: batch_size {batch_size} done!")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")
        print(f"support graph batch size: ", list(self.graph_runners.keys()))

    def __del__(self) -> None:
        # Delete the CUDA graphs before deleting the pynccl communicator.
        # NOTE(woosuk): This is necessary because otherwise deadlocks can
        # happen.
        # FIXME(woosuk): This is a bit hacky. Find a more robust solution.
        # TODO(youkaichao): when we get enough user feedback that pynccl is
        # more stable than cupy, we can remove this, e.g. in v0.4.1.
        self.graph_runners.clear()
        self.pynccl_backend = None

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        cumsum_batchsize_list: List[int],
        memory_pool,
        **kwargs,
    ) -> None:
        assert self.graph is None
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        with _maybe_pynccl():
            self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                cumsum_batchsize_list,
                **kwargs,
            )
        torch.cuda.synchronize()

        # Capture the graph.
        # NOTE(woosuk): Python 3.8 does not support multi-line with statements.
        # https://stackoverflow.com/questions/31039022/python-multi-line-with-statement
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=memory_pool):  # noqa: SIM117
            with _maybe_pynccl():
                hidden_states = self.model(
                    input_ids,
                    positions,
                    kv_caches,
                    attn_metadata,
                    cumsum_batchsize_list,
                    **kwargs,
                )
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": attn_metadata.slot_mapping,
            "context_lens": attn_metadata.decode_metadata.context_lens,
            "block_tables": attn_metadata.decode_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        return

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        cumsum_batchsize_list: List[int],
        **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["context_lens"].copy_(
            attn_metadata.decode_metadata.context_lens, non_blocking=True)
        self.input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@contextlib.contextmanager
def _maybe_pynccl():
    if pynccl_utils.is_initialized(
    ) and not custom_all_reduce.is_initialized():
        with with_pynccl_for_all_reduce():
            yield
    else:
        yield


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)


def _prepare_fake_inputs(
        seq_len: int, vision_language_config: Optional[VisionLanguageConfig]):
    """Prepare fake inputs for profile run."""
    if vision_language_config:
        prompt_tokens = [
            vision_language_config.image_token_id
        ] * vision_language_config.image_feature_size + [0] * (
            seq_len - vision_language_config.image_feature_size)
        fake_image_input = MultiModalData(
            type=MultiModalData.Type.IMAGE,
            data=torch.zeros(vision_language_config.image_input_shape,
                             dtype=torch.float16))
    else:
        prompt_tokens = [0] * seq_len
        fake_image_input = None
    return SequenceData(prompt_tokens), fake_image_input
