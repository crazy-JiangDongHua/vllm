from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import accumulate

import torch
from torch.distributed import ProcessGroup
import torch.distributed

from .parallel_state import (get_tensor_model_parallel_group,
                             get_atten_data_model_parallel_group,
                             get_atten_tensor_model_parallel_group,
                             get_tensor_model_parallel_rank,
                             get_atten_tensor_model_parallel_rank,
                             get_tensor_model_parallel_world_size,
                             get_atten_tensor_model_parallel_world_size,
                             get_atten_data_model_parallel_world_size,
                             is_pynccl_enabled_for_all_reduce)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation will be applied in-place on the input tensor if
    disable_custom_all_reduce is set to True. Otherwise, this operation may or
    may not be applied in place depending on whether custom all reduce is
    invoked for a particular tensor, which further depends on the tensor size
    and GPU topology.

    TLDR: always assume this function modifies its input, but use the return
    value as the output.
    """
    from vllm.distributed.device_communicators import pynccl_utils
    from vllm.distributed.device_communicators.custom_all_reduce import (
        custom_all_reduce)

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    out = custom_all_reduce(input_)
    if out is not None:
        return out
    if is_pynccl_enabled_for_all_reduce():
        # TODO: support multiple parallel groups.
        pynccl_utils.all_reduce(input_)
    else:
        torch.distributed.all_reduce(input_,
                                     group=get_tensor_model_parallel_group())

    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())
    return input_


def atten_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention model parallel group."""

    from vllm.distributed.device_communicators import pynccl_utils
    from vllm.distributed.device_communicators.custom_all_reduce import (
        custom_sub_all_reduce)
    
    # Bypass the function if we are using only 1 GPU.
    if get_atten_tensor_model_parallel_world_size() == 1:
        return input_
    out = custom_sub_all_reduce(input_)
    if out is not None:
        return out
    
    # pynccl_utils did't support sub group allreduce

    torch.distributed.all_reduce(input_,
                                 group=get_atten_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor


def atten_tensor_model_parallel_all_gather(input_: torch.Tensor,
                                           dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across attention model parallel group."""
    world_size = get_atten_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_atten_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor


def atten_data_model_parallel_all_gather_diff_size(input_: torch.Tensor,
                                                   output_size_list: List[torch.Size] = [],
                                                   dim: int = -1) -> torch.Tensor:
    """
    All-gather the different size input tensor across attention data model parallel group.
    Assume that only the shape of dim dimension is different.
    """
    world_size = get_atten_data_model_parallel_world_size()
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    if not output_size_list:
        # All-gather tensor size
        input_size = input_.size()
        output_size_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(output_size_list, 
                                            input_size, 
                                            group=get_atten_data_model_parallel_group())

    # Allocate output tensor.
    output_tensor_list = [torch.empty(size,
                                dtype=input_.dtype,
                                device=input_.device) 
                          for size in output_size_list]
    # All-gather.
    torch.distributed.all_gather(
        output_tensor_list, input_, group=get_atten_data_model_parallel_group())
    # Reshape
    output_tensor = torch.cat(output_tensor_list, dim=dim)
    return output_tensor


def atten_data_model_parallel_all_gather_diff_batchsize(
    input_: torch.Tensor,
    cumsum_batchsize_list: List[int],
) -> torch.Tensor:
    """
    All-gather the different batchsize input tensor across attention data model parallel group.
    Assume that only the shape of dim dimension is different and the dim is 0
    """
    world_size = get_atten_data_model_parallel_world_size()
    if world_size == 1:
        return input_
    
    assert cumsum_batchsize_list is not None
    
    # Allocate output tensor.
    size = list(input_.size())
    size[0] = cumsum_batchsize_list[-1]
    output_tensor = torch.empty(
        size=size,
        dtype=input_.dtype,
        device=input_.device
    )
    output_tensor_list = [output_tensor[cumsum_batchsize_list[i]:cumsum_batchsize_list[i+1]]
                          for i in range(len(cumsum_batchsize_list)-1)]
    # All-gather.
    torch.distributed.all_gather(
        output_tensor_list, input_, group=get_atten_data_model_parallel_group())
    return output_tensor


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group.

    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    # Allocate output tensor.
    if get_tensor_model_parallel_rank() == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    # Gather.
    torch.distributed.gather(input_,
                             gather_list,
                             dst=dst,
                             group=get_tensor_model_parallel_group())
    if get_tensor_model_parallel_rank() == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor


def atten_tensor_model_parallel_gather(input_: torch.Tensor,
                                        dst: int = 0,
                                        dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group.

    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    """
    world_size = get_atten_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    # Allocate output tensor.
    if get_atten_tensor_model_parallel_rank() == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    # Gather.
    torch.distributed.gather(input_,
                             gather_list,
                             dst=dst,
                             group=get_atten_tensor_model_parallel_group())
    if get_atten_tensor_model_parallel_rank() == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor


def broadcast(input_: torch.Tensor,
              src: int = 0,
              group: Optional[ProcessGroup] = None):
    """Broadcast the input tensor."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_
    # Broadcast.
    torch.distributed.broadcast(input_, src=src, group=group)
    return input_


def broadcast_object_list(obj_list: List[Any],
                          src: int = 0,
                          group: Optional[ProcessGroup] = None):
    """Broadcast the input object list."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return obj_list
    # Broadcast.
    torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
    return obj_list


TensorMetadata = namedtuple("TensorMetadata", ["dtype", "size"])


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
) -> Optional[Dict[Any, Union[torch.Tensor, Any]]]:
    """Broadcast the input tensor dictionary."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor_dict

    rank = torch.distributed.get_rank()
    if rank == src:
        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                assert value.is_cuda, (
                    f"Tensor {key}: {value} is not on cuda. Currently we only "
                    f"support broadcasting tensors on cuda.")
                metadata_list.append(
                    (key, TensorMetadata(value.dtype, value.size())))
            else:
                metadata_list.append((key, value))
        torch.distributed.broadcast_object_list([metadata_list],
                                                src=src,
                                                group=group)
        async_handles = []
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = tensor_dict[key]
                async_handles.append(
                    torch.distributed.broadcast(tensor,
                                                src=src,
                                                group=group,
                                                async_op=True))
        for async_handle in async_handles:
            async_handle.wait()

    else:
        recv_metadata_list = [None]
        torch.distributed.broadcast_object_list(recv_metadata_list,
                                                src=src,
                                                group=group)
        assert recv_metadata_list[0] is not None
        tensor_dict = {}
        async_handles = []
        for key, value in recv_metadata_list[0]:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device="cuda")
                async_handle = torch.distributed.broadcast(tensor,
                                                           src=src,
                                                           async_op=True,
                                                           group=group)
                async_handles.append(async_handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    return tensor_dict


# copy from torch.distributed 
import io
import pickle
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


def _object_to_tensor(obj, device):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size):
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


def batch_send_tensor_dict(
    tensor_dict_list: List[Optional[Dict[Any, Union[torch.Tensor, Any]]]] = [],
    dst_group_list: List[List[int]] = [],
    group: Optional[ProcessGroup] = None,
) -> None:
    """
    p2p send the input tensor dictionary to a list of peer group, 
    just like a collection of part group broadcast
    """
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)

    assert len(tensor_dict_list) == len(dst_group_list), \
        f"the number of tensor dict({len(tensor_dict_list)} is not equal)" \
        f"the number of dst group({len(dst_group_list)})"

    for dst_group in dst_group_list:
        for dst in dst_group:
            assert dst in ranks, f"Invalid dst rank ({dst})"
    
    src = torch.distributed.get_rank(group=group)
    for dst_group in dst_group_list:
        while src in dst_group:
            dst_group.remove(src)

    current_device = torch.device("cuda")
    metadata_list_list: List[List[Tuple[Any, Any]]] = []
    meta_tensor_list: List[torch.Tensor] = []
    meta_size_tensor_list: List[torch.Tensor] = []
    for tensor_dict in tensor_dict_list:
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        metadata_list = []
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                assert value.is_cuda, (
                    f"Tensor {key}: {value} is not on cuda. Currently we only "
                    f"support broadcasting tensors on cuda.")
                metadata_list.append(
                    (key, TensorMetadata(value.dtype, value.size())))
            else:
                metadata_list.append((key, value))
        metadata_list_list.append(metadata_list)
        # Serialize object elements to tensors on src rank.
        meta_tensor, meta_size_tensor = _object_to_tensor(metadata_list, current_device)
        meta_tensor_list.append(meta_tensor)
        meta_size_tensor_list.append(meta_size_tensor)

    p2p_op_list = []
    for dst_group, meta_size_tensor in zip(dst_group_list, meta_size_tensor_list):
        for dst in dst_group:
            p2p_op_list.append(
                torch.distributed.P2POp(
                    torch.distributed.isend,
                    meta_size_tensor,
                    dst,
                    group=group
                )
            )
    async_handles = torch.distributed.batch_isend_irecv(p2p_op_list)
    for async_handle in async_handles:
        async_handle.wait()

    p2p_op_list = []
    for dst_group, meta_tensor in zip(dst_group_list, meta_tensor_list):
        for dst in dst_group:
            p2p_op_list.append(
                torch.distributed.P2POp(
                    torch.distributed.isend,
                    meta_tensor,
                    dst,
                    group=group
                )
            )
    async_handles = torch.distributed.batch_isend_irecv(p2p_op_list)
    for async_handle in async_handles:
        async_handle.wait()
    
    p2p_op_list = []
    for dst_group, metadata_list, tensor_dict in \
        zip(dst_group_list, metadata_list_list, tensor_dict_list):
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = tensor_dict[key]
                for dst in dst_group:
                    p2p_op_list.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend,
                            tensor,
                            dst,
                            group=group
                        )
                    )
    async_handles = torch.distributed.batch_isend_irecv(p2p_op_list)
    for async_handle in async_handles:
        async_handle.wait()
    return


def recv_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
) -> Optional[Dict[Any, Union[torch.Tensor, Any]]]:
    """p2p recv the input tensor dictionary."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass if src is dst.
    dst = torch.distributed.get_rank(group=group)
    if src == dst:
        return tensor_dict

    current_device = torch.device("cuda")
    meta_size_tensor = torch.empty(1,
                                   dtype=torch.long,
                                   device=current_device)
    (handle, ) = torch.distributed.batch_isend_irecv([
        torch.distributed.P2POp(
            torch.distributed.irecv,
            meta_size_tensor,
            src,
            group=group
        )
    ]) 
    handle.wait()

    meta_size = meta_size_tensor[0]
    meta_tensor = torch.empty(
        meta_size,
        dtype=torch.uint8,
        device=current_device
    )
    (handle, ) = torch.distributed.batch_isend_irecv([
        torch.distributed.P2POp(
            torch.distributed.irecv,
            meta_tensor,
            src,
            group=group
        )
    ]) 
    handle.wait()
    metadata_list = _tensor_to_object(meta_tensor, meta_size)
    assert metadata_list is not None

    tensor_dict = {}
    p2p_op_list = []
    for key, value in metadata_list:
        if isinstance(value, TensorMetadata):
            tensor = torch.empty(value.size,
                                 dtype=value.dtype,
                                 device=current_device)
            p2p_op_list.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv,
                    tensor,
                    src,
                    group=group
                )
            )
            tensor_dict[key] = tensor
        else:
            tensor_dict[key] = value

    async_handles = torch.distributed.batch_isend_irecv(p2p_op_list)
    for async_handle in async_handles:
        async_handle.wait()
    
    return tensor_dict
