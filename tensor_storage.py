"""Runtime saved-tensor storage for this memory-limited CUDA run.

Large contiguous tensors may reside in pinned CPU memory between forward and
backward. Native RK4/GRU/loss and their gradients still execute on CUDA. Keep
strided views on their original device for compiled backward layout contracts.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch


def release_saved_tensor_cache(device):
    """Return inactive pinned allocations to the OS after the update graph dies."""
    if torch.device(device).type != 'cuda':
        return {}
    before = torch.cuda.memory.host_memory_stats()['allocated_bytes.current']
    # PyTorch 2.10 exposes host-cache release here; cuda.empty_cache is GPU-only.
    torch._C._host_emptyCache()
    after = torch.cuda.memory.host_memory_stats()['allocated_bytes.current']
    return {'pinned_reserved_before_bytes': before, 'pinned_reserved_after_bytes': after}


@contextmanager
def saved_tensor_storage(device):
    stats = {'offloaded_tensors': 0, 'offloaded_bytes': 0}
    if torch.device(device).type != 'cuda':
        yield stats
        return

    def pack(tensor):
        if tensor.device.type != "cuda" or not tensor.is_contiguous() or tensor.numel() < 65536:
            return (None, tensor.detach())
        cpu = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout, pin_memory=True)
        cpu.copy_(tensor.detach())
        stats['offloaded_tensors'] += 1
        stats['offloaded_bytes'] += tensor.numel()*tensor.element_size()
        return ((tensor.device, tensor.stride()), cpu)

    def unpack(saved):
        metadata, tensor = saved
        if metadata is None:
            return tensor
        device, stride = metadata
        restored = tensor.to(device, non_blocking=True)
        if restored.stride() != stride:
            layout = torch.empty_strided(restored.size(), stride, device=device, dtype=restored.dtype)
            layout.copy_(restored)
            restored = layout
        return restored

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats
