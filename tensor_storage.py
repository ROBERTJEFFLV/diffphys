"""Runtime saved-tensor storage for this memory-limited CUDA run.

Large contiguous tensors may reside in pinned CPU memory between forward and
backward. Native RK4/GRU/loss and their gradients still execute on CUDA. Keep
strided views on their original device for compiled backward layout contracts.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch

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
