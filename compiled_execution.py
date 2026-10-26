"""Explicit tensor compilation boundaries; native modules/hooks retain ownership."""
from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def compiled_kernel(function):
    from strict_inductor import strict_inductor, configure_compiler
    configure_compiler()
    return torch.compile(function, backend=strict_inductor, fullgraph=True, dynamic=True)


def tensor_kernel(function, *args, compiled=False, **kwargs):
    return (compiled_kernel(function) if compiled else function)(*args, **kwargs)


def measured_kernel(state):
    from response_noise import measured_observation
    return measured_observation(state, dt=.01)


def compile_actor(policy, device):
    if torch.device(device).type != 'cuda' or getattr(policy, '_compiled_execution', False):
        return policy
    # Hooks stay on the native modules, outside their compiled forward bodies.
    features = 'goal_features' if hasattr(policy, 'goal_features') else 'control_features'
    # Instance-bound callables are owned by that Actor, never a global cache.
    from strict_inductor import strict_inductor, configure_compiler
    configure_compiler()
    def compile_method(method):
        return torch.compile(method, backend=strict_inductor, fullgraph=True, dynamic=True)
    setattr(policy, features, compile_method(getattr(policy, features)))
    # PyTorch 2.10 CUDA's fused GRU lacks FakeTensor support. Keep its native
    # fused forward/backward and public dispatcher; never substitute gate math.
    policy.readout.forward = compile_method(policy.readout.forward)
    policy._compiled_execution = True
    return policy


def execution_contract(device, *, goal_task=False):
    if torch.device(device).type != 'cuda':
        return {'version': 'compiled-prefetch-v1', 'tensor_execution': 'native-cpu-reference',
                'sampling': 'spawn-one-batch'}
    from strict_inductor import compiler_contract
    regions = ['rk4-step', 'measurement', 'actor-features', 'readout', 'group-partials']
    if goal_task:
        regions += ['phase', 'task-loss']
    return {'version': 'compiled-prefetch-v1', 'sampling': 'spawn-one-batch',
            'tensor_execution': 'strict-inductor', 'compiler': compiler_contract(),
            'saved_tensors': {'storage': 'cpu-pinned-large-contiguous', 'minimum_elements': 65536},
            'regions': regions,
            'native_regions': ['fused-gru', 'Adam', 'validation-and-hooks'],
            'backward': 'AOTAutograd of the same tensor regions'}
