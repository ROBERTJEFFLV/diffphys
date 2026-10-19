"""Compile one original RK4 step; never compile the Actor or rollout loop.

All
equations, pulse lookup and state/history updates come from the original bound
method. AOTAutograd differentiates that same region. Compiler errors propagate.
"""
from __future__ import annotations

import torch

from .env_raptor import RaptorParams, RaptorSimulator, RaptorState


class CompiledRaptorSimulator(RaptorSimulator):
    def __init__(self, params: RaptorParams = RaptorParams()):
        super().__init__(params)
        from .strict_inductor import strict_inductor
        self._compiled_step = torch.compile(super().step, backend=strict_inductor,
                                            fullgraph=True, dynamic=True)

    def step(self, state: RaptorState, action: torch.Tensor) -> RaptorState:
        # Keep the existing public validation error outside Dynamo's exception wrapping.
        if action.shape != state.motor.shape or action.device != state.motor.device or action.dtype != state.motor.dtype:
            raise ValueError("action must match rotor state [batch,4], dtype and device")
        return self._compiled_step(state, action)


def make_simulator(params: RaptorParams, device) -> RaptorSimulator:
    """CUDA uses verified full-step compilation; CPU is the native reference."""
    return CompiledRaptorSimulator(params) if torch.device(device).type == 'cuda' else RaptorSimulator(params)


def physics_backend_contract(device) -> dict:
    if torch.device(device).type == 'cuda':
        from .strict_inductor import compiler_contract
        return {'physics':'compiled', **compiler_contract()}
    return {'physics':'native-cpu-reference'}
