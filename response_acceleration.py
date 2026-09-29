"""Explicit execution backend; never change Actor, integrator or gradient rules.

Only the per-step physical transition is compiled. Dynamic survivor compaction
and native GRU/Linear hooks remain outside the compiled region. CUDA Graphs,
mixed precision and a custom adjoint are deliberately not enabled here.
"""
from __future__ import annotations

import torch

from env_raptor import RaptorParams, RaptorSimulator


def execution_contract(physics_backend: str = 'eager') -> dict:
    if physics_backend not in ('eager', 'compile'):
        raise ValueError('physics_backend must be eager or compile')
    return {
        'version': 'verified-hotpath-v1',
        'physics_backend': physics_backend,
        'record_training_observations': False,
        'reuse_task_feature_squares': True,
        'compiler': ({'backend': 'inductor', 'fullgraph': True, 'dynamic': True,
                      'mode': 'default', 'cudagraphs': False}
                     if physics_backend == 'compile' else None),
    }


def make_simulator(params: RaptorParams, physics_backend: str = 'eager') -> RaptorSimulator:
    """Build once per run, so compiler caches survive optimizer updates.

    Eager remains the default until a target-device benchmark passes. Compiling
    H500 or the Actor would interfere with the observer and is NOT done. Initial
    compilation and shape/grad-mode specializations have a separate startup cost.
    Errors are propagated, not retried using another backend.
    """
    execution_contract(physics_backend)
    simulator = RaptorSimulator(params)
    if physics_backend == 'compile':
        simulator.step = torch.compile(
            simulator.step, backend='inductor', fullgraph=True, dynamic=True,
            options={'triton.cudagraphs': False},
        )
    return simulator
