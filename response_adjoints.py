"""Exact block adjoints and explicit closed-loop derivative coordinates.

The oracle retains ambient matrix covectors and independent history aliases.
Only the regression coordinates project rotations onto SO(3) tangent directions.
No normalization statistics, policy inputs, or physical equations change here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import hashlib

import torch

from response_phase1 import compare_boundary, dynamic_closed_state
from response_task import rollout, step_costs

ADJOINT_OBJECTIVE = 'exact-task-boundary-adjoint-v1'
STATE_SCALES = {
    'physical.position': 5., 'physical.velocity': 5., 'physical.rotation': 1.,
    'physical.omega': 10., 'physical.motor': 1., 'physical.previous_action': 1.,
    'policy.memory': 1., 'policy.integral': .5, 'policy.previous_velocity': 5.,
    'policy.previous_omega': 10., 'policy.previous_rotation': 1., 'policy.older_action': 1.,
}


def state_field(closed, name):
    group, field = name.split('.')
    return getattr(getattr(closed, group), field)


def gradient_coordinates(name, gradient, closed):
    """dV/d(x/s), or dV/d(delta) for R exp([delta]_x), in radians."""
    if name.endswith('rotation'):
        product = state_field(closed, name).detach().transpose(-1, -2) @ gradient
        return torch.stack((product[:, 2, 1]-product[:, 1, 2],
                            product[:, 0, 2]-product[:, 2, 0],
                            product[:, 1, 0]-product[:, 0, 1]), dim=-1)
    return gradient.flatten(1) * STATE_SCALES[name]


@contextmanager
def frozen_parameters(module):
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


@dataclass(frozen=True)
class BoundaryAdjoints:
    gradients: dict
    horizon: int
    window_steps: int
    actor_sha256: str
    loss_config: dict
    continuity: list
    initial_sha256: str
    objective: str = ADJOINT_OBJECTIVE

    def validate(self, policy, horizon, window_steps, loss_config, record):
        from response_training import model_hash
        if self.actor_sha256 != model_hash(policy):
            raise ValueError('boundary adjoints belong to another Actor')
        if (self.objective != ADJOINT_OBJECTIVE or self.horizon != horizon or
                self.window_steps != window_steps or self.loss_config != asdict(loss_config)):
            raise ValueError('boundary adjoint objective/time configuration changed')
        if self.initial_sha256 != boundary_fingerprint(record.boundaries[0]):
            raise ValueError('boundary adjoints belong to different initial states')


def boundary_fingerprint(closed):
    digest = hashlib.sha256()
    for group in ('physical', 'policy'):
        state = getattr(closed, group)
        for field in fields(state):
            tensor = getattr(state, field.name).detach().contiguous().cpu()
            digest.update((group+'.'+field.name+str(tensor.dtype)+str(tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def linearized_terminal(closed, gradients, remaining):
    """Exact value at the reference state with a prescribed first derivative."""
    result = remaining.detach()
    for name, costate in gradients.items():
        z = state_field(closed, name)
        result = result + (costate.detach() * (z-z.detach())).flatten(1).sum(-1)
    return result


def collect_boundary_adjoints(policy, simulator, record, horizon, window_steps, loss_config):
    """One reverse H-window pass, unweighted per-scene full dynamic adjoints.

    Each graph contains only one window. The adjoint itself carries the exact
    long-horizon chain, including any real explosion; this is not an approximation.
    """
    from response_training import model_hash
    if horizon < 1 or window_steps < 1 or horizon % window_steps:
        raise ValueError('adjoint horizon must be divisible by window_steps')
    if set(record.boundaries) != set(range(0, horizon+1, window_steps)):
        raise ValueError('adjoints need all boundaries including the initial state')
    actor_sha = model_hash(policy)
    _, terminal_leaves, names = dynamic_closed_state(record.boundaries[horizon])
    gradients = {horizon: {n: torch.zeros_like(z) for n, z in zip(names, terminal_leaves)}}
    continuity = []
    with frozen_parameters(policy), torch.enable_grad():
        for start in reversed(range(0, horizon, window_steps)):
            end = start+window_steps
            state, leaves, current_names = dynamic_closed_state(record.boundaries[start])
            if names != current_names:
                raise ValueError('dynamic boundary layout changed')
            trace = rollout(policy, simulator, state, window_steps)
            continuity.append(compare_boundary(record.boundaries[end], trace.end, end))
            local = step_costs(trace, loss_config, start=start, horizon=horizon).sum(0)
            objective = local + linearized_terminal(trace.end, gradients[end], record.returns[end])
            raw = torch.autograd.grad(objective.sum(), leaves, allow_unused=True)
            gradients[start] = {n: torch.zeros_like(z) if g is None else g.detach()
                                for n, z, g in zip(names, leaves, raw)}
            if not bool(torch.isfinite(objective).all()) or not all(
                    bool(torch.isfinite(g).all()) for g in gradients[start].values()):
                raise FloatingPointError('nonfinite full boundary adjoint')
            del trace, objective, local, raw, leaves, state
    return BoundaryAdjoints(gradients, horizon, window_steps, actor_sha, asdict(loss_config), continuity,
                            boundary_fingerprint(record.boundaries[0]))
