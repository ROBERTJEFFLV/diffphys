"""Exact Actor BPTT: one reverse state+parameter VJP per rematerialized window."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import torch

from response_task import (
    ResponseClosedLoopState,
    TaskLossConfig,
    FlightStatistics,
    initialize,
    rollout,
    step_costs,
    risk_weights,
)

PHYSICAL_DYNAMIC = ("position", "velocity", "rotation", "omega", "motor", "previous_action")
POLICY_DYNAMIC = (
    "memory",
    "integral",
    "previous_velocity",
    "previous_omega",
    "previous_rotation",
    "older_action",
)
POLICY_NONDIFFERENTIABLE = ("calls", "last_action")


def snapshot(closed: ResponseClosedLoopState) -> ResponseClosedLoopState:
    return type(closed)(
        *(
            type(state)(**{f.name: getattr(state, f.name).detach().clone() for f in fields(state)})
            for state in (closed.physical, closed.policy)
        )
    )


def dynamic_closed_state(closed):
    """Independent physical/history aliases; truth and discrete call state fixed."""
    if {f.name for f in fields(closed.policy)} != set(POLICY_DYNAMIC + POLICY_NONDIFFERENTIABLE):
        raise ValueError("register new policy state fields in the adjoint layout")
    states, leaves, names = [], [], []
    for group, state, active in (
        ("physical", closed.physical, PHYSICAL_DYNAMIC),
        ("policy", closed.policy, POLICY_DYNAMIC),
    ):
        values = {}
        for f in fields(state):
            z = getattr(state, f.name).detach()
            if f.name in active:
                z = z.requires_grad_(True)
                leaves.append(z)
                names.append((group, f.name))
            values[f.name] = z
        states.append(type(state)(**values))
    return ResponseClosedLoopState(*states), leaves, names


@torch.no_grad()
def compare_boundary(expected, actual, step):
    atol, rtol = (1e-6, 1e-5) if actual.physical.position.dtype == torch.float32 else (1e-12, 1e-10)
    errors, passed, exact = [], [], []
    for group in ("physical", "policy"):
        for f in fields(getattr(expected, group)):
            a, b = (
                getattr(getattr(expected, group), f.name),
                getattr(getattr(actual, group), f.name).detach(),
            )
            errors.append((a - b).abs().max())
            passed.append(
                (
                    torch.isfinite(a) & torch.isfinite(b) & ((a - b).abs() <= atol + rtol * b.abs())
                ).all()
            )
            exact.append((a == b).all())
    if not bool(torch.stack(passed).all()):
        raise RuntimeError("nonfinite or inconsistent boundary at step %d" % step)
    return {
        "step": step,
        "exact": bool(torch.stack(exact).all()),
        "max_error": float(torch.stack(errors).max()),
    }


@dataclass(frozen=True)
class BoundaryRecord:
    boundaries: dict[int, ResponseClosedLoopState]
    costs: torch.Tensor
    weights: torch.Tensor
    horizon: int
    window_steps: int
    loss_config: TaskLossConfig
    metrics: dict


@torch.no_grad()
def collect_boundary_rollout(policy, simulator, initial, config, *, horizon=500, window_steps=50):
    if horizon < 1 or window_steps < 1 or horizon % window_steps:
        raise ValueError("horizon must be divisible by window_steps")
    closed = initialize(policy, initial)
    boundaries = {0: snapshot(closed)}
    statistics = FlightStatistics(initial, horizon, config)
    cost_chunks = []
    for start in range(0, horizon, window_steps):
        trace = rollout(policy, simulator, closed, window_steps)
        # One full time reduction preserves the CVaR input and learning trajectory.
        cost_chunks.append(step_costs(trace, config, start=start, horizon=horizon))
        statistics.add(trace, start)
        closed = trace.end
        boundaries[start + window_steps] = snapshot(closed)
        del trace
    costs = torch.cat(cost_chunks).sum(0)
    if not bool(torch.isfinite(costs).all()):
        raise FloatingPointError("nonfinite continuous task costs")
    weights = risk_weights(costs, config)
    return BoundaryRecord(
        boundaries, costs, weights, horizon, window_steps, config, statistics.finish(costs, weights)
    )


def backward_actor(policy, simulator, record, config, *, gradient_scale=0.1):
    """Compute parameter and boundary VJPs together; publish grads only on success."""
    if config != record.loss_config:
        raise ValueError("recorded task objective changed")
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("gradient_scale must be positive and finite")
    parameters = [p for p in policy.parameters() if p.requires_grad]
    accumulated = [None] * len(parameters)
    adjoint, rows, starts = {}, [], []
    for start in reversed(range(0, record.horizon, record.window_steps)):
        end = start + record.window_steps
        state, leaves, names = dynamic_closed_state(record.boundaries[start])
        trace = rollout(policy, simulator, state, record.window_steps)
        rows.append(compare_boundary(record.boundaries[end], trace.end, end))
        objective = (
            record.weights * step_costs(trace, config, start=start, horizon=record.horizon).sum(0)
        ).sum()
        for (group, name), costate in adjoint.items():
            z = getattr(getattr(trace.end, group), name)
            objective = objective + (costate * (z - z.detach())).sum()
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("nonfinite reverse objective at %d" % start)
        gradients = torch.autograd.grad(objective, leaves + parameters, allow_unused=True)
        if not all(bool(torch.isfinite(g).all()) for g in gradients if g is not None):
            raise FloatingPointError("nonfinite full-horizon gradient at %d" % start)
        # Covectors already contain scene weights; no decay or boundary clipping.
        adjoint = {
            n: torch.zeros_like(z) if g is None else g.detach()
            for n, z, g in zip(names, leaves, gradients[: len(leaves)])
        }
        for i, g in enumerate(gradients[len(leaves) :]):
            if g is not None:
                if accumulated[i] is None:
                    accumulated[i] = g.detach().clone().mul_(gradient_scale)
                else:
                    accumulated[i].add_(g.detach(), alpha=gradient_scale)
        starts.append(start)
        del trace, state, leaves, gradients, objective
    if not all(bool(torch.isfinite(g).all()) for g in accumulated if g is not None):
        raise FloatingPointError("nonfinite accumulated Actor gradient")
    for p, g in zip(parameters, accumulated):
        p.grad = g
    return {
        "reverse_starts": starts,
        "boundaries": rows,
        "gradient_scale": gradient_scale,
        "active_state_groups": len(adjoint),
        "finite": True,
    }
