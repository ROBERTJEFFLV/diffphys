"""Full or recomputed Actor gradients, exact at zero temporal decay."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import torch

from response_task import (
    ResponseClosedLoopState,
    PHYSICAL_DYNAMIC,
    POLICY_DYNAMIC,
    TaskLossConfig,
    FlightStatistics,
    initialize,
    rollout,
    step_costs,
    risk_weights,
    tensors_finite,
)

from response_groups import (GroupBalanceConfig, group_gradient_coefficients,
                             backward_group_gradients)

POLICY_NONDIFFERENTIABLE = ()


def snapshot(closed: ResponseClosedLoopState) -> ResponseClosedLoopState:
    return type(closed)(
        *(
            type(state)(**{f.name: (getattr(state, f.name).detach() if f.name == "noise_tape"
                                  else getattr(state, f.name).detach().clone())
                          for f in fields(state)})
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
            if f.name == "noise_tape" and a.data_ptr() == b.data_ptr() and a.shape == b.shape:
                continue  # Immutable tape is shared, not recomputed or modified.
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
    valid: torch.Tensor
    backprop_mode: str = "windowed"
    time_decay: float = 0.0
    group_coefficients: torch.Tensor | None = None
    group_balance: dict | None = None
    group_config: GroupBalanceConfig | None = None


def collect_boundary_rollout(
    policy, simulator, initial, config, *, horizon=500, window_steps=50,
    backprop_mode="windowed", time_decay=0.0, group_config: GroupBalanceConfig | None = None,
):
    if horizon < 1 or window_steps < 1 or horizon % window_steps:
        raise ValueError("horizon must be divisible by window_steps")
    if backprop_mode not in ("full", "windowed"):
        raise ValueError("unknown backprop mode")
    group_config = group_config or GroupBalanceConfig()
    if group_config.enabled and backprop_mode != "full":
        raise ValueError("group gradient normalization requires full retained-graph BPTT")
    with torch.set_grad_enabled(backprop_mode == "full"):
        closed = initialize(policy, initial)
        boundaries = {0: snapshot(closed)}
        statistics = FlightStatistics(initial, horizon, config)
        cost_chunks, valid_chunks = [], []
        for start in range(0, horizon, window_steps):
            trace = rollout(policy, simulator, closed, window_steps, time_decay=time_decay)
            # No detach between windows: full mode retains the entire H500 graph.
            cost_chunks.append(step_costs(trace, config, start=start, horizon=horizon))
            valid_chunks.append(trace.valid)
            statistics.add(trace, start)
            closed = trace.end
            boundaries[start + window_steps] = snapshot(closed)
            del trace
        costs = torch.cat(cost_chunks).sum(0)
        if not bool(torch.isfinite(costs).all()):
            raise FloatingPointError("nonfinite continuous task costs")
        weights = risk_weights(costs, config)
    coefficients, group_report = group_gradient_coefficients(
        weights, boundaries[0].physical, group_config)
    metrics = statistics.finish(costs.detach(), weights)
    return BoundaryRecord(
        boundaries, costs, weights, horizon, window_steps, config,
        metrics, torch.cat(valid_chunks), backprop_mode, time_decay,
        coefficients, group_report, group_config,
    )


def backward_actor(policy, simulator, record, config, *, gradient_scale=0.1):
    """Compute group-normalized gradients or the unchanged pooled baseline."""
    if config != record.loss_config:
        raise ValueError("recorded task objective changed")
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("gradient_scale must be positive and finite")
    parameters = [p for p in policy.parameters() if p.requires_grad]
    weights = record.weights
    if record.group_coefficients is not None:
        if record.backprop_mode != "full" or record.group_config is None:
            raise ValueError("group gradients require a full graph and bound group configuration")
        gradients, report = backward_group_gradients(
            record.costs, record.group_coefficients, parameters, record.group_config,
            gradient_scale=gradient_scale)
        for p, g in zip(parameters, gradients):
            p.grad = g
        return {"boundaries": [], "group_gradient": report}
    if record.backprop_mode == "full":
        objective = gradient_scale * (weights * record.costs).sum()
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("nonfinite full-horizon objective")
        gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
        if not tensors_finite(gradients):
            raise FloatingPointError("nonfinite full-horizon Actor gradient")
        for p, g in zip(parameters, gradients):
            p.grad = g
        # No re-computation took place; callers must not report a boundary match.
        return {"boundaries": []}
    accumulated = [None] * len(parameters)
    adjoint, rows = {}, []
    for start in reversed(range(0, record.horizon, record.window_steps)):
        end = start + record.window_steps
        state, leaves, names = dynamic_closed_state(record.boundaries[start])
        trace = rollout(policy, simulator, state, record.window_steps, time_decay=record.time_decay)
        if not torch.equal(trace.valid, record.valid[start:end]):
            raise RuntimeError("inconsistent termination mask at step %d" % start)
        rows.append(compare_boundary(record.boundaries[end], trace.end, end))
        objective = (
            weights * step_costs(trace, config, start=start, horizon=record.horizon).sum(0)
        ).sum()
        for (group, name), costate in adjoint.items():
            z = getattr(getattr(trace.end, group), name)
            objective = objective + (costate * (z - z.detach())).sum()
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("nonfinite reverse objective at %d" % start)
        gradients = torch.autograd.grad(objective, leaves + parameters, allow_unused=True)
        if not tensors_finite(gradients):
            raise FloatingPointError("nonfinite full-horizon gradient at %d" % start)
        # Decay already occurred per physical step; do not apply it again at boundaries.
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
        del trace, state, leaves, gradients, objective
    if not tensors_finite(accumulated):
        raise FloatingPointError("nonfinite accumulated Actor gradient")
    for p, g in zip(parameters, accumulated):
        p.grad = g
    return {
        "boundaries": rows,
    }
