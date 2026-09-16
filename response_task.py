"""Teacher-free causal rollout, physical task objective, and scenario banks."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Optional

import torch

from env_l2f import L2FParams, L2FSimulator, L2FState
from response_policy import (
    ResponseMotorPolicy, ResponsePolicyState, body_vector,
)


# Shared by per-step gradient decay and exact reverse-window state covectors.
PHYSICAL_DYNAMIC = ("position", "velocity", "orientation", "omega", "motor", "previous_action")
POLICY_DYNAMIC = (
    "memory",
    "integral",
    "previous_velocity",
    "previous_omega",
    "previous_rotation",
    "older_action",
)


@dataclass(frozen=True)
class TaskLossConfig:
    position_weight: float = 1.0
    velocity_weight: float = 0.3
    omega_weight: float = 0.1
    action_weight: float = 0.0001
    action_delta_weight: float = 0.01
    omega_delta_weight: float = 0.005
    steady_weight: float = 2.0
    steady_steps: int = 100
    tail_weight: float = 0.5
    tail_fraction: float = 0.2
    huber_delta: float = 1.0
    dead_cost: float = 3.0  # Raw cost per unflown step after failure.
    terminal_cost: float = 200.0  # Raw one-off cost, including failure at H.

    def __post_init__(self) -> None:
        weights = [getattr(self, f.name) for f in fields(self) if f.name.endswith("weight")]
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("task weights must be finite and non-negative")
        if self.position_weight <= 0 or self.omega_weight <= 0:
            raise ValueError("position holding and full angular velocity must be penalized")
        if self.steady_steps < 1 or not 0 < self.tail_fraction <= 1:
            raise ValueError("invalid steady window or CVaR tail fraction")
        if not math.isfinite(self.huber_delta) or self.huber_delta < 0:
            raise ValueError("Huber delta must be finite and non-negative (0 selects historical squares)")
        if any(not math.isfinite(c) or c < 0 for c in (self.dead_cost, self.terminal_cost)):
            raise ValueError("failure costs must be finite and non-negative")


@dataclass(frozen=True)
class ResponseClosedLoopState:
    physical: L2FState
    policy: ResponsePolicyState


@dataclass(frozen=True)
class TaskTrajectory:
    end: ResponseClosedLoopState
    observations: torch.Tensor
    actions: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor
    omegas: torch.Tensor
    action_deltas: torch.Tensor
    omega_deltas: torch.Tensor
    initial: L2FState
    valid: torch.Tensor  # [time, scene], including the first terminal transition


def observation(physical: L2FState, integral: torch.Tensor) -> torch.Tensor:
    """Noisy deployable state only; one fixed noise sample per physical time step.

    Re-observing a window boundary MUST NOT draw a second noise realization.
    The simulator's true orientation is never changed by observation noise.
    """
    clean = torch.cat((physical.position, physical.velocity,
                       physical.rotation.flatten(1), physical.omega), -1)
    if physical.noise_tape.shape[1] == 1:
        noise = physical.noise_tape[:, 0]
    else:
        rows = torch.arange(clean.shape[0], device=clean.device)
        noise = physical.noise_tape[rows, physical.step_index]
    measured = clean + noise
    rotation_measured = measured[:, 6:15].reshape(-1, 3, 3)
    return torch.cat((measured, body_vector(rotation_measured, integral),
                      physical.previous_action), -1)


def initialize(policy: ResponseMotorPolicy, physical: L2FState) -> ResponseClosedLoopState:
    obs = observation(physical, torch.zeros_like(physical.position))
    return ResponseClosedLoopState(physical, policy.initial_state(obs))


def _select_rows(state, indices: torch.Tensor):
    """Compact live rows without detaching their physical or recurrent graph."""
    if indices.numel() == getattr(state, fields(state)[0].name).shape[0]:
        return state
    return type(state)(**{f.name: getattr(state, f.name).index_select(0, indices)
                          for f in fields(state)})


def _merge_rows(full, before, after, indices: torch.Tensor):
    if indices.numel() == getattr(full, fields(full)[0].name).shape[0]:
        return after
    # Only dynamic fields changed by step/Actor are scattered. In particular,
    # do not copy the full immutable noise tape on every physical time step.
    return type(full)(**{
        f.name: (getattr(full, f.name) if getattr(before, f.name) is getattr(after, f.name)
                 else getattr(full, f.name).index_copy(0, indices, getattr(after, f.name)))
        for f in fields(full)
    })


class _GradientDecay(torch.autograd.Function):
    """Identity forward; deliberately replace the state VJP by rho times itself."""
    @staticmethod
    def forward(ctx, value, rho):
        ctx.rho = rho
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.rho, None


def _decay_closed_state(closed: ResponseClosedLoopState, rho: float) -> ResponseClosedLoopState:
    # No tensor clones: immutable physics parameters and the noise tape stay shared.
    return ResponseClosedLoopState(*(
        replace(state, **{name: _GradientDecay.apply(getattr(state, name), rho)
                          for name in names})
        for state, names in ((closed.physical, PHYSICAL_DYNAMIC), (closed.policy, POLICY_DYNAMIC))
    ))


def rollout(
    policy: ResponseMotorPolicy,
    simulator: L2FSimulator,
    initial: L2FState | ResponseClosedLoopState,
    steps: int,
    *,
    time_decay: float = 0.0,
) -> TaskTrajectory:
    """Unchanged flight; optional backward-only decay at every control step.

    time_decay is alpha in s^-1, rho=exp(-alpha*dt). Zero keeps exact BPTT.
    Positive values define a surrogate gradient, not a discounted forward loss.
    """
    if not math.isfinite(time_decay) or time_decay < 0:
        raise ValueError("time_decay must be finite and non-negative")
    rho = math.exp(-time_decay * simulator.params.dt)
    decay = time_decay > 0 and torch.is_grad_enabled()
    if steps < 1:
        raise ValueError("rollout must contain physical transitions")
    if abs(simulator.params.dt - policy.config.dt) > 1.0e-12:
        raise ValueError("training and deployment dt must agree")
    closed = initialize(policy, initial) if isinstance(initial, L2FState) else initial
    expected_code = int(simulator.params.protocol == "raptor")
    if not bool((closed.physical.profile_code == expected_code).all()):
        raise ValueError("sampler and simulator reference protocols disagree")
    if (closed.physical.noise_tape.shape[1] > 1
            and int(closed.physical.step_index.max()) + steps >= closed.physical.noise_tape.shape[1]):
        raise ValueError("noise tape too short: sample scenarios for the full requested horizon")
    initial_physical = closed.physical
    observations = [observation(closed.physical, closed.policy.integral)]
    actions, positions, velocities, omegas, action_deltas, omega_deltas = [], [], [], [], [], []
    valid = []
    # A terminal row stays frozen outside its boundary, so it cannot become live
    # again when this rollout is resumed at the next reverse-window boundary.
    indices = (~simulator.terminated(closed.physical)).nonzero(as_tuple=True)[0]
    live = ResponseClosedLoopState(_select_rows(closed.physical, indices),
                                   _select_rows(closed.policy, indices))
    for step in range(steps):
        current_observation = observations[-1]
        if decay and indices.numel():
            # Full and compact tensors are parallel aliases, not serial gates.
            # Cover observation, physics, GRU/history and delta-cost paths once.
            closed = _decay_closed_state(closed, rho)
            live = _decay_closed_state(live, rho)
            current_observation = observation(closed.physical, closed.policy.integral)
        before = closed.physical
        action = before.previous_action
        active = torch.zeros_like(before.step_index, dtype=torch.bool)
        if indices.numel():
            active[indices] = True  # The upcoming crossing transition counts.
            output = policy(current_observation.index_select(0, indices), live.policy)
            physical = simulator.step(live.physical, output.action)
            closed = ResponseClosedLoopState(
                _merge_rows(closed.physical, live.physical, physical, indices),
                _merge_rows(closed.policy, live.policy, output.next_state, indices),
            )
            action = action.index_copy(0, indices, output.action)
            keep = (~simulator.terminated(physical)).nonzero(as_tuple=True)[0]
            indices = indices.index_select(0, keep)
            live = ResponseClosedLoopState(_select_rows(physical, keep),
                                           _select_rows(output.next_state, keep))
        # Frozen terminal rows are storage padding only: no further Actor, RK4,
        # memory or noise-index updates, and no costs/statistics for padded steps.
        physical = closed.physical
        valid.append(active)
        actions.append(action)
        positions.append(physical.position)
        velocities.append(physical.velocity)
        omegas.append(physical.omega)
        action_deltas.append(action - before.previous_action)
        omega_deltas.append(physical.omega - before.omega)
        observations.append(observation(physical, closed.policy.integral))
    return TaskTrajectory(
        closed, torch.stack(observations), torch.stack(actions), torch.stack(positions),
        torch.stack(velocities), torch.stack(omegas), torch.stack(action_deltas),
        torch.stack(omega_deltas), initial_physical, torch.stack(valid),
    )


def weighted_task_features(
    trajectory: TaskTrajectory, config: TaskLossConfig, *, start: int = 0,
    horizon: Optional[int] = None,
) -> torch.Tensor:
    """Signed sqrt(2 Huber) residuals, with the original small-error curvature.

    Robustify physical errors before applying their weights. Global time indices
    retain the full-flight mean and final steady window when used on short slices.
    Delta=0 is available only to reproduce historical quadratic objectives.
    Two constant residuals add ((H-X)*dead_cost + terminal_cost)/H at the
    first failed transition. They bypass Huber and steady-window weighting;
    no post-failure physics or gradient through the discrete failure time.
    """
    def robust(value):
        # Physical costs and their normalization are unchanged. Failure
        # bookkeeping is added separately after applying these time weights.
        value = torch.where(trajectory.valid[..., None], value, 0.0)
        delta = config.huber_delta
        if delta == 0:
            return value
        outer = value.sign() * (2 * delta * value.abs() - delta ** 2).clamp_min(delta ** 2).sqrt()
        return torch.where(value.abs() <= delta, value, outer)

    features = torch.cat((
        math.sqrt(config.position_weight) * robust(trajectory.positions),
        math.sqrt(config.velocity_weight) * robust(trajectory.velocities),
        math.sqrt(config.omega_weight) * robust(trajectory.omegas),
        math.sqrt(config.action_weight) * robust(0.5 * trajectory.actions),
        math.sqrt(config.action_delta_weight) * robust(0.5 * trajectory.action_deltas),
        math.sqrt(config.omega_delta_weight) * robust(trajectory.omega_deltas),
    ), -1)
    steps = features.shape[0]
    horizon = steps if horizon is None else horizon
    if start < 0 or horizon < 1 or start + steps > horizon:
        raise ValueError("cost slice must lie inside the full horizon")
    tail = min(config.steady_steps, horizon)
    time_weights = features.new_full((steps,), 1.0 / horizon)
    time_weights = time_weights + (
        torch.arange(start, start + steps, device=features.device) >= horizon - tail
    ).to(features) * (config.steady_weight / tail)
    features = features * time_weights.sqrt()[:, None, None]
    with torch.no_grad():
        state = trajectory.initial
        terminal = trajectory.valid & (
            (trajectory.positions.abs() > state.position_limit[None, :, None]).any(-1)
            | (trajectory.velocities.abs() > state.velocity_limit[None, :, None]).any(-1)
            | (trajectory.omegas.abs() > state.omega_limit[None, :, None]).any(-1)
        )
        remaining = horizon - torch.arange(start + 1, start + steps + 1,
                                            device=features.device, dtype=features.dtype)
        # Book all missing steps once at failure, not at each window end.
        # valid excludes frozen padding; reaching H without a violation costs zero.
        failed = terminal.to(features)
        failure = torch.stack((failed * remaining[:, None] * (config.dead_cost / horizon),
                               failed * (config.terminal_cost / horizon)), -1)
    return torch.cat((features, failure.sqrt()), -1)


def scenario_costs(trajectory: TaskTrajectory, config: TaskLossConfig) -> torch.Tensor:
    return step_costs(trajectory, config).sum(0)


def step_costs(trajectory, config, *, start=0, horizon=None) -> torch.Tensor:
    """Additive [time, scenario] costs, without mean/CVaR scenario weights."""
    return weighted_task_features(trajectory, config, start=start, horizon=horizon).square().sum(-1)


def warning_risk_steps(trajectory) -> dict[str, torch.Tensor]:
    """Unchanged hover warning exposure, used only as a physical observation."""
    return {
        "omega": (trajectory.omegas.detach().norm(dim=-1) / 10.0 - 1).clamp_min(0).square()
                 * trajectory.valid,
        "saturation": ((trajectory.actions.detach().abs() - .95) / (1.0 - .95))
                      .clamp_min(0).square().mean(-1) * trajectory.valid,
    }


def hard_risk_metrics(trajectory, loss_config: TaskLossConfig) -> dict:
    risks = {name: value.sum(0) for name, value in warning_risk_steps(trajectory).items()}
    count = max(1, math.ceil(loss_config.tail_fraction * trajectory.actions.shape[1]))
    risks = {name: float(value.mean() + loss_config.tail_weight * value.topk(count).values.mean())
             for name, value in risks.items()}
    position = torch.cat((trajectory.initial.position[None], trajectory.positions)).detach().norm(dim=-1)
    velocity = torch.cat((trajectory.initial.velocity[None], trajectory.velocities)).detach().norm(dim=-1)
    return {"hard_risk_components": risks,
            "hard_risk_peaks": {"omega": float(trajectory.omegas.detach().norm(dim=-1).max()),
                                "action_abs": float(trajectory.actions.detach().abs().max()),
                                "position": float(position.max()), "velocity": float(velocity.max())}}


def risk_weights(costs: torch.Tensor, config: TaskLossConfig) -> torch.Tensor:
    """Choose the pooled full-flight tail once; reuse these detached weights."""
    if costs.ndim != 1 or costs.numel() == 0 or not bool(torch.isfinite(costs).all()):
        raise ValueError("risk weights need finite per-scenario full-flight costs")
    count = max(1, math.ceil(config.tail_fraction * costs.numel()))
    weights = torch.full_like(costs, 1.0 / costs.numel())
    weights[costs.detach().topk(count).indices] += config.tail_weight / count
    return weights.detach()


def task_residual(trajectory: TaskTrajectory, config: TaskLossConfig) -> torch.Tensor:
    features = weighted_task_features(trajectory, config)
    costs = features.square().sum(dim=(0, 2))
    count = max(1, math.ceil(config.tail_fraction * costs.numel()))
    selected = costs.detach().topk(count).indices
    # .5 ||residual||^2 equals mean cost plus upper-tail CVaR cost.
    return torch.cat((
        (math.sqrt(2.0 / costs.numel()) * features).reshape(-1),
        (math.sqrt(2.0 * config.tail_weight / count) * features[:, selected]).reshape(-1),
    ))


def task_loss(trajectory: TaskTrajectory, config: TaskLossConfig) -> torch.Tensor:
    return 0.5 * task_residual(trajectory, config).square().sum()


def sample_scenarios(
    count: int, *, seed: int, dt: float = 0.01,
    device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32,
    scenario_mode: str = "raptor", horizon: int = 500,
) -> L2FState:
    """Source-defined airframes, paper-first initial conditions and fixed noise.

    A fresh episode sample uses an independent local seed. There is no 4x4
    authority reweighting and no environment-default fallback in this path.
    """
    simulator = L2FSimulator(L2FParams(dt=dt, protocol=scenario_mode))
    return simulator.reset(count, seed=seed, horizon=horizon, device=device, dtype=dtype)


def trajectory_metrics(trajectory: TaskTrajectory, config: TaskLossConfig) -> dict:
    """Aggregate control metrics over every scene, with the original finite mask."""
    costs = scenario_costs(trajectory, config).detach()
    p, v, w = [x.detach().norm(dim=-1) for x in (
        trajectory.positions, trajectory.velocities, trajectory.omegas
    )]
    finite = torch.stack([
        torch.isfinite(x).flatten(2).all(-1)
        for x in (trajectory.positions, trajectory.velocities, trajectory.omegas, trajectory.actions)
    ]).all(dim=(0, 1))
    if hasattr(trajectory, "end") and trajectory.end is not None:
        for field in fields(trajectory.end.physical):
            value = getattr(trajectory.end.physical, field.name)
            finite = finite & torch.isfinite(value).reshape(value.shape[0], -1).all(-1)
        finite = finite & torch.isfinite(trajectory.end.policy.memory).all(-1)
    valid = trajectory.valid
    count = valid.sum().clamp_min(1)
    saturation = (trajectory.actions.detach().abs() >= 1.0 - 1.0e-6).to(p.dtype).mean(-1)
    return {
        "task_objective": float(task_loss(trajectory, config).detach()),
        "position_rms": float((p.square()[valid].sum() / count).sqrt()),
        "velocity_rms": float((v.square()[valid].sum() / count).sqrt()),
        "omega_rms": float((w.square()[valid].sum() / count).sqrt()),
        "motor_saturation_fraction": float(saturation[valid].sum() / count),
        "physical_transitions": int(valid.sum()),
        "finite": bool(finite.all()),
        "scenario_count": costs.numel(),
    }


@torch.no_grad()
def reference_episode_metrics(trajectory: TaskTrajectory) -> dict:
    """Score first termination with each airframe's own reference boundaries.

    Only publish the matching profile, never apply a L2F box to RAPTOR scenes.
    The terminal transition is retained; subsequent entries are frozen padding.
    The first violation remains a failure even on the final allowed transition.
    """
    state = trajectory.initial
    code = int(state.profile_code[0])
    if not bool((state.profile_code == code).all()):
        raise ValueError("a reference evaluation bank cannot mix protocols")
    if bool(L2FSimulator.terminated(state).any()):
        raise ValueError("reference evaluation starts outside its termination set")
    name = "raptor" if code == 1 else "l2f"
    horizon = trajectory.positions.shape[0]
    terminated = ((trajectory.positions.abs() > state.position_limit[None, :, None]).any(-1)
                  | (trajectory.velocities.abs() > state.velocity_limit[None, :, None]).any(-1)
                  | (trajectory.omegas.abs() > state.omega_limit[None, :, None]).any(-1))
    steps = torch.arange(1, horizon+1, device=terminated.device)[:, None]
    lengths = torch.where(terminated, steps, horizon).amin(dim=0)
    result = {
        "reference_protocol": name,
        name + "_episode_length_mean": float(lengths.double().mean()),
        name + "_episode_length_std": float(lengths.double().std(unbiased=False)),
        name + "_share_terminated": float(terminated.any(0).double().mean()),
        "reference_position_limit_min_m": float(state.position_limit.min()),
        "reference_position_limit_max_m": float(state.position_limit.max()),
    }
    if name == "l2f":
        settled = (lengths == horizon) & (trajectory.positions[-1].norm(dim=-1) < 0.20)
        result["l2f_settling_fraction_200mm"] = float(settled.double().mean())
    return result


def task_loss_components(trajectory, config, *, start=0, horizon=None, weights=None):
    """Additive attribution of the existing task objective, including its CVaR."""
    features = weighted_task_features(trajectory, config, start=start, horizon=horizon).square()
    if weights is None:
        weights = risk_weights(features.sum(dim=(0, 2)), config)
    return {name: float((weights * features[:, :, section].sum(dim=(0, 2))).sum().detach())
            for name, section in (('position', slice(0, 3)), ('velocity', slice(3, 6)),
                                  ('omega', slice(6, 9)), ('regularization', slice(9, 20)),
                                  ('dead', slice(20, 21)), ('terminal', slice(21, 22)))}


def tensors_finite(values):
    """Reduce on each device before reading; Adam step counters may live on CPU."""
    checks = {}
    for value in values:
        if value is not None:
            checks.setdefault(value.device, []).append(torch.isfinite(value).all())
    return all(bool(torch.stack(flags).all()) for flags in checks.values())


class FlightStatistics:
    """Streaming observations; no trajectory, labels or training gates retained."""

    def __init__(self, initial, horizon, config):
        self.horizon, self.config = horizon, config
        self.squares = initial.position.new_zeros(3, initial.position.shape[0])
        self.saturation = self.squares[0].clone()
        self.valid_steps = torch.zeros_like(initial.step_index)
        self.components = initial.position.new_zeros(6, initial.position.shape[0])
        self.risks = initial.position.new_zeros(2, initial.position.shape[0])

    @torch.no_grad()
    def add(self, trace, start):
        magnitudes = torch.stack([x.norm(dim=-1) for x in
                                  (trace.positions, trace.velocities, trace.omegas)])
        values = [magnitudes, trace.actions]
        values += [getattr(s, f.name) for s in (trace.end.physical, trace.end.policy) for f in fields(s)]
        if not tensors_finite(values):
            raise FloatingPointError('nonfinite closed-loop state')
        self.valid_steps.add_(trace.valid.sum(0))
        self.squares.add_((magnitudes.square() * trace.valid[None]).sum(1))
        self.saturation.add_(((trace.actions.abs() >= 1.0-1e-6).to(magnitudes.dtype).mean(2)
                              * trace.valid).sum(0))
        features = weighted_task_features(trace, self.config, start=start, horizon=self.horizon).square()
        for i, section in enumerate((slice(0, 3), slice(3, 6), slice(6, 9),
                                     slice(9, 20), slice(20, 21), slice(21, 22))):
            self.components[i].add_(features[:, :, section].sum(dim=(0, 2)))
        for i, value in enumerate(warning_risk_steps(trace).values()):
            self.risks[i].add_(value.sum(0))

    def finish(self, costs, weights):
        count = self.valid_steps.sum().clamp_min(1)
        result = {name: float((self.squares[i].sum()/count).sqrt())
                  for i, name in enumerate(('position_rms', 'velocity_rms', 'omega_rms'))}
        result.update(task_objective=float((costs*weights).sum()), finite=True,
                      scenario_count=costs.numel(),
                      motor_saturation_fraction=float(self.saturation.sum()/count),
                      physical_transitions=int(self.valid_steps.sum()),
                      task_components={name: float((self.components[i]*weights).sum())
                                       for i, name in enumerate(('position','velocity','omega','regularization','dead','terminal'))},
                      omega_risk=float((self.risks[0]*risk_weights(self.risks[0], self.config)).sum()),
                      saturation_risk=float((self.risks[1]*risk_weights(self.risks[1], self.config)).sum()))
        return result
