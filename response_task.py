"""Teacher-free causal rollout, physical task objective, and scenario banks."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Optional

import torch

from env_l2f import L2FParams, L2FSimulator, L2FState
from response_policy import (
    ResponseMotorPolicy, ResponsePolicyState, body_vector,
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


def observation(physical: L2FState, integral: torch.Tensor) -> torch.Tensor:
    """World p/v, full R, body omega/integral, executed previous action."""
    return torch.cat((
        physical.position, physical.velocity, physical.rotation.flatten(1),
        physical.omega, body_vector(physical.rotation, integral),
        physical.previous_action,
    ), -1)


def initialize(policy: ResponseMotorPolicy, physical: L2FState) -> ResponseClosedLoopState:
    obs = observation(physical, torch.zeros_like(physical.position))
    return ResponseClosedLoopState(physical, policy.initial_state(obs))


def rollout(
    policy: ResponseMotorPolicy,
    simulator: L2FSimulator,
    initial: L2FState | ResponseClosedLoopState,
    steps: int,
) -> TaskTrajectory:
    if steps < 1:
        raise ValueError("rollout must contain physical transitions")
    if abs(simulator.params.dt - policy.config.dt) > 1.0e-12:
        raise ValueError("training and deployment dt must agree")
    closed = initialize(policy, initial) if isinstance(initial, L2FState) else initial
    observations = [observation(closed.physical, closed.policy.integral)]
    actions, positions, velocities, omegas, action_deltas, omega_deltas = [], [], [], [], [], []
    for step in range(steps):
        output = policy(observations[-1], closed.policy)
        before = closed.physical
        # The same action is executed and recorded, and the policy state is
        # advanced once. No truth reset, detached burn-in, or hidden-state splice.
        physical = simulator.step(before, output.action)
        closed = ResponseClosedLoopState(physical, output.next_state)
        actions.append(output.action)
        positions.append(physical.position)
        velocities.append(physical.velocity)
        omegas.append(physical.omega)
        action_deltas.append(output.action - before.previous_action)
        omega_deltas.append(physical.omega - before.omega)
        observations.append(observation(physical, closed.policy.integral))
    return TaskTrajectory(
        closed, torch.stack(observations), torch.stack(actions), torch.stack(positions),
        torch.stack(velocities), torch.stack(omegas), torch.stack(action_deltas),
        torch.stack(omega_deltas),
    )


def weighted_task_features(
    trajectory: TaskTrajectory, config: TaskLossConfig, *, start: int = 0,
    horizon: Optional[int] = None,
) -> torch.Tensor:
    """Signed sqrt(2 Huber) residuals, with the original small-error curvature.

    Robustify physical errors before applying their weights. Global time indices
    retain the full-flight mean and final steady window when used on short slices.
    Delta=0 is available only to reproduce historical quadratic objectives.
    """
    def robust(value):
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
    return features * time_weights.sqrt()[:, None, None]


def scenario_costs(trajectory: TaskTrajectory, config: TaskLossConfig) -> torch.Tensor:
    return step_costs(trajectory, config).sum(0)


def step_costs(trajectory, config, *, start=0, horizon=None) -> torch.Tensor:
    """Additive [time, scenario] costs, without mean/CVaR scenario weights."""
    return weighted_task_features(trajectory, config, start=start, horizon=horizon).square().sum(-1)


def warning_risk_steps(trajectory) -> dict[str, torch.Tensor]:
    """Unchanged hover warning exposure, used only as a physical observation."""
    return {
        "omega": (trajectory.omegas.detach().norm(dim=-1) / 10.0 - 1).clamp_min(0).square(),
        "saturation": ((trajectory.actions.detach().abs() - .95) / (1.0 - .95))
                      .clamp_min(0).square().mean(-1),
    }


def hard_risk_metrics(trajectory, loss_config: TaskLossConfig) -> dict:
    risks = {name: value.sum(0) for name, value in warning_risk_steps(trajectory).items()}
    count = max(1, math.ceil(loss_config.tail_fraction * trajectory.actions.shape[1]))
    risks = {name: float(value.mean() + loss_config.tail_weight * value.topk(count).values.mean())
             for name, value in risks.items()}
    position = torch.cat((trajectory.observations[:1, ..., :3], trajectory.positions)).detach().norm(dim=-1)
    velocity = torch.cat((trajectory.observations[:1, ..., 3:6], trajectory.velocities)).detach().norm(dim=-1)
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
    scenario_mode: str = "physical-fit",
) -> L2FState:
    """Independent physical-fit bank or nominal airframe with random kinematics."""
    if scenario_mode not in ("physical-fit", "fixed-airframe"):
        raise ValueError("unknown response scenario mode")
    if count < 16 or count % 16:
        raise ValueError("scenario count must be a positive multiple of 16")
    simulator = L2FSimulator(L2FParams(dt=dt))
    if scenario_mode == "fixed-airframe":
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            fixed = simulator.reset(count, device="cpu", dtype=torch.float32,
                                    sample_dynamics=False, sample_external_force=False)
        state = L2FState(**{f.name: getattr(fixed, f.name).to(device=device, dtype=dtype)
                           for f in fields(fixed)})
        return state
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        pool = simulator.reset(
            max(4096, count * 128), device=torch.device("cpu"), dtype=torch.float32,
            sample_dynamics=True,
            sample_external_force=True,
        )
    tw_edges = torch.linspace(1.45, 5.50, 5)
    alpha_edges = torch.logspace(math.log10(35.0), math.log10(2200.0), 5)
    tw = torch.bucketize(pool.thrust_to_weight, tw_edges[1:-1])
    alpha = torch.bucketize(pool.alpha_roll_max, alpha_edges[1:-1])
    selections = []
    for i in range(4):
        for j in range(4):
            choices = torch.nonzero((tw == i) & (alpha == j), as_tuple=False).flatten()
            if choices.numel() < count // 16:
                raise RuntimeError("physical-fit sampler did not fill the registered strata")
            selections.append(choices[:count // 16])
    indices = torch.cat(selections)
    state = L2FState(**{
        field.name: getattr(pool, field.name).index_select(0, indices).to(device=device, dtype=dtype)
        for field in fields(pool)
    })
    return state


def trajectory_metrics(trajectory: TaskTrajectory, config: TaskLossConfig) -> dict:
    """Aggregate control metrics over every scene, with the original finite mask."""
    costs = scenario_costs(trajectory, config).detach()
    p, v, w = [x.detach().norm(dim=-1) for x in (
        trajectory.positions, trajectory.velocities, trajectory.omegas
    )]
    tail = min(config.steady_steps, p.shape[0])
    finite = torch.stack([
        torch.isfinite(x).flatten(2).all(-1)
        for x in (trajectory.positions, trajectory.velocities, trajectory.omegas, trajectory.actions)
    ]).all(dim=(0, 1))
    if hasattr(trajectory, "end") and trajectory.end is not None:
        for field in fields(trajectory.end.physical):
            value = getattr(trajectory.end.physical, field.name)
            finite = finite & torch.isfinite(value).reshape(value.shape[0], -1).all(-1)
        finite = finite & torch.isfinite(trajectory.end.policy.memory).all(-1)
    success = ((p[-tail:] < 0.05) & (v[-tail:] < 0.10) & (w[-tail:] < 0.50)).all(0)
    saturation = (trajectory.actions.detach().abs() >= 1.0 - 1.0e-6).to(p.dtype).mean(dim=(0, 2))
    return {
        "task_objective": float(task_loss(trajectory, config).detach()),
        "position_rms": float(p.square().mean().sqrt()),
        "velocity_rms": float(v.square().mean().sqrt()),
        "omega_rms": float(w.square().mean().sqrt()),
        "steady_success_rate": float((success & finite).to(p.dtype).mean()),
        "motor_saturation_fraction": float(saturation.mean()),
        "finite": bool(finite.all()), "success_count": int((success & finite).sum()),
        "scenario_count": costs.numel(),
    }


def task_loss_components(trajectory, config, *, start=0, horizon=None, weights=None):
    """Additive attribution of the existing task objective, including its CVaR."""
    features = weighted_task_features(trajectory, config, start=start, horizon=horizon).square()
    if weights is None:
        weights = risk_weights(features.sum(dim=(0, 2)), config)
    return {name: float((weights * features[:, :, section].sum(dim=(0, 2))).sum().detach())
            for name, section in (('position', slice(0, 3)), ('velocity', slice(3, 6)),
                                  ('omega', slice(6, 9)), ('regularization', slice(9, None)))}


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
        self.success = torch.ones_like(self.saturation, dtype=torch.bool)
        self.components = initial.position.new_zeros(4, initial.position.shape[0])
        self.risks = initial.position.new_zeros(2, initial.position.shape[0])

    @torch.no_grad()
    def add(self, trace, start):
        magnitudes = torch.stack([x.norm(dim=-1) for x in
                                  (trace.positions, trace.velocities, trace.omegas)])
        values = [magnitudes, trace.actions]
        values += [getattr(s, f.name) for s in (trace.end.physical, trace.end.policy) for f in fields(s)]
        if not tensors_finite(values):
            raise FloatingPointError('nonfinite continuous closed-loop state')
        self.squares.add_(magnitudes.square().sum(1))
        self.saturation.add_((trace.actions.abs() >= 1.0-1e-6).to(magnitudes.dtype).mean(2).sum(0))
        tail_start = max(0, self.horizon-min(self.config.steady_steps, self.horizon)-start)
        p, v, w = magnitudes[:, tail_start:]
        self.success &= ((p < .05) & (v < .1) & (w < .5)).all(0)
        features = weighted_task_features(trace, self.config, start=start, horizon=self.horizon).square()
        for i, section in enumerate((slice(0, 3), slice(3, 6), slice(6, 9), slice(9, None))):
            self.components[i].add_(features[:, :, section].sum(dim=(0, 2)))
        for i, value in enumerate(warning_risk_steps(trace).values()):
            self.risks[i].add_(value.sum(0))

    def finish(self, costs, weights):
        result = {name: float((self.squares[i].mean()/self.horizon).sqrt())
                  for i, name in enumerate(('position_rms', 'velocity_rms', 'omega_rms'))}
        result.update(task_objective=float((costs*weights).sum()), finite=True,
                      steady_success_rate=float(self.success.float().mean()),
                      success_count=int(self.success.sum()), scenario_count=costs.numel(),
                      motor_saturation_fraction=float(self.saturation.mean()/self.horizon),
                      task_components={name: float((self.components[i]*weights).sum())
                                       for i, name in enumerate(('position','velocity','omega','regularization'))},
                      omega_risk=float((self.risks[0]*risk_weights(self.risks[0], self.config)).sum()),
                      saturation_risk=float((self.risks[1]*risk_weights(self.risks[1], self.config)).sum()))
        return result
