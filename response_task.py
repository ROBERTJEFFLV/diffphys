"""Teacher-free causal rollout, physical task objective, and scenario banks."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping, Optional

import torch
from torch.nn.utils.stateless import functional_call

from env_l2f import L2FParams, L2FSimulator, L2FState
from response_policy import (
    ResponseMotorPolicy, ResponsePolicyState, body_vector, measured_response,
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
    prediction_weight: float = 0.01
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
class RiskConfig:
    """Fixed hover operating limits, independent of task precision weights.

    These are configurable engineering starting points, not certified limits.
    Risk is an undiscounted sum of per-transition dimensionless barriers.
    """
    position_limit: float = 5.0
    velocity_limit: float = 5.0
    omega_limit: float = 10.0
    saturation_limit: float = 0.95
    sharpness: float = 10.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(getattr(self, f.name)) or getattr(self, f.name) <= 0
               for f in fields(self)):
            raise ValueError("risk limits and sharpness must be finite and positive")
        if self.saturation_limit > 1:
            raise ValueError("risk saturation limit must be at most normalized full command 1")


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


def policy_step(policy, obs, state, parameters=None, **kwargs):
    if parameters is None:
        return policy(obs, state, **kwargs)
    return functional_call(policy, parameters, (obs, state), kwargs)


def rollout(
    policy: ResponseMotorPolicy,
    simulator: L2FSimulator,
    initial: L2FState | ResponseClosedLoopState,
    steps: int,
    *,
    parameters: Optional[Mapping[str, torch.Tensor]] = None,
    memory_enabled: bool = True,
) -> TaskTrajectory:
    if steps < 1:
        raise ValueError("rollout must contain physical transitions")
    if abs(simulator.params.dt - policy.config.dt) > 1.0e-12:
        raise ValueError("training and deployment dt must agree")
    closed = initialize(policy, initial) if isinstance(initial, L2FState) else initial
    observations = [observation(closed.physical, closed.policy.integral)]
    actions, positions, velocities, omegas, action_deltas, omega_deltas = [], [], [], [], [], []
    for _ in range(steps):
        output = policy_step(
            policy, observations[-1], closed.policy, parameters,
            memory_enabled=memory_enabled,
        )
        before = closed.physical
        # The same action is executed and recorded, and the policy state is
        # advanced once. No truth reset, detached burn-in, or hidden-state splice.
        physical = simulator.step(before, output.action, grad_decay=1.0)
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


def concatenate(parts: list[TaskTrajectory]) -> TaskTrajectory:
    if not parts:
        raise ValueError("empty trajectory list")
    names = ("actions", "positions", "velocities", "omegas", "action_deltas", "omega_deltas")
    values = {name: torch.cat([getattr(p, name) for p in parts]) for name in names}
    observations = torch.cat([parts[0].observations] + [p.observations[1:] for p in parts[1:]])
    return TaskTrajectory(parts[-1].end, observations=observations, **values)


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


def risk_components(trajectory, config: RiskConfig, *, position_reference=0.,
                    velocity_reference=0., omega_reference=0.) -> dict[str, torch.Tensor]:
    """[time, scene] barriers on task errors and normalized command saturation.

    Hover references are zero. A future tracking task must supply its references
    consistently in collection, continuation, Actor loss and acceptance.
    """
    def barrier(magnitude, limit):
        return torch.nn.functional.softplus(config.sharpness * (magnitude / limit - 1)) / config.sharpness

    return {
        "position": barrier((trajectory.positions - position_reference).norm(dim=-1), config.position_limit),
        "velocity": barrier((trajectory.velocities - velocity_reference).norm(dim=-1), config.velocity_limit),
        "omega": barrier((trajectory.omegas - omega_reference).norm(dim=-1), config.omega_limit),
        "saturation": barrier(trajectory.actions.abs(), config.saturation_limit).mean(-1),
    }


def step_risks(trajectory, config: RiskConfig, **references) -> torch.Tensor:
    return torch.stack(tuple(risk_components(trajectory, config, **references).values())).sum(0)


def suffix_risks(risks: torch.Tensor) -> torch.Tensor:
    """Exact remaining risk at Z_0 ... Z_H; b_t belongs to Z_t -> Z_{t+1}."""
    return torch.cat((risks.flip(0).cumsum(0).flip(0), torch.zeros_like(risks[:1])))


def physical_risk_metrics(trajectory, config: RiskConfig, loss_config: TaskLossConfig) -> dict:
    """Real flight risk, with independent full-flight mean+CVaR per component."""
    components = {name: value.detach().sum(0) for name, value in risk_components(trajectory, config).items()}
    total = torch.stack(tuple(components.values())).sum(0)

    def aggregate(value):
        count = max(1, math.ceil(loss_config.tail_fraction * value.numel()))
        return float(value.mean() + loss_config.tail_weight * value.topk(count).values.mean())

    return {"risk_objective": aggregate(total),
            "risk_components": {name: aggregate(value) for name, value in components.items()}}


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


def prediction_residual(policy, observations, actions, *, parameters=None) -> torch.Tensor:
    """Fixed self-collected response supervision, with NO actor-label target.

    Detaching observations/actions here prevents the actor making the prediction
    problem artificially easy. Memory remains differentiable through the record.
    The separate task rollout keeps the complete action/physics/memory gradient.
    """
    observations, actions = observations.detach(), actions.detach()
    if observations.shape[0] != actions.shape[0] + 1:
        raise ValueError("H executed actions require H+1 observation rows")
    state = policy.initial_state(observations[0])
    errors = []
    for step in range(actions.shape[0]):
        output = policy_step(
            policy, observations[step], state, parameters,
            applied_action=actions[step],
        )
        target = measured_response(observations[step], observations[step + 1], policy.config.dt)
        errors.append(output.response_prediction - target)
        state = output.next_state
    return torch.stack(errors).reshape(-1) / math.sqrt(actions.shape[0] * actions.shape[1] * 6)


def sample_scenarios(
    count: int, *, seed: int, dt: float = 0.01,
    device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32,
) -> tuple[L2FState, torch.Tensor]:
    """Independent 4x4 physical-fit bank, without importing distillation code."""
    if count < 16 or count % 16:
        raise ValueError("scenario count must be a positive multiple of 16")
    simulator = L2FSimulator(L2FParams(dt=dt))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        pool = simulator.reset(
            max(4096, count * 128), device=torch.device("cpu"), dtype=torch.float32,
            sample_dynamics=True, sampled_dynamics_level="broad",
            broad_sampler="physical-fit", balanced_dynamics_sampling=False,
            sample_external_force=True,
        )
    tw_edges = torch.linspace(1.45, 5.50, 5)
    alpha_edges = torch.logspace(math.log10(35.0), math.log10(2200.0), 5)
    tw = torch.bucketize(pool.thrust_to_weight, tw_edges[1:-1])
    alpha = torch.bucketize(pool.alpha_roll_max, alpha_edges[1:-1])
    selections, cells = [], []
    for i in range(4):
        for j in range(4):
            choices = torch.nonzero((tw == i) & (alpha == j), as_tuple=False).flatten()
            if choices.numel() < count // 16:
                raise RuntimeError("physical-fit sampler did not fill the registered strata")
            selections.append(choices[:count // 16])
            cells.extend([(i, j)] * (count // 16))
    indices = torch.cat(selections)
    state = L2FState(**{
        field.name: getattr(pool, field.name).index_select(0, indices).to(device=device, dtype=dtype)
        for field in fields(pool)
    })
    return state, torch.tensor(cells, dtype=torch.long, device=device)


def trajectory_metrics(trajectory: TaskTrajectory, config: TaskLossConfig) -> dict:
    """Minimal control metrics; keep every scene, not a tail/JVP audit suite."""
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
    rows = [{
        "scenario": i, "task_cost": float(costs[i]), "finite": bool(finite[i]),
        "success": bool(success[i] and finite[i]),
        "position_rms": float(p[:, i].square().mean().sqrt()),
        "velocity_rms": float(v[:, i].square().mean().sqrt()),
        "omega_rms": float(w[:, i].square().mean().sqrt()),
        "motor_saturation_fraction": float(saturation[i]),
    } for i in range(costs.numel())]
    return {
        "task_objective": float(task_loss(trajectory, config).detach()),
        "position_rms": float(p.square().mean().sqrt()),
        "velocity_rms": float(v.square().mean().sqrt()),
        "omega_rms": float(w.square().mean().sqrt()),
        "steady_success_rate": float((success & finite).to(p.dtype).mean()),
        "motor_saturation_fraction": float(saturation.mean()),
        "finite": bool(finite.all()), "success_count": sum(r["success"] for r in rows),
        "scenario_count": len(rows), "scenarios": rows,
    }
