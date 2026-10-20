"""Causal rollout; position, true attitude-transition and command-change costs."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Optional

import torch

from env_raptor import RaptorParams, RaptorSimulator, RaptorState, IMMUTABLE_TAPES
from response_noise import DisturbanceConfig, measured_observation
from response_policy import (
    ResponseMotorPolicy, ResponsePolicyState,
)


# Differentiable closed-loop state edges covered by per-step gradient decay.
PHYSICAL_DYNAMIC = ("position", "velocity", "orientation", "omega", "motor", "previous_action", "previous_velocity")
POLICY_DYNAMIC = ("memory",)
TASK_OBJECTIVE_VERSION = "pre-position-so3-transition-action-delta-equal-scenes-v2"
TASK_COMPONENTS = ("position", "attitude_delta", "action_delta", "dead", "terminal")


@dataclass(frozen=True)
class TaskLossConfig:
    # No guessed smoothing defaults: each experiment supplies fixed scales.
    epsilon_p: float
    epsilon_a: float
    lambda_R: float  # Fixed multiplier of 1-cos(relative rotation per control step).
    dead_cost: float = 3.0  # Raw cost per unflown step after failure.
    terminal_cost: float = 200.0  # Raw one-off cost, including failure at H.

    def __post_init__(self) -> None:
        for name in ("epsilon_p", "epsilon_a", "lambda_R"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if any(not math.isfinite(c) or c < 0 for c in (self.dead_cost, self.terminal_cost)):
            raise ValueError("failure costs must be finite and non-negative")


@dataclass(frozen=True)
class ResponseClosedLoopState:
    physical: RaptorState
    policy: ResponsePolicyState


@dataclass(frozen=True)
class TaskTrajectory:
    end: ResponseClosedLoopState
    observations: torch.Tensor | None  # EVAL/replay retain these; training can omit storage.
    actions: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor
    omegas: torch.Tensor
    action_deltas: torch.Tensor
    omega_deltas: torch.Tensor
    initial: RaptorState
    valid: torch.Tensor  # [time, scene], including the first terminal transition
    pre_positions: torch.Tensor  # True p_t, before the command's physical transition.
    pre_orientations: torch.Tensor  # True unit q_t=[w,x,y,z], not noisy observations.
    post_orientations: torch.Tensor  # True q_(t+1), including the terminal transition.


def observation(physical: RaptorState) -> torch.Tensor:
    return measured_observation(physical)


def initialize(policy: ResponseMotorPolicy, physical: RaptorState) -> ResponseClosedLoopState:
    obs = observation(physical)
    return ResponseClosedLoopState(physical, policy.initial_state(obs))


def _select_rows(state, indices: torch.Tensor):
    """Compact live rows without detaching their physical or recurrent graph."""
    if indices.numel() == getattr(state, fields(state)[0].name).shape[0]:
        return state
    return type(state)(**{f.name: (getattr(state, f.name) if f.name in IMMUTABLE_TAPES
                          else getattr(state, f.name).index_select(0, indices))
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
    simulator: RaptorSimulator,
    initial: RaptorState | ResponseClosedLoopState,
    steps: int,
    *,
    time_decay: float = 0.0,
    actor_probe=None,
    record_observations: bool = True,
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
    closed = initialize(policy, initial) if isinstance(initial, RaptorState) else initial
    if (closed.physical.noise_tape.shape[1] > 1
            and int(closed.physical.step_index.max()) + steps >= closed.physical.noise_tape.shape[1]):
        raise ValueError("noise tape too short: sample scenarios for the full requested horizon")
    initial_physical = closed.physical
    observations = [observation(closed.physical)] if record_observations else []
    actions, positions, velocities, omegas, action_deltas, omega_deltas = [], [], [], [], [], []
    valid = []
    pre_positions, pre_orientations, post_orientations = [], [], []
    # A terminal row stays frozen outside its boundary, so it cannot become live
    # again when this rollout is resumed at the next metrics-chunk boundary.
    indices = (~simulator.terminated(closed.physical)).nonzero(as_tuple=True)[0]
    live = ResponseClosedLoopState(_select_rows(closed.physical, indices),
                                   _select_rows(closed.policy, indices))
    for step in range(steps):
        current_observation = observations[-1] if record_observations else None
        if decay and indices.numel():
            # Full and compact tensors are parallel aliases, not serial gates.
            # Cover observation, physics, GRU/history and delta-cost paths once.
            closed = _decay_closed_state(closed, rho)
            live = _decay_closed_state(live, rho)
            current_observation = observation(closed.physical)
        elif indices.numel() and not record_observations:
            current_observation = observation(closed.physical)
        before = closed.physical
        pre_positions.append(before.position)
        pre_orientations.append(before.orientation)
        action = before.previous_action
        active = torch.zeros_like(before.step_index, dtype=torch.bool)
        if indices.numel():
            active[indices] = True  # The upcoming crossing transition counts.
            if actor_probe is not None:
                actor_probe.begin_step(indices)  # Training-only original-row metadata, not Actor input.
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
        post_orientations.append(physical.orientation)
        valid.append(active)
        actions.append(action)
        positions.append(physical.position)
        velocities.append(physical.velocity)
        omegas.append(physical.omega)
        action_deltas.append(action - before.previous_action)
        omega_deltas.append(physical.omega - before.omega)
        if record_observations:
            observations.append(observation(physical))
    return TaskTrajectory(
        closed, torch.stack(observations) if record_observations else None,
        torch.stack(actions), torch.stack(positions),
        torch.stack(velocities), torch.stack(omegas), torch.stack(action_deltas),
        torch.stack(omega_deltas), initial_physical, torch.stack(valid),
        torch.stack(pre_positions), torch.stack(pre_orientations), torch.stack(post_orientations),
    )


def smooth_l2(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Smooth L2 of a whole vector; rationalized to avoid small-error cancellation."""
    squared = value.square().sum(-1)
    # hypot computes sqrt(||x||^2+epsilon^2) without squaring epsilon into
    # underflow/overflow. vector_norm's zero VJP also keeps zero-error gradients finite.
    radius = torch.hypot(value.norm(dim=-1), value.new_full((), epsilon))
    return squared / (radius + epsilon)


def attitude_delta_cost(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """1-cos(theta) for true unit-quaternion endpoints, with no absolute target.

    Twice the squared vector part of conj(before)*after equals
    (3-trace(R_before.T@R_after))/2, or ||R_after-R_before||_F^2/4.
    Form it using after-before to avoid cancellation at tiny rotations and
    to make identical endpoints exactly zero. Quaternion signs are equivalent.
    No acos, division by dt, learnable metric or angular-velocity input is used.
    """
    w, x, y, z = before.unbind(-1)
    dw, dx, dy, dz = (after-before).unbind(-1)
    relative_vector = torch.stack((
        (w*dx-x*dw) + (z*dy-y*dz),
        (w*dy-y*dw) + (x*dz-z*dx),
        (w*dz-z*dw) + (y*dx-x*dy),
    ), -1)
    return 2*relative_vector.square().sum(-1)


def step_cost_components(
    trajectory: TaskTrajectory, config: TaskLossConfig, *, start: int = 0,
    horizon: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Direct additive costs, once per executed control, normalized by planned H.

    Position is pre-action truth; attitude uses true q_t and q_(t+1), not a
    world-upright target. Commands are final tanh outputs. Failure and attitude
    change include the last physical transition, including a crossing on H.
    Normal recovery also rotates: this is a soft motion cost, not instability.
    """
    steps = trajectory.valid.shape[0]
    horizon = steps if horizon is None else horizon
    if start < 0 or horizon < 1 or start + steps > horizon:
        raise ValueError("cost slice must lie inside the full horizon")
    valid = trajectory.valid
    # Mask BEFORE nonlinear math: frozen padding cannot create NaN*0 gradients.
    position, before_q, after_q, delta, post_position, action = (
        torch.where(valid[..., None], value, 0.)
        for value in (trajectory.pre_positions, trajectory.pre_orientations,
                      trajectory.post_orientations, trajectory.action_deltas,
                      trajectory.positions, trajectory.actions)
    )
    components = {
        "position": smooth_l2(position, config.epsilon_p) / horizon,
        "attitude_delta": (config.lambda_R/horizon) * attitude_delta_cost(before_q, after_q),
        "action_delta": smooth_l2(delta, config.epsilon_a) / horizon,
    }
    with torch.no_grad():
        state = trajectory.initial
        terminal = valid & RaptorSimulator.position_terminated(post_position, state.position_limit)
        remaining = horizon - torch.arange(start + 1, start + steps + 1,
                                            device=position.device, dtype=position.dtype)
        # Book all missing steps once at failure, not at each window end.
        # valid excludes frozen padding; reaching H without a violation costs zero.
        failed = terminal.to(position)
        components["dead"] = failed * remaining[:, None] * (config.dead_cost / horizon)
        components["terminal"] = failed * (config.terminal_cost / horizon)
    if not tensors_finite((position, before_q, after_q, delta, post_position, action, *components.values())):
        raise FloatingPointError("nonfinite valid task state or cost")
    return components


def scenario_costs(trajectory: TaskTrajectory, config: TaskLossConfig) -> torch.Tensor:
    return step_costs(trajectory, config).sum(0)


def step_costs(trajectory, config, *, start=0, horizon=None) -> torch.Tensor:
    """Additive [time, scene] costs; never square costs into residuals."""
    return sum(step_cost_components(trajectory, config, start=start, horizon=horizon).values())


def warning_risk_steps(trajectory) -> dict[str, torch.Tensor]:
    """Unchanged hover warning exposure, used only as a physical observation."""
    return {
        "omega": (trajectory.omegas.detach().norm(dim=-1) / 10.0 - 1).clamp_min(0).square()
                 * trajectory.valid,
        "saturation": ((trajectory.actions.detach().abs() - .95) / (1.0 - .95))
                      .clamp_min(0).square().mean(-1) * trajectory.valid,
    }


def _warning_risk_score(values):
    # Retain the old read-only warning report; these weights NEVER enter loss.
    count = max(1, math.ceil(.2 * values.numel()))
    return values.mean() + .5*values.topk(count).values.mean()


def hard_risk_metrics(trajectory) -> dict:
    risks = {name: value.sum(0) for name, value in warning_risk_steps(trajectory).items()}
    risks = {name: float(_warning_risk_score(value)) for name, value in risks.items()}
    position = torch.cat((trajectory.initial.position[None], trajectory.positions)).detach().norm(dim=-1)
    velocity = torch.cat((trajectory.initial.velocity[None], trajectory.velocities)).detach().norm(dim=-1)
    return {"hard_risk_components": risks,
            "hard_risk_peaks": {"omega": float(trajectory.omegas.detach().norm(dim=-1).max()),
                                "action_abs": float(trajectory.actions.detach().abs().max()),
                                "position": float(position.max()), "velocity": float(velocity.max())}}


def uniform_scene_weights(costs: torch.Tensor) -> torch.Tensor:
    """Exactly 1/N, independent of cost order, survival and failure constants."""
    if costs.ndim != 1 or costs.numel() == 0 or not bool(torch.isfinite(costs).all()):
        raise ValueError("uniform weights need finite nonempty per-scene costs")
    return torch.full_like(costs, 1.0 / costs.numel()).detach()


def task_loss(trajectory: TaskTrajectory, config: TaskLossConfig) -> torch.Tensor:
    return scenario_costs(trajectory, config).mean()


def sample_scenarios(
    count: int, *, seed: int, dt: float = .01,
    device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32,
    horizon: int = 500, disturbances: DisturbanceConfig = DisturbanceConfig(),
) -> RaptorState:
    return RaptorSimulator(RaptorParams(dt)).reset(
        count, seed=seed, horizon=horizon, device=device, dtype=dtype, disturbances=disturbances)


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
            if field.name in IMMUTABLE_TAPES:
                continue  # Original-pool tapes need not match a selected batch size.
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
    """Score first position-only termination with each airframe's own boundary.

    Every scene uses its own size-scaled RAPTOR boundary.
    The terminal transition is retained; subsequent entries are frozen padding.
    The first violation remains a failure even on the final allowed transition.
    """
    state = trajectory.initial
    if bool(RaptorSimulator.terminated(state).any()):
        raise ValueError("reference evaluation starts outside its termination set")
    name = "raptor"
    horizon = trajectory.positions.shape[0]
    terminated = RaptorSimulator.position_terminated(trajectory.positions, state.position_limit)
    steps = torch.arange(1, horizon+1, device=terminated.device)[:, None]
    lengths = torch.where(terminated, steps, horizon).amin(dim=0)
    result = {
        "reference_protocol": name,
        "episode_termination": "position-only",
        name + "_episode_length_mean": float(lengths.double().mean()),
        name + "_episode_length_std": float(lengths.double().std(unbiased=False)),
        name + "_share_terminated": float(terminated.any(0).double().mean()),
        "reference_position_limit_min_m": float(state.position_limit.min()),
        "reference_position_limit_max_m": float(state.position_limit.max()),
    }
    return result


def task_loss_components(trajectory, config, *, start=0, horizon=None):
    """Same five direct terms and same initial-scene mean as TRAIN and EVAL."""
    return {name: float(value.sum(0).mean().detach()) for name, value in
            step_cost_components(trajectory, config, start=start, horizon=horizon).items()}


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
        self.components = initial.position.new_zeros(len(TASK_COMPONENTS), initial.position.shape[0])
        self.risks = initial.position.new_zeros(2, initial.position.shape[0])

    @torch.no_grad()
    def add(self, trace, start, *, components=None):
        magnitudes = torch.stack([x.norm(dim=-1) for x in
                                  (trace.positions, trace.velocities, trace.omegas)])
        values = [magnitudes, trace.actions]
        values += [getattr(s, f.name) for s in (trace.end.physical, trace.end.policy) for f in fields(s)
                   if f.name not in IMMUTABLE_TAPES]
        if not tensors_finite(values):
            raise FloatingPointError('nonfinite closed-loop state')
        self.valid_steps.add_(trace.valid.sum(0))
        self.squares.add_((magnitudes.square() * trace.valid[None]).sum(1))
        self.saturation.add_(((trace.actions.abs() >= 1.0-1e-6).to(magnitudes.dtype).mean(2)
                              * trace.valid).sum(0))
        if components is None:
            components = step_cost_components(trace, self.config, start=start, horizon=self.horizon)
        for i, name in enumerate(TASK_COMPONENTS):
            self.components[i].add_(components[name].sum(0))
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
                                       for i, name in enumerate(TASK_COMPONENTS)},
                      omega_risk=float(_warning_risk_score(self.risks[0])),
                      saturation_risk=float(_warning_risk_score(self.risks[1])))
        return result
