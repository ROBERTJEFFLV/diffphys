from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from env_l2f import (
    L2FLossConfig,
    L2FSimulator,
    L2FState,
    apply_gradient_decay,
)
from l2f_cuda_backend import cuda_step
from policy_observation import (
    PolicyObservationState,
    build_policy_observation,
    update_position_integral,
)
from training_objectives import independent_cvar_tail_loss


DYNAMIC_STATE_FIELDS = (
    "position",
    "velocity",
    "rotation",
    "omega",
    "motor",
    "previous_action",
)
CONTINUITY_FIELDS = (
    "position",
    "velocity",
    "orientation",
    "omega",
    "motor",
    "previous_action",
    "hidden",
    "integral",
)
SHOOTING_SCALES = {
    "position": 0.10,
    "velocity": 0.10,
    "orientation": 0.10,
    "omega": 0.50,
    "motor": 0.10,
    "previous_action": 0.10,
    "hidden": 0.10,
    "integral": 0.10,
}


@dataclass
class RecurrentSystemState:
    state: L2FState
    hidden: torch.Tensor
    integral: torch.Tensor


@dataclass
class SegmentResult:
    end: RecurrentSystemState
    task_loss: torch.Tensor
    position_history: torch.Tensor
    omega_history: torch.Tensor
    first_action: torch.Tensor
    components: dict[str, torch.Tensor]


def clone_recurrent(value: RecurrentSystemState) -> RecurrentSystemState:
    return RecurrentSystemState(
        state=L2FState(
            **{
                field.name: getattr(value.state, field.name).detach().clone()
                for field in fields(L2FState)
            }
        ),
        hidden=value.hidden.detach().clone(),
        integral=value.integral.detach().clone(),
    )


def detach_recurrent(value: RecurrentSystemState) -> RecurrentSystemState:
    return clone_recurrent(value)


def recurrent_tensors(value: RecurrentSystemState) -> tuple[torch.Tensor, ...]:
    return (
        value.state.position,
        value.state.velocity,
        value.state.rotation,
        value.state.omega,
        value.state.motor,
        value.state.previous_action,
        value.hidden,
        value.integral,
    )


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map from axis-angle vectors to SO(3)."""

    if rotation_vector.shape[-1] != 3:
        raise ValueError("SO(3) exponential input must end in dimension 3")
    theta_square = rotation_vector.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_square.clamp_min(1.0e-30))
    small = theta_square < 1.0e-8
    a = torch.where(
        small,
        1.0 - theta_square / 6.0 + theta_square.square() / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta_square / 24.0 + theta_square.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta_square.clamp_min(1.0e-30),
    )
    skew = _skew(rotation_vector)
    identity = torch.eye(
        3, device=rotation_vector.device, dtype=rotation_vector.dtype
    ).expand(*rotation_vector.shape[:-1], 3, 3)
    return identity + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Differentiable local logarithm map used by orientation continuity."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError("SO(3) logarithm input must end in shape [3,3]")
    antisymmetric = 0.5 * torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    epsilon = torch.finfo(rotation.dtype).eps
    sine = torch.sqrt(
        antisymmetric.square().sum(dim=-1, keepdim=True) + epsilon * epsilon
    )
    cosine = (
        rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) - 1.0
    ) * 0.5
    theta = torch.atan2(sine, cosine.clamp(-1.0, 1.0))
    factor = theta / sine
    return factor * antisymmetric


class ShootingBoundary(nn.Module):
    """One complete H250 boundary state in normalized local coordinates."""

    def __init__(self, base: RecurrentSystemState) -> None:
        super().__init__()
        for name in (
            "position",
            "velocity",
            "rotation",
            "omega",
            "motor",
            "previous_action",
        ):
            self.register_buffer(f"base_{name}", getattr(base.state, name).detach().clone())
        self.register_buffer("base_hidden", base.hidden.detach().clone())
        self.register_buffer("base_integral", base.integral.detach().clone())
        self.position = nn.Parameter(torch.zeros_like(base.state.position))
        self.velocity = nn.Parameter(torch.zeros_like(base.state.velocity))
        self.orientation = nn.Parameter(torch.zeros_like(base.state.omega))
        self.omega = nn.Parameter(torch.zeros_like(base.state.omega))
        self.motor = nn.Parameter(torch.zeros_like(base.state.motor))
        self.previous_action = nn.Parameter(torch.zeros_like(base.state.previous_action))
        self.hidden = nn.Parameter(torch.zeros_like(base.hidden))
        self.integral = nn.Parameter(torch.zeros_like(base.integral))

    def materialize(self, static_template: L2FState) -> RecurrentSystemState:
        dynamic: dict[str, torch.Tensor] = {
            "position": self.base_position + SHOOTING_SCALES["position"] * self.position,
            "velocity": self.base_velocity + SHOOTING_SCALES["velocity"] * self.velocity,
            "rotation": self.base_rotation
            @ so3_exp(SHOOTING_SCALES["orientation"] * self.orientation),
            "omega": self.base_omega + SHOOTING_SCALES["omega"] * self.omega,
            "motor": self.base_motor + SHOOTING_SCALES["motor"] * self.motor,
            "previous_action": self.base_previous_action
            + SHOOTING_SCALES["previous_action"] * self.previous_action,
        }
        state_values = {
            field.name: (
                dynamic[field.name]
                if field.name in dynamic
                else getattr(static_template, field.name)
            )
            for field in fields(L2FState)
        }
        return RecurrentSystemState(
            state=L2FState(**state_values),
            hidden=self.base_hidden + SHOOTING_SCALES["hidden"] * self.hidden,
            integral=self.base_integral + SHOOTING_SCALES["integral"] * self.integral,
        )


def continuity_residuals(
    predicted: RecurrentSystemState,
    shooting: RecurrentSystemState,
) -> dict[str, torch.Tensor]:
    relative_rotation = predicted.state.rotation.transpose(-1, -2) @ shooting.state.rotation
    return {
        "position": (shooting.state.position - predicted.state.position)
        / SHOOTING_SCALES["position"],
        "velocity": (shooting.state.velocity - predicted.state.velocity)
        / SHOOTING_SCALES["velocity"],
        "orientation": so3_log(relative_rotation) / SHOOTING_SCALES["orientation"],
        "omega": (shooting.state.omega - predicted.state.omega)
        / SHOOTING_SCALES["omega"],
        "motor": (shooting.state.motor - predicted.state.motor)
        / SHOOTING_SCALES["motor"],
        "previous_action": (
            shooting.state.previous_action - predicted.state.previous_action
        )
        / SHOOTING_SCALES["previous_action"],
        "hidden": (shooting.hidden - predicted.hidden) / SHOOTING_SCALES["hidden"],
        "integral": (shooting.integral - predicted.integral)
        / SHOOTING_SCALES["integral"],
    }


def continuity_rms(residuals: dict[str, torch.Tensor]) -> torch.Tensor:
    block_mean_squares = torch.stack(
        tuple(residuals[name].square().mean() for name in CONTINUITY_FIELDS)
    )
    epsilon = torch.finfo(block_mean_squares.dtype).eps
    return torch.sqrt(block_mean_squares.mean() + epsilon * epsilon) - epsilon


def continuity_max_abs(residuals: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(tuple(residuals[name].abs().max() for name in CONTINUITY_FIELDS)).max()


def zero_duals(
    residuals: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    return [
        {name: torch.zeros_like(value) for name, value in blocks.items()}
        for blocks in residuals
    ]


def augmented_lagrangian_terms(
    residuals: list[dict[str, torch.Tensor]],
    duals: list[dict[str, torch.Tensor]],
    *,
    rho: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if len(residuals) != len(duals):
        raise ValueError("continuity residual and dual counts must match")
    boundary_terms: list[torch.Tensor] = []
    for blocks, multipliers in zip(residuals, duals):
        field_terms = [
            (multipliers[name] * blocks[name] + 0.5 * float(rho) * blocks[name].square()).mean()
            for name in CONTINUITY_FIELDS
        ]
        boundary_terms.append(torch.stack(field_terms).mean())
    if not boundary_terms:
        raise ValueError("multiple shooting requires at least one internal boundary")
    return torch.stack(boundary_terms).sum(), boundary_terms


@torch.no_grad()
def update_duals_(
    duals: list[dict[str, torch.Tensor]],
    residuals: list[dict[str, torch.Tensor]],
    *,
    rho: float,
) -> None:
    for multipliers, blocks in zip(duals, residuals):
        for name in CONTINUITY_FIELDS:
            multipliers[name].add_(float(rho) * blocks[name].detach())


def _masked_smooth_l1_sum(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    per_sample = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    return (per_sample * eligible.to(dtype=per_sample.dtype)).sum()


def _omega_decay_loss(
    omega_history: torch.Tensor,
    sample_mask: torch.Tensor,
    args: Any,
) -> torch.Tensor:
    # Exact Python-3.8-compatible form of training_objectives.multistep_omega_decay_loss.
    weighted: list[torch.Tensor] = []
    for horizon, weight, target in zip(
        args.omega_decay_horizons,
        args.omega_decay_beta,
        args.omega_decay_rho,
    ):
        start_norm = torch.linalg.vector_norm(omega_history[:-horizon], dim=-1)
        future_norm = torch.linalg.vector_norm(omega_history[horizon:], dim=-1)
        active = sample_mask.unsqueeze(0) & (
            start_norm.detach() >= float(args.success_omega)
        )
        ratio = future_norm / (start_norm.detach() + float(args.omega_decay_eps))
        penalty = F.relu(ratio - float(target)).square()
        loss = penalty[active].mean() if bool(active.any().item()) else future_norm.sum() * 0.0
        weighted.append(float(weight) * loss)
    return torch.stack(weighted).sum()


def rollout_q2_segment(
    policy: nn.Module,
    sim: L2FSimulator,
    initial: RecurrentSystemState,
    *,
    train_args: Any,
    loss_config: L2FLossConfig,
    state_step_decay: float,
    hidden_step_decay: float,
    retain_mask: torch.Tensor,
    segment_index: int,
    segment_steps: int,
    segment_count: int,
    motor_aux_weight: float,
    backend: str,
) -> SegmentResult:
    if backend not in ("cuda", "torch"):
        raise ValueError("multiple-shooting segment backend must be cuda or torch")
    state = initial.state
    hidden = initial.hidden
    observation_state = PolicyObservationState(initial.integral)
    zero = state.position.sum() * 0.0
    tracking_sum = zero
    clf_sum = zero
    outward_sum = zero
    du_sum = zero
    ddu_sum = zero
    sat_sum = zero
    motor_aux_sum = zero
    previous_potential = sim.tracking_potential(state, loss_config)
    previous_action_delta: torch.Tensor | None = None
    tail_potentials: list[torch.Tensor] = []
    position_history: list[torch.Tensor] = []
    omega_history: list[torch.Tensor] = [state.omega]
    first_action: torch.Tensor | None = None
    decay_sample_mask = retain_mask & (
        (state.alpha_roll_max <= train_args.omega_decay_alpha_roll_max)
        | (state.alpha_yaw_max <= train_args.omega_decay_alpha_yaw_max)
    )

    for local_step in range(segment_steps):
        previous_action = state.previous_action
        motor_target = state.motor.detach()
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode=train_args.observation_mode,
            noise_max=train_args.observation_noise_max,
            integral_input_frame=train_args.integral_input_frame,
            integral_input_multiplier=train_args.integral_input_multiplier,
        )
        action, hidden, auxiliary = policy.forward_with_aux(observation, hidden)
        if first_action is None:
            first_action = action
        hidden = apply_gradient_decay(hidden, hidden_step_decay)
        observation_state = update_position_integral(
            observation_state,
            observed_position,
            dt=train_args.dt,
            integral_limit=train_args.integral_limit,
            integral_leak=train_args.integral_leak,
        )
        sample_step = segment_index * segment_steps + local_step
        if sample_step >= train_args.motor_aux_burn_in:
            eligible = torch.ones(
                state.position.shape[0], device=state.position.device, dtype=torch.bool
            )
            motor_aux_sum = motor_aux_sum + _masked_smooth_l1_sum(
                auxiliary["motor_state"], motor_target, eligible
            )
        action_delta = action - previous_action
        state = (
            cuda_step(state, action, sim.params, grad_decay=state_step_decay)
            if backend == "cuda"
            else sim.step(state, action, grad_decay=state_step_decay)
        )
        position_history.append(state.position)
        omega_history.append(state.omega)
        tracking_components = sim.tracking_components(state, loss_config)
        potential = sum(tracking_components.values())
        tracking_sum = tracking_sum + potential.mean()
        clf_target = (1.0 - train_args.clf_kappa * train_args.dt) * previous_potential.detach()
        clf_sum = clf_sum + F.relu(potential - clf_target).square().mean()
        outward_sum = outward_sum + sim.outward_velocity_loss(state, loss_config)
        du_sum = du_sum + action_delta.square().mean()
        sat_sum = sat_sum + F.relu(action.abs() - train_args.u_soft).square().mean()
        if previous_action_delta is not None:
            ddu_sum = ddu_sum + (action_delta - previous_action_delta).square().mean()
        tail_potentials.append(potential)
        previous_potential = potential
        previous_action_delta = action_delta

    if first_action is None:
        raise RuntimeError("segment rollout produced no action")
    horizon = float(segment_steps)
    tracking_loss = tracking_sum / horizon
    clf_loss = clf_sum / horizon
    outward_loss = outward_sum / horizon
    du_loss = du_sum / horizon
    ddu_loss = ddu_sum / max(segment_steps - 1, 1)
    sat_loss = sat_sum / horizon
    tail_count = min(max(train_args.tail_steps, 1), len(tail_potentials))
    tail_loss = torch.stack(tail_potentials[-tail_count:]).mean()
    omega_history_tensor = torch.stack(omega_history, dim=0)
    omega_decay_loss = _omega_decay_loss(omega_history_tensor, decay_sample_mask, train_args)
    total_episode_steps = segment_steps * segment_count
    motor_count = state.position.shape[0] * max(
        total_episode_steps - train_args.motor_aux_burn_in, 0
    )
    motor_objective = (
        motor_aux_sum * float(segment_count) / float(motor_count)
        if motor_count > 0
        else motor_aux_sum * 0.0
    )
    task_loss = (
        tracking_loss
        + train_args.lambda_clf * clf_loss
        + train_args.lambda_out * outward_loss
        + train_args.lambda_tail * tail_loss
        + train_args.lambda_du * du_loss
        + train_args.lambda_ddu * ddu_loss
        + train_args.lambda_sat * sat_loss
        + float(motor_aux_weight) * motor_objective
        + train_args.w_omega_decay * omega_decay_loss
    )
    return SegmentResult(
        end=RecurrentSystemState(
            state=state,
            hidden=hidden,
            integral=observation_state.integral_position,
        ),
        task_loss=task_loss,
        position_history=torch.stack(position_history, dim=0),
        omega_history=omega_history_tensor[1:],
        first_action=first_action,
        components={
            "tracking": tracking_loss,
            "clf": clf_loss,
            "outward": outward_loss,
            "tail": tail_loss,
            "du": du_loss,
            "ddu": ddu_loss,
            "sat": sat_loss,
            "motor_aux": motor_objective,
            "omega_decay": omega_decay_loss,
        },
    )


def q2_h1000_task_loss(
    segments: list[SegmentResult],
    train_args: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(segments) != 4:
        raise ValueError("the minimal H1000 experiment requires exactly four H250 segments")
    dense = torch.stack(tuple(segment.task_loss for segment in segments)).mean()
    early = independent_cvar_tail_loss(
        segments[0].position_history,
        segments[0].omega_history,
        position_threshold=train_args.success_position_m,
        omega_threshold=train_args.success_omega,
        q_position=train_args.q_position,
        q_omega=train_args.q_omega,
        w_position_cvar=train_args.w_position_cvar,
        w_omega_cvar=train_args.w_omega_cvar,
        window_steps=train_args.tail_window_steps,
    ).loss
    final = independent_cvar_tail_loss(
        segments[-1].position_history,
        segments[-1].omega_history,
        position_threshold=train_args.success_position_m,
        omega_threshold=train_args.success_omega,
        q_position=train_args.q_position,
        q_omega=train_args.q_omega,
        w_position_cvar=train_args.w_position_cvar,
        w_omega_cvar=train_args.w_omega_cvar,
        window_steps=train_args.tail_window_steps,
    ).loss
    event = float(train_args.early_tail_weight) * early + float(
        train_args.final_tail_weight
    ) * final
    return dense + event, early, final
