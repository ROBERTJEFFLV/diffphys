from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from env_l2f import L2FSimulator, L2FState
from model import ACTION_DIM, CAPABILITY_DIM, INTEGRAL_INPUT_DIM, RESPONSE_DIM, MotorGRUPolicy
from policy_observation import (
    build_policy_observation,
    initial_observation_state,
)


EQUILIBRIUM_ERROR_DIM = 16
EQUILIBRIUM_POLICY_ARCHITECTURE = "equilibrium-centered-motor-gru"
EQUILIBRIUM_POLICY_ARCHITECTURE_VERSION = 1


@dataclass(frozen=True)
class EquilibriumTarget:
    """Analytic disturbance-dependent hover equilibrium for one scenario batch."""

    body_z: torch.Tensor
    total_thrust: torch.Tensor
    motor_trim: torch.Tensor
    feasible: torch.Tensor


@dataclass(frozen=True)
class PhaseSpaceConfig:
    position_scale: float = 0.10
    velocity_scale: float = 0.10
    tilt_scale: float = 0.10
    omega_scale: float = 0.50
    motor_scale: float = 0.10
    action_scale: float = 0.10
    lambda_position: float = 1.0
    lambda_tilt: float = 2.0
    weight_position: float = 1.0
    weight_sliding_position: float = 1.0
    weight_tilt: float = 1.0
    weight_sliding_tilt: float = 1.0
    weight_yaw_rate: float = 0.25
    weight_motor: float = 0.10
    weight_previous_action: float = 0.10


@dataclass(frozen=True)
class InvarianceResult:
    loss: torch.Tensor
    physics_loss: torch.Tensor
    hidden_loss: torch.Tensor
    direction_loss: torch.Tensor
    trim_loss: torch.Tensor
    action_trim_loss: torch.Tensor
    hidden_residual_rms: torch.Tensor
    physical_residual_rms: torch.Tensor
    feasible_fraction: float


@dataclass(frozen=True)
class ContractionResult:
    loss: torch.Tensor
    mean_violation: torch.Tensor
    violation_cvar: torch.Tensor
    terminal_energy: torch.Tensor
    terminal_cvar: torch.Tensor
    per_scenario_violation: torch.Tensor


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool or mask.shape != value.shape[:1]:
        raise ValueError("mask must be boolean and match the first value dimension")
    if not bool(mask.any().item()):
        return value.sum() * 0.0
    return value[mask].mean()


def _thrust_from_motor(
    motor: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
) -> torch.Tensor:
    return (c0 + c1 * motor + c2 * motor.square()).clamp_min(0.0)


def analytic_equilibrium_target(
    state: L2FState,
    *,
    gravity: float = 9.80665,
    bisection_steps: int = 48,
    feasibility_tolerance: float = 1.0e-6,
) -> EquilibriumTarget:
    """Return the exact equal-thrust trim for gravity plus constant force.

    The simulator uses ``a = b3*T/m - g*e3 + f_ext/m``.  Static hover
    therefore requires ``b3*T = m*g*e3 - f_ext``.  Each rotor is assigned
    one quarter of the required total thrust and its monotone thrust curve is
    inverted on the deployable motor interval [-1, 1].
    """

    if bisection_steps < 1:
        raise ValueError("bisection_steps must be positive")
    required = -state.external_force.clone()
    required[:, 2] = required[:, 2] + state.mass * float(gravity)
    total_thrust = torch.linalg.vector_norm(required, dim=-1)
    body_z = required / total_thrust[:, None].clamp_min(1.0e-12)
    per_rotor = 0.25 * total_thrust[:, None]

    c0 = state.thrust_coeff_c0
    c1 = state.thrust_coeff_c1
    c2 = state.thrust_coeff_c2
    lower = torch.full_like(per_rotor.expand_as(c0), -1.0)
    upper = torch.full_like(lower, 1.0)
    thrust_lower = _thrust_from_motor(lower, c0, c1, c2)
    thrust_upper = _thrust_from_motor(upper, c0, c1, c2)
    derivative_lower = c1 - 2.0 * c2
    derivative_upper = c1 + 2.0 * c2
    monotone = (derivative_lower > 0.0) & (derivative_upper > 0.0)
    tolerance = float(feasibility_tolerance)
    rotor_feasible = (
        monotone
        & (per_rotor >= thrust_lower - tolerance)
        & (per_rotor <= thrust_upper + tolerance)
    )

    for _ in range(bisection_steps):
        midpoint = 0.5 * (lower + upper)
        midpoint_thrust = _thrust_from_motor(midpoint, c0, c1, c2)
        below = midpoint_thrust < per_rotor
        lower = torch.where(below, midpoint, lower)
        upper = torch.where(below, upper, midpoint)
    motor_trim = 0.5 * (lower + upper)
    feasible = (
        rotor_feasible.all(dim=-1)
        & torch.isfinite(body_z).all(dim=-1)
        & torch.isfinite(motor_trim).all(dim=-1)
        & (total_thrust > 1.0e-9)
        & (body_z[:, 2] > 0.0)
    )
    return EquilibriumTarget(
        body_z=body_z.detach(),
        total_thrust=total_thrust.detach(),
        motor_trim=motor_trim.detach(),
        feasible=feasible.detach(),
    )


def rotation_from_body_z(body_z: torch.Tensor) -> torch.Tensor:
    """Build a yaw-canonical body-to-world rotation with the requested z axis."""

    if body_z.ndim != 2 or body_z.shape[-1] != 3:
        raise ValueError("body_z must have shape [batch, 3]")
    z_axis = F.normalize(body_z, dim=-1, eps=1.0e-12)
    world_x = z_axis.new_tensor((1.0, 0.0, 0.0)).expand_as(z_axis)
    world_y = z_axis.new_tensor((0.0, 1.0, 0.0)).expand_as(z_axis)
    use_y = (z_axis * world_x).sum(dim=-1, keepdim=True).abs() > 0.90
    reference = torch.where(use_y, world_y, world_x)
    x_axis = F.normalize(
        reference - (reference * z_axis).sum(dim=-1, keepdim=True) * z_axis,
        dim=-1,
        eps=1.0e-12,
    )
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def materialize_equilibrium_state(
    template: L2FState,
    target: EquilibriumTarget,
) -> L2FState:
    dynamic = {
        "position": torch.zeros_like(template.position),
        "velocity": torch.zeros_like(template.velocity),
        "rotation": rotation_from_body_z(target.body_z),
        "omega": torch.zeros_like(template.omega),
        "motor": target.motor_trim.clone(),
        "previous_action": target.motor_trim.clone(),
    }
    return L2FState(
        **{
            field.name: dynamic.get(field.name, getattr(template, field.name))
            for field in fields(L2FState)
        }
    )


def body_z_error_body_frame(
    rotation_body_to_world: torch.Tensor,
    desired_body_z_world: torch.Tensor,
) -> torch.Tensor:
    """Yaw-free thrust-direction error expressed in body coordinates."""

    desired_body = torch.bmm(
        rotation_body_to_world.transpose(1, 2),
        desired_body_z_world.unsqueeze(-1),
    ).squeeze(-1)
    body_z = desired_body.new_tensor((0.0, 0.0, 1.0)).expand_as(desired_body)
    return torch.linalg.cross(body_z, desired_body, dim=-1)


class EquilibriumCenteredPolicy(nn.Module):
    """GRU trim estimator plus feedback that is exactly zero at equilibrium."""

    def __init__(
        self,
        *,
        observation_dim: int = INTEGRAL_INPUT_DIM,
        encoder_dim: int = 192,
        hidden_dim: int = 192,
        encoder_depth: int = 2,
        negative_slope: float = 0.05,
        feedback_context_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if observation_dim != INTEGRAL_INPUT_DIM:
            raise ValueError("equilibrium-centered policy currently requires integral25")
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be positive")
        layers: list[nn.Module] = []
        for index in range(encoder_depth):
            input_width = observation_dim if index == 0 else encoder_dim
            layers.extend(
                (
                    nn.Linear(input_width, encoder_dim),
                    nn.LeakyReLU(negative_slope=negative_slope),
                )
            )
        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRUCell(encoder_dim, hidden_dim)
        self.trim_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.direction_head = nn.Linear(hidden_dim, 3)
        self.feedback_gain_head = nn.Linear(
            hidden_dim, ACTION_DIM * EQUILIBRIUM_ERROR_DIM
        )
        self.base_feedback_gain = nn.Parameter(
            torch.zeros(ACTION_DIM, EQUILIBRIUM_ERROR_DIM)
        )
        self.motor_state_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.capability_head = nn.Linear(hidden_dim, CAPABILITY_DIM)
        self.response_head = nn.Linear(hidden_dim + ACTION_DIM, RESPONSE_DIM)
        self.observation_dim = observation_dim
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.encoder_depth = encoder_depth
        self.negative_slope = negative_slope
        self.feedback_context_scale = float(feedback_context_scale)
        self.register_buffer(
            "feedback_error_scales",
            torch.tensor(
                (
                    0.10,
                    0.10,
                    0.10,
                    0.10,
                    0.10,
                    0.10,
                    0.10,
                    0.10,
                    0.10,
                    0.50,
                    0.50,
                    0.50,
                    0.10,
                    0.10,
                    0.10,
                    0.10,
                )
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.uniform_(self.trim_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.trim_head.bias)
        nn.init.zeros_(self.direction_head.weight)
        nn.init.zeros_(self.direction_head.bias)
        nn.init.zeros_(self.feedback_gain_head.weight)
        nn.init.zeros_(self.feedback_gain_head.bias)
        nn.init.zeros_(self.base_feedback_gain)
        for head in (self.motor_state_head, self.capability_head, self.response_head):
            nn.init.orthogonal_(head.weight)
            nn.init.zeros_(head.bias)

    @classmethod
    def from_motor_gru(
        cls,
        source: MotorGRUPolicy,
        *,
        feedback_context_scale: float = 0.25,
    ) -> "EquilibriumCenteredPolicy":
        """Reuse a Q2 encoder/GRU while replacing all three action branches."""

        if source.observation_dim != INTEGRAL_INPUT_DIM:
            raise ValueError("source policy must use the integral25 observation")
        first = source.encoder[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("source encoder must begin with Linear")
        target = cls(
            observation_dim=source.observation_dim,
            encoder_dim=first.out_features,
            hidden_dim=source.hidden_dim,
            encoder_depth=len(source.encoder) // 2,
            negative_slope=source.negative_slope,
            feedback_context_scale=feedback_context_scale,
        ).to(device=first.weight.device, dtype=first.weight.dtype)
        target.encoder.load_state_dict(source.encoder.state_dict())
        target.gru.load_state_dict(source.gru.state_dict())
        # This is initialization only, not a persistent imitation objective.
        target.trim_head.load_state_dict(source.motor_head.state_dict())
        target.motor_state_head.load_state_dict(source.motor_state_head.state_dict())
        target.capability_head.load_state_dict(source.capability_head.state_dict())
        target.response_head.load_state_dict(source.response_head.state_dict())
        return target

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "architecture": EQUILIBRIUM_POLICY_ARCHITECTURE,
            "architecture_version": EQUILIBRIUM_POLICY_ARCHITECTURE_VERSION,
            "observation_dim": self.observation_dim,
            "encoder_dim": self.encoder_dim,
            "encoder_depth": self.encoder_depth,
            "hidden_dim": self.hidden_dim,
            "action_dim": ACTION_DIM,
            "equilibrium_error_dim": EQUILIBRIUM_ERROR_DIM,
            "feedback_zero_at_equilibrium": True,
            "deployment_privileged_inputs": False,
        }

    def _encode(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observation must have shape [batch,{self.observation_dim}]"
            )
        if hidden is None:
            hidden = self.initial_hidden(
                observation.shape[0],
                device=observation.device,
                dtype=observation.dtype,
            )
        hidden = self.gru(self.encoder(observation), hidden)
        latent = F.leaky_relu(hidden, negative_slope=self.negative_slope)
        return latent, hidden

    @staticmethod
    def _direction_from_raw(raw: torch.Tensor) -> torch.Tensor:
        positive_z = F.softplus(raw[:, 2:3]) + 1.0e-4
        return F.normalize(torch.cat((raw[:, :2], positive_z), dim=-1), dim=-1)

    def _action_from_latent(
        self,
        observation: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        trim_logits = self.trim_head(latent)
        trim_action = torch.tanh(trim_logits)
        body_z_star = self._direction_from_raw(self.direction_head(latent))
        rotation = observation[:, 6:15].reshape(-1, 3, 3)
        tilt_error = body_z_error_body_frame(rotation, body_z_star)
        error = torch.cat(
            (
                observation[:, 0:3],
                observation[:, 3:6],
                tilt_error,
                observation[:, 15:18],
                observation[:, 21:25] - trim_action,
            ),
            dim=-1,
        )
        phi = torch.tanh(error / self.feedback_error_scales.to(error))
        contextual_gain = self.feedback_gain_head(latent).reshape(
            -1, ACTION_DIM, EQUILIBRIUM_ERROR_DIM
        )
        gain = self.base_feedback_gain.unsqueeze(0) + self.feedback_context_scale * torch.tanh(
            contextual_gain
        )
        feedback_logits = torch.einsum("bij,bj->bi", gain, phi)
        action = torch.tanh(trim_logits + feedback_logits)
        return action, {
            "trim_logits": trim_logits,
            "trim_action": trim_action,
            "body_z_star": body_z_star,
            "feedback_error": error,
            "feedback_features": phi,
            "feedback_gain": gain,
            "feedback_logits": feedback_logits,
        }

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, hidden = self._encode(observation, hidden)
        action, _ = self._action_from_latent(observation, latent)
        return action, hidden

    def forward_with_aux(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        latent, hidden = self._encode(observation, hidden)
        action, details = self._action_from_latent(observation, latent)
        motor_state = torch.tanh(self.motor_state_head(latent))
        auxiliary = {
            **details,
            "motor_state": motor_state,
            "capability": self.capability_head(latent),
            "response": self.response_head(torch.cat((latent, action.detach()), dim=-1)),
        }
        return action, hidden, auxiliary


def equilibrium_prediction_loss(
    details: dict[str, torch.Tensor],
    target: EquilibriumTarget,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction_per_sample = 1.0 - (
        details["body_z_star"] * target.body_z
    ).sum(dim=-1).clamp(-1.0, 1.0)
    trim_per_sample = F.smooth_l1_loss(
        details["trim_action"], target.motor_trim, reduction="none"
    ).mean(dim=-1)
    direction = _masked_mean(direction_per_sample, target.feasible)
    trim = _masked_mean(trim_per_sample, target.feasible)
    return direction + trim, direction, trim


def phase_space_energy(
    state: L2FState,
    target: EquilibriumTarget,
    config: PhaseSpaceConfig,
) -> torch.Tensor:
    """Positive phase-space energy around the disturbance-dependent equilibrium."""

    position = state.position
    velocity = state.velocity
    tilt_error = body_z_error_body_frame(state.rotation, target.body_z)
    omega = state.omega
    p = position / float(config.position_scale)
    sliding_p = (
        velocity + float(config.lambda_position) * position
    ) / float(config.velocity_scale)
    tilt = tilt_error / float(config.tilt_scale)
    sliding_tilt = (
        omega[:, :2] + float(config.lambda_tilt) * tilt_error[:, :2]
    ) / float(config.omega_scale)
    motor = (state.motor - target.motor_trim) / float(config.motor_scale)
    previous_action = (state.previous_action - target.motor_trim) / float(
        config.action_scale
    )
    return (
        float(config.weight_position) * p.square().sum(dim=-1)
        + float(config.weight_sliding_position) * sliding_p.square().sum(dim=-1)
        + float(config.weight_tilt) * tilt[:, :2].square().sum(dim=-1)
        + float(config.weight_sliding_tilt) * sliding_tilt.square().sum(dim=-1)
        + float(config.weight_yaw_rate)
        * (omega[:, 2] / float(config.omega_scale)).square()
        + float(config.weight_motor) * motor.square().sum(dim=-1)
        + float(config.weight_previous_action) * previous_action.square().sum(dim=-1)
    )


def contraction_objective(
    energy: torch.Tensor,
    *,
    interval_seconds: float,
    contraction_rate: float,
    epsilon: float = 1.0e-6,
    cvar_fraction: float = 0.20,
    terminal_weight: float = 1.0,
    violation_cvar_weight: float = 1.0,
) -> ContractionResult:
    if energy.ndim != 2 or energy.shape[0] < 2:
        raise ValueError("energy must have shape [sample_time>=2, batch]")
    if interval_seconds <= 0.0 or contraction_rate < 0.0:
        raise ValueError("interval_seconds must be positive and rate non-negative")
    if not 0.0 < cvar_fraction <= 1.0:
        raise ValueError("cvar_fraction must be in (0,1]")
    decay = math.exp(-2.0 * float(contraction_rate) * float(interval_seconds))
    margin = energy[1:] - decay * energy[:-1].detach() - float(epsilon)
    violations = F.relu(margin).square()
    per_scenario = violations.mean(dim=0)
    count = max(1, int(math.ceil(cvar_fraction * energy.shape[1])))
    violation_cvar = torch.topk(per_scenario, count, largest=True).values.mean()
    terminal = energy[-1]
    terminal_cvar = torch.topk(terminal, count, largest=True).values.mean()
    mean_violation = violations.mean()
    loss = (
        mean_violation
        + float(violation_cvar_weight) * violation_cvar
        + float(terminal_weight) * terminal_cvar
    )
    return ContractionResult(
        loss=loss,
        mean_violation=mean_violation,
        violation_cvar=violation_cvar,
        terminal_energy=terminal.mean(),
        terminal_cvar=terminal_cvar,
        per_scenario_violation=per_scenario,
    )


def _so3_local_residual(rotation: torch.Tensor) -> torch.Tensor:
    vector = 0.5 * torch.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1.0e-12)
    cosine = 0.5 * (rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) - 1.0)
    angle = torch.atan2(sine, cosine.clamp(-1.0, 1.0))
    return angle * vector / sine


def equilibrium_invariance_loss(
    policy: EquilibriumCenteredPolicy,
    sim: L2FSimulator,
    template: L2FState,
    target: EquilibriumTarget,
    *,
    integral_input_frame: str = "body",
    integral_limit: float = 0.5,
    fixed_point_steps: int = 32,
) -> InvarianceResult:
    """Require the complete physical/recurrent equilibrium to remain fixed."""

    if fixed_point_steps < 1:
        raise ValueError("fixed_point_steps must be positive")
    equilibrium = materialize_equilibrium_state(template, target)
    observation_state = initial_observation_state(
        equilibrium.position.shape[0],
        device=equilibrium.position.device,
        dtype=equilibrium.position.dtype,
    )
    observation, _ = build_policy_observation(
        equilibrium,
        observation_state,
        mode="integral25",
        noise_max=0.0,
        integral_input_frame=integral_input_frame,
    )
    hidden_star = policy.initial_hidden(
        equilibrium.position.shape[0],
        device=equilibrium.position.device,
        dtype=equilibrium.position.dtype,
    )
    with torch.no_grad():
        for _ in range(fixed_point_steps):
            _, hidden_star = policy(observation, hidden_star)
    hidden_star = hidden_star.detach()
    action, hidden_next, details = policy.forward_with_aux(observation, hidden_star)
    next_state = sim.step(equilibrium, action, grad_decay=1.0)

    relative_rotation = equilibrium.rotation.transpose(1, 2) @ next_state.rotation
    physical_blocks = (
        (next_state.position - equilibrium.position) / 0.10,
        (next_state.velocity - equilibrium.velocity) / 0.10,
        _so3_local_residual(relative_rotation) / 0.10,
        (next_state.omega - equilibrium.omega) / 0.50,
        (next_state.motor - equilibrium.motor) / 0.10,
        (next_state.previous_action - equilibrium.previous_action) / 0.10,
    )
    physical_per_sample = torch.stack(
        tuple(block.reshape(block.shape[0], -1).square().mean(dim=-1) for block in physical_blocks),
        dim=0,
    ).mean(dim=0)
    hidden_per_sample = (
        (hidden_next - hidden_star).square().mean(dim=-1) / (0.10 * 0.10)
    )
    action_per_sample = (
        (action - target.motor_trim).square().mean(dim=-1) / (0.10 * 0.10)
    )
    physics_loss = _masked_mean(physical_per_sample, target.feasible)
    hidden_loss = _masked_mean(hidden_per_sample, target.feasible)
    action_trim_loss = _masked_mean(action_per_sample, target.feasible)
    prediction, direction_loss, trim_loss = equilibrium_prediction_loss(details, target)
    loss = physics_loss + hidden_loss + action_trim_loss + prediction
    return InvarianceResult(
        loss=loss,
        physics_loss=physics_loss,
        hidden_loss=hidden_loss,
        direction_loss=direction_loss,
        trim_loss=trim_loss,
        action_trim_loss=action_trim_loss,
        hidden_residual_rms=torch.sqrt(hidden_loss.clamp_min(0.0)),
        physical_residual_rms=torch.sqrt(physics_loss.clamp_min(0.0)),
        feasible_fraction=float(target.feasible.float().mean().item()),
    )
