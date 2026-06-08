from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class L2FParams:
    dt: float = 0.01
    mass: float = 0.05
    gravity: float = 9.80665
    arm_length: float = 0.046
    yaw_drag: float = 0.012
    motor_tau: float = 0.06
    motor_authority: float = 1.35
    inertia_x: float = 1.4e-5
    inertia_y: float = 1.4e-5
    inertia_z: float = 2.17e-5
    max_initial_position: float = 1.0
    max_initial_velocity: float = 0.6
    max_initial_angle: float = 0.45
    max_initial_omega: float = 1.0


@dataclass
class L2FState:
    position: torch.Tensor
    velocity: torch.Tensor
    rotation: torch.Tensor
    omega: torch.Tensor
    motor: torch.Tensor
    previous_action: torch.Tensor


@dataclass(frozen=True)
class L2FLossConfig:
    p_scale: float = 2.0
    v_scale: float = 3.0
    omega_scale: float = 10.0
    huber_beta: float = 1.0
    w_p: float = 1.0
    w_v: float = 0.3
    w_r: float = 0.1
    w_omega: float = 0.05
    attitude_mode: str = "full"


class _GradientDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, decay: float) -> torch.Tensor:
        ctx.decay = float(decay)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output * ctx.decay, None


def apply_gradient_decay(tensor: torch.Tensor, decay: float) -> torch.Tensor:
    if decay >= 1.0:
        return tensor
    if decay < 0.0:
        raise ValueError("gradient decay must be non-negative")
    return _GradientDecay.apply(tensor, float(decay))


def decay_state_gradient(state: L2FState, decay: float) -> L2FState:
    if decay >= 1.0:
        return state
    return L2FState(
        position=apply_gradient_decay(state.position, decay),
        velocity=apply_gradient_decay(state.velocity, decay),
        rotation=apply_gradient_decay(state.rotation, decay),
        omega=apply_gradient_decay(state.omega, decay),
        motor=apply_gradient_decay(state.motor, decay),
        previous_action=apply_gradient_decay(state.previous_action, decay),
    )


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vector[..., 0])
    x, y, z = vector.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )


def _orthonormalize(rotation: torch.Tensor) -> torch.Tensor:
    x = F.normalize(rotation[:, :, 0], dim=-1, eps=1.0e-6)
    y = rotation[:, :, 1]
    y = y - (x * y).sum(dim=-1, keepdim=True) * x
    y = F.normalize(y, dim=-1, eps=1.0e-6)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _so3_exp(phi: torch.Tensor) -> torch.Tensor:
    theta_sq = phi.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq)
    theta_sq_matrix = theta_sq.view(-1, 1, 1)
    theta_matrix = theta.view(-1, 1, 1)
    theta_sq_safe = torch.clamp(theta_sq_matrix, min=1.0e-8)
    theta_safe = torch.clamp(theta_matrix, min=1.0e-4)
    small = theta_sq_matrix < 1.0e-8
    a_small = 1.0 - theta_sq_matrix / 6.0 + theta_sq_matrix.square() / 120.0
    b_small = 0.5 - theta_sq_matrix / 24.0 + theta_sq_matrix.square() / 720.0
    a = torch.where(small, a_small, torch.sin(theta_matrix) / theta_safe)
    b = torch.where(
        small,
        b_small,
        (1.0 - torch.cos(theta_matrix)) / theta_sq_safe,
    )
    k = _skew(phi)
    k2 = k @ k
    identity = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(phi.shape[0], 3, 3)
    return identity + a * k + b * k2


def _smooth_l1_sum(value: torch.Tensor, beta: float) -> torch.Tensor:
    zeros = torch.zeros_like(value)
    loss = F.smooth_l1_loss(value, zeros, beta=beta, reduction="none")
    return loss.reshape(value.shape[0], -1).sum(dim=-1)


class L2FSimulator:
    """Differentiable L2F-style quadrotor simulator using CUDA tensors when available."""

    def __init__(self, params: L2FParams | None = None) -> None:
        self.params = params or L2FParams()

    def reset(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> L2FState:
        p = self.params
        device = torch.device(device)
        position = torch.empty(batch_size, 3, device=device, dtype=dtype).uniform_(
            -p.max_initial_position, p.max_initial_position
        )
        velocity = torch.empty(batch_size, 3, device=device, dtype=dtype).uniform_(
            -p.max_initial_velocity, p.max_initial_velocity
        )
        omega = torch.empty(batch_size, 3, device=device, dtype=dtype).uniform_(
            -p.max_initial_omega, p.max_initial_omega
        )
        motor = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        previous_action = torch.zeros(batch_size, 4, device=device, dtype=dtype)

        axis = torch.randn(batch_size, 3, device=device, dtype=dtype)
        axis = F.normalize(axis, dim=-1, eps=1.0e-6)
        angle = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(
            -p.max_initial_angle, p.max_initial_angle
        )
        identity = torch.eye(3, device=device, dtype=dtype).expand(batch_size, 3, 3)
        k = _skew(axis)
        rotation = identity + torch.sin(angle).view(-1, 1, 1) * k
        rotation = rotation + (1.0 - torch.cos(angle)).view(-1, 1, 1) * (k @ k)

        return L2FState(position, velocity, rotation, omega, motor, previous_action)

    def state_features(self, state: L2FState) -> torch.Tensor:
        return torch.cat(
            (
                state.position,
                state.velocity,
                state.rotation.reshape(state.position.shape[0], 9),
                state.omega,
            ),
            dim=-1,
        )

    def error_features(self, state: L2FState) -> torch.Tensor:
        batch_size = state.position.shape[0]
        identity = torch.eye(
            3, device=state.position.device, dtype=state.position.dtype
        ).expand(batch_size, 3, 3)
        return torch.cat(
            (
                state.position,
                state.velocity,
                (state.rotation - identity).reshape(batch_size, 9),
                state.omega,
            ),
            dim=-1,
        )

    def step(
        self,
        state: L2FState,
        action: torch.Tensor,
        *,
        grad_decay: float = 1.0,
        external_force: torch.Tensor | None = None,
        external_torque: torch.Tensor | None = None,
    ) -> L2FState:
        p = self.params
        dt = p.dt
        command = action.clamp(-1.0, 1.0)
        alpha = min(max(dt / p.motor_tau, 0.0), 1.0)
        motor = state.motor + alpha * (command - state.motor)

        hover_thrust = p.mass * p.gravity / 4.0
        thrust = hover_thrust * (1.0 + p.motor_authority * motor)
        thrust = thrust.clamp_min(0.0)
        total_thrust = thrust.sum(dim=-1, keepdim=True)

        body_z = state.rotation[:, :, 2]
        gravity = torch.tensor(
            (0.0, 0.0, -p.gravity),
            device=state.position.device,
            dtype=state.position.dtype,
        )
        acceleration = body_z * (total_thrust / p.mass) + gravity
        if external_force is not None:
            acceleration = acceleration + external_force / p.mass
        velocity = state.velocity + dt * acceleration
        position = state.position + dt * velocity

        torque = torch.stack(
            (
                p.arm_length * (thrust[:, 1] - thrust[:, 3]),
                p.arm_length * (thrust[:, 2] - thrust[:, 0]),
                p.yaw_drag * (thrust[:, 0] - thrust[:, 1] + thrust[:, 2] - thrust[:, 3]),
            ),
            dim=-1,
        )
        if external_torque is not None:
            torque = torque + external_torque
        inertia = torch.tensor(
            (p.inertia_x, p.inertia_y, p.inertia_z),
            device=state.position.device,
            dtype=state.position.dtype,
        )
        omega_cross = torch.cross(state.omega, state.omega * inertia, dim=-1)
        omega = state.omega + dt * ((torque - omega_cross) / inertia)

        rotation = state.rotation @ _so3_exp(dt * omega)

        next_state = L2FState(position, velocity, rotation, omega, motor, command)
        return decay_state_gradient(next_state, grad_decay)

    def tracking_components(
        self,
        state: L2FState,
        config: L2FLossConfig,
    ) -> dict[str, torch.Tensor]:
        batch_size = state.position.shape[0]
        identity = torch.eye(
            3, device=state.position.device, dtype=state.position.dtype
        ).expand(batch_size, 3, 3)

        p_error = state.position / config.p_scale
        v_error = state.velocity / config.v_scale
        omega_error = state.omega / config.omega_scale
        if config.attitude_mode == "tilt":
            attitude_error = state.rotation[:, :, 2] - identity[:, :, 2]
        elif config.attitude_mode == "full":
            attitude_error = state.rotation - identity
        else:
            raise ValueError("attitude_mode must be 'full' or 'tilt'")

        return {
            "position": config.w_p * _smooth_l1_sum(p_error, config.huber_beta),
            "velocity": config.w_v * _smooth_l1_sum(v_error, config.huber_beta),
            "attitude": config.w_r * _smooth_l1_sum(attitude_error, config.huber_beta),
            "omega": config.w_omega * _smooth_l1_sum(omega_error, config.huber_beta),
        }

    def tracking_potential(
        self,
        state: L2FState,
        config: L2FLossConfig,
    ) -> torch.Tensor:
        components = self.tracking_components(state, config)
        return sum(components.values())

    def outward_velocity_loss(
        self,
        state: L2FState,
        config: L2FLossConfig,
    ) -> torch.Tensor:
        radial_velocity = (
            (state.position / config.p_scale) * (state.velocity / config.v_scale)
        ).sum(dim=-1)
        return F.relu(radial_velocity).square().mean()

    def loss_terms(
        self,
        state: L2FState,
        action: torch.Tensor,
        action_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = state.position.shape[0]
        identity = torch.eye(
            3, device=state.position.device, dtype=state.position.dtype
        ).expand(batch_size, 3, 3)
        return {
            "position": 8.0 * state.position.square().sum(dim=-1).mean(),
            "velocity": 1.0 * state.velocity.square().sum(dim=-1).mean(),
            "attitude": 3.0 * (state.rotation - identity).square().sum(dim=(1, 2)).mean(),
            "omega": 0.12 * state.omega.square().sum(dim=-1).mean(),
            "motor": 0.01 * action.square().mean(),
            "smooth": 0.04 * action_delta.square().mean(),
        }
