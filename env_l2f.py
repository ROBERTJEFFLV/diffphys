from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class L2FParams:
    dt: float = 0.01
    action_max: float = 1.0
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
    disturbance_force_max: float = 0.0
    external_force_ratio: float = 0.0


@dataclass(frozen=True)
class L2FRaptorEpisodeDynamics:
    mass: torch.Tensor
    thrust_coeff_c0: torch.Tensor
    thrust_coeff_c1: torch.Tensor
    thrust_coeff_c2: torch.Tensor
    thrust_to_weight: torch.Tensor
    torque_to_inertia: torch.Tensor
    rotor_distance_factor: torch.Tensor
    inertia_factor: torch.Tensor
    motor_time_rising: torch.Tensor
    motor_time_falling: torch.Tensor
    rotor_torque_constant: torch.Tensor
    cbrt_mass: torch.Tensor
    force_std: torch.Tensor
    arm_length: torch.Tensor
    inertia_x: torch.Tensor
    inertia_y: torch.Tensor
    inertia_z: torch.Tensor


@dataclass
class L2FState:
    position: torch.Tensor
    velocity: torch.Tensor
    rotation: torch.Tensor
    omega: torch.Tensor
    motor: torch.Tensor
    previous_action: torch.Tensor
    external_force: torch.Tensor
    mass: torch.Tensor
    thrust_coeff_c0: torch.Tensor
    thrust_coeff_c1: torch.Tensor
    thrust_coeff_c2: torch.Tensor
    thrust_to_weight: torch.Tensor
    torque_to_inertia: torch.Tensor
    rotor_distance_factor: torch.Tensor
    inertia_factor: torch.Tensor
    motor_time_rising: torch.Tensor
    motor_time_falling: torch.Tensor
    rotor_torque_constant: torch.Tensor
    cbrt_mass: torch.Tensor
    force_std: torch.Tensor
    arm_length: torch.Tensor
    inertia_x: torch.Tensor
    inertia_y: torch.Tensor
    inertia_z: torch.Tensor


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
        external_force=state.external_force,
        mass=state.mass,
        thrust_coeff_c0=state.thrust_coeff_c0,
        thrust_coeff_c1=state.thrust_coeff_c1,
        thrust_coeff_c2=state.thrust_coeff_c2,
        thrust_to_weight=state.thrust_to_weight,
        torque_to_inertia=state.torque_to_inertia,
        rotor_distance_factor=state.rotor_distance_factor,
        inertia_factor=state.inertia_factor,
        motor_time_rising=state.motor_time_rising,
        motor_time_falling=state.motor_time_falling,
        rotor_torque_constant=state.rotor_torque_constant,
        cbrt_mass=state.cbrt_mass,
        force_std=state.force_std,
        arm_length=state.arm_length,
        inertia_x=state.inertia_x,
        inertia_y=state.inertia_y,
        inertia_z=state.inertia_z,
    )


def _sample_reciprocal_style_factor(unit: torch.Tensor, deviation: float) -> torch.Tensor:
    upper = max(1.0 + deviation, 1.0)
    lower = 1.0 / upper
    return lower + unit * (upper - lower)


def sample_raptor_episode_dynamics(
    nominal: L2FParams,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> L2FRaptorEpisodeDynamics:
    batch = batch_size
    sample = torch.rand
    normal = torch.randn

    nominal_mass = torch.full((batch,), float(nominal.mass), device=device, dtype=dtype)
    nominal_arm = float(nominal.arm_length)
    nominal_inertia_x = torch.full((batch,), float(nominal.inertia_x), device=device, dtype=dtype)
    nominal_inertia_y = torch.full((batch,), float(nominal.inertia_y), device=device, dtype=dtype)
    nominal_inertia_z = torch.full((batch,), float(nominal.inertia_z), device=device, dtype=dtype)
    nominal_gravity = float(nominal.gravity)

    nominal_thrust_coeff_c0 = nominal_mass * nominal_gravity / 4.0
    nominal_thrust_coeff_c1 = nominal_thrust_coeff_c0 * float(nominal.motor_authority)
    nominal_thrust_coeff_c2 = torch.zeros((batch,), device=device, dtype=dtype)
    max_action = float(nominal.action_max)
    max_thrust_nominal = (
        nominal_thrust_coeff_c0
        + nominal_thrust_coeff_c1 * max_action
        + nominal_thrust_coeff_c2 * (max_action * max_action)
    ) * 4.0
    thrust_to_weight_nominal = max_thrust_nominal / (nominal_mass * nominal_gravity)

    thrust_to_weight = sample((batch,), device=device, dtype=dtype) * (5.0 - 1.5) + 1.5
    factor_thrust_to_weight = thrust_to_weight / thrust_to_weight_nominal

    relative_size_min = math.pow(0.02, 1.0 / 3.0)
    relative_size_max = math.pow(5.0, 1.0 / 3.0)
    size_new = sample((batch,), device=device, dtype=dtype) * (relative_size_max - relative_size_min) + relative_size_min
    mass_new = size_new * size_new * size_new
    mass_new = torch.clamp(mass_new, min=1.0e-12)
    scale_relative = (mass_new / nominal_mass).pow(1.0 / 3.0)
    factor_mass = mass_new / nominal_mass
    size_factor = _sample_reciprocal_style_factor(sample((batch,), device=device, dtype=dtype), 0.1)
    rotor_distance_factor = scale_relative * size_factor
    arm_length = torch.full((batch,), nominal_arm, device=device, dtype=dtype) * rotor_distance_factor

    thrust_factor = factor_thrust_to_weight * factor_mass
    thrust_coeff_c0 = nominal_thrust_coeff_c0 * thrust_factor
    thrust_coeff_c1 = nominal_thrust_coeff_c1 * thrust_factor
    thrust_coeff_c2 = nominal_thrust_coeff_c2 * thrust_factor

    max_thrust_per_rotor = thrust_to_weight * mass_new * nominal_gravity / 4.0
    max_torque = math.sqrt(2.0) * abs(nominal_arm) * max_thrust_per_rotor
    torque_to_inertia_nominal = max_torque / nominal_inertia_x
    torque_to_inertia = sample((batch,), device=device, dtype=dtype) * (1200.0 - 40.0) + 40.0
    torque_to_inertia_factor = torque_to_inertia / torque_to_inertia_nominal
    inertia_factor = torque_to_inertia_factor / torch.clamp(rotor_distance_factor, min=1.0e-6)
    inertia_x = nominal_inertia_x / torch.clamp(inertia_factor, min=1.0e-6)
    inertia_y = nominal_inertia_y / torch.clamp(inertia_factor, min=1.0e-6)
    inertia_z = nominal_inertia_z / torch.clamp(inertia_factor, min=1.0e-6)

    rotor_torque_constant = sample((batch,), device=device, dtype=dtype) * (0.05 - 0.005) + 0.005
    motor_time_rising = sample((batch,), device=device, dtype=dtype) * (0.10 - 0.03) + 0.03
    motor_time_falling = sample((batch,), device=device, dtype=dtype) * (0.30 - 0.03) + 0.03

    surplus = (thrust_to_weight - 1.0).clamp_min(0.0)
    multiple = sample((batch,), device=device, dtype=dtype) * (surplus * 0.3)
    force_std = multiple * thrust_to_weight * mass_new / 3.0
    external_force = normal((batch, 3), device=device, dtype=dtype) * force_std[:, None]

    return L2FRaptorEpisodeDynamics(
        mass=mass_new,
        thrust_coeff_c0=thrust_coeff_c0[:, None].expand(batch, 4),
        thrust_coeff_c1=thrust_coeff_c1[:, None].expand(batch, 4),
        thrust_coeff_c2=thrust_coeff_c2[:, None].expand(batch, 4),
        thrust_to_weight=thrust_to_weight,
        torque_to_inertia=torque_to_inertia,
        rotor_distance_factor=rotor_distance_factor,
        inertia_factor=inertia_factor,
        motor_time_rising=motor_time_rising,
        motor_time_falling=motor_time_falling,
        rotor_torque_constant=rotor_torque_constant,
        cbrt_mass=size_new,
        force_std=force_std,
        arm_length=arm_length,
        inertia_x=inertia_x,
        inertia_y=inertia_y,
        inertia_z=inertia_z,
    )


def _fixed_episode_dynamics(
    nominal: L2FParams,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> L2FRaptorEpisodeDynamics:
    batch = batch_size
    mass = torch.full((batch,), float(nominal.mass), device=device, dtype=dtype)
    thrust_coeff_c0_nom = torch.full((batch,), float(nominal.mass * nominal.gravity / 4.0), device=device, dtype=dtype)
    thrust_coeff_c1_nom = thrust_coeff_c0_nom * float(nominal.motor_authority)
    thrust_coeff_c2_nom = torch.zeros((batch,), device=device, dtype=dtype)
    thrust_to_weight = torch.full((batch,), float(nominal.motor_authority + 1.0), device=device, dtype=dtype)
    torque_to_inertia = torch.full((batch,), 250.0, device=device, dtype=dtype)
    rotor_distance_factor = torch.ones((batch,), device=device, dtype=dtype)
    inertia_factor = torch.ones((batch,), device=device, dtype=dtype)
    motor_time_rising = torch.full((batch,), float(nominal.motor_tau), device=device, dtype=dtype)
    motor_time_falling = torch.full((batch,), float(nominal.motor_tau), device=device, dtype=dtype)
    rotor_torque_constant = torch.full((batch,), 0.02, device=device, dtype=dtype)
    cbrt_mass = mass.pow(1.0 / 3.0)
    force_std = torch.zeros((batch,), device=device, dtype=dtype)
    arm_length = torch.full((batch,), float(nominal.arm_length), device=device, dtype=dtype)
    inertia_x = torch.full((batch,), float(nominal.inertia_x), device=device, dtype=dtype)
    inertia_y = torch.full((batch,), float(nominal.inertia_y), device=device, dtype=dtype)
    inertia_z = torch.full((batch,), float(nominal.inertia_z), device=device, dtype=dtype)
    return L2FRaptorEpisodeDynamics(
        mass=mass,
        thrust_coeff_c0=thrust_coeff_c0_nom[:, None].expand(batch, 4),
        thrust_coeff_c1=thrust_coeff_c1_nom[:, None].expand(batch, 4),
        thrust_coeff_c2=thrust_coeff_c2_nom[:, None].expand(batch, 4),
        thrust_to_weight=thrust_to_weight,
        torque_to_inertia=torque_to_inertia,
        rotor_distance_factor=rotor_distance_factor,
        inertia_factor=inertia_factor,
        motor_time_rising=motor_time_rising,
        motor_time_falling=motor_time_falling,
        rotor_torque_constant=rotor_torque_constant,
        cbrt_mass=cbrt_mass,
        force_std=force_std,
        arm_length=arm_length,
        inertia_x=inertia_x,
        inertia_y=inertia_y,
        inertia_z=inertia_z,
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
        sample_dynamics: bool = False,
        sampled_dynamics_level: str = "small",
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
        external_force = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        axis = torch.randn(batch_size, 3, device=device, dtype=dtype)
        axis = F.normalize(axis, dim=-1, eps=1.0e-6)
        angle = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(
            -p.max_initial_angle, p.max_initial_angle
        )
        identity = torch.eye(3, device=device, dtype=dtype).expand(batch_size, 3, 3)
        k = _skew(axis)
        rotation = identity + torch.sin(angle).view(-1, 1, 1) * k
        rotation = rotation + (1.0 - torch.cos(angle)).view(-1, 1, 1) * (k @ k)

        if sample_dynamics and sampled_dynamics_level == "broad":
            dynamics = sample_raptor_episode_dynamics(
                p,
                batch_size,
                device=device,
                dtype=dtype,
            )
            external_force = dynamics.external_force
        else:
            dynamics = _fixed_episode_dynamics(
                p,
                batch_size,
                device=device,
                dtype=dtype,
            )
            disturbance_force_max = p.disturbance_force_max
            if disturbance_force_max <= 0.0:
                disturbance_force_max = p.external_force_ratio
            max_force_std = (dynamics.thrust_to_weight - 1.0).clamp_min(0.0) * disturbance_force_max
            force_std = max_force_std * dynamics.thrust_to_weight * dynamics.mass / 3.0
            if torch.any(force_std > 0.0):
                force_scale = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(0.0, 1.0) * force_std[:, None]
                external_force = torch.randn(batch_size, 3, device=device, dtype=dtype) * force_scale
            else:
                external_force = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        return L2FState(
            position,
            velocity,
            rotation,
            omega,
            motor,
            previous_action,
            external_force,
            mass=dynamics.mass,
            thrust_coeff_c0=dynamics.thrust_coeff_c0,
            thrust_coeff_c1=dynamics.thrust_coeff_c1,
            thrust_coeff_c2=dynamics.thrust_coeff_c2,
            thrust_to_weight=dynamics.thrust_to_weight,
            torque_to_inertia=dynamics.torque_to_inertia,
            rotor_distance_factor=dynamics.rotor_distance_factor,
            inertia_factor=dynamics.inertia_factor,
            motor_time_rising=dynamics.motor_time_rising,
            motor_time_falling=dynamics.motor_time_falling,
            rotor_torque_constant=dynamics.rotor_torque_constant,
            cbrt_mass=dynamics.cbrt_mass,
            force_std=getattr(dynamics, "force_std", torch.zeros_like(dynamics.mass)),
            arm_length=dynamics.arm_length,
            inertia_x=dynamics.inertia_x,
            inertia_y=dynamics.inertia_y,
            inertia_z=dynamics.inertia_z,
        )

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
    ) -> L2FState:
        p = self.params
        dt = p.dt
        command = action.clamp(-1.0, 1.0)
        motor_time_rising = state.motor_time_rising[:, None]
        motor_time_falling = state.motor_time_falling[:, None]
        motor_tau = torch.where(command >= state.motor, motor_time_rising, motor_time_falling)
        alpha = torch.clamp(dt / motor_tau, 0.0, 1.0)
        motor = state.motor + alpha * (command - state.motor)
        thrust = (
            state.thrust_coeff_c0 * 1.0
            + state.thrust_coeff_c1 * motor
            + state.thrust_coeff_c2 * motor * motor
        )
        thrust = thrust.clamp_min(0.0)
        total_thrust = thrust.sum(dim=-1, keepdim=True)

        body_z = state.rotation[:, :, 2]
        gravity = torch.tensor(
            (0.0, 0.0, -p.gravity),
            device=state.position.device,
            dtype=state.position.dtype,
        )
        inv_mass = 1.0 / state.mass[:, None]
        acceleration = body_z * (total_thrust * inv_mass) + gravity + state.external_force * inv_mass
        velocity = state.velocity + dt * acceleration
        position = state.position + dt * velocity

        torque = torch.stack(
            (
                state.arm_length * (thrust[:, 1] - thrust[:, 3]),
                state.arm_length * (thrust[:, 2] - thrust[:, 0]),
                p.yaw_drag * (thrust[:, 0] - thrust[:, 1] + thrust[:, 2] - thrust[:, 3]),
            ),
            dim=-1,
        )
        inertia = torch.stack(
            (
                state.inertia_x,
                state.inertia_y,
                state.inertia_z,
            ),
            dim=-1,
        )
        omega_cross = torch.cross(state.omega, state.omega * inertia, dim=-1)
        omega = state.omega + dt * ((torque - omega_cross) / inertia)

        rotation = state.rotation @ _so3_exp(dt * omega)

        next_state = L2FState(
            position,
            velocity,
            rotation,
            omega,
            motor,
            command,
            state.external_force,
            mass=state.mass,
            thrust_coeff_c0=state.thrust_coeff_c0,
            thrust_coeff_c1=state.thrust_coeff_c1,
            thrust_coeff_c2=state.thrust_coeff_c2,
            thrust_to_weight=state.thrust_to_weight,
            torque_to_inertia=state.torque_to_inertia,
            rotor_distance_factor=state.rotor_distance_factor,
            inertia_factor=state.inertia_factor,
            motor_time_rising=state.motor_time_rising,
            motor_time_falling=state.motor_time_falling,
            rotor_torque_constant=state.rotor_torque_constant,
            cbrt_mass=state.cbrt_mass,
            force_std=state.force_std,
            arm_length=state.arm_length,
            inertia_x=state.inertia_x,
            inertia_y=state.inertia_y,
            inertia_z=state.inertia_z,
        )
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
