from __future__ import annotations

from contextlib import contextmanager

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
    external_force: torch.Tensor
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
    alpha_roll_max: torch.Tensor
    alpha_pitch_max: torch.Tensor
    alpha_yaw_max: torch.Tensor
    eta_yaw: torch.Tensor
    jz_over_jxy: torch.Tensor
    dt_alpha_roll_max: torch.Tensor
    dt_alpha_yaw_max: torch.Tensor


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
    alpha_roll_max: torch.Tensor
    alpha_pitch_max: torch.Tensor
    alpha_yaw_max: torch.Tensor
    eta_yaw: torch.Tensor
    jz_over_jxy: torch.Tensor
    dt_alpha_roll_max: torch.Tensor
    dt_alpha_yaw_max: torch.Tensor


def _sample_reciprocal_style_factor(unit: torch.Tensor, deviation: float) -> torch.Tensor:
    upper = max(1.0 + deviation, 1.0)
    lower = 1.0 / upper
    return lower + unit * (upper - lower)


def _range_from_unit(unit: torch.Tensor, low: torch.Tensor | float, high: torch.Tensor | float) -> torch.Tensor:
    return low + unit * (high - low)


def sample_physical_fit_episode_dynamics(
    nominal: L2FParams,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> L2FRaptorEpisodeDynamics:
    """Physically fitted broad sampler.

    Root variables are mass, thrust reserve, inertia coefficient, yaw torque
    constant, and motor time constant. Arm length, inertia, roll authority, and
    yaw authority are derived from rigid-body relationships.
    """
    batch = batch_size
    gravity = float(nominal.gravity)
    u_mass = torch.rand((batch,), device=device, dtype=dtype)
    u_tw = torch.rand((batch,), device=device, dtype=dtype)
    u_k = torch.rand((batch,), device=device, dtype=dtype)
    u_tau = torch.rand((batch,), device=device, dtype=dtype)
    u_misc = torch.rand((batch,), device=device, dtype=dtype)
    u_misc2 = torch.rand((batch,), device=device, dtype=dtype)
    u_misc3 = torch.rand((batch,), device=device, dtype=dtype)

    mass_min = torch.tensor(0.02, device=device, dtype=dtype)
    mass_max = torch.tensor(5.00, device=device, dtype=dtype)
    log_mass = torch.log(mass_min) + u_mass * (torch.log(mass_max) - torch.log(mass_min))
    mass = torch.exp(log_mass).clamp_min(1.0e-12)
    cbrt_mass = mass.pow(1.0 / 3.0)

    arm_center = 0.21257 * mass.pow(0.5028)
    arm_length = arm_center * _sample_reciprocal_style_factor(u_misc, 0.20)
    arm_length = arm_length.clamp(0.028, 0.50)
    rotor_distance_factor = arm_length / float(nominal.arm_length)

    thrust_to_weight_center = 3.19478 * mass.pow(0.09285)
    thrust_to_weight_low = torch.clamp(thrust_to_weight_center / 1.55, min=1.45, max=5.50)
    thrust_to_weight_high = torch.clamp(thrust_to_weight_center * 1.70, min=1.45, max=5.50)
    thrust_to_weight = _range_from_unit(u_tw, thrust_to_weight_low, thrust_to_weight_high).clamp(1.45, 5.50)

    hover_thrust_per_rotor = mass * gravity / 4.0
    max_thrust_per_rotor = thrust_to_weight * mass * gravity / 4.0
    min_thrust_per_rotor = torch.clamp(2.0 * hover_thrust_per_rotor - max_thrust_per_rotor, min=0.0)
    thrust_delta_per_rotor = torch.clamp(max_thrust_per_rotor - min_thrust_per_rotor, min=1.0e-9)
    thrust_coeff_c0 = hover_thrust_per_rotor
    thrust_coeff_c1 = max_thrust_per_rotor - hover_thrust_per_rotor
    thrust_coeff_c2 = torch.zeros((batch,), device=device, dtype=dtype)

    kxy_center = 0.19013 * mass.pow(0.23165)
    kxy = kxy_center * _sample_reciprocal_style_factor(u_k, 0.50)
    kxy = kxy.clamp(0.045, 0.50)
    # Principal moments of every rigid body satisfy Jz <= Jx + Jy.  This
    # sampler uses Jx == Jy, so keeping Jz/Jxy below two is mandatory.  The
    # tighter upper end still covers the named 50 g, 150 g, and X500 anchors.
    jz_over_jxy = _range_from_unit(u_misc2, 1.45, 1.95)
    inertia_x = torch.clamp(kxy * mass * arm_length * arm_length, min=1.0e-12)
    inertia_y = inertia_x
    inertia_z = jz_over_jxy * inertia_x

    alpha_raw = arm_length * thrust_delta_per_rotor / torch.clamp(inertia_x, min=1.0e-12)
    alpha_roll_max = alpha_raw.clamp(35.0, 2200.0)
    inertia_scale = alpha_raw / torch.clamp(alpha_roll_max, min=1.0e-9)
    inertia_x = inertia_x * inertia_scale
    inertia_y = inertia_y * inertia_scale
    inertia_z = inertia_z * inertia_scale
    alpha_pitch_max = alpha_roll_max

    rotor_torque_constant_center = 0.01523 * mass.pow(-0.09015)
    rotor_torque_constant = rotor_torque_constant_center * _sample_reciprocal_style_factor(u_misc3, 0.45)
    rotor_torque_constant = rotor_torque_constant.clamp(0.006, 0.035)
    yaw_mix_thrust_max = 2.0 * thrust_delta_per_rotor
    alpha_yaw_raw = rotor_torque_constant * yaw_mix_thrust_max / torch.clamp(inertia_z, min=1.0e-12)
    eta_raw = alpha_yaw_raw / torch.clamp(alpha_roll_max, min=1.0e-9)
    eta_yaw = eta_raw.clamp(0.02, 1.00)
    rotor_torque_constant = rotor_torque_constant * eta_yaw / torch.clamp(eta_raw, min=1.0e-9)
    alpha_yaw_max = eta_yaw * alpha_roll_max

    motor_time_rising_center = 0.10658 * mass.pow(0.18987)
    motor_time_rising = motor_time_rising_center * _sample_reciprocal_style_factor(u_tau, 0.35)
    motor_time_rising = motor_time_rising.clamp(0.025, 0.18)
    fall_factor = _range_from_unit(torch.rand((batch,), device=device, dtype=dtype), 1.0, 2.6)
    motor_time_falling = (motor_time_rising * fall_factor).clamp(0.03, 0.35)

    disturbance_acc_std = _range_from_unit(torch.rand((batch,), device=device, dtype=dtype), 0.0, 0.35)
    disturbance_acc_std = disturbance_acc_std * torch.clamp(thrust_to_weight - 1.0, min=0.0)
    force_std = mass * disturbance_acc_std
    external_force = torch.randn((batch, 3), device=device, dtype=dtype) * force_std[:, None]

    torque_to_inertia = alpha_roll_max
    inertia_factor = torch.full((batch,), float(nominal.inertia_x), device=device, dtype=dtype) / torch.clamp(inertia_x, min=1.0e-12)

    return L2FRaptorEpisodeDynamics(
        mass=mass,
        thrust_coeff_c0=thrust_coeff_c0[:, None].expand(batch, 4),
        thrust_coeff_c1=thrust_coeff_c1[:, None].expand(batch, 4),
        thrust_coeff_c2=thrust_coeff_c2[:, None].expand(batch, 4),
        external_force=external_force,
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
        alpha_roll_max=alpha_roll_max,
        alpha_pitch_max=alpha_pitch_max,
        alpha_yaw_max=alpha_yaw_max,
        eta_yaw=eta_yaw,
        jz_over_jxy=jz_over_jxy,
        dt_alpha_roll_max=alpha_roll_max * float(nominal.dt),
        dt_alpha_yaw_max=alpha_yaw_max * float(nominal.dt),
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
    external_force = torch.zeros((batch, 3), device=device, dtype=dtype)
    max_thrust_per_rotor = thrust_coeff_c0_nom + thrust_coeff_c1_nom + thrust_coeff_c2_nom
    min_thrust_per_rotor = torch.clamp(thrust_coeff_c0_nom - thrust_coeff_c1_nom + thrust_coeff_c2_nom, min=0.0)
    thrust_delta_per_rotor = torch.clamp(max_thrust_per_rotor - min_thrust_per_rotor, min=1.0e-9)
    roll_torque_max = arm_length * thrust_delta_per_rotor
    yaw_mix_thrust_max = 2.0 * thrust_delta_per_rotor
    alpha_roll_max = roll_torque_max / torch.clamp(inertia_x, min=1.0e-12)
    alpha_pitch_max = roll_torque_max / torch.clamp(inertia_y, min=1.0e-12)
    alpha_yaw_max = rotor_torque_constant * yaw_mix_thrust_max / torch.clamp(inertia_z, min=1.0e-12)
    eta_yaw = alpha_yaw_max / torch.clamp(alpha_roll_max, min=1.0e-9)
    jz_over_jxy = inertia_z / torch.clamp(inertia_x, min=1.0e-12)
    return L2FRaptorEpisodeDynamics(
        mass=mass,
        thrust_coeff_c0=thrust_coeff_c0_nom[:, None].expand(batch, 4),
        thrust_coeff_c1=thrust_coeff_c1_nom[:, None].expand(batch, 4),
        thrust_coeff_c2=thrust_coeff_c2_nom[:, None].expand(batch, 4),
        external_force=external_force,
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
        alpha_roll_max=alpha_roll_max,
        alpha_pitch_max=alpha_pitch_max,
        alpha_yaw_max=alpha_yaw_max,
        eta_yaw=eta_yaw,
        jz_over_jxy=jz_over_jxy,
        dt_alpha_roll_max=alpha_roll_max * float(nominal.dt),
        dt_alpha_yaw_max=alpha_yaw_max * float(nominal.dt),
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


def _so3_exp(phi: torch.Tensor) -> torch.Tensor:
    theta_sq = phi.square().sum(dim=-1, keepdim=True)
    # Guard before sqrt: masking its result with where still permits 0 * inf
    # in backward at phi=0. Keep theta_sq unchanged for the small-angle series.
    theta = torch.sqrt(theta_sq.clamp_min(torch.finfo(phi.dtype).tiny))
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


def _gyro_jacobian(omega: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    wx, wy, wz = omega.unbind(dim=-1)
    jx, jy, jz = inertia.unbind(dim=-1)
    zeros = torch.zeros_like(wx)
    return torch.stack(
        (
            zeros,
            (jz - jy) * wz,
            (jz - jy) * wy,
            (jx - jz) * wz,
            zeros,
            (jx - jz) * wx,
            (jy - jx) * wy,
            (jy - jx) * wx,
            zeros,
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _implicit_midpoint_matrix(omega_mid: torch.Tensor, inertia: torch.Tensor, dt: float) -> torch.Tensor:
    identity = torch.eye(3, device=omega_mid.device, dtype=omega_mid.dtype).expand(omega_mid.shape[0], 3, 3)
    gyro_jac = _gyro_jacobian(omega_mid, inertia)
    return identity + (0.5 * float(dt)) * gyro_jac / torch.clamp(inertia[:, :, None], min=1.0e-12)


def _implicit_midpoint_residual(
    omega_next: torch.Tensor,
    omega: torch.Tensor,
    torque: torch.Tensor,
    inertia: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    omega_mid = 0.5 * (omega + omega_next)
    gyro_mid = torch.cross(omega_mid, omega_mid * inertia, dim=-1)
    return omega_next - omega - float(dt) * ((torque - gyro_mid) / torch.clamp(inertia, min=1.0e-12))


class _SolveErrors:
    """Retain device-side LAPACK status until the surrounding update completes."""

    def __init__(self):
        self.infos = []
        self.active = True

    def check(self):
        if self.infos and bool(torch.cat([info.reshape(-1) for info in self.infos]).any()):
            raise FloatingPointError("implicit midpoint solve reported a nonzero LAPACK info")
        self.infos.clear()


def _solve(matrix, rhs, errors):
    solution, info = torch.linalg.solve_ex(matrix, rhs, check_errors=False)
    if errors is None or not errors.active:
        if bool(info.any()):
            raise FloatingPointError("implicit midpoint solve reported a nonzero LAPACK info")
    else:
        errors.infos.append(info)
    return solution


def _solve_implicit_midpoint_omega(
    omega: torch.Tensor,
    torque: torch.Tensor,
    inertia: torch.Tensor,
    dt: float,
    iterations: int = 4,
    errors: _SolveErrors | None = None,
) -> torch.Tensor:
    gyro = torch.cross(omega, omega * inertia, dim=-1)
    omega_next = omega + float(dt) * ((torque - gyro) / torch.clamp(inertia, min=1.0e-12))
    identity_eps = 1.0e-9 * torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[0], 3, 3)
    for _ in range(iterations):
        omega_mid = 0.5 * (omega + omega_next)
        residual = _implicit_midpoint_residual(omega_next, omega, torque, inertia, dt)
        jacobian = _implicit_midpoint_matrix(omega_mid, inertia, dt) + identity_eps
        delta = _solve(jacobian, residual[:, :, None], errors).squeeze(-1)
        omega_next = omega_next - delta
    return omega_next


class _ImplicitMidpointOmega(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        omega: torch.Tensor,
        torque: torch.Tensor,
        inertia: torch.Tensor,
        dt: float,
        errors: _SolveErrors | None,
    ) -> torch.Tensor:
        omega_next = _solve_implicit_midpoint_omega(omega, torque, inertia, float(dt), errors=errors)
        ctx.dt = float(dt)
        ctx.errors = errors
        ctx.save_for_backward(omega, omega_next, inertia)
        return omega_next

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        omega, omega_next, inertia = ctx.saved_tensors
        dt = ctx.dt
        omega_mid = 0.5 * (omega + omega_next)
        gyro_jac = _gyro_jacobian(omega_mid, inertia)
        jacobian = _implicit_midpoint_matrix(omega_mid, inertia, dt)
        lambda_vec = _solve(
            jacobian.transpose(-1, -2),
            grad_output[:, :, None],
            ctx.errors,
        ).squeeze(-1)
        inv_inertia_lambda = lambda_vec / torch.clamp(inertia, min=1.0e-12)
        gyro_term = torch.matmul(
            gyro_jac.transpose(-1, -2),
            inv_inertia_lambda[:, :, None],
        ).squeeze(-1)
        grad_omega = lambda_vec - (0.5 * dt) * gyro_term
        grad_torque = dt * inv_inertia_lambda
        return grad_omega, grad_torque, None, None, None


class L2FSimulator:
    """Differentiable L2F-style quadrotor simulator using CUDA tensors when available."""

    def __init__(self, params: L2FParams | None = None) -> None:
        self.params = params or L2FParams()
        self._solve_errors = None

    @contextmanager
    def defer_solve_errors(self):
        """Scope both forward and backward; check all statuses before Adam."""
        if self._solve_errors is not None:
            raise RuntimeError("nested deferred solve checks are not supported")
        errors = _SolveErrors()
        self._solve_errors = errors
        try:
            yield
            errors.check()
        finally:
            errors.active = False
            errors.infos.clear()
            self._solve_errors = None

    def reset(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        sample_dynamics: bool = False,
        sample_external_force: bool = True,
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

        if sample_dynamics:
            dynamics = sample_physical_fit_episode_dynamics(
                p,
                batch_size,
                device=device,
                dtype=dtype,
            )
            external_force = dynamics.external_force
            if not sample_external_force:
                external_force = torch.zeros_like(external_force)
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
            alpha_roll_max=dynamics.alpha_roll_max,
            alpha_pitch_max=dynamics.alpha_pitch_max,
            alpha_yaw_max=dynamics.alpha_yaw_max,
            eta_yaw=dynamics.eta_yaw,
            jz_over_jxy=dynamics.jz_over_jxy,
            dt_alpha_roll_max=dynamics.dt_alpha_roll_max,
            dt_alpha_yaw_max=dynamics.dt_alpha_yaw_max,
        )

    def step(
        self,
        state: L2FState,
        action: torch.Tensor,
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
                state.rotor_torque_constant * (thrust[:, 0] - thrust[:, 1] + thrust[:, 2] - thrust[:, 3]),
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
        omega = _ImplicitMidpointOmega.apply(state.omega, torque, inertia, dt, self._solve_errors)
        omega_mid = 0.5 * (state.omega + omega)

        rotation = state.rotation @ _so3_exp(dt * omega_mid)

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
            alpha_roll_max=state.alpha_roll_max,
            alpha_pitch_max=state.alpha_pitch_max,
            alpha_yaw_max=state.alpha_yaw_max,
            eta_yaw=state.eta_yaw,
            jz_over_jxy=state.jz_over_jxy,
            dt_alpha_roll_max=state.dt_alpha_roll_max,
            dt_alpha_yaw_max=state.dt_alpha_yaw_max,
        )
        return next_state
