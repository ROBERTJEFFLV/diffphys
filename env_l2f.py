from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


CAPABILITY_NAMES = (
    "log_thrust_to_weight",
    "log_alpha_roll",
    "log_eta_yaw",
    "log_jz_over_jxy",
    "log_tau_rise",
    "log_tau_fall",
)
# Backward-compatible normalization bounds cover both the corrected physical-fit
# sampler and historical physical checkpoints. Log-space midpoints map to zero
# and endpoints map to +/-1; these are not a recommendation for new inertia data.
CAPABILITY_BOUNDS = (
    (1.45, 5.50),
    (35.0, 2200.0),
    (0.02, 1.00),
    (1.45, 2.80),
    (0.025, 0.18),
    (0.03, 0.35),
)


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


@dataclass(frozen=True)
class L2FLossConfig:
    p_scale: float = 2.0
    v_scale: float = 3.0
    omega_scale: float = 10.0
    huber_beta: float = 1.0
    w_p: float = 1.0
    w_v: float = 0.3
    w_omega: float = 0.05


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


def normalized_capability_target(state: L2FState) -> torch.Tensor:
    """Return six privileged aggregate dynamics quantities in normalized log space."""
    values = torch.stack(
        (
            state.thrust_to_weight,
            state.alpha_roll_max,
            state.eta_yaw,
            state.jz_over_jxy,
            state.motor_time_rising,
            state.motor_time_falling,
        ),
        dim=-1,
    )
    bounds = values.new_tensor(CAPABILITY_BOUNDS)
    log_bounds = bounds.log()
    center = log_bounds.mean(dim=-1)
    half_range = 0.5 * (log_bounds[:, 1] - log_bounds[:, 0])
    return (values.clamp_min(1.0e-12).log() - center) / half_range


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
        alpha_roll_max=state.alpha_roll_max,
        alpha_pitch_max=state.alpha_pitch_max,
        alpha_yaw_max=state.alpha_yaw_max,
        eta_yaw=state.eta_yaw,
        jz_over_jxy=state.jz_over_jxy,
        dt_alpha_roll_max=state.dt_alpha_roll_max,
        dt_alpha_yaw_max=state.dt_alpha_yaw_max,
    )


def _sample_reciprocal_style_factor(unit: torch.Tensor, deviation: float) -> torch.Tensor:
    upper = max(1.0 + deviation, 1.0)
    lower = 1.0 / upper
    return lower + unit * (upper - lower)


def _stratified_unit(batch_size: int, device: torch.device, dtype: torch.dtype, bins: int, offset: int = 0) -> torch.Tensor:
    # Balance every marginal even when batch_size is smaller than a full
    # Cartesian grid; independent permutations avoid a fixed cross-variable tie.
    index = torch.randperm(batch_size, device=device)
    phase = torch.randint(0, int(bins), (1,), device=device)
    bin_id = ((index + int(offset) + phase) % int(bins)).to(dtype=dtype)
    return (bin_id + torch.rand((batch_size,), device=device, dtype=dtype)) / float(bins)


def _joint_stratified_units(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    bins: int,
    dimensions: int,
) -> tuple[torch.Tensor, ...]:
    """Stratify several roots jointly when a complete grid fits the batch.

    A 256-sample batch with four roots and four bins contains every 4^4 cell
    exactly once.  Other batch sizes retain the marginal balancing used by the
    historical sampler rather than biasing a partial Cartesian grid.
    """
    if batch_size <= 0 or bins <= 0 or dimensions <= 0:
        raise ValueError("batch_size, bins, and dimensions must be positive")
    cell_count = int(bins) ** int(dimensions)
    if batch_size % cell_count != 0:
        return tuple(
            _stratified_unit(batch_size, device, dtype, bins, offset=bins**dimension)
            for dimension in range(dimensions)
        )
    cell = torch.arange(batch_size, device=device) % cell_count
    bin_ids = torch.stack(
        tuple((cell // (bins**dimension)) % bins for dimension in range(dimensions)),
        dim=-1,
    )
    units = (
        bin_ids.to(dtype=dtype)
        + torch.rand((batch_size, dimensions), device=device, dtype=dtype)
    ) / float(bins)
    units = units[torch.randperm(batch_size, device=device)]
    return tuple(units[:, dimension] for dimension in range(dimensions))


def _range_from_unit(unit: torch.Tensor, low: torch.Tensor | float, high: torch.Tensor | float) -> torch.Tensor:
    return low + unit * (high - low)


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

    min_thrust_per_rotor = torch.clamp(thrust_coeff_c0 - thrust_coeff_c1 + thrust_coeff_c2, min=0.0)
    thrust_delta_per_rotor = torch.clamp(max_thrust_per_rotor - min_thrust_per_rotor, min=1.0e-9)
    roll_torque_max = arm_length * thrust_delta_per_rotor
    yaw_mix_thrust_max = 2.0 * thrust_delta_per_rotor
    alpha_roll_max = roll_torque_max / torch.clamp(inertia_x, min=1.0e-12)
    alpha_pitch_max = roll_torque_max / torch.clamp(inertia_y, min=1.0e-12)
    alpha_yaw_max = rotor_torque_constant * yaw_mix_thrust_max / torch.clamp(inertia_z, min=1.0e-12)
    eta_yaw = alpha_yaw_max / torch.clamp(alpha_roll_max, min=1.0e-9)
    jz_over_jxy = inertia_z / torch.clamp(inertia_x, min=1.0e-12)

    surplus = (thrust_to_weight - 1.0).clamp_min(0.0)
    multiple = sample((batch,), device=device, dtype=dtype) * (surplus * 0.3)
    force_std = multiple * thrust_to_weight * mass_new / 3.0
    external_force = normal((batch, 3), device=device, dtype=dtype) * force_std[:, None]

    return L2FRaptorEpisodeDynamics(
        mass=mass_new,
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
        cbrt_mass=size_new,
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


def sample_physical_broad_episode_dynamics(
    nominal: L2FParams,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> L2FRaptorEpisodeDynamics:
    batch = batch_size
    sample = torch.rand
    normal = torch.randn
    gravity = float(nominal.gravity)
    nominal_mass = torch.full((batch,), float(nominal.mass), device=device, dtype=dtype)

    relative_size_min = math.pow(0.02, 1.0 / 3.0)
    relative_size_max = math.pow(5.0, 1.0 / 3.0)
    cbrt_mass = sample((batch,), device=device, dtype=dtype) * (relative_size_max - relative_size_min) + relative_size_min
    mass = torch.clamp(cbrt_mass * cbrt_mass * cbrt_mass, min=1.0e-12)
    scale_relative = (mass / nominal_mass).pow(1.0 / 3.0)
    size_factor = _sample_reciprocal_style_factor(sample((batch,), device=device, dtype=dtype), 0.1)
    rotor_distance_factor = scale_relative * size_factor
    arm_length = torch.full((batch,), float(nominal.arm_length), device=device, dtype=dtype) * rotor_distance_factor

    thrust_to_weight = sample((batch,), device=device, dtype=dtype) * (5.0 - 1.5) + 1.5
    hover_thrust_per_rotor = mass * gravity / 4.0
    max_thrust_per_rotor = thrust_to_weight * mass * gravity / 4.0
    min_thrust_per_rotor = torch.clamp(2.0 * hover_thrust_per_rotor - max_thrust_per_rotor, min=0.0)
    thrust_delta_per_rotor = torch.clamp(max_thrust_per_rotor - min_thrust_per_rotor, min=1.0e-9)
    thrust_coeff_c0 = hover_thrust_per_rotor
    thrust_coeff_c1 = max_thrust_per_rotor - hover_thrust_per_rotor
    thrust_coeff_c2 = torch.zeros((batch,), device=device, dtype=dtype)

    alpha_roll_max = sample((batch,), device=device, dtype=dtype) * (600.0 - 40.0) + 40.0
    alpha_pitch_max = alpha_roll_max
    eta_yaw = sample((batch,), device=device, dtype=dtype) * (0.15 - 0.05) + 0.05
    jz_over_jxy = sample((batch,), device=device, dtype=dtype) * (2.8 - 1.5) + 1.5
    alpha_yaw_max = eta_yaw * alpha_roll_max

    roll_torque_max = arm_length * thrust_delta_per_rotor
    inertia_x = roll_torque_max / torch.clamp(alpha_roll_max, min=1.0e-9)
    inertia_y = roll_torque_max / torch.clamp(alpha_pitch_max, min=1.0e-9)
    inertia_z = jz_over_jxy * inertia_x
    yaw_mix_thrust_max = 2.0 * thrust_delta_per_rotor
    rotor_torque_constant = alpha_yaw_max * inertia_z / torch.clamp(yaw_mix_thrust_max, min=1.0e-9)

    torque_to_inertia = alpha_roll_max
    inertia_factor = torch.full((batch,), float(nominal.inertia_x), device=device, dtype=dtype) / torch.clamp(inertia_x, min=1.0e-12)
    motor_time_rising = sample((batch,), device=device, dtype=dtype) * (0.10 - 0.03) + 0.03
    motor_time_falling = sample((batch,), device=device, dtype=dtype) * (0.30 - 0.03) + 0.03

    surplus = (thrust_to_weight - 1.0).clamp_min(0.0)
    multiple = sample((batch,), device=device, dtype=dtype) * (surplus * 0.3)
    force_std = multiple * thrust_to_weight * mass / 3.0
    external_force = normal((batch, 3), device=device, dtype=dtype) * force_std[:, None]

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


def sample_physical_fit_episode_dynamics(
    nominal: L2FParams,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    balanced: bool = False,
) -> L2FRaptorEpisodeDynamics:
    """Physically fitted broad sampler.

    Root variables are mass, thrust reserve, inertia coefficient, yaw torque
    constant, and motor time constant. Arm length, inertia, roll authority, and
    yaw authority are derived from rigid-body relationships.
    """
    batch = batch_size
    gravity = float(nominal.gravity)
    bins = 4

    if balanced:
        u_mass, u_tw, u_k, u_tau = _joint_stratified_units(
            batch,
            device,
            dtype,
            bins=bins,
            dimensions=4,
        )
    else:
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


def _orthonormalize(rotation: torch.Tensor) -> torch.Tensor:
    x = F.normalize(rotation[:, :, 0], dim=-1, eps=1.0e-6)
    y = rotation[:, :, 1]
    y = y - (x * y).sum(dim=-1, keepdim=True) * x
    y = F.normalize(y, dim=-1, eps=1.0e-6)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


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


def _solve_implicit_midpoint_omega(
    omega: torch.Tensor,
    torque: torch.Tensor,
    inertia: torch.Tensor,
    dt: float,
    iterations: int = 4,
) -> torch.Tensor:
    gyro = torch.cross(omega, omega * inertia, dim=-1)
    omega_next = omega + float(dt) * ((torque - gyro) / torch.clamp(inertia, min=1.0e-12))
    identity_eps = 1.0e-9 * torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[0], 3, 3)
    for _ in range(iterations):
        omega_mid = 0.5 * (omega + omega_next)
        residual = _implicit_midpoint_residual(omega_next, omega, torque, inertia, dt)
        jacobian = _implicit_midpoint_matrix(omega_mid, inertia, dt) + identity_eps
        delta = torch.linalg.solve(jacobian, residual[:, :, None]).squeeze(-1)
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
    ) -> torch.Tensor:
        omega_next = _solve_implicit_midpoint_omega(omega, torque, inertia, float(dt))
        ctx.dt = float(dt)
        ctx.save_for_backward(omega, omega_next, inertia)
        return omega_next

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        omega, omega_next, inertia = ctx.saved_tensors
        dt = ctx.dt
        omega_mid = 0.5 * (omega + omega_next)
        gyro_jac = _gyro_jacobian(omega_mid, inertia)
        jacobian = _implicit_midpoint_matrix(omega_mid, inertia, dt)
        lambda_vec = torch.linalg.solve(
            jacobian.transpose(-1, -2),
            grad_output[:, :, None],
        ).squeeze(-1)
        inv_inertia_lambda = lambda_vec / torch.clamp(inertia, min=1.0e-12)
        gyro_term = torch.matmul(
            gyro_jac.transpose(-1, -2),
            inv_inertia_lambda[:, :, None],
        ).squeeze(-1)
        grad_omega = lambda_vec - (0.5 * dt) * gyro_term
        grad_torque = dt * inv_inertia_lambda
        return grad_omega, grad_torque, None, None


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
        broad_sampler: str = "legacy",
        balanced_dynamics_sampling: bool = False,
        sample_external_force: bool = True,
    ) -> L2FState:
        p = self.params
        device = torch.device(device)
        if sample_dynamics and sampled_dynamics_level != "broad":
            raise ValueError(
                "sampled dynamics levels 'small' and 'medium' are not implemented; "
                "use --sampled-dynamics-level broad or fixed dynamics"
            )
        if balanced_dynamics_sampling and broad_sampler != "physical-fit":
            raise ValueError(
                "balanced dynamics sampling is currently implemented only for physical-fit"
            )
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
            if broad_sampler == "legacy":
                dynamics = sample_raptor_episode_dynamics(
                    p,
                    batch_size,
                    device=device,
                    dtype=dtype,
                )
            elif broad_sampler == "physical":
                dynamics = sample_physical_broad_episode_dynamics(
                    p,
                    batch_size,
                    device=device,
                    dtype=dtype,
                )
            elif broad_sampler == "physical-fit":
                dynamics = sample_physical_fit_episode_dynamics(
                    p,
                    batch_size,
                    device=device,
                    dtype=dtype,
                    balanced=balanced_dynamics_sampling,
                )
            else:
                raise ValueError("broad_sampler must be 'legacy', 'physical', or 'physical-fit'")
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
        omega = _ImplicitMidpointOmega.apply(state.omega, torque, inertia, dt)
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
        return decay_state_gradient(next_state, grad_decay)

    def tracking_components(
        self,
        state: L2FState,
        config: L2FLossConfig,
    ) -> dict[str, torch.Tensor]:
        p_error = state.position / config.p_scale
        v_error = state.velocity / config.v_scale
        omega_error = state.omega / config.omega_scale
        return {
            "position": config.w_p * _smooth_l1_sum(p_error, config.huber_beta),
            "velocity": config.w_v * _smooth_l1_sum(v_error, config.huber_beta),
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
        return {
            "position": 8.0 * state.position.square().sum(dim=-1).mean(),
            "velocity": 1.0 * state.velocity.square().sum(dim=-1).mean(),
            "omega": 0.12 * state.omega.square().sum(dim=-1).mean(),
            "motor": 0.01 * action.square().mean(),
            "smooth": 0.04 * action_delta.square().mean(),
        }
