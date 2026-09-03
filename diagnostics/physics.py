from __future__ import annotations

from dataclasses import dataclass

import torch

from env_l2f import L2FParams, L2FState


@dataclass(frozen=True)
class BranchCounterfactuals:
    """Motor commands for the four required branch counterfactuals."""

    main_only: torch.Tensor
    main_integral: torch.Tensor
    main_damping: torch.Tensor
    full: torch.Tensor
    main_logits: torch.Tensor
    integral_logits: torch.Tensor
    damping_logits: torch.Tensor

    @property
    def full_local_logit_gain(self) -> torch.Tensor:
        """Diagonal of d tanh(logits) / d logits at the deployed action."""

        return 1.0 - self.full.square()


@dataclass(frozen=True)
class BranchWrenchDecomposition:
    """Next-step wrench counterfactuals in ``[T,tau_x,tau_y,tau_z]`` order."""

    main_only: torch.Tensor
    main_integral: torch.Tensor
    main_damping: torch.Tensor
    full: torch.Tensor
    integral_delta: torch.Tensor
    damping_delta: torch.Tensor
    damping_delta_after_integral: torch.Tensor
    full_delta: torch.Tensor
    counterfactuals: BranchCounterfactuals
    main_only_motor: torch.Tensor
    main_integral_motor: torch.Tensor
    main_damping_motor: torch.Tensor
    full_motor: torch.Tensor
    main_only_thrust: torch.Tensor
    main_integral_thrust: torch.Tensor
    main_damping_thrust: torch.Tensor
    full_thrust: torch.Tensor


@dataclass(frozen=True)
class SteadyStateFeasibility:
    """Analytic constant-force trim for the symmetric four-rotor mixer."""

    required_body_z_world: torch.Tensor
    required_total_thrust: torch.Tensor
    required_thrust_ratio: torch.Tensor
    required_tilt_rad: torch.Tensor
    required_motor_thrust: torch.Tensor
    required_motor_command: torch.Tensor
    min_motor_thrust: torch.Tensor
    max_motor_thrust: torch.Tensor
    lower_command_headroom: torch.Tensor
    upper_command_headroom: torch.Tensor
    required_trim_ratio: torch.Tensor
    feasible: torch.Tensor
    numerical_failure: torch.Tensor
    lower_thrust_violation: torch.Tensor
    upper_thrust_violation: torch.Tensor


def command_to_next_motor(
    state: L2FState,
    action: torch.Tensor,
    *,
    dt: float,
) -> torch.Tensor:
    """Apply the exact legacy clamp and asymmetric first-order motor lag."""

    if action.shape != state.motor.shape:
        raise ValueError("action must match state.motor shape")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    command = action.clamp(-1.0, 1.0)
    rise = state.motor_time_rising[:, None]
    fall = state.motor_time_falling[:, None]
    tau = torch.where(command >= state.motor, rise, fall)
    alpha = (float(dt) / tau).clamp(0.0, 1.0)
    return state.motor + alpha * (command - state.motor)


def motor_to_thrust(state: L2FState, motor: torch.Tensor) -> torch.Tensor:
    """Evaluate the exact simulator thrust polynomial and non-negative clamp."""

    if motor.shape != state.motor.shape:
        raise ValueError("motor must match state.motor shape")
    thrust = (
        state.thrust_coeff_c0
        + state.thrust_coeff_c1 * motor
        + state.thrust_coeff_c2 * motor.square()
    )
    return thrust.clamp_min(0.0)


def thrust_to_wrench(state: L2FState, thrust: torch.Tensor) -> torch.Tensor:
    """Map rotor thrust to the simulator body wrench ``[T,tau_x,tau_y,tau_z]``."""

    if thrust.shape != state.motor.shape or thrust.shape[-1] != 4:
        raise ValueError("thrust must have shape [batch,4]")
    total = thrust.sum(dim=-1)
    torque = torch.stack(
        (
            state.arm_length * (thrust[:, 1] - thrust[:, 3]),
            state.arm_length * (thrust[:, 2] - thrust[:, 0]),
            state.rotor_torque_constant
            * (thrust[:, 0] - thrust[:, 1] + thrust[:, 2] - thrust[:, 3]),
        ),
        dim=-1,
    )
    return torch.cat((total[:, None], torque), dim=-1)


def action_to_next_wrench(
    state: L2FState,
    action: torch.Tensor,
    *,
    dt: float,
) -> torch.Tensor:
    motor = command_to_next_motor(state, action, dt=dt)
    return thrust_to_wrench(state, motor_to_thrust(state, motor))


def branch_counterfactual_actions(
    auxiliary: dict[str, torch.Tensor],
) -> BranchCounterfactuals:
    """Reconstruct all branch counterfactuals from one recurrent forward pass."""

    required = (
        "main_logits",
        "integral_residual_logits",
        "damping_residual_logits",
    )
    missing = [name for name in required if name not in auxiliary]
    if missing:
        raise KeyError(f"policy auxiliary output is missing branch logits: {missing}")
    main = auxiliary["main_logits"]
    integral = auxiliary["integral_residual_logits"]
    damping = auxiliary["damping_residual_logits"]
    return BranchCounterfactuals(
        main_only=torch.tanh(main),
        main_integral=torch.tanh(main + integral),
        main_damping=torch.tanh(main + damping),
        full=torch.tanh(main + integral + damping),
        main_logits=main,
        integral_logits=integral,
        damping_logits=damping,
    )


def branch_wrench_decomposition(
    state: L2FState,
    auxiliary: dict[str, torch.Tensor],
    *,
    dt: float,
) -> BranchWrenchDecomposition:
    """Measure branch effects after tanh, motor lag, thrust curve, and mixer."""

    actions = branch_counterfactual_actions(auxiliary)
    main_motor = command_to_next_motor(state, actions.main_only, dt=dt)
    integral_motor = command_to_next_motor(state, actions.main_integral, dt=dt)
    damping_motor = command_to_next_motor(state, actions.main_damping, dt=dt)
    full_motor = command_to_next_motor(state, actions.full, dt=dt)
    main_thrust = motor_to_thrust(state, main_motor)
    integral_thrust = motor_to_thrust(state, integral_motor)
    damping_thrust = motor_to_thrust(state, damping_motor)
    full_thrust = motor_to_thrust(state, full_motor)
    main_wrench = thrust_to_wrench(state, main_thrust)
    integral_wrench = thrust_to_wrench(state, integral_thrust)
    damping_wrench = thrust_to_wrench(state, damping_thrust)
    full_wrench = thrust_to_wrench(state, full_thrust)
    return BranchWrenchDecomposition(
        main_only=main_wrench,
        main_integral=integral_wrench,
        main_damping=damping_wrench,
        full=full_wrench,
        integral_delta=integral_wrench - main_wrench,
        damping_delta=damping_wrench - main_wrench,
        damping_delta_after_integral=full_wrench - integral_wrench,
        full_delta=full_wrench - main_wrench,
        counterfactuals=actions,
        main_only_motor=main_motor,
        main_integral_motor=integral_motor,
        main_damping_motor=damping_motor,
        full_motor=full_motor,
        main_only_thrust=main_thrust,
        main_integral_thrust=integral_thrust,
        main_damping_thrust=damping_thrust,
        full_thrust=full_thrust,
    )


def _thrust_extrema(state: L2FState) -> tuple[torch.Tensor, torch.Tensor]:
    """Return polynomial thrust extrema over command [-1,1], including vertices."""

    candidates = [
        motor_to_thrust(state, torch.full_like(state.motor, -1.0)),
        motor_to_thrust(state, torch.full_like(state.motor, 1.0)),
    ]
    c2 = state.thrust_coeff_c2
    nonzero = c2.abs() > torch.finfo(c2.dtype).eps
    vertex = torch.where(
        nonzero,
        -state.thrust_coeff_c1 / (2.0 * torch.where(nonzero, c2, torch.ones_like(c2))),
        torch.zeros_like(c2),
    )
    vertex_valid = nonzero & (vertex >= -1.0) & (vertex <= 1.0)
    vertex_thrust = motor_to_thrust(state, vertex.clamp(-1.0, 1.0))
    candidates.append(
        torch.where(vertex_valid, vertex_thrust, torch.full_like(vertex_thrust, float("nan")))
    )
    stacked = torch.stack(candidates, dim=0)
    minimum = torch.nan_to_num(stacked, nan=float("inf")).amin(dim=0)
    maximum = torch.nan_to_num(stacked, nan=-float("inf")).amax(dim=0)
    return minimum, maximum


def _invert_thrust_polynomial(
    state: L2FState,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert each rotor polynomial, preferring the in-range root nearest hover."""

    if target.shape != state.motor.shape:
        raise ValueError("target thrust must have shape [batch,4]")
    c0 = state.thrust_coeff_c0
    c1 = state.thrust_coeff_c1
    c2 = state.thrust_coeff_c2
    eps = torch.finfo(target.dtype).eps * 32.0
    linear = c2.abs() <= eps
    linear_valid = linear & (c1.abs() > eps)
    linear_root = (target - c0) / torch.where(linear_valid, c1, torch.ones_like(c1))

    discriminant = c1.square() - 4.0 * c2 * (c0 - target)
    quadratic_valid = (~linear) & (discriminant >= 0.0)
    sqrt_discriminant = discriminant.clamp_min(0.0).sqrt()
    denominator = 2.0 * torch.where(~linear, c2, torch.ones_like(c2))
    root_a = (-c1 + sqrt_discriminant) / denominator
    root_b = (-c1 - sqrt_discriminant) / denominator
    root_a_valid = quadratic_valid & (root_a >= -1.0 - 1.0e-6) & (root_a <= 1.0 + 1.0e-6)
    root_b_valid = quadratic_valid & (root_b >= -1.0 - 1.0e-6) & (root_b <= 1.0 + 1.0e-6)
    choose_a = root_a_valid & (~root_b_valid | (root_a.abs() <= root_b.abs()))
    quadratic_root = torch.where(choose_a, root_a, root_b)
    quadratic_has_root = root_a_valid | root_b_valid

    command = torch.where(linear, linear_root, quadratic_root)
    valid = (linear_valid & (linear_root >= -1.0 - 1.0e-6) & (linear_root <= 1.0 + 1.0e-6))
    valid = valid | quadratic_has_root
    command = torch.where(valid, command.clamp(-1.0, 1.0), torch.full_like(command, float("nan")))
    return command, valid


def steady_state_feasibility(
    state: L2FState,
    params: L2FParams,
) -> SteadyStateFeasibility:
    """Solve the yaw-free constant-force trim and classify actuator feasibility.

    This is an analytic feasibility calculation, not a controller evaluation.
    The symmetric legacy mixer has zero steady torque at equal rotor thrust.
    Motor lag does not change a constant command's equilibrium.
    """

    gravity_force = torch.zeros_like(state.external_force)
    gravity_force[:, 2] = state.mass * float(params.gravity)
    required_vector = gravity_force - state.external_force
    required_total = torch.linalg.vector_norm(required_vector, dim=-1)
    finite_required = torch.isfinite(required_vector).all(dim=-1) & torch.isfinite(required_total)
    nonzero_required = required_total > torch.finfo(required_total.dtype).eps
    body_z = required_vector / required_total.clamp_min(torch.finfo(required_total.dtype).eps)[:, None]
    body_z = torch.where(
        nonzero_required[:, None],
        body_z,
        body_z.new_tensor((0.0, 0.0, 1.0)).expand_as(body_z),
    )
    tilt = torch.acos(body_z[:, 2].clamp(-1.0, 1.0))
    required_motor_thrust = required_total[:, None].expand(-1, 4) / 4.0
    minimum, maximum = _thrust_extrema(state)
    tolerance = 1.0e-6 * maximum.abs().clamp_min(1.0)
    lower_violation = required_motor_thrust < minimum - tolerance
    upper_violation = required_motor_thrust > maximum + tolerance
    command, root_valid = _invert_thrust_polynomial(state, required_motor_thrust)
    numerical_failure = finite_required & (~root_valid.all(dim=-1)) & (~lower_violation.any(dim=-1)) & (~upper_violation.any(dim=-1))
    feasible = (
        finite_required
        & root_valid.all(dim=-1)
        & (~lower_violation.any(dim=-1))
        & (~upper_violation.any(dim=-1))
    )
    max_total = maximum.sum(dim=-1)
    return SteadyStateFeasibility(
        required_body_z_world=body_z,
        required_total_thrust=required_total,
        required_thrust_ratio=required_total / (state.mass * float(params.gravity)).clamp_min(1.0e-12),
        required_tilt_rad=tilt,
        required_motor_thrust=required_motor_thrust,
        required_motor_command=command,
        min_motor_thrust=minimum,
        max_motor_thrust=maximum,
        lower_command_headroom=command + 1.0,
        upper_command_headroom=1.0 - command,
        required_trim_ratio=required_total / max_total.clamp_min(1.0e-12),
        feasible=feasible,
        numerical_failure=numerical_failure | (~finite_required),
        lower_thrust_violation=lower_violation,
        upper_thrust_violation=upper_violation,
    )
