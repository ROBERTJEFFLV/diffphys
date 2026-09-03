from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from env_l2f import L2FState, L2FSimulator, L2FParams
from model import MotorGRUPolicy
from policy_observation import (
    PolicyObservationState,
    build_policy_observation,
    initial_observation_state,
    mode_from_observation_dim,
    update_position_integral,
)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_seed_list(seed_arg: str, seed: int, count: int) -> list[int]:
    if seed_arg.strip():
        return [int(part.strip()) for part in seed_arg.split(",") if part.strip()]
    return [seed + offset for offset in range(max(count, 1))]


def _parse_controllers(controller_arg: str) -> list[str]:
    controllers = []
    for item in controller_arg.split(","):
        name = item.strip().lower()
        if name:
            controllers.append(name)
    if not controllers:
        raise ValueError("--controllers must include at least one entry")
    return controllers


def _clone_state(state: L2FState) -> L2FState:
    return L2FState(
        position=state.position.clone(),
        velocity=state.velocity.clone(),
        rotation=state.rotation.clone(),
        omega=state.omega.clone(),
        motor=state.motor.clone(),
        previous_action=state.previous_action.clone(),
        external_force=state.external_force.clone(),
        mass=state.mass.clone(),
        thrust_coeff_c0=state.thrust_coeff_c0.clone(),
        thrust_coeff_c1=state.thrust_coeff_c1.clone(),
        thrust_coeff_c2=state.thrust_coeff_c2.clone(),
        thrust_to_weight=state.thrust_to_weight.clone(),
        torque_to_inertia=state.torque_to_inertia.clone(),
        rotor_distance_factor=state.rotor_distance_factor.clone(),
        inertia_factor=state.inertia_factor.clone(),
        motor_time_rising=state.motor_time_rising.clone(),
        motor_time_falling=state.motor_time_falling.clone(),
        rotor_torque_constant=state.rotor_torque_constant.clone(),
        cbrt_mass=state.cbrt_mass.clone(),
        force_std=state.force_std.clone(),
        arm_length=state.arm_length.clone(),
        inertia_x=state.inertia_x.clone(),
        inertia_y=state.inertia_y.clone(),
        inertia_z=state.inertia_z.clone(),
        alpha_roll_max=state.alpha_roll_max.clone(),
        alpha_pitch_max=state.alpha_pitch_max.clone(),
        alpha_yaw_max=state.alpha_yaw_max.clone(),
        eta_yaw=state.eta_yaw.clone(),
        jz_over_jxy=state.jz_over_jxy.clone(),
        dt_alpha_roll_max=state.dt_alpha_roll_max.clone(),
        dt_alpha_yaw_max=state.dt_alpha_yaw_max.clone(),
    )


def _matrix_to_euler_zyx(rotation: torch.Tensor) -> torch.Tensor:
    rotation = rotation.to(torch.float64)
    roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = -torch.asin(torch.clamp(rotation[:, 2, 0], min=-1.0, max=1.0))
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return torch.stack((roll, pitch, yaw), dim=-1)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _thrust_bounds(c0: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    minus_action = -torch.ones_like(c1)
    plus_action = torch.ones_like(c1)
    t_minus = c0 + c1 * minus_action + c2 * (minus_action * minus_action)
    t_plus = c0 + c1 * plus_action + c2 * (plus_action * plus_action)
    return torch.minimum(t_minus, t_plus), torch.maximum(t_minus, t_plus)


def _thrust_to_action(
    thrust: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
) -> torch.Tensor:
    min_thrust, max_thrust = _thrust_bounds(c0, c1, c2)
    thrust = torch.where(
        torch.isfinite(thrust),
        thrust,
        max_thrust,
    )
    thrust = thrust.clamp(min=min_thrust, max=max_thrust)
    eps = 1.0e-8
    quadratic = c2.abs() > 1.0e-8
    discr = c1 * c1 - 4.0 * c2 * (c0 - thrust)
    discr = torch.clamp(discr, min=0.0)
    linear_action = (thrust - c0) / (torch.where(c1.abs() > eps, c1, torch.ones_like(c1)))
    sqrt_discr = torch.sqrt(discr)
    quadratic_action = (-c1 + sqrt_discr) / (2.0 * (torch.where(torch.abs(c2) > eps, c2, torch.ones_like(c2)))
    )
    action = torch.where(quadratic, quadratic_action, linear_action)
    action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
    if not quadratic.all():
        action = torch.where(quadratic, action, linear_action)
    return action.clamp(-1.0, 1.0)


class _ControllerBase:
    name = "controller"

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        raise NotImplementedError

    def action(self, state: L2FState, *, step: int) -> torch.Tensor:
        raise NotImplementedError


class LearnedController(_ControllerBase):
    name = "learned"

    def __init__(
        self,
        policy: MotorGRUPolicy,
        sim: L2FSimulator,
        *,
        hidden_decay: float = 1.0,
    ) -> None:
        self.policy = policy
        self.sim = sim
        self.hidden_decay = float(hidden_decay)
        self.hidden: torch.Tensor | None = None
        self.observation_state: PolicyObservationState | None = None
        self.observation_mode = str(getattr(policy, "observation_mode", mode_from_observation_dim(policy.observation_dim)))
        self.integral_limit = float(getattr(policy, "integral_limit", 0.5))
        self.integral_leak = float(getattr(policy, "integral_leak", 0.0))
        self.integral_input_frame = str(getattr(policy, "integral_input_frame", "world"))

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.hidden = None
        self.hidden = torch.zeros(batch_size, self.policy.hidden_dim, device=device, dtype=dtype)
        self.observation_state = initial_observation_state(batch_size, device=device, dtype=dtype)

    def action(self, state: L2FState, *, step: int) -> torch.Tensor:  # noqa: ARG002
        if self.observation_state is None:
            raise RuntimeError("controller must be reset before action()")
        observation, observed_position = build_policy_observation(
            state,
            self.observation_state,
            mode=self.observation_mode,
            integral_input_frame=self.integral_input_frame,
        )
        action, hidden = self.policy(observation, self.hidden)
        self.hidden = hidden * self.hidden_decay
        self.observation_state = update_position_integral(
            self.observation_state,
            observed_position,
            dt=self.sim.params.dt,
            integral_limit=self.integral_limit,
            integral_leak=self.integral_leak,
        )
        return action


class PIDController(_ControllerBase):
    name = "pid"

    def __init__(
        self,
        sim: L2FSimulator,
        *,
        kp_xy: float,
        kd_xy: float,
        kp_z: float,
        kd_z: float,
        kp_att: float,
        kd_att: float,
        kd_yaw: float,
        max_tilt: float,
    ) -> None:
        self.sim = sim
        self.kp_xy = float(kp_xy)
        self.kd_xy = float(kd_xy)
        self.kp_z = float(kp_z)
        self.kd_z = float(kd_z)
        self.kp_att = float(kp_att)
        self.kd_att = float(kd_att)
        self.kd_yaw = float(kd_yaw)
        self.max_tilt = float(max_tilt)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        pass

    def action(self, state: L2FState, *, step: int) -> torch.Tensor:  # noqa: ARG002
        # Position/velocity outer loop (to world frame origin).
        gravity = torch.tensor(self.sim.params.gravity, device=state.position.device, dtype=state.position.dtype)
        pos_err = -state.position
        vel_err = -state.velocity

        z_acc_des = self.kp_z * pos_err[:, 2] + self.kd_z * vel_err[:, 2]
        x_acc_des = self.kp_xy * pos_err[:, 0] + self.kd_xy * vel_err[:, 0]
        y_acc_des = self.kp_xy * pos_err[:, 1] + self.kd_xy * vel_err[:, 1]

        total_thrust = state.mass * (gravity + z_acc_des)

        # Desired small-angle attitude command (body -> world Euler xyz convention).
        roll_des = torch.clamp(y_acc_des / gravity, min=-self.max_tilt, max=self.max_tilt)
        pitch_des = torch.clamp(x_acc_des / gravity, min=-self.max_tilt, max=self.max_tilt)
        euler = _matrix_to_euler_zyx(state.rotation)
        roll, pitch = euler[:, 0], euler[:, 1]
        roll_err = _wrap_to_pi(roll_des - roll)
        pitch_err = _wrap_to_pi(pitch_des - pitch)

        target_angular_acc = torch.stack(
            (
                self.kp_att * roll_err - self.kd_att * state.omega[:, 0],
                self.kp_att * pitch_err - self.kd_att * state.omega[:, 1],
                -self.kd_yaw * state.omega[:, 2],
            ),
            dim=-1,
        )
        inertia = torch.stack((state.inertia_x, state.inertia_y, state.inertia_z), dim=-1)
        torque = inertia * target_angular_acc
        torque = torque - torch.cross(state.omega, inertia * state.omega, dim=-1)

        # Inverse rotor allocation used by env_l2f: 
        # [T, tau_x, tau_y, tau_z] -> [f0,f1,f2,f3].
        arm = torch.clamp(state.arm_length, min=1.0e-8)
        k_t = torch.clamp(state.rotor_torque_constant, min=1.0e-8)
        bx = torque[:, 0] / arm
        by = torque[:, 1] / arm
        bz = torque[:, 2] / k_t

        t0 = (total_thrust + bz - 2.0 * by) / 4.0
        t1 = (total_thrust + 2.0 * bx - bz) / 4.0
        t2 = (total_thrust + bz + 2.0 * by) / 4.0
        t3 = (total_thrust - 2.0 * bx - bz) / 4.0
        thrust = torch.stack((t0, t1, t2, t3), dim=1)

        min_thrust, max_thrust = _thrust_bounds(state.thrust_coeff_c0, state.thrust_coeff_c1, state.thrust_coeff_c2)
        thrust = thrust.clamp(min=min_thrust, max=max_thrust)
        return _thrust_to_action(
            thrust,
            state.thrust_coeff_c0,
            state.thrust_coeff_c1,
            state.thrust_coeff_c2,
        )


@dataclass
class RolloutData:
    controller: str
    seed: int
    sample: int
    time_s: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    euler_deg: np.ndarray
    omega: np.ndarray
    action: np.ndarray
    motor: np.ndarray


def _rollout_controller(
    controller: _ControllerBase,
    sim: L2FSimulator,
    initial_state: L2FState,
    *,
    controller_name: str,
    seed: int,
    horizon: int,
    dt: float,
    device: torch.device,
) -> list[RolloutData]:
    with torch.no_grad():
        state = _clone_state(initial_state)
        batch_size = state.position.shape[0]
        controller.reset(batch_size, device=device, dtype=state.position.dtype)

        position = torch.empty((horizon + 1, batch_size, 3), device=device, dtype=torch.float64)
        velocity = torch.empty_like(position)
        rotation = torch.empty((horizon + 1, batch_size, 3, 3), device=device, dtype=torch.float64)
        omega = torch.empty_like(position)
        action = torch.empty((horizon, batch_size, 4), device=device, dtype=torch.float64)
        motor = torch.empty_like(action)

        position[0] = state.position.double()
        velocity[0] = state.velocity.double()
        rotation[0] = state.rotation.double()
        omega[0] = state.omega.double()

        for step_i in range(horizon):
            command = controller.action(state, step=step_i).double()
            command = command.clamp(-1.0, 1.0)
            next_state = sim.step(
                state,
                command.to(state.position.dtype),
                grad_decay=1.0,
            )
            action[step_i] = command
            motor[step_i] = next_state.motor.double()
            state = next_state

            position[step_i + 1] = state.position.double()
            velocity[step_i + 1] = state.velocity.double()
            rotation[step_i + 1] = state.rotation.double()
            omega[step_i + 1] = state.omega.double()

        euler_deg = _matrix_to_euler_zyx(rotation[:, :, :, :].reshape(-1, 3, 3)).reshape(
            horizon + 1,
            batch_size,
            3,
        ) * 180.0 / np.pi
        time = torch.arange(horizon + 1, device=device, dtype=torch.float64) * float(dt)

        out: list[RolloutData] = []
        for sample_i in range(batch_size):
            out.append(
                RolloutData(
                    controller=controller_name,
                    seed=seed,
                    sample=sample_i,
                    time_s=time.cpu().numpy(),
                    position=position[:, sample_i, :].cpu().numpy(),
                    velocity=velocity[:, sample_i, :].cpu().numpy(),
                    euler_deg=euler_deg[:, sample_i, :].cpu().numpy(),
                    omega=omega[:, sample_i, :].cpu().numpy(),
                    action=action[:, sample_i, :].cpu().numpy(),
                    motor=motor[:, sample_i, :].cpu().numpy(),
                )
            )
        return out


def _dynamics_rows(seed: int, state: L2FState) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    batch_size = state.position.shape[0]
    state_cpu = _clone_state(state)
    for sample_i in range(batch_size):
        rows.append(
            {
                "seed": seed,
                "sample": sample_i,
                "mass": float(state_cpu.mass[sample_i].item()),
                "cbrt_mass": float(state_cpu.cbrt_mass[sample_i].item()),
                "thrust_to_weight": float(state_cpu.thrust_to_weight[sample_i].item()),
                "torque_to_inertia": float(state_cpu.torque_to_inertia[sample_i].item()),
                "rotor_distance_factor": float(state_cpu.rotor_distance_factor[sample_i].item()),
                "inertia_factor": float(state_cpu.inertia_factor[sample_i].item()),
                "motor_time_rising": float(state_cpu.motor_time_rising[sample_i].item()),
                "motor_time_falling": float(state_cpu.motor_time_falling[sample_i].item()),
                "rotor_torque_constant": float(state_cpu.rotor_torque_constant[sample_i].item()),
                "force_std": float(state_cpu.force_std[sample_i].item()),
                "arm_length": float(state_cpu.arm_length[sample_i].item()),
                "inertia_x": float(state_cpu.inertia_x[sample_i].item()),
                "inertia_y": float(state_cpu.inertia_y[sample_i].item()),
                "inertia_z": float(state_cpu.inertia_z[sample_i].item()),
                "alpha_roll_max": float(state_cpu.alpha_roll_max[sample_i].item()),
                "alpha_pitch_max": float(state_cpu.alpha_pitch_max[sample_i].item()),
                "alpha_yaw_max": float(state_cpu.alpha_yaw_max[sample_i].item()),
                "eta_yaw": float(state_cpu.eta_yaw[sample_i].item()),
                "jz_over_jxy": float(state_cpu.jz_over_jxy[sample_i].item()),
                "dt_alpha_roll_max": float(state_cpu.dt_alpha_roll_max[sample_i].item()),
                "dt_alpha_yaw_max": float(state_cpu.dt_alpha_yaw_max[sample_i].item()),
                "external_force_x": float(state_cpu.external_force[sample_i, 0].item()),
                "external_force_y": float(state_cpu.external_force[sample_i, 1].item()),
                "external_force_z": float(state_cpu.external_force[sample_i, 2].item()),
                "thrust_coeff_c0": float(state_cpu.thrust_coeff_c0[sample_i].mean().item()),
                "thrust_coeff_c1": float(state_cpu.thrust_coeff_c1[sample_i].mean().item()),
                "thrust_coeff_c2": float(state_cpu.thrust_coeff_c2[sample_i].mean().item()),
            }
        )
    return rows


def _trace_metrics(trace: RolloutData, args: argparse.Namespace) -> dict[str, float | int | str]:
    position_norm = np.linalg.norm(trace.position, axis=1)
    velocity_norm = np.linalg.norm(trace.velocity, axis=1)
    omega_norm = np.linalg.norm(trace.omega, axis=1)

    # Index zero is the reset state. Every success flag below corresponds to a
    # completed physical step and is independent of roll, pitch, and yaw.
    step_success = (
        (position_norm[1:] < args.success_position_m)
        & (velocity_norm[1:] < args.success_velocity)
        & (omega_norm[1:] < args.success_omega)
    )
    finite = (
        np.all(np.isfinite(trace.position[1:]), axis=1)
        & np.all(np.isfinite(trace.velocity[1:]), axis=1)
        & np.all(np.isfinite(trace.omega[1:]), axis=1)
    )
    alive = np.logical_and.accumulate(finite)
    step_success &= alive
    rolling_fraction = np.full(step_success.shape, np.nan, dtype=np.float64)
    window = args.steady_window_steps
    if step_success.size >= window:
        counts = np.convolve(step_success.astype(np.float64), np.ones(window), mode="valid")
        rolling_fraction[window - 1 :] = counts / float(window)
    steady_entries = np.flatnonzero(rolling_fraction >= args.steady_required_fraction)
    first_steady = int(steady_entries[0]) if steady_entries.size else None
    final_window_fraction = float(rolling_fraction[-1]) if step_success.size else float("nan")
    steady_success = bool(
        np.isfinite(final_window_fraction)
        and final_window_fraction >= args.steady_required_fraction
    )
    snapshot_success = bool(step_success[-1]) if step_success.size else False
    stay_start = None if first_steady is None else first_steady + 1
    stay = (
        float(step_success[stay_start:].mean())
        if stay_start is not None and stay_start < step_success.size
        else float("nan")
    )

    initial_position = float(position_norm[0])
    initial_nonzero = max(initial_position, 1.0e-6)
    overshoot = float(position_norm.max() / initial_nonzero - 1.0)

    return {
        "controller": trace.controller,
        "seed": trace.seed,
        "sample": trace.sample,
        "position_max": float(position_norm.max()),
        "position_final": float(position_norm[-1]),
        "velocity_max": float(velocity_norm.max()),
        "velocity_final": float(velocity_norm[-1]),
        "omega_max": float(omega_norm.max()),
        "omega_final": float(omega_norm[-1]),
        "action_abs_mean": float(np.mean(np.abs(trace.action))),
        "action_abs_max": float(np.max(np.abs(trace.action))),
        "motor_abs_mean": float(np.mean(np.abs(trace.motor))),
        "motor_abs_max": float(np.max(np.abs(trace.motor))),
        "overshoot": float(overshoot),
        "position_hold_snapshot": float(snapshot_success),
        "position_hold_steady": float(steady_success),
        "final_window_success_fraction": final_window_fraction,
        "settling_time_s": (
            float(first_steady + 1) * args.dt if first_steady is not None else float("nan")
        ),
        "stay": stay,
        "survival": float(alive[-1]) if alive.size else 0.0,
        "recoverable": float(steady_success),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_aggregate_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    by_controller: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        cleaned: dict[str, float] = {}
        for key, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating, bool)):
                cleaned[key] = float(value)
        by_controller.setdefault(str(row["controller"]), []).append(cleaned)

    def _mean(values: list[dict[str, float]], key: str) -> float:
        return float(np.mean([v[key] for v in values if key in v]))

    fieldnames = (
        "controller", "count", "position_hold_steady_rate", "position_hold_steady_count",
        "position_hold_snapshot_rate", "final_window_success_fraction_mean", "stay_mean",
        "survival_rate", "pos_final_mean", "pos_max_mean",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for controller, controller_rows in by_controller.items():
            recover = [float(r["recoverable"]) for r in controller_rows]
            writer.writerow(
                {
                    "controller": controller,
                    "count": len(controller_rows),
                    "position_hold_steady_rate": float(np.mean(recover)),
                    "position_hold_steady_count": float(np.sum(recover)),
                    "position_hold_snapshot_rate": _mean(controller_rows, "position_hold_snapshot"),
                    "final_window_success_fraction_mean": _mean(controller_rows, "final_window_success_fraction"),
                    "stay_mean": _mean(controller_rows, "stay"),
                    "survival_rate": _mean(controller_rows, "survival"),
                    "pos_final_mean": _mean(controller_rows, "position_final"),
                    "pos_max_mean": _mean(controller_rows, "position_max"),
                }
            )


def _write_dynamics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_trajectory_csv(path: Path, records: list[RolloutData]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for record in records:
        for step in range(record.position.shape[0]):
            row = {
                "controller": record.controller,
                "seed": record.seed,
                "sample": record.sample,
                "step": step,
                "time_s": float(record.time_s[step]),
                "pos_x": float(record.position[step, 0]),
                "pos_y": float(record.position[step, 1]),
                "pos_z": float(record.position[step, 2]),
                "vel_x": float(record.velocity[step, 0]),
                "vel_y": float(record.velocity[step, 1]),
                "vel_z": float(record.velocity[step, 2]),
                "roll_deg": float(record.euler_deg[step, 0]),
                "pitch_deg": float(record.euler_deg[step, 1]),
                "yaw_deg": float(record.euler_deg[step, 2]),
                "omega_x": float(record.omega[step, 0]),
                "omega_y": float(record.omega[step, 1]),
                "omega_z": float(record.omega[step, 2]),
            }
            if step < record.action.shape[0]:
                row.update(
                    {
                        "action_0": float(record.action[step, 0]),
                        "action_1": float(record.action[step, 1]),
                        "action_2": float(record.action[step, 2]),
                        "action_3": float(record.action[step, 3]),
                        "motor_0": float(record.motor[step, 0]),
                        "motor_1": float(record.motor[step, 1]),
                        "motor_2": float(record.motor[step, 2]),
                        "motor_3": float(record.motor[step, 3]),
                    }
                )
            rows.append(row)
    fieldnames = tuple(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_by_controller(records: list[RolloutData]) -> dict[str, list[RolloutData]]:
    groups: dict[str, list[RolloutData]] = {}
    for record in records:
        groups.setdefault(record.controller, []).append(record)
    return groups


def _plot_rollouts(records: list[RolloutData], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not records:
        return

    colors = plt.get_cmap("tab10").colors
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = _group_by_controller(records)

    # draw_real-like output: 3D trajectories and xyz-time.
    fig_real = plt.figure(figsize=(12, 9))
    ax3d = fig_real.add_subplot(2, 2, 1, projection="3d")
    ax_xyz = fig_real.add_subplot(2, 2, 2)
    ax_xy = fig_real.add_subplot(2, 2, 3)
    ax_norm = fig_real.add_subplot(2, 2, 4)

    color_i = 0
    for controller, traces in groups.items():
        color = colors[color_i % len(colors)]
        color_i += 1
        for trace in traces[:8]:
            t = trace.time_s
            p = trace.position
            ax3d.plot(p[:, 0], p[:, 1], p[:, 2], color=color, alpha=0.35, label=None)
            if trace == traces[0]:
                ax_xyz.plot(t, p[:, 0], color=color, alpha=0.35, linestyle="--", label=f"{controller} x")
                ax_xyz.plot(t, p[:, 1], color=color, alpha=0.35, linestyle="-", label=f"{controller} y")
                ax_xyz.plot(t, p[:, 2], color=color, alpha=0.35, linestyle="-.", label=f"{controller} z")
            else:
                ax_xyz.plot(t, p[:, 0], color=color, alpha=0.35, linestyle="--")
                ax_xyz.plot(t, p[:, 1], color=color, alpha=0.35, linestyle="-")
                ax_xyz.plot(t, p[:, 2], color=color, alpha=0.35, linestyle="-.")
            pos_norm = np.linalg.norm(p, axis=1)
            ax_norm.plot(t, pos_norm, color=color, alpha=0.35)
        stacked = np.stack([t.position for t in traces], axis=0)
        mean_pos = stacked.mean(axis=0)
        ax3d.plot(mean_pos[:, 0], mean_pos[:, 1], mean_pos[:, 2], label=controller, linewidth=2.2, color=color)
        ax_xy.plot(mean_pos[:, 0], mean_pos[:, 1], label=f"{controller} xy", color=color)
        ax_norm.plot(traces[0].time_s, np.linalg.norm(mean_pos, axis=1), label=controller, linewidth=2.2, color=color)

    ax3d.set_title("draw_real: 3D trajectory")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax_xyz.set_title("draw_real: x/y/z")
    ax_xyz.set_xlabel("time (s)")
    ax_xyz.set_ylabel("position (m)")
    ax_xyz.legend(fontsize="small")
    ax_xy.set_title("draw_real: xy projection")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend(fontsize="small")
    ax_norm.set_title("draw_real: position norm")
    ax_norm.set_xlabel("time (s)")
    ax_norm.set_ylabel("|p| (m)")
    ax_norm.legend(fontsize="small")
    fig_real.tight_layout()
    fig_real.savefig(out_dir / "draw_real.png", dpi=180)
    plt.close(fig_real)

    # draw_q-like output: angle / attitude over time.
    fig_q, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].set_title("draw_q: roll/pitch/yaw (deg)")
    for controller, traces in groups.items():
        color = colors[hash(controller) % len(colors)]
        stacked = np.stack([t.euler_deg for t in traces], axis=0)
        mean_euler = stacked.mean(axis=0)
        std_euler = stacked.std(axis=0)
        time = traces[0].time_s
        for axis_i, label in enumerate(("roll", "pitch", "yaw")):
            axes[axis_i].plot(time, mean_euler[:, axis_i], label=f"{controller} {label}")
            axes[axis_i].fill_between(
                time,
                mean_euler[:, axis_i] - std_euler[:, axis_i],
                mean_euler[:, axis_i] + std_euler[:, axis_i],
                alpha=0.2,
            )
    for ax in axes:
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(fontsize="small")
    fig_q.tight_layout()
    fig_q.savefig(out_dir / "draw_q.png", dpi=180)
    plt.close(fig_q)

    # draw_w-like output: angular velocity over time.
    fig_w, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].set_title("draw_w: angular velocity (rad/s)")
    for controller, traces in groups.items():
        stacked = np.stack([t.omega for t in traces], axis=0)
        mean_omega = stacked.mean(axis=0)
        std_omega = stacked.std(axis=0)
        time = traces[0].time_s
        for axis_i, label in enumerate(("omega_x", "omega_y", "omega_z")):
            axes[axis_i].plot(time, mean_omega[:, axis_i], label=f"{controller} {label}")
            axes[axis_i].fill_between(
                time,
                mean_omega[:, axis_i] - std_omega[:, axis_i],
                mean_omega[:, axis_i] + std_omega[:, axis_i],
                alpha=0.2,
            )
    for ax in axes:
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(fontsize="small")
    fig_w.tight_layout()
    fig_w.savefig(out_dir / "draw_w.png", dpi=180)
    plt.close(fig_w)

    # motor/control view
    fig_u, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].set_title("motor inputs (normalized action)")
    axes[1].set_title("motor thrust states")
    for controller, traces in groups.items():
        color = colors[hash(controller) % len(colors)]
        action_stack = np.stack([t.action for t in traces], axis=0)
        mean_action = action_stack.mean(axis=0)
        time = traces[0].time_s[:-1]
        for i in range(4):
            axes[0].plot(time, mean_action[:, i], color=color, alpha=0.8, label=f"{controller} a{i}")
        motor_stack = np.stack([t.motor for t in traces], axis=0)
        mean_motor = motor_stack.mean(axis=0)
        axes[1].plot(time, mean_motor[:, 0], color=color, alpha=0.8, label=f"{controller} m0")
        for i in range(1, 4):
            axes[1].plot(time, mean_motor[:, i], color=color, alpha=0.8, label=f"{controller} m{i}")
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="x-small", ncol=2)
    fig_u.tight_layout()
    fig_u.savefig(out_dir / "draw_u.png", dpi=180)
    plt.close(fig_u)


def _load_checkpoint(path: Path, device: torch.device, *, args: argparse.Namespace) -> MotorGRUPolicy:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    source_dim = int(state_dict["encoder.0.weight"].shape[1])
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    get_arg = checkpoint_args.get if isinstance(checkpoint_args, dict) else lambda key, default: getattr(checkpoint_args, key, default)
    policy = MotorGRUPolicy(
        observation_dim=source_dim,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth,
        enable_integral_residual=bool(get_arg("enable_integral_residual", False)),
        enable_damping_residual=bool(get_arg("enable_rate_damping_residual", False)),
        integral_residual_hidden_dim=int(get_arg("integral_residual_hidden_dim", 16)),
        damping_residual_hidden_dim=int(get_arg("damping_residual_hidden_dim", 32)),
        integral_residual_scale=float(get_arg("integral_residual_scale", 1.0)),
        damping_residual_scale=float(get_arg("damping_residual_scale", 1.0)),
    ).to(device)
    policy.load_compatible_state_dict(state_dict)
    policy.observation_mode = str(get_arg("observation_mode", mode_from_observation_dim(source_dim)))
    policy.integral_limit = float(get_arg("integral_limit", 0.5))
    policy.integral_leak = float(get_arg("integral_leak", 0.0))
    policy.integral_input_frame = str(get_arg("integral_input_frame", "world"))
    policy.eval()
    return policy


def _build_controllers(
    args: argparse.Namespace,
    *,
    device: torch.device,
    sim: L2FSimulator,
) -> dict[str, _ControllerBase]:
    names = _parse_controllers(args.controllers)
    available: dict[str, _ControllerBase] = {}
    for name in names:
        if name == "learned":
            if not args.checkpoint_path:
                raise ValueError("learned controller requested but --checkpoint-path is empty")
            policy = _load_checkpoint(Path(args.checkpoint_path), device=device, args=args)
            available[name] = LearnedController(
                policy=policy,
                sim=sim,
                hidden_decay=args.hidden_state_decay,
            )
        elif name == "pid":
            available[name] = PIDController(
                sim,
                kp_xy=args.pid_kp_xy,
                kd_xy=args.pid_kd_xy,
                kp_z=args.pid_kp_z,
                kd_z=args.pid_kd_z,
                kp_att=args.pid_kp_att,
                kd_att=args.pid_kd_att,
                kd_yaw=args.pid_kd_yaw,
                max_tilt=args.pid_max_tilt,
            )
        else:
            raise ValueError(f"unknown controller: {name}")
    return available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare learned and reference controllers with common dynamics.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seeds", default="")
    parser.add_argument("--eval-seed-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--controllers", default="learned,pid")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--hidden-state-decay", type=float, default=1.0)
    parser.add_argument("--save-dir", default="reports/controller_compare")
    parser.add_argument("--save-prefix", default="compare")
    parser.add_argument("--trajectory-path", default="trajectory_compare.csv")
    parser.add_argument("--summary-path", default="comparison_summary.csv")
    parser.add_argument("--aggregate-summary-path", default="comparison_summary_aggregate.csv")
    parser.add_argument("--dynamics-path", default="dynamics.csv")
    parser.add_argument("--no-plot", action="store_true")

    parser.add_argument("--mass", type=float, default=0.05)
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--arm-length", type=float, default=0.046)
    parser.add_argument("--yaw-drag", type=float, default=0.012)
    parser.add_argument("--motor-tau", type=float, default=0.06)
    parser.add_argument("--motor-authority", type=float, default=1.35)
    parser.add_argument("--inertia-x", type=float, default=1.4e-5)
    parser.add_argument("--inertia-y", type=float, default=1.4e-5)
    parser.add_argument("--inertia-z", type=float, default=2.17e-5)
    parser.add_argument("--max-initial-position", type=float, default=1.0)
    parser.add_argument("--max-initial-velocity", type=float, default=0.6)
    parser.add_argument("--max-initial-angle", type=float, default=0.45)
    parser.add_argument("--max-initial-omega", type=float, default=1.0)
    parser.add_argument("--disturbance-force-max", type=float, default=0.0)
    parser.add_argument("--external-force-ratio", type=float, default=0.0)
    parser.add_argument(
        "--dynamics-profile",
        default="fixed",
        choices=("fixed", "raptor-broad", "physical-broad"),
        help="fixed uses the nominal model; broad profiles reuse env_l2f training-time dynamics randomization.",
    )
    parser.add_argument("--sample-dynamics", action="store_true")
    parser.add_argument("--sampled-dynamics-level", default="small", choices=("small", "broad"))
    parser.add_argument("--broad-sampler", default="legacy", choices=("legacy", "physical"))

    parser.add_argument("--success-position-m", type=float, default=0.05)
    parser.add_argument("--success-velocity", type=float, default=0.10)
    parser.add_argument("--success-omega", type=float, default=0.20)
    parser.add_argument("--steady-window-steps", type=int, default=100)
    parser.add_argument("--steady-required-fraction", type=float, default=0.95)

    parser.add_argument("--pid-kp-xy", type=float, default=2.2)
    parser.add_argument("--pid-kd-xy", type=float, default=1.0)
    parser.add_argument("--pid-kp-z", type=float, default=2.8)
    parser.add_argument("--pid-kd-z", type=float, default=1.0)
    parser.add_argument("--pid-kp-att", type=float, default=2.0)
    parser.add_argument("--pid-kd-att", type=float, default=0.3)
    parser.add_argument("--pid-kd-yaw", type=float, default=0.2)
    parser.add_argument("--pid-max-tilt", type=float, default=0.45)

    parser.add_argument("--encoder-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--encoder-depth", type=int, default=2)

    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.steady_window_steps <= 0:
        raise ValueError("--steady-window-steps must be positive")
    if not 0.0 < args.steady_required_fraction <= 1.0:
        raise ValueError("--steady-required-fraction must be in (0, 1]")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0")
    if args.mass <= 0.0:
        raise ValueError("--mass must be > 0")
    if args.gravity <= 0.0:
        raise ValueError("--gravity must be > 0")
    if args.arm_length <= 0.0:
        raise ValueError("--arm-length must be > 0")
    if args.motor_tau <= 0.0:
        raise ValueError("--motor-tau must be > 0")
    if args.motor_authority <= 0.0:
        raise ValueError("--motor-authority must be > 0")
    if args.inertia_x <= 0.0 or args.inertia_y <= 0.0 or args.inertia_z <= 0.0:
        raise ValueError("--inertia-x/y/z must be > 0")
    if args.dynamics_profile == "raptor-broad":
        args.sample_dynamics = True
        args.sampled_dynamics_level = "broad"
        args.broad_sampler = "legacy"
    elif args.dynamics_profile == "physical-broad":
        args.sample_dynamics = True
        args.sampled_dynamics_level = "broad"
        args.broad_sampler = "physical"
    return args


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    sim = L2FSimulator(
        L2FParams(
            dt=args.dt,
            mass=args.mass,
            gravity=args.gravity,
            arm_length=args.arm_length,
            yaw_drag=args.yaw_drag,
            motor_tau=args.motor_tau,
            motor_authority=args.motor_authority,
            inertia_x=args.inertia_x,
            inertia_y=args.inertia_y,
            inertia_z=args.inertia_z,
            max_initial_position=args.max_initial_position,
            max_initial_velocity=args.max_initial_velocity,
            max_initial_angle=args.max_initial_angle,
            max_initial_omega=args.max_initial_omega,
            disturbance_force_max=args.disturbance_force_max,
            external_force_ratio=args.external_force_ratio,
        )
    )
    controllers = _build_controllers(args, device=device, sim=sim)
    seeds = _parse_seed_list(args.eval_seeds, args.seed, args.eval_seed_count)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[RolloutData] = []
    all_metrics: list[dict[str, float | int | str]] = []
    all_dynamics_rows: list[dict[str, float | int]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        initial_state = sim.reset(
            args.batch_size,
            device=device,
            sample_dynamics=args.sample_dynamics,
            sampled_dynamics_level=args.sampled_dynamics_level,
            broad_sampler=args.broad_sampler,
        )
        all_dynamics_rows.extend(_dynamics_rows(seed, initial_state))
        for controller_name, controller in controllers.items():
            records = _rollout_controller(
                controller=controller,
                sim=sim,
                initial_state=initial_state,
                controller_name=controller_name,
                seed=seed,
                horizon=args.horizon,
                dt=args.dt,
                device=device,
            )
            all_records.extend(records)
            all_metrics.extend(_trace_metrics(record, args) for record in records)

    _write_trajectory_csv(save_dir / args.trajectory_path, all_records)
    _write_summary_csv(save_dir / args.summary_path, all_metrics)
    _write_summary_aggregate_csv(save_dir / args.aggregate_summary_path, all_metrics)
    _write_dynamics_csv(save_dir / args.dynamics_path, all_dynamics_rows)
    with (save_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    for controller, traces in _group_by_controller(all_records).items():
        if not traces:
            continue
        pos_final = np.mean([np.linalg.norm(trace.position[-1]) for trace in traces])
        pos_max = np.mean([np.max(np.linalg.norm(trace.position, axis=1)) for trace in traces])
        print(f"{controller}: final_pos_mean={pos_final:.4f}, max_pos_mean={pos_max:.4f}")

    if not args.no_plot:
        _plot_rollouts(all_records, save_dir / f"{args.save_prefix}_plots")


if __name__ == "__main__":
    main()
