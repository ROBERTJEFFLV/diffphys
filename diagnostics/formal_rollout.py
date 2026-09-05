from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from diagnostics.physics import branch_wrench_decomposition, motor_to_thrust
from diagnostics.streaming_phase1 import Phase1StreamingAccumulator
from env_l2f import L2FLossConfig, L2FParams, L2FSimulator, L2FState
from l2f_cuda_backend import cuda_step, load_extension
from model import MotorGRUPolicy
from policy_observation import (
    LEGACY_BOX_INTEGRAL_CLAMP_MODE,
    build_policy_observation,
    initial_observation_state,
    integral_clamp_active_mask,
    update_position_integral,
    validate_integral_clamp_mode,
)


Intervention = Literal["full", "damping_zero", "explicit_integral_zero", "integral_state_zero"]


@dataclass
class FormalRolloutResult:
    horizon_rows: list[dict[str, Any]]
    branch_rows: list[dict[str, Any]]
    phase_rows: list[dict[str, Any]]
    integral_rows: list[dict[str, Any]]


@dataclass
class TrajectoryTrace:
    position: torch.Tensor
    velocity: torch.Tensor
    rotation: torch.Tensor
    omega: torch.Tensor
    motor: torch.Tensor
    action: torch.Tensor
    integral: torch.Tensor
    hidden: torch.Tensor


def select_state(state: L2FState, indices: torch.Tensor) -> L2FState:
    return L2FState(**{
        name: getattr(state, name).index_select(0, indices)
        for name in L2FState.__dataclass_fields__
    })


def clone_state(state: L2FState) -> L2FState:
    return L2FState(**{
        name: getattr(state, name).detach().clone()
        for name in L2FState.__dataclass_fields__
    })


def load_q2_policy(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[MotorGRUPolicy, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    policy = MotorGRUPolicy(
        observation_dim=25,
        encoder_dim=int(args.get("encoder_dim", 192)),
        hidden_dim=int(args.get("hidden_dim", 192)),
        encoder_depth=int(args.get("encoder_depth", 2)),
        enable_integral_residual=bool(args.get("enable_integral_residual", False)),
        enable_damping_residual=bool(args.get("enable_rate_damping_residual", False)),
        integral_residual_hidden_dim=int(args.get("integral_residual_hidden_dim", 16)),
        damping_residual_hidden_dim=int(args.get("damping_residual_hidden_dim", 32)),
        integral_residual_scale=float(args.get("integral_residual_scale", 1.0)),
        damping_residual_scale=float(args.get("damping_residual_scale", 1.0)),
    ).to(device=device, dtype=dtype)
    missing, unexpected = policy.load_compatible_state_dict(checkpoint.get("model", checkpoint))
    if missing or unexpected:
        raise RuntimeError(f"formal Q2 load mismatch: missing={missing}, unexpected={unexpected}")
    policy.eval()
    # Preserve the teacher's observation contract wherever it is replayed.
    # In particular DAgger must not replace its integral with a student's.
    policy.q2_observation_settings = {
        "mode": str(args.get("observation_mode", "integral25")),
        "integral_input_frame": str(args.get("integral_input_frame", "world")),
        "integral_input_multiplier": float(args.get("integral_input_multiplier", 1.0)),
        "noise_max": float(args.get("observation_noise_max", 0.0)),
        "integral_limit": float(args.get("integral_limit", 0.5)),
        "integral_leak": float(args.get("integral_leak", 0.0)),
        "integral_clamp_mode": str(args.get("integral_clamp_mode", LEGACY_BOX_INTEGRAL_CLAMP_MODE)),
    }
    return policy, args


def _finite_state(state: L2FState) -> torch.Tensor:
    finite = torch.ones(state.position.shape[0], device=state.position.device, dtype=torch.bool)
    for name in ("position", "velocity", "rotation", "omega", "motor", "previous_action"):
        finite &= torch.isfinite(getattr(state, name).reshape(state.position.shape[0], -1)).all(dim=-1)
    return finite


def _failure_label(
    joint: int,
    position: int,
    velocity: int,
    omega: int,
    survived: bool,
) -> str:
    if not survived:
        return "survival_failure"
    if joint >= 95:
        return "success"
    failures = [
        name for name, count in (("position", position), ("velocity", velocity), ("omega", omega))
        if count < 95
    ]
    if not failures:
        return "window_overlap_failure"
    if len(failures) == 3:
        return "combined_failure"
    return "+".join(failures) if len(failures) > 1 else f"{failures[0]}-only_failure"


def _tail_proxy(values: torch.Tensor, threshold: float) -> dict[str, torch.Tensor]:
    violation = torch.relu(values / float(threshold) - 1.0).square()
    sorted_values = violation.sort(dim=0, descending=True).values
    top_count = max(1, int(np.ceil(0.05 * values.shape[0])))
    return {
        "mean": violation.mean(dim=0),
        "maximum": violation.max(dim=0).values,
        "top5_mean": sorted_values[:top_count].mean(dim=0),
        "sixth_largest": sorted_values[min(5, sorted_values.shape[0] - 1)],
        "nonzero_fraction": (violation > 0).to(values.dtype).mean(dim=0),
    }


@torch.no_grad()
def trace_q2(
    policy: MotorGRUPolicy,
    initial_state: L2FState,
    *,
    horizon: int,
    integral_clamp_mode: str = LEGACY_BOX_INTEGRAL_CLAMP_MODE,
) -> TrajectoryTrace:
    """Record the exact Torch Q2 path for MATLAB/Python short-trace parity."""

    sim = L2FSimulator(L2FParams(dt=0.01))
    state = clone_state(initial_state)
    batch = state.position.shape[0]
    hidden = policy.initial_hidden(batch, device=state.position.device, dtype=state.position.dtype)
    observation_state = initial_observation_state(batch, device=state.position.device, dtype=state.position.dtype)
    logs: dict[str, list[torch.Tensor]] = {
        "position": [state.position.clone()], "velocity": [state.velocity.clone()],
        "rotation": [state.rotation.clone()], "omega": [state.omega.clone()],
        "motor": [state.motor.clone()], "integral": [observation_state.integral_position.clone()],
        "hidden": [hidden.clone()], "action": [],
    }
    for _ in range(horizon):
        observation, observed_position = build_policy_observation(
            state, observation_state, mode="integral25", integral_input_frame="body",
            integral_input_multiplier=1.0,
        )
        action, hidden = policy(observation, hidden)
        observation_state = update_position_integral(
            observation_state, observed_position, dt=0.01, integral_limit=0.5, integral_leak=0.0,
            integral_clamp_mode=integral_clamp_mode,
        )
        state = sim.step(state, action, grad_decay=1.0)
        logs["action"].append(action.clone())
        for name in ("position", "velocity", "rotation", "omega", "motor"):
            logs[name].append(getattr(state, name).clone())
        logs["integral"].append(observation_state.integral_position.clone())
        logs["hidden"].append(hidden.clone())
    return TrajectoryTrace(**{name: torch.stack(value) for name, value in logs.items()})


@torch.no_grad()
def rollout_q2(
    policy: MotorGRUPolicy,
    initial_state: L2FState,
    scenario_uids: list[str],
    *,
    checkpoint_label: str,
    seed: int,
    group: str,
    horizon: int,
    snapshot_horizons: tuple[int, ...] = (500, 1000, 2000, 5000, 10000),
    backend: Literal["cuda", "torch"] = "cuda",
    intervention: Intervention = "full",
    record_branches: bool = False,
    branch_stride: int = 1,
    record_phase: bool = False,
    streaming_accumulator: Phase1StreamingAccumulator | None = None,
    integral_clamp_mode: str = LEGACY_BOX_INTEGRAL_CLAMP_MODE,
) -> FormalRolloutResult:
    if len(scenario_uids) != initial_state.position.shape[0]:
        raise ValueError("scenario UID count must match batch")
    if horizon < 100:
        raise ValueError("formal steady labels require at least 100 steps")
    integral_clamp_mode = validate_integral_clamp_mode(integral_clamp_mode)
    device = initial_state.position.device
    if backend == "cuda":
        if device.type != "cuda" or initial_state.position.dtype != torch.float32:
            raise ValueError("compact CUDA rollout requires CUDA float32 state")
        load_extension()
    params = L2FParams(dt=0.01)
    sim = L2FSimulator(params)
    loss_config = L2FLossConfig()
    state = clone_state(initial_state)
    batch = state.position.shape[0]
    if streaming_accumulator is not None and streaming_accumulator.config.batch_size != batch:
        raise ValueError("streaming accumulator batch size does not match rollout batch")
    hidden = policy.initial_hidden(batch, device=device, dtype=state.position.dtype)
    observation_state = initial_observation_state(batch, device=device, dtype=state.position.dtype)
    alive = torch.ones(batch, device=device, dtype=torch.bool)
    windows: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=100)
    norm_windows: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=100)
    horizon_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    integral_rows: list[dict[str, Any]] = []
    requested = {value for value in snapshot_horizons if value <= horizon}
    energy = torch.zeros(batch, device=device, dtype=state.position.dtype)
    saturation_count = torch.zeros(batch, device=device, dtype=state.position.dtype)

    clamp_limit = 0.5
    clamp_tolerance = 1.0e-7
    first_clamp = torch.full((batch, 3), -1, device=device, dtype=torch.int64)
    first_unclamp = torch.full((batch, 3), -1, device=device, dtype=torch.int64)
    clamp_total = torch.zeros((batch, 3), device=device, dtype=torch.int64)
    clamp_run = torch.zeros((batch, 3), device=device, dtype=torch.int64)
    clamp_longest = torch.zeros((batch, 3), device=device, dtype=torch.int64)
    ever_clamped = torch.zeros((batch, 3), device=device, dtype=torch.bool)
    interval_clamp = torch.zeros((3, batch, 3), device=device, dtype=torch.int64)

    min_thrust = motor_to_thrust(state, torch.full_like(state.motor, -1.0))
    max_thrust = motor_to_thrust(state, torch.full_like(state.motor, 1.0))
    thrust_delta = (max_thrust - min_thrust).clamp_min(1.0e-12)
    torque_authority = torch.stack(
        (state.arm_length * thrust_delta[:, 0], state.arm_length * thrust_delta[:, 0],
         2.0 * state.rotor_torque_constant * thrust_delta[:, 0]), dim=-1,
    )

    for step in range(1, horizon + 1):
        if intervention == "integral_state_zero":
            observation_state.integral_position.zero_()
        integral_used = observation_state.integral_position.clone()
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode="integral25",
            noise_max=0.0,
            integral_input_frame="body",
            integral_input_multiplier=1.0,
        )
        _, hidden, auxiliary = policy.forward_with_aux(observation, hidden)
        decomposition = branch_wrench_decomposition(state, auxiliary, dt=params.dt)
        if intervention == "full" or intervention == "integral_state_zero":
            action = decomposition.counterfactuals.full
        elif intervention == "damping_zero":
            action = decomposition.counterfactuals.main_integral
        elif intervention == "explicit_integral_zero":
            action = decomposition.counterfactuals.main_damping
        else:
            raise ValueError(f"unknown intervention: {intervention}")

        if intervention != "integral_state_zero":
            observation_state = update_position_integral(
                observation_state,
                observed_position,
                dt=params.dt,
                integral_limit=clamp_limit,
                integral_leak=0.0,
                integral_clamp_mode=integral_clamp_mode,
            )
        integral_next = observation_state.integral_position
        clamped = integral_clamp_active_mask(
            integral_next,
            integral_limit=clamp_limit,
            integral_clamp_mode=integral_clamp_mode,
            tolerance=clamp_tolerance,
        )
        newly = clamped & (~ever_clamped)
        first_clamp[newly] = step
        recovered = (~clamped) & ever_clamped & (first_unclamp < 0)
        first_unclamp[recovered] = step
        ever_clamped |= clamped
        clamp_total += clamped
        clamp_run = torch.where(clamped, clamp_run + 1, torch.zeros_like(clamp_run))
        clamp_longest = torch.maximum(clamp_longest, clamp_run)
        interval = 0 if step <= 500 else (1 if step <= 2000 else 2)
        interval_clamp[interval] += clamped

        if record_branches and (step - 1) % branch_stride == 0:
            delta_d = decomposition.damping_delta_after_integral
            delta_d_no_i = decomposition.damping_delta
            interaction = (
                decomposition.full - decomposition.main_integral
                - decomposition.main_damping + decomposition.main_only
            )
            power_axis = state.omega * delta_d[:, 1:]
            power = power_axis.sum(dim=-1)
            cosine = power / (
                torch.linalg.vector_norm(state.omega, dim=-1)
                * torch.linalg.vector_norm(delta_d[:, 1:], dim=-1) + 1.0e-12
            )
            collective_headroom = torch.where(
                delta_d[:, 0] >= 0,
                max_thrust.sum(dim=-1) - decomposition.full[:, 0],
                decomposition.full[:, 0] - min_thrust.sum(dim=-1),
            ).clamp_min(1.0e-12)
            position_unit = state.position / torch.linalg.vector_norm(state.position, dim=-1, keepdim=True).clamp_min(1.0e-12)
            world_accel = state.rotation[:, :, 2] * (delta_d[:, :1] / state.mass[:, None])
            position_recovery_accel = (-position_unit * world_accel).sum(dim=-1)
            arrays = {
                "main_wrench": decomposition.main_only,
                "main_integral_wrench": decomposition.main_integral,
                "main_damping_wrench": decomposition.main_damping,
                "full_wrench": decomposition.full,
                "integral_delta": decomposition.integral_delta,
                "damping_delta": delta_d,
                "damping_delta_no_integral": delta_d_no_i,
                "interaction": interaction,
                "full_action": decomposition.counterfactuals.full,
                "main_action": decomposition.counterfactuals.main_only,
                "main_integral_action": decomposition.counterfactuals.main_integral,
                "main_damping_action": decomposition.counterfactuals.main_damping,
                "tanh_gain": decomposition.counterfactuals.full_local_logit_gain,
                "full_motor": decomposition.full_motor,
                "main_motor": decomposition.main_only_motor,
                "full_thrust": decomposition.full_thrust,
                "main_thrust": decomposition.main_only_thrust,
                "power_axis": power_axis,
            }
            cpu = {name: value.detach().cpu().numpy() for name, value in arrays.items()}
            scalar_cpu = {
                "power": power.cpu().numpy(),
                "cosine": cosine.cpu().numpy(),
                "collective_headroom": collective_headroom.cpu().numpy(),
                "position_recovery_accel": position_recovery_accel.cpu().numpy(),
                "mass_g": (state.mass * 9.80665).cpu().numpy(),
                "current_collective": decomposition.full[:, 0].cpu().numpy(),
                "omega_norm": torch.linalg.vector_norm(state.omega, dim=-1).cpu().numpy(),
                "damping_torque_norm": torch.linalg.vector_norm(delta_d[:, 1:], dim=-1).cpu().numpy(),
            }
            omega_cpu = state.omega.cpu().numpy()
            torque_auth_cpu = torque_authority.cpu().numpy()
            for index, uid in enumerate(scenario_uids):
                row: dict[str, Any] = {
                    "checkpoint": checkpoint_label, "seed": seed, "training_group": group,
                    "scenario_uid": uid, "step": step, "time_s": step * params.dt,
                    "intervention": intervention, "integral_clamp_mode": integral_clamp_mode,
                    "damping_power": float(scalar_cpu["power"][index]),
                    "damping_cosine": float(scalar_cpu["cosine"][index]),
                    "damping_collective_over_mg": abs(float(cpu["damping_delta"][index, 0])) / max(float(scalar_cpu["mass_g"][index]), 1.0e-12),
                    "damping_collective_over_current": abs(float(cpu["damping_delta"][index, 0])) / max(abs(float(scalar_cpu["current_collective"][index])), 1.0e-12),
                    "damping_collective_over_headroom": abs(float(cpu["damping_delta"][index, 0])) / max(float(scalar_cpu["collective_headroom"][index]), 1.0e-12),
                    "damping_position_recovery_accel": float(scalar_cpu["position_recovery_accel"][index]),
                    "omega_norm": float(scalar_cpu["omega_norm"][index]),
                    "damping_torque_norm": float(scalar_cpu["damping_torque_norm"][index]),
                }
                for name in ("main_wrench", "main_integral_wrench", "main_damping_wrench", "full_wrench", "integral_delta", "damping_delta", "damping_delta_no_integral", "interaction"):
                    for component, suffix in enumerate(("collective", "tau_x", "tau_y", "tau_z")):
                        row[f"{name}_{suffix}"] = float(cpu[name][index, component])
                for name in ("full_action", "main_action", "main_integral_action", "main_damping_action", "tanh_gain", "full_motor", "main_motor", "full_thrust", "main_thrust"):
                    for motor in range(4):
                        row[f"{name}_{motor}"] = float(cpu[name][index, motor])
                for axis, suffix in enumerate(("x", "y", "z")):
                    row[f"omega_{suffix}"] = float(omega_cpu[index, axis])
                    row[f"damping_power_{suffix}"] = float(cpu["power_axis"][index, axis])
                    row[f"damping_tau_over_authority_{suffix}"] = float(cpu["damping_delta"][index, axis + 1] / max(torque_auth_cpu[index, axis], 1.0e-12))
                branch_rows.append(row)

        previous_action = state.previous_action
        state = cuda_step(state, action, params, grad_decay=1.0) if backend == "cuda" else sim.step(state, action, grad_decay=1.0)
        alive &= _finite_state(state)
        energy += action.square().sum(dim=-1)
        saturation_count += (action.abs() >= 0.98).to(action.dtype).sum(dim=-1)
        position_norm = torch.linalg.vector_norm(state.position, dim=-1)
        velocity_norm = torch.linalg.vector_norm(state.velocity, dim=-1)
        omega_norm = torch.linalg.vector_norm(state.omega, dim=-1)
        p_pass = position_norm < 0.05
        v_pass = velocity_norm < 0.10
        o_pass = omega_norm < 0.20
        windows.append((p_pass, v_pass, o_pass, p_pass & v_pass & o_pass & alive))
        norm_windows.append((position_norm, velocity_norm, omega_norm))

        if record_phase or streaming_accumulator is not None:
            action_delta = torch.linalg.vector_norm(action - previous_action, dim=-1)
            potential = sim.tracking_potential(state, loss_config)
            signal_tensors = {
                "position_norm": position_norm,
                "velocity_norm": velocity_norm,
                "omega_norm": omega_norm,
                "action_norm": torch.linalg.vector_norm(action, dim=-1),
                "action_delta_norm": action_delta,
                "motor_norm": torch.linalg.vector_norm(state.motor, dim=-1),
                "hidden_norm": torch.linalg.vector_norm(hidden, dim=-1),
                "integral_norm": torch.linalg.vector_norm(integral_used, dim=-1),
                "integral_clamp_fraction": clamped.to(state.position.dtype).mean(dim=-1),
                "main_logits_norm": torch.linalg.vector_norm(auxiliary["main_logits"], dim=-1),
                "integral_logits_norm": torch.linalg.vector_norm(auxiliary["integral_residual_logits"], dim=-1),
                "damping_logits_norm": torch.linalg.vector_norm(auxiliary["damping_residual_logits"], dim=-1),
                "branch_collective": decomposition.damping_delta_after_integral[:, 0],
                "branch_torque_norm": torch.linalg.vector_norm(
                    decomposition.damping_delta_after_integral[:, 1:], dim=-1
                ),
                "damping_power": (
                    state.omega * decomposition.damping_delta_after_integral[:, 1:]
                ).sum(dim=-1),
                "damping_torque_authority_ratio": (
                    torch.linalg.vector_norm(
                        decomposition.damping_delta_after_integral[:, 1:], dim=-1
                    )
                    / torch.linalg.vector_norm(torque_authority, dim=-1).clamp_min(1.0e-12)
                ),
                "motor_prediction_rmse": torch.sqrt(
                    torch.mean(torch.square(auxiliary["motor_state"] - state.motor), dim=-1)
                ),
                "tanh_gain_mean": decomposition.counterfactuals.full_local_logit_gain.mean(dim=-1),
                "dense_potential": potential,
            }
            if streaming_accumulator is not None:
                missing_signals = set(streaming_accumulator.config.signal_names).difference(signal_tensors)
                if missing_signals:
                    raise KeyError(f"unsupported streaming signals: {sorted(missing_signals)}")
                streaming_values = torch.stack(
                    [signal_tensors[name] for name in streaming_accumulator.config.signal_names], dim=-1
                ).cpu().numpy()
                streaming_accumulator.update(streaming_values)
            if record_phase:
                names = tuple(signal_tensors)
                phase_values = torch.stack([signal_tensors[name] for name in names], dim=-1).cpu().numpy()
                for index, uid in enumerate(scenario_uids):
                    phase_rows.append({
                        "checkpoint": checkpoint_label, "seed": seed, "training_group": group,
                        "scenario_uid": uid, "step": step, "phase": (step - 1) % 250,
                        "integral_clamp_mode": integral_clamp_mode,
                        **{name: float(phase_values[index, j]) for j, name in enumerate(names)},
                    })

        if step in requested:
            stack = [torch.stack([entry[i] for entry in windows], dim=0) for i in range(4)]
            counts = [value.sum(dim=0) for value in stack]
            norm_stack = [torch.stack([entry[i] for entry in norm_windows], dim=0) for i in range(3)]
            proxies = [_tail_proxy(value, threshold) for value, threshold in zip(norm_stack, (0.05, 0.10, 0.20))]
            count_cpu = [value.cpu().numpy() for value in counts]
            norm_cpu = [value.cpu().numpy() for value in norm_stack]
            proxy_cpu = [
                {name: value.cpu().numpy() for name, value in proxy.items()}
                for proxy in proxies
            ]
            alive_cpu = alive.cpu().numpy()
            energy_cpu = energy.cpu().numpy()
            saturation_cpu = saturation_count.cpu().numpy()
            for index, uid in enumerate(scenario_uids):
                p_count, v_count, o_count, joint_count = (int(value[index]) for value in count_cpu)
                row: dict[str, Any] = {
                    "checkpoint": checkpoint_label, "seed": seed, "training_group": group,
                    "scenario_uid": uid, "intervention": intervention, "horizon": step,
                    "integral_clamp_mode": integral_clamp_mode,
                    "position_pass_count": p_count, "velocity_pass_count": v_count,
                    "omega_pass_count": o_count, "joint_pass_count": joint_count,
                    "steady_success": int(bool(alive_cpu[index]) and joint_count >= 95),
                    "survival": int(bool(alive_cpu[index])),
                    "failure_label": _failure_label(joint_count, p_count, v_count, o_count, bool(alive_cpu[index])),
                    "control_energy": float(energy_cpu[index]),
                    "action_saturation_fraction": float(saturation_cpu[index] / (4 * step)),
                    "position_final": float(position_norm[index]),
                    "velocity_final": float(velocity_norm[index]),
                    "omega_final": float(omega_norm[index]),
                }
                for channel, channel_index in (("position", 0), ("velocity", 1), ("omega", 2)):
                    channel_norm = norm_cpu[channel_index][:, index]
                    row[f"{channel}_tail_rms"] = float(np.sqrt(np.mean(channel_norm ** 2)))
                    row[f"{channel}_tail_mean"] = float(np.mean(channel_norm))
                    row[f"{channel}_tail_max"] = float(np.max(channel_norm))
                    for proxy_name, proxy_value in proxy_cpu[channel_index].items():
                        row[f"{channel}_violation_{proxy_name}"] = float(proxy_value[index])
                horizon_rows.append(row)

    first_clamp_cpu = first_clamp.cpu().numpy()
    first_unclamp_cpu = first_unclamp.cpu().numpy()
    total_cpu = clamp_total.cpu().numpy()
    longest_cpu = clamp_longest.cpu().numpy()
    interval_cpu = interval_clamp.cpu().numpy()
    final_integral = observation_state.integral_position.cpu().numpy()
    interval_lengths = (min(horizon, 500), max(0, min(horizon, 2000) - 500), max(0, horizon - 2000))
    for index, uid in enumerate(scenario_uids):
        for axis, axis_name in enumerate(("x", "y", "z")):
            row = {
                "checkpoint": checkpoint_label, "seed": seed, "training_group": group,
                "scenario_uid": uid, "intervention": intervention, "axis": axis_name,
                "integral_clamp_mode": integral_clamp_mode,
                "first_clamp_step": int(first_clamp_cpu[index, axis]),
                "first_clamp_time_s": float(first_clamp_cpu[index, axis] * params.dt) if first_clamp_cpu[index, axis] >= 0 else float("nan"),
                "first_unclamp_step": int(first_unclamp_cpu[index, axis]),
                "unclamp_recovery_time_s": float((first_unclamp_cpu[index, axis] - first_clamp_cpu[index, axis]) * params.dt) if first_unclamp_cpu[index, axis] >= 0 else float("nan"),
                "total_clamp_duration_s": float(total_cpu[index, axis] * params.dt),
                "longest_clamp_duration_s": float(longest_cpu[index, axis] * params.dt),
                "clamp_fraction": float(total_cpu[index, axis] / horizon),
                "final_integral": float(final_integral[index, axis]),
            }
            for interval, name in enumerate(("H0_H500", "H500_H2000", "H2000_H10000")):
                length = interval_lengths[interval]
                row[f"clamp_fraction_{name}"] = float(interval_cpu[interval, index, axis] / length) if length else float("nan")
            integral_rows.append(row)
    return FormalRolloutResult(horizon_rows, branch_rows, phase_rows, integral_rows)


# Descriptive public name for experiment runners; keep ``rollout_q2`` for all
# existing diagnostics and historical call sites.
run_formal_rollout = rollout_q2
