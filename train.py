from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from env_l2f import (
    L2FState,
    L2FLossConfig,
    L2FParams,
    L2FSimulator,
    apply_gradient_decay,
)
from l2f_full_cuda_backend import METRIC_NAMES, full_cuda_rollout_metrics
from l2f_cuda_backend import cuda_backend_available, cuda_step, load_extension
from model import MotorGRUPolicy

RAPTOR_QUANTITIES = (
    "position",
    "angle",
    "linear_velocity",
    "angular_velocity",
    "angular_acceleration",
    "action",
    "action_relative",
)


def _metric_summary(values: list[torch.Tensor]) -> tuple[float, float, float]:
    if len(values) == 0:
        nan = float("nan")
        return nan, nan, nan
    stacked = torch.stack(values, dim=0)
    mean = stacked.mean().item()
    max_per_episode = stacked.max(dim=0).values
    max_mean = max_per_episode.mean().item()
    if max_per_episode.numel() <= 1:
        max_std = 0.0
    else:
        max_std = max_per_episode.std(unbiased=False).item()
    return mean, max_mean, max_std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train state/error/action GRU motor policy.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--sim-backend", default="auto", choices=("auto", "torch", "cuda", "cuda-full"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-initial-position", type=float, default=1.0)
    parser.add_argument("--max-initial-velocity", type=float, default=0.6)
    parser.add_argument("--max-initial-angle", type=float, default=0.45)
    parser.add_argument("--max-initial-omega", type=float, default=1.0)
    parser.add_argument("--disturbance-force-max", type=float, default=0.0)
    parser.add_argument("--external-force-ratio", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--encoder-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--grad-skip-threshold", type=float, default=1.0e4)
    parser.add_argument("--state-grad-decay", type=float, default=0.5)
    parser.add_argument("--hidden-grad-decay", type=float, default=0.7)
    parser.add_argument("--p-scale", type=float, default=2.0)
    parser.add_argument("--v-scale", type=float, default=3.0)
    parser.add_argument("--omega-scale", type=float, default=10.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--w-p", type=float, default=1.0)
    parser.add_argument("--w-v", type=float, default=0.3)
    parser.add_argument("--w-r", type=float, default=0.1)
    parser.add_argument("--w-omega", type=float, default=0.05)
    parser.add_argument("--attitude-mode", choices=("full", "tilt"), default="full")
    parser.add_argument("--clf-kappa", type=float, default=1.0)
    parser.add_argument("--tail-steps", type=int, default=50)
    parser.add_argument("--u-soft", type=float, default=0.9)
    parser.add_argument("--lambda-clf", type=float, default=0.5)
    parser.add_argument("--lambda-out", type=float, default=0.1)
    parser.add_argument("--lambda-tail", type=float, default=1.0)
    parser.add_argument("--lambda-du", type=float, default=3.0e-3)
    parser.add_argument("--lambda-ddu", type=float, default=3.0e-4)
    parser.add_argument("--lambda-sat", type=float, default=0.03)
    parser.add_argument("--external-force-max", type=float, default=0.0)
    parser.add_argument("--external-torque-max", type=float, default=0.0)
    parser.add_argument("--action-noise-max", type=float, default=0.0)
    parser.add_argument("--observation-noise-max", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-path", default="runs/direct_h500/train.csv")
    parser.add_argument("--checkpoint-path", default="checkpoints/direct_h500_motor_gru.pt")
    parser.add_argument("--checkpoint-steps", default="")
    parser.add_argument("--init-checkpoint-path", default="")
    parser.add_argument("--init-optimizer-state", action="store_true")
    parser.add_argument("--settling-position-mm", type=float, default=10.0)
    parser.add_argument("--tail-start-step", type=int, default=450)
    parser.add_argument("--sample-dynamics", action="store_true")
    parser.add_argument("--fixed-dynamics", action="store_true")
    parser.add_argument("--sampled-dynamics-level", default="small", choices=("small", "medium", "broad"))
    parser.add_argument("--broad-sampler", default="legacy", choices=("legacy", "physical"))
    parser.add_argument("--balanced-dynamics-sampling", action="store_true")
    parser.add_argument("--disable-balanced-dynamics-sampling", action="store_true")
    parser.add_argument("--correlated-size-mass-sampling", action="store_true")
    parser.add_argument("--disable-correlated-size-mass-sampling", action="store_true")
    parser.add_argument("--persistent-episode-training", action="store_true")
    parser.add_argument("--training-episode-steps", type=int, default=500)
    parser.add_argument("--direct-h500-training", action="store_true")
    parser.add_argument("--gpu-rollout", action="store_true")
    parser.add_argument("--sampler-audit-path", default="")
    parser.add_argument("--numerics-audit", action="store_true")
    parser.add_argument("--numerics-audit-path", default="")
    parser.add_argument("--angular-audit-path", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-seeds", default="")
    parser.add_argument("--eval-seed-count", type=int, default=1)
    parser.add_argument("--trajectory-path", default="")
    parser.add_argument("--trajectory-count", type=int, default=5)
    parser.add_argument("--success-position-m", type=float, default=0.05)
    parser.add_argument("--success-angle-rad", type=float, default=0.10)
    parser.add_argument("--success-velocity", type=float, default=0.10)
    parser.add_argument("--success-omega", type=float, default=0.20)
    args = parser.parse_args()

    raw_args = set(sys.argv[1:])
    args.sample_dynamics_overridden = args.sample_dynamics or args.fixed_dynamics
    if args.fixed_dynamics:
        args.sample_dynamics = False
    args.sampled_dynamics_level_overridden = "--sampled-dynamics-level" in raw_args
    args.balanced_dynamics_sampling_overridden = (
        "--balanced-dynamics-sampling" in raw_args or "--disable-balanced-dynamics-sampling" in raw_args
    )
    args.correlated_size_mass_sampling_overridden = (
        "--correlated-size-mass-sampling" in raw_args or "--disable-correlated-size-mass-sampling" in raw_args
    )
    if args.disable_balanced_dynamics_sampling:
        args.balanced_dynamics_sampling = False
    if args.disable_correlated_size_mass_sampling:
        args.correlated_size_mass_sampling = False
    if args.gpu_rollout:
        args.sim_backend = "cuda-full"
    if args.numerics_audit_path == "":
        args.numerics_audit_path = str(Path(args.log_path).with_name(Path(args.log_path).name.replace(".csv", "_numerics_audit.csv")))
    if args.angular_audit_path == "":
        args.angular_audit_path = str(Path(args.log_path).with_name(Path(args.log_path).name.replace(".csv", "_angular_audit.csv")))
    if args.training_episode_steps <= 0:
        raise ValueError("--training-episode-steps must be positive")
    return args


def apply_direct_h500_training_defaults(options: argparse.Namespace) -> None:
    if not options.direct_h500_training:
        return
    if not options.sample_dynamics_overridden:
        options.sample_dynamics = False
    if not options.sampled_dynamics_level_overridden:
        options.sampled_dynamics_level = "small"
    if not options.balanced_dynamics_sampling_overridden:
        options.balanced_dynamics_sampling = False
    if not options.correlated_size_mass_sampling_overridden:
        options.correlated_size_mass_sampling = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_sim_backend(name: str, device: torch.device) -> str:
    if name == "torch":
        return "torch"
    if name == "cuda":
        if device.type != "cuda":
            raise ValueError("--sim-backend cuda requires --device cuda or auto with CUDA available")
        load_extension()
        return "cuda"
    if name == "cuda-full":
        if device.type != "cuda":
            raise ValueError("--sim-backend cuda-full requires --device cuda or auto with CUDA available")
        load_extension()
        return "cuda-full"
    if device.type == "cuda" and cuda_backend_available():
        return "cuda-full"
    return "torch"


def _base_fieldnames() -> tuple[str, ...]:
    return (
        "step",
        "loss",
        "tracking",
        "position",
        "velocity",
        "attitude",
        "omega",
        "clf",
        "outward",
        "tail",
        "du",
        "ddu",
        "sat",
        "grad_norm",
        "grad_norm_fp64_before_clip",
        "grad_norm_fp64_after_clip",
        "max_abs_grad_before_clip",
        "grad_scale",
        "update_applied",
        "skip_reason",
        "max_abs_param_delta",
        "seconds",
        "episode_id",
        "segment_id",
        "reset_mask",
        "mass",
        "cbrt_mass",
        "thrust_to_weight",
        "torque_to_inertia",
        "alpha_roll_max",
        "alpha_pitch_max",
        "alpha_yaw_max",
        "eta_yaw",
        "jz_over_jxy",
        "dt_alpha_roll_max",
        "dt_alpha_yaw_max",
        "rotor_distance_factor",
        "inertia_factor",
        "tau_rise",
        "tau_fall",
        "rotor_torque_constant",
        "force_std",
        "f_ext_x",
        "f_ext_y",
        "f_ext_z",
        "f_ext_norm",
    )


def _status_fieldnames() -> tuple[str, ...]:
    return (
        "success_rate",
        "tail_success_rate",
        "invalid_fraction",
    )


def _raptor_fieldnames(settling_position_mm: float) -> tuple[str, ...]:
    names: list[str] = []
    for prefix in ("full", "tail"):
        for quantity in RAPTOR_QUANTITIES:
            names.append(f"{prefix}_{quantity}_mean")
            names.append(f"{prefix}_{quantity}_max_mean")
            names.append(f"{prefix}_{quantity}_max_std")
    names.append(f"full_position_settling_fraction_{int(settling_position_mm)}mm")
    names.append(f"tail_position_settling_fraction_{int(settling_position_mm)}mm")
    return tuple(names)


def open_log(path: Path, settling_position_mm: float) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=_base_fieldnames() + _status_fieldnames() + _raptor_fieldnames(settling_position_mm),
    )
    writer.writeheader()
    return handle, writer


def open_sampler_audit_log(path: Path) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "step",
            "outer_step",
            "batch_index",
            "episode_id",
            "segment_id",
            "reset_mask",
            "mass",
            "cbrt_mass",
            "thrust_to_weight",
            "torque_to_inertia",
            "alpha_roll_max",
            "alpha_pitch_max",
            "alpha_yaw_max",
            "eta_yaw",
            "jz_over_jxy",
            "dt_alpha_roll_max",
            "dt_alpha_yaw_max",
            "rotor_distance_factor",
            "inertia_factor",
            "tau_rise",
            "tau_fall",
            "rotor_torque_constant",
            "force_std",
            "f_ext_x",
            "f_ext_y",
            "f_ext_z",
            "f_ext_norm",
        ),
    )
    writer.writeheader()
    return handle, writer


def open_numerics_audit_log(path: Path) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "train_step",
            "rollout_step",
            "phase",
            "first_bad_step",
            "first_bad_batch",
            "bad_code",
            "loss_finite",
            "state_finite",
            "grad_finite",
            "param_finite_before",
            "param_finite_after",
            "loss",
            "tracking",
            "position_loss",
            "velocity_loss",
            "attitude_loss",
            "omega_loss",
            "clf",
            "outward",
            "tail",
            "du",
            "ddu",
            "sat",
            "max_abs_p",
            "max_abs_v",
            "max_abs_omega",
            "max_abs_motor",
            "max_abs_action",
            "max_thrust",
            "max_acc",
            "max_torque",
            "max_external_acc",
            "max_rotation_orthogonality_error",
            "min_mass",
            "max_mass",
            "min_inertia",
            "max_inertia",
            "min_tau_rise",
            "max_tau_rise",
            "min_tau_fall",
            "max_tau_fall",
            "max_potential",
            "max_clf_delta",
            "max_lp_adj",
            "max_lv_adj",
            "max_lR_adj",
            "max_lw_adj",
            "max_lm_adj",
            "max_lpa_adj",
            "max_action_adj",
            "max_hidden_adj",
            "grad_norm",
            "grad_norm_fp64_before_clip",
            "grad_norm_fp64_after_clip",
            "max_abs_grad_before_clip",
            "grad_scale",
            "update_applied",
            "skip_reason",
            "max_abs_param_delta",
            "max_abs_grad",
            "max_abs_grad_encoder",
            "max_abs_grad_gru",
            "max_abs_grad_head",
            "max_abs_param_before",
            "max_abs_param_after",
            "first_nan_param_name",
        ),
    )
    writer.writeheader()
    return handle, writer


def open_angular_audit_log(path: Path) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "train_step",
            "first_bad_step",
            "precursor_step",
            "batch_index",
            "bad_code",
            "omega_before_norm",
            "omega_after_norm",
            "phi_norm",
            "torque_norm",
            "torque_over_inertia_norm",
            "gyro_cross_norm",
            "gyro_cross_over_inertia_norm",
            "angular_rhs_norm",
            "finite_difference_omega_dot_norm",
            "inertia_x",
            "inertia_y",
            "inertia_z",
            "rotor_torque_constant",
            "torque_x",
            "torque_y",
            "torque_z",
            "omega_before_x",
            "omega_before_y",
            "omega_before_z",
            "omega_after_x",
            "omega_after_y",
            "omega_after_z",
            "action_0",
            "action_1",
            "action_2",
            "action_3",
            "motor_before_0",
            "motor_before_1",
            "motor_before_2",
            "motor_before_3",
            "motor_after_0",
            "motor_after_1",
            "motor_after_2",
            "motor_after_3",
            "action_roll_asym",
            "action_pitch_asym",
            "action_yaw_mix",
            "motor_roll_asym",
            "motor_pitch_asym",
            "motor_yaw_mix",
            "thrust_0",
            "thrust_1",
            "thrust_2",
            "thrust_3",
            "r_orth_error_before",
            "r_orth_error_after",
            "mass",
            "thrust_to_weight",
            "torque_to_inertia",
            "alpha_roll_max",
            "alpha_pitch_max",
            "alpha_yaw_max",
            "eta_yaw",
            "jz_over_jxy",
            "dt_alpha_roll_max",
            "dt_alpha_yaw_max",
            "rotor_distance_factor",
            "inertia_factor",
            "tau_rise",
            "tau_fall",
            "force_std",
            "f_ext_norm",
        ),
    )
    writer.writeheader()
    return handle, writer


def open_eval_log(path: Path, settling_position_mm: float) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "eval_seed",
            "eval_batch_size",
            "eval_horizon",
            "seconds",
        )
        + _status_fieldnames()
        + _raptor_fieldnames(settling_position_mm),
    )
    writer.writeheader()
    return handle, writer


def _quat_angle_from_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]
    qw = torch.sqrt(torch.clamp(1.0 + trace, min=0.0)) * 0.5
    angle = 2.0 * torch.acos(torch.clamp(qw, -1.0, 1.0))
    return angle.abs()


def _action_metrics(action: torch.Tensor, action_min: float = -1.0, action_max: float = 1.0, hover_relative: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    half_range = (action_max - action_min) * 0.5
    action_value = action * half_range + action_min + half_range
    hovering_value = hover_relative * (action_max - action_min) + action_min
    action_metric = (action_value - hovering_value).abs().mean(dim=-1)
    action_relative_metric = ((action + 1.0) * 0.5 - hover_relative).abs().mean(dim=-1)
    return action_metric, action_relative_metric


def _clone_state(state):
    return type(state)(
        position=state.position.detach().clone(),
        velocity=state.velocity.detach().clone(),
        rotation=state.rotation.detach().clone(),
        omega=state.omega.detach().clone(),
        motor=state.motor.detach().clone(),
        previous_action=state.previous_action.detach().clone(),
        external_force=state.external_force.detach().clone(),
        mass=state.mass.detach().clone(),
        thrust_coeff_c0=state.thrust_coeff_c0.detach().clone(),
        thrust_coeff_c1=state.thrust_coeff_c1.detach().clone(),
        thrust_coeff_c2=state.thrust_coeff_c2.detach().clone(),
        thrust_to_weight=state.thrust_to_weight.detach().clone(),
        torque_to_inertia=state.torque_to_inertia.detach().clone(),
        rotor_distance_factor=state.rotor_distance_factor.detach().clone(),
        inertia_factor=state.inertia_factor.detach().clone(),
        motor_time_rising=state.motor_time_rising.detach().clone(),
        motor_time_falling=state.motor_time_falling.detach().clone(),
        rotor_torque_constant=state.rotor_torque_constant.detach().clone(),
        cbrt_mass=state.cbrt_mass.detach().clone(),
        force_std=state.force_std.detach().clone(),
        arm_length=state.arm_length.detach().clone(),
        inertia_x=state.inertia_x.detach().clone(),
        inertia_y=state.inertia_y.detach().clone(),
        inertia_z=state.inertia_z.detach().clone(),
        alpha_roll_max=state.alpha_roll_max.detach().clone(),
        alpha_pitch_max=state.alpha_pitch_max.detach().clone(),
        alpha_yaw_max=state.alpha_yaw_max.detach().clone(),
        eta_yaw=state.eta_yaw.detach().clone(),
        jz_over_jxy=state.jz_over_jxy.detach().clone(),
        dt_alpha_roll_max=state.dt_alpha_roll_max.detach().clone(),
        dt_alpha_yaw_max=state.dt_alpha_yaw_max.detach().clone(),
    )


def _sampler_audit_row(
    step_idx: int,
    outer_step: int,
    batch_index: int,
    episode_id: int,
    segment_id: int,
    reset_mask: bool,
    state,
) -> dict[str, float | int]:
    external = state.external_force[batch_index]
    return {
        "step": step_idx,
        "outer_step": outer_step,
        "batch_index": batch_index,
        "episode_id": episode_id,
        "segment_id": segment_id,
        "reset_mask": int(reset_mask),
        "mass": float(state.mass[batch_index].item()),
        "cbrt_mass": float(state.cbrt_mass[batch_index].item()),
        "thrust_to_weight": float(state.thrust_to_weight[batch_index].item()),
        "torque_to_inertia": float(state.torque_to_inertia[batch_index].item()),
        "alpha_roll_max": float(state.alpha_roll_max[batch_index].item()),
        "alpha_pitch_max": float(state.alpha_pitch_max[batch_index].item()),
        "alpha_yaw_max": float(state.alpha_yaw_max[batch_index].item()),
        "eta_yaw": float(state.eta_yaw[batch_index].item()),
        "jz_over_jxy": float(state.jz_over_jxy[batch_index].item()),
        "dt_alpha_roll_max": float(state.dt_alpha_roll_max[batch_index].item()),
        "dt_alpha_yaw_max": float(state.dt_alpha_yaw_max[batch_index].item()),
        "rotor_distance_factor": float(state.rotor_distance_factor[batch_index].item()),
        "inertia_factor": float(state.inertia_factor[batch_index].item()),
        "tau_rise": float(state.motor_time_rising[batch_index].item()),
        "tau_fall": float(state.motor_time_falling[batch_index].item()),
        "rotor_torque_constant": float(state.rotor_torque_constant[batch_index].item()),
        "force_std": float(state.force_std[batch_index].item()),
        "f_ext_x": float(external[0].item()),
        "f_ext_y": float(external[1].item()),
        "f_ext_z": float(external[2].item()),
        "f_ext_norm": float(torch.linalg.norm(external).item()),
    }


def _numerics_log_steps(horizon: int) -> list[int]:
    points = [1, 2, 4, 8, 16, 32, 64, 128, 250, horizon]
    return sorted({step for step in points if 1 <= step <= horizon})


def _safe_max_abs(tensor: torch.Tensor) -> float:
    value = torch.nan_to_num(tensor.detach().abs(), nan=float("inf"), posinf=float("inf"), neginf=float("inf")).max()
    return float(value.item())


def _safe_min(tensor: torch.Tensor) -> float:
    value = torch.nan_to_num(tensor.detach(), nan=float("inf"), posinf=float("inf"), neginf=-float("inf")).min()
    return float(value.item())


def _safe_max(tensor: torch.Tensor) -> float:
    value = torch.nan_to_num(tensor.detach(), nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")).max()
    return float(value.item())


def _safe_scalar_norm(tensor: torch.Tensor) -> float:
    value = torch.nan_to_num(tensor.detach(), nan=float("inf"), posinf=float("inf"), neginf=-float("inf"))
    return float(torch.linalg.norm(value).item())


def _rotation_orth_error_single(rotation: torch.Tensor) -> float:
    if not torch.isfinite(rotation).all().item():
        return float("inf")
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    return float(torch.linalg.norm(rotation.transpose(0, 1) @ rotation - identity).item())


def _first_nonfinite_batch(tensor: torch.Tensor) -> int:
    flat = tensor.detach().reshape(tensor.shape[0], -1)
    finite = torch.isfinite(flat).all(dim=-1)
    bad = torch.where(~finite)[0]
    if bad.numel() == 0:
        return -1
    return int(bad[0].item())


def _first_bad_from_checks(checks: list[tuple[torch.Tensor, int]]) -> tuple[int, int]:
    for tensor, code in checks:
        batch_index = _first_nonfinite_batch(tensor)
        if batch_index >= 0:
            return batch_index, code
    return -1, 0


def _first_nonfinite_step_batch(tensor: torch.Tensor, *, step_offset: int) -> tuple[int, int]:
    flat = tensor.detach().reshape(tensor.shape[0], tensor.shape[1], -1)
    finite = torch.isfinite(flat).all(dim=-1)
    bad = torch.where(~finite)
    if bad[0].numel() == 0:
        return -1, -1
    linear = bad[0] * tensor.shape[1] + bad[1]
    arg = torch.argmin(linear)
    return int(bad[0][arg].item()) + step_offset, int(bad[1][arg].item())


def _first_nonfinite_step_from_sequences(
    checks: list[tuple[torch.Tensor, int, int]],
) -> tuple[int, int, int]:
    best_step = -1
    best_batch = -1
    best_code = 0
    for tensor, code, step_offset in checks:
        step, batch = _first_nonfinite_step_batch(tensor, step_offset=step_offset)
        if step < 0:
            continue
        if best_step < 0 or step < best_step or (step == best_step and batch < best_batch):
            best_step = step
            best_batch = batch
            best_code = code
    return best_step, best_batch, best_code


def _param_stats(policy: MotorGRUPolicy) -> tuple[float, bool, str]:
    max_abs = 0.0
    finite = True
    first_bad = ""
    for name, param in policy.named_parameters():
        data = param.detach()
        if data.numel() > 0:
            max_abs = max(max_abs, _safe_max_abs(data))
        if not torch.isfinite(data).all().item():
            finite = False
            if first_bad == "":
                first_bad = name
    return max_abs, finite, first_bad


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _set_optimizer_hparams(optimizer: torch.optim.Optimizer, *, lr: float, weight_decay: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = weight_decay


def _param_delta_snapshot(policy: MotorGRUPolicy) -> list[torch.Tensor]:
    return [param.detach().clone() for param in policy.parameters()]


def _max_abs_param_delta(policy: MotorGRUPolicy, before: list[torch.Tensor]) -> float:
    values = [
        (param.detach() - previous).abs().amax()
        for param, previous in zip(policy.parameters(), before)
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).amax().item())


def _grad_norm_fp64_tensor(policy: MotorGRUPolicy) -> torch.Tensor:
    device = next(policy.parameters()).device
    total_sq = torch.zeros((), device=device, dtype=torch.float64)
    for param in policy.parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            total_sq = total_sq + grad.double().square().sum()
    return torch.sqrt(total_sq)


def _apply_fp64_global_grad_clip(policy: MotorGRUPolicy, grad_clip: float) -> tuple[float, float, float]:
    norm_before = _grad_norm_fp64_tensor(policy)
    norm_before_value = float(norm_before.item())
    scale = 1.0
    if grad_clip > 0.0 and torch.isfinite(norm_before).item():
        scale = min(1.0, grad_clip / (norm_before_value + 1.0e-12))
        for param in policy.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
    norm_after = _grad_norm_fp64_tensor(policy)
    return norm_before_value, float(norm_after.item()), float(scale)


def _grad_stats(policy: MotorGRUPolicy) -> dict[str, float | bool | str]:
    stats: dict[str, float | bool | str] = {
        "max_abs_grad": 0.0,
        "max_abs_grad_encoder": 0.0,
        "max_abs_grad_gru": 0.0,
        "max_abs_grad_head": 0.0,
        "grad_finite": True,
        "first_nan_param_name": "",
    }
    for name, param in policy.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        max_abs = _safe_max_abs(grad)
        stats["max_abs_grad"] = max(float(stats["max_abs_grad"]), max_abs)
        if name.startswith("encoder"):
            stats["max_abs_grad_encoder"] = max(float(stats["max_abs_grad_encoder"]), max_abs)
        elif name.startswith("gru"):
            stats["max_abs_grad_gru"] = max(float(stats["max_abs_grad_gru"]), max_abs)
        elif name.startswith("motor_head"):
            stats["max_abs_grad_head"] = max(float(stats["max_abs_grad_head"]), max_abs)
        if not torch.isfinite(grad).all().item():
            stats["grad_finite"] = False
            if stats["first_nan_param_name"] == "":
                stats["first_nan_param_name"] = name
    return stats


def _empty_numerics_row(train_step: int, rollout_step: int, phase: str) -> dict[str, float | int | str]:
    return {
        "train_step": train_step,
        "rollout_step": rollout_step,
        "phase": phase,
        "first_bad_step": -1,
        "first_bad_batch": -1,
        "bad_code": 0,
        "loss_finite": "",
        "state_finite": "",
        "grad_finite": "",
        "param_finite_before": "",
        "param_finite_after": "",
        "loss": float("nan"),
        "tracking": float("nan"),
        "position_loss": float("nan"),
        "velocity_loss": float("nan"),
        "attitude_loss": float("nan"),
        "omega_loss": float("nan"),
        "clf": float("nan"),
        "outward": float("nan"),
        "tail": float("nan"),
        "du": float("nan"),
        "ddu": float("nan"),
        "sat": float("nan"),
        "max_abs_p": float("nan"),
        "max_abs_v": float("nan"),
        "max_abs_omega": float("nan"),
        "max_abs_motor": float("nan"),
        "max_abs_action": float("nan"),
        "max_thrust": float("nan"),
        "max_acc": float("nan"),
        "max_torque": float("nan"),
        "max_external_acc": float("nan"),
        "max_rotation_orthogonality_error": float("nan"),
        "min_mass": float("nan"),
        "max_mass": float("nan"),
        "min_inertia": float("nan"),
        "max_inertia": float("nan"),
        "min_tau_rise": float("nan"),
        "max_tau_rise": float("nan"),
        "min_tau_fall": float("nan"),
        "max_tau_fall": float("nan"),
        "max_potential": float("nan"),
        "max_clf_delta": float("nan"),
        "max_lp_adj": float("nan"),
        "max_lv_adj": float("nan"),
        "max_lR_adj": float("nan"),
        "max_lw_adj": float("nan"),
        "max_lm_adj": float("nan"),
        "max_lpa_adj": float("nan"),
        "max_action_adj": float("nan"),
            "max_hidden_adj": float("nan"),
            "grad_norm": float("nan"),
            "grad_norm_fp64_before_clip": float("nan"),
            "grad_norm_fp64_after_clip": float("nan"),
            "max_abs_grad_before_clip": float("nan"),
            "grad_scale": float("nan"),
            "update_applied": "",
            "skip_reason": "",
            "max_abs_param_delta": float("nan"),
            "max_abs_grad": float("nan"),
        "max_abs_grad_encoder": float("nan"),
        "max_abs_grad_gru": float("nan"),
        "max_abs_grad_head": float("nan"),
        "max_abs_param_before": float("nan"),
        "max_abs_param_after": float("nan"),
        "first_nan_param_name": "",
    }


def _write_cuda_numerics_rows(
    writer: csv.DictWriter,
    *,
    angular_writer: csv.DictWriter | None = None,
    train_step: int,
    horizon: int,
    metrics_tensor: torch.Tensor,
    debug_tensors: tuple[torch.Tensor, ...],
    initial_state: L2FState,
    params: L2FParams,
    clf_kappa: float,
) -> None:
    (
        actions,
        p_states,
        v_states,
        r_states,
        w_states,
        motor_states,
        previous_action_states,
        potentials,
        hidden_adj_mag,
        lp_adj,
        lv_adj,
        lR_adj,
        lw_adj,
        lm_adj,
        lpa_adj,
        action_adj,
    ) = debug_tensors
    del previous_action_states
    batch = initial_state.mass.shape[0]
    identity = torch.eye(3, device=r_states.device, dtype=r_states.dtype).expand(batch, 3, 3)
    inertia_stack = torch.stack((initial_state.inertia_x, initial_state.inertia_y, initial_state.inertia_z), dim=-1)
    mass = initial_state.mass[:, None]
    clf_decay = max(0.0, 1.0 - clf_kappa * params.dt)
    detached_metrics = metrics_tensor.detach()
    thrust_sequence = (
        initial_state.thrust_coeff_c0[None, :, :]
        + initial_state.thrust_coeff_c1[None, :, :] * motor_states[1:]
        + initial_state.thrust_coeff_c2[None, :, :] * motor_states[1:] * motor_states[1:]
    ).clamp_min(0.0)
    first_forward_step, first_forward_batch, first_forward_code = _first_nonfinite_step_from_sequences(
        [
            (p_states, 1, 0),
            (v_states, 2, 0),
            (r_states, 3, 0),
            (w_states, 4, 0),
            (motor_states, 5, 0),
            (actions, 6, 1),
            (thrust_sequence, 7, 1),
        ]
    )
    first_loss_step, first_loss_batch, first_loss_code = _first_nonfinite_step_from_sequences(
        [(potentials[1:], 8, 1)]
    )
    if not torch.isfinite(detached_metrics).all().item() and first_loss_step < 0:
        first_loss_step, first_loss_batch, first_loss_code = horizon, -1, 8
    first_adjoint_step, first_adjoint_batch, first_adjoint_code = _first_nonfinite_step_from_sequences(
        [
            (lp_adj, 9, 0),
            (lv_adj, 9, 0),
            (lR_adj, 9, 0),
            (lw_adj, 9, 0),
            (lm_adj, 9, 0),
            (lpa_adj, 9, 0),
            (action_adj, 9, 1),
            (hidden_adj_mag, 9, 0),
        ]
    )
    if first_forward_step >= 0:
        first_bad_step, first_bad_batch, first_bad_code = first_forward_step, first_forward_batch, first_forward_code
    elif first_loss_step >= 0:
        first_bad_step, first_bad_batch, first_bad_code = first_loss_step, first_loss_batch, first_loss_code
    else:
        first_bad_step, first_bad_batch, first_bad_code = first_adjoint_step, first_adjoint_batch, first_adjoint_code

    if angular_writer is not None and first_bad_step >= 1 and first_bad_batch >= 0:
        precursor_step = max(first_bad_step - 1, 0)
        action_idx = min(max(first_bad_step - 1, 0), horizon - 1)
        after_step = min(max(first_bad_step, 0), horizon)
        b = first_bad_batch
        omega_before = w_states[precursor_step, b]
        omega_after = w_states[after_step, b]
        motor_before = motor_states[precursor_step, b]
        motor_after = motor_states[after_step, b]
        action = actions[action_idx, b]
        thrust = (
            initial_state.thrust_coeff_c0[b]
            + initial_state.thrust_coeff_c1[b] * motor_after
            + initial_state.thrust_coeff_c2[b] * motor_after * motor_after
        ).clamp_min(0.0)
        torque = torch.stack(
            (
                initial_state.arm_length[b] * (thrust[1] - thrust[3]),
                initial_state.arm_length[b] * (thrust[2] - thrust[0]),
                initial_state.rotor_torque_constant[b] * (thrust[0] - thrust[1] + thrust[2] - thrust[3]),
            )
        )
        inertia = torch.stack((initial_state.inertia_x[b], initial_state.inertia_y[b], initial_state.inertia_z[b]))
        gyro_cross = torch.cross(omega_before, omega_before * inertia, dim=0)
        angular_rhs = (torque - gyro_cross) / inertia
        fd_omega_dot = (omega_after - omega_before) / params.dt
        phi = params.dt * omega_after
        row = {
            "train_step": train_step,
            "first_bad_step": first_bad_step,
            "precursor_step": precursor_step,
            "batch_index": b,
            "bad_code": first_bad_code,
            "omega_before_norm": _safe_scalar_norm(omega_before),
            "omega_after_norm": _safe_scalar_norm(omega_after),
            "phi_norm": _safe_scalar_norm(phi),
            "torque_norm": _safe_scalar_norm(torque),
            "torque_over_inertia_norm": _safe_scalar_norm(torque / inertia),
            "gyro_cross_norm": _safe_scalar_norm(gyro_cross),
            "gyro_cross_over_inertia_norm": _safe_scalar_norm(gyro_cross / inertia),
            "angular_rhs_norm": _safe_scalar_norm(angular_rhs),
            "finite_difference_omega_dot_norm": _safe_scalar_norm(fd_omega_dot),
            "inertia_x": float(initial_state.inertia_x[b].item()),
            "inertia_y": float(initial_state.inertia_y[b].item()),
            "inertia_z": float(initial_state.inertia_z[b].item()),
            "rotor_torque_constant": float(initial_state.rotor_torque_constant[b].item()),
            "torque_x": float(torque[0].item()),
            "torque_y": float(torque[1].item()),
            "torque_z": float(torque[2].item()),
            "omega_before_x": float(omega_before[0].item()),
            "omega_before_y": float(omega_before[1].item()),
            "omega_before_z": float(omega_before[2].item()),
            "omega_after_x": float(omega_after[0].item()),
            "omega_after_y": float(omega_after[1].item()),
            "omega_after_z": float(omega_after[2].item()),
            "action_0": float(action[0].item()),
            "action_1": float(action[1].item()),
            "action_2": float(action[2].item()),
            "action_3": float(action[3].item()),
            "motor_before_0": float(motor_before[0].item()),
            "motor_before_1": float(motor_before[1].item()),
            "motor_before_2": float(motor_before[2].item()),
            "motor_before_3": float(motor_before[3].item()),
            "motor_after_0": float(motor_after[0].item()),
            "motor_after_1": float(motor_after[1].item()),
            "motor_after_2": float(motor_after[2].item()),
            "motor_after_3": float(motor_after[3].item()),
            "action_roll_asym": float((action[1] - action[3]).abs().item()),
            "action_pitch_asym": float((action[2] - action[0]).abs().item()),
            "action_yaw_mix": float((action[0] - action[1] + action[2] - action[3]).abs().item()),
            "motor_roll_asym": float((motor_after[1] - motor_after[3]).abs().item()),
            "motor_pitch_asym": float((motor_after[2] - motor_after[0]).abs().item()),
            "motor_yaw_mix": float((motor_after[0] - motor_after[1] + motor_after[2] - motor_after[3]).abs().item()),
            "thrust_0": float(thrust[0].item()),
            "thrust_1": float(thrust[1].item()),
            "thrust_2": float(thrust[2].item()),
            "thrust_3": float(thrust[3].item()),
            "r_orth_error_before": _rotation_orth_error_single(r_states[precursor_step, b]),
            "r_orth_error_after": _rotation_orth_error_single(r_states[after_step, b]),
            "mass": float(initial_state.mass[b].item()),
            "thrust_to_weight": float(initial_state.thrust_to_weight[b].item()),
            "torque_to_inertia": float(initial_state.torque_to_inertia[b].item()),
            "alpha_roll_max": float(initial_state.alpha_roll_max[b].item()),
            "alpha_pitch_max": float(initial_state.alpha_pitch_max[b].item()),
            "alpha_yaw_max": float(initial_state.alpha_yaw_max[b].item()),
            "eta_yaw": float(initial_state.eta_yaw[b].item()),
            "jz_over_jxy": float(initial_state.jz_over_jxy[b].item()),
            "dt_alpha_roll_max": float(initial_state.dt_alpha_roll_max[b].item()),
            "dt_alpha_yaw_max": float(initial_state.dt_alpha_yaw_max[b].item()),
            "rotor_distance_factor": float(initial_state.rotor_distance_factor[b].item()),
            "inertia_factor": float(initial_state.inertia_factor[b].item()),
            "tau_rise": float(initial_state.motor_time_rising[b].item()),
            "tau_fall": float(initial_state.motor_time_falling[b].item()),
            "force_std": float(initial_state.force_std[b].item()),
            "f_ext_norm": float(torch.linalg.norm(initial_state.external_force[b]).item()),
        }
        angular_writer.writerow(row)

    for rollout_step in _numerics_log_steps(horizon):
        action_step = rollout_step - 1
        p_s = p_states[rollout_step]
        v_s = v_states[rollout_step]
        r_s = r_states[rollout_step]
        w_s = w_states[rollout_step]
        motor_s = motor_states[rollout_step]
        action_s = actions[action_step]
        thrust = (
            initial_state.thrust_coeff_c0
            + initial_state.thrust_coeff_c1 * motor_s
            + initial_state.thrust_coeff_c2 * motor_s * motor_s
        ).clamp_min(0.0)
        total_thrust = thrust.sum(dim=-1, keepdim=True)
        r_prev = r_states[action_step]
        body_z = r_prev[:, :, 2]
        gravity = torch.tensor((0.0, 0.0, -params.gravity), device=p_s.device, dtype=p_s.dtype)
        acc = body_z * (total_thrust / mass) + gravity + initial_state.external_force / mass
        torque = torch.stack(
            (
                initial_state.arm_length * (thrust[:, 1] - thrust[:, 3]),
                initial_state.arm_length * (thrust[:, 2] - thrust[:, 0]),
                initial_state.rotor_torque_constant * (thrust[:, 0] - thrust[:, 1] + thrust[:, 2] - thrust[:, 3]),
            ),
            dim=-1,
        )
        rot_orth = torch.linalg.norm(r_s.transpose(1, 2) @ r_s - identity, dim=(1, 2))
        clf_delta = potentials[rollout_step] - clf_decay * potentials[action_step]
        row = _empty_numerics_row(train_step, rollout_step, "rollout")
        row.update(
            {
                "first_bad_step": first_bad_step,
                "first_bad_batch": first_bad_batch,
                "bad_code": first_bad_code,
                "loss_finite": int(torch.isfinite(detached_metrics).all().item()),
                "state_finite": int(first_forward_step < 0),
                "loss": float(detached_metrics[0].item()),
                "tracking": float(detached_metrics[1].item()),
                "position_loss": float(detached_metrics[2].item()),
                "velocity_loss": float(detached_metrics[3].item()),
                "attitude_loss": float(detached_metrics[4].item()),
                "omega_loss": float(detached_metrics[5].item()),
                "clf": float(detached_metrics[6].item()),
                "outward": float(detached_metrics[7].item()),
                "tail": float(detached_metrics[8].item()),
                "du": float(detached_metrics[9].item()),
                "ddu": float(detached_metrics[10].item()),
                "sat": float(detached_metrics[11].item()),
                "max_abs_p": _safe_max_abs(p_s),
                "max_abs_v": _safe_max_abs(v_s),
                "max_abs_omega": _safe_max_abs(w_s),
                "max_abs_motor": _safe_max_abs(motor_s),
                "max_abs_action": _safe_max_abs(action_s),
                "max_thrust": _safe_max(thrust),
                "max_acc": _safe_max_abs(acc),
                "max_torque": _safe_max_abs(torque),
                "max_external_acc": _safe_max_abs(initial_state.external_force / mass),
                "max_rotation_orthogonality_error": _safe_max(rot_orth),
                "min_mass": _safe_min(initial_state.mass),
                "max_mass": _safe_max(initial_state.mass),
                "min_inertia": _safe_min(inertia_stack),
                "max_inertia": _safe_max(inertia_stack),
                "min_tau_rise": _safe_min(initial_state.motor_time_rising),
                "max_tau_rise": _safe_max(initial_state.motor_time_rising),
                "min_tau_fall": _safe_min(initial_state.motor_time_falling),
                "max_tau_fall": _safe_max(initial_state.motor_time_falling),
                "max_potential": _safe_max(potentials[rollout_step]),
                "max_clf_delta": _safe_max(clf_delta),
                "max_lp_adj": _safe_max_abs(lp_adj[rollout_step]),
                "max_lv_adj": _safe_max_abs(lv_adj[rollout_step]),
                "max_lR_adj": _safe_max_abs(lR_adj[rollout_step]),
                "max_lw_adj": _safe_max_abs(lw_adj[rollout_step]),
                "max_lm_adj": _safe_max_abs(lm_adj[rollout_step]),
                "max_lpa_adj": _safe_max_abs(lpa_adj[rollout_step]),
                "max_action_adj": _safe_max_abs(action_adj[action_step]),
                "max_hidden_adj": _safe_max_abs(hidden_adj_mag[action_step]),
            }
        )
        writer.writerow(row)


def _write_optimizer_numerics_row(
    writer: csv.DictWriter,
    *,
    train_step: int,
    loss_finite: bool,
    state_finite: bool,
    grad_norm: float,
    grad_stats: dict[str, float | bool | str],
    grad_norm_fp64_before_clip: float,
    grad_norm_fp64_after_clip: float,
    max_abs_grad_before_clip: float,
    grad_scale: float,
    update_applied: bool,
    skip_reason: str,
    max_abs_param_delta: float,
    param_before: tuple[float, bool, str],
    param_after: tuple[float, bool, str],
) -> None:
    row = _empty_numerics_row(train_step, -1, "optimizer")
    first_bad = str(grad_stats["first_nan_param_name"] or param_after[2] or param_before[2])
    row.update(
        {
            "bad_code": 10 if not bool(grad_stats["grad_finite"]) else 0,
            "loss_finite": int(loss_finite),
            "state_finite": int(state_finite),
            "grad_finite": int(bool(grad_stats["grad_finite"])),
            "param_finite_before": int(param_before[1]),
            "param_finite_after": int(param_after[1]),
            "grad_norm": float(grad_norm),
            "grad_norm_fp64_before_clip": float(grad_norm_fp64_before_clip),
            "grad_norm_fp64_after_clip": float(grad_norm_fp64_after_clip),
            "max_abs_grad_before_clip": float(max_abs_grad_before_clip),
            "grad_scale": float(grad_scale),
            "update_applied": int(update_applied),
            "skip_reason": skip_reason,
            "max_abs_param_delta": float(max_abs_param_delta),
            "max_abs_grad": float(grad_stats["max_abs_grad"]),
            "max_abs_grad_encoder": float(grad_stats["max_abs_grad_encoder"]),
            "max_abs_grad_gru": float(grad_stats["max_abs_grad_gru"]),
            "max_abs_grad_head": float(grad_stats["max_abs_grad_head"]),
            "max_abs_param_before": param_before[0],
            "max_abs_param_after": param_after[0],
            "first_nan_param_name": first_bad,
        }
    )
    writer.writerow(row)


def _copy_state_mask(target_state, source_state, mask: torch.Tensor) -> None:
    if not torch.any(mask):
        return
    idx = torch.where(mask)[0]
    target_state.position[idx] = source_state.position[idx]
    target_state.velocity[idx] = source_state.velocity[idx]
    target_state.rotation[idx] = source_state.rotation[idx]
    target_state.omega[idx] = source_state.omega[idx]
    target_state.motor[idx] = source_state.motor[idx]
    target_state.previous_action[idx] = source_state.previous_action[idx]
    target_state.external_force[idx] = source_state.external_force[idx]
    target_state.mass[idx] = source_state.mass[idx]
    target_state.thrust_coeff_c0[idx] = source_state.thrust_coeff_c0[idx]
    target_state.thrust_coeff_c1[idx] = source_state.thrust_coeff_c1[idx]
    target_state.thrust_coeff_c2[idx] = source_state.thrust_coeff_c2[idx]
    target_state.thrust_to_weight[idx] = source_state.thrust_to_weight[idx]
    target_state.torque_to_inertia[idx] = source_state.torque_to_inertia[idx]
    target_state.rotor_distance_factor[idx] = source_state.rotor_distance_factor[idx]
    target_state.inertia_factor[idx] = source_state.inertia_factor[idx]
    target_state.motor_time_rising[idx] = source_state.motor_time_rising[idx]
    target_state.motor_time_falling[idx] = source_state.motor_time_falling[idx]
    target_state.rotor_torque_constant[idx] = source_state.rotor_torque_constant[idx]
    target_state.cbrt_mass[idx] = source_state.cbrt_mass[idx]
    target_state.force_std[idx] = source_state.force_std[idx]
    target_state.arm_length[idx] = source_state.arm_length[idx]
    target_state.inertia_x[idx] = source_state.inertia_x[idx]
    target_state.inertia_y[idx] = source_state.inertia_y[idx]
    target_state.inertia_z[idx] = source_state.inertia_z[idx]
    target_state.alpha_roll_max[idx] = source_state.alpha_roll_max[idx]
    target_state.alpha_pitch_max[idx] = source_state.alpha_pitch_max[idx]
    target_state.alpha_yaw_max[idx] = source_state.alpha_yaw_max[idx]
    target_state.eta_yaw[idx] = source_state.eta_yaw[idx]
    target_state.jz_over_jxy[idx] = source_state.jz_over_jxy[idx]
    target_state.dt_alpha_roll_max[idx] = source_state.dt_alpha_roll_max[idx]
    target_state.dt_alpha_yaw_max[idx] = source_state.dt_alpha_yaw_max[idx]


def _empty_raptor_metrics(settling_position_mm: float) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in _status_fieldnames() + _raptor_fieldnames(settling_position_mm):
        metrics[name] = float("nan")
    return metrics


def _parse_eval_seeds(seed_arg: str, seed: int, count: int) -> list[int]:
    if seed_arg.strip():
        return [int(part.strip()) for part in seed_arg.split(",") if part.strip()]
    return [seed + offset for offset in range(max(count, 1))]


def _window_stats(
    samples: dict[str, list[torch.Tensor]],
    *,
    prefix: str,
    start_step: int,
    horizon: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    start = min(max(start_step, 0), max(horizon - 1, 0))
    for quantity in RAPTOR_QUANTITIES:
        mean, max_mean, max_std = _metric_summary(samples[quantity][start:])
        metrics[f"{prefix}_{quantity}_mean"] = mean
        metrics[f"{prefix}_{quantity}_max_mean"] = max_mean
        metrics[f"{prefix}_{quantity}_max_std"] = max_std
    return metrics


def _write_trajectory_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "eval_seed",
        "sample",
        "step",
        "time_s",
        "p_norm",
        "v_norm",
        "angle_error",
        "omega_norm",
        "action_0",
        "action_1",
        "action_2",
        "action_3",
        "motor_0",
        "motor_1",
        "motor_2",
        "motor_3",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _uniform_noise_like(tensor: torch.Tensor, max_abs: float) -> torch.Tensor:
    if max_abs <= 0.0:
        return torch.zeros_like(tensor)
    return torch.empty_like(tensor).uniform_(-max_abs, max_abs)


@torch.no_grad()
def rollout_diagnostics(
    policy: MotorGRUPolicy,
    sim: L2FSimulator,
    initial_state,
    args: argparse.Namespace,
    *,
    step_backend: str,
    eval_seed: int | None = None,
    trajectory_count: int = 0,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    state = _clone_state(initial_state)
    hidden = None
    samples = {quantity: [] for quantity in RAPTOR_QUANTITIES}
    trajectory_rows: list[dict[str, float | int]] = []
    finite_mask = torch.ones(state.position.shape[0], device=state.position.device, dtype=torch.bool)
    capture_count = min(max(trajectory_count, 0), state.position.shape[0])

    for step_i in range(args.horizon):
        omega_before = state.omega
        state_features = sim.state_features(state)
        error_features = sim.error_features(state)
        previous_action = state.previous_action
        if args.observation_noise_max > 0.0:
            state_features = state_features + _uniform_noise_like(state_features, args.observation_noise_max)
            error_features = error_features + _uniform_noise_like(error_features, args.observation_noise_max)
            previous_action_input = previous_action + _uniform_noise_like(previous_action, args.observation_noise_max)
        else:
            previous_action_input = previous_action
        action, hidden = policy(state_features, error_features, previous_action_input, hidden)

        position_norm = state.position.norm(dim=-1)
        angle_error = _quat_angle_from_rotation_matrix(state.rotation)
        velocity_norm = state.velocity.norm(dim=-1)
        omega_norm = state.omega.norm(dim=-1)
        action_metric, action_relative_metric = _action_metrics(action)

        if args.action_noise_max > 0.0:
            physics_action = (action + _uniform_noise_like(action, args.action_noise_max)).clamp(-1.0, 1.0)
        else:
            physics_action = action
        if step_backend == "cuda":
            next_state = cuda_step(state, physics_action, sim.params, grad_decay=1.0)
        else:
            next_state = sim.step(
                state,
                physics_action,
                grad_decay=1.0,
            )

        angular_acceleration = (next_state.omega - omega_before).norm(dim=-1) / args.dt

        samples["position"].append(position_norm.detach())
        samples["angle"].append(angle_error.detach())
        samples["linear_velocity"].append(velocity_norm.detach())
        samples["angular_velocity"].append(omega_norm.detach())
        samples["angular_acceleration"].append(angular_acceleration.detach())
        samples["action"].append(action_metric.detach())
        samples["action_relative"].append(action_relative_metric.detach())

        finite_mask &= torch.isfinite(position_norm)
        finite_mask &= torch.isfinite(angle_error)
        finite_mask &= torch.isfinite(velocity_norm)
        finite_mask &= torch.isfinite(omega_norm)
        finite_mask &= torch.isfinite(angular_acceleration)
        finite_mask &= torch.isfinite(action).all(dim=-1)
        finite_mask &= torch.isfinite(next_state.motor).all(dim=-1)

        if capture_count > 0:
            p_cpu = position_norm[:capture_count].detach().cpu()
            v_cpu = velocity_norm[:capture_count].detach().cpu()
            angle_cpu = angle_error[:capture_count].detach().cpu()
            omega_cpu = omega_norm[:capture_count].detach().cpu()
            action_cpu = action[:capture_count].detach().cpu()
            motor_cpu = next_state.motor[:capture_count].detach().cpu()
            for sample_i in range(capture_count):
                trajectory_rows.append(
                    {
                        "eval_seed": -1 if eval_seed is None else eval_seed,
                        "sample": sample_i,
                        "step": step_i,
                        "time_s": step_i * args.dt,
                        "p_norm": float(p_cpu[sample_i]),
                        "v_norm": float(v_cpu[sample_i]),
                        "angle_error": float(angle_cpu[sample_i]),
                        "omega_norm": float(omega_cpu[sample_i]),
                        "action_0": float(action_cpu[sample_i, 0]),
                        "action_1": float(action_cpu[sample_i, 1]),
                        "action_2": float(action_cpu[sample_i, 2]),
                        "action_3": float(action_cpu[sample_i, 3]),
                        "motor_0": float(motor_cpu[sample_i, 0]),
                        "motor_1": float(motor_cpu[sample_i, 1]),
                        "motor_2": float(motor_cpu[sample_i, 2]),
                        "motor_3": float(motor_cpu[sample_i, 3]),
                    }
                )

        state = next_state

    metrics = {}
    metrics.update(_window_stats(samples, prefix="full", start_step=0, horizon=args.horizon))
    metrics.update(_window_stats(samples, prefix="tail", start_step=args.tail_start_step, horizon=args.horizon))

    settle_thr = args.settling_position_mm / 1000.0
    final_position = samples["position"][-1]
    tail_start = min(max(args.tail_start_step, 0), max(args.horizon - 1, 0))
    tail_position_stack = torch.stack(samples["position"][tail_start:], dim=0)
    metrics[f"full_position_settling_fraction_{int(args.settling_position_mm)}mm"] = (
        (final_position < settle_thr) & finite_mask
    ).float().mean().item()
    metrics[f"tail_position_settling_fraction_{int(args.settling_position_mm)}mm"] = (
        (tail_position_stack.max(dim=0).values < settle_thr) & finite_mask
    ).float().mean().item()

    final_success = (
        (samples["position"][-1] < args.success_position_m)
        & (samples["angle"][-1] < args.success_angle_rad)
        & (samples["linear_velocity"][-1] < args.success_velocity)
        & (samples["angular_velocity"][-1] < args.success_omega)
        & finite_mask
    )
    tail_success = (
        (torch.stack(samples["position"][tail_start:], dim=0).mean(dim=0) < args.success_position_m)
        & (torch.stack(samples["angle"][tail_start:], dim=0).mean(dim=0) < args.success_angle_rad)
        & (torch.stack(samples["linear_velocity"][tail_start:], dim=0).mean(dim=0) < args.success_velocity)
        & (torch.stack(samples["angular_velocity"][tail_start:], dim=0).mean(dim=0) < args.success_omega)
        & finite_mask
    )
    metrics["success_rate"] = final_success.float().mean().item()
    metrics["tail_success_rate"] = tail_success.float().mean().item()
    metrics["invalid_fraction"] = (~finite_mask).float().mean().item()
    return metrics, trajectory_rows


def main() -> None:
    args = parse_args()
    apply_direct_h500_training_defaults(args)
    if args.external_force_max != 0.0:
        raise ValueError("--external-force-max is deprecated; use --disturbance-force-max for episode-level hidden force")
    if args.external_torque_max != 0.0:
        raise ValueError("--external-torque-max is not part of the RAPTOR-style external-force path")
    device = resolve_device(args.device)
    sim_backend = resolve_sim_backend(args.sim_backend, device)
    if args.sample_dynamics and args.sampled_dynamics_level == "broad" and sim_backend == "cuda":
        sim_backend = "torch"
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    sim = L2FSimulator(
        L2FParams(
            dt=args.dt,
            max_initial_position=args.max_initial_position,
            max_initial_velocity=args.max_initial_velocity,
            max_initial_angle=args.max_initial_angle,
            max_initial_omega=args.max_initial_omega,
            disturbance_force_max=args.disturbance_force_max,
            external_force_ratio=args.external_force_ratio,
        )
    )
    loss_config = L2FLossConfig(
        p_scale=args.p_scale,
        v_scale=args.v_scale,
        omega_scale=args.omega_scale,
        huber_beta=args.huber_beta,
        w_p=args.w_p,
        w_v=args.w_v,
        w_r=args.w_r,
        w_omega=args.w_omega,
        attitude_mode=args.attitude_mode,
    )
    if sim_backend == "cuda-full" and args.attitude_mode != "full":
        raise ValueError("cuda-full backend currently requires --attitude-mode full")
    state_step_decay = args.state_grad_decay ** args.dt
    hidden_step_decay = args.hidden_grad_decay ** args.dt
    policy = MotorGRUPolicy(
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.init_checkpoint_path:
        init_checkpoint = torch.load(args.init_checkpoint_path, map_location=device)
        init_state_dict = init_checkpoint.get("model", init_checkpoint)
        policy.load_state_dict(init_state_dict)
        if args.init_optimizer_state and isinstance(init_checkpoint, dict) and "optimizer" in init_checkpoint:
            optimizer.load_state_dict(init_checkpoint["optimizer"])
            _optimizer_to_device(optimizer, device)
            _set_optimizer_hparams(optimizer, lr=args.lr, weight_decay=args.weight_decay)
            print(f"loaded initial optimizer state: {args.init_checkpoint_path}")
        print(f"loaded initial model checkpoint: {args.init_checkpoint_path}")

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_steps: set[int] = set()
    if args.checkpoint_steps.strip():
        checkpoint_steps = {
            int(part.strip())
            for part in args.checkpoint_steps.split(",")
            if part.strip()
        }
    diagnostic_backend = "cuda" if device.type == "cuda" and sim_backend == "cuda" else "torch"

    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        policy.load_state_dict(state_dict)
        policy.eval()
        eval_seeds = _parse_eval_seeds(args.eval_seeds, args.seed, args.eval_seed_count)
        eval_handle, eval_writer = open_eval_log(Path(args.log_path), args.settling_position_mm)
        trajectory_rows_all: list[dict[str, float | int]] = []
        eval_rows: list[dict[str, float | int]] = []
        start = perf_counter()
        print(
            f"device={device} sim_backend={sim_backend} eval_only=true "
            f"step_backend={diagnostic_backend} eval_seeds={','.join(str(seed) for seed in eval_seeds)}"
        )
        try:
            for seed_i, eval_seed in enumerate(eval_seeds):
                torch.manual_seed(eval_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(eval_seed)
                state = sim.reset(
                    args.batch_size,
                    device=device,
                    sample_dynamics=args.sample_dynamics,
                    sampled_dynamics_level=args.sampled_dynamics_level,
                    broad_sampler=args.broad_sampler,
                )
                metrics, trajectory_rows = rollout_diagnostics(
                    policy,
                    sim,
                    state,
                    args,
                    step_backend=diagnostic_backend,
                    eval_seed=eval_seed,
                    trajectory_count=args.trajectory_count if seed_i == 0 else 0,
                )
                elapsed = perf_counter() - start
                row = {
                    "eval_seed": eval_seed,
                    "eval_batch_size": args.batch_size,
                    "eval_horizon": args.horizon,
                    "seconds": elapsed,
                    **metrics,
                }
                eval_writer.writerow(row)
                eval_handle.flush()
                eval_rows.append(row)
                trajectory_rows_all.extend(trajectory_rows)
                print(
                    "eval_seed={seed} success={success:.6f} tail_success={tail_success:.6f} "
                    "tail_p={tail_p:.6f} tail_angle={tail_angle:.6f} tail_v={tail_v:.6f} tail_w={tail_w:.6f}".format(
                        seed=eval_seed,
                        success=row["success_rate"],
                        tail_success=row["tail_success_rate"],
                        tail_p=row["tail_position_mean"],
                        tail_angle=row["tail_angle_mean"],
                        tail_v=row["tail_linear_velocity_mean"],
                        tail_w=row["tail_angular_velocity_mean"],
                    ),
                    flush=True,
                )

            numeric_fields = _status_fieldnames() + _raptor_fieldnames(args.settling_position_mm)
            for aggregate_name, reducer in (
                ("aggregate_mean", lambda tensor: tensor.mean()),
                ("aggregate_std", lambda tensor: tensor.std(unbiased=False) if tensor.numel() > 1 else torch.zeros_like(tensor.mean())),
            ):
                aggregate_row: dict[str, float | int | str] = {
                    "eval_seed": aggregate_name,
                    "eval_batch_size": args.batch_size,
                    "eval_horizon": args.horizon,
                    "seconds": perf_counter() - start,
                }
                for field in numeric_fields:
                    values = torch.tensor([float(row[field]) for row in eval_rows], dtype=torch.float64)
                    aggregate_row[field] = float(reducer(values))
                eval_writer.writerow(aggregate_row)
                eval_handle.flush()
        finally:
            eval_handle.close()

        if args.trajectory_path:
            _write_trajectory_csv(Path(args.trajectory_path), trajectory_rows_all)
            print(f"saved trajectories: {args.trajectory_path}")
        print(f"saved eval log: {args.log_path}")
        return

    log_handle, log_writer = open_log(Path(args.log_path), args.settling_position_mm)
    sampler_audit_handle: object | None = None
    sampler_audit_writer: csv.DictWriter | None = None
    if args.sample_dynamics and args.sampler_audit_path:
        sampler_audit_handle, sampler_audit_writer = open_sampler_audit_log(Path(args.sampler_audit_path))
    numerics_audit_handle: object | None = None
    numerics_audit_writer: csv.DictWriter | None = None
    angular_audit_handle: object | None = None
    angular_audit_writer: csv.DictWriter | None = None
    if args.numerics_audit:
        numerics_audit_handle, numerics_audit_writer = open_numerics_audit_log(Path(args.numerics_audit_path))
        angular_audit_handle, angular_audit_writer = open_angular_audit_log(Path(args.angular_audit_path))
    start = perf_counter()
    print(f"device={device} sim_backend={sim_backend}")

    episode_id = torch.zeros(args.batch_size, device=device, dtype=torch.long)
    segment_id = torch.zeros(args.batch_size, device=device, dtype=torch.long)
    episode_steps = torch.zeros(args.batch_size, device=device, dtype=torch.long)
    invalid_mask = torch.zeros(args.batch_size, device=device, dtype=torch.bool)

    try:
        for step_idx in range(1, args.steps + 1):
            if args.persistent_episode_training:
                if step_idx == 1:
                    state = sim.reset(
                        args.batch_size,
                        device=device,
                        sample_dynamics=args.sample_dynamics,
                        sampled_dynamics_level=args.sampled_dynamics_level,
                        broad_sampler=args.broad_sampler,
                    )
                    episode_id = torch.ones_like(episode_id)
                    segment_id = torch.ones_like(segment_id)
                    episode_steps = torch.zeros_like(episode_steps)
                    reset_mask = torch.ones(args.batch_size, device=device, dtype=torch.bool)
                else:
                    reset_mask = invalid_mask | (episode_steps >= args.training_episode_steps)
                    if reset_mask.any():
                        replacement = sim.reset(
                            args.batch_size,
                            device=device,
                            sample_dynamics=args.sample_dynamics,
                            sampled_dynamics_level=args.sampled_dynamics_level,
                            broad_sampler=args.broad_sampler,
                        )
                        _copy_state_mask(state, replacement, reset_mask)
                        episode_id = torch.where(reset_mask, episode_id + 1, episode_id)
                        episode_steps = torch.where(reset_mask, torch.zeros_like(episode_steps), episode_steps)
                        segment_id = torch.where(reset_mask, torch.zeros_like(segment_id), segment_id)
                    segment_id = torch.where(reset_mask, torch.ones_like(segment_id), segment_id + 1)
                diagnostic_initial_state = _clone_state(state)
            else:
                reset_mask = torch.ones(args.batch_size, device=device, dtype=torch.bool)
                state = sim.reset(
                    args.batch_size,
                    device=device,
                    sample_dynamics=args.sample_dynamics,
                    sampled_dynamics_level=args.sampled_dynamics_level,
                    broad_sampler=args.broad_sampler,
                )
                segment_id = torch.ones_like(segment_id)
                episode_id = torch.zeros_like(episode_id)
                episode_steps = torch.zeros_like(episode_steps)
                diagnostic_initial_state = _clone_state(state)

            if sampler_audit_writer is not None:
                for batch_index in range(args.batch_size):
                    row = _sampler_audit_row(
                        step_idx=step_idx,
                        outer_step=step_idx,
                        batch_index=batch_index,
                        episode_id=int(episode_id[batch_index].item()),
                        segment_id=int(segment_id[batch_index].item()),
                        reset_mask=bool(reset_mask[batch_index].item()),
                        state=diagnostic_initial_state,
                    )
                    sampler_audit_writer.writerow(row)
                sampler_audit_handle.flush()

            f_ext_norm = torch.linalg.norm(diagnostic_initial_state.external_force, dim=-1).mean()
            current_finite_mask = torch.ones(args.batch_size, device=device, dtype=torch.bool)

            if sim_backend == "cuda-full":
                (
                    metrics_tensor,
                    final_position,
                    final_velocity,
                    final_rotation,
                    final_omega,
                    final_motor,
                    final_previous_action,
                    *cuda_debug_tensors,
                ) = full_cuda_rollout_metrics(
                    policy,
                    state,
                    sim.params,
                    loss_config,
                    horizon=args.horizon,
                    tail_steps=args.tail_steps,
                    state_step_decay=state_step_decay,
                    hidden_step_decay=hidden_step_decay,
                    clf_kappa=args.clf_kappa,
                    u_soft=args.u_soft,
                    lambda_clf=args.lambda_clf,
                    lambda_out=args.lambda_out,
                    lambda_tail=args.lambda_tail,
                    lambda_du=args.lambda_du,
                    lambda_ddu=args.lambda_ddu,
                    lambda_sat=args.lambda_sat,
                    noise_seed=args.seed * 1000003 + step_idx,
                    external_torque_max=args.external_torque_max,
                    action_noise_max=args.action_noise_max,
                    observation_noise_max=args.observation_noise_max,
                    collect_debug=args.numerics_audit,
                )
                loss = metrics_tensor[0]
                if numerics_audit_writer is not None:
                    _write_cuda_numerics_rows(
                        numerics_audit_writer,
                        angular_writer=angular_audit_writer,
                        train_step=step_idx,
                        horizon=args.horizon,
                        metrics_tensor=metrics_tensor,
                        debug_tensors=tuple(cuda_debug_tensors),
                        initial_state=diagnostic_initial_state,
                        params=sim.params,
                        clf_kappa=args.clf_kappa,
                    )
                    numerics_audit_handle.flush()
                    angular_audit_handle.flush()
                if args.persistent_episode_training:
                    state = L2FState(
                        position=final_position.detach(),
                        velocity=final_velocity.detach(),
                        rotation=final_rotation.detach(),
                        omega=final_omega.detach(),
                        motor=final_motor.detach(),
                        previous_action=final_previous_action.detach(),
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

                current_finite_mask = torch.isfinite(final_position).all(dim=-1)
                current_finite_mask &= torch.isfinite(final_velocity).all(dim=-1)
                current_finite_mask &= torch.isfinite(final_rotation).reshape(args.batch_size, 9).all(dim=-1)
                current_finite_mask &= torch.isfinite(final_omega).all(dim=-1)
                current_finite_mask &= torch.isfinite(final_motor).all(dim=-1)
                current_finite_mask &= torch.isfinite(final_previous_action).all(dim=-1)
                current_finite_mask &= torch.isfinite(loss)

                detached_metrics = metrics_tensor.detach()
                metrics = {
                    name: detached_metrics[idx].item()
                    for idx, name in enumerate(METRIC_NAMES)
                }
                # CUDA-FULL backend currently does not expose step-wise trajectory tensors for RAPTOR-style metrics.
                raptor_metrics = {
                    "position_mean": float("nan"),
                    "position_max_mean": float("nan"),
                    "position_max_std": float("nan"),
                    "angle_mean": float("nan"),
                    "angle_max_mean": float("nan"),
                    "angle_max_std": float("nan"),
                    "linear_velocity_mean": float("nan"),
                    "linear_velocity_max_mean": float("nan"),
                    "linear_velocity_max_std": float("nan"),
                    "angular_velocity_mean": float("nan"),
                    "angular_velocity_max_mean": float("nan"),
                    "angular_velocity_max_std": float("nan"),
                    "angular_acceleration_mean": float("nan"),
                    "angular_acceleration_max_mean": float("nan"),
                    "angular_acceleration_max_std": float("nan"),
                    "action_mean": float("nan"),
                    "action_max_mean": float("nan"),
                    "action_max_std": float("nan"),
                    "action_relative_mean": float("nan"),
                    "action_relative_max_mean": float("nan"),
                    "action_relative_max_std": float("nan"),
                    f"position_settling_fraction_{int(args.settling_position_mm)}mm": float("nan"),
                }
            else:
                hidden = None
                tracking_sum = torch.zeros((), device=device)
                clf_sum = torch.zeros((), device=device)
                outward_sum = torch.zeros((), device=device)
                du_sum = torch.zeros((), device=device)
                ddu_sum = torch.zeros((), device=device)
                sat_sum = torch.zeros((), device=device)
                previous_potential = sim.tracking_potential(state, loss_config)
                tail_potentials: list[torch.Tensor] = []
                previous_action_delta: torch.Tensor | None = None
                previous_omega: torch.Tensor | None = None
                metric_sums: dict[str, torch.Tensor] = {}

                position_errors: list[torch.Tensor] = []
                angle_errors: list[torch.Tensor] = []
                linear_velocity_errors: list[torch.Tensor] = []
                angular_velocity_errors: list[torch.Tensor] = []
                angular_acceleration_errors: list[torch.Tensor] = []
                action_errors: list[torch.Tensor] = []
                action_relative_errors: list[torch.Tensor] = []

                for _ in range(args.horizon):
                    state_features = sim.state_features(state)
                    error_features = sim.error_features(state)
                    previous_action = state.previous_action
                    action, hidden = policy(state_features, error_features, previous_action, hidden)
                    hidden = apply_gradient_decay(hidden, hidden_step_decay)
                    action_delta = action - previous_action

                    position_errors.append(state.position.norm(dim=-1))
                    angle_errors.append(_quat_angle_from_rotation_matrix(state.rotation))
                    linear_velocity_errors.append(state.velocity.norm(dim=-1))
                    angular_velocity_errors.append(state.omega.norm(dim=-1))
                    action_metric, action_relative_metric = _action_metrics(action)
                    action_errors.append(action_metric)
                    action_relative_errors.append(action_relative_metric)

                    if sim_backend == "cuda":
                        state = cuda_step(state, action, sim.params, grad_decay=state_step_decay)
                    else:
                        state = sim.step(state, action, grad_decay=state_step_decay)
                    # angular acceleration uses next-state omega
                    if previous_omega is None:
                        angular_acceleration_errors.append(torch.zeros_like(state.omega.norm(dim=-1)))
                    else:
                        angular_acceleration_errors.append((state.omega - previous_omega).norm(dim=-1) / args.dt)

                    tracking_components = sim.tracking_components(state, loss_config)
                    potential = sum(tracking_components.values())
                    tracking_sum = tracking_sum + potential.mean()
                    clf_target = (1.0 - args.clf_kappa * args.dt) * previous_potential.detach()
                    clf_sum = clf_sum + F.relu(potential - clf_target).square().mean()
                    outward_sum = outward_sum + sim.outward_velocity_loss(state, loss_config)
                    du_sum = du_sum + action_delta.square().mean()
                    sat_sum = sat_sum + F.relu(action.abs() - args.u_soft).square().mean()
                    if previous_action_delta is not None:
                        ddu_sum = ddu_sum + (action_delta - previous_action_delta).square().mean()

                    tail_potentials.append(potential)
                    previous_potential = potential
                    previous_action_delta = action_delta
                    previous_omega = state.omega.detach().clone()
                    for name, value in tracking_components.items():
                        value_mean = value.mean()
                        metric_sums[name] = metric_sums.get(name, torch.zeros_like(value_mean)) + value_mean.detach()

                horizon = float(args.horizon)
                tracking_loss = tracking_sum / horizon
                clf_loss = clf_sum / horizon
                outward_loss = outward_sum / horizon
                du_loss = du_sum / horizon
                ddu_count = max(args.horizon - 1, 1)
                ddu_loss = ddu_sum / ddu_count
                sat_loss = sat_sum / horizon
                tail_count = min(max(args.tail_steps, 1), len(tail_potentials))
                tail_loss = torch.stack(tail_potentials[-tail_count:]).mean()
                loss = (
                    tracking_loss
                    + args.lambda_clf * clf_loss
                    + args.lambda_out * outward_loss
                    + args.lambda_tail * tail_loss
                    + args.lambda_du * du_loss
                    + args.lambda_ddu * ddu_loss
                    + args.lambda_sat * sat_loss
                )

                position_mean, position_max_mean, position_max_std = _metric_summary(position_errors)
                angle_mean, angle_max_mean, angle_max_std = _metric_summary(angle_errors)
                linear_velocity_mean, linear_velocity_max_mean, linear_velocity_max_std = _metric_summary(linear_velocity_errors)
                angular_velocity_mean, angular_velocity_max_mean, angular_velocity_max_std = _metric_summary(angular_velocity_errors)
                angular_acceleration_mean, angular_acceleration_max_mean, angular_acceleration_max_std = _metric_summary(angular_acceleration_errors)
                action_mean, action_max_mean, action_max_std = _metric_summary(action_errors)
                action_relative_mean, action_relative_max_mean, action_relative_max_std = _metric_summary(action_relative_errors)

                settle_thr = args.settling_position_mm / 1000.0
                final_position = state.position.norm(dim=-1)
                settling_fraction = (final_position < settle_thr).float().mean().item()
                raptor_metrics = {
                    "position_mean": position_mean,
                    "position_max_mean": position_max_mean,
                    "position_max_std": position_max_std,
                    "angle_mean": angle_mean,
                    "angle_max_mean": angle_max_mean,
                    "angle_max_std": angle_max_std,
                    "linear_velocity_mean": linear_velocity_mean,
                    "linear_velocity_max_mean": linear_velocity_max_mean,
                    "linear_velocity_max_std": linear_velocity_max_std,
                    "angular_velocity_mean": angular_velocity_mean,
                    "angular_velocity_max_mean": angular_velocity_max_mean,
                    "angular_velocity_max_std": angular_velocity_max_std,
                    "angular_acceleration_mean": angular_acceleration_mean,
                    "angular_acceleration_max_mean": angular_acceleration_max_mean,
                    "angular_acceleration_max_std": angular_acceleration_max_std,
                    "action_mean": action_mean,
                    "action_max_mean": action_max_mean,
                    "action_max_std": action_max_std,
                    "action_relative_mean": action_relative_mean,
                    "action_relative_max_mean": action_relative_max_mean,
                    "action_relative_max_std": action_relative_max_std,
                    f"position_settling_fraction_{int(args.settling_position_mm)}mm": settling_fraction,
                }

                current_finite_mask = torch.ones(args.batch_size, device=device, dtype=torch.bool)
                current_finite_mask &= torch.isfinite(loss)
                metrics = {
                    name: (value / args.horizon).item()
                    for name, value in metric_sums.items()
                }
                metrics.update(
                    {
                        "loss": loss.item(),
                        "tracking": tracking_loss.item(),
                        "clf": clf_loss.item(),
                        "outward": outward_loss.item(),
                        "tail": tail_loss.item(),
                        "du": du_loss.item(),
                        "ddu": ddu_loss.item(),
                        "sat": sat_loss.item(),
                    }
                )
            loss_finite = bool(torch.isfinite(loss).item())
            state_finite = bool(current_finite_mask.all().item())
            do_backward = loss_finite and state_finite
            param_before = _param_stats(policy) if numerics_audit_writer is not None else (float("nan"), True, "")
            grad_stats: dict[str, float | bool | str] = {
                "max_abs_grad": float("nan"),
                "max_abs_grad_encoder": float("nan"),
                "max_abs_grad_gru": float("nan"),
                "max_abs_grad_head": float("nan"),
                "grad_finite": False,
                "first_nan_param_name": "",
            }
            grad_norm_fp64_before_clip = float("nan")
            grad_norm_fp64_after_clip = float("nan")
            max_abs_grad_before_clip = float("nan")
            grad_scale = float("nan")
            update_applied = False
            skip_reason = ""
            params_before_update = _param_delta_snapshot(policy)
            if do_backward:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_stats = _grad_stats(policy)
                if bool(grad_stats["grad_finite"]):
                    max_abs_grad_before_clip = float(grad_stats["max_abs_grad"])
                    (
                        grad_norm_fp64_before_clip,
                        grad_norm_fp64_after_clip,
                        grad_scale,
                    ) = _apply_fp64_global_grad_clip(policy, args.grad_clip)
                    grad_norm = grad_norm_fp64_before_clip
                    if not math.isfinite(grad_norm_fp64_before_clip):
                        skip_reason = "grad_norm_nonfinite"
                        grad_stats["grad_finite"] = False
                    elif args.grad_skip_threshold > 0.0 and grad_norm_fp64_before_clip > args.grad_skip_threshold:
                        skip_reason = "grad_skip_threshold"
                    else:
                        optimizer.step()
                        update_applied = True
                else:
                    grad_norm = float("nan")
                    skip_reason = str(grad_stats["first_nan_param_name"] or "grad_tensor_nonfinite")
            else:
                optimizer.zero_grad(set_to_none=True)
                grad_norm = float("nan")
                skip_reason = "loss_or_state_nonfinite"
            max_abs_param_delta = _max_abs_param_delta(policy, params_before_update)
            param_after = _param_stats(policy) if numerics_audit_writer is not None else (float("nan"), True, "")
            if numerics_audit_writer is not None:
                _write_optimizer_numerics_row(
                    numerics_audit_writer,
                    train_step=step_idx,
                    loss_finite=loss_finite,
                    state_finite=state_finite,
                    grad_norm=float(grad_norm),
                    grad_stats=grad_stats,
                    grad_norm_fp64_before_clip=grad_norm_fp64_before_clip,
                    grad_norm_fp64_after_clip=grad_norm_fp64_after_clip,
                    max_abs_grad_before_clip=max_abs_grad_before_clip,
                    grad_scale=grad_scale,
                    update_applied=update_applied,
                    skip_reason=skip_reason,
                    max_abs_param_delta=max_abs_param_delta,
                    param_before=param_before,
                    param_after=param_after,
                )
                numerics_audit_handle.flush()
            if args.persistent_episode_training and sim_backend == "torch":
                state = _clone_state(state)
            if args.numerics_audit and sim_backend == "cuda-full":
                raptor_metrics = _empty_raptor_metrics(args.settling_position_mm)
            else:
                policy.eval()
                raptor_metrics, _ = rollout_diagnostics(
                    policy,
                    sim,
                    diagnostic_initial_state,
                    args,
                    step_backend=diagnostic_backend,
                    eval_seed=args.seed,
                    trajectory_count=0,
                )
                policy.train()
            segment_mask_mean = reset_mask.float().mean().item()

            elapsed = perf_counter() - start
            row = {
                "step": step_idx,
                "grad_norm": float(grad_norm),
                "grad_norm_fp64_before_clip": float(grad_norm_fp64_before_clip),
                "grad_norm_fp64_after_clip": float(grad_norm_fp64_after_clip),
                "max_abs_grad_before_clip": float(max_abs_grad_before_clip),
                "grad_scale": float(grad_scale),
                "update_applied": int(update_applied),
                "skip_reason": skip_reason,
                "max_abs_param_delta": float(max_abs_param_delta),
                "seconds": elapsed,
                "episode_id": float(episode_id.float().mean().item()),
                "segment_id": float(segment_id.float().mean().item()),
                "reset_mask": float(segment_mask_mean),
                "mass": float(diagnostic_initial_state.mass.mean().item()),
                "cbrt_mass": float(diagnostic_initial_state.cbrt_mass.mean().item()),
                "thrust_to_weight": float(diagnostic_initial_state.thrust_to_weight.mean().item()),
                "torque_to_inertia": float(diagnostic_initial_state.torque_to_inertia.mean().item()),
                "alpha_roll_max": float(diagnostic_initial_state.alpha_roll_max.mean().item()),
                "alpha_pitch_max": float(diagnostic_initial_state.alpha_pitch_max.mean().item()),
                "alpha_yaw_max": float(diagnostic_initial_state.alpha_yaw_max.mean().item()),
                "eta_yaw": float(diagnostic_initial_state.eta_yaw.mean().item()),
                "jz_over_jxy": float(diagnostic_initial_state.jz_over_jxy.mean().item()),
                "dt_alpha_roll_max": float(diagnostic_initial_state.dt_alpha_roll_max.mean().item()),
                "dt_alpha_yaw_max": float(diagnostic_initial_state.dt_alpha_yaw_max.mean().item()),
                "rotor_distance_factor": float(diagnostic_initial_state.rotor_distance_factor.mean().item()),
                "inertia_factor": float(diagnostic_initial_state.inertia_factor.mean().item()),
                "tau_rise": float(diagnostic_initial_state.motor_time_rising.mean().item()),
                "tau_fall": float(diagnostic_initial_state.motor_time_falling.mean().item()),
                "rotor_torque_constant": float(diagnostic_initial_state.rotor_torque_constant.mean().item()),
                "force_std": float(diagnostic_initial_state.force_std.mean().item()),
                "f_ext_x": float(diagnostic_initial_state.external_force[:, 0].mean().item()),
                "f_ext_y": float(diagnostic_initial_state.external_force[:, 1].mean().item()),
                "f_ext_z": float(diagnostic_initial_state.external_force[:, 2].mean().item()),
                "f_ext_norm": float(f_ext_norm.item()),
                **metrics,
                **raptor_metrics,
            }
            log_writer.writerow(row)
            log_handle.flush()

            if step_idx % args.log_every == 0 or step_idx == 1:
                print(
                    "step={step} loss={loss:.6f} track={tracking:.6f} "
                    "clf={clf:.6f} tail={tail:.6f} grad={grad_norm:.3f} "
                    "update={update_applied} skip={skip_reason}".format(
                        step=step_idx,
                        loss=row["loss"],
                        tracking=row["tracking"],
                        clf=row["clf"],
                        tail=row["tail"],
                        grad_norm=row["grad_norm"],
                        update_applied=row["update_applied"],
                        skip_reason=row["skip_reason"],
                    ),
                    flush=True,
                )

            should_save_main = args.save_every > 0 and step_idx % args.save_every == 0
            should_save_history = step_idx in checkpoint_steps
            if should_save_main or should_save_history:
                checkpoint_payload = {
                    "step": step_idx,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
                if should_save_main:
                    torch.save(checkpoint_payload, checkpoint_path)
                if should_save_history:
                    step_checkpoint_path = checkpoint_path.with_name(
                        f"{checkpoint_path.stem}_step_{step_idx}{checkpoint_path.suffix}"
                    )
                    torch.save(checkpoint_payload, step_checkpoint_path)

            if args.persistent_episode_training:
                episode_steps = torch.where(reset_mask, episode_steps, episode_steps + args.horizon)
            finite_mask = current_finite_mask
            invalid_mask = ~finite_mask
        torch.save(
            {
                "step": args.steps,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            checkpoint_path,
        )
        if args.steps in checkpoint_steps:
            torch.save(
                {
                    "step": args.steps,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                checkpoint_path.with_name(f"{checkpoint_path.stem}_step_{args.steps}{checkpoint_path.suffix}"),
            )
    finally:
        log_handle.close()
        if sampler_audit_handle is not None:
            sampler_audit_handle.close()
        if numerics_audit_handle is not None:
            numerics_audit_handle.close()
        if angular_audit_handle is not None:
            angular_audit_handle.close()

    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved log: {args.log_path}")


if __name__ == "__main__":
    main()
