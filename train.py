from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shlex
import sys
from collections import deque
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
    normalized_capability_target,
)
from l2f_full_cuda_backend import METRIC_NAMES, full_cuda_rollout_metrics
from l2f_cuda_backend import cuda_backend_available, cuda_step, load_extension
from model import MotorGRUPolicy, compensate_integral_input_scale_
from policy_observation import (
    COMPACT_OBSERVATION_MODE,
    INTEGRAL_OBSERVATION_MODE,
    INTEGRAL_INPUT_FRAMES,
    LEGACY_OBSERVATION_MODE,
    OBSERVATION_MODES,
    PolicyObservationState,
    build_policy_observation,
    initial_observation_state,
    observation_dim,
    reset_observation_state,
    update_position_integral,
)
from retain_bank import (
    RetainBank,
    apply_retain_bank_samples,
    load_retain_bank,
    validate_retain_bank_for_sampler,
)
from training_objectives import (
    accumulated_episode_objective,
    independent_cvar_tail_loss,
    multistep_omega_decay_loss,
    retain_action_mse,
    time_normalized_segment_sum,
    threshold_cvar_tail_loss,
)
from training_schedule import (
    COMPRESSED_T2_10PCT,
    FIXED_H500,
    MIXED_HORIZON_CURRICULUM,
    select_episode_horizon,
)
from temporal_decay import (
    CURRENT_DECAY_MODE,
    GRADIENT_DECAY_MODES,
    resolve_step_gradient_decay,
)

RAPTOR_QUANTITIES = (
    "position",
    "linear_velocity",
    "angular_velocity",
    "angular_acceleration",
    "action",
    "action_relative",
)

POSITION_HOLD_CHECKPOINTS = (500, 2000, 5000, 10000)

RESET_AFTER_SKIP_REASONS = {
    "adaptive_gate_suspicious",
    "adaptive_gate_hard_grad",
    "grad_skip_threshold",
    "post_update_rejected",
    "loss_spike",
    "episode_invalid",
    "loss_or_state_nonfinite",
    "grad_norm_nonfinite",
    "grad_tensor_nonfinite",
}


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


def _masked_rate(value: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().item())
    if count == 0:
        return float("nan")
    return float(value[mask].float().mean().item())


def _ramped_auxiliary_weight(base_weight: float, accepted_updates: int, ramp_updates: int) -> float:
    if base_weight <= 0.0:
        return 0.0
    if ramp_updates <= 0:
        return float(base_weight)
    progress = min(1.0, float(accepted_updates + 1) / float(ramp_updates))
    return float(base_weight) * progress


def _ramped_auxiliary_weight_by_physical_steps(
    base_weight: float,
    physical_steps_after_segment: int,
    ramp_physical_steps: int,
) -> float:
    if base_weight <= 0.0:
        return 0.0
    if ramp_physical_steps <= 0:
        return float(base_weight)
    progress = min(1.0, float(physical_steps_after_segment) / float(ramp_physical_steps))
    return float(base_weight) * progress


def _masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_sample = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    eligible_weight = eligible.to(dtype=per_sample.dtype)
    return (per_sample * eligible_weight).sum(), eligible_weight.sum()


def _parse_csv_tuple(value: str, cast) -> tuple:
    parsed = tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("comma-separated option must contain at least one value")
    return parsed


class _ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, line: str) -> list[str]:
        return shlex.split(line, comments=True, posix=True)


def parse_args() -> argparse.Namespace:
    parser = _ArgumentParser(
        description="Train deployable observation/GRU motor policy.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--sim-backend", default="auto", choices=("auto", "torch", "cuda", "cuda-full"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--optimizer-updates",
        type=int,
        default=0,
        help="Stop after this many accepted optimizer updates; zero keeps legacy --steps behavior.",
    )
    parser.add_argument(
        "--max-outer-steps",
        type=int,
        default=0,
        help="Safety limit when --optimizer-updates is active; zero derives a conservative limit.",
    )
    parser.add_argument(
        "--physical-step-budget",
        type=int,
        default=0,
        help=(
            "Stop after this many batched simulator steps. A H500 episode at "
            "batch 256 consumes 128000 physical steps."
        ),
    )
    parser.add_argument(
        "--checkpoint-physical-steps",
        default="",
        help="Comma-separated accumulated physical-step checkpoints.",
    )
    parser.add_argument(
        "--episode-horizon-schedule",
        choices=(FIXED_H500, MIXED_HORIZON_CURRICULUM, COMPRESSED_T2_10PCT),
        default=FIXED_H500,
    )
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
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default=INTEGRAL_OBSERVATION_MODE)
    parser.add_argument("--integral-limit", type=float, default=0.5)
    parser.add_argument("--integral-leak", type=float, default=0.0)
    parser.add_argument("--integral-input-frame", choices=INTEGRAL_INPUT_FRAMES, default="world")
    parser.add_argument("--integral-input-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--compensate-integral-input-scale-on-load",
        action="store_true",
        help=(
            "After loading the initialization checkpoint, divide every first-layer "
            "weight that directly consumes the integral feature by its multiplier."
        ),
    )
    parser.add_argument("--enable-integral-residual", action="store_true")
    parser.add_argument("--integral-residual-hidden-dim", type=int, default=16)
    parser.add_argument("--integral-residual-scale", type=float, default=1.0)
    parser.add_argument("--enable-rate-damping-residual", action="store_true")
    parser.add_argument("--damping-residual-hidden-dim", type=int, default=32)
    parser.add_argument("--damping-residual-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--grad-skip-threshold", type=float, default=1.0e4)
    parser.add_argument("--adaptive-update-gate", action="store_true")
    parser.add_argument("--gate-warmup-updates", type=int, default=50)
    parser.add_argument("--gate-ema-beta", type=float, default=0.98)
    parser.add_argument("--gate-loss-factor", type=float, default=5.0)
    parser.add_argument("--gate-loss-add", type=float, default=0.05)
    parser.add_argument("--gate-grad-factor", type=float, default=30.0)
    parser.add_argument("--gate-grad-floor", type=float, default=10.0)
    parser.add_argument(
        "--post-update-check",
        default="off",
        choices=("off", "finite", "suspicious", "all"),
        help=(
            "Transactional rollout check after an optimizer step. 'finite' only "
            "rejects non-finite rollout/loss values and is compatible with CVaR/retain; "
            "'suspicious'/'all' also compare the replayed dense objective."
        ),
    )
    parser.add_argument("--post-update-next-check", action="store_true")
    parser.add_argument("--hard-reject-suspicious-grad", type=float, default=1000.0)
    parser.add_argument("--post-loss-factor", type=float, default=1.5)
    parser.add_argument("--post-loss-add", type=float, default=0.05)
    parser.add_argument("--post-ema-loss-factor", type=float, default=5.0)
    parser.add_argument("--post-ema-loss-add", type=float, default=0.05)
    parser.add_argument("--post-next-loss-factor", type=float, default=1.5)
    parser.add_argument("--post-next-loss-add", type=float, default=0.05)
    parser.add_argument("--post-next-ema-loss-factor", type=float, default=5.0)
    parser.add_argument("--post-next-ema-loss-add", type=float, default=0.05)
    parser.add_argument("--reset-after-skipped-update", action="store_true")
    parser.add_argument("--reset-loss-spike-factor", type=float, default=0.0)
    parser.add_argument("--reset-loss-spike-add", type=float, default=0.0)
    parser.add_argument("--state-grad-decay", type=float, default=0.5)
    parser.add_argument("--hidden-grad-decay", type=float, default=0.7)
    parser.add_argument(
        "--gradient-decay-mode",
        choices=GRADIENT_DECAY_MODES,
        default=CURRENT_DECAY_MODE,
        help=(
            "'current' preserves base**dt; 'nmi' uses exp(-alpha*dt) with "
            "the state/hidden alpha options."
        ),
    )
    parser.add_argument("--state-grad-alpha", type=float, default=None)
    parser.add_argument("--hidden-grad-alpha", type=float, default=None)
    parser.add_argument("--p-scale", type=float, default=2.0)
    parser.add_argument("--v-scale", type=float, default=3.0)
    parser.add_argument("--omega-scale", type=float, default=10.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--w-p", type=float, default=1.0)
    parser.add_argument("--w-v", type=float, default=0.3)
    parser.add_argument("--w-omega", type=float, default=0.05)
    parser.add_argument("--clf-kappa", type=float, default=1.0)
    parser.add_argument("--tail-steps", type=int, default=50)
    parser.add_argument("--u-soft", type=float, default=0.9)
    parser.add_argument("--lambda-clf", type=float, default=0.5)
    parser.add_argument("--lambda-out", type=float, default=0.1)
    parser.add_argument("--lambda-tail", type=float, default=1.0)
    parser.add_argument("--w-tail", type=float, default=0.0)
    parser.add_argument("--lambda-tail-omega", type=float, default=1.0)
    parser.add_argument("--cvar-fraction", type=float, default=0.20)
    parser.add_argument("--tail-window-steps", type=int, default=100)
    parser.add_argument("--tail-selection-mode", choices=("combined", "independent"), default="independent")
    parser.add_argument("--q-position", type=float, default=0.20)
    parser.add_argument("--q-omega", type=float, default=0.20)
    parser.add_argument("--w-position-cvar", type=float, default=0.0)
    parser.add_argument("--w-omega-cvar", type=float, default=0.0)
    parser.add_argument("--early-tail-weight", type=float, default=0.25)
    parser.add_argument("--final-tail-weight", type=float, default=1.0)
    parser.add_argument("--correct-episode-boundary-weighting", action="store_true")
    parser.add_argument("--w-omega-decay", type=float, default=0.0)
    parser.add_argument("--omega-decay-horizons", default="5,10,25")
    parser.add_argument("--omega-decay-beta", default="0.2,0.3,0.5")
    parser.add_argument("--omega-decay-rho", default="0.90,0.75,0.50")
    parser.add_argument("--omega-decay-eps", type=float, default=1.0e-6)
    parser.add_argument("--omega-decay-alpha-roll-max", type=float, default=154.0)
    parser.add_argument("--omega-decay-alpha-yaw-max", type=float, default=13.81)
    parser.add_argument("--lambda-du", type=float, default=3.0e-3)
    parser.add_argument("--lambda-ddu", type=float, default=3.0e-4)
    parser.add_argument("--lambda-sat", type=float, default=0.03)
    parser.add_argument("--lambda-motor-aux", type=float, default=0.0)
    parser.add_argument("--lambda-capability-aux", type=float, default=0.0)
    parser.add_argument("--lambda-response-aux", type=float, default=0.0)
    parser.add_argument("--motor-aux-burn-in", type=int, default=15)
    parser.add_argument("--capability-aux-burn-in", type=int, default=35)
    parser.add_argument("--response-aux-burn-in", type=int, default=10)
    parser.add_argument("--response-dv-scale", type=float, default=0.1)
    parser.add_argument("--response-domega-scale", type=float, default=1.0)
    parser.add_argument("--aux-weight-ramp-updates", type=int, default=300)
    parser.add_argument(
        "--aux-weight-ramp-physical-steps",
        type=int,
        default=0,
        help="When positive, ramp auxiliary weights by accumulated physical steps.",
    )
    parser.add_argument("--retain-bank-path", default="")
    parser.add_argument("--retain-fraction", type=float, default=0.25)
    parser.add_argument("--w-retain", type=float, default=0.0)
    parser.add_argument("--baseline-checkpoint-path", default="")
    parser.add_argument("--debug-terminal-loss-only", action="store_true")
    parser.add_argument("--external-force-max", type=float, default=0.0)
    parser.add_argument("--external-torque-max", type=float, default=0.0)
    parser.add_argument("--action-noise-max", type=float, default=0.0)
    parser.add_argument("--observation-noise-max", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--training-diagnostics-every-updates",
        type=int,
        default=1,
        help="Run the extra no-grad rollout and per-aux gradient attribution every N accepted updates.",
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-path", default="runs/direct_h500/train.csv")
    parser.add_argument("--checkpoint-path", default="checkpoints/direct_h500_motor_gru.pt")
    parser.add_argument("--checkpoint-steps", default="")
    parser.add_argument(
        "--checkpoint-updates",
        default="",
        help="Comma-separated accepted optimizer-update indices to save.",
    )
    parser.add_argument("--init-checkpoint-path", default="")
    parser.add_argument("--init-optimizer-state", action="store_true")
    parser.add_argument(
        "--resume-training-state",
        action="store_true",
        help=(
            "Resume optimizer/update counters and RNG/episode state from --init-checkpoint-path. "
            "Checkpoints without a training_state payload are rejected."
        ),
    )
    parser.add_argument("--settling-position-mm", type=float, default=10.0)
    parser.add_argument("--tail-start-step", type=int, default=450)
    parser.add_argument("--sample-dynamics", action="store_true")
    parser.add_argument("--fixed-dynamics", action="store_true")
    parser.add_argument("--sampled-dynamics-level", default="small", choices=("small", "medium", "broad"))
    parser.add_argument("--broad-sampler", default="legacy", choices=("legacy", "physical", "physical-fit"))
    parser.add_argument("--balanced-dynamics-sampling", action="store_true")
    parser.add_argument("--disable-balanced-dynamics-sampling", action="store_true")
    parser.add_argument("--disable-sampled-external-force", action="store_true")
    parser.add_argument("--correlated-size-mass-sampling", action="store_true")
    parser.add_argument("--disable-correlated-size-mass-sampling", action="store_true")
    parser.add_argument("--persistent-episode-training", action="store_true")
    parser.add_argument("--training-episode-steps", type=int, default=500)
    parser.add_argument("--update-timing", default="segment", choices=("segment", "episode-boundary"))
    parser.add_argument(
        "--optimization-block-horizon",
        type=int,
        default=0,
        help=(
            "Experimental, default-off optimizer/CVaR boundary in physical steps. "
            "Zero preserves legacy coupling to each reset episode; H500 permits "
            "updates inside longer reset episodes while carrying physical, hidden, "
            "and integral state."
        ),
    )
    parser.add_argument(
        "--tail-supervision-block-horizon",
        type=int,
        default=0,
        help=(
            "Experimental, default-off cadence for independent position/omega "
            "CVaR events. Zero preserves the historical coupling to the optimizer "
            "block. A shorter value permits multiple complete supervision blocks "
            "to accumulate before one optimizer commit."
        ),
    )
    parser.add_argument("--direct-h500-training", action="store_true")
    parser.add_argument("--gpu-rollout", action="store_true")
    parser.add_argument("--sampler-audit-path", default="")
    parser.add_argument("--sampler-audit-resets-only", action="store_true")
    parser.add_argument("--sampler-audit-hash-only", action="store_true")
    parser.add_argument("--numerics-audit", action="store_true")
    parser.add_argument("--numerics-audit-full-steps", action="store_true")
    parser.add_argument("--numerics-audit-path", default="")
    parser.add_argument("--angular-audit-path", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-seeds", default="")
    parser.add_argument("--eval-seed-count", type=int, default=1)
    parser.add_argument("--eval-samples-path", default="")
    parser.add_argument("--trajectory-path", default="")
    parser.add_argument("--trajectory-count", type=int, default=5)
    parser.add_argument("--success-position-m", type=float, default=0.05)
    parser.add_argument("--success-velocity", type=float, default=0.10)
    parser.add_argument("--success-omega", type=float, default=0.20)
    parser.add_argument("--steady-window-steps", type=int, default=100)
    parser.add_argument("--steady-required-fraction", type=float, default=0.95)
    parser.add_argument("--subgroup-tail-fraction", type=float, default=0.20)
    expanded_args = parser._read_args_from_files(sys.argv[1:])  # argparse has no public expansion helper.
    args = parser.parse_args(expanded_args)
    args.omega_decay_horizons = _parse_csv_tuple(args.omega_decay_horizons, int)
    args.omega_decay_beta = _parse_csv_tuple(args.omega_decay_beta, float)
    args.omega_decay_rho = _parse_csv_tuple(args.omega_decay_rho, float)

    raw_args = set(expanded_args)
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
    if args.correlated_size_mass_sampling_overridden:
        raise ValueError(
            "--correlated-size-mass-sampling/--disable-correlated-size-mass-sampling "
            "were historical no-op flags; select an explicit --broad-sampler instead"
        )
    if args.gpu_rollout:
        args.sim_backend = "cuda-full"
    if args.numerics_audit_path == "":
        args.numerics_audit_path = str(Path(args.log_path).with_name(Path(args.log_path).name.replace(".csv", "_numerics_audit.csv")))
    if args.angular_audit_path == "":
        args.angular_audit_path = str(Path(args.log_path).with_name(Path(args.log_path).name.replace(".csv", "_angular_audit.csv")))
    if args.training_episode_steps <= 0:
        raise ValueError("--training-episode-steps must be positive")
    if args.persistent_episode_training and args.training_episode_steps <= args.horizon:
        raise ValueError(
            "persistent episode training requires --training-episode-steps greater than --horizon"
        )
    if min(args.lambda_motor_aux, args.lambda_capability_aux, args.lambda_response_aux) < 0.0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if min(args.motor_aux_burn_in, args.capability_aux_burn_in, args.response_aux_burn_in) < 0:
        raise ValueError("auxiliary burn-in steps must be non-negative")
    if args.response_dv_scale <= 0.0 or args.response_domega_scale <= 0.0:
        raise ValueError("response target scales must be positive")
    if args.gradient_decay_mode == CURRENT_DECAY_MODE:
        if args.state_grad_alpha is not None or args.hidden_grad_alpha is not None:
            raise ValueError("gradient-decay alpha options require --gradient-decay-mode nmi")
    elif args.state_grad_alpha is None or args.hidden_grad_alpha is None:
        raise ValueError("nmi gradient decay requires both state and hidden alpha")
    if args.optimizer_updates < 0 or args.max_outer_steps < 0 or args.physical_step_budget < 0:
        raise ValueError("optimizer-update limits must be non-negative")
    if args.physical_step_budget > 0 and args.optimizer_updates > 0:
        raise ValueError("physical-step-budget and optimizer-updates are mutually exclusive")
    if args.physical_step_budget > 0:
        if args.update_timing != "episode-boundary" or not args.persistent_episode_training:
            raise ValueError("physical-step training requires persistent episode-boundary updates")
        if args.horizon != 250:
            raise ValueError("physical-step curriculum requires H250 truncated-BPTT segments")
        if args.batch_size <= 0 or args.physical_step_budget % (args.batch_size * 500) != 0:
            raise ValueError("physical-step budget must align to one batched H500 episode")
        if args.sim_backend == "cuda-full":
            raise ValueError("physical-step curriculum is not implemented by cuda-full")
    elif args.episode_horizon_schedule != FIXED_H500:
        raise ValueError("mixed episode horizons require --physical-step-budget")
    if args.resume_training_state and not args.init_checkpoint_path:
        raise ValueError("--resume-training-state requires --init-checkpoint-path")
    if args.w_tail < 0.0 or args.lambda_tail_omega < 0.0:
        raise ValueError("threshold tail weights must be non-negative")
    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("--cvar-fraction must be in (0, 1]")
    if args.tail_window_steps <= 0 or args.tail_window_steps > args.horizon:
        raise ValueError("--tail-window-steps must be in [1, horizon] for truncated BPTT")
    if not 0.0 < args.q_position <= 1.0 or not 0.0 < args.q_omega <= 1.0:
        raise ValueError("--q-position and --q-omega must be in (0, 1]")
    if min(args.w_position_cvar, args.w_omega_cvar, args.early_tail_weight, args.final_tail_weight) < 0.0:
        raise ValueError("independent CVaR and early/final weights must be non-negative")
    if args.integral_limit <= 0.0 or args.integral_leak < 0.0:
        raise ValueError("integral limit must be positive and leak must be non-negative")
    if args.integral_input_multiplier < 0.0:
        raise ValueError("--integral-input-multiplier must be non-negative")
    if args.compensate_integral_input_scale_on_load:
        if args.integral_input_multiplier <= 0.0:
            raise ValueError("integral scale compensation requires a positive multiplier")
        if not args.init_checkpoint_path:
            raise ValueError("integral scale compensation requires --init-checkpoint-path")
        if args.init_optimizer_state or args.resume_training_state:
            raise ValueError("integral scale compensation cannot restore optimizer/training state")
    if args.integral_residual_scale < 0.0 or args.damping_residual_scale < 0.0:
        raise ValueError("residual scales must be non-negative")
    if args.integral_residual_hidden_dim <= 0 or args.damping_residual_hidden_dim <= 0:
        raise ValueError("residual hidden dimensions must be positive")
    if (args.enable_integral_residual or args.enable_rate_damping_residual) and (
        args.observation_mode != INTEGRAL_OBSERVATION_MODE
    ):
        raise ValueError("residual branches require --observation-mode integral25")
    if args.w_omega_decay < 0.0 or args.omega_decay_eps <= 0.0:
        raise ValueError("omega decay weight must be non-negative and eps positive")
    if not (
        len(args.omega_decay_horizons)
        == len(args.omega_decay_beta)
        == len(args.omega_decay_rho)
        == 3
    ):
        raise ValueError("omega decay horizons, beta and rho must each contain three values")
    if args.w_omega_decay > 0.0 and any(
        k <= 0 or k >= args.horizon for k in args.omega_decay_horizons
    ):
        raise ValueError("omega decay horizons must be in [1, horizon-1]")
    if any(value < 0.0 for value in (*args.omega_decay_beta, *args.omega_decay_rho)):
        raise ValueError("omega decay beta/rho values must be non-negative")
    if args.w_omega_decay > 0.0 and not args.retain_bank_path:
        raise ValueError("omega decay requires low-authority retain-bank sampling")
    if args.w_retain < 0.0 or not 0.0 <= args.retain_fraction <= 1.0:
        raise ValueError("retain weight/fraction must be non-negative and fraction at most one")
    if args.retain_fraction > 0.0 and not args.retain_bank_path:
        # Legacy configs remain usable without a bank; formal retain configs
        # explicitly provide both path and fraction.
        args.retain_fraction = 0.0
    if args.w_retain > 0.0 and (not args.retain_bank_path or not args.baseline_checkpoint_path):
        raise ValueError("baseline retain requires --retain-bank-path and --baseline-checkpoint-path")
    has_cvar = args.w_tail > 0.0 or args.w_position_cvar > 0.0 or args.w_omega_cvar > 0.0
    if (has_cvar or args.w_retain > 0.0) and args.post_update_check not in {"off", "finite"}:
        raise ValueError(
            "CVaR/retain objectives support only --post-update-check off or finite; "
            "dense-objective acceptance thresholds are not comparable"
        )
    if args.steady_window_steps <= 0:
        raise ValueError("--steady-window-steps must be positive")
    if not 0.0 < args.steady_required_fraction <= 1.0:
        raise ValueError("--steady-required-fraction must be in (0, 1]")
    if not 0.0 < args.subgroup_tail_fraction <= 0.5:
        raise ValueError("--subgroup-tail-fraction must be in (0, 0.5]")
    if args.aux_weight_ramp_updates < 0:
        raise ValueError("--aux-weight-ramp-updates must be non-negative")
    if args.aux_weight_ramp_physical_steps < 0:
        raise ValueError("--aux-weight-ramp-physical-steps must be non-negative")
    if args.aux_weight_ramp_physical_steps > 0 and args.physical_step_budget <= 0:
        raise ValueError("physical-step auxiliary ramp requires --physical-step-budget")
    if args.training_diagnostics_every_updates <= 0:
        raise ValueError("--training-diagnostics-every-updates must be positive")
    if args.update_timing == "episode-boundary" and not args.persistent_episode_training:
        raise ValueError("--update-timing episode-boundary requires --persistent-episode-training")
    if args.physical_step_budget <= 0:
        possible_reset_horizons = (args.training_episode_steps,)
    elif args.episode_horizon_schedule == FIXED_H500:
        possible_reset_horizons = (500,)
    elif args.episode_horizon_schedule == COMPRESSED_T2_10PCT:
        possible_reset_horizons = (500, 1000)
    else:
        possible_reset_horizons = (500, 1000, 2000)
    if args.optimization_block_horizon < 0:
        raise ValueError("--optimization-block-horizon must be non-negative")
    if args.optimization_block_horizon > 0:
        if args.physical_step_budget <= 0:
            raise ValueError("--optimization-block-horizon requires --physical-step-budget")
        if not args.persistent_episode_training or args.update_timing != "episode-boundary":
            raise ValueError(
                "--optimization-block-horizon requires persistent episode-boundary training"
            )
        if args.optimization_block_horizon % args.horizon != 0:
            raise ValueError("--optimization-block-horizon must be divisible by --horizon")
        if any(
            reset_horizon % args.optimization_block_horizon != 0
            for reset_horizon in possible_reset_horizons
        ):
            raise ValueError(
                "--optimization-block-horizon must divide every scheduled reset horizon"
            )
    if args.tail_supervision_block_horizon < 0:
        raise ValueError("--tail-supervision-block-horizon must be non-negative")
    if args.tail_supervision_block_horizon > 0:
        if not args.persistent_episode_training or args.update_timing != "episode-boundary":
            raise ValueError(
                "--tail-supervision-block-horizon requires persistent "
                "episode-boundary training"
            )
        if args.tail_selection_mode != "independent" or args.w_tail > 0.0:
            raise ValueError(
                "--tail-supervision-block-horizon supports only independent CVaR events"
            )
        if args.tail_supervision_block_horizon % args.horizon != 0:
            raise ValueError("--tail-supervision-block-horizon must be divisible by --horizon")
        if args.tail_supervision_block_horizon < 2 * args.horizon:
            raise ValueError(
                "--tail-supervision-block-horizon must contain at least two segments"
            )
        if any(
            reset_horizon % args.tail_supervision_block_horizon != 0
            for reset_horizon in possible_reset_horizons
        ):
            raise ValueError(
                "--tail-supervision-block-horizon must divide every scheduled reset horizon"
            )
        for reset_horizon in possible_reset_horizons:
            optimizer_horizon = (
                reset_horizon
                if args.optimization_block_horizon == 0
                else args.optimization_block_horizon
            )
            if optimizer_horizon % args.tail_supervision_block_horizon != 0:
                raise ValueError(
                    "--tail-supervision-block-horizon must divide every optimizer block"
                )
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
        return "cuda"
    return "torch"


def _validate_cuda_full_observation_mode(sim_backend: str, observation_mode: str) -> None:
    if sim_backend == "cuda-full" and observation_mode != LEGACY_OBSERVATION_MODE:
        raise ValueError(
            "cuda-full is hard-coded for the legacy 40D observation and does not "
            "support 22D/25D; use --sim-backend cuda or torch"
        )


def _validate_cuda_full_residual_support(sim_backend: str, args: argparse.Namespace) -> None:
    if sim_backend == "cuda-full" and (
        args.enable_integral_residual
        or args.enable_rate_damping_residual
        or args.w_omega_decay > 0.0
    ):
        raise ValueError(
            "cuda-full does not implement integral/damping residual branches or "
            "multi-step omega decay; use --sim-backend cuda or torch"
        )


def _base_fieldnames() -> tuple[str, ...]:
    return (
        "step",
        "optimizer_update",
        "physical_steps",
        "episode_target_steps",
        "episode_horizon_phase",
        "optimization_block_horizon",
        "optimization_block_boundary",
        "reset_episode_boundary",
        "first_optimization_segment",
        "tail_supervision_block_horizon",
        "tail_supervision_block_boundary",
        "first_tail_supervision_segment",
        "hidden_state_norm_initial",
        "integral_state_norm_initial",
        "loss",
        "tracking",
        "position",
        "velocity",
        "omega",
        "clf",
        "outward",
        "tail",
        "threshold_tail",
        "position_tail",
        "omega_tail",
        "early_position_cvar",
        "early_omega_cvar",
        "final_position_cvar",
        "final_omega_cvar",
        "position_cvar_selected_fraction",
        "omega_cvar_selected_fraction",
        "cvar_selected_overlap_fraction",
        "position_cvar_selected_indices",
        "omega_cvar_selected_indices",
        "position_cvar_selected_alpha_roll_mean",
        "position_cvar_selected_alpha_yaw_mean",
        "position_cvar_selected_tau_fall_mean",
        "position_cvar_selected_thrust_to_weight_mean",
        "position_cvar_selected_force_std_mean",
        "omega_cvar_selected_alpha_roll_mean",
        "omega_cvar_selected_alpha_yaw_mean",
        "omega_cvar_selected_tau_fall_mean",
        "omega_cvar_selected_thrust_to_weight_mean",
        "omega_cvar_selected_force_std_mean",
        "position_cvar_encoder_grad_norm",
        "position_cvar_gru_grad_norm",
        "omega_cvar_encoder_grad_norm",
        "omega_cvar_gru_grad_norm",
        "segment_loss",
        "episode_only_loss",
        "episode_boundary_gradient_scale",
        "w_position_cvar_effective",
        "w_omega_cvar_effective",
        "early_tail_weight_effective",
        "final_tail_weight_effective",
        "cvar_selected_fraction",
        "cvar_selected_alpha_roll_mean",
        "cvar_selected_alpha_yaw_mean",
        "cvar_selected_tau_fall_mean",
        "cvar_selected_thrust_to_weight_mean",
        "cvar_selected_force_std_mean",
        "retain_action_mse",
        "w_tail_effective",
        "w_retain_effective",
        "retain_fraction_actual",
        "retain_bank_h500_success",
        "retain_bank_h10000_success",
        "du",
        "ddu",
        "sat",
        "motor_aux_loss",
        "capability_aux_loss",
        "response_aux_loss",
        "omega_decay_loss",
        "omega_decay_active_fraction",
        "omega_decay_horizon_0",
        "omega_decay_horizon_1",
        "omega_decay_horizon_2",
        "omega_decay_ratio_0",
        "omega_decay_ratio_1",
        "omega_decay_ratio_2",
        "omega_decay_component_loss_0",
        "omega_decay_component_loss_1",
        "omega_decay_component_loss_2",
        "omega_decay_encoder_grad_norm",
        "omega_decay_gru_grad_norm",
        "integral_world_norm_mean",
        "integral_body_x_mean",
        "integral_body_y_mean",
        "integral_body_z_mean",
        "integral_clamp_ratio",
        "integral_residual_action_rms",
        "damping_residual_action_rms",
        "steady_motor_bias_0",
        "steady_motor_bias_1",
        "steady_motor_bias_2",
        "steady_motor_bias_3",
        "high_force_position_tail_rms",
        "lambda_motor_aux_effective",
        "lambda_capability_aux_effective",
        "lambda_response_aux_effective",
        "motor_aux_encoder_grad_norm",
        "motor_aux_gru_grad_norm",
        "capability_aux_encoder_grad_norm",
        "capability_aux_gru_grad_norm",
        "response_aux_encoder_grad_norm",
        "response_aux_gru_grad_norm",
        "grad_norm",
        "grad_norm_encoder",
        "grad_norm_gru",
        "grad_norm_fp64_before_clip",
        "grad_norm_fp64_after_clip",
        "max_abs_grad_before_clip",
        "grad_scale",
        "update_applied",
        "skip_reason",
        "gate_loss_ema",
        "gate_grad_ema",
        "gate_ready",
        "gate_suspicious",
        "post_loss_after",
        "post_loss_limit",
        "post_next_loss_after",
        "post_next_loss_limit",
        "post_hard_reject",
        "post_update_accepted",
        "force_reset_next",
        "loss_spike_reset",
        "reset_loss_limit",
        "rollout_valid",
        "episode_valid",
        "max_abs_param_delta",
        "seconds",
        "update_timing",
        "episode_boundary",
        "grad_accum_segments",
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
    names = [
        "success_rate",
        "tail_success_rate",
        "invalid_fraction",
        "omega_failure_rate",
        "settling_time",
        "stay",
        "survival",
        "strict_bounded_angular_motion_rate",
        "loose_bounded_angular_motion_rate",
    ]
    for axis in ("x", "y", "z"):
        names.extend(
            (
                f"omega_{axis}_tail_rms_mean",
                f"omega_{axis}_tail_max_mean",
                f"omega_{axis}_tail_spectral_peak_hz_mean",
            )
        )
    for motor_index in range(4):
        names.extend(
            (
                f"action_{motor_index}_tail_rms_mean",
                f"action_{motor_index}_delta_tail_rms_mean",
            )
        )
    for checkpoint in POSITION_HOLD_CHECKPOINTS:
        suffix = f"H{checkpoint}"
        names.extend(
            (
                f"position_hold_snapshot_{suffix}",
                f"position_hold_steady_{suffix}",
                f"final_window_success_fraction_{suffix}",
                f"settling_time_{suffix}",
                f"stay_{suffix}",
                f"survival_{suffix}",
            )
        )
    names.extend(
        (
        # Legacy aggregate aliases.  h500_success_rate now means steady
        # position hold, and survival is deliberately independent of hold.
        "h500_success_rate",
        "h500_to_final_survival_rate",
        "h500_to_final_stay_success_rate",
        "low_alpha_roll_fraction",
        "low_alpha_roll_success_rate",
        "low_alpha_roll_omega_failure_rate",
        "low_alpha_yaw_fraction",
        "low_alpha_yaw_success_rate",
        "low_alpha_yaw_omega_failure_rate",
        "large_tau_fall_fraction",
        "large_tau_fall_success_rate",
        "large_tau_fall_omega_failure_rate",
        )
    )
    return tuple(names)


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


def open_log(
    path: Path,
    settling_position_mm: float,
    *,
    append: bool = False,
) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _base_fieldnames() + _status_fieldnames() + _raptor_fieldnames(
        settling_position_mm
    )
    write_header = not append or not path.exists() or path.stat().st_size == 0
    if append and not write_header:
        with path.open("r", newline="") as existing_handle:
            existing_header = next(csv.reader(existing_handle), [])
        if tuple(existing_header) != fieldnames:
            raise ValueError(
                "cannot append to a training log with a different schema; "
                "start a new log path for this code version"
            )
    handle = path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )
    if write_header:
        writer.writeheader()
    return handle, writer


def open_sampler_audit_log(path: Path, *, append: bool = False) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    handle = path.open("a" if append else "w", newline="")
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
            "position_x",
            "position_y",
            "position_z",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "rotation_00",
            "rotation_01",
            "rotation_02",
            "rotation_10",
            "rotation_11",
            "rotation_12",
            "rotation_20",
            "rotation_21",
            "rotation_22",
            "omega_x",
            "omega_y",
            "omega_z",
            "motor_0",
            "motor_1",
            "motor_2",
            "motor_3",
            "previous_action_0",
            "previous_action_1",
            "previous_action_2",
            "previous_action_3",
        ),
    )
    if write_header:
        writer.writeheader()
    return handle, writer


def open_sampler_audit_hash_log(path: Path, *, append: bool = False) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    handle = path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "step",
            "outer_step",
            "reset_count",
            "episode_id_min",
            "episode_id_max",
            "state_sha256",
        ),
    )
    if write_header:
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
        "max_hidden_adj_before",
        "max_hidden_adj_after",
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


def _validate_resume_has_no_pending_episode_gradients(
    *,
    args: argparse.Namespace,
    training_state: dict[str, object],
) -> None:
    """Reject checkpoints whose next segment needs unsaved accumulated grads."""

    if not args.persistent_episode_training or args.update_timing != "episode-boundary":
        return
    episode_steps = training_state["episode_steps"]
    invalid_mask = training_state["invalid_mask"]
    if not torch.is_tensor(episode_steps) or not torch.is_tensor(invalid_mask):
        raise TypeError("resume episode_steps and invalid_mask must be tensors")
    episode_target_steps = int(
        training_state.get("episode_target_steps", args.training_episode_steps)
    )
    optimization_block_horizon = _resolve_optimization_block_horizon(
        int(getattr(args, "optimization_block_horizon", 0)),
        episode_target_steps,
    )
    next_reset_mask = invalid_mask.to(dtype=torch.bool) | (
        episode_steps >= episode_target_steps
    )
    episode_valid = bool(training_state.get("episode_valid", True))
    has_completed_segment = bool(
        (episode_steps.remainder(optimization_block_horizon) > 0).any().item()
    )
    will_discard_episode = bool(next_reset_mask.any().item()) or not episode_valid
    if has_completed_segment and not will_discard_episode:
        raise RuntimeError(
            "cannot resume an unfinished persistent episode with episode-boundary "
            "updates: the checkpoint depends on accumulated parameter gradients "
            "that are not stored; resume from a complete episode-boundary checkpoint"
        )


def _resolve_optimization_block_horizon(
    configured_horizon: int,
    reset_horizon: int,
) -> int:
    """Resolve the optimizer boundary while preserving legacy reset coupling."""

    if configured_horizon < 0 or reset_horizon <= 0:
        raise ValueError("optimization/reset horizons must be non-negative/positive")
    block_horizon = reset_horizon if configured_horizon == 0 else configured_horizon
    if reset_horizon % block_horizon != 0:
        raise ValueError(
            f"optimization block H{block_horizon} does not divide reset horizon H{reset_horizon}"
        )
    return block_horizon


def _resolve_tail_supervision_block_horizon(
    configured_horizon: int,
    optimization_block_horizon: int,
    reset_horizon: int,
) -> int:
    """Resolve independent CVaR cadence while preserving legacy coupling."""

    if configured_horizon < 0 or optimization_block_horizon <= 0 or reset_horizon <= 0:
        raise ValueError(
            "tail/optimization/reset horizons must be non-negative/positive"
        )
    block_horizon = (
        optimization_block_horizon if configured_horizon == 0 else configured_horizon
    )
    if reset_horizon % block_horizon != 0:
        raise ValueError(
            f"tail supervision block H{block_horizon} does not divide reset horizon "
            f"H{reset_horizon}"
        )
    if optimization_block_horizon % block_horizon != 0:
        raise ValueError(
            f"tail supervision block H{block_horizon} must divide optimizer block "
            f"H{optimization_block_horizon}"
        )
    return block_horizon


def _active_steps_in_optimization_block(
    *,
    block_start_episode_step: int,
    block_horizon: int,
    burn_in: int,
) -> int:
    """Count supervised steps in one optimizer block after reset-relative burn-in."""

    if block_start_episode_step < 0 or block_horizon <= 0 or burn_in < 0:
        raise ValueError("block start/burn-in must be non-negative and horizon positive")
    block_end = block_start_episode_step + block_horizon
    return max(block_end - max(block_start_episode_step, burn_in), 0)


def _capture_training_state(
    *,
    state,
    persistent_hidden: torch.Tensor,
    persistent_observation_state: torch.Tensor,
    baseline_persistent_hidden: torch.Tensor | None,
    episode_id: torch.Tensor,
    segment_id: torch.Tensor,
    episode_steps: torch.Tensor,
    current_finite_mask: torch.Tensor,
    retain_mask: torch.Tensor,
    retain_bank_indices: torch.Tensor,
    force_reset_next: bool,
    must_reset_next: bool,
    episode_valid: bool,
    gate_loss_ema: float | None,
    gate_grad_ema: float | None,
    gate_accepted_updates: int,
    physical_steps_completed: int,
    episode_target_steps: int,
    curriculum_episode_index: int,
    step_idx: int,
    horizon: int,
    persistent_episode_training: bool,
) -> dict[str, object]:
    """Capture state at the start of the next outer segment.

    Formal checkpoints are written before the bookkeeping at the bottom of the
    loop.  Store the post-bookkeeping values here so a resumed run consumes the
    same next reset/RNG stream as an uninterrupted run.
    """

    next_episode_steps = episode_steps.detach().clone()
    if persistent_episode_training:
        next_episode_steps = next_episode_steps + int(horizon)
    next_invalid_mask = ~current_finite_mask.detach().clone()
    if persistent_episode_training and force_reset_next:
        next_invalid_mask = torch.ones_like(next_invalid_mask)
    payload: dict[str, object] = {
        "step": int(step_idx),
        "optimizer_update": int(gate_accepted_updates),
        "physical_steps_completed": int(physical_steps_completed),
        "episode_target_steps": int(episode_target_steps),
        "curriculum_episode_index": int(curriculum_episode_index),
        "state": {
            name: getattr(state, name).detach().clone()
            for name in state.__dataclass_fields__
        },
        "persistent_hidden": persistent_hidden.detach().clone(),
        "persistent_observation_state": {
            "integral_position": persistent_observation_state.integral_position.detach().clone(),
        },
        "baseline_persistent_hidden": (
            None
            if baseline_persistent_hidden is None
            else baseline_persistent_hidden.detach().clone()
        ),
        "episode_id": episode_id.detach().clone(),
        "segment_id": segment_id.detach().clone(),
        "episode_steps": next_episode_steps,
        "invalid_mask": next_invalid_mask,
        "retain_mask": retain_mask.detach().clone(),
        "retain_bank_indices": retain_bank_indices.detach().clone(),
        "must_reset_next": bool(must_reset_next),
        "episode_valid": bool(episode_valid),
        "gate_loss_ema": gate_loss_ema,
        "gate_grad_ema": gate_grad_ema,
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state_all": (
            [rng.cpu() for rng in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }
    return payload


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
    position = state.position[batch_index]
    velocity = state.velocity[batch_index]
    rotation = state.rotation[batch_index]
    omega = state.omega[batch_index]
    motor = state.motor[batch_index]
    previous_action = state.previous_action[batch_index]
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
        "position_x": float(position[0].item()),
        "position_y": float(position[1].item()),
        "position_z": float(position[2].item()),
        "velocity_x": float(velocity[0].item()),
        "velocity_y": float(velocity[1].item()),
        "velocity_z": float(velocity[2].item()),
        "rotation_00": float(rotation[0, 0].item()),
        "rotation_01": float(rotation[0, 1].item()),
        "rotation_02": float(rotation[0, 2].item()),
        "rotation_10": float(rotation[1, 0].item()),
        "rotation_11": float(rotation[1, 1].item()),
        "rotation_12": float(rotation[1, 2].item()),
        "rotation_20": float(rotation[2, 0].item()),
        "rotation_21": float(rotation[2, 1].item()),
        "rotation_22": float(rotation[2, 2].item()),
        "omega_x": float(omega[0].item()),
        "omega_y": float(omega[1].item()),
        "omega_z": float(omega[2].item()),
        "motor_0": float(motor[0].item()),
        "motor_1": float(motor[1].item()),
        "motor_2": float(motor[2].item()),
        "motor_3": float(motor[3].item()),
        "previous_action_0": float(previous_action[0].item()),
        "previous_action_1": float(previous_action[1].item()),
        "previous_action_2": float(previous_action[2].item()),
        "previous_action_3": float(previous_action[3].item()),
    }


def _sampler_audit_hash_row(
    step_idx: int,
    outer_step: int,
    episode_id: torch.Tensor,
    reset_mask: torch.Tensor,
    state: L2FState,
) -> dict[str, int | str]:
    digest = hashlib.sha256()
    reset_count = int(reset_mask.sum().item())
    for name in state.__dataclass_fields__:
        value = getattr(state, name)[reset_mask].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    reset_episode_id = episode_id[reset_mask]
    return {
        "step": step_idx,
        "outer_step": outer_step,
        "reset_count": reset_count,
        "episode_id_min": int(reset_episode_id.min().item()),
        "episode_id_max": int(reset_episode_id.max().item()),
        "state_sha256": digest.hexdigest(),
    }


def _numerics_log_steps(horizon: int, *, full_steps: bool = False) -> list[int]:
    if full_steps:
        return list(range(1, horizon + 1))
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


def _initialize_missing_auxiliary_parameters(
    policy: MotorGRUPolicy,
    missing: list[str],
    *,
    seed: int,
    device: torch.device,
) -> None:
    """Initialize new belief heads identically across observation widths."""
    missing_prefixes = {name.split(".", 1)[0] for name in missing}
    if not missing_prefixes:
        return
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed) + 104729)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed) + 104729)
        for name in ("motor_state_head", "capability_head", "response_head"):
            if name not in missing_prefixes:
                continue
            head = getattr(policy, name)
            torch.nn.init.orthogonal_(head.weight)
            torch.nn.init.zeros_(head.bias)
        for name in ("integral_residual_head", "damping_residual_head"):
            if name not in missing_prefixes:
                continue
            head = getattr(policy, name)
            if head is None:
                raise RuntimeError(f"checkpoint expects disabled optional module {name}")
            policy._reset_residual_head(head)


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


def _scale_policy_grads(policy: MotorGRUPolicy, scale: float) -> None:
    for param in policy.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)


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
        "grad_norm_encoder": 0.0,
        "grad_norm_gru": 0.0,
        "grad_finite": True,
        "first_nan_param_name": "",
    }
    encoder_sq = torch.zeros((), device=next(policy.parameters()).device, dtype=torch.float64)
    gru_sq = torch.zeros_like(encoder_sq)
    for name, param in policy.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        max_abs = _safe_max_abs(grad)
        stats["max_abs_grad"] = max(float(stats["max_abs_grad"]), max_abs)
        if name.startswith("encoder"):
            stats["max_abs_grad_encoder"] = max(float(stats["max_abs_grad_encoder"]), max_abs)
            encoder_sq = encoder_sq + grad.double().square().sum()
        elif name.startswith("gru"):
            stats["max_abs_grad_gru"] = max(float(stats["max_abs_grad_gru"]), max_abs)
            gru_sq = gru_sq + grad.double().square().sum()
        elif name.startswith("motor_head"):
            stats["max_abs_grad_head"] = max(float(stats["max_abs_grad_head"]), max_abs)
        if not torch.isfinite(grad).all().item():
            stats["grad_finite"] = False
            if stats["first_nan_param_name"] == "":
                stats["first_nan_param_name"] = name
    stats["grad_norm_encoder"] = float(torch.sqrt(encoder_sq).item())
    stats["grad_norm_gru"] = float(torch.sqrt(gru_sq).item())
    return stats


def _component_encoder_gru_grad_norms(
    loss: torch.Tensor,
    policy: MotorGRUPolicy,
) -> tuple[float, float]:
    encoder_parameters = [parameter for name, parameter in policy.named_parameters() if name.startswith("encoder")]
    gru_parameters = [parameter for name, parameter in policy.named_parameters() if name.startswith("gru")]
    parameters = encoder_parameters + gru_parameters
    if not loss.requires_grad or not parameters:
        return 0.0, 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    encoder_sq = torch.zeros((), device=loss.device, dtype=torch.float64)
    gru_sq = torch.zeros_like(encoder_sq)
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        contribution = gradient.detach().double().square().sum()
        if index < len(encoder_parameters):
            encoder_sq = encoder_sq + contribution
        else:
            gru_sq = gru_sq + contribution
    return float(torch.sqrt(encoder_sq).item()), float(torch.sqrt(gru_sq).item())


def _reset_detach_hidden(hidden: torch.Tensor, reset_mask: torch.Tensor) -> torch.Tensor:
    """Detach recurrent history and zero only environments that reset."""
    if hidden.ndim != 2 or reset_mask.shape != hidden.shape[:1] or reset_mask.dtype != torch.bool:
        raise ValueError("hidden must be [batch, hidden] and reset_mask a boolean [batch]")
    return torch.where(reset_mask[:, None], torch.zeros_like(hidden), hidden.detach())


def _update_ema(old: float | None, value: float, beta: float) -> float:
    if old is None or not math.isfinite(old):
        return float(value)
    return float(beta * old + (1.0 - beta) * value)


def _clone_state_dict_tensors(state_dict: dict) -> dict:
    cloned = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        elif isinstance(value, dict):
            cloned[key] = _clone_state_dict_tensors(value)
        elif isinstance(value, list):
            cloned[key] = [
                item.detach().clone() if torch.is_tensor(item) else item
                for item in value
            ]
        elif isinstance(value, tuple):
            cloned[key] = tuple(
                item.detach().clone() if torch.is_tensor(item) else item
                for item in value
            )
        else:
            cloned[key] = value
    return cloned


def _snapshot_module_state(policy: MotorGRUPolicy) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in policy.state_dict().items()
    }


def _snapshot_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    return _clone_state_dict_tensors(optimizer.state_dict())


@torch.no_grad()
def _post_update_loss_check(
    *,
    policy: MotorGRUPolicy,
    sim: L2FSimulator,
    initial_state: L2FState,
    loss_config: L2FLossConfig,
    args: argparse.Namespace,
    sim_backend: str,
    state_step_decay: float,
    hidden_step_decay: float,
    noise_seed: int,
    initial_hidden: torch.Tensor | None = None,
    initial_observation_state_value: PolicyObservationState | None = None,
    episode_step_offset: torch.Tensor | None = None,
    lambda_motor_aux_effective: float = 0.0,
    lambda_capability_aux_effective: float = 0.0,
    lambda_response_aux_effective: float = 0.0,
) -> tuple[float, bool]:
    policy.eval()
    last_completed_step = 0
    try:
        if sim_backend == "cuda-full":
            outputs = full_cuda_rollout_metrics(
                policy,
                initial_state,
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
                noise_seed=noise_seed,
                external_torque_max=args.external_torque_max,
                action_noise_max=args.action_noise_max,
                observation_noise_max=args.observation_noise_max,
                terminal_loss_only=args.debug_terminal_loss_only,
                collect_debug=False,
            )
            metrics_tensor = outputs[0]
            final_position = outputs[1]
            final_velocity = outputs[2]
            final_rotation = outputs[3]
            final_omega = outputs[4]
            final_motor = outputs[5]
            finite = bool(torch.isfinite(metrics_tensor).all().item())
            finite = finite and bool(torch.isfinite(final_position).all().item())
            finite = finite and bool(torch.isfinite(final_velocity).all().item())
            finite = finite and bool(torch.isfinite(final_rotation).all().item())
            finite = finite and bool(torch.isfinite(final_omega).all().item())
            finite = finite and bool(torch.isfinite(final_motor).all().item())
            return float(metrics_tensor[0].item()), finite

        state = _clone_state(initial_state)
        hidden = None if initial_hidden is None else initial_hidden.detach().clone()
        observation_state = (
            initial_observation_state(
                state.position.shape[0],
                device=state.position.device,
                dtype=state.position.dtype,
            )
            if initial_observation_state_value is None
            else initial_observation_state_value.detach().clone()
        )
        tracking_sum = torch.zeros((), device=state.position.device)
        clf_sum = torch.zeros((), device=state.position.device)
        outward_sum = torch.zeros((), device=state.position.device)
        du_sum = torch.zeros((), device=state.position.device)
        ddu_sum = torch.zeros((), device=state.position.device)
        sat_sum = torch.zeros((), device=state.position.device)
        motor_aux_sum = torch.zeros((), device=state.position.device)
        capability_aux_sum = torch.zeros((), device=state.position.device)
        response_aux_sum = torch.zeros((), device=state.position.device)
        motor_aux_count = torch.zeros((), device=state.position.device)
        capability_aux_count = torch.zeros((), device=state.position.device)
        response_aux_count = torch.zeros((), device=state.position.device)
        previous_potential = sim.tracking_potential(state, loss_config)
        previous_action_delta: torch.Tensor | None = None
        tail_potentials: list[torch.Tensor] = []

        use_auxiliary = any(
            weight > 0.0
            for weight in (
                lambda_motor_aux_effective,
                lambda_capability_aux_effective,
                lambda_response_aux_effective,
            )
        )
        for rollout_step in range(args.horizon):
            observation, observed_position = build_policy_observation(
                state,
                observation_state,
                mode=args.observation_mode,
                noise_max=args.observation_noise_max,
                integral_input_frame=args.integral_input_frame,
                integral_input_multiplier=args.integral_input_multiplier,
            )
            if use_auxiliary:
                motor_target = state.motor.detach()
                capability_target = normalized_capability_target(state).detach()
                action, hidden, auxiliary = policy.forward_with_aux(
                    observation,
                    hidden,
                )
                if episode_step_offset is None:
                    sample_step = torch.full(
                        (state.position.shape[0],),
                        rollout_step,
                        device=state.position.device,
                        dtype=torch.long,
                    )
                else:
                    sample_step = episode_step_offset + rollout_step
                if lambda_motor_aux_effective > 0.0:
                    value_sum, value_count = _masked_smooth_l1_loss(
                        auxiliary["motor_state"],
                        motor_target,
                        sample_step >= args.motor_aux_burn_in,
                    )
                    motor_aux_sum = motor_aux_sum + value_sum
                    motor_aux_count = motor_aux_count + value_count
                if lambda_capability_aux_effective > 0.0:
                    value_sum, value_count = _masked_smooth_l1_loss(
                        auxiliary["capability"],
                        capability_target,
                        sample_step >= args.capability_aux_burn_in,
                    )
                    capability_aux_sum = capability_aux_sum + value_sum
                    capability_aux_count = capability_aux_count + value_count
                if lambda_response_aux_effective > 0.0:
                    response_velocity = state.velocity.detach()
                    response_omega = state.omega.detach()
            else:
                action, hidden = policy(observation, hidden)
            hidden = apply_gradient_decay(hidden, hidden_step_decay)
            observation_state = update_position_integral(
                observation_state,
                observed_position,
                dt=args.dt,
                integral_limit=args.integral_limit,
                integral_leak=args.integral_leak,
            )
            action_delta = action - state.previous_action

            if sim_backend == "cuda":
                state = cuda_step(state, action, sim.params, grad_decay=state_step_decay)
            else:
                state = sim.step(state, action, grad_decay=state_step_decay)

            if use_auxiliary and lambda_response_aux_effective > 0.0:
                response_target = torch.cat(
                    (
                        (state.velocity.detach() - response_velocity) / args.response_dv_scale,
                        (state.omega.detach() - response_omega) / args.response_domega_scale,
                    ),
                    dim=-1,
                )
                value_sum, value_count = _masked_smooth_l1_loss(
                    auxiliary["response"],
                    response_target,
                    sample_step >= args.response_aux_burn_in,
                )
                response_aux_sum = response_aux_sum + value_sum
                response_aux_count = response_aux_count + value_count

            tracking_components = sim.tracking_components(state, loss_config)
            potential = sum(tracking_components.values())
            tracking_sum = tracking_sum + potential.mean()
            clf_target = (1.0 - args.clf_kappa * args.dt) * previous_potential
            clf_sum = clf_sum + F.relu(potential - clf_target).square().mean()
            outward_sum = outward_sum + sim.outward_velocity_loss(state, loss_config)
            du_sum = du_sum + action_delta.square().mean()
            sat_sum = sat_sum + F.relu(action.abs() - args.u_soft).square().mean()
            if previous_action_delta is not None:
                ddu_sum = ddu_sum + (action_delta - previous_action_delta).square().mean()

            tail_potentials.append(potential)
            previous_potential = potential
            previous_action_delta = action_delta

        horizon = float(args.horizon)
        tracking_loss = tracking_sum / horizon
        ddu_count = max(args.horizon - 1, 1)
        ddu_loss = ddu_sum / ddu_count
        tail_count = min(max(args.tail_steps, 1), len(tail_potentials))
        tail_loss = torch.stack(tail_potentials[-tail_count:]).mean()
        motor_aux_loss = motor_aux_sum / motor_aux_count.clamp_min(1.0)
        capability_aux_loss = capability_aux_sum / capability_aux_count.clamp_min(1.0)
        response_aux_loss = response_aux_sum / response_aux_count.clamp_min(1.0)
        loss = (
            tracking_loss
            + args.lambda_clf * clf_sum / horizon
            + args.lambda_out * outward_sum / horizon
            + args.lambda_tail * tail_loss
            + args.lambda_du * du_sum / horizon
            + args.lambda_ddu * ddu_loss
            + args.lambda_sat * sat_sum / horizon
            + lambda_motor_aux_effective * motor_aux_loss
            + lambda_capability_aux_effective * capability_aux_loss
            + lambda_response_aux_effective * response_aux_loss
        )

        finite = bool(torch.isfinite(loss).item())
        finite = finite and bool(torch.isfinite(state.position).all().item())
        finite = finite and bool(torch.isfinite(state.velocity).all().item())
        finite = finite and bool(torch.isfinite(state.rotation).all().item())
        finite = finite and bool(torch.isfinite(state.omega).all().item())
        finite = finite and bool(torch.isfinite(state.motor).all().item())
        finite = finite and bool(torch.isfinite(state.previous_action).all().item())
        return float(loss.item()), finite
    finally:
        policy.train()


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
        "max_hidden_adj_before": float("nan"),
        "max_hidden_adj_after": float("nan"),
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
    full_steps: bool = False,
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
        hidden_adj_before_mag,
        hidden_adj_after_mag,
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
            (hidden_adj_before_mag, 9, 0),
            (hidden_adj_after_mag, 9, 0),
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

    for rollout_step in _numerics_log_steps(horizon, full_steps=full_steps):
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
                "omega_loss": float(detached_metrics[4].item()),
                "clf": float(detached_metrics[5].item()),
                "outward": float(detached_metrics[6].item()),
                "tail": float(detached_metrics[7].item()),
                "du": float(detached_metrics[8].item()),
                "ddu": float(detached_metrics[9].item()),
                "sat": float(detached_metrics[10].item()),
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
                "max_hidden_adj_before": _safe_max_abs(hidden_adj_before_mag[action_step]),
                "max_hidden_adj_after": _safe_max_abs(hidden_adj_after_mag[action_step]),
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
    for name in L2FState.__dataclass_fields__:
        destination = getattr(target_state, name)
        if any(stride == 0 for stride in destination.stride()):
            destination = destination.clone()
            setattr(target_state, name, destination)
        destination[idx] = getattr(source_state, name)[idx]


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


def _position_hold_histories(
    hold_history: torch.Tensor,
    survival_history: torch.Tensor,
    *,
    window_steps: int,
    required_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trailing hold fractions and steady decisions for every completed step."""
    if hold_history.ndim != 2 or survival_history.shape != hold_history.shape:
        raise ValueError("position-hold histories must both have shape [time, batch]")
    horizon = hold_history.shape[0]
    if horizon <= 0:
        raise ValueError("position-hold history must not be empty")

    hold_float = hold_history.to(dtype=torch.float32)
    cumulative = torch.cat(
        (
            torch.zeros(
                1,
                hold_history.shape[1],
                device=hold_history.device,
                dtype=hold_float.dtype,
            ),
            hold_float.cumsum(dim=0),
        ),
        dim=0,
    )
    ends = torch.arange(1, horizon + 1, device=hold_history.device)
    starts = torch.clamp(ends - int(window_steps), min=0)
    counts = (ends - starts).to(dtype=hold_float.dtype)
    fractions = (cumulative[ends] - cumulative[starts]) / counts[:, None]

    eligible = ends >= int(window_steps)
    steady = (
        (fractions >= float(required_fraction))
        & survival_history
        & eligible[:, None]
    )
    return fractions, steady


def _position_hold_step_success(
    position: torch.Tensor,
    velocity: torch.Tensor,
    omega: torch.Tensor,
    *,
    success_position: float,
    success_velocity: float,
    success_omega: float,
    survival: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the yaw/attitude-independent success flag for one physical step."""
    success = (
        (position.norm(dim=-1) < float(success_position))
        & (velocity.norm(dim=-1) < float(success_velocity))
        & (omega.norm(dim=-1) < float(success_omega))
    )
    if survival is not None:
        success &= survival
    return success


def _position_hold_checkpoint(
    hold_history: torch.Tensor,
    survival_history: torch.Tensor,
    fraction_history: torch.Tensor,
    steady_history: torch.Tensor,
    *,
    completed_steps: int,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Return per-sample snapshot, steady, settling, stay, and survival values."""
    if not 1 <= completed_steps <= hold_history.shape[0]:
        raise ValueError("completed_steps is outside the recorded rollout")
    end_index = completed_steps - 1
    steady_prefix = steady_history[:completed_steps]
    has_settled = steady_prefix.any(dim=0)
    first_steady_index = steady_prefix.to(dtype=torch.int64).argmax(dim=0)
    settling_time = torch.where(
        has_settled,
        (first_steady_index + 1).to(dtype=fraction_history.dtype) * float(dt),
        torch.full_like(first_steady_index, float("nan"), dtype=fraction_history.dtype),
    )

    time_index = torch.arange(completed_steps, device=hold_history.device)[:, None]
    after_settling = time_index > first_steady_index[None, :]
    stay_numerator = (hold_history[:completed_steps] & after_settling).sum(dim=0)
    stay_denominator = after_settling.sum(dim=0)
    has_stay_interval = has_settled & (stay_denominator > 0)
    stay = torch.where(
        has_stay_interval,
        stay_numerator.to(dtype=fraction_history.dtype)
        / stay_denominator.clamp_min(1).to(dtype=fraction_history.dtype),
        torch.full_like(stay_numerator, float("nan"), dtype=fraction_history.dtype),
    )
    return {
        "snapshot": hold_history[end_index] & survival_history[end_index],
        "steady": steady_history[end_index],
        "final_window_fraction": fraction_history[end_index],
        "settling_time": settling_time,
        "stay": stay,
        "survival": survival_history[end_index],
    }


def _finite_mean_or_nan(values: torch.Tensor) -> float:
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        return float("nan")
    return float(values[finite].mean().item())


def _dominant_frequency(values: torch.Tensor, dt: float) -> torch.Tensor:
    """Return the non-DC dominant frequency for [time, batch, axis] signals."""
    if values.ndim != 3:
        raise ValueError("spectral input must have shape [time, batch, axis]")
    if values.shape[0] < 2:
        return torch.zeros(values.shape[1:], device=values.device, dtype=values.dtype)
    centered = values - values.mean(dim=0, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=0).abs().square()
    spectrum[0] = 0.0
    indices = spectrum.argmax(dim=0)
    frequencies = indices.to(values.dtype) / (float(values.shape[0]) * float(dt))
    has_energy = spectrum.amax(dim=0) > torch.finfo(values.dtype).eps
    return torch.where(has_energy, frequencies, torch.zeros_like(frequencies))


def _tail_axis_diagnostics(
    position_norm: torch.Tensor,
    velocity_norm: torch.Tensor,
    omega: torch.Tensor,
    action: torch.Tensor,
    *,
    dt: float,
) -> dict[str, torch.Tensor]:
    if omega.ndim != 3 or omega.shape[-1] != 3:
        raise ValueError("omega tail must have shape [time, batch, 3]")
    if action.ndim != 3 or action.shape[:2] != omega.shape[:2] or action.shape[-1] != 4:
        raise ValueError("action tail must have shape [time, batch, 4]")
    omega_rms_axis = omega.square().mean(dim=0).sqrt()
    omega_max_axis = omega.abs().amax(dim=0)
    action_rms_axis = action.square().mean(dim=0).sqrt()
    if action.shape[0] >= 2:
        action_delta_rms_axis = action.diff(dim=0).square().mean(dim=0).sqrt()
    else:
        action_delta_rms_axis = torch.zeros_like(action_rms_axis)
    position_rms = position_norm.square().mean(dim=0).sqrt()
    velocity_rms = velocity_norm.square().mean(dim=0).sqrt()
    omega_rms = omega.square().sum(dim=-1).mean(dim=0).sqrt()
    return {
        "omega_rms_axis": omega_rms_axis,
        "omega_max_axis": omega_max_axis,
        "omega_peak_hz_axis": _dominant_frequency(omega, dt),
        "action_rms_axis": action_rms_axis,
        "action_delta_rms_axis": action_delta_rms_axis,
        "position_rms": position_rms,
        "velocity_rms": velocity_rms,
        "omega_rms": omega_rms,
        "strict_bounded_angular_motion": (
            (position_rms < 0.05) & (velocity_rms < 0.10) & (omega_rms >= 0.20)
        ),
        "loose_bounded_angular_motion": (
            (position_rms < 0.10) & (velocity_rms < 0.20) & (omega_rms >= 0.20)
        ),
    }


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
        "omega_norm",
        "omega_x",
        "omega_y",
        "omega_z",
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


def _write_eval_samples_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
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
    sample_count: int = 0,
    initial_hidden: torch.Tensor | None = None,
) -> tuple[
    dict[str, float],
    list[dict[str, float | int]],
    list[dict[str, float | int]],
]:
    state = _clone_state(initial_state)
    hidden = None if initial_hidden is None else initial_hidden.detach().clone()
    observation_state = initial_observation_state(
        state.position.shape[0],
        device=state.position.device,
        dtype=state.position.dtype,
    )
    samples = {quantity: [] for quantity in RAPTOR_QUANTITIES}
    trajectory_rows: list[dict[str, float | int]] = []
    finite_mask = torch.ones(state.position.shape[0], device=state.position.device, dtype=torch.bool)
    position_hold_steps: list[torch.Tensor] = []
    survival_steps: list[torch.Tensor] = []
    diagnostic_position_norm: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
    diagnostic_velocity_norm: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
    diagnostic_omega: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
    diagnostic_action: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
    capture_count = min(max(trajectory_count, 0), state.position.shape[0])
    eval_sample_count = min(max(sample_count, 0), state.position.shape[0])

    for step_i in range(args.horizon):
        omega_before = state.omega
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode=args.observation_mode,
            noise_max=args.observation_noise_max,
            integral_input_frame=args.integral_input_frame,
            integral_input_multiplier=args.integral_input_multiplier,
        )
        action, hidden = policy(observation, hidden)
        observation_state = update_position_integral(
            observation_state,
            observed_position,
            dt=args.dt,
            integral_limit=args.integral_limit,
            integral_leak=args.integral_leak,
        )

        position_norm = state.position.norm(dim=-1)
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
        samples["linear_velocity"].append(velocity_norm.detach())
        samples["angular_velocity"].append(omega_norm.detach())
        samples["angular_acceleration"].append(angular_acceleration.detach())
        samples["action"].append(action_metric.detach())
        samples["action_relative"].append(action_relative_metric.detach())

        finite_mask &= torch.isfinite(position_norm)
        finite_mask &= torch.isfinite(velocity_norm)
        finite_mask &= torch.isfinite(omega_norm)
        finite_mask &= torch.isfinite(angular_acceleration)
        finite_mask &= torch.isfinite(action).all(dim=-1)
        finite_mask &= torch.isfinite(hidden).all(dim=-1)
        finite_mask &= torch.isfinite(next_state.position).all(dim=-1)
        finite_mask &= torch.isfinite(next_state.velocity).all(dim=-1)
        finite_mask &= torch.isfinite(next_state.rotation).reshape(state.position.shape[0], 9).all(dim=-1)
        finite_mask &= torch.isfinite(next_state.omega).all(dim=-1)
        finite_mask &= torch.isfinite(next_state.motor).all(dim=-1)

        position_hold_steps.append(
            _position_hold_step_success(
                next_state.position,
                next_state.velocity,
                next_state.omega,
                success_position=args.success_position_m,
                success_velocity=args.success_velocity,
                success_omega=args.success_omega,
                survival=finite_mask,
            ).clone()
        )
        survival_steps.append(finite_mask.clone())
        diagnostic_position_norm.append(next_state.position.norm(dim=-1).detach())
        diagnostic_velocity_norm.append(next_state.velocity.norm(dim=-1).detach())
        diagnostic_omega.append(next_state.omega.detach())
        diagnostic_action.append(action.detach())

        if capture_count > 0:
            p_cpu = position_norm[:capture_count].detach().cpu()
            v_cpu = velocity_norm[:capture_count].detach().cpu()
            omega_cpu = omega_norm[:capture_count].detach().cpu()
            omega_axis_cpu = state.omega[:capture_count].detach().cpu()
            action_cpu = action[:capture_count].detach().cpu()
            # Keep trajectory state fields aligned at the pre-action time t.
            motor_cpu = state.motor[:capture_count].detach().cpu()
            for sample_i in range(capture_count):
                trajectory_rows.append(
                    {
                        "eval_seed": -1 if eval_seed is None else eval_seed,
                        "sample": sample_i,
                        "step": step_i,
                        "time_s": step_i * args.dt,
                        "p_norm": float(p_cpu[sample_i]),
                        "v_norm": float(v_cpu[sample_i]),
                        "omega_norm": float(omega_cpu[sample_i]),
                        "omega_x": float(omega_axis_cpu[sample_i, 0]),
                        "omega_y": float(omega_axis_cpu[sample_i, 1]),
                        "omega_z": float(omega_axis_cpu[sample_i, 2]),
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

    metrics: dict[str, float] = {}
    metrics.update(_window_stats(samples, prefix="full", start_step=0, horizon=args.horizon))
    metrics.update(_window_stats(samples, prefix="tail", start_step=args.tail_start_step, horizon=args.horizon))

    settle_thr = args.settling_position_mm / 1000.0
    final_position = state.position.norm(dim=-1)
    final_velocity = state.velocity.norm(dim=-1)
    final_omega = state.omega.norm(dim=-1)
    position_hold_history = torch.stack(position_hold_steps, dim=0)
    survival_history = torch.stack(survival_steps, dim=0)
    final_window_fractions, steady_history = _position_hold_histories(
        position_hold_history,
        survival_history,
        window_steps=args.steady_window_steps,
        required_fraction=args.steady_required_fraction,
    )
    final_summary = _position_hold_checkpoint(
        position_hold_history,
        survival_history,
        final_window_fractions,
        steady_history,
        completed_steps=args.horizon,
        dt=args.dt,
    )
    final_success = final_summary["steady"]
    final_survival = final_summary["survival"]
    tail_start = min(max(args.tail_start_step, 0), max(args.horizon - 1, 0))
    tail_position_stack = torch.stack(samples["position"][tail_start:], dim=0)
    metrics[f"full_position_settling_fraction_{int(args.settling_position_mm)}mm"] = (
        (final_position < settle_thr) & final_survival
    ).float().mean().item()
    metrics[f"tail_position_settling_fraction_{int(args.settling_position_mm)}mm"] = (
        (tail_position_stack.max(dim=0).values < settle_thr) & final_survival
    ).float().mean().item()

    metrics["success_rate"] = final_success.float().mean().item()
    # Deprecated compatibility alias: success now always means the configured
    # final steady window, never a separately averaged legacy tail.
    metrics["tail_success_rate"] = metrics["success_rate"]
    metrics["invalid_fraction"] = (~final_survival).float().mean().item()
    metrics["settling_time"] = _finite_mean_or_nan(final_summary["settling_time"])
    metrics["stay"] = _finite_mean_or_nan(final_summary["stay"])
    metrics["survival"] = float(final_survival.float().mean().item())
    omega_failure = (final_omega >= args.success_omega) | (~final_survival)
    metrics["omega_failure_rate"] = omega_failure.float().mean().item()

    axis_diagnostics = _tail_axis_diagnostics(
        torch.stack(tuple(diagnostic_position_norm), dim=0),
        torch.stack(tuple(diagnostic_velocity_norm), dim=0),
        torch.stack(tuple(diagnostic_omega), dim=0),
        torch.stack(tuple(diagnostic_action), dim=0),
        dt=args.dt,
    )
    metrics["strict_bounded_angular_motion_rate"] = float(
        axis_diagnostics["strict_bounded_angular_motion"].float().mean().item()
    )
    metrics["loose_bounded_angular_motion_rate"] = float(
        axis_diagnostics["loose_bounded_angular_motion"].float().mean().item()
    )
    for axis_index, axis in enumerate(("x", "y", "z")):
        metrics[f"omega_{axis}_tail_rms_mean"] = float(
            axis_diagnostics["omega_rms_axis"][:, axis_index].mean().item()
        )
        metrics[f"omega_{axis}_tail_max_mean"] = float(
            axis_diagnostics["omega_max_axis"][:, axis_index].mean().item()
        )
        metrics[f"omega_{axis}_tail_spectral_peak_hz_mean"] = float(
            axis_diagnostics["omega_peak_hz_axis"][:, axis_index].mean().item()
        )
    for motor_index in range(4):
        metrics[f"action_{motor_index}_tail_rms_mean"] = float(
            axis_diagnostics["action_rms_axis"][:, motor_index].mean().item()
        )
        metrics[f"action_{motor_index}_delta_tail_rms_mean"] = float(
            axis_diagnostics["action_delta_rms_axis"][:, motor_index].mean().item()
        )

    checkpoint_summaries: dict[int, dict[str, torch.Tensor]] = {}
    for checkpoint in POSITION_HOLD_CHECKPOINTS:
        suffix = f"H{checkpoint}"
        for name in (
            "position_hold_snapshot",
            "position_hold_steady",
            "final_window_success_fraction",
            "settling_time",
            "stay",
            "survival",
        ):
            metrics[f"{name}_{suffix}"] = float("nan")
        if checkpoint > args.horizon:
            continue
        summary = _position_hold_checkpoint(
            position_hold_history,
            survival_history,
            final_window_fractions,
            steady_history,
            completed_steps=checkpoint,
            dt=args.dt,
        )
        checkpoint_summaries[checkpoint] = summary
        metrics[f"position_hold_snapshot_{suffix}"] = float(summary["snapshot"].float().mean().item())
        metrics[f"position_hold_steady_{suffix}"] = float(summary["steady"].float().mean().item())
        metrics[f"final_window_success_fraction_{suffix}"] = float(
            summary["final_window_fraction"].mean().item()
        )
        metrics[f"settling_time_{suffix}"] = _finite_mean_or_nan(summary["settling_time"])
        metrics[f"stay_{suffix}"] = _finite_mean_or_nan(summary["stay"])
        metrics[f"survival_{suffix}"] = float(summary["survival"].float().mean().item())

    h500_summary = checkpoint_summaries.get(500)
    if h500_summary is None:
        metrics["h500_success_rate"] = float("nan")
        metrics["h500_to_final_survival_rate"] = float("nan")
        metrics["h500_to_final_stay_success_rate"] = float("nan")
        h500_success = torch.zeros_like(final_success)
        h500_survived = torch.zeros_like(final_success)
        h500_stay_after = torch.full_like(final_window_fractions[-1], float("nan"))
    else:
        h500_success = h500_summary["steady"]
        h500_survival = h500_summary["survival"]
        h500_survived = h500_survival & final_survival
        survival_denominator = max(int(h500_survival.sum().item()), 1)
        if args.horizon > 500:
            h500_stay_after = position_hold_history[500:].float().mean(dim=0)
        else:
            h500_stay_after = torch.full_like(final_window_fractions[-1], float("nan"))
        metrics["h500_success_rate"] = float(h500_success.float().mean().item())
        metrics["h500_to_final_survival_rate"] = float(
            h500_survived.sum().item() / survival_denominator
        )
        metrics["h500_to_final_stay_success_rate"] = _finite_mean_or_nan(
            h500_stay_after[h500_success]
        )

    q = args.subgroup_tail_fraction
    alpha_roll_threshold = torch.quantile(initial_state.alpha_roll_max, q)
    alpha_yaw_threshold = torch.quantile(initial_state.alpha_yaw_max, q)
    tau_fall_threshold = torch.quantile(initial_state.motor_time_falling, 1.0 - q)
    subgroups = {
        "low_alpha_roll": initial_state.alpha_roll_max <= alpha_roll_threshold,
        "low_alpha_yaw": initial_state.alpha_yaw_max <= alpha_yaw_threshold,
        "large_tau_fall": initial_state.motor_time_falling >= tau_fall_threshold,
    }
    for name, mask in subgroups.items():
        metrics[f"{name}_fraction"] = float(mask.float().mean().item())
        metrics[f"{name}_success_rate"] = _masked_rate(final_success, mask)
        metrics[f"{name}_omega_failure_rate"] = _masked_rate(omega_failure, mask)

    sample_rows: list[dict[str, float | int]] = []
    if eval_sample_count > 0:
        for sample_i in range(eval_sample_count):
            external = initial_state.external_force[sample_i]
            initial_position = initial_state.position[sample_i]
            initial_velocity = initial_state.velocity[sample_i]
            initial_rotation = initial_state.rotation[sample_i]
            initial_omega = initial_state.omega[sample_i]
            sample_row: dict[str, float | int] = {
                    "eval_seed": -1 if eval_seed is None else eval_seed,
                    "sample_index": sample_i,
                    "horizon": args.horizon,
                    "success": int(final_success[sample_i].item()),
                    "settling_time": float(final_summary["settling_time"][sample_i].item()),
                    "stay": float(final_summary["stay"][sample_i].item()),
                    "survival": int(final_survival[sample_i].item()),
                    "position_m": float(final_position[sample_i].item()),
                    "velocity": float(final_velocity[sample_i].item()),
                    "omega": float(final_omega[sample_i].item()),
                    "omega_failure": int(omega_failure[sample_i].item()),
                    "position_tail_rms": float(axis_diagnostics["position_rms"][sample_i].item()),
                    "velocity_tail_rms": float(axis_diagnostics["velocity_rms"][sample_i].item()),
                    "omega_tail_rms": float(axis_diagnostics["omega_rms"][sample_i].item()),
                    "strict_bounded_angular_motion": int(
                        axis_diagnostics["strict_bounded_angular_motion"][sample_i].item()
                    ),
                    "loose_bounded_angular_motion": int(
                        axis_diagnostics["loose_bounded_angular_motion"][sample_i].item()
                    ),
                    "h500_success": int(h500_success[sample_i].item()),
                    "h500_to_final_survived": int(h500_survived[sample_i].item()),
                    "h500_to_final_stayed_success": float(h500_stay_after[sample_i].item()),
                    "low_alpha_roll": int(subgroups["low_alpha_roll"][sample_i].item()),
                    "low_alpha_yaw": int(subgroups["low_alpha_yaw"][sample_i].item()),
                    "large_tau_fall": int(subgroups["large_tau_fall"][sample_i].item()),
                    "mass": float(initial_state.mass[sample_i].item()),
                    "thrust_to_weight": float(initial_state.thrust_to_weight[sample_i].item()),
                    "alpha_roll_max": float(initial_state.alpha_roll_max[sample_i].item()),
                    "alpha_yaw_max": float(initial_state.alpha_yaw_max[sample_i].item()),
                    "tau_rise": float(initial_state.motor_time_rising[sample_i].item()),
                    "tau_fall": float(initial_state.motor_time_falling[sample_i].item()),
                    "f_ext_x": float(external[0].item()),
                    "f_ext_y": float(external[1].item()),
                    "f_ext_z": float(external[2].item()),
                    "initial_position_x": float(initial_position[0].item()),
                    "initial_position_y": float(initial_position[1].item()),
                    "initial_position_z": float(initial_position[2].item()),
                    "initial_velocity_x": float(initial_velocity[0].item()),
                    "initial_velocity_y": float(initial_velocity[1].item()),
                    "initial_velocity_z": float(initial_velocity[2].item()),
                    "initial_rotation_00": float(initial_rotation[0, 0].item()),
                    "initial_rotation_01": float(initial_rotation[0, 1].item()),
                    "initial_rotation_02": float(initial_rotation[0, 2].item()),
                    "initial_rotation_10": float(initial_rotation[1, 0].item()),
                    "initial_rotation_11": float(initial_rotation[1, 1].item()),
                    "initial_rotation_12": float(initial_rotation[1, 2].item()),
                    "initial_rotation_20": float(initial_rotation[2, 0].item()),
                    "initial_rotation_21": float(initial_rotation[2, 1].item()),
                    "initial_rotation_22": float(initial_rotation[2, 2].item()),
                    "initial_omega_x": float(initial_omega[0].item()),
                    "initial_omega_y": float(initial_omega[1].item()),
                    "initial_omega_z": float(initial_omega[2].item()),
            }
            for axis_index, axis in enumerate(("x", "y", "z")):
                sample_row[f"omega_{axis}_tail_rms"] = float(
                    axis_diagnostics["omega_rms_axis"][sample_i, axis_index].item()
                )
                sample_row[f"omega_{axis}_tail_max"] = float(
                    axis_diagnostics["omega_max_axis"][sample_i, axis_index].item()
                )
                sample_row[f"omega_{axis}_tail_spectral_peak_hz"] = float(
                    axis_diagnostics["omega_peak_hz_axis"][sample_i, axis_index].item()
                )
            for motor_index in range(4):
                sample_row[f"action_{motor_index}_tail_rms"] = float(
                    axis_diagnostics["action_rms_axis"][sample_i, motor_index].item()
                )
                sample_row[f"action_{motor_index}_delta_tail_rms"] = float(
                    axis_diagnostics["action_delta_rms_axis"][sample_i, motor_index].item()
                )
            for checkpoint, summary in checkpoint_summaries.items():
                suffix = f"H{checkpoint}"
                sample_row[f"position_hold_snapshot_{suffix}"] = int(summary["snapshot"][sample_i].item())
                sample_row[f"position_hold_steady_{suffix}"] = int(summary["steady"][sample_i].item())
                sample_row[f"final_window_success_fraction_{suffix}"] = float(
                    summary["final_window_fraction"][sample_i].item()
                )
                sample_row[f"settling_time_{suffix}"] = float(summary["settling_time"][sample_i].item())
                sample_row[f"stay_{suffix}"] = float(summary["stay"][sample_i].item())
                sample_row[f"survival_{suffix}"] = int(summary["survival"][sample_i].item())
            sample_rows.append(sample_row)
    return metrics, trajectory_rows, sample_rows


def main() -> None:
    args = parse_args()
    apply_direct_h500_training_defaults(args)
    if args.external_force_max != 0.0:
        raise ValueError("--external-force-max is deprecated; use --disturbance-force-max for episode-level hidden force")
    if args.external_torque_max != 0.0:
        raise ValueError("--external-torque-max is not part of the RAPTOR-style external-force path")
    device = resolve_device(args.device)
    sim_backend = resolve_sim_backend(args.sim_backend, device)
    _validate_cuda_full_observation_mode(sim_backend, args.observation_mode)
    _validate_cuda_full_residual_support(sim_backend, args)
    if sim_backend == "cuda-full" and (
        args.w_tail > 0.0
        or args.w_position_cvar > 0.0
        or args.w_omega_cvar > 0.0
        or args.w_retain > 0.0
    ):
        raise ValueError(
            "threshold CVaR and baseline retain require step-wise trajectories; "
            "use --sim-backend cuda or torch"
        )
    auxiliary_training = any(
        weight > 0.0
        for weight in (
            args.lambda_motor_aux,
            args.lambda_capability_aux,
            args.lambda_response_aux,
        )
    )
    if auxiliary_training and sim_backend == "cuda-full":
        raise ValueError(
            "privileged auxiliary losses are not implemented by cuda-full; "
            "use --sim-backend cuda or torch"
        )
    if args.persistent_episode_training and sim_backend == "cuda-full":
        raise ValueError(
            "cuda-full does not expose hidden0/final_hidden; use --sim-backend torch or cuda"
        )
    if args.persistent_episode_training and args.update_timing == "segment":
        print(
            "warning: accepted segment updates reset persistent hidden; "
            "use --update-timing episode-boundary for recurrent identification"
        )
    if args.lambda_capability_aux > 0.0 and not args.sample_dynamics:
        print("warning: capability target is constant because dynamics sampling is disabled")
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
        w_omega=args.w_omega,
    )
    state_step_decay = resolve_step_gradient_decay(
        mode=args.gradient_decay_mode,
        dt=args.dt,
        current_base=args.state_grad_decay,
        alpha=args.state_grad_alpha,
    )
    hidden_step_decay = resolve_step_gradient_decay(
        mode=args.gradient_decay_mode,
        dt=args.dt,
        current_base=args.hidden_grad_decay,
        alpha=args.hidden_grad_alpha,
    )
    policy = MotorGRUPolicy(
        observation_dim=observation_dim(args.observation_mode),
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth,
        enable_integral_residual=args.enable_integral_residual,
        enable_damping_residual=args.enable_rate_damping_residual,
        integral_residual_hidden_dim=args.integral_residual_hidden_dim,
        damping_residual_hidden_dim=args.damping_residual_hidden_dim,
        integral_residual_scale=args.integral_residual_scale,
        damping_residual_scale=args.damping_residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    resume_training_state: dict[str, object] | None = None
    if args.init_checkpoint_path:
        init_checkpoint = torch.load(args.init_checkpoint_path, map_location=device)
        init_state_dict = init_checkpoint.get("model", init_checkpoint)
        missing, _ = policy.load_compatible_state_dict(init_state_dict)
        _initialize_missing_auxiliary_parameters(
            policy,
            missing,
            seed=args.seed,
            device=device,
        )
        restore_optimizer = args.init_optimizer_state or args.resume_training_state
        if restore_optimizer and isinstance(init_checkpoint, dict) and "optimizer" in init_checkpoint:
            if missing:
                raise ValueError(
                    "cannot restore an action-only optimizer after adding auxiliary heads; "
                    "omit --init-optimizer-state"
                )
            optimizer.load_state_dict(init_checkpoint["optimizer"])
            _optimizer_to_device(optimizer, device)
            _set_optimizer_hparams(optimizer, lr=args.lr, weight_decay=args.weight_decay)
            print(f"loaded initial optimizer state: {args.init_checkpoint_path}")
        elif args.resume_training_state:
            raise ValueError("resume checkpoint has no optimizer state")
        if args.resume_training_state:
            if not isinstance(init_checkpoint, dict) or not isinstance(
                init_checkpoint.get("training_state"), dict
            ):
                raise ValueError(
                    "resume checkpoint has no training_state payload; restart this paired run "
                    "from its common initialization checkpoint"
                )
            resume_training_state = init_checkpoint["training_state"]
            _validate_resume_has_no_pending_episode_gradients(
                args=args,
                training_state=resume_training_state,
            )
        if missing:
            print(f"initialized new auxiliary parameters: {','.join(missing)}")
        if args.compensate_integral_input_scale_on_load:
            compensate_integral_input_scale_(policy, args.integral_input_multiplier)
            print(
                "compensated integral-consuming first-layer weights for "
                f"multiplier={args.integral_input_multiplier:g}"
            )
        print(f"loaded initial model checkpoint: {args.init_checkpoint_path}")

    retain_bank: RetainBank | None = None
    if args.retain_bank_path:
        retain_bank = load_retain_bank(args.retain_bank_path)
        if args.retain_fraction > 0.0:
            validate_retain_bank_for_sampler(retain_bank, args.broad_sampler)
        print(
            f"loaded retain bank: {args.retain_bank_path} count={len(retain_bank)} "
            f"H500={retain_bank.baseline_h500_rate:.6f} "
            f"H10000={retain_bank.baseline_h10000_rate:.6f}"
        )
    baseline_policy: MotorGRUPolicy | None = None
    if args.w_retain > 0.0:
        fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            baseline_policy = MotorGRUPolicy(
                observation_dim=observation_dim(args.observation_mode),
                encoder_dim=args.encoder_dim,
                hidden_dim=args.hidden_dim,
                encoder_depth=args.encoder_depth,
            ).to(device)
        baseline_checkpoint = torch.load(args.baseline_checkpoint_path, map_location=device)
        baseline_state_dict = baseline_checkpoint.get("model", baseline_checkpoint)
        baseline_policy.load_compatible_state_dict(baseline_state_dict)
        baseline_policy.eval()
        for parameter in baseline_policy.parameters():
            parameter.requires_grad_(False)
        print(f"loaded frozen retain baseline: {args.baseline_checkpoint_path}")

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_steps: set[int] = set()
    if args.checkpoint_steps.strip():
        checkpoint_steps = {
            int(part.strip())
            for part in args.checkpoint_steps.split(",")
            if part.strip()
        }
    checkpoint_updates: set[int] = set()
    if args.checkpoint_updates.strip():
        checkpoint_updates = {
            int(part.strip())
            for part in args.checkpoint_updates.split(",")
            if part.strip()
        }
    if any(value <= 0 for value in checkpoint_updates):
        raise ValueError("--checkpoint-updates values must be positive")
    checkpoint_physical_steps: set[int] = set()
    if args.checkpoint_physical_steps.strip():
        checkpoint_physical_steps = {
            int(part.strip())
            for part in args.checkpoint_physical_steps.split(",")
            if part.strip()
        }
    if any(value <= 0 for value in checkpoint_physical_steps):
        raise ValueError("--checkpoint-physical-steps values must be positive")
    if checkpoint_physical_steps and args.physical_step_budget <= 0:
        raise ValueError("physical-step checkpoints require --physical-step-budget")
    if any(value > args.physical_step_budget for value in checkpoint_physical_steps):
        raise ValueError("physical-step checkpoints cannot exceed the physical-step budget")
    if any(value % (args.batch_size * 500) != 0 for value in checkpoint_physical_steps):
        raise ValueError("physical-step checkpoints must align to batched H500 work")
    diagnostic_backend = "cuda" if device.type == "cuda" and sim_backend == "cuda" else "torch"
    gate_loss_ema: float | None = None
    gate_grad_ema: float | None = None
    gate_accepted_updates = 0

    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        policy.load_compatible_state_dict(state_dict)
        policy.eval()
        eval_seeds = _parse_eval_seeds(args.eval_seeds, args.seed, args.eval_seed_count)
        eval_handle, eval_writer = open_eval_log(Path(args.log_path), args.settling_position_mm)
        trajectory_rows_all: list[dict[str, float | int]] = []
        sample_rows_all: list[dict[str, float | int]] = []
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
                    balanced_dynamics_sampling=args.balanced_dynamics_sampling,
                    sample_external_force=not args.disable_sampled_external_force,
                )
                metrics, trajectory_rows, sample_rows = rollout_diagnostics(
                    policy,
                    sim,
                    state,
                    args,
                    step_backend=diagnostic_backend,
                    eval_seed=eval_seed,
                    trajectory_count=args.trajectory_count if seed_i == 0 else 0,
                    sample_count=args.batch_size if args.eval_samples_path else 0,
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
                sample_rows_all.extend(sample_rows)
                print(
                    "eval_seed={seed} success={success:.6f} survival={survival:.6f} "
                    "stay={stay:.6f} tail_p={tail_p:.6f} tail_v={tail_v:.6f} tail_w={tail_w:.6f}".format(
                        seed=eval_seed,
                        success=row["success_rate"],
                        survival=row["survival"],
                        stay=row["stay"],
                        tail_p=row["tail_position_mean"],
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
        if args.eval_samples_path:
            _write_eval_samples_csv(Path(args.eval_samples_path), sample_rows_all)
            print(f"saved eval samples: {args.eval_samples_path}")
        print(f"saved eval log: {args.log_path}")
        return

    log_handle, log_writer = open_log(
        Path(args.log_path),
        args.settling_position_mm,
        append=args.resume_training_state,
    )
    sampler_audit_handle: object | None = None
    sampler_audit_writer: csv.DictWriter | None = None
    if args.sample_dynamics and args.sampler_audit_path:
        if args.sampler_audit_hash_only:
            sampler_audit_handle, sampler_audit_writer = open_sampler_audit_hash_log(
                Path(args.sampler_audit_path),
                append=args.resume_training_state,
            )
        else:
            sampler_audit_handle, sampler_audit_writer = open_sampler_audit_log(
                Path(args.sampler_audit_path),
                append=args.resume_training_state,
            )
    numerics_audit_handle: object | None = None
    numerics_audit_writer: csv.DictWriter | None = None
    angular_audit_handle: object | None = None
    angular_audit_writer: csv.DictWriter | None = None
    if args.numerics_audit:
        numerics_audit_handle, numerics_audit_writer = open_numerics_audit_log(Path(args.numerics_audit_path))
        angular_audit_handle, angular_audit_writer = open_angular_audit_log(Path(args.angular_audit_path))
    # Policy construction consumes a width-dependent number of random values.
    # Restart the training stream so paired 40D/22D/25D groups see identical
    # reset, dynamics, force and deployable-observation noise streams.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    start = perf_counter()
    print(f"device={device} sim_backend={sim_backend}")

    episode_id = torch.zeros(args.batch_size, device=device, dtype=torch.long)
    segment_id = torch.zeros(args.batch_size, device=device, dtype=torch.long)
    episode_steps = torch.zeros(args.batch_size, device=device, dtype=torch.long)
    invalid_mask = torch.zeros(args.batch_size, device=device, dtype=torch.bool)
    persistent_hidden = policy.initial_hidden(
        args.batch_size,
        device=device,
        dtype=next(policy.parameters()).dtype,
    )
    persistent_observation_state = initial_observation_state(
        args.batch_size,
        device=device,
        dtype=next(policy.parameters()).dtype,
    )
    baseline_persistent_hidden = (
        baseline_policy.initial_hidden(
            args.batch_size,
            device=device,
            dtype=next(baseline_policy.parameters()).dtype,
        )
        if baseline_policy is not None
        else None
    )
    retain_mask = torch.zeros(args.batch_size, device=device, dtype=torch.bool)
    retain_bank_indices = torch.full(
        (args.batch_size,),
        -1,
        device=device,
        dtype=torch.long,
    )
    grad_accum_segments = 0
    grad_accum_segment_loss_sum = 0.0
    grad_accum_episode_only_loss_sum = 0.0
    episode_valid = True
    must_reset_next = False
    first_outer_step = 1
    physical_steps_completed = 0
    curriculum_episode_index = 0
    episode_target_steps = int(args.training_episode_steps)
    episode_horizon_phase = "fixed_config"
    if args.physical_step_budget > 0:
        initial_decision = select_episode_horizon(
            completed_physical_steps=physical_steps_completed,
            physical_step_budget=args.physical_step_budget,
            checkpoint_physical_steps=checkpoint_physical_steps,
            batch_size=args.batch_size,
            segment_steps=args.horizon,
            episode_index=curriculum_episode_index,
            mode=args.episode_horizon_schedule,
        )
        episode_target_steps = initial_decision.horizon_steps
        episode_horizon_phase = initial_decision.phase
        curriculum_episode_index += 1

    if resume_training_state is not None:
        state = L2FState(**resume_training_state["state"])
        persistent_hidden = resume_training_state["persistent_hidden"]
        persistent_observation_state = PolicyObservationState(
            integral_position=resume_training_state["persistent_observation_state"][
                "integral_position"
            ]
        )
        baseline_persistent_hidden = resume_training_state["baseline_persistent_hidden"]
        episode_id = resume_training_state["episode_id"]
        segment_id = resume_training_state["segment_id"]
        episode_steps = resume_training_state["episode_steps"]
        invalid_mask = resume_training_state["invalid_mask"]
        retain_mask = resume_training_state["retain_mask"]
        retain_bank_indices = resume_training_state["retain_bank_indices"]
        must_reset_next = bool(resume_training_state["must_reset_next"])
        episode_valid = bool(resume_training_state["episode_valid"])
        gate_loss_ema = resume_training_state["gate_loss_ema"]
        gate_grad_ema = resume_training_state["gate_grad_ema"]
        gate_accepted_updates = int(resume_training_state["optimizer_update"])
        physical_steps_completed = int(resume_training_state.get("physical_steps_completed", 0))
        episode_target_steps = int(
            resume_training_state.get("episode_target_steps", args.training_episode_steps)
        )
        curriculum_episode_index = int(
            resume_training_state.get("curriculum_episode_index", 0)
        )
        first_outer_step = int(resume_training_state["step"]) + 1
        torch.set_rng_state(resume_training_state["torch_rng_state"].cpu())
        cuda_rng_state_all = resume_training_state.get("cuda_rng_state_all", [])
        if device.type == "cuda":
            if not cuda_rng_state_all:
                raise ValueError("CUDA resume checkpoint has no CUDA RNG state")
            torch.cuda.set_rng_state_all([rng.cpu() for rng in cuda_rng_state_all])
        print(
            f"resuming training at step={first_outer_step} "
            f"optimizer_update={gate_accepted_updates}",
            flush=True,
        )

    if args.physical_step_budget > 0:
        remaining_physical_steps = max(args.physical_step_budget - physical_steps_completed, 0)
        remaining_segments = math.ceil(
            remaining_physical_steps / float(args.batch_size * args.horizon)
        )
        derived_outer_limit = first_outer_step - 1 + int(remaining_segments) * 3
        outer_step_limit = args.max_outer_steps or derived_outer_limit
    elif args.optimizer_updates > 0:
        segments_per_episode = max(1, math.ceil(args.training_episode_steps / args.horizon))
        remaining_updates = max(args.optimizer_updates - gate_accepted_updates, 0)
        derived_outer_limit = first_outer_step - 1 + remaining_updates * segments_per_episode * 3
        outer_step_limit = args.max_outer_steps or derived_outer_limit
    else:
        outer_step_limit = args.steps

    try:
        for step_idx in range(first_outer_step, outer_step_limit + 1):
            if args.physical_step_budget > 0 and physical_steps_completed >= args.physical_step_budget:
                break
            if args.optimizer_updates > 0 and gate_accepted_updates >= args.optimizer_updates:
                break
            if args.persistent_episode_training:
                if step_idx == 1:
                    state = sim.reset(
                        args.batch_size,
                        device=device,
                        sample_dynamics=args.sample_dynamics,
                        sampled_dynamics_level=args.sampled_dynamics_level,
                        broad_sampler=args.broad_sampler,
                        balanced_dynamics_sampling=args.balanced_dynamics_sampling,
                        sample_external_force=not args.disable_sampled_external_force,
                    )
                    episode_id = torch.ones_like(episode_id)
                    segment_id = torch.ones_like(segment_id)
                    episode_steps = torch.zeros_like(episode_steps)
                    reset_mask = torch.ones(args.batch_size, device=device, dtype=torch.bool)
                    episode_valid = True
                    must_reset_next = False
                else:
                    reset_mask = invalid_mask | (episode_steps >= episode_target_steps)
                    if must_reset_next and not bool(torch.all(reset_mask).item()):
                        raise RuntimeError(
                            "parameter update was applied, but next persistent rollout did not reset all samples"
                        )
                    if args.update_timing == "episode-boundary" and reset_mask.any():
                        reset_mask = torch.ones_like(reset_mask)
                        optimizer.zero_grad(set_to_none=True)
                        grad_accum_segments = 0
                        grad_accum_segment_loss_sum = 0.0
                        grad_accum_episode_only_loss_sum = 0.0
                    if reset_mask.any():
                        replacement = sim.reset(
                            args.batch_size,
                            device=device,
                            sample_dynamics=args.sample_dynamics,
                            sampled_dynamics_level=args.sampled_dynamics_level,
                            broad_sampler=args.broad_sampler,
                            balanced_dynamics_sampling=args.balanced_dynamics_sampling,
                            sample_external_force=not args.disable_sampled_external_force,
                        )
                        _copy_state_mask(state, replacement, reset_mask)
                        episode_id = torch.where(reset_mask, episode_id + 1, episode_id)
                        episode_steps = torch.where(reset_mask, torch.zeros_like(episode_steps), episode_steps)
                        segment_id = torch.where(reset_mask, torch.zeros_like(segment_id), segment_id)
                        if bool(torch.all(reset_mask).item()):
                            episode_valid = True
                            must_reset_next = False
                            if args.physical_step_budget > 0:
                                decision = select_episode_horizon(
                                    completed_physical_steps=physical_steps_completed,
                                    physical_step_budget=args.physical_step_budget,
                                    checkpoint_physical_steps=checkpoint_physical_steps,
                                    batch_size=args.batch_size,
                                    segment_steps=args.horizon,
                                    episode_index=curriculum_episode_index,
                                    mode=args.episode_horizon_schedule,
                                )
                                episode_target_steps = decision.horizon_steps
                                episode_horizon_phase = decision.phase
                                curriculum_episode_index += 1
                    segment_id = torch.where(reset_mask, torch.ones_like(segment_id), segment_id + 1)
                if retain_bank is not None and reset_mask.any():
                    sampled_retain_mask, sampled_bank_indices = apply_retain_bank_samples(
                        state,
                        retain_bank,
                        reset_mask,
                        fraction=args.retain_fraction,
                    )
                    retain_mask = torch.where(reset_mask, sampled_retain_mask, retain_mask)
                    retain_bank_indices = torch.where(
                        reset_mask,
                        sampled_bank_indices,
                        retain_bank_indices,
                    )
                diagnostic_initial_state = _clone_state(state)
            else:
                reset_mask = torch.ones(args.batch_size, device=device, dtype=torch.bool)
                state = sim.reset(
                    args.batch_size,
                    device=device,
                    sample_dynamics=args.sample_dynamics,
                    sampled_dynamics_level=args.sampled_dynamics_level,
                    broad_sampler=args.broad_sampler,
                    balanced_dynamics_sampling=args.balanced_dynamics_sampling,
                    sample_external_force=not args.disable_sampled_external_force,
                )
                segment_id = torch.ones_like(segment_id)
                episode_id = torch.zeros_like(episode_id)
                episode_steps = torch.zeros_like(episode_steps)
                episode_valid = True
                must_reset_next = False
                if retain_bank is not None:
                    retain_mask, retain_bank_indices = apply_retain_bank_samples(
                        state,
                        retain_bank,
                        reset_mask,
                        fraction=args.retain_fraction,
                    )
                diagnostic_initial_state = _clone_state(state)

            if args.persistent_episode_training:
                persistent_hidden = _reset_detach_hidden(persistent_hidden, reset_mask)
                persistent_observation_state = persistent_observation_state.detach()
                reset_observation_state(persistent_observation_state, reset_mask)
                observation_state = persistent_observation_state
                diagnostic_initial_observation_state = observation_state.clone()
                diagnostic_initial_hidden = persistent_hidden.detach().clone()
                if baseline_persistent_hidden is not None:
                    baseline_persistent_hidden = _reset_detach_hidden(
                        baseline_persistent_hidden,
                        reset_mask,
                    )
            else:
                observation_state = initial_observation_state(
                    args.batch_size,
                    device=device,
                    dtype=next(policy.parameters()).dtype,
                )
                diagnostic_initial_observation_state = observation_state.clone()
                diagnostic_initial_hidden = None
                if baseline_persistent_hidden is not None:
                    baseline_persistent_hidden.zero_()
            diagnostic_episode_steps = episode_steps.detach().clone()
            optimization_block_horizon = _resolve_optimization_block_horizon(
                args.optimization_block_horizon,
                episode_target_steps,
            )
            tail_supervision_block_horizon = _resolve_tail_supervision_block_horizon(
                args.tail_supervision_block_horizon,
                optimization_block_horizon,
                episode_target_steps,
            )
            optimization_block_progress = episode_steps.remainder(
                optimization_block_horizon
            )
            if not bool(
                torch.all(optimization_block_progress == optimization_block_progress[0]).item()
            ):
                raise RuntimeError("optimizer block progress diverged across the paired batch")
            if bool(
                torch.any(optimization_block_progress + args.horizon > optimization_block_horizon).item()
            ):
                raise RuntimeError("H250 segment would cross an optimizer block boundary")
            first_optimization_segment = bool(
                torch.all(optimization_block_progress == 0).item()
            )
            optimization_block_boundary = bool(
                torch.all(
                    optimization_block_progress + args.horizon
                    >= optimization_block_horizon
                ).item()
            )
            tail_supervision_block_progress = episode_steps.remainder(
                tail_supervision_block_horizon
            )
            if not bool(
                torch.all(
                    tail_supervision_block_progress
                    == tail_supervision_block_progress[0]
                ).item()
            ):
                raise RuntimeError(
                    "tail supervision block progress diverged across the paired batch"
                )
            if bool(
                torch.any(
                    tail_supervision_block_progress + args.horizon
                    > tail_supervision_block_horizon
                ).item()
            ):
                raise RuntimeError(
                    "H250 segment would cross a tail supervision block boundary"
                )
            first_tail_supervision_segment = bool(
                torch.all(tail_supervision_block_progress == 0).item()
            )
            tail_supervision_block_boundary = bool(
                torch.all(
                    tail_supervision_block_progress + args.horizon
                    >= tail_supervision_block_horizon
                ).item()
            )
            reset_episode_boundary = bool(
                torch.all(episode_steps + args.horizon >= episode_target_steps).item()
            )
            # Preserve the historical log column. In the default coupled mode
            # this is also the optimizer/CVaR boundary.
            episode_boundary = reset_episode_boundary
            optimization_block_start_episode_step = int(
                (episode_steps[0] - optimization_block_progress[0]).item()
            )
            segments_per_optimization_block = max(
                1,
                optimization_block_horizon // args.horizon,
            )
            if (
                args.optimization_block_horizon > 0
                and first_optimization_segment
                and not bool(reset_mask.any().item())
            ):
                if grad_accum_segments != 0:
                    raise RuntimeError(
                        "new optimization block started with pending accumulated gradients"
                    )
                episode_valid = True
            segment_physical_steps = args.batch_size * args.horizon
            physical_steps_after_segment = physical_steps_completed + (
                segment_physical_steps if args.physical_step_budget > 0 else 0
            )
            if args.aux_weight_ramp_physical_steps > 0:
                lambda_motor_aux_effective = _ramped_auxiliary_weight_by_physical_steps(
                    args.lambda_motor_aux,
                    physical_steps_after_segment,
                    args.aux_weight_ramp_physical_steps,
                )
                lambda_capability_aux_effective = _ramped_auxiliary_weight_by_physical_steps(
                    args.lambda_capability_aux,
                    physical_steps_after_segment,
                    args.aux_weight_ramp_physical_steps,
                )
                lambda_response_aux_effective = _ramped_auxiliary_weight_by_physical_steps(
                    args.lambda_response_aux,
                    physical_steps_after_segment,
                    args.aux_weight_ramp_physical_steps,
                )
            else:
                lambda_motor_aux_effective = _ramped_auxiliary_weight(
                    args.lambda_motor_aux,
                    gate_accepted_updates,
                    args.aux_weight_ramp_updates,
                )
                lambda_capability_aux_effective = _ramped_auxiliary_weight(
                    args.lambda_capability_aux,
                    gate_accepted_updates,
                    args.aux_weight_ramp_updates,
                )
                lambda_response_aux_effective = _ramped_auxiliary_weight(
                    args.lambda_response_aux,
                    gate_accepted_updates,
                    args.aux_weight_ramp_updates,
                )

            if sampler_audit_writer is not None and (
                not args.sampler_audit_resets_only or bool(reset_mask.any().item())
            ):
                if args.sampler_audit_hash_only:
                    sampler_audit_writer.writerow(
                        _sampler_audit_hash_row(
                            step_idx=step_idx,
                            outer_step=step_idx,
                            episode_id=episode_id,
                            reset_mask=reset_mask,
                            state=diagnostic_initial_state,
                        )
                    )
                else:
                    for batch_index in range(args.batch_size):
                        if args.sampler_audit_resets_only and not bool(reset_mask[batch_index].item()):
                            continue
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
            auxiliary_gradient_metrics = {
                "motor_aux_encoder_grad_norm": float("nan"),
                "motor_aux_gru_grad_norm": float("nan"),
                "capability_aux_encoder_grad_norm": float("nan"),
                "capability_aux_gru_grad_norm": float("nan"),
                "response_aux_encoder_grad_norm": float("nan"),
                "response_aux_gru_grad_norm": float("nan"),
            }
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
                    terminal_loss_only=args.debug_terminal_loss_only,
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
                        full_steps=args.numerics_audit_full_steps,
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
                metrics.update(
                    {
                        "motor_aux_loss": 0.0,
                        "capability_aux_loss": 0.0,
                        "response_aux_loss": 0.0,
                        "threshold_tail": 0.0,
                        "position_tail": 0.0,
                        "omega_tail": 0.0,
                        "cvar_selected_fraction": 0.0,
                        "cvar_selected_alpha_roll_mean": float("nan"),
                        "cvar_selected_alpha_yaw_mean": float("nan"),
                        "cvar_selected_tau_fall_mean": float("nan"),
                        "cvar_selected_thrust_to_weight_mean": float("nan"),
                        "cvar_selected_force_std_mean": float("nan"),
                        "retain_action_mse": 0.0,
                        "w_tail_effective": 0.0,
                        "w_retain_effective": 0.0,
                        "retain_fraction_actual": float(retain_mask.float().mean().item()),
                        "retain_bank_h500_success": float("nan"),
                        "retain_bank_h10000_success": float("nan"),
                        "lambda_motor_aux_effective": 0.0,
                        "lambda_capability_aux_effective": 0.0,
                        "lambda_response_aux_effective": 0.0,
                        **auxiliary_gradient_metrics,
                    }
                )
                # CUDA-FULL backend currently does not expose step-wise trajectory tensors for RAPTOR-style metrics.
                raptor_metrics = {
                    "position_mean": float("nan"),
                    "position_max_mean": float("nan"),
                    "position_max_std": float("nan"),
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
                hidden = persistent_hidden if args.persistent_episode_training else None
                baseline_hidden = baseline_persistent_hidden
                tracking_sum = torch.zeros((), device=device)
                clf_sum = torch.zeros((), device=device)
                outward_sum = torch.zeros((), device=device)
                du_sum = torch.zeros((), device=device)
                ddu_sum = torch.zeros((), device=device)
                sat_sum = torch.zeros((), device=device)
                motor_aux_sum = torch.zeros((), device=device)
                capability_aux_sum = torch.zeros((), device=device)
                response_aux_sum = torch.zeros((), device=device)
                motor_aux_count = torch.zeros((), device=device)
                capability_aux_count = torch.zeros((), device=device)
                response_aux_count = torch.zeros((), device=device)
                retain_sum = torch.zeros((), device=device)
                previous_potential = sim.tracking_potential(state, loss_config)
                tail_potentials: list[torch.Tensor] = []
                previous_action_delta: torch.Tensor | None = None
                previous_omega: torch.Tensor | None = None
                metric_sums: dict[str, torch.Tensor] = {}
                tail_position_history: list[torch.Tensor] = []
                tail_omega_history: list[torch.Tensor] = []

                position_errors: list[torch.Tensor] = []
                linear_velocity_errors: list[torch.Tensor] = []
                angular_velocity_errors: list[torch.Tensor] = []
                angular_acceleration_errors: list[torch.Tensor] = []
                action_errors: list[torch.Tensor] = []
                action_relative_errors: list[torch.Tensor] = []
                action_history: list[torch.Tensor] = []
                omega_decay_history: list[torch.Tensor] = [state.omega]
                integral_world_norm_sum = torch.zeros((), device=device)
                integral_body_sum = torch.zeros(3, device=device)
                integral_clamp_sum = torch.zeros((), device=device)
                integral_residual_square_sum = torch.zeros((), device=device)
                damping_residual_square_sum = torch.zeros((), device=device)
                decay_sample_mask = retain_mask & (
                    (state.alpha_roll_max <= args.omega_decay_alpha_roll_max)
                    | (state.alpha_yaw_max <= args.omega_decay_alpha_yaw_max)
                )

                for rollout_step in range(args.horizon):
                    previous_action = state.previous_action
                    observation, observed_position = build_policy_observation(
                        state,
                        observation_state,
                        mode=args.observation_mode,
                        noise_max=args.observation_noise_max,
                        integral_input_frame=args.integral_input_frame,
                        integral_input_multiplier=args.integral_input_multiplier,
                    )
                    if args.observation_mode == INTEGRAL_OBSERVATION_MODE:
                        integral_body = observation[:, 18:21]
                        integral_world_norm_sum = integral_world_norm_sum + (
                            observation_state.integral_position.detach().norm(dim=-1).mean()
                        )
                        integral_body_sum = integral_body_sum + integral_body.detach().mean(dim=0)
                    if auxiliary_training:
                        # Targets describe the pre-action state at t and must not
                        # send gradients into earlier simulator transitions.
                        motor_target = state.motor.detach()
                        capability_target = normalized_capability_target(state).detach()
                        action, hidden, auxiliary = policy.forward_with_aux(
                            observation,
                            hidden,
                        )
                        integral_residual_square_sum = integral_residual_square_sum + auxiliary[
                            "integral_action_contribution"
                        ].detach().square().mean()
                        damping_residual_square_sum = damping_residual_square_sum + auxiliary[
                            "damping_action_contribution"
                        ].detach().square().mean()
                        sample_step = diagnostic_episode_steps + rollout_step
                        if lambda_motor_aux_effective > 0.0:
                            value_sum, value_count = _masked_smooth_l1_loss(
                                auxiliary["motor_state"],
                                motor_target,
                                sample_step >= args.motor_aux_burn_in,
                            )
                            motor_aux_sum = motor_aux_sum + value_sum
                            motor_aux_count = motor_aux_count + value_count
                        if lambda_capability_aux_effective > 0.0:
                            value_sum, value_count = _masked_smooth_l1_loss(
                                auxiliary["capability"],
                                capability_target,
                                sample_step >= args.capability_aux_burn_in,
                            )
                            capability_aux_sum = capability_aux_sum + value_sum
                            capability_aux_count = capability_aux_count + value_count
                        if lambda_response_aux_effective > 0.0:
                            response_velocity = state.velocity.detach()
                            response_omega = state.omega.detach()
                    else:
                        action, hidden = policy(observation, hidden)
                    hidden = apply_gradient_decay(hidden, hidden_step_decay)
                    observation_state = update_position_integral(
                        observation_state,
                        observed_position,
                        dt=args.dt,
                        integral_limit=args.integral_limit,
                        integral_leak=args.integral_leak,
                    )
                    integral_clamp_sum = integral_clamp_sum + (
                        observation_state.integral_position.detach().abs()
                        >= (float(args.integral_limit) - 1.0e-7)
                    ).float().mean()
                    if baseline_policy is not None:
                        with torch.no_grad():
                            baseline_action, baseline_hidden = baseline_policy(
                                observation.detach(),
                                baseline_hidden,
                            )
                        retain_sum = retain_sum + retain_action_mse(
                            action,
                            baseline_action,
                            retain_mask,
                        )
                    action_delta = action - previous_action

                    position_errors.append(state.position.norm(dim=-1))
                    linear_velocity_errors.append(state.velocity.norm(dim=-1))
                    angular_velocity_errors.append(state.omega.norm(dim=-1))
                    action_metric, action_relative_metric = _action_metrics(action)
                    action_errors.append(action_metric)
                    action_relative_errors.append(action_relative_metric)
                    action_history.append(action.detach())

                    if sim_backend == "cuda":
                        state = cuda_step(state, action, sim.params, grad_decay=state_step_decay)
                    else:
                        state = sim.step(state, action, grad_decay=state_step_decay)
                    tail_position_history.append(state.position)
                    tail_omega_history.append(state.omega)
                    omega_decay_history.append(state.omega)
                    if auxiliary_training and lambda_response_aux_effective > 0.0:
                        response_target = torch.cat(
                            (
                                (state.velocity.detach() - response_velocity) / args.response_dv_scale,
                                (state.omega.detach() - response_omega) / args.response_domega_scale,
                            ),
                            dim=-1,
                        )
                        value_sum, value_count = _masked_smooth_l1_loss(
                            auxiliary["response"],
                            response_target,
                            sample_step >= args.response_aux_burn_in,
                        )
                        response_aux_sum = response_aux_sum + value_sum
                        response_aux_count = response_aux_count + value_count
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
                motor_aux_loss = motor_aux_sum / motor_aux_count.clamp_min(1.0)
                capability_aux_loss = capability_aux_sum / capability_aux_count.clamp_min(1.0)
                response_aux_loss = response_aux_sum / response_aux_count.clamp_min(1.0)
                motor_aux_objective_loss = time_normalized_segment_sum(
                    motor_aux_sum,
                    total_episode_count=(
                        args.batch_size
                        * _active_steps_in_optimization_block(
                            block_start_episode_step=optimization_block_start_episode_step,
                            block_horizon=optimization_block_horizon,
                            burn_in=args.motor_aux_burn_in,
                        )
                    ),
                    segments_per_episode=segments_per_optimization_block,
                )
                capability_aux_objective_loss = time_normalized_segment_sum(
                    capability_aux_sum,
                    total_episode_count=(
                        args.batch_size
                        * _active_steps_in_optimization_block(
                            block_start_episode_step=optimization_block_start_episode_step,
                            block_horizon=optimization_block_horizon,
                            burn_in=args.capability_aux_burn_in,
                        )
                    ),
                    segments_per_episode=segments_per_optimization_block,
                )
                response_aux_objective_loss = time_normalized_segment_sum(
                    response_aux_sum,
                    total_episode_count=(
                        args.batch_size
                        * _active_steps_in_optimization_block(
                            block_start_episode_step=optimization_block_start_episode_step,
                            block_horizon=optimization_block_horizon,
                            burn_in=args.response_aux_burn_in,
                        )
                    ),
                    segments_per_episode=segments_per_optimization_block,
                )
                retain_loss = retain_sum / horizon
                zero = tracking_loss * 0.0
                threshold_tail_loss = zero
                position_tail_loss = zero
                omega_tail_loss = zero
                early_position_cvar = zero
                early_omega_cvar = zero
                final_position_cvar = zero
                final_omega_cvar = zero
                position_cvar_weighted = zero
                omega_cvar_weighted = zero
                position_cvar_selected_fraction = 0.0
                omega_cvar_selected_fraction = 0.0
                cvar_selected_overlap_fraction = 0.0
                position_cvar_selected_indices = ""
                omega_cvar_selected_indices = ""
                position_cvar_selected_alpha_roll_mean = float("nan")
                position_cvar_selected_alpha_yaw_mean = float("nan")
                position_cvar_selected_tau_fall_mean = float("nan")
                position_cvar_selected_thrust_to_weight_mean = float("nan")
                position_cvar_selected_force_std_mean = float("nan")
                omega_cvar_selected_alpha_roll_mean = float("nan")
                omega_cvar_selected_alpha_yaw_mean = float("nan")
                omega_cvar_selected_tau_fall_mean = float("nan")
                omega_cvar_selected_thrust_to_weight_mean = float("nan")
                omega_cvar_selected_force_std_mean = float("nan")
                cvar_selected_fraction = 0.0
                cvar_selected_alpha_roll_mean = float("nan")
                cvar_selected_alpha_yaw_mean = float("nan")
                cvar_selected_tau_fall_mean = float("nan")
                cvar_selected_thrust_to_weight_mean = float("nan")
                cvar_selected_force_std_mean = float("nan")
                episode_only_loss = zero
                omega_decay_loss = zero
                omega_decay_ratios = (zero.detach(), zero.detach(), zero.detach())
                omega_decay_component_losses = (zero, zero, zero)
                omega_decay_active_fraction = 0.0
                if args.w_omega_decay > 0.0:
                    omega_decay = multistep_omega_decay_loss(
                        torch.stack(omega_decay_history, dim=0),
                        decay_sample_mask,
                        horizons=args.omega_decay_horizons,
                        beta=args.omega_decay_beta,
                        rho=args.omega_decay_rho,
                        success_omega=args.success_omega,
                        eps=args.omega_decay_eps,
                    )
                    omega_decay_loss = omega_decay.loss
                    omega_decay_ratios = omega_decay.mean_ratios
                    omega_decay_component_losses = omega_decay.horizon_losses
                    omega_decay_active_fraction = sum(omega_decay.active_fractions) / len(
                        omega_decay.active_fractions
                    )
                position_history_tensor = torch.stack(tail_position_history, dim=0)
                omega_history_tensor = torch.stack(tail_omega_history, dim=0)
                if args.tail_selection_mode == "combined" and optimization_block_boundary:
                    threshold_tail = threshold_cvar_tail_loss(
                        position_history_tensor,
                        omega_history_tensor,
                        position_threshold=args.success_position_m,
                        omega_threshold=args.success_omega,
                        lambda_tail_omega=args.lambda_tail_omega,
                        cvar_fraction=args.cvar_fraction,
                        window_steps=args.tail_window_steps,
                    )
                    threshold_tail_loss = threshold_tail.loss
                    position_tail_loss = threshold_tail.position_tail
                    omega_tail_loss = threshold_tail.omega_tail
                    selected = threshold_tail.selected_mask
                    cvar_selected_fraction = threshold_tail.selected_fraction
                    cvar_selected_alpha_roll_mean = float(state.alpha_roll_max[selected].mean().detach().item())
                    cvar_selected_alpha_yaw_mean = float(state.alpha_yaw_max[selected].mean().detach().item())
                    cvar_selected_tau_fall_mean = float(state.motor_time_falling[selected].mean().detach().item())
                    cvar_selected_thrust_to_weight_mean = float(state.thrust_to_weight[selected].mean().detach().item())
                    cvar_selected_force_std_mean = float(state.force_std[selected].mean().detach().item())
                    episode_only_loss = episode_only_loss + args.final_tail_weight * args.w_tail * threshold_tail_loss
                elif args.tail_selection_mode == "independent":
                    independent_events: list[tuple[str, float]] = []
                    if first_tail_supervision_segment and args.early_tail_weight > 0.0:
                        independent_events.append(("early", args.early_tail_weight))
                    if tail_supervision_block_boundary and args.final_tail_weight > 0.0:
                        independent_events.append(("final", args.final_tail_weight))
                    for event_name, event_weight in independent_events:
                        independent_tail = independent_cvar_tail_loss(
                            position_history_tensor,
                            omega_history_tensor,
                            position_threshold=args.success_position_m,
                            omega_threshold=args.success_omega,
                            q_position=args.q_position,
                            q_omega=args.q_omega,
                            w_position_cvar=args.w_position_cvar,
                            w_omega_cvar=args.w_omega_cvar,
                            window_steps=args.tail_window_steps,
                        )
                        if event_name == "early":
                            early_position_cvar = independent_tail.position_cvar
                            early_omega_cvar = independent_tail.omega_cvar
                        else:
                            final_position_cvar = independent_tail.position_cvar
                            final_omega_cvar = independent_tail.omega_cvar
                        position_tail_loss = position_tail_loss + event_weight * independent_tail.position_cvar
                        omega_tail_loss = omega_tail_loss + event_weight * independent_tail.omega_cvar
                        position_component = (
                            event_weight * args.w_position_cvar * independent_tail.position_cvar
                        )
                        omega_component = event_weight * args.w_omega_cvar * independent_tail.omega_cvar
                        position_cvar_weighted = position_cvar_weighted + position_component
                        omega_cvar_weighted = omega_cvar_weighted + omega_component
                        episode_only_loss = episode_only_loss + position_component + omega_component
                        threshold_tail_loss = threshold_tail_loss + position_component + omega_component
                        position_selected = independent_tail.position_selected_mask
                        omega_selected = independent_tail.omega_selected_mask
                        position_cvar_selected_fraction = independent_tail.position_selected_fraction
                        omega_cvar_selected_fraction = independent_tail.omega_selected_fraction
                        cvar_selected_overlap_fraction = independent_tail.selected_overlap_fraction
                        position_cvar_selected_indices = ";".join(
                            str(int(index))
                            for index in independent_tail.position_selected_indices.detach().cpu().tolist()
                        )
                        omega_cvar_selected_indices = ";".join(
                            str(int(index))
                            for index in independent_tail.omega_selected_indices.detach().cpu().tolist()
                        )
                        position_cvar_selected_alpha_roll_mean = float(
                            state.alpha_roll_max[position_selected].mean().detach().item()
                        )
                        position_cvar_selected_alpha_yaw_mean = float(
                            state.alpha_yaw_max[position_selected].mean().detach().item()
                        )
                        position_cvar_selected_tau_fall_mean = float(
                            state.motor_time_falling[position_selected].mean().detach().item()
                        )
                        position_cvar_selected_thrust_to_weight_mean = float(
                            state.thrust_to_weight[position_selected].mean().detach().item()
                        )
                        position_cvar_selected_force_std_mean = float(
                            state.force_std[position_selected].mean().detach().item()
                        )
                        omega_cvar_selected_alpha_roll_mean = float(
                            state.alpha_roll_max[omega_selected].mean().detach().item()
                        )
                        omega_cvar_selected_alpha_yaw_mean = float(
                            state.alpha_yaw_max[omega_selected].mean().detach().item()
                        )
                        omega_cvar_selected_tau_fall_mean = float(
                            state.motor_time_falling[omega_selected].mean().detach().item()
                        )
                        omega_cvar_selected_thrust_to_weight_mean = float(
                            state.thrust_to_weight[omega_selected].mean().detach().item()
                        )
                        omega_cvar_selected_force_std_mean = float(
                            state.force_std[omega_selected].mean().detach().item()
                        )
                segment_loss = (
                    tracking_loss
                    + args.lambda_clf * clf_loss
                    + args.lambda_out * outward_loss
                    + args.lambda_tail * tail_loss
                    + args.lambda_du * du_loss
                    + args.lambda_ddu * ddu_loss
                    + args.lambda_sat * sat_loss
                    + args.w_retain * retain_loss
                    + lambda_motor_aux_effective * motor_aux_objective_loss
                    + lambda_capability_aux_effective * capability_aux_objective_loss
                    + lambda_response_aux_effective * response_aux_objective_loss
                    + args.w_omega_decay * omega_decay_loss
                )
                loss = segment_loss + episode_only_loss

                position_cvar_encoder_grad_norm = float("nan")
                position_cvar_gru_grad_norm = float("nan")
                omega_cvar_encoder_grad_norm = float("nan")
                omega_cvar_gru_grad_norm = float("nan")
                omega_decay_encoder_grad_norm = float("nan")
                omega_decay_gru_grad_norm = float("nan")
                attribute_cvar_gradients = (
                    bool(independent_events) if args.tail_selection_mode == "independent" else False
                ) and (gate_accepted_updates + 1) % args.training_diagnostics_every_updates == 0
                if attribute_cvar_gradients:
                    if float(position_cvar_weighted.detach().abs().item()) > 0.0:
                        position_cvar_encoder_grad_norm, position_cvar_gru_grad_norm = (
                            _component_encoder_gru_grad_norms(position_cvar_weighted, policy)
                        )
                    if float(omega_cvar_weighted.detach().abs().item()) > 0.0:
                        omega_cvar_encoder_grad_norm, omega_cvar_gru_grad_norm = (
                            _component_encoder_gru_grad_norms(omega_cvar_weighted, policy)
                        )
                attribute_decay_gradients = (
                    args.w_omega_decay > 0.0
                    and (gate_accepted_updates + 1) % args.training_diagnostics_every_updates == 0
                    and float(omega_decay_loss.detach().abs().item()) > 0.0
                )
                if attribute_decay_gradients:
                    omega_decay_encoder_grad_norm, omega_decay_gru_grad_norm = (
                        _component_encoder_gru_grad_norms(
                            args.w_omega_decay * omega_decay_loss,
                            policy,
                        )
                    )

                attribute_auxiliary_gradients = (
                    optimization_block_boundary
                    and (gate_accepted_updates + 1) % args.training_diagnostics_every_updates == 0
                )
                if attribute_auxiliary_gradients:
                    for prefix, weighted_auxiliary_loss in (
                        ("motor_aux", lambda_motor_aux_effective * motor_aux_loss),
                        ("capability_aux", lambda_capability_aux_effective * capability_aux_loss),
                        ("response_aux", lambda_response_aux_effective * response_aux_loss),
                    ):
                        if float(weighted_auxiliary_loss.detach().abs().item()) == 0.0:
                            continue
                        encoder_norm, gru_norm = _component_encoder_gru_grad_norms(
                            weighted_auxiliary_loss,
                            policy,
                        )
                        auxiliary_gradient_metrics[f"{prefix}_encoder_grad_norm"] = encoder_norm
                        auxiliary_gradient_metrics[f"{prefix}_gru_grad_norm"] = gru_norm

                if args.persistent_episode_training:
                    persistent_hidden = hidden.detach()
                    persistent_observation_state = observation_state.detach()
                    if baseline_hidden is not None:
                        baseline_persistent_hidden = baseline_hidden.detach()

                position_mean, position_max_mean, position_max_std = _metric_summary(position_errors)
                linear_velocity_mean, linear_velocity_max_mean, linear_velocity_max_std = _metric_summary(linear_velocity_errors)
                angular_velocity_mean, angular_velocity_max_mean, angular_velocity_max_std = _metric_summary(angular_velocity_errors)
                angular_acceleration_mean, angular_acceleration_max_mean, angular_acceleration_max_std = _metric_summary(angular_acceleration_errors)
                action_mean, action_max_mean, action_max_std = _metric_summary(action_errors)
                action_relative_mean, action_relative_max_mean, action_relative_max_std = _metric_summary(action_relative_errors)
                integral_world_norm_mean = float((integral_world_norm_sum / horizon).item())
                integral_body_mean = integral_body_sum / horizon
                integral_clamp_ratio = float((integral_clamp_sum / horizon).item())
                integral_residual_action_rms = float(
                    torch.sqrt(integral_residual_square_sum / horizon).item()
                )
                damping_residual_action_rms = float(
                    torch.sqrt(damping_residual_square_sum / horizon).item()
                )
                action_history_tensor = torch.stack(action_history, dim=0)
                motor_bias = action_history_tensor[-min(args.tail_window_steps, args.horizon):].mean(
                    dim=(0, 1)
                )
                force_count = max(1, int(math.ceil(0.20 * args.batch_size)))
                high_force_indices = torch.topk(
                    state.external_force.detach().norm(dim=-1),
                    k=force_count,
                    largest=True,
                    sorted=False,
                ).indices
                high_force_position_tail_rms = float(
                    torch.sqrt(
                        position_history_tensor[-args.tail_window_steps:, high_force_indices]
                        .square()
                        .sum(dim=-1)
                        .mean()
                    ).detach().item()
                )

                settle_thr = args.settling_position_mm / 1000.0
                final_position = state.position.norm(dim=-1)
                settling_fraction = (final_position < settle_thr).float().mean().item()
                raptor_metrics = {
                    "position_mean": position_mean,
                    "position_max_mean": position_max_mean,
                    "position_max_std": position_max_std,
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

                current_finite_mask = torch.isfinite(state.position).all(dim=-1)
                current_finite_mask &= torch.isfinite(state.velocity).all(dim=-1)
                current_finite_mask &= torch.isfinite(state.rotation).reshape(args.batch_size, 9).all(dim=-1)
                current_finite_mask &= torch.isfinite(state.omega).all(dim=-1)
                current_finite_mask &= torch.isfinite(state.motor).all(dim=-1)
                current_finite_mask &= torch.isfinite(state.previous_action).all(dim=-1)
                current_finite_mask &= torch.isfinite(hidden).all(dim=-1)
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
                        "motor_aux_loss": motor_aux_loss.item(),
                        "capability_aux_loss": capability_aux_loss.item(),
                        "response_aux_loss": response_aux_loss.item(),
                        "omega_decay_loss": omega_decay_loss.item(),
                        "omega_decay_active_fraction": omega_decay_active_fraction,
                        "omega_decay_horizon_0": args.omega_decay_horizons[0],
                        "omega_decay_horizon_1": args.omega_decay_horizons[1],
                        "omega_decay_horizon_2": args.omega_decay_horizons[2],
                        "omega_decay_ratio_0": omega_decay_ratios[0].item(),
                        "omega_decay_ratio_1": omega_decay_ratios[1].item(),
                        "omega_decay_ratio_2": omega_decay_ratios[2].item(),
                        "omega_decay_component_loss_0": omega_decay_component_losses[0].item(),
                        "omega_decay_component_loss_1": omega_decay_component_losses[1].item(),
                        "omega_decay_component_loss_2": omega_decay_component_losses[2].item(),
                        "omega_decay_encoder_grad_norm": omega_decay_encoder_grad_norm,
                        "omega_decay_gru_grad_norm": omega_decay_gru_grad_norm,
                        "integral_world_norm_mean": integral_world_norm_mean,
                        "integral_body_x_mean": float(integral_body_mean[0].item()),
                        "integral_body_y_mean": float(integral_body_mean[1].item()),
                        "integral_body_z_mean": float(integral_body_mean[2].item()),
                        "integral_clamp_ratio": integral_clamp_ratio,
                        "integral_residual_action_rms": integral_residual_action_rms,
                        "damping_residual_action_rms": damping_residual_action_rms,
                        "steady_motor_bias_0": float(motor_bias[0].item()),
                        "steady_motor_bias_1": float(motor_bias[1].item()),
                        "steady_motor_bias_2": float(motor_bias[2].item()),
                        "steady_motor_bias_3": float(motor_bias[3].item()),
                        "high_force_position_tail_rms": high_force_position_tail_rms,
                        "threshold_tail": threshold_tail_loss.item(),
                        "position_tail": position_tail_loss.item(),
                        "omega_tail": omega_tail_loss.item(),
                        "early_position_cvar": early_position_cvar.item(),
                        "early_omega_cvar": early_omega_cvar.item(),
                        "final_position_cvar": final_position_cvar.item(),
                        "final_omega_cvar": final_omega_cvar.item(),
                        "position_cvar_selected_fraction": position_cvar_selected_fraction,
                        "omega_cvar_selected_fraction": omega_cvar_selected_fraction,
                        "cvar_selected_overlap_fraction": cvar_selected_overlap_fraction,
                        "position_cvar_selected_indices": position_cvar_selected_indices,
                        "omega_cvar_selected_indices": omega_cvar_selected_indices,
                        "position_cvar_selected_alpha_roll_mean": position_cvar_selected_alpha_roll_mean,
                        "position_cvar_selected_alpha_yaw_mean": position_cvar_selected_alpha_yaw_mean,
                        "position_cvar_selected_tau_fall_mean": position_cvar_selected_tau_fall_mean,
                        "position_cvar_selected_thrust_to_weight_mean": position_cvar_selected_thrust_to_weight_mean,
                        "position_cvar_selected_force_std_mean": position_cvar_selected_force_std_mean,
                        "omega_cvar_selected_alpha_roll_mean": omega_cvar_selected_alpha_roll_mean,
                        "omega_cvar_selected_alpha_yaw_mean": omega_cvar_selected_alpha_yaw_mean,
                        "omega_cvar_selected_tau_fall_mean": omega_cvar_selected_tau_fall_mean,
                        "omega_cvar_selected_thrust_to_weight_mean": omega_cvar_selected_thrust_to_weight_mean,
                        "omega_cvar_selected_force_std_mean": omega_cvar_selected_force_std_mean,
                        "position_cvar_encoder_grad_norm": position_cvar_encoder_grad_norm,
                        "position_cvar_gru_grad_norm": position_cvar_gru_grad_norm,
                        "omega_cvar_encoder_grad_norm": omega_cvar_encoder_grad_norm,
                        "omega_cvar_gru_grad_norm": omega_cvar_gru_grad_norm,
                        "segment_loss": segment_loss.item(),
                        "episode_only_loss": episode_only_loss.item(),
                        "episode_boundary_gradient_scale": 1.0,
                        "w_position_cvar_effective": args.w_position_cvar,
                        "w_omega_cvar_effective": args.w_omega_cvar,
                        "early_tail_weight_effective": args.early_tail_weight,
                        "final_tail_weight_effective": args.final_tail_weight,
                        "cvar_selected_fraction": cvar_selected_fraction,
                        "cvar_selected_alpha_roll_mean": cvar_selected_alpha_roll_mean,
                        "cvar_selected_alpha_yaw_mean": cvar_selected_alpha_yaw_mean,
                        "cvar_selected_tau_fall_mean": cvar_selected_tau_fall_mean,
                        "cvar_selected_thrust_to_weight_mean": cvar_selected_thrust_to_weight_mean,
                        "cvar_selected_force_std_mean": cvar_selected_force_std_mean,
                        "retain_action_mse": retain_loss.item(),
                        "w_tail_effective": args.w_tail,
                        "w_retain_effective": args.w_retain,
                        "retain_fraction_actual": float(retain_mask.float().mean().item()),
                        "retain_bank_h500_success": (
                            float("nan") if retain_bank is None else retain_bank.baseline_h500_rate
                        ),
                        "retain_bank_h10000_success": (
                            float("nan") if retain_bank is None else retain_bank.baseline_h10000_rate
                        ),
                        "lambda_motor_aux_effective": lambda_motor_aux_effective,
                        "lambda_capability_aux_effective": lambda_capability_aux_effective,
                        "lambda_response_aux_effective": lambda_response_aux_effective,
                        **auxiliary_gradient_metrics,
                    }
                )
            loss_finite = bool(torch.isfinite(loss).item())
            state_finite = bool(current_finite_mask.all().item())
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
            gate_ready = False
            gate_suspicious = False
            post_loss_after = float("nan")
            post_loss_limit = float("nan")
            post_next_loss_after = float("nan")
            post_next_loss_limit = float("nan")
            post_hard_reject = False
            post_update_accepted = False
            force_reset_next = False
            loss_spike_reset = False
            reset_loss_limit = float("nan")
            update_applied = False
            next_segment_initial_state = _clone_state(state) if args.persistent_episode_training else None
            next_segment_initial_hidden = (
                persistent_hidden.detach().clone()
                if args.persistent_episode_training
                else None
            )
            next_segment_initial_observation_state = observation_state.detach().clone()
            skip_reason = ""
            current_loss_value = float(loss.detach().item()) if loss_finite else float("nan")
            params_before_update = _param_delta_snapshot(policy)
            grad_accum_segments_for_log = grad_accum_segments
            should_consume_update = (
                args.update_timing == "segment" or optimization_block_boundary
            )
            if (
                args.reset_loss_spike_factor > 0.0
                and gate_loss_ema is not None
                and math.isfinite(gate_loss_ema)
                and math.isfinite(current_loss_value)
            ):
                reset_loss_limit = max(
                    gate_loss_ema * args.reset_loss_spike_factor,
                    gate_loss_ema + args.reset_loss_spike_add,
                )
                if current_loss_value > reset_loss_limit:
                    loss_spike_reset = True
            rollout_valid = loss_finite and state_finite and not loss_spike_reset
            do_backward = rollout_valid
            if not rollout_valid:
                episode_valid = False
                if loss_spike_reset:
                    skip_reason = "loss_spike"
                else:
                    skip_reason = "loss_or_state_nonfinite"
            if do_backward:
                if args.update_timing == "segment":
                    optimizer.zero_grad(set_to_none=True)
                backward_loss = loss
                episode_boundary_gradient_scale = 1.0
                if (
                    args.update_timing == "episode-boundary"
                    and args.correct_episode_boundary_weighting
                    and sim_backend != "cuda-full"
                ):
                    episode_boundary_gradient_scale = float(
                        segments_per_optimization_block
                    )
                    backward_loss = accumulated_episode_objective(
                        segment_loss,
                        episode_only_loss,
                        segments_per_episode=segments_per_optimization_block,
                        correct_episode_event_weighting=True,
                    )
                    metrics["episode_boundary_gradient_scale"] = episode_boundary_gradient_scale
                backward_loss.backward()
                if args.update_timing == "episode-boundary":
                    grad_accum_segments += 1
                    grad_accum_segments_for_log = grad_accum_segments
                    if math.isfinite(current_loss_value):
                        if sim_backend == "cuda-full":
                            grad_accum_segment_loss_sum += current_loss_value
                        else:
                            grad_accum_segment_loss_sum += float(segment_loss.detach().item())
                            grad_accum_episode_only_loss_sum += float(episode_only_loss.detach().item())

                if not should_consume_update:
                    grad_stats = _grad_stats(policy)
                    if bool(grad_stats["grad_finite"]):
                        max_abs_grad_before_clip = float(grad_stats["max_abs_grad"])
                        grad_norm_fp64_before_clip = float(_grad_norm_fp64_tensor(policy).item())
                        grad_norm = grad_norm_fp64_before_clip
                        skip_reason = "defer_update_until_episode_boundary"
                    else:
                        grad_norm = float("nan")
                        skip_reason = str(grad_stats["first_nan_param_name"] or "grad_tensor_nonfinite")
                        optimizer.zero_grad(set_to_none=True)
                        grad_accum_segments = 0
                        grad_accum_segment_loss_sum = 0.0
                        grad_accum_episode_only_loss_sum = 0.0
                        episode_valid = False
                else:
                    if args.update_timing == "episode-boundary" and not episode_valid:
                        grad_stats = _grad_stats(policy)
                        if bool(grad_stats["grad_finite"]):
                            max_abs_grad_before_clip = float(grad_stats["max_abs_grad"])
                            grad_norm_fp64_before_clip = float(_grad_norm_fp64_tensor(policy).item())
                            grad_norm = grad_norm_fp64_before_clip
                        else:
                            grad_norm = float("nan")
                        skip_reason = "episode_invalid"
                    elif args.update_timing == "episode-boundary":
                        if grad_accum_segments <= 0:
                            raise RuntimeError("episode-boundary update reached with no accumulated gradients")
                        _scale_policy_grads(policy, 1.0 / float(grad_accum_segments))
                        loss_value = grad_accum_segment_loss_sum / float(grad_accum_segments)
                        if args.correct_episode_boundary_weighting:
                            loss_value += grad_accum_episode_only_loss_sum
                        else:
                            loss_value += grad_accum_episode_only_loss_sum / float(grad_accum_segments)
                    else:
                        loss_value = float(loss.detach().item())

                    if skip_reason != "episode_invalid":
                        grad_stats = _grad_stats(policy)
                    if skip_reason != "episode_invalid" and bool(grad_stats["grad_finite"]):
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
                            gate_ready = (
                                args.adaptive_update_gate
                                and gate_accepted_updates >= args.gate_warmup_updates
                                and gate_loss_ema is not None
                                and gate_grad_ema is not None
                                and math.isfinite(gate_loss_ema)
                                and math.isfinite(gate_grad_ema)
                            )
                            if gate_ready:
                                loss_limit = max(
                                    gate_loss_ema * args.gate_loss_factor,
                                    gate_loss_ema + args.gate_loss_add,
                                )
                                grad_limit = max(
                                    gate_grad_ema * args.gate_grad_factor,
                                    args.gate_grad_floor,
                                )
                                if loss_value > loss_limit or grad_norm_fp64_before_clip > grad_limit:
                                    gate_suspicious = True
                                if (
                                    gate_suspicious
                                    and args.hard_reject_suspicious_grad > 0.0
                                    and grad_norm_fp64_before_clip > args.hard_reject_suspicious_grad
                                ):
                                    post_hard_reject = True

                            need_post_check = (
                                args.post_update_check in {"all", "finite"}
                                or (args.post_update_check == "suspicious" and gate_suspicious)
                            )
                            if post_hard_reject:
                                skip_reason = "adaptive_gate_hard_grad"
                            elif args.adaptive_update_gate and gate_suspicious and args.post_update_check == "off":
                                skip_reason = "adaptive_gate_suspicious"
                            else:
                                saved_model_state = None
                                saved_optimizer_state = None
                                if need_post_check:
                                    saved_model_state = _snapshot_module_state(policy)
                                    saved_optimizer_state = _snapshot_optimizer_state(optimizer)

                                optimizer.step()
                                update_applied = True
                                post_update_accepted = True

                                if need_post_check:
                                    (
                                        post_loss_after,
                                        post_finite,
                                    ) = _post_update_loss_check(
                                        policy=policy,
                                        sim=sim,
                                        initial_state=diagnostic_initial_state,
                                        loss_config=loss_config,
                                        args=args,
                                        sim_backend=sim_backend,
                                        state_step_decay=state_step_decay,
                                        hidden_step_decay=hidden_step_decay,
                                        noise_seed=args.seed * 1000003 + step_idx,
                                        initial_hidden=diagnostic_initial_hidden,
                                        initial_observation_state_value=diagnostic_initial_observation_state,
                                        episode_step_offset=diagnostic_episode_steps,
                                        lambda_motor_aux_effective=lambda_motor_aux_effective,
                                        lambda_capability_aux_effective=lambda_capability_aux_effective,
                                        lambda_response_aux_effective=lambda_response_aux_effective,
                                    )
                                    if args.post_update_check == "finite":
                                        post_loss_limit = float("inf")
                                        post_update_accepted = (
                                            post_finite and math.isfinite(post_loss_after)
                                        )
                                    else:
                                        before_limit = max(
                                            loss_value * args.post_loss_factor,
                                            loss_value + args.post_loss_add,
                                        )
                                        post_loss_limit = before_limit
                                        if gate_loss_ema is not None and math.isfinite(gate_loss_ema):
                                            ema_limit = max(
                                                gate_loss_ema * args.post_ema_loss_factor,
                                                gate_loss_ema + args.post_ema_loss_add,
                                            )
                                            post_loss_limit = min(before_limit, ema_limit)

                                        post_update_accepted = (
                                            post_finite
                                            and math.isfinite(post_loss_after)
                                            and post_loss_after <= post_loss_limit
                                        )

                                    if (
                                        post_update_accepted
                                        and args.post_update_next_check
                                        and args.persistent_episode_training
                                        and (
                                            args.update_timing == "segment"
                                            or (
                                                args.optimization_block_horizon > 0
                                                and not reset_episode_boundary
                                            )
                                        )
                                        and next_segment_initial_state is not None
                                    ):
                                        (
                                            post_next_loss_after,
                                            post_next_finite,
                                        ) = _post_update_loss_check(
                                            policy=policy,
                                            sim=sim,
                                            initial_state=next_segment_initial_state,
                                            loss_config=loss_config,
                                            args=args,
                                            sim_backend=sim_backend,
                                            state_step_decay=state_step_decay,
                                            hidden_step_decay=hidden_step_decay,
                                            noise_seed=args.seed * 1000003 + step_idx + 7919,
                                            initial_hidden=next_segment_initial_hidden,
                                            initial_observation_state_value=next_segment_initial_observation_state,
                                            episode_step_offset=diagnostic_episode_steps + args.horizon,
                                            lambda_motor_aux_effective=lambda_motor_aux_effective,
                                            lambda_capability_aux_effective=lambda_capability_aux_effective,
                                            lambda_response_aux_effective=lambda_response_aux_effective,
                                        )
                                        if args.post_update_check == "finite":
                                            post_next_loss_limit = float("inf")
                                            post_update_accepted = (
                                                post_next_finite
                                                and math.isfinite(post_next_loss_after)
                                            )
                                        else:
                                            next_before_limit = max(
                                                loss_value * args.post_next_loss_factor,
                                                loss_value + args.post_next_loss_add,
                                            )
                                            post_next_loss_limit = next_before_limit
                                            if gate_loss_ema is not None and math.isfinite(gate_loss_ema):
                                                next_ema_limit = max(
                                                    gate_loss_ema * args.post_next_ema_loss_factor,
                                                    gate_loss_ema + args.post_next_ema_loss_add,
                                                )
                                                post_next_loss_limit = min(next_before_limit, next_ema_limit)
                                            post_update_accepted = (
                                                post_next_finite
                                                and math.isfinite(post_next_loss_after)
                                                and post_next_loss_after <= post_next_loss_limit
                                            )

                                    if not post_update_accepted:
                                        if saved_model_state is None or saved_optimizer_state is None:
                                            raise RuntimeError("post-update rollback requires saved states")
                                        policy.load_state_dict(saved_model_state)
                                        optimizer.load_state_dict(saved_optimizer_state)
                                        _optimizer_to_device(optimizer, device)
                                        _set_optimizer_hparams(
                                            optimizer,
                                            lr=args.lr,
                                            weight_decay=args.weight_decay,
                                        )
                                        update_applied = False
                                        skip_reason = "post_update_rejected"

                            if update_applied:
                                gate_loss_ema = _update_ema(gate_loss_ema, loss_value, args.gate_ema_beta)
                                gate_grad_ema = _update_ema(gate_grad_ema, grad_norm_fp64_before_clip, args.gate_ema_beta)
                                gate_accepted_updates += 1
                    else:
                        if skip_reason != "episode_invalid":
                            grad_norm = float("nan")
                            skip_reason = str(grad_stats["first_nan_param_name"] or "grad_tensor_nonfinite")
                        episode_valid = False

                    if args.update_timing == "episode-boundary":
                        grad_accum_segments_for_log = grad_accum_segments
                        optimizer.zero_grad(set_to_none=True)
                        grad_accum_segments = 0
                        grad_accum_segment_loss_sum = 0.0
                        grad_accum_episode_only_loss_sum = 0.0
            else:
                optimizer.zero_grad(set_to_none=True)
                grad_accum_segments = 0
                grad_accum_segment_loss_sum = 0.0
                grad_accum_episode_only_loss_sum = 0.0
                grad_norm = float("nan")
                episode_valid = False
                if not skip_reason:
                    skip_reason = "loss_or_state_nonfinite"
            if args.reset_after_skipped_update and skip_reason in RESET_AFTER_SKIP_REASONS:
                force_reset_next = True
            if loss_spike_reset:
                force_reset_next = True
            if (
                update_applied
                and args.persistent_episode_training
                and (
                    args.optimization_block_horizon == 0
                    or reset_episode_boundary
                )
            ):
                force_reset_next = True
                must_reset_next = True
            if skip_reason in RESET_AFTER_SKIP_REASONS:
                optimizer.zero_grad(set_to_none=True)
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
            if args.persistent_episode_training and sim_backend != "cuda-full":
                state = _clone_state(state)
            diagnostics_due = (
                step_idx == 1
                or args.training_diagnostics_every_updates == 1
                or (
                    update_applied
                    and gate_accepted_updates % args.training_diagnostics_every_updates == 0
                )
            )
            if not diagnostics_due:
                raptor_metrics = _empty_raptor_metrics(args.settling_position_mm)
            elif args.numerics_audit and sim_backend == "cuda-full":
                raptor_metrics = _empty_raptor_metrics(args.settling_position_mm)
            else:
                policy.eval()
                raptor_metrics, _, _ = rollout_diagnostics(
                    policy,
                    sim,
                    diagnostic_initial_state,
                    args,
                    step_backend=diagnostic_backend,
                    eval_seed=args.seed,
                    trajectory_count=0,
                    # An accepted update changes the GRU parameters, so hidden
                    # produced by the old parameters is no longer reusable.
                    initial_hidden=None if update_applied else diagnostic_initial_hidden,
                )
                policy.train()
            segment_mask_mean = reset_mask.float().mean().item()
            hidden_state_norm_initial = (
                float("nan")
                if diagnostic_initial_hidden is None
                else float(
                    torch.linalg.vector_norm(diagnostic_initial_hidden, dim=-1)
                    .mean()
                    .item()
                )
            )
            integral_state_norm_initial = float(
                torch.linalg.vector_norm(
                    diagnostic_initial_observation_state.integral_position,
                    dim=-1,
                )
                .mean()
                .item()
            )

            elapsed = perf_counter() - start
            row = {
                "step": step_idx,
                "optimizer_update": gate_accepted_updates,
                "physical_steps": physical_steps_after_segment,
                "episode_target_steps": episode_target_steps,
                "episode_horizon_phase": episode_horizon_phase,
                "optimization_block_horizon": optimization_block_horizon,
                "optimization_block_boundary": int(optimization_block_boundary),
                "reset_episode_boundary": int(reset_episode_boundary),
                "first_optimization_segment": int(first_optimization_segment),
                "tail_supervision_block_horizon": tail_supervision_block_horizon,
                "tail_supervision_block_boundary": int(tail_supervision_block_boundary),
                "first_tail_supervision_segment": int(first_tail_supervision_segment),
                "hidden_state_norm_initial": hidden_state_norm_initial,
                "integral_state_norm_initial": integral_state_norm_initial,
                "grad_norm": float(grad_norm),
                "grad_norm_encoder": float(grad_stats.get("grad_norm_encoder", float("nan"))),
                "grad_norm_gru": float(grad_stats.get("grad_norm_gru", float("nan"))),
                "grad_norm_fp64_before_clip": float(grad_norm_fp64_before_clip),
                "grad_norm_fp64_after_clip": float(grad_norm_fp64_after_clip),
                "max_abs_grad_before_clip": float(max_abs_grad_before_clip),
                "grad_scale": float(grad_scale),
                "update_applied": int(update_applied),
                "skip_reason": skip_reason,
                "gate_loss_ema": float("nan") if gate_loss_ema is None else float(gate_loss_ema),
                "gate_grad_ema": float("nan") if gate_grad_ema is None else float(gate_grad_ema),
                "gate_ready": int(gate_ready),
                "gate_suspicious": int(gate_suspicious),
                "post_loss_after": float(post_loss_after),
                "post_loss_limit": float(post_loss_limit),
                "post_next_loss_after": float(post_next_loss_after),
                "post_next_loss_limit": float(post_next_loss_limit),
                "post_hard_reject": int(post_hard_reject),
                "post_update_accepted": int(post_update_accepted),
                "force_reset_next": int(force_reset_next),
                "loss_spike_reset": int(loss_spike_reset),
                "reset_loss_limit": float(reset_loss_limit),
                "rollout_valid": int(rollout_valid),
                "episode_valid": int(episode_valid),
                "max_abs_param_delta": float(max_abs_param_delta),
                "seconds": elapsed,
                "update_timing": args.update_timing,
                "episode_boundary": int(episode_boundary),
                "grad_accum_segments": int(grad_accum_segments_for_log),
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
                    "step={step} update={optimizer_update} loss={loss:.6f} track={tracking:.6f} "
                    "clf={clf:.6f} tail={tail:.6f} grad={grad_norm:.3f} "
                    "update={update_applied} skip={skip_reason}".format(
                        step=step_idx,
                        optimizer_update=gate_accepted_updates,
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

            should_save_main = (
                update_applied
                and args.save_every > 0
                and gate_accepted_updates % args.save_every == 0
            )
            physical_checkpoint_due = (
                update_applied
                and args.physical_step_budget > 0
                and physical_steps_after_segment in checkpoint_physical_steps
            )
            should_save_history = (
                (update_applied and gate_accepted_updates in checkpoint_updates)
                or physical_checkpoint_due
                or (args.optimizer_updates == 0 and step_idx in checkpoint_steps)
            )
            if should_save_main or should_save_history:
                training_state_payload = _capture_training_state(
                    state=state,
                    persistent_hidden=persistent_hidden,
                    persistent_observation_state=persistent_observation_state,
                    baseline_persistent_hidden=baseline_persistent_hidden,
                    episode_id=episode_id,
                    segment_id=segment_id,
                    episode_steps=episode_steps,
                    current_finite_mask=current_finite_mask,
                    retain_mask=retain_mask,
                    retain_bank_indices=retain_bank_indices,
                    force_reset_next=force_reset_next,
                    must_reset_next=must_reset_next,
                    episode_valid=episode_valid,
                    gate_loss_ema=gate_loss_ema,
                    gate_grad_ema=gate_grad_ema,
                    gate_accepted_updates=gate_accepted_updates,
                    physical_steps_completed=physical_steps_after_segment,
                    episode_target_steps=episode_target_steps,
                    curriculum_episode_index=curriculum_episode_index,
                    step_idx=step_idx,
                    horizon=args.horizon,
                    persistent_episode_training=args.persistent_episode_training,
                )
                checkpoint_payload = {
                    "architecture": policy.architecture_metadata(),
                    "step": step_idx,
                    "optimizer_update": gate_accepted_updates,
                    "physical_steps": physical_steps_after_segment,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "training_state": training_state_payload,
                }
                if should_save_main:
                    torch.save(checkpoint_payload, checkpoint_path)
                if should_save_history:
                    if physical_checkpoint_due:
                        history_label = f"physical_steps_{physical_steps_after_segment}"
                    elif update_applied and gate_accepted_updates in checkpoint_updates:
                        history_label = f"update_{gate_accepted_updates}"
                    else:
                        history_label = f"step_{step_idx}"
                    step_checkpoint_path = checkpoint_path.with_name(
                        f"{checkpoint_path.stem}_{history_label}{checkpoint_path.suffix}"
                    )
                    torch.save(checkpoint_payload, step_checkpoint_path)

            if args.persistent_episode_training:
                episode_steps = episode_steps + args.horizon
            if args.physical_step_budget > 0:
                physical_steps_completed = physical_steps_after_segment
            finite_mask = current_finite_mask
            invalid_mask = ~finite_mask
            if args.persistent_episode_training and force_reset_next:
                invalid_mask = torch.ones_like(invalid_mask)
            last_completed_step = step_idx
        if args.physical_step_budget > 0 and physical_steps_completed != args.physical_step_budget:
            raise RuntimeError(
                f"training stopped at {physical_steps_completed} physical steps; "
                f"target was {args.physical_step_budget} within {outer_step_limit} outer steps"
            )
        if args.optimizer_updates > 0 and gate_accepted_updates != args.optimizer_updates:
            raise RuntimeError(
                f"training stopped at {gate_accepted_updates} optimizer updates; "
                f"target was {args.optimizer_updates} within {outer_step_limit} outer steps"
            )
        final_step = last_completed_step
        final_training_state = _capture_training_state(
            state=state,
            persistent_hidden=persistent_hidden,
            persistent_observation_state=persistent_observation_state,
            baseline_persistent_hidden=baseline_persistent_hidden,
            episode_id=episode_id,
            segment_id=segment_id,
            episode_steps=episode_steps,
            current_finite_mask=current_finite_mask,
            retain_mask=retain_mask,
            retain_bank_indices=retain_bank_indices,
            force_reset_next=force_reset_next,
            must_reset_next=must_reset_next,
            episode_valid=episode_valid,
            gate_loss_ema=gate_loss_ema,
            gate_grad_ema=gate_grad_ema,
            gate_accepted_updates=gate_accepted_updates,
            physical_steps_completed=physical_steps_completed,
            episode_target_steps=episode_target_steps,
            curriculum_episode_index=curriculum_episode_index,
            step_idx=final_step,
            # The loop-bottom bookkeeping has already advanced episode_steps.
            horizon=0,
            persistent_episode_training=args.persistent_episode_training,
        )
        torch.save(
            {
                "architecture": policy.architecture_metadata(),
                "step": final_step,
                "optimizer_update": gate_accepted_updates,
                "physical_steps": physical_steps_completed,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "training_state": final_training_state,
            },
            checkpoint_path,
        )
        if args.optimizer_updates == 0 and final_step in checkpoint_steps:
            torch.save(
                {
                    "architecture": policy.architecture_metadata(),
                    "step": final_step,
                    "optimizer_update": gate_accepted_updates,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "training_state": final_training_state,
                },
                checkpoint_path.with_name(f"{checkpoint_path.stem}_step_{final_step}{checkpoint_path.suffix}"),
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
