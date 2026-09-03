from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train  # noqa: E402
from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from l2f_cuda_backend import cuda_step, load_extension  # noqa: E402
from model import MotorGRUPolicy  # noqa: E402
from policy_observation import (  # noqa: E402
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)
from retain_bank import apply_retain_bank_samples, load_retain_bank  # noqa: E402


SNAPSHOTS = (500, 2000, 5000)
TAIL_STEPS = 100
PRIMARY_ROWS = (
    ("H5000 tail position", "h5000_position", "error"),
    ("H5000 CVaR20 position", "h5000_position", "cvar"),
    ("H2000 tail position", "h2000_position", "error"),
    ("H500 recovery position", "h500_position", "error"),
    ("H500 recovery omega", "h500_omega", "error"),
    ("H5000 success", "h5000_success", "success"),
)


class _ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, line: str) -> list[str]:
        return shlex.split(line, comments=True, posix=True)


def parse_args() -> argparse.Namespace:
    parser = _ArgumentParser(
        description="30-minute paired TBPTT/multiple-shooting value screen.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--base-config", type=Path, default=ROOT / "configs/time_horizon_T2.args"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/multiple_shooting_timed_value_screen",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timing-updates", type=int, default=2)
    parser.add_argument("--paired-training-seconds", type=float, default=18.0 * 60.0)
    parser.add_argument("--total-wall-seconds", type=float, default=30.0 * 60.0)
    parser.add_argument("--eval-reserve-seconds", type=float, default=7.0 * 60.0)
    parser.add_argument("--ordinary-count", type=int, default=64)
    parser.add_argument("--hard-count", type=int, default=64)
    parser.add_argument("--hard-candidate-count", type=int, default=384)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ordinary-seed", type=int, default=7001)
    parser.add_argument("--hard-seed", type=int, default=7002)
    parser.add_argument("--bootstrap-seed", type=int, default=7003)
    return parser.parse_args()


def _load_train_args(path: Path) -> argparse.Namespace:
    candidate = path if path.is_absolute() else ROOT / path
    previous = sys.argv
    try:
        sys.argv = ["train.py", f"@{candidate}"]
        args = train.parse_args()
        train.apply_direct_h500_training_defaults(args)
        return args
    finally:
        sys.argv = previous


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_payload(state: L2FState) -> dict[str, torch.Tensor]:
    return {
        field.name: getattr(state, field.name).detach().cpu()
        for field in fields(L2FState)
    }


def _clone_state(state: L2FState) -> L2FState:
    return L2FState(
        **{
            field.name: getattr(state, field.name).detach().clone()
            for field in fields(L2FState)
        }
    )


def _load_q2_policy(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> MotorGRUPolicy:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    saved_args = checkpoint.get("args", {})
    policy = MotorGRUPolicy(
        observation_dim=25,
        encoder_dim=int(saved_args.get("encoder_dim", 192)),
        hidden_dim=int(saved_args.get("hidden_dim", 192)),
        encoder_depth=int(saved_args.get("encoder_depth", 2)),
        enable_integral_residual=bool(
            saved_args.get("enable_integral_residual", False)
        ),
        enable_damping_residual=bool(
            saved_args.get("enable_rate_damping_residual", False)
        ),
        integral_residual_hidden_dim=int(
            saved_args.get("integral_residual_hidden_dim", 16)
        ),
        damping_residual_hidden_dim=int(
            saved_args.get("damping_residual_hidden_dim", 32)
        ),
        integral_residual_scale=float(
            saved_args.get("integral_residual_scale", 1.0)
        ),
        damping_residual_scale=float(
            saved_args.get("damping_residual_scale", 1.0)
        ),
    ).to(device=device, dtype=torch.float32)
    missing, unexpected = policy.load_compatible_state_dict(
        checkpoint.get("model", checkpoint)
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Q2 load mismatch: missing={missing}, unexpected={unexpected}"
        )
    policy.eval()
    return policy


def _state_from_payload(
    payload: dict[str, torch.Tensor], *, device: torch.device, dtype: torch.dtype
) -> L2FState:
    return L2FState(
        **{
            field.name: torch.as_tensor(
                payload[field.name], device=device, dtype=dtype
            )
            for field in fields(L2FState)
        }
    )


def _select_state(state: L2FState, indices: torch.Tensor) -> L2FState:
    return L2FState(
        **{
            field.name: getattr(state, field.name).index_select(0, indices)
            for field in fields(L2FState)
        }
    )


def _concat_states(states: Iterable[L2FState]) -> L2FState:
    items = tuple(states)
    return L2FState(
        **{
            field.name: torch.cat(
                tuple(getattr(state, field.name) for state in items), dim=0
            )
            for field in fields(L2FState)
        }
    )


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _make_sim(train_args: argparse.Namespace) -> L2FSimulator:
    return L2FSimulator(
        L2FParams(
            dt=train_args.dt,
            max_initial_position=train_args.max_initial_position,
            max_initial_velocity=train_args.max_initial_velocity,
            max_initial_angle=train_args.max_initial_angle,
            max_initial_omega=train_args.max_initial_omega,
            disturbance_force_max=train_args.disturbance_force_max,
            external_force_ratio=train_args.external_force_ratio,
        )
    )


def _sample_q2_state(
    sim: L2FSimulator,
    train_args: argparse.Namespace,
    count: int,
    *,
    seed: int,
    device: torch.device,
) -> L2FState:
    _set_seed(seed, device)
    return sim.reset(
        count,
        device=device,
        dtype=torch.float32,
        sample_dynamics=train_args.sample_dynamics,
        sampled_dynamics_level=train_args.sampled_dynamics_level,
        broad_sampler=train_args.broad_sampler,
        balanced_dynamics_sampling=train_args.balanced_dynamics_sampling,
        sample_external_force=not train_args.disable_sampled_external_force,
    )


def _write_training_schedule(
    path: Path,
    *,
    sim: L2FSimulator,
    train_args: argparse.Namespace,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    state = _sample_q2_state(
        sim, train_args, batch_size, seed=seed, device=device
    )
    retain_path = Path(train_args.retain_bank_path)
    if not retain_path.is_absolute():
        retain_path = ROOT / retain_path
    retain_bank = load_retain_bank(retain_path)
    reset_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
    retain_mask, retain_indices = apply_retain_bank_samples(
        state,
        retain_bank,
        reset_mask,
        fraction=train_args.retain_fraction,
    )
    payload = {
        "format": "fixed-q2-training-schedule-v1",
        "schedule_semantics": "one fixed batch replayed for every optimizer update",
        "seed": seed,
        "state": _state_payload(state),
        "retain_mask": retain_mask.detach().cpu(),
        "retain_indices": retain_indices.detach().cpu(),
    }
    torch.save(payload, path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "batch_size": batch_size,
        "retain_count": int(retain_mask.sum().item()),
        "retain_indices": [int(value) for value in retain_indices.detach().cpu().tolist()],
    }


def _finite_state(state: L2FState) -> torch.Tensor:
    finite = torch.ones(
        state.position.shape[0], device=state.position.device, dtype=torch.bool
    )
    for name in ("position", "velocity", "rotation", "omega", "motor", "previous_action"):
        value = getattr(state, name).reshape(state.position.shape[0], -1)
        finite &= torch.isfinite(value).all(dim=-1)
    return finite


@torch.no_grad()
def native_l2f_rollout(
    policy: torch.nn.Module,
    initial_state: L2FState,
    train_args: argparse.Namespace,
    *,
    horizon: int = 5000,
    backend: str = "cuda",
) -> dict[str, np.ndarray]:
    """Evaluate fixed states only through the repository's native L2F simulator."""

    if horizon < max(SNAPSHOTS):
        raise ValueError("the value screen requires a full H5000 rollout")
    if backend == "cuda":
        load_extension()
    sim = _make_sim(train_args)
    state = _clone_state(initial_state)
    batch_size = state.position.shape[0]
    hidden = policy.initial_hidden(
        batch_size, device=state.position.device, dtype=state.position.dtype
    )
    observation_state = initial_observation_state(
        batch_size, device=state.position.device, dtype=state.position.dtype
    )
    alive = torch.ones(batch_size, device=state.position.device, dtype=torch.bool)
    position_window: list[torch.Tensor] = []
    velocity_window: list[torch.Tensor] = []
    omega_window: list[torch.Tensor] = []
    output: dict[str, np.ndarray] = {}
    policy.eval()
    for step in range(1, horizon + 1):
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode=train_args.observation_mode,
            noise_max=0.0,
            integral_input_frame=train_args.integral_input_frame,
            integral_input_multiplier=train_args.integral_input_multiplier,
        )
        action, hidden = policy(observation, hidden)
        observation_state = update_position_integral(
            observation_state,
            observed_position,
            dt=train_args.dt,
            integral_limit=train_args.integral_limit,
            integral_leak=train_args.integral_leak,
            integral_clamp_mode=getattr(train_args, "integral_clamp_mode", None) or "box",
        )
        state = (
            cuda_step(state, action, sim.params, grad_decay=1.0)
            if backend == "cuda"
            else sim.step(state, action, grad_decay=1.0)
        )
        alive &= _finite_state(state)
        position = torch.nan_to_num(
            state.position.norm(dim=-1), nan=1.0e6, posinf=1.0e6, neginf=1.0e6
        )
        velocity = torch.nan_to_num(
            state.velocity.norm(dim=-1), nan=1.0e6, posinf=1.0e6, neginf=1.0e6
        )
        omega = torch.nan_to_num(
            state.omega.norm(dim=-1), nan=1.0e6, posinf=1.0e6, neginf=1.0e6
        )
        position_window.append(position)
        velocity_window.append(velocity)
        omega_window.append(omega)
        if len(position_window) > TAIL_STEPS:
            position_window.pop(0)
            velocity_window.pop(0)
            omega_window.pop(0)
        if step in SNAPSHOTS:
            position_stack = torch.stack(tuple(position_window), dim=0)
            velocity_stack = torch.stack(tuple(velocity_window), dim=0)
            omega_stack = torch.stack(tuple(omega_window), dim=0)
            joint_success = (
                (position_stack < float(train_args.success_position_m))
                & (velocity_stack < float(train_args.success_velocity))
                & (omega_stack < float(train_args.success_omega))
                & alive.unsqueeze(0)
            )
            prefix = f"h{step}"
            output[f"{prefix}_position"] = position_stack.mean(dim=0).cpu().numpy()
            output[f"{prefix}_velocity"] = velocity_stack.mean(dim=0).cpu().numpy()
            output[f"{prefix}_omega"] = omega_stack.mean(dim=0).cpu().numpy()
            output[f"{prefix}_success"] = (
                joint_success.float().mean(dim=0) >= 0.95
            ).float().cpu().numpy()
            output[f"{prefix}_finite"] = alive.float().cpu().numpy()
    return output


def _hard_selection(
    state: L2FState,
    metrics: dict[str, np.ndarray],
    *,
    count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    force = state.external_force.detach().norm(dim=-1).cpu().numpy()
    finite = (
        (metrics["h500_finite"] > 0.5)
        & (metrics["h2000_finite"] > 0.5)
        & (metrics["h5000_finite"] > 0.5)
    )
    tiers = (
        ("strict", 0.075, 0.15, 0.30, 0.25),
        ("standard", 0.10, 0.20, 0.40, 0.35),
        ("relaxed", 0.15, 0.30, 0.60, 0.50),
    )
    selected_mask = None
    selected_tier = ""
    for name, p_limit, v_limit, omega_limit, long_p_cap in tiers:
        mask = (
            finite
            & (force > 1.0e-6)
            & (metrics["h500_position"] <= p_limit)
            & (metrics["h500_velocity"] <= v_limit)
            & (metrics["h500_omega"] <= omega_limit)
            & (metrics["h5000_position"] <= long_p_cap)
        )
        if int(mask.sum()) >= count:
            selected_mask = mask
            selected_tier = name
            break
    if selected_mask is None:
        raise RuntimeError(
            "fixed hard candidate pool did not contain enough hard-but-recoverable "
            f"scenarios: need {count}"
        )
    candidates = np.flatnonzero(selected_mask)
    hard_score = metrics["h5000_position"] + 0.25 * metrics["h2000_position"]
    order = candidates[np.argsort(hard_score[candidates])[::-1]]
    chosen = order[:count]
    return chosen, {
        "selection_tier": selected_tier,
        "eligible_count": int(selected_mask.sum()),
        "candidate_count": int(state.position.shape[0]),
        "selected_count": int(chosen.size),
        "selected_candidate_indices": [int(index) for index in chosen.tolist()],
        "selected_force_norm_min": float(force[chosen].min()),
        "selected_h500_position_max": float(metrics["h500_position"][chosen].max()),
        "selected_h5000_position_min": float(metrics["h5000_position"][chosen].min()),
        "selected_h5000_position_max": float(metrics["h5000_position"][chosen].max()),
    }


def _write_eval_set(
    path: Path,
    *,
    sim: L2FSimulator,
    policy: torch.nn.Module,
    train_args: argparse.Namespace,
    ordinary_count: int,
    hard_count: int,
    candidate_count: int,
    ordinary_seed: int,
    hard_seed: int,
    device: torch.device,
    backend: str,
) -> tuple[L2FState, list[str], dict[str, Any]]:
    ordinary = _sample_q2_state(
        sim, train_args, ordinary_count, seed=ordinary_seed, device=device
    )
    candidates = _sample_q2_state(
        sim, train_args, candidate_count, seed=hard_seed, device=device
    )
    candidate_metrics = native_l2f_rollout(
        policy, candidates, train_args, horizon=5000, backend=backend
    )
    selected, selection = _hard_selection(candidates, candidate_metrics, count=hard_count)
    hard = _select_state(
        candidates, torch.as_tensor(selected, device=device, dtype=torch.long)
    )
    combined = _concat_states((ordinary, hard))
    groups = ["ordinary"] * ordinary_count + ["hard"] * hard_count
    payload = {
        "format": "native-l2f-paired-eval-v1",
        "state": _state_payload(combined),
        "groups": groups,
        "ordinary_seed": ordinary_seed,
        "hard_seed": hard_seed,
        "hard_selection": selection,
    }
    torch.save(payload, path)
    manifest = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "ordinary_count": ordinary_count,
        "hard_count": hard_count,
        "hard_selection": selection,
        "evaluation_backend": f"native L2F {backend} step",
    }
    return combined, groups, manifest


def _load_arm_policy(
    base_checkpoint: Path,
    arm_checkpoint: Path | None,
    *,
    device: torch.device,
) -> torch.nn.Module:
    policy = _load_q2_policy(base_checkpoint, device=device)
    if arm_checkpoint is not None:
        payload = torch.load(arm_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = policy.load_compatible_state_dict(payload["model"])
        if missing or unexpected:
            raise RuntimeError(
                f"arm checkpoint load mismatch: missing={missing}, unexpected={unexpected}"
            )
    policy.eval()
    return policy


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_value(values: np.ndarray, mode: str) -> float:
    if mode == "cvar":
        count = max(1, int(math.ceil(0.20 * values.size)))
        return float(np.sort(values)[-count:].mean())
    return float(values.mean())


def paired_bootstrap_extra_improvement(
    tbptt: np.ndarray,
    multiple_shooting: np.ndarray,
    *,
    mode: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return MS extra improvement and its paired percentile interval."""

    if tbptt.shape != multiple_shooting.shape or tbptt.ndim != 1:
        raise ValueError("paired bootstrap inputs must be equal-length vectors")
    if mode not in ("error", "cvar", "success"):
        raise ValueError(f"unknown bootstrap mode: {mode}")
    point = (
        _metric_value(multiple_shooting, mode)
        - _metric_value(tbptt, mode)
        if mode == "success"
        else _metric_value(tbptt, mode) - _metric_value(multiple_shooting, mode)
    )
    rng = np.random.default_rng(seed)
    statistics: list[np.ndarray] = []
    remaining = replicates
    while remaining > 0:
        block = min(remaining, 1000)
        indices = rng.integers(0, tbptt.size, size=(block, tbptt.size))
        left = tbptt[indices]
        right = multiple_shooting[indices]
        if mode == "cvar":
            count = max(1, int(math.ceil(0.20 * tbptt.size)))
            left_stat = np.sort(left, axis=1)[:, -count:].mean(axis=1)
            right_stat = np.sort(right, axis=1)[:, -count:].mean(axis=1)
            statistics.append(left_stat - right_stat)
        elif mode == "success":
            statistics.append(right.mean(axis=1) - left.mean(axis=1))
        else:
            statistics.append(left.mean(axis=1) - right.mean(axis=1))
        remaining -= block
    bootstrap = np.concatenate(statistics)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return point, float(low), float(high)


def _run_arm(
    *,
    mode: str,
    output_dir: Path,
    schedule_path: Path,
    updates: int,
    device: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    config_name = (
        "multiple_shooting_h1000_tbptt.args"
        if mode == "tbptt"
        else "multiple_shooting_h1000_penalty.args"
    )
    command = [
        sys.executable,
        str(ROOT / "tools/train_multiple_shooting_h1000.py"),
        f"@{ROOT / 'configs' / config_name}",
        "--output-dir",
        str(output_dir),
        "--training-schedule",
        str(schedule_path),
        "--updates",
        str(updates),
        "--future-credit-probe-steps",
        "0",
        "--device",
        device,
        "--log-every",
        str(max(updates // 10, 1)),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=max(timeout_seconds, 60.0),
    )
    wall_seconds = time.perf_counter() - started
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    summary["wall_seconds"] = wall_seconds
    summary["command"] = command
    return summary


def _evaluation_rows(
    labels: list[str], groups: list[str], metrics: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(len(groups)):
        row: dict[str, Any] = {
            "checkpoint": labels[index],
            "scenario_index": index,
            "scenario_group": groups[index],
        }
        for name, values in metrics.items():
            row[name] = float(values[index])
        rows.append(row)
    return rows


def _summarize_results(
    evaluations: dict[str, dict[str, np.ndarray]],
    *,
    replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table: list[dict[str, Any]] = []
    for row_index, (label, key, mode) in enumerate(PRIMARY_ROWS):
        before = _metric_value(evaluations["before"][key], mode)
        tbptt = _metric_value(evaluations["tbptt"][key], mode)
        ms = _metric_value(evaluations["multiple_shooting"][key], mode)
        extra, low, high = paired_bootstrap_extra_improvement(
            evaluations["tbptt"][key],
            evaluations["multiple_shooting"][key],
            mode=mode,
            replicates=replicates,
            seed=bootstrap_seed + row_index,
        )
        if mode == "success":
            gain_tbptt = tbptt - before
            gain_ms = ms - before
        else:
            gain_tbptt = before - tbptt
            gain_ms = before - ms
        table.append(
            {
                "metric": label,
                "before": before,
                "tbptt": tbptt,
                "multiple_shooting": ms,
                "tbptt_improvement": gain_tbptt,
                "ms_improvement": gain_ms,
                "ms_extra_improvement": extra,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    auxiliary: list[dict[str, Any]] = []
    for group in ("all", "ordinary", "hard"):
        mask = np.ones(len(next(iter(evaluations["before"].values()))), dtype=bool)
        if group != "all":
            mask = np.asarray(evaluations["groups"]) == group
        for horizon in SNAPSHOTS:
            for channel in ("position", "velocity", "omega", "success"):
                key = f"h{horizon}_{channel}"
                auxiliary.append(
                    {
                        "scenario_group": group,
                        "metric": key,
                        "before": float(evaluations["before"][key][mask].mean()),
                        "tbptt": float(evaluations["tbptt"][key][mask].mean()),
                        "multiple_shooting": float(
                            evaluations["multiple_shooting"][key][mask].mean()
                        ),
                    }
                )
    return table, auxiliary


def _gate_decision(
    table: list[dict[str, Any]],
    tbptt_summary: dict[str, Any],
    ms_summary: dict[str, Any],
) -> dict[str, Any]:
    by_name = {row["metric"]: row for row in table}
    early_channels = (
        "H500 recovery position",
        "H500 recovery omega",
    )
    early_ok = all(
        by_name[name]["multiple_shooting"]
        <= 1.02 * max(by_name[name]["tbptt"], 1.0e-12)
        for name in early_channels
    )
    # Velocity is checked from the raw auxiliary report by the caller and added below.
    tbptt_clip_fraction = float(tbptt_summary["clip_count"]) / max(
        int(tbptt_summary["completed_updates"]), 1
    )
    ms_clip_fraction = float(ms_summary["clip_count"]) / max(
        int(ms_summary["completed_updates"]), 1
    )
    checks = {
        "primary_extra_improvement_positive_ci": (
            by_name["H5000 tail position"]["ms_extra_improvement"] > 0.0
            and by_name["H5000 tail position"]["ci95_low"] > 0.0
        ),
        "h5000_cvar_same_direction": (
            by_name["H5000 CVaR20 position"]["ms_extra_improvement"] > 0.0
        ),
        "h2000_position_same_direction": (
            by_name["H2000 tail position"]["ms_extra_improvement"] > 0.0
        ),
        "h500_position_and_omega_within_2pct": early_ok,
        "h5000_success_drop_at_most_1pp": (
            by_name["H5000 success"]["multiple_shooting"]
            >= by_name["H5000 success"]["tbptt"] - 0.01
        ),
        "continuity_final_below_0p05": (
            float(ms_summary["final_continuity_rms"]) < 0.05
        ),
        "no_nan_inf_or_skip": (
            int(tbptt_summary["all_finite"]) == 1
            and int(ms_summary["all_finite"]) == 1
            and int(tbptt_summary["skip_count"]) == 0
            and int(ms_summary["skip_count"]) == 0
        ),
        "ms_clip_fraction_not_materially_higher": (
            ms_clip_fraction <= tbptt_clip_fraction + 0.05
        ),
    }
    return {
        "decision": "support" if all(checks.values()) else "not_support",
        "checks": checks,
        "tbptt_clip_fraction": tbptt_clip_fraction,
        "ms_clip_fraction": ms_clip_fraction,
    }


def _write_markdown_table(path: Path, table: list[dict[str, Any]], decision: str) -> None:
    lines = [
        "| 指标 | Before | TBPTT | MS | MS额外改善 | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            "| {metric} | {before:.6g} | {tbptt:.6g} | {multiple_shooting:.6g} | "
            "{ms_extra_improvement:+.6g} | [{ci95_low:+.6g}, {ci95_high:+.6g}] |".format(
                **row
            )
        )
    lines.extend(("", f"Gate decision: **{decision}**"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if min(args.ordinary_count, args.hard_count, args.hard_candidate_count) <= 0:
        raise ValueError("evaluation counts must be positive")
    if args.hard_candidate_count < args.hard_count:
        raise ValueError("hard candidate count must be at least hard count")
    if args.timing_updates != 2:
        raise ValueError("the locked screen uses exactly two timing updates per arm")

    started = time.perf_counter()
    deadline = started + args.total_wall_seconds
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_args = _load_train_args(args.base_config)
    if int(train_args.seed) != args.seed:
        raise ValueError(f"base config seed {train_args.seed} does not match locked seed {args.seed}")
    device = train.resolve_device(args.device)
    backend = "cuda" if device.type == "cuda" else "torch"
    sim = _make_sim(train_args)
    checkpoint_path = Path(train_args.init_checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    schedule_path = output_dir / "training_schedule.pt"
    schedule_manifest = _write_training_schedule(
        schedule_path,
        sim=sim,
        train_args=train_args,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    before_policy = _load_arm_policy(checkpoint_path, None, device=device)
    eval_path = output_dir / "eval_set.pt"
    eval_state, groups, eval_manifest = _write_eval_set(
        eval_path,
        sim=sim,
        policy=before_policy,
        train_args=train_args,
        ordinary_count=args.ordinary_count,
        hard_count=args.hard_count,
        candidate_count=args.hard_candidate_count,
        ordinary_seed=args.ordinary_seed,
        hard_seed=args.hard_seed,
        device=device,
        backend=backend,
    )
    del before_policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    timing: dict[str, dict[str, Any]] = {}
    for mode in ("tbptt", "multiple-shooting"):
        arm_dir = output_dir / "timing" / mode.replace("-", "_")
        arm_dir.mkdir(parents=True, exist_ok=True)
        timing[mode] = _run_arm(
            mode=mode,
            output_dir=arm_dir,
            schedule_path=schedule_path,
            updates=args.timing_updates,
            device=args.device,
            timeout_seconds=min(5.0 * 60.0, deadline - time.perf_counter()),
        )
    slower_update_seconds = max(
        float(timing["tbptt"]["mean_update_seconds"]),
        float(timing["multiple-shooting"]["mean_update_seconds"]),
    )
    formal_updates = max(
        1, int(math.floor(args.paired_training_seconds / (2.0 * slower_update_seconds)))
    )
    budget = {
        "timing_updates_per_arm": args.timing_updates,
        "tbptt_mean_update_seconds": timing["tbptt"]["mean_update_seconds"],
        "ms_mean_update_seconds": timing["multiple-shooting"]["mean_update_seconds"],
        "slower_update_seconds_T": slower_update_seconds,
        "formula": "floor(paired_training_seconds / (2*T))",
        "paired_training_seconds": args.paired_training_seconds,
        "formal_updates_per_arm_N": formal_updates,
        "physical_steps_per_arm": formal_updates * args.batch_size * 1000,
    }
    (output_dir / "budget.json").write_text(
        json.dumps(budget, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"timing T={slower_update_seconds:.3f}s, locked N={formal_updates}, "
        f"physical_steps/arm={budget['physical_steps_per_arm']}",
        flush=True,
    )

    formal: dict[str, dict[str, Any]] = {}
    for mode in ("tbptt", "multiple-shooting"):
        remaining = deadline - time.perf_counter() - args.eval_reserve_seconds
        if remaining <= 60.0:
            raise TimeoutError("30-minute wall budget left insufficient time for paired training")
        arm_dir = output_dir / "formal" / mode.replace("-", "_")
        arm_dir.mkdir(parents=True, exist_ok=True)
        formal[mode] = _run_arm(
            mode=mode,
            output_dir=arm_dir,
            schedule_path=schedule_path,
            updates=formal_updates,
            device=args.device,
            timeout_seconds=remaining,
        )

    if (
        formal["tbptt"]["completed_updates"] != formal_updates
        or formal["multiple-shooting"]["completed_updates"] != formal_updates
    ):
        raise RuntimeError("one arm did not complete the locked equal update budget")

    checkpoints = {
        "before": None,
        "tbptt": output_dir / "formal/tbptt/model.pt",
        "multiple_shooting": output_dir / "formal/multiple_shooting/model.pt",
    }
    evaluations: dict[str, Any] = {"groups": groups}
    all_eval_rows: list[dict[str, Any]] = []
    for label, arm_checkpoint in checkpoints.items():
        if deadline - time.perf_counter() <= 30.0:
            raise TimeoutError("30-minute wall deadline reached before unified evaluation")
        policy = _load_arm_policy(checkpoint_path, arm_checkpoint, device=device)
        metrics = native_l2f_rollout(
            policy, eval_state, train_args, horizon=5000, backend=backend
        )
        evaluations[label] = metrics
        label_rows = _evaluation_rows([label] * len(groups), groups, metrics)
        all_eval_rows.extend(label_rows)
        del policy
    _write_csv(output_dir / "paired_scenario_metrics.csv", all_eval_rows)

    table, auxiliary = _summarize_results(
        evaluations,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_csv(output_dir / "compact_table.csv", table)
    _write_csv(output_dir / "auxiliary_metrics.csv", auxiliary)
    decision = _gate_decision(
        table, formal["tbptt"], formal["multiple-shooting"]
    )
    # Lock the omitted H500 velocity component to the same 2% early-recovery rule.
    auxiliary_lookup = {
        (row["scenario_group"], row["metric"]): row for row in auxiliary
    }
    h500_velocity = auxiliary_lookup[("all", "h500_velocity")]
    velocity_ok = float(h500_velocity["multiple_shooting"]) <= 1.02 * max(
        float(h500_velocity["tbptt"]), 1.0e-12
    )
    decision["checks"]["h500_velocity_within_2pct"] = velocity_ok
    decision["decision"] = (
        "support" if all(decision["checks"].values()) else "not_support"
    )

    elapsed = time.perf_counter() - started
    report = {
        "decision": decision,
        "budget": budget,
        "schedule": schedule_manifest,
        "eval_set": eval_manifest,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training": formal,
        "timing": timing,
        "elapsed_seconds": elapsed,
        "within_30_minute_cap": elapsed <= args.total_wall_seconds,
        "pre_registered_rules": {
            "primary_ci": "H5000 tail position MS extra improvement > 0 and 95% CI low > 0",
            "cvar_direction": "H5000 position CVaR20 MS extra improvement > 0",
            "h2000_direction": "H2000 position MS extra improvement > 0",
            "early_recovery": "MS H500 position/velocity/omega <= 1.02 * TBPTT",
            "steady_success": "MS H5000 success >= TBPTT - 0.01",
            "continuity": "MS final normalized RMS < 0.05",
            "numeric": "both finite, zero skipped updates",
            "clipping": "MS clip fraction <= TBPTT clip fraction + 0.05",
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown_table(output_dir / "compact_table.md", table, decision["decision"])
    print(f"decision={decision['decision']} elapsed={elapsed:.1f}s", flush=True)
    print(f"wrote {output_dir / 'compact_table.md'}", flush=True)


if __name__ == "__main__":
    main()
