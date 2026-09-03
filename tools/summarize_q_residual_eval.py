from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


HORIZONS = (500, 1000, 2000, 5000, 10000)
DYNAMIC_HARD_IDS = np.asarray(
    (53, 69, 97, 147, 161, 237, 351, 355, 396, 467, 514,
     569, 608, 636, 650, 677, 682, 732, 768, 834, 864, 904),
    dtype=np.int64,
)


def _items(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _num(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def _bool(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return _num(rows, field) > 0.5


def _rate(mask: np.ndarray, subgroup: np.ndarray | None = None) -> float:
    values = mask if subgroup is None else mask[subgroup]
    return float(np.mean(values)) if values.size else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the paired Q0/Q1/Q2 MATLAB evaluation.")
    parser.add_argument("--eval-root", default="reports/q_residual_h500_u2000_matlab")
    parser.add_argument("--training-root", default="reports/q_residual_h500_u2000_gpu2")
    parser.add_argument("--groups", default="Q0,Q1,Q2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-label", default="update_2000")
    parser.add_argument("--output-subdir", default="aggregate")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--secondary-baseline-samples", default="")
    parser.add_argument("--secondary-baseline-label", default="secondary_baseline")
    args = parser.parse_args()
    args.groups = _items(args.groups)
    return args


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    training_root = Path(args.training_root)
    baseline_rows = _read(eval_root / "baseline" / "samples.csv")
    secondary_baseline_rows = (
        _read(Path(args.secondary_baseline_samples))
        if args.secondary_baseline_samples
        else None
    )
    rows_by_group = {
        group: _read(
            eval_root / f"seed_{args.seed}" / f"group_{group}"
            / args.checkpoint_label / "samples.csv"
        )
        for group in args.groups
    }
    sample_id = _num(baseline_rows, "sample_id").astype(np.int64)
    if secondary_baseline_rows is not None:
        secondary_ids = _num(secondary_baseline_rows, "sample_id").astype(np.int64)
        if not np.array_equal(sample_id, secondary_ids):
            raise ValueError("primary and secondary baseline sample IDs differ")
    force = np.sqrt(
        _num(baseline_rows, "external_force_x") ** 2
        + _num(baseline_rows, "external_force_y") ** 2
        + _num(baseline_rows, "external_force_z") ** 2
    )
    alpha_roll = _num(baseline_rows, "alpha_roll_max")
    alpha_yaw = _num(baseline_rows, "alpha_yaw_max")
    high_force = force >= np.quantile(force, 0.80)
    low_roll = alpha_roll <= np.quantile(alpha_roll, 0.20)
    low_yaw = alpha_yaw <= np.quantile(alpha_yaw, 0.20)
    dynamic_hard = np.isin(sample_id, DYNAMIC_HARD_IDS)

    summary_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    flip_rows: list[dict[str, object]] = []
    secondary_flip_rows: list[dict[str, object]] = []
    for group, rows in rows_by_group.items():
        group_summary: dict[str, object] = {
            "group": group,
            "control_energy_mean": float(np.mean(_num(rows, "control_energy"))),
            "action_delta_rms_mean": float(np.mean(_num(rows, "action_delta_rms"))),
            "integral_residual_action_rms_mean": float(np.mean(_num(rows, "integral_residual_action_rms"))),
            "damping_residual_action_rms_mean": float(np.mean(_num(rows, "damping_residual_action_rms"))),
            "integral_clamp_ratio_mean": float(np.mean(_num(rows, "integral_clamp_ratio"))),
            "integral_world_norm_mean": float(np.mean(_num(rows, "integral_world_norm_mean"))),
            "high_force_integral_residual_action_rms": float(np.mean(
                _num(rows, "integral_residual_action_rms")[high_force]
            )),
            "low_roll_damping_residual_action_rms": float(np.mean(
                _num(rows, "damping_residual_action_rms")[low_roll]
            )),
            "low_yaw_damping_residual_action_rms": float(np.mean(
                _num(rows, "damping_residual_action_rms")[low_yaw]
            )),
        }
        for motor_index in range(4):
            group_summary[f"steady_motor_bias_{motor_index}_mean"] = float(
                np.mean(_num(rows, f"steady_motor_bias_{motor_index}"))
            )
        for horizon in HORIZONS:
            suffix = f"H{horizon}"
            success = _bool(rows, f"position_hold_steady_{suffix}")
            baseline_success = _bool(baseline_rows, f"position_hold_steady_{suffix}")
            secondary_baseline_success = (
                _bool(secondary_baseline_rows, f"position_hold_steady_{suffix}")
                if secondary_baseline_rows is not None
                else None
            )
            position = _num(rows, f"position_tail_rms_{suffix}")
            velocity = _num(rows, f"velocity_tail_rms_{suffix}")
            omega = _num(rows, f"omega_tail_rms_{suffix}")
            failed = ~success
            survival = _bool(rows, f"survival_{suffix}")
            position_bad = position >= 0.05
            velocity_bad = velocity >= 0.10
            omega_bad = omega >= 0.20
            rms_bad_count = position_bad.astype(np.int8) + velocity_bad.astype(np.int8) + omega_bad.astype(np.int8)
            group_summary[f"steady_{suffix}"] = _rate(success)
            group_summary[f"settling_time_{suffix}_mean"] = float(
                np.nanmean(_num(rows, f"position_hold_settling_time_s_{suffix}"))
            )
            settling = _num(rows, f"position_hold_settling_time_s_{suffix}")
            finite_settling = settling[np.isfinite(settling)]
            group_summary[f"settling_time_{suffix}_median"] = (
                float(np.median(finite_settling)) if finite_settling.size else float("nan")
            )
            group_summary[f"settling_time_{suffix}_p95"] = (
                float(np.quantile(finite_settling, 0.95)) if finite_settling.size else float("nan")
            )
            group_summary[f"stay_{suffix}_mean"] = float(
                np.nanmean(_num(rows, f"position_hold_stay_fraction_{suffix}"))
            )
            group_summary[f"survival_{suffix}"] = _rate(
                survival
            )
            group_summary[f"dynamic_hard_22_success_{suffix}"] = _rate(success, dynamic_hard)
            group_summary[f"highest_force_20_success_{suffix}"] = _rate(success, high_force)
            group_summary[f"lowest_alpha_roll_20_success_{suffix}"] = _rate(success, low_roll)
            group_summary[f"lowest_alpha_yaw_20_success_{suffix}"] = _rate(success, low_yaw)
            group_summary[f"highest_force_20_position_rms_{suffix}"] = float(
                np.mean(position[high_force])
            )
            omega_x = _num(rows, f"omega_x_tail_rms_{suffix}")
            omega_y = _num(rows, f"omega_y_tail_rms_{suffix}")
            omega_z = _num(rows, f"omega_z_tail_rms_{suffix}")
            group_summary[f"lowest_alpha_roll_20_omega_xy_rms_{suffix}"] = float(
                np.mean(np.sqrt(omega_x[low_roll] ** 2 + omega_y[low_roll] ** 2))
            )
            group_summary[f"lowest_alpha_yaw_20_omega_z_rms_{suffix}"] = float(
                np.mean(omega_z[low_yaw])
            )
            group_summary[f"omega_x_rms_{suffix}"] = float(np.mean(omega_x))
            group_summary[f"omega_y_rms_{suffix}"] = float(np.mean(omega_y))
            group_summary[f"omega_z_rms_{suffix}"] = float(np.mean(omega_z))
            failure_rows.append(
                {
                    "group": group,
                    "horizon": horizon,
                    "failures": int(np.sum(failed)),
                    "position_only_failures": int(np.sum(failed & position_bad & ~velocity_bad & ~omega_bad)),
                    "omega_related_failures": int(np.sum(failed & omega_bad)),
                    "velocity_related_failures": int(np.sum(failed & velocity_bad)),
                    "multiple_metric_failures": int(np.sum(failed & (rms_bad_count >= 2))),
                    "window_only_failures": int(np.sum(failed & (rms_bad_count == 0) & survival)),
                    "survival_failures": int(np.sum(~survival)),
                    "position_bias_5_6cm": int(np.sum(
                        failed & (position >= 0.05) & (position < 0.06)
                        & (velocity < 0.10) & (omega < 0.20)
                    )),
                    "position_bias_6_10cm": int(np.sum(
                        failed & (position >= 0.06) & (position < 0.10)
                        & (velocity < 0.10) & (omega < 0.20)
                    )),
                    "position_rms_ge_10cm": int(np.sum(position >= 0.10)),
                    "bounded_angular_motion_strict": int(np.sum(_bool(rows, f"strict_bounded_angular_motion_{suffix}"))),
                    "bounded_angular_motion_loose": int(np.sum(_bool(rows, f"loose_bounded_angular_motion_{suffix}"))),
                    "omega_x_rms_mean": float(np.mean(omega_x)),
                    "omega_y_rms_mean": float(np.mean(omega_y)),
                    "omega_z_rms_mean": float(np.mean(omega_z)),
                }
            )
            flip_rows.append(
                {
                    "group": group,
                    "horizon": horizon,
                    "baseline_S_to_F": int(np.sum(baseline_success & ~success)),
                    "baseline_F_to_S": int(np.sum(~baseline_success & success)),
                    "S_to_S": int(np.sum(baseline_success & success)),
                    "F_to_F": int(np.sum(~baseline_success & ~success)),
                }
            )
            if secondary_baseline_success is not None:
                secondary_flip_rows.append(
                    {
                        "group": group,
                        "horizon": horizon,
                        "baseline_S_to_F": int(np.sum(secondary_baseline_success & ~success)),
                        "baseline_F_to_S": int(np.sum(~secondary_baseline_success & success)),
                        "S_to_S": int(np.sum(secondary_baseline_success & success)),
                        "F_to_F": int(np.sum(~secondary_baseline_success & ~success)),
                    }
                )
        summary_rows.append(group_summary)

    learning_rows: list[dict[str, object]] = []
    for group in args.groups:
        rows = _read(training_root / f"seed_{args.seed}" / f"group_{group}" / "train.csv")
        for row in rows:
            if row.get("update_applied") != "1":
                continue
            learning_rows.append(
                {
                    "group": group,
                    "optimizer_update": int(float(row["optimizer_update"])),
                    "physical_steps": int(float(row.get("physical_steps", 0))),
                    "episode_target_steps": int(float(row.get("episode_target_steps", 0))),
                    "loss": float(row["loss"]),
                    "position_cvar": float(row["position_tail"]),
                    "omega_cvar": float(row["omega_tail"]),
                    "omega_decay_loss": float(row["omega_decay_loss"]),
                    "integral_residual_action_rms": float(row["integral_residual_action_rms"]),
                    "damping_residual_action_rms": float(row["damping_residual_action_rms"]),
                    "integral_clamp_ratio": float(row["integral_clamp_ratio"]),
                    "high_force_position_tail_rms": float(row["high_force_position_tail_rms"]),
                }
            )

    output = eval_root / args.output_subdir
    _write(output / "group_summary.csv", summary_rows)
    _write(output / "failure_classification.csv", failure_rows)
    _write(output / f"paired_flips_vs_{args.baseline_label}.csv", flip_rows)
    if secondary_flip_rows:
        _write(
            output / f"paired_flips_vs_{args.secondary_baseline_label}.csv",
            secondary_flip_rows,
        )
    _write(output / "learning_curve.csv", learning_rows)
    print(output)


if __name__ == "__main__":
    main()
