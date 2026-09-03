from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np


HORIZONS = (500, 10000)
DYNAMIC_HARD_IDS = (
    53, 69, 97, 147, 161, 237, 351, 355, 396, 467, 514,
    569, 608, 636, 650, 677, 682, 732, 768, 834, 864, 904,
)


def items(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curve(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = (
        ("loss", "total loss", "log"),
        ("threshold_tail", "unweighted threshold tail", "log"),
        ("retain_action_mse", "retain action MSE", "symlog"),
        ("motor_aux_loss", "motor prediction loss", "log"),
    )
    for group in sorted({str(row["group"]) for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        updates = np.asarray([float(row["optimizer_update"]) for row in group_rows])
        for axis, (field, title, scale) in zip(axes.flat, fields):
            values = np.asarray([float(row[field]) for row in group_rows])
            window = min(25, len(values))
            if window > 1:
                kernel = np.ones(window) / window
                values = np.convolve(values, kernel, mode="valid")
                x = updates[window - 1:]
            else:
                x = updates
            axis.plot(x, values, label=group, linewidth=1.3)
            axis.set_title(title)
            axis.set_xlabel("optimizer update")
            axis.set_yscale(scale)
            axis.grid(True, alpha=0.25)
    axes[0, 0].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def boolean(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) > 0.5 for row in rows], dtype=bool)


def numeric(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired compact-input/CVaR experiments.")
    parser.add_argument("--eval-root", default="reports/compact_cvar_ablation_h500_u2000_matlab")
    parser.add_argument("--training-root", default="reports/compact_cvar_ablation_h500_u2000_gpu2")
    parser.add_argument("--groups", default="P0,P1,P2,P3,P4a,P4b")
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--checkpoint-label", default="update_2000")
    parser.add_argument(
        "--baseline-samples",
        default="",
    )
    parser.add_argument(
        "--oracle-results",
        default="reports/privileged_oracle_standalone_u2000_r3/scenario_results.csv",
    )
    parser.add_argument(
        "--retain-root",
        default="reports/compact_cvar_ablation_h500_u2000_retain",
    )
    args = parser.parse_args()
    args.groups = items(args.groups)
    args.seeds = [int(value) for value in items(args.seeds)]
    return args


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    training_root = Path(args.training_root)
    samples = {
        (seed, group): read_rows(
            eval_root / f"seed_{seed}" / f"group_{group}" / args.checkpoint_label / "samples.csv"
        )
        for seed in args.seeds
        for group in args.groups
    }
    baseline_path = (
        Path(args.baseline_samples)
        if args.baseline_samples
        else eval_root / "baseline" / "samples.csv"
    )
    baseline_rows = read_rows(baseline_path)
    group_summary: list[dict[str, object]] = []
    flip_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for group in args.groups:
        summary: dict[str, object] = {"group": group}
        for horizon in HORIZONS:
            rates = [
                float(boolean(samples[seed, group], f"position_hold_steady_H{horizon}").mean())
                for seed in args.seeds
            ]
            summary[f"steady_H{horizon}_mean"] = statistics.fmean(rates)
            summary[f"steady_H{horizon}_std"] = statistics.pstdev(rates)
            summary[f"steady_H{horizon}_per_seed"] = ";".join(f"{value:.9f}" for value in rates)
        summary["stay_mean"] = statistics.fmean(
            float(row["stay"])
            for seed in args.seeds
            for row in samples[seed, group]
            if row.get("stay", "").lower() not in ("", "nan")
        )
        summary["survival_mean"] = statistics.fmean(
            float(row["survival"])
            for seed in args.seeds
            for row in samples[seed, group]
        )
        summary["settling_time_mean"] = statistics.fmean(
            float(row["settling_time"])
            for seed in args.seeds
            for row in samples[seed, group]
            if row.get("settling_time", "").lower() not in ("", "nan")
        )
        group_summary.append(summary)

        for horizon in HORIZONS:
            baseline = boolean(baseline_rows, f"position_hold_steady_H{horizon}")
            counts = {"S_to_S": 0, "S_to_F": 0, "F_to_S": 0, "F_to_F": 0}
            for seed in args.seeds:
                candidate = boolean(samples[seed, group], f"position_hold_steady_H{horizon}")
                counts["S_to_S"] += int(np.sum(baseline & candidate))
                counts["S_to_F"] += int(np.sum(baseline & ~candidate))
                counts["F_to_S"] += int(np.sum(~baseline & candidate))
                counts["F_to_F"] += int(np.sum(~baseline & ~candidate))
            flip_rows.append({"group": group, "horizon": horizon, **counts})

            per_seed_failures: list[dict[str, float]] = []
            for seed in args.seeds:
                rows = samples[seed, group]
                steady = boolean(rows, f"position_hold_steady_H{horizon}")
                position_rms = numeric(rows, f"position_tail_rms_H{horizon}")
                velocity_rms = numeric(rows, f"velocity_tail_rms_H{horizon}")
                omega_rms = numeric(rows, f"omega_tail_rms_H{horizon}")
                failed = ~steady
                position_only = (
                    failed
                    & (position_rms >= 0.05)
                    & (velocity_rms < 0.10)
                    & (omega_rms < 0.20)
                )
                omega_related = failed & (omega_rms >= 0.20)
                velocity_related = failed & (velocity_rms >= 0.10)
                strict_bounded = boolean(rows, f"strict_bounded_angular_motion_H{horizon}")
                loose_bounded = boolean(rows, f"loose_bounded_angular_motion_H{horizon}")
                position_bias_6_10cm = (
                    (position_rms >= 0.06)
                    & (position_rms < 0.10)
                    & (velocity_rms < 0.10)
                    & (omega_rms < 0.20)
                )
                per_seed_failures.append(
                    {
                        "failure_rate": float(failed.mean()),
                        "position_only_rate": float(position_only.mean()),
                        "omega_related_rate": float(omega_related.mean()),
                        "velocity_related_rate": float(velocity_related.mean()),
                        "strict_bounded_angular_motion_rate": float(strict_bounded.mean()),
                        "loose_bounded_angular_motion_rate": float(loose_bounded.mean()),
                        "position_bias_6_10cm_count": float(position_bias_6_10cm.sum()),
                        "position_bias_6_10cm_rate": float(position_bias_6_10cm.mean()),
                    }
                )
            failure_rows.append(
                {
                    "group": group,
                    "horizon": horizon,
                    **{
                        name: statistics.fmean(row[name] for row in per_seed_failures)
                        for name in per_seed_failures[0]
                    },
                }
            )

    learning_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for group in args.groups:
            rows = read_rows(training_root / f"seed_{seed}" / f"group_{group}" / "train.csv")
            for row in rows:
                if row.get("update_applied") != "1":
                    continue
                learning_rows.append(
                    {
                        "seed": seed,
                        "group": group,
                        "optimizer_update": int(float(row["optimizer_update"])),
                        "loss": float(row["loss"]),
                        "position_tail": float(row["position_tail"]),
                        "omega_tail": float(row["omega_tail"]),
                        "threshold_tail": float(row["threshold_tail"]),
                        "early_position_cvar": float(row["early_position_cvar"]),
                        "early_omega_cvar": float(row["early_omega_cvar"]),
                        "final_position_cvar": float(row["final_position_cvar"]),
                        "final_omega_cvar": float(row["final_omega_cvar"]),
                        "retain_action_mse": float(row["retain_action_mse"]),
                        "motor_aux_loss": float(row["motor_aux_loss"]),
                        "capability_aux_loss": float(row["capability_aux_loss"]),
                        "response_aux_loss": float(row["response_aux_loss"]),
                        "lambda_motor_aux_effective": float(row["lambda_motor_aux_effective"]),
                        "lambda_capability_aux_effective": float(
                            row["lambda_capability_aux_effective"]
                        ),
                        "lambda_response_aux_effective": float(
                            row["lambda_response_aux_effective"]
                        ),
                        "w_tail_effective": float(row["w_tail_effective"]),
                        "w_retain_effective": float(row["w_retain_effective"]),
                        "cvar_selected_fraction": float(row["cvar_selected_fraction"]),
                        "cvar_selected_alpha_roll_mean": float(
                            row["cvar_selected_alpha_roll_mean"]
                        ),
                        "cvar_selected_alpha_yaw_mean": float(
                            row["cvar_selected_alpha_yaw_mean"]
                        ),
                        "cvar_selected_tau_fall_mean": float(
                            row["cvar_selected_tau_fall_mean"]
                        ),
                        "cvar_selected_thrust_to_weight_mean": float(
                            row["cvar_selected_thrust_to_weight_mean"]
                        ),
                        "cvar_selected_force_std_mean": float(
                            row["cvar_selected_force_std_mean"]
                        ),
                        "motor_aux_encoder_grad_norm": float(
                            row["motor_aux_encoder_grad_norm"]
                        ),
                        "motor_aux_gru_grad_norm": float(row["motor_aux_gru_grad_norm"]),
                        "capability_aux_encoder_grad_norm": float(
                            row["capability_aux_encoder_grad_norm"]
                        ),
                        "capability_aux_gru_grad_norm": float(
                            row["capability_aux_gru_grad_norm"]
                        ),
                        "response_aux_encoder_grad_norm": float(
                            row["response_aux_encoder_grad_norm"]
                        ),
                        "response_aux_gru_grad_norm": float(
                            row["response_aux_gru_grad_norm"]
                        ),
                        "encoder_grad_norm": float(row["grad_norm_encoder"]),
                        "gru_grad_norm": float(row["grad_norm_gru"]),
                    }
                )

    ranking_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        ordered = sorted(group_summary, key=lambda row: float(row[f"steady_H{horizon}_mean"]), reverse=True)
        for rank, row in enumerate(ordered, 1):
            ranking_rows.append(
                {
                    "horizon": horizon,
                    "rank": rank,
                    "group": row["group"],
                    "steady_success": row[f"steady_H{horizon}_mean"],
                }
            )

    axis_rows: list[dict[str, object]] = []
    evaluated = {"baseline": baseline_rows}
    evaluated.update({group: samples[args.seeds[0], group] for group in args.groups})
    for group, rows in evaluated.items():
        for horizon in HORIZONS:
            axis_row: dict[str, object] = {"group": group, "horizon": horizon}
            for axis in ("x", "y", "z"):
                for statistic in ("rms", "max", "spectral_peak_hz"):
                    field = f"omega_{axis}_tail_{statistic}_H{horizon}"
                    axis_row[f"omega_{axis}_tail_{statistic}_mean"] = float(
                        numeric(rows, field).mean()
                    )
            for action in range(4):
                for statistic in ("rms", "delta_tail_rms"):
                    suffix = "tail_rms" if statistic == "rms" else statistic
                    field = f"action_{action}_{suffix}_H{horizon}"
                    axis_row[f"action_{action}_{suffix}_mean"] = float(
                        numeric(rows, field).mean()
                    )
            axis_rows.append(axis_row)

    oracle_path = Path(args.oracle_results)
    oracle_by_id = {
        int(float(row["scenario_id"])): row
        for row in read_rows(oracle_path)
    } if oracle_path.exists() else {}
    dynamic_rows: list[dict[str, object]] = []
    dynamic_summary_rows: list[dict[str, object]] = []
    for group, rows in evaluated.items():
        by_id = {int(float(row["sample_id"])): row for row in rows}
        selected = [by_id[scenario_id] for scenario_id in DYNAMIC_HARD_IDS]
        for row in selected:
            scenario_id = int(float(row["sample_id"]))
            output: dict[str, object] = {
                "group": group,
                "scenario_id": scenario_id,
                "oracle_status": oracle_by_id.get(scenario_id, {}).get("status", "not_run"),
            }
            for horizon in HORIZONS:
                for field in (
                    "position_hold_steady",
                    "final_window_success_fraction",
                    "survival",
                    "position_tail_rms",
                    "velocity_tail_rms",
                    "omega_tail_rms",
                    "strict_bounded_angular_motion",
                    "loose_bounded_angular_motion",
                ):
                    source = f"{field}_H{horizon}"
                    output[source] = row[source]
            dynamic_rows.append(output)
        for horizon in HORIZONS:
            steady = boolean(selected, f"position_hold_steady_H{horizon}")
            position_rms = numeric(selected, f"position_tail_rms_H{horizon}")
            velocity_rms = numeric(selected, f"velocity_tail_rms_H{horizon}")
            omega_rms = numeric(selected, f"omega_tail_rms_H{horizon}")
            dynamic_summary_rows.append(
                {
                    "group": group,
                    "horizon": horizon,
                    "count": len(selected),
                    "steady_success_count": int(steady.sum()),
                    "steady_success_rate": float(steady.mean()),
                    "position_only_failure_count": int(
                        np.sum(
                            ~steady & (position_rms >= 0.05)
                            & (velocity_rms < 0.10) & (omega_rms < 0.20)
                        )
                    ),
                    "position_bias_6_10cm_count": int(
                        np.sum(
                            (position_rms >= 0.06) & (position_rms < 0.10)
                            & (velocity_rms < 0.10) & (omega_rms < 0.20)
                        )
                    ),
                    "omega_related_failure_count": int(np.sum(~steady & (omega_rms >= 0.20))),
                    "strict_bounded_angular_motion_count": int(
                        boolean(selected, f"strict_bounded_angular_motion_H{horizon}").sum()
                    ),
                    "loose_bounded_angular_motion_count": int(
                        boolean(selected, f"loose_bounded_angular_motion_H{horizon}").sum()
                    ),
                    "survival_count": int(boolean(selected, f"survival_H{horizon}").sum()),
                }
            )

    retain_rows: list[dict[str, object]] = []
    retain_root = Path(args.retain_root)
    for seed in args.seeds:
        for group in args.groups:
            path = retain_root / f"seed_{seed}" / f"group_{group}" / "summary.csv"
            if not path.exists():
                continue
            for row in read_rows(path):
                retain_rows.append({"seed": seed, "group": group, **row})
    write_rows(eval_root / "P_group_summary.csv", group_summary)
    write_rows(eval_root / "P_ranking.csv", ranking_rows)
    write_rows(eval_root / "P_baseline_paired_flips.csv", flip_rows)
    write_rows(eval_root / "P_failure_modes.csv", failure_rows)
    write_rows(eval_root / "P_axis_diagnostics.csv", axis_rows)
    write_rows(eval_root / "P_dynamic_hard_22.csv", dynamic_rows)
    write_rows(eval_root / "P_dynamic_hard_22_summary.csv", dynamic_summary_rows)
    write_rows(eval_root / "P_retain_bank_summary.csv", retain_rows)
    write_rows(eval_root / "P_learning_curve.csv", learning_rows)
    plot_learning_curve(eval_root / "P_learning_curve.png", learning_rows)
    print(f"wrote P0-P4b summaries under {eval_root.resolve()}")


if __name__ == "__main__":
    main()
