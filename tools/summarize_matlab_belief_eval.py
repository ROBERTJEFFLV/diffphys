from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np


GROUPS = ("A", "B", "C", "D", "E")
SEEDS = (7, 17, 27, 37, 47)
SUMMARY_METRICS = (
    "success_rate_H500",
    "success_rate_H10000",
    "H500_success_to_H10000_survival_rate",
    "H500_success_to_H10000_stay_success_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final MATLAB A-E evaluations.")
    parser.add_argument(
        "--root",
        default="reports/formal_belief_ablation_h500_s500_gpu2",
    )
    return parser.parse_args()


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def read_many(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    eval_root = root / "matlab_eval_streaming"
    summaries = {
        (seed, group): read_one(
            eval_root / f"seed_{seed}" / f"group_{group}" / "step_500/summary.csv"
        )
        for seed in SEEDS
        for group in GROUPS
    }

    group_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for group in GROUPS:
        row: dict[str, object] = {"group": group, "training_seed_count": len(SEEDS)}
        for metric in SUMMARY_METRICS:
            values = [float(summaries[seed, group][metric]) for seed in SEEDS]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.pstdev(values)
        group_rows.append(row)
        if group == "A":
            continue
        for metric in SUMMARY_METRICS:
            deltas = [
                float(summaries[seed, group][metric])
                - float(summaries[seed, "A"][metric])
                for seed in SEEDS
            ]
            paired_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "mean_delta_vs_A": statistics.fmean(deltas),
                    "std_delta_vs_A": statistics.pstdev(deltas),
                    "positive_seed_count": sum(delta > 0.0 for delta in deltas),
                    "zero_seed_count": sum(delta == 0.0 for delta in deltas),
                    "negative_seed_count": sum(delta < 0.0 for delta in deltas),
                }
            )

    baseline = read_many(eval_root / "baseline/step_0/samples.csv")
    alpha_roll = np.array([float(row["alpha_roll_max"]) for row in baseline])
    alpha_yaw = np.array([float(row["alpha_yaw_max"]) for row in baseline])
    tau_fall = np.array([float(row["motor_time_falling_s"]) for row in baseline])
    baseline_h500 = np.array([bool(int(float(row["success_H500"]))) for row in baseline])
    baseline_h10000 = np.array([bool(int(float(row["success_H10000"]))) for row in baseline])
    masks = {
        "low_alpha_roll": alpha_roll <= np.quantile(alpha_roll, 0.2),
        "low_alpha_yaw": alpha_yaw <= np.quantile(alpha_yaw, 0.2),
        "large_tau_fall": tau_fall >= np.quantile(tau_fall, 0.8),
    }

    per_seed_samples: dict[tuple[int, str], dict[str, float]] = {}
    subgroup_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for group in GROUPS:
            samples = read_many(
                eval_root / f"seed_{seed}" / f"group_{group}" / "step_500/samples.csv"
            )
            success_h500 = np.array(
                [bool(int(float(row["success_H500"]))) for row in samples]
            )
            success_h10000 = np.array(
                [bool(int(float(row["success_H10000"]))) for row in samples]
            )
            omega_h500 = np.array([float(row["omega_H500"]) for row in samples])
            values = {
                "low_alpha_roll_success_rate": float(success_h500[masks["low_alpha_roll"]].mean()),
                "low_alpha_yaw_success_rate": float(success_h500[masks["low_alpha_yaw"]].mean()),
                "large_tau_fall_success_rate": float(success_h500[masks["large_tau_fall"]].mean()),
                "omega_failure_rate_H500": float((omega_h500 >= 0.2).mean()),
                "baseline_retain_rate_H500": float(success_h500[baseline_h500].mean()),
                "baseline_retain_rate_H10000": float(success_h10000[baseline_h10000].mean()),
            }
            per_seed_samples[seed, group] = values

    sample_metrics = tuple(next(iter(per_seed_samples.values())).keys())
    for group in GROUPS:
        for metric in sample_metrics:
            values = [per_seed_samples[seed, group][metric] for seed in SEEDS]
            deltas = [
                per_seed_samples[seed, group][metric]
                - per_seed_samples[seed, "A"][metric]
                for seed in SEEDS
            ]
            lower_is_better = metric == "omega_failure_rate_H500"
            improved = [delta < 0.0 if lower_is_better else delta > 0.0 for delta in deltas]
            subgroup_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values),
                    "mean_delta_vs_A": statistics.fmean(deltas),
                    "improved_seed_count": sum(improved),
                }
            )

    prediction_rows: list[dict[str, object]] = []
    for group in GROUPS:
        by_seed: list[dict[str, float]] = []
        for seed in SEEDS:
            rows = [
                row
                for row in read_many(root / f"seed_{seed}/group_{group}/train.csv")
                if row["update_applied"] == "1"
            ][-25:]
            by_seed.append(
                {
                    name: statistics.fmean(float(row[name]) for row in rows)
                    for name in (
                        "motor_aux_loss",
                        "capability_aux_loss",
                        "response_aux_loss",
                    )
                }
            )
        prediction_rows.append(
            {
                "group": group,
                **{
                    f"{name}_last25_mean": statistics.fmean(row[name] for row in by_seed)
                    for name in by_seed[0]
                },
            }
        )

    write_rows(eval_root / "group_summary.csv", group_rows)
    write_rows(eval_root / "paired_vs_A.csv", paired_rows)
    write_rows(eval_root / "subgroup_and_retain.csv", subgroup_rows)
    write_rows(eval_root / "prediction_losses.csv", prediction_rows)
    print(f"wrote summaries under {eval_root}")


if __name__ == "__main__":
    main()
