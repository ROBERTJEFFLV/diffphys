from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np

HORIZONS = (500, 2000, 5000, 10000)

def _items(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bool(row: dict[str, str], key: str) -> bool:
    return bool(int(float(row[key])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize H500 position-hold MATLAB results.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--training-root", default="")
    parser.add_argument("--baseline-samples", default="")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--control-group", required=True)
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument(
        "--legacy-failure-table",
        default=(
            "reports/formal_belief_ablation_h500_s500_gpu2/matlab_eval_streaming/"
            "failure_analysis/failure_frequency_by_sample.csv"
        ),
    )
    parser.add_argument(
        "--dynamic-hard-table",
        default=(
            "reports/formal_belief_ablation_h500_s500_gpu2/matlab_eval_streaming/"
            "failure_analysis/dynamic_hard_feasible_h500.csv"
        ),
    )
    args = parser.parse_args()
    args.groups = _items(args.groups)
    args.seeds = [int(value) for value in _items(args.seeds)]
    if args.control_group not in args.groups:
        raise ValueError("--control-group must appear in --groups")
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.eval_root).resolve()
    by_run: dict[tuple[int, str], list[dict[str, str]]] = {
        (seed, group): _rows(root / f"seed_{seed}" / f"group_{group}" / "step_500" / "samples.csv")
        for seed in args.seeds
        for group in args.groups
    }
    group_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    run_success: dict[tuple[int, str, int], np.ndarray] = {}
    for group in args.groups:
        group_row: dict[str, object] = {"group": group}
        stays: list[float] = []
        survivals: list[float] = []
        for horizon in HORIZONS:
            values: list[float] = []
            snapshots: list[float] = []
            fractions: list[float] = []
            for seed in args.seeds:
                rows = by_run[seed, group]
                success = np.array(
                    [_bool(row, f"position_hold_steady_H{horizon}") for row in rows], dtype=bool
                )
                run_success[seed, group, horizon] = success
                values.append(float(success.mean()))
                snapshots.append(float(np.mean([
                    _bool(row, f"position_hold_snapshot_H{horizon}") for row in rows
                ])))
                fractions.append(float(np.mean([
                    float(row[f"final_window_success_fraction_H{horizon}"]) for row in rows
                ])))
            group_row[f"position_hold_steady_H{horizon}_mean"] = statistics.fmean(values)
            group_row[f"position_hold_steady_H{horizon}_std"] = statistics.pstdev(values)
            group_row[f"position_hold_snapshot_H{horizon}_mean"] = statistics.fmean(snapshots)
            group_row[f"final_window_success_fraction_H{horizon}_mean"] = statistics.fmean(fractions)
        for seed in args.seeds:
            rows = by_run[seed, group]
            stays.extend(float(row["stay"]) for row in rows if row.get("stay", "") not in ("", "NaN", "nan"))
            survivals.extend(float(row["survival"]) for row in rows if row.get("survival", "") not in ("", "NaN", "nan"))
        group_row["stay_mean"] = statistics.fmean(stays) if stays else float("nan")
        group_row["survival_mean"] = statistics.fmean(survivals) if survivals else float("nan")
        group_rows.append(group_row)

    for horizon in HORIZONS:
        ordered = sorted(
            group_rows,
            key=lambda row: float(row[f"position_hold_steady_H{horizon}_mean"]),
            reverse=True,
        )
        ranking_rows.extend(
            {
                "horizon": horizon,
                "rank": rank,
                "group": row["group"],
                "position_hold_steady_mean": row[f"position_hold_steady_H{horizon}_mean"],
            }
            for rank, row in enumerate(ordered, start=1)
        )

    flip_rows: list[dict[str, object]] = []
    for group in args.groups:
        if group == args.control_group:
            continue
        for horizon in HORIZONS:
            counts = {"A_success_candidate_success": 0, "A_success_candidate_failure": 0,
                      "A_failure_candidate_success": 0, "A_failure_candidate_failure": 0}
            for seed in args.seeds:
                control = run_success[seed, args.control_group, horizon]
                candidate = run_success[seed, group, horizon]
                counts["A_success_candidate_success"] += int(np.sum(control & candidate))
                counts["A_success_candidate_failure"] += int(np.sum(control & ~candidate))
                counts["A_failure_candidate_success"] += int(np.sum(~control & candidate))
                counts["A_failure_candidate_failure"] += int(np.sum(~control & ~candidate))
            flip_rows.append({"horizon": horizon, "candidate": group, **counts})

    legacy = _rows(Path(args.legacy_failure_table))
    old_fail_ids = {int(float(row["sample_id"])) for row in legacy if int(float(row["fail_count_H500"])) == 25}
    conflict_ids = {
        int(float(row["sample_id"])) for row in legacy
        if row.get("failure_category_H500") == "always_attitude_target_infeasible"
    }
    dynamic_ids = {
        int(float(row["sample_id"]))
        for row in _rows(Path(args.dynamic_hard_table))
        if row.get("failure_category_H500", "always_dynamic_omega")
        == "always_dynamic_omega"
    }
    subsets = {"old_25_of_25_fail": old_fail_ids, "legacy_identity_conflict": conflict_ids,
               "dynamic_hard": dynamic_ids}
    subset_rows: list[dict[str, object]] = []
    for subset_name, ids in subsets.items():
        indices = np.array(sorted(sample_id - 1 for sample_id in ids), dtype=np.int64)
        for group in args.groups:
            matrix = np.stack(
                [run_success[seed, group, 500][indices] for seed in args.seeds], axis=0
            )
            scenario_any = matrix.any(axis=0)
            scenario_all = matrix.all(axis=0)
            subset_rows.append(
                {
                    "subset": subset_name,
                    "scenario_count": len(ids),
                    "group": group,
                    "model_scenario_success_rate": float(matrix.mean()),
                    "scenarios_success_any_seed": int(scenario_any.sum()),
                    "scenarios_success_all_seeds": int(scenario_all.sum()),
                }
            )
        all_models = np.stack(
            [
                run_success[seed, group, 500][indices]
                for seed in args.seeds
                for group in args.groups
            ],
            axis=0,
        )
        subset_rows.append(
            {
                "subset": subset_name,
                "scenario_count": len(ids),
                "group": "ALL_MODELS",
                "model_scenario_success_rate": float(all_models.mean()),
                "scenarios_success_any_seed": int(all_models.any(axis=0).sum()),
                "scenarios_success_all_seeds": int(all_models.all(axis=0).sum()),
            }
        )

    reference_rows = by_run[args.seeds[0], args.control_group]
    alpha_roll = np.array([float(row["alpha_roll_max"]) for row in reference_rows])
    alpha_yaw = np.array([float(row["alpha_yaw_max"]) for row in reference_rows])
    tau_fall = np.array([float(row["motor_time_falling_s"]) for row in reference_rows])
    subgroup_masks = {
        "low_alpha_roll": alpha_roll <= np.quantile(alpha_roll, 0.2),
        "low_alpha_yaw": alpha_yaw <= np.quantile(alpha_yaw, 0.2),
        "large_tau_fall": tau_fall >= np.quantile(tau_fall, 0.8),
    }
    subgroup_rows: list[dict[str, object]] = []
    for group in args.groups:
        for subgroup, mask in subgroup_masks.items():
            h500 = np.stack(
                [run_success[seed, group, 500][mask] for seed in args.seeds], axis=0
            )
            h10000 = np.stack(
                [run_success[seed, group, 10000][mask] for seed in args.seeds], axis=0
            )
            omega_failure = np.stack(
                [
                    np.array(
                        [float(row["omega_H500"]) >= 0.20 for row in by_run[seed, group]],
                        dtype=bool,
                    )[mask]
                    for seed in args.seeds
                ],
                axis=0,
            )
            subgroup_rows.append(
                {
                    "group": group,
                    "subgroup": subgroup,
                    "scenario_count": int(mask.sum()),
                    "position_hold_steady_H500": float(h500.mean()),
                    "position_hold_steady_H10000": float(h10000.mean()),
                    "omega_failure_H500": float(omega_failure.mean()),
                }
            )

    _write(root / "position_hold_group_ranking.csv", group_rows)
    _write(root / "position_hold_ranking_by_horizon.csv", ranking_rows)
    _write(root / "position_hold_paired_flips.csv", flip_rows)
    _write(root / "position_hold_subset_summary.csv", subset_rows)
    _write(root / "position_hold_dynamics_subgroups.csv", subgroup_rows)
    if args.baseline_samples:
        baseline_rows = _rows(Path(args.baseline_samples))
        baseline_flip_rows: list[dict[str, object]] = []
        for horizon in HORIZONS:
            baseline = np.array(
                [_bool(row, f"position_hold_steady_H{horizon}") for row in baseline_rows],
                dtype=bool,
            )
            for group in args.groups:
                counts = {
                    "baseline_success_candidate_success": 0,
                    "baseline_success_candidate_failure": 0,
                    "baseline_failure_candidate_success": 0,
                    "baseline_failure_candidate_failure": 0,
                }
                for seed in args.seeds:
                    candidate = run_success[seed, group, horizon]
                    counts["baseline_success_candidate_success"] += int(
                        np.sum(baseline & candidate)
                    )
                    counts["baseline_success_candidate_failure"] += int(
                        np.sum(baseline & ~candidate)
                    )
                    counts["baseline_failure_candidate_success"] += int(
                        np.sum(~baseline & candidate)
                    )
                    counts["baseline_failure_candidate_failure"] += int(
                        np.sum(~baseline & ~candidate)
                    )
                baseline_flip_rows.append(
                    {"horizon": horizon, "candidate": group, **counts}
                )
        _write(root / "position_hold_vs_initial_baseline.csv", baseline_flip_rows)
    if args.training_root:
        training_root = Path(args.training_root).resolve()
        prediction_rows: list[dict[str, object]] = []
        for group in args.groups:
            per_seed: list[dict[str, float]] = []
            for seed in args.seeds:
                rows = _rows(training_root / f"seed_{seed}" / f"group_{group}" / "train.csv")
                accepted = [row for row in rows if row.get("update_applied") == "1"]
                tail = accepted[-25:]
                per_seed.append(
                    {
                        name: statistics.fmean(float(row[name]) for row in tail)
                        for name in ("motor_aux_loss", "capability_aux_loss", "response_aux_loss")
                    }
                )
            prediction_rows.append(
                {
                    "group": group,
                    **{
                        f"{name}_last25_mean": statistics.fmean(seed_row[name] for seed_row in per_seed)
                        for name in per_seed[0]
                    },
                }
            )
        _write(root / "position_hold_prediction_losses.csv", prediction_rows)
    print(f"wrote position-hold summaries under {root}")


if __name__ == "__main__":
    main()
