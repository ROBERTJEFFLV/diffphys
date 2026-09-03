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


def _items(value: str, cast) -> list:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize final T control/candidate seeds.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--control-group", required=True)
    parser.add_argument("--candidate-group", required=True)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--physical-step", type=int, default=256_000_000)
    parser.add_argument(
        "--p4b-samples",
        default="reports/q_residual_h500_u2000_matlab/baseline/samples.csv",
    )
    args = parser.parse_args()
    root = Path(args.eval_root)
    seeds = _items(args.seeds, int)
    groups = (args.control_group, args.candidate_group)
    baseline = _read(root / "baseline" / "samples.csv")
    p4b_baseline = _read(Path(args.p4b_samples))
    sample_id = _num(baseline, "sample_id").astype(np.int64)
    if not np.array_equal(
        sample_id,
        _num(p4b_baseline, "sample_id").astype(np.int64),
    ):
        raise ValueError("Q2 and P4b baseline sample IDs differ")
    force = np.sqrt(
        _num(baseline, "external_force_x") ** 2
        + _num(baseline, "external_force_y") ** 2
        + _num(baseline, "external_force_z") ** 2
    )
    alpha_roll = _num(baseline, "alpha_roll_max")
    alpha_yaw = _num(baseline, "alpha_yaw_max")
    high_force = force >= np.quantile(force, 0.80)
    low_roll = alpha_roll <= np.quantile(alpha_roll, 0.20)
    low_yaw = alpha_yaw <= np.quantile(alpha_yaw, 0.20)
    dynamic_hard = np.isin(sample_id, DYNAMIC_HARD_IDS)

    run_rows: list[dict[str, object]] = []
    flip_rows: list[dict[str, object]] = []
    samples: dict[tuple[int, str], list[dict[str, str]]] = {}
    for seed in seeds:
        for group in groups:
            rows = _read(
                root
                / f"seed_{seed}"
                / f"group_{group}"
                / f"physical_steps_{args.physical_step}"
                / "samples.csv"
            )
            samples[(seed, group)] = rows
            for horizon in HORIZONS:
                suffix = f"H{horizon}"
                success = _bool(rows, f"position_hold_steady_{suffix}")
                baseline_success = _bool(baseline, f"position_hold_steady_{suffix}")
                p4b_success = _bool(p4b_baseline, f"position_hold_steady_{suffix}")
                position = _num(rows, f"position_tail_rms_{suffix}")
                velocity = _num(rows, f"velocity_tail_rms_{suffix}")
                omega = _num(rows, f"omega_tail_rms_{suffix}")
                failed = ~success
                position_bad = position >= 0.05
                velocity_bad = velocity >= 0.10
                omega_bad = omega >= 0.20
                settling = _num(rows, f"position_hold_settling_time_s_{suffix}")
                finite_settling = settling[np.isfinite(settling)]
                row: dict[str, object] = {
                    "seed": seed,
                    "group": group,
                    "horizon": horizon,
                    "steady_success": float(np.mean(success)),
                    "settling_median": float(np.median(finite_settling)),
                    "settling_p95": float(np.quantile(finite_settling, 0.95)),
                    "stay": float(np.nanmean(_num(rows, f"position_hold_stay_fraction_{suffix}"))),
                    "survival": float(np.mean(_bool(rows, f"survival_{suffix}"))),
                    "position_only_failures": int(np.sum(failed & position_bad & ~velocity_bad & ~omega_bad)),
                    "omega_related_failures": int(np.sum(failed & omega_bad)),
                    "velocity_related_failures": int(np.sum(failed & velocity_bad)),
                    "position_bias_5_6cm": int(np.sum(
                        failed & (position >= 0.05) & (position < 0.06)
                        & (velocity < 0.10) & (omega < 0.20)
                    )),
                    "position_bias_6_10cm": int(np.sum(
                        failed & (position >= 0.06) & (position < 0.10)
                        & (velocity < 0.10) & (omega < 0.20)
                    )),
                    "position_rms_ge_10cm": int(np.sum(position >= 0.10)),
                    "highest_force_20_success": float(np.mean(success[high_force])),
                    "lowest_alpha_roll_20_success": float(np.mean(success[low_roll])),
                    "lowest_alpha_yaw_20_success": float(np.mean(success[low_yaw])),
                    "dynamic_hard_22_success": float(np.mean(success[dynamic_hard])),
                    "omega_x_rms": float(np.mean(_num(rows, f"omega_x_tail_rms_{suffix}"))),
                    "omega_y_rms": float(np.mean(_num(rows, f"omega_y_tail_rms_{suffix}"))),
                    "omega_z_rms": float(np.mean(_num(rows, f"omega_z_tail_rms_{suffix}"))),
                    "action_delta_rms": float(np.mean(_num(rows, "action_delta_rms"))),
                    "control_energy": float(np.mean(_num(rows, "control_energy"))),
                    "integral_norm": float(np.mean(_num(rows, "integral_world_norm_mean"))),
                    "integral_clamp_ratio": float(np.mean(_num(rows, "integral_clamp_ratio"))),
                    "baseline_S_to_F": int(np.sum(baseline_success & ~success)),
                    "baseline_F_to_S": int(np.sum(~baseline_success & success)),
                    "p4b_S_to_F": int(np.sum(p4b_success & ~success)),
                    "p4b_F_to_S": int(np.sum(~p4b_success & success)),
                }
                run_rows.append(row)

        control_rows = samples[(seed, args.control_group)]
        candidate_rows = samples[(seed, args.candidate_group)]
        for horizon in HORIZONS:
            suffix = f"H{horizon}"
            control_success = _bool(control_rows, f"position_hold_steady_{suffix}")
            candidate_success = _bool(candidate_rows, f"position_hold_steady_{suffix}")
            flip_rows.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "control_S_to_candidate_F": int(np.sum(control_success & ~candidate_success)),
                    "control_F_to_candidate_S": int(np.sum(~control_success & candidate_success)),
                    "S_to_S": int(np.sum(control_success & candidate_success)),
                    "F_to_F": int(np.sum(~control_success & ~candidate_success)),
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    numeric_fields = [
        field
        for field in run_rows[0]
        if field not in {"seed", "group", "horizon"}
    ]
    for group in groups:
        for horizon in HORIZONS:
            selected = [
                row for row in run_rows
                if row["group"] == group and row["horizon"] == horizon
            ]
            aggregate: dict[str, object] = {"group": group, "horizon": horizon, "seeds": len(selected)}
            for field in numeric_fields:
                values = np.asarray([float(row[field]) for row in selected], dtype=np.float64)
                aggregate[f"{field}_mean"] = float(np.mean(values))
                aggregate[f"{field}_std"] = float(np.std(values))
            aggregate_rows.append(aggregate)

    output = root / f"aggregate_multiseed_physical_steps_{args.physical_step}"
    _write(output / "per_seed_metrics.csv", run_rows)
    _write(output / "aggregate_metrics.csv", aggregate_rows)
    _write(output / "paired_flips_control_candidate.csv", flip_rows)
    print(output)


if __name__ == "__main__":
    main()
