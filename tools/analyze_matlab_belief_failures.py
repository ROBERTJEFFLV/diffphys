from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (7, 17, 27, 37, 47)
GROUPS = ("A", "B", "C", "D", "E")
ANGLE_LIMIT_DEG = 0.1 * 180.0 / np.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired MATLAB evaluation failures.")
    parser.add_argument(
        "--root",
        default="reports/formal_belief_ablation_h500_s500_gpu2/matlab_eval_streaming",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output = root / "failure_analysis"
    output.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(root / "baseline/step_0/samples.csv")
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        for group in GROUPS:
            frame = pd.read_csv(root / f"seed_{seed}/group_{group}/step_500/samples.csv")
            frame["training_seed"] = seed
            frame["group"] = group
            frames.append(frame)
    evaluations = pd.concat(frames, ignore_index=True)

    reason_rows: list[dict[str, object]] = []
    for horizon in (500, 10000):
        evaluations[f"fail_H{horizon}"] = evaluations[f"success_H{horizon}"] < 0.5
        tests = {
            "position": evaluations[f"position_H{horizon}_m"] >= 0.05,
            "velocity": evaluations[f"velocity_H{horizon}"] >= 0.10,
            "angle": evaluations[f"angle_H{horizon}_deg"] >= ANGLE_LIMIT_DEG,
            "omega": evaluations[f"omega_H{horizon}"] >= 0.20,
        }
        for name, values in tests.items():
            evaluations[f"{name}_fail_H{horizon}"] = values
        failed = evaluations[evaluations[f"fail_H{horizon}"]]
        for name in tests:
            reason_rows.append(
                {
                    "horizon": horizon,
                    "reason": name,
                    "failure_instances": int(failed[f"{name}_fail_H{horizon}"].sum()),
                    "fraction_of_failure_instances": float(
                        failed[f"{name}_fail_H{horizon}"].mean()
                    ),
                }
            )

        grouped = evaluations.groupby("sample_id")
        baseline[f"fail_count_H{horizon}"] = baseline["sample_id"].map(
            grouped[f"fail_H{horizon}"].sum()
        )
        for name in tests:
            baseline[f"{name}_fail_models_H{horizon}"] = baseline["sample_id"].map(
                grouped[f"{name}_fail_H{horizon}"].sum()
            )

    baseline["force_norm"] = np.sqrt(
        baseline["external_force_x"] ** 2
        + baseline["external_force_y"] ** 2
        + baseline["external_force_z"] ** 2
    )
    baseline["kxy"] = baseline["inertia_x"] / (
        baseline["mass_kg"] * baseline["arm_length_m"] ** 2
    )

    always = baseline["fail_count_H500"] == 25
    feasible = baseline["attitude_target_feasible"] > 0.5
    dynamic_omega = baseline["omega_fail_models_H500"] >= 13
    baseline["failure_category_H500"] = "never_failed"
    baseline.loc[baseline["fail_count_H500"].between(1, 12), "failure_category_H500"] = (
        "intermittent"
    )
    baseline.loc[baseline["fail_count_H500"].between(13, 24), "failure_category_H500"] = (
        "majority_failed"
    )
    baseline.loc[always & ~feasible, "failure_category_H500"] = (
        "always_attitude_target_infeasible"
    )
    baseline.loc[always & feasible & dynamic_omega, "failure_category_H500"] = (
        "always_dynamic_omega"
    )
    baseline.loc[always & feasible & ~dynamic_omega, "failure_category_H500"] = (
        "always_quasistatic_position"
    )

    baseline.to_csv(output / "failure_frequency_by_sample.csv", index=False)
    baseline[always].to_csv(output / "always_fail_h500.csv", index=False)
    baseline[always & feasible].to_csv(output / "dynamic_hard_feasible_h500.csv", index=False)
    pd.DataFrame(reason_rows).to_csv(output / "failure_reason_summary.csv", index=False)

    always_infeasible = baseline[always & ~feasible]
    always_feasible = baseline[always & feasible]
    omega_hard = baseline[always & feasible & dynamic_omega]
    static_hard = baseline[always & feasible & ~dynamic_omega]
    never = baseline[baseline["fail_count_H500"] == 0]
    report = f"""# Final step500 failure analysis

- Fixed scenarios: {len(baseline)}
- Evaluated trained models: {len(SEEDS) * len(GROUPS)}
- H500 always failed (25/25): {int(always.sum())}
- H500 failed in a majority (>=13/25): {int((baseline['fail_count_H500'] >= 13).sum())}
- H500 ever failed: {int((baseline['fail_count_H500'] > 0).sum())}
- H500 never failed: {len(never)}

## Always-failed decomposition

- Attitude target physically incompatible with required equilibrium tilt: {len(always_infeasible)}
- Attitude-feasible dynamic/omega hard vehicles: {len(omega_hard)}
- Attitude-feasible quasi-static position hard vehicles: {len(static_hard)}

The attitude-infeasible set has median required tilt
{always_infeasible['required_tilt_deg'].median():.3f} deg, while the success
angle limit is {ANGLE_LIMIT_DEG:.3f} deg. The never-failed set has median
required tilt {never['required_tilt_deg'].median():.3f} deg.

The dynamic/omega set has median alpha_roll
{omega_hard['alpha_roll_max'].median():.3f}, alpha_yaw
{omega_hard['alpha_yaw_max'].median():.3f}, and tau_fall
{omega_hard['motor_time_falling_s'].median():.4f} s.

Dynamic/omega sample IDs:

{', '.join(str(value) for value in omega_hard['sample_id'].tolist())}

Quasi-static position sample IDs:

{', '.join(str(value) for value in static_hard['sample_id'].tolist())}
"""
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(f"saved failure analysis: {output}")


if __name__ == "__main__":
    main()
