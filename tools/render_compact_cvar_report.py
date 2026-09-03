from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: str) -> str:
    return f"{100.0 * float(value):.3f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the final compact-CVaR experiment report.")
    parser.add_argument("--eval-root", default="reports/compact_cvar_ablation_h500_u2000_matlab")
    parser.add_argument("--oracle-root", default="reports/privileged_oracle_standalone_u2000_r3")
    args = parser.parse_args()
    root = Path(args.eval_root)
    summary = rows(root / "P_group_summary.csv")
    flips = rows(root / "P_baseline_paired_flips.csv")
    failures = rows(root / "P_failure_modes.csv")
    dynamic = rows(root / "P_dynamic_hard_22_summary.csv")
    oracle_path = Path(args.oracle_root) / "scenario_results.csv"
    oracle = rows(oracle_path) if oracle_path.exists() else []

    lines = [
        "# Compact input / independent CVaR formal result",
        "",
        "Fixed training seed 7, MATLAB evaluation seed 1007, 1024 identical physical-broad scenarios, "
        "H500 training and H10000 streaming validation. Main success is the final 100-step steady window.",
        "",
        "## Overall result",
        "",
        "| Group | H500 steady | H10000 steady | Settling (s) | Stay | Survival |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['group']} | {pct(row['steady_H500_mean'])} | {pct(row['steady_H10000_mean'])} "
            f"| {float(row['settling_time_mean']):.3f} | {pct(row['stay_mean'])} "
            f"| {pct(row['survival_mean'])} |"
        )

    lines += [
        "",
        "## Paired flips relative to initialization baseline",
        "",
        "| Group | Horizon | S→F | F→S | S→S | F→F |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in flips:
        lines.append(
            f"| {row['group']} | H{row['horizon']} | {row['S_to_F']} | {row['F_to_S']} "
            f"| {row['S_to_S']} | {row['F_to_F']} |"
        )

    lines += [
        "",
        "## Failure modes (rate over 1024 scenarios)",
        "",
        "| Group | Horizon | Position-only | Omega-related | 6–10 cm bias | Strict bounded angular | Loose bounded angular |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in failures:
        lines.append(
            f"| {row['group']} | H{row['horizon']} | {pct(row['position_only_rate'])} "
            f"| {pct(row['omega_related_rate'])} | {int(round(float(row['position_bias_6_10cm_count'])))} "
            f"| {pct(row['strict_bounded_angular_motion_rate'])} "
            f"| {pct(row['loose_bounded_angular_motion_rate'])} |"
        )

    lines += [
        "",
        "## Dynamic-hard 22",
        "",
        "| Group | Horizon | Steady | Position-only failures | Omega-related failures | Bounded angular |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in dynamic:
        lines.append(
            f"| {row['group']} | H{row['horizon']} | {row['steady_success_count']}/{row['count']} "
            f"| {row['position_only_failure_count']} | {row['omega_related_failure_count']} "
            f"| {row['strict_bounded_angular_motion_count']} |"
        )

    reachable = sum(row.get("status") == "oracle_reachable" for row in oracle)
    lines += [
        "",
        "## Privileged oracle",
        "",
        f"MATLAB H10000 validated oracle_reachable: **{reachable}/{len(oracle)}**. "
        "All remaining scenarios are `unproven`; optimization failure is never called physically infeasible.",
        "",
        "Raw tables: `P_group_summary.csv`, `P_baseline_paired_flips.csv`, "
        "`P_failure_modes.csv`, `P_dynamic_hard_22_summary.csv`, and `P_learning_curve.csv`.",
        "",
    ]
    output = root / "FINAL_REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
