from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.phase1_fast_common import (  # noqa: E402
    artifact_fingerprint,
    atomic_write_dataframe,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


SEEDS = (7, 17, 27)
ARMS = ("A", "B", "C")
HORIZONS = (500, 10_000)
COMPARISONS = (("C", "A"), ("C", "B"))
REJECT_REASONS = {
    "adaptive_gate_suspicious",
    "adaptive_gate_hard_grad",
    "post_update_rejected",
}
GRADIENT_SKIP_REASONS = {
    "grad_skip_threshold",
    "grad_norm_nonfinite",
    "grad_tensor_nonfinite",
}


def _bootstrap_intervals(
    difference: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return scenario-block and crossed seed+scenario percentile intervals."""

    if difference.ndim != 2:
        raise ValueError("difference must have shape [seed, scenario]")
    seed_count, scenario_count = difference.shape
    rng = np.random.default_rng(seed)
    scenario_estimates = np.empty(n_bootstrap, dtype=np.float64)
    crossed_estimates = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 250
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        count = stop - start
        scenario_indices = rng.integers(0, scenario_count, size=(count, scenario_count))
        scenario_estimates[start:stop] = difference[:, scenario_indices].mean(axis=(0, 2))
        seed_indices = rng.integers(0, seed_count, size=(count, seed_count))
        crossed = difference[
            seed_indices[:, :, None],
            scenario_indices[:, None, :],
        ]
        crossed_estimates[start:stop] = crossed.mean(axis=(1, 2))
    quantiles = (0.025, 0.975)
    scenario_ci = tuple(float(value) for value in np.quantile(scenario_estimates, quantiles))
    crossed_ci = tuple(float(value) for value in np.quantile(crossed_estimates, quantiles))
    return scenario_ci, crossed_ci


def _inertia_feasible_mask(scenarios: pd.DataFrame) -> pd.Series:
    x = scenarios["inertia_x"].to_numpy(dtype=np.float64)
    y = scenarios["inertia_y"].to_numpy(dtype=np.float64)
    z = scenarios["inertia_z"].to_numpy(dtype=np.float64)
    tolerance = 1.0e-12
    feasible = (
        (x <= y + z + tolerance)
        & (y <= x + z + tolerance)
        & (z <= x + y + tolerance)
    )
    return pd.Series(feasible, index=scenarios["scenario_uid"].astype(str))


def _load_metrics(root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    inputs: list[Path] = []
    for seed in SEEDS:
        for arm in ARMS:
            path = root / f"seed_{seed}" / f"arm_{arm}" / "scenario_metrics.csv"
            provenance = path.with_name("RUN_PROVENANCE.json")
            if not path.is_file() or not provenance.is_file():
                raise FileNotFoundError(path if not path.is_file() else provenance)
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            if payload.get("outputs", {}).get(path.name) != sha256_file(path):
                raise RuntimeError(f"output hash mismatch: {path}")
            frame = pd.read_csv(path)
            if set(frame["label_status"].unique()) != {
                "corrected_internal_not_yet_matlab_confirmed"
            }:
                raise RuntimeError(f"unexpected label status: {path}")
            frames.append(frame)
            inputs.extend((path, provenance))
    metrics = pd.concat(frames, ignore_index=True)
    expected_rows = len(SEEDS) * len(ARMS) * len(HORIZONS) * 1024
    if len(metrics) != expected_rows:
        raise RuntimeError(f"metric rows={len(metrics)}, expected={expected_rows}")
    return metrics, inputs


def _apply_formal_matlab_labels(
    metrics: pd.DataFrame,
    *,
    matlab_root: Path,
    scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Path], int, dict[str, float]]:
    """Replace internal success labels with strictly paired formal MATLAB labels."""

    mappings = {
        "mass": "mass_kg",
        "cbrt_mass": "cbrt_mass",
        "arm_length": "arm_length_m",
        "thrust_to_weight": "thrust_to_weight",
        "torque_to_inertia": "torque_to_inertia",
        "motor_time_rising": "motor_time_rising_s",
        "motor_time_falling": "motor_time_falling_s",
        "force_std": "force_std",
        "inertia_x": "inertia_x",
        "inertia_y": "inertia_y",
        "inertia_z": "inertia_z",
        "alpha_roll_max": "alpha_roll_max",
        "alpha_yaw_max": "alpha_yaw_max",
        "eta_yaw": "eta_yaw",
        "jz_over_jxy": "jz_over_jxy",
        "external_force_0": "external_force_x",
        "external_force_1": "external_force_y",
        "external_force_2": "external_force_z",
    }
    expected_ids = scenarios["scenario_id"].to_numpy(dtype=np.int64)
    uid_by_id = scenarios.set_index("scenario_id")["scenario_uid"].astype(str)
    formal_frames: list[pd.DataFrame] = []
    inputs: list[Path] = []
    maximum_errors = {source: 0.0 for source in mappings}
    for seed in SEEDS:
        for arm in ARMS:
            path = matlab_root / f"seed_{seed}" / f"arm_{arm}" / "samples.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            samples = pd.read_csv(path)
            sample_ids = samples["sample_id"].to_numpy(dtype=np.int64)
            if not np.array_equal(sample_ids, expected_ids):
                raise RuntimeError(f"sample IDs/order do not match manifest: {path}")
            for source, target in mappings.items():
                error = float(
                    np.max(
                        np.abs(
                            scenarios[source].to_numpy(dtype=np.float64)
                            - samples[target].to_numpy(dtype=np.float64)
                        )
                    )
                )
                maximum_errors[source] = max(maximum_errors[source], error)
                if error > 1.0e-12:
                    raise RuntimeError(
                        f"MATLAB reset mismatch for {source}: {error:.3e} in {path}"
                    )
            for horizon in HORIZONS:
                column = f"position_hold_steady_H{horizon}"
                if column not in samples.columns:
                    if horizon == 10_000 and "position_hold_steady" in samples.columns:
                        column = "position_hold_steady"
                    else:
                        raise RuntimeError(f"missing {column}: {path}")
                formal_frames.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "training_group": arm,
                            "horizon": horizon,
                            "scenario_uid": [uid_by_id.loc[value] for value in sample_ids],
                            "formal_success": samples[column].to_numpy(dtype=np.int64),
                        }
                    )
                )
            inputs.append(path)

    formal = pd.concat(formal_frames, ignore_index=True)
    keys = ["seed", "training_group", "horizon", "scenario_uid"]
    merged = metrics.merge(formal, on=keys, how="left", validate="one_to_one")
    if merged["formal_success"].isna().any():
        raise RuntimeError("formal MATLAB labels did not cover every internal metric row")
    mismatch_count = int(
        (
            merged["steady_success"].to_numpy(dtype=np.int64)
            != merged["formal_success"].to_numpy(dtype=np.int64)
        ).sum()
    )
    merged["steady_success"] = merged.pop("formal_success").astype(np.int64)
    merged["label_status"] = "formal_matlab_confirmed"
    return merged, inputs, mismatch_count, maximum_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=ROOT / "reports" / "continuity_cadence_internal_eval_20260804",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=ROOT / "reports" / "continuity_cadence_9p6m_20260804",
    )
    parser.add_argument(
        "--matlab-root",
        type=Path,
        help="Use formally evaluated MATLAB success labels from this directory.",
    )
    parser.add_argument(
        "--scenario-csv",
        type=Path,
        default=(
            ROOT
            / "diagnostic_inputs"
            / "h10000_paired_96m_20260804"
            / "manifests"
            / "SCENARIO_MANIFEST.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "continuity_cadence_analysis_20260804",
    )
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    args = parser.parse_args()

    scenarios = pd.read_csv(args.scenario_csv)
    if scenarios["scenario_uid"].nunique() != 1024:
        raise RuntimeError("scenario manifest must contain 1024 unique UIDs")
    metrics, inputs = _load_metrics(args.eval_root.resolve())
    label_status = "corrected_internal_not_yet_matlab_confirmed"
    matlab_label_mismatch_count: int | None = None
    matlab_reset_max_errors: dict[str, float] | None = None
    if args.matlab_root is not None:
        metrics, matlab_inputs, matlab_label_mismatch_count, matlab_reset_max_errors = (
            _apply_formal_matlab_labels(
                metrics,
                matlab_root=args.matlab_root.resolve(),
                scenarios=scenarios,
            )
        )
        inputs.extend(matlab_inputs)
        label_status = "formal_matlab_confirmed"
    feasible = _inertia_feasible_mask(scenarios)
    if int(feasible.sum()) != 400:
        raise RuntimeError(f"inertia-feasible count={int(feasible.sum())}, expected=400")

    performance = (
        metrics.groupby(["seed", "training_group", "horizon"], as_index=False)
        .agg(
            success_rate=("steady_success", "mean"),
            survival_rate=("survival", "mean"),
            action_saturation_fraction=("action_saturation_fraction", "mean"),
            position_tail_rms=("position_tail_rms", "mean"),
            velocity_tail_rms=("velocity_tail_rms", "mean"),
            omega_tail_rms=("omega_tail_rms", "mean"),
        )
    )

    pair_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    difference_cache: dict[tuple[str, str, int, str], np.ndarray] = {}
    for horizon in HORIZONS:
        horizon_frame = metrics[metrics["horizon"].eq(horizon)]
        success_pivot = horizon_frame.pivot(
            index="scenario_uid",
            columns=["seed", "training_group"],
            values="steady_success",
        ).sort_index()
        saturation_pivot = horizon_frame.pivot(
            index="scenario_uid",
            columns=["seed", "training_group"],
            values="action_saturation_fraction",
        ).reindex(success_pivot.index)
        feasible_for_pivot = feasible.reindex(success_pivot.index).to_numpy(dtype=bool)
        for left, right in COMPARISONS:
            difference = np.stack(
                [
                    success_pivot[(seed, left)].to_numpy(dtype=np.float64)
                    - success_pivot[(seed, right)].to_numpy(dtype=np.float64)
                    for seed in SEEDS
                ]
            )
            saturation_difference = np.stack(
                [
                    saturation_pivot[(seed, left)].to_numpy(dtype=np.float64)
                    - saturation_pivot[(seed, right)].to_numpy(dtype=np.float64)
                    for seed in SEEDS
                ]
            )
            for stratum, mask in (
                ("all", np.ones(difference.shape[1], dtype=bool)),
                ("inertia_feasible", feasible_for_pivot),
            ):
                selected = difference[:, mask]
                scenario_ci, crossed_ci = _bootstrap_intervals(
                    selected,
                    n_bootstrap=args.bootstrap,
                    seed=args.bootstrap_seed + horizon + (0 if right == "A" else 1),
                )
                point = float(selected.mean())
                difference_cache[(left, right, horizon, stratum)] = selected
                pair_rows.append(
                    {
                        "left_arm": left,
                        "right_arm": right,
                        "horizon": horizon,
                        "stratum": stratum,
                        "scenario_count": int(mask.sum()),
                        "point_difference": point,
                        "point_difference_pp": 100.0 * point,
                        "scenario_ci_low": scenario_ci[0],
                        "scenario_ci_high": scenario_ci[1],
                        "scenario_ci_low_pp": 100.0 * scenario_ci[0],
                        "scenario_ci_high_pp": 100.0 * scenario_ci[1],
                        "crossed_ci_low": crossed_ci[0],
                        "crossed_ci_high": crossed_ci[1],
                        "crossed_ci_low_pp": 100.0 * crossed_ci[0],
                        "crossed_ci_high_pp": 100.0 * crossed_ci[1],
                        "positive_seed_count": int((selected.mean(axis=1) > 0.0).sum()),
                        "minimum_seed_difference": float(selected.mean(axis=1).min()),
                        "action_saturation_difference": float(
                            saturation_difference[:, mask].mean()
                        ),
                    }
                )
                for seed_index, seed in enumerate(SEEDS):
                    seed_rows.append(
                        {
                            "seed": seed,
                            "left_arm": left,
                            "right_arm": right,
                            "horizon": horizon,
                            "stratum": stratum,
                            "scenario_count": int(mask.sum()),
                            "difference": float(selected[seed_index].mean()),
                            "difference_pp": float(100.0 * selected[seed_index].mean()),
                        }
                    )

    pair_effects = pd.DataFrame(pair_rows)
    seed_effects = pd.DataFrame(seed_rows)

    def effect(left: str, right: str, horizon: int, stratum: str = "all") -> pd.Series:
        match = pair_effects[
            pair_effects["left_arm"].eq(left)
            & pair_effects["right_arm"].eq(right)
            & pair_effects["horizon"].eq(horizon)
            & pair_effects["stratum"].eq(stratum)
        ]
        if len(match) != 1:
            raise AssertionError("pairwise effect lookup is not unique")
        return match.iloc[0]

    ca500 = effect("C", "A", 500)
    cb500 = effect("C", "B", 500)
    ca10000 = effect("C", "A", 10_000)
    cb10000 = effect("C", "B", 10_000)
    direction_checks = []
    for left, right in COMPARISONS:
        for horizon in HORIZONS:
            overall = effect(left, right, horizon, "all")["point_difference"]
            feasible_point = effect(left, right, horizon, "inertia_feasible")[
                "point_difference"
            ]
            direction_checks.append(bool(overall == 0.0 or overall * feasible_point >= 0.0))

    training_rows: list[pd.DataFrame] = []
    for seed in SEEDS:
        for arm in ARMS:
            path = args.training_root.resolve() / f"seed_{seed}" / f"arm_{arm}" / "train.csv"
            training_rows.append(pd.read_csv(path).assign(seed=seed, arm=arm))
            inputs.append(path)
    training = pd.concat(training_rows, ignore_index=True)
    boundary_count = int(training["optimization_block_boundary"].sum())
    reject_count = int(training["skip_reason"].isin(REJECT_REASONS).sum())
    gradient_skip_count = int(training["skip_reason"].isin(GRADIENT_SKIP_REASONS).sum())
    invalid_count = int((training["rollout_valid"] == 0).sum())
    max_saturation_rise = float(
        pair_effects[pair_effects["stratum"].eq("all")][
            "action_saturation_difference"
        ].max()
    )

    gates = pd.DataFrame(
        [
            {
                "gate": "C_vs_A_H500_crossed_lower_above_minus_0p5pp",
                "value": ca500["crossed_ci_low"],
                "threshold": -0.005,
                "comparison": ">",
                "passed": bool(ca500["crossed_ci_low"] > -0.005),
            },
            {
                "gate": "C_vs_A_H500_no_seed_loses_more_than_1pp",
                "value": ca500["minimum_seed_difference"],
                "threshold": -0.01,
                "comparison": ">=",
                "passed": bool(ca500["minimum_seed_difference"] >= -0.01),
            },
            {
                "gate": "C_vs_B_H500_scenario_lower_above_zero",
                "value": cb500["scenario_ci_low"],
                "threshold": 0.0,
                "comparison": ">",
                "passed": bool(cb500["scenario_ci_low"] > 0.0),
            },
            {
                "gate": "C_vs_B_H10000_scenario_lower_above_minus_0p5pp",
                "value": cb10000["scenario_ci_low"],
                "threshold": -0.005,
                "comparison": ">",
                "passed": bool(cb10000["scenario_ci_low"] > -0.005),
            },
            {
                "gate": "C_vs_A_H10000_point_gain_at_least_0p5pp",
                "value": ca10000["point_difference"],
                "threshold": 0.005,
                "comparison": ">=",
                "passed": bool(ca10000["point_difference"] >= 0.005),
            },
            {
                "gate": "C_vs_A_H10000_positive_in_at_least_two_seeds",
                "value": ca10000["positive_seed_count"],
                "threshold": 2,
                "comparison": ">=",
                "passed": bool(ca10000["positive_seed_count"] >= 2),
            },
            {
                "gate": "inertia_feasible_effects_do_not_reverse",
                "value": int(sum(direction_checks)),
                "threshold": len(direction_checks),
                "comparison": "==",
                "passed": bool(all(direction_checks)),
            },
            {
                "gate": "zero_nonfinite_rollouts",
                "value": invalid_count,
                "threshold": 0,
                "comparison": "==",
                "passed": invalid_count == 0,
            },
            {
                "gate": "rejected_update_rate_at_most_0p5pct",
                "value": reject_count / max(boundary_count, 1),
                "threshold": 0.005,
                "comparison": "<=",
                "passed": reject_count / max(boundary_count, 1) <= 0.005,
            },
            {
                "gate": "gradient_skip_rate_at_most_0p5pct",
                "value": gradient_skip_count / max(boundary_count, 1),
                "threshold": 0.005,
                "comparison": "<=",
                "passed": gradient_skip_count / max(boundary_count, 1) <= 0.005,
            },
            {
                "gate": "action_saturation_rise_at_most_2pp",
                "value": max_saturation_rise,
                "threshold": 0.02,
                "comparison": "<=",
                "passed": max_saturation_rise <= 0.02,
            },
        ]
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe(performance, output / "PERFORMANCE_SUMMARY.csv")
    atomic_write_dataframe(pair_effects, output / "PAIRWISE_EFFECTS.csv")
    atomic_write_dataframe(seed_effects, output / "SEED_EFFECTS.csv")
    atomic_write_dataframe(gates, output / "DECISION_GATES.csv")
    all_passed = bool(gates["passed"].all())
    if label_status == "formal_matlab_confirmed":
        evaluation_statement = (
            "Success labels come from the formal MATLAB evaluation on the exact "
            "1,024-scenario manifest. Tail and action diagnostics come from the "
            "paired corrected Torch/CUDA replay. Internal/MATLAB label mismatches: "
            f"{matlab_label_mismatch_count}/{len(metrics)}."
        )
        title_suffix = "formal MATLAB evaluation"
    else:
        evaluation_statement = (
            "These outcomes use the corrected Torch/CUDA frozen rollout on the exact "
            "1,024-scenario manifest. The nine new policies have not yet been confirmed "
            "by MATLAB; this report is provisional rather than a formal decision."
        )
        title_suffix = "corrected internal evaluation"
    summary = f"""# Continuity/cadence screen - {title_suffix}

Status: **{'all preregistered gates passed' if all_passed else 'do not promote Arm C'}**.

{evaluation_statement}

## Primary effects

- C vs A, H500: {100 * ca500['point_difference']:+.3f} pp; crossed 95% CI [{100 * ca500['crossed_ci_low']:+.3f}, {100 * ca500['crossed_ci_high']:+.3f}] pp; worst seed {100 * ca500['minimum_seed_difference']:+.3f} pp.
- C vs B, H500: {100 * cb500['point_difference']:+.3f} pp; scenario-block 95% CI [{100 * cb500['scenario_ci_low']:+.3f}, {100 * cb500['scenario_ci_high']:+.3f}] pp.
- C vs B, H10000: {100 * cb10000['point_difference']:+.3f} pp; scenario-block 95% CI [{100 * cb10000['scenario_ci_low']:+.3f}, {100 * cb10000['scenario_ci_high']:+.3f}] pp.
- C vs A, H10000: {100 * ca10000['point_difference']:+.3f} pp; crossed 95% CI [{100 * ca10000['crossed_ci_low']:+.3f}, {100 * ca10000['crossed_ci_high']:+.3f}] pp; positive seeds {int(ca10000['positive_seed_count'])}/3.

## Training safety

- Non-finite rollout rows: {invalid_count}.
- Rejected updates: {reject_count}/{boundary_count}.
- Gradient skips: {gradient_skip_count}/{boundary_count}.
- Maximum measured action-saturation rise: {100 * max_saturation_rise:+.3f} pp.

The 400 inertia-feasible historical scenarios are reported separately in `PAIRWISE_EFFECTS.csv`. Failed gates must not be repaired by changing the sampler, clamp, action mapping, or network in this same experiment.
"""
    atomic_write_text(summary, output / "SUMMARY.md")
    provenance = artifact_fingerprint(
        inputs=(args.scenario_csv, *inputs),
        parameters={
            "seeds": SEEDS,
            "arms": ARMS,
            "horizons": HORIZONS,
            "bootstrap": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "label_status": label_status,
            "matlab_label_mismatch_count": matlab_label_mismatch_count,
            "matlab_reset_max_errors": matlab_reset_max_errors,
            "all_gates_passed": all_passed,
        },
        code_paths=(Path(__file__),),
    )
    provenance["outputs"] = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "RUN_PROVENANCE.json"
    }
    atomic_write_json(provenance, output / "RUN_PROVENANCE.json")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
