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
DEFAULT_ARMS = ("A", "B", "C")
ALLOWED_ARMS = ("A", "B", "C", "D")
HORIZONS = (500, 10_000)
RESET_MAPPINGS = {
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


def _parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    unknown = tuple(arm for arm in arms if arm not in ALLOWED_ARMS)
    if not arms or unknown:
        raise argparse.ArgumentTypeError(
            f"arms must be drawn from {','.join(ALLOWED_ARMS)}; unknown={unknown}"
        )
    if len(set(arms)) != len(arms):
        raise argparse.ArgumentTypeError("arms must not contain duplicates")
    return arms


def _formal_columns(horizon: int) -> tuple[str, str, str, str]:
    suffix = f"_H{horizon}"
    final_column = "position_H500_m" if horizon == 500 else "position_final_m"
    return (
        f"position_hold_steady{suffix}",
        f"position_pass_count{suffix}",
        f"position_tail_rms{suffix}",
        final_column,
    )


def _cpu_row(
    root: Path | None,
    *,
    seed: int,
    arm: str,
    scenario_uid: str,
    horizon: int,
    cache: dict[Path, pd.DataFrame],
    inputs: list[Path],
) -> pd.Series | None:
    if root is None:
        return None
    path = root / f"seed_{seed}" / f"arm_{arm}" / "scenario_metrics.csv"
    if path not in cache:
        if not path.is_file():
            return None
        cache[path] = pd.read_csv(path)
        inputs.append(path)
        provenance = path.with_name("RUN_PROVENANCE.json")
        if provenance.is_file():
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            if payload.get("outputs", {}).get(path.name) != sha256_file(path):
                raise RuntimeError(f"CPU confirmation output hash mismatch: {path}")
            inputs.append(provenance)
    frame = cache[path]
    match = frame[
        frame["scenario_uid"].eq(scenario_uid) & frame["horizon"].eq(horizon)
    ]
    if len(match) != 1:
        return None
    return match.iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate CUDA, formal MATLAB, and CPU-float64 label parity."
    )
    parser.add_argument(
        "--internal-root",
        type=Path,
        default=ROOT / "reports" / "continuity_cadence_internal_eval_20260804",
    )
    parser.add_argument(
        "--matlab-root",
        type=Path,
        default=ROOT / "reports" / "continuity_cadence_matlab_20260804",
    )
    parser.add_argument(
        "--cpu-root",
        type=Path,
        default=ROOT / "reports" / "continuity_cadence_cpu_confirmation_20260804",
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
        default=ROOT / "reports" / "continuity_cadence_label_parity_20260804",
    )
    parser.add_argument(
        "--arms",
        type=_parse_arms,
        default=DEFAULT_ARMS,
        help="Comma-separated policy arms to validate (default: A,B,C).",
    )
    args = parser.parse_args()

    scenarios = pd.read_csv(args.scenario_csv.resolve())
    expected_ids = scenarios["scenario_id"].to_numpy(dtype=np.int64)
    expected_uids = scenarios["scenario_uid"].astype(str).tolist()
    if len(expected_ids) != 1024 or len(set(expected_ids)) != 1024:
        raise RuntimeError("scenario manifest must contain 1,024 unique IDs")

    inputs: list[Path] = [args.scenario_csv.resolve()]
    mismatch_rows: list[dict[str, object]] = []
    reset_errors = {source: 0.0 for source in RESET_MAPPINGS}
    cpu_cache: dict[Path, pd.DataFrame] = {}
    comparison_count = 0
    for seed in SEEDS:
        for arm in args.arms:
            internal_path = (
                args.internal_root.resolve()
                / f"seed_{seed}"
                / f"arm_{arm}"
                / "scenario_metrics.csv"
            )
            matlab_path = (
                args.matlab_root.resolve()
                / f"seed_{seed}"
                / f"arm_{arm}"
                / "samples.csv"
            )
            internal = pd.read_csv(internal_path)
            samples = pd.read_csv(matlab_path)
            inputs.extend((internal_path, matlab_path))
            sample_ids = samples["sample_id"].to_numpy(dtype=np.int64)
            if not np.array_equal(sample_ids, expected_ids):
                raise RuntimeError(f"sample IDs/order do not match manifest: {matlab_path}")
            for source, target in RESET_MAPPINGS.items():
                error = float(
                    np.max(
                        np.abs(
                            scenarios[source].to_numpy(dtype=np.float64)
                            - samples[target].to_numpy(dtype=np.float64)
                        )
                    )
                )
                reset_errors[source] = max(reset_errors[source], error)
            for horizon in HORIZONS:
                success_column, pass_column, rms_column, final_column = _formal_columns(
                    horizon
                )
                internal_horizon = internal[internal["horizon"].eq(horizon)].set_index(
                    "scenario_uid"
                )
                if len(internal_horizon) != 1024:
                    raise RuntimeError(
                        f"internal row count is not 1,024: {internal_path}, H{horizon}"
                    )
                internal_horizon = internal_horizon.loc[expected_uids]
                cuda_success = internal_horizon["steady_success"].to_numpy(
                    dtype=np.int64
                )
                matlab_success = samples[success_column].to_numpy(dtype=np.int64)
                comparison_count += len(cuda_success)
                for index in np.flatnonzero(cuda_success != matlab_success):
                    scenario_uid = expected_uids[index]
                    cuda_row = internal_horizon.loc[scenario_uid]
                    cpu = _cpu_row(
                        args.cpu_root.resolve() if args.cpu_root else None,
                        seed=seed,
                        arm=arm,
                        scenario_uid=scenario_uid,
                        horizon=horizon,
                        cache=cpu_cache,
                        inputs=inputs,
                    )
                    mismatch_rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "horizon": horizon,
                            "sample_id": int(sample_ids[index]),
                            "scenario_uid": scenario_uid,
                            "cuda_success": int(cuda_success[index]),
                            "matlab_success": int(matlab_success[index]),
                            "cpu_float64_success": (
                                np.nan if cpu is None else int(cpu["steady_success"])
                            ),
                            "cuda_position_pass_count": int(
                                cuda_row["position_pass_count"]
                            ),
                            "matlab_position_pass_count": int(samples.iloc[index][pass_column]),
                            "cpu_position_pass_count": (
                                np.nan if cpu is None else int(cpu["position_pass_count"])
                            ),
                            "cuda_position_tail_rms": float(cuda_row["position_tail_rms"]),
                            "matlab_position_tail_rms": float(samples.iloc[index][rms_column]),
                            "cpu_position_tail_rms": (
                                np.nan if cpu is None else float(cpu["position_tail_rms"])
                            ),
                            "cuda_position_final": float(cuda_row["position_final"]),
                            "matlab_position_final": float(samples.iloc[index][final_column]),
                            "cpu_position_final": (
                                np.nan if cpu is None else float(cpu["position_final"])
                            ),
                        }
                    )

    if max(reset_errors.values()) > 1.0e-12:
        raise RuntimeError(f"MATLAB reset mismatch exceeds tolerance: {reset_errors}")
    mismatch_columns = (
        "seed",
        "arm",
        "horizon",
        "sample_id",
        "scenario_uid",
        "cuda_success",
        "matlab_success",
        "cpu_float64_success",
        "cuda_position_pass_count",
        "matlab_position_pass_count",
        "cpu_position_pass_count",
        "cuda_position_tail_rms",
        "matlab_position_tail_rms",
        "cpu_position_tail_rms",
        "cuda_position_final",
        "matlab_position_final",
        "cpu_position_final",
    )
    mismatches = pd.DataFrame(mismatch_rows, columns=mismatch_columns)
    if len(mismatches):
        cpu_confirmed = (
            mismatches["cpu_float64_success"].eq(mismatches["matlab_success"])
            & mismatches["cpu_position_pass_count"].eq(
                mismatches["matlab_position_pass_count"]
            )
            & (
                mismatches["cpu_position_tail_rms"]
                - mismatches["matlab_position_tail_rms"]
            ).abs().le(1.0e-12)
            & (
                mismatches["cpu_position_final"]
                - mismatches["matlab_position_final"]
            ).abs().le(1.0e-12)
        )
        cpu_confirmed_count = int(cpu_confirmed.sum())
    else:
        cpu_confirmed_count = 0

    reset_frame = pd.DataFrame(
        [
            {"scenario_column": source, "maximum_absolute_error": error}
            for source, error in reset_errors.items()
        ]
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe(mismatches, output / "LABEL_MISMATCHES.csv")
    atomic_write_dataframe(reset_frame, output / "RESET_PARAMETER_MAX_ERRORS.csv")
    summary = f"""# Continuity/cadence label parity

- CUDA float32 versus formal MATLAB comparisons: {comparison_count}.
- Label mismatches: {len(mismatches)}.
- Mismatches reproduced by Torch CPU float64 as the MATLAB label: {cpu_confirmed_count}/{len(mismatches)}.
- Maximum reset-parameter absolute error: {max(reset_errors.values()):.3e}.

Formal MATLAB labels are used for the decision report. A mismatch confirmed by CPU float64 is classified as a numerical boundary case, not a provenance mismatch.
"""
    atomic_write_text(summary, output / "SUMMARY.md")
    provenance = artifact_fingerprint(
        inputs=tuple(dict.fromkeys(inputs)),
        parameters={
            "seeds": SEEDS,
            "arms": args.arms,
            "horizons": HORIZONS,
            "comparison_count": comparison_count,
            "label_mismatch_count": len(mismatches),
            "cpu_confirmed_count": cpu_confirmed_count,
            "reset_tolerance": 1.0e-12,
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
