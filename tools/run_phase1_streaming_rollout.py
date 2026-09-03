from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.formal_rollout import load_q2_policy, run_formal_rollout, select_state
from diagnostics.phase1_fast_common import (
    artifact_fingerprint,
    atomic_write_dataframe,
    atomic_write_json,
    sha256_file,
)
from diagnostics.scenarios import (
    load_matlab_scenarios,
    read_scenario_rows,
    validate_against_matlab_samples,
)
from diagnostics.streaming_phase1 import Phase1StreamingAccumulator, StreamingConfig
from policy_observation import (
    INTEGRAL_CLAMP_MODES,
    LEGACY_BOX_INTEGRAL_CLAMP_MODE,
)


SCENARIO_ORDER_FIELDS = (
    "sample_id", "mass_kg", "cbrt_mass", "arm_length_m", "motor_span_m",
    "thrust_to_weight", "torque_to_inertia", "motor_time_rising_s",
    "motor_time_falling_s", "force_std", "inertia_x", "inertia_y", "inertia_z",
    "alpha_roll_max", "alpha_yaw_max", "eta_yaw", "jz_over_jxy",
    "external_force_x", "external_force_y", "external_force_z", "required_tilt_deg",
)

STREAMING_OUTPUT_NAMES = (
    "streaming_phase_profiles.csv",
    "streaming_frequency.csv",
    "tail_window.csv",
    "early_window.csv",
    "snapshot_selection_window.csv",
    "streaming_manifest.json",
    "matlab_labels.csv",
    "scenario_metrics.csv",
    "integral_diagnostics.csv",
)
SUMMARY_ONLY_OUTPUT_NAMES = (
    "matlab_labels.csv",
    "scenario_metrics.csv",
    "integral_diagnostics.csv",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _checkpoint(checkpoints_csv: Path, label: str) -> tuple[Path, int, str]:
    match = next((row for row in _read_csv(checkpoints_csv) if row["label"] == label), None)
    if match is None:
        raise KeyError(f"checkpoint label not found: {label}")
    path = Path(match["checkpoint_path"])
    if not path.is_file() or sha256_file(path) != match["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint path/hash mismatch: {path}")
    return path, int(match["seed"]), match["group"]


def _portable_label(row: dict[str, str]) -> str:
    return (
        f"seed_{int(row['seed'])}_group_{row['group']}_"
        f"physical_steps_{int(row['physical_steps'])}"
    )


def _portable_path(bundle_index: Path, relative_path: str) -> Path:
    root = bundle_index.resolve().parent
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise RuntimeError(f"portable bundle path must be relative: {relative_path}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise RuntimeError(f"portable bundle path escapes bundle root: {relative_path}")
    return resolved


def _scenario_order_sha256(rows: list[dict[str, str]]) -> str:
    table = [[row[field] for field in SCENARIO_ORDER_FIELDS] for row in rows]
    payload = json.dumps(table, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bundle_row(bundle_index: Path, checkpoint_label: str) -> dict[str, str]:
    matches = [
        row for row in _read_csv(bundle_index)
        if _portable_label(row) == checkpoint_label
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "portable bundle index must contain exactly one row for "
            f"checkpoint label {checkpoint_label!r}; found {len(matches)}"
        )
    return matches[0]


def _validate_bundle_provenance(
    *,
    checkpoint_label: str,
    bundle_index: Path,
    matlab_samples: Path,
) -> tuple[Path, int, str]:
    """Resolve a relocated diagnostic bundle without rewriting source manifests."""

    row = _bundle_row(bundle_index, checkpoint_label)
    checkpoint_path = _portable_path(bundle_index, row["checkpoint_path"])
    indexed_samples = _portable_path(bundle_index, row["samples_path"])
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
        raise RuntimeError(f"portable checkpoint path/hash mismatch: {checkpoint_path}")
    if not indexed_samples.is_file() or sha256_file(indexed_samples) != row["samples_sha256"]:
        raise RuntimeError(f"portable MATLAB sample path/hash mismatch: {indexed_samples}")
    if indexed_samples != matlab_samples.resolve():
        raise RuntimeError(
            "portable MATLAB sample provenance mismatch for "
            f"{checkpoint_label!r}: index={indexed_samples}, supplied={matlab_samples.resolve()}"
        )
    sample_rows = _read_csv(indexed_samples)
    if len(sample_rows) != int(row["sample_rows"]):
        raise RuntimeError(
            f"portable MATLAB sample row-count mismatch: {len(sample_rows)} != {row['sample_rows']}"
        )
    if not bool(int(row.get("motor_state_head_present", "0"))):
        raise RuntimeError(f"portable checkpoint lacks required motor_state_head: {checkpoint_path}")
    scenario_order = _scenario_order_sha256(sample_rows)
    if scenario_order != row.get("scenario_order_sha256"):
        raise RuntimeError(
            "portable MATLAB scenario-order hash mismatch: "
            f"{scenario_order} != {row.get('scenario_order_sha256')}"
        )
    return checkpoint_path, int(row["seed"]), row["group"]


def _validate_matlab_provenance(
    *,
    checkpoint_label: str,
    checkpoints_csv: Path,
    matlab_samples: Path,
    matlab_eval_manifest: Path,
) -> tuple[Path, int, str]:
    """Resolve and cross-check the checkpoint and MATLAB label sources."""

    checkpoint_path, seed, group = _checkpoint(checkpoints_csv, checkpoint_label)
    matches = [
        row
        for row in _read_csv(matlab_eval_manifest)
        if row.get("label") == checkpoint_label
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "MATLAB eval manifest must contain exactly one row for "
            f"checkpoint label {checkpoint_label!r}; found {len(matches)}"
        )
    manifest_row = matches[0]
    required = ("sample_output_path", "checkpoint_path")
    missing = [name for name in required if not manifest_row.get(name)]
    if missing:
        raise RuntimeError(
            f"MATLAB eval manifest row {checkpoint_label!r} lacks {missing}"
        )

    resolved_samples = matlab_samples.resolve()
    manifest_samples = Path(manifest_row["sample_output_path"]).resolve()
    if manifest_samples != resolved_samples:
        raise RuntimeError(
            "MATLAB sample provenance mismatch for "
            f"{checkpoint_label!r}: manifest={manifest_samples}, supplied={resolved_samples}"
        )

    resolved_checkpoint = checkpoint_path.resolve()
    manifest_checkpoint = Path(manifest_row["checkpoint_path"]).resolve()
    if manifest_checkpoint != resolved_checkpoint:
        raise RuntimeError(
            "MATLAB checkpoint provenance mismatch for "
            f"{checkpoint_label!r}: manifest={manifest_checkpoint}, "
            f"CHECKPOINTS={resolved_checkpoint}"
        )
    return resolved_checkpoint, seed, group


def _failure_label(row: dict[str, str], scenario: dict[str, str]) -> str:
    pass_columns = tuple(
        f"{channel}_pass_count_H10000"
        for channel in ("position", "velocity", "omega")
    )
    if all(column in row and row[column] != "" for column in pass_columns):
        p, v, o = (int(float(row[column])) for column in pass_columns)
        joint_success = bool(int(float(row["position_hold_steady_H10000"])))
        if p < 95 and v >= 95 and o >= 95:
            return "position-only"
        if o < 95:
            return "omega-related"
        if bool(int(float(scenario["dynamic_hard"]))):
            return "dynamic-hard"
        if joint_success:
            return "success"
        return "other-failure"

    # The formal six-run MATLAB tables contain exact joint 95/100 success but
    # not the three per-channel pass counts. Keep that label authoritative and
    # explicitly untyped; the Torch rollout's measured per-channel counts are
    # written to scenario_metrics.csv and can classify mechanisms afterwards.
    joint_success = bool(int(float(row["position_hold_steady_H10000"])))
    return "formal-joint-success" if joint_success else "formal-joint-failure-untyped"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-pass bounded-memory Q2 Phase 1 continuous rollout.")
    parser.add_argument("--checkpoint-label", default="seed_7_group_T0_physical_steps_96000000")
    parser.add_argument("--checkpoints-csv", type=Path, default=PROJECT_ROOT / "reports/q3_root_cause_diagnostics/CHECKPOINTS.csv")
    parser.add_argument(
        "--bundle-index",
        type=Path,
        help="Portable BUNDLE_INDEX.csv; resolves relative checkpoint/sample paths after relocation.",
    )
    parser.add_argument("--scenario-csv", type=Path, default=PROJECT_ROOT / "reports/q3_root_cause_diagnostics/SCENARIO_MANIFEST.csv")
    parser.add_argument("--matlab-samples", type=Path, required=True)
    parser.add_argument("--matlab-eval-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/q3_root_cause_diagnostics/streaming_seed7_t0")
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--scenario-limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--integral-clamp-mode",
        choices=INTEGRAL_CLAMP_MODES,
        default=LEGACY_BOX_INTEGRAL_CLAMP_MODE,
        help="Frozen-policy counterfactual only; training retains the legacy box default.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Write only paired horizon/integral summaries and provenance; skip large "
            "streaming phase, frequency, early, and tail artifacts."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _provenance_parameters(
    args: argparse.Namespace,
    *,
    seed: int,
    group: str,
) -> dict[str, object]:
    return {
        "checkpoint_label": args.checkpoint_label,
        "seed": seed,
        "training_group": group,
        "horizon": args.horizon,
        "scenario_limit": args.scenario_limit,
        "device": args.device,
        "integral_clamp_mode": args.integral_clamp_mode,
        "summary_only": args.summary_only,
    }


def main() -> int:
    parser = _argument_parser()
    args = parser.parse_args()
    if args.horizon < 100:
        raise ValueError("horizon must be at least 100")
    output = args.output.resolve()
    bundle_row: dict[str, str] | None = None
    if args.bundle_index is not None:
        bundle_row = _bundle_row(args.bundle_index, args.checkpoint_label)
        checkpoint_path, seed, group = _validate_bundle_provenance(
            checkpoint_label=args.checkpoint_label,
            bundle_index=args.bundle_index,
            matlab_samples=args.matlab_samples,
        )
        provenance_inputs = [
            checkpoint_path,
            args.scenario_csv,
            args.matlab_samples,
            args.bundle_index,
        ]
    else:
        if args.matlab_eval_manifest is None:
            parser.error("--matlab-eval-manifest is required unless --bundle-index is supplied")
        checkpoint_path, seed, group = _validate_matlab_provenance(
            checkpoint_label=args.checkpoint_label,
            checkpoints_csv=args.checkpoints_csv,
            matlab_samples=args.matlab_samples,
            matlab_eval_manifest=args.matlab_eval_manifest,
        )
        provenance_inputs = [
            checkpoint_path,
            args.scenario_csv,
            args.matlab_samples,
            args.checkpoints_csv,
            args.matlab_eval_manifest,
        ]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    output.mkdir(parents=True, exist_ok=True)
    code_paths = [
        Path(__file__),
        PROJECT_ROOT / "diagnostics/formal_rollout.py",
        PROJECT_ROOT / "diagnostics/physics.py",
        PROJECT_ROOT / "diagnostics/scenarios.py",
        PROJECT_ROOT / "diagnostics/streaming_phase1.py",
        PROJECT_ROOT / "l2f_cuda_backend.py",
        PROJECT_ROOT / "model.py",
        PROJECT_ROOT / "env_l2f.py",
        PROJECT_ROOT / "policy_observation.py",
    ]
    provenance = artifact_fingerprint(
        provenance_inputs,
        parameters=_provenance_parameters(args, seed=seed, group=group),
        code_paths=code_paths,
    )
    run_manifest = output / "RUN_PROVENANCE.json"
    if run_manifest.exists() and not args.force:
        existing = __import__("json").loads(run_manifest.read_text(encoding="utf-8"))
        outputs = existing.get("outputs", {})
        if existing.get("pipeline_fingerprint") == provenance["pipeline_fingerprint"] and outputs and all(
            (output / name).is_file() and sha256_file(output / name) == digest
            for name, digest in outputs.items()
        ):
            return 0
        raise RuntimeError("existing streaming output fingerprint differs; use a new output directory or --force")

    dtype = torch.float32 if device.type == "cuda" else torch.float64
    scenario_rows = read_scenario_rows(args.scenario_csv)
    if bundle_row is not None:
        formal_horizon = int(bundle_row["horizon"])
        if args.horizon > formal_horizon:
            raise RuntimeError(
                f"requested horizon {args.horizon} exceeds bundle horizon {formal_horizon}"
            )
        expected_eval_seed = int(bundle_row["eval_seed"])
        observed_eval_seeds = {int(float(row["eval_seed"])) for row in scenario_rows}
        if observed_eval_seeds != {expected_eval_seed}:
            raise RuntimeError(
                "scenario manifest evaluation seed does not match portable bundle: "
                f"{sorted(observed_eval_seeds)} != {[expected_eval_seed]}"
            )
    scenario_ids, state = load_matlab_scenarios(args.scenario_csv, device=device, dtype=dtype)
    matlab_rows = _read_csv(args.matlab_samples)
    if len(matlab_rows) != len(scenario_rows) or scenario_ids != [int(float(row["sample_id"])) for row in matlab_rows]:
        raise RuntimeError("MATLAB sample rows do not align with scenario manifest")
    reset_errors = validate_against_matlab_samples(scenario_rows, matlab_rows)
    if max(reset_errors.values(), default=float("inf")) > 1.0e-12:
        raise RuntimeError(
            "MATLAB sample parameters do not align with scenario manifest: "
            f"max_abs_error={max(reset_errors.values()):.3e}"
        )
    if args.scenario_limit is not None:
        if not 0 < args.scenario_limit <= len(scenario_rows):
            raise ValueError("scenario-limit is outside the manifest")
        indices = torch.arange(args.scenario_limit, device=device)
        state = select_state(state, indices)
        scenario_rows = scenario_rows[: args.scenario_limit]
        matlab_rows = matlab_rows[: args.scenario_limit]
    scenario_uids = [row["scenario_uid"] for row in scenario_rows]
    failure_groups = [_failure_label(matlab, scenario) for matlab, scenario in zip(matlab_rows, scenario_rows)]
    has_channel_counts = all(
        all(f"{channel}_pass_count_H10000" in row for channel in ("position", "velocity", "omega"))
        for row in matlab_rows
    )
    label_source = "matlab-per-channel-counts" if has_channel_counts else "matlab-joint-success-only"
    formal_success = [int(float(row["position_hold_steady_H10000"])) for row in matlab_rows]
    formal_by_uid = {
        uid: (success, group_name)
        for uid, success, group_name in zip(scenario_uids, formal_success, failure_groups)
    }
    policy, _ = load_q2_policy(checkpoint_path, device=device, dtype=dtype)
    signal_names = (
        "position_norm", "velocity_norm", "omega_norm", "action_delta_norm", "hidden_norm",
        "integral_norm", "integral_clamp_fraction", "branch_collective", "branch_torque_norm",
        "damping_power", "damping_torque_authority_ratio", "motor_prediction_rmse",
        "tanh_gain_mean", "dense_potential",
    )
    accumulator = None
    if not args.summary_only:
        accumulator = Phase1StreamingAccumulator(StreamingConfig(
            batch_size=len(scenario_uids), signal_names=signal_names
        ))
    metadata: dict[str, list[object]] = {
        "checkpoint": [args.checkpoint_label] * len(scenario_uids),
        "seed": [seed] * len(scenario_uids),
        "training_group": [group] * len(scenario_uids),
        "failure_group": failure_groups,
        "integral_clamp_mode": [args.integral_clamp_mode] * len(scenario_uids),
    }
    for column in ("dynamic_hard", "high_force", "low_roll", "low_yaw"):
        metadata[column] = [int(float(row[column])) for row in scenario_rows]

    started = time.perf_counter()
    result = run_formal_rollout(
        policy,
        state,
        scenario_uids,
        checkpoint_label=args.checkpoint_label,
        seed=seed,
        group=group,
        horizon=args.horizon,
        snapshot_horizons=tuple(sorted({
            args.horizon,
            *(value for value in (500, 1000, 2000, 5000, 10000) if value <= args.horizon),
        })),
        backend="cuda" if device.type == "cuda" else "torch",
        intervention="full",
        record_branches=False,
        record_phase=False,
        streaming_accumulator=accumulator,
        integral_clamp_mode=args.integral_clamp_mode,
    )
    rollout_elapsed = time.perf_counter() - started
    for row in result.horizon_rows:
        row["formal_steady_success_H10000"] = formal_by_uid[row["scenario_uid"]][0]
        row["formal_failure_group_H10000"] = formal_by_uid[row["scenario_uid"]][1]
        row["formal_label_source"] = label_source
    if accumulator is not None:
        accumulator.write(output, scenario_uid=scenario_uids, metadata=metadata)
    atomic_write_dataframe(pd.DataFrame({
        "scenario_uid": scenario_uids,
        "formal_steady_success_H10000": formal_success,
        "formal_failure_group_H10000": failure_groups,
        "formal_label_source": [label_source] * len(scenario_uids),
        "integral_clamp_mode": [args.integral_clamp_mode] * len(scenario_uids),
    }), output / "matlab_labels.csv")
    atomic_write_dataframe(pd.DataFrame(result.horizon_rows), output / "scenario_metrics.csv")
    atomic_write_dataframe(pd.DataFrame(result.integral_rows), output / "integral_diagnostics.csv")
    output_names = SUMMARY_ONLY_OUTPUT_NAMES if args.summary_only else STREAMING_OUTPUT_NAMES
    outputs = [output / name for name in output_names]
    if missing_outputs := [path.name for path in outputs if not path.is_file()]:
        raise RuntimeError(f"streaming rollout did not produce required outputs: {missing_outputs}")
    provenance.update({
        "scenario_count": len(scenario_uids),
        "formal_label_source": label_source,
        "integral_clamp_mode": args.integral_clamp_mode,
        "summary_only": args.summary_only,
        "rollout_elapsed_s": rollout_elapsed,
        "total_physical_steps": len(scenario_uids) * args.horizon,
        "outputs": {path.name: sha256_file(path) for path in outputs},
    })
    atomic_write_json(provenance, run_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
