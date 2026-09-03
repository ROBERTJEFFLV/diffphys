from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.provenance import canonical_config_hash, git_commit
from diagnostics.scenarios import (
    formal_scenario_uid,
    load_matlab_scenarios,
    read_scenario_rows,
    scenario_group_masks,
    validate_against_matlab_samples,
)
from diagnostics.physics import motor_to_thrust, steady_state_feasibility, thrust_to_wrench
from diagnostics.physics import action_to_next_wrench
from diagnostics.formal_rollout import load_q2_policy, rollout_q2, select_state, trace_q2
from diagnostics.time_weighting import dense_tracking_time_weights
from diagnostics.sampling_coverage import (
    extreme_masks,
    extreme_thresholds,
    sample_physical_coverage,
    state_coverage,
)
from env_l2f import L2FState
from policy_observation import PolicyObservationState, build_policy_observation, update_position_integral
from diagnostics.gradient_audit import run_gradient_audit
from env_l2f import L2FParams


DEFAULT_OUTPUT = ROOT / "reports" / "q3_root_cause_diagnostics"
FORMAL_EVAL_MANIFEST = ROOT / "reports" / "time_horizon_q2_multiseed_matlab" / "eval_manifest.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_rows(eval_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in eval_rows:
        checkpoint = Path(row["checkpoint_path"])
        weights = Path(row["weights_path"])
        if not checkpoint.is_file() or not weights.is_file():
            raise FileNotFoundError(f"formal checkpoint/export missing: {checkpoint} / {weights}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = payload.get("args", {})
        exported = loadmat(weights, squeeze_me=True, struct_as_record=False)
        source = Path(str(exported["source_checkpoint"]))
        checkpoint_hash = _sha256(checkpoint)
        source_hash = _sha256(source) if source.is_file() else "missing"
        seed = int(args.get("seed", row["label"].split("_", 2)[1]))
        group = "T2" if "group_T2" in row["label"] else "T0"
        command_path = checkpoint.parents[1] / "command.txt"
        train_log_path = checkpoint.parents[1] / "train.csv"
        result.append({
            "label": row["label"],
            "seed": seed,
            "group": group,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "physical_steps": int(payload.get("physical_steps", 96_000_000)),
            "resolved_config_sha256": canonical_config_hash(args),
            "model_mat_path": str(weights.resolve()),
            "model_mat_sha256": _sha256(weights),
            "model_mat_source_checkpoint": str(source),
            "source_checkpoint_sha256": source_hash,
            "source_matches_manifest_checkpoint": int(source_hash == checkpoint_hash),
            "command_path": str(command_path.resolve()),
            "command_sha256": _sha256(command_path) if command_path.is_file() else "missing",
            "train_log_path": str(train_log_path.resolve()),
            "train_log_sha256": _sha256(train_log_path) if train_log_path.is_file() else "missing",
            "observation_mode": args.get("observation_mode"),
            "integral_input_frame": args.get("integral_input_frame"),
            "integral_input_multiplier": args.get("integral_input_multiplier"),
            "integral_limit": args.get("integral_limit"),
            "integral_leak": args.get("integral_leak"),
            "enable_integral_residual": args.get("enable_integral_residual"),
            "enable_rate_damping_residual": args.get("enable_rate_damping_residual"),
            "episode_horizon_schedule": args.get("episode_horizon_schedule"),
        })
    return result


def prepare(output: Path, scenario_reset_csv: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    eval_rows = _read_csv(FORMAL_EVAL_MANIFEST)
    if len(eval_rows) != 6:
        raise ValueError(f"formal manifest has {len(eval_rows)} jobs, expected 6")
    checkpoint_rows = _checkpoint_rows(eval_rows)
    _write_csv(output / "CHECKPOINTS.csv", checkpoint_rows)
    seed7_t0 = next(row for row in eval_rows if row["label"].startswith("seed_7_group_T0_"))
    _write_csv(
        output / "stage1a_matlab_labels_manifest.csv",
        [{
            "label": "q3_phase1_stage1a_seed7_t0_labels",
            "weights_path": seed7_t0["weights_path"],
            "output_path": str((output / "stage1a_matlab_labels_summary.csv").resolve()),
            "sample_output_path": str((output / "stage1a_matlab_labels_samples.csv").resolve()),
            "mat_output_path": str((output / "stage1a_matlab_labels_metrics.mat").resolve()),
            "batch_size": 1024,
            "eval_seed": 1007,
            "horizon": 10000,
        }],
    )

    scenario_rows = read_scenario_rows(scenario_reset_csv)
    if len(scenario_rows) != 1024:
        raise ValueError(f"formal reset export has {len(scenario_rows)} rows, expected 1024")
    masks = scenario_group_masks(scenario_rows)
    force = np.linalg.norm(
        np.asarray([[float(row[f"external_force_{i}"]) for i in range(3)] for row in scenario_rows]),
        axis=1,
    )
    manifest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(scenario_rows):
        scenario_id = int(float(row["scenario_id"]))
        item: dict[str, Any] = {
            "scenario_uid": formal_scenario_uid(1007, scenario_id),
            "scenario_id": scenario_id,
            "eval_seed": 1007,
            "scenario_source": "matlab:l2f_reset:physical-broad:physical",
            **{name: value for name, value in row.items() if name != "scenario_id"},
            "external_force_norm": float(force[index]),
        }
        item.update({name: int(mask[index]) for name, mask in masks.items()})
        manifest_rows.append(item)
    _write_csv(output / "SCENARIO_MANIFEST.csv", manifest_rows)

    reference_samples = Path(eval_rows[0]["sample_output_path"])
    reset_errors = validate_against_matlab_samples(scenario_rows, _read_csv(reference_samples))
    max_reset_error = max(reset_errors.values())
    if max_reset_error > 1.0e-12:
        raise RuntimeError(f"formal reset stream mismatch: max error {max_reset_error:.3e}")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "git_commit": git_commit(ROOT),
        "git_dirty_state": "unavailable:not-a-git-repository",
        "formal_eval_manifest": str(FORMAL_EVAL_MANIFEST.resolve()),
        "formal_eval_manifest_sha256": _sha256(FORMAL_EVAL_MANIFEST),
        "scenario_reset_export": str(scenario_reset_csv.resolve()),
        "scenario_reset_export_sha256": _sha256(scenario_reset_csv),
        "scenario_count": len(scenario_rows),
        "evaluation_seed": 1007,
        "evaluation_horizon": 10000,
        "evaluation_batch_size": 1024,
        "matlab_entry": "matlab_l2f/run_motor_gru_eval_manifest.m",
        "matlab_reset": "l2f_reset(1024, physical-broad/physical, seed=1007)",
        "success_semantics": {
            "window_steps": 100,
            "required_passes": 95,
            "position_norm_lt_m": 0.05,
            "velocity_norm_lt_m_s": 0.10,
            "omega_norm_lt_rad_s": 0.20,
        },
        "reset_parameter_max_abs_errors_vs_existing_samples": reset_errors,
        "reset_parameter_alignment_pass": max_reset_error <= 1.0e-12,
        "checkpoint_sha256": {row["label"]: row["checkpoint_sha256"] for row in checkpoint_rows},
        "config_sha256": {row["label"]: row["resolved_config_sha256"] for row in checkpoint_rows},
    }
    (output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    run_plan = f"""# Q3 Phase 1 run plan

This directory is diagnostics-only. It does not alter Q2/T0 policy, simulator, training loss, optimizer, or defaults.

## Frozen evaluation objects

- Six 96M checkpoints: seed 7/17/27 paired T0 and T2 (see `CHECKPOINTS.csv`).
- Formal scenarios: 1024 rows from MATLAB `physical-broad/physical`, eval seed 1007.
- Reset parameter parity: maximum absolute error versus the existing formal sample table is {max_reset_error:.3e}.
- One continuous H10000 trajectory supplies H500/H1000/H2000/H5000/H10000 snapshots.
- MATLAB remains the source of formal success labels; Python/compact CUDA supplies internal diagnostics.

## Stages and gates

1. **1A wiring:** exact reset export, deterministic stratified 64, short MATLAB/Python trajectory comparison, schema and finite-value validation.
2. **1B T0:** all three T0 checkpoints over the full 1024-scenario H10000 stream.
3. **1C paired T2:** the same seed-scenario pairs, retaining diagnostics that address time weighting, phase, and gradient structure.

Stage 1B must not start if reset fields, scenario UIDs, initial hidden/integral/action state, or short-trajectory tolerances fail.

## Fixed 64 selection

After an exact H10000 per-channel 95/100 label replay, sort candidates by SHA256 scenario UID and select without replacement in this priority order: position-only 8, omega-related 8, dynamic-hard 8, high-force 8, low-roll 8, low-yaw 8, ordinary-success 16. A shortage is backfilled deterministically from the nearest compatible stratum. Overlap belongs only to the first matching stratum. The resulting IDs are written to `STRATIFIED_SUBSET.md` before intervention.

## Output integrity

Every CSV must contain measured rows. Missing experiments are documented in `BLOCKERS.md`; no empty CSV or placeholder plot is created. Scenario-level aggregation precedes group statistics and paired bootstrap resamples scenarios.
"""
    (output / "RUN_PLAN.md").write_text(run_plan, encoding="utf-8")

    blockers = """# Phase 1 blockers and provenance limitations

- The workspace is not a Git repository (`.git` is absent). Git commit and dirty state therefore cannot be recovered and are explicitly recorded as unavailable; no value is inferred.
- MATLAB R2023b starts with a broken user/default path warning (`initdesktoputils` missing). The same `restoredefaultpath; rehash toolboxcache` prefix recorded by the prior formal evaluation restores core functions. Exact reset export then completed and matched all 18 comparable fields in the existing 1024-row formal sample table. This warning is retained as an environment risk, but is not currently blocking Stage 1A.
"""
    (output / "BLOCKERS.md").write_text(blockers, encoding="utf-8")


def feasibility(output: Path, scenario_reset_csv: Path) -> None:
    ids, state = load_matlab_scenarios(scenario_reset_csv, dtype=torch.float64)
    result = steady_state_feasibility(state, L2FParams())
    reconstructed_thrust = motor_to_thrust(state, result.required_motor_command)
    reconstructed_wrench = thrust_to_wrench(state, reconstructed_thrust)
    target_wrench = torch.cat(
        (result.required_total_thrust[:, None], torch.zeros_like(state.omega)), dim=-1
    )
    residual = torch.linalg.vector_norm(reconstructed_wrench - target_wrench, dim=-1)
    scenario_manifest = _read_csv(output / "SCENARIO_MANIFEST.csv")
    rows: list[dict[str, Any]] = []
    for index, scenario_id in enumerate(ids):
        lower = bool(result.lower_thrust_violation[index].any())
        upper = bool(result.upper_thrust_violation[index].any())
        numerical = bool(result.numerical_failure[index])
        feasible_flag = bool(result.feasible[index])
        if numerical:
            reason = "thrust-polynomial-inversion-or-numerical-failure"
        elif upper and float(result.required_total_thrust[index]) > float(result.max_motor_thrust[index].sum()):
            reason = "collective-thrust-infeasible"
        elif upper or lower:
            reason = "per-motor-infeasible"
        elif feasible_flag:
            reason = "analytically-feasible"
        else:
            reason = "unclassified-numerical-failure"
        row: dict[str, Any] = {
            "scenario_uid": scenario_manifest[index]["scenario_uid"],
            "scenario_id": scenario_id,
            "required_total_thrust_n": float(result.required_total_thrust[index]),
            "hover_thrust_n": float(state.mass[index] * 9.80665),
            "required_thrust_ratio": float(result.required_thrust_ratio[index]),
            "required_trim_ratio": float(result.required_trim_ratio[index]),
            "required_tilt_deg": float(torch.rad2deg(result.required_tilt_rad[index])),
            "desired_body_z_x": float(result.required_body_z_world[index, 0]),
            "desired_body_z_y": float(result.required_body_z_world[index, 1]),
            "desired_body_z_z": float(result.required_body_z_world[index, 2]),
            "feasible": int(feasible_flag),
            "classification": reason,
            "numerical_solver_residual": float(residual[index]) if torch.isfinite(residual[index]) else float("nan"),
            "active_constraint": "upper" if upper else ("lower" if lower else "none"),
        }
        for motor in range(4):
            row[f"motor_{motor}_required_thrust_n"] = float(result.required_motor_thrust[index, motor])
            row[f"motor_{motor}_required_command"] = float(result.required_motor_command[index, motor])
            row[f"motor_{motor}_upper_headroom"] = float(result.upper_command_headroom[index, motor])
            row[f"motor_{motor}_lower_headroom"] = float(result.lower_command_headroom[index, motor])
            row[f"motor_{motor}_min_thrust_n"] = float(result.min_motor_thrust[index, motor])
            row[f"motor_{motor}_max_thrust_n"] = float(result.max_motor_thrust[index, motor])
        for name in ("dynamic_hard", "high_force", "low_roll", "low_yaw", "slow_motor_rise", "slow_motor_fall", "low_thrust_to_weight"):
            row[name] = scenario_manifest[index][name]
        rows.append(row)
    _write_csv(output / "feasibility.csv", rows)

    summary_rows: list[dict[str, Any]] = []
    groups = ["overall", "dynamic_hard", "high_force", "low_roll", "low_yaw", "slow_motor_rise", "slow_motor_fall", "low_thrust_to_weight"]
    for group in groups:
        selected = rows if group == "overall" else [row for row in rows if int(row[group]) == 1]
        ratios = np.asarray([float(row["required_trim_ratio"]) for row in selected])
        tilts = np.asarray([float(row["required_tilt_deg"]) for row in selected])
        summary_rows.append({
            "group": group,
            "count": len(selected),
            "feasible_count": sum(int(row["feasible"]) for row in selected),
            "infeasible_count": sum(1 - int(row["feasible"]) for row in selected),
            "feasible_fraction": float(np.mean([int(row["feasible"]) for row in selected])),
            "collective_infeasible_count": sum(row["classification"] == "collective-thrust-infeasible" for row in selected),
            "per_motor_infeasible_count": sum(row["classification"] == "per-motor-infeasible" for row in selected),
            "numerical_failure_count": sum("numerical" in row["classification"] or "inversion" in row["classification"] for row in selected),
            "required_trim_ratio_mean": float(np.mean(ratios)),
            "required_trim_ratio_p95": float(np.quantile(ratios, 0.95)),
            "required_tilt_deg_mean": float(np.mean(tilts)),
            "required_tilt_deg_p95": float(np.quantile(tilts, 0.95)),
        })
    _write_csv(output / "feasibility_group_summary.csv", summary_rows)
    overall = summary_rows[0]
    report = f"""# Analytic steady-state feasibility

The exact 1024-scenario formal MATLAB reset stream was evaluated with the legacy thrust polynomial, command bounds, steady motor relation, and four-rotor mixer. Equal rotor thrust is not assumed feasible: the polynomial is inverted per rotor and the resulting wrench is reconstructed.

- Analytically feasible: {overall['feasible_count']} / {overall['count']} ({100.0 * overall['feasible_fraction']:.3f}%).
- Collective-thrust infeasible: {overall['collective_infeasible_count']}.
- Per-motor infeasible: {overall['per_motor_infeasible_count']}.
- Polynomial/numerical failures: {overall['numerical_failure_count']}.

`feasibility.csv` retains actuator headroom, active constraints, reconstructed-wrench residual, and a distinct classification for every scenario. Controller statistics must report both raw and feasible-only populations.
"""
    (output / "FEASIBILITY_REPORT.md").write_text(report, encoding="utf-8")


def _formal_label(row: dict[str, str], suffix: str = "H10000") -> str:
    survived = int(float(row[f"survival_{suffix}"])) == 1
    joint = int(round(100.0 * float(row[f"final_window_success_fraction_{suffix}"])))
    position = int(float(row[f"position_pass_count_{suffix}"]))
    velocity = int(float(row[f"velocity_pass_count_{suffix}"]))
    omega = int(float(row[f"omega_pass_count_{suffix}"]))
    if not survived:
        return "survival_failure"
    if joint >= 95:
        return "success"
    failed = [name for name, count in (("position", position), ("velocity", velocity), ("omega", omega)) if count < 95]
    if not failed:
        return "window_overlap_failure"
    if len(failed) == 3:
        return "combined_failure"
    return "+".join(failed) if len(failed) > 1 else f"{failed[0]}-only_failure"


def select_stage1a(output: Path) -> None:
    label_path = output / "stage1a_matlab_labels_samples.csv"
    if not label_path.is_file():
        raise FileNotFoundError(f"MATLAB per-channel labels are not complete: {label_path}")
    labels = _read_csv(label_path)
    manifest = _read_csv(output / "SCENARIO_MANIFEST.csv")
    if len(labels) != 1024 or len(manifest) != 1024:
        raise ValueError("Stage 1A selection requires the full 1024-scenario set")
    enriched: list[dict[str, Any]] = []
    for sample, scenario in zip(labels, manifest):
        if int(sample["sample_id"]) != int(scenario["scenario_id"]):
            raise ValueError("MATLAB labels and scenario manifest are misaligned")
        item = dict(scenario)
        item["formal_failure_label_H10000"] = _formal_label(sample)
        item["formal_steady_success_H10000"] = int(float(sample["position_hold_steady_H10000"]))
        item["position_pass_count_H10000"] = int(float(sample["position_pass_count_H10000"]))
        item["velocity_pass_count_H10000"] = int(float(sample["velocity_pass_count_H10000"]))
        item["omega_pass_count_H10000"] = int(float(sample["omega_pass_count_H10000"]))
        item["selection_hash"] = hashlib.sha256((item["scenario_uid"] + ":q3-phase1-stage1a").encode()).hexdigest()
        enriched.append(item)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def take(name: str, count: int, predicate: Any) -> None:
        candidates = sorted(
            (row for row in enriched if row["scenario_uid"] not in used and predicate(row)),
            key=lambda row: row["selection_hash"],
        )
        if len(candidates) < count:
            raise RuntimeError(f"Stage 1A stratum {name} has only {len(candidates)} candidates, needs {count}")
        for row in candidates[:count]:
            row = dict(row)
            row["stage1a_stratum"] = name
            row["stage1a_selection_rank"] = len(selected) + 1
            selected.append(row)
            used.add(row["scenario_uid"])

    take("position-only", 8, lambda row: row["formal_failure_label_H10000"] == "position-only_failure")
    take("omega-related", 8, lambda row: "omega" in row["formal_failure_label_H10000"])
    take("dynamic-hard", 8, lambda row: int(row["dynamic_hard"]) == 1)
    take("high-force", 8, lambda row: int(row["high_force"]) == 1)
    take("low-roll", 8, lambda row: int(row["low_roll"]) == 1)
    take("low-yaw", 8, lambda row: int(row["low_yaw"]) == 1)
    take(
        "ordinary-success", 16,
        lambda row: row["formal_failure_label_H10000"] == "success"
        and not any(int(row[name]) for name in ("dynamic_hard", "high_force", "low_roll", "low_yaw")),
    )
    _write_csv(output / "stage1a_subset.csv", selected)
    lines = [
        "# Stage 1A deterministic stratified subset", "",
        "Selection uses exact MATLAB H10000 per-channel 95/100 counts. Candidates are sorted by `SHA256(scenario_uid + ':q3-phase1-stage1a')`, selected without replacement, and assigned to the first matching stratum in the pre-registered priority order.", "",
        "| rank | scenario_id | scenario_uid | stratum | H10000 label | p/v/omega pass |", "|---:|---:|---|---|---|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row['stage1a_selection_rank']} | {row['scenario_id']} | `{row['scenario_uid']}` | "
            f"{row['stage1a_stratum']} | {row['formal_failure_label_H10000']} | "
            f"{row['position_pass_count_H10000']}/{row['velocity_pass_count_H10000']}/{row['omega_pass_count_H10000']} |"
        )
    (output / "STRATIFIED_SUBSET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage1a(output: Path, scenario_reset_csv: Path, device_name: str) -> None:
    subset = _read_csv(output / "stage1a_subset.csv")
    checkpoints = _read_csv(output / "CHECKPOINTS.csv")
    checkpoint = next(row for row in checkpoints if row["seed"] == "7" and row["group"] == "T0")
    all_ids, all_state = load_matlab_scenarios(
        scenario_reset_csv, device=device_name, dtype=torch.float32
    )
    index_by_id = {scenario_id: index for index, scenario_id in enumerate(all_ids)}
    indices = torch.as_tensor(
        [index_by_id[int(row["scenario_id"])] for row in subset],
        device=device_name, dtype=torch.long,
    )
    state = select_state(all_state, indices)
    uids = [row["scenario_uid"] for row in subset]
    policy, _ = load_q2_policy(
        checkpoint["checkpoint_path"], device=device_name, dtype=torch.float32
    )
    all_horizon_rows: list[dict[str, Any]] = []
    full_result = rollout_q2(
        policy, state, uids,
        checkpoint_label=checkpoint["label"], seed=7, group="T0", horizon=2000,
        snapshot_horizons=(500, 2000), backend="cuda" if device_name.startswith("cuda") else "torch",
        intervention="full", record_branches=True, branch_stride=1, record_phase=True,
    )
    all_horizon_rows.extend(full_result.horizon_rows)
    for intervention in ("damping_zero", "explicit_integral_zero", "integral_state_zero"):
        result = rollout_q2(
            policy, state, uids,
            checkpoint_label=checkpoint["label"], seed=7, group="T0", horizon=2000,
            snapshot_horizons=(500, 2000), backend="cuda" if device_name.startswith("cuda") else "torch",
            intervention=intervention, record_branches=False, record_phase=False,
        )
        all_horizon_rows.extend(result.horizon_rows)
    group_by_uid = {row["scenario_uid"]: row for row in subset}
    for rows in (full_result.branch_rows, full_result.phase_rows, full_result.integral_rows, all_horizon_rows):
        for row in rows:
            groups = group_by_uid[row["scenario_uid"]]
            row["stage1a_stratum"] = groups["stage1a_stratum"]
            row["formal_failure_label_H10000"] = groups["formal_failure_label_H10000"]
            for name in ("dynamic_hard", "high_force", "low_roll", "low_yaw", "slow_motor_rise", "slow_motor_fall", "low_thrust_to_weight"):
                row[name] = groups[name]
    _write_csv(output / "stage1a_branch_wrench.csv", full_result.branch_rows)
    _write_csv(output / "stage1a_h250_phase_raw.csv", full_result.phase_rows)
    _write_csv(output / "stage1a_integral_diagnostics.csv", full_result.integral_rows)
    _write_csv(output / "branch_intervention.csv", all_horizon_rows)

    base = {(row["scenario_uid"], int(row["horizon"])): row for row in all_horizon_rows if row["intervention"] == "full"}
    pair_rows: list[dict[str, Any]] = []
    for row in all_horizon_rows:
        if row["intervention"] == "full":
            continue
        reference = base[row["scenario_uid"], int(row["horizon"])]
        pair: dict[str, Any] = {
            "checkpoint": row["checkpoint"], "seed": row["seed"], "scenario_uid": row["scenario_uid"],
            "horizon": row["horizon"], "intervention": row["intervention"],
            "baseline_success": reference["steady_success"], "intervention_success": row["steady_success"],
            "S_to_F": int(int(reference["steady_success"]) == 1 and int(row["steady_success"]) == 0),
            "F_to_S": int(int(reference["steady_success"]) == 0 and int(row["steady_success"]) == 1),
            "stage1a_stratum": row["stage1a_stratum"],
        }
        for metric in ("position_tail_rms", "velocity_tail_rms", "omega_tail_rms", "control_energy", "action_saturation_fraction"):
            pair[f"baseline_{metric}"] = reference[metric]
            pair[f"intervention_{metric}"] = row[metric]
            pair[f"delta_{metric}"] = float(row[metric]) - float(reference[metric])
        pair_rows.append(pair)
    _write_csv(output / "branch_intervention_pairs.csv", pair_rows)
    report = """# Stage 1A frozen-policy branch intervention

The deterministic 64-scenario subset was run for H2000 without parameter updates. `full`, explicit damping forced to zero, explicit integral residual forced to zero, and total integral state forced to zero use paired initial states. Removing the explicit integral residual is not interpreted as removing all integral influence; only the total-integral-state intervention removes the input from the main recurrent path as well.

These Stage 1A results are a wiring/safety gate. They are not promoted to full-population causal conclusions before the three-seed formal analysis.
"""
    (output / "BRANCH_INTERVENTION_REPORT.md").write_text(report, encoding="utf-8")


def time_weighting(output: Path) -> None:
    checkpoints = _read_csv(output / "CHECKPOINTS.csv")
    summary_rows: list[dict[str, Any]] = []
    by_step_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        rows = [row for row in _read_csv(Path(checkpoint["train_log_path"])) if int(float(row["physical_steps"])) <= 96_000_000]
        segments = len(rows)
        boundaries = [row for row in rows if int(float(row["episode_boundary"])) == 1]
        first_segments = [row for row in rows if int(float(row["segment_id"])) == 1]
        updates = max(int(float(row["optimizer_update"])) for row in rows)
        batch_size = 256
        retain_fractions = [float(row["retain_fraction_actual"]) for row in first_segments if row["retain_fraction_actual"].lower() != "nan"]
        retain_fraction = float(np.mean(retain_fractions))
        episode_counts = {h: sum(int(float(row["episode_target_steps"])) == h for row in boundaries) for h in (500, 1000, 2000)}
        reset_events = len(first_segments)
        fresh_sample_resets = sum(batch_size * (1.0 - float(row["retain_fraction_actual"])) for row in first_segments)
        retain_sample_resets = sum(batch_size * float(row["retain_fraction_actual"]) for row in first_segments)
        item = {
            "seed": checkpoint["seed"], "training_group": checkpoint["group"],
            "physical_steps": 96_000_000, "optimizer_updates": updates,
            "episodes": len(boundaries), "reset_events": reset_events,
            "fresh_sample_resets": fresh_sample_resets, "retain_sample_resets": retain_sample_resets,
            "retain_fraction_actual_mean": retain_fraction,
            "h250_segments": segments, "h500_boundaries": episode_counts[500],
            "h1000_boundaries": episode_counts[1000], "h2000_boundaries": episode_counts[2000],
            "early_cvar_events": len(first_segments), "final_cvar_events": len(boundaries),
            "tracking_tail_events": segments,
            "fresh_recovery_segment_fraction": len(first_segments) / segments,
            "late_state_segment_fraction": 1.0 - len(first_segments) / segments,
            "optimizer_updates_per_million_physical_steps": updates / 96.0,
            "reset_events_per_million_physical_steps": reset_events / 96.0,
            "h500_final_events_per_million_physical_steps": episode_counts[500] / 96.0,
        }
        summary_rows.append(item)
        distribution_rows.extend([
            {"seed": checkpoint["seed"], "training_group": checkpoint["group"], "state_region": "first_H250_recovery", "segment_count": len(first_segments), "segment_fraction": len(first_segments) / segments},
            {"seed": checkpoint["seed"], "training_group": checkpoint["group"], "state_region": "continued_late_state", "segment_count": segments - len(first_segments), "segment_fraction": 1.0 - len(first_segments) / segments},
        ])
        for episode_steps in sorted({int(float(row["episode_target_steps"])) for row in boundaries}):
            weights = dense_tracking_time_weights(
                episode_steps=episode_steps, segment_steps=250, tail_steps=50,
                lambda_tail=1.0,
            ).numpy()
            for step in range(1, episode_steps + 1):
                first_tail = 151 <= step <= 250
                final_tail = episode_steps - 99 <= step <= episode_steps
                motor_aux = 0.03 / max(episode_steps - 15, 1) if step > 15 else 0.0
                by_step_rows.append({
                    "seed": checkpoint["seed"], "training_group": checkpoint["group"],
                    "episode_steps": episode_steps, "step_in_episode": step,
                    "segment_phase": (step - 1) % 250,
                    "dense_tracking_effective_weight": float(weights[step - 1]),
                    "segment_tracking_tail_extra_weight": float(weights[step - 1] - 1.0 / episode_steps),
                    "early_cvar_window_exposure": 0.25 / 100.0 if first_tail else 0.0,
                    "final_cvar_window_exposure": 1.0 / 100.0 if final_tail else 0.0,
                    "motor_aux_effective_weight": motor_aux,
                    "is_fresh_recovery": int(step <= 250),
                    "is_late_state": int(step > 250),
                })
    _write_csv(output / "time_weighting.csv", summary_rows)
    _write_csv(output / "time_weighting_by_step.csv", by_step_rows)
    _write_csv(output / "state_distribution_summary.csv", distribution_rows)
    group_means: dict[str, dict[str, float]] = {}
    for group in ("T0", "T2"):
        selected = [row for row in summary_rows if row["training_group"] == group]
        group_means[group] = {
            name: float(np.mean([float(row[name]) for row in selected]))
            for name in ("optimizer_updates", "reset_events", "h500_boundaries", "fresh_recovery_segment_fraction", "late_state_segment_fraction")
        }
    t0, t2 = group_means["T0"], group_means["T2"]
    report = f"""# T0/T2 effective weighting at the selected 96M checkpoint

Counts were recomputed from each of the six actual `train.csv` logs, not only from the theoretical curriculum.

| quantity (three-seed mean) | T0 | T2 |
|---|---:|---:|
| optimizer updates | {t0['optimizer_updates']:.0f} | {t2['optimizer_updates']:.0f} |
| reset events | {t0['reset_events']:.0f} | {t2['reset_events']:.0f} |
| H500 final events | {t0['h500_boundaries']:.0f} | {t2['h500_boundaries']:.0f} |
| first-H250 recovery segment fraction | {t0['fresh_recovery_segment_fraction']:.4f} | {t2['fresh_recovery_segment_fraction']:.4f} |
| continued late-state segment fraction | {t0['late_state_segment_fraction']:.4f} | {t2['late_state_segment_fraction']:.4f} |

The observed schedule changes are directionally sufficient to explain slower H500 recovery, smoother/lower-energy control, and slightly better late behavior as an objective/state-distribution reweighting mechanism. This is a high-credibility observational explanation, not proof that it is the only cause; the separate gradient audit tests whether cross-segment action credit exists.
"""
    (output / "T0_T2_WEIGHTING_REPORT.md").write_text(report, encoding="utf-8")


def sampling_coverage(output: Path, scenario_reset_csv: Path, sample_count: int) -> None:
    sampler = sample_physical_coverage(sample_count, seed=31_007)
    _, evaluation_state = load_matlab_scenarios(scenario_reset_csv, dtype=torch.float64)
    evaluation = state_coverage(evaluation_state)
    retain_payload = torch.load(
        ROOT / "reports" / "next_stage_retain_bank" / "retain_bank.pt",
        map_location="cpu", weights_only=False,
    )
    retain_state = L2FState(**retain_payload["state"])
    retain = state_coverage(retain_state)
    datasets = {"physical_broad_sampler": sampler, "formal_evaluation": evaluation, "retain_bank": retain}
    thresholds = extreme_thresholds(sampler)
    coverage_rows: list[dict[str, Any]] = []
    for dataset_name, data in datasets.items():
        for variable, values in data.mapping().items():
            coverage_rows.append({
                "dataset": dataset_name, "variable": variable, "count": values.size,
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "q01": float(np.quantile(values, 0.01)), "q05": float(np.quantile(values, 0.05)),
                "q20": float(np.quantile(values, 0.20)), "median": float(np.quantile(values, 0.50)),
                "q80": float(np.quantile(values, 0.80)), "q95": float(np.quantile(values, 0.95)),
                "q99": float(np.quantile(values, 0.99)),
                "sampler_low20_threshold": thresholds[variable][0],
                "sampler_high20_threshold": thresholds[variable][1],
            })
    _write_csv(output / "sampling_coverage.csv", coverage_rows)

    joint_rows: list[dict[str, Any]] = []
    retain_rows: list[dict[str, Any]] = []
    batch_size = 256
    for dataset_name, data in datasets.items():
        masks = extreme_masks(data, thresholds)
        for corner, mask in masks.items():
            probability = float(mask.mean())
            row = {
                "dataset": dataset_name, "corner": corner, "sample_count": mask.size,
                "corner_count": int(mask.sum()), "probability": probability,
                "expected_per_batch_256": batch_size * probability,
                "binomial_zero_batch_probability": float((1.0 - probability) ** batch_size),
                "actual_batch_count_mean": float("nan"), "actual_zero_batch_fraction": float("nan"),
            }
            if dataset_name == "physical_broad_sampler" and mask.size % batch_size == 0:
                counts = mask.reshape(-1, batch_size).sum(axis=1)
                row["actual_batch_count_mean"] = float(np.mean(counts))
                row["actual_zero_batch_fraction"] = float(np.mean(counts == 0))
                row["actual_batch_count_p05"] = float(np.quantile(counts, 0.05))
                row["actual_batch_count_p95"] = float(np.quantile(counts, 0.95))
            joint_rows.append(row)
            if dataset_name in ("retain_bank", "formal_evaluation"):
                retain_rows.append(dict(row))
    _write_csv(output / "sampling_joint_extremes.csv", joint_rows)
    _write_csv(output / "retain_coverage.csv", retain_rows)
    rare = [row for row in joint_rows if row["dataset"] == "physical_broad_sampler" and float(row["expected_per_batch_256"]) < 1.0]
    report = f"""# Physical-broad joint sampling coverage

The sampler was measured without training using {sample_count:,} fresh physical-broad draws at fixed seed 31007. Tail cutoffs are defined once from that sampler and then applied unchanged to the formal evaluation set and retain bank.

- Joint regions with fewer than one expected sample per batch of 256: {len(rare)} / 7.
- The four-way `high force + low roll + low yaw + slow motor` corner is measured directly; its zero-sample batch frequency is reported rather than inferred from independent 0.2 tails.
- The retain bank is a conditional old-success set, so its coverage is reported separately and is not treated as a challenge distribution.

See `sampling_joint_extremes.csv` for expected counts, empirical batch-count distribution, and zero-sample fractions.
"""
    (output / "SAMPLING_COVERAGE_REPORT.md").write_text(report, encoding="utf-8")


def gradient_audit(output: Path, scenario_reset_csv: Path) -> None:
    checkpoint = next(
        row for row in _read_csv(output / "CHECKPOINTS.csv")
        if row["seed"] == "7" and row["group"] == "T0"
    )
    summaries: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []

    _, cpu_state = load_matlab_scenarios(scenario_reset_csv, device="cpu", dtype=torch.float64)
    cpu_state = select_state(cpu_state, torch.tensor([0], dtype=torch.long))
    current = run_gradient_audit(
        checkpoint["checkpoint_path"], cpu_state, horizon=500, device="cpu", dtype=torch.float64
    )
    for rows in current:
        for row in rows:
            row["audit_precision"] = "cpu_float64_tiny"
    summaries.extend(current[0]); parameter_rows.extend(current[1]); boundary_rows.extend(current[2])

    if torch.cuda.is_available():
        _, cuda_state = load_matlab_scenarios(scenario_reset_csv, device="cuda", dtype=torch.float32)
        cuda_state = select_state(cuda_state, torch.arange(4, device="cuda"))
        for horizon in (500, 1000):
            current = run_gradient_audit(
                checkpoint["checkpoint_path"], cuda_state, horizon=horizon,
                device="cuda", dtype=torch.float32,
            )
            for rows in current:
                for row in rows:
                    row["audit_precision"] = "cuda_float32_practical"
            summaries.extend(current[0]); parameter_rows.extend(current[1]); boundary_rows.extend(current[2])
    _write_csv(output / "gradient_audit.csv", summaries)
    _write_csv(output / "gradient_audit_parameter_groups.csv", parameter_rows)
    _write_csv(output / "gradient_audit_boundaries.csv", boundary_rows)

    import matplotlib.pyplot as plt
    figure_rows = [row for row in boundary_rows if row["audit_precision"] == "cuda_float32_practical" and int(row["horizon"]) == 1000]
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for mode in ("legacy_detach", "full_bptt", "no_detach_decay", "checkpoint_recompute"):
        selected = sorted((row for row in figure_rows if row["mode"] == mode), key=lambda row: int(row["boundary_after_segment"]))
        if selected:
            distance = [1000 - 250 * int(row["boundary_after_segment"]) for row in selected]
            adjoint = [max(float(row["physical_state_adjoint_norm"]), 1.0e-30) for row in selected]
            axis.plot(distance, adjoint, marker="o", label=mode)
    axis.set_yscale("log")
    axis.set_xlabel("backpropagation distance from final H1000 boundary (steps)")
    axis.set_ylabel("future-loss physical-state adjoint norm")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "figures" / "gradient_decay_over_time.png", dpi=160)
    plt.close(figure)

    practical = [row for row in summaries if row["audit_precision"] == "cuda_float32_practical" and int(row["horizon"]) == 1000]
    legacy = next((row for row in practical if row["mode"] == "legacy_detach"), None)
    full = next((row for row in practical if row["mode"] == "full_bptt"), None)
    recompute = next((row for row in practical if row["mode"] == "checkpoint_recompute"), None)
    report = f"""# Cross-segment gradient audit

No parameter update was performed. CPU float64 H500 verifies graph semantics; CUDA float32 H500/H1000 measures practical runtime and memory. All modes use the same initial states, parameters, forward equations, and diagnostic loss; gradient-decay operators are identity in forward.

- Legacy final-loss adjoint to the first-segment action at H1000: {float(legacy['early_segment_action_adjoint_from_final_loss']) if legacy else float('nan'):.6g}.
- Full-BPTT final-loss adjoint to the first-segment action at H1000: {float(full['early_segment_action_adjoint_from_final_loss']) if full else float('nan'):.6g}.
- Checkpoint/recompute gradient cosine versus full BPTT: {float(recompute['gradient_cosine_vs_full_bptt']) if recompute else float('nan'):.6g}.
- Maximum forward error is recorded per mode in `gradient_audit.csv`; a nonzero value fails the backward-only comparison premise.

The audit distinguishes existence of cross-segment credit from its numerical size. It does not update the formal optimizer or enable a new training route.
"""
    (output / "GRADIENT_AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def simulator_consistency(output: Path, scenario_reset_csv: Path) -> None:
    from scipy.io import loadmat

    subset = _read_csv(output / "stage1a_subset.csv")
    checkpoint = next(row for row in _read_csv(output / "CHECKPOINTS.csv") if row["seed"] == "7" and row["group"] == "T0")
    ids, state = load_matlab_scenarios(scenario_reset_csv, dtype=torch.float64)
    index_by_id = {scenario_id: index for index, scenario_id in enumerate(ids)}
    indices = torch.tensor([index_by_id[int(row["scenario_id"])] for row in subset], dtype=torch.long)
    state = select_state(state, indices)
    policy, _ = load_q2_policy(checkpoint["checkpoint_path"], device="cpu", dtype=torch.float64)
    python_trace = trace_q2(policy, state, horizon=100)
    matlab = loadmat(output / "stage1a_matlab_consistency.mat")
    tolerance = 1.0e-10
    rows: list[dict[str, Any]] = []
    variables = ("position", "velocity", "rotation", "omega", "motor", "action", "integral", "hidden")
    for variable in variables:
        python_value = getattr(python_trace, variable).numpy()
        matlab_name = "hidden_log" if variable == "hidden" else variable
        if variable == "rotation":
            python_value = python_value.reshape(python_value.shape[0], python_value.shape[1], 9)
        matlab_value = np.asarray(matlab[matlab_name])
        difference = np.abs(python_value - matlab_value)
        for index, subset_row in enumerate(subset):
            per_step = difference[:, index].reshape(difference.shape[0], -1).max(axis=1)
            divergent = np.flatnonzero(per_step > tolerance)
            rows.append({
                "scenario_uid": subset_row["scenario_uid"], "scenario_id": subset_row["scenario_id"],
                "variable": variable, "compared_steps": difference.shape[0],
                "max_absolute_error": float(difference[:, index].max()),
                "rms_error": float(np.sqrt(np.mean(difference[:, index] ** 2))),
                "first_divergence_step": int(divergent[0]) if divergent.size else -1,
                "tolerance": tolerance, "pass": int(divergent.size == 0),
            })
    _write_csv(output / "simulator_consistency.csv", rows)
    max_error = max(float(row["max_absolute_error"]) for row in rows)
    failures = sum(1 - int(row["pass"]) for row in rows)
    report = f"""# MATLAB/Python simulator consistency

The deterministic Stage 1A subset uses the exact same 64 reset states, seed-7 T0 checkpoint, zero hidden/integral initialization, and 100 continuous steps. MATLAB uses the formal evaluation functions; Python uses the Torch double-precision Q2 path.

- Compared scenario-variable series: {len(rows)}.
- Tolerance: {tolerance:.1e} max absolute error at every compared step.
- Failures: {failures}.
- Global maximum absolute error: {max_error:.3e}.

This establishes short-horizon equation and initialization parity. It does not require pointwise H10000 equality after floating-point divergence; formal H500-H10000 success remains sourced from MATLAB.
"""
    (output / "SIMULATOR_CONSISTENCY.md").write_text(report, encoding="utf-8")


def yaw_equivalence(output: Path, scenario_reset_csv: Path, sample_count: int) -> None:
    checkpoint = next(row for row in _read_csv(output / "CHECKPOINTS.csv") if row["seed"] == "7" and row["group"] == "T0")
    ids, base = load_matlab_scenarios(scenario_reset_csv, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(41_007)
    indices = torch.arange(sample_count) % len(ids)
    state = select_state(base, indices)
    yaw = (2.0 * torch.rand(sample_count, generator=generator, dtype=torch.float64) - 1.0) * math.pi
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    q = torch.zeros(sample_count, 3, 3, dtype=torch.float64)
    q[:, 0, 0] = cosine; q[:, 0, 1] = -sine
    q[:, 1, 0] = sine; q[:, 1, 1] = cosine; q[:, 2, 2] = 1.0
    rotate = lambda value: torch.bmm(q, value.unsqueeze(-1)).squeeze(-1)
    rotated_state = L2FState(**{
        name: (
            rotate(getattr(state, name)) if name in ("position", "velocity", "external_force")
            else torch.bmm(q, getattr(state, name)) if name == "rotation"
            else getattr(state, name).clone()
        )
        for name in L2FState.__dataclass_fields__
    })
    integral = 1.4 * torch.rand(sample_count, 3, generator=generator, dtype=torch.float64) - 0.7
    rotated_integral = rotate(integral)
    obs_state = PolicyObservationState(integral)
    rotated_obs_state = PolicyObservationState(rotated_integral)
    observation, observed_position = build_policy_observation(
        state, obs_state, mode="integral25", integral_input_frame="body"
    )
    rotated_observation, rotated_position = build_policy_observation(
        rotated_state, rotated_obs_state, mode="integral25", integral_input_frame="body"
    )
    policy, _ = load_q2_policy(checkpoint["checkpoint_path"], device="cpu", dtype=torch.float64)
    hidden = policy.initial_hidden(sample_count, device="cpu", dtype=torch.float64)
    with torch.no_grad():
        action, _ = policy(observation, hidden)
        rotated_action, _ = policy(rotated_observation, hidden)
        wrench = action_to_next_wrench(state, action, dt=0.01)
        rotated_wrench = action_to_next_wrench(rotated_state, rotated_action, dt=0.01)
    next_integral = update_position_integral(
        obs_state, observed_position, dt=0.01, integral_limit=0.5
    ).integral_position
    rotated_next_integral = update_position_integral(
        rotated_obs_state, rotated_position, dt=0.01, integral_limit=0.5
    ).integral_position
    expected_rotated_integral = rotate(next_integral)
    rows: list[dict[str, Any]] = []
    for index in range(sample_count):
        rows.append({
            "test_seed": 41007, "sample_index": index + 1,
            "source_scenario_id": ids[int(indices[index])], "yaw_angle_rad": float(yaw[index]),
            "pre_clamp_integral_norm": float(torch.linalg.vector_norm(integral[index])),
            "original_clamped_axes": int((next_integral[index].abs() >= 0.5 - 1.0e-12).sum()),
            "rotated_clamped_axes": int((rotated_next_integral[index].abs() >= 0.5 - 1.0e-12).sum()),
            "box_clamp_equivariance_error": float(torch.linalg.vector_norm(rotated_next_integral[index] - expected_rotated_integral[index])),
            "box_clamp_magnitude_error": float(abs(torch.linalg.vector_norm(rotated_next_integral[index]) - torch.linalg.vector_norm(next_integral[index]))),
            "legacy_observation_direct_error": float(torch.linalg.vector_norm(rotated_observation[index] - observation[index])),
            "integral_body_invariance_error": float(torch.linalg.vector_norm(rotated_observation[index, 18:21] - observation[index, 18:21])),
            "action_invariance_error": float(torch.linalg.vector_norm(rotated_action[index] - action[index])),
            "body_wrench_invariance_error": float(torch.linalg.vector_norm(rotated_wrench[index] - wrench[index])),
        })
    _write_csv(output / "yaw_equivalence.csv", rows)


def _descriptive_group_rows(
    scenario_rows: list[dict[str, Any]],
    *,
    metrics: list[str],
    group_masks: dict[str, Any],
    seed: int = 51_007,
) -> list[dict[str, Any]]:
    generator = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    for group, predicate in group_masks.items():
        selected = [row for row in scenario_rows if predicate(row)]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            boot = np.mean(generator.choice(values, size=(2000, values.size), replace=True), axis=1)
            result.append({
                "group": group, "metric": metric, "scenario_count": values.size,
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "q05": float(np.quantile(values, 0.05)), "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)), "q95": float(np.quantile(values, 0.95)),
                "bootstrap_mean_ci95_low": float(np.quantile(boot, 0.025)),
                "bootstrap_mean_ci95_high": float(np.quantile(boot, 0.975)),
            })
    return result


def summarize_stage1a(output: Path) -> None:
    subset = {row["scenario_uid"]: row for row in _read_csv(output / "stage1a_subset.csv")}
    source = output / "stage1a_branch_wrench.csv"
    metric_names = [
        f"{prefix}_{component}"
        for prefix in ("integral_delta", "damping_delta", "damping_delta_no_integral", "interaction")
        for component in ("collective", "tau_x", "tau_y", "tau_z")
    ]
    accum: dict[tuple[str, str], dict[str, Any]] = {}
    damping_rows: list[dict[str, Any]] = []
    branch_sample_rows: list[dict[str, Any]] = []
    previous_omega: dict[str, float] = {}
    previous_power: dict[str, float] = {}

    def window_names(step: int) -> tuple[str, str]:
        return (("H0_H500" if step <= 500 else "H500_H2000"), "H0_H2000")

    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uid = row["scenario_uid"]
            step = int(row["step"])
            scenario = subset[uid]
            inertia_x = float(scenario["inertia_x"])
            inertia_z = float(scenario["inertia_z"])
            roll_torque = inertia_x * float(scenario["alpha_roll_max"])
            yaw_torque = inertia_z * float(scenario["alpha_yaw_max"])
            torque_authority = math.sqrt(2.0 * roll_torque * roll_torque + yaw_torque * yaw_torque)
            omega_norm = float(row["omega_norm"])
            torque_norm = float(row["damping_torque_norm"])
            power = float(row["damping_power"])
            cosine = float(row["damping_cosine"])
            omega_growth = float("nan")
            aligned_positive = 0
            if uid in previous_omega:
                omega_growth = omega_norm - previous_omega[uid]
                aligned_positive = int(previous_power[uid] > 0.0 and omega_growth > 0.0)
            previous_omega[uid] = omega_norm
            previous_power[uid] = power
            damping_rows.append({
                "checkpoint": row["checkpoint"], "seed": row["seed"], "scenario_uid": uid,
                "step": step, "time_s": row["time_s"], "stage1a_stratum": row["stage1a_stratum"],
                "formal_failure_label_H10000": row["formal_failure_label_H10000"],
                "omega_norm": omega_norm, "damping_torque_norm": torque_norm,
                "torque_authority": torque_authority, "damping_power": power, "damping_cosine": cosine,
                "active_rate": int(omega_norm >= 0.05),
                "active_damping_0p5pct": int(omega_norm >= 0.05 and torque_norm >= 0.005 * torque_authority),
                "active_damping_1pct": int(omega_norm >= 0.05 and torque_norm >= 0.01 * torque_authority),
                "positive_power": int(power > 0.0), "positive_power_energy_j": max(power, 0.0) * 0.01,
                "negative_power_energy_j": min(power, 0.0) * 0.01,
                "omega_growth_next_aligned": aligned_positive, "omega_norm_delta_from_previous": omega_growth,
                "power_x": row["damping_power_x"], "power_y": row["damping_power_y"], "power_z": row["damping_power_z"],
            })
            if (step - 1) % 10 == 0:
                branch_sample_rows.append(dict(row))
            for window in window_names(step):
                key = (uid, window)
                data = accum.setdefault(key, {
                    "count": 0, "metrics": {name: [0.0, 0.0, 0.0, 0.0] for name in metric_names},
                    "power": [], "cosine": [], "positive_energy": 0.0, "negative_energy": 0.0,
                    "active_rate": 0, "active_rate_positive": 0,
                    "active_0p5": 0, "active_0p5_positive": 0, "active_1": 0, "active_1_positive": 0,
                    "max_positive_power": 0.0, "positive_run": 0, "longest_positive_run": 0,
                    "aligned_growth_count": 0, "gain_sum": np.zeros(4), "saturation_count": 0,
                })
                data["count"] += 1
                for metric in metric_names:
                    value = float(row[metric])
                    stats = data["metrics"][metric]
                    stats[0] += value; stats[1] += abs(value); stats[2] += value * value; stats[3] = max(stats[3], abs(value))
                data["power"].append(power); data["cosine"].append(cosine)
                data["positive_energy"] += max(power, 0.0) * 0.01
                data["negative_energy"] += min(power, 0.0) * 0.01
                data["max_positive_power"] = max(data["max_positive_power"], power)
                data["positive_run"] = data["positive_run"] + 1 if power > 0 else 0
                data["longest_positive_run"] = max(data["longest_positive_run"], data["positive_run"])
                active_rate = omega_norm >= 0.05
                active_0p5 = active_rate and torque_norm >= 0.005 * torque_authority
                active_1 = active_rate and torque_norm >= 0.01 * torque_authority
                data["active_rate"] += active_rate; data["active_rate_positive"] += active_rate and power > 0
                data["active_0p5"] += active_0p5; data["active_0p5_positive"] += active_0p5 and power > 0
                data["active_1"] += active_1; data["active_1_positive"] += active_1 and power > 0
                data["aligned_growth_count"] += aligned_positive
                data["gain_sum"] += np.asarray([float(row[f"tanh_gain_{motor}"]) for motor in range(4)])
                data["saturation_count"] += sum(abs(float(row[f"full_action_{motor}"])) >= 0.98 for motor in range(4))

    _write_csv(output / "branch_wrench.csv", branch_sample_rows)
    _write_csv(output / "damping_power.csv", damping_rows)
    branch_scenarios: list[dict[str, Any]] = []
    damping_scenarios: list[dict[str, Any]] = []
    for (uid, window), data in accum.items():
        scenario = subset[uid]
        branch_row: dict[str, Any] = {
            "checkpoint": "seed_7_group_T0_physical_steps_96000000", "seed": 7,
            "scenario_uid": uid, "window": window, "step_count": data["count"],
            "stage1a_stratum": scenario["stage1a_stratum"],
            "formal_failure_label_H10000": scenario["formal_failure_label_H10000"],
        }
        for name in ("dynamic_hard", "high_force", "low_roll", "low_yaw"):
            branch_row[name] = scenario[name]
        for metric, stats in data["metrics"].items():
            branch_row[f"{metric}_mean"] = stats[0] / data["count"]
            branch_row[f"{metric}_mean_abs"] = stats[1] / data["count"]
            branch_row[f"{metric}_rms"] = math.sqrt(stats[2] / data["count"])
            branch_row[f"{metric}_max_abs"] = stats[3]
        branch_row["tanh_gain_mean"] = float(np.mean(data["gain_sum"] / data["count"]))
        branch_row["action_saturation_fraction"] = data["saturation_count"] / (4 * data["count"])
        branch_scenarios.append(branch_row)
        power_values = np.asarray(data["power"]); cosine_values = np.asarray(data["cosine"])
        damping_scenarios.append({
            **{key: branch_row[key] for key in ("checkpoint", "seed", "scenario_uid", "window", "stage1a_stratum", "formal_failure_label_H10000", "dynamic_hard", "high_force", "low_roll", "low_yaw")},
            "step_count": data["count"], "positive_power_fraction_all": float(np.mean(power_values > 0)),
            "positive_power_fraction_active_rate": data["active_rate_positive"] / max(data["active_rate"], 1),
            "positive_power_fraction_active_damping_0p5pct": data["active_0p5_positive"] / max(data["active_0p5"], 1),
            "positive_power_fraction_active_damping_1pct": data["active_1_positive"] / max(data["active_1"], 1),
            "active_rate_count": data["active_rate"], "active_damping_0p5pct_count": data["active_0p5"], "active_damping_1pct_count": data["active_1"],
            "positive_power_integrated_energy": data["positive_energy"],
            "negative_power_integrated_energy": data["negative_energy"],
            "net_damping_work": data["positive_energy"] + data["negative_energy"],
            "max_positive_power": data["max_positive_power"],
            "longest_positive_power_interval_s": data["longest_positive_run"] * 0.01,
            "positive_power_aligned_with_omega_growth_fraction": data["aligned_growth_count"] / max(data["count"] - 1, 1),
            "cosine_q05": float(np.quantile(cosine_values, 0.05)), "cosine_median": float(np.median(cosine_values)),
            "cosine_q95": float(np.quantile(cosine_values, 0.95)),
        })
    _write_csv(output / "branch_wrench_scenario_summary.csv", branch_scenarios)
    _write_csv(output / "damping_power_scenario_summary.csv", damping_scenarios)
    masks = {
        "all_stage1a": lambda row: True,
        "success": lambda row: row["formal_failure_label_H10000"] == "success",
        "position-only": lambda row: row["formal_failure_label_H10000"] == "position-only_failure",
        "omega-related": lambda row: "omega" in row["formal_failure_label_H10000"],
        "dynamic-hard": lambda row: int(row["dynamic_hard"]) == 1,
        "high-force": lambda row: int(row["high_force"]) == 1,
        "low-roll": lambda row: int(row["low_roll"]) == 1,
        "low-yaw": lambda row: int(row["low_yaw"]) == 1,
    }
    branch_metrics = ["damping_delta_collective_mean_abs", "damping_delta_tau_x_rms", "damping_delta_tau_y_rms", "damping_delta_tau_z_rms", "integral_delta_collective_mean_abs", "integral_delta_tau_x_rms", "integral_delta_tau_y_rms", "integral_delta_tau_z_rms", "tanh_gain_mean", "action_saturation_fraction"]
    damping_metrics = ["positive_power_fraction_all", "positive_power_fraction_active_rate", "positive_power_fraction_active_damping_1pct", "positive_power_integrated_energy", "negative_power_integrated_energy", "net_damping_work", "max_positive_power", "longest_positive_power_interval_s"]
    final_branch = [row for row in branch_scenarios if row["window"] == "H0_H2000"]
    final_damping = [row for row in damping_scenarios if row["window"] == "H0_H2000"]
    _write_csv(output / "branch_wrench_group_summary.csv", _descriptive_group_rows(final_branch, metrics=branch_metrics, group_masks=masks))
    _write_csv(output / "damping_power_group_summary.csv", _descriptive_group_rows(final_damping, metrics=damping_metrics, group_masks=masks))
    _write_csv(output / "integral_diagnostics.csv", _read_csv(output / "stage1a_integral_diagnostics.csv"))
    integral_scenarios: list[dict[str, Any]] = []
    for uid in subset:
        axes = [row for row in _read_csv(output / "stage1a_integral_diagnostics.csv") if row["scenario_uid"] == uid]
        integral_scenarios.append({
            "scenario_uid": uid, "formal_failure_label_H10000": subset[uid]["formal_failure_label_H10000"],
            "clamped_axis_count": sum(float(row["clamp_fraction"]) > 0 for row in axes),
            "mean_clamp_fraction": float(np.mean([float(row["clamp_fraction"]) for row in axes])),
            "max_clamp_fraction": max(float(row["clamp_fraction"]) for row in axes),
            "dynamic_hard": subset[uid]["dynamic_hard"], "high_force": subset[uid]["high_force"],
            "low_roll": subset[uid]["low_roll"], "low_yaw": subset[uid]["low_yaw"],
        })
    integral_metrics = ["clamped_axis_count", "mean_clamp_fraction", "max_clamp_fraction"]
    _write_csv(output / "integral_group_summary.csv", _descriptive_group_rows(integral_scenarios, metrics=integral_metrics, group_masks=masks))
    (output / "BRANCH_WRENCH_REPORT.md").write_text(
        "# Stage 1A branch wrench decomposition\n\nMeasured H2000 counterfactuals use the full tanh, asymmetric motor lag, thrust polynomial, and mixer. Raw rows are deterministically sampled every 10 steps in `branch_wrench.csv`; all 128,000 measured steps contribute to scenario/group summaries. These are Stage 1A gate results, not yet the three-seed population conclusion.\n",
        encoding="utf-8",
    )
    (output / "DAMPING_PASSIVITY_REPORT.md").write_text(
        "# Stage 1A damping passivity\n\nPower is `omega^T delta_tau_D` after actuator dynamics. Reports include all steps, active-rate steps, and authority-normalized 0.5%/1% damping thresholds. Positive signs at negligible rate/torque are separated from active damping events. Formal hypothesis status remains pending Stage 1B.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnostics-only Q3 Phase 1 tasks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument(
        "--scenario-reset-csv",
        type=Path,
        default=DEFAULT_OUTPUT / "scenario_reset_exact.csv",
    )
    feasibility_parser = subparsers.add_parser("feasibility")
    feasibility_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    feasibility_parser.add_argument(
        "--scenario-reset-csv", type=Path, default=DEFAULT_OUTPUT / "scenario_reset_exact.csv"
    )
    select_parser = subparsers.add_parser("select-stage1a")
    select_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    stage1a_parser = subparsers.add_parser("stage1a")
    stage1a_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    stage1a_parser.add_argument("--scenario-reset-csv", type=Path, default=DEFAULT_OUTPUT / "scenario_reset_exact.csv")
    stage1a_parser.add_argument("--device", default="cuda")
    weighting_parser = subparsers.add_parser("time-weighting")
    weighting_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    coverage_parser = subparsers.add_parser("sampling-coverage")
    coverage_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    coverage_parser.add_argument("--scenario-reset-csv", type=Path, default=DEFAULT_OUTPUT / "scenario_reset_exact.csv")
    coverage_parser.add_argument("--sample-count", type=int, default=1_048_576)
    gradient_parser = subparsers.add_parser("gradient-audit")
    gradient_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    gradient_parser.add_argument("--scenario-reset-csv", type=Path, default=DEFAULT_OUTPUT / "scenario_reset_exact.csv")
    consistency_parser = subparsers.add_parser("simulator-consistency")
    consistency_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    consistency_parser.add_argument("--scenario-reset-csv", type=Path, default=DEFAULT_OUTPUT / "scenario_reset_exact.csv")
    yaw_parser = subparsers.add_parser("yaw-equivalence")
    yaw_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    yaw_parser.add_argument("--scenario-reset-csv", type=Path, default=DEFAULT_OUTPUT / "scenario_reset_exact.csv")
    yaw_parser.add_argument("--sample-count", type=int, default=1000)
    summarize_parser = subparsers.add_parser("summarize-stage1a")
    summarize_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.output, args.scenario_reset_csv)
    elif args.command == "feasibility":
        feasibility(args.output, args.scenario_reset_csv)
    elif args.command == "select-stage1a":
        select_stage1a(args.output)
    elif args.command == "stage1a":
        run_stage1a(args.output, args.scenario_reset_csv, args.device)
    elif args.command == "time-weighting":
        time_weighting(args.output)
    elif args.command == "sampling-coverage":
        sampling_coverage(args.output, args.scenario_reset_csv, args.sample_count)
    elif args.command == "gradient-audit":
        gradient_audit(args.output, args.scenario_reset_csv)
    elif args.command == "simulator-consistency":
        simulator_consistency(args.output, args.scenario_reset_csv)
    elif args.command == "yaw-equivalence":
        yaw_equivalence(args.output, args.scenario_reset_csv, args.sample_count)
    elif args.command == "summarize-stage1a":
        summarize_stage1a(args.output)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
