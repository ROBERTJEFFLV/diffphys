"""Honest v4 shared-tau contract layered on the v3 read-only diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_contract import (  # noqa: E402
    PROBE_AMPLITUDE,
    PROBE_CONTRACT_VERSION,
    PROBE_ENTRIES,
    PROBE_PERIOD,
    SHARED_TAU_MIN_CALLS,
    SHARED_TAU_MIN_MOTORS,
    SHARED_TAU_MIN_POOLED,
    SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION,
    WAVEFORM,
    WAVEFORM_SHA256,
    canonical_waveform_json,
    waveform_metadata,
)
from tools.diagnose_probe_v3 import (  # noqa: E402
    _rollout_one,
    _formal_gate,
    modal_coordinates,
    projected_coriolis_residual_ratio,
    standardized_rank_condition,
    validate_waveform,
)
TRAIN_SEEDS = (3707, 4707, 5707, 6707)
VALIDATION_SEED = 7707
FORMAL_SCENARIOS = 16
FORMAL_HORIZON = 125
DEFAULT_Q2_CHECKPOINT = ROOT / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_v4_candidate_family() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return the bounded deterministic family, sourced from v4 artifact."""
    return (WAVEFORM,
            tuple(tuple(-value for value in row) for row in WAVEFORM),
            tuple(reversed(WAVEFORM)))


def validate_shared_tau_configuration(
    rising: torch.Tensor, falling: torch.Tensor, *, tolerance: float = 1.0e-8
) -> dict[str, Any]:
    """Reject per-motor tau configurations; v4 requires one tau per branch."""
    if rising.shape != falling.shape or rising.ndim not in (1, 2):
        raise ValueError("rising and falling tau must have shape [4] or [batch,4]")
    if rising.shape[-1] != 4 or not bool(torch.isfinite(rising).all() and torch.isfinite(falling).all()):
        raise ValueError("tau values must be finite and have four motors")
    rise_spread = float((rising - rising[..., :1]).abs().max())
    fall_spread = float((falling - falling[..., :1]).abs().max())
    passed = rise_spread <= tolerance and fall_spread <= tolerance
    return {"shared": passed, "rising_spread": rise_spread, "falling_spread": fall_spread,
            "tolerance": tolerance, "gate_passed": passed}


assert_shared_tau_configuration = validate_shared_tau_configuration


def shared_tau_diagnostics(
    command: torch.Tensor,
    motor_before: torch.Tensor,
    *,
    motor_after: torch.Tensor | None = None,
    amplitude: float = PROBE_AMPLITUDE,
    support_steps: int = PROBE_PERIOD,
) -> dict[str, Any]:
    """Compute pooled branch support and weighted Fisher information per scene."""
    if command.shape != motor_before.shape or command.ndim != 3 or command.shape[-1] != 4:
        raise ValueError("command and motor_before must have shape [time,scene,4]")
    if motor_after is not None and motor_after.shape != command.shape:
        raise ValueError("motor_after must match command shape")
    if command.shape[0] < support_steps or support_steps < PROBE_PERIOD:
        raise ValueError("shared-tau diagnostics require the 50-step probe")
    u, m = command[:support_steps].double(), motor_before[:support_steps].double()
    delta = u - m
    threshold = float(amplitude) / 2.0
    rows: list[dict[str, Any]] = []
    for scene in range(u.shape[1]):
        scene_row: dict[str, Any] = {"scene": scene, "branches": {}}
        for name, branch in (("rise", delta >= 0), ("fall", delta < 0)):
            valid = branch & (delta.abs() >= threshold)
            scene_valid = valid[:, scene]
            counts = scene_valid.sum(0)
            calls = int(scene_valid.any(-1).sum())
            motors = int((counts > 0).sum())
            excitation_sq = (delta[:, scene].square() * scene_valid).sum()
            # This is the diagonal Fisher information for a first-order
            # transition in tau, weighted by actual excitation magnitude.
            weighted_information = (delta[:, scene].square() * delta[:, scene].abs() * scene_valid).sum()
            tau_estimate = None
            if motor_after is not None:
                dm = motor_after[:support_steps, scene].double() - m[:, scene]
                valid_tau = valid[:, scene] & (dm.abs() > 1.0e-10)
                tau_values = (0.01 * delta[:, scene] / dm.clamp_min(1.0e-10).where(
                    dm >= 0, dm.clamp_max(-1.0e-10)
                )).clamp(1.0e-4, 10.0)
                weights = delta[:, scene].abs() * valid_tau
                tau_estimate = float((tau_values * weights).sum() / weights.sum().clamp_min(1.0e-12))
            gate = (int(counts.sum()) >= SHARED_TAU_MIN_POOLED and
                    calls >= SHARED_TAU_MIN_CALLS and motors >= SHARED_TAU_MIN_MOTORS and
                    float(excitation_sq) >= SHARED_TAU_MIN_POOLED * threshold ** 2 and
                    float(weighted_information) >= SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION)
            scene_row["branches"][name] = {
                "pooled_valid": int(counts.sum()), "distinct_time_calls": calls,
                "motors_contributing": motors, "per_motor_counts": counts.tolist(),
                "excitation_sq": float(excitation_sq),
                "weighted_fisher_information": float(weighted_information),
                "tau_estimate": tau_estimate, "gate_passed": bool(gate),
            }
        scene_row["gate_passed"] = bool(all(value["gate_passed"] for value in scene_row["branches"].values()))
        rows.append(scene_row)
    return {"threshold": threshold, "min_pooled": SHARED_TAU_MIN_POOLED,
            "min_distinct_calls": SHARED_TAU_MIN_CALLS, "min_motors": SHARED_TAU_MIN_MOTORS,
            "min_weighted_fisher_information": SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION,
            "per_scene": rows, "gate_passed": bool(rows) and all(row["gate_passed"] for row in rows)}


def v4_design_report() -> dict[str, Any]:
    checks = validate_waveform(WAVEFORM)
    checks["contract_version"] = PROBE_CONTRACT_VERSION
    checks["waveform_sha256"] = WAVEFORM_SHA256
    return checks


def _v4_gate(rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Apply v3 checks, replacing only per-motor support with shared-tau gates."""
    base_pass, checks = _formal_gate(rows)
    for item, row in zip(checks["seed_results"], rows):
        item["checks"].pop("support_each_motor_rise_fall_ge_3", None)
        item["checks"].pop("physical_support_upper_bound_ge_3", None)
        raw = row["probe"]["raw_transition"]
        command = torch.tensor(raw["command"], dtype=torch.float64)
        before = torch.tensor(raw["motor_before"], dtype=torch.float64)
        after = torch.tensor(raw["motor_after"], dtype=torch.float64)
        tau = shared_tau_diagnostics(command, before, motor_after=after)
        item["shared_tau"] = tau
        physical_tau = raw.get("physical_tau", {})
        item["exact_tau_ceiling"] = {
            "physical_rising_max": max(physical_tau.get("rising", [float("nan")]), default=float("nan")),
            "physical_falling_max": max(physical_tau.get("falling", [float("nan")]), default=float("nan")),
            "per_scene_branch_estimates": [
                {name: value["tau_estimate"] for name, value in scene["branches"].items()}
                for scene in tau["per_scene"]
            ],
        }
        item["checks"]["shared_tau_pooled_time_motor_fisher"] = bool(tau["gate_passed"])
        item["passed"] = bool(all(item["checks"].values()))
        probe, zero = row["probe"], row["zero"]
        paired_metrics: dict[str, Any] = {}
        for section in ("h75", "h125", "tail_h75_h125"):
            for metric in ("position", "velocity", "omega"):
                for statistic in ("mean", "p99"):
                    name = f"paired_{section}_{metric}_{statistic}"
                    actual = float(probe[section][metric][statistic])
                    baseline = float(zero[section][metric][statistic])
                    allowed = 1.05 * baseline + 1.0e-7
                    paired_metrics[name] = {
                        "actual": actual,
                        "q2_zero": baseline,
                        "allowed": allowed,
                        "actual_over_q2": actual / max(baseline, 1.0e-12),
                        "passed": actual <= allowed,
                    }
        peak_metrics: dict[str, Any] = {}
        for metric, absolute in (("position", 5.0), ("velocity", 20.0), ("omega", 5.0)):
            name = f"peak_{metric}"
            actual = float(probe["stats"][f"{metric}_max"])
            baseline = float(zero["stats"][f"{metric}_max"])
            allowed = max(absolute, 1.5 * baseline)
            peak_metrics[name] = {
                "actual": actual,
                "q2_zero": baseline,
                "absolute_floor": absolute,
                "allowed": allowed,
                "passed": actual <= allowed,
            }
        item["safety_metrics"] = {
            "paired": paired_metrics,
            "peaks": peak_metrics,
            "energy_retention_modal_min": min(
                value for scene in probe["energy_retention_modal"] for value in scene
            ),
            "standardized_X": probe["standardized_X"],
            "coriolis": probe["coriolis"],
            "zero_arm_canonical_parity_max_abs": row["zero_arm_canonical_parity_max_abs"],
        }
    return bool(rows) and all(item["passed"] for item in checks["seed_results"]), checks


def run(args: argparse.Namespace) -> dict[str, Any]:
    design = v4_design_report()
    family = generate_v4_candidate_family()
    report: dict[str, Any] = {
        "diagnostic": "probe-v4-shared-tau-q2-candidate-scoring",
        "contract": waveform_metadata(), "design": design,
        "seed_split": {"train": list(TRAIN_SEEDS), "validation": [VALIDATION_SEED],
                        "blind_consumed": []},
        "candidate_count": len(family), "formal_frozen_sha256": None,
        "source_checkpoint": None, "source_checkpoint_sha256": None,
        "producer_code_sha256": _hash_file(Path(__file__)),
        "protocol": {
            "scenarios": int(args.scenarios),
            "horizon": int(args.horizon),
            "formal_scenarios": FORMAL_SCENARIOS,
            "formal_horizon": FORMAL_HORIZON,
        },
        "formal": {"eligible": False, "frozen_sha256": None},
        "gate_passed": False,
    }
    if args.dry_run:
        report["status"] = "dry_run_static_candidate_not_frozen"
        return report
    if args.checkpoint is None or not args.checkpoint.is_file():
        report["status"] = "STOP_no_frozen_candidate"
        report["failure"] = "frozen Q2 checkpoint is required"
        return report
    report["source_checkpoint"] = str(args.checkpoint.resolve())
    report["source_checkpoint_sha256"] = _hash_file(args.checkpoint)
    if args.scenarios < 16 or args.scenarios % 16:
        raise ValueError("scenarios must be a positive multiple of 16")
    if bool(getattr(args, "formal_freeze", False)) and (
        args.scenarios != FORMAL_SCENARIOS or args.horizon != FORMAL_HORIZON
    ):
        report["status"] = "STOP_formal_protocol_mismatch"
        report["failure"] = (
            f"formal freeze requires scenarios={FORMAL_SCENARIOS} and "
            f"horizon={FORMAL_HORIZON}"
        )
        return report
    work = [(index, seed, table) for index, table in enumerate(family) for seed in TRAIN_SEEDS]
    def evaluate(index: int, seed: int, table: tuple[tuple[int, ...], ...]):
        return index, _rollout_one(args.checkpoint, seed, args.scenarios, args.horizon,
                                   torch.tensor(table, dtype=torch.float32))
    if Parallel is not None and args.n_jobs > 1:
        evaluated = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(evaluate)(*item) for item in work)
    else:
        evaluated = [evaluate(*item) for item in work]
    scores = []
    by_index = {index: [row for idx, row in evaluated if idx == index] for index in range(len(family))}
    for index, table in enumerate(family):
        passed, checks = _v4_gate(by_index[index])
        supports = [value for row in by_index[index] for branch in (row["probe"]["rise_support"], row["probe"]["fall_support"])
                    for scene in branch for value in scene]
        cond = max(row["probe"]["standardized_X"]["condition_max"] for row in by_index[index])
        scores.append({"index": index, "sha256": hashlib.sha256(canonical_waveform_json(table)).hexdigest(),
                       "train_gate_passed": passed, "min_actual_support": min(supports),
                       "worst_X_condition": cond, "checks": checks})
    report["candidate_scores"] = scores
    passers = [item for item in scores if item["train_gate_passed"]]
    if not passers:
        report["status"] = "STOP_no_frozen_candidate"
        report["failure"] = "no candidate passed all train shared-tau and safety gates"
        return report
    selected = min(passers, key=lambda item: (item["worst_X_condition"], item["sha256"]))
    val_index = selected["index"]
    val_row = _rollout_one(args.checkpoint, VALIDATION_SEED, args.scenarios, args.horizon,
                           torch.tensor(family[val_index], dtype=torch.float32))
    val_pass, val_checks = _v4_gate([val_row])
    report["validation"] = {"index": val_index, "gate_passed": val_pass, "checks": val_checks}
    if val_pass:
        sha = selected["sha256"]
        report["formal_frozen_sha256"] = sha
        if bool(getattr(args, "formal_freeze", False)):
            # Formal promotion is allowed only for the waveform already
            # registered in probe_contract.py.  Candidate search cannot
            # silently replace that contract after observing outcomes.
            if sha != WAVEFORM_SHA256:
                report["status"] = "STOP_selected_waveform_not_registered_contract"
                report["failure"] = (
                    "validation winner does not match the pre-registered v4 waveform"
                )
                report["formal"] = {"eligible": False, "frozen_sha256": None}
                report["formal_frozen_sha256"] = None
            else:
                report["status"] = "formal_frozen"
                report["gate_passed"] = True
                report["formal"] = {
                    "eligible": True,
                    "frozen_sha256": sha,
                    "gate_passed": True,
                    "selection_train_gate_passed": True,
                    "independent_validation_gate_passed": True,
                    "blind_consumed": [],
                }
        else:
            report["status"] = "debug_selected_not_formal"
            report["formal"] = {"eligible": False, "frozen_sha256": sha,
                                 "debug_selected": True}
    else:
        report["status"] = "STOP_no_frozen_candidate"
        report["failure"] = "selected train candidate failed validation 7707"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_Q2_CHECKPOINT)
    parser.add_argument("--scenarios", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=125)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument(
        "--formal-freeze", action="store_true",
        help=(
            "freeze only if train+validation pass and the winner exactly matches "
            "the pre-registered probe_contract waveform; never consumes blind seeds"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostic": report["diagnostic"],
                      "frozen_sha256": report.get("formal_frozen_sha256"),
                      "gate_passed": report.get("gate_passed", False),
                      "failure": report.get("failure")}, sort_keys=True))
    # A formal-freeze invocation is a release gate, not just a diagnostic
    # report generator.  Keep the report for auditability, but make shell
    # pipelines fail closed unless the pre-registered waveform was actually
    # frozen after passing both the train and independent-validation gates.
    if args.formal_freeze and (
        report.get("status") != "formal_frozen"
        or report.get("gate_passed") is not True
        or report.get("formal", {}).get("eligible") is not True
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
