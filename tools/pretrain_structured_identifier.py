"""Pretrain the deployable structured identifier from a frozen Q2 teacher.

This entry point exists solely to produce the production ``identifier_init``
artifact consumed by Phase A1.  It never constructs the historical causal
oracle GRU.  The teacher supplies the action that is executed in the
simulator, while capability values are simulator-side training targets and
never part of the policy observation.

The formal path is deliberately fail-closed: a frozen v4 probe record and a
hash-bound causal-oracle collector release report with all pretraining checks
passed must already exist before
the Q2 checkpoint is loaded or any optimizer is created.  ``--dry-run`` only
prints the resolved contract and performs no filesystem writes.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator  # noqa: E402
from structured_checkpoint import (  # noqa: E402
    CADENCE_SEMANTICS_VERSION,
    identifier_weights_hash,
    sha256_file,
    write_identifier_init_artifact,
)
from structured_distillation import (  # noqa: E402
    build_dagger_scenario_bank,
    collect_dagger_episode,
    dagger_window_loss,
    phase_a_equilibrium_gate,
    set_distillation_phase,
)
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy  # noqa: E402
from tools.diagnose_causal_identifier_oracle import (  # noqa: E402
    DEFAULT_PROBE_V4_REPORT,
    probe_v4_eligibility,
)
from probe_contract import WAVEFORM_SHA256  # noqa: E402


DEFAULT_Q2 = ROOT / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
DEFAULT_CAUSAL_ORACLE = ROOT / "runs/structured_pipeline/causal_identifier_oracle.json"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def validate_pretraining_gates(
    *,
    source_checkpoint: Path,
    probe_report: Path,
    causal_oracle_report: Path,
) -> dict[str, Any]:
    """Validate all immutable gates required before identifier pretraining.

    The causal report is intentionally checked independently of the v4 probe.
    In particular, an eligible probe with a failed causal gate cannot authorize
    training.  The returned mapping is JSON-safe and is copied into the run
    report for provenance.
    """

    source_checkpoint = Path(source_checkpoint)
    probe_report = Path(probe_report)
    causal_oracle_report = Path(causal_oracle_report)
    if not source_checkpoint.is_file():
        raise RuntimeError(f"Q2 source checkpoint is missing: {source_checkpoint}")
    probe_payload = _read_json(probe_report, "v4 probe report")
    probe = probe_v4_eligibility(
        probe_report, q2_checkpoint=source_checkpoint
    )
    formal = probe_payload.get("formal")
    if not isinstance(formal, Mapping):
        raise RuntimeError("v4 probe report has no formal freeze record")
    # probe_v4_eligibility checks the contract and frozen SHA.  Keep the gate
    # check explicit here because old debug reports contain eligible/frozen
    # fields but no release gate.
    frozen_sha = probe.get("frozen_sha256")
    formal_gate = formal.get("gate_passed")
    top_gate = probe_payload.get("gate_passed")
    if not probe.get("eligible") or frozen_sha != WAVEFORM_SHA256:
        raise RuntimeError(
            "identifier pretraining requires an eligible frozen v4 probe "
            f"(reason={probe.get('reason')})"
        )
    if formal.get("eligible") is not True:
        raise RuntimeError("identifier pretraining requires formal.eligible=true in v4 probe report")
    if formal_gate is False or top_gate is False or (formal_gate is not True and top_gate is not True):
        raise RuntimeError("identifier pretraining requires the v4 probe formal gate to pass")

    causal = _read_json(causal_oracle_report, "causal oracle report")
    required = ("diagnostic", "requested_formal_shape", "pretraining_gate_passed",
                "pretraining_checks", "checkpoint_sha256", "cadence_semantics_version")
    missing = [key for key in required if key not in causal]
    if missing:
        raise RuntimeError(
            "causal oracle report is missing release fields: " + ", ".join(missing)
        )
    if causal.get("diagnostic") != "causal-identifier-sequence-oracle":
        raise RuntimeError("causal oracle report has an unexpected diagnostic kind")
    # ``formal_eligible``/top-level ``gate_passed`` intentionally remain false
    # in the collector until a production sequence artifact exists.  The
    # pretrainer is that artifact's producer, so its upstream release gate is
    # the collector's immutable physics/K35/zero-parity gate instead.
    if causal.get("requested_formal_shape") is not True:
        raise RuntimeError("causal oracle report does not have the registered formal shape")
    if causal.get("pretraining_gate_passed") is not True:
        raise RuntimeError("causal oracle pretraining gate is not passed")
    checks = causal.get("pretraining_checks")
    required_checks = (
        "coverage", "physics_ceiling", "physics_finite", "physics_branch_support",
        "finite_collection", "no_identification_failure", "zero_q2_parity",
    )
    if not isinstance(checks, Mapping) or any(checks.get(key) is not True for key in required_checks):
        raise RuntimeError("causal oracle report is missing a passing K35/physics/zero-parity check")
    coverage = causal.get("coverage")
    ceiling = causal.get("ceiling")
    safety = causal.get("safety")
    parity = causal.get("zero_parity")
    if (not isinstance(coverage, Mapping) or coverage.get("K35_gate_passed") is not True
            or not isinstance(ceiling, Mapping) or ceiling.get("gate_passed") is not True
            or ceiling.get("tau_wls_finite") is not True
            or ceiling.get("effectiveness_finite") is not True
            or ceiling.get("tau_wls_supported") is not True
            or not isinstance(safety, Mapping) or safety.get("finite_collected") is not True
            or safety.get("no_identification_failure") is not True
            or not isinstance(parity, Mapping) or parity.get("gate_passed") is not True):
        raise RuntimeError("causal oracle K35/physics/zero-parity formal gate is not passed")
    if causal.get("cadence_semantics_version") != CADENCE_SEMANTICS_VERSION:
        raise RuntimeError("causal oracle report has stale cadence semantics")

    # Bind the causal report to this exact frozen Q2 file.  Importing the
    # shared hash helper avoids a second, subtly different digest definition.
    source_sha = sha256_file(source_checkpoint)
    if causal.get("checkpoint_sha256") != source_sha:
        raise RuntimeError("causal oracle report does not bind the Q2 source checkpoint")
    return {
        "probe_v4": {
            "path": str(probe_report.resolve()),
            "eligible": True,
            "formal_gate_passed": True,
            "frozen_sha256": frozen_sha,
            "contract_version": probe.get("contract_version"),
            "report_sha256": sha256_file(probe_report),
        },
        "causal_oracle": {
            "path": str(causal_oracle_report.resolve()),
            # These two collector fields remain false until a sequence
            # identifier artifact exists.  Preserve their real values rather
            # than relabeling the collector as a formal production oracle.
            "formal_eligible": bool(causal.get("formal_eligible", False)),
            "gate_passed": bool(causal.get("gate_passed", False)),
            "pretraining_gate_passed": True,
            "checkpoint_sha256": source_sha,
            "report_sha256": sha256_file(causal_oracle_report),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--output", type=Path, required=True,
                        help="production identifier_init.pt artifact")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--causal-oracle-report", "--identification-oracle-report",
                        dest="causal_oracle_report", type=Path, default=DEFAULT_CAUSAL_ORACLE)
    parser.add_argument("--probe-v4-report", "--probe-report", dest="probe_v4_report",
                        type=Path, default=DEFAULT_PROBE_V4_REPORT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=125)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--identifier-dim", type=int, default=64)
    parser.add_argument("--slow-cadence", type=int, default=25)
    parser.add_argument("--identification-publish-start", type=int, default=50)
    parser.add_argument("--motor-observer-bank-size", type=int, choices=(0, 15, 35), default=35)
    parser.add_argument("--motor-observer-mode", choices=("legacy", "fixed_multi_tau_v1", "fixed_multi_tau_v2"), default="fixed_multi_tau_v2")
    parser.add_argument("--motor-tau-grid-version", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--burn-in-probe-amplitude", type=float, default=0.005)
    parser.add_argument("--allocator-solver", choices=("smooth_dls", "box_qp"), default="box_qp")
    parser.add_argument("--allocator-rate-limit", type=float, default=50.0)
    parser.add_argument("--prefix", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve the command without validating inputs or writing files")
    return parser.parse_args(argv)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def _move_bank(bank, device: torch.device):
    if device.type == "cpu":
        return bank
    from env_l2f import L2FState
    state = L2FState(**{
        name: getattr(bank.state, name).to(device)
        for name in bank.state.__dataclass_fields__
    })
    return type(bank)(state, bank.tw_bin.to(device), bank.log_alpha_bin.to(device), bank.stratum)


def _config(args: argparse.Namespace, dt: float) -> StructuredPolicyConfig:
    bank_size = int(args.motor_observer_bank_size)
    mode = args.motor_observer_mode if bank_size else "legacy"
    grid_version = int(args.motor_tau_grid_version) if bank_size else 0
    return StructuredPolicyConfig(
        hidden_dim=args.hidden_dim, identifier_dim=args.identifier_dim, dt=dt,
        slow_cadence=args.slow_cadence,
        identification_publish_start=args.identification_publish_start,
        motor_observer_bank_size=bank_size,
        motor_observer_mode=mode,
        motor_tau_grid_version=grid_version,
        burn_in_probe_amplitude=args.burn_in_probe_amplitude,
        allocator_solver=args.allocator_solver,
        allocator_rate_limit=args.allocator_rate_limit,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        return {
            "stage": "identifier_pretrain",
            "dry_run": True,
            "source_checkpoint": str(args.source_checkpoint),
            "output": str(args.output),
            "report": str(args.report),
            "probe_v4_report": str(args.probe_v4_report),
            "causal_oracle_report": str(args.causal_oracle_report),
        }
    if args.horizon < 76 or args.prefix < 1 or args.prefix >= args.horizon:
        raise ValueError("horizon must include calls 50 and 75 and prefix must be inside it")
    if args.updates < 1 or args.scenarios < 16 or args.scenarios % 16:
        raise ValueError("scenarios must be a positive 4x4 bank and updates must be positive")
    # This is intentionally the first operation in a real run.  A failed
    # probe/oracle gate must not even initialize an optimizer or teacher.
    gate = validate_pretraining_gates(
        source_checkpoint=args.source_checkpoint,
        probe_report=args.probe_v4_report,
        causal_oracle_report=args.causal_oracle_report,
    )
    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    teacher, teacher_args = load_q2_policy(args.source_checkpoint, device=device, dtype=torch.float32)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    dt = float(teacher_args.get("dt", 0.01))
    config = _config(args, dt)
    student = StructuredRecurrentPolicy(config).to(device)
    active = set_distillation_phase(student, "A1")
    expected = tuple(name for name in student.state_dict()
                     if name.startswith(("identifier.", "bank_adapter.", "capability_head.")))
    if tuple(active) != expected or not active:
        raise RuntimeError("A1 pretraining parameter set is not exactly production identifier weights")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=1.0e-5,
    )
    history: list[dict[str, Any]] = []
    training_bank_seeds: list[int] = []
    simulator = L2FSimulator(L2FParams(dt=dt))
    for update in range(args.updates):
        # Use a fresh deterministic 4x4 authority-stratified bank for every
        # update.  Validation and final banks occupy disjoint seed ranges.
        bank_seed = args.seed + 100 + update
        training_bank_seeds.append(bank_seed)
        bank = _move_bank(build_dagger_scenario_bank(
            args.scenarios, seed=bank_seed, dt=dt,
            per_cell=args.scenarios // 16,
        ), device)
        episode = collect_dagger_episode(
            teacher, student, simulator,
            bank, beta=1.0, horizon=args.horizon,
            episode_seed=args.seed + 1000 + update,
        )
        # Phase A1 uses simulator capability targets and analytic equilibrium
        # targets.  It deliberately leaves the uncertainty scale frozen.
        loss, components = dagger_window_loss(
            student, episode, prefix=args.prefix, phase="A1",
            capability_weight=0.0, capability_mean_weight=1.0,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        if not bool(torch.isfinite(gradient_norm).item()) or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite identifier pretraining update")
        optimizer.step()
        teacher_steps = int(episode.intervention_mask.sum().item())
        expected_teacher_steps = int(episode.intervention_mask.numel())
        if teacher_steps != expected_teacher_steps:
            raise RuntimeError("identifier pretraining must execute the frozen teacher action")
        history.append({
            "update": update + 1, "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm.detach()), **components,
            "teacher_execution_count": teacher_steps,
            "student_execution_count": expected_teacher_steps - teacher_steps,
        })

    def evaluate_bank(seed: int) -> dict[str, Any]:
        episode = collect_dagger_episode(
            teacher, student, simulator, _move_bank(build_dagger_scenario_bank(
                args.scenarios, seed=seed, dt=dt, per_cell=args.scenarios // 16,
            ), device), beta=1.0, horizon=args.horizon, episode_seed=seed + 10000,
        )
        teacher_steps = int(episode.intervention_mask.sum().item())
        expected_steps = int(episode.intervention_mask.numel())
        with torch.no_grad():
            loss, components = dagger_window_loss(
                student, episode, prefix=args.prefix, phase="A1",
                capability_weight=0.0, capability_mean_weight=1.0,
            )
        if teacher_steps != expected_steps or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite independent identifier validation or teacher execution mismatch")
        mean_gate, mean_gate_passed = phase_a_equilibrium_gate(
            episode, require_identification_width=False,
        )
        return {
            "seed": seed, "scenario_count": args.scenarios,
            "teacher_execution_count": teacher_steps,
            "student_execution_count": expected_steps - teacher_steps,
            "loss": float(loss), "components": components,
            "finite": True, "teacher_action_is_executed": True,
            "phase_a_mean_gate": mean_gate,
            "phase_a_mean_gate_passed": bool(mean_gate_passed),
        }

    validation = evaluate_bank(args.seed + 10000)
    final = evaluate_bank(args.seed + 20000)
    formal_gate_passed = bool(
        validation["finite"] and final["finite"]
        and validation["phase_a_mean_gate_passed"]
        and final["phase_a_mean_gate_passed"]
    )
    if not formal_gate_passed:
        failure_payload: dict[str, Any] = {
            "stage": "identifier_pretrain",
            "diagnostic": "production-structured-identifier-pretraining",
            "status": "STOP_identifier_accuracy_gate_failed",
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
            "causal_oracle_report": gate["causal_oracle"],
            "probe_v4": gate["probe_v4"],
            "config": asdict(config),
            "device": str(device), "seed": args.seed,
            "scenario_count": args.scenarios, "horizon": args.horizon,
            "updates": args.updates, "history": history,
            "validation": validation, "final": final,
            "bank_seeds": {
                "train": training_bank_seeds,
                "validation": args.seed + 10000,
                "final": args.seed + 20000,
                "blind": [],
            },
            "active_parameter_names": list(active),
            "identifier_weights_sha256": identifier_weights_hash(student),
            "teacher_is_frozen": True,
            "teacher_action_is_executed": True,
            "runtime_features_from_deployable_observation_action_history": True,
            "capability_labels_used_as_runtime_features": False,
            "sidecar_oracle_gru": False,
            "independent_validation_passed": False,
            "independent_final_passed": False,
            "formal_gate_passed": False,
            "pretraining_gate_passed": False,
            "identifier_artifact_written": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(failure_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "production identifier failed independent calls50/75 mean-accuracy gates; "
            "no identifier artifact was written"
        )
    artifact = write_identifier_init_artifact(
        student, args.output, q2_checkpoint=args.source_checkpoint,
        causal_oracle_report=args.causal_oracle_report,
    )
    payload: dict[str, Any] = {
        "stage": "identifier_pretrain",
        "diagnostic": "production-structured-identifier-pretraining",
        "artifact_type": artifact["artifact_type"],
        "artifact_schema_version": artifact["artifact_schema_version"],
        "identifier_init_artifact": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "identifier_weights_sha256": artifact["identifier_weights_sha256"],
        },
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": artifact["q2_checkpoint_sha256"],
        "causal_oracle_report": gate["causal_oracle"],
        "probe_v4": gate["probe_v4"],
        "config": asdict(config),
        "device": str(device), "seed": args.seed,
        "scenario_count": args.scenarios, "horizon": args.horizon,
        "updates": args.updates, "history": history,
        "validation": validation, "final": final,
        "bank_seeds": {"train": training_bank_seeds,
                       "validation": args.seed + 10000,
                       "final": args.seed + 20000, "blind": []},
        "active_parameter_names": list(active),
        "identifier_weights_sha256": identifier_weights_hash(student),
        "teacher_is_frozen": True,
        "teacher_action_is_executed": True,
        "runtime_features_from_deployable_observation_action_history": True,
        "capability_labels_used_as_runtime_features": False,
        "sidecar_oracle_gru": False,
        "independent_validation_passed": bool(
            validation["finite"] and validation["phase_a_mean_gate_passed"]
        ),
        "independent_final_passed": bool(
            final["finite"] and final["phase_a_mean_gate_passed"]
        ),
        "formal_gate_passed": formal_gate_passed,
        "pretraining_gate_passed": formal_gate_passed,
        "identifier_artifact_written": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"identifier pretraining stopped: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in
                      ("stage", "dry_run", "formal_gate_passed", "pretraining_gate_passed")
                      if key in result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
