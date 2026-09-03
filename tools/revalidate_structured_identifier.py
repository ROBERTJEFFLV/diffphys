"""Revalidate the deployable identifier after Phase-A1 changes.

Phase A1 changes the production identifier weights, so the immutable causal
probe that authorized its initialization is no longer sufficient.  This
read-only stage runs two fresh, authority-stratified banks: teacher-forced
(``beta=1``) and fully on-policy (``beta=0``).  It never trains, consumes no
blind bank, and writes a passing report only when both banks satisfy the
registered Phase-A physical/equilibrium gate.
"""
from __future__ import annotations

import argparse
import json
import sys
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
    IDENTIFIER_ARTIFACT_SCHEMA_VERSION,
    identifier_weights_hash,
    load_identifier_init_artifact,
    sha256_file,
    validate_identifier_pretraining_report,
)
from structured_distillation import (  # noqa: E402
    build_dagger_scenario_bank,
    collect_dagger_episode,
    phase_a_equilibrium_gate,
)
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy  # noqa: E402
from structured_rollout import load_structured_policy  # noqa: E402
from identification_features import feature_schema_sha256  # noqa: E402
from probe_contract import PROBE_CONTRACT_VERSION, WAVEFORM_SHA256  # noqa: E402
from tools.diagnose_causal_identifier_oracle import probe_v4_eligibility  # noqa: E402


DEFAULT_Q2 = ROOT / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
DEFAULT_WORK = ROOT / "runs/structured_pipeline"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _move_bank(bank, device: torch.device):
    if device.type == "cpu":
        return bank
    from env_l2f import L2FState
    state = L2FState(**{
        name: getattr(bank.state, name).to(device)
        for name in bank.state.__dataclass_fields__
    })
    return type(bank)(state, bank.tw_bin.to(device), bank.log_alpha_bin.to(device), bank.stratum)


def _strata(bank) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in bank.stratum:
        counts[name] = counts.get(name, 0) + 1
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a1-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-a1-report", type=Path, required=True)
    parser.add_argument("--identifier-init-artifact", type=Path, required=True)
    parser.add_argument("--identifier-pretrain-report", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe-v4-report", type=Path, default=ROOT / "reports/probe_v4_formal.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=125)
    parser.add_argument("--teacher-forced-seed-offset", type=int, default=30001)
    parser.add_argument("--on-policy-seed-offset", type=int, default=40002)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    probe = probe_v4_eligibility(
        args.probe_v4_report, q2_checkpoint=args.source_checkpoint
    )
    if not probe.get("eligible"):
        raise RuntimeError(
            "causal revalidation requires the current formal v4 freeze record "
            f"(reason={probe.get('reason')})"
        )
    a1 = _read_json(args.phase_a1_report, "Phase A1 report")
    pretrain = _read_json(args.identifier_pretrain_report, "identifier pretraining report")
    if a1.get("active_phase") != "A1":
        raise RuntimeError("causal revalidation requires an active Phase-A1 checkpoint")
    if a1.get("causal_revalidation_required") is not True:
        raise RuntimeError("Phase A1 did not mark its identifier causal gate stale")
    if a1.get("causal_gate_status") != "stale_revalidation_required":
        raise RuntimeError("Phase A1 causal gate status is not stale_revalidation_required")
    if a1.get("phase_a_mean_gate_passed") is not True:
        raise RuntimeError("causal revalidation requires a passed A1 physical-mean gate")
    if pretrain.get("stage") != "identifier_pretrain" or pretrain.get("pretraining_gate_passed") is not True:
        raise RuntimeError("identifier pretraining report is not a passed production artifact report")
    if not args.source_checkpoint.is_file() or not args.phase_a1_checkpoint.is_file():
        raise RuntimeError("Q2 and Phase-A1 checkpoints must both exist")
    if Path(str(a1.get("source_checkpoint", ""))).resolve() != args.source_checkpoint.resolve():
        raise RuntimeError("Phase A1 report is not bound to the requested Q2 checkpoint")
    artifact_record = pretrain.get("identifier_init_artifact")
    if not isinstance(artifact_record, Mapping):
        raise RuntimeError("pretraining report has no identifier artifact binding")
    if Path(str(artifact_record.get("path", ""))).resolve() != args.identifier_init_artifact.resolve():
        raise RuntimeError("requested identifier artifact differs from pretraining report")
    artifact_sha = sha256_file(args.identifier_init_artifact)
    if artifact_record.get("sha256") != artifact_sha:
        raise RuntimeError("identifier artifact hash differs from pretraining report")
    a1_artifact = a1.get("identifier_init_artifact")
    if not isinstance(a1_artifact, Mapping) or a1_artifact.get("sha256") != artifact_sha:
        raise RuntimeError("Phase A1 did not carry the exact identifier artifact hash")
    return a1, pretrain, {"artifact_sha256": artifact_sha}


def _validate_artifact_binding(
    args: argparse.Namespace,
    student: StructuredRecurrentPolicy,
    pretrain: Mapping[str, Any],
) -> dict[str, Any]:
    # Install the artifact into an isolated policy solely to exercise the
    # canonical fail-closed loader.  The A1 policy itself is never modified.
    isolated = StructuredRecurrentPolicy(student.config).to(device="cpu")
    loaded = load_identifier_init_artifact(
        args.identifier_init_artifact, isolated,
        q2_checkpoint=args.source_checkpoint,
        causal_oracle_report=Path(str(pretrain["causal_oracle_report"]["path"])),
    )
    payload = torch.load(args.identifier_init_artifact, map_location="cpu", weights_only=False)
    if payload.get("artifact_schema_version") != IDENTIFIER_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("identifier artifact schema is stale")
    if payload.get("cadence_semantics_version") != CADENCE_SEMANTICS_VERSION:
        raise RuntimeError("identifier artifact cadence semantics are stale")
    if payload.get("q2_checkpoint_sha256") != sha256_file(args.source_checkpoint):
        raise RuntimeError("identifier artifact is not bound to Q2")
    if payload.get("feature_schema_sha256") != feature_schema_sha256():
        raise RuntimeError("identifier artifact feature schema is stale")
    probe = payload.get("probe")
    if not isinstance(probe, Mapping) or probe.get("contract_version") != PROBE_CONTRACT_VERSION \
            or probe.get("waveform_sha256") != WAVEFORM_SHA256:
        raise RuntimeError("identifier artifact probe contract is stale")
    expected_hash = pretrain.get("identifier_weights_sha256")
    if expected_hash != payload.get("identifier_weights_sha256"):
        raise RuntimeError("pretraining report and identifier artifact weight hashes disagree")
    return {
        **loaded,
        "artifact_sha256": sha256_file(args.identifier_init_artifact),
        "q2_checkpoint_sha256": payload["q2_checkpoint_sha256"],
        "causal_oracle_report_sha256": payload.get("causal_oracle_report_sha256"),
        "probe_contract_version": probe["contract_version"],
        "probe_waveform_sha256": probe["waveform_sha256"],
        "feature_schema_sha256": payload["feature_schema_sha256"],
    }


def _bank_result(
    teacher, student, simulator, bank, *, beta: float, horizon: int, episode_seed: int,
) -> dict[str, Any]:
    with torch.no_grad():
        episode = collect_dagger_episode(
            teacher, student, simulator, bank, beta=beta,
            horizon=horizon, episode_seed=episode_seed,
        )
        gate, passed = phase_a_equilibrium_gate(
            episode, phase_steps=(50, 75), require_identification_width=False,
        )
    teacher_count = int(episode.intervention_mask.sum().item())
    total = int(episode.intervention_mask.numel())
    finite = bool(episode.finite.all().item())
    failure_count = int(episode.identification_failure_t50.sum().item())
    if beta == 1.0 and teacher_count != total:
        raise RuntimeError("teacher-forced bank did not execute the frozen Q2 action")
    if beta == 0.0 and teacher_count != 0:
        raise RuntimeError("on-policy bank unexpectedly executed teacher actions")
    return {
        "beta": float(beta),
        "execution_mode": "teacher_forced" if beta == 1.0 else "on_policy",
        "episode_seed": int(episode_seed),
        "scenario_count": int(bank.count),
        "authority_strata": _strata(bank),
        "teacher_execution_count": teacher_count,
        "student_execution_count": total - teacher_count,
        "finite": finite,
        "identification_failure_t50": failure_count,
        "capability_calls_50_75": [
            {
                "call": int(row["phase_step"]),
                "capability_z_rms": float(row["capability_z_rms"]),
                "effectiveness_z_rms": float(row["effectiveness_z_rms"]),
                "capability_axis_z_rms": row["capability_axis_z_rms"],
                "passed": bool(row["passed"]),
            }
            for row in gate["rows"]
        ],
        "phase_a_gate": gate,
        "phase_a_gate_passed": bool(passed and finite and failure_count == 0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        return {"stage": "identifier_revalidation", "dry_run": True,
                "output": str(args.output), "report": str(args.report)}
    if args.scenarios < 16 or args.scenarios % 16 or args.horizon < 76:
        raise ValueError("scenarios must be a 4x4 bank and horizon must include calls 50/75")
    a1_report, pretrain, _ = _validate_inputs(args)
    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    student, a1_payload = load_structured_policy(args.phase_a1_checkpoint, device=device)
    embedded_a1_report = a1_payload.get("report", {})
    if embedded_a1_report.get("identifier_weights_sha256_after") != identifier_weights_hash(student):
        raise RuntimeError("Phase A1 checkpoint identifier hash does not match its report")
    # Torch payloads may retain tuples while JSON turns them into lists.  A
    # JSON round trip gives a canonical semantic comparison without trusting
    # a same-named but unrelated external report.
    if json.dumps(
        embedded_a1_report, sort_keys=True, separators=(",", ":")
    ) != json.dumps(a1_report, sort_keys=True, separators=(",", ":")):
        raise RuntimeError("external Phase A1 report differs from the report embedded in checkpoint")
    artifact = _validate_artifact_binding(args, student, pretrain)
    causal_report_path = Path(str(pretrain["causal_oracle_report"]["path"]))
    pretraining_contract = validate_identifier_pretraining_report(
        args.identifier_pretrain_report,
        identifier_artifact=args.identifier_init_artifact,
        identifier_contract=artifact,
        policy=student,
        q2_checkpoint=args.source_checkpoint,
        causal_oracle_report=causal_report_path,
        probe_v4_report=args.probe_v4_report,
    )
    probe_record = pretrain.get("probe_v4")
    if (not isinstance(probe_record, Mapping)
            or probe_record.get("report_sha256") != sha256_file(args.probe_v4_report)):
        raise RuntimeError("revalidation probe-v4 report differs from identifier pretraining")
    teacher, teacher_args = load_q2_policy(args.source_checkpoint, device=device, dtype=torch.float32)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.eval()
    simulator = L2FSimulator(L2FParams(dt=float(teacher_args.get("dt", student.config.dt))))
    forced_bank = _move_bank(build_dagger_scenario_bank(
        args.scenarios, seed=args.seed + args.teacher_forced_seed_offset,
        dt=simulator.params.dt, per_cell=args.scenarios // 16,
    ), device)
    on_policy_bank = _move_bank(build_dagger_scenario_bank(
        args.scenarios, seed=args.seed + args.on_policy_seed_offset,
        dt=simulator.params.dt, per_cell=args.scenarios // 16,
    ), device)
    forced = _bank_result(
        teacher, student, simulator, forced_bank, beta=1.0,
        horizon=args.horizon, episode_seed=args.seed + 50001,
    )
    on_policy = _bank_result(
        teacher, student, simulator, on_policy_bank, beta=0.0,
        horizon=args.horizon, episode_seed=args.seed + 60002,
    )
    passed = bool(forced["phase_a_gate_passed"] and on_policy["phase_a_gate_passed"])
    payload: dict[str, Any] = {
        "stage": "identifier_revalidation",
        "diagnostic": "causal-identifier-revalidation",
        "causal_gate_revalidated": passed,
        "formal_gate_passed": passed,
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": a1_report.get("cadence_semantics"),
        "post_a1_checkpoint": str(args.phase_a1_checkpoint.resolve()),
        "post_a1_checkpoint_sha256": sha256_file(args.phase_a1_checkpoint),
        "phase_a1_report_sha256": sha256_file(args.phase_a1_report),
        "identifier_init_artifact": artifact,
        "identifier_weights_sha256": identifier_weights_hash(student),
        "identifier_pretrain_report": str(args.identifier_pretrain_report.resolve()),
        "identifier_pretrain_report_sha256": sha256_file(args.identifier_pretrain_report),
        "identifier_pretraining_contract": pretraining_contract,
        "q2_checkpoint": str(args.source_checkpoint.resolve()),
        "q2_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "probe_v4_report": str(args.probe_v4_report.resolve()),
        "probe_v4_report_sha256": sha256_file(args.probe_v4_report),
        "probe_contract_version": PROBE_CONTRACT_VERSION,
        "probe_waveform_sha256": WAVEFORM_SHA256,
        "feature_schema_sha256": feature_schema_sha256(),
        "device": str(device), "seed": int(args.seed), "horizon": int(args.horizon),
        "scenario_count_per_bank": int(args.scenarios),
        "bank_seeds": {
            "teacher_forced": int(args.seed + args.teacher_forced_seed_offset),
            "on_policy": int(args.seed + args.on_policy_seed_offset),
            "blind": [],
        },
        "banks": {"teacher_forced": forced, "on_policy": on_policy},
        "teacher_is_frozen": True,
        "capability_publication_calls": [50, 75],
        "revalidation_contract": "both independent authority-stratified teacher-forced and beta0/on-policy banks must pass phase-A mean/equilibrium gate",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("causal identifier revalidation gate failed; A2 remains blocked")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "structured-recurrent-motor-policy",
        "diagnostic": "causal-identifier-revalidation",
        "report": payload,
    }, args.output)
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"identifier revalidation stopped: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("stage", "dry_run", "causal_gate_revalidated") if key in result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
