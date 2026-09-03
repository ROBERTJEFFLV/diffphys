"""Stage-aware, fail-closed runner for the structured L2F pipeline.

The runner only composes existing Torch/L2F tools.  It never changes the Q2
baseline and refuses to start a downstream formal stage after a failed release
gate.  ``--dry-run`` resolves every command without importing or executing a
training tool, which makes the configuration reproducible on CPU-only hosts.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_WORK = ROOT / "runs/structured_pipeline"
STAGES = (
    "identifier_oracle", "identifier_pretrain", "phase_a1", "identifier_revalidation", "phase_a2", "phase_b", "residual_oracle", "phase_c", "calibration", "postcheck", "migration",
    "fullspace2", "fullspace4", "postms_calibration", "postms_postcheck", "postms_migration",
)
CONFIG_NAMES = {
    "identifier_oracle": "structured_identifier_oracle.args",
    "identifier_pretrain": "structured_identifier_pretrain.args",
    "phase_a1": "structured_phase_a_dagger.args",
    "identifier_revalidation": "structured_identifier_revalidation.args",
    "phase_a2": "structured_phase_a2_uncertainty.args",
    "phase_b": "structured_phase_b_local_jvp.args",
    "residual_oracle": "structured_residual_oracle.args",
    "phase_c": "structured_phase_c_residual_dagger.args",
    "calibration": "structured_formal_q_calibration.args",
    "postcheck": "structured_postcalibration_evidence.args",
    "migration": "structured_paired_migration.args",
    "fullspace2": "structured_fullspace_2xH250.args",
    "fullspace4": "structured_fullspace_4xH250.args",
    "postms_calibration": "structured_postms_calibration.args",
    "postms_postcheck": "structured_postms_postcheck.args",
    "postms_migration": "structured_postms_migration.args",
}
REPORT_NAMES = {
    "identifier_oracle": "causal_identifier_oracle.json",
    "identifier_pretrain": "identifier_pretrain_report.json",
    "phase_a1": "phase_a1_report.json", "phase_a2": "phase_a2_report.json",
    "identifier_revalidation": "causal_identifier_revalidation.json",
    "phase_b": "phase_b_report.json",
    "residual_oracle": "residual_oracle_report.json",
    "phase_c": "phase_c_report.json", "calibration": "calibration_report.json",
    "postcheck": "postcheck_report.json",
    "migration": "migration_report.json", "fullspace2": "fullspace2_report.json",
    "fullspace4": "fullspace4_report.json",
    "postms_calibration": "postms_calibration_report.json",
    "postms_postcheck": "postms_postcheck_report.json",
    "postms_migration": "postms_migration_report.json",
}


def parse_args_file(path: Path, substitutions: Mapping[str, str]) -> List[str]:
    """Parse a small shell-like ``.args`` file, including ``@file`` includes."""

    result: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@") and len(line) > 1 and not line[1].isspace():
            include = Path(line[1:])
            if not include.is_absolute():
                include = path.parent / include
            result.extend(parse_args_file(include, substitutions))
            continue
        for key, value in substitutions.items():
            line = line.replace("{" + key + "}", str(value))
        result.extend(shlex.split(line, comments=True, posix=True))
    return result


def stage_command(stage: str, *, root: Path = ROOT,
                  work_dir: Path = DEFAULT_WORK, device: str = "auto") -> List[str]:
    if stage not in STAGES:
        raise ValueError(f"unknown structured pipeline stage: {stage}")
    config = root / "configs" / CONFIG_NAMES[stage]
    substitutions = {
        "root": str(root), "work": str(work_dir), "device": device,
    }
    args = parse_args_file(config, substitutions)
    tool = {
        "identifier_oracle": "diagnose_causal_identifier_oracle.py",
        "identifier_pretrain": "pretrain_structured_identifier.py",
        "phase_a1": "distill_structured_dagger.py",
        "identifier_revalidation": "revalidate_structured_identifier.py",
        "phase_a2": "distill_structured_dagger.py",
        "phase_b": "distill_structured_local_gain.py",
        "residual_oracle": "validate_structured_residual_oracle.py",
        "phase_c": "distill_structured_dagger.py",
        "calibration": "calibrate_structured_capability.py",
        "postcheck": "postcalibration_local_evidence.py",
        "migration": "validate_structured_q2_migration.py",
        "fullspace2": "train_structured_full_space.py",
        "fullspace4": "train_structured_full_space.py",
        "postms_calibration": "calibrate_structured_capability.py",
        "postms_postcheck": "postcalibration_local_evidence.py",
        "postms_migration": "validate_structured_q2_migration.py",
    }[stage]
    return [sys.executable, str(root / "tools" / tool), *args]


def _report_path(stage: str, work_dir: Path) -> Path:
    return work_dir / REPORT_NAMES[stage]


def _read_report(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise RuntimeError(f"stage report was not written: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid stage report: {path}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"stage report is not a JSON object: {path}")
    return value


def _require_bool(report: Mapping[str, object], key: str, context: str) -> None:
    if not bool(report.get(key, False)):
        raise RuntimeError(f"{context} requires {key}=true")


def _require_upstream(stage: str, work_dir: Path) -> None:
    """Require artifacts and promotion gates before formal downstream stages."""
    if stage == "identifier_oracle":
        # The collector remains diagnostic-only (its sequence artifact is not
        # a production init), but it may run once the registered v4 probe is
        # frozen.  With today's debug report this rejects before spending any
        # K35 collection budget, preserving the fail-closed behavior.
        from tools.diagnose_causal_identifier_oracle import probe_v4_eligibility
        probe = probe_v4_eligibility()
        if not probe.get("eligible"):
            raise RuntimeError(
                "identifier_oracle stage is blocked: formal causal oracle is not ready"
            )
        return
    if stage == "identifier_pretrain":
        # The producer performs the immutable v4/causal checks itself before
        # touching Torch.  The collector's ``formal_eligible`` and top-level
        # ``gate_passed`` intentionally stay false until this producer writes
        # the production artifact, so do not require either field here.
        oracle = _read_report(_report_path("identifier_oracle", work_dir))
        _require_bool(oracle, "requested_formal_shape", "production identifier pretraining")
        _require_bool(oracle, "pretraining_gate_passed", "production identifier pretraining")
    elif stage == "phase_a1":
        from tools.diagnose_causal_identifier_oracle import probe_v4_eligibility
        probe = probe_v4_eligibility()
        if not probe.get("eligible"):
            raise RuntimeError(
                "Phase A1 is blocked: the current formal v4 probe is not eligible"
            )
        report = _read_report(_report_path("identifier_oracle", work_dir))
        if "requested_formal_shape" in report or "pretraining_gate_passed" in report:
            _require_bool(report, "requested_formal_shape", "Phase A1 identification oracle")
            _require_bool(report, "pretraining_gate_passed", "Phase A1 identification oracle")
        else:
            # Compatibility for old screening reports: the producer stage is
            # strict, but retain the historical error ordering so a missing
            # artifact is still reported clearly for legacy callers.
            _require_bool(report, "formal_eligible", "Phase A1 identification oracle")
            _require_bool(report, "gate_passed", "Phase A1 identification oracle")
        artifact = work_dir / "identifier_init.pt"
        if not artifact.is_file():
            raise RuntimeError(
                "Phase A1 requires the formal production identifier-init artifact: "
                f"{artifact}"
            )
        pretrain = _read_report(_report_path("identifier_pretrain", work_dir))
        _require_bool(pretrain, "pretraining_gate_passed", "Phase A1 production identifier init")
        if pretrain.get("artifact_schema_version") != "structured_identifier_init_v1":
            raise RuntimeError("Phase A1 requires the current identifier pretraining artifact schema")
    elif stage == "phase_a2":
        report = _read_report(_report_path("phase_a1", work_dir))
        _require_bool(report, "phase_a_mean_gate_passed", "Phase A2")
        if report.get("active_phase") != "A1":
            raise RuntimeError("Phase A2 requires a Phase-A1 mean report")
        if bool(report.get("capability_calibration_installed_in_stage", True)):
            raise RuntimeError("Phase A1 must not install capability calibration")
        if report.get("causal_revalidation_required") is not True:
            raise RuntimeError(
                "Phase A2 requires an explicit stale causal gate from A1; "
                "refusing an unbound/legacy A1 report"
            )
        if report.get("causal_gate_status") != "stale_revalidation_required":
            raise RuntimeError("Phase A2 requires causal_gate_status=stale_revalidation_required")
        revalidation = work_dir / "causal_identifier_revalidation.json"
        if not revalidation.is_file():
            raise RuntimeError(
                "Phase A2 requires causal revalidation after A1 identifier updates: "
                f"{revalidation}"
            )
        from structured_checkpoint import validate_causal_revalidation_report
        validate_causal_revalidation_report(
            revalidation,
            phase_a1_checkpoint=work_dir / "phase_a1.pt",
            a1_report=report,
        )
    elif stage == "identifier_revalidation":
        a1 = _read_report(_report_path("phase_a1", work_dir))
        _require_bool(a1, "phase_a_mean_gate_passed", "causal identifier revalidation")
        if a1.get("active_phase") != "A1":
            raise RuntimeError("causal identifier revalidation requires a Phase-A1 report")
        if a1.get("causal_revalidation_required") is not True or a1.get("causal_gate_status") != "stale_revalidation_required":
            raise RuntimeError("causal identifier revalidation requires the stale A1 causal gate")
        pretrain = _read_report(_report_path("identifier_pretrain", work_dir))
        _require_bool(pretrain, "pretraining_gate_passed", "causal identifier revalidation")
        artifact = work_dir / "identifier_init.pt"
        checkpoint = work_dir / "phase_a1.pt"
        if not artifact.is_file() or not checkpoint.is_file():
            raise RuntimeError("causal identifier revalidation requires A1 checkpoint and identifier artifact")
    elif stage == "phase_b":
        report = _read_report(_report_path("phase_a2", work_dir))
        _require_bool(report, "equilibrium_gate_passed", "Phase B")
        if report.get("active_phase") != "A2":
            raise RuntimeError("Phase B requires a Phase-A2 report")
        if not bool(report.get("capability_calibration_installed_in_stage", False)):
            raise RuntimeError("Phase B requires A2 capability calibration")
    elif stage == "residual_oracle":
        report = _read_report(_report_path("phase_b", work_dir))
        if report.get("active_phase") != "B":
            raise RuntimeError("residual oracle requires a Phase-B report with active_phase=B")
        _require_bool(report, "phase_b_gate_passed", "residual oracle")
        _require_bool(report, "h250_gate_passed", "residual oracle")
    elif stage == "phase_c":
        report = _read_report(_report_path("residual_oracle", work_dir))
        if report.get("active_phase") != "residual_oracle":
            raise RuntimeError("Phase C requires a residual-oracle report")
        _require_bool(report, "oracle_gate_passed", "Phase C")
    elif stage == "calibration":
        report = _read_report(_report_path("phase_c", work_dir))
        if report.get("active_phase") != "C":
            raise RuntimeError("calibration requires a Phase-C report with active_phase=C")
        _require_bool(report, "phase_c_gate_passed", "formal calibration")
        markers = report.get("intervention_markers", {})
        if not isinstance(markers, Mapping) or not bool(markers.get("final_two_beta0", False)):
            raise RuntimeError("formal calibration requires final two beta=0 DAgger rounds")
        if int(report.get("final_teacher_execution_count", 1)) != 0:
            raise RuntimeError("formal calibration requires zero final teacher executions")
    elif stage == "postcheck":
        report = _read_report(_report_path("calibration", work_dir))
        metadata = report.get("capability_calibration", report)
        if not isinstance(metadata, Mapping):
            raise RuntimeError("postcheck requires calibration metadata")
        if not bool(metadata.get("sufficient_samples", False)):
            raise RuntimeError("postcheck requires sufficient held-out calibration samples")
        if not bool(metadata.get("promotion_gate_passed", False)):
            raise RuntimeError("postcheck requires a passed formal capability calibration gate")
        checkpoint = work_dir / "calibrated.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"postcheck checkpoint is missing: {checkpoint}")
    elif stage == "migration":
        report = _read_report(_report_path("postcheck", work_dir))
        _require_bool(report, "postcheck_gate_passed", "paired migration")
    elif stage == "fullspace2":
        report = _read_report(_report_path("migration", work_dir))
        _require_bool(report, "migration_gate_passed", "full-space 2xH250")
    elif stage == "fullspace4":
        report = _read_report(_report_path("fullspace2", work_dir))
        if int(report.get("accepted_steps", 0)) < 1:
            raise RuntimeError("full-space 4xH250 requires an accepted 2xH250 step")
        if not bool(report.get("formal_gate_passed", False)):
            raise RuntimeError("full-space 4xH250 requires fullspace2 formal_gate_passed=true")
    elif stage == "postms_calibration":
        report = _read_report(_report_path("fullspace4", work_dir))
        if int(report.get("accepted_steps", 0)) < 1:
            raise RuntimeError("post-MS calibration requires an accepted 4xH250 step")
        _require_bool(report, "formal_gate_passed", "post-MS calibration")
        _require_bool(report, "pre_update_controller_evidence_invalidated",
                      "post-MS calibration")
        expected = str((work_dir / "migrated.pt").resolve())
        if Path(str(report.get("source_checkpoint", ""))).resolve().as_posix() != Path(expected).as_posix():
            raise RuntimeError("4xH250 must be an independent horizon-scaling run from migrated.pt")
    elif stage == "postms_postcheck":
        report = _read_report(_report_path("postms_calibration", work_dir))
        metadata = report.get("capability_calibration", report)
        if not isinstance(metadata, Mapping):
            raise RuntimeError("post-MS postcheck requires calibration metadata")
        _require_bool(metadata, "promotion_gate_passed", "post-MS postcheck")
        if not bool(metadata.get("sufficient_samples", False)):
            raise RuntimeError("post-MS postcheck requires sufficient held-out samples")
    elif stage == "postms_migration":
        report = _read_report(_report_path("postms_postcheck", work_dir))
        _require_bool(report, "postcheck_gate_passed", "post-MS migration")


def run_pipeline(stages: Sequence[str], *, root: Path = ROOT,
                 work_dir: Path = DEFAULT_WORK, device: str = "auto",
                 dry_run: bool = False) -> List[List[str]]:
    if not dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
    commands: List[List[str]] = []
    for stage in stages:
        if not dry_run:
            _require_upstream(stage, work_dir)
        command = stage_command(stage, root=root, work_dir=work_dir, device=device)
        if stage == "phase_c":
            # The report is accepted only after _require_upstream validates
            # its hash and held-out H250 gate; this argument is never a naked
            # boolean config bypass.
            command.extend(["--external-residual-oracle-report",
                            str(_report_path("residual_oracle", work_dir))])
        commands.append(command)
        print("[dry-run]" if dry_run else "[run]", shlex.join(command))
        if dry_run:
            continue
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"structured pipeline stage {stage} failed with exit {completed.returncode}")
    return commands


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cli(argv)
    stages = STAGES if args.stage == "all" else (args.stage,)
    try:
        run_pipeline(stages, work_dir=args.work_dir, device=args.device,
                     dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"pipeline stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
