from __future__ import annotations

from pathlib import Path

import pytest

from tools.run_structured_pipeline import (
    ROOT,
    STAGES,
    parse_args_file,
    run_pipeline,
    stage_command,
)


def _flag_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_args_parser_expands_substitutions_and_comments(tmp_path: Path) -> None:
    config = tmp_path / "sample.args"
    config.write_text(
        "# ignored\n--output {work}/out.pt\n--seed 7 # inline comment\n",
        encoding="utf-8",
    )
    parsed = parse_args_file(config, {"work": "/tmp/pipeline"})
    assert parsed[:2] == ["--output", "/tmp/pipeline/out.pt"]
    assert parsed == ["--output", "/tmp/pipeline/out.pt", "--seed", "7"]


def test_formal_stage_commands_are_fixed_and_dependencies_are_explicit(tmp_path: Path) -> None:
    phase_a = stage_command("phase_a1", work_dir=tmp_path, device="cpu")
    revalidation = stage_command("identifier_revalidation", work_dir=tmp_path, device="cpu")
    phase_a2 = stage_command("phase_a2", work_dir=tmp_path, device="cpu")
    phase_c = stage_command("phase_c", work_dir=tmp_path, device="cpu")
    fullspace = stage_command("fullspace4", work_dir=tmp_path, device="cpu")
    assert _flag_value(phase_a, "--seed") == "7"
    assert _flag_value(phase_a, "--phase") == "A1"
    assert _flag_value(phase_a, "--identifier-pretraining-report") == str(
        tmp_path / "identifier_pretrain_report.json"
    )
    assert _flag_value(phase_a, "--probe-v4-report") == str(
        ROOT / "reports" / "probe_v4_formal.json"
    )
    assert _flag_value(revalidation, "--phase-a1-checkpoint") == str(
        tmp_path / "phase_a1.pt"
    )
    assert _flag_value(revalidation, "--teacher-forced-seed-offset") == "30001"
    assert _flag_value(revalidation, "--on-policy-seed-offset") == "40002"
    assert _flag_value(phase_a2, "--phase") == "A2"
    assert _flag_value(phase_a2, "--student-checkpoint") == str(tmp_path / "phase_a1.pt")
    assert _flag_value(phase_c, "--student-checkpoint") == str(tmp_path / "phase_b.pt")
    assert "--external-h250-report" not in phase_c
    assert "--external-residual-oracle-report" not in phase_c
    assert _flag_value(fullspace, "--batch-size") == "64"
    assert _flag_value(fullspace, "--segment-steps") == "250"
    assert _flag_value(fullspace, "--segments") == "4"
    assert _flag_value(fullspace, "--alpha") == "0.8"
    assert _flag_value(fullspace, "--action-probe-stride") == "1"
    assert _flag_value(fullspace, "--trainable-prefix") == "residual_head."


def test_all_dry_run_expands_every_stage_without_writing_artifacts(tmp_path: Path) -> None:
    commands = run_pipeline(STAGES, root=ROOT, work_dir=tmp_path,
                            device="cpu", dry_run=True)
    assert len(commands) == len(STAGES)
    assert list(tmp_path.iterdir()) == []
    assert all(command[0].endswith("python3") or command[0].endswith("python")
               for command in commands)


def test_identifier_oracle_is_first_and_explicitly_blocked_until_ready(tmp_path: Path) -> None:
    command = stage_command("identifier_oracle", work_dir=tmp_path, device="cpu")
    assert command[-1] == "4"
    assert "diagnose_causal_identifier_oracle.py" in command[1]
    with pytest.raises(RuntimeError, match="identifier_oracle stage is blocked"):
        run_pipeline(("identifier_oracle",), root=ROOT, work_dir=tmp_path,
                     device="cpu", dry_run=False)


def test_downstream_formal_stage_stops_on_failed_upstream_gate(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "postcheck_report.json").write_text(
        '{"postcheck_gate_passed": false}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="postcheck_gate_passed"):
        run_pipeline(("migration",), root=ROOT, work_dir=tmp_path,
                     device="cpu", dry_run=False)


def test_phase_a2_requires_mean_gate_without_a1_calibration(tmp_path: Path) -> None:
    (tmp_path / "phase_a1_report.json").write_text(
        '{"active_phase":"A1", "phase_a_mean_gate_passed":false, '
        '"capability_calibration_installed_in_stage":false}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="phase_a_mean_gate_passed"):
        run_pipeline(("phase_a2",), root=ROOT, work_dir=tmp_path,
                     device="cpu", dry_run=False)


def test_phase_b_requires_calibrated_phase_a2(tmp_path: Path) -> None:
    (tmp_path / "phase_a2_report.json").write_text(
        '{"active_phase":"A2", "equilibrium_gate_passed":true, '
        '"capability_calibration_installed_in_stage":false}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="A2 capability calibration"):
        run_pipeline(("phase_b",), root=ROOT, work_dir=tmp_path,
                     device="cpu", dry_run=False)


def test_oracle_requires_phase_b_gate_and_phase_c_injects_oracle_report(tmp_path: Path) -> None:
    (tmp_path / "phase_b_report.json").write_text(
        '{"active_phase":"B", "phase_b_gate_passed":false, "h250_gate_passed":true}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="phase_b_gate_passed"):
        run_pipeline(("residual_oracle",), root=ROOT, work_dir=tmp_path,
                     device="cpu", dry_run=False)


def test_postms_calibration_requires_accepted_formal_4x_horizon_update(tmp_path: Path) -> None:
    (tmp_path / "fullspace4_report.json").write_text(
        '{"accepted_steps": 1, "formal_gate_passed": false, '
        '"pre_update_controller_evidence_invalidated": true}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="formal_gate_passed"):
        run_pipeline(("postms_calibration",), root=ROOT, work_dir=tmp_path,
                     device="cpu", dry_run=False)
