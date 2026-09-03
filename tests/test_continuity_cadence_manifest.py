from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

from tools import run_continuity_cadence_screen as runner
from tools import validate_continuity_cadence_screen as validator
from training_schedule import COMPRESSED_T2_10PCT_SEQUENCE


ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_d_run(root: Path) -> Path:
    run_dir = root / "seed_7" / "arm_D"
    command = runner._command(7, "D", run_dir)
    runner._write_run_manifest(run_dir, seed=7, arm="D", command=command)

    rows: list[dict[str, object]] = []
    physical_steps = 0
    optimizer_updates = 0
    for episode_target in COMPRESSED_T2_10PCT_SEQUENCE:
        for episode_step in range(0, episode_target, 250):
            physical_steps += 256 * 250
            reset_boundary = episode_step + 250 == episode_target
            if reset_boundary:
                optimizer_updates += 1
            tail_progress = episode_step % 500
            rows.append(
                {
                    "physical_steps": physical_steps,
                    "optimizer_update": optimizer_updates,
                    "update_applied": int(reset_boundary),
                    "reset_episode_boundary": int(reset_boundary),
                    "optimization_block_boundary": int(reset_boundary),
                    "episode_target_steps": episode_target,
                    "tail_supervision_block_horizon": 500,
                    "tail_supervision_block_boundary": int(tail_progress + 250 == 500),
                    "first_tail_supervision_segment": int(tail_progress == 0),
                    "rollout_valid": 1,
                    "skip_reason": "",
                    "force_reset_next": int(reset_boundary),
                    "reset_mask": float(episode_step == 0),
                    "hidden_state_norm_initial": 1.0,
                    "integral_state_norm_initial": 1.0,
                }
            )

    with (run_dir / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "checkpoints" / "model.pt").write_bytes(b"synthetic-checkpoint")
    (run_dir / "reset_samples.csv").write_text("reset_hash\npaired\n", encoding="utf-8")
    return run_dir


def test_arm_d_schema_v2_manifest_and_preregistered_counts(tmp_path: Path) -> None:
    run_dir = _write_synthetic_d_run(tmp_path)
    summary, errors = validator._validate_run(run_dir, "D")

    assert errors == []
    assert summary["complete"] is True
    assert summary["rows"] == 150
    assert summary["updates"] == 67
    assert summary["reset_boundaries"] == 67
    assert summary["optimization_boundaries"] == 67
    assert summary["tail_supervision_starts"] == 75
    assert summary["tail_supervision_boundaries"] == 75
    assert summary["mid_h1000_updates"] == 0

    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 2
    assert manifest["initial_checkpoint"] == validator.INITIAL_CHECKPOINT_RELATIVE.as_posix()
    assert manifest["initial_checkpoint_sha256"] == validator.INITIAL_CHECKPOINT_SHA256
    assert tuple(record["path"] for record in manifest["code_files"]) == tuple(
        path.as_posix() for path in validator.EXPECTED_CODE_FILES
    )
    assert tuple(record["path"] for record in manifest["config_closure"]) == tuple(
        path.as_posix() for path in validator._expected_config_closure("D")
    )


def test_schema_v2_rejects_incomplete_provenance(tmp_path: Path) -> None:
    run_dir = _write_synthetic_d_run(tmp_path)
    manifest_path = run_dir / "RUN_MANIFEST.json"
    valid = json.loads(manifest_path.read_text(encoding="utf-8"))

    cases: list[tuple[str, dict[str, object], str]] = []
    missing_code = copy.deepcopy(valid)
    missing_code["code_files"].pop()
    cases.append(("missing-code", missing_code, "code_files is missing"))

    duplicate_code = copy.deepcopy(valid)
    duplicate_code["code_files"].append(copy.deepcopy(duplicate_code["code_files"][0]))
    cases.append(("duplicate-code", duplicate_code, "code_files has duplicate paths"))

    empty_config = copy.deepcopy(valid)
    empty_config["config_closure"] = []
    cases.append(("empty-config", empty_config, "config_closure must be a non-empty list"))

    wrong_checkpoint = copy.deepcopy(valid)
    wrong_checkpoint["initial_checkpoint"] = "checkpoints/other.pt"
    cases.append(("wrong-checkpoint", wrong_checkpoint, "manifest initial_checkpoint="))

    wrong_checkpoint_hash = copy.deepcopy(valid)
    wrong_checkpoint_hash["initial_checkpoint_sha256"] = "0" * 64
    cases.append(
        (
            "wrong-checkpoint-hash",
            wrong_checkpoint_hash,
            "initial checkpoint SHA-256 is not the frozen value",
        )
    )

    for label, payload, expected_error in cases:
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _, errors = validator._validate_run(run_dir, "D")
        with_error = "\n".join(errors)
        assert expected_error in with_error, f"{label}: {with_error}"


def test_arm_d_runner_dry_run_manifest_and_skip_guard(tmp_path: Path) -> None:
    output_root = tmp_path / "dry-run"
    dry_run = subprocess.run(
        (
            sys.executable,
            "tools/run_continuity_cadence_screen.py",
            "--seeds",
            "7",
            "--arms",
            "D",
            "--output-root",
            str(output_root),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
    run_dir = output_root / "seed_7" / "arm_D"
    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["expected_segments"] == 150
    assert manifest["expected_reset_episodes"] == 67
    assert manifest["expected_optimizer_commits"] == 67
    assert manifest["expected_tail_supervision_blocks"] == 75
    assert manifest["initial_checkpoint_sha256"] == runner.INITIAL_CHECKPOINT_SHA256
    assert (run_dir / "command.txt").is_file()
    assert not (run_dir / "train.csv").exists()
    assert not (run_dir / "checkpoints" / "model.pt").exists()

    forbidden_root = tmp_path / "forbidden-skip"
    forbidden = subprocess.run(
        (
            sys.executable,
            "tools/run_continuity_cadence_screen.py",
            "--seeds",
            "7",
            "--arms",
            "D",
            "--output-root",
            str(forbidden_root),
            "--skip-existing",
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert forbidden.returncode != 0
    assert "Arm D forbids --skip-existing" in forbidden.stdout + forbidden.stderr
    assert not forbidden_root.exists()
