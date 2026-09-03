from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
import pandas as pd
import torch

from diagnostics.phase1_fast_common import atomic_write_dataframe
from diagnostics.formal_rollout import rollout_q2
from env_l2f import L2FSimulator
from model import MotorGRUPolicy
from policy_observation import (
    CYLINDRICAL_INTEGRAL_CLAMP_MODE,
    LEGACY_BOX_INTEGRAL_CLAMP_MODE,
)
from tools.run_phase1_streaming_rollout import (
    SCENARIO_ORDER_FIELDS,
    STREAMING_OUTPUT_NAMES,
    SUMMARY_ONLY_OUTPUT_NAMES,
    _argument_parser,
    _failure_label,
    _provenance_parameters,
    _scenario_order_sha256,
    _validate_bundle_provenance,
    _validate_matlab_provenance,
)


LABEL = "seed_17_group_T0_physical_steps_96000000"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "seed_17" / "group_T0" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    samples = tmp_path / "seed_17" / "group_T0" / "samples.csv"
    _write_csv(samples, [{"sample_id": 1}])
    checkpoints_csv = tmp_path / "CHECKPOINTS.csv"
    _write_csv(
        checkpoints_csv,
        [{
            "label": LABEL,
            "checkpoint_path": checkpoint,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seed": 17,
            "group": "T0",
        }],
    )
    eval_manifest = tmp_path / "eval_manifest.csv"
    _write_csv(
        eval_manifest,
        [{
            "label": LABEL,
            "checkpoint_path": checkpoint,
            "sample_output_path": samples,
        }],
    )
    return {
        "checkpoint": checkpoint,
        "samples": samples,
        "checkpoints_csv": checkpoints_csv,
        "eval_manifest": eval_manifest,
    }


def test_matching_matlab_and_checkpoint_provenance_passes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    checkpoint, seed, group = _validate_matlab_provenance(
        checkpoint_label=LABEL,
        checkpoints_csv=paths["checkpoints_csv"],
        matlab_samples=paths["samples"],
        matlab_eval_manifest=paths["eval_manifest"],
    )

    assert checkpoint == paths["checkpoint"].resolve()
    assert seed == 17
    assert group == "T0"


def test_seed_group_sample_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    wrong_samples = tmp_path / "seed_17" / "group_T2" / "samples.csv"
    _write_csv(wrong_samples, [{"sample_id": 1}])

    with pytest.raises(RuntimeError, match="MATLAB sample provenance mismatch"):
        _validate_matlab_provenance(
            checkpoint_label=LABEL,
            checkpoints_csv=paths["checkpoints_csv"],
            matlab_samples=wrong_samples,
            matlab_eval_manifest=paths["eval_manifest"],
        )


def test_seed_group_manifest_label_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write_csv(
        paths["eval_manifest"],
        [{
            "label": "seed_17_group_T2_physical_steps_96000000",
            "checkpoint_path": paths["checkpoint"],
            "sample_output_path": paths["samples"],
        }],
    )

    with pytest.raises(RuntimeError, match="must contain exactly one row"):
        _validate_matlab_provenance(
            checkpoint_label=LABEL,
            checkpoints_csv=paths["checkpoints_csv"],
            matlab_samples=paths["samples"],
            matlab_eval_manifest=paths["eval_manifest"],
        )


def test_seed_group_checkpoint_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    wrong_checkpoint = tmp_path / "seed_17" / "group_T2" / "model.pt"
    wrong_checkpoint.parent.mkdir(parents=True)
    wrong_checkpoint.write_bytes(b"other checkpoint")
    _write_csv(
        paths["eval_manifest"],
        [{
            "label": LABEL,
            "checkpoint_path": wrong_checkpoint,
            "sample_output_path": paths["samples"],
        }],
    )

    with pytest.raises(RuntimeError, match="MATLAB checkpoint provenance mismatch"):
        _validate_matlab_provenance(
            checkpoint_label=LABEL,
            checkpoints_csv=paths["checkpoints_csv"],
            matlab_samples=paths["samples"],
            matlab_eval_manifest=paths["eval_manifest"],
        )


def _portable_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "bundle"
    checkpoint = root / "checkpoints" / "seed_17" / "T0" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    samples = root / "matlab_eval" / "seed_17" / "T0" / "samples.csv"
    sample_row = {field: str(index) for index, field in enumerate(SCENARIO_ORDER_FIELDS)}
    sample_row["sample_id"] = "1"
    _write_csv(samples, [sample_row])
    bundle_index = root / "BUNDLE_INDEX.csv"
    _write_csv(
        bundle_index,
        [{
            "seed": 17,
            "group": "T0",
            "physical_steps": 96000000,
            "checkpoint_path": checkpoint.relative_to(root),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "samples_path": samples.relative_to(root),
            "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
            "sample_rows": 1,
            "scenario_order_sha256": _scenario_order_sha256([sample_row]),
            "motor_state_head_present": 1,
        }],
    )
    return {"checkpoint": checkpoint, "samples": samples, "bundle_index": bundle_index}


def test_portable_bundle_resolves_relocated_relative_paths(tmp_path: Path) -> None:
    paths = _portable_fixture(tmp_path)

    checkpoint, seed, group = _validate_bundle_provenance(
        checkpoint_label=LABEL,
        bundle_index=paths["bundle_index"],
        matlab_samples=paths["samples"],
    )

    assert checkpoint == paths["checkpoint"].resolve()
    assert seed == 17
    assert group == "T0"


def test_portable_bundle_rejects_sample_hash_mismatch(tmp_path: Path) -> None:
    paths = _portable_fixture(tmp_path)
    paths["samples"].write_text("sample_id\n2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sample path/hash mismatch"):
        _validate_bundle_provenance(
            checkpoint_label=LABEL,
            bundle_index=paths["bundle_index"],
            matlab_samples=paths["samples"],
        )


def test_formal_joint_label_is_explicitly_untyped_without_channel_counts() -> None:
    scenario = {"dynamic_hard": "0"}
    assert _failure_label({"position_hold_steady_H10000": "1"}, scenario) == "formal-joint-success"
    assert (
        _failure_label({"position_hold_steady_H10000": "0"}, scenario)
        == "formal-joint-failure-untyped"
    )


def test_exact_channel_counts_keep_legacy_mechanism_groups() -> None:
    row = {
        "position_hold_steady_H10000": "0",
        "position_pass_count_H10000": "94",
        "velocity_pass_count_H10000": "100",
        "omega_pass_count_H10000": "100",
    }
    assert _failure_label(row, {"dynamic_hard": "0"}) == "position-only"


def test_streaming_output_manifest_is_an_explicit_non_hidden_whitelist() -> None:
    assert len(STREAMING_OUTPUT_NAMES) == len(set(STREAMING_OUTPUT_NAMES))
    assert all(not name.startswith(".") for name in STREAMING_OUTPUT_NAMES)
    assert "RUN_PROVENANCE.json" not in STREAMING_OUTPUT_NAMES
    assert set(SUMMARY_ONLY_OUTPUT_NAMES) < set(STREAMING_OUTPUT_NAMES)


def test_runner_provenance_records_default_and_counterfactual_clamp_modes() -> None:
    parser = _argument_parser()
    legacy = parser.parse_args(["--matlab-samples", "samples.csv"])
    cylinder = parser.parse_args([
        "--matlab-samples", "samples.csv",
        "--integral-clamp-mode", CYLINDRICAL_INTEGRAL_CLAMP_MODE,
        "--summary-only",
    ])

    legacy_parameters = _provenance_parameters(legacy, seed=7, group="T0")
    cylinder_parameters = _provenance_parameters(cylinder, seed=7, group="T0")
    assert legacy_parameters["integral_clamp_mode"] == LEGACY_BOX_INTEGRAL_CLAMP_MODE
    assert legacy_parameters["summary_only"] is False
    assert cylinder_parameters["integral_clamp_mode"] == CYLINDRICAL_INTEGRAL_CLAMP_MODE
    assert cylinder_parameters["summary_only"] is True
    assert cylinder_parameters != legacy_parameters


def test_summary_only_cylindrical_h100_formal_rollout_smoke() -> None:
    torch.manual_seed(20260804)
    state = L2FSimulator().reset(
        2,
        device="cpu",
        dtype=torch.float64,
        sample_dynamics=False,
        sample_external_force=False,
    )
    policy = MotorGRUPolicy(
        observation_dim=25,
        encoder_dim=12,
        hidden_dim=12,
        encoder_depth=1,
        enable_integral_residual=True,
        enable_damping_residual=True,
        integral_residual_hidden_dim=4,
        damping_residual_hidden_dim=4,
    ).to(dtype=torch.float64).eval()

    result = rollout_q2(
        policy,
        state,
        ["scenario-a", "scenario-b"],
        checkpoint_label="smoke",
        seed=7,
        group="T0",
        horizon=100,
        snapshot_horizons=(100,),
        backend="torch",
        streaming_accumulator=None,
        integral_clamp_mode=CYLINDRICAL_INTEGRAL_CLAMP_MODE,
    )

    assert len(result.horizon_rows) == 2
    assert len(result.integral_rows) == 6
    assert result.branch_rows == []
    assert result.phase_rows == []
    assert {row["integral_clamp_mode"] for row in result.horizon_rows} == {
        CYLINDRICAL_INTEGRAL_CLAMP_MODE
    }


def test_atomic_dataframe_write_removes_destination_specific_orphans(tmp_path: Path) -> None:
    destination = tmp_path / "large.csv"
    orphan = tmp_path / ".large.csv.stale.csv.extra"
    orphan.write_text("partial", encoding="utf-8")

    atomic_write_dataframe(pd.DataFrame({"value": [1, 2]}), destination)

    assert destination.is_file()
    assert not orphan.exists()
