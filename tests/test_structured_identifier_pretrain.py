from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from probe_contract import (
    WAVEFORM,
    WAVEFORM_SHA256,
    canonical_waveform_json,
    waveform_metadata,
)
from probe_contract_v5 import CONTRACT_SHA256 as WAVEFORM_SHA256
from structured_checkpoint import sha256_file
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy
from tools.pretrain_structured_identifier import (
    run,
    validate_pretraining_gates,
)


def _passing_probe(path: Path, monkeypatch) -> None:
    from test_probe_v5 import passing_freeze
    passing_freeze(path, monkeypatch)


def _passing_oracle(path: Path, source: Path) -> None:
    checks = {
        key: True for key in (
            "coverage", "paired_safety", "physics_ceiling", "physics_finite", "physics_branch_support",
            "finite_collection", "no_identification_failure", "zero_q2_parity",
        )
    }
    collection = path.with_suffix(".pt")
    torch.save([], collection)
    path.write_text(json.dumps({
        "collection_path": str(collection), "collection_sha256": sha256_file(collection),
        "diagnostic": "causal-identifier-sequence-oracle",
        "requested_formal_shape": True,
        "pretraining_gate_passed": True,
        "pretraining_checks": checks,
        "checkpoint_sha256": sha256_file(source),
        "cadence_semantics_version": "call100_passive_identification_v5",
        "coverage": {"K35_gate_passed": True, "continuous_observer_gate_passed": True},
        "ceiling": {"gate_passed": True, "tau_wls_finite": True,
                     "effectiveness_finite": True, "tau_wls_supported": True},
        "safety": {"finite_collected": True, "no_identification_failure": True},
        "zero_parity": {"gate_passed": True},
    }), encoding="utf-8")


def test_identifier_pretrain_dry_run_has_no_side_effects(tmp_path: Path) -> None:
    output = tmp_path / "identifier_init.pt"
    report = tmp_path / "report.json"
    result = run(type("Args", (), {
        "dry_run": True, "source_checkpoint": tmp_path / "missing.pt",
        "output": output, "report": report,
        "probe_v4_report": tmp_path / "probe.json",
        "causal_oracle_report": tmp_path / "oracle.json",
    })())
    assert result["stage"] == "identifier_pretrain"
    assert not output.exists() and not report.exists()


def test_identifier_pretrain_rejects_unfrozen_current_probe(tmp_path: Path) -> None:
    source = Path("reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt")
    probe = tmp_path / "unfrozen.json"
    probe.write_text(json.dumps({"formal": {"eligible": False}, "gate_passed": False}))
    with pytest.raises(RuntimeError, match="eligible frozen v5 identification"):
        validate_pretraining_gates(
            source_checkpoint=source,
            probe_report=probe,
            causal_oracle_report=tmp_path / "missing-oracle.json",
        )


def test_identifier_pretrain_gate_binds_source_and_reports_all_checks(tmp_path: Path, monkeypatch) -> None:
    source = Path("reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt")
    probe = tmp_path / "probe.json"
    oracle = tmp_path / "oracle.json"
    _passing_probe(probe, monkeypatch)
    _passing_oracle(oracle, source)
    result = validate_pretraining_gates(
        source_checkpoint=source, probe_report=probe, causal_oracle_report=oracle,
    )
    assert result["probe_v4"]["frozen_sha256"] == WAVEFORM_SHA256
    assert result["causal_oracle"]["checkpoint_sha256"] == sha256_file(source)


def test_identifier_artifact_contains_only_production_weights(tmp_path: Path) -> None:
    from structured_checkpoint import load_identifier_init_artifact, write_identifier_init_artifact

    source = Path("reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt")
    oracle = tmp_path / "oracle.json"
    oracle.write_text("{}", encoding="utf-8")
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4))
    artifact_path = tmp_path / "identifier_init.pt"
    artifact = write_identifier_init_artifact(
        policy, artifact_path, q2_checkpoint=source, causal_oracle_report=oracle,
    )
    assert artifact["sidecar_oracle_gru"] is False
    assert all("oracle" not in name and "label" not in name
               for name in artifact["weight_names"])
    restored = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4))
    loaded = load_identifier_init_artifact(
        artifact_path, restored, q2_checkpoint=source, causal_oracle_report=oracle,
    )
    assert loaded["artifact_schema_version"] == "structured_identifier_init_v1"


def test_identifier_pretraining_report_binds_artifact_and_mean_gates(tmp_path: Path) -> None:
    from structured_checkpoint import (
        load_identifier_init_artifact,
        validate_identifier_pretraining_report,
        write_identifier_init_artifact,
    )

    source = Path("reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt")
    oracle = tmp_path / "oracle.json"
    oracle.write_text("{}", encoding="utf-8")
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4))
    artifact_path = tmp_path / "identifier_init.pt"
    write_identifier_init_artifact(
        policy, artifact_path, q2_checkpoint=source, causal_oracle_report=oracle,
    )
    contract = load_identifier_init_artifact(
        artifact_path, policy, q2_checkpoint=source, causal_oracle_report=oracle,
    )
    passed_gate = {"gate_passed": True, "rows": [
        {"phase_step": 100, "passed": True},
        {"phase_step": 125, "passed": True},
    ]}
    report = {
        "stage": "identifier_pretrain",
        "diagnostic": "production-structured-identifier-pretraining",
        "formal_gate_passed": True,
        "pretraining_gate_passed": True,
        "independent_validation_passed": True,
        "independent_final_passed": True,
        "identifier_artifact_written": True,
        "teacher_is_frozen": True,
        "teacher_action_is_executed": True,
        "runtime_features_from_deployable_observation_action_history": True,
        "capability_labels_used_as_runtime_features": False,
        "sidecar_oracle_gru": False,
        "artifact_schema_version": "structured_identifier_init_v1",
        "source_checkpoint_sha256": sha256_file(source),
        "identifier_init_artifact": {
            "sha256": sha256_file(artifact_path),
            "identifier_weights_sha256": contract["identifier_weights_sha256"],
        },
        "identifier_weights_sha256": contract["identifier_weights_sha256"],
        "config": asdict(policy.config),
        "causal_oracle_report": {"report_sha256": sha256_file(oracle)},
        "probe_v4": {"frozen_sha256": WAVEFORM_SHA256},
        "bank_seeds": {"train": [107], "validation": 10007,
                       "final": 20007, "blind": []},
        "validation": {"phase_a_mean_gate_passed": True,
                       "phase_a_mean_gate": passed_gate},
        "final": {"phase_a_mean_gate_passed": True,
                  "phase_a_mean_gate": passed_gate},
    }
    report_path = tmp_path / "pretrain.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    validated = validate_identifier_pretraining_report(
        report_path, identifier_artifact=artifact_path,
        identifier_contract=contract, policy=policy,
        q2_checkpoint=source, causal_oracle_report=oracle,
    )
    assert validated["sha256"] == sha256_file(report_path)

    report["final"]["phase_a_mean_gate_passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="final calls100/125 mean gate"):
        validate_identifier_pretraining_report(
            report_path, identifier_artifact=artifact_path,
            identifier_contract=contract, policy=policy,
            q2_checkpoint=source, causal_oracle_report=oracle,
        )
