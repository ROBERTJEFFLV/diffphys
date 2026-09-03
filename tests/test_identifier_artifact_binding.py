from __future__ import annotations

from pathlib import Path

import pytest
import torch

from structured_checkpoint import (
    build_identifier_init_artifact,
    load_identifier_init_artifact,
)
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy
from tools.run_structured_pipeline import run_pipeline


def _artifact(tmp_path: Path):
    q2 = tmp_path / "q2.pt"
    q2.write_bytes(b"q2-baseline")
    oracle = tmp_path / "oracle.json"
    oracle.write_text("{\"formal_eligible\":true}\n", encoding="utf-8")
    config = StructuredPolicyConfig(hidden_dim=4, identifier_dim=4)
    source = StructuredRecurrentPolicy(config)
    payload = build_identifier_init_artifact(
        source, q2_checkpoint=q2, causal_oracle_report=oracle
    )
    artifact = tmp_path / "identifier_init.pt"
    torch.save(payload, artifact)
    return q2, oracle, artifact, config


def test_identifier_artifact_installs_exact_production_weights(tmp_path: Path) -> None:
    q2, oracle, artifact, config = _artifact(tmp_path)
    target = StructuredRecurrentPolicy(config)
    before = target.identifier.cell.weight_ih.detach().clone()
    details = load_identifier_init_artifact(
        artifact, target, q2_checkpoint=q2, causal_oracle_report=oracle
    )
    assert details["artifact_schema_version"] == "structured_identifier_init_v1"
    assert not torch.equal(before, target.identifier.cell.weight_ih)


@pytest.mark.parametrize("mutation", ("extra", "q2", "oracle_gru", "labels"))
def test_identifier_artifact_fails_closed(tmp_path: Path, mutation: str) -> None:
    q2, oracle, artifact, config = _artifact(tmp_path)
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if mutation == "extra":
        payload["weights"]["oracle_gru.weight"] = torch.zeros(1)
    elif mutation == "q2":
        payload["q2_checkpoint_sha256"] = "0" * 64
    elif mutation == "oracle_gru":
        payload["sidecar_oracle_gru"] = True
    else:
        payload["capability_labels_runtime_input"] = True
    torch.save(payload, artifact)
    with pytest.raises(RuntimeError, match="invalid identifier-init artifact"):
        load_identifier_init_artifact(
            artifact, StructuredRecurrentPolicy(config), q2_checkpoint=q2,
            causal_oracle_report=oracle,
        )


def test_pipeline_requires_current_formal_probe_before_starting_a1(tmp_path: Path) -> None:
    (tmp_path / "causal_identifier_oracle.json").write_text(
        '{"formal_eligible":true,"gate_passed":true}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="current formal v4 probe"):
        run_pipeline(("phase_a1",), work_dir=tmp_path, device="cpu")


def test_pipeline_requires_revalidation_after_a1(tmp_path: Path) -> None:
    (tmp_path / "phase_a1_report.json").write_text(
        '{"active_phase":"A1","phase_a_mean_gate_passed":true, '
        '"capability_calibration_installed_in_stage":false, '
        '"causal_revalidation_required":true, '
        '"causal_gate_status":"stale_revalidation_required"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="causal revalidation"):
        run_pipeline(("phase_a2",), work_dir=tmp_path, device="cpu")
