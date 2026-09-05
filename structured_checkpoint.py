"""Shared checkpoint provenance and deployment-hash helpers."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from identification_features import feature_schema_metadata, feature_schema_sha256
from probe_contract_v5 import (VERSION as PROBE_CONTRACT_VERSION, ACTIVE_STEPS as PROBE_PERIOD, CONTRACT_SHA256 as WAVEFORM_SHA256)


CALIBRATION_STATE_NAMES = frozenset({
    "capability_conformal_q",
    "capability_calibration_n",
    "capability_calibration_valid",
})

# Bump whenever call-index/publication semantics change.  Including this in
# structured deployment hashes prevents pre-migration artifacts from being
# accepted as same-behavior calibration or oracle inputs.
CADENCE_SEMANTICS_VERSION = "call100_passive_identification_v5"

IDENTIFIER_ARTIFACT_SCHEMA_VERSION = "structured_identifier_init_v1"
IDENTIFIER_ARTIFACT_ARCHITECTURE = "structured-recurrent-motor-policy"
IDENTIFIER_ARTIFACT_CADENCE = {
    "call_index_completed_transitions": True,
    "call0_has_response": False,
    "publication_calls": [100, 125],
    "availability_t25": [0, 0, 0, 0, 0, 0],
    "t50_call_index": 100,
}


def sha256_file(path: str | Path) -> str:
    """Return the content hash used to bind formal artifacts to a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def runtime_contract_hash() -> str:
    """Bind artifacts to inference equations and state/probe semantics."""
    root = Path(__file__).resolve().parent
    names = ("structured_policy.py", "structured_allocator.py", "structured_rollout.py",
             "identification_features.py", "identification_information.py",
             "probe_contract_v5.py", "env_l2f.py")
    return hashlib.sha256(json.dumps({name: sha256_file(root / name) for name in names},
                                   sort_keys=True).encode()).hexdigest()


def identifier_artifact_weight_names(config: Any) -> tuple[str, ...]:
    """Exact trainable production weights allowed in an identifier artifact."""

    names = [
        "identifier.cell.weight_ih", "identifier.cell.weight_hh",
        "identifier.cell.bias_ih", "identifier.cell.bias_hh",
    ]
    if int(getattr(config, "motor_observer_bank_size", 0)):
        names.append("bank_adapter.weight")
    names.extend(("capability_head.weight", "capability_head.bias"))
    return tuple(names)


def identifier_weights_hash(weights_or_policy: Mapping[str, torch.Tensor] | Any) -> str:
    """Hash exactly the production identifier weights in canonical name order."""

    if isinstance(weights_or_policy, Mapping):
        values = weights_or_policy
        config = None
    else:
        values = weights_or_policy.state_dict()
        config = weights_or_policy.config
    if config is not None:
        names = identifier_artifact_weight_names(config)
    else:
        # Artifact mappings do not carry a policy object.  Recover the same
        # canonical order used for a policy hash; sorting would put bias keys
        # before weights and make an unchanged init appear modified.
        canonical = (
            "identifier.cell.weight_ih", "identifier.cell.weight_hh",
            "identifier.cell.bias_ih", "identifier.cell.bias_hh",
            "bank_adapter.weight", "capability_head.weight", "capability_head.bias",
        )
        names = tuple(name for name in canonical if name in values)
    digest = hashlib.sha256()
    for name in names:
        if name not in values:
            raise RuntimeError(f"identifier weight is missing: {name}")
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_identifier_init_artifact(
    policy: Any,
    *,
    q2_checkpoint: str | Path,
    causal_oracle_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a portable init artifact from a production policy.

    Only the production identifier, optional bank adapter, and capability mean
    head are copied.  In particular, no oracle GRU, simulator labels, or other
    policy state is included.
    """

    names = identifier_artifact_weight_names(policy.config)
    state = policy.state_dict()
    weights = {name: state[name].detach().cpu().clone() for name in names}
    artifact: dict[str, Any] = {
        "artifact_schema_version": IDENTIFIER_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "production-structured-identifier-init",
        "architecture": IDENTIFIER_ARTIFACT_ARCHITECTURE,
        "config": asdict(policy.config),
        "weights": weights,
        "weight_names": list(names),
        "identifier_weights_sha256": identifier_weights_hash(weights),
        "runtime_contract_sha256": runtime_contract_hash(),
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": dict(IDENTIFIER_ARTIFACT_CADENCE),
        "probe": {
            "contract_version": PROBE_CONTRACT_VERSION,
            "period": PROBE_PERIOD,
            "amplitude": float(policy.config.burn_in_probe_amplitude),
            "waveform_sha256": WAVEFORM_SHA256,
        },
        "feature_schema": feature_schema_metadata(),
        "feature_schema_sha256": feature_schema_sha256(),
        "q2_checkpoint_sha256": sha256_file(q2_checkpoint),
        "q2_checkpoint": str(Path(q2_checkpoint).resolve()),
        "causal_oracle_report_sha256": (
            sha256_file(causal_oracle_report) if causal_oracle_report is not None else None
        ),
        "sidecar_oracle_gru": False,
        "capability_labels_runtime_input": False,
    }
    return artifact


def write_identifier_init_artifact(
    policy: Any,
    path: str | Path,
    *,
    q2_checkpoint: str | Path,
    causal_oracle_report: str | Path | None = None,
) -> dict[str, Any]:
    """Write a reversible production identifier-init artifact."""

    artifact = build_identifier_init_artifact(
        policy,
        q2_checkpoint=q2_checkpoint,
        causal_oracle_report=causal_oracle_report,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    return artifact


def _artifact_error(message: str) -> RuntimeError:
    return RuntimeError(f"invalid identifier-init artifact: {message}")


def load_identifier_init_artifact(
    path: str | Path,
    policy: Any,
    *,
    q2_checkpoint: str | Path,
    causal_oracle_report: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and install a production identifier init, fail-closed."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise _artifact_error(f"artifact is missing: {artifact_path}")
    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise _artifact_error(f"cannot load {artifact_path}") from exc
    if not isinstance(payload, dict):
        raise _artifact_error("top-level payload must be a mapping")
    if payload.get("artifact_schema_version") != IDENTIFIER_ARTIFACT_SCHEMA_VERSION:
        raise _artifact_error("schema version is stale or unknown")
    if payload.get("artifact_type") != "production-structured-identifier-init":
        raise _artifact_error("artifact type is not a production identifier init")
    if payload.get("architecture") != IDENTIFIER_ARTIFACT_ARCHITECTURE:
        raise _artifact_error("architecture does not name StructuredRecurrentPolicy")
    if payload.get("sidecar_oracle_gru") is not False:
        raise _artifact_error("sidecar oracle GRU artifacts are not accepted")
    if payload.get("capability_labels_runtime_input") is not False:
        raise _artifact_error("capability labels must not be runtime inputs")
    forbidden_payload_keys = {
        "labels", "target", "capability_labels", "oracle_model",
        "oracle_state_dict", "sidecar_oracle_state_dict",
    }
    if forbidden_payload_keys.intersection(payload):
        raise _artifact_error("artifact contains labels or a sidecar oracle payload")
    expected_names = identifier_artifact_weight_names(policy.config)
    weights = payload.get("weights")
    if not isinstance(weights, dict) or tuple(payload.get("weight_names", ())) != expected_names:
        raise _artifact_error("weight_names do not exactly match production identifier weights")
    if set(weights) != set(expected_names):
        raise _artifact_error("weights contain missing or unexpected parameters")
    if any("oracle" in name.lower() or "label" in name.lower() for name in weights):
        raise _artifact_error("oracle/label parameters are not production weights")
    expected_config = asdict(policy.config)
    if payload.get("config") != expected_config:
        raise _artifact_error("policy configuration does not match A1")
    if payload.get("cadence_semantics_version") != CADENCE_SEMANTICS_VERSION:
        raise _artifact_error("cadence semantics are stale")
    if payload.get("runtime_contract_sha256") != runtime_contract_hash():
        raise _artifact_error("runtime equations or state contract changed")
    if payload.get("cadence_semantics") != IDENTIFIER_ARTIFACT_CADENCE:
        raise _artifact_error("cadence contract does not match production")
    probe = payload.get("probe")
    expected_probe = {
        "contract_version": PROBE_CONTRACT_VERSION,
        "period": PROBE_PERIOD,
        "amplitude": float(policy.config.burn_in_probe_amplitude),
        "waveform_sha256": WAVEFORM_SHA256,
    }
    if probe != expected_probe:
        raise _artifact_error("probe contract/hash does not match production")
    if payload.get("feature_schema_sha256") != feature_schema_sha256():
        raise _artifact_error("feature-schema hash does not match production")
    if payload.get("feature_schema") != feature_schema_metadata():
        raise _artifact_error("feature schema metadata does not match production")
    expected_q2_hash = sha256_file(q2_checkpoint)
    if payload.get("q2_checkpoint_sha256") != expected_q2_hash:
        raise _artifact_error("Q2 checkpoint hash does not match source checkpoint")
    if causal_oracle_report is not None:
        expected_oracle_hash = sha256_file(causal_oracle_report)
        if payload.get("causal_oracle_report_sha256") != expected_oracle_hash:
            raise _artifact_error("causal oracle report hash does not match")
    elif payload.get("causal_oracle_report_sha256") is not None:
        raise _artifact_error("causal oracle report path is required for this bound artifact")
    if payload.get("identifier_weights_sha256") != identifier_weights_hash(weights):
        raise _artifact_error("identifier weight hash is malformed")
    model_state = policy.state_dict()
    for name in expected_names:
        value = weights[name]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(model_state[name].shape):
            raise _artifact_error(f"shape mismatch for {name}")
        if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
            raise _artifact_error(f"non-finite/non-floating weight {name}")
    with torch.no_grad():
        for name in expected_names:
            model_state[name].copy_(weights[name].to(model_state[name]))
    return {
        "path": str(artifact_path.resolve()),
        "sha256": sha256_file(artifact_path),
        "identifier_weights_sha256": payload["identifier_weights_sha256"],
        "q2_checkpoint_sha256": expected_q2_hash,
        "feature_schema_sha256": payload["feature_schema_sha256"],
        "probe_waveform_sha256": WAVEFORM_SHA256,
        "artifact_schema_version": IDENTIFIER_ARTIFACT_SCHEMA_VERSION,
    }


def validate_identifier_pretraining_report(
    path: str | Path,
    *,
    identifier_artifact: str | Path,
    identifier_contract: Mapping[str, Any],
    policy: Any,
    q2_checkpoint: str | Path,
    causal_oracle_report: str | Path,
    probe_v4_report: str | Path | None = None,
) -> dict[str, Any]:
    """Bind an A1 identifier init to its independent accuracy gates.

    Loading a structurally valid tensor artifact is not sufficient: a stale
    artifact from a failed producer run must not bypass the calls100/125
    capability-mean validation.  This report is therefore a mandatory second
    half of the production identifier-init contract.
    """

    report_path = Path(path)
    if not report_path.is_file():
        raise RuntimeError(f"identifier pretraining report is missing: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid identifier pretraining report: {report_path}") from exc
    if not isinstance(report, dict):
        raise RuntimeError("identifier pretraining report must be a JSON object")
    if report.get("stage") != "identifier_pretrain" or report.get(
        "diagnostic"
    ) != "production-structured-identifier-pretraining":
        raise RuntimeError("identifier pretraining report has an unexpected stage/diagnostic")
    required_true = (
        "formal_gate_passed", "pretraining_gate_passed",
        "independent_validation_passed", "independent_final_passed",
        "identifier_artifact_written", "teacher_is_frozen",
        "teacher_action_is_executed",
        "runtime_features_from_deployable_observation_action_history",
    )
    failed = [name for name in required_true if report.get(name) is not True]
    if failed:
        raise RuntimeError(
            "identifier pretraining accuracy/provenance gate is not passed: "
            + ", ".join(failed)
        )
    if report.get("capability_labels_used_as_runtime_features") is not False:
        raise RuntimeError("identifier pretraining used capability labels as runtime features")
    if report.get("sidecar_oracle_gru") is not False:
        raise RuntimeError("identifier pretraining report contains a sidecar oracle GRU")
    if report.get("artifact_schema_version") != IDENTIFIER_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("identifier pretraining report has a stale artifact schema")
    if report.get("source_checkpoint_sha256") != sha256_file(q2_checkpoint):
        raise RuntimeError("identifier pretraining report does not bind the Q2 checkpoint")
    artifact_record = report.get("identifier_init_artifact")
    if not isinstance(artifact_record, Mapping):
        raise RuntimeError("identifier pretraining report has no artifact record")
    if artifact_record.get("sha256") != sha256_file(identifier_artifact):
        raise RuntimeError("identifier pretraining report does not bind the identifier artifact")
    expected_weights = identifier_contract.get("identifier_weights_sha256")
    if (artifact_record.get("identifier_weights_sha256") != expected_weights
            or report.get("identifier_weights_sha256") != expected_weights):
        raise RuntimeError("identifier pretraining report identifier hash mismatch")
    if report.get("config") != asdict(policy.config):
        raise RuntimeError("identifier pretraining report policy config mismatch")
    oracle = report.get("causal_oracle_report")
    if not isinstance(oracle, Mapping) or oracle.get("report_sha256") != sha256_file(
        causal_oracle_report
    ):
        raise RuntimeError("identifier pretraining report causal-oracle hash mismatch")
    probe = report.get("probe_v4")
    if not isinstance(probe, Mapping) or probe.get("frozen_sha256") != WAVEFORM_SHA256:
        raise RuntimeError("identifier pretraining report does not bind the frozen v4 probe")
    if probe_v4_report is not None:
        current_probe = Path(probe_v4_report)
        if not current_probe.is_file():
            raise RuntimeError("the current frozen v4 probe report is missing")
        if probe.get("report_sha256") != sha256_file(current_probe):
            raise RuntimeError(
                "identifier pretraining report does not bind the current frozen v4 report"
            )
    seeds = report.get("bank_seeds")
    if not isinstance(seeds, Mapping) or list(seeds.get("blind", ())) != []:
        raise RuntimeError("identifier pretraining report consumed an undeclared blind bank")
    for split in ("validation", "final"):
        record = report.get(split)
        if (not isinstance(record, Mapping)
                or record.get("phase_a_mean_gate_passed") is not True
                or not isinstance(record.get("phase_a_mean_gate"), Mapping)
                or record["phase_a_mean_gate"].get("gate_passed") is not True):
            raise RuntimeError(
                f"identifier pretraining {split} calls100/125 mean gate is not passed"
            )
    return {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)}


def validate_causal_revalidation_report(
    path: str | Path,
    *,
    phase_a1_checkpoint: str | Path,
    a1_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the separate post-A1 causal gate before allowing A2."""

    report_path = Path(path)
    if not report_path.is_file():
        raise RuntimeError(f"causal revalidation report is missing: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid causal revalidation report: {report_path}") from exc
    if not isinstance(report, dict) or report.get("diagnostic") != "causal-identifier-revalidation":
        raise RuntimeError("causal revalidation report has an unexpected diagnostic kind")
    if report.get("causal_gate_revalidated") is not True:
        raise RuntimeError("causal gate must be revalidated after A1 identifier updates")
    if report.get("cadence_semantics_version") != CADENCE_SEMANTICS_VERSION:
        raise RuntimeError("causal revalidation report has stale cadence semantics")
    if report.get("feature_schema_sha256") != feature_schema_sha256():
        raise RuntimeError("causal revalidation feature-schema hash mismatch")
    if report.get("post_a1_checkpoint_sha256") != sha256_file(phase_a1_checkpoint):
        raise RuntimeError("causal revalidation does not bind the A1 checkpoint")
    expected_identifier_hash = a1_report.get("identifier_weights_sha256_after")
    if report.get("identifier_weights_sha256") != expected_identifier_hash:
        raise RuntimeError("causal revalidation identifier hash does not match A1")
    return {"path": str(report_path.resolve()), "sha256": sha256_file(report_path), **report}


def require_current_cadence_semantics(payload: dict, *, context: str) -> None:
    """Reject structured artifacts made before the active publication rule."""

    version = payload.get("cadence_semantics_version")
    if version is None and isinstance(payload.get("report"), dict):
        version = payload["report"].get("cadence_semantics_version")
    if version != CADENCE_SEMANTICS_VERSION:
        raise RuntimeError(
            f"{context} requires cadence semantics {CADENCE_SEMANTICS_VERSION!r}; "
            f"checkpoint has {version!r} and is stale"
        )


def require_formal_identification_config(config) -> None:
    if (config.identification_publish_start != 100 or config.slow_cadence != 25
            or config.burn_in_probe_amplitude != 0.0 or config.motor_observer_bank_size != 35
            or config.motor_tau_grid_version != 2):
        raise RuntimeError("formal identification requires passive v5, call100, cadence25, and K35 grid v2")


def deployment_policy_hash(policy) -> str:
    """Hash the exact deployed policy, including calibration buffers.

    The quantile, validity and sample-count buffers affect release semantics,
    so all three are included.  Keeping this in one module prevents a
    calibrated checkpoint from being compared with a pre-calibration hash.
    """

    return _hash_policy(policy, excluded=())


def base_policy_hash(policy) -> str:
    """Hash weights/config while excluding mutable calibration buffers."""

    return _hash_policy(policy, excluded=CALIBRATION_STATE_NAMES)


def _hash_policy(policy, *, excluded) -> str:
    digest = hashlib.sha256()
    for name, value in policy.state_dict().items():
        if name in excluded:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps(asdict(policy.config), sort_keys=True).encode("utf-8"))
    digest.update(CADENCE_SEMANTICS_VERSION.encode("utf-8"))
    digest.update(runtime_contract_hash().encode("utf-8"))
    return digest.hexdigest()
