"""Explicit, weights-only transfer into the goal-conditioned native Actor.

This module never restores an optimizer, RNG or sampling index.
Exact continuation remains the training module's strict resume operation.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
from pathlib import Path

import torch

from .env_raptor import ACTION_CONVENTION, RaptorParams
from .response_policy import (
    AGILE_ARCHITECTURE,
    ARCHITECTURE,
    CONTROL_FEATURE_DIM,
    ResponseMotorPolicy,
    ResponsePolicyConfig,
)

IMPORT_VERSION = "native-hover-to-goal36-weights-only-v1"
REFERENCE_SCHEMA = "raptor-multi-airframe-gaussian-v6"
ACTOR_KEYS = frozenset((
    "response_memory.weight_ih", "response_memory.weight_hh",
    "response_memory.bias_ih", "response_memory.bias_hh",
    "readout.weight", "readout.bias",
))


def _model_digest(state: dict[str, torch.Tensor]) -> str:
    """Match the existing checkpoint digest before any padding or dtype cast."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        for metadata in (name, str(value.dtype), str(tuple(value.shape))):
            digest.update(metadata.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _source_config(value: dict) -> ResponsePolicyConfig:
    if value.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("initialization requires a supported structured RAPTOR checkpoint schema")
    architecture = value.get("architecture")
    if architecture not in (ARCHITECTURE, AGILE_ARCHITECTURE):
        raise ValueError("initialization Actor architecture is incompatible")
    try:
        source_policy = value["policy_config"]
        binding = value["binding"]
        protocol = binding["protocol"]
        environment = protocol["environment"]
        if (not all(isinstance(item, dict) for item in
                    (source_policy, binding, protocol, environment, protocol["policy"]))
                or not {"memory_dim", "dt"}.issubset(source_policy)
                or not {"memory_dim", "dt"}.issubset(protocol["policy"])):
            raise ValueError("incomplete policy/environment metadata")
        config = ResponsePolicyConfig(**source_policy)
        params = RaptorParams(**protocol["environment_params"])
        recorded_policy = ResponsePolicyConfig(**protocol["policy"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("malformed initialization policy/environment metadata") from error
    if (protocol.get("version") != value["schema"]
            or protocol.get("architecture") != architecture
            or recorded_policy != config):
        raise ValueError("initialization policy metadata is inconsistent")
    if (config.goal_conditioning != (architecture == AGILE_ARCHITECTURE)
            or environment.get("action_convention") != ACTION_CONVENTION):
        raise ValueError("initialization goal interface or motor semantics mismatch")
    if (config.dt != params.dt or environment.get("dt") != params.dt
            or binding.get("dtype") not in ("float32", "float64")):
        raise ValueError("initialization policy, environment timestep or dtype mismatch")
    return config


def import_hover_checkpoint(policy: ResponseMotorPolicy, path: str | Path) -> dict:
    """Import a compatible structured hover/agile checkpoint, using fresh state.

    Hover: preserve all six tensors and zero-pad GRU input columns 16 -> 36.
    Agile: copy the same six tensors strictly, retaining learned goal columns.
    A raw state_dict cannot establish the action convention and is rejected.
    All metadata, shapes, finite values and the original model hash are checked
    before any destination tensor is modified. The caller owns fresh Adam.
    """
    if not isinstance(policy, ResponseMotorPolicy) or not policy.config.goal_conditioning:
        raise ValueError("weights-only agile initialization needs a goal-conditioned policy")
    source = Path(path)
    source_bytes = source.read_bytes()
    file_digest = hashlib.sha256(source_bytes).hexdigest()
    value = torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("initialization requires a structured checkpoint, not raw weights")
    config = _source_config(value)
    if config.memory_dim != policy.config.memory_dim or config.dt != policy.config.dt:
        raise ValueError("initialization hidden size or timestep does not match the destination")
    expected = policy.state_dict()
    state = value.get("model")
    if not isinstance(state, dict) or set(state) != ACTOR_KEYS or set(expected) != ACTOR_KEYS:
        raise ValueError("initialization must contain exactly the six native Actor tensors")
    input_columns = policy.response_memory.input_size if config.goal_conditioning else CONTROL_FEATURE_DIM
    source_dtype = getattr(torch, value["binding"]["dtype"])
    for name, target in expected.items():
        tensor = state[name]
        shape = (target.shape[0], input_columns) if name == "response_memory.weight_ih" else target.shape
        if (target.dtype not in (torch.float32, torch.float64)
                or not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.dtype != source_dtype or tensor.shape != shape
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("initialization tensor shape/dtype/nonfinite mismatch: " + name)
    source_digest = _model_digest(state)
    if value.get("model_sha256") != source_digest:
        raise ValueError("initialization checkpoint model digest mismatch")

    # Validate conversion before mutating the destination. In particular, finite
    # FP64 values that overflow the requested FP32 dtype must not be imported.
    converted = {}
    for name, target in expected.items():
        tensor = state[name].to(device=target.device, dtype=target.dtype)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("initialization weights overflow the destination dtype: " + name)
        if name == "response_memory.weight_ih" and not config.goal_conditioning:
            padded = torch.zeros_like(target)
            padded[:, :CONTROL_FEATURE_DIM].copy_(tensor)
            tensor = padded
        converted[name] = tensor
    policy.load_state_dict(converted, strict=True)
    return {
        "file_sha256": file_digest,
        "source_model_sha256": source_digest,
        "source_architecture": value["architecture"],
        "target_architecture": policy.architecture,
        "source_policy_config": asdict(config),
        "migration_version": IMPORT_VERSION,
        "weights_only": True,
        "input_columns_copied": input_columns,
        "input_columns_added": policy.response_memory.input_size - input_columns,
    }
