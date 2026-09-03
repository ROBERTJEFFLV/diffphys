"""Causal, version-independent identification feature primitives.

This module is intentionally free of policy state.  The production policy and
the read-only observer oracle both call these functions so that normalization,
modal mixing, and feature ordering cannot silently diverge.
"""
from __future__ import annotations

from typing import Sequence
import math
import hashlib
import json

import torch


ACTION_DIM = 4
GRAVITY = 9.80665
EXCITATION_SCALE = 0.10
EXCITATION_LAGS = (0, 4, 12)
EXCITATION_HISTORY_LEN = 13

# This describes the feature ordering consumed by production's identifier and
# by the read-only oracle.  Keep it data-only and versioned: an artifact made
# with a different ordering must never be loaded as an identifier init.
FEATURE_SCHEMA_VERSION = "production_identifier_features_v1"


def feature_schema_metadata() -> dict[str, object]:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "legacy24_dim": 24,
        "legacy24_lags": list(EXCITATION_LAGS),
        "legacy24_history_len": EXCITATION_HISTORY_LEN,
        "legacy24_excitation_scale": EXCITATION_SCALE,
        "bank_modal_channels_per_mode": 8,
        "bank_modal_order": ["response_times_modal_state", "modal_state_squared"],
    }


def feature_schema_sha256() -> str:
    canonical = json.dumps(feature_schema_metadata(), sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def normalize_response(
    specific_force_body: torch.Tensor,
    angular_acceleration: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact production force, angular, and collective channels.

    Inputs are the body-frame specific force (m/s²) and angular acceleration
    (rad/s²) produced by the *same* transition as the supplied excitation.
    ``response_mask`` is applied after normalization, matching the burn-in
    semantics of the policy.
    """

    if specific_force_body.shape[-1] != 3 or angular_acceleration.shape[-1] != 3:
        raise ValueError("response tensors must have final dimension 3")
    force = (torch.asinh(specific_force_body / GRAVITY) /
             specific_force_body.new_tensor(math.asinh(5.5))).clamp(-1.0, 1.0)
    angular = (torch.asinh(angular_acceleration / 35.0) /
               angular_acceleration.new_tensor(math.asinh(2200.0 / 35.0))).clamp(-1.0, 1.0)
    collective = (torch.asinh(specific_force_body[..., 2:3] / GRAVITY - 1.0) /
                  specific_force_body.new_tensor(math.asinh(4.5)))
    if response_mask is not None:
        mask = response_mask.to(dtype=force.dtype)
        while mask.ndim < force.ndim:
            mask = mask.unsqueeze(-1)
        force = force * mask
        angular = angular * mask
        collective = collective * mask
    return force, angular, collective


def modal(motor_state: torch.Tensor) -> torch.Tensor:
    """Map four motor channels to collective/roll/pitch/yaw coordinates."""

    if motor_state.shape[-1] != ACTION_DIM:
        raise ValueError("motor_state must have final dimension 4")
    return torch.stack((
        motor_state.mean(dim=-1),
        (motor_state[..., 1] - motor_state[..., 3]) / 2.0,
        (motor_state[..., 2] - motor_state[..., 0]) / 2.0,
        (motor_state[..., 0] - motor_state[..., 1]
         + motor_state[..., 2] - motor_state[..., 3]) / 4.0,
    ), dim=-1)


def sol_response(collective: torch.Tensor, angular: torch.Tensor) -> torch.Tensor:
    """Return the registered [collective, roll, pitch, yaw] response vector."""

    if collective.shape[-1:] != (1,) or angular.shape[-1] != 3:
        raise ValueError("collective must end in 1 and angular in 3 channels")
    return torch.cat((collective, angular), dim=-1)


def production_legacy24(
    excitation_history: torch.Tensor,
    force_response: torch.Tensor,
    angular_response: torch.Tensor,
    motor_delta: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    lags: Sequence[int] = EXCITATION_LAGS,
    excitation_scale: float = EXCITATION_SCALE,
) -> torch.Tensor:
    """Build the production 24-D causal identifier context.

    ``excitation_history[:, 0]`` and ``motor_delta`` belong to the transition
    whose response is being passed in.  The explicit arguments make this
    alignment auditable and prevent callers from accidentally using predicted
    actions or a post-transition response.
    """

    if excitation_history.ndim != 3 or excitation_history.shape[-2:] != (EXCITATION_HISTORY_LEN, ACTION_DIM):
        raise ValueError("excitation_history must have shape [batch,13,4]")
    if force_response.shape[-1] != 3 or angular_response.shape[-1] != 3:
        raise ValueError("response tensors must have final dimension 3")
    mask = response_mask.to(force_response.dtype)
    while mask.ndim < force_response.ndim:
        mask = mask.unsqueeze(-1)
    force_response = force_response * mask
    angular_response = angular_response * mask
    collective = excitation_history.mean(dim=-1)
    roll = excitation_history[..., 1] - excitation_history[..., 3]
    pitch = excitation_history[..., 2] - excitation_history[..., 0]
    yaw = (excitation_history[..., 0] - excitation_history[..., 1]
           + excitation_history[..., 2] - excitation_history[..., 3])
    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    force_blocks = []
    angular_blocks = []
    for lag in tuple(lags):
        if lag < 0 or lag >= EXCITATION_HISTORY_LEN:
            raise ValueError("identification lag must be in [0,12]")
        force_blocks.append((collective[:, lag, None] * force_response).clamp(-5.0, 5.0))
        angular_blocks.append((rpy[:, lag] * angular_response).clamp(-5.0, 5.0))
    aligned = excitation_history[:, 0]
    normalized_delta = (motor_delta / float(excitation_scale)).clamp(-5.0, 5.0)
    rise = (torch.relu(aligned) * torch.relu(normalized_delta)).mean(dim=-1, keepdim=True).clamp(0.0, 25.0)
    fall = (torch.relu(-aligned) * torch.relu(-normalized_delta)).mean(dim=-1, keepdim=True).clamp(0.0, 25.0)
    energy = aligned.square().clamp(0.0, 25.0)
    result = torch.cat((*force_blocks, *angular_blocks,
                        mask * rise, mask * fall, energy), dim=-1)
    if result.shape[-1] != 24:
        raise RuntimeError(f"legacy feature contract produced {result.shape[-1]} channels")
    return result


def bank_modal_features(
    motor_state: torch.Tensor,
    response_y: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-observer ``[x*y, x²]`` features, shape ``[..., K, 8]``."""

    if motor_state.ndim < 2 or motor_state.shape[-1] != ACTION_DIM:
        raise ValueError("motor_state must have shape [...,K,4]")
    if response_y.shape[-1] != 4:
        raise ValueError("response_y must have final dimension 4")
    x = modal(motor_state).clamp(-1.0, 1.0)
    y = response_y
    if response_mask is not None:
        mask = response_mask.to(dtype=y.dtype)
        while mask.ndim < y.ndim:
            mask = mask.unsqueeze(-1)
        y = y * mask
    else:
        mask = None
    y = y.unsqueeze(-2)
    result = torch.cat(((x * y).clamp(-25.0, 25.0),
                        x.square().clamp(0.0, 25.0)), dim=-1)
    if mask is not None:
        result = result * mask.unsqueeze(-2)
    return result


__all__ = [
    "ACTION_DIM", "EXCITATION_HISTORY_LEN", "EXCITATION_LAGS",
    "EXCITATION_SCALE", "FEATURE_SCHEMA_VERSION", "GRAVITY",
    "bank_modal_features", "feature_schema_metadata", "feature_schema_sha256", "modal",
    "normalize_response", "production_legacy24", "sol_response",
]
