"""Versioned read-only motor probe artifact shared by policy and diagnostics."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import torch

PROBE_CONTRACT_VERSION = "v4"
PROBE_PERIOD = 50
PROBE_BLOCK = 25
PROBE_ENTRIES = (-1, 0, 1)
PROBE_AMPLITUDE = 0.005
SHARED_TAU_MIN_POOLED = 12
SHARED_TAU_MIN_CALLS = 3
SHARED_TAU_MIN_MOTORS = 2
# The pooled support count alone can be satisfied by numerically tiny rows.
# Keep an explicit, contract-level information floor for both tau branches.
SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION = (
    SHARED_TAU_MIN_POOLED * (PROBE_AMPLITUDE / 2.0) ** 3
)

WAVEFORM: tuple[tuple[int, int, int, int], ...] = (
    (-1, 1, 1, 1), (-1, 1, 1, 1), (0, 0, 0, 0), (1, 1, -1, -1),
    (1, 1, -1, -1), (0, 0, 0, 0), (-1, 0, 0, -1), (-1, 1, -1, -1),
    (0, 1, -1, 0), (1, 0, 0, 1), (1, -1, 1, 1), (0, -1, 1, 0),
    (1, 0, 0, 0), (1, -1, 1, -1), (0, -1, 1, -1), (-1, 0, 0, 0),
    (-1, -1, -1, -1), (0, -1, -1, -1), (0, 0, 0, 0), (1, 1, 1, 1),
    (1, 1, 1, 1), (0, 0, 0, 0), (-1, -1, -1, 1), (-1, -1, -1, 1),
    (0, 0, 0, 0), (-1, -1, -1, -1), (-1, -1, -1, -1), (0, 0, 0, 0),
    (1, 0, 1, 0), (1, 1, 1, 1), (0, 1, 0, 1), (1, 0, 0, 0),
    (1, -1, 1, 1), (0, -1, 1, 1), (-1, 0, 0, 0), (-1, 1, -1, 1),
    (0, 1, -1, 1), (-1, 0, 0, 0), (-1, 1, -1, 1), (0, 1, -1, 1),
    (1, 0, 0, 0), (1, -1, -1, -1), (0, -1, -1, -1), (1, 0, 0, 0),
    (1, 1, 1, -1), (0, 1, 1, -1), (-1, 0, 0, 0), (-1, -1, 1, -1),
    (0, -1, 1, -1), (0, 0, 0, 0),
)


def waveform_tensor(*, device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.int64) -> torch.Tensor:
    return torch.tensor(WAVEFORM, device=device, dtype=dtype)


def canonical_waveform_json(waveform: Iterable[Iterable[int]] = WAVEFORM) -> bytes:
    return json.dumps([[int(value) for value in row] for row in waveform],
                      separators=(",", ":"), ensure_ascii=True).encode("utf-8")


WAVEFORM_SHA256 = hashlib.sha256(canonical_waveform_json()).hexdigest()


def waveform_metadata() -> dict[str, object]:
    return {"contract_version": PROBE_CONTRACT_VERSION, "shape": [50, 4],
            "entries": list(PROBE_ENTRIES), "amplitude": PROBE_AMPLITUDE,
            "sha256": WAVEFORM_SHA256,
            "shared_tau_min_pooled": SHARED_TAU_MIN_POOLED,
            "shared_tau_min_calls": SHARED_TAU_MIN_CALLS,
            "shared_tau_min_motors": SHARED_TAU_MIN_MOTORS,
            "shared_tau_min_weighted_fisher_information":
                SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION}
