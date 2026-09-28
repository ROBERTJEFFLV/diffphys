"""TRAIN-only, fixed-quota physical coverage using unchanged clean reset draws.

128 sampling cells are NOT gradient groups. No trajectory, Actor, or EVAL result
is used to choose a row. Noise/pulses are attached only to the selected pool by
the caller. Old random sampling and fixed EVAL remain in response_training.
"""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import torch

from env_raptor import IMMUTABLE_TAPES, RaptorParams, RaptorSimulator, RaptorState
from response_noise import DisturbanceConfig

SAMPLING_VERSION = "physics-coverage128-conditional-yaw-v1"
CELL_COUNT = 128
TTI_EDGES = (330.0, 620.0, 910.0)
RISING_EDGES = (0.0475, 0.065, 0.0825)
FALLING_EDGES = (0.0975, 0.165, 0.2325)
MAX_CANDIDATE_ROUNDS = 16
DEFINITION_FILE = "configs/physics_coverage.json"
ROOT = Path(__file__).resolve().parent


def validate_sampling(mode: str, count: int) -> None:
    if mode not in ("random", "coverage128"):
        raise ValueError("train-sampling must be random or coverage128")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("TRAIN count must be a positive integer")
    if mode == "coverage128" and count % CELL_COUNT:
        raise ValueError("coverage128 needs a total TRAIN count divisible by 128 (4 * --scenarios)")


def sampling_contract(mode: str) -> dict:
    """Bind fixed thresholds, candidate protocol and generator to checkpoints."""
    validate_sampling(mode, CELL_COUNT)
    if mode == "random":
        return {"mode": mode, "version": "original-four-bank-random-v1"}
    data = json.loads((ROOT / DEFINITION_FILE).read_text())
    if data.get("version") != SAMPLING_VERSION:
        raise ValueError("unsupported physical coverage definition")
    for key, expected in (("tti_edges", TTI_EDGES), ("rising_edges", RISING_EDGES),
                          ("falling_edges", FALLING_EDGES)):
        if data.get(key) != list(expected):
            raise ValueError("physical coverage bin definition mismatch: " + key)
    thresholds = data.get("yaw_medians_by_tti_bin", [])
    if len(thresholds) != 4 or any(not math.isfinite(x) or x <= 0 for x in thresholds):
        raise ValueError("four finite positive conditional yaw thresholds are required")
    env_hash = hashlib.sha256((ROOT / "env_raptor.py").read_bytes()).hexdigest()
    if data.get("environment_source_sha256") != env_hash:
        raise ValueError("physical generator changed: review and recalibrate coverage thresholds")
    return {**data, "mode": mode, "cells": CELL_COUNT,
            "candidate_batch_rule": "max(1024,2*total_train)",
            "max_candidate_rounds": MAX_CANDIDATE_ROUNDS,
            "selection": "first-unique-draws-per-cell-then-seeded-shuffle",
            "definition_sha256": hashlib.sha256((ROOT / DEFINITION_FILE).read_bytes()).hexdigest()}


def yaw_authority(state: RaptorState) -> torch.Tensor:
    """Single-rotor km*Tmax/Jz, evaluated in FP64; not a stability certificate.

    All four rotors share km/thrust curves in this reference family. No extra
    arm-length division: the equivalent expression is TTI*(km/arm)*(Jx/Jz).
    """
    c = state.thrust_coefficients[:, 0].double()
    maximum = state.motor_max.double()
    thrust = c[:, 0] + c[:, 1] * maximum + c[:, 2] * maximum.square()
    return state.rotor_torque_constant[:, 0].double() * thrust / state.inertia[:, 2].double()


def physics_cell_ids(state: RaptorState, definition: dict | None = None) -> torch.Tensor:
    """Stable IDs: (((TTI*4 + rise)*4 + fall)*2 + conditional_yaw)."""
    definition = sampling_contract("coverage128") if definition is None else definition
    values = (state.torque_to_inertia, state.motor_time_rising[:, 0],
              state.motor_time_falling[:, 0])
    bounds = ((40.0, 1200.0), (0.03, 0.10), (0.03, 0.30))
    bins = []
    for value, edges, (low, high) in zip(values, (TTI_EDGES, RISING_EDGES, FALLING_EDGES), bounds):
        # Bounds/edges are represented in the source dtype, so FP32 endpoint
        # rounding does not reject legitimate reference samples.
        if not bool((torch.isfinite(value) & (value >= low) & (value <= high)).all()):
            raise ValueError("physical coverage feature is outside the reference range")
        bins.append(torch.bucketize(value.contiguous(), value.new_tensor(edges), right=True))
    authority = yaw_authority(state)
    if not bool((torch.isfinite(authority) & (authority > 0)).all()):
        raise ValueError("invalid yaw authority")
    thresholds = authority.new_tensor(definition["yaw_medians_by_tti_bin"])
    yaw_bin = (authority >= thresholds[bins[0]]).long()
    return ((bins[0] * 4 + bins[1]) * 4 + bins[2]) * 2 + yaw_bin


def _stream_seed(seeds: Sequence[int], label: str) -> int:
    payload = json.dumps([SAMPLING_VERSION, list(seeds), label], separators=(",", ":")).encode()
    # High positive namespace, disjoint from the fixed low-valued EVAL seeds.
    return (1 << 62) | (int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 62)-1))


def _select_clean(state: RaptorState, indices: torch.Tensor) -> RaptorState:
    if any(getattr(state, name).shape[1] != 1 for name in IMMUTABLE_TAPES):
        raise ValueError("select physical candidates BEFORE attaching noise/pulse tapes")
    return RaptorState(**{f.name: getattr(state, f.name).index_select(0, indices) for f in fields(state)})


@torch.no_grad()
def sample_coverage(count: int, seeds: Sequence[int], *, dt: float = .01,
                    dtype: torch.dtype = torch.float32) -> tuple[RaptorState, dict]:
    """Bounded rejection sampling from the original joint physical distribution.

    Original initial kinematics/motors are retained, with no success filtering.
    Exhaustion raises; there is no duplication, jitter, relaxed bin or fallback.
    Identical count/seeds/dtype/source reproduce the complete selected pool.
    """
    validate_sampling("coverage128", count)
    if not seeds or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
        raise ValueError("coverage needs integer TRAIN seeds")
    definition = sampling_contract("coverage128")
    simulator = RaptorSimulator(RaptorParams(dt))
    quota, batch_size = count // CELL_COUNT, max(1024, 2 * count)
    counts = torch.zeros(CELL_COUNT, dtype=torch.long)
    parts, candidate_seeds = [], []
    for round_index in range(MAX_CANDIDATE_ROUNDS):
        seed = _stream_seed(seeds, f"candidates:{round_index}")
        candidate_seeds.append(seed)
        candidate = simulator.reset(batch_size, seed=seed, dtype=dtype, horizon=1,
                                    disturbances=DisturbanceConfig.clean())
        ids = physics_cell_ids(candidate, definition)
        order = ids.argsort(stable=True)
        frequencies = torch.bincount(ids, minlength=CELL_COUNT)
        offsets = frequencies.cumsum(0) - frequencies
        rank = torch.arange(batch_size) - offsets[ids[order]]
        selected = order[rank < (quota - counts)[ids[order]]]
        if selected.numel():
            parts.append(_select_clean(candidate, selected))
            counts += torch.bincount(ids[selected], minlength=CELL_COUNT)
        if bool((counts == quota).all()):
            break
    else:
        missing = (counts < quota).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(f"coverage128 candidate budget exhausted; unfilled cells: {missing}")
    pooled = RaptorState(**{f.name: torch.cat([getattr(s, f.name) for s in parts]) for f in fields(parts[0])})
    generator = torch.Generator(device="cpu").manual_seed(_stream_seed(seeds, "shuffle"))
    initial = _select_clean(pooled, torch.randperm(count, generator=generator))
    initial = replace(initial, noise_row=torch.arange(count))
    report = {"mode": "coverage128", "version": SAMPLING_VERSION,
              "definition_sha256": definition["definition_sha256"],
              "total_train": count, "quota_per_cell": quota, "cell_counts": counts.tolist(),
              "candidate_batch_size": batch_size, "candidate_rounds": len(candidate_seeds),
              "candidates_drawn": batch_size * len(candidate_seeds), "candidate_seeds": candidate_seeds}
    return initial, report
