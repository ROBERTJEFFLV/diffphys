"""Collective motor-lag experiment, separate from the historical v4 artifact.

The shared rise/fall parameters do not require four independent test inputs.
Equal command increments lie in the static differential mixer nullspace.
This is a command-space property, not a guarantee of zero physical torque:
unequal motor states, asymmetric lags, thrust clipping and feedback still act.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

VERSION = "passive_identification_v5"
AMPLITUDE = 0.005
DEFAULT_AMPLITUDE = 0.0
DWELL_STEPS = 25
ACTIVE_STEPS = 50
RECOVERY_STEPS = 25
PUBLISH_START = DWELL_STEPS + ACTIVE_STEPS + RECOVERY_STEPS
PUBLICATION_CALLS = (PUBLISH_START, PUBLISH_START + 25)
MIN_COLLECTION_HORIZON = PUBLICATION_CALLS[-1] + 1
# Balanced short/long pulses.  All four commands receive the SAME scalar.
PULSE = (1,) * 2 + (0,) * 2 + (-1,) * 2 + (0,) * 2 + (1,) * 4 + (0,) * 2 + (-1,) * 4 + (0,) * 7
WAVEFORM = PULSE * 2
assert len(WAVEFORM) == ACTIVE_STEPS and sum(WAVEFORM) == 0
TRAIN_SEEDS = (3707, 4707, 5707, 6707)
VALIDATION_SEED = 7707
FORMAL_SCENARIOS = 64
FORMAL_HORIZON = 125
Q2_SHA256 = "b401dc6f02beadf51d1b55b24b9056f0b00f17a7a8a5d94d3675554fdb15370d"


def metadata() -> dict:
    return {
        "version": VERSION, "amplitude": DEFAULT_AMPLITUDE,
        "mode": "passive_first", "experimental_active_amplitude": AMPLITUDE,
        "waveform": list(WAVEFORM),
        "dwell_steps": DWELL_STEPS, "active_steps": ACTIVE_STEPS,
        "recovery_steps": RECOVERY_STEPS, "publication_calls": list(PUBLICATION_CALLS),
        "guard_active_calls": [DWELL_STEPS, PUBLISH_START - 1],
        "command_direction": [1, 1, 1, 1], "residual_slew_per_call": AMPLITUDE,
        "guards": {"position_norm": 5.0, "velocity_norm": 5.0,
                   "omega_norm": "max(5, 1.5*dwell_peak)", "body_z_min": 0.0},
        "safety_ratio": 1.05, "safety_atol": 1e-7,
        "coverage_min": 0.90, "coverage_per_cell_min": 0.50,
        "energy_retention_min_for_active_experiments": 0.90,
        "passive_support_window": [0, PUBLISH_START],
        "train_seeds": list(TRAIN_SEEDS), "validation_seed": VALIDATION_SEED,
        "formal_scenarios": FORMAL_SCENARIOS, "formal_horizon": FORMAL_HORIZON,
        "q2_sha256": Q2_SHA256,
        "identifies": ["shared_tau_rise", "shared_tau_fall"],
        "angular_capability_authorization": False,
    }


CONTRACT_SHA256 = hashlib.sha256(json.dumps(metadata(), sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()


@dataclass
class ProbeState:
    residual: torch.Tensor  # [batch,1], physically executed scalar increment
    aborted: torch.Tensor  # [batch], sticky for the entire episode
    omega_reference: torch.Tensor | None = None  # measured, probe-free dwell peak

    @classmethod
    def initial(cls, action: torch.Tensor) -> "ProbeState":
        return cls(torch.zeros_like(action[:, :1]),
                   torch.zeros(action.shape[0], dtype=torch.bool, device=action.device),
                   torch.zeros_like(action[:, :1]))


def requested_probe(call: int | torch.Tensor, action: torch.Tensor,
                    amplitude: float = AMPLITUDE) -> torch.Tensor:
    if amplitude not in (0.0, AMPLITUDE):
        raise ValueError("v5 amplitude is registered: only zero or 0.005 is allowed")
    index = torch.as_tensor(call, device=action.device, dtype=torch.long).reshape(-1)
    index = index.expand(action.shape[0]) - DWELL_STEPS
    table = action.new_tensor(WAVEFORM)
    active = (index >= 0) & (index < ACTIVE_STEPS)
    return (amplitude * table[index.clamp(0, ACTIVE_STEPS - 1)] * active)[:, None]


def apply_probe(action: torch.Tensor, state: ProbeState, call: int | torch.Tensor,
                *, position: torch.Tensor, velocity: torch.Tensor,
                omega: torch.Tensor, body_z: torch.Tensor,
                amplitude: float = DEFAULT_AMPLITUDE,
                lower: torch.Tensor | None = None,
                upper: torch.Tensor | None = None) -> tuple[torch.Tensor, ProbeState, torch.Tensor]:
    """Intersect ALL motor bounds before projecting the one scalar input.

No privileged authority/motor/force is read.  Additional allocator box bounds
can be supplied; an empty intersection aborts rather than adding a torque.
An emergency abort overrides residual slew and returns the base action.
"""
    if action.ndim != 2 or action.shape[-1] != 4:
        raise ValueError("action must be [batch,4]")
    if state.residual.shape != action[:, :1].shape or state.aborted.shape != action.shape[:1]:
        raise ValueError("probe state has incompatible shape")
    if not bool(torch.isfinite(action).all()) or bool((action.abs() > 1.0).any()):
        raise ValueError("base action must be finite and inside [-1,1]")
    requested = requested_probe(call, action, amplitude)
    calls = torch.as_tensor(call, device=action.device).reshape(-1).expand(action.shape[0])
    reference = torch.zeros_like(action[:, :1]) if state.omega_reference is None else state.omega_reference
    reference = torch.where((calls < DWELL_STEPS)[:, None],
                            torch.maximum(reference, omega.norm(dim=-1, keepdim=True)), reference)
    if amplitude == 0.0:
        return action, ProbeState(torch.zeros_like(state.residual), state.aborted, reference), requested
    observations = torch.cat((position, velocity, omega, body_z), -1)
    safe = (torch.isfinite(observations).all(-1)
            & (position.norm(dim=-1) <= 5.0) & (velocity.norm(dim=-1) <= 5.0)
            & (omega.norm(dim=-1) <= (1.5 * reference.squeeze(-1)).clamp_min(5.0))
            & (body_z[:, 2] > 0.0))
    lo = torch.full_like(action, -1.0) if lower is None else lower
    hi = torch.full_like(action, 1.0) if upper is None else upper
    scalar_lo = torch.maximum((lo - action).amax(-1, keepdim=True),
                              (state.residual - AMPLITUDE).clamp_min(-AMPLITUDE))
    scalar_hi = torch.minimum((hi - action).amin(-1, keepdim=True),
                              (state.residual + AMPLITUDE).clamp_max(AMPLITUDE))
    monitored = (calls >= DWELL_STEPS) & (calls < PUBLISH_START)
    infeasible = (scalar_lo > scalar_hi).squeeze(-1)
    aborted = state.aborted | (monitored & (~safe | infeasible))
    scalar = torch.minimum(torch.maximum(requested, scalar_lo), scalar_hi)
    scalar = torch.where((aborted | infeasible | ~monitored)[:, None], torch.zeros_like(scalar), scalar)
    return action + scalar, ProbeState(scalar, aborted, reference), requested
