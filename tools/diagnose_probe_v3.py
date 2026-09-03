"""Design and diagnose the pre-registered 50-step motor probe (v3).

The probe is intentionally a diagnostic artifact.  It does not modify a
policy, consume blind seeds, or use simulator motor truth for a deployability
claim.  The design checks are kept here (instead of in the policy module) so a
waveform hash can be frozen before a training/validation run.

The formal arm uses a frozen Q2 policy and applies the requested residual with
the exact deployable annulus::

    L=max(-A, d_previous-A, -1-q)
    U=min( A, d_previous+A,  1-q)
    d=clip(A*W, L, U),       action=q+d.

``python tools/diagnose_probe_v3.py --dry-run`` only checks and publishes the
waveform, which makes the contract cheap to test on machines without a Q2
checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from policy_observation import (  # noqa: E402
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)
from structured_distillation import build_dagger_scenario_bank  # noqa: E402


ACTION_DIM = 4
PROBE_PERIOD = 50
PROBE_BLOCK = 25
PROBE_AMPLITUDE = 0.005
PROBE_ENTRIES = (-1, 0, 1)
SEARCH_SEED = 20260806
TRAIN_SEEDS = (3707, 4707, 5707, 6707)
VALIDATION_SEED = 7707
BLIND_SEEDS = (10707, 11707)
FORMAL_HORIZON = 125
TAIL_START = 75
OBSERVER_TAU = 0.06

# This table was selected by the fixed-seed, lexicographic design search.  It
# is data, not a policy schedule: changing one entry changes the SHA and must
# therefore be reviewed as a new probe release.
PRE_REGISTERED_WAVEFORM: tuple[tuple[int, int, int, int], ...] = (
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


def _canonical_waveform(waveform: Iterable[Iterable[int]]) -> bytes:
    table = [[int(value) for value in row] for row in waveform]
    return json.dumps(table, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def waveform_sha256(waveform: Iterable[Iterable[int]] = PRE_REGISTERED_WAVEFORM) -> str:
    return hashlib.sha256(_canonical_waveform(waveform)).hexdigest()


def modal_coordinates(waveform: torch.Tensor) -> torch.Tensor:
    """Return [collective, roll, pitch, yaw] coordinates for [T,4] rows."""
    if waveform.ndim != 2 or waveform.shape[-1] != 4:
        raise ValueError("waveform must have shape [time,4]")
    return torch.stack((
        waveform.mean(-1),
        (waveform[:, 1] - waveform[:, 3]) / 2,
        (waveform[:, 2] - waveform[:, 0]) / 2,
        (waveform[:, 0] - waveform[:, 1] + waveform[:, 2] - waveform[:, 3]) / 4,
    ), dim=-1)


def _pulse_lengths(values: torch.Tensor, sign: int) -> list[int]:
    locations = torch.nonzero(values == sign, as_tuple=False).flatten().tolist()
    lengths: list[int] = []
    cursor = 0
    while cursor < len(locations):
        end = cursor + 1
        while end < len(locations) and locations[end] == locations[end - 1] + 1:
            end += 1
        lengths.append(end - cursor)
        cursor = end
    return lengths


def lag_design(waveform: torch.Tensor) -> torch.Tensor:
    """Build X=[z_t,z_t-4,z_t-12] using the actual four motor coordinates."""
    if waveform.shape[0] <= 12:
        raise ValueError("waveform must contain more than 12 rows")
    return torch.cat((waveform[12:], waveform[8:-4], waveform[:-12]), dim=-1)


def standardized_rank_condition(design: torch.Tensor) -> dict[str, Any]:
    if design.ndim != 2:
        raise ValueError("design must be a matrix")
    centered = design.double() - design.double().mean(0)
    scale = centered.std(0, unbiased=False)
    standardized = centered / scale.clamp_min(1.0e-12)
    singular = torch.linalg.svdvals(standardized)
    threshold = float(singular.max().item()) * 1.0e-8 if singular.numel() else 0.0
    rank = int((singular > threshold).sum().item())
    condition = float(singular.max().item() / max(float(singular.min().item()), 1.0e-12))
    return {
        "feature_order": ["z_t", "z_t_minus_4", "z_t_minus_12"],
        "feature_dim": int(design.shape[-1]), "sample_count": int(design.shape[0]),
        "rank": rank, "condition": condition,
        "singular_values": [float(v) for v in singular], "rank12": rank == 12,
    }


def projected_coriolis_residual_ratio(actuation: torch.Tensor,
                                      coriolis: torch.Tensor) -> torch.Tensor:
    """Return ``||c-P_a c||/||c||`` per scene for 2-D exact-regression blocks."""
    if actuation.shape != coriolis.shape or actuation.shape[-1] != 2 or actuation.ndim not in (2, 3):
        raise ValueError("actuation and coriolis must have shape [time,2] or [scene,time,2]")
    if actuation.ndim == 2:
        actuation, coriolis = actuation.unsqueeze(0), coriolis.unsqueeze(0)
    a = actuation.reshape(actuation.shape[0], -1).double()
    c = coriolis.reshape(coriolis.shape[0], -1).double()
    coefficient = (a * c).sum(-1) / (a.square().sum(-1).clamp_min(1.0e-12))
    residual = c - coefficient[:, None] * a
    return torch.linalg.vector_norm(residual, dim=-1) / torch.linalg.vector_norm(c, dim=-1).clamp_min(1.0e-12)


def validate_waveform(waveform: Iterable[Iterable[int]] = PRE_REGISTERED_WAVEFORM) -> dict[str, Any]:
    """Return all fixed design gates; no metric is softened on failure."""
    table = torch.as_tensor(waveform, dtype=torch.int64)
    if tuple(table.shape) != (PROBE_PERIOD, ACTION_DIM):
        raise ValueError("probe waveform must have fixed shape 50x4")
    entries_ok = bool(torch.isin(table, torch.tensor(PROBE_ENTRIES)).all())
    support: dict[str, Any] = {}
    pulses_ok = True
    # The no-opposite-sign rule applies to the complete 50-step sequence as
    # well as to each publication block (a block boundary is not a physical
    # reset).
    adjacency_ok = not bool(((table[:-1] * table[1:]) == -1).any())
    block_sums = []
    for block in range(2):
        for motor in range(4):
            values = table[block * PROBE_BLOCK:(block + 1) * PROBE_BLOCK, motor]
            block_sums.append(int(values.sum()))
            counts = {str(sign): int((values == sign).sum()) for sign in (-1, 0, 1)}
            lengths = {str(sign): _pulse_lengths(values, sign) for sign in (-1, 1)}
            pulses_ok &= all(length >= 2 for ls in lengths.values() for length in ls)
            adjacency_ok &= not bool(((values[:-1] * values[1:]) == -1).any())
            support[f"block{block + 1}_motor{motor + 1}"] = {"counts": counts, "pulse_lengths": lengths}
    modal = modal_coordinates(table.double())
    covariance = torch.cov((modal - modal.mean(0)).T, correction=0)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    covariance_ratio = float(eigenvalues[0] / eigenvalues[-1].clamp_min(1.0e-12))
    rotational = modal[:, 1:]
    overlap = (rotational.abs() > 0).sum(-1) >= 2
    overlap_rows = torch.nonzero(overlap, as_tuple=False).flatten().tolist()
    rotational_keys = {tuple(int(v) for v in row.tolist()) for row in rotational[overlap]}
    reverse_pairs = sum(tuple(-v for v in key) in rotational_keys for key in rotational_keys) // 2
    design = lag_design(table.double())
    standardized = standardized_rank_condition(design)
    gates = {
        "shape": tuple(table.shape) == (50, 4), "entries": entries_ok,
        "per_block_counts": all(
            min(item["counts"].values()) >= 4 for item in support.values()
        ), "per_block_zero_sum": all(value == 0 for value in block_sums),
        "pulse_length_ge_2": pulses_ok, "no_opposite_adjacency": adjacency_ok,
        "modal_covariance_ratio_ge_0p35": covariance_ratio >= 0.35,
        "multi_rotational_overlap_ge_8": len(overlap_rows) >= 8,
        "reverse_rotational_pair": reverse_pairs >= 1,
        "lag_design_rank12": standardized["rank"] == 12,
    }
    return {
        "shape": list(table.shape), "entries": list(PROBE_ENTRIES), "table": table.tolist(), "support": support,
        "block_sums": [block_sums[:4], block_sums[4:]],
        "modal_covariance": {"eigenvalues": [float(v) for v in eigenvalues],
                             "lambda_min_over_lambda_max": covariance_ratio},
        "multi_rotational_overlap_steps": overlap_rows,
        "reverse_rotational_pairs": int(reverse_pairs), "lag_design": standardized,
        "gates": gates, "gate_passed": bool(all(gates.values())),
    }


def generate_candidate_family(*, seed: int = SEARCH_SEED) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return the bounded, deterministic structural candidate family."""
    if int(seed) != SEARCH_SEED:
        raise ValueError(f"probe v3 search seed is fixed at {SEARCH_SEED}")
    return (
        PRE_REGISTERED_WAVEFORM,
        tuple(tuple(-value for value in row) for row in PRE_REGISTERED_WAVEFORM),
        tuple(reversed(PRE_REGISTERED_WAVEFORM)),
    )


def search_probe_v3(*, seed: int = SEARCH_SEED) -> dict[str, Any] | None:
    """Generate deterministic structural candidates, without claiming freeze.

    The release table is a static candidate.  Q2 train-seed scoring is a
    separate step in :func:`run`; until that step passes, this result is only
    a structural candidate and its SHA must not be called frozen.
    """
    if int(seed) != SEARCH_SEED:
        raise ValueError(f"probe v3 search seed is fixed at {SEARCH_SEED}")
    candidates = generate_candidate_family(seed=seed)
    valid = []
    for table in candidates:
        checks = validate_waveform(table)
        if checks["gate_passed"]:
            key = (-checks["modal_covariance"]["lambda_min_over_lambda_max"],
                   tuple(value for row in table for value in row))
            valid.append((key, table, checks))
    if not valid:
        return None
    _, table, checks = min(valid, key=lambda item: item[0])
    return {
        "waveform": [list(row) for row in table],
        "sha256": waveform_sha256(table), "search_seed": SEARCH_SEED,
        "status": "static_candidate",
        "objective": ["maximize modal covariance ratio", "lexicographically smallest table"],
        "candidate_count": len(candidates), "valid_candidate_count": len(valid),
        "checks": checks,
    }


def clip_residual(requested_normalized: torch.Tensor, q2_action: torch.Tensor,
                  previous_residual: torch.Tensor, amplitude: float = PROBE_AMPLITUDE
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the registered residual annulus elementwise and return d,L,U."""
    if requested_normalized.shape != q2_action.shape or q2_action.shape != previous_residual.shape:
        raise ValueError("requested residual, q2 action, and previous residual must have equal shape")
    if amplitude <= 0:
        raise ValueError("residual amplitude must be positive")
    a = q2_action.new_tensor(float(amplitude))
    lower = torch.maximum(torch.maximum(-a, previous_residual - a), -1.0 - q2_action)
    upper = torch.minimum(torch.minimum(a, previous_residual + a), 1.0 - q2_action)
    if bool((lower > upper).any()):
        raise RuntimeError("residual clip bounds are inconsistent")
    return torch.clamp(a * requested_normalized, min=lower, max=upper), lower, upper


# Descriptive aliases make the small contract convenient to import from tests
# and downstream experiment runners without exposing any private rollout code.
apply_residual_clip = clip_residual


def generate_probe_waveform(*, device: torch.device | str = "cpu",
                            dtype: torch.dtype = torch.int64) -> torch.Tensor:
    return torch.tensor(PRE_REGISTERED_WAVEFORM, device=device, dtype=dtype)


def _clone_state(state: L2FState) -> L2FState:
    return L2FState(**{name: getattr(state, name).detach().clone()
                       for name in state.__dataclass_fields__})


def _safe_stats(position: torch.Tensor, velocity: torch.Tensor, omega: torch.Tensor,
                action: torch.Tensor, zero: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    finite = bool(torch.isfinite(torch.cat((position, velocity, omega, action), dim=-1)).all())
    result: dict[str, Any] = {
        "finite": finite,
        "position_max": float(torch.linalg.vector_norm(position, dim=-1).amax()),
        "velocity_max": float(torch.linalg.vector_norm(velocity, dim=-1).amax()),
        "omega_max": float(torch.linalg.vector_norm(omega, dim=-1).amax()),
    }
    if zero is not None:
        for name in ("position", "velocity", "omega"):
            value = result[f"{name}_max"]
            result[f"{name}_ratio_to_zero"] = value / max(float(zero[f"{name}_max"]), 1.0e-12)
    return result


def _max_numeric_delta(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        return max((_max_numeric_delta(left[key], right[key]) for key in left if key in right), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        return max((_max_numeric_delta(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0


@torch.no_grad()
def _rollout_one(checkpoint: Path, seed: int, scenarios: int, horizon: int,
                 waveform: torch.Tensor, amplitude: float = PROBE_AMPLITUDE) -> dict[str, Any]:
    if horizon < FORMAL_HORIZON:
        raise ValueError("formal probe rollout requires horizon >= 125")
    torch.set_num_threads(1)
    bank = build_dagger_scenario_bank(scenarios, seed=int(seed), dt=0.01, per_cell=scenarios // 16)
    policy, _ = load_q2_policy(checkpoint, device="cpu", dtype=torch.float32)
    sim = L2FSimulator(L2FParams(dt=0.01))

    def arm(use_probe: bool) -> dict[str, Any]:
        state = _clone_state(bank.state)
        batch = state.position.shape[0]
        hidden = policy.initial_hidden(batch, device="cpu", dtype=torch.float32)
        obs_state = initial_observation_state(batch, device="cpu", dtype=torch.float32)
        observer = state.previous_action.clone()  # deployable initialization only
        previous_residual = torch.zeros_like(observer)
        positions: list[torch.Tensor] = []
        velocities: list[torch.Tensor] = []
        omegas: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        requested_rows: list[torch.Tensor] = []
        executed_rows: list[torch.Tensor] = []
        support_deployable = torch.zeros(2, batch, 4, dtype=torch.int64)
        support_physical = torch.zeros(2, batch, 4, dtype=torch.int64)
        support_physical_upper = torch.zeros(2, batch, 4, dtype=torch.int64)
        energy_requested = torch.zeros(batch)
        energy_executed = torch.zeros(batch)
        coriolis_term: list[torch.Tensor] = []
        actuation_coordinates: list[torch.Tensor] = []
        coriolis_c: list[torch.Tensor] = []
        motor_before_rows: list[torch.Tensor] = []
        motor_after_rows: list[torch.Tensor] = []
        command_rows: list[torch.Tensor] = []
        for step in range(horizon):
            observation, observed_position = build_policy_observation(
                state, obs_state, mode="integral25", integral_input_frame="body",
                integral_input_multiplier=1.0,
            )
            q2, hidden = policy(observation, hidden)
            probe_active = step < PROBE_PERIOD
            requested = (waveform[step] if (probe_active and use_probe)
                         else torch.zeros(4)).expand(batch, -1)
            if use_probe:
                residual, _, _ = clip_residual(requested, q2, previous_residual, amplitude)
            else:
                residual = torch.zeros_like(q2)
            action = (q2 + residual).clamp(-1.0, 1.0)
            # Both support definitions are retained: the deployable observer
            # cannot see simulator motor truth, while the physical ceiling is
            # explicitly allowed to use the true motor state.
            if probe_active and use_probe:
                support_deployable[0] += (((action - observer).abs() >= amplitude / 2) &
                                          (action >= observer)).to(torch.int64)
                support_deployable[1] += (((action - observer).abs() >= amplitude / 2) &
                                          (action < observer)).to(torch.int64)
                physical_delta = action - state.motor
                support_physical[0] += (((physical_delta.abs() >= amplitude / 2) &
                                         (action >= state.motor))).to(torch.int64)
                support_physical[1] += (((physical_delta.abs() >= amplitude / 2) &
                                         (action < state.motor))).to(torch.int64)
                # A bounded Q2-guided search cannot exceed these counts:
                # residual d is confined to [-A,A], irrespective of W.
                q_delta = q2 - state.motor
                support_physical_upper[0] += (q_delta >= -amplitude / 2).to(torch.int64)
                support_physical_upper[1] += (q_delta <= amplitude / 2).to(torch.int64)
            requested_rows.append((amplitude * requested).clone())
            executed_rows.append(residual.clone())
            if probe_active:
                energy_requested += (amplitude * requested).square().sum(-1)
                energy_executed += residual.square().sum(-1)
            next_state = sim.step(state, action, grad_decay=1.0)
            omega_mid = 0.5 * (state.omega + next_state.omega)
            inertia = torch.stack((state.inertia_x, state.inertia_y, state.inertia_z), dim=-1)
            gyro = torch.cross(omega_mid, omega_mid * inertia, dim=-1)
            motor = state.motor + (0.01 / torch.where(action >= state.motor,
                                                       state.motor_time_rising[:, None],
                                                       state.motor_time_falling[:, None])).clamp(0, 1) * (action - state.motor)
            thrust = (state.thrust_coeff_c0 + state.thrust_coeff_c1 * motor +
                      state.thrust_coeff_c2 * motor.square()).clamp_min(0)
            torque = torch.stack((state.arm_length * (thrust[:, 1] - thrust[:, 3]),
                                  state.arm_length * (thrust[:, 2] - thrust[:, 0]),
                                  state.rotor_torque_constant * (thrust[:, 0] - thrust[:, 1] + thrust[:, 2] - thrust[:, 3])), dim=-1)
            measured = (next_state.omega - state.omega) / 0.01
            motor_before_rows.append(state.motor.clone())
            motor_after_rows.append(motor.clone())
            command_rows.append(action.clone())
            coriolis_term.append((gyro / inertia).clone())
            motor_modal = modal_coordinates(motor)
            actuation_coordinates.append(motor_modal[:, 1:3].clone())
            coriolis_c.append(torch.stack((-omega_mid[:, 1] * omega_mid[:, 2],
                                           omega_mid[:, 2] * omega_mid[:, 0]), dim=-1).clone())
            positions.append(next_state.position.clone()); velocities.append(next_state.velocity.clone())
            omegas.append(next_state.omega.clone()); actions.append(action.clone())
            observer = observer + (0.01 / OBSERVER_TAU) * (action - observer)
            previous_residual = residual
            obs_state = update_position_integral(obs_state, observed_position, dt=0.01,
                                                 integral_limit=0.5, integral_leak=0.0)
            state = next_state
        pos, vel, omg, act = map(lambda xs: torch.stack(xs), (positions, velocities, omegas, actions))
        tail = slice(max(0, TAIL_START - 1), horizon)
        def endpoint_metrics(value: torch.Tensor, index: int | slice) -> dict[str, float]:
            norms = torch.linalg.vector_norm(value[index], dim=-1)
            return {"mean": float(norms.mean()), "p99": float(norms.quantile(0.99))}
        # Standardized X is intentionally computed per scene on the 50 active
        # probe steps.  Pooling scenes can hide an unidentifiable scenario.
        executed_innovation = torch.stack(executed_rows)
        per_scene_x = [
            standardized_rank_condition(lag_design(executed_innovation[:PROBE_PERIOD, scene]))
            for scene in range(batch)
        ]
        requested_modal = modal_coordinates(
            torch.stack(requested_rows)[:PROBE_PERIOD].reshape(-1, 4)
        ).reshape(PROBE_PERIOD, batch, 4)
        executed_modal = modal_coordinates(
            torch.stack(executed_rows)[:PROBE_PERIOD].reshape(-1, 4)
        ).reshape(PROBE_PERIOD, batch, 4)
        requested_modal_energy = requested_modal.square().sum(0)
        executed_modal_energy = executed_modal.square().sum(0)
        energy_retention_modal = executed_modal_energy / requested_modal_energy.clamp_min(1.0e-12)
        coriolis_values = torch.stack(coriolis_term)
        a_values = torch.stack(actuation_coordinates)[:PROBE_PERIOD]
        c_values = torch.stack(coriolis_c)[:PROBE_PERIOD]
        raw_rms_by_scene = torch.linalg.vector_norm(
            c_values.transpose(0, 1).reshape(batch, -1), dim=-1
        ) / math.sqrt(PROBE_PERIOD * 2.0)
        ratio_by_scene = projected_coriolis_residual_ratio(
            a_values.permute(1, 0, 2), c_values.permute(1, 0, 2)
        )
        coriolis_term_rms = float(c_values.square().mean().sqrt())
        return {
            "stats": _safe_stats(pos, vel, omg, act),
            "h75": {k: endpoint_metrics(v, 74) for k, v in (("position", pos), ("velocity", vel), ("omega", omg))},
            "h125": {k: endpoint_metrics(v, 124) for k, v in (("position", pos), ("velocity", vel), ("omega", omg))},
            "tail_h75_h125": {k: endpoint_metrics(v, tail) for k, v in (("position", pos), ("velocity", vel), ("omega", omg))},
            "rise_support": support_physical[0].tolist(), "fall_support": support_physical[1].tolist(),
            "deployable_rise_support": support_deployable[0].tolist(),
            "deployable_fall_support": support_deployable[1].tolist(),
            "physical_support_upper_bound": support_physical_upper.tolist(),
            "support_contract": {"threshold": amplitude / 2.0, "minimum_per_motor": 3,
                                 "physical_ceiling_uses": "simulator motor truth",
                                 "deployable_observer_uses": "previous_action and fixed causal observer"},
            "requested_energy": energy_requested.tolist(), "executed_energy": energy_executed.tolist(),
            "energy_retention": (energy_executed / energy_requested.clamp_min(1.0e-12)).tolist(),
            "energy_retention_modal": energy_retention_modal.tolist(),
            "energy_retention_contract": {"axes": ["collective", "roll", "pitch", "yaw"],
                                          "minimum": 0.90},
            "standardized_X_per_scenario": per_scene_x,
            "standardized_X": {
                "rank_min": min(item["rank"] for item in per_scene_x),
                "condition_p95": float(torch.tensor([item["condition"] for item in per_scene_x]).quantile(0.95)),
                "condition_max": max(item["condition"] for item in per_scene_x),
                "rank12_all": all(item["rank"] == 12 for item in per_scene_x),
            },
            "coriolis": {"term_rms": coriolis_term_rms,
                         "raw_p5_norm": float(raw_rms_by_scene.quantile(0.05)),
                         "projected_residual_ratio_min": float(ratio_by_scene.min()),
                         "projected_residual_ratio_p5": float(ratio_by_scene.quantile(0.05)),
                         "gate_contract": {"raw_p5_norm_min": 0.01,
                                           "projected_residual_ratio_min": 0.10}},
            "raw_transition": {"command": torch.stack(command_rows).tolist(),
                                "motor_before": torch.stack(motor_before_rows).tolist(),
                                "motor_after": torch.stack(motor_after_rows).tolist(),
                                "physical_tau": {"rising": bank.state.motor_time_rising.tolist(),
                                                  "falling": bank.state.motor_time_falling.tolist()}},
        }

    probe = arm(True)
    zero = arm(False)
    zero_repeat = arm(False)
    zero_parity = _max_numeric_delta(zero, zero_repeat)
    zero["canonical_parity_max_abs"] = zero_parity
    ratios = {
        section: {
            name: {stat: probe[section][name][stat] / max(zero[section][name][stat], 1.0e-12)
                   for stat in ("mean", "p99")}
            for name in probe[section]
        }
        for section in ("h75", "h125", "tail_h75_h125")
    }
    probe["q2_zero_safety"] = {
        "h75": probe["h75"], "h125": probe["h125"],
        "tail_h75_h125": probe["tail_h75_h125"], "zero": zero["stats"],
        "ratios": ratios,
        "criteria": {"ratio": 1.5, "absolute": {"position": 5.0,
                                                   "velocity": 20.0, "omega": 5.0}},
    }
    absolute_limits = {"position": 5.0, "velocity": 20.0, "omega": 5.0}
    paired = {
        f"{section}_{metric}_{stat}": probe[section][metric][stat] <=
        1.05 * zero[section][metric][stat] + 1.0e-7
        for section in ratios for metric in ratios[section] for stat in ("mean", "p99")
    }
    peaks = {
        metric: probe["stats"][f"{metric}_max"] <= max(
            absolute_limits[metric], 1.5 * zero["stats"][f"{metric}_max"]
        ) for metric in absolute_limits
    }
    probe["q2_zero_safety"]["paired_mean_p99"] = paired
    probe["q2_zero_safety"]["peak_bounds"] = peaks
    probe["q2_zero_safety"]["safe"] = bool(all(paired.values()) and all(peaks.values()))
    return {"seed": int(seed), "scenario_count": int(scenarios), "probe": probe, "zero": zero,
            "zero_arm_canonical_parity_max_abs": zero_parity}


def _formal_gate(rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, Any] = {"rows": len(rows), "seed_results": []}
    for row in rows:
        probe, zero = row["probe"], row["zero"]
        safety = {
            "finite": bool(probe["stats"]["finite"] and zero["stats"]["finite"]),
            "zero_arm_canonical_parity_le_1e-7": row["zero_arm_canonical_parity_max_abs"] <= 1.0e-7,
            "paired_mean_p99_max_absolute_or_ratio": bool(probe["q2_zero_safety"]["safe"]),
            "energy_retention_modal_ge_0p90": bool(all(
                value >= 0.90 for scene in probe["energy_retention_modal"] for value in scene
            )),
            "support_each_motor_rise_fall_ge_3": all(
                value >= 3 for branch in (probe["rise_support"], probe["fall_support"])
                for scene in branch for value in scene
            ),
            "physical_support_upper_bound_ge_3": all(
                value >= 3 for branch in probe["physical_support_upper_bound"]
                for scene in branch for value in scene
            ),
            "standardized_X_rank12_condition_gates": bool(
                probe["standardized_X"]["rank12_all"] and
                probe["standardized_X"]["condition_p95"] <= 10.0 and
                probe["standardized_X"]["condition_max"] <= 20.0
            ),
            "coriolis_raw_p5_ge_0p01": probe["coriolis"]["raw_p5_norm"] >= 0.01,
            "coriolis_projected_residual_ratio_ge_0p10": probe["coriolis"]["projected_residual_ratio_min"] >= 0.10,
        }
        safety.update({f"paired_{key}": value
                       for key, value in probe["q2_zero_safety"]["paired_mean_p99"].items()})
        safety.update({f"peak_{key}": value
                       for key, value in probe["q2_zero_safety"]["peak_bounds"].items()})
        checks["seed_results"].append({"seed": row["seed"], "checks": safety, "passed": all(safety.values())})
    return bool(rows) and all(item["passed"] for item in checks["seed_results"]), checks


def _candidate_quality(rows: list[dict[str, Any]], design: dict[str, Any]) -> tuple[Any, ...]:
    """Registered lexicographic ranking; smaller tuples are preferred."""
    passed, checks = _formal_gate(rows)
    supports = [value for row in rows for branch in (row["probe"]["rise_support"], row["probe"]["fall_support"])
                for scene in branch for value in scene]
    rank_all = all(row["probe"]["standardized_X"]["rank12_all"] for row in rows)
    worst_condition = max(row["probe"]["standardized_X"]["condition_max"] for row in rows)
    margins = []
    for row in rows:
        probe, zero = row["probe"], row["zero"]
        for section in ("h75", "h125", "tail_h75_h125"):
            for metric in ("position", "velocity", "omega"):
                for stat in ("mean", "p99"):
                    margins.append(1.05 * zero[section][metric][stat] /
                                   max(probe[section][metric][stat], 1.0e-12))
        for metric, absolute in (("position", 5.0), ("velocity", 20.0), ("omega", 5.0)):
            margins.append(max(absolute, 1.5 * zero["stats"][f"{metric}_max"]) /
                           max(probe["stats"][f"{metric}_max"], 1.0e-12))
    safety_margin = min(margins) if margins else float("-inf")
    return (0 if passed else 1, -min(supports, default=0), 0 if rank_all else 1,
            worst_condition, -design["modal_covariance"]["lambda_min_over_lambda_max"],
            -safety_margin, tuple(value for row in design.get("table", PRE_REGISTERED_WAVEFORM) for value in row))


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate = search_probe_v3(seed=SEARCH_SEED)
    design = validate_waveform()
    result: dict[str, Any] = {
        "diagnostic": "probe-v3-design-and-q2-residual-diagnostic",
        "waveform": candidate["waveform"] if candidate else None,
        "waveform_sha256": candidate["sha256"] if candidate else waveform_sha256(),
        "waveform_sha_status": "static_candidate_not_frozen",
        "formal_frozen_sha256": None,
        "waveform_json": _canonical_waveform(PRE_REGISTERED_WAVEFORM).decode(),
        "waveform_artifact": {"shape": [PROBE_PERIOD, ACTION_DIM],
                              "entries": list(PROBE_ENTRIES),
                              "sha256": waveform_sha256()},
        "search": {"seed": SEARCH_SEED, "objective": candidate["objective"] if candidate else None,
                    "candidate_found": False,
                    "static_candidate_found": candidate is not None,
                    "status": "static_candidate_not_frozen",
                    "q2_guided_train_seed_scoring": "required_before_freeze",
                    "scoring_order": [*TRAIN_SEEDS, VALIDATION_SEED],
                    "blind_seeds_consumed": [],},
        "design": design,
        "seed_split": {"train_search": list(TRAIN_SEEDS), "validation": [VALIDATION_SEED],
                        "blind_excluded": list(BLIND_SEEDS)},
        "amplitude": PROBE_AMPLITUDE,
        "residual_clip": {"formula": "L=max(-A,dprev-A,-1-q); U=min(A,dprev+A,1-q); d=clip(AW,L,U)",
                          "A": PROBE_AMPLITUDE},
        "formal": {"eligible": False, "gate_passed": False, "frozen_candidate": False},
    }
    if candidate is None:
        result["failure"] = "no candidate passed the fixed design gates"
    elif not args.dry_run:
        if args.checkpoint is None or not args.checkpoint.is_file():
            result["failure"] = "frozen Q2 checkpoint is required for formal rollout"
        else:
            if args.scenarios < 16 or args.scenarios % 16:
                raise ValueError("scenarios must be a positive multiple of 16")
            # Q2 scoring is performed for every structurally valid candidate
            # and every train seed before validation is touched.
            family = generate_candidate_family(seed=SEARCH_SEED)
            structural = [(index, table, validate_waveform(table))
                          for index, table in enumerate(family)]
            structural = [item for item in structural if item[2]["gate_passed"]]
            work = [(index, seed, table) for index, table, _ in structural for seed in TRAIN_SEEDS]
            fn = lambda index, seed, table: (index, _rollout_one(
                args.checkpoint, seed, args.scenarios, args.horizon,
                torch.tensor(table, dtype=torch.float32),
            ))
            if Parallel is not None and int(args.n_jobs) > 1:
                rows = Parallel(n_jobs=int(args.n_jobs), backend="loky")(
                    delayed(fn)(*item) for item in work
                )
            else:
                rows = [fn(*item) for item in work]
            by_candidate = {index: [row for row_index, row in rows if row_index == index]
                            for index, _, _ in structural}
            scored = []
            for index, table, candidate_design in structural:
                candidate_rows = by_candidate[index]
                passed, checks = _formal_gate(candidate_rows)
                scored.append({"index": index, "sha256": waveform_sha256(table),
                               "train_gate_passed": passed, "checks": checks,
                               "quality_key": list(_candidate_quality(candidate_rows, candidate_design)),
                               "failure_seeds": [item["seed"] for item in checks["seed_results"] if not item["passed"]]})
            train_passers = [item for item in scored if item["train_gate_passed"]]
            result["formal"] = {"eligible": False, "gate_passed": False,
                                 "frozen_candidate": False, "candidate_scores": scored,
                                 "checkpoint": str(args.checkpoint.resolve()),
                                 "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
            if not train_passers:
                result["failure"] = "STOP: no candidate passed all Q2 train-seed gates"
                result["search"]["status"] = "stop_no_frozen_candidate"
                result["search"]["stop_reason"] = "all deterministic candidates failed train-seed gates"
            else:
                selected_score = min(train_passers, key=lambda item: tuple(item["quality_key"]))
                selected_index = selected_score["index"]
                selected_table = family[selected_index]
                validation_row = _rollout_one(
                    args.checkpoint, VALIDATION_SEED, args.scenarios, args.horizon,
                    torch.tensor(selected_table, dtype=torch.float32),
                )
                validation_passed, validation_checks = _formal_gate([validation_row])
                result["formal"].update({"validation_checks": validation_checks,
                                         "validation_gate_passed": validation_passed,
                                         "selected_train_index": selected_index,
                                         "rows": by_candidate[selected_index] + [validation_row]})
                if validation_passed:
                    selected_sha = waveform_sha256(selected_table)
                    result["search"]["status"] = "frozen_after_q2_scoring"
                    result["search"]["candidate_found"] = True
                    result["formal"]["eligible"] = True
                    result["formal"]["gate_passed"] = True
                    result["formal"]["frozen_candidate"] = True
                    result["formal_frozen_sha256"] = selected_sha
                    result["waveform"] = [list(row) for row in selected_table]
                    result["waveform_sha256"] = selected_sha
                    result["waveform_sha_status"] = "frozen_after_train_and_validation"
                else:
                    result["failure"] = "STOP: selected train candidate failed validation seed 7707"
                    result["search"]["status"] = "stop_no_frozen_candidate"
                    result["search"]["stop_reason"] = "validation gate failed"
    result["design_gate_passed"] = design["gate_passed"]
    result["gate_passed"] = bool(result["formal"].get("frozen_candidate", False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=FORMAL_HORIZON)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    waveform_path = args.output.with_suffix(".waveform.json")
    # The sidecar hash is over the exact bytes of the JSON artifact.
    artifact = report.get("waveform") or PRE_REGISTERED_WAVEFORM
    waveform_path.write_bytes(_canonical_waveform(artifact))
    args.output.with_suffix(".waveform.sha256").write_text(waveform_sha256(artifact) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostic": report["diagnostic"], "gate_passed": report["gate_passed"],
                      "waveform_sha256": report["waveform_sha256"],
                      "failure": report.get("failure")}, sort_keys=True))


if __name__ == "__main__":
    main()
