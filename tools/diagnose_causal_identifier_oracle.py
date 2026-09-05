"""Fail-closed causal identifier oracle.

This is a read-only diagnostic.  It never updates a production policy and it
does not use capability labels to construct features.  The collector records
the transition ``(state_t, u_t) -> state_(t+1)`` and publishes windows at
calls 25/100/125.  The six arms are intentionally diagnostic representations:

The collector has the historical six-arm feature schema for auditability, but
does not fit or release those arms.  All collection is driven by the frozen
raw Q2 checkpoint; blind seeds remain unreachable until a later protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from identification_features import (  # noqa: E402
    bank_modal_features,
    feature_schema_metadata,
    feature_schema_sha256,
    modal,
    normalize_response,
    production_legacy24,
    sol_response,
)
from structured_distillation import build_dagger_scenario_bank, normalize_log_capability  # noqa: E402
from structured_checkpoint import CADENCE_SEMANTICS_VERSION  # noqa: E402
from probe_contract import (  # noqa: E402
    PROBE_AMPLITUDE,
    PROBE_CONTRACT_VERSION,
    PROBE_PERIOD,
    SHARED_TAU_MIN_CALLS,
    SHARED_TAU_MIN_MOTORS,
    SHARED_TAU_MIN_POOLED,
    SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION,
    WAVEFORM,
    WAVEFORM_SHA256,
    canonical_waveform_json,
    waveform_metadata,
    waveform_tensor,
)
from policy_observation import (  # noqa: E402
    LEGACY_BOX_INTEGRAL_CLAMP_MODE,
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)
from structured_policy import motor_observer_tau_grid  # noqa: E402


FORMAL_TRAIN_SEEDS = (13707, 14707, 15707, 16707)
FORMAL_VALIDATION_SEED = 17707
FORMAL_FINAL_SEEDS = (18707, 19707)
FORMAL_BLIND_SEEDS = (10707, 11707)
FORMAL_AMPLITUDES = (0.0,)
PUBLICATION_STEPS = (25, 100, 125)
LAMBDA_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
BOOTSTRAP_SEED = 314159
BANK_DIM = 35 * 8
DEFAULT_Q2_CHECKPOINT = (
    ROOT / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
)
OBSERVER_LEGACY_TAU = 0.06
ZERO_PARITY_TOLERANCE = 1.0e-7
ACTIVE_PUBLICATIONS = (100, 125)
PROBE_WAVEFORM_VERSION = PROBE_CONTRACT_VERSION
DEFAULT_PROBE_V4_REPORT = ROOT / "reports/probe_v5_formal.json"


def probe_v5_eligibility(path: Path = DEFAULT_PROBE_V4_REPORT, *, q2_checkpoint: Path = DEFAULT_Q2_CHECKPOINT) -> dict[str, Any]:
    """Compatibility API name; current production requires the v5 record."""
    from tools.diagnose_probe_v5 import eligibility
    return eligibility(path, q2_checkpoint=q2_checkpoint)


def _tau_grid(version: int) -> tuple[tuple[float, float], ...]:
    """Call the shared versioned grid while remaining compatible with K15."""
    try:
        return tuple(motor_observer_tau_grid(version=version))
    except TypeError:
        pairs = tuple(motor_observer_tau_grid())
        return pairs if version == 1 else pairs


def _clone_state(state: L2FState) -> L2FState:
    return L2FState(**{name: getattr(state, name).detach().clone()
                       for name in state.__dataclass_fields__})


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_probe_v4_eligibility(
    path: Path = DEFAULT_PROBE_V4_REPORT,
    *,
    q2_checkpoint: Path = DEFAULT_Q2_CHECKPOINT,
) -> dict[str, Any]:
    """Read the v4 freeze record with a fail-closed default.

    A missing, malformed, non-v4, or non-frozen report is never interpreted as
    an eligible probe.  In particular, ``formal_frozen_sha256: null`` in the
    checked-in debug report must keep the causal oracle ineligible.
    """
    FORMAL_TRAIN_SEEDS = (3707, 4707, 5707, 6707)
    FORMAL_VALIDATION_SEED = 7707
    result: dict[str, Any] = {
        "eligible": False,
        "report": str(path.resolve()),
        "contract_version": None,
        "frozen_sha256": None,
        "reason": "missing_probe_v4_freeze_record",
    }
    if not path.is_file():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result["reason"] = "invalid_probe_v4_freeze_record"
        return result
    if not isinstance(payload, dict):
        result["reason"] = "invalid_probe_v4_freeze_record"
        return result
    contract = payload.get("contract")
    formal = payload.get("formal")
    result["contract_version"] = contract.get("contract_version") if isinstance(contract, dict) else None
    frozen_sha = payload.get("formal_frozen_sha256")
    if isinstance(formal, dict) and formal.get("frozen_sha256") is not None:
        frozen_sha = formal.get("frozen_sha256")
    result["frozen_sha256"] = frozen_sha
    contract_sha = contract.get("sha256") if isinstance(contract, dict) else None
    design = payload.get("design")
    report_sha = payload.get("waveform_sha256")
    if report_sha is None and isinstance(design, dict):
        report_sha = design.get("waveform_sha256")
    expected_metadata = waveform_metadata()
    if not isinstance(contract, dict) or contract != expected_metadata:
        result["reason"] = "probe_contract_metadata_mismatch"
        return result
    if contract_sha != WAVEFORM_SHA256 or report_sha != WAVEFORM_SHA256:
        result["reason"] = "probe_waveform_sha256_mismatch"
        return result
    if not isinstance(formal, dict) or formal.get("eligible") is not True:
        result["reason"] = "no_frozen_probe_candidate"
        return result
    if formal.get("gate_passed") is not True or payload.get("gate_passed") is not True:
        result["reason"] = "frozen_probe_gate_not_passed"
        return result
    if frozen_sha != WAVEFORM_SHA256:
        result["reason"] = "frozen_probe_sha256_missing_or_mismatch"
        return result
    if payload.get("status") != "formal_frozen":
        result["reason"] = "probe_report_not_formally_frozen"
        return result
    if (formal.get("selection_train_gate_passed") is not True
            or formal.get("independent_validation_gate_passed") is not True):
        result["reason"] = "probe_train_or_validation_gate_missing"
        return result
    seed_split = payload.get("seed_split")
    if (not isinstance(seed_split, dict)
            or seed_split.get("train") != list(FORMAL_TRAIN_SEEDS)
            or seed_split.get("validation") != [FORMAL_VALIDATION_SEED]
            or seed_split.get("blind_consumed") != []
            or formal.get("blind_consumed") != []):
        result["reason"] = "probe_seed_split_or_blind_status_mismatch"
        return result
    protocol = payload.get("protocol")
    if (not isinstance(protocol, dict)
            or protocol.get("scenarios") != 16
            or protocol.get("horizon") != 125
            or protocol.get("formal_scenarios") != 16
            or protocol.get("formal_horizon") != 125):
        result["reason"] = "probe_formal_protocol_mismatch"
        return result
    producer = ROOT / "tools" / "diagnose_probe_v4.py"
    if (not producer.is_file()
            or payload.get("producer_code_sha256") != _hash_file(producer)):
        result["reason"] = "probe_producer_code_sha256_mismatch"
        return result
    scores = payload.get("candidate_scores")
    expected_tables = (
        WAVEFORM,
        tuple(tuple(-value for value in row) for row in WAVEFORM),
        tuple(reversed(WAVEFORM)),
    )
    expected_candidate_sha = {
        hashlib.sha256(canonical_waveform_json(table)).hexdigest()
        for table in expected_tables
    }
    if not isinstance(scores, list) or len(scores) != len(expected_candidate_sha):
        result["reason"] = "probe_candidate_evidence_missing"
        return result
    actual_candidate_sha = {
        item.get("sha256") for item in scores if isinstance(item, dict)
    }
    if actual_candidate_sha != expected_candidate_sha:
        result["reason"] = "probe_candidate_family_mismatch"
        return result
    passers = []
    for item in scores:
        checks = item.get("checks")
        rows = checks.get("seed_results") if isinstance(checks, dict) else None
        if not rows:
            result["reason"] = "probe_candidate_gate_evidence_inconsistent"
            return result
        for row in rows:
            row_checks = row.get("checks") if isinstance(row, dict) else None
            metrics = row.get("safety_metrics") if isinstance(row, dict) else None
            paired = metrics.get("paired") if isinstance(metrics, dict) else None
            peaks = metrics.get("peaks") if isinstance(metrics, dict) else None
            expected_paired = {
                f"paired_{section}_{metric}_{statistic}"
                for section in ("h75", "h125", "tail_h75_h125")
                for metric in ("position", "velocity", "omega")
                for statistic in ("mean", "p99")
            }
            expected_peaks = {"peak_position", "peak_velocity", "peak_omega"}
            if (not isinstance(row_checks, dict)
                    or not isinstance(paired, dict) or set(paired) != expected_paired
                    or not isinstance(peaks, dict) or set(peaks) != expected_peaks):
                result["reason"] = "probe_numeric_safety_evidence_missing"
                return result
            numeric_results = []
            for name, value in tuple(paired.items()) + tuple(peaks.items()):
                if not isinstance(value, dict):
                    result["reason"] = "probe_numeric_safety_evidence_missing"
                    return result
                actual = value.get("actual")
                allowed = value.get("allowed")
                if (not isinstance(actual, (int, float))
                        or not isinstance(allowed, (int, float))
                        or not math.isfinite(float(actual))
                        or not math.isfinite(float(allowed))):
                    result["reason"] = "probe_numeric_safety_evidence_nonfinite"
                    return result
                expected = float(actual) <= float(allowed)
                if value.get("passed") is not expected or row_checks.get(name) is not expected:
                    result["reason"] = "probe_numeric_safety_gate_inconsistent"
                    return result
                numeric_results.append(expected)
            if row_checks.get("paired_mean_p99_max_absolute_or_ratio") is not all(numeric_results):
                result["reason"] = "probe_numeric_safety_gate_inconsistent"
                return result
            shared_tau = row.get("shared_tau")
            if (not isinstance(shared_tau, dict)
                    or row_checks.get("shared_tau_pooled_time_motor_fisher")
                    is not bool(shared_tau.get("gate_passed"))):
                result["reason"] = "probe_shared_tau_gate_evidence_inconsistent"
                return result
        evidence_passed = bool(rows) and all(
            isinstance(row, dict) and row.get("passed") is True for row in rows
        )
        if bool(item.get("train_gate_passed")) != evidence_passed:
            result["reason"] = "probe_candidate_gate_evidence_inconsistent"
            return result
        if evidence_passed:
            passers.append(item)
    if not passers:
        result["reason"] = "probe_no_passing_train_candidate"
        return result
    selected = min(
        passers,
        key=lambda item: (float(item.get("worst_X_condition", float("inf"))), item["sha256"]),
    )
    validation = payload.get("validation")
    validation_checks = validation.get("checks") if isinstance(validation, dict) else None
    validation_rows = (
        validation_checks.get("seed_results")
        if isinstance(validation_checks, dict) else None
    )
    if (selected.get("sha256") != frozen_sha
            or not isinstance(validation, dict)
            or validation.get("index") != selected.get("index")
            or validation.get("gate_passed") is not True
            or not validation_rows
            or not all(isinstance(row, dict) and row.get("passed") is True
                       for row in validation_rows)):
        result["reason"] = "probe_selection_or_validation_evidence_inconsistent"
        return result
    if not q2_checkpoint.is_file():
        result["reason"] = "frozen_q2_checkpoint_missing"
        return result
    expected_q2_sha256 = _hash_file(q2_checkpoint)
    result["source_checkpoint_sha256"] = payload.get("source_checkpoint_sha256")
    result["expected_checkpoint_sha256"] = expected_q2_sha256
    if payload.get("source_checkpoint_sha256") != expected_q2_sha256:
        result["reason"] = "probe_q2_checkpoint_sha256_mismatch"
        return result
    result["eligible"] = True
    result["reason"] = "frozen_v4_probe_candidate"
    return result


probe_v4_eligibility = legacy_probe_v4_eligibility

def _response(state: L2FState, next_state: L2FState, dt: float):
    acceleration = (next_state.velocity - state.velocity) / float(dt)
    gravity = acceleration.new_tensor((0.0, 0.0, 9.80665)).expand_as(acceleration)
    specific = torch.bmm(state.rotation.transpose(1, 2),
                         (acceleration + gravity).unsqueeze(-1)).squeeze(-1)
    angular = (next_state.omega - state.omega) / float(dt)
    force, angular_n, collective = normalize_response(specific, angular)
    return force, angular_n, sol_response(collective, angular_n), specific, angular


def exact_physics_block_fit(
    motor_before: torch.Tensor,
    motor_next: torch.Tensor,
    command: torch.Tensor,
    specific_body: torch.Tensor,
    rotation_before: torch.Tensor,
    omega_before: torch.Tensor,
    omega_after: torch.Tensor,
    external_force: torch.Tensor,
    mass: torch.Tensor,
    dt: float,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """Fit the six causal physical coefficients from one scene/window.

    This is deliberately a read-only physics oracle.  The feature matrix uses
    only the transition observations and the applied command.  Dynamics
    labels are used by the caller only for the reported error.  The force
    channel subtracts the external acceleration in the *pre-transition body
    frame*.  Motor-state regressions pool rows for fitting, while branch
    support is retained per motor so an unexcited rotor cannot be hidden.
    """
    tensors = (motor_before, motor_next, command, specific_body,
               rotation_before, omega_before, omega_after, external_force)
    if any(value.ndim != 2 for value in tensors[:4]) or any(value.shape[-1] != 4 for value in tensors[:3]):
        raise ValueError("motor/command tensors must have shape [time,4]")
    if specific_body.shape[-1] != 3 or rotation_before.shape[-2:] != (3, 3):
        raise ValueError("specific_body and rotation_before have incompatible shapes")
    if omega_before.shape[-1] != 3 or omega_after.shape[-1] != 3:
        raise ValueError("omega tensors must have shape [time,3]")
    if external_force.shape[-1] != 3 or mass.ndim not in (0, 1):
        raise ValueError("external_force/mass have incompatible shapes")
    if not (motor_before.shape == motor_next.shape == command.shape == specific_body.shape[:1] + (4,)):
        raise ValueError("transition tensors must have a common time dimension")
    n = motor_before.shape[0]
    if rotation_before.shape[0] != n or omega_before.shape[0] != n or omega_after.shape[0] != n or external_force.shape[0] != n:
        raise ValueError("transition tensors must have a common time dimension")
    if mass.ndim == 0:
        mass = mass.expand(n)
    if mass.shape != (n,):
        raise ValueError("mass must be scalar or shape [time]")

    output_dtype = motor_before.dtype
    motor_before, motor_next, command, specific_body, rotation_before, omega_before, omega_after, external_force, mass = (
        value.double() for value in (motor_before, motor_next, command, specific_body,
        rotation_before, omega_before, omega_after, external_force, mass))
    ext_body = torch.bmm(rotation_before.transpose(-1, -2),
                         (external_force / mass[:, None]).unsqueeze(-1)).squeeze(-1)
    corrected_specific = specific_body - ext_body
    measured_collective = corrected_specific[:, 2] / float(gravity)
    omega_mid = 0.5 * (omega_before + omega_after)
    angular_accel = (omega_after - omega_before) / float(dt)
    # The simulator clips each rotor's thrust at zero.  A global linear
    # regression in motor command is therefore misspecified when TW>2.
    # Enumerate the scalar TW breakpoints; within each active set the force
    # equation is affine, so its constrained least-squares optimum is exact.
    breaks = (-1.0 / motor_next[motor_next < 0]).clamp(.45, 4.5)
    breaks = torch.unique(torch.cat((breaks, motor_next.new_tensor((.45, 4.5))))).sort().values
    middle = .5 * (breaks[:-1] + breaks[1:])
    active = 1.0 + middle[:, None, None] * motor_next[None] > 0
    slopes = (active * motor_next[None]).mean(-1)
    offsets = active.double().mean(-1)
    candidates = (slopes * (measured_collective[None] - offsets)).sum(-1) / slopes.square().sum(-1).clamp_min(1e-20)
    candidates = candidates.clamp(min=breaks[:-1], max=breaks[1:])
    errors = (offsets + slopes * candidates[:, None] - measured_collective[None]).square().sum(-1)
    surplus = candidates[errors.argmin()]
    rotor_force = (1.0 + surplus * motor_next).clamp_min(0)
    force_active = rotor_force > 0
    tw_response = measured_collective - force_active.double().mean(-1)
    force_slope = (force_active * motor_next).mean(-1)
    full_range = 1.0 + surplus - (1.0 - surplus).clamp_min(0)
    x = torch.stack((force_slope,
        (rotor_force[:, 1] - rotor_force[:, 3]) / full_range,
        (rotor_force[:, 2] - rotor_force[:, 0]) / full_range,
        (rotor_force[:, 0] - rotor_force[:, 1] + rotor_force[:, 2] - rotor_force[:, 3]) / (2 * full_range)), -1)

    # Each block is one row family in a common six-column linear model:
    # [TW-1, local_roll, local_yaw, beta, inv_tau_up, inv_tau_down].
    # The angular coefficients are local command slopes around hover.  They
    # differ from the full motor-range authority labels by the piecewise
    # thrust-curve slope and are converted back below.
    designs, responses, weights = [], [], []
    thrust_design = torch.zeros(n, 6, dtype=motor_before.dtype, device=motor_before.device)
    thrust_design[:, 0] = x[:, 0]
    designs.append(thrust_design)
    responses.append(tw_response)
    weights.append(torch.ones(n, dtype=tw_response.dtype, device=tw_response.device))

    roll_design = torch.zeros(n, 6, dtype=motor_before.dtype, device=motor_before.device)
    roll_design[:, 1] = x[:, 1]
    roll_design[:, 3] = -omega_mid[:, 1] * omega_mid[:, 2]
    designs.append(roll_design)
    responses.append(angular_accel[:, 0])
    weights.append(torch.ones(n, dtype=tw_response.dtype, device=tw_response.device))

    pitch_design = torch.zeros(n, 6, dtype=motor_before.dtype, device=motor_before.device)
    pitch_design[:, 1] = x[:, 2]
    pitch_design[:, 3] = omega_mid[:, 2] * omega_mid[:, 0]
    designs.append(pitch_design)
    responses.append(angular_accel[:, 1])
    weights.append(torch.ones(n, dtype=tw_response.dtype, device=tw_response.device))

    yaw_design = torch.zeros(n, 6, dtype=motor_before.dtype, device=motor_before.device)
    yaw_design[:, 2] = x[:, 3]
    designs.append(yaw_design)
    responses.append(angular_accel[:, 2])
    weights.append(torch.ones(n, dtype=tw_response.dtype, device=tw_response.device))

    motor_delta = command - motor_before
    motor_rate = (motor_next - motor_before) / float(dt)
    branch = command >= motor_before
    tau_design = torch.zeros(n * 4, 6, dtype=motor_before.dtype, device=motor_before.device)
    tau_design[:, 4] = torch.where(branch, motor_delta, torch.zeros_like(motor_delta)).reshape(-1)
    tau_design[:, 5] = torch.where(~branch, motor_delta, torch.zeros_like(motor_delta)).reshape(-1)
    tau_response = motor_rate.reshape(-1)
    tau_valid = tau_design[:, 4].abs().add(tau_design[:, 5].abs()) > 1.0e-8
    designs.append(tau_design[tau_valid])
    responses.append(tau_response[tau_valid])
    weights.append(torch.ones(int(tau_valid.sum()), dtype=tw_response.dtype, device=tw_response.device))

    design = torch.cat(designs, dim=0).double()
    response = torch.cat(responses, dim=0).double()
    weight = torch.cat(weights, dim=0).double().sqrt()
    weighted_design = design * weight[:, None]
    weighted_response = response * weight
    column_norm = torch.linalg.vector_norm(weighted_design, dim=0)
    normalized_design = weighted_design / column_norm.clamp_min(1.0e-12)
    rank = int(torch.linalg.matrix_rank(normalized_design).item())
    condition = float(torch.linalg.cond(normalized_design).item()) if rank == 6 else float("inf")
    # Solve in the same column-normalized coordinates used by the rank and
    # conditioning diagnostic.  The safe probe deliberately creates tiny
    # angular/coupling columns; solving the raw matrix lets lstsq's default
    # rank cutoff discard physically valid columns solely because their units
    # differ by orders of magnitude.
    normalized_coefficients = torch.linalg.lstsq(
        normalized_design, weighted_response
    ).solution
    coefficients = normalized_coefficients / column_norm.clamp_min(1.0e-12)
    inv_tau = coefficients[4:6]
    tw_estimate = coefficients[0] + 1.0
    estimate = torch.stack((
        tw_estimate,
        coefficients[1],
        coefficients[2] / coefficients[1].clamp_min(1.0e-8),
        coefficients[3] + 1.0,
        1.0 / inv_tau[0].clamp_min(1.0e-8),
        1.0 / inv_tau[1].clamp_min(1.0e-8),
    ))
    prediction = design @ coefficients
    # Per-motor counts remain useful diagnostics, but the simulator has one
    # rise tau and one fall tau shared by all rotors.  Eligibility therefore
    # uses the pooled/time/motor/Fisher contract below, not four fictitious
    # independent tau parameters.
    command_delta = command - motor_before
    valid_delta = command_delta.abs() > 1.0e-8
    rise_support = (valid_delta & (command_delta >= 0.0)).sum(dim=0)
    fall_support = (valid_delta & (command_delta < 0.0)).sum(dim=0)
    shared_support = shared_tau_support(command, motor_before)
    return {"estimate": estimate.to(output_dtype), "coefficients": coefficients.to(output_dtype),
            "rank": rank, "condition": condition, "design": normalized_design.to(motor_before.dtype),
            "fit_rms": float((prediction - response).square().mean().sqrt()),
            "rise_support": int(rise_support.sum()), "fall_support": int(fall_support.sum()),
            "rise_support_per_motor": rise_support.cpu(),
            "fall_support_per_motor": fall_support.cpu(),
            "shared_tau_support": shared_support}


def _advance_bank(estimate: torch.Tensor, command: torch.Tensor,
                  pairs: tuple[tuple[float, float], ...], dt: float) -> torch.Tensor:
    rise = estimate.new_tensor([p[0] for p in pairs]).view(1, -1, 1)
    fall = estimate.new_tensor([p[1] for p in pairs]).view(1, -1, 1)
    command = command[:, None, :].expand_as(estimate)
    tau = torch.where(command >= estimate, rise, fall).clamp_min(1.0e-4)
    return estimate + (float(dt) / tau).clamp(max=1.0) * (command.clamp(-1, 1) - estimate)


def _fixed_bank_dim(value: torch.Tensor) -> torch.Tensor:
    """Pad K15 smoke banks to the registered K35 input without new math."""
    flat = value.reshape(value.shape[0], -1)
    if flat.shape[-1] > BANK_DIM:
        raise ValueError("candidate bank exceeds registered K35 dimension")
    return torch.nn.functional.pad(flat, (0, BANK_DIM - flat.shape[-1]))


def _validate_probe_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Return the only waveform permitted by the registered v4 contract."""
    waveform = torch.as_tensor(waveform, dtype=torch.float32).cpu()
    if waveform.shape != (PROBE_PERIOD, 4) or not bool(torch.isfinite(waveform).all()):
        raise ValueError("probe waveform must be finite with shape [50,4]")
    if not bool((waveform == waveform.round()).all()):
        raise ValueError("v4 probe waveform entries must be integer contract values")
    actual_sha = _probe_waveform_sha256(waveform)
    if actual_sha != WAVEFORM_SHA256:
        raise ValueError(
            "probe waveform does not match the unique probe_contract.py v4 artifact "
            f"(expected {WAVEFORM_SHA256}, got {actual_sha})"
        )
    return waveform


def load_probe_waveform(path: Path | None = None) -> torch.Tensor:
    """Load the registered v4 waveform; reject all fallback/alternate tables."""
    if path is None:
        return waveform_tensor(dtype=torch.float32).clone()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in (".pt", ".pth"):
        value = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError("probe waveform must be JSON or torch .pt/.pth")
    return _validate_probe_waveform(torch.as_tensor(value, dtype=torch.float32))


def _probe_waveform_sha256(waveform: torch.Tensor) -> str:
    canonical = [[int(value) for value in row]
                 for row in waveform.detach().cpu().tolist()]
    return hashlib.sha256(canonical_waveform_json(canonical)).hexdigest()


def residual_probe_step(
    previous: torch.Tensor,
    call_index: int,
    amplitude: float,
    q_action: torch.Tensor | None = None,
    waveform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the exact Sol residual annulus clip for one zero-based call.

    ``requested=A*W_t`` for calls ``u0..u49`` and is zero from call 50 onward;
    the returned residual is clipped by amplitude, slew, and action headroom.
    """
    if previous.ndim != 2 or previous.shape[-1] != 4:
        raise ValueError("previous residual must have shape [batch,4]")
    if amplitude < 0.0 or not math.isfinite(float(amplitude)):
        raise ValueError("probe amplitude must be finite and non-negative")
    if call_index < 0:
        raise ValueError("call_index is zero-based")
    if q_action is None:
        q_action = torch.zeros_like(previous)
    if q_action.shape != previous.shape:
        raise ValueError("q_action and previous residual must have equal shape")
    if waveform is None:
        waveform = load_probe_waveform()
    waveform = torch.as_tensor(waveform, device=previous.device, dtype=previous.dtype)
    if waveform.shape != (PROBE_PERIOD, 4):
        raise ValueError("probe waveform must have shape [50,4]")
    requested = waveform[call_index] if call_index < PROBE_PERIOD else torch.zeros(4, device=previous.device, dtype=previous.dtype)
    requested = float(amplitude) * requested.view(1, 4).expand_as(previous)
    if amplitude == 0.0:
        return torch.zeros_like(previous)
    a = previous.new_tensor(float(amplitude))
    lower = torch.maximum(torch.maximum(-a, previous - a), -1.0 - q_action)
    upper = torch.minimum(torch.minimum(a, previous + a), 1.0 - q_action)
    if bool((lower > upper).any()):
        raise RuntimeError("residual clip bounds are inconsistent")
    return torch.clamp(requested, min=lower, max=upper)


def _q2_settings(policy_args: dict[str, Any], policy: torch.nn.Module) -> dict[str, Any]:
    """Read every observation/integral knob from the frozen Q2 checkpoint."""
    mode = str(policy_args.get("observation_mode", "integral25"))
    if int(getattr(policy, "observation_dim", 25)) != {"legacy40": 40, "compact22": 22,
                                                        "integral25": 25}.get(mode, -1):
        raise ValueError(f"Q2 checkpoint observation mode/dimension mismatch: {mode!r}")
    return {
        "mode": mode,
        "integral_input_frame": str(policy_args.get("integral_input_frame", "world")),
        "integral_input_multiplier": float(policy_args.get("integral_input_multiplier", 1.0)),
        "noise_max": float(policy_args.get("observation_noise_max", 0.0)),
        "integral_limit": float(policy_args.get("integral_limit", 0.5)),
        "integral_leak": float(policy_args.get("integral_leak", 0.0)),
        "integral_clamp_mode": str(policy_args.get(
            "integral_clamp_mode", LEGACY_BOX_INTEGRAL_CLAMP_MODE)),
        "dt": float(policy_args.get("dt", 0.01)),
    }


def _state_error(left: L2FState, right: L2FState) -> float:
    fields = ("position", "velocity", "rotation", "omega", "motor", "previous_action")
    return max(float((getattr(left, name) - getattr(right, name)).abs().max()) for name in fields)


def _publication_windows(values: torch.Tensor) -> torch.Tensor:
    """Align transitions as [publication, scenario, 25 calls, channels]."""
    if values.ndim != 3:
        raise ValueError("transition values must have shape [time,scenario,channels]")
    return torch.stack([values[step - 25:step] for step in PUBLICATION_STEPS], dim=0).transpose(1, 2)


def initialize_observer_bank(
    previous_action: torch.Tensor,
    tau_grid: tuple[tuple[float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize legacy/K15/K35 observers from deployable action history."""
    if previous_action.ndim != 2 or previous_action.shape[-1] != 4:
        raise ValueError("previous_action must have shape [batch,4]")
    previous_action = previous_action.clamp(-1.0, 1.0)
    bank = previous_action[:, None, :].expand(-1, len(tau_grid), -1).clone()
    return previous_action.clone(), bank.clone(), bank.clone()


def physics_gate_failed_checks(report: dict[str, Any]) -> list[str]:
    """Return exact failed checks for one publication's physics gate."""
    failed: list[str] = []
    if not report.get("finite", False):
        failed.append("finite")
    if report.get("rank_min", 0) != 6:
        failed.append("rank6_per_scene")
    if report.get("condition_median", float("inf")) > 30.0:
        failed.append("condition_median<=30")
    if report.get("condition_p95", float("inf")) > 100.0:
        failed.append("condition_p95<=100")
    if report.get("condition_max", float("inf")) > 300.0:
        failed.append("condition_max<=300")
    z_rms = report.get("normalized_z_rms_per_dim", ())
    if len(z_rms) != 6 or any(
        not math.isfinite(float(value)) or float(value) > 0.02
        for value in z_rms
    ):
        failed.append("normalized_z_rms_per_dim<=0.02")
    abs_p95 = report.get("absolute_error_p95_per_dim", ())
    if len(abs_p95) != 6 or any(
        not math.isfinite(float(value)) or float(value) > 0.05
        for value in abs_p95
    ):
        failed.append("absolute_error_p95_per_dim<=0.05")
    # v4 estimates one rise and one fall tau shared across rotors.  Per-motor
    # support remains in reports for diagnosis, but is not an eligibility
    # condition (it incorrectly assumes four independent tau parameters).
    if not report.get("shared_tau_support_passed", False):
        failed.append("shared_tau_pooled_time_motor_fisher")
    return failed


def coverage_gate_failed_checks(summary: dict[str, Any], *, strict_motor_rms: bool) -> list[str]:
    """Return exact failed checks for one K15/K35 coverage publication."""
    failed: list[str] = []
    if not summary.get("finite", False):
        failed.append("finite_best_to_legacy_ratio")
    if summary.get("median_best_to_legacy", float("inf")) > 0.50:
        failed.append("median_best_to_legacy<=0.50")
    if summary.get("p95_best_to_legacy", float("inf")) > 0.80:
        failed.append("p95_best_to_legacy<=0.80")
    if strict_motor_rms and summary.get("motor_rms_p95", float("inf")) > 0.001:
        failed.append("motor_rms_p95<=0.001")
    if strict_motor_rms and summary.get("motor_rms_max", float("inf")) > 0.005:
        failed.append("motor_rms_max<=0.005")
    if strict_motor_rms and not summary.get("authority_cells_finite", False):
        failed.append("authority_cells_finite")
    return failed


def _collect_one(checkpoint: Path, seed: int, amplitude: float, scenarios: int,
                 horizon: int, waveform: torch.Tensor | None = None) -> dict[str, Any]:
    torch.set_num_threads(1)
    if horizon < max(PUBLICATION_STEPS):
        raise ValueError("horizon must include calls 25/100/125")
    # Formal collection always loads the untouched Q2 artifact through the
    # shared loader.  In particular, this path never reconstructs a structured
    # policy from a migration/config payload.
    policy, policy_args = load_q2_policy(checkpoint, device="cpu", dtype=torch.float32)
    settings = _q2_settings(policy_args, policy)
    waveform = (load_probe_waveform() if waveform is None
                else _validate_probe_waveform(waveform))
    bank = build_dagger_scenario_bank(scenarios, seed=int(seed), dt=settings["dt"],
                                      per_cell=scenarios // 16)
    simulator = L2FSimulator(L2FParams(dt=settings["dt"]))
    physical = _clone_state(bank.state)
    canonical_physical = _clone_state(physical)
    observation_state = initial_observation_state(
        scenarios, device=physical.position.device, dtype=physical.position.dtype)
    canonical_observation_state = observation_state.clone()
    hidden = policy.initial_hidden(scenarios, device="cpu", dtype=physical.position.dtype)
    canonical_hidden = hidden.clone()
    # All three observer paths start from the deployable previous action.  The
    # simulator's motor field is intentionally never used for initialization.
    pairs15, pairs35 = _tau_grid(1), _tau_grid(2)
    legacy, candidate15, candidate = initialize_observer_bank(physical.previous_action, pairs35)
    # The helper above uses one bank shape for the registered K35 path.  K15 is
    # initialized independently with the same deployable previous action.
    candidate15 = physical.previous_action[:, None, :].expand(-1, len(pairs15), -1).clone()
    observer_initial_legacy = legacy.clone()
    observer_initial_k15 = candidate15.clone()
    observer_initial_k35 = candidate.clone()
    residual_previous = torch.zeros_like(physical.previous_action)
    from probe_contract_v5 import ProbeState, apply_probe, CONTRACT_SHA256
    safe_probe = ProbeState.initial(physical.previous_action)
    metric_rows = {name: [] for name in ("position", "velocity", "omega")}
    excitation_history = torch.zeros(scenarios, 13, 4, dtype=physical.position.dtype)
    sequence: dict[str, list[torch.Tensor]] = {name: [] for name in ("L", "S", "U", "T")}
    true_motor_rows, initial_motor_rows, legacy_rows, candidate_rows, candidate15_rows = [], [], [], [], []
    legacy_sequence: list[torch.Tensor] = []
    failure_rows, residual_rows, requested_rows, q_rows, excitation_rows, previous_action_rows = [], [], [], [], [], []
    raw_specific, raw_angular, response_rows, command_rows = [], [], [], []
    rotation_before_rows, omega_before_rows, omega_after_rows = [], [], []
    external_force_rows, mass_rows, external_accel_body_rows = [], [], []
    canonical_actions, canonical_states, canonical_integrals = [], [], []
    parity_action, parity_state, parity_integral, parity_hidden = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for step in range(horizon):
            obs, observed_position = build_policy_observation(
                physical, observation_state, mode=settings["mode"],
                noise_max=settings["noise_max"],
                integral_input_frame=settings["integral_input_frame"],
                integral_input_multiplier=settings["integral_input_multiplier"],
            )
            q, next_hidden = policy(obs, hidden)
            canonical_obs, canonical_position = build_policy_observation(
                canonical_physical, canonical_observation_state, mode=settings["mode"],
                noise_max=settings["noise_max"],
                integral_input_frame=settings["integral_input_frame"],
                integral_input_multiplier=settings["integral_input_multiplier"],
            )
            canonical_q, canonical_next_hidden = policy(canonical_obs, canonical_hidden)
            action, safe_probe, requested_scalar = apply_probe(q, safe_probe, step,
                position=physical.position, velocity=physical.velocity,
                omega=physical.omega, body_z=physical.rotation[:, :, 2], amplitude=float(amplitude))
            residual = action - q
            requested_rows.append(requested_scalar.expand_as(q).clone())
            # Update the exact Q2 integral with the observed position, then
            # advance physics.  The simulator stores the actual action in
            # previous_action, which is what Q2 sees on the next call.
            next_observation_state = update_position_integral(
                observation_state, observed_position, dt=settings["dt"],
                integral_limit=settings["integral_limit"], integral_leak=settings["integral_leak"],
                integral_clamp_mode=settings["integral_clamp_mode"])
            next_canonical_observation_state = update_position_integral(
                canonical_observation_state, canonical_position, dt=settings["dt"],
                integral_limit=settings["integral_limit"], integral_leak=settings["integral_leak"],
                integral_clamp_mode=settings["integral_clamp_mode"])
            next_physical = simulator.step(physical, action, grad_decay=1.0)
            next_canonical_physical = simulator.step(canonical_physical, canonical_q, grad_decay=1.0)
            force, angular, y, specific, angular_raw = _response(
                physical, next_physical, simulator.params.dt)
            legacy_next = _advance_bank(legacy[:, None, :], action, ((OBSERVER_LEGACY_TAU,
                                                                       OBSERVER_LEGACY_TAU),), simulator.params.dt)[:, 0]
            candidate15_next = _advance_bank(candidate15, action, pairs15, simulator.params.dt)
            candidate_next = _advance_bank(candidate, action, pairs35, simulator.params.dt)
            # Production legacy24 uses the deployable observer innovation as
            # excitation; physical motor truth and previous action are not
            # interchangeable here.
            excitation = ((action - legacy) / 0.10).clamp(-5.0, 5.0)
            excitation_history = torch.cat((excitation[:, None], excitation_history[:, :-1]), dim=1)
            legacy_row = production_legacy24(
                excitation_history, force, angular, legacy_next - legacy,
                torch.ones(scenarios, dtype=action.dtype))
            response_y = y
            bank_row = bank_modal_features(candidate_next, response_y)
            true_row = bank_modal_features(next_physical.motor[:, None, :], response_y)
            bank_flat = _fixed_bank_dim(bank_row)
            sequence["L"].append(torch.zeros_like(bank_flat))
            sequence["S"].append(bank_flat)
            sequence["U"].append(_fixed_bank_dim(torch.cat((bank_row[..., :4].abs(), bank_row[..., 4:]), dim=-1)))
            sequence["T"].append(_fixed_bank_dim(true_row[:, 0]))
            legacy_sequence.append(legacy_row)
            true_motor_rows.append(next_physical.motor)
            initial_motor_rows.append(physical.motor)
            legacy_rows.append(legacy_next)
            candidate_rows.append(candidate_next)
            candidate15_rows.append(candidate15_next)
            failure_rows.append(safe_probe.aborted.clone())
            for name in metric_rows:
                metric_rows[name].append(getattr(next_physical, name).norm(dim=-1))
            residual_rows.append(residual)
            q_rows.append(q)
            excitation_rows.append(excitation_history.clone())
            previous_action_rows.append(physical.previous_action.clone())
            raw_specific.append(specific)
            raw_angular.append(angular_raw)
            response_rows.append(response_y)
            command_rows.append(action)
            rotation_before_rows.append(physical.rotation)
            omega_before_rows.append(physical.omega)
            omega_after_rows.append(next_physical.omega)
            external_force_rows.append(physical.external_force)
            mass_rows.append(physical.mass)
            external_accel_body_rows.append(torch.bmm(
                physical.rotation.transpose(1, 2),
                (physical.external_force / physical.mass[:, None]).unsqueeze(-1)).squeeze(-1))
            if float(amplitude) == 0.0:
                parity_action = max(parity_action, float((action - canonical_q).abs().max()))
                parity_state = max(parity_state, _state_error(next_physical, next_canonical_physical))
                parity_integral = max(parity_integral, float((next_observation_state.integral_position - next_canonical_observation_state.integral_position).abs().max()))
                parity_hidden = max(parity_hidden, float((next_hidden - canonical_next_hidden).abs().max()))
            canonical_actions.append(canonical_q)
            canonical_states.append(next_canonical_physical)
            canonical_integrals.append(next_canonical_observation_state.integral_position.clone())
            residual_previous = residual
            legacy, candidate15, candidate = legacy_next, candidate15_next, candidate_next
            physical, observation_state, hidden = next_physical, next_observation_state, next_hidden
            canonical_physical, canonical_observation_state, canonical_hidden = next_canonical_physical, next_canonical_observation_state, canonical_next_hidden
    rows = {name: _publication_windows(torch.stack(value)) for name, value in sequence.items()}
    legacy_values = torch.stack(legacy_sequence)
    legacy_windows = _publication_windows(legacy_values)
    true_motor = torch.stack(true_motor_rows)
    initial_motor = torch.stack(initial_motor_rows)
    legacy_motor = torch.stack(legacy_rows)
    candidate_motor = torch.stack(candidate_rows)
    target = normalize_log_capability(torch.stack((bank.state.thrust_to_weight,
        bank.state.alpha_roll_max, bank.state.eta_yaw, bank.state.jz_over_jxy,
        bank.state.motor_time_rising, bank.state.motor_time_falling), dim=-1))
    windows: dict[str, torch.Tensor] = rows
    s = windows["S"]
    windows["M"] = s.mean(dim=2, keepdim=True).expand_as(s)
    windows["P"] = torch.zeros_like(windows["T"])
    return {
        "seed": int(seed), "amplitude": float(amplitude), "strata": list(bank.stratum),
        "metrics": {name: torch.stack(values).cpu() for name, values in metric_rows.items()},
        "cells": (bank.tw_bin * 4 + bank.log_alpha_bin).cpu(),
        "features": {k: v.cpu() for k, v in windows.items()}, "legacy_features": legacy_windows.cpu(),
        "target": target.cpu(), "p_available": False,
        "q_actions": torch.stack(q_rows).cpu(), "commands": torch.stack(command_rows).cpu(),
        "residual_probe": torch.stack(residual_rows).cpu(),
        "requested_probe": torch.stack(requested_rows).cpu(),
        "excitation_history": torch.stack(excitation_rows).cpu(),
        "previous_action_seen": torch.stack(previous_action_rows).cpu(),
        "canonical_actions": torch.stack(canonical_actions).cpu(),
        "canonical_states": [{k: getattr(v, k).cpu() for k in ("position", "velocity", "rotation", "omega", "motor", "previous_action")} for v in canonical_states],
        "canonical_integral": torch.stack(canonical_integrals).cpu(),
        "zero_parity": {"max_action_error": parity_action, "max_state_error": parity_state,
                        "max_integral_error": parity_integral, "max_hidden_error": parity_hidden,
                        "gate_passed": bool(float(amplitude) == 0.0 and max(parity_action, parity_state, parity_integral, parity_hidden) <= ZERO_PARITY_TOLERANCE)},
        "true_motor": true_motor.cpu(), "initial_motor": initial_motor.cpu(), "legacy_motor": legacy_motor.cpu(),
        "candidate_motor": candidate_motor.cpu(), "candidate15_motor": torch.stack(candidate15_rows).cpu(),
        "identification_failed": torch.stack(failure_rows).cpu(), "tau_pairs": pairs35,
        "raw_specific": torch.stack(raw_specific).cpu(), "raw_angular": torch.stack(raw_angular).cpu(),
        "response_features": torch.stack(response_rows).cpu(),
        "response_windows": _publication_windows(torch.stack(response_rows)).cpu(),
        "rotation_before": torch.stack(rotation_before_rows).cpu(), "omega_before": torch.stack(omega_before_rows).cpu(),
        "omega_after": torch.stack(omega_after_rows).cpu(), "external_force": torch.stack(external_force_rows).cpu(),
        "mass": torch.stack(mass_rows).cpu(), "external_accel_body": torch.stack(external_accel_body_rows).cpu(),
        "physical_tau": torch.stack((bank.state.motor_time_rising, bank.state.motor_time_falling), dim=-1).cpu(),
        "observer_initial": {
            "legacy": observer_initial_legacy.cpu(),
            "K15": observer_initial_k15.cpu(),
            "K35": observer_initial_k35.cpu(),
        },
        "q2_observation": settings,
        "probe_waveform": torch.zeros(PROBE_PERIOD, 4),
        "probe_waveform_sha256": CONTRACT_SHA256,
    }


def _ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    xz, ym = (x - mean) / scale, y.mean(0)
    eye = torch.eye(x.shape[-1], dtype=torch.float64)
    coef = torch.linalg.solve(xz.double().T @ xz.double() + ridge * eye,
                              xz.double().T @ (y - ym).double())
    return mean, scale, coef


def _predict(model, x: torch.Tensor) -> torch.Tensor:
    mean, scale, coef, ym = model
    return ((x - mean) / scale).double() @ coef + ym


def _fit_sequence(train: list[dict[str, Any]], val: dict[str, Any], arm: str):
    # Same architecture for every arm: Linear(bankdim,24), GRUCell(24,64,
    # leak=.08), Linear(64,6).  The ridge oracle is a cheap upper-bound probe;
    # the recurrent fit is used by ``--stage train``.
    x = torch.cat([r["features"][arm] for r in train], dim=1)
    y = torch.cat([r["target"] for r in train], dim=0).unsqueeze(0).expand(3, -1, -1).reshape(-1, 6)
    xv = val["features"][arm]
    yv = val["target"].unsqueeze(0).expand(3, -1, -1).reshape(-1, 6)
    # Sequence mean is only for the transparent linear ceiling baseline.
    xm, xvmean = x.mean(2), xv.mean(2)
    fitted = []
    for lam in LAMBDA_GRID:
        mean, scale, coef = _ridge(xm.reshape(-1, xm.shape[-1]), y, lam)
        pred = _predict((mean, scale, coef, y.mean(0).double()), xvmean.reshape(-1, xvmean.shape[-1])).float()
        fitted.append((float(lam), float((pred - yv).square().mean().sqrt()), (mean, scale, coef, y.mean(0).double())))
    selected = min(fitted, key=lambda q: q[1])
    return selected[2], {"selected_lambda": selected[0], "validation_rms": selected[1],
                        "input_dim": int(x.shape[-1]), "architecture": "Linear->GRUCell(leak=.08)->Linear"}


class _SequenceIdentifier(nn.Module):
    """The registered six-arm recurrent probe, kept local to this diagnostic."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, 24, bias=False)
        self.cell = nn.GRUCell(24, 64)
        self.head = nn.Linear(64, 6)

    def forward(self, sequence: torch.Tensor, legacy: torch.Tensor) -> torch.Tensor:
        hidden = sequence.new_zeros(sequence.shape[0], 64)
        for index in range(sequence.shape[1]):
            context = self.input(sequence[:, index]) + legacy[:, index]
            candidate = self.cell(torch.tanh(context), hidden)
            hidden = torch.tanh(0.92 * hidden + 0.08 * candidate)
        return self.head(hidden)


def _train_sequence(train: list[dict[str, Any]], val: dict[str, Any], arm: str,
                    max_updates: int) -> dict[str, Any]:
    """Short, bounded recurrent fit; formal defaults are never swept."""
    x = torch.cat([r["features"][arm] for r in train], dim=1).reshape(-1, 25, train[0]["features"][arm].shape[-1])
    y = torch.cat([r["target"] for r in train], dim=0).unsqueeze(0).expand(3, -1, -1).reshape(-1, 6)
    xv = val["features"][arm].reshape(-1, 25, val["features"][arm].shape[-1])
    l = torch.cat([r["legacy_features"] for r in train], dim=1).reshape(-1, 25, 24)
    lv = val["legacy_features"].reshape(-1, 25, 24)
    yv = val["target"].unsqueeze(0).expand(3, -1, -1).reshape(-1, 6)
    model = _SequenceIdentifier(x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_update, stale, grad_norm = float("inf"), 0, 0, 0.0
    generator = torch.Generator().manual_seed(101)
    for update in range(max(1, int(max_updates))):
        indices = torch.randint(x.shape[0], (min(64, x.shape[0]),), generator=generator)
        prediction = model(x.index_select(0, indices), l.index_select(0, indices))
        loss = (prediction - y.index_select(0, indices)).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        with torch.no_grad():
            val_rms = float((model(xv, lv) - yv).square().mean().sqrt())
        if math.isfinite(val_rms) and val_rms < best:
            best, best_update, stale = val_rms, update + 1, 0
        else:
            stale += 1
            if stale >= 10:
                break
    return {"input_dim": int(x.shape[-1]), "architecture": "Linear->GRUCell(24,64,leak=.08)->Linear",
            "updates": int(best_update if best_update else update + 1), "best_validation_rms": best,
            "final_grad_norm": grad_norm, "nan_inf": not math.isfinite(best)}


def shared_tau_support(
    command: torch.Tensor,
    motor_before: torch.Tensor,
    *,
    amplitude: float = PROBE_AMPLITUDE,
) -> dict[str, Any]:
    """Measure pooled shared rise/fall support for one scene.

    Tau is one parameter per branch, shared by all four motors.  Per-motor
    counts are retained as diagnostics only and never determine this gate.
    The information statistic is deliberately weighted by excitation size so
    twelve threshold-touching rows cannot pass by count alone.
    """
    if command.shape != motor_before.shape or command.ndim != 2 or command.shape[-1] != 4:
        raise ValueError("command and motor_before must have shape [time,4]")
    if not math.isfinite(float(amplitude)) or amplitude <= 0.0:
        raise ValueError("amplitude must be finite and positive")
    delta = command.double() - motor_before.double()
    threshold = float(amplitude) / 2.0
    result: dict[str, Any] = {
        "threshold": threshold,
        "min_pooled": SHARED_TAU_MIN_POOLED,
        "min_distinct_calls": SHARED_TAU_MIN_CALLS,
        "min_motors": SHARED_TAU_MIN_MOTORS,
        "min_weighted_fisher_information": SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION,
        "branches": {},
    }
    for name, branch in (("rise", delta >= 0.0), ("fall", delta < 0.0)):
        valid = branch & (delta.abs() >= threshold)
        counts = valid.sum(0)
        pooled = int(counts.sum())
        calls = int(valid.any(-1).sum())
        motors = int((counts > 0).sum())
        excitation_sq = float((delta.square() * valid).sum())
        weighted_information = float((delta.square() * delta.abs() * valid).sum())
        gate = (
            pooled >= SHARED_TAU_MIN_POOLED
            and calls >= SHARED_TAU_MIN_CALLS
            and motors >= SHARED_TAU_MIN_MOTORS
            and weighted_information >= SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION
        )
        result["branches"][name] = {
            "pooled_valid": pooled,
            "distinct_time_calls": calls,
            "motors_contributing": motors,
            "per_motor_counts": counts.cpu(),
            "excitation_sq": excitation_sq,
            "weighted_fisher_information": weighted_information,
            "gate_passed": bool(gate),
        }
    result["gate_passed"] = bool(all(value["gate_passed"] for value in result["branches"].values()))
    return result


def fit_motor_tau_wls(command: torch.Tensor, motor: torch.Tensor, dt: float) -> torch.Tensor:
    """Analytic tau ceiling from the known first-order motor equation."""
    if command.shape != motor.shape or command.shape[-1] != 4 or command.shape[0] < 2:
        raise ValueError("command and motor must have shape [time,4]")
    m, nxt, u = motor[:-1].double(), motor[1:].double(), command[:-1].double()
    delta = nxt - m
    numerator = float(dt) * (u - m)
    valid = delta.abs() > 1e-8
    estimates = numerator / delta.clamp_min(1e-8).where(delta >= 0, delta.clamp_max(-1e-8))
    weights = delta.abs() * valid
    result = (estimates.clamp(1e-4, 10.0) * weights).sum(0) / weights.sum(0).clamp_min(1e-8)
    return result.to(dtype=command.dtype)


def fit_motor_tau_wls_split(command: torch.Tensor, motor: torch.Tensor,
                            dt: float) -> torch.Tensor:
    """Return separate rising/falling WLS estimates, shape ``[2,4]``."""
    if command.shape != motor.shape or command.ndim != 2:
        raise ValueError("command and motor must have shape [time,4]")
    m, nxt, u = motor[:-1].double(), motor[1:].double(), command[:-1].double()
    delta = nxt - m
    estimates = float(dt) * (u - m) / delta.where(delta.abs() > 1e-8, torch.ones_like(delta))
    estimates = estimates.clamp(1e-4, 10.0)
    masks = (delta >= 0.0, delta < 0.0)
    outputs = []
    for mask in masks:
        weight = delta.abs() * mask
        outputs.append((estimates * weight).sum(0) / weight.sum(0).clamp_min(1e-8))
    return torch.stack(outputs).to(dtype=command.dtype)


def analytic_privileged_ceiling(rows: list[dict[str, Any]], dt: float = 0.01) -> dict[str, Any]:
    from probe_contract_v5 import metadata as v5_metadata
    """Report exact six-parameter physics ceilings at each publication."""
    by_publication: dict[str, list[dict[str, Any]]] = {str(step): [] for step in PUBLICATION_STEPS}
    for row in rows:
        for scene in range(row["commands"].shape[1]):
            for step in PUBLICATION_STEPS:
                # Publications are cumulative so that a short early window
                # does not silently erase the opposite motor branch before
                # the scene has had time to excite it.
                sl = slice(0, step)
                fit = exact_physics_block_fit(
                    row["initial_motor"][sl, scene], row["true_motor"][sl, scene],
                    row["commands"][sl, scene], row["raw_specific"][sl, scene],
                    row["rotation_before"][sl, scene], row["omega_before"][sl, scene],
                    row["omega_after"][sl, scene], row["external_force"][sl, scene],
                    row["mass"][sl, scene], dt,
                )
                # The capability vector is a scoring target only; it never
                # enters exact_physics_block_fit's feature matrix.
                estimate_z = normalize_log_capability(fit["estimate"].unsqueeze(0))[0]
                by_publication[str(step)].append({
                    "estimate": fit["estimate"],
                    "error_z": estimate_z - row["target"][scene],
                    "rank": fit["rank"], "condition": fit["condition"],
                    "fit_rms": fit["fit_rms"],
                    "rise_support": fit["rise_support"],
                    "fall_support": fit["fall_support"],
                    "rise_support_per_motor": fit["rise_support_per_motor"],
                    "fall_support_per_motor": fit["fall_support_per_motor"],
                    "shared_tau_support": fit["shared_tau_support"],
                })

    publication_report: dict[str, Any] = {}
    all_errors = []
    for step, values in by_publication.items():
        errors = torch.stack([v["error_z"] for v in values])
        estimates = torch.stack([v["estimate"] for v in values])
        ranks = torch.tensor([v["rank"] for v in values])
        conditions = torch.tensor([v["condition"] for v in values])
        rise_per_motor = torch.stack([v["rise_support_per_motor"] for v in values])
        fall_per_motor = torch.stack([v["fall_support_per_motor"] for v in values])
        shared_values = [v["shared_tau_support"] for v in values]
        shared_rise = [v["shared_tau_support"]["branches"]["rise"] for v in values]
        shared_fall = [v["shared_tau_support"]["branches"]["fall"] for v in values]
        finite_conditions = conditions[torch.isfinite(conditions)]
        condition_median = (float(torch.quantile(finite_conditions, 0.50))
                            if finite_conditions.numel() == conditions.numel()
                            else float("inf"))
        condition_p95 = (float(torch.quantile(finite_conditions, 0.95))
                         if finite_conditions.numel() == conditions.numel()
                         else float("inf"))
        z_rms_per_dim = errors.square().mean(0).sqrt()
        abs_error_p95_per_dim = torch.quantile(errors.abs(), 0.95, dim=0)
        failed_checks: list[str] = []
        publication_report[step] = {
            "count": len(values), "rank_min": int(ranks.min()),
            "rank_full_fraction": float((ranks == 6).float().mean()),
            "condition_median": condition_median,
            "condition_p95": condition_p95,
            "condition_max": float(conditions.max()),
            "capability_z_rms_per_dim": z_rms_per_dim.tolist(),
            "normalized_z_rms_per_dim": z_rms_per_dim.tolist(),
            "absolute_error_p95_per_dim": abs_error_p95_per_dim.tolist(),
            "effectiveness_z_rms": float(errors[:, :4].square().mean().sqrt()),
            "effectiveness_dims_0_3_z_rms": float(errors[:, :3].square().mean().sqrt()),
            "tau_z_rms": float(errors[:, 4:].square().mean().sqrt()),
            "finite": bool(torch.isfinite(errors).all() and torch.isfinite(estimates).all()),
            "tau_support_pooled": bool(all(v["gate_passed"] for v in shared_values)),
            "rise_support_min_per_motor": rise_per_motor.min(dim=0).values.tolist(),
            "fall_support_min_per_motor": fall_per_motor.min(dim=0).values.tolist(),
            # These are the v4 gate quantities.  The per-motor minima above
            # are retained as diagnostics and intentionally do not gate.
            "tau_rise_support_min_pooled": min(v["pooled_valid"] for v in shared_rise),
            "tau_fall_support_min_pooled": min(v["pooled_valid"] for v in shared_fall),
            "tau_rise_distinct_calls_min": min(v["distinct_time_calls"] for v in shared_rise),
            "tau_fall_distinct_calls_min": min(v["distinct_time_calls"] for v in shared_fall),
            "tau_rise_motors_contributing_min": min(v["motors_contributing"] for v in shared_rise),
            "tau_fall_motors_contributing_min": min(v["motors_contributing"] for v in shared_fall),
            "tau_weighted_fisher_information_min": min(
                min(v["weighted_fisher_information"] for v in shared_rise),
                min(v["weighted_fisher_information"] for v in shared_fall),
            ),
            "tau_weighted_fisher_information_floor": SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION,
            "shared_tau_support_passed": bool(all(v["gate_passed"] for v in shared_values)),
            "support_per_motor_passed": bool((rise_per_motor >= 1).all() and (fall_per_motor >= 1).all()),
            "active_gate": int(step) in ACTIVE_PUBLICATIONS,
        }
        failed_checks = physics_gate_failed_checks(publication_report[step])
        publication_report[step]["failed_checks"] = failed_checks
        all_errors.append(errors)
    errors = torch.cat(all_errors, dim=0) if all_errors else torch.empty(0, 6)
    active_reports = [publication_report.get(str(step), {}) for step in ACTIVE_PUBLICATIONS]
    gate = bool(all(not value.get("failed_checks") for value in active_reports))
    failed_checks = {str(step): value.get("failed_checks", [])
                     for step, value in zip(ACTIVE_PUBLICATIONS, active_reports)
                     if value.get("failed_checks")}
    return {
        "publication": publication_report,
        # call25 is retained as a diagnostic row only; neither physical
        # pretraining gate below is allowed to depend on its quality.
        "tau_wls_finite": bool(all(value.get("finite", False) for value in active_reports)),
        "effectiveness_finite": bool(all(value.get("finite", False) for value in active_reports)),
        "effectiveness_normalized_z_rms": float(errors[:, :4].square().mean().sqrt()) if errors.numel() else float("inf"),
        "tau_wls_normalized_z_rms": float(errors[:, 4:].square().mean().sqrt()) if errors.numel() else float("inf"),
        "response_regression_rank_min": min(v["rank_min"] for v in publication_report.values()) if publication_report else 0,
        "response_regression_condition_max": max(v["condition_max"] for v in publication_report.values()) if publication_report else float("inf"),
        "response_regression_rank_passed": bool(all(publication_report[str(step)]["rank_min"] == 6
                                                     for step in ACTIVE_PUBLICATIONS)),
        "tau_wls_supported": bool(all(
            publication_report[str(step)]["shared_tau_support_passed"]
            for step in ACTIVE_PUBLICATIONS
        )),
        "uses_capability_labels_as_features": False,
        "block_design_columns": ["TW-1", "alpha_roll", "alpha_yaw", "beta", "inv_tau_up", "inv_tau_down"],
        "gate_passed": gate,
        "failed_checks": failed_checks,
        "probe_contract": v5_metadata(),
        "note": "external acceleration is projected with rotation_before; rise/fall tau are shared and gated on pooled time/motor support plus a weighted excitation proxy (not a statistical Fisher matrix); per-motor counts are diagnostic only",
    }


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize K15/K35 motor-state coverage before identifier fitting."""
    result: dict[str, Any] = {}
    for key, label in (("candidate15_motor", "K15"), ("candidate_motor", "K35")):
        by_step = {}
        for step in PUBLICATION_STEPS:
            errors, baseline = [], []
            for row in rows:
                truth = row["true_motor"][step - 1]
                errors.append((row[key][step - 1] - truth[:, None, :]).square().mean(-1).sqrt().min(-1).values)
                baseline.append((row["legacy_motor"][step - 1] - truth).square().mean(-1).sqrt())
            error, reference = torch.cat(errors), torch.cat(baseline)
            ratio = error / reference.clamp_min(1e-8)
            authority_finite = True
            for row in rows:
                truth_row = row["true_motor"][step - 1]
                candidate_row = row[key][step - 1]
                scene_error = (candidate_row - truth_row[:, None, :]).square().mean(-1).sqrt().min(-1).values
                for cell in set(row["strata"]):
                    cell_mask = torch.tensor([value == cell for value in row["strata"]], dtype=torch.bool)
                    authority_finite = authority_finite and bool(torch.isfinite(scene_error[cell_mask]).all())
            by_step[str(step)] = {
                "median_best_to_legacy": float(torch.median(ratio)),
                "p95_best_to_legacy": float(torch.quantile(ratio, 0.95)),
                "rms": float(error.mean()), "legacy_rms": float(reference.mean()),
                "motor_rms_p95": float(torch.quantile(error, 0.95)),
                "motor_rms_max": float(error.max()),
                "authority_cells_finite": authority_finite,
                "finite": bool(torch.isfinite(ratio).all()),
            }
            by_step[str(step)]["nearest_mode_failed_checks"] = coverage_gate_failed_checks(
                by_step[str(step)], strict_motor_rms=(label == "K35"))
            by_step[str(step)]["failed_checks"] = coverage_gate_failed_checks(
                by_step[str(step)], strict_motor_rms=False)
        result[label] = by_step
    # K15 is retained as a registered control arm.  The promotion coverage
    # gate is K35-only; a failing K15 reference is reported but cannot be
    # hidden by the K35 result.
    result["K15_reference_passed"] = bool(all(not result["K15"][str(step)]["failed_checks"]
                                              for step in ACTIVE_PUBLICATIONS))
    result["K35_gate_passed"] = bool(all(not result["K35"][str(step)]["failed_checks"]
                                         for step in ACTIVE_PUBLICATIONS))
    continuous = {}
    for step in ACTIVE_PUBLICATIONS:
        errors = []
        for row in rows:
            prefix = step - 25
            delta = (row["commands"][:prefix] - row["initial_motor"][:prefix]).double()
            change = (row["true_motor"][:prefix] - row["initial_motor"][:prefix]).double()
            taus = []
            for sign in (1, -1):
                mask = delta * sign > 1e-8
                denominator = (delta * change * mask).sum((0, 2))
                tau = .01 * (delta.square() * mask).sum((0, 2)) / denominator.clamp_min(1e-20)
                taus.append(tau)
            # Exactly the production continuous-tau observer, initialized
            # from previous_action, with no future target/true-state reset.
            motor = row["previous_action_seen"][0].double().clone()
            for command in row["commands"][:step].double():
                tau = torch.where(command >= motor, taus[0][:, None], taus[1][:, None])
                motor = motor + (.01 / tau.clamp_min(1e-8)).clamp_max(1) * (command - motor)
            errors.append((motor - row["true_motor"][step - 1]).square().mean(-1).sqrt())
        error = torch.cat(errors)
        continuous[str(step)] = {"fit_through_transition": step - 26,
            "held_out_transition_count": 25, "motor_rms_p95": float(error.quantile(.95)),
            "motor_rms_max": float(error.max()),
            "passed": bool(torch.isfinite(error).all() and error.quantile(.95) <= .001 and error.max() <= .005)}
    result["continuous_observer_ceiling"] = continuous
    result["continuous_observer_gate_passed"] = all(row["passed"] for row in continuous.values())
    result["passed"] = result["K35_gate_passed"] and result["continuous_observer_gate_passed"]
    result["scope"] = "K35 relative representation coverage plus continuous-tau observer ceiling; not trained-identifier accuracy"
    result["failed_checks"] = {
        label: {str(step): result[label][str(step)]["failed_checks"]
                for step in ACTIVE_PUBLICATIONS if result[label][str(step)]["failed_checks"]}
        for label in ("K15", "K35")
    }
    return result


def _finite_collection(row: dict[str, Any]) -> bool:
    """Check collected actions, states, observers, and response tensors."""
    names = (
        "q_actions", "commands", "residual_probe", "requested_probe",
        "previous_action_seen", "excitation_history", "true_motor",
        "initial_motor", "legacy_motor", "candidate15_motor", "candidate_motor",
        "raw_specific", "raw_angular", "response_features", "rotation_before",
        "omega_before", "omega_after", "external_force", "mass",
        "external_accel_body", "target",
    )
    if not all(torch.isfinite(row[name]).all().item() for name in names):
        return False
    return all(torch.isfinite(value).all().item()
               for state in row["canonical_states"]
               for value in state.values())


def _collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Blind seeds are deliberately excluded here.  They may only be consumed
    # by a later release stage after the feature/config/model hashes have been
    # frozen; this evolving diagnostic must not spend them accidentally.
    seeds = FORMAL_TRAIN_SEEDS
    if args.dry_run:
        seeds, args.scenarios, args.horizon = (FORMAL_TRAIN_SEEDS[:1], 16, 126)
    if getattr(args, "waveform", None) is not None:
        raise ValueError("v5 passive identification does not consume a v4 waveform file")
    waveform = load_probe_waveform()
    jobs = max(1, int(args.n_jobs))
    fn = lambda seed, amp: _collect_one(
        args.checkpoint, seed, amp, args.scenarios, args.horizon, waveform)
    pairs = [(seed, amp) for seed in seeds for amp in FORMAL_AMPLITUDES]
    if Parallel is not None and jobs > 1:
        return Parallel(n_jobs=jobs, backend="loky")([delayed(fn)(seed, amp) for seed, amp in pairs])
    return [fn(seed, amp) for seed, amp in pairs]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage not in ("collect-ceiling", "train", "all"):
        raise ValueError("--stage must be collect-ceiling, train, or all")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.dry_run:
        gate = probe_v5_eligibility(Path(args.probe_report), q2_checkpoint=args.checkpoint)
        if gate.get("eligible") is not True:
            raise RuntimeError("K35 collection requires the current frozen v5 probe")
    collected = _collect(args)
    active_rows = collected
    from tools.diagnose_probe_v5 import paired_metrics
    from probe_contract_v5 import metadata as v5_metadata, VERSION
    zero_by_seed = {r["seed"]: r for r in collected if r["amplitude"] == 0.0}
    paired_safety = {}
    for row in active_rows:
        zero = zero_by_seed[row["seed"]]
        actual = dict(row["metrics"], action=row["commands"])
        baseline = dict(zero["metrics"], action=zero["commands"])
        paired_safety[str(row["seed"])] = paired_metrics(baseline, actual, row["cells"])
    ceiling = analytic_privileged_ceiling(active_rows)
    coverage = _coverage_summary(active_rows)
    probe_eligibility = probe_v5_eligibility(
        Path(getattr(args, "probe_report", DEFAULT_PROBE_V4_REPORT)),
        q2_checkpoint=args.checkpoint,
    )
    failure_count = sum(int(r["identification_failed"].any().item()) for r in active_rows)
    result: dict[str, Any] = {"diagnostic": "causal-identifier-sequence-oracle", "stage": args.stage,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _hash_file(args.checkpoint),
        "code_sha256": _hash_file(Path(__file__)),
        # The six-arm sequence fit and blind-release schema are intentionally
        # unfinished.  A correctly sized run is still diagnostic-only until
        # those contracts are frozen and independently gated.
        # Eligibility is intentionally computed below only after the explicit
        # v4 freeze record and all pretraining checks are available.  The
        # sequence model remains unfrozen, so this stays false today.
        "formal_eligible": False,
        "probe_eligibility": probe_eligibility,
        "release_status": "blocked_pending_frozen_sequence_and_blind_protocol",
        "requested_formal_shape": bool(
            not args.dry_run and args.scenarios == 128 and args.horizon == 126 and args.n_jobs == 4
        ), "seed_split": {"train": list(FORMAL_TRAIN_SEEDS), "validation": [FORMAL_VALIDATION_SEED],
            "final": list(FORMAL_FINAL_SEEDS), "blind": list(FORMAL_BLIND_SEEDS)},
        "amplitudes": list(FORMAL_AMPLITUDES), "publications": list(ACTIVE_PUBLICATIONS),
        "consumed_seeds": {"train": sorted({row["seed"] for row in collected}),
                           "validation": [], "final": [], "blind": []},
        "collection_split": "training_only",
        "diagnostic_publications": [25],
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": {
            "call_index_completed_transitions": True,
            "call0_has_response": False,
            "publication_calls": [100, 125],
            "t50_call_index": 100,
            "diagnostic_calls": [25],
            "diagnostic_publications": [25],
            "active_publications": list(ACTIVE_PUBLICATIONS),
            "availability_t25": [0, 0, 0, 0, 0, 0],
            "publication_rule": "call100 and call125 only; call25 diagnostic/unavailable",
        },
        "probe_waveform_version": VERSION,
        "probe_amplitudes": list(FORMAL_AMPLITUDES),
        "probe_amplitude": float(FORMAL_AMPLITUDES[-1]),
        "probe_waveform_sha256": collected[0]["probe_waveform_sha256"] if collected else None,
        "probe_waveform_source": "probe_contract_v5.py",
        "probe_waveform_metadata": v5_metadata(),
        "feature_schema": feature_schema_metadata(),
        "feature_schema_sha256": feature_schema_sha256(),
        # The sequence model used by this diagnostic is intentionally a
        # sidecar upper-bound probe.  It is never a production identifier init
        # and never emits weights consumed by StructuredRecurrentPolicy.
        "oracle_sequence_model_is_sidecar": True,
        "production_identifier_init_artifact": None,
        "capability_labels_runtime_input": False,
        "training_protocol": {"optimizer": "AdamW", "lr": 3.0e-3, "weight_decay": 1.0e-4,
                              "batch_size": 64, "max_updates": int(args.max_updates),
                              "clip_norm": 1.0, "validation_every": 1, "patience": 10,
                              "initial_seeds": [101, 202, 303], "bootstrap": {"repeats": 10000, "seed": BOOTSTRAP_SEED}},
        "blind_status": "deferred_until_model_and_feature_hash_are_frozen",
        "arm_contract": {"L": "legacy24 + zero K35 bank (control)",
                         "M": "legacy24 + repeated 25-call signed bank mean (non-causal diagnostic)",
                         "U": "legacy24 + ordered absolute cross bank",
                         "S": "legacy24 + ordered signed K35 bank (production comparison)",
                         "T": "legacy24 + privileged true-motor bank",
                         "P": "unavailable: no pre-registered deconfounded response model"},
        "p_available": False,
        "representation_grid_version": 2,
        "tau_grid_sizes": sorted({len(r["tau_pairs"]) for r in collected}), "ceiling": ceiling,
        "coverage": coverage,
        "safety": {"finite_collected": bool(all(_finite_collection(r) for r in collected)),
                   "identification_failure_count": int(failure_count),
                   "no_identification_failure": failure_count == 0}}
    # Keep collection reusable without putting tensors in the JSON report.
    tensor_path = args.output.with_suffix(args.output.suffix + ".pt")
    torch.save(collected, tensor_path)
    result["collection_path"] = str(tensor_path.resolve())
    result["collection_sha256"] = _hash_file(tensor_path)
    result["paired_safety"] = paired_safety
    zero_rows = [r for r in collected if r["amplitude"] == 0.0]
    parity_values = [r["zero_parity"] for r in zero_rows]
    result["zero_parity"] = {
        "max_action_error": max((v["max_action_error"] for v in parity_values), default=float("inf")),
        "max_state_error": max((v["max_state_error"] for v in parity_values), default=float("inf")),
        "max_integral_error": max((v["max_integral_error"] for v in parity_values), default=float("inf")),
        "max_hidden_error": max((v["max_hidden_error"] for v in parity_values), default=float("inf")),
        "gate_passed": bool(parity_values and all(v["gate_passed"] for v in parity_values)),
        "tolerance": ZERO_PARITY_TOLERANCE,
    }
    pretraining_checks = {
        "paired_safety": bool(paired_safety) and all(v["passed"] for row in paired_safety.values() for v in row.values()),
        "coverage": bool(coverage["passed"]),
        "physics_ceiling": bool(ceiling.get("gate_passed", False)),
        "physics_finite": bool(
            ceiling["tau_wls_finite"] and ceiling["effectiveness_finite"]
        ),
        "physics_branch_support": bool(ceiling["tau_wls_supported"]),
        "finite_collection": bool(result["safety"]["finite_collected"]),
        "no_identification_failure": bool(
            result["safety"]["no_identification_failure"]
        ),
        "zero_q2_parity": bool(result["zero_parity"]["gate_passed"]),
    }
    result["pretraining_checks"] = pretraining_checks
    result["pretraining_gate_passed"] = bool(all(pretraining_checks.values()))
    formal_blockers = []
    if not probe_eligibility["eligible"]:
        formal_blockers.append(probe_eligibility["reason"])
    if not result["requested_formal_shape"]:
        formal_blockers.append("requested_formal_shape")
    if not result["pretraining_gate_passed"]:
        formal_blockers.append("pretraining_gate")
    # No sequence identifier artifact is frozen by this diagnostic.  Keep the
    # final condition explicit so a future v4 candidate cannot accidentally
    # turn collection into a release merely by editing the debug report.
    formal_blockers.append("sequence_identifier_artifact_not_frozen")
    result["formal_eligibility"] = {
        "eligible": False,
        "blockers": formal_blockers,
        "fail_closed": True,
    }
    # This release only freezes and audits collection.  The six-arm fitting
    # stage remains deliberately absent: no model, bootstrap, or blind release
    # may be inferred from this diagnostic while its protocol is still open.
    if args.stage in ("train", "all"):
        result["training_skipped_reason"] = "six-arm training is disabled; collection is fail-closed"
    result["gate_passed"] = bool(result["formal_eligible"] and result["coverage"]["passed"]
                                  and result["ceiling"].get("gate_passed", False)
                                  and result["ceiling"]["tau_wls_finite"]
                                  and result["ceiling"]["tau_wls_supported"]
                                  and result["ceiling"]["effectiveness_finite"]
                                  and result["safety"]["finite_collected"]
                                  and result["safety"]["no_identification_failure"]
                                  and result["zero_parity"]["gate_passed"])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_Q2_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("collect-ceiling", "train", "all"), default="all")
    parser.add_argument("--scenarios", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=126)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--waveform", type=Path, default=None,
                        help="optional file containing the exact v4 waveform (alternate tables are rejected)")
    parser.add_argument("--probe-report", type=Path, default=DEFAULT_PROBE_V4_REPORT,
                        help="v4 freeze report used for fail-closed eligibility")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.stage = "collect-ceiling"
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("diagnostic", "stage", "formal_eligible", "gate_passed", "ceiling")}, sort_keys=True))


if __name__ == "__main__":
    main()
