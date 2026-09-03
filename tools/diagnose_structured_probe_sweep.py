"""Paired H50 identification-probe safety sweep.

This diagnostic is intentionally independent of training.  For each fixed
authority-balanced scenario bank it starts every amplitude from the same
physical/recurrent state, runs the real differentiable L2F simulator through
call125 (H126 policy calls),
and reports the effect of the deterministic burn-in probe.  The policy module
is not changed: amplitudes are injected through a replaced immutable config
when constructing a fresh policy copy for each arm.

Example::

    python3 tools/diagnose_structured_probe_sweep.py \
      --checkpoint checkpoints/structured_full_space_current_smoke.pt \
      --output reports/structured_probe_sweep.json \
      --seeds 1707 1708 1709 2707 --n-jobs 4

The scenario bank and amplitude arms are paired within each seed.  ``amp=0``
is the per-seed reference arm.  A candidate is marked safe only when all
scenarios are finite, its angular-rate peak is no more than 1.5 times the
paired reference (with a 5 rad/s absolute guard when the reference is smaller), its translational peaks stay
within a conservative 2x reference/absolute bound, and action increments are
finite and in the deployable range.  These criteria are pre-registered in the
JSON output; they are a screening gate, not a stability certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - sequential fallback
    Parallel = None

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from equilibrium_control import analytic_equilibrium_target  # noqa: E402
from structured_distillation import (  # noqa: E402
    build_dagger_scenario_bank,
    normalize_log_capability,
)
from structured_checkpoint import CADENCE_SEMANTICS_VERSION  # noqa: E402
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy  # noqa: E402
from structured_rollout import StructuredClosedLoopState, structured_observation  # noqa: E402


DEFAULT_AMPLITUDES = (0.0, 0.005, 0.01, 0.02, 0.05)
DEFAULT_SEEDS = (1707, 1708, 1709, 2707)
HORIZON = 126
RECOVERY_STEPS = (75, 125)
TAIL_START = 75


def _clone_state(state: L2FState) -> L2FState:
    return L2FState(**{
        name: getattr(state, name).detach().clone()
        for name in state.__dataclass_fields__
    })


def _load_policy(checkpoint: Path, amplitude: float) -> StructuredRecurrentPolicy:
    payload = torch.load(checkpoint, map_location="cpu")
    config = StructuredPolicyConfig(**payload["config"])
    config = replace(config, burn_in_probe_amplitude=float(amplitude))
    policy = StructuredRecurrentPolicy(config).to(device="cpu", dtype=torch.float32)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    return policy


def _percentile(value: torch.Tensor, q: float) -> float:
    return float(torch.quantile(value.reshape(-1).float(), q).item())


def _modal(action: torch.Tensor) -> torch.Tensor:
    """Return collective/roll/pitch/yaw modal coordinates for motor commands."""

    return torch.stack(
        (
            action.mean(dim=-1),
            action[:, 1] - action[:, 3],
            action[:, 2] - action[:, 0],
            action[:, 0] - action[:, 1] + action[:, 2] - action[:, 3],
        ),
        dim=-1,
    )


def _modal_trace(requested: torch.Tensor, executed: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for step in range(requested.shape[0]):
        row: dict[str, Any] = {"step": int(step + 1)}
        for name, values in (("requested", requested[step]), ("executed", executed[step])):
            row[name] = {
                "mean": [float(v) for v in values.mean(dim=0)],
                "rms": [float(v) for v in values.square().mean(dim=0).sqrt()],
                "p99_abs": [float(v) for v in values.abs().quantile(0.99, dim=0)],
                "max_abs": [float(v) for v in values.abs().amax(dim=0)],
            }
        rows.append(row)
    return rows


def _gramian(innovation: torch.Tensor) -> dict[str, Any]:
    """Build the actual-command [u_t,u_t-4,u_t-12] Gramian."""

    if innovation.ndim != 3 or innovation.shape[-1] != 4:
        raise ValueError("innovation must have shape [time,batch,4]")
    if innovation.shape[0] <= 12:
        raise ValueError("Horizon must exceed the largest Gramian lag")
    rows = []
    for index in range(12, innovation.shape[0]):
        rows.append(torch.cat((innovation[index], innovation[index - 4], innovation[index - 12]), dim=-1))
    design = torch.cat(rows, dim=0)
    gram = design.transpose(0, 1) @ design / max(design.shape[0], 1)
    singular = torch.linalg.svdvals(gram)
    threshold = float(singular.max().item()) * 1.0e-6
    rank = int((singular > threshold).sum().item())
    smallest = float(singular.min().item())
    condition = float(singular.max().item() / max(smallest, 1.0e-12))
    return {
        "feature_order": ["u_t", "u_t_minus_4", "u_t_minus_12"],
        "feature_dim": 12,
        "sample_count": int(design.shape[0]),
        "rank": rank,
        "condition": condition,
        "singular_values": [float(v) for v in singular],
        "rank12": bool(rank == 12),
        "condition_lt_30": bool(condition < 30.0),
        "identifiable": bool(rank == 12 and condition < 30.0),
    }


def _one_arm(
    checkpoint: Path,
    bank_state: L2FState,
    amplitude: float,
    *,
    horizon: int = HORIZON,
) -> dict[str, Any]:
    """Run one arm over a complete fixed bank and return raw aggregate metrics."""

    policy = _load_policy(checkpoint, amplitude)
    physical = _clone_state(bank_state)
    simulator = L2FSimulator(L2FParams(dt=policy.config.dt))
    observation = torch.cat(
        (
            physical.position,
            physical.velocity,
            physical.rotation.reshape(physical.position.shape[0], 9),
            physical.omega,
            torch.zeros_like(physical.position),
            physical.previous_action,
        ),
        dim=-1,
    )
    recurrent = policy.initial_state(observation)
    current = StructuredClosedLoopState(physical=physical, policy=recurrent)
    action_trace = []
    requested_trace = []
    executed_trace = []
    innovation_trace = []
    position_trace = []
    velocity_trace = []
    omega_trace = []
    rate_trace = []
    rate_limit_trace = []
    action_cap_trace = []
    capability_trace = []
    capability_z_trace = []
    trim_trace = []
    body_z_trace = []
    disturbance_trace = []
    identification_failure_trace = []
    finite = True
    with torch.no_grad():
        for _ in range(int(horizon)):
            observation = structured_observation(current)
            output = policy.forward_with_aux(
                observation, current.policy, simulator.params.dt
            )
            # The observer's motor estimate is the state immediately before
            # this command.  This is the same causal command innovation used
            # by the production identifier, and uses the command that really
            # reaches the simulator rather than the requested probe.
            executed_action = output.auxiliary["applied_action"]
            innovation_trace.append(executed_action - current.policy.motor_estimate)
            next_physical = simulator.step(
                current.physical, output.action, grad_decay=1.0
            )
            current = StructuredClosedLoopState(
                physical=next_physical, policy=output.next_state
            )
            allocator = output.auxiliary["allocator"]
            action_trace.append(output.action)
            requested_trace.append(output.auxiliary["identification_probe_action"])
            executed_trace.append(executed_action)
            position_trace.append(next_physical.position)
            velocity_trace.append(next_physical.velocity)
            omega_trace.append(next_physical.omega)
            rate_trace.append(allocator.rate_limited)
            rate_limit_trace.append(output.auxiliary["rate_limit"])
            action_cap_trace.append(output.auxiliary["action_delta_cap"])
            capability_trace.append(output.auxiliary["capability"])
            capability_z_trace.append(output.auxiliary["capability_z_mean"])
            trim_trace.append(output.auxiliary["trim_action"])
            body_z_trace.append(output.auxiliary["body_z"])
            disturbance_trace.append(output.auxiliary["disturbance_accel"])
            identification_failure_trace.append(output.auxiliary["identification_failed"])
            finite = finite and bool(
                torch.isfinite(output.action).all()
                and torch.isfinite(next_physical.position).all()
                and torch.isfinite(next_physical.velocity).all()
                and torch.isfinite(next_physical.omega).all()
                and torch.isfinite(allocator.rate_limited).all()
            )

    action = torch.stack(action_trace)
    requested = torch.stack(requested_trace)
    executed = torch.stack(executed_trace)
    innovation = torch.stack(innovation_trace)
    position = torch.linalg.vector_norm(torch.stack(position_trace), dim=-1)
    velocity = torch.linalg.vector_norm(torch.stack(velocity_trace), dim=-1)
    omega = torch.linalg.vector_norm(torch.stack(omega_trace), dim=-1)
    # Include the first command relative to the scenario's previous action so
    # H50 reports an actual command-rate sequence, not only T-1 differences.
    previous_action = bank_state.previous_action.detach()
    action_delta = torch.cat(
        (
            (action[:1] - previous_action.unsqueeze(0)).abs(),
            (action[1:] - action[:-1]).abs(),
        ),
        dim=0,
    )
    rate_limited = torch.stack(rate_trace)
    rate_limit = torch.stack(rate_limit_trace)
    action_cap = torch.stack(action_cap_trace)
    capability = torch.stack(capability_trace)
    capability_z = torch.stack(capability_z_trace)
    trim = torch.stack(trim_trace)
    body_z = torch.stack(body_z_trace)
    disturbance = torch.stack(disturbance_trace)
    identification_failures = torch.stack(identification_failure_trace)
    true_capability = torch.stack(
        (
            bank_state.thrust_to_weight,
            bank_state.alpha_roll_max,
            bank_state.eta_yaw,
            bank_state.jz_over_jxy,
            bank_state.motor_time_rising,
            bank_state.motor_time_falling,
        ), dim=-1,
    )
    true_capability_z = normalize_log_capability(true_capability)
    equilibrium = analytic_equilibrium_target(
        bank_state, gravity=simulator.params.gravity
    )
    capability_error = (capability_z - true_capability_z.unsqueeze(0)).square().mean(dim=-1).sqrt()
    trim_error = (trim - equilibrium.motor_trim.unsqueeze(0)).square().mean(dim=-1).sqrt()
    body_z_error = (body_z - equilibrium.body_z.unsqueeze(0)).square().sum(dim=-1).sqrt()
    disturbance_target = bank_state.external_force / bank_state.mass[:, None]
    disturbance_error = (disturbance - disturbance_target.unsqueeze(0)).square().mean(dim=-1).sqrt()
    gramian = _gramian(innovation)
    modal_requested = _modal(requested.reshape(-1, 4)).reshape(requested.shape[0], requested.shape[1], 4)
    modal_executed = _modal(executed.reshape(-1, 4)).reshape(executed.shape[0], executed.shape[1], 4)
    recovery = {}
    for step in RECOVERY_STEPS:
        # Physical traces are recorded after each command (row0 is state1),
        # unlike capability traces which are indexed by policy call.
        index = step - 1
        recovery[f"h{step}"] = {
            "position": float(position[index].mean()),
            "position_p99": _percentile(position[index], 0.99),
            "velocity": float(velocity[index].mean()),
            "velocity_p99": _percentile(velocity[index], 0.99),
            "omega": float(omega[index].mean()),
            "omega_p99": _percentile(omega[index], 0.99),
        }
    tail = {
        "start": TAIL_START,
        "end": int(horizon),
        "position": float(position[TAIL_START - 1:].mean()),
        "position_p99": _percentile(position[TAIL_START - 1:], 0.99),
        "velocity": float(velocity[TAIL_START - 1:].mean()),
        "velocity_p99": _percentile(velocity[TAIL_START - 1:], 0.99),
        "omega": float(omega[TAIL_START - 1:].mean()),
        "omega_p99": _percentile(omega[TAIL_START - 1:], 0.99),
    }
    capability_checkpoints = {}
    for step in (50, 75, 125):
        index = step
        capability_checkpoints[f"t{step}"] = {
            "capability_mean_error_rms": float(capability_error[index].mean()),
            "capability_mean_error_p99": _percentile(capability_error[index], 0.99),
            "trim_error_rms": float(trim_error[index].mean()),
            "trim_error_p99": _percentile(trim_error[index], 0.99),
            "body_z_error_rms": float(body_z_error[index].mean()),
            "body_z_error_p99": _percentile(body_z_error[index], 0.99),
            "disturbance_error_rms": float(disturbance_error[index].mean()),
            "disturbance_error_p99": _percentile(disturbance_error[index], 0.99),
            "identification_failed_count": int(identification_failures[index].sum()),
        }
    return {
        "amplitude": float(amplitude),
        "horizon": int(horizon),
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": {
            "call_index_completed_transitions": True,
            "call0_has_response": False,
            "t50_call_index": 50,
        },
        "scenario_count": int(bank_state.position.shape[0]),
        "finite": int(finite),
        "position_max": float(position.max()),
        "position_p99": _percentile(position, 0.99),
        "velocity_max": float(velocity.max()),
        "velocity_p99": _percentile(velocity, 0.99),
        "omega_max": float(omega.max()),
        "omega_p99": _percentile(omega, 0.99),
        "action_delta_rms": float(action_delta.square().mean().sqrt()),
        "action_delta_max": float(action_delta.max()),
        "rate_bound_activity_mean": float(rate_limited.mean()),
        "rate_bound_activity_max": float(rate_limited.max()),
        "rate_bound_fraction": float((rate_limited > 1.0e-8).float().mean()),
        "rate_limit_min": float(rate_limit.min()),
        "action_delta_cap_max": float(action_cap.max()),
        "recovery": recovery,
        "tail_h75_h125": tail,
        "capability_checkpoints": capability_checkpoints,
        "requested_executed_modal_trace": _modal_trace(
            modal_requested, modal_executed
        ),
        "command_innovation_gramian": gramian,
    }


def _seed_result(
    checkpoint: Path,
    seed: int,
    amplitudes: tuple[float, ...],
    *,
    scenarios: int,
    horizon: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    bank = build_dagger_scenario_bank(
        int(scenarios), seed=int(seed), dt=0.01, per_cell=int(scenarios) // 16
    )
    rows = [
        _one_arm(checkpoint, bank.state, amplitude, horizon=horizon)
        for amplitude in amplitudes
    ]
    reference = rows[0]
    for row in rows:
        row["paired_reference_amplitude"] = float(amplitudes[0])
        row["omega_peak_ratio_to_zero"] = (
            float(row["omega_max"]) / max(float(reference["omega_max"]), 1.0e-12)
        )
        row["position_peak_ratio_to_zero"] = (
            float(row["position_max"]) / max(float(reference["position_max"]), 1.0e-12)
        )
        row["velocity_peak_ratio_to_zero"] = (
            float(row["velocity_max"]) / max(float(reference["velocity_max"]), 1.0e-12)
        )
    return {"seed": int(seed), "scenario_count": int(scenarios), "arms": rows}


def _safety(row: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "finite": bool(row["finite"]),
        "omega_peak_le_1p5x_reference_and_5": bool(
            row["omega_max"] <= max(5.0, 1.5 * max(reference["omega_max"], 1.0e-12))
        ),
        "position_peak_le_2x_reference_and_5": bool(
            row["position_max"] <= max(5.0, 2.0 * max(reference["position_max"], 1.0e-12))
        ),
        "velocity_peak_le_2x_reference_and_20": bool(
            row["velocity_max"] <= max(20.0, 2.0 * max(reference["velocity_max"], 1.0e-12))
        ),
        "action_delta_finite_and_deployable": bool(
            math.isfinite(row["action_delta_max"]) and row["action_delta_max"] <= 1.0
        ),
    }
    return all(checks.values()), checks


def _recovery_checks(
    row: dict[str, Any], reference: dict[str, Any], *, tolerance: float = 0.05
) -> tuple[bool, dict[str, bool]]:
    """Compare H75/H125 endpoints and H75-H125 tail to the zero-probe arm."""

    checks: dict[str, bool] = {}
    for section in ("h75", "h125"):
        left = row["recovery"][section]
        right = reference["recovery"][section]
        for signal in (
            "position", "position_p99", "velocity", "velocity_p99",
            "omega", "omega_p99",
        ):
            checks[f"{section}_{signal}_within_5pct"] = bool(
                abs(float(left[signal]) - float(right[signal]))
                <= float(tolerance) * max(abs(float(right[signal])), 1.0e-6)
            )
    left = row["tail_h75_h125"]
    right = reference["tail_h75_h125"]
    for signal in (
        "position", "position_p99", "velocity", "velocity_p99",
        "omega", "omega_p99",
    ):
        checks[f"tail_{signal}_within_5pct"] = bool(
            abs(float(left[signal]) - float(right[signal]))
            <= float(tolerance) * max(abs(float(right[signal])), 1.0e-6)
        )
    return all(checks.values()), checks


def summarize(
    seed_results: list[dict[str, Any]], amplitudes: tuple[float, ...]
) -> tuple[list[dict[str, Any]], float | None]:
    summary = []
    max_safe: float | None = None
    for amplitude in amplitudes:
        candidates = []
        checks_by_seed = []
        for result in seed_results:
            rows = result["arms"]
            reference = rows[0]
            row = next(r for r in rows if r["amplitude"] == amplitude)
            safe, checks = _safety(row, reference)
            recovery_safe, recovery_checks = _recovery_checks(row, reference)
            candidates.append(row)
            identifiable = bool(row["command_innovation_gramian"]["identifiable"])
            checks_by_seed.append({
                "seed": result["seed"],
                "safe": bool(safe and recovery_safe),
                "dynamic_safe": safe,
                "recovery_safe_5pct": recovery_safe,
                "identifiable": identifiable,
                "checks": checks,
                "recovery_checks": recovery_checks,
            })
        # Summary is a mean only for descriptive comparison; safety is an
        # all-seed/all-scenario gate and never inferred from the mean.
        summary_row = {
            "amplitude": float(amplitude),
            "seed_count": len(candidates),
            "safe_all_seeds": bool(all(item["safe"] for item in checks_by_seed)),
            "dynamic_safe_all_seeds": bool(all(item["dynamic_safe"] for item in checks_by_seed)),
            "recovery_5pct_all_seeds": bool(all(item["recovery_safe_5pct"] for item in checks_by_seed)),
            "identifiable_all_seeds": bool(all(item["identifiable"] for item in checks_by_seed)),
            "safety_by_seed": checks_by_seed,
        }
        for key in (
            "position_max", "position_p99", "velocity_max", "velocity_p99",
            "omega_max", "omega_p99", "action_delta_rms", "action_delta_max",
            "rate_bound_activity_mean", "rate_bound_activity_max", "rate_bound_fraction",
        ):
            values = [float(row[key]) for row in candidates]
            summary_row[f"mean_{key}"] = sum(values) / len(values)
            summary_row[f"max_{key}"] = max(values)
        for section, source in (
            ("h75", lambda row: row["recovery"]["h75"]),
            ("h125", lambda row: row["recovery"]["h125"]),
            ("tail_h75_h125", lambda row: row["tail_h75_h125"]),
        ):
            for signal in ("position", "velocity", "omega"):
                values = [float(source(row)[signal]) for row in candidates]
                summary_row[f"mean_{section}_{signal}"] = sum(values) / len(values)
                summary_row[f"max_{section}_{signal}"] = max(values)
        for checkpoint in ("t50", "t75", "t125"):
            for metric in (
                "capability_mean_error_rms", "trim_error_rms",
                "body_z_error_rms", "disturbance_error_rms",
            ):
                values = [float(row["capability_checkpoints"][checkpoint][metric]) for row in candidates]
                summary_row[f"mean_{checkpoint}_{metric}"] = sum(values) / len(values)
                summary_row[f"max_{checkpoint}_{metric}"] = max(values)
        gramian_conditions = [float(row["command_innovation_gramian"]["condition"]) for row in candidates]
        gramian_ranks = [int(row["command_innovation_gramian"]["rank"]) for row in candidates]
        summary_row["mean_gramian_condition"] = sum(gramian_conditions) / len(gramian_conditions)
        summary_row["max_gramian_condition"] = max(gramian_conditions)
        summary_row["min_gramian_rank"] = min(gramian_ranks)
        summary.append(summary_row)
        if summary_row["safe_all_seeds"]:
            max_safe = float(amplitude)
    return summary, max_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--amplitudes", type=float, nargs="+", default=list(DEFAULT_AMPLITUDES))
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.horizon != HORIZON:
        raise ValueError("registered probe sweep requires --horizon 126")
    if args.scenarios < 16 or args.scenarios % 16:
        raise ValueError("scenarios must be a positive multiple of 16")
    amplitudes = tuple(float(v) for v in args.amplitudes)
    if len(amplitudes) < 2 or abs(amplitudes[0]) > 1.0e-12:
        raise ValueError("the first amplitude must be the paired zero reference")
    if any(value < 0.0 for value in amplitudes):
        raise ValueError("probe amplitudes must be non-negative")
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    jobs = args.n_jobs
    if Parallel is None or jobs == 1:
        results = [
            _seed_result(args.checkpoint, seed, amplitudes,
                         scenarios=args.scenarios, horizon=args.horizon)
            for seed in seeds
        ]
    else:
        work = (
            delayed(_seed_result)(
                args.checkpoint, seed, amplitudes,
                scenarios=args.scenarios, horizon=args.horizon,
            )
            for seed in seeds
        )
        results = Parallel(n_jobs=jobs, backend="loky", verbose=0)(work)
    results.sort(key=lambda item: int(item["seed"]))
    compact, max_safe = summarize(results, amplitudes)
    payload = {
        "diagnostic": "structured-policy-paired-integrated-probe-sweep-v2",
        "checkpoint": str(args.checkpoint.resolve()),
        "seeds": list(seeds),
        "amplitudes": list(amplitudes),
        "horizon": int(args.horizon),
        "scenario_count": int(args.scenarios),
        "scenario_bank": "4x4 thrust-to-weight/log-alpha physical-fit, paired per seed",
        "execution": "L2FSimulator.step, CPU float32, no training, fresh policy per arm",
        "safety_criteria": {
            "finite": "all actions and position/velocity/omega/rate diagnostics finite",
            "omega": "peak <= max(5 rad/s, 1.5x paired amp=0 peak), per seed; 5 is an absolute guard when the reference is below it",
            "position": "peak <= max(5, 2x paired amp=0 peak), per seed; the ratio guard remains active for high-error references",
            "velocity": "peak <= max(20, 2x paired amp=0 peak), per seed; the ratio guard remains active for high-error references",
            "action_delta": "finite and <= 1.0 deployable normalized-action units",
            "recovery": "H75/H125 endpoints and H75-H125 tail position/velocity/omega within 5% of paired amp=0",
            "aggregation": "candidate is safe only if every seed passes every check",
        },
        "identifiability_criteria": {
            "input": "actual executed command innovation = applied_action - pre-command motor estimate",
            "regressor": "[u_t, u_t-4, u_t-12] over call t=13..125, concatenated across scenarios",
            "rank": "exactly 12 at 1e-6 relative singular-value threshold",
            "condition": "Gramian condition number < 30",
            "aggregation": "identifiable only if every seed passes rank and condition",
        },
        "maximum_safe_amplitude": max_safe,
        "compact_summary": compact,
        "per_seed": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnostic": payload["diagnostic"],
        "maximum_safe_amplitude": max_safe,
        "compact_summary": compact,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
