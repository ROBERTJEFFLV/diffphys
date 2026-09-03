"""Fixed-policy, paired authority-stratified risk evaluation.

This is deliberately independent of training.  It creates one frozen scenario
bank, evaluates reference/candidate policies on the same states, and computes
smooth RU-CVaR with at least eight effective tail scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - sequential fallback for minimal installs
    Parallel = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from smooth_risk import rockafellar_uryasev_cvar  # noqa: E402
from structured_rollout import (  # noqa: E402
    StructuredBoundaryCodec,
    StructuredClosedLoopState,
    load_structured_policy,
    rollout_structured_segment,
)


def _initial_observation(state: L2FState) -> torch.Tensor:
    return torch.cat(
        (
            state.position,
            state.velocity,
            state.rotation.reshape(state.position.shape[0], 9),
            state.omega,
            torch.zeros_like(state.position),
            state.previous_action,
        ),
        dim=-1,
    )


def _clone_state(state: L2FState) -> L2FState:
    return L2FState(
        **{name: getattr(state, name).detach().clone() for name in state.__dataclass_fields__}
    )


def build_scenario_bank(
    count: int = 64,
    *,
    seed: int = 7,
    dt: float = 0.01,
) -> list[dict[str, Any]]:
    """Build a deterministic bank and assign equal low/mid/high authority strata."""

    if count < 64:
        raise ValueError("fixed-policy risk bank requires at least 64 scenarios")
    simulator = L2FSimulator(L2FParams(dt=dt))
    scenarios = []
    for index in range(count):
        torch.manual_seed(int(seed) + index)
        state = simulator.reset(
            1,
            device="cpu",
            dtype=torch.float32,
            sample_dynamics=True,
            sampled_dynamics_level="broad",
            broad_sampler="physical-fit",
            balanced_dynamics_sampling=False,
            sample_external_force=True,
        )
        authority = float(state.alpha_roll_max.item())
        scenarios.append({"scenario_id": int(index), "state": _clone_state(state), "authority": authority})
    order = sorted(range(count), key=lambda i: scenarios[i]["authority"])
    low_end = count // 3
    high_start = count - count // 3
    for rank, index in enumerate(order):
        scenarios[index]["authority_stratum"] = "low" if rank < low_end else "high" if rank >= high_start else "mid"
    return scenarios


_POLICY_CACHE: dict[str, tuple[Any, dict]] = {}


def _load_cached(checkpoint: str):
    loaded = _POLICY_CACHE.get(checkpoint)
    if loaded is None:
        loaded = load_structured_policy(Path(checkpoint), device=torch.device("cpu"), dtype=torch.float32)
        loaded[0].eval()
        _POLICY_CACHE[checkpoint] = loaded
    return loaded


def _evaluate_one(checkpoint: str, scenario: dict[str, Any], horizon: int) -> dict[str, Any]:
    policy, payload = _load_cached(checkpoint)
    physical = _clone_state(scenario["state"])
    simulator = L2FSimulator(L2FParams(dt=policy.config.dt))
    recurrent = policy.initial_state(_initial_observation(physical))
    closed = StructuredClosedLoopState(physical=physical, policy=recurrent)
    codec = StructuredBoundaryCodec(physical, policy)
    end, trace = rollout_structured_segment(policy, simulator, closed, steps=horizon, collect=True)
    endpoint = codec.pack(end)
    control_error = codec.unpack(endpoint)
    p_norm = torch.linalg.vector_norm(trace["position"], dim=-1)
    v_norm = torch.linalg.vector_norm(trace["velocity"], dim=-1)
    w_norm = torch.linalg.vector_norm(trace["omega"], dim=-1)
    terminal_error = torch.linalg.vector_norm(
        torch.cat((control_error.physical.position / 0.10,
                   control_error.physical.velocity / 0.10,
                   control_error.physical.omega / 0.50), dim=-1), dim=-1
    )
    return {
        "scenario_id": scenario["scenario_id"],
        "authority": scenario["authority"],
        "authority_stratum": scenario["authority_stratum"],
        "checkpoint": checkpoint,
        "horizon": int(horizon),
        "finite": int(all(bool(torch.isfinite(value).all()) for value in trace.values())),
        "terminal_energy": float(terminal_error.square().mean()),
        "terminal_position": float(p_norm[-1]),
        "terminal_velocity": float(v_norm[-1]),
        "terminal_omega": float(w_norm[-1]),
        "max_position": float(p_norm.max()),
        "max_velocity": float(v_norm.max()),
        "max_omega": float(w_norm.max()),
        "action_rms": float(trace["action"].square().mean().sqrt()),
        "allocator_condition_max": float(trace["allocator_condition"].max()),
        "minimum_headroom_min": float(trace["minimum_headroom"].min()),
        "wrench_residual_rms": float(trace["wrench_residual"].square().mean().sqrt()),
        "fast_feedback_verified": int(bool(payload.get("fast_feedback_verified", False))),
    }


def _evaluate_checkpoint(checkpoint: str, scenarios: list[dict[str, Any]], horizon: int, n_jobs: int) -> list[dict[str, Any]]:
    if Parallel is None or n_jobs == 1:
        return [_evaluate_one(checkpoint, scenario, horizon) for scenario in scenarios]
    jobs = (delayed(_evaluate_one)(checkpoint, scenario, horizon) for scenario in scenarios)
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(jobs)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _risk(values: list[float], *, alpha: float, beta: float) -> float:
    tail = int(math.ceil((1.0 - alpha) * len(values)))
    if tail < 8:
        raise ValueError("effective CVaR tail must contain at least eight scenarios")
    return float(rockafellar_uryasev_cvar(torch.tensor(values), alpha=alpha, beta=beta, mode="softplus"))


def paired_metrics(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    alpha: float = 0.875,
    beta: float = 1.0e-2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Join same-bank rows and summarize deltas by authority stratum."""

    reference = {int(row["scenario_id"]): row for row in reference_rows}
    candidate = {int(row["scenario_id"]): row for row in candidate_rows}
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate scenario banks differ")
    metrics = ("terminal_energy", "terminal_position", "terminal_velocity", "terminal_omega", "max_position", "max_velocity", "max_omega", "action_rms", "wrench_residual_rms")
    paired = []
    for scenario_id in sorted(reference):
        left, right = reference[scenario_id], candidate[scenario_id]
        row = {"scenario_id": scenario_id, "authority": right["authority"], "authority_stratum": right["authority_stratum"]}
        for metric in metrics:
            row[f"reference_{metric}"] = left[metric]
            row[f"candidate_{metric}"] = right[metric]
            row[f"delta_{metric}"] = right[metric] - left[metric]
        paired.append(row)
    groups = []
    for stratum in ("low", "mid", "high"):
        rows = [row for row in paired if row["authority_stratum"] == stratum]
        if not rows:
            continue
        # Keep every authority-group CVaR statistically meaningful.  The
        # global bank uses the requested alpha, while a 21/22-sample stratum
        # lowers alpha just enough to retain at least eight tail samples.
        group_alpha = min(float(alpha), 1.0 - 8.0 / float(len(rows)))
        group = {"authority_stratum": stratum, "scenario_count": len(rows), "risk_alpha": group_alpha, "effective_tail_count": int(math.ceil((1.0 - group_alpha) * len(rows)))}
        for metric in metrics:
            ref_values = [float(row[f"reference_{metric}"]) for row in rows]
            cand_values = [float(row[f"candidate_{metric}"]) for row in rows]
            group[f"reference_mean_{metric}"] = sum(ref_values) / len(ref_values)
            group[f"candidate_mean_{metric}"] = sum(cand_values) / len(cand_values)
            group[f"delta_mean_{metric}"] = group[f"candidate_mean_{metric}"] - group[f"reference_mean_{metric}"]
            group[f"reference_cvar_{metric}"] = _risk(ref_values, alpha=group_alpha, beta=beta)
            group[f"candidate_cvar_{metric}"] = _risk(cand_values, alpha=group_alpha, beta=beta)
        groups.append(group)
    all_energy_reference = [float(row["reference_terminal_energy"]) for row in paired]
    all_energy_candidate = [float(row["candidate_terminal_energy"]) for row in paired]
    summary = {
        "scenario_count": len(paired),
        "authority_strata": {key: sum(row["authority_stratum"] == key for row in paired) for key in ("low", "mid", "high")},
        "alpha": alpha,
        "risk_beta": beta,
        "effective_tail_count": int(math.ceil((1.0 - alpha) * len(paired))),
        "reference_terminal_energy_cvar": _risk(all_energy_reference, alpha=alpha, beta=beta),
        "candidate_terminal_energy_cvar": _risk(all_energy_candidate, alpha=alpha, beta=beta),
        "candidate_minus_reference_terminal_energy_cvar": _risk(all_energy_candidate, alpha=alpha, beta=beta) - _risk(all_energy_reference, alpha=alpha, beta=beta),
    }
    return paired, groups, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired fixed-policy structured L2F risk evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.875)
    parser.add_argument("--risk-beta", type=float, default=1.0e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon < 1 or args.scenarios < 64:
        raise ValueError("horizon must be positive and scenarios must be at least 64")
    if int(math.ceil((1.0 - args.alpha) * args.scenarios)) < 8:
        raise ValueError("alpha leaves fewer than eight effective tail scenarios")
    reference_path = args.reference_checkpoint or args.checkpoint
    scenarios = build_scenario_bank(args.scenarios, seed=args.seed)
    reference_rows = _evaluate_checkpoint(str(reference_path.resolve()), scenarios, args.horizon, args.n_jobs)
    candidate_rows = _evaluate_checkpoint(str(args.checkpoint.resolve()), scenarios, args.horizon, args.n_jobs)
    paired, groups, summary = paired_metrics(reference_rows, candidate_rows, alpha=args.alpha, beta=args.risk_beta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "reference_per_scenario.csv", reference_rows)
    _write_csv(args.output_dir / "candidate_per_scenario.csv", candidate_rows)
    _write_csv(args.output_dir / "paired_per_scenario.csv", paired)
    _write_csv(args.output_dir / "authority_group_metrics.csv", groups)
    payload = {"checkpoint": str(args.checkpoint.resolve()), "reference_checkpoint": str(reference_path.resolve()), "horizon": args.horizon, "seed": args.seed, "n_jobs": args.n_jobs, **summary}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
