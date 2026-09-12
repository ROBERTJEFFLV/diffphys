"""Offline exact H500 versus reverse-window BPTT; never updates an Actor."""

from __future__ import annotations

import argparse
from dataclasses import fields
import importlib.util
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env_l2f import L2FParams, L2FSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, rollout, step_costs, risk_weights
from response_adjoints import collect_boundary_rollout, backward_actor, snapshot, compare_boundary
from response_training import (
    atomic_json,
    source_hash,
    file_hash,
    migrate_actor_weights,
    sample_training_scenarios,
    load_legacy_checkpoint,
)


def gradient_vector(policy):
    return torch.cat(
        [
            torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten()
            for p in policy.parameters()
        ]
    )


def check(policy, simulator, initial, horizon, windows, config):
    reference_boundaries = {}

    def capture(step, closed):
        if any(step % window == 0 for window in windows):
            reference_boundaries[step] = snapshot(closed)

    policy.zero_grad(set_to_none=True)
    full = rollout(policy, simulator, initial, horizon, boundary_observer=capture)
    costs = step_costs(full, config).sum(0)
    weights = risk_weights(costs, config)
    (0.1 * (weights * costs).sum()).backward()
    exact = gradient_vector(policy).detach().double()
    if not bool(torch.isfinite(exact).all()) or not bool(exact.norm() > 0):
        raise FloatingPointError("full H500 diagnostic gradient is nonfinite or zero")
    del full, costs, weights
    rows = []
    for window in windows:
        policy.zero_grad(set_to_none=True)
        record = collect_boundary_rollout(
            policy, simulator, initial, config, horizon=horizon, window_steps=window
        )
        continuity = [
            compare_boundary(reference_boundaries[t], closed, t)
            for t, closed in record.boundaries.items()
            if t
        ]
        report = backward_actor(policy, simulator, record, config, gradient_scale=0.1)
        gradient = gradient_vector(policy).double()
        relative_error = float((gradient - exact).norm() / exact.norm())
        cosine = float(torch.dot(gradient, exact) / (gradient.norm() * exact.norm()))
        rows.append(
            {
                "window": window,
                "relative_error": relative_error,
                "cosine": cosine,
                "full_gradient_norm": float(exact.norm()),
                "full_forward_boundaries": continuity,
                "reverse_boundaries": report["boundaries"],
                "record_tensor_bytes": sum(
                    x.numel() * x.element_size()
                    for closed in record.boundaries.values()
                    for state in (closed.physical, closed.policy)
                    for f in fields(state)
                    for x in [getattr(state, f.name)]
                )
                + record.costs.numel() * record.costs.element_size()
                + record.weights.numel() * record.weights.element_size(),
            }
        )
        if relative_error > 1e-4:
            raise RuntimeError(
                "window %d differs from full H500 gradient: %.6g" % (window, relative_error)
            )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--legacy-policy-path",
        type=Path,
        help="optional local historical policy source for all-action parity",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--windows", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--scenarios", type=int, default=32, help="two pooled TRAIN banks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(w < 1 or args.horizon % w for w in args.windows):
        parser.error("invalid windows")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "device": args.device,
        "torch": str(torch.__version__),
        "dtype": "float32",
        "source_sha256": source_hash(),
        "scenarios": 2 * args.scenarios,
        "horizon": args.horizon,
        "actor_updates": 0,
        "status": "running",
        "cases": [],
    }
    if args.device == "cuda":
        result["gpu"] = torch.cuda.get_device_name()
    started = time.monotonic()
    legacy = None
    if args.legacy_policy_path:
        spec = importlib.util.spec_from_file_location(
            "_historical_response_policy", args.legacy_policy_path
        )
        legacy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = legacy
        spec.loader.exec_module(legacy)
        result["legacy_policy_sha256"] = file_hash(args.legacy_policy_path)
    try:
        for path in args.checkpoints or [None]:
            saved = load_legacy_checkpoint(path) if path else {}
            config = ResponsePolicyConfig(**saved.get("policy_config", {}))
            policy = ResponseMotorPolicy(config).to(args.device)
            state = saved.get("model", saved.get("actor", policy.state_dict()))
            migrate_actor_weights(policy, state)
            initial, seeds = sample_training_scenarios(
                args.scenarios, 0, dt=config.dt, device=torch.device(args.device)
            )
            sim = L2FSimulator(L2FParams(dt=config.dt))
            row = {
                "checkpoint": str(path),
                "checkpoint_sha256": file_hash(path) if path else None,
                "seeds": seeds,
            }
            result["cases"].append(row)
            if legacy is not None:
                old = legacy.ResponseMotorPolicy(
                    legacy.ResponsePolicyConfig(**saved.get("policy_config", {}))
                ).to(args.device)
                old.load_state_dict(state, strict=True)
                with torch.no_grad():
                    before, after = [rollout(p, sim, initial, args.horizon) for p in (old, policy)]
                    row["auxiliary_removal"] = {
                        n: torch.equal(getattr(before, n), getattr(after, n))
                        for n in ("actions", "positions", "velocities", "omegas", "observations")
                    }
                    if not all(row["auxiliary_removal"].values()):
                        raise RuntimeError("auxiliary removal changed forward")
                del old, before, after
            row["gradients"] = check(
                policy, sim, initial, args.horizon, args.windows, TaskLossConfig()
            )
            atomic_json(args.output, result)
            print(row, flush=True)
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, result)
    return result


if __name__ == "__main__":
    main()
