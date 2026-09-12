"""One-time task-gradient/information contract check; no optimizer updates."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FParams, L2FSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, observation, rollout, sample_scenarios, task_loss
from response_training import TRAIN_SEED_BASE, atomic_json, source_hash, train


def inspect_calls(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))}


def run_contract(*, device="cuda", loss_config=None, scenario_mode="fixed-airframe"):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; do not call a skipped check passed")
    config = ResponsePolicyConfig()
    loss_config = TaskLossConfig() if loss_config is None else loss_config
    policy = ResponseMotorPolicy(config).to(device)
    simulator = L2FSimulator(L2FParams(dt=config.dt))
    initial, _ = sample_scenarios(16, seed=TRAIN_SEED_BASE, dt=config.dt, device=torch.device(device),
                                  scenario_mode=scenario_mode)
    captured = []
    handle = policy.register_forward_hook(lambda module, inputs, output: captured.append(output.action))
    trace = rollout(policy, simulator, initial, 16)
    handle.remove()
    loss = task_loss(trace, loss_config)
    groups = {
        "response_recurrent": [p for name, p in policy.named_parameters()
                               if name.startswith(("response_encoder.", "response_memory."))],
        "control_head": list(policy.controller.parameters()),
    }
    gradients = {}
    for name, parameters in groups.items():
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        present = all(value is not None for value in values)
        finite = present and all(bool(torch.isfinite(value).all()) for value in values)
        norm = sum(float(value.detach().double().square().sum()) for value in values if value is not None) ** .5
        gradients[name] = {"finite": finite, "norm": norm}
    first_action_gradient = torch.autograd.grad(loss, captured[0], retain_graph=True)[0]
    obs = observation(initial, torch.zeros_like(initial.position))
    changed_truth = replace(
        initial, mass=initial.mass * 2, inertia_x=initial.inertia_x * 3,
        motor_time_rising=initial.motor_time_rising * 2,
        motor=torch.ones_like(initial.motor), external_force=initial.external_force + 99,
    )
    changed_obs = observation(changed_truth, torch.zeros_like(initial.position))
    with torch.no_grad():
        no_leak = torch.equal(obs, changed_obs) and torch.equal(policy(obs).action, policy(changed_obs).action)
    forbidden = {"load_q2_policy", "_reference_rollout", "collect_dagger_episode",
                 "dagger_window_loss", "fit_contextual_gain_local"}
    calls = set().union(*(inspect_calls(fn) for fn in (
        train, rollout, task_loss, ResponseMotorPolicy.forward
    )))
    checks = {
        "no_q2_targets": not bool(calls & forbidden),
        "response_recurrent_gradient": gradients["response_recurrent"]["finite"] and gradients["response_recurrent"]["norm"] > 0,
        "control_head_gradient": gradients["control_head"]["finite"] and gradients["control_head"]["norm"] > 0,
        "no_privileged_actor_input": no_leak,
        "no_unused_prediction_head": not hasattr(policy, "response_predictor"),
        "startup_connected": bool(torch.isfinite(first_action_gradient).all()) and bool(first_action_gradient.any()),
    }
    report = {
        "schema": "actor-only-training-contract-v2", "source_sha256": source_hash(),
        "policy_config": asdict(config), "loss_config": asdict(loss_config),
        "model_seed": 7, "scenario_seed": TRAIN_SEED_BASE, "horizon": 16,
        "scenario_mode": scenario_mode,
        "device": device, "optimizer_updates": 0, "checks": checks, "gradient_groups": gradients,
        "passed": all(checks.values()) and bool(torch.isfinite(loss)),
        "scope": "short causal contract; full-H500 parity is a separate boundary-adjoint check",
        "final_seeds_consumed": [], "deployment_authorized": False,
    }
    report["evidence_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--scenario-mode", choices=("physical-fit", "fixed-airframe"), default="fixed-airframe")
    parser.add_argument("--output", type=Path, default=Path("reports/actor_only_simplification/contract.json"))
    parser.add_argument("--huber-delta", type=float, default=1.)
    args = parser.parse_args()
    report = run_contract(device=args.device, scenario_mode=args.scenario_mode, loss_config=TaskLossConfig(
        huber_delta=args.huber_delta,
    ))
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
