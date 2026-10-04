#!/usr/bin/env python3
"""Read-only finite-perturbation audit of the complete geometric/GRU motor loop.

No Time Decay, optimizer update, loss change or checkpoint selection. Sampled
finite differences are not a proof, and the weights below define the metric.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from env_raptor import RaptorSimulator
from response_noise import DisturbanceConfig
from response_policy import ResponsePolicyState
from response_task import ResponseClosedLoopState, initialize, observation
from response_training import load_policy_checkpoint, atomic_json, model_hash

# Each component is divided by its scale before the joint Euclidean norm.
# Attitude uses an SO(3) chordal distance, locally equivalent to radians.
STATE_SCALES = {"position": 1., "velocity": 3., "attitude": 1., "omega": 10.,
                "motor": 1., "previous_action": 1., "previous_velocity": 3., "memory": 1.}


def state_distance(a, b):
    squares = []
    for name, scale in STATE_SCALES.items():
        if name == "attitude":
            d = a.physical.rotation-b.physical.rotation
            squares.append(d.square().sum((-1, -2))/(2*scale*scale))
        else:
            aa = a.policy.memory if name == "memory" else getattr(a.physical, name)
            bb = b.policy.memory if name == "memory" else getattr(b.physical, name)
            squares.append(((aa-bb)/scale).flatten(1).square().sum(-1))
    return torch.stack(squares).sum(0).sqrt()


def perturb(closed, vector):
    values, offset = {}, 0
    for name, scale in STATE_SCALES.items():
        if name == "attitude":
            delta = vector[:, offset:offset+3]*scale
            angle = delta.norm(dim=-1, keepdim=True)
            dw = torch.cos(angle/2)
            dv = .5*torch.sinc(angle/(2*math.pi))*delta
            q = closed.physical.orientation
            qw, qv = q[:, :1], q[:, 1:]
            values["orientation"] = F.normalize(torch.cat((qw*dw-(qv*dv).sum(-1, keepdim=True),
                qw*dv+dw*qv+torch.linalg.cross(qv,dv,dim=-1)), -1), dim=-1)
            offset += 3
            continue
        source = closed.policy.memory if name == "memory" else getattr(closed.physical, name)
        width = source[0].numel()
        value = source + vector[:, offset:offset+width].reshape_as(source)*scale
        if name == "motor": value = value.clamp(0, 1)
        if name in ("previous_action", "memory"): value = value.clamp(-1, 1)
        values[name] = value
        offset += width
    memory = values.pop("memory")
    return ResponseClosedLoopState(replace(closed.physical, **values), ResponsePolicyState(memory))


@torch.no_grad()
def audit_policy(policy, simulator, initial, *, steps=25, directions=4, epsilon=1e-4,
                 seed=1007, base_only=False, warmup=0):
    if steps < 1 or directions < 1 or warmup < 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive steps/directions/epsilon and nonnegative warmup required")
    n = initial.position.shape[0]
    if n < 1: raise ValueError("audit needs at least one scene")
    closed = initialize(policy, initial)
    def advance(state):
        parts = policy.components(observation(state.physical), state.policy)
        if base_only:
            from response_policy import channels_to_motors
            action = torch.tanh(channels_to_motors(parts.base_command))
        else:
            action = torch.tanh(parts.motor_logits)
        return ResponseClosedLoopState(simulator.step(state.physical, action),
                                       ResponsePolicyState(parts.memory)), action
    for _ in range(warmup): closed, _ = advance(closed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    results = []
    for direction in range(directions):
        vector = torch.randn(n, 29+policy.config.memory_dim, generator=generator,
                             dtype=initial.position.dtype).to(initial.position.device)
        vector = F.normalize(vector, dim=-1)*epsilon
        a, b = perturb(closed, vector), perturb(closed, -vector)
        initial_distance = state_distance(a, b)
        if not bool(torch.isfinite(initial_distance).all()) or bool((initial_distance <= 0).any()):
            raise ValueError("perturbation is unresolved in this dtype; increase epsilon or use float64")
        peak = torch.ones_like(initial_distance)
        action_peak = torch.zeros_like(peak)
        boundary = simulator.terminated(a.physical) | simulator.terminated(b.physical)
        finite = True
        for step in range(steps):
            a, ua = advance(a); b, ub = advance(b)
            gain = state_distance(a, b)/initial_distance
            action_gain = (ua-ub).norm(dim=-1)/initial_distance
            finite = bool(torch.isfinite(gain).all() & torch.isfinite(action_gain).all())
            if not finite: break
            peak = torch.maximum(peak, gain)
            action_peak = torch.maximum(action_peak, action_gain)
            boundary |= simulator.terminated(a.physical) | simulator.terminated(b.physical)
        results.append({"direction": direction, "finite": finite, "completed_steps": step+1,
                        "terminal_state_gain": gain.cpu().tolist() if finite else None,
                        "peak_state_gain_including_initial": peak.cpu().tolist() if finite else None,
                        "peak_motor_command_gain": action_peak.cpu().tolist() if finite else None,
                        "outside_task_domain": boundary.cpu().tolist()})
    return {"method": "paired_forward_finite_perturbations", "certified": False,
            "deployment_authorized": False, "time_decay_used": False,
            "mode": "base_only" if base_only else "full_actor", "state_scales": STATE_SCALES,
            "epsilon": epsilon, "directions": directions, "scenes": n, "steps": steps,
            "warmup": warmup, "seed": seed, "actor_sha256": model_hash(policy),
            "termination_handling": "reported, not frozen; post-boundary gains are out-of-task diagnostics",
            "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args(argv)
    if args.scenarios < 1: parser.error("scenarios must be positive")
    torch.set_num_threads(1)
    policy, saved = load_policy_checkpoint(args.checkpoint, args.device, torch.float64)
    disturbances = DisturbanceConfig(**saved["binding"]["protocol"]["disturbances"])
    sim = RaptorSimulator()
    initial = sim.reset(args.scenarios, device=args.device, dtype=torch.float64,
                        seed=args.seed, horizon=args.steps+args.warmup, disturbances=disturbances)
    report = audit_policy(policy, sim, initial, steps=args.steps, directions=args.directions,
                          epsilon=args.epsilon, seed=args.seed, base_only=args.base_only, warmup=args.warmup)
    atomic_json(args.output, report)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
