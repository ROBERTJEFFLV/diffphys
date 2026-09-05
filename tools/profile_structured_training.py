"""Measure production A1 collection/backprop on training scenarios only."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy
from env_l2f import L2FParams, L2FSimulator
from structured_distillation import (build_dagger_scenario_bank, collect_dagger_episode,
    dagger_window_loss, phase_a_equilibrium_gate, set_distillation_phase)
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy
from tools.pretrain_structured_identifier import DEFAULT_Q2, _move_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = StructuredPolicyConfig(motor_observer_bank_size=35,
        motor_observer_mode="fixed_multi_tau_v2", motor_tau_grid_version=2,
        allocator_solver="box_qp", allocator_rate_limit=50.0)
    result = {"purpose": "training-only production profiling", "formal_eligible": False,
              "training_seed": 7, "config": asdict(config), "measurements": []}
    for name in (("cuda", "cpu") if args.device == "both" else (args.device,)):
        device = torch.device(name)
        torch.manual_seed(7)
        teacher, _ = load_q2_policy(DEFAULT_Q2, device=device, dtype=torch.float32)
        teacher.requires_grad_(False).eval()
        policy = StructuredRecurrentPolicy(config).to(device)
        set_distillation_phase(policy, "A1")
        optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),
                                      lr=3e-4, weight_decay=1e-5)
        simulator = L2FSimulator(L2FParams())
        for repeat in range(args.repeats):
            started = time.perf_counter()
            bank = _move_bank(build_dagger_scenario_bank(64, seed=107 + repeat), device)
            bank_seconds = time.perf_counter() - started
            if name == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            episode = collect_dagger_episode(teacher, policy, simulator, bank,
                beta=1.0, horizon=126, teacher_probe=True, episode_seed=1007 + repeat)
            if name == "cuda":
                torch.cuda.synchronize()
            collect_seconds = time.perf_counter() - started
            started = time.perf_counter()
            loss, components = dagger_window_loss(policy, episode, prefix=25,
                phase="A1", capability_weight=0.0, capability_mean_weight=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("profiling encountered a non-finite gradient")
            optimizer.step()
            if name == "cuda":
                torch.cuda.synchronize()
            update_seconds = time.perf_counter() - started
            gate, _ = phase_a_equilibrium_gate(episode, require_identification_width=False)
            row = dict(device=name, repeat=repeat, scenario_seed=107 + repeat,
                scenarios=64, horizon=126, bank_seconds=bank_seconds,
                collect_seconds=collect_seconds, update_seconds=update_seconds,
                loss=float(loss), components=components, gradient_norm=float(norm),
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if name == "cuda" else None,
                peak_cuda_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if name == "cuda" else None,
                initial_training_bank_gate=gate)
            result["measurements"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "initial_training_bank_gate"}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            del loss, episode, bank
        del policy, teacher, optimizer
        if name == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
