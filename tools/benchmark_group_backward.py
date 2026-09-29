#!/usr/bin/env python3
"""Bounded, read-only comparison of native one-pass probes and grouped VJPs.

No optimizer update, training service, checkpoint, or sampling definition is
modified. Reported time is rollout + gradient computation, excluding the
trainer's Adam and update-audit I/O. Use a matching GPU for CUDA measurements.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from env_raptor import RaptorSimulator
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import sample_training_scenarios, source_hash, atomic_json


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(policy, initial, horizon, decay, cap, backend, chunk):
    device = initial.mass.device
    policy.zero_grad(set_to_none=True)
    gc.collect()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    config = GroupBalanceConfig(max_groups=128, min_scenarios=1, layout="coverage128",
                                 clip_norm=cap, backward_backend=backend, vjp_chunk_size=chunk)
    simulator, loss = RaptorSimulator(), TaskLossConfig()
    start = time.perf_counter()
    record = collect_rollout(policy, simulator, initial, loss, horizon=horizon,
                              time_decay=decay, group_config=config)
    synchronize(device)
    forward_end = time.perf_counter()
    report = backward_actor(policy, simulator, record, loss)["group_gradient"]
    synchronize(device)
    end = time.perf_counter()
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    gradient = torch.cat([p.grad.detach().flatten() for p in policy.parameters()]).cpu()
    result = {
        "backend": backend,
        "forward_seconds": forward_end-start,
        "backward_seconds": end-forward_end,
        "total_seconds": end-start,
        "cuda_peak_bytes": peak,
        "group_count": report["group_count"],
        "graph_traversals": report["graph_traversals"],
        "max_clip_multiplier": float(report["values"][:, 2].max()),
    }
    costs, norms = record.costs.detach().cpu().clone(), report["values"].detach().cpu().clone()
    del record, report
    return result, gradient, costs, norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--scenes", type=int, default=2048, help="TOTAL scenes, multiple of 128")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-decay", type=float, default=1.)
    parser.add_argument("--clip-norm", type=float, default=1.)
    parser.add_argument("--vjp-chunk-size", type=int, default=16)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (args.scenes < 128 or args.scenes % 128 or args.horizon < 1
            or args.repeats < 1 or args.warmup < 0):
        parser.error("positive horizon/repeats, nonnegative warmup, scenes divisible by 128 required")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no silent CPU substitution")
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    initial, seeds = sample_training_scenarios(args.scenes//4, 0, horizon=args.horizon,
                                               sampling="coverage128", device=device, dtype=dtype)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=args.memory_dim)).to(device=device,dtype=dtype)
    samples = {"vjp": [], "probe": []}
    outputs = {}
    for index in range(args.warmup+args.repeats):
        # Alternate order to avoid giving one backend every warm-cache position.
        for backend in (("vjp", "probe") if index % 2 == 0 else ("probe", "vjp")):
            result, gradient, costs, norms = measure(policy, initial, args.horizon,
                                                     args.time_decay, args.clip_norm,
                                                     backend, args.vjp_chunk_size)
            print(json.dumps({"iteration": index, "warmup": index < args.warmup, **result}),
                  file=sys.stderr, flush=True)
            if index >= args.warmup:
                samples[backend].append(result)
            outputs[backend] = gradient,costs,norms
    reference, actual = outputs["vjp"], outputs["probe"]
    tol = dict(rtol=5e-4, atol=3e-5) if dtype == torch.float32 else dict(rtol=3e-9, atol=2e-10)
    checks = {
        "forward_costs_bitwise_equal": torch.equal(reference[1],actual[1]),
        "clipped_gradient_close": torch.allclose(reference[0],actual[0],**tol),
        "group_norms_multipliers_close": torch.allclose(reference[2],actual[2],**tol),
        "gradient_max_absolute_difference": float((actual[0]-reference[0]).abs().max()),
        "gradient_relative_l2_difference": float((actual[0].double()-reference[0].double()).norm()
                                                 / reference[0].double().norm().clamp_min(1e-30)),
        "tolerance":tol,
    }
    medians = {b: {k: statistics.median(s[k] for s in records)
                   for k in ("forward_seconds","backward_seconds","total_seconds")}
               for b,records in samples.items()}
    result = {
        "source_sha256": source_hash(), "torch_version": str(torch.__version__),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
        "dtype": args.dtype, "scenes":args.scenes, "horizon":args.horizon,
        "policy_config":asdict(policy.config), "train_seeds":seeds,
        "time_decay":args.time_decay, "group_cap":args.clip_norm, "gradient_scale":.1,
        "samples":samples, "medians":medians, "checks":checks,
        "gradient_phase_speedup":medians["vjp"]["total_seconds"]/medians["probe"]["total_seconds"],
        "scope":"No Adam update or audit I/O; not learned performance or cross-hardware speed prediction.",
    }
    if args.report:
        atomic_json(args.report,result)
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if all(checks[k] for k in ("forward_costs_bitwise_equal",
                  "clipped_gradient_close","group_norms_multipliers_close")) else 1)


if __name__ == "__main__":
    main()
