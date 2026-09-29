#!/usr/bin/env python3
"""Read-only eager/compiled RK4 preflight on identical fixed TRAIN scenarios.

Reports cold compilation separately from warm rollout+backward timings. Never
runs Adam, writes checkpoints, changes a training run, or substitutes CPU for
CUDA. Run without profiling for latency numbers; use Nsight in a separate run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from env_raptor import RaptorParams
from response_acceleration import execution_contract, make_simulator
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import atomic_json, sample_training_scenarios, source_hash


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(policy, simulator, initial, horizon, decay, config):
    device = initial.mass.device
    policy.zero_grad(set_to_none=True)
    gc.collect()
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    loss = TaskLossConfig()
    start = time.perf_counter()
    record = collect_rollout(policy, simulator, initial, loss, horizon=horizon,
                             time_decay=decay, group_config=config)
    synchronize(device)
    forward_end = time.perf_counter()
    report = backward_actor(policy, simulator, record, loss)['group_gradient']
    synchronize(device)
    end = time.perf_counter()
    times = {
        'forward_seconds': forward_end - start,
        'backward_seconds': end - forward_end,
        'gradient_phase_seconds': end - start,
        'cuda_peak_allocated_bytes': (torch.cuda.max_memory_allocated(device)
                                      if device.type == 'cuda' else None),
        'physical_transitions': record.metrics['physical_transitions'],
    }
    # Small copies outside the timing window. Do not retain a flight graph.
    output = {
        'costs': record.costs.detach().cpu().clone(),
        'raw_groups': record.probe.rows.detach().cpu().clone(),
        'norms_and_multipliers': report['values'].detach().cpu().clone(),
        'clipped_gradient': torch.cat([p.grad.detach().flatten() for p in policy.parameters()]).cpu(),
    }
    return times, output


def compare(reference, actual, *, rtol, atol):
    checks = {}
    for name in reference:
        a, b = reference[name], actual[name]
        checks[name + '_close'] = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
        checks[name + '_max_abs_error'] = float((a.double() - b.double()).abs().max())
    a, b = reference['clipped_gradient'].double(), actual['clipped_gradient'].double()
    checks['gradient_relative_l2_error'] = float((a - b).norm() / a.norm().clamp_min(1e-30))
    checks['passed'] = all(v for k, v in checks.items() if k.endswith('_close'))
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--backends', nargs='+', choices=('eager', 'compile'), default=['eager', 'compile'])
    parser.add_argument('--scenes', type=int, default=2048, help='TOTAL scenes, not four-bank size')
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--memory-dim', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--time-decay', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=1.)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.scenes < 128 or args.scenes % 128 or args.horizon < 1 or args.memory_dim < 1
            or args.repeats < 1 or args.warmup < 1):
        parser.error('positive horizon/memory/repeats/warmup; scenes must be divisible by 128')
    if len(set(args.backends)) != len(args.backends):
        parser.error('backends must be unique')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; refusing silent CPU substitution')
    if not math.isfinite(args.time_decay) or args.time_decay < 0:
        parser.error('time-decay must be finite and nonnegative')
    if not math.isfinite(args.clip_norm) or args.clip_norm <= 0:
        parser.error('clip-norm must be finite and positive')
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    initial, seeds = sample_training_scenarios(args.scenes // 4, 0, sampling='coverage128',
                                               device=device, dtype=dtype, horizon=args.horizon)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=args.memory_dim)).to(device=device, dtype=dtype)
    config = GroupBalanceConfig(max_groups=128, min_scenarios=1, layout='coverage128',
                                clip_norm=args.clip_norm, backward_backend='probe')
    simulators = {name: make_simulator(RaptorParams(), name) for name in args.backends}
    tolerance = dict(rtol=5e-4, atol=3e-5) if dtype == torch.float32 else dict(rtol=3e-8, atol=2e-9)
    samples, checks = {name: [] for name in args.backends}, []
    for i in range(args.warmup + args.repeats):
        outputs, times = {}, {}
        order = args.backends if i % 2 == 0 else list(reversed(args.backends))
        for name in order:
            timing, output = measure(policy, simulators[name], initial, args.horizon,
                                      args.time_decay, config)
            timing.update(iteration=i, warmup=i < args.warmup)
            samples[name].append(timing)
            outputs[name], times[name] = output, timing
            print(json.dumps({'backend': name, **timing}), file=sys.stderr, flush=True)
        if len(outputs) == 2:
            check = compare(outputs['eager'], outputs['compile'], **tolerance)
            check['physical_transitions_equal'] = (
                times['eager']['physical_transitions'] == times['compile']['physical_transitions'])
            check['passed'] = check['passed'] and check['physical_transitions_equal']
            checks.append({'iteration': i, **check})
    medians = {
        name: {key: statistics.median(row[key] for row in rows if not row['warmup'])
               for key in ('forward_seconds', 'backward_seconds', 'gradient_phase_seconds')}
        for name, rows in samples.items()
    }
    result = {
        'source_sha256': source_hash(), 'torch_version': str(torch.__version__),
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'dtype': args.dtype, 'scenes': args.scenes, 'horizon': args.horizon,
        'policy_config': asdict(policy.config), 'train_seeds': seeds,
        'time_decay': args.time_decay, 'group_cap': args.clip_norm, 'gradient_scale': .1,
        'execution': {name: execution_contract(name) for name in args.backends},
        'samples': samples, 'warm_medians': medians, 'tolerance': tolerance, 'checks': checks,
        'passed': all(c['passed'] for c in checks) if checks else None,
        'scope': 'Numerical gradient-phase preflight only. Excludes sampling, Adam and audit I/O; not training convergence or deployment safety.',
    }
    if len(medians) == 2:
        result['warm_gradient_phase_speedup'] = (
            medians['eager']['gradient_phase_seconds'] / medians['compile']['gradient_phase_seconds'])
    atomic_json(args.report, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result['passed'] is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
