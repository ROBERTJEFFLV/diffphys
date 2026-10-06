#!/usr/bin/env python3
"""Read-only synchronized production rollout/backward timing on fixed TRAIN tapes.

Supports mature hidden-only B checkpoints through an explicitly weights-only
in-memory import. Does not run Adam, save a training checkpoint, or alter inputs.
Warm-up/compile cost is separate from timed samples. Peak memory is PyTorch
allocation, excluding the display, driver and other processes.
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
from env_raptor import RaptorSimulator, rotation_backend_contract
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig
from response_policy import ARCHITECTURE, ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import sample_training_scenarios, model_hash, source_hash, atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--scenes', type=int, default=2048, help='TOTAL scenes, multiple of 128')
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--rotation-backend', choices=('eager', 'compile'), default='eager')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--epsilon-p', type=float, required=True)
    parser.add_argument('--epsilon-a', type=float, required=True)
    parser.add_argument('--lambda-R', type=float, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.scenes < 128 or args.scenes % 128 or args.horizon < 1
            or args.warmup < 0 or args.repeats < 1):
        parser.error('scenes divisible by 128, positive horizon/repeats, nonnegative warmup required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; no silent CPU replacement')
    if args.checkpoint and args.report.resolve() == args.checkpoint.resolve():
        parser.error('report cannot overwrite the checkpoint')
    loss = TaskLossConfig(args.epsilon_p, args.epsilon_a, args.lambda_R)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    if saved is not None and saved.get('architecture') != ARCHITECTURE:
        raise ValueError('profiling requires a native hidden-only B checkpoint; no architecture conversion')
    config = ResponsePolicyConfig(**saved['policy_config']) if saved is not None else ResponsePolicyConfig()
    policy = ResponseMotorPolicy(config).to(args.device)
    if saved is not None:
        policy.load_state_dict(saved['model'], strict=True)
        if model_hash(policy) != saved.get('model_sha256'):
            raise ValueError('checkpoint model digest mismatch')
    original = model_hash(policy)
    simulator = RaptorSimulator(rotation_backend=args.rotation_backend)
    groups = GroupBalanceConfig(max_groups=128, min_scenarios=args.scenes//128, layout='coverage128')
    def synchronize():
        if args.device == 'cuda':
            torch.cuda.synchronize()
    start = time.perf_counter()
    initial, seeds = sample_training_scenarios(args.scenes//4, 0, horizon=args.horizon,
                                               sampling='coverage128', device=args.device)
    synchronize()
    sampling = time.perf_counter()-start
    samples, warmups = [], []
    for index in range(args.warmup+args.repeats):
        policy.zero_grad(set_to_none=True)
        gc.collect();synchronize()
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        record = collect_rollout(policy, simulator, initial, loss, horizon=args.horizon,
                                  time_decay=1., group_config=groups)
        synchronize();middle = time.perf_counter()
        backward_actor(policy, simulator, record, loss)
        synchronize();end = time.perf_counter()
        row = {'forward_seconds':middle-start, 'backward_seconds':end-middle,
               'total_seconds':end-start, 'physical_transitions':record.metrics['physical_transitions'],
               'physical_transitions_per_second':record.metrics['physical_transitions']/(end-start),
               'peak_bytes':torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
               'task_objective':record.metrics['task_objective']}
        (warmups if index < args.warmup else samples).append(row)
        print(json.dumps({'warmup':index < args.warmup, **row}), file=sys.stderr, flush=True)
        del record
    if model_hash(policy) != original:
        raise RuntimeError('read-only profiler changed Actor parameters')
    report = {'source_sha256':source_hash(), 'model_sha256':original, 'model_unchanged':True,
              'checkpoint_source_sha256':saved.get('binding', {}).get('source_sha256') if saved else None,
              'device':args.device, 'device_name':torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU',
              'torch_version':str(torch.__version__), 'dtype':'float32', 'policy_config':asdict(config),
              'scenes':args.scenes, 'horizon':args.horizon, 'train_seeds':seeds,
              'loss_config':asdict(loss), 'time_decay':1., 'gradient_scale':.1,
              'numerical_backend':rotation_backend_contract(args.rotation_backend),
              'sampling_seconds':sampling, 'warmups':warmups, 'samples':samples,
              'median_seconds':statistics.median(r['total_seconds'] for r in samples),
              'scope':'Frozen Actor float32 forward/backward only; no Adam, audit I/O, learning or resume claim.'}
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
