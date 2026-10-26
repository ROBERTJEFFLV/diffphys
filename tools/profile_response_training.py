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
from env_raptor import RaptorParams, RaptorSimulator, rotation_backend_contract
from compiled_physics import make_simulator, physics_backend_contract
from compiled_execution import compile_actor, execution_contract
from tensor_storage import saved_tensor_storage
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig
from response_policy import ARCHITECTURE, ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig
from pass_stop import (ARCHITECTURE as GOAL_ARCHITECTURE, GoalConfig, PassStopLossConfig,
                       GoalMotorPolicy, import_motor_weights)
from response_training import sample_training_scenarios, model_hash, source_hash, atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--scenes', type=int, default=2048, help='TOTAL scenes, multiple of 128')
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--native-reference', action='store_true', help='profiling only: original eager path')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--mission-config', type=Path)
    parser.add_argument('--time-decay', type=float)
    parser.add_argument('--profile', action='store_true', help='one extra read-only operator profile')
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
    if args.mission_config:
        mission = GoalConfig(**json.loads(args.mission_config.read_text()))
        loss = PassStopLossConfig(args.epsilon_p, args.epsilon_a, args.lambda_R, mission=asdict(mission))
    time_decay = args.time_decay if args.time_decay is not None else (0. if args.mission_config else 1.)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    if saved is not None and saved.get('architecture') not in (ARCHITECTURE, GOAL_ARCHITECTURE):
        raise ValueError('profiling requires a native hidden-only Actor checkpoint')
    if saved is not None and saved['architecture'] == GOAL_ARCHITECTURE and not args.mission_config:
        raise ValueError('goal Actor profiling requires --mission-config')
    config = ResponsePolicyConfig(**saved['policy_config']) if saved is not None else ResponsePolicyConfig()
    policy = (GoalMotorPolicy(config) if args.mission_config else ResponseMotorPolicy(config)).to(args.device)
    if saved is not None:
        reference = (GoalMotorPolicy if saved['architecture'] == GOAL_ARCHITECTURE
                     else ResponseMotorPolicy)(config)
        reference.load_state_dict(saved['model'], strict=True)
        if model_hash(reference) != saved.get('model_sha256'):
            raise ValueError('checkpoint model digest mismatch')
        import_motor_weights(policy, reference.state_dict())
        del reference
    original = model_hash(policy)
    simulator = (RaptorSimulator() if args.native_reference
                 else make_simulator(RaptorParams(dt=config.dt), args.device))
    if not args.native_reference:
        compile_actor(policy, args.device)
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
        with saved_tensor_storage(args.device) as storage:
            record = collect_rollout(policy, simulator, initial, loss, horizon=args.horizon,
                                      time_decay=time_decay, group_config=groups)
        synchronize();middle = time.perf_counter()
        backward_actor(policy, simulator, record, loss)
        synchronize();end = time.perf_counter()
        row = {'forward_seconds':middle-start, 'backward_seconds':end-middle,
               'total_seconds':end-start, 'physical_transitions':record.metrics['physical_transitions'],
               'physical_transitions_per_second':record.metrics['physical_transitions']/(end-start),
               'peak_bytes':torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
               'task_objective':record.metrics['task_objective'], 'tensor_storage':dict(storage)}
        (warmups if index < args.warmup else samples).append(row)
        print(json.dumps({'warmup':index < args.warmup, **row}), file=sys.stderr, flush=True)
        del record
    operator_table = None
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profiler:
            with saved_tensor_storage(args.device):
                record = collect_rollout(policy, simulator, initial, loss, horizon=args.horizon,
                                          time_decay=time_decay, group_config=groups)
            backward_actor(policy, simulator, record, loss)
            synchronize()
            del record
        operator_table = profiler.key_averages().table(
            sort_by='self_cuda_time_total' if args.device == 'cuda' else 'self_cpu_time_total', row_limit=30)
    if model_hash(policy) != original:
        raise RuntimeError('read-only profiler changed Actor parameters')
    report = {'source_sha256':source_hash(), 'model_sha256':original, 'model_unchanged':True,
              'checkpoint_source_sha256':saved.get('binding', {}).get('source_sha256') if saved else None,
              'device':args.device, 'device_name':torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU',
              'torch_version':str(torch.__version__), 'dtype':'float32', 'policy_config':asdict(config),
              'scenes':args.scenes, 'horizon':args.horizon, 'train_seeds':seeds,
              'loss_config':asdict(loss), 'time_decay':time_decay, 'gradient_scale':.1,
              'actor_input_size':policy.response_memory.input_size, 'operator_table':operator_table,
              'numerical_backend':(rotation_backend_contract('eager') if args.native_reference
                                   else physics_backend_contract(args.device)),
              'execution':{'tensor_execution':'native-reference'} if args.native_reference else execution_contract(args.device, goal_task=bool(args.mission_config)),
              'sampling_seconds':sampling, 'warmups':warmups, 'samples':samples,
              'median_seconds':statistics.median(r['total_seconds'] for r in samples),
              'scope':'Frozen Actor float32 forward/backward only; no Adam, audit I/O, learning or resume claim.'}
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
