#!/usr/bin/env python3
"""Read-only replay of a recorded Actor/Adam update (same source/backend required)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from response_audit import AUDIT_VERSION, parameter_changes
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_groups import GroupBalanceConfig
from response_task import TaskLossConfig
from response_noise import DisturbanceConfig
from env_raptor import RaptorSimulator, RaptorParams
from response_adjoints import collect_rollout, backward_actor
from response_training import (source_hash, migrate_actor_weights, migrate_named_adam,
                               restore_rng, capture_rng, sample_training_scenarios,
                               safe_global_clip, atomic_json)


def compare_values(actual, expected):
    """Report bitwise equality plus largest tensor difference; never forgive failure."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            return False, float('inf')
        a, b = actual.detach().cpu(), expected.detach().cpu()
        same = torch.equal(a, b)
        diff = float((a.double()-b.double()).abs().max()) if a.numel() else 0.
        return same, diff
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return False, float('inf')
        pairs = [compare_values(actual[k], v) for k,v in expected.items()]
    elif isinstance(expected, (list,tuple)):
        if not isinstance(actual, (list,tuple)) or len(actual) != len(expected):
            return False, float('inf')
        pairs = [compare_values(a,b) for a,b in zip(actual,expected)]
    else:
        return actual == expected, 0. if actual == expected else float('inf')
    return all(p[0] for p in pairs), max((p[1] for p in pairs), default=0.)


def replay(path, mode='full'):
    capsule = torch.load(path, map_location='cpu', weights_only=True)
    if capsule.get('version') != AUDIT_VERSION:
        raise ValueError('unsupported update capsule')
    if capsule.get('failure') is not None:
        raise ValueError('failed attempt retained for diagnosis, not an accepted-step replay')
    cfg = capsule['binding']
    if cfg['source_sha256'] != source_hash():
        raise ValueError('replay source mismatch; use the accompanying audit/source.zip')
    if str(torch.__version__) != cfg['torch_version']:
        raise ValueError('exact replay needs the saved PyTorch version')
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('saved CUDA update requires CUDA; no silent CPU substitution')
    dtype = getattr(torch, cfg['dtype'])
    torch.set_num_threads(cfg['threads'])
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    policy = ResponseMotorPolicy(ResponsePolicyConfig(**capsule['policy_config'])).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg['lr'])
    migrate_actor_weights(policy, capsule['before']['model'])
    migrate_named_adam(optimizer, policy, capsule['before']['optimizer'], capsule['optimizer_parameter_names'])
    restore_rng(capsule['before']['rng'])
    optimizer.zero_grad(set_to_none=True)
    if mode == 'full':
        sampling = {}
        initial, seeds = sample_training_scenarios(
            cfg['protocol']['scenarios_per_bank'], capsule['before']['sampling_index'],
            dt=policy.config.dt, device=device, dtype=dtype, horizon=cfg['horizon'],
            disturbances=DisturbanceConfig(**cfg['protocol']['disturbances']),
            sampling=cfg['training_sampling']['mode'], sampling_report=sampling)
        if {'seeds':seeds, 'report':sampling} != capsule['sampling']:
            raise ValueError('replayed TRAIN sampling does not match recorded batch')
        simulator = RaptorSimulator(RaptorParams(dt=policy.config.dt))
        loss = TaskLossConfig(**cfg['protocol']['loss'])
        group = GroupBalanceConfig(**{k:v for k,v in cfg['group_balance'].items() if k != 'version'})
        record = collect_rollout(policy, simulator, initial, loss, horizon=cfg['horizon'],
                                 time_decay=cfg['time_decay'], group_config=group)
        backward_actor(policy, simulator, record, loss, gradient_scale=cfg['gradient_scale'])
        safe_global_clip(policy.parameters(), cfg['gradient_clip'])
    elif mode == 'adam':
        for name, parameter in policy.named_parameters():
            grad = capsule['gradients_entering_adam'][name]
            parameter.grad = None if grad is None else grad.to(parameter).clone()
    else:
        raise ValueError('mode must be full or adam')
    gradients = {name:p.grad for name,p in policy.named_parameters()}
    gradient_match, gradient_error = compare_values(gradients, capsule['gradients_entering_adam'])
    optimizer.step()
    model_match, model_error = compare_values(policy.state_dict(), capsule['after']['model'])
    adam_match, adam_error = compare_values(optimizer.state_dict(), capsule['after']['optimizer'])
    rng_match, _ = compare_values(capture_rng(), capsule['after']['rng'])
    return {'update': capsule['update'], 'mode': mode,
            'gradient_equal': gradient_match, 'gradient_max_abs_difference': gradient_error,
            'model_equal': model_match, 'model_max_abs_difference': model_error,
            'adam_equal': adam_match, 'adam_max_abs_difference': adam_error,
            'rng_equal': rng_match,
            'passed': gradient_match and model_match and adam_match and rng_match,
            'parameter_changes': parameter_changes(policy, capsule['before']['model'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capsule', type=Path)
    parser.add_argument('--mode', choices=('full','adam'), default='full')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = replay(args.capsule, args.mode)
    if args.report:
        if args.report.resolve() == args.capsule.resolve():
            parser.error('report cannot overwrite the input capsule')
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
