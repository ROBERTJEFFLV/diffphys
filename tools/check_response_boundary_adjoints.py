"""Bounded saved-Actor oracle check and optional frozen-Critic/Adam diagnostics.

Reads historical weights explicitly, never resumes their Critic optimizer/schema,
never retains an Actor update, and never consumes FINAL. Run only when requested.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import response_training as common
from env_l2f import L2FParams, L2FSimulator, L2FState
from response_adjoints import STATE_SCALES
from response_phase1 import parameter_gradient_metrics
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, initialize, rollout, sample_scenarios
from response_value import (TaskValueConfig, TaskValueCritic, TaskValueTrainer,
    collect_task_trajectory, collect_boundary_adjoints, collect_derivative_samples,
    task_gradient_vector, audit_actor_gradient, task_metrics)
from tools.diagnose_response_update import exact_gradient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--fits', type=int, nargs='+', default=[1, 6])
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--chunk-size', type=int, default=64,
                        help='use the original 64-scene batch; smaller chunks may alter CUDA numerics')
    parser.add_argument('--frozen-fits', type=int, default=0,
                        help='paired memory/full supervision on the first saved Actor only')
    parser.add_argument('--shadow-steps', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=300)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.chunk_size < 1 or args.frozen_fits < 0 or args.max_seconds <= 0:
        parser.error('invalid diagnostic budget')
    torch.set_num_threads(1)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'source_sha256': common.source_hash(),
              'device': args.device, 'torch': str(torch.__version__), 'dtype': 'float32',
              'tool_sha256': common.file_hash(Path(__file__)), 'full_bptt_chunk_size': args.chunk_size,
              'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
              'actor_updates_committed': 0, 'fits': []}

    def save():
        result['elapsed_seconds'] = time.monotonic()-started
        common.atomic_json(args.output/'results.json', result)

    def budget():
        if time.monotonic()-started >= args.max_seconds:
            raise TimeoutError('bounded diagnostic time budget exhausted')

    try:
        for fit in args.fits:
            budget()
            before = torch.load(args.snapshots/('%04d.pt' % (fit-1)), map_location='cpu')
            after = torch.load(args.snapshots/('%04d.pt' % fit), map_location='cpu')
            binding = before['binding']; horizon = binding['horizon']; window = binding['critic']['window_steps']
            policy = ResponseMotorPolicy(ResponsePolicyConfig(**before['policy_config'])).to(args.device)
            policy.load_state_dict(before['model']); actor_hash = common.model_hash(policy)
            initial, seeds = common.sample_training_scenarios(binding['protocol']['scenarios_per_bank'], fit-1,
                batches=2, dt=policy.config.dt, device=torch.device(args.device), dtype=torch.float32,
                scenario_mode='fixed-airframe')
            if seeds != after['progress']['history'][fit-1]['scenario_seeds']:
                raise ValueError('saved scenario seeds do not match the requested update')
            sim = L2FSimulator(L2FParams(dt=policy.config.dt))
            loss = TaskLossConfig(**binding['training_loss'])
            record = collect_task_trajectory(policy, sim, initial, horizon, window, loss)
            target = TaskValueCritic(record.inputs.shape[-1]).to(args.device).requires_grad_(False)
            target.load_state_dict(after['critic_training']['target'])  # weights for diagnosis, not resume
            tick = time.monotonic()
            adjoints = collect_boundary_adjoints(policy, sim, record, horizon, window, loss)
            row = {'fit': fit, 'actor_sha256': actor_hash, 'seeds': seeds, 'scenarios': len(record.weights),
                   'adjoint_seconds': time.monotonic()-tick,
                   'boundary_exact': sum(r['exact'] for r in adjoints.continuity),
                   'boundary_continuity': adjoints.continuity, 'variants': {}}
            result['fits'].append(row); save()
            budget()
            if args.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            full, evidence = exact_gradient(policy, sim, initial, horizon, loss, chunk_size=args.chunk_size)
            evidence['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if args.device == 'cuda' else None
            reference = full/(horizon//window)
            row['full_bptt'] = evidence
            gradients = {}
            for mode in ('oracle_full_state', 'critic', 'none'):
                budget()
                gradient = task_gradient_vector(policy, target, sim, initial, horizon, window, loss,
                    record, terminal_mode=mode, adjoints=adjoints)
                gradients[mode] = gradient
                row['variants'][mode] = parameter_gradient_metrics(gradient, reference)
                save()
            if row['variants']['oracle_full_state']['relative_error'] > 1e-4:
                raise RuntimeError('full boundary adjoint differs from full BPTT beyond 1e-4')
            torch.save({'full': full.cpu(), 'variants': {n:g.cpu() for n,g in gradients.items()}},
                        args.output/('fit_%02d_gradients.pt' % fit))
            print(json.dumps({'fit':fit, 'variants':row['variants']}), flush=True)

            if args.shadow_steps:
                banks = [sample_scenarios(binding['protocol']['scenarios_per_bank'], seed=seed,
                    dt=policy.config.dt, device=torch.device(args.device), scenario_mode='fixed-airframe')[0]
                    for seed in binding['evaluation_seeds']]
                from dataclasses import fields
                evaluation = L2FState(**{f.name:torch.cat([getattr(b,f.name) for b in banks]) for f in fields(banks[0])})
                with torch.no_grad():
                    row['baseline'] = {n:task_metrics(rollout(policy,sim,state,horizon),loss)
                                       for n,state in (('train',initial),('eval',evaluation))}
                    for m in row['baseline'].values(): m.pop('scenarios',None)
                row['shadow_steps'] = {}
                for mode, gradient in gradients.items():
                    budget()
                    shadow = copy.deepcopy(policy)
                    optimizer = torch.optim.Adam(shadow.parameters(), lr=binding['lr'])
                    optimizer.load_state_dict(copy.deepcopy(before['optimizer']))
                    offset = 0
                    for parameter in shadow.parameters():
                        parameter.grad = gradient[offset:offset+parameter.numel()].reshape_as(parameter).clone()
                        offset += parameter.numel()
                    norm = torch.nn.utils.clip_grad_norm_(shadow.parameters(),binding['gradient_clip'],error_if_nonfinite=True)
                    theta = torch.nn.utils.parameters_to_vector(shadow.parameters()).detach().clone()
                    optimizer.step()
                    delta = torch.nn.utils.parameters_to_vector(shadow.parameters()).detach()-theta
                    metrics = {'clip_input_norm': float(norm), 'step_l2': float(delta.double().norm()),
                               'oracle_gradient_dot_step': float(torch.dot(full.double(),delta.double()))}
                    with torch.no_grad():
                        for name, state in (('train',initial),('eval',evaluation)):
                            metrics[name] = task_metrics(rollout(shadow,sim,state,horizon),loss)
                            metrics[name].pop('scenarios',None)
                    row['shadow_steps'][mode] = metrics; save()

            if args.frozen_fits and fit == args.fits[0]:
                row['frozen_comparison'] = {}
                # Identical split, initial scalar MLP, minibatch RNG, and fresh Adam.
                torch.manual_seed(7)
                cfg = TaskValueConfig(window_steps=window)
                train, held = collect_derivative_samples(policy,sim,record,horizon,loss,cfg,adjoints=adjoints)
                for name, groups in (('memory',('policy.memory',)), ('full',tuple(STATE_SCALES))):
                    budget()
                    trainer = TaskValueTrainer(policy,initialize(policy,initial),horizon,replace(cfg,derivative_state_groups=groups))
                    trainer.critic.load_state_dict(before['critic_training']['critic'])
                    trainer.target.load_state_dict(before['critic_training']['critic'])
                    pools = [replace(p,state_groups=groups,gradients={n:p.gradients[n] for n in groups}) for p in (train,held)]
                    torch.manual_seed(711)
                    rounds = []
                    for _ in range(args.frozen_fits):
                        budget()
                        fitted = trainer.fit(record,*pools)
                        audit, _ = audit_actor_gradient(policy,trainer.target,sim,initial,horizon,window,
                                                loss,record,adjoints,reference=reference)
                        fitted['target_actor_gradient'] = audit
                        rounds.append(fitted)
                        row['frozen_comparison'][name] = rounds; save()
                    torch.save(trainer.state_dict(),args.output/('fit_%02d_%s_critic.pt' % (fit,name)))
                row['frozen_comparison_scope'] = 'same fixed Actor/data; fresh Critic Adam; old initial MLP weights only; no Actor update'
            if common.model_hash(policy) != actor_hash:
                raise RuntimeError('diagnostic mutated the saved Actor')
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='stopped', error=repr(error))
        save()
        raise
    save()
    return result


if __name__ == '__main__':
    main()
