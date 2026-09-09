"""Phase 1 task Actor-Critic orchestration: persistent Adam, observation-only EVAL."""
from __future__ import annotations

from dataclasses import asdict, fields
import copy
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from env_l2f import L2FParams, L2FSimulator, L2FState
from response_policy import ResponseMotorPolicy
from response_task import initialize, rollout, sample_scenarios
from response_value import TASK_VALUE_OBJECTIVE, TaskValueTrainer, task_metrics
import response_training as common


def task_binding(args, policy_config, loss_config):
    return {'source_sha256': common.source_hash(), 'optimizer': 'task-adam',
            'protocol': common.protocol(policy_config, loss_config, args.scenarios, args.scenario_mode),
            'training_loss': asdict(loss_config), 'horizon': args.horizon,
            'critic': {'objective': TASK_VALUE_OBJECTIVE, **asdict(common.critic_configuration(args))},
            'lr': args.lr, 'weight_decay': 0., 'gradient_clip': args.gradient_clip,
            'device': args.device, 'dtype': args.dtype, 'torch_version': str(torch.__version__),
            'training_sampling': {'batches': 2, 'scenarios': 2 * args.scenarios, 'pooled_cvar': True},
            'evaluation_every': args.development_every, 'evaluation_seeds': list(common.DEVELOPMENT_SEEDS),
            'evaluation_role': 'observation_and_best_checkpoint_only', 'phase1_probes': args.phase1_probes,
            'contract_sha256': None if args.contract_report is None else common.file_hash(args.contract_report)}


def _setup(args, policy_config):
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    policy = ResponseMotorPolicy(policy_config).to(device=device, dtype=dtype)
    if args.initialize_from is not None:
        old, _ = common.load_policy_checkpoint(args.initialize_from, device, dtype)
        if old.config != policy_config:
            raise ValueError('weights-only initialization requires the same Actor architecture')
        policy.load_state_dict(old.state_dict())
    initial, _ = common.sample_training_scenarios(args.scenarios, 0, batches=2,
        dt=policy_config.dt, device=device, dtype=dtype, scenario_mode='fixed-airframe')
    trainer = TaskValueTrainer(policy, initialize(policy, initial), args.horizon, common.critic_configuration(args))
    return policy, L2FSimulator(L2FParams(dt=policy_config.dt)), trainer, common.actor_optimizer(args, policy)


def _evaluation_initials(args, policy):
    ref = next(policy.parameters())
    banks = [sample_scenarios(args.scenarios, seed=seed, dt=policy.config.dt,
             device=ref.device, dtype=ref.dtype, scenario_mode='fixed-airframe')[0]
             for seed in common.DEVELOPMENT_SEEDS]
    return L2FState(**{f.name: torch.cat([getattr(bank, f.name) for bank in banks]) for f in fields(banks[0])})


@torch.no_grad()
def evaluate_task_policy(policy, simulator, initial, horizon, config):
    """One pooled continuous EVAL with precisely the same task/CVaR definition."""
    return task_metrics(rollout(policy, simulator, initial, horizon), config)


def _synchronize(policy):
    if next(policy.parameters()).is_cuda:
        torch.cuda.synchronize(next(policy.parameters()).device)


def train_task_value(args, policy_config, loss_config):
    if args.updates is None or not 1 <= args.updates < 500_000:
        raise ValueError('specify a positive task-Adam update budget below the reserved seed range')
    if args.scenario_mode != 'fixed-airframe' or loss_config.prediction_weight != 0:
        raise ValueError('task-Adam Phase 1 uses nominal dynamics and no auxiliary objective')
    if common.FINAL_CLAIM.exists():
        raise RuntimeError('FINAL consumed; register a new protocol before training')
    if args.q2_checkpoint is not None:
        raise ValueError('Q2 cannot enter task-value training')
    if args.updates > 5:
        common.validate_training_contract(args.contract_report, policy_config, loss_config)
    work = args.work_dir
    latest = work/'latest.training.pt'
    if args.initialize_from is not None and latest.exists():
        raise ValueError('weights-only initialization needs a new experiment directory')
    policy, simulator, trainer, optimizer = _setup(args, policy_config)
    run_binding = task_binding(args, policy_config, loss_config)
    initial_model = copy.deepcopy(policy.state_dict())
    progress = {'attempts': 0, 'updates': 0, 'elapsed_seconds': 0., 'status': 'training',
                'history': [], 'evaluation': [], 'training_seeds': [], 'best_score': None, 'best_update': None,
                'best_success_rate': None, 'best_success_score': None, 'best_success_update': None,
                'significant_score': None, 'bad_checks': 0, 'initialization_checkpoint_sha256':
                None if args.initialize_from is None else common.file_hash(args.initialize_from)}
    resume = args.resume if args.resume is not None else latest if latest.exists() else None
    if resume is not None:
        saved = torch.load(resume, map_location='cpu')
        if saved.get('schema') != common.PROTOCOL_VERSION or saved.get('binding') != run_binding:
            raise ValueError('task-Adam resume objective/config/source changed; start a new experiment')
        policy.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        trainer.load_state_dict(saved['critic_training'])
        progress, initial_model = saved['progress'], saved['initial_model']
        common.restore_rng(saved['rng'])
        if progress['status'] == 'failed':
            raise RuntimeError('failed task-Adam run must be diagnosed, not automatically resumed')
    work.mkdir(parents=True, exist_ok=True)
    common.atomic_json(work/'configuration.json', {'binding': run_binding,
        'execution': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    eval_initial = _evaluation_initials(args, policy)
    previous_seconds, started = progress['elapsed_seconds'], time.monotonic()
    stop = {'requested': False}
    handlers = {s: signal.signal(s, lambda *unused: stop.update(requested=True))
                for s in (signal.SIGTERM, signal.SIGINT)}

    def save(path=latest):
        progress['elapsed_seconds'] = previous_seconds + time.monotonic() - started
        common.save_training(path, policy, optimizer, progress, {}, run_binding, initial_model, critic=trainer)

    def evaluate():
        trainer.stage = 'evaluation'
        rng = common.capture_rng()
        try:
            report = evaluate_task_policy(policy, simulator, eval_initial, args.horizon, loss_config)
        finally:
            common.restore_rng(rng)
        row = {k: v for k, v in report.items() if k != 'scenarios'}
        row.update(update=progress['updates'], attempt=progress['attempts'],
                   score=report['task_objective'], model_sha256=common.model_hash(policy),
                   elapsed_seconds=previous_seconds + time.monotonic() - started)
        progress['evaluation'].append(row)
        selected = common.record_development_selection(progress, row, args.min_relative_improvement)
        if selected['cost']:
            save(work/'best.training.pt')
        if selected['success']:
            save(work/'best_success.training.pt')
        common.atomic_json(work/'evaluation'/('%07d.json' % progress['updates']),
            {'seeds': list(common.DEVELOPMENT_SEEDS), 'scenarios': 2 * args.scenarios,
             'used_for_rollback': False, 'pooled_cvar': True, **row, 'scenarios_detail': report['scenarios']})
        print(json.dumps({'evaluation': row}, allow_nan=False), flush=True)
        trainer.stage = 'idle'

    current_initial = None
    try:
        if not progress['evaluation']:
            evaluate(); save()
        progress['status'] = 'training'
        while progress['attempts'] < args.updates:
            if stop['requested'] or previous_seconds + time.monotonic() - started >= args.max_seconds:
                progress['status'] = 'interrupted' if stop['requested'] else 'time_budget'
                break
            current_initial, seeds = common.sample_training_scenarios(args.scenarios, progress['attempts'],
                batches=2, dt=policy_config.dt, device=next(policy.parameters()).device,
                dtype=next(policy.parameters()).dtype, scenario_mode='fixed-airframe')
            _synchronize(policy)
            if next(policy.parameters()).is_cuda:
                torch.cuda.reset_peak_memory_stats()
            clock = time.monotonic()
            row = trainer.update(policy, optimizer, simulator, current_initial, args.horizon, loss_config,
                                 gradient_clip=args.gradient_clip, probes=args.phase1_probes)
            _synchronize(policy)
            row['update_seconds'] = time.monotonic() - clock
            row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if next(policy.parameters()).is_cuda else None
            progress['attempts'] += 1; progress['updates'] += 1
            progress['training_seeds'].extend(seeds)
            row.update(attempt=progress['attempts'], update=progress['updates'], scenario_seeds=seeds,
                       training_scenarios=2 * args.scenarios, model_sha256=common.model_hash(policy))
            # Metrics describe the episode BEFORE this Adam update; no candidate replay.
            row['train_before'].pop('scenarios', None)
            progress['history'].append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
            if progress['updates'] % args.development_every == 0:
                evaluate()
            if progress['updates'] % args.checkpoint_every == 0:
                save()
        if progress['status'] == 'training':
            progress['status'] = 'update_budget'
        if progress['evaluation'][-1]['update'] != progress['updates']:
            evaluate()
        save()
    except BaseException as error:
        progress.update(status='failed', error=repr(error), failure_stage=trainer.stage)
        common.atomic_torch(work/'failure.pt', {'stage': trainer.stage, 'error': repr(error),
            'next_attempt': progress['attempts'] + 1, 'initial': current_initial,
            'actor_gradients': {n: None if p.grad is None else p.grad.detach().cpu() for n, p in policy.named_parameters()}})
        save()
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    summary = {k: progress[k] for k in ('status', 'attempts', 'updates', 'elapsed_seconds',
                                      'best_score', 'best_update', 'best_success_rate', 'best_success_update')}
    summary.update(optimizer='task-adam', candidate_rollouts=0, rollback_count=0,
                   evaluation_checks=len(progress['evaluation']), final_evaluation=progress['evaluation'][-1])
    common.atomic_json(work/'training_report.json', summary)
    return summary


def profile_task_value(args, policy_config, loss_config):
    policy, simulator, trainer, optimizer = _setup(args, policy_config)
    initial, seeds = common.sample_training_scenarios(args.scenarios, 0, batches=2,
        dt=policy_config.dt, device=next(policy.parameters()).device,
        dtype=next(policy.parameters()).dtype, scenario_mode='fixed-airframe')
    _synchronize(policy)
    if next(policy.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    row = trainer.update(policy, optimizer, simulator, initial, args.horizon, loss_config,
                         gradient_clip=args.gradient_clip, probes=args.phase1_probes)
    _synchronize(policy)
    report = {'scope': 'one fresh task-value fit and short-window Adam update; no EVAL or candidate replay',
              'seconds': time.monotonic() - started, 'source_sha256': common.source_hash(),
              'device': args.device, 'dtype': args.dtype, 'scenarios': 2 * args.scenarios,
              'horizon': args.horizon, 'scenario_seeds': seeds,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated() if next(policy.parameters()).is_cuda else None,
              'result': row}
    common.atomic_json(args.work_dir/'task_value_profile.json', report)
    return report
