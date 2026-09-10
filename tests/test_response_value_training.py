from __future__ import annotations

import copy
import json

import pytest
import torch

from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
import response_training as training
from test_response_guarded_updates import assert_nested_equal
from tools.train_response_control import parse_args


def arguments(work, updates=2):
    return parse_args(['--optimizer', 'task-adam', '--scenario-mode', 'fixed-airframe',
        '--phase1-probes', '--value-terminal-mode', 'oracle_full_state', '--updates', str(updates), '--horizon', '4', '--window-steps', '2',
        '--scenarios', '16', '--memory-dim', '4', '--hidden-dim', '8',
        '--development-every', '1', '--checkpoint-every', '1', '--work-dir', str(work)])


def run(work, monkeypatch, updates=2):
    monkeypatch.setattr(training, 'FINAL_CLAIM', work.parent/'unused-final')
    args = arguments(work, updates)
    training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8), TaskLossConfig(prediction_weight=0))
    return torch.load(work/'latest.training.pt', map_location='cpu')


def test_task_adam_runs_pooled_objective_and_never_rolls_back_worsening_eval(tmp_path, monkeypatch):
    args = arguments(tmp_path/'run')  # first fail must be the absent CLI route
    import response_value_training as value_training
    original = value_training.evaluate_task_policy
    counter = []
    def worsening(*a, **kw):
        report = original(*a, **kw)
        counter.append(1)
        report['task_objective'] = float(len(counter) * 1.e6)
        return report
    monkeypatch.setattr(value_training, 'evaluate_task_policy', worsening)
    saved = run(args.work_dir, monkeypatch)
    assert saved['progress']['updates'] == 2
    assert len(saved['progress']['evaluation']) == 3  # initial + two periodic checks
    assert saved['progress']['best_update'] == 0
    assert all(float(s['step']) == 2 for s in saved['optimizer']['state'].values())
    assert saved['binding']['training_loss'] == saved['binding']['protocol']['loss']
    assert saved['binding']['critic']['objective'] == saved['critic_training']['objective']
    assert saved['binding']['optimizer'] == 'task-adam'
    for row in saved['progress']['history']:
        assert row['training_scenarios'] == 32
        assert row['updated'] and len(row['windows']) == 2
    best = torch.load(args.work_dir/'best.training.pt', map_location='cpu')
    assert best['progress']['updates'] == 0
    assert any(not torch.equal(best['model'][k], saved['model'][k]) for k in saved['model'])


def test_exact_resume_preserves_both_adams_target_and_sampling(tmp_path, monkeypatch):
    whole = run(tmp_path/'whole', monkeypatch, updates=2)
    first = run(tmp_path/'split', monkeypatch, updates=1)
    assert first['progress']['updates'] == 1
    resumed = run(tmp_path/'split', monkeypatch, updates=2)
    for key in ('model', 'optimizer', 'critic_training', 'rng'):
        assert_nested_equal(whole[key], resumed[key])
    assert whole['progress']['training_seeds'] == resumed['progress']['training_seeds']
    assert resumed['progress']['updates'] == 2


def test_task_adam_failure_saves_stage_and_does_not_submit_actor(tmp_path, monkeypatch):
    arguments(tmp_path/'failed')
    import response_value
    original = response_value.accumulate_task_gradients
    def bad(*args, **kwargs):
        original(*args, **kwargs)
        next(args[0].parameters()).grad.fill_(float('nan'))
    monkeypatch.setattr(response_value, 'accumulate_task_gradients', bad)
    with pytest.raises(RuntimeError):
        run(tmp_path/'failed', monkeypatch, updates=1)
    saved = torch.load(tmp_path/'failed/latest.training.pt', map_location='cpu')
    assert saved['progress']['status'] == 'failed'
    assert saved['progress']['failure_stage'] == 'short_window_backward'
    assert saved['optimizer']['state'] == {}
    assert saved['critic_training']['completed_fits'] == 1
    assert (tmp_path/'failed/failure.pt').exists()
