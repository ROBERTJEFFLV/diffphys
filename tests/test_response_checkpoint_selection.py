from __future__ import annotations

import copy

import torch

import response_training as training
import response_task as task
import response_critic as critic
from response_policy import ResponsePolicyConfig
from tools.train_response_control import parse_args
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


def test_small_cost_improvements_all_save_and_cumulatively_reset_patience():
    assert hasattr(training, 'record_development_selection'), 'best saving still tied to significant improvement'
    progress = {'updates': 0, 'best_score': None, 'bad_checks': 0}
    saves, counts = [], []
    for index, score in enumerate((1., .999, .998, .997)):
        progress['updates'] = index
        selected = training.record_development_selection(progress,
            {'score': score, 'omega_rms': .1, 'steady_success_rate': .1}, .002)
        saves.append(selected['cost'])
        counts.append(progress['bad_checks'])
    assert saves == [True, True, True, True]
    assert counts == [0, 1, 2, 0]
    assert progress['best_score'] == progress['significant_score'] == .997


def test_best_cost_and_success_files_contain_the_actual_distinct_actor_weights(tmp_path, monkeypatch):
    # Exercise the real train/save path; replace only proposal outcomes and DEV
    # measurements to create exact, reproducible selection conflicts.
    monkeypatch.setattr(training, 'FINAL_CLAIM', tmp_path / 'unused')
    snapshots = []
    values = iter(((1., 0.), (.999, .2), (.998, .1), (.997, .05)))
    def evaluate(policy, *args, **kwargs):
        snapshots.append(copy.deepcopy(policy.state_dict()))
        score, success = next(values)
        return {'score': score, 'steady_success_rate': success, 'finite': True,
                'position_rms': .1, 'velocity_rms': .1, 'omega_rms': .1,
                'motor_saturation_fraction': 0., 'risk_objective': 0., 'risk_components': {},
                'hard_risk_components': {}, 'hard_risk_bounds_violated': []}
    def propose(self, policy, *args, **kwargs):
        with torch.no_grad():
            policy.controller[-1].bias.add_(.001)
        return {'accepted': True, 'proposal_finite': True, 'rejection_reason': None}
    monkeypatch.setattr(training, 'evaluate', evaluate)
    monkeypatch.setattr(critic.CriticTrainer, 'guarded_step', propose)
    args = parse_args(['--work-dir', str(tmp_path), '--updates', '3', '--horizon', '4',
                      '--window-steps', '2', '--scenarios', '16', '--development-every', '1'])
    training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8), task.TaskLossConfig(prediction_weight=0))
    cost = torch.load(tmp_path / 'best.training.pt', map_location='cpu')
    assert cost['progress']['best_score'] == .997
    success_path = tmp_path / 'best_success.training.pt'
    assert success_path.exists(), 'success winner must have its own actual checkpoint'
    success = torch.load(success_path, map_location='cpu')
    assert_nested_equal(cost['model'], snapshots[3])
    assert_nested_equal(success['model'], snapshots[1])
    assert success['progress']['best_success_rate'] == .2
