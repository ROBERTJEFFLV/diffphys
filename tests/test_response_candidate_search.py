from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

import response_critic as critic
import response_proposals as proposals
import response_task as task
from response_training import capture_rng
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


def metrics(performance=10., omega=0., saturation=0., *, finite=True, bounds=()):
    return {'finite': finite, 'task_objective': performance,
            'risk_objective': 100.,
            'risk_components': {'position': 50., 'velocity': 40., 'omega': 10., 'saturation': 0.},
            'hard_risk_components': {'omega': omega, 'saturation': saturation},
            'hard_risk_bounds_violated': list(bounds)}


def test_search_selects_best_safe_real_candidate_across_both_scales_and_restores_state():
    assert hasattr(proposals, 'search_candidates'), 'missing direct rollout candidate search'
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    before = copy.deepcopy(model.state_dict())
    rng = capture_rng()
    visits, draws = [], []
    def measure():
        x, y = model.weight.detach()[0].tolist()
        visits.append((x, y))
        draws.append(float(torch.rand(())))
        # The best task candidate (+x) is unsafe. The best safe y step is rho/4.
        return metrics(10.-5*x+(y-.25)**2-.0625, omega=max(x, 0.))
    result = proposals.search_candidates(model, torch.eye(2, dtype=torch.float64), measure,
        proposals.SubspaceConfig(parameter_relative_step=1.), baseline=metrics(),
        accept_candidate=lambda row: critic.acceptance_rejection(metrics(), row, (), (),
                                                                 dev_relative_tolerance=.002))
    assert len(visits) == 8 and len(set(visits)) == 8
    assert set(visits) == {(1.,0.),(-1.,0.),(.25,0.),(-.25,0.),(0.,1.),(0.,-1.),(0.,.25),(0.,-.25)}
    assert len(set(draws)) == 1, 'every candidate must see the same RNG'
    torch.testing.assert_close(result.parameters, torch.tensor([0., .25], dtype=torch.float64))
    assert result.metrics['task_objective'] == 9.9375
    assert result.evidence['candidate_rollouts'] == 8
    assert_nested_equal(model.state_dict(), before)
    assert_nested_equal(capture_rng(), rng)


def test_bad_candidate_does_not_abort_search_or_hide_later_safe_improvement():
    assert hasattr(proposals, 'search_candidates')
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    visits = []
    def measure():
        value = float(model.weight[0, 0])
        visits.append(value)
        if value > 0:
            raise FloatingPointError('nonfinite physical candidate')
        return metrics(10.+value)
    result = proposals.search_candidates(model, torch.ones(1, 1, dtype=torch.float64), measure,
        proposals.SubspaceConfig(parameter_relative_step=1.), baseline=metrics(),
        accept_candidate=lambda row: critic.acceptance_rejection(metrics(), row, (), (),
                                                                 dev_relative_tolerance=.002))
    assert len(visits) == 4 and result.metrics['task_objective'] == 9.
    assert sum(row['rejection_reason'] == 'candidate_nonfinite' for row in result.evidence['candidates']) == 2
    assert float(model.weight[0, 0]) == 0.


def test_search_restores_actor_and_rng_on_unexpected_error():
    assert hasattr(proposals, 'search_candidates')
    model = torch.nn.Linear(1, 1, bias=False).double()
    before, rng = copy.deepcopy(model.state_dict()), capture_rng()
    def broken():
        torch.rand(())
        raise RuntimeError('simulator failed')
    with pytest.raises(RuntimeError, match='simulator failed'):
        proposals.search_candidates(model, torch.ones(1, 1, dtype=torch.float64), broken,
            proposals.SubspaceConfig(), baseline=metrics(), accept_candidate=lambda row: None)
    assert_nested_equal(before, model.state_dict())
    assert_nested_equal(rng, capture_rng())


def test_no_improving_candidate_rejects_the_proposal_without_a_derivative_gate():
    assert hasattr(proposals, 'search_candidates')
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    result = proposals.search_candidates(model, torch.ones(1, 1, dtype=torch.float64),
        lambda: metrics(10.+float(model.weight.square().sum())), proposals.SubspaceConfig(),
        baseline=metrics(), accept_candidate=lambda row: critic.acceptance_rejection(
            metrics(), row, (), (), dev_relative_tolerance=.002))
    assert result.parameters is None and result.reason == 'no_acceptable_candidate'
    assert result.evidence['candidate_rollouts'] == 4


def test_search_does_not_stop_at_the_first_safe_improvement():
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    def measure():
        x = float(model.weight[0, 0])
        return metrics(10.+x*x-1.2*x)
    result = proposals.search_candidates(model, torch.ones(1, 1, dtype=torch.float64), measure,
        proposals.SubspaceConfig(parameter_relative_step=1.), baseline=metrics(),
        accept_candidate=lambda row: critic.acceptance_rejection(metrics(), row, (), (),
                                                                 dev_relative_tolerance=.002))
    assert result.evidence['candidates'][0]['rejection_reason'] is None
    assert float(result.parameters[0]) == .25
    assert result.metrics['task_objective'] == pytest.approx(9.7625)
    assert result.evidence['candidate_rollouts'] == 4


def test_full_rank_search_uses_at_most_twenty_real_candidates():
    model = torch.nn.Linear(5, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    result = proposals.search_candidates(model, torch.eye(5, dtype=torch.float64),
        lambda: metrics(), proposals.SubspaceConfig(), baseline=metrics(),
        accept_candidate=lambda row: 'train_not_improved')
    assert result.evidence['candidate_rollouts'] == 20
    assert result.reason == 'no_acceptable_candidate'


def test_performance_can_trade_position_and_velocity_while_hard_risk_stays_equal():
    before = metrics()
    after = metrics(9.)
    after['risk_objective'] = 1.e6
    after['risk_components'].update(position=1.e5, velocity=1.e5, omega=1.e5)
    assert critic.acceptance_rejection(before, after, [before, before], [after, after],
                                      dev_relative_tolerance=.002) is None


@pytest.mark.parametrize('bank', ['train', 'dev'])
@pytest.mark.parametrize('reason', ['omega', 'saturation', 'bounds', 'nonfinite'])
def test_true_hard_risks_reject_on_train_and_each_dev_bank(bank, reason):
    before, good, bad = metrics(), metrics(9.), metrics(8.)
    if reason in ('omega', 'saturation'):
        bad['hard_risk_components'][reason] = 1.
    elif reason == 'bounds':
        bad['hard_risk_bounds_violated'] = ['position']
    else:
        bad = {'finite': False}
    rejected = critic.acceptance_rejection(before, bad if bank == 'train' else good,
        [before, before], [good, bad] if bank == 'dev' else [good, good], dev_relative_tolerance=.002)
    assert rejected is not None and rejected.startswith('train_' if bank == 'train' else 'development_')


def test_hard_risk_is_zero_in_normal_operation_and_bounds_are_separate_from_tracking():
    assert hasattr(task, 'hard_risk_metrics'), 'missing separate physical danger metrics'
    policy, sim, initial = fixture()
    trace = task.rollout(policy, sim, initial, 1)
    omega = torch.zeros_like(trace.omegas)
    omega[..., 0] = 9.
    normal = replace(trace, omegas=omega, actions=torch.full_like(trace.actions, .9),
                     positions=torch.full_like(trace.positions, 2.), velocities=torch.full_like(trace.velocities, 2.))
    row = task.hard_risk_metrics(normal, task.RiskConfig(), task.TaskLossConfig(), task.HardRiskConfig())
    assert row['hard_risk_components'] == {'omega': 0., 'saturation': 0.}
    assert row['hard_risk_bounds_violated'] == []
    bound = task.hard_risk_metrics(normal, task.RiskConfig(), task.TaskLossConfig(),
                                 task.HardRiskConfig(position_bound=3.))
    assert bound['hard_risk_bounds_violated'] == ['position']
    omega[..., 0] = 15.
    danger = replace(normal, omegas=omega, actions=torch.full_like(trace.actions, .975))
    row = task.hard_risk_metrics(danger, task.RiskConfig(), task.TaskLossConfig(), task.HardRiskConfig())
    assert row['hard_risk_components']['omega'] == pytest.approx(.375)
    assert row['hard_risk_components']['saturation'] == pytest.approx(.375)


def test_exceeding_the_hard_motor_command_limit_is_always_out_of_bounds():
    policy, sim, initial = fixture()
    trace = task.rollout(policy, sim, initial, 1)
    row = task.hard_risk_metrics(replace(trace, actions=torch.full_like(trace.actions, 1.01)),
                                task.RiskConfig(), task.TaskLossConfig(), task.HardRiskConfig())
    assert row['hard_risk_bounds_violated'] == ['action_limit']


def test_hard_risk_allows_only_the_declared_small_budget_including_zero_baseline():
    assert hasattr(task, 'HardRiskConfig')
    config = task.HardRiskConfig(relative_tolerance=.001, absolute_tolerance=1.e-8)
    before = metrics(10., omega=1.)
    acceptable = metrics(9., omega=1.0005, saturation=5.e-9)
    assert critic.acceptance_rejection(before, acceptable, (), (), dev_relative_tolerance=.002,
                                      hard_risk_config=config) is None
    unacceptable = metrics(9., omega=1.002)
    assert critic.acceptance_rejection(before, unacceptable, (), (), dev_relative_tolerance=.002,
                                      hard_risk_config=config) == 'train_hard_risk_deteriorated'


def test_periodic_report_does_not_add_a_third_actor_acceptance_gate(tmp_path, monkeypatch):
    import response_training as training
    from tools.train_response_control import parse_args
    from response_policy import ResponsePolicyConfig
    monkeypatch.setattr(training, 'FINAL_CLAIM', tmp_path / 'unused')
    reports = []
    def report(*args, **kwargs):
        reports.append(1)
        return {'score': 10. if len(reports) == 1 else 21., 'finite': True,
                'position_rms': 1., 'velocity_rms': 1., 'omega_rms': 1.,
                'steady_success_rate': 0., 'motor_saturation_fraction': 0.,
                'risk_objective': 1., 'risk_components': {}, 'hard_risk_components': {},
                'hard_risk_bounds_violated': []}
    monkeypatch.setattr(training, 'evaluate', report)
    def accepted(self, policy, *args, **kwargs):
        # Isolate the run state machine after its primary TRAIN/DEV guard.
        with torch.no_grad():
            policy.controller[-1].bias.add_(1.e-5)
        return {'accepted': True, 'proposal_finite': True, 'rejection_reason': None}
    monkeypatch.setattr(critic.CriticTrainer, 'guarded_step', accepted)
    args = parse_args(['--work-dir', str(tmp_path), '--updates', '2', '--horizon', '4',
                       '--window-steps', '2', '--scenarios', '16', '--development-every', '1'])
    result = training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8),
                            task.TaskLossConfig(prediction_weight=0))
    assert result['actual_updates'] == 2 and result['status'] == 'update_budget'
