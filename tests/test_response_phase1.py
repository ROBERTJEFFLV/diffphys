from __future__ import annotations

import copy
from dataclasses import fields, replace
import importlib.util

import pytest
import torch

import response_critic as critic
import response_task as task
from env_l2f import L2FParams, L2FSimulator
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


def phase1():
    assert importlib.util.find_spec('response_phase1'), 'missing Phase 1 probes'
    import response_phase1
    return response_phase1


def test_fixed_airframe_changes_only_initial_kinematics_and_preserves_rng():
    rng = torch.get_rng_state().clone()
    a, cells = task.sample_scenarios(32, seed=31000007, scenario_mode='fixed-airframe', dtype=torch.float64)
    b, _ = task.sample_scenarios(32, seed=32000007, scenario_mode='fixed-airframe', dtype=torch.float64)
    assert torch.equal(torch.get_rng_state(), rng)
    with torch.random.fork_rng(devices=[]):
        nominal = L2FSimulator(L2FParams()).reset(32, device='cpu', dtype=torch.float32,
                                               sample_dynamics=False, sample_external_force=False)
    varying = {'position', 'velocity', 'rotation', 'omega'}
    for f in fields(a):
        if f.name in varying:
            assert not torch.equal(getattr(a, f.name), getattr(b, f.name))
        else:
            torch.testing.assert_close(getattr(a, f.name), getattr(nominal, f.name).double(), rtol=0, atol=0)
            assert torch.equal(getattr(a, f.name), getattr(b, f.name))
    assert not a.external_force.any()
    assert (cells == -1).all()  # No fictitious dynamics strata.


def test_boundary_probe_covers_memory_history_and_detects_a_splice():
    p = phase1()
    policy, sim, initial = fixture()
    reference = p.boundary_reference(policy, sim, initial, 6, 2)
    assert set(reference) == {2, 4, 6}
    closed = task.initialize(policy, initial)
    for end in (2, 4, 6):
        closed = task.rollout(policy, sim, critic.detach_closed_state(closed), 2).end
        report = p.compare_boundary(reference[end], closed, end)
        assert report['passed'] and report['exact']
        assert len(report['fields']) == len(fields(closed.physical)) + len(fields(closed.policy))
    bad = replace(closed, policy=replace(closed.policy, memory=closed.policy.memory + .01))
    with pytest.raises(p.Phase1ProbeError, match='boundary'):
        p.compare_boundary(reference[6], bad, 6)
    bad = replace(closed, physical=replace(closed.physical, motor=closed.physical.motor * float('nan')))
    with pytest.raises(p.Phase1ProbeError, match='nonfinite'):
        p.compare_boundary(reference[6], bad, 6)


def test_loss_components_reconstruct_original_and_minimal_cvar_objectives():
    p = phase1()
    policy, sim, initial = fixture()
    trace = task.rollout(policy, sim, initial, 6)
    config = task.TaskLossConfig(steady_steps=2, tail_fraction=.5)
    for selected in (config, task.training_loss_config(config)):
        report = p.task_loss_components(trace, selected)
        expected = float(task.task_loss(trace, selected))
        assert sum(report[k] for k in ('position', 'velocity', 'omega', 'regularization')) == pytest.approx(expected)
        assert report['total'] == pytest.approx(expected)
        assert report['mean'] + report['cvar_addition'] == pytest.approx(expected)


def test_probes_do_not_change_actor_gradients_or_critic_state():
    p = phase1()
    policy, sim, initial = fixture()
    config = task.TaskLossConfig()
    record = critic.collect_trajectory(policy, sim, initial, 6, config)
    target = critic.RiskToGoCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    before = copy.deepcopy(target.state_dict())
    kwargs = dict(baseline_returns=record.returns, separate_objectives=True)
    plain = critic.accumulate_actor_gradients(policy, target, sim, initial, 6, 2, config, record.weights, **kwargs)
    observed = critic.accumulate_actor_gradients(policy, target, sim, initial, 6, 2, config, record.weights,
        phase1_reference=p.boundary_reference(policy, sim, initial, 6, 2), **kwargs)
    torch.testing.assert_close(plain['objective_gradients'], observed['objective_gradients'], rtol=0, atol=0)
    assert_nested_equal(before, target.state_dict())
    probes = observed['phase1_windows']
    assert len(probes) == 3 and probes[-1]['terminal_state_gradient_norm'] == 0
    assert all(row['performance_gradient_norm'] > 0 and row['risk_gradient_norm'] > 0 for row in probes)
    assert all(row['terminal_state_gradient_norm'] > 0 for row in probes[:-1])


def test_proposal_summary_distinguishes_best_loss_from_acceptable_candidate():
    p = phase1()
    evidence = {'continuous_loss_before': 10., 'search': {'candidates': [
        {'performance': 8., 'rejection_reason': 'hard_risk'},
        {'performance': 9., 'rejection_reason': None},
        {'performance': 11., 'rejection_reason': 'performance'},
    ]}}
    report = p.proposal_summary(evidence)
    assert report['best_candidate_h500_loss'] == 8.
    assert report['best_acceptable_h500_loss'] == 9.
    assert report['best_improvement_fraction'] == pytest.approx(.2)
    assert report['has_acceptable_train_candidate']
    assert report['acceptable_train_candidates'] == 1


def test_phase1_training_routes_train_and_dev_and_persists_probes(tmp_path, monkeypatch):
    import response_training as training
    from response_policy import ResponsePolicyConfig
    from tools.train_response_control import parse_args
    monkeypatch.setattr(training, 'FINAL_CLAIM', tmp_path/'unused-final')
    args = parse_args(['--scenario-mode', 'fixed-airframe', '--phase1-probes', '--updates', '1',
                      '--horizon', '4', '--window-steps', '2', '--scenarios', '16',
                      '--memory-dim', '4', '--hidden-dim', '8', '--development-every', '1',
                      '--work-dir', str(tmp_path/'run')])
    training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8), task.TaskLossConfig(prediction_weight=0))
    saved = torch.load(tmp_path/'run/latest.training.pt', map_location='cpu')
    assert saved['binding']['protocol']['scenario_mode'] == 'fixed-airframe'
    row = saved['progress']['history'][0]
    assert row['phase1_windows'] and row['phase1_proposal']['baseline_h500_loss'] is not None
    assert row['proposal_seconds'] > 0
    import json
    dev = json.loads((tmp_path/'run/development/0000000.json').read_text())
    assert dev['scenario_mode'] == 'fixed-airframe'
    assert not dev['unseen_parameter_draws_within_registered_family']
    assert all('task_loss_components' in r['policy'] for r in dev['records'])


def test_candidate_nonfinite_stops_phase1_and_keeps_completed_critic_fit(monkeypatch):
    policy, sim, initial = fixture()
    trainer = critic.CriticTrainer(policy, task.initialize(policy, initial), 6,
        critic.CriticConfig(window_steps=2, phase1_probes=True, direction_samples=0))
    before = copy.deepcopy(policy.state_dict())
    original = trainer._metrics
    calls = []
    def corrupt_candidate(trace, loss_config, *, candidate=False):
        if candidate:
            calls.append(1)
            trace = replace(trace, positions=trace.positions * float('nan'))
        return original(trace, loss_config, candidate=candidate)
    monkeypatch.setattr(trainer, '_metrics', corrupt_candidate)
    result = trainer.guarded_step(policy, None, sim, initial, 6, task.TaskLossConfig(),
                                  development_initials=(initial,), gradient_clip=10)
    assert len(calls) == 1
    assert not result['proposal_finite'] and not result['accepted']
    assert result['phase1_failure']['stage'] == 'candidate_nonfinite'
    assert result['critic_update_retained'] and trainer.completed_fits == 1
    assert_nested_equal(before, policy.state_dict())
