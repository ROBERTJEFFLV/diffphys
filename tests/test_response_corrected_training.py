from __future__ import annotations

import copy
from dataclasses import replace
import inspect

import torch
import pytest

import response_critic as critic
import response_task as task
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


def test_separate_window_gradients_match_detached_physics_and_keep_actor_continuous():
    policy, sim, initial = fixture()
    config = task.TaskLossConfig(steady_steps=2)
    record = critic.collect_trajectory(policy, sim, initial, 6, config)
    target = critic.RiskToGoCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    before = copy.deepcopy(policy.state_dict())
    assert 'separate_objectives' in inspect.signature(critic.accumulate_actor_gradients).parameters
    result = critic.accumulate_actor_gradients(policy, target, sim, initial, 6, 2, config,
        record.weights, baseline_returns=record.returns, separate_objectives=True)
    weights = torch.stack([task.risk_weights(record.returns[0, :, j], config) for j in range(4)], -1)
    totals = []
    closed = task.initialize(policy, initial)
    for start in (0, 2, 4):
        trace = task.rollout(policy, sim, critic.detach_closed_state(closed), 2)
        risks = torch.stack(tuple(task.risk_components(trace, task.RiskConfig()).values()), -1).sum(0)
        if start != 4:
            risks = risks + target(critic.critic_features(trace.end, start+2, 6))
        performance = (record.weights * task.step_costs(trace, config, start=start, horizon=6).sum(0)).sum()
        totals.append(torch.cat((performance.view(1), (weights * risks).sum(0))) / 3)
        closed = trace.end
    objectives = torch.stack(totals).sum(0)
    parameters = list(policy.parameters())
    for j in range(5):
        rows = torch.autograd.grad(objectives[j], parameters, retain_graph=j < 4, allow_unused=True)
        expected = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                              for p, g in zip(parameters, rows)])
        torch.testing.assert_close(result['objective_gradients'][j], expected, rtol=1.e-9, atol=1.e-10)
    assert_nested_equal(policy.state_dict(), before)
    assert torch.equal(result['end'].policy.memory, record.trajectory.end.policy.memory)
    assert all(p.grad is None for p in target.parameters())


def test_failed_train_gate_skips_dev_and_retains_completed_fit(monkeypatch):
    policy, sim, initial = fixture()
    state = critic.CriticTrainer(policy, task.initialize(policy, initial), 6,
                                critic.CriticConfig(window_steps=2, direction_samples=0, proposal='smoothmax-adam'))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.)
    dev = replace(initial, position=initial.position + 10.)
    original = critic.rollout
    visits = []
    def observed(policy, sim, start, *args, **kwargs):
        if start is dev:
            visits.append(1)
        return original(policy, sim, start, *args, **kwargs)
    monkeypatch.setattr(critic, 'rollout', observed)
    result = state.guarded_step(policy, optimizer, sim, initial, 6, task.TaskLossConfig(),
                                development_initials=(dev,), gradient_clip=10.)
    assert not result['accepted']
    assert visits == [], 'rejected TRAIN must not spend time on candidate or baseline DEV'
    assert state.completed_fits == 1 and result['critic_update_retained']


def test_checked_increment_is_applied_directly_without_adam_or_weight_decay(monkeypatch):
    import response_proposals as proposals
    assert 'proposal' in critic.CriticConfig.__dataclass_fields__
    policy, sim, initial = fixture()
    with torch.no_grad():
        policy.controller[-1].weight.zero_()
        policy.controller[-1].bias.fill_(.2)
    initial = replace(initial, position=initial.position.new_tensor([[0.,0.,.3]]).repeat(2,1),
                      velocity=initial.velocity.new_tensor([[0.,0.,.2]]).repeat(2,1),
                      omega=torch.zeros_like(initial.omega))
    state = critic.CriticTrainer(policy, task.initialize(policy, initial), 6,
        critic.CriticConfig(window_steps=2, direction_samples=0, proposal='physics-subspace'))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=10., weight_decay=1.)
    # Seed stale moments deliberately: the checked proposal must ignore all of them.
    for p in policy.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    # Re-create the controlled baseline after seeding optimizer history.
    baseline_policy, _, _ = fixture()
    policy.load_state_dict(baseline_policy.state_dict())
    with torch.no_grad():
        policy.controller[-1].weight.zero_()
        policy.controller[-1].bias.fill_(.2)
    before = torch.nn.utils.parameters_to_vector(policy.parameters()).detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    direction = torch.cat([torch.full_like(p, -.5).flatten() if name == 'controller.4.bias'
                           else torch.zeros_like(p).flatten() for name, p in policy.named_parameters()])
    assert direction.norm().item() == 1
    def checked(*args, **kwargs):
        return proposals.CorrectedDirection(direction, None, {'parameter_step': .002, 'basis_rank': 1,
            'probe_rollouts': 4, 'objective_gradient_norms': [1.]*5})
    monkeypatch.setattr(proposals, 'correct_direction', checked)
    def forbidden(*args, **kwargs):
        raise AssertionError('checked increment passed through Adam')
    optimizer.step = forbidden
    result = state.guarded_step(policy, optimizer, sim, initial, 6, task.TaskLossConfig(),
                               development_initials=(initial, initial), gradient_clip=10.)
    assert result['accepted'], result
    torch.testing.assert_close(torch.nn.utils.parameters_to_vector(policy.parameters()), before + .002*direction)
    assert_nested_equal(optimizer.state_dict(), saved_optimizer)


def test_dev_cache_survives_critic_fit_and_invalidates_on_actor_states_and_objective(monkeypatch):
    policy, sim, initial = fixture()
    config = task.TaskLossConfig()
    state = critic.CriticTrainer(policy, task.initialize(policy, initial), 4,
                                critic.CriticConfig(window_steps=2, direction_samples=0))
    assert hasattr(state, 'development_baseline'), 'missing keyed DEV baseline cache'
    first, hit = state.development_baseline(policy, sim, (initial,), 4, config)
    assert not hit
    state.fit(critic.collect_trajectory(policy, sim, initial, 4, config))
    second, hit = state.development_baseline(policy, sim, (initial,), 4, config)
    assert hit and first == second
    with torch.no_grad():
        policy.controller[-1].bias.add_(.01)
    _, hit = state.development_baseline(policy, sim, (initial,), 4, config)
    assert not hit
    initial.position.add_(1.)
    _, hit = state.development_baseline(policy, sim, (initial,), 4, config)
    assert not hit
    _, hit = state.development_baseline(policy, sim, (initial,), 4, replace(config, position_weight=2.))
    assert not hit
    _, hit = state.development_baseline(policy, sim, (initial,), 4, config)
    policy.config = replace(policy.config, action_rate=1.)
    _, hit = state.development_baseline(policy, sim, (initial,), 4, config)
    assert not hit


def test_fullspace_debug_profile_still_runs_its_historical_gradient_probe(tmp_path):
    import response_training as training
    from response_policy import ResponsePolicyConfig
    from tools.train_response_control import parse_args
    args = parse_args(['--mode', 'profile', '--optimizer', 'full-space-ms', '--segment-steps', '2',
                       '--scenarios', '16', '--memory-dim', '4', '--hidden-dim', '8', '--work-dir', str(tmp_path)])
    report = training.profile(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8), task.TaskLossConfig())
    assert report['horizon'] == 4 and report['production_checkpoint_written'] is False


@pytest.mark.parametrize('reason', ['no_feasible_direction', 'fd_unreliable', 'line_search_exhausted'])
def test_rejected_proposals_keep_fitting_until_plateau_and_cannot_auto_resume(tmp_path, monkeypatch, reason):
    import response_training as training
    import response_proposals as proposals
    from response_policy import ResponsePolicyConfig
    from tools.train_response_control import parse_args
    monkeypatch.setattr(training, 'FINAL_CLAIM', tmp_path / 'unused')
    def unavailable(*args, **kwargs):
        return proposals.CorrectedDirection(None, reason, {'basis_rank': 2, 'probe_rollouts': 8})
    monkeypatch.setattr(proposals, 'correct_direction', unavailable)
    evaluations = []
    evaluate = training.evaluate
    def observed(*args, **kwargs):
        evaluations.append(1)
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(training, 'evaluate', observed)
    args = parse_args(['--work-dir', str(tmp_path), '--updates', '4', '--horizon', '4', '--window-steps', '2',
                       '--maximum-proposal-rejections', '3',
                       '--scenarios', '16', '--memory-dim', '4', '--hidden-dim', '8', '--critic-direction-samples', '0'])
    result = training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8),
                            task.TaskLossConfig(prediction_weight=0))
    assert result['status'] == 'proposal_plateau' and result['exit_class'] == 'business_stop'
    assert result['actual_attempts'] == 3 and len(evaluations) == 1
    saved = torch.load(tmp_path / 'latest.training.pt', map_location='cpu')
    assert saved['optimizer'] is None, 'direct proposals must not carry Actor Adam history'
    assert saved['critic_training']['completed_fits'] == 3
    assert saved['solver']['consecutive_proposal_rejections'] == 3
    assert [row['rejection_reason'] for row in saved['progress']['history']] == [reason]*3
    assert all(row['critic_update_retained'] for row in saved['progress']['history'])
    resumed = training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8),
                             task.TaskLossConfig(prediction_weight=0))
    assert resumed['resume_blocked'] and resumed['actual_attempts'] == 3
    assert len(evaluations) == 1
    assert_nested_equal(saved, torch.load(tmp_path / 'latest.training.pt', map_location='cpu'))


def test_acceptance_resets_consecutive_proposal_rejections(tmp_path, monkeypatch):
    import response_training as training
    from response_policy import ResponsePolicyConfig
    from tools.train_response_control import parse_args
    monkeypatch.setattr(training, 'FINAL_CLAIM', tmp_path / 'unused')
    outcomes = iter([False, True, False, False])
    # Isolate the run state machine; physical acceptance has separate real-rollout tests.
    def outcome(self, *args, **kwargs):
        accepted = next(outcomes)
        return {'accepted': accepted, 'proposal_finite': True,
                'rejection_reason': None if accepted else 'no_feasible_direction'}
    monkeypatch.setattr(critic.CriticTrainer, 'guarded_step', outcome)
    args = parse_args(['--work-dir', str(tmp_path), '--updates', '4', '--horizon', '4', '--window-steps', '2',
                       '--scenarios', '16', '--maximum-proposal-rejections', '2'])
    result = training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8), task.TaskLossConfig(prediction_weight=0))
    assert result['status'] == 'proposal_plateau' and result['actual_attempts'] == 4
    assert result['actual_updates'] == 1
    saved = torch.load(tmp_path / 'latest.training.pt', map_location='cpu')
    assert saved['solver']['consecutive_proposal_rejections'] == 2
    assert [row['consecutive_proposal_rejections'] for row in saved['progress']['history']] == [1, 0, 1, 2]


def test_nonfinite_proposal_is_error_stop_and_does_not_retry(tmp_path, monkeypatch):
    import response_training as training
    from response_policy import ResponsePolicyConfig
    from tools.train_response_control import parse_args
    monkeypatch.setattr(training, 'FINAL_CLAIM', tmp_path / 'unused')
    def nonfinite(self, *args, **kwargs):
        return {'accepted': False, 'proposal_finite': False, 'rejection_reason': 'nonfinite probe'}
    monkeypatch.setattr(critic.CriticTrainer, 'guarded_step', nonfinite)
    args = parse_args(['--work-dir', str(tmp_path), '--updates', '4', '--horizon', '4', '--window-steps', '2',
                       '--scenarios', '16'])
    with pytest.raises(FloatingPointError, match='nonfinite probe'):
        training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8), task.TaskLossConfig(prediction_weight=0))
    saved = torch.load(tmp_path / 'latest.training.pt', map_location='cpu')
    assert saved['progress']['status'] == 'failed' and saved['progress']['exit_class'] == 'crash'
    assert saved['progress']['attempts'] == 1
