from __future__ import annotations

import copy
from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch

import response_task as task
from response_critic import critic_features, detach_closed_state
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


def value_module():
    assert importlib.util.find_spec('response_value'), 'missing unified task-value training'
    return importlib.import_module('response_value')


def flat_grad(policy):
    return torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten()
                      for p in policy.parameters()])


def test_mc_labels_are_exact_task_suffixes_and_share_full_flight_cvar():
    value = value_module()
    policy, sim, initial = fixture()
    config = task.TaskLossConfig(steady_steps=2, prediction_weight=0)
    record = value.collect_task_trajectory(policy, sim, initial, 6, 2, config)
    expected = task.step_costs(record.trajectory, config)
    for t in range(7):
        torch.testing.assert_close(record.returns[t], expected[t:].sum(0))
    assert not record.inputs.requires_grad and not record.returns.requires_grad
    assert record.returns.shape == (7, initial.position.shape[0])
    assert (record.weights * record.returns[0]).sum() == pytest.approx(float(task.task_loss(record.trajectory, config)))
    assert set(record.boundaries) == {0, 2, 4, 6}
    # At identical kinematics, last-two-step costs are 7 times normal steps:
    # (1/6 + 2/2) / (1/6) = 7. Windows must not restart the tail clock.
    frozen = replace(record.trajectory,
        positions=torch.ones_like(record.trajectory.positions),
        velocities=torch.zeros_like(record.trajectory.velocities),
        omegas=torch.zeros_like(record.trajectory.omegas),
        actions=torch.zeros_like(record.trajectory.actions),
        action_deltas=torch.zeros_like(record.trajectory.action_deltas),
        omega_deltas=torch.zeros_like(record.trajectory.omega_deltas))
    costs = task.step_costs(frozen, config)
    torch.testing.assert_close(costs[-1], 7 * costs[0])


def test_one_window_actor_gradient_equals_full_evaluation_objective_gradient():
    value = value_module()
    policy, sim, initial = fixture()
    config = task.TaskLossConfig(steady_steps=2, prediction_weight=0)
    record = value.collect_task_trajectory(policy, sim, initial, 6, 6, config)
    target = value.TaskValueCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    policy.zero_grad(set_to_none=True)
    task.task_loss(task.rollout(policy, sim, initial, 6), config).backward()
    expected = flat_grad(policy).clone()
    report = value.accumulate_task_gradients(policy, target, sim, initial, 6, 6, config, record)
    torch.testing.assert_close(flat_grad(policy), expected, rtol=1e-9, atol=1e-10)
    assert report['windows'][0]['terminal_value'] == 0
    assert report['windows'][0]['boundary']['exact']
    assert all(p.grad is None for p in target.parameters())


def test_window_gradient_matches_frozen_boundary_surrogate_and_keeps_terminal_derivative():
    value = value_module()
    policy, sim, initial = fixture()
    config = task.TaskLossConfig(steady_steps=2, prediction_weight=0)
    record = value.collect_task_trajectory(policy, sim, initial, 6, 2, config)
    # A known linear terminal value catches no_grad, missing tail time and
    # accidental mean-risk (H-t) rescaling of task-value units.
    target = torch.nn.Linear(record.inputs.shape[-1], 1, bias=False).double().requires_grad_(False)
    with torch.no_grad():
        target.weight.zero_(); target.weight[0, 0] = 7.; target.weight[0, 20] = .3
    policy.zero_grad(set_to_none=True)
    expected_loss = 0
    for start in (0, 2, 4):
        closed = task.initialize(policy, initial) if start == 0 else record.boundaries[start]
        trace = task.rollout(policy, sim, detach_closed_state(closed), 2)
        local = task.step_costs(trace, config, start=start, horizon=6).sum(0)
        future = target(critic_features(trace.end, start + 2, 6)).squeeze(-1) if start < 4 else 0
        expected_loss = expected_loss + (record.weights * (local + future)).sum() / 3
    expected_loss.backward()
    expected = flat_grad(policy).clone()
    state = copy.deepcopy(target.state_dict())
    result = value.accumulate_task_gradients(policy, target, sim, initial, 6, 2, config, record)
    torch.testing.assert_close(flat_grad(policy), expected, rtol=1e-9, atol=1e-10)
    assert all(w['boundary']['exact'] for w in result['windows'])
    assert all(w['terminal_state_gradient_norm'] > 0 for w in result['windows'][:-1])
    assert_nested_equal(state, target.state_dict())
    assert all(p.grad is None for p in target.parameters())


def test_critic_fit_updates_only_critic_then_polyak_target_in_task_units():
    value = value_module()
    policy, sim, initial = fixture()
    config = task.TaskLossConfig(prediction_weight=0)
    record = value.collect_task_trajectory(policy, sim, initial, 6, 2, config)
    trainer = value.TaskValueTrainer(policy, task.initialize(policy, initial), 6,
        value.TaskValueConfig(window_steps=2, target_tau=.25, epochs=2, batch_size=8))
    before = copy.deepcopy(policy.state_dict())
    old_target = copy.deepcopy(trainer.target.state_dict())
    result = trainer.fit(record)
    assert result['critic_loss_after'] < result['critic_loss_before']
    for name, current in trainer.target.state_dict().items():
        torch.testing.assert_close(current, .75 * old_target[name] + .25 * trainer.critic.state_dict()[name])
    assert_nested_equal(before, policy.state_dict())
    assert all(p.grad is None for p in policy.parameters())
    assert all(not p.requires_grad for p in trainer.target.parameters())
    restored = value.TaskValueTrainer(policy, task.initialize(policy, initial), 6, trainer.config)
    restored.load_state_dict(copy.deepcopy(trainer.state_dict()))
    assert_nested_equal(trainer.state_dict(), restored.state_dict())
    stale = copy.deepcopy(trainer.state_dict()); stale['objective'] = 'old-risk'
    with pytest.raises(ValueError, match='objective'):
        restored.load_state_dict(stale)


def test_update_uses_persistent_adam_without_candidate_replay(monkeypatch):
    value = value_module()
    import response_proposals
    policy, sim, initial = fixture()
    config = task.TaskLossConfig(prediction_weight=0)
    trainer = value.TaskValueTrainer(policy, task.initialize(policy, initial), 6,
        value.TaskValueConfig(window_steps=2, batch_size=16, terminal_mode='oracle_full_state'))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    def forbidden(*args, **kwargs):
        raise AssertionError('candidate search must not run')
    monkeypatch.setattr(response_proposals, 'search_candidates', forbidden)
    calls = []; original = value.rollout
    def count(*args, **kwargs):
        calls.append(args[3]); return original(*args, **kwargs)
    monkeypatch.setattr(value, 'rollout', count)
    for _ in range(2):
        result = trainer.update(policy, optimizer, sim, initial, 6, config, gradient_clip=10)
        assert result['updated'] and result['finite']
        assert result['derivative_train_samples'] > 0 and result['derivative_minibatches'] > 0
    # Continuation labels add short suffix flights. There is still exactly one
    # full initial-state episode per update, with no candidate replay.
    assert calls.count(6) == 2 and all(0 < n <= 6 for n in calls)
    assert all(float(s['step']) == 2 for s in optimizer.state.values())
    assert trainer.completed_fits == 2


def test_nonfinite_backward_does_not_submit_adam_and_keeps_completed_fit(monkeypatch):
    value = value_module()
    policy, sim, initial = fixture()
    trainer = value.TaskValueTrainer(policy, task.initialize(policy, initial), 6,
        value.TaskValueConfig(window_steps=2, terminal_mode='oracle_full_state'))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    before = copy.deepcopy(policy.state_dict())
    # Inject at the actual parameter-gradient boundary, after supervised fit.
    handle = next(policy.parameters()).register_hook(lambda g: g * float('nan'))
    try:
        with pytest.raises(FloatingPointError):
            trainer.update(policy, optimizer, sim, initial, 6, task.TaskLossConfig(prediction_weight=0), gradient_clip=10)
    finally:
        handle.remove()
    assert_nested_equal(before, policy.state_dict())
    assert optimizer.state == {}
    assert trainer.completed_fits == 1
