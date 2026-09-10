from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

import response_value as value
from response_task import TaskLossConfig, initialize, rollout, step_costs
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


def samples(group='policy.memory', balance='minibatch'):
    assert hasattr(value, 'collect_derivative_samples'), 'missing continuation derivative supervision'
    policy, sim, initial = fixture()
    config = TaskLossConfig(steady_steps=2)
    record = value.collect_task_trajectory(policy, sim, initial, 6, 2, config)
    derivative_config = value.TaskValueConfig(window_steps=2, batch_size=8,
        derivative_state_groups=(group,), derivative_holdout_scenes=1,
        derivative_batch_size=2, derivative_balance_mode=balance)
    train, heldout = value.collect_derivative_samples(policy, sim, record, 6, config, derivative_config)
    return policy, sim, initial, config, record, derivative_config, train, heldout


@pytest.mark.parametrize('group,scale', [('policy.memory', 1.), ('physical.velocity', 5.)])
def test_true_labels_match_frozen_actor_continuation_and_fixed_coordinates(group, scale):
    policy, sim, _, config, record, _, train, heldout = samples(group)
    before = copy.deepcopy(policy.state_dict())
    from response_critic import critic_features
    critic = value.TaskValueCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    assert set(train.scene_ids.tolist()).isdisjoint(heldout.scene_ids.tolist())
    for batch in (train, heldout):
        for i, step in enumerate(batch.steps.tolist()):
            closed = value.select_derivative_samples(batch, torch.tensor([i])).closed
            leaf = getattr(getattr(closed, group.split('.')[0]), group.split('.')[1]).detach().requires_grad_(True)
            parent, field = group.split('.')
            closed = replace(closed, **{parent: replace(getattr(closed, parent), **{field: leaf})})
            continuation = rollout(policy, sim, closed, 6-step,
                parameters={n:p.detach() for n,p in policy.named_parameters()})
            cost = step_costs(continuation, config, start=step, horizon=6).sum()
            expected = torch.autograd.grad(cost, leaf)[0].flatten(1) * scale
            torch.testing.assert_close(batch.gradients[group][i:i+1], expected, rtol=1e-9, atol=1e-10)
        assert not batch.gradients[group].requires_grad
        predicted = value.predict_derivatives(critic, batch)
        for i, step in enumerate(batch.steps.tolist()):
            closed = value.select_derivative_samples(batch, torch.tensor([i])).closed
            parent, field = group.split('.')
            leaf = getattr(getattr(closed, parent), field).detach().requires_grad_(True)
            closed = replace(closed, **{parent: replace(getattr(closed, parent), **{field: leaf})})
            terminal = critic(critic_features(closed, step, 6)).sum()
            expected = torch.autograd.grad(terminal, leaf)[0].flatten(1)*scale
            torch.testing.assert_close(predicted[group][i:i+1], expected, rtol=1e-9, atol=1e-10)
    assert_nested_equal(before, policy.state_dict())
    assert all(p.requires_grad and p.grad is None for p in policy.parameters())


def test_cosine_log_norm_loss_separates_direction_and_scale_and_backpropagates():
    assert hasattr(value, 'derivative_losses'), 'missing cosine/log-norm losses'
    true = torch.tensor([[3., 4.], [0., 0.]], dtype=torch.float64)
    predicted = (true*100).requires_grad_(True)
    direction, magnitude = value.derivative_losses(predicted, true, epsilon=1e-8)
    assert float(direction) == pytest.approx(0, abs=1e-12)
    assert magnitude > 1
    magnitude.backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad[0].norm() > 0
    reverse, _ = value.derivative_losses(-true[:1], true[:1], epsilon=1e-8)
    assert float(reverse) == pytest.approx(2.)


def test_derivative_fit_preserves_actor_calibrates_once_and_restores_weights():
    policy, _, initial, _, record, config, train, heldout = samples(balance='fixed')
    trainer = value.TaskValueTrainer(policy, initialize(policy, initial), 6, config)
    actor = copy.deepcopy(policy.state_dict())
    old_target = copy.deepcopy(trainer.target.state_dict())
    # Training only the derivative loss must reach critic parameters via a
    # second-order graph, while leaving the Actor untouched.
    predicted = value.predict_derivatives(trainer.critic, train, create_graph=True)
    direction, magnitude = value.derivative_losses(predicted, train.gradients)
    grads = torch.autograd.grad(direction + magnitude, list(trainer.critic.parameters()), allow_unused=True)
    assert sum(float(g.norm()) for g in grads if g is not None) > 0
    fitted = trainer.fit(record, train, heldout)
    weights = copy.deepcopy(trainer.derivative_balance)
    assert weights is not None
    norms = fitted['derivative_gradient_contributions'][0]['raw_norms']
    assert weights['direction'] == pytest.approx(norms['value']/norms['direction'])
    assert weights['magnitude'] == pytest.approx(norms['value']/norms['magnitude'])
    assert fitted['derivative_train_samples'] == 2
    assert fitted['target_derivative_heldout_after']['samples'] == 2
    for n, p in trainer.target.state_dict().items():
        torch.testing.assert_close(p, (1-config.target_tau)*old_target[n]+config.target_tau*trainer.critic.state_dict()[n])
    trainer.fit(record, train, heldout)
    assert trainer.derivative_balance == weights
    restored = value.TaskValueTrainer(policy, initialize(policy, initial), 6, config)
    restored.load_state_dict(copy.deepcopy(trainer.state_dict()))
    assert restored.derivative_balance == weights
    assert_nested_equal(actor, policy.state_dict())
    assert all(p.grad is None for p in policy.parameters())
    stale = copy.deepcopy(trainer.state_dict()); stale['objective']='task-value-v2-memory-cosine-lognorm'
    with pytest.raises(ValueError, match='objective'):
        restored.load_state_dict(stale)


def test_failed_fit_rolls_back_derivative_balance_with_critic(monkeypatch):
    policy, _, initial, _, record, config, train, heldout = samples()
    trainer = value.TaskValueTrainer(policy, initialize(policy, initial), 6, config)
    before = copy.deepcopy(trainer.state_dict())
    original = trainer.optimizer.step
    def fail():
        original()
        raise RuntimeError('injected fit failure')
    monkeypatch.setattr(trainer.optimizer, 'step', fail)
    with pytest.raises(RuntimeError, match='injected'):
        trainer.fit(record, train, heldout)
    assert_nested_equal(before, trainer.state_dict())


def test_target_gradient_metrics_have_no_parameter_side_effects():
    from response_phase1 import derivative_gradient_metrics
    target = torch.tensor([[3.,4.], [0.,0.]])
    result = derivative_gradient_metrics(target*2, target)
    assert result['gradient_cosine'] == pytest.approx(1)
    assert result['norm_ratio'] == pytest.approx(2)
    assert result['lognorm_error'] == pytest.approx(torch.log(torch.tensor(2.)).item(), rel=1e-6)
    assert result['direction_samples'] == 1 and result['samples'] == 2


def test_derivative_configuration_routes_explicit_sampling_and_new_objective():
    from tools.train_response_control import parse_args
    from response_training import critic_configuration
    args = parse_args(['--optimizer', 'task-adam', '--scenario-mode', 'fixed-airframe',
        '--value-derivative-state-groups', 'policy.memory', '--value-derivative-boundaries', '50', '250', '450',
        '--value-derivative-holdout-scenes', '12',
        '--value-derivative-batch-size', '32'])
    config = critic_configuration(args)
    assert config.sampling_boundaries(500) == (50,250,450)
    assert config.derivative_state_groups == ('policy.memory',) and config.derivative_holdout_scenes == 12
    assert config.derivative_batch_size == 32
    assert value.TASK_VALUE_OBJECTIVE != 'task-value-v1-exact-step-cost-suffix'
    with pytest.raises(ValueError, match='horizon'):
        config.sampling_boundaries(250)
