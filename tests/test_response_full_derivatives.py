import copy
from dataclasses import replace

import pytest
import torch

import response_value as value
from response_adjoints import STATE_SCALES, gradient_coordinates
from response_phase1 import dynamic_closed_state
from response_critic import critic_features
from response_task import TaskLossConfig, initialize
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


def setup():
    policy, sim, initial = fixture()
    loss = TaskLossConfig(steady_steps=2)
    record = value.collect_task_trajectory(policy, sim, initial, 6, 2, loss)
    config = value.TaskValueConfig(window_steps=2, batch_size=8,
        derivative_state_groups=tuple(STATE_SCALES), derivative_holdout_scenes=1,
        derivative_batch_size=2)
    labels = value.collect_boundary_adjoints(policy, sim, record, 6, 2, loss)
    train, held = value.collect_derivative_samples(policy, sim, record, 6, loss, config, adjoints=labels)
    trainer = value.TaskValueTrainer(policy, initialize(policy, initial), 6, config)
    return policy, sim, initial, record, config, labels, train, held, trainer


def test_full_supervision_covers_every_scene_and_nonterminal_boundary_in_fixed_coordinates():
    assert 'derivative_state_groups' in value.TaskValueConfig.__dataclass_fields__, 'missing multiple state groups'
    _, _, _, record, config, labels, train, held, trainer = setup()
    assert set(train.scene_ids.tolist()).isdisjoint(held.scene_ids.tolist())
    for pool in (train, held):
        assert set(pool.steps.tolist()) == {2,4}
        assert len(pool.steps) == 2
        assert set(pool.gradients) == set(STATE_SCALES)
        assert pool.metadata['actor_sha256'] == labels.actor_sha256
        predicted = value.predict_derivatives(trainer.critic, pool, create_graph=True)
        for i, step in enumerate(pool.steps.tolist()):
            sample = value.select_derivative_samples(pool, torch.tensor([i]))
            state, leaves, names = dynamic_closed_state(sample.closed)
            raw = torch.autograd.grad(trainer.critic(critic_features(state, step, 6)).sum(),
                                       leaves, allow_unused=True)
            for name, leaf, grad in zip(names, leaves, raw):
                if name not in STATE_SCALES:
                    continue
                expected = gradient_coordinates(name, torch.zeros_like(leaf) if grad is None else grad, state)
                torch.testing.assert_close(predicted[name][i:i+1], expected)
                true = labels.gradients[step][name][pool.scene_ids[i:i+1]]
                torch.testing.assert_close(pool.gradients[name][i:i+1], gradient_coordinates(name, true, state))
        direction, magnitude = value.derivative_losses(predicted, pool.gradients)
        grads = torch.autograd.grad(direction+magnitude, list(trainer.critic.parameters()), allow_unused=True)
        assert all(torch.isfinite(g).all() for g in grads if g is not None)
        assert sum(float(g.norm()) for g in grads if g is not None) > 0


def test_minibatch_balance_is_remeasured_and_failed_fit_restores_all_critic_state(monkeypatch):
    assert 'derivative_state_groups' in value.TaskValueConfig.__dataclass_fields__, 'missing multiple state groups'
    policy, _, _, record, config, _, train, held, trainer = setup()
    actor = copy.deepcopy(policy.state_dict())
    fitted = trainer.fit(record, train, held)
    assert trainer.balance_updates == 2  # 14 value samples, batch size 8
    rows = fitted['derivative_gradient_contributions']
    assert len(rows) == 2
    for row in rows:
        norms, weights = row['raw_norms'], row['weights']
        assert weights['direction'] == pytest.approx(norms['value']/norms['direction'])
        assert weights['magnitude'] == pytest.approx(norms['value']/norms['magnitude'])
    assert 'physical.position' in fitted['target_derivative_heldout_after']['state_groups']
    before = copy.deepcopy(trainer.state_dict())
    original = trainer.optimizer.step
    def fail():
        original()
        raise RuntimeError('injected fit failure')
    monkeypatch.setattr(trainer.optimizer, 'step', fail)
    with pytest.raises(RuntimeError, match='injected'):
        trainer.fit(record, train, held)
    assert_nested_equal(before, trainer.state_dict())
    assert_nested_equal(actor, policy.state_dict())
    assert all(p.grad is None for p in policy.parameters())


def test_near_zero_supervision_norm_is_not_amplified_by_epsilon():
    assert 'derivative_state_groups' in value.TaskValueConfig.__dataclass_fields__, 'missing multiple state groups'
    *_, trainer = setup()
    p = next(trainer.critic.parameters())
    v, d, m = p.sum(), p.sum()*1e-20, p.sum()*0.
    row = trainer._balance_derivative_losses(v,d,m)
    assert row['weights']['direction'] == 0 and row['weights']['magnitude'] == 0
    assert set(row['inactive']) == {'direction','magnitude'}


def test_derivative_metric_reports_negative_fraction_and_ratio_quantiles():
    from response_phase1 import derivative_gradient_metrics
    true = torch.tensor([[1.,0.],[1.,0.],[0.,0.]])
    pred = torch.tensor([[-1.,0.],[3.,0.],[0.,0.]])
    result = derivative_gradient_metrics(pred,true)
    assert 'norm_ratio_p90' in result, 'missing per-group distribution probes'
    assert result['negative_cosine_fraction'] == pytest.approx(.5)
    assert result['norm_ratio_median'] == pytest.approx(2.)
    assert result['norm_ratio_p90'] == pytest.approx(2.8)
    assert result['zero_true_samples'] == 1
