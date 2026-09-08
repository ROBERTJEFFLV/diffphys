from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

import response_critic as critic
import response_task as task
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


def test_normal_motor_compensation_has_no_saturation_risk_or_safety_gradient():
    policy, sim, initial = fixture()
    trace = task.rollout(policy, sim, initial, 1)
    commands = trace.actions.new_tensor([0., .0741, .95, .975]).view(1, 1, 4).requires_grad_()
    risks = task.risk_components(replace(trace, actions=commands), task.RiskConfig())
    # Warning at .95, hard limit at 1: .975 is halfway through the warning band.
    torch.testing.assert_close(risks['saturation'], commands.new_tensor([[.0625]]))
    derivative = torch.autograd.grad(risks['saturation'].sum(), commands)[0]
    torch.testing.assert_close(derivative[..., :3], torch.zeros_like(derivative[..., :3]))
    assert derivative[..., 3].item() > 0
    for normal in (0., .0741, -.0741):
        value = task.risk_components(replace(trace, actions=torch.full_like(trace.actions, normal)),
                                     task.RiskConfig())['saturation']
        assert torch.equal(value, torch.zeros_like(value))


def test_critic_training_is_invariant_to_component_units_and_scales_are_frozen():
    policy, sim, initial = fixture()
    record = critic.collect_trajectory(policy, sim, initial, 4, task.TaskLossConfig())
    config = critic.CriticConfig(window_steps=2, batch_size=32, direction_samples=0)
    first = critic.CriticTrainer(policy, task.initialize(policy, initial), 4, config)
    second = critic.CriticTrainer(policy, task.initialize(policy, initial), 4, config)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    # All components describe the same normalized target, but have different units.
    units = record.returns.new_tensor([1.e8, 1.e4, 10., 1.])
    record = replace(record, inputs=torch.zeros_like(record.inputs), returns=torch.ones_like(record.returns))
    rng = torch.get_rng_state()
    first.fit(record)
    torch.set_rng_state(rng)
    report = second.fit(replace(record, returns=record.returns * units))
    torch.testing.assert_close(second.critic(record.inputs) / units, first.critic(record.inputs),
                               rtol=1.e-9, atol=1.e-10)
    torch.testing.assert_close(second.critic.output_scales, units)
    saved = copy.deepcopy(second.state_dict())
    second.fit(replace(record, returns=record.returns * units * 10))
    torch.testing.assert_close(second.critic.output_scales, units)
    second.load_state_dict(saved)
    assert_nested_equal(second.state_dict(), saved)
    assert report['critic_scale_source'] == 'first_train_trajectory'
    assert len(report['critic_component_normalized_mae_after']) == 4


def test_direction_supervision_uses_same_component_units_as_value_loss():
    scale = torch.tensor([1.e8, 1., .1, 1.e-3], dtype=torch.float64)
    plus = torch.tensor([[.2, .3, .4, .5]], dtype=torch.float64, requires_grad=True)
    minus = torch.zeros_like(plus)
    truth = torch.ones_like(plus)
    assert 'scales' in __import__('inspect').signature(critic.direction_ranking_loss).parameters
    actual = critic.direction_ranking_loss(plus * scale, minus, truth * scale, minus,
                                          scales=scale, temperature=.1)
    expected = torch.nn.functional.softplus(-plus / .1).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, plus, retain_graph=True)[0],
                               torch.autograd.grad(expected, plus)[0])


def test_old_risk_checkpoint_cannot_silently_reuse_incompatible_supervision():
    policy, _, initial = fixture()
    state = critic.CriticTrainer(policy, task.initialize(policy, initial), 4,
                                critic.CriticConfig(window_steps=2))
    saved = copy.deepcopy(state.state_dict())
    saved['objective'] = 'component-risk-to-go-v2-fixed-physical-scales'
    with pytest.raises(ValueError, match='objective|experiment'):
        state.load_state_dict(saved)
