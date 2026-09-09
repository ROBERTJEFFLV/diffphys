from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

import response_critic as critic
import response_task as task
from env_l2f import CAPABILITY_BOUNDS, L2FParams
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


@pytest.mark.parametrize('multiplier', [.4, 100.])
def test_capability_features_ignore_common_mass_force_inertia_units(multiplier):
    policy, sim, initial = fixture()
    initial = replace(initial, external_force=torch.ones_like(initial.external_force) * .01)
    closed = task.initialize(policy, initial)
    changed = replace(closed, physical=replace(initial,
        mass=initial.mass*multiplier, external_force=initial.external_force*multiplier,
        inertia_x=initial.inertia_x*multiplier, inertia_y=initial.inertia_y*multiplier,
        inertia_z=initial.inertia_z*multiplier,
        thrust_coeff_c0=initial.thrust_coeff_c0*multiplier,
        thrust_coeff_c1=initial.thrust_coeff_c1*multiplier,
        thrust_coeff_c2=initial.thrust_coeff_c2*multiplier))
    torch.testing.assert_close(critic.critic_features(closed, 2, 6), critic.critic_features(changed, 2, 6))
    # Verify that this change of size really preserves physical response,
    # rather than only checking that the feature selector ignores the fields.
    first, second = initial, changed.physical
    for _ in range(3):
        actions = initial.motor.new_tensor([[.1, -.1, .2, -.2]]).expand_as(initial.motor)
        first, second = sim.step(first, actions), sim.step(second, actions)
        for name in ('position', 'velocity', 'omega', 'rotation', 'motor'):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))


def test_feature_tail_is_existing_log_capability_then_acceleration_over_g_and_time():
    policy, _, initial = fixture()
    bounds = initial.position.new_tensor(CAPABILITY_BOUNDS)
    values = (bounds[:, 0] * bounds[:, 1]).sqrt()
    changes = {name: torch.full_like(getattr(initial, name), float(value)) for name, value in zip(
        ('thrust_to_weight', 'alpha_roll_max', 'eta_yaw', 'jz_over_jxy', 'motor_time_rising', 'motor_time_falling'), values)}
    initial = replace(initial, **changes, external_force=initial.mass[:, None] * L2FParams().gravity
                      * initial.position.new_tensor([.1, -.2, .3]))
    features = critic.critic_features(task.initialize(policy, initial), 2, 4)
    torch.testing.assert_close(features[:, -10:-4], features.new_zeros((2, 6)), atol=1.e-6, rtol=0)
    torch.testing.assert_close(features[:, -4:], features.new_tensor([[.1, -.2, .3, .5]]).expand(2, 4))


@pytest.mark.parametrize('horizon', [2, 50, 500, 1000])
def test_mean_risk_labels_remove_only_the_remaining_step_factor(horizon):
    assert hasattr(critic, 'mean_future_risks'), 'missing mean-risk target conversion'
    per_step = torch.tensor([.2, 1., 3., 0.], dtype=torch.float64)
    sums = torch.arange(horizon, -1, -1, dtype=torch.float64)[:, None, None] * per_step[None, None, :]
    means = critic.mean_future_risks(sums)
    torch.testing.assert_close(means[:-1], per_step.expand(horizon, 1, 4))
    assert torch.equal(means[-1], torch.zeros_like(means[-1]))


def test_linear_head_predicts_signed_mean_risk_without_calibration_or_inverse_transform():
    net = critic.RiskToGoCritic(2).double()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.network[-1].bias.fill_(-2.)
    torch.testing.assert_close(net(torch.ones(2, 2, dtype=torch.float64)), torch.full((2, 4), -2., dtype=torch.float64))
    assert list(net.named_buffers()) == []


def test_fit_learns_mean_labels_directly_and_requires_no_calibration():
    policy, sim, initial = fixture()
    trainer = critic.CriticTrainer(policy, task.initialize(policy, initial), 4,
                                  critic.CriticConfig(window_steps=2, direction_samples=0))
    record = critic.collect_trajectory(policy, sim, initial, 4, task.TaskLossConfig())
    with torch.no_grad():
        for p in trainer.critic.parameters():
            p.zero_()
    sums = torch.arange(4, -1, -1, dtype=record.returns.dtype)[:, None, None].expand(5, 2, 4)
    record = replace(record, returns=sums)
    report = trainer.fit(record)
    # Four nonterminal rows have mean-risk target 1; Z_H has target 0.
    assert report['critic_loss_before'] == pytest.approx(.4)
    assert report['critic_loss_after'] < report['critic_loss_before']
    assert trainer.completed_fits == 1


def test_actor_restores_total_future_risk_and_its_gradient_at_each_boundary():
    policy, sim, initial = fixture()
    config = task.TaskLossConfig()
    record = critic.collect_trajectory(policy, sim, initial, 6, config)
    class MeanRisk(torch.nn.Module):
        def forward(self, inputs):
            return inputs[:, :1].expand(-1, 4)
    target = MeanRisk()
    actual = critic.accumulate_actor_gradients(policy, target, sim, initial, 6, 2, config,
        record.weights, baseline_returns=record.returns, separate_objectives=True)['objective_gradients']
    weights = torch.stack([task.risk_weights(record.returns[0, :, j], config) for j in range(4)], -1)
    closed = task.initialize(policy, initial)
    terms = []
    for start in (0, 2, 4):
        trace = task.rollout(policy, sim, critic.detach_closed_state(closed), 2)
        risk = torch.stack(tuple(task.risk_components(trace, task.RiskConfig()).values()), -1).sum(0)
        if start < 4:
            # Independent oracle: explicit remaining count times mean state risk.
            risk = risk + (4-start) * (trace.end.physical.position[:, :1]/5.).expand(-1, 4)
        terms.append((weights * risk).sum(0)/3)
        closed = trace.end
    parameters = list(policy.parameters())
    objectives = torch.stack(terms).sum(0)
    for j in range(4):
        grad = torch.autograd.grad(objectives[j], parameters, retain_graph=j<3, allow_unused=True)
        expected = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p,g in zip(parameters,grad)])
        torch.testing.assert_close(actual[j+1], expected, rtol=1.e-9, atol=1.e-11)


@pytest.mark.parametrize('objective', ['component-risk-v3-warning-saturation-train-scaled', 'component-risk-v4-train-affine-asinh'])
def test_old_cumulative_critic_checkpoint_is_rejected_before_mutating_state(objective):
    policy, _, initial = fixture()
    trainer = critic.CriticTrainer(policy, task.initialize(policy, initial), 4, critic.CriticConfig(window_steps=2))
    before = copy.deepcopy(trainer.state_dict())
    old = copy.deepcopy(before)
    old['objective'] = objective
    with pytest.raises(ValueError, match='objective|experiment'):
        trainer.load_state_dict(old)
    assert_nested_equal(trainer.state_dict(), before)


def test_terminal_zero_convention_cannot_hide_nonfinite_raw_supervision():
    policy, sim, initial = fixture()
    trainer = critic.CriticTrainer(policy, task.initialize(policy, initial), 4, critic.CriticConfig(window_steps=2))
    record = critic.collect_trajectory(policy, sim, initial, 4, task.TaskLossConfig())
    bad = record.returns.clone()
    bad[-1] = float('nan')
    before = copy.deepcopy(trainer.state_dict())
    with pytest.raises(FloatingPointError, match='supervision'):
        trainer.fit(replace(record, returns=bad))
    assert_nested_equal(trainer.state_dict(), before)
