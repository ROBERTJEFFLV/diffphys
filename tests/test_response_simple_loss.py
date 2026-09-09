from __future__ import annotations

from dataclasses import replace

import torch

import response_task as task
import response_critic as critic
from test_response_control import fixture


def test_training_loss_has_only_dense_tracking_first_action_difference_and_cvar():
    assert hasattr(task, 'training_step_costs'), 'missing separate minimal training objective'
    policy, sim, initial = fixture()
    trace = task.rollout(policy, sim, initial, 2)
    tensors = {}
    for name in ('positions', 'velocities', 'omegas', 'actions', 'action_deltas', 'omega_deltas'):
        tensors[name] = torch.full_like(getattr(trace, name), .2).requires_grad_()
    trace = replace(trace, **tensors)
    # Large legacy shaping weights must have no effect on the training graph.
    config = task.TaskLossConfig(action_weight=99., omega_delta_weight=88., steady_weight=77.,
                                prediction_weight=66., tail_weight=.5, tail_fraction=.5)
    costs = task.training_step_costs(trace, config)
    # Three axes of p/v/w plus four motors: original du convention is .5*du.
    expected_scene = 3*.2**2*(1.+.3+.1)+4*.1**2*.01
    torch.testing.assert_close(costs, costs.new_full((2, 2), expected_scene/2))
    loss = task.training_task_loss(trace, config)
    torch.testing.assert_close(loss, loss.new_tensor(expected_scene*1.5))
    gradients = torch.autograd.grad(loss, tuple(tensors.values()), allow_unused=True)
    for name, gradient in zip(tensors, gradients):
        if name in ('actions', 'omega_deltas'):
            assert gradient is None or not bool(gradient.any())
        else:
            assert torch.isfinite(gradient).all() and bool(gradient.any())


def test_training_windows_are_dense_and_keep_full_flight_cvar_definition():
    assert hasattr(task, 'training_step_costs')
    policy, sim, initial = fixture()
    config = task.TaskLossConfig(steady_steps=2, tail_weight=.5, tail_fraction=.5)
    full = task.rollout(policy, sim, initial, 6)
    first = task.rollout(policy, sim, initial, 3)
    second = task.rollout(policy, sim, first.end, 3)
    costs = task.training_step_costs(full, config)
    split = torch.cat([task.training_step_costs(first, config, start=0, horizon=6),
                       task.training_step_costs(second, config, start=3, horizon=6)])
    torch.testing.assert_close(costs, split)
    scene = costs.sum(0)
    torch.testing.assert_close(task.training_task_loss(full, config), scene.mean()+.5*scene.max())
    record = critic.collect_trajectory(policy, sim, initial, 6, config)
    torch.testing.assert_close(record.weights, task.risk_weights(scene, config))


def test_evaluation_cost_and_success_thresholds_keep_the_existing_definition():
    assert hasattr(task, 'training_task_loss')
    policy, sim, initial = fixture()
    trace = task.rollout(policy, sim, initial, 2)
    trace = replace(trace, positions=torch.full_like(trace.positions, .2),
        velocities=torch.zeros_like(trace.velocities), omegas=torch.zeros_like(trace.omegas),
        actions=torch.full_like(trace.actions, .4), action_deltas=torch.full_like(trace.action_deltas, .2),
        omega_deltas=torch.full_like(trace.omega_deltas, .3))
    config = task.TaskLossConfig(steady_steps=1)
    # Evaluation retains steady weighting, effort and omega-delta measurements.
    expected = (3*.2**2 + .0001*4*.2**2 + .01*4*.1**2 + .005*3*.3**2)*3*1.5
    metrics = task.trajectory_metrics(trace, config)
    assert abs(metrics['task_objective'] - expected) < 1.e-12
    assert metrics['success_count'] == 0
    minimal = task.training_task_loss(trace, config)
    torch.testing.assert_close(minimal, minimal.new_tensor((3*.2**2+.01*4*.1**2)*1.5))
