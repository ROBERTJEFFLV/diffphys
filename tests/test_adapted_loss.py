"""Independent examples for smooth norms/attitude transitions, equal-scene scoring."""
from dataclasses import replace
import copy
import math
from types import SimpleNamespace

import pytest
import torch

import response_task as task
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig
from env_raptor import RaptorSimulator
from test_episode_termination import actor, CountingSimulator, scheduled, flat_grads


def config(**changes):
    # Explicit numerical fixtures, NOT recommended production settings.
    values = dict(epsilon_p=.5, epsilon_a=.25, lambda_R=.4)
    values.update(changes)
    return task.TaskLossConfig(**values)


def trace_fixture(steps=3, scenes=2, dtype=torch.float64):
    q = torch.zeros(steps, scenes, 4, dtype=dtype)
    q[..., 0] = 1
    return SimpleNamespace(
        pre_positions=torch.zeros(steps, scenes, 3, dtype=dtype, requires_grad=True),
        pre_orientations=q.requires_grad_(),
        post_orientations=q.detach().clone().requires_grad_(),
        positions=torch.zeros(steps, scenes, 3, dtype=dtype),
        velocities=torch.zeros(steps, scenes, 3, dtype=dtype),
        omegas=torch.zeros(steps, scenes, 3, dtype=dtype),
        actions=torch.zeros(steps, scenes, 4, dtype=dtype),
        action_deltas=torch.zeros(steps, scenes, 4, dtype=dtype, requires_grad=True),
        valid=torch.ones(steps, scenes, dtype=torch.bool),
        initial=SimpleNamespace(position_limit=torch.full((scenes,), 10., dtype=dtype)),
    )


def test_hand_computed_vector_norm_attitude_transition_and_component_total():
    trace = trace_fixture(1, 1)
    trace.pre_positions = torch.tensor([[[3., 4., 0.]]], dtype=torch.float64)
    trace.pre_orientations = torch.tensor([[[math.sqrt(.5), math.sqrt(.5), 0., 0.]]], dtype=torch.float64)
    trace.action_deltas = torch.tensor([[[.3, .4, 0., 0.]]], dtype=torch.float64)
    expected = math.sqrt(25.25)-.5 + .4 + math.sqrt(.3125)-.25
    assert task.task_loss(trace, config()).item() == pytest.approx(expected)
    components = task.task_loss_components(trace, config())
    assert set(components) == {'position', 'attitude_delta', 'action_delta', 'dead', 'terminal'}
    assert sum(components.values()) == pytest.approx(expected)
    assert components['position'] == pytest.approx(math.sqrt(25.25)-.5)
    assert components['attitude_delta'] == pytest.approx(.4)
    assert components['action_delta'] == pytest.approx(math.sqrt(.3125)-.25)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_zero_error_has_zero_cost_and_finite_zero_gradient(dtype):
    trace = trace_fixture(dtype=dtype)
    loss = task.task_loss(trace, config())
    assert loss.item() == 0
    gradients = torch.autograd.grad(loss, (trace.pre_positions, trace.pre_orientations,
                                          trace.post_orientations, trace.action_deltas))
    assert all(torch.isfinite(g).all() and not g.any() for g in gradients)


@pytest.mark.parametrize('width', [3, 4])
def test_smooth_vector_norm_matches_analytic_gradient_and_finite_difference(width):
    x = torch.linspace(-.3, .6, width, dtype=torch.float64, requires_grad=True)
    epsilon = .25
    rho = task.smooth_l2(x, epsilon)
    gradient, = torch.autograd.grad(rho, (x,))
    expected = x.detach() / math.sqrt(float(x.detach().square().sum()) + epsilon**2)
    torch.testing.assert_close(gradient, expected, rtol=1e-13, atol=1e-13)
    direction = torch.linspace(.2, .7, width, dtype=x.dtype)
    delta = 1e-5
    finite_difference = (task.smooth_l2(x.detach()+delta*direction, epsilon)
                         - task.smooth_l2(x.detach()-delta*direction, epsilon)) / (2*delta)
    assert float(gradient @ direction) == pytest.approx(float(finite_difference), rel=1e-8, abs=1e-10)
    tiny = torch.full_like(x, 1e-10)
    assert task.smooth_l2(tiny, epsilon).item() > 0  # Avoid subtractive cancellation.


@pytest.mark.parametrize('dtype,epsilon', [(torch.float32, 1e-30), (torch.float32, 1e20),
                                         (torch.float64, 1e-200), (torch.float64, 1e200)])
def test_finite_positive_scale_cannot_make_zero_error_gradient_nonfinite(dtype, epsilon):
    x = torch.zeros(4, dtype=dtype, requires_grad=True)
    cost = task.smooth_l2(x, epsilon)
    gradient, = torch.autograd.grad(cost, (x,))
    assert cost.item() == 0
    assert torch.isfinite(gradient).all() and not gradient.any()


@pytest.mark.parametrize('name', ['epsilon_p', 'epsilon_a', 'lambda_R'])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_fixed_scales_and_rotation_weight_must_be_finite_positive(name, value):
    values = dict(epsilon_p=.5, epsilon_a=.25, lambda_R=.4)
    values[name] = value
    with pytest.raises(ValueError):
        task.TaskLossConfig(**values)


@pytest.mark.parametrize('angle,expected', [(0., 0.), (math.pi/2, .4), (math.pi, .8)])
def test_relative_rotation_sign_symmetry_and_half_turn_stationary_point(angle, expected):
    trace = trace_fixture(1, 1)
    theta = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
    q = torch.stack((torch.cos(theta/2), torch.sin(theta/2), theta*0, theta*0)).reshape(1, 1, 4)
    trace.pre_orientations = q
    loss = task.task_loss(trace, config())
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-14)
    derivative, = torch.autograd.grad(loss, (theta,), retain_graph=True)
    if angle == math.pi:
        assert abs(float(derivative)) < 1e-14
    trace.pre_orientations = -q
    assert float(task.task_loss(trace, config()).detach()) == pytest.approx(expected, abs=1e-14)


def test_unchanging_yaw_has_no_attitude_transition_cost():
    trace = trace_fixture(1, 1)
    trace.pre_orientations = torch.tensor([[[.6, 0., 0., .8]]], dtype=torch.float64)
    trace.post_orientations = trace.pre_orientations.clone()
    assert task.task_loss(trace, config()).item() == 0


def test_loss_uses_pre_action_truth_not_post_action_states_or_noisy_observations():
    policy = actor()
    initial = scheduled([3], 3)
    trace = task.rollout(policy, CountingSimulator(), initial, 3)
    assert trace.pre_positions[:, 0, 0].tolist() == [0., 1., 2.]
    assert trace.positions[:, 0, 0].tolist() == [1., 2., 3.]
    components = task.task_loss_components(trace, config())
    assert components['position'] == pytest.approx((math.sqrt(1.25)-.5+math.sqrt(4.25)-.5)/3)
    altered = replace(trace, observations=trace.observations+100,
                      velocities=trace.velocities+100, omegas=trace.omegas+100)
    assert torch.equal(task.step_costs(trace, config()), task.step_costs(altered, config()))


@pytest.mark.parametrize('horizon', [8, 500])
def test_failure_accounting_includes_first_middle_and_final_transition(horizon):
    trace = task.rollout(actor(), CountingSimulator(), scheduled([1, horizon//2, horizon, horizon+1], horizon), horizon)
    components = task.step_cost_components(trace, config())
    lengths = [1, horizon//2, horizon, horizon]
    expected_dead = [(horizon-t)*3/horizon if i < 3 else 0 for i, t in enumerate(lengths)]
    expected_terminal = [200/horizon]*3 + [0]
    torch.testing.assert_close(components['dead'].sum(0), torch.tensor(expected_dead, dtype=torch.float64))
    torch.testing.assert_close(components['terminal'].sum(0), torch.tensor(expected_terminal, dtype=torch.float64))
    assert all(not value[~trace.valid].any() for value in components.values())


def test_invalid_padding_is_sanitized_before_math_valid_nonfinite_is_rejected():
    trace = trace_fixture(2, 1)
    trace.valid[1] = False
    for name in ('pre_positions', 'pre_orientations', 'post_orientations', 'action_deltas'):
        value = getattr(trace, name).detach().clone()
        value[1] = float('nan')
        setattr(trace, name, value.requires_grad_())
    loss = task.task_loss(trace, config())
    gradients = torch.autograd.grad(loss, (trace.pre_positions, trace.pre_orientations,
                                          trace.post_orientations, trace.action_deltas))
    assert loss.item() == 0 and all(torch.isfinite(g).all() for g in gradients)
    trace.pre_positions = trace.pre_positions.detach().clone()
    trace.pre_positions[0] = float('nan')
    with pytest.raises(FloatingPointError):
        task.task_loss(trace, config())


def test_command_zero_placeholder_and_fifty_step_boundary_preserve_loss_and_gradient():
    initial = scheduled([71, 71], 70)
    policy = actor()
    full = task.rollout(policy, CountingSimulator(), initial, 70)
    first = task.rollout(policy, CountingSimulator(), initial, 50)
    second = task.rollout(policy, CountingSimulator(), first.end, 20)
    torch.testing.assert_close(full.action_deltas[0], full.actions[0], rtol=0, atol=0)
    torch.testing.assert_close(second.action_deltas[0], second.actions[0]-first.actions[-1], rtol=0, atol=0)
    full_cost = task.step_costs(full, config())
    parts = torch.cat((task.step_costs(first, config(), start=0, horizon=70),
                       task.step_costs(second, config(), start=50, horizon=70)))
    torch.testing.assert_close(full_cost, parts, rtol=0, atol=0)
    parameters = tuple(policy.parameters())
    a = torch.autograd.grad(full_cost.mean(1).sum(), parameters)
    b = torch.autograd.grad(parts.mean(1).sum(), parameters)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-12)


def test_uniform_scene_weights_do_not_depend_on_cost_order_or_failure_constants():
    costs = torch.tensor([1., 5., 100., 2.], dtype=torch.float64, requires_grad=True)
    weights = task.uniform_scene_weights(costs)
    assert not weights.requires_grad
    assert weights.tolist() == [.25]*4
    assert torch.equal(weights, task.uniform_scene_weights(costs.flip(0)))
    policy = actor()
    trace = task.rollout(policy, CountingSimulator(), scheduled([1, 8, 9], 8), 8)
    low = config()
    high = replace(low, dead_cost=17., terminal_cost=900.)
    a, b = task.task_loss(trace, low), task.task_loss(trace, high)
    assert b.item() > a.item()
    ga = torch.autograd.grad(a, tuple(policy.parameters()), retain_graph=True)
    gb = torch.autograd.grad(b, tuple(policy.parameters()))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize('decay', [0., 1.])
def test_uncapped_equal_group_probe_matches_independent_pooled_vjp(decay):
    initial = scheduled([9]*64, 8)
    a = actor()
    b = copy.deepcopy(a)
    simulator = CountingSimulator()
    record = collect_rollout(a, simulator, initial, config(), horizon=8, time_decay=decay,
                             group_config=GroupBalanceConfig(clip_norm=1e9))
    assert record.weights.tolist() == [1/64]*64
    backward_actor(a, simulator, record, config(), gradient_scale=.1)
    trace = task.rollout(b, CountingSimulator(), initial, 8, time_decay=decay)
    objective = .1*task.scenario_costs(trace, config()).mean()
    objective.backward()
    torch.testing.assert_close(flat_grads(a), flat_grads(b), rtol=2e-10, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_new_cost_keeps_float32_and_all_actor_gradients_finite():
    initial = scheduled([1, 3, 9], 8, torch.float32).to('cuda', torch.float32)
    policy = actor(torch.float32).cuda()
    trace = task.rollout(policy, CountingSimulator(), initial, 8)
    loss = task.task_loss(trace, config())
    assert loss.dtype == torch.float32 and loss.device.type == 'cuda'
    loss.backward()
    assert torch.isfinite(flat_grads(policy)).all()


@pytest.mark.parametrize('old_field', ['velocity_weight', 'omega_weight', 'action_weight',
                                    'omega_delta_weight', 'huber_delta', 'tail_weight',
                                    'tail_fraction', 'steady_weight', 'steady_steps'])
def test_deleted_loss_fields_are_not_silently_accepted(old_field):
    with pytest.raises(TypeError):
        task.TaskLossConfig(epsilon_p=.5, epsilon_a=.25, lambda_R=.4, **{old_field: .2})


def test_objective_version_is_bound_and_old_checkpoint_cannot_resume(tmp_path):
    from response_training import binding, _checkpoint, require_reference_checkpoint
    from response_policy import ResponsePolicyConfig
    from tools.train_response_control import parse_args
    args = parse_args(['--device', 'cpu', '--epsilon-p', '.5', '--epsilon-a', '.25', '--lambda-R', '.4'])
    policy = actor()
    args.dtype = 'float64'
    bound = binding(args, policy.config, config())
    assert bound['protocol']['task_objective'] == task.TASK_OBJECTIVE_VERSION
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    checkpoint = _checkpoint(policy, optimizer, dict(updates=0), bound)
    require_reference_checkpoint(checkpoint)
    old_tilt = copy.deepcopy(checkpoint)
    old_tilt['binding']['protocol']['task_objective'] = 'pre-action-smooth-vector-norm-tilt-equal-scenes-v1'
    with pytest.raises(ValueError, match='objective'):
        require_reference_checkpoint(old_tilt)
    old = copy.deepcopy(checkpoint)
    old['binding']['protocol'].pop('task_objective')
    old['binding']['protocol']['loss'] = {'position_weight': 1., 'tail_weight': .5}
    with pytest.raises(ValueError, match='objective'):
        require_reference_checkpoint(old)


def test_old_objective_only_imports_weights_not_adam_best_score_or_sampling_index(tmp_path):
    from test_reference_training import args_for, run
    from response_training import evaluate_checkpoint
    source, imported = tmp_path/'source', tmp_path/'imported'
    run(args_for(source, 1))
    saved = torch.load(source/'latest.pt', weights_only=True)
    saved['binding']['protocol'].pop('task_objective')
    saved['binding']['protocol']['loss'] = {'position_weight': 1., 'tail_weight': .5}
    saved['progress']['best_score'] = -999.
    legacy = tmp_path/'legacy.pt'
    torch.save(saved, legacy)
    with pytest.raises(ValueError, match='objective'):
        run(args_for(tmp_path/'bad_resume', 2, ('--resume', str(legacy))))
    args = args_for(tmp_path/'bad_eval', 0); args.checkpoint = legacy
    with pytest.raises(ValueError, match='objective'):
        evaluate_checkpoint(args)
    run(args_for(imported, 0, ('--init-checkpoint', str(legacy))))
    new = torch.load(imported/'latest.pt', weights_only=True)
    assert new['model_sha256'] == saved['model_sha256']
    assert new['optimizer']['state'] == {} and new['next_update'] == 0
    assert new['progress']['best_score'] >= 0
    assert new['progress']['initialization']['weights_only']
    assert new['binding']['protocol']['task_objective'] == task.TASK_OBJECTIVE_VERSION


def test_streaming_components_match_the_full_eval_objective_and_physical_metrics():
    from response_training import evaluate
    initial = scheduled([1, 51, 71]*32, 70)
    policy = actor()
    record = collect_rollout(policy, CountingSimulator(), initial, config(), horizon=70)
    actual = evaluate(policy, CountingSimulator(), initial, 70, config())
    for key in ('task_objective', 'position_rms', 'velocity_rms', 'omega_rms', 'motor_saturation_fraction'):
        assert record.metrics[key] == pytest.approx(actual[key], rel=1e-12, abs=1e-12)
    for name, value in record.metrics['task_components'].items():
        assert value == pytest.approx(actual['task_components'][name], rel=1e-12, abs=1e-12)
    assert sum(actual['task_components'].values()) == pytest.approx(actual['task_objective'])
    record.probe.close()
