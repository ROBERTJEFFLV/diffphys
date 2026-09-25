"""Verify the declared backward rule, not a finite difference of its identity forward."""
from dataclasses import fields, replace
from pathlib import Path
import copy
import math
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import response_task as task
from env_l2f import L2FParams, L2FSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_adjoints import collect_boundary_rollout, backward_actor
from tools.train_response_control import parse_args
from test_episode_termination import actor, CountingSimulator, scheduled, flat_grads, select


def gradients(policy, objective):
    result = torch.autograd.grad(objective, list(policy.parameters()), allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                      for p, g in zip(policy.parameters(), result)])


def manual_prefix(policy, simulator, initial, horizon, alpha):
    """Independent scalar-episode recurrence using ordinary stop-gradient arithmetic."""
    rho = math.exp(-alpha * simulator.params.dt)
    closed = task.initialize(policy, initial)
    observations = [task.observation(initial)]
    data = {n: [] for n in ('actions', 'positions', 'velocities', 'omegas',
                            'action_deltas', 'omega_deltas')}
    for _ in range(horizon):
        def incoming(state, names):
            return replace(state, **{n: getattr(state, n).detach()
                                     + rho * (getattr(state, n) - getattr(state, n).detach())
                                     for n in names})
        before = incoming(closed.physical, task.PHYSICAL_DYNAMIC)
        history = incoming(closed.policy, task.POLICY_DYNAMIC)
        output = policy(task.observation(before), history)
        after = simulator.step(before, output.action)
        closed = task.ResponseClosedLoopState(after, output.next_state)
        for name, value in zip(data, (output.action, after.position, after.velocity,
                                     after.omega, output.action - before.previous_action,
                                     after.omega - before.omega)):
            data[name].append(value)
        observations.append(task.observation(after))
        if bool(simulator.terminated(after).all()):
            break
    return task.TaskTrajectory(end=closed, observations=torch.stack(observations),
                               initial=initial,
                               valid=torch.ones(len(data['actions']), 1, dtype=torch.bool),
                               **{n: torch.stack(v) for n, v in data.items()})


@pytest.mark.parametrize('alpha', [0., 1., 4.])
def test_identity_and_exponential_relative_time_rule(alpha):
    rho = math.exp(-alpha * .01)
    x = torch.tensor([.1, -.8, 4.], dtype=torch.float64, requires_grad=True)
    y = task._GradientDecay.apply(x, rho)
    assert torch.equal(y, x)
    torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0],
                               torch.full_like(x, rho), rtol=0, atol=0)
    # Each shared parameter injection keeps its local derivative; only the
    # elapsed steps between that injection and a later cost get rho**lag.
    theta = torch.tensor(.02, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(.3, dtype=torch.float64); states = []; gain = 1.04
    for _ in range(17):
        z = gain * task._GradientDecay.apply(z, rho) + theta
        states.append(z)
    actual = torch.autograd.grad(sum(v.square()/2 for v in states), theta)[0]
    expected = sum(float(states[t].detach()) * sum((gain*rho)**(t-k) for k in range(t+1))
                   for t in range(len(states)))
    assert float(actual) == pytest.approx(expected, rel=1e-13)


def test_every_adjoint_state_field_is_decayed_but_metadata_is_shared():
    from response_adjoints import dynamic_closed_state
    s = L2FSimulator().reset(2, dtype=torch.float64, horizon=2)
    closed, leaves, _ = dynamic_closed_state(task.initialize(actor(), s))
    rho = math.exp(-.01)
    decayed = task._decay_closed_state(closed, rho)
    values = [getattr(getattr(decayed, group), n)
              for group, names in [('physical', task.PHYSICAL_DYNAMIC), ('policy', task.POLICY_DYNAMIC)]
              for n in names]
    actual = torch.autograd.grad(sum(z.sum() for z in values), leaves)
    for z, g in zip(leaves, actual):
        torch.testing.assert_close(g, torch.full_like(z, rho), rtol=0, atol=0)
    assert decayed.physical.noise_tape is closed.physical.noise_tape
    assert decayed.physical.mass is closed.physical.mass
    assert {f.name for f in fields(decayed.policy)} == {'memory'}
    assert decayed.physical.step_index is closed.physical.step_index


@pytest.mark.parametrize('profile', ['l2f', 'raptor'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_forward_loss_masks_and_noise_unchanged_with_grad_enabled(profile, dtype):
    p = actor(dtype); sim = L2FSimulator(L2FParams(protocol=profile))
    s = sim.reset(4, seed=123, dtype=dtype, horizon=20)
    pos = s.position.clone(); vel = s.velocity.clone()
    pos[0] = 0; pos[0, 0] = s.position_limit[0] - 1e-6; vel[0, 0] = 1
    s = replace(s, position=pos, velocity=vel)
    rng = torch.get_rng_state().clone()
    a = task.rollout(p, sim, s, 20, time_decay=0.)
    b = task.rollout(p, sim, s, 20, time_decay=1.)
    for name in ('observations', 'actions', 'positions', 'velocities', 'omegas',
                 'action_deltas', 'omega_deltas', 'valid'):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    for group in ('physical', 'policy'):
        for field in fields(getattr(a.end, group)):
            assert torch.equal(getattr(getattr(a.end, group), field.name),
                               getattr(getattr(b.end, group), field.name)), field.name
    assert int(b.valid[:, 0].sum()) == 1
    ca = task.scenario_costs(a, task.TaskLossConfig())
    cb = task.scenario_costs(b, task.TaskLossConfig())
    assert torch.equal(ca, cb)
    assert torch.equal(task.risk_weights(ca, task.TaskLossConfig()),
                       task.risk_weights(cb, task.TaskLossConfig()))
    with torch.no_grad():
        c = task.rollout(p, sim, s, 20, time_decay=1.)
    assert torch.equal(c.actions, a.actions)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('alpha', [0., 1.])
@pytest.mark.parametrize('mode', ['full', 'windowed'])
def test_gradient_equals_independent_decayed_prefixes(alpha, mode):
    h = 20; initial = scheduled([1, 5, 6, 19, 20, 21], h); config = task.TaskLossConfig()
    a = actor(); b = copy.deepcopy(a); sim = CountingSimulator()
    record = collect_boundary_rollout(a, sim, initial, config, horizon=h, window_steps=5,
                                     backprop_mode=mode, time_decay=alpha)
    assert record.time_decay == alpha
    backward_actor(a, sim, record, config, gradient_scale=1.)
    costs = torch.cat([task.step_costs(manual_prefix(b, CountingSimulator(),
                        select(initial, slice(i, i+1)), h, alpha), config, horizon=h).sum(0)
                       for i in range(6)])
    weights = task.risk_weights(costs, config)
    torch.testing.assert_close(record.costs, costs, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(flat_grads(a), gradients(b, (weights*costs).sum()),
                               rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize('profile', ['l2f', 'raptor'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_full_windowed_and_window_size_agree(profile, dtype):
    p = actor(dtype); sim = L2FSimulator(L2FParams(protocol=profile))
    initial = sim.reset(4, seed=17, dtype=dtype, horizon=20); config = task.TaskLossConfig()
    pos = initial.position.clone(); vel = initial.velocity.clone()
    pos[0] = 0; pos[0, 0] = initial.position_limit[0] - 1e-6; vel[0, 0] = 1
    initial = replace(initial, position=pos, velocity=vel)
    ref = None
    for mode, window in [('full', 20), ('full', 5), ('windowed', 5), ('windowed', 10)]:
        policy = copy.deepcopy(p)
        record = collect_boundary_rollout(policy, sim, initial, config, horizon=20,
                        window_steps=window, backprop_mode=mode, time_decay=1.)
        checks = backward_actor(policy, sim, record, config, gradient_scale=1.)
        assert all(c['exact'] for c in checks['boundaries'])
        vec = flat_grads(policy)
        if ref is None:
            ref = vec; costs = record.costs.detach(); mask = record.valid
        else:
            tol = 2e-5 if dtype == torch.float32 else 2e-11
            torch.testing.assert_close(vec, ref, rtol=tol, atol=tol)
            torch.testing.assert_close(record.costs, costs, rtol=tol, atol=tol)
            assert torch.equal(mask, record.valid)


def test_one_step_parameter_gradient_is_not_discounted():
    p = actor(); sim = L2FSimulator(); s = sim.reset(3, dtype=torch.float64, horizon=1)
    a = task.rollout(p, sim, s, 1, time_decay=0.)
    b = task.rollout(p, sim, s, 1, time_decay=4.)
    config = task.TaskLossConfig()
    torch.testing.assert_close(gradients(p, task.task_loss(a, config)),
                               gradients(p, task.task_loss(b, config)), rtol=0, atol=0)


def test_termination_does_not_restart_or_skip_crossing():
    p = actor(); sim = CountingSimulator(); h = 50
    initial = scheduled([1, 19, 50, 51], h)
    seen = []
    hook = p.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[0]))
    trace = task.rollout(p, sim, initial, h, time_decay=1.)
    hook.remove()
    lengths = torch.tensor([1, 19, 50, 50])
    assert torch.equal(trace.valid.sum(0), lengths)
    assert torch.equal(trace.end.physical.step_index, lengths)
    assert sum(seen) == len(sim.ids) == int(lengths.sum())
    with torch.no_grad():
        for i in range(len(lengths)):
            single = manual_prefix(p, CountingSimulator(), select(initial, slice(i, i+1)), h, 1.)
            torch.testing.assert_close(trace.end.policy.memory[i:i+1], single.end.policy.memory,
                                       rtol=1e-11, atol=1e-12)
    task.task_loss(trace, task.TaskLossConfig()).backward()
    assert torch.isfinite(flat_grads(p)).all()


@pytest.mark.parametrize('alpha', [-1., float('nan'), float('inf')])
def test_invalid_time_decay_rejected(alpha):
    p = actor(); sim = L2FSimulator(); s = sim.reset(1, dtype=torch.float64, horizon=1)
    with pytest.raises(ValueError, match='time_decay'):
        task.rollout(p, sim, s, 1, time_decay=alpha)
    with pytest.raises(SystemExit):
        parse_args(['--time-decay', str(alpha)])


def test_default_cli_and_checkpoint_binding(tmp_path):
    from test_failure_cost_compatibility import training_args, run
    from response_training import evaluate_checkpoint
    assert parse_args([]).time_decay == 1.
    path = tmp_path/'run'
    run(training_args(path, updates=1, extra=('--time-decay', '0.3')))
    saved = torch.load(path/'latest.pt', weights_only=True)
    assert saved['binding']['time_decay'] == .3
    assert saved['binding']['algorithm'] == 'time-decayed-bptt-adam'
    with pytest.raises(ValueError, match='configuration'):
        run(training_args(path, updates=2, extra=('--resume', str(path/'latest.pt'), '--time-decay', '0')))
    args = training_args(tmp_path/'eval'); args.checkpoint = path/'latest.pt'
    # EVAL is forward-only; no new policy/environment fields or migration.
    result = evaluate_checkpoint(args)
    assert result['finite']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_surrogate_gradient_and_device():
    p = actor(torch.float32).cuda(); sim = L2FSimulator()
    s = sim.reset(4, device='cuda', dtype=torch.float32, horizon=20)
    traces = [task.rollout(p, sim, s, 20, time_decay=alpha) for alpha in (0., 1.)]
    assert torch.equal(traces[0].actions, traces[1].actions)
    task.task_loss(traces[1], task.TaskLossConfig()).backward()
    assert flat_grads(p).is_cuda and torch.isfinite(flat_grads(p)).all()
