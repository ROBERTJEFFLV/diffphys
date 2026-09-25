"""Independent episode termination, including the terminal transition and its BPTT."""
from dataclasses import fields, replace
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_l2f import L2FParams, L2FSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import (ResponseClosedLoopState, TaskTrajectory, TaskLossConfig,
                           initialize, observation, rollout, step_costs, risk_weights,
                           trajectory_metrics, FlightStatistics, reference_episode_metrics)
from response_adjoints import collect_boundary_rollout, backward_actor


def select(state, index):
    return type(state)(**{f.name: getattr(state, f.name)[index] for f in fields(state)})


def actor(dtype=torch.float64):
    torch.manual_seed(17)
    return ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype=dtype)


class CountingSimulator(L2FSimulator):
    """Known per-row failure times; fail loudly on ANY post-termination call."""
    def __init__(self):
        super().__init__()
        self.ids = []

    def step(self, state, action):
        assert not self.terminated(state).any(), 'physics called after termination'
        self.ids.extend(state.mass.tolist())
        position = torch.cat((state.position[:, :1] + 1, state.position[:, 1:]), -1)
        return replace(state, position=position, velocity=state.velocity + .02 * action[:, :3],
                       omega=.95 * state.omega + .1 * action[:, :3],
                       motor=.9 * state.motor + .1 * (action + 1) / 2,
                       previous_action=action, step_index=state.step_index + 1)


def scheduled(lengths, horizon, dtype=torch.float64):
    s = L2FSimulator().reset(len(lengths), seed=7, dtype=dtype, horizon=horizon)
    return replace(s, position=torch.zeros_like(s.position), velocity=torch.zeros_like(s.velocity),
                   omega=torch.zeros_like(s.omega), position_limit=s.mass.new_tensor(lengths) - .5,
                   velocity_limit=torch.full_like(s.mass, 1000), omega_limit=torch.full_like(s.mass, 1000),
                   mass=torch.arange(1, len(lengths)+1, dtype=dtype))


def manual_prefix(policy, simulator, initial, horizon):
    """Independent serial implementation: no padding and no production rollout()."""
    closed = initialize(policy, initial)
    obs = [observation(initial)]
    actions, positions, velocities, omegas, da, dw = [], [], [], [], [], []
    for _ in range(horizon):
        before = closed.physical
        output = policy(obs[-1], closed.policy)
        after = simulator.step(before, output.action)
        closed = ResponseClosedLoopState(after, output.next_state)
        actions.append(output.action); positions.append(after.position)
        velocities.append(after.velocity); omegas.append(after.omega)
        da.append(output.action - before.previous_action); dw.append(after.omega - before.omega)
        obs.append(observation(after))
        if bool(simulator.terminated(after).all()):
            break
    return TaskTrajectory(closed, torch.stack(obs), torch.stack(actions), torch.stack(positions),
                          torch.stack(velocities), torch.stack(omegas), torch.stack(da), torch.stack(dw),
                          initial, torch.ones(len(actions), 1, dtype=torch.bool, device=initial.mass.device))


def flat_grads(policy):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in policy.parameters()])


@torch.no_grad()
def test_first_crossing_independent_stop_and_no_post_terminal_calls():
    horizon = 500
    initial = scheduled([1, 19, 50, 500, 501], horizon)
    policy = actor(); sim = CountingSimulator(); actor_rows = []
    hook = policy.register_forward_pre_hook(lambda _, args: actor_rows.append(args[0].shape[0]))
    trace = rollout(policy, sim, initial, horizon)
    hook.remove()
    lengths = torch.tensor([1, 19, 50, 500, 500])
    assert torch.equal(trace.valid.sum(0), lengths)
    assert torch.equal(trace.end.physical.step_index, lengths)
    # No production call counter: independently compare each terminal hidden state.
    for i in range(len(lengths)):
        single = manual_prefix(policy, CountingSimulator(), select(initial, slice(i, i+1)), horizon)
        torch.testing.assert_close(trace.end.policy.memory[i:i+1], single.end.policy.memory,
                                   rtol=1e-11, atol=1e-12)
    assert sum(actor_rows) == int(lengths.sum()) == len(sim.ids)
    assert [sim.ids.count(float(i+1)) for i in range(5)] == lengths.tolist()
    for i, length in enumerate(lengths):
        t = int(length)
        assert trace.valid[t-1, i]  # Keep the action/transition causing failure.
        assert not trace.valid[t:, i].any()
        assert (trace.positions[t-1:, i, 0] == t).all()
    assert torch.equal(L2FSimulator.terminated(trace.end.physical), torch.tensor([True]*4+[False]))
    report = reference_episode_metrics(trace)
    assert report['raptor_share_terminated'] == .8  # Failure on step 500 is still failure.


@pytest.mark.parametrize('mode', ['full', 'windowed'])
@pytest.mark.parametrize('lengths', [[1, 3, 4, 7, 8, 9], [1, 1], [2, 3], [9, 9]])
def test_batched_cost_and_full_gradient_equal_independent_prefixes(mode, lengths):
    h = 8
    initial = scheduled(lengths, h); loss = TaskLossConfig()
    batched = actor(); serial = copy.deepcopy(batched)
    record = collect_boundary_rollout(batched, CountingSimulator(), initial, loss,
                                     horizon=h, window_steps=4, backprop_mode=mode)
    result = backward_actor(batched, CountingSimulator(), record, loss)
    costs = torch.cat([step_costs(manual_prefix(serial, CountingSimulator(),
                      select(initial, slice(i,i+1)), h), loss, horizon=h).sum(0)
                      for i in range(len(lengths))])
    weights = risk_weights(costs, loss)
    (.1 * (weights * costs).sum()).backward()
    torch.testing.assert_close(record.costs, costs, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(record.weights, weights, rtol=0, atol=0)
    torch.testing.assert_close(flat_grads(batched), flat_grads(serial), rtol=1e-9, atol=1e-10)
    if mode == 'windowed':
        assert all(r['exact'] for r in result['boundaries'])
    for policy in [batched, serial]:
        torch.optim.Adam(policy.parameters(), lr=3e-4).step()
    for x, y in zip(batched.parameters(), serial.parameters()):
        torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('profile', ['l2f', 'raptor'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_real_physics_noise_and_windowed_gradients(profile, dtype):
    h = 20
    sim = L2FSimulator(L2FParams(protocol=profile))
    initial = sim.reset(3, seed=7, dtype=dtype, horizon=h)
    # One known early failure, plus two independent untouched scenes.
    p = initial.position.clone(); v = initial.velocity.clone()
    p[0] = 0; p[0, 0] = initial.position_limit[0] - .00001; v[0, 0] = 1
    initial = replace(initial, position=p, velocity=v)
    a = actor(dtype); b = copy.deepcopy(a); loss = TaskLossConfig()
    records = []
    for policy, mode in [(a, 'full'), (b, 'windowed')]:
        record = collect_boundary_rollout(policy, sim, initial, loss, horizon=h,
                                         window_steps=5, backprop_mode=mode)
        checks = backward_actor(policy, sim, record, loss)
        if mode == 'windowed':
            assert all(r['exact'] for r in checks['boundaries'])
        records.append(record)
    assert records[0].valid[:, 0].sum() == 1
    assert torch.equal(records[0].valid, records[1].valid)
    torch.testing.assert_close(records[0].costs, records[1].costs, rtol=0, atol=0)
    tol = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(flat_grads(a), flat_grads(b), rtol=tol, atol=tol)
    assert records[1].boundaries[h].physical.noise_tape.data_ptr() == initial.noise_tape.data_ptr()
    # Frozen memory and noise index: no repeated observation updates after failure.
    assert records[0].boundaries[h].physical.step_index[0] == 1
    first = a(observation(initial)).next_state.memory[0]
    torch.testing.assert_close(records[0].boundaries[h].policy.memory[0], first,
                               rtol=0, atol=0)
    trace = rollout(a, sim, initial, h)
    # Batch compaction must not change another aircraft's flight or noise tape.
    for i in (1, 2):
        single = manual_prefix(a, sim, select(initial, slice(i, i+1)), h)
        length = single.actions.shape[0]
        assert int(trace.valid[:, i].sum()) == length
        for name in ['actions', 'positions', 'velocities', 'omegas']:
            torch.testing.assert_close(getattr(trace, name)[:length, i:i+1],
                                       getattr(single, name), rtol=tol, atol=tol)
    metrics = trajectory_metrics(trace, loss)
    stats = FlightStatistics(initial, h, loss)
    stats.add(trace, 0)
    costs = step_costs(trace, loss).sum(0)
    streamed = stats.finish(costs.detach(), risk_weights(costs, loss))
    for k in ['position_rms', 'velocity_rms', 'omega_rms', 'motor_saturation_fraction', 'physical_transitions']:
        assert streamed[k] == pytest.approx(metrics[k], rel=tol, abs=tol)
    if profile == 'l2f':
        tape = initial.noise_tape.clone(); tape[0, 2:] += 100
        other = rollout(a, sim, replace(initial, noise_tape=tape), h)
        assert torch.equal(trace.actions, other.actions)
        assert torch.equal(trace.valid, other.valid)


def test_terminal_transition_keeps_gradient_and_padding_has_none():
    initial = scheduled([1, 3], 8)
    policy = actor(); trace = rollout(policy, CountingSimulator(), initial, 8)
    trace.actions.retain_grad(); trace.velocities.retain_grad()
    step_costs(trace, TaskLossConfig()).sum().backward()
    assert trace.velocities.grad[0,0].abs().sum() > 0
    assert trace.actions.grad[0,0].abs().sum() > 0
    assert (trace.velocities.grad[~trace.valid] == 0).all()
    assert (trace.actions.grad[~trace.valid] == 0).all()


def test_recompute_checks_transition_mask():
    initial = scheduled([1, 7], 8); policy = actor(); loss = TaskLossConfig()
    record = collect_boundary_rollout(policy, CountingSimulator(), initial, loss,
                                     horizon=8, window_steps=4, backprop_mode='windowed')
    invalid = record.valid.clone(); invalid[-1,0] = True
    with pytest.raises(RuntimeError, match='termination'):
        backward_actor(policy, CountingSimulator(), replace(record, valid=invalid), loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_terminal_batch():
    initial = scheduled([1, 3, 9], 8, torch.float32).to('cuda', torch.float32)
    policy = actor(torch.float32).cuda()
    trace = rollout(policy, CountingSimulator(), initial, 8)
    assert trace.valid.sum(0).tolist() == [1,3,8]
    step_costs(trace, TaskLossConfig()).sum().backward()
    assert torch.isfinite(flat_grads(policy)).all()


def test_terminal_prefix_actor_directional_derivative():
    initial = scheduled([2, 7], 8); policy = actor(); sim = CountingSimulator()
    loss = TaskLossConfig(); parameters = list(policy.parameters())
    def objective():
        trace = rollout(policy, sim, initial, 8)
        costs = step_costs(trace, loss).sum(0)
        return (costs * risk_weights(costs, loss)).sum()
    value = objective()
    gradients = torch.autograd.grad(value, parameters, allow_unused=True)
    directions = [torch.randn_like(p) for p in parameters]
    length = torch.cat([d.flatten() for d in directions]).norm()
    directions = [d / length for d in directions]
    analytical = sum((g*d).sum() for g,d in zip(gradients,directions) if g is not None)
    saved = [p.detach().clone() for p in parameters]; eps = 1e-5
    with torch.no_grad():
        for p,x,d in zip(parameters,saved,directions):
            p.copy_(x + eps*d)
        plus = objective()
        for p,x,d in zip(parameters,saved,directions):
            p.copy_(x - eps*d)
        minus = objective()
        for p,x in zip(parameters,saved):
            p.copy_(x)
    torch.testing.assert_close(analytical, (plus-minus)/(2*eps), rtol=2e-4, atol=1e-8)
