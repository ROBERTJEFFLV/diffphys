"""Performance changes must preserve sensing, physics and every Actor derivative."""
from dataclasses import fields, replace
from unittest.mock import patch
import copy

import pytest
import torch

from env_raptor import RaptorSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import rollout, step_costs, TaskLossConfig
from response_groups import GroupBalanceConfig
from response_adjoints import collect_rollout, backward_actor

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('decay', [0., 1.])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_optional_observation_storage_preserves_flight_and_gradients(device, decay, dtype):
    torch.manual_seed(913)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(device=device, dtype=dtype)
    other = copy.deepcopy(policy)
    simulator = RaptorSimulator()
    initial = simulator.reset(8, seed=49, horizon=60, device=device, dtype=dtype)
    limit = initial.position_limit.clone()
    limit[0] = .001  # Frozen padding must retain its original failure semantics.
    initial = replace(initial, position_limit=limit)
    loss = TaskLossConfig(.01, .01, .2)
    reference = rollout(policy, simulator, initial, 60, time_decay=decay)
    reduced = rollout(other, simulator, initial, 60, time_decay=decay,
                      record_observations=False)
    assert reference.observations.shape == (61, 8, 22)
    assert reduced.observations is None
    for field in fields(reference):
        if field.name in ('observations', 'end', 'initial'):
            continue
        assert torch.equal(getattr(reference, field.name), getattr(reduced, field.name)), field.name
    for group in ('physical', 'policy'):
        a, b = getattr(reference.end, group), getattr(reduced.end, group)
        for field in fields(a):
            assert torch.equal(getattr(a, field.name), getattr(b, field.name)), (group, field.name)
    step_costs(reference, loss).sum().backward()
    step_costs(reduced, loss).sum().backward()
    for (name, a), (_, b) in zip(policy.named_parameters(), other.named_parameters()):
        assert torch.equal(a.grad, b.grad), name


@pytest.mark.parametrize('decay', [0., 1.])
def test_training_omits_observation_storage_but_keeps_probe_and_chunk_gradients(decay, monkeypatch):
    import response_adjoints as adjoints
    simulator = RaptorSimulator()
    initial = simulator.reset(32, seed=91, horizon=60, dtype=torch.float64)
    torch.manual_seed(47)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    other = copy.deepcopy(policy)
    loss = TaskLossConfig(.01, .01, .2)
    original = adjoints.rollout
    calls = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((kwargs.get('record_observations'), result.observations))
        return result
    monkeypatch.setattr(adjoints, 'rollout', observe)
    record = collect_rollout(policy, simulator, initial, loss, horizon=60, time_decay=decay)
    assert calls == [(False, None), (False, None)]
    backward_actor(policy, simulator, record, loss)
    monkeypatch.setattr(adjoints, 'rollout', lambda *args, **kwargs: original(
        *args, **{**kwargs, 'record_observations': True}))
    reference = collect_rollout(other, simulator, initial, loss, horizon=60, time_decay=decay)
    backward_actor(other, simulator, reference, loss)
    assert torch.equal(record.costs, reference.costs)
    assert record.metrics == reference.metrics
    # In the exact (no-decay) CPU graph, removal of unused observation nodes
    # can reorder a floating-point sum at a join (measured max 6.94e-18).
    # The production Time Decay path retains bitwise agreement.
    if decay:
        assert torch.equal(record.probe.rows, reference.probe.rows)
    else:
        torch.testing.assert_close(record.probe.rows, reference.probe.rows, rtol=1e-13, atol=1e-15)
    for a, b in zip(policy.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cached_physics_vectors_reuse_allocation_without_changing_rk4(device, dtype, monkeypatch):
    import env_raptor as env
    env._constant_vector.cache_clear()
    simulator = RaptorSimulator()
    initial = simulator.reset(3, seed=57, horizon=3, device=device, dtype=dtype)
    action = torch.full_like(initial.motor, .13, requires_grad=True)
    rng = torch.random.get_rng_state().clone()
    with patch('env_raptor.torch.tensor', wraps=torch.tensor) as factory:
        first = simulator.step(initial, action)
        allocated = factory.call_count
        second = simulator.step(initial, action)
        assert allocated == 2
        assert factory.call_count == allocated
    assert torch.equal(rng, torch.random.get_rng_state())
    gradient = torch.autograd.grad(first.omega.sum() + first.position.sum() + first.motor.sum(), action)[0]
    monkeypatch.setattr(env, '_constant_vector', lambda values, device, dtype: torch.tensor(
        values, device=device, dtype=dtype))
    reference = simulator.step(initial, action)
    expected = torch.autograd.grad(reference.omega.sum() + reference.position.sum() + reference.motor.sum(), action)[0]
    for field in fields(initial):
        assert torch.equal(getattr(first, field.name), getattr(reference, field.name)), field.name
        assert torch.equal(getattr(first, field.name), getattr(second, field.name)), field.name
    assert torch.equal(gradient, expected)


def test_probe_keeps_fp64_accumulation_without_explicit_cast_temporary():
    from response_grad_probe import GroupGradientProbe
    torch.manual_seed(57)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4))
    probe = GroupGradientProbe(policy, torch.arange(4).reshape(2, 2), torch.tensor([2, 2]))
    x, h, delta = torch.randn(4, 16), torch.randn(4, 4), torch.randn(4, 4)
    with patch.object(torch.Tensor, 'double', side_effect=AssertionError('unneeded FP64 temporary')):
        probe._linear_partial(h, delta, probe.slots)
        probe._gru_partial(x, h, delta, probe.slots)
    assert probe.rows.dtype == torch.float64
    assert torch.isfinite(probe.rows).all()
    assert probe.rows.norm() > 0
    probe.close()


@pytest.mark.parametrize('device', DEVICES)
def test_compiled_rotation_backend_preserves_rk4_and_all_actor_gradients(device):
    torch.manual_seed(827)
    initial = RaptorSimulator().reset(8, seed=63, horizon=16, device=device)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(device)
    other = copy.deepcopy(policy)
    loss = TaskLossConfig(.01, .01, .2)
    eager = rollout(policy, RaptorSimulator(rotation_backend='eager'), initial, 16, time_decay=1.)
    compiled = rollout(other, RaptorSimulator(rotation_backend='compile'), initial, 16, time_decay=1.)
    for name in ('actions', 'observations', 'positions', 'velocities', 'omegas', 'valid'):
        torch.testing.assert_close(getattr(eager, name), getattr(compiled, name), rtol=1e-6, atol=1e-6)
    step_costs(eager, loss).sum().backward()
    step_costs(compiled, loss).sum().backward()
    for a, b in zip(policy.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)


def test_rotation_backend_is_explicit_in_binding_and_rejects_invalid_choice():
    from tools.train_response_control import parse_args
    from response_training import binding
    loss = TaskLossConfig(.01, .01, .2)
    extra = ['--epsilon-p', '.01', '--epsilon-a', '.01', '--lambda-R', '.2']
    eager = parse_args(extra + ['--rotation-backend', 'eager'])
    compiled = parse_args(extra + ['--rotation-backend', 'compile'])
    a, b = [binding(args, ResponsePolicyConfig(), loss) for args in (eager, compiled)]
    assert a['numerical_backend']['rotation'] == 'eager'
    assert b['numerical_backend']['rotation'] == 'compile'
    assert b['numerical_backend']['options']['emulate_precision_casts'] is True
    assert b['numerical_backend']['options']['triton.cudagraphs'] is False
    assert a != b
    with pytest.raises(ValueError, match='rotation backend'):
        RaptorSimulator(rotation_backend='unsupported')
