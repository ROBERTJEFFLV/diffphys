"""Optional real Inductor tests (set DIFFPHYS_TEST_COMPILE=1; compilation is costly)."""
from dataclasses import fields, replace
import copy
import os

import pytest
import torch

from env_raptor import RaptorParams, RaptorSimulator
from response_acceleration import make_simulator, execution_contract


def test_execution_backend_is_explicit_and_checkpoint_bound():
    from tools.train_response_control import parse_args
    from response_training import binding, SOURCE_FILES
    from response_policy import ResponsePolicyConfig
    from response_task import TaskLossConfig
    a = parse_args(['--device', 'cpu'])
    b = parse_args(['--device', 'cpu', '--physics-backend', 'compile'])
    assert a.physics_backend == 'eager'
    assert binding(a, ResponsePolicyConfig(), TaskLossConfig())['execution'] == execution_contract('eager')
    assert binding(b, ResponsePolicyConfig(), TaskLossConfig())['execution'] == execution_contract('compile')
    assert 'response_acceleration.py' in SOURCE_FILES
    with pytest.raises(ValueError, match='physics_backend'):
        make_simulator(RaptorParams(), 'automatic-fallback')


compile_test = pytest.mark.skipif(os.environ.get('DIFFPHYS_TEST_COMPILE') != '1',
                                  reason='set DIFFPHYS_TEST_COMPILE=1 for real Inductor checks')


@compile_test
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_compiled_rk4_preserves_all_fields_and_state_action_vjps(dtype):
    torch.compiler.reset()
    torch.manual_seed(63)
    eager = RaptorSimulator()
    compiled = make_simulator(RaptorParams(), 'compile')
    names = ('position', 'velocity', 'orientation', 'omega', 'motor')
    # Reuse the compiled callable across two batch sizes: compaction stays dynamic.
    for n in (32, 9):
        state = eager.reset(n, seed=43, horizon=4, dtype=dtype)
        state = replace(state, **{k: getattr(state, k).clone().requires_grad_() for k in names})
        action = torch.randn(n, 4, dtype=dtype, requires_grad=True)
        inputs = [getattr(state, k) for k in names] + [action]
        a, b = eager.step(state, action), compiled.step(state, action)
        tol = dict(rtol=5e-4, atol=3e-5) if dtype == torch.float32 else dict(rtol=3e-9, atol=2e-10)
        for field in fields(state):
            torch.testing.assert_close(getattr(a, field.name), getattr(b, field.name), **tol)
        grads = [torch.autograd.grad(sum(getattr(s, k).square().sum() for k in names), inputs)
                 for s in (a, b)]
        for x, y in zip(*grads):
            torch.testing.assert_close(x, y, **tol)


@compile_test
@pytest.mark.parametrize('decay', [0., 1.])
def test_compiled_full_bptt_retains_all_group_vectors_delay_and_pulses(decay):
    from response_adjoints import collect_rollout, backward_actor
    from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
    from response_groups import GroupBalanceConfig
    from response_task import TaskLossConfig
    from response_training import sample_training_scenarios
    torch.compiler.reset()
    torch.manual_seed(7)
    state, _ = sample_training_scenarios(32, 7, horizon=55, sampling='coverage128', dtype=torch.float64)
    a = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    b = copy.deepcopy(a)
    cfg = GroupBalanceConfig(max_groups=128, min_scenarios=1, layout='coverage128', clip_norm=.1)
    loss = TaskLossConfig()
    eager = make_simulator(RaptorParams())
    compiled = make_simulator(RaptorParams(), 'compile')
    r = collect_rollout(a, eager, state, loss, horizon=55, time_decay=decay, group_config=cfg)
    s = collect_rollout(b, compiled, state, loss, horizon=55, time_decay=decay, group_config=cfg)
    assert r.metrics['physical_transitions'] == s.metrics['physical_transitions']
    torch.testing.assert_close(r.costs, s.costs, rtol=3e-9, atol=2e-10)
    backward_actor(a, eager, r, loss)
    backward_actor(b, compiled, s, loss)
    torch.testing.assert_close(r.probe.rows, s.probe.rows, rtol=3e-8, atol=2e-9)
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x.grad, y.grad, rtol=3e-8, atol=2e-9)


@compile_test
def test_compiled_one_update_replays_actor_adam_rng_and_rejects_eager_resume(tmp_path):
    from test_update_audit import args_for, run
    from tools.replay_response_update import replay
    torch.compiler.reset()
    run(args_for(tmp_path, 1, ('--physics-backend', 'compile', '--max-seconds', '900')))
    capsules = list((tmp_path / 'audit/updates').glob('*.pt'))
    assert len(capsules) == 1
    result = replay(capsules[0])
    assert result['passed'], result
    with pytest.raises(ValueError, match='configuration'):
        run(args_for(tmp_path, 2, ('--resume', str(tmp_path / 'latest.pt'))))
