"""The user-selected fixed scales are configured, bound, and differentiable."""
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path
import copy
import math

import pytest
import torch

from env_raptor import RaptorSimulator
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, smooth_l2, rollout, scenario_costs, task_loss_components
from response_training import binding, _checkpoint, require_reference_checkpoint, sample_training_scenarios
from tools.train_response_control import parse_args
from test_adapted_loss import trace_fixture


CONFIRMED = dict(epsilon_p=.01, epsilon_a=.01, lambda_R=25.,
                 dead_cost=3., terminal_cost=200.)
ARGUMENT_FILE = Path(__file__).resolve().parents[1] / 'configs/response_raptor_multi_airframe.args'


def test_checked_in_config_supplies_all_confirmed_constants_without_fixture_defaults():
    # Use the real CLI: the general test fixture must not fill missing config flags.
    args = parse_args(['@' + str(ARGUMENT_FILE)])
    assert {field.name: getattr(args, field.name) for field in fields(TaskLossConfig)} == CONFIRMED


def test_confirmed_scales_match_independent_hand_values():
    trace = trace_fixture(1, 1)
    trace.pre_positions = torch.tensor([[[.03, .04, 0.]]], dtype=torch.float64)
    trace.action_deltas = torch.tensor([[[.06, .08, 0., 0.]]], dtype=torch.float64)
    trace.pre_orientations = torch.tensor([[[math.sqrt(.5), math.sqrt(.5), 0., 0.]]],
                                          dtype=torch.float64)
    components = task_loss_components(trace, TaskLossConfig(**CONFIRMED))
    expected = dict(position=math.sqrt(.05**2 + .01**2) - .01, attitude_delta=25.,
                    action_delta=math.sqrt(.1**2 + .01**2) - .01, dead=0., terminal=0.)
    assert components == pytest.approx(expected, rel=1e-13, abs=1e-14)
    assert float(scenario_costs(trace, TaskLossConfig(**CONFIRMED)).detach()) == pytest.approx(sum(expected.values()))


@pytest.mark.parametrize('width', [3, 4])
def test_confirmed_smoothing_has_the_declared_analytic_and_finite_difference_gradient(width):
    x = torch.linspace(-.02, .03, width, dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(smooth_l2(x, .01), (x,))
    expected = x.detach() / torch.sqrt(x.detach().square().sum() + .01**2)
    torch.testing.assert_close(gradient, expected, rtol=1e-13, atol=1e-13)
    direction = torch.linspace(.2, .6, width, dtype=x.dtype)
    step = 1e-7
    fd = (smooth_l2(x.detach() + step*direction, .01)
          - smooth_l2(x.detach() - step*direction, .01)) / (2*step)
    assert float(gradient @ direction) == pytest.approx(float(fd), rel=1e-9, abs=1e-10)


def test_checkpoint_serializes_fixed_constants_and_binds_each_change(tmp_path):
    args = parse_args(['@' + str(ARGUMENT_FILE), '--device', 'cpu', '--dtype', 'float64'])
    policy_config = ResponsePolicyConfig(**{field.name: getattr(args, field.name)
                                          for field in fields(ResponsePolicyConfig)})
    policy = ResponseMotorPolicy(policy_config).double()
    loss = TaskLossConfig(**{field.name: getattr(args, field.name) for field in fields(TaskLossConfig)})
    bound = binding(args, policy_config, loss)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    path = tmp_path / 'untrained.pt'
    torch.save(_checkpoint(policy, optimizer, dict(updates=0), bound), path)
    saved = torch.load(path, weights_only=True)
    require_reference_checkpoint(saved)
    assert saved['binding']['protocol']['loss'] == CONFIRMED
    assert saved['binding'] == bound and saved['optimizer']['state'] == {}
    assert all(type(value) is float for value in asdict(loss).values())
    with pytest.raises(FrozenInstanceError):
        loss.epsilon_p = .02
    for name in CONFIRMED:
        changed = replace(loss, **{name: getattr(loss, name)*2})
        assert binding(args, policy_config, changed) != saved['binding']


@pytest.mark.parametrize('decay', [0., 1.])
@pytest.mark.parametrize('device,dtype', [
    ('cpu', torch.float64),
    pytest.param('cuda', torch.float32,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')),
])
def test_confirmed_loss_native_group_probe_matches_independent_mean_backward(device, dtype, decay):
    # Bounded real-physics regression: 128 equally sized coverage groups, H8.
    initial, _ = sample_training_scenarios(32, 0, horizon=8, sampling='coverage128',
                                          device=device, dtype=dtype)
    torch.manual_seed(7)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(device=device, dtype=dtype)
    reference = copy.deepcopy(policy)
    original = {name: value.clone() for name, value in policy.state_dict().items()}
    loss = TaskLossConfig(**CONFIRMED)
    group = GroupBalanceConfig(max_groups=128, min_scenarios=1, layout='coverage128',
                               clip_norm=1e9)
    record = collect_rollout(policy, RaptorSimulator(), initial, loss, horizon=8,
                             time_decay=decay, group_config=group)
    assert torch.equal(record.weights, torch.full_like(record.costs, 1/128))
    backward_actor(policy, RaptorSimulator(), record, loss, gradient_scale=.1)
    trace = rollout(reference, RaptorSimulator(), initial, 8, time_decay=decay)
    (.1*scenario_costs(trace, loss).mean()).backward()
    tolerance = dict(rtol=5e-4, atol=3e-5) if dtype == torch.float32 else dict(rtol=3e-10, atol=1e-11)
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        assert torch.isfinite(actual.grad).all() and torch.isfinite(expected.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
    assert all(torch.equal(value, original[name]) for name, value in policy.state_dict().items())
    assert asdict(loss) == CONFIRMED
