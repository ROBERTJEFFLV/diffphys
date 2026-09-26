"""Architecture tests; physical dynamics, group update and BPTT stay inherited."""
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from env_raptor import RaptorSimulator
from response_policy import CONTROL_FEATURE_DIM, OBSERVATION_DIM, ResponseMotorPolicy, ResponsePolicyConfig, ResponsePolicyState
from response_task import observation
from tools.train_response_control import parse_args
from test_episode_termination import actor

ROOT = Path(__file__).resolve().parents[1]


def observations(batch=5, dtype=torch.float64):
    return observation(RaptorSimulator().reset(batch, seed=917, dtype=dtype, horizon=8))


def test_only_one_native_gru_and_one_affine_readout():
    p = ResponseMotorPolicy()
    assert [type(m) for m in p.children()] == [torch.nn.GRUCell, torch.nn.Linear]
    assert p.response_memory.input_size == CONTROL_FEATURE_DIM == 16
    assert p.response_memory.hidden_size == 64
    assert (p.readout.in_features, p.readout.out_features) == (80, 4)
    assert sum(x.numel() for x in p.parameters()) == 16068
    assert {f.name for f in fields(ResponsePolicyState)} == {'memory'}
    assert {f.name for f in fields(ResponsePolicyConfig)} == {'memory_dim', 'dt'}
    assert not p.readout.weight[:, :16].any()
    assert p.readout.weight[:, 16:].abs().sum() > 0
    assert not p.readout.bias.any()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_features_preserve_existing_frames_scales_and_known_command(dtype):
    p = ResponseMotorPolicy().to(dtype=dtype)
    obs = observations(dtype=dtype)
    R = obs[:, 6:15].reshape(-1, 3, 3)
    up = obs.new_tensor([0., 0., 1.]).expand(len(obs), 3)
    def body(x): return (R.transpose(-1, -2) @ x[..., None]).squeeze(-1)
    expected = torch.cat((body(obs[:, :3]), body(obs[:, 3:6])/3,
                          body(up), obs[:, 15:18]/10, obs[:, 18:22]), -1)
    assert obs.shape == (5, OBSERVATION_DIM) and OBSERVATION_DIM == 22
    torch.testing.assert_close(p.control_features(obs), expected, rtol=0, atol=0)
    # The known command is supplied, not reconstructed from hidden state.
    changed = obs.clone(); changed[:, 18:22] = -.7
    assert torch.equal(p.control_features(changed)[:, 12:16], changed[:, 18:22])


def test_observation_has_no_parameter_or_motor_truth_inputs():
    s = RaptorSimulator().reset(3, dtype=torch.float64, horizon=8)
    changed = replace(s, mass=s.mass*7, inertia=s.inertia*3, motor=s.motor*.4,
                      external_force=s.external_force+10)
    assert torch.equal(observation(s), observation(changed))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_combined_readout_equals_two_blocks_in_forward_and_backward(dtype):
    torch.manual_seed(101)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype=dtype)
    with torch.no_grad(): p.readout.weight[:, :16].normal_(0, .05)
    x = observations(dtype=dtype).requires_grad_()
    old = torch.randn(len(x), 8, dtype=dtype, requires_grad=True)
    c = p.control_features(x); h = p.response_memory(c, old)
    combined = p(x, ResponsePolicyState(old)).action
    split = torch.tanh(F.linear(c, p.readout.weight[:, :16])
                      + F.linear(h, p.readout.weight[:, 16:], p.readout.bias))
    tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(combined, split, **tolerance)
    variables = [x, old, *p.parameters()]
    a = torch.autograd.grad(combined.square().sum(), variables, retain_graph=True)
    b = torch.autograd.grad(split.square().sum(), variables)
    for aa, bb in zip(a, b): torch.testing.assert_close(aa, bb, **tolerance)


def test_first_observation_updates_hidden_and_trains_zero_wc():
    p = actor()
    obs = observations().requires_grad_()
    state = p.initial_state(obs)
    assert not state.memory.any()
    out = p(obs, state)
    expected_hidden = p.response_memory(p.control_features(obs), state.memory)
    torch.testing.assert_close(out.next_state.memory, expected_hidden, rtol=0, atol=0)
    assert out.next_state.memory.abs().sum() > 0
    out.action.square().sum().backward()
    assert p.readout.weight.grad[:, :16].abs().sum() > 0
    assert p.readout.weight.grad[:, 16:].abs().sum() > 0
    assert p.response_memory.weight_ih.grad.abs().sum() > 0
    assert obs.grad[:, :3].abs().sum() > 0
    # First output already depends on the current observation, without W_c.
    shifted = obs.detach().clone(); shifted[:, 0] += .1
    assert not torch.equal(p(obs.detach()).action, p(shifted).action)


def test_wc_gradient_has_the_expected_direct_outer_product():
    p = actor(); obs = observations()
    c = p.control_features(obs)
    h = p.response_memory(c, p.initial_state(obs).memory)
    z = p.readout(torch.cat((c, h), -1))
    loss = torch.tanh(z).square().sum()
    delta = torch.autograd.grad(loss, z, retain_graph=True)[0]
    weight_grad = torch.autograd.grad(loss, p.readout.weight)[0]
    torch.testing.assert_close(weight_grad[:, :16], delta.T @ c, rtol=1e-12, atol=1e-12)


def test_hidden_retains_actual_action_history_when_current_input_is_equal():
    p = actor(); now = observations()
    first = now.clone(); second = now.clone()
    first[:, 18:22] = -.6; second[:, 18:22] = .6
    ha, hb = p(first).next_state, p(second).next_state
    assert not torch.equal(ha.memory, hb.memory)
    assert not torch.equal(p(now, ha).action, p(now, hb).action)
    torch.testing.assert_close(p(now, p.initial_state(now)).action, p(now).action, rtol=0, atol=0)


@pytest.mark.parametrize('width', [0, -1, True, 3.5])
def test_invalid_hidden_width_rejected(width):
    with pytest.raises(ValueError, match='memory_dim'): ResponsePolicyConfig(memory_dim=width)


def test_legacy_input_and_removed_architecture_flags_fail_explicitly():
    p = actor()
    with pytest.raises(ValueError, match='22'): p(torch.zeros(3, 25, dtype=torch.float64))
    for flag in ['--hidden-dim', '--integral-limit', '--integral-leak']:
        with pytest.raises(SystemExit): parse_args([flag, '1'])
    with pytest.raises(ValueError, match='memory'):
        p(observations(), ResponsePolicyState(torch.zeros(4, 8, dtype=torch.float64)))
