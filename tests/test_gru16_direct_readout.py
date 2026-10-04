"""Deployment-interface regressions for the structured replacement of the readout.

The filename is retained to keep the existing regression entrypoints intact.
"""
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from env_raptor import RaptorSimulator
from response_policy import (CONTROL_FEATURE_DIM, OBSERVATION_DIM, ResponseMotorPolicy,
                             ResponsePolicyConfig, ResponsePolicyState,
                             IncrementalResidualReadout, GeometricFeedback)
from response_task import observation
from tools.train_response_control import parse_args
from test_episode_termination import actor

ROOT = Path(__file__).resolve().parents[1]


def observations(batch=5, dtype=torch.float64):
    return observation(RaptorSimulator().reset(batch, seed=917, dtype=dtype, horizon=8))


def test_native_gru_and_explicit_geometric_and_residual_paths():
    p = ResponseMotorPolicy()
    assert [type(m) for m in p.children()] == [torch.nn.GRUCell, IncrementalResidualReadout, GeometricFeedback]
    assert p.response_memory.input_size == CONTROL_FEATURE_DIM == 16
    assert p.response_memory.hidden_size == 64
    assert (p.readout.in_features, p.readout.out_features) == (64, 4)
    assert sum(x.numel() for x in p.parameters()) == 16013
    assert {f.name for f in fields(ResponsePolicyState)} == {'memory'}
    assert {f.name for f in fields(ResponsePolicyConfig)} == {
        'memory_dim', 'dt', 'residual_amplitude', 'residual_gain',
        'horizontal_accel_limit', 'vertical_fraction', 'antipodal_epsilon'}
    assert p.readout.weight.abs().sum() > 0
    assert not p.readout.bias.any()
    assert 'base_feedback.coefficients.raw' in dict(p.named_parameters())


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
def test_explicit_paths_match_manual_mixing_in_forward_and_backward(dtype):
    torch.manual_seed(101)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype=dtype)
    x = observations(dtype=dtype).requires_grad_()
    old = (torch.rand(len(x), 8, dtype=dtype)*2-1).requires_grad_()
    c = p.control_features(x); h = p.response_memory(c, old)
    combined = p(x, ResponsePolicyState(old)).action
    weight, bias = p.readout.effective_parameters(p.response_memory)
    correction = p.readout.amplitude*torch.tanh(F.linear(h, weight, bias))
    mixer = x.new_tensor([[1,-1,-1,-1], [1,-1,1,1], [1,1,1,-1], [1,1,-1,1]])
    split = torch.tanh(F.linear(p.base_feedback(x)+correction, mixer))
    tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(combined, split, **tolerance)
    variables = [x, old, *p.parameters()]
    a = torch.autograd.grad(combined.square().sum(), variables, retain_graph=True)
    b = torch.autograd.grad(split.square().sum(), variables)
    for aa, bb in zip(a, b): torch.testing.assert_close(aa, bb, **tolerance)


def test_first_observation_updates_hidden_and_trains_both_paths():
    p = actor()
    obs = observations().requires_grad_()
    state = p.initial_state(obs)
    assert not state.memory.any()
    out = p(obs, state)
    expected_hidden = p.response_memory(p.control_features(obs), state.memory)
    torch.testing.assert_close(out.next_state.memory, expected_hidden, rtol=0, atol=0)
    assert out.next_state.memory.abs().sum() > 0
    out.action.square().sum().backward()
    assert p.base_feedback.coefficients.raw.grad.abs().sum() > 0
    assert p.readout.weight.grad.abs().sum() > 0
    assert p.response_memory.weight_ih.grad.abs().sum() > 0
    assert obs.grad[:, :3].abs().sum() > 0
    # First output already depends on the current observation, with zero-initialized memory.
    shifted = obs.detach().clone(); shifted[:, 0] += .1
    assert not torch.equal(p(obs.detach()).action, p(shifted).action)


@pytest.mark.parametrize('zero_weights', [False, True])
def test_effective_readout_pullback_matches_autograd_including_gru_bounds(zero_weights):
    from response_grad_probe import bounded_readout_pullback
    p = actor()
    if zero_weights:
        with torch.no_grad():
            for v in p.parameters(): v.zero_()
    gw = torch.randn(3, *p.readout.weight.shape, dtype=torch.float64)
    gb = torch.randn(3, 4, dtype=torch.float64)
    actual = bounded_readout_pullback(p, gw, gb)
    weight, bias = p.readout.effective_parameters(p.response_memory)
    named = list(p.named_parameters())
    for group in range(3):
        expected = torch.autograd.grad((weight*gw[group]).sum()+(bias*gb[group]).sum(),
                                       [v for _,v in named], retain_graph=True, allow_unused=True)
        for (name, param), derivative in zip(named, expected):
            a = actual[name][group] if name in actual else torch.zeros_like(param)
            b = derivative if derivative is not None else torch.zeros_like(param)
            torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-12)


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
