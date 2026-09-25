"""Architecture tests; physical dynamics, group update and BPTT stay inherited."""
from dataclasses import fields, replace
from pathlib import Path
import ast
import copy
import hashlib
import json

import pytest
import torch
from torch.nn import functional as F

from env_l2f import L2FSimulator
from response_policy import (
    ARCHITECTURE, CONTROL_FEATURE_DIM, OBSERVATION_DIM,
    ResponseMotorPolicy, ResponsePolicyConfig, ResponsePolicyState,
)
from response_task import observation, rollout, TaskLossConfig
from response_adjoints import collect_boundary_rollout, backward_actor
from response_groups import GroupBalanceConfig, normalize_group_rows
from response_training import migrate_actor_weights, require_reference_checkpoint
from tools.train_response_control import parse_args
from tools.verify_group_balance import require_same_actor, fixture
from test_episode_termination import actor, CountingSimulator, scheduled, flat_grads

ROOT = Path(__file__).resolve().parents[1]


def observations(batch=5, dtype=torch.float64):
    return observation(L2FSimulator().reset(batch, seed=917, dtype=dtype, horizon=8))


def test_only_one_native_gru_and_one_affine_readout():
    p = ResponseMotorPolicy()
    assert [type(m) for m in p.children()] == [torch.nn.GRUCell, torch.nn.Linear]
    assert p.response_memory.input_size == CONTROL_FEATURE_DIM == 16
    assert p.response_memory.hidden_size == 64
    assert (p.readout.in_features, p.readout.out_features) == (80, 4)
    assert sum(x.numel() for x in p.parameters()) == 16068
    assert {f.name for f in fields(ResponsePolicyState)} == {'memory'}
    assert {f.name for f in fields(ResponsePolicyConfig)} == {'memory_dim', 'dt', 'action_rate'}
    assert not p.readout.weight[:, :16].any()
    assert p.readout.weight[:, 16:].abs().sum() > 0
    assert not p.readout.bias.any()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_features_preserve_existing_frames_scales_and_executed_action(dtype):
    p = ResponseMotorPolicy().to(dtype=dtype)
    obs = observations(dtype=dtype)
    R = obs[:, 6:15].reshape(-1, 3, 3)
    up = obs.new_tensor([0., 0., 1.]).expand(len(obs), 3)
    def body(x): return (R.transpose(-1, -2) @ x[..., None]).squeeze(-1)
    expected = torch.cat((body(obs[:, :3]), body(obs[:, 3:6])/3,
                          body(up), obs[:, 15:18]/10, obs[:, 18:22]), -1)
    assert obs.shape == (5, OBSERVATION_DIM) and OBSERVATION_DIM == 22
    torch.testing.assert_close(p.control_features(obs), expected, rtol=0, atol=0)
    # The executed command is supplied, not reconstructed from hidden state.
    changed = obs.clone(); changed[:, 18:22] = -.7
    assert torch.equal(p.control_features(changed)[:, 12:16], changed[:, 18:22])


def test_observation_has_no_parameter_or_motor_truth_inputs():
    s = L2FSimulator().reset(3, dtype=torch.float64, horizon=8)
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


@pytest.mark.parametrize('rate', [0., .7])
def test_optional_slew_projection_preserves_parent_rule(rate):
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8, action_rate=rate)).double()
    obs = observations()
    with torch.no_grad():
        for parameter in p.parameters(): parameter.zero_()
        p.readout.bias.fill_(10)
    proposed = torch.tanh(obs.new_full((len(obs), 4), 10))
    expected = proposed
    if rate > 0:
        previous = obs[:, 18:22]; radius = rate*p.config.dt
        expected = torch.maximum((previous-radius).clamp(-1, 1),
                                 torch.minimum((previous+radius).clamp(-1, 1), proposed))
    torch.testing.assert_close(p(obs).action, expected, rtol=0, atol=0)


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


@pytest.mark.parametrize('alpha', [0., 1.])
def test_nonzero_direct_feedback_keeps_h500_full_windowed_bptt(alpha):
    # All 500 graph steps are exercised, including a first and final-step failure.
    h = 500; initial = scheduled([1, 499, 500, 501], h)
    p = actor()
    with torch.no_grad(): p.readout.weight[:, :16].normal_(0, .001)
    rows = []
    for mode in ['full', 'windowed']:
        model = copy.deepcopy(p); loss = TaskLossConfig()
        record = collect_boundary_rollout(model, CountingSimulator(), initial, loss,
            horizon=h, window_steps=50, backprop_mode=mode, time_decay=alpha)
        checks = backward_actor(model, CountingSimulator(), record, loss)
        assert all(c['exact'] for c in checks['boundaries'])
        assert record.valid.sum(0).tolist() == [1, 499, 500, 500]
        rows.append((record.costs.detach(), flat_grads(model)))
    torch.testing.assert_close(rows[0][0], rows[1][0], rtol=0, atol=0)
    torch.testing.assert_close(rows[0][1], rows[1][1], rtol=1e-9, atol=1e-10)


def test_wc_and_gru_share_whole_actor_group_norm_and_one_adam_step():
    p = actor()
    with torch.no_grad(): p.readout.weight[:, :16].normal_(0, .01)
    optimizer = torch.optim.Adam(p.parameters(), lr=3e-4)
    s = L2FSimulator().reset(64, seed=101, dtype=torch.float64, horizon=8)
    cfg = GroupBalanceConfig(enabled=True); loss = TaskLossConfig()
    record = collect_boundary_rollout(p, L2FSimulator(), s, loss, horizon=8, window_steps=4,
        backprop_mode='full', time_decay=1., group_config=cfg)
    parameters = list(p.parameters())
    raw_rows = []
    for seed in record.group_coefficients:
        gs = torch.autograd.grad(record.costs, parameters, grad_outputs=.1*seed, retain_graph=True)
        raw_rows.append(torch.cat([g.flatten() for g in gs]))
    expected, _ = normalize_group_rows(torch.stack(raw_rows), cfg.gradient_epsilon)
    backward_actor(p, L2FSimulator(), record, loss)
    torch.testing.assert_close(flat_grads(p), expected, rtol=1e-10, atol=1e-12)
    assert p.readout.weight.grad[:, :16].norm() > 0
    assert p.response_memory.weight_hh.grad.norm() > 0
    optimizer.step()
    assert len(optimizer.param_groups) == 1
    assert all(float(value['step']) == 1 for value in optimizer.state.values())
    assert set(optimizer.state) == set(parameters)


@pytest.mark.parametrize('entry', ['resume', 'init-checkpoint', 'evaluate'])
def test_old_response_mlp_checkpoint_rejected_before_loading(tmp_path, entry):
    from test_reference_training import args_for, run
    from response_training import evaluate_checkpoint
    path = tmp_path / 'old.pt'
    torch.save({'schema': 'response-actor-only-reference-v3',
                'architecture': 'response-conditioned-absolute-motor-policy-v3'}, path)
    if entry == 'evaluate':
        args = args_for(tmp_path/'eval'); args.checkpoint = path
        function = lambda: evaluate_checkpoint(args)
    else:
        function = lambda: run(args_for(tmp_path/entry, extra=('--'+entry, str(path))))
    with pytest.raises(ValueError, match='Actor architecture'): function()


def test_old_weights_and_cross_architecture_equivalence_are_not_silently_accepted(tmp_path):
    p = actor(); state = dict(p.state_dict()); state['controller.4.bias'] = torch.zeros(4)
    with pytest.raises(ValueError, match='keys'): migrate_actor_weights(p, state)
    (tmp_path/'response_policy.py').write_text('ARCHITECTURE = "old"\n')
    with pytest.raises(ValueError, match='same Actor architecture'): require_same_actor(tmp_path)
    require_same_actor(ROOT)


def test_existing_hover_diagnostic_uses_new_readout():
    p, s = fixture('cpu', True)
    with torch.no_grad(): trace = rollout(p, L2FSimulator(), s, 8)
    assert trace.valid.all()
    assert trace.positions.abs().max() < 1e-5


def test_parent_training_and_simulator_contract_is_preserved():
    manifest = json.loads((ROOT/'tests/gru16_parent_contract.json').read_text())
    for name, expected in manifest['unchanged_files'].items():
        data = (ROOT/name).read_bytes()
        digest = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert digest == expected, name
    for name, symbols in manifest['unchanged_source_segments'].items():
        text = (ROOT/name).read_text()
        nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for symbol, expected in symbols.items():
            segment = ast.get_source_segment(text, nodes[symbol])
            assert hashlib.sha256(segment.encode()).hexdigest() == expected, (name, symbol)


def test_real_rk4_h500_nonzero_feedback_full_and_windowed_agree():
    """Constructed near-hover controller; no claim about the random initialization."""
    from env_l2f import L2FParams
    sim = L2FSimulator(L2FParams(protocol='l2f'))
    s = sim.reset(2, seed=7, dtype=torch.float64, horizon=500)
    rotor = (s.mass*9.81/(4*s.thrust_coefficients[:, 0, 2])).sqrt()
    command = 2*rotor/s.motor_max-1
    q = torch.zeros_like(s.orientation); q[:, 0] = 1
    pos = torch.zeros_like(s.position); pos[:, 2] = pos.new_tensor([-.02, .02])
    s = replace(s, position=pos, velocity=torch.zeros_like(s.velocity), orientation=q,
                omega=torch.zeros_like(s.omega), motor=rotor[:, None].expand(-1, 4),
                previous_action=command[:, None].expand(-1, 4),
                external_force=torch.zeros_like(s.external_force),
                external_torque=torch.zeros_like(s.external_torque))
    p = actor()
    with torch.no_grad():
        # Equal motor outputs prevent rotational excitation in this fixture.
        row = p.readout.weight[0, 16:].clone()*.01
        p.readout.weight.zero_()
        p.readout.weight[:, 2] = -.4
        p.readout.weight[:, 5] = -1.2
        p.readout.weight[:, 16:] = row
        p.readout.bias.fill_(float(command[0].atanh()))
    results = []
    for mode in ('full', 'windowed'):
        model = copy.deepcopy(p); loss = TaskLossConfig()
        record = collect_boundary_rollout(model, sim, s, loss, horizon=500,
            window_steps=50, backprop_mode=mode, time_decay=1.)
        info = backward_actor(model, sim, record, loss)
        assert record.valid.all()
        assert all(row['exact'] for row in info['boundaries'])
        assert model.response_memory.weight_ih.grad.norm() > 0
        assert model.readout.weight.grad[:, :16].norm() > 0
        results.append((record.costs.detach(), flat_grads(model)))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-9, atol=1e-10)
