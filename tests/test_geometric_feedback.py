"""Actor-only geometry and differential-gain regressions, not flight certification."""
from dataclasses import replace
import copy
import math

import pytest
import torch

from response_policy import (ResponseMotorPolicy, ResponsePolicyConfig,
                             ResponsePolicyState, channels_to_motors,
                             motors_to_channels, gru_incremental_bounds)
from env_raptor import RaptorSimulator, quaternion_rotation
from response_noise import DisturbanceConfig
from response_task import observation, rollout, TaskLossConfig


def upright(n=1, dtype=torch.float64):
    x = torch.zeros(n, 22, dtype=dtype)
    x[:, 6:15] = torch.eye(3, dtype=dtype).reshape(1, 9)
    return x


def rotated(angle, axis):
    x = upright()
    q = x.new_zeros(1, 4)
    q[:, 0] = math.cos(angle / 2)
    q[:, axis + 1] = math.sin(angle / 2)
    x[:, 6:15] = quaternion_rotation(q).reshape(1, 9)
    return x


def test_mixer_roundtrip_and_physical_flu_motor_signs():
    q = torch.eye(4, dtype=torch.float64)
    motor = channels_to_motors(q)
    expected = q.new_tensor([[1, 1, 1, 1], [-1, -1, 1, 1],
                            [-1, 1, 1, -1], [-1, 1, -1, 1]])
    torch.testing.assert_close(motor, expected, rtol=0, atol=0)
    torch.testing.assert_close(motors_to_channels(motor), q, rtol=0, atol=0)
    s = RaptorSimulator().reset(4, dtype=torch.float64, disturbances=DisturbanceConfig.clean())
    force = .5 + .01 * motor
    torque = RaptorSimulator.body_torque(s, force)
    for row in (1, 2, 3):
        assert torque[row, row - 1] > 0
        assert torch.count_nonzero(torque[row].abs() > 1e-12) == 1


def test_base_is_instantaneous_and_independent_of_previous_command_and_memory():
    p = ResponseMotorPolicy().double()
    x = upright(2); x[:, 0] = .2
    other = x.clone(); other[:, 18:] = .9
    torch.testing.assert_close(p.base_feedback(x), p.base_feedback(other), rtol=0, atol=0)
    a = p.components(x)
    b = p.components(x, ResponsePolicyState(torch.ones(2, 64, dtype=x.dtype) * .2))
    torch.testing.assert_close(a.base_command, b.base_command, rtol=0, atol=0)
    assert not torch.equal(a.residual_command, b.residual_command)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('degrees', [1, 45, 90, 120, 179])
def test_geometric_feedback_restores_large_tilts_without_euler_division(axis, degrees):
    p = ResponseMotorPolicy().double()
    x = rotated(math.radians(degrees), axis).requires_grad_()
    base = p.base_feedback(x)
    assert base[0, 1 + axis] < 0
    assert base[0, 1 + (1 - axis)].abs() < 1e-12
    grad = torch.autograd.grad(base.square().sum(), x)[0]
    assert torch.isfinite(grad).all()


def test_exact_inversion_has_finite_declared_axis_and_no_upright_bias():
    p = ResponseMotorPolicy().double()
    zero = p.base_feedback(upright())
    torch.testing.assert_close(zero[:, 1:], torch.zeros_like(zero[:, 1:]), rtol=0, atol=0)
    x = rotated(math.pi, 0).requires_grad_()
    y = p.base_feedback(x)
    assert y[0, 1] > 0  # Explicit +body-x convention at the antipode only.
    assert torch.isfinite(torch.autograd.grad(y.sum(), x)[0]).all()


def test_position_velocity_and_rate_feedback_signs_and_no_heading_target():
    p = ResponseMotorPolicy().double()
    for field in (0, 3):
        x = upright(); x[0, field] = .1
        assert p.base_feedback(x)[0, 2] < 0
        x = upright(); x[0, field + 1] = .1
        assert p.base_feedback(x)[0, 1] > 0
        x = upright(); x[0, field + 2] = .1
        assert p.base_feedback(x)[0, 0] < p.base_feedback(upright())[0, 0]
    for axis in range(3):
        x = upright(); x[0, 15 + axis] = .1
        assert p.base_feedback(x)[0, 1 + axis] < 0
    torch.testing.assert_close(p.base_feedback(rotated(2.2, 2)),
                               p.base_feedback(upright()), rtol=0, atol=1e-12)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_first_action_heads_and_final_saturation_have_finite_gradients(dtype):
    torch.manual_seed(7)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype=dtype)
    x = upright(4, dtype); x[:, 0] = .2; x[:, 3] = -.15
    x.requires_grad_()
    parts = p.components(x)
    out = p(x)
    torch.testing.assert_close(out.action, torch.tanh(parts.motor_logits), rtol=0, atol=0)
    torch.testing.assert_close(parts.motor_logits,
                               channels_to_motors(parts.base_command + parts.residual_command),
                               rtol=0, atol=0)
    grads = torch.autograd.grad(out.action.square().sum(), (x, *p.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert p.initial_state(x).memory.count_nonzero() == 0
    assert out.next_state.memory.abs().sum() > 0
    assert out.action.abs().max() <= 1


def test_joint_input_hidden_residual_jacobian_obeys_each_channel_budget():
    torch.manual_seed(3)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=5)).double()
    # Test large raw weights as well as their initialized range; the actual
    # bound must include GRU input/recurrent sensitivities, not just readout W.
    for scale in (1., 4.):
        with torch.no_grad():
            p.response_memory.weight_ih.mul_(scale)
            p.response_memory.weight_hh.mul_(scale)
            p.readout.weight.mul_(scale)
            p.response_memory.bias_hh.add_(.2)
        x = torch.randn(21, dtype=torch.float64) * .2
        def correction(z):
            memory = p.response_memory(z[:16][None], z[16:][None])
            return p.readout.amplitude * torch.tanh(p.readout(memory, p.response_memory)[0])
        jac = torch.autograd.functional.jacobian(correction, x)
        assert (torch.linalg.vector_norm(jac, dim=-1) <= p.readout.gain_limit + 1e-12).all()
        y = correction(x)
        assert (y.abs() <= p.readout.amplitude).all()


def test_gru_bound_covers_native_one_step_derivatives_on_reachable_hidden_cube():
    torch.manual_seed(102)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    lx, lh = gru_incremental_bounds(p.response_memory)
    for _ in range(4):
        x = torch.randn(16, dtype=torch.float64)
        h = torch.rand(4, dtype=torch.float64) * 2 - 1
        jx, jh = torch.autograd.functional.jacobian(
            lambda a, b: p.response_memory(a[None], b[None])[0], (x, h))
        assert torch.linalg.matrix_norm(jx, ord=2) <= lx
        assert torch.linalg.matrix_norm(jh, ord=2) <= lh


def test_exact_gradient_matches_finite_difference_through_bound_and_geometry():
    torch.manual_seed(72)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    x = rotated(2., 1); x[:, :3] = x.new_tensor([.2, -.1, .1])
    state = ResponsePolicyState(torch.full((1, 4), .25, dtype=x.dtype))
    variables = [p.base_feedback.coefficients.raw, p.readout.weight,
                 p.response_memory.weight_ih, p.response_memory.weight_hh,
                 p.response_memory.bias_hh]
    # Residual-only term makes parameter-normalization derivatives observable.
    def loss():
        parts = p.components(x, state)
        return parts.base_command.square().sum() + 100 * parts.residual_command.sum()
    exact = torch.autograd.grad(loss(), variables)
    for param, grad in zip(variables, exact):
        direction = torch.randn_like(param); direction /= direction.norm()
        original = param.detach().clone(); eps = 1e-6
        with torch.no_grad(): param.copy_(original + eps * direction)
        plus = loss().item()
        with torch.no_grad(): param.copy_(original - eps * direction)
        minus = loss().item()
        with torch.no_grad(): param.copy_(original)
        assert (grad * direction).sum().item() == pytest.approx((plus-minus)/(2*eps), rel=2e-5, abs=2e-8)


def test_geometry_coefficients_stay_in_declared_ranges_after_large_updates():
    p = ResponseMotorPolicy().double()
    for raw in (-1e3, 1e3):
        with torch.no_grad(): p.base_feedback.coefficients.raw.fill_(raw)
        values = p.base_feedback.coefficients.values(upright())
        assert (values >= p.base_feedback.coefficients.lower).all()
        assert (values <= p.base_feedback.coefficients.upper).all()
        x = rotated(2.1, 1).requires_grad_()
        assert torch.isfinite(torch.autograd.grad(p(x).action.sum(), x)[0]).all()


@pytest.mark.parametrize('kwargs', [
    {'residual_amplitude': (-.1, .1, .1, .1)},
    {'residual_gain': (0., .1, .1, .1)},
    {'residual_gain': (float('nan'), .1, .1, .1)},
    {'residual_amplitude': (.1, .1)},
    {'vertical_fraction': 1.}, {'horizontal_accel_limit': 0.},
    {'antipodal_epsilon': 0.},
])
def test_invalid_structural_settings_fail(kwargs):
    with pytest.raises(ValueError): ResponsePolicyConfig(**kwargs)


def test_zero_residual_ablation_retains_native_causal_memory_without_physics_truth():
    p = ResponseMotorPolicy(ResponsePolicyConfig(residual_amplitude=(0., 0., 0., 0.))).double()
    x = upright(); parts = p.components(x)
    assert not parts.residual_command.any()
    s = RaptorSimulator().reset(1, horizon=5, dtype=torch.float64)
    other = replace(s, mass=s.mass*2, inertia=s.inertia*3, motor=s.motor*.2)
    torch.testing.assert_close(p(observation(s)).action, p(observation(other)).action, rtol=0, atol=0)


def test_cli_channel_limits_are_validated_and_roundtrip_as_checkpoint_tuples():
    from tools.train_response_control import parse_args
    args = parse_args(['--residual-amplitude', '.5', '.1', '.1', '.05',
                       '--residual-gain', '.2', '.03', '.03', '.01'])
    assert args.residual_amplitude == (.5, .1, .1, .05)
    assert args.residual_gain == (.2, .03, .03, .01)
    with pytest.raises(SystemExit): parse_args(['--residual-gain', '0', '1', '1', '1'])


def test_checkpoint_cannot_silently_change_constraint_configuration(tmp_path):
    from test_reference_training import args_for, run
    from response_training import require_reference_checkpoint
    run(args_for(tmp_path/'source', 0))
    saved = torch.load(tmp_path/'source/latest.pt', weights_only=True)
    saved['policy_config']['residual_gain'] = (9., 9., 9., 9.)
    with pytest.raises(ValueError, match='constraint binding'):
        require_reference_checkpoint(saved)


@pytest.mark.parametrize('base_only', [False, True])
def test_whole_loop_audit_is_read_only_uses_physics_and_delay_history(base_only):
    from tools.audit_geometric_feedback import audit_policy, perturb, state_distance
    from response_task import initialize
    torch.manual_seed(712)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    sim = RaptorSimulator()
    initial = sim.reset(3, seed=25, horizon=10, dtype=torch.float64)
    before = copy.deepcopy(p.state_dict()); rng = torch.get_rng_state().clone()
    report = audit_policy(p, sim, initial, steps=5, directions=2, base_only=base_only, warmup=2)
    assert not report['certified'] and not report['time_decay_used']
    assert all(row['finite'] for row in report['results'])
    assert all(torch.equal(before[k], value) for k, value in p.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(param.grad is None for param in p.parameters())
    closed = initialize(p, initial)
    vector = torch.zeros(3, 33, dtype=torch.float64); vector[:, 20] = .001
    perturbed = perturb(closed, vector)
    assert not torch.equal(closed.physical.previous_velocity, perturbed.physical.previous_velocity)
    assert (state_distance(closed, perturbed) > 0).all()
    assert perturbed.physical.noise_tape is initial.noise_tape
    assert perturbed.physical.pulse_tape is initial.pulse_tape


def test_whole_loop_audit_follows_the_actual_policy_step():
    from tools.audit_geometric_feedback import audit_policy
    class CheckingSimulator(RaptorSimulator):
        calls = 0
        def step(self, state, action):
            self.calls += 1
            return super().step(state, action)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    sim = CheckingSimulator(); initial = sim.reset(2, horizon=3, dtype=torch.float64)
    report = audit_policy(p, sim, initial, steps=3, directions=2)
    assert sim.calls == 12
    assert len(report['results']) == 2
