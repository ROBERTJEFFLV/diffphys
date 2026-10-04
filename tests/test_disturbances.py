"""Gaussian law, timestamp causality, privileged-input isolation and gradients."""
from dataclasses import fields, replace
import copy
import math

import pytest
import torch

from env_raptor import RaptorSimulator
from response_noise import (DisturbanceConfig, attach_disturbances, measured_observation,
                            executed_command, raptor_force_std, rotation_error)
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import (rollout, step_costs, TaskLossConfig, observation,
                           _select_rows, initialize, _decay_closed_state)
from response_training import sample_pool, sample_training_scenarios, DEVELOPMENT_SEEDS


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_raptor_force_scale_formula_not_g_times_and_not_hard_clipped(dtype):
    # Endpoint arithmetic is analytical, independent of the RNG or typical draw.
    m = torch.tensor([.02, .027, 5.], dtype=dtype)
    t = torch.tensor([1.5, 2.25, 5.], dtype=dtype)
    u = torch.tensor([0., .5, 1.], dtype=dtype)
    expected = u * .3 * (t - 1) * t * m / 3
    torch.testing.assert_close(raptor_force_std(m, t, u), expected)
    assert float(expected[-1]) == pytest.approx(10.)
    s = RaptorSimulator().reset(8192, seed=49, horizon=2, dtype=dtype)
    normalized = s.external_force / s.force_std[:, None]
    assert abs(float(normalized.mean())) < .025
    assert abs(float(normalized.std()) - 1) < .025
    assert (normalized.abs() > 3).any()  # /3 did NOT become a 3-sigma cap.
    assert (s.external_force.norm(dim=-1) > .1 * s.mass * 9.81).any()
    assert not s.external_torque.any()
    limit = .3 * (s.thrust_to_weight - 1) * s.thrust_to_weight * s.mass / 3
    assert (s.force_std <= limit + 1e-6).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_independent_measurement_draws_have_requested_per_sample_units(dtype):
    s = RaptorSimulator().reset(4096, seed=61, horizon=16, dtype=dtype)
    expected = s.mass.new_tensor([.001]*3 + [.002]*3 + [.001]*3 + [.002]*3)
    assert s.noise_tape.shape == (4096, 17, 12)
    torch.testing.assert_close(s.noise_std, expected.expand(4096, 12), rtol=0, atol=0)
    z = (s.noise_tape / expected).reshape(-1, 12).double()
    assert z.mean(0).abs().max() < .025
    assert (z.std(0) - 1).abs().max() < .025
    cov = torch.cov(z.T)
    assert (cov - torch.diag(cov.diag())).abs().max() < .025
    assert s.velocity_delay.min() >= .010 - 1e-8
    assert s.velocity_delay.max() <= .030 + 1e-8
    assert float(s.velocity_delay.mean()) == pytest.approx(.020, abs=.0004)
    assert float(s.velocity_delay.std()) == pytest.approx(.020/math.sqrt(12), abs=.0004)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_reset_history_holds_first_measurement_and_does_not_leak_motor(dtype):
    s = RaptorSimulator().reset(32, seed=7, horizon=6, dtype=dtype)
    obs = observation(s)
    assert not s.previous_action.any()  # Not 2*true_motor-1.
    assert not torch.equal(s.previous_action, 2*s.motor-1)
    assert s.previous_velocity.shape == (32, 3, 3)
    for lag in range(3):
        assert torch.equal(s.previous_velocity[:, lag], s.velocity)
    torch.testing.assert_close(obs[:, 3:6], s.velocity + s.noise_tape[:, 0, 3:6])
    assert torch.equal(obs[:, 18:22], s.previous_action)


@pytest.mark.parametrize('delay', [0., .010, .015, .020, .025, .030])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_delay_reads_original_noisy_world_measurement_and_both_endpoints(delay, dtype):
    s = RaptorSimulator().reset(1, horizon=5, dtype=dtype)
    tape = torch.zeros_like(s.noise_tape)
    for t in range(6):
        tape[:, t, 3:6] = .1 * t
    s = replace(s, step_index=torch.tensor([4]), noise_tape=tape,
                velocity=torch.full((1, 3), 4., dtype=dtype),
                previous_velocity=torch.tensor([[[3.]*3, [2.]*3, [1.]*3]], dtype=dtype),
                velocity_delay=torch.tensor([delay], dtype=dtype))
    expected = 1.1 * (4 - delay/.01)
    torch.testing.assert_close(observation(s)[:, 3:6], torch.full((1, 3), expected, dtype=dtype))
    future = tape.clone(); future[:, 5] = 1000
    assert torch.equal(observation(s), observation(replace(s, noise_tape=future)))


def test_delayed_observation_has_no_current_velocity_or_future_noise_leak():
    s = RaptorSimulator().reset(2, horizon=6, dtype=torch.float64)
    s = replace(s, step_index=torch.full_like(s.step_index, 4),
                velocity_delay=torch.full_like(s.velocity_delay, .025))
    old = observation(s)
    tape = s.noise_tape.clone(); tape[:, 3:, 3:6] += 100
    altered = replace(s, velocity=s.velocity+1000, noise_tape=tape)
    assert torch.equal(old, observation(altered))  # Only times 1 and 2 contribute.


def test_fractional_history_weights_have_exact_gradients():
    s = RaptorSimulator().reset(1, horizon=5, dtype=torch.float64)
    now = s.velocity.detach().requires_grad_()
    history = s.previous_velocity.detach().requires_grad_()
    s = replace(s, velocity=now, previous_velocity=history,
                step_index=torch.full_like(s.step_index, 4),
                velocity_delay=torch.full_like(s.velocity_delay, .025))
    out = observation(s)[:, 3:6].sum()
    gn, gh = torch.autograd.grad(out, (now, history))
    assert not gn.any()
    torch.testing.assert_close(gh, history.new_tensor([[[0.]*3, [.5]*3, [.5]*3]]))
    def fn(x, h): return observation(replace(s, velocity=x, previous_velocity=h))[:, 3:6]
    assert torch.autograd.gradcheck(fn, (now, history), eps=1e-6)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_so3_observation_and_command_path(dtype):
    sim = RaptorSimulator(); s = sim.reset(64, horizon=4, dtype=dtype)
    r = observation(s)[:, 6:15].reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=dtype).expand(64, 3, 3)
    torch.testing.assert_close(r.transpose(1, 2) @ r, eye)
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(64, dtype=dtype))
    action = torch.linspace(-1, 1, 256, dtype=dtype).reshape(64, 4)
    assert torch.equal(executed_command(s, action), action)
    new = sim.step(s, action)
    assert torch.equal(new.previous_action, action)
    assert torch.equal(new.previous_velocity[:, 0], s.velocity)
    assert torch.equal(new.previous_velocity[:, 1:], s.previous_velocity[:, :-1])
    assert torch.equal(new.external_force, s.external_force)
    assert new.noise_tape.data_ptr() == s.noise_tape.data_ptr()


def test_actor_instantaneous_noninterference_for_all_privileged_fields():
    torch.manual_seed(21)
    model = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    s = RaptorSimulator().reset(4, horizon=6, dtype=torch.float64)
    # Fix accessible measurements/history. Change every physical parameter and
    # private diagnostic field; changing real subsequent dynamics is allowed.
    hidden = ('motor', 'external_force', 'external_torque', 'mass', 'inertia',
              'rotor_positions', 'thrust_coefficients', 'rotor_torque_constant',
              'motor_time_rising', 'motor_time_falling', 'motor_min', 'motor_max',
              'arm_length', 'thrust_to_weight', 'torque_to_inertia',
              'initial_position_limit', 'position_limit', 'guidance', 'force_std', 'noise_std')
    altered = replace(s, **{name: getattr(s, name)*3+7 for name in hidden})
    assert torch.equal(observation(s), observation(altered))
    a = model(observation(s)); b = model(observation(altered))
    assert torch.equal(a.action, b.action)
    assert torch.equal(a.next_state.memory, b.next_state.memory)


def test_cost_and_termination_never_read_measurement_noise():
    s = RaptorSimulator().reset(4, seed=7, horizon=8, dtype=torch.float64)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    trace = rollout(p, RaptorSimulator(), s, 8)
    changed = replace(trace, observations=trace.observations+10000)
    assert torch.equal(step_costs(trace, TaskLossConfig()), step_costs(changed, TaskLossConfig()))
    altered = replace(s, noise_tape=s.noise_tape+10000)
    assert torch.equal(RaptorSimulator.terminated(s), RaptorSimulator.terminated(altered))


def test_rng_replay_compaction_and_physical_sampling_are_separated():
    sim = RaptorSimulator()
    a = sim.reset(16, seed=47, horizon=8)
    b = sim.reset(16, seed=47, horizon=8,
                  disturbances=DisturbanceConfig(position_std=.2, velocity_std=.3))
    for name in ('position', 'velocity', 'orientation', 'omega', 'motor', 'mass',
                 'inertia', 'external_force', 'force_std', 'velocity_delay'):
        assert torch.equal(getattr(a, name), getattr(b, name))
    before = torch.get_rng_state().clone()
    obs = observation(a); observation(a); sim.step(a, torch.zeros_like(a.motor))
    assert torch.equal(before, torch.get_rng_state())
    indices = torch.tensor([9, 1, 14])
    small = _select_rows(a, indices)
    assert small.noise_tape.data_ptr() == a.noise_tape.data_ptr()
    assert torch.equal(observation(small), obs[indices])
    c = sim.reset(16, seed=47, horizon=8, disturbances=DisturbanceConfig.clean())
    for name in ('position', 'velocity', 'orientation', 'omega', 'motor', 'mass', 'previous_action'):
        assert torch.equal(getattr(a, name), getattr(c, name))
    assert not c.external_force.any() and not c.velocity_delay.any()


def test_train_resamples_but_single_eval_bank_is_fixed():
    a, _ = sample_training_scenarios(8, 0, horizon=5)
    b, _ = sample_training_scenarios(8, 1, horizon=5)
    x = sample_pool(4, DEVELOPMENT_SEEDS, horizon=5)
    y = sample_pool(4, DEVELOPMENT_SEEDS, horizon=5)
    for f in fields(x):
        assert torch.equal(getattr(x, f.name), getattr(y, f.name))
    assert not torch.equal(a.mass, b.mass)
    assert not torch.equal(a.noise_tape, b.noise_tape)
    assert torch.equal(a.noise_std[0], x.noise_std[0])


@pytest.mark.parametrize('kwargs', [
    {'position_std': -1.}, {'velocity_std': float('nan')}, {'attitude_std': float('inf')},
    {'omega_std': -1.}, {'velocity_delay_min': -.001}, {'velocity_delay_max': .03001},
    {'velocity_delay_min': .02, 'velocity_delay_max': .01}, {'enabled': 1},
])
def test_invalid_config_fails(kwargs):
    with pytest.raises(ValueError): DisturbanceConfig(**kwargs)


def test_no_sampling_or_reattaching_midflight():
    sim=RaptorSimulator(); s=sim.reset(1,horizon=3)
    s=sim.step(s,torch.zeros_like(s.motor))
    with pytest.raises(ValueError, match='reset'):
        attach_disturbances(s,DisturbanceConfig(),seed=7,horizon=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_noisy_rollout_and_backward():
    s=RaptorSimulator().reset(32,seed=7,horizon=15,device='cuda')
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).cuda()
    tr=rollout(p,RaptorSimulator(),s,15,time_decay=1.)
    step_costs(tr,TaskLossConfig()).sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters())


def test_actual_h500_default_width_graph_with_delayed_measurements():
    """Constructed no-termination fixture, NOT a learned flight-performance test."""
    from response_training import pool_states
    from response_adjoints import collect_rollout, backward_actor
    torch.manual_seed(30)
    sim=RaptorSimulator()
    one=sim.reset(1,seed=7,horizon=500,dtype=torch.float64,disturbances=DisturbanceConfig.clean())
    s=pool_states([one]*32)
    s=attach_disturbances(s,DisturbanceConfig(),seed=47,horizon=500)
    c=s.thrust_coefficients
    motor=(-c[...,1]+(c[...,1].square()+4*c[...,2]*(s.mass[:,None]*9.81/4-c[...,0])).sqrt())/(2*c[...,2])
    s=replace(s,position=torch.zeros_like(s.position),velocity=torch.zeros_like(s.velocity),
              previous_velocity=torch.zeros_like(s.previous_velocity),omega=torch.zeros_like(s.omega),
              orientation=s.orientation.new_tensor([1.,0.,0.,0.]).expand(32,4).clone(),
              motor=motor,position_limit=torch.full_like(s.position_limit,1000))
    p=ResponseMotorPolicy().double()
    with torch.no_grad():
        p.readout.weight.zero_()
        p.readout.bias.zero_()
        # Set only the common operating-point bias in this single-airframe
        # fixture; sampled motor truth is still never an Actor observation.
        hover_logit = torch.atanh(2*motor[0,0]-1)
        coefficients = p.base_feedback.coefficients
        fraction = (hover_logit-coefficients.lower[-1])/(coefficients.upper[-1]-coefficients.lower[-1])
        coefficients.raw[-1].copy_(torch.logit(fraction))
    record=collect_rollout(p,sim,s,TaskLossConfig(),horizon=500,time_decay=1.)
    assert record.metrics['physical_transitions']==32*500
    backward_actor(p,sim,record,TaskLossConfig())
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters())
    assert p.readout.weight.grad.abs().sum()>0
