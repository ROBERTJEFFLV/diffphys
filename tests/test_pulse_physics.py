"""Pulse timing, force-at-point mechanics and reset distribution (no training)."""
from dataclasses import fields, replace
import math
import numpy as np
import pytest
import torch

from env_raptor import RaptorSimulator, IMMUTABLE_TAPES
from response_noise import DisturbanceConfig, pulse_at, measured_observation


def with_pulse(s, force, point, start=0, duration=10):
    tape = torch.zeros_like(s.pulse_tape)
    active = torch.zeros_like(s.pulse_active_tape)
    tape[:, start:start+duration, :3] = torch.as_tensor(force, dtype=s.mass.dtype, device=s.mass.device)
    tape[:, start:start+duration, 3:] = torch.as_tensor(point, dtype=s.mass.dtype, device=s.mass.device)
    active[:, start:start+duration] = True
    return replace(s, pulse_tape=tape, pulse_active_tape=active)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_harder_reset_distribution_and_guidance(dtype):
    s = RaptorSimulator().reset(8192, seed=74, horizon=1, dtype=dtype,
                               disturbances=DisturbanceConfig.clean())
    normal = s.guidance == 0
    angle = 2*s.orientation[:, 0].clamp(-1, 1).acos()
    assert angle.max() <= 2*math.pi/3 + 1e-6
    assert angle.max() > math.radians(119)
    assert float((angle[normal] > math.pi/2).float().mean()) == pytest.approx(.25, abs=.025)
    for value, limit in [(s.velocity, 2.5), (s.omega, 2.2)]:
        assert value.abs().max() <= limit + 1e-6
        assert value[normal].max() > .99*limit and value[normal].min() < -.99*limit
        assert abs(float(value[normal].double().std()) - limit/math.sqrt(3)) < .03
        assert not value[~normal].any()
    torch.testing.assert_close(s.initial_position_limit, 10*s.arm_length)
    torch.testing.assert_close(s.position_limit, 20*s.arm_length)
    assert (s.position.abs() <= s.initial_position_limit[:, None]).all()
    assert 0 <= s.motor.min() < s.motor.max() <= .5
    assert not s.previous_action.any()
    assert not s.position[~normal].any() and not angle[~normal].any()
    assert .08 < float((~normal).float().mean()) < .12


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_pulse_schedule_durations_gaussian_scale_and_body_points(dtype):
    s = RaptorSimulator().reset(512, seed=41, horizon=500, dtype=dtype)
    assert s.pulse_tape.shape == (512, 501, 6)
    assert s.pulse_active_tape.shape == (512, 501)
    assert s.pulse_active_tape.dtype == torch.bool
    assert not s.pulse_active_tape[:, -1].any()  # No transition beyond H.
    first, standardized, fractions = [], [], []
    for row in range(512):
        active = s.pulse_active_tape[row]
        previous = torch.cat((torch.zeros(1, dtype=torch.bool), active[:-1]))
        starts = (active & ~previous).nonzero().flatten()
        assert 4 <= len(starts) <= 7
        first.append(int(starts[0]))
        assert ((starts[1:]-starts[:-1] >= 80) & (starts[1:]-starts[:-1] <= 120)).all()
        for start in starts.tolist():
            end = min(start+10, 500)
            assert active[start:end].all()
            assert not active[end]
            value = s.pulse_tape[row, start]
            assert torch.equal(s.pulse_tape[row, start:end], value.expand(end-start, 6))
            force, point = value[:3], value[3:]
            standardized.append(force/s.force_std[row])
            # Uniform point along one of the four COM-to-rotor arms, not a guessed mesh.
            assert point[2] == 0
            torch.testing.assert_close(point[0].abs(), point[1].abs())
            fraction = point.norm()/s.arm_length[row]
            assert 0 <= fraction <= 1+1e-6
            fractions.append(float(fraction))
        assert not s.pulse_tape[row, ~active].any()
    assert 0 <= min(first) < max(first) < 100
    assert np.std(first) > 25
    assert np.mean(first) == pytest.approx(49.5, abs=4)
    z = torch.stack(standardized).double()
    assert z.mean(0).abs().max() < .08
    assert (z.std(0)-1).abs().max() < .06
    assert (z.abs() > 3).any()  # No three-sigma clipping.
    assert np.mean(fractions) == pytest.approx(.5, abs=.025)


def test_enabling_pulses_preserves_other_rng_streams_and_hidden_observations():
    sim = RaptorSimulator()
    a = sim.reset(32, seed=9, horizon=160, dtype=torch.float64)
    b = sim.reset(32, seed=9, horizon=160, dtype=torch.float64,
                  disturbances=DisturbanceConfig(pulse_enabled=False))
    for f in fields(a):
        if f.name not in ('pulse_tape', 'pulse_active_tape'):
            assert torch.equal(getattr(a, f.name), getattr(b, f.name)), f.name
    assert torch.equal(measured_observation(a), measured_observation(b))
    assert a.pulse_active_tape.any() and not b.pulse_active_tape.any()
    altered = replace(a, pulse_tape=a.pulse_tape+1000,
                      pulse_active_tape=~a.pulse_active_tape, force_std=a.force_std+100)
    assert torch.equal(measured_observation(a), measured_observation(altered))
    before = torch.get_rng_state().clone()
    out = sim.step(a, torch.zeros_like(a.motor))
    assert torch.equal(before, torch.get_rng_state())
    for name in ('pulse_tape', 'pulse_active_tape'):
        assert name in IMMUTABLE_TAPES
        assert getattr(out, name).data_ptr() == getattr(a, name).data_ptr()
    assert torch.equal(out.external_force, a.external_force)


def test_pulse_zero_lever_gives_exact_translational_impulse_and_no_rotation():
    sim = RaptorSimulator()
    s = sim.reset(1, seed=7, horizon=20, dtype=torch.float64)
    s = replace(s, position=s.position*0, velocity=s.velocity*0, omega=s.omega*0,
                orientation=s.orientation.new_tensor([[1., 0., 0., 0.]]),
                external_force=s.external_force*0, thrust_coefficients=s.thrust_coefficients*0)
    force = s.mass[0]*s.mass.new_tensor([.7, -.2, .5])
    a = with_pulse(s, force, [0., 0., 0.])
    b = replace(a, pulse_tape=torch.zeros_like(a.pulse_tape))
    for step in range(20):
        a = sim.step(a, torch.zeros_like(a.motor)); b = sim.step(b, torch.zeros_like(b.motor))
        elapsed = (step+1)*.01
        on_time = min(elapsed, .1)
        torch.testing.assert_close(a.velocity-b.velocity, (force/s.mass[0]*on_time)[None], atol=1e-13, rtol=1e-12)
        distance = .5*on_time**2 + max(0, elapsed-.1)*.1
        torch.testing.assert_close(a.position-b.position, (force/s.mass[0]*distance)[None], atol=1e-13, rtol=1e-12)
        assert not a.omega.any()


def numpy_step(s, action):
    """Independent NumPy RK4, using quaternion double-cross rotation both ways."""
    def x(name): return getattr(s, name).detach().numpy()[0]
    rotor, c, inertia, mass = x('rotor_positions'), x('thrust_coefficients'), x('inertia'), x('mass')
    row, time = int(x('noise_row')), int(x('step_index'))
    tape = s.pulse_tape.detach().numpy()
    pulse = tape[row, 0 if tape.shape[1] == 1 else time]
    fw, rb = pulse[:3], pulse[3:]
    command = np.clip(action.detach().numpy()[0], -1, 1)
    target = x('motor_min')+(command+1)*.5*(x('motor_max')-x('motor_min'))
    initial = np.concatenate([x('position'), x('velocity'), x('orientation'), x('omega'), x('motor')])
    def rotate(q, v):
        twice = 2*np.cross(q[1:], v)
        return v+q[0]*twice+np.cross(q[1:], twice)
    def rhs(z):
        p, v, q, w, motor = z[:3], z[3:6], z[6:10], z[10:13], z[13:]
        f = c[:, 0]+c[:, 1]*motor+c[:, 2]*motor**2
        rotor_forces = np.column_stack((np.zeros(4), np.zeros(4), f))
        torque = np.cross(rotor, rotor_forces).sum(0)
        torque[2] += np.sum(np.array([-1, 1, -1, 1])*x('rotor_torque_constant')*f)
        conjugate = q*np.array([1, -1, -1, -1])
        torque += x('external_torque')+np.cross(rb, rotate(conjugate, fw))
        acc = rotate(q, rotor_forces.sum(0))/mass+np.array([0, 0, -9.81])+(x('external_force')+fw)/mass
        qdot = .5*np.r_[-np.dot(q[1:], w), q[0]*w+np.cross(q[1:], w)]
        wdot = (torque-np.cross(w, inertia*w))/inertia
        tau = np.where(target >= motor, x('motor_time_rising'), x('motor_time_falling'))
        return np.r_[v, acc, qdot, wdot, (target-motor)/tau]
    k1 = rhs(initial); k2 = rhs(initial+.005*k1)
    k3 = rhs(initial+.005*k2); k4 = rhs(initial+.01*k3)
    result = initial+.01/6*(k1+2*k2+2*k3+k4)
    result[6:10] /= np.linalg.norm(result[6:10])
    result[13:] = np.clip(result[13:], x('motor_min'), x('motor_max'))
    return torch.from_numpy(result)


@pytest.mark.parametrize('point_scale', [0., .35, 1.])
def test_force_at_body_point_matches_independent_rk4(point_scale):
    sim = RaptorSimulator(); s = sim.reset(1, seed=57, horizon=25, dtype=torch.float64)
    force = s.mass[0]*s.mass.new_tensor([.4, -.7, 1.3])
    s = with_pulse(s, force, point_scale*s.rotor_positions[0, 2], start=2)
    for step in range(25):
        action = s.motor.new_tensor([[.2, .1, -.2, .15]])
        expected = numpy_step(s, action)
        s = sim.step(s, action)
        actual = torch.cat((s.position[0], s.velocity[0], s.orientation[0], s.omega[0], s.motor[0]))
        torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-11)


def test_off_center_pulse_orientation_and_action_gradients():
    sim = RaptorSimulator(); s = sim.reset(1, seed=57, horizon=2, dtype=torch.float64)
    s = with_pulse(s, s.mass[0]*s.mass.new_tensor([.4, -.7, 1.3]), .6*s.rotor_positions[0, 2])
    action = torch.full((1, 4), .12, dtype=torch.float64, requires_grad=True)
    q = s.orientation.detach().requires_grad_()
    def fn(a, orientation):
        out = sim.step(replace(s, orientation=orientation), a)
        return torch.cat((out.position, out.velocity, out.orientation, out.omega, out.motor), -1)
    assert torch.autograd.gradcheck(fn, (action, q), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_invalid_pulse_switch_and_clean_fixture():
    with pytest.raises(ValueError): DisturbanceConfig(pulse_enabled=1)
    s = RaptorSimulator().reset(3, horizon=100, disturbances=DisturbanceConfig.clean())
    assert s.pulse_tape.shape == (3, 1, 6)
    assert not s.pulse_tape.any() and not s.pulse_active_tape.any()
    force, point = pulse_at(s)
    assert not force.any() and not point.any()
