"""Attitude-transition regularization: independent SO(3) values and gradients."""
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from env_raptor import RaptorSimulator, quaternion_rotation
from response_noise import DisturbanceConfig
from response_policy import ResponseMotorPolicy
import response_task as task


def rotation_q(angle, axis=2, dtype=torch.float64):
    angle = torch.as_tensor(angle, dtype=dtype)
    entries = [torch.cos(angle/2), angle*0, angle*0, angle*0]
    entries[axis+1] = torch.sin(angle/2)
    return torch.stack(entries)


def fixture(before, after, *, steps=1):
    dtype, device = before.dtype, before.device
    return SimpleNamespace(
        pre_positions=torch.zeros(steps, 1, 3, dtype=dtype, device=device, requires_grad=True),
        pre_orientations=before.reshape(1, 1, 4).expand(steps, 1, 4),
        post_orientations=after.reshape(1, 1, 4).expand(steps, 1, 4),
        positions=torch.zeros(steps, 1, 3, dtype=dtype, device=device),
        actions=torch.zeros(steps, 1, 4, dtype=dtype, device=device),
        action_deltas=torch.zeros(steps, 1, 4, dtype=dtype, device=device, requires_grad=True),
        valid=torch.ones(steps, 1, dtype=torch.bool, device=device),
        initial=SimpleNamespace(position_limit=torch.tensor([10.], dtype=dtype, device=device)))


def config():
    return task.TaskLossConfig(.01, .01, .2)


def test_unchanging_tilted_attitude_has_no_attitude_cost():
    q = rotation_q(math.atan(.2), axis=1)
    trace = fixture(q, q)
    assert task.task_loss(trace, config()).item() == 0


def test_yaw_transition_is_penalized_without_an_absolute_heading_target():
    trace = fixture(rotation_q(0), rotation_q(.3))
    assert task.task_loss(trace, config()).item() == pytest.approx(.2*(1-math.cos(.3)), rel=1e-13)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('angle', [0., 1e-7, .01, .3, math.pi/2, math.pi])
def test_relative_rotation_value_and_angle_gradient(axis, angle):
    theta = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
    trace = fixture(rotation_q(0), rotation_q(theta, axis=axis))
    loss = task.task_loss(trace, config())
    expected = .4*math.sin(angle/2)**2
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-12, abs=1e-28)
    gradient, = torch.autograd.grad(loss, (theta,))
    assert float(gradient) == pytest.approx(.2*math.sin(angle), rel=1e-12, abs=1e-16)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_identical_arbitrary_attitude_has_zero_cost_and_finite_zero_gradient(dtype):
    raw = torch.tensor([.3, .2, -.5, .7], dtype=dtype)
    before = (raw/raw.norm()).detach().requires_grad_()
    after = before.detach().clone().requires_grad_()
    trace = fixture(before, after, steps=50)
    loss = task.task_loss(trace, config())
    gradients = torch.autograd.grad(loss, (before, after))
    assert loss.item() == 0
    assert all(torch.isfinite(g).all() and not g.any() for g in gradients)


def test_quaternion_sign_equivalence_and_matrix_reference():
    before = torch.tensor([.3, .2, -.5, .7], dtype=torch.float64)
    before = before/before.norm()
    after = rotation_q(.17, axis=1)
    base = task.task_loss(fixture(before, after), config())
    for a, b in ((-before, after), (before, -after), (-before, -after)):
        torch.testing.assert_close(task.task_loss(fixture(a, b), config()), base, rtol=1e-14, atol=1e-15)
    rb, ra = quaternion_rotation(before), quaternion_rotation(after)
    matrix_reference = .2*(3-torch.trace(rb.T@ra))/2
    torch.testing.assert_close(base, matrix_reference, rtol=1e-14, atol=1e-15)


def test_failure_transition_is_charged_once_and_frozen_attitudes_are_not_charged():
    # One executed 90-degree rotation crosses the boundary; the rest is padding.
    trace = fixture(rotation_q(0), rotation_q(math.pi/2), steps=3)
    trace.valid[1:] = False
    trace.positions[0, 0, 0] = 11.
    components = task.step_cost_components(trace, config())
    assert components['attitude_delta'][:, 0].tolist() == pytest.approx([.2/3, 0., 0.])
    assert components['dead'][:, 0].tolist() == [2., 0., 0.]
    assert components['terminal'][:, 0].tolist() == pytest.approx([200/3, 0., 0.])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_attitude_value_and_both_endpoint_gradients_match_float64_reference():
    angles = torch.tensor([.19, -.07], dtype=torch.float64, requires_grad=True)
    cpu_value = task.task_loss(fixture(rotation_q(angles[0], axis=0),
                                      rotation_q(angles[1], axis=1)), config())
    cpu_gradient, = torch.autograd.grad(cpu_value, (angles,))
    cuda_angles = angles.detach().float().cuda().requires_grad_()
    cuda_value = task.task_loss(fixture(rotation_q(cuda_angles[0], axis=0, dtype=torch.float32),
                                       rotation_q(cuda_angles[1], axis=1, dtype=torch.float32)), config())
    cuda_gradient, = torch.autograd.grad(cuda_value, (cuda_angles,))
    assert cuda_value.dtype == torch.float32 and cuda_value.device.type == 'cuda'
    torch.testing.assert_close(cuda_value.detach().cpu().double(), cpu_value.detach(), rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(cuda_gradient.detach().cpu().double(), cpu_gradient, rtol=2e-6, atol=1e-8)


def test_both_endpoint_gradients_match_directional_finite_difference():
    x = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
    def value(angles):
        return task.task_loss(fixture(rotation_q(angles[0], axis=0),
                                     rotation_q(angles[1], axis=1)), config())
    gradient, = torch.autograd.grad(value(x), (x,))
    direction = torch.tensor([.6, -.3], dtype=torch.float64)
    epsilon = 1e-6
    fd = (value(x.detach()+epsilon*direction)-value(x.detach()-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close(gradient@direction, fd, rtol=1e-8, atol=1e-11)
    assert gradient[0] != 0 and gradient[1] != 0


def test_logs_name_attitude_changes_not_tilt():
    trace = fixture(rotation_q(0), rotation_q(.3))
    components = task.task_loss_components(trace, config())
    assert set(components) == {'position', 'attitude_delta', 'action_delta', 'dead', 'terminal'}
    assert sum(components.values()) == pytest.approx(float(task.task_loss(trace, config()).detach()))


def test_invalid_post_attitude_padding_is_masked_but_valid_nonfinite_is_rejected():
    trace = fixture(rotation_q(0), rotation_q(0), steps=2)
    trace.post_orientations = trace.post_orientations.clone().requires_grad_()
    trace.valid[1] = False
    with torch.no_grad():
        trace.post_orientations[1] = float('nan')
    loss = task.task_loss(trace, config())
    gradient, = torch.autograd.grad(loss, (trace.post_orientations,))
    assert loss.item() == 0 and torch.isfinite(gradient).all()
    trace.valid[1] = True
    with pytest.raises(FloatingPointError):
        task.task_loss(trace, config())


def test_every_executed_transition_including_last_and_chunk_boundary_counts():
    torch.manual_seed(123)
    policy = ResponseMotorPolicy().double()
    simulator = RaptorSimulator()
    initial = simulator.reset(2, seed=123, dtype=torch.float64, horizon=60,
                              disturbances=DisturbanceConfig.clean())
    # Remove early position termination without altering the actual equations.
    initial = replace(initial, position_limit=torch.full_like(initial.position_limit, 1000.))
    full = task.rollout(policy, simulator, initial, 60)
    first = task.rollout(policy, simulator, initial, 50)
    second = task.rollout(policy, simulator, first.end, 10)
    torch.testing.assert_close(full.post_orientations[49], second.pre_orientations[0], rtol=0, atol=0)
    full_cost = task.step_costs(full, config())
    chunks = torch.cat((task.step_costs(first, config(), start=0, horizon=60),
                        task.step_costs(second, config(), start=50, horizon=60)))
    for name in ('pre_positions', 'pre_orientations', 'post_orientations', 'actions', 'action_deltas'):
        torch.testing.assert_close(getattr(full, name),
                                   torch.cat((getattr(first, name), getattr(second, name))),
                                   rtol=0, atol=0)
    # Existing smooth_l2 reductions can round differently across tensor lengths;
    # the actual trajectory and the new attitude costs remain bitwise identical.
    torch.testing.assert_close(full_cost, chunks, rtol=1e-14, atol=1e-16)
    parameters = tuple(policy.parameters())
    a = torch.autograd.grad(full_cost.mean(1).sum(), parameters)
    b = torch.autograd.grad(chunks.mean(1).sum(), parameters)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=2e-11, atol=1e-12)
