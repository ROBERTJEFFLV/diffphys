"""Regressions for fixed-command, event-constrained policy training."""
from dataclasses import replace
import math
import pytest
import torch
from diffphys.waypoint_config import WaypointConfig
from diffphys.waypoint_mission import build_mission, goal_packet, advance_target
from diffphys.waypoint_gradients import phr, update_duals, corrected_gradient
from diffphys.env_raptor import RaptorSimulator
from diffphys.response_noise import DisturbanceConfig, measured_observation


def scene(n=2):
    state = RaptorSimulator().reset(n, seed=71, dtype=torch.float64,
                                    horizon=500, disturbances=DisturbanceConfig.clean())
    return build_mission(state, WaypointConfig(), seed=13)


def test_birth_advances_and_final_latches():
    s, m = scene()
    index = torch.zeros(2, dtype=torch.long)
    index = advance_target(s.position, m, index, .1)
    assert index.tolist() == [1, 1]
    index = advance_target(m.points[:, 1] + torch.tensor([.0999, 0., 0.]), m, index, .1)
    assert index.tolist() == [2, 2]
    index = advance_target(s.position, m, index, .1)
    assert index.tolist() == [2, 2]
    packet = goal_packet(measured_observation(s), m, index, .1)
    torch.testing.assert_close(packet[:, :3], m.points[:, 2])


def test_measured_position_and_no_angle_gate():
    s, m = scene()
    index = torch.ones(2, dtype=torch.long)
    far = m.points[:, 1] + .11 * m.direction
    assert advance_target(far, m, index, .1).tolist() == [1, 1]
    near = m.points[:, 1] + .09 * m.direction
    assert advance_target(near, m, index, .1).tolist() == [2, 2]


def test_augmented_lagrangian_no_scene_cancellation():
    g = torch.tensor([[1.], [-1.]], dtype=torch.float64, requires_grad=True)
    mu = torch.zeros_like(g)
    value = phr(torch.zeros(2), g, mu, 2.)
    assert value.tolist() == [1., 0.]
    assert torch.autograd.grad(value.sum(), g)[0].tolist() == [[2.], [0.]]
    assert update_duals(g, mu, 2.).tolist() == [[2.], [0.]]


def test_corrected_gradient_exact_quadratic():
    theta = torch.tensor([.3, -.4, .2], dtype=torch.float64)
    center = 2 * theta
    def evaluate(x):
        return x.square().sum()
    out, info = corrected_gradient(theta, center, evaluate, sigma=.01,
                                   directions=4, seed=123)
    torch.testing.assert_close(out, center, atol=1e-13, rtol=1e-13)
    assert info['extra_forward_rollouts'] == 8
