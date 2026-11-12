"""Keep speed-angle recovery and the restored independent quadratic point costs."""
from __future__ import annotations

import math

import pytest
import torch

from plane_tasks import directed_velocity_ratio, velocity_z_orthogonality_ratio


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_velocity_angle_ratios_cannot_improve_when_speed_decreases(dtype):
    generator = torch.Generator().manual_seed(413)
    directions = torch.randn(32, 3, generator=generator, dtype=dtype)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    q = torch.randn(32, 4, generator=generator, dtype=dtype)
    q = torch.nn.functional.normalize(q, dim=-1)
    target = torch.tensor((1., 0., 0.), dtype=dtype)
    speeds = torch.tensor((0., .01, .1, .75, 1.49, 1.5, 3., 10.), dtype=dtype)
    velocity = speeds[:, None, None]*directions[None]
    floor = torch.full((32,), 1.5, dtype=dtype)
    for ratio in (directed_velocity_ratio(velocity, target, floor, 1.),
                  velocity_z_orthogonality_ratio(q, velocity, floor, 1.)):
        assert bool(torch.isfinite(ratio).all())
        assert bool((ratio[1:] <= ratio[:-1]+2e-5).all())
        torch.testing.assert_close(ratio[-1], ratio[-2], rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('speed', [0., .1, .75, 1.499, 1.5, 1.501, 3.])
def test_velocity_angle_gradients_are_finite_at_zero_and_speed_floor(dtype, speed):
    v = torch.tensor([[speed, .2*speed, .3*speed]], dtype=dtype, requires_grad=True)
    q = torch.tensor([[math.sqrt(.5), 0., math.sqrt(.5), 0.]], dtype=dtype,
                     requires_grad=True)
    floor = v.new_tensor([1.5])
    value = (directed_velocity_ratio(v, v.new_tensor((1., 0., 0.)), floor, 1.)
             + velocity_z_orthogonality_ratio(q, v, floor, 1.)).sum()
    gradients = torch.autograd.grad(value, (v, q))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)


def test_correct_direction_has_forward_recovery_gradient_at_rest():
    v = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    ratio = directed_velocity_ratio(v, v.new_tensor((1., 0., 0.)), v.new_tensor([1.5]), 1.)
    gradient, = torch.autograd.grad(ratio.sum(), v)
    assert gradient[0, 0] < 0 and bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize('speed', [.7, 1.7])
def test_velocity_angle_gradients_match_finite_differences(speed):
    v = torch.tensor([[speed, .2, .3]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[.8, .1, .2, .3]], dtype=torch.float64)
    q = (q/q.norm(dim=-1, keepdim=True)).requires_grad_()
    def ratios(velocity, quaternion):
        floor = velocity.new_tensor([1.5])
        return torch.stack((directed_velocity_ratio(velocity, velocity.new_tensor((1., 0., 0.)), floor, 1.),
                            velocity_z_orthogonality_ratio(quaternion, velocity, floor, 1.)))
    assert torch.autograd.gradcheck(ratios, (v, q), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_terminal_quadratic_cost_does_not_change_with_pass_quality(dtype):
    from dataclasses import asdict, replace
    from pass_stop import GoalConfig, PassStopLossConfig, typed_costs
    from test_pass_stop import synthetic

    goal = GoalConfig(relative_to_boundary=False,
                      points=((-1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                      via_quaternion=(1., 0., 0., 0.),
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    base = PassStopLossConfig(.01, .01, 25., mission=asdict(goal))
    active = replace(base, mission=asdict(replace(goal, random_via=True, plane_task=True,
                                               plane_crossing_only=True)))
    p = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]], dtype=dtype)
    v = p.new_tensor((2., 0., 0.)).expand(3, 1, 3)
    good_q = p.new_tensor((1., 0., 0., 0.)).expand(3, 1, 4).clone()
    bad_q = good_q.clone()
    bad_q[1, 0] = p.new_tensor((math.sqrt(.5), 0., math.sqrt(.5), 0.))
    good, _ = synthetic(p, q=good_q, v=v, config=base)
    bad, _ = synthetic(p, q=bad_q, v=v, config=base)
    good_cost, good_metrics, good_parts = typed_costs(good, active)
    bad_cost, bad_metrics, bad_parts = typed_costs(bad, active)
    assert good_metrics['pass_fraction'] == 1 and bad_metrics['pass_fraction'] == 0
    torch.testing.assert_close(bad_parts['stop'], good_parts['stop'], rtol=0, atol=0)
    slower_v = v.clone()
    slower_v[-1] = p.new_tensor((1., 0., 0.))
    slower, _ = synthetic(p, q=good_q, v=slower_v, config=base)
    _, _, slower_parts = typed_costs(slower, active)
    # Same end balance; changing only final speed 2 -> 1m/s costs
    # 10*((20-1)^2-(10-1)^2) = 2800 at the 0.1m/s threshold.
    assert (good_parts['stop']-slower_parts['stop']).item() == pytest.approx(2800., abs=1e-3)
    assert bad_cost.item() > good_cost.item()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_missed_pass_uses_original_quadratic_position_penalty(dtype):
    from dataclasses import asdict, replace
    from pass_stop import GoalConfig, PassStopLossConfig, typed_costs
    from test_pass_stop import synthetic

    goal = GoalConfig(relative_to_boundary=False,
                      points=((-1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                      via_quaternion=(1., 0., 0., 0.),
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    base = PassStopLossConfig(.01, .01, 25., mission=asdict(goal))
    active = replace(base, mission=asdict(replace(goal, random_via=True, plane_task=True,
                                               plane_crossing_only=True)))
    p = torch.tensor([[-1., .3, 0.], [0., .3, 0.], [1., .3, 0.]], dtype=dtype)
    v = p.new_tensor((2., 0., 0.)).expand(3, 1, 3)
    trace, _ = synthetic(p, v=v, config=base)
    _, metrics, parts = typed_costs(trace, active)
    assert metrics['pass_fraction'] == 0
    # Closest approach 0.3m at a 0.1m tolerance: 10*(3-1)^2.
    assert parts['pass_position'].item() == pytest.approx(40., abs=2e-5)
