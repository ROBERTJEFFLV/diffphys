"""Small crossings and lower-bound speed scoring must work in production FP32."""
from __future__ import annotations

import pytest
import torch

from pass_stop import typed_costs
from pass_stop_physics import segment_ball_geometry
from test_pass_stop import synthetic, GoalConfig


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('scale', [.01, 1., 100.])
def test_millimetre_crossing_keeps_real_entry_and_position_gradient(dtype, scale):
    a = torch.tensor([[-.101, 0., 0.]], dtype=dtype, requires_grad=True) * scale
    b = torch.tensor([[-.099, 0., 0.]], dtype=dtype, requires_grad=True) * scale
    hit, _, enter, exit_ = segment_ball_geometry(a, b, torch.zeros_like(a), a.new_tensor([.1*scale]))
    assert hit.item()
    torch.testing.assert_close(enter, enter.new_tensor([.5]), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(exit_, exit_.new_tensor([1.]))
    ga, gb = torch.autograd.grad(enter.sum(), (a, b))
    expected = a.new_tensor([[-250./scale, 0., 0.]])
    torch.testing.assert_close(ga, expected, rtol=2e-4, atol=1e-6)
    torch.testing.assert_close(gb, expected, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize('velocity', [(2., 0., 0.), (3.5, 0., 0.),
                                      (0., 4., 0.), (-2., 0., 0.)])
def test_at_or_above_minimum_has_no_region_speed_penalty(velocity):
    p = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    q = p.new_tensor(GoalConfig().via_quaternion).expand(3, 1, 4)
    v = p.new_tensor(velocity).expand(3, 1, 3).clone().requires_grad_()
    trace, config = synthetic(p, q=q, v=v)
    _, _, parts = typed_costs(trace, config)
    assert parts['pass_speed'].item() == 0.
    assert 'velocity_window' not in parts
    grad, = torch.autograd.grad(parts['pass_speed'].sum(), (v,))
    assert not grad.any()


def test_subminimum_speed_is_penalized_without_old_velocity_tolerance_slack():
    p = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    v = p.new_tensor([1.9, 0., 0.]).expand(3, 1, 3).clone().requires_grad_()
    trace, config = synthetic(p, v=v)
    parts = typed_costs(trace, config)[2]
    assert parts['pass_speed'].item() == pytest.approx(.4)
    grad, = torch.autograd.grad(parts['pass_speed'].sum(), (v,))
    assert torch.isfinite(grad).all()
    assert (grad*v).sum() < 0


def test_later_fast_revisit_does_not_erase_first_region_speed_shortfall():
    p = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.],
                      [0., 0., 0.], [-1., 0., 0.]], dtype=torch.float64)
    v = p.new_tensor([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
                      [4., 0., 0.], [4., 0., 0.]])[:, None]
    trace, config = synthetic(p, v=v)
    assert typed_costs(trace, config)[2]['pass_speed'].item() == pytest.approx(40.)


def test_constant_velocity_checkpoint_is_not_reinterpreted_as_speed_floor(tmp_path):
    from test_pass_stop_integration import cli
    from response_training import require_reference_checkpoint
    cli(tmp_path/'run', 0)
    saved = torch.load(tmp_path/'run/latest.pt', weights_only=True)
    saved['binding']['protocol']['loss']['mission']['version'] = 'pass-region-pose-velocity-v2'
    with pytest.raises(ValueError, match='objective'):
        require_reference_checkpoint(saved)
