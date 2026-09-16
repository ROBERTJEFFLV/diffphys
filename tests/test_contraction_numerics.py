"""Numerical state clipping is not evidence of physical non-expansion."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_l2f import L2FSimulator
from response_contraction import contraction_loss, worst_direction
from test_contraction import setup


@pytest.mark.parametrize('field', ['position', 'velocity', 'omega'])
@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('audit', ['sampled', 'worst'])
def test_numerical_clamp_is_not_accepted_as_contraction(field, sign, audit):
    actor, simulator, closed, metric, config = setup(n=1)

    class ClampedSimulator(L2FSimulator):
        def step(self, state, action):
            result = super().step(state, action)
            value = getattr(result, field)
            # Model the already-existing physical numerical clamp firing on a
            # terminal transition. Its zero derivative must not look stabilizing.
            value = (value * 0 + sign * 200000).clamp(-100000, 100000)
            return replace(result, **{field: value})

    simulated = ClampedSimulator(simulator.params)
    with pytest.raises(FloatingPointError, match='numerical state clamp'):
        if audit == 'sampled':
            contraction_loss(actor, simulated, metric, closed, config, seed=9)
        else:
            worst_direction(actor, simulated, metric, closed, config)


def test_ordinary_boundary_crossing_is_still_a_valid_terminal_prefix():
    actor, simulator, closed, metric, config = setup(n=1)
    position = closed.physical.position.clone()
    velocity = closed.physical.velocity.clone()
    position[0] = 0
    position[0, 0] = closed.physical.position_limit[0] - 1e-8
    velocity[0, 0] = 1
    closed = replace(closed, physical=replace(closed.physical,
                                             position=position, velocity=velocity))
    loss, report = contraction_loss(actor, simulator, metric, closed, config, seed=9)
    assert report['terminal_intervals'] == 1
    assert report['complete_intervals'] == 0
    assert report['true_forward_transitions'] == config.directions + 1
    assert report['certified'] is False
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
