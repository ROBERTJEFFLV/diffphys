from __future__ import annotations

import math
import unittest

import torch

from env_l2f import L2FParams, L2FSimulator
from multiple_shooting import (
    RecurrentSystemState,
    ShootingBoundary,
    continuity_residuals,
    continuity_rms,
    so3_exp,
    so3_log,
)


class MultipleShootingGeometryTest(unittest.TestCase):
    def test_so3_round_trip_and_orthogonality(self) -> None:
        value = torch.tensor(
            [[0.05, -0.08, 0.12], [-0.15, 0.03, 0.02]], dtype=torch.float64
        )
        rotation = so3_exp(value)
        recovered = so3_log(rotation)
        identity = torch.eye(3, dtype=torch.float64).expand_as(rotation)
        self.assertTrue(torch.allclose(recovered, value, atol=1.0e-10, rtol=1.0e-10))
        self.assertTrue(
            torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1.0e-12)
        )

    def test_zero_continuity_has_finite_zero_gradient(self) -> None:
        torch.manual_seed(3)
        sim = L2FSimulator(L2FParams())
        state = sim.reset(2, device="cpu", dtype=torch.float64)
        hidden = torch.zeros(2, 7, dtype=torch.float64)
        integral = torch.zeros(2, 3, dtype=torch.float64)
        base = RecurrentSystemState(state=state, hidden=hidden, integral=integral)
        boundary = ShootingBoundary(base)
        materialized = boundary.materialize(state)
        residuals = continuity_residuals(base, materialized)
        loss = continuity_rms(residuals)
        gradients = torch.autograd.grad(loss.square(), tuple(boundary.parameters()))
        self.assertEqual(float(loss.item()), 0.0)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertTrue(all(float(gradient.abs().max().item()) == 0.0 for gradient in gradients))

    def test_orientation_continuity_is_log_map_residual(self) -> None:
        sim = L2FSimulator(L2FParams())
        state = sim.reset(1, device="cpu", dtype=torch.float64)
        hidden = torch.zeros(1, 4, dtype=torch.float64)
        integral = torch.zeros(1, 3, dtype=torch.float64)
        predicted = RecurrentSystemState(state=state, hidden=hidden, integral=integral)
        boundary = ShootingBoundary(predicted)
        with torch.no_grad():
            boundary.orientation[0, 2] = 1.0
        shooting = boundary.materialize(state)
        residual = continuity_residuals(predicted, shooting)["orientation"]
        expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(residual, expected, atol=1.0e-10, rtol=1.0e-10))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(so3_log(predicted.state.rotation.transpose(-1, -2) @ shooting.state.rotation)).item()),
            0.1,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
