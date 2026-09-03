from __future__ import annotations

import torch

from structured_stability import (
    default_phase_space_metric,
    local_stability_report,
    quadratic_metric,
    smooth_contraction_objective,
    solve_discrete_lyapunov,
)


def test_default_phase_space_metric_is_positive_and_pole_derived() -> None:
    metric = default_phase_space_metric(dtype=torch.float64)
    assert metric.matrix.shape == (15, 15)
    assert float(torch.linalg.eigvalsh(metric.matrix).min()) > 0.0
    assert 0.0 <= metric.linear_model_retention < 1.0


def test_phase_metric_uses_five_to_one_tilt_omega_normalization() -> None:
    metric = default_phase_space_metric(dtype=torch.float64, tilt_scale=0.1, omega_scale=0.5)
    changed = default_phase_space_metric(dtype=torch.float64, tilt_scale=0.1, omega_scale=1.0)
    assert not torch.allclose(metric.matrix, changed.matrix)
    assert torch.isfinite(metric.matrix).all()
    # The endpoint chart q=(R^T z_d)[:2] has derivatives (-omega_y,+omega_x).
    assert metric.matrix[6, 9] < 0.0
    assert metric.matrix[7, 8] > 0.0


def test_discrete_lyapunov_metric_matches_equation() -> None:
    a = torch.tensor(((0.8, 0.1), (0.0, 0.7)), dtype=torch.float64)
    q = torch.eye(2, dtype=torch.float64)
    metric = solve_discrete_lyapunov(a, q)
    residual = a.T @ metric.matrix @ a - metric.matrix + q
    assert torch.linalg.matrix_norm(residual) < 1.0e-9
    assert 0.0 <= metric.linear_model_retention < 1.0


def test_metric_and_smooth_contraction_have_gradients_on_both_times() -> None:
    error = torch.tensor(((1.0, 0.0), (0.9, 0.0)), requires_grad=True)
    energy = quadratic_metric(error, torch.eye(2))
    loss = smooth_contraction_objective(energy, retention=0.8).sum()
    loss.backward()
    assert error.grad is not None
    assert torch.isfinite(error.grad).all()
    assert float(error.grad[0].abs().sum()) > 0.0
    assert float(error.grad[1].abs().sum()) > 0.0


def test_local_report_detects_stable_and_unstable_maps() -> None:
    stable = local_stability_report(lambda x: 0.8 * x, torch.zeros(3), horizon_steps=5)
    unstable = local_stability_report(lambda x: 1.1 * x, torch.zeros(3), horizon_steps=5)
    assert stable.locally_schur_stable
    assert stable.spectral_radius < 1.0
    assert not unstable.locally_schur_stable
    assert unstable.finite_time_gain > 1.0
