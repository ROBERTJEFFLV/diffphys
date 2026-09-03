from __future__ import annotations

import torch

from smooth_risk import (
    FixedThetaMicrobatchCVaR,
    rockafellar_uryasev_cvar,
    smooth_positive,
    solve_smooth_cvar_eta,
)


def test_huber_positive_is_zero_below_and_linear_above() -> None:
    values = torch.tensor([-1.0, 0.0, 0.1, 2.0])
    result = smooth_positive(values, beta=0.2, mode="huber")
    assert result[0] == 0
    assert torch.allclose(result[-1], torch.tensor(1.9))


def test_fixed_eta_microbatch_matches_concatenated_ru_expression() -> None:
    values = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0], requires_grad=True)
    eta = torch.tensor(1.0)
    expected = rockafellar_uryasev_cvar(values, alpha=0.8, eta=eta, beta=0.05, mode="huber")
    accumulator = FixedThetaMicrobatchCVaR(alpha=0.8, eta=eta, beta=0.05, mode="huber")
    accumulator.update(values[:2])
    accumulator.update(values[2:])
    actual = accumulator.finalize().value
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(values.grad).all()


def test_default_eta_solves_smooth_tail_mass_and_gradient_is_finite() -> None:
    values = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    summary = rockafellar_uryasev_cvar(values, alpha=2.0 / 3.0, beta=0.1, return_summary=True)
    tail_mass = torch.sigmoid((values.detach() - summary.eta) / 0.1).mean()
    torch.testing.assert_close(tail_mass, torch.tensor(1.0 / 3.0), atol=1.0e-5, rtol=0.0)
    summary.value.backward()
    assert summary.sample_count == 3
    assert torch.isfinite(values.grad).all()


def test_smooth_eta_is_stationary_and_detached() -> None:
    values = torch.tensor([0.0, 0.2, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    eta = solve_smooth_cvar_eta(values, alpha=0.75, beta=0.2)
    assert not eta.requires_grad
    tail_mass = torch.sigmoid((values.detach() - eta) / 0.2).mean()
    torch.testing.assert_close(tail_mass, torch.tensor(0.25, dtype=torch.float64), atol=1.0e-10, rtol=0.0)


def test_quantile_free_microbatch_path_fits_one_global_smooth_eta() -> None:
    values = torch.tensor([0.0, 0.2, 1.0, 3.0], dtype=torch.float64)
    expected = rockafellar_uryasev_cvar(values, alpha=0.75, beta=0.2)
    accumulator = FixedThetaMicrobatchCVaR(alpha=0.75, beta=0.2)
    accumulator.update(values[:2])
    accumulator.update(values[2:])
    summary = accumulator.finalize()
    torch.testing.assert_close(summary.eta, solve_smooth_cvar_eta(values, alpha=0.75, beta=0.2))
    torch.testing.assert_close(summary.value, expected)
