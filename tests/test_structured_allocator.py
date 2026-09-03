from __future__ import annotations

import itertools

import torch

from structured_allocator import ActiveSetBoxQPAllocator


def test_box_qp_matches_clamped_identity_solution_and_has_gradient() -> None:
    allocator = ActiveSetBoxQPAllocator(damping=1.0e-3).double()
    desired = torch.tensor(((2.0, -3.0, 0.25, -0.5),), dtype=torch.float64,
                           requires_grad=True)
    mixer = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    action, diagnostics = allocator(desired, mixer=mixer)
    expected = (desired.detach() / 1.001).clamp(-1.0, 1.0)
    torch.testing.assert_close(action, expected, atol=1.0e-10, rtol=1.0e-10)
    assert float(diagnostics.primal_violation.max()) <= 1.0e-12
    assert float(diagnostics.kkt_residual.max()) <= 1.0e-10
    action.square().sum().backward()
    assert desired.grad is not None and bool(torch.isfinite(desired.grad).all())


def test_box_qp_enforces_intersection_of_box_trust_and_rate_bounds() -> None:
    allocator = ActiveSetBoxQPAllocator(damping=1.0e-3)
    desired = torch.full((2, 4), 100.0)
    mixer = torch.eye(4).expand(2, 4, 4)
    # Keep the trust and rate boxes non-empty; an empty intersection is a
    # genuine infeasible control request and is tested separately below.
    trim = torch.tensor(((0.2, -0.2, 0.1, -0.1),) * 2)
    previous = torch.tensor(((0.0, 0.0, 0.0, 0.0),) * 2)
    action, diagnostics = allocator(
        desired, previous, mixer=mixer, trim=trim, dt=0.01,
        action_delta_cap=0.3, rate_limit=0.5,
    )
    # Rate is the tightest bound here: 0.5 action units/s * .01 s.
    assert bool((action.abs() <= 0.005 + 1.0e-7).all())
    assert float(diagnostics.primal_violation.max()) <= 1.0e-7
    assert bool((diagnostics.rate_limited > 0).all())


def test_box_qp_rejects_incompatible_trust_and_rate_boxes() -> None:
    allocator = ActiveSetBoxQPAllocator()
    desired = torch.zeros(1, 4)
    mixer = torch.eye(4).unsqueeze(0)
    trim = torch.full((1, 4), 0.9)
    previous = torch.zeros(1, 4)
    try:
        allocator(
            desired, previous, mixer=mixer, trim=trim, dt=0.01,
            action_delta_cap=0.1, rate_limit=0.5,
        )
    except RuntimeError as exc:
        assert "empty intersection" in str(exc)
    else:  # pragma: no cover - a silent constraint violation is unacceptable
        raise AssertionError("incompatible allocator constraints must fail closed")


def test_box_qp_objective_is_no_worse_than_all_corners() -> None:
    torch.manual_seed(7)
    allocator = ActiveSetBoxQPAllocator(damping=0.03).double()
    mixer = torch.randn(3, 4, 4, dtype=torch.float64)
    desired = torch.randn(3, 4, dtype=torch.float64)
    trim = torch.randn(3, 4, dtype=torch.float64).clamp(-0.5, 0.5)
    action, diagnostics = allocator(desired, mixer=mixer, trim=trim)
    corners = torch.tensor(tuple(itertools.product((-1.0, 1.0), repeat=4)),
                           dtype=torch.float64)
    for scene in range(3):
        residual = corners @ mixer[scene].T - desired[scene]
        objective = 0.5 * residual.square().sum(-1)
        objective += 0.5 * 0.03 * (corners - trim[scene]).square().sum(-1)
        assert float(diagnostics.objective[scene]) <= float(objective.min()) + 1.0e-9
        assert bool((action[scene].abs() <= 1.0 + 1.0e-12).all())


def test_box_qp_zero_delta_preserves_feasible_trim() -> None:
    allocator = ActiveSetBoxQPAllocator(damping=1.0e-3).double()
    mixer = torch.randn(2, 4, 4, dtype=torch.float64)
    trim = torch.tensor(((0.1, -0.2, 0.3, -0.4), (0.2, 0.3, -0.1, -0.2)),
                        dtype=torch.float64)
    desired = torch.bmm(mixer, trim.unsqueeze(-1)).squeeze(-1)
    action, diagnostics = allocator(desired, mixer=mixer, trim=trim)
    torch.testing.assert_close(action, trim, atol=1.0e-9, rtol=1.0e-9)
    assert float(diagnostics.wrench_residual.max()) <= 1.0e-8
    assert float(diagnostics.kkt_residual.max()) <= 1.0e-8


def test_constraint_diagnostics_report_only_the_bound_that_actually_limits() -> None:
    allocator = ActiveSetBoxQPAllocator(damping=1.0e-3)
    mixer = torch.eye(4).unsqueeze(0)
    desired = torch.full((1, 4), 100.0)
    zero = torch.zeros_like(desired)

    physical, physical_diag = allocator(
        desired, zero, mixer=mixer, trim=zero,
        action_delta_cap=10.0, rate_limit=0.0,
    )
    assert torch.allclose(physical, torch.ones_like(physical))
    assert float(physical_diag.trust_limited.item()) == 0.0
    assert float(physical_diag.rate_limited.item()) == 0.0

    trust, trust_diag = allocator(
        desired, zero, mixer=mixer, trim=zero,
        action_delta_cap=0.2, rate_limit=0.0,
    )
    assert torch.allclose(trust, torch.full_like(trust, 0.2))
    assert float(trust_diag.trust_limited.item()) == 1.0
    assert float(trust_diag.rate_limited.item()) == 0.0

    rate, rate_diag = allocator(
        desired, zero, mixer=mixer, trim=zero, dt=0.01,
        action_delta_cap=0.5, rate_limit=1.0,
    )
    assert torch.allclose(rate, torch.full_like(rate, 0.01))
    assert float(rate_diag.trust_limited.item()) == 0.0
    assert float(rate_diag.rate_limited.item()) == 1.0
