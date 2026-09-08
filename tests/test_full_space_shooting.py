from __future__ import annotations

import torch
import pytest

from full_space_shooting import (
    BoundaryLayout,
    FullSpaceProblem,
    boundary_jvp,
    boundary_vjp,
    solve_boundary_lm,
    solve_joint_sqp_step,
    so3_exp,
    so3_local_residual,
    so3_retract,
    trust_region_diagnostics,
    _cg_solve,
)


def _problem() -> FullSpaceProblem:
    initial = torch.tensor([1.0, -0.5], dtype=torch.float64)
    theta = torch.tensor([0.25], dtype=torch.float64)
    # Stable affine map; two independently stored boundaries.
    def segment(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.stack((0.7 * z[0] + t[0], 0.6 * z[1] - 0.5 * t[0]))

    boundaries = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    return FullSpaceProblem(initial, boundaries, theta, segment, BoundaryLayout(2))


def test_boundaries_are_independent_and_lm_defect_converges() -> None:
    problem = _problem()
    before = problem.defects()
    result = solve_boundary_lm(problem, damping=1.0e-8, cg_iterations=32, max_backtracks=8)
    assert result.accepted
    assert result.defects_after < result.defects_before
    assert result.defects_after < 1.0e-6
    # The stored boundary is not silently replaced by direct substitution.
    assert not torch.allclose(problem.boundaries, result.boundaries)


def test_matrix_free_jvp_vjp_match_directional_derivative() -> None:
    problem = _problem()
    direction = torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=torch.float64)
    jvp = boundary_jvp(problem, direction)
    eps = 1.0e-6
    finite = (problem.defects(problem.boundaries + eps * direction) - problem.defects(problem.boundaries - eps * direction)) / (2 * eps)
    assert torch.allclose(jvp, finite, rtol=1.0e-6, atol=1.0e-8)
    cotangent = torch.ones_like(jvp)
    vjp = boundary_vjp(problem, cotangent)
    assert torch.allclose(vjp, torch.autograd.functional.vjp(lambda x: problem.defects(x), problem.boundaries, v=cotangent)[1])


def test_so3_residual_is_zero_at_same_rotation() -> None:
    rotation = torch.eye(3, dtype=torch.float64).expand(2, 3, 3).clone()
    residual = so3_local_residual(rotation, rotation)
    assert torch.allclose(residual, torch.zeros_like(residual))


def test_boundary_layout_uses_so3_residual_for_three_coordinate_orientation() -> None:
    layout = BoundaryLayout(5, rotation_slice=slice(1, 4))
    predicted = torch.tensor((2.0, 0.2, -0.1, 0.3, 4.0), dtype=torch.float64)
    actual = torch.tensor((1.0, -0.1, 0.05, 0.2, 3.0), dtype=torch.float64)
    residual = layout.residual(predicted, actual)
    expected_orientation = so3_local_residual(
        so3_exp(predicted[1:4]), so3_exp(actual[1:4])
    )
    torch.testing.assert_close(residual[0], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(residual[1:4], expected_orientation)
    torch.testing.assert_close(residual[4], torch.tensor(1.0, dtype=torch.float64))


def test_normalized_so3_boundary_applies_exp_in_physical_radians() -> None:
    layout = BoundaryLayout(3, rotation_slice=slice(0, 3), rotation_scale=0.1)
    predicted = torch.tensor([1.2, -0.3, 0.4], dtype=torch.float64)
    actual = torch.tensor([-0.2, 0.7, -0.5], dtype=torch.float64)
    expected = so3_local_residual(
        so3_exp(0.1 * predicted), so3_exp(0.1 * actual)
    ) / 0.1
    torch.testing.assert_close(layout.residual(predicted, actual), expected)


def test_so3_boundary_uses_three_tangent_variables_and_retracts_on_manifold() -> None:
    layout = BoundaryLayout(state_dim=3, rotation_slice=slice(0, 3))
    assert layout.residual_dim == 3
    base = torch.eye(3, dtype=torch.float64)
    tangent = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    rotation = so3_retract(base, tangent)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, torch.eye(3, dtype=torch.float64), atol=1.0e-12)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(() , dtype=torch.float64), atol=1.0e-12)


def test_trust_region_reports_action_and_parameter_rejections() -> None:
    diagnostic = trust_region_diagnostics(
        torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]),
        action_before=torch.zeros(2), action_after=torch.tensor([0.3, 0.4]),
        parameter_radius=1.1, action_radius=0.2,
    )
    assert diagnostic.within_parameter_radius
    assert not diagnostic.within_action_radius
    assert not diagnostic.accepted


def _joint_problem() -> FullSpaceProblem:
    initial = torch.tensor([1.0], dtype=torch.float64)
    theta = torch.tensor([0.2], dtype=torch.float64)

    def segment(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 0.7 * z + t

    def terminal_task(starts: torch.Tensor, ends: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # A terminal-only task still has policy credit through both segments.
        return ends[-1] - 2.0

    return FullSpaceProblem(
        initial, torch.zeros(2, 1, dtype=torch.float64), theta, segment,
        BoundaryLayout(1), task_residual=terminal_task,
    )


def test_joint_sqp_updates_theta_and_boundaries_with_linearized_equality() -> None:
    problem = _joint_problem()
    result = solve_joint_sqp_step(problem, linear_solver="legacy-cg", damping=1.0e-3, cg_iterations=64, max_backtracks=4)
    assert result.accepted
    assert not torch.allclose(result.theta, problem.theta)
    assert not torch.allclose(result.boundaries, problem.boundaries)
    assert result.defects_after < result.defects_before
    assert result.task_norm_after < result.task_norm_before
    assert result.linearized_constraint_residual < 1.0e-8
    assert result.kkt_stationarity_residual < 1.0e-1
    assert not result.linear_solver_breakdown
    assert result.linear_solver_iterations > 0


def test_joint_sqp_rejects_dangerous_action_step() -> None:
    problem = _joint_problem()

    def actions(theta: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        return theta.expand(4)

    result = solve_joint_sqp_step(
        problem, linear_solver="legacy-cg", damping=1.0e-3, cg_iterations=64, action_evaluator=actions,
        action_radius=1.0e-6, max_backtracks=4,
    )
    assert not result.accepted
    assert torch.allclose(result.theta, problem.theta)
    assert result.action_step_norm > 1.0e-6


def test_joint_sqp_backtracks_to_action_radius_and_recomputes_reduction() -> None:
    problem = _joint_problem()

    def actions(theta: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        return theta.expand(4)

    result = solve_joint_sqp_step(
        problem, linear_solver="legacy-cg", damping=1.0e-3, cg_iterations=64, action_evaluator=actions,
        action_radius=0.5, max_backtracks=8,
    )
    assert result.accepted
    assert result.backtracks > 0
    assert result.action_step_norm <= 0.5
    assert result.predicted_reduction > 0.0
    assert abs(result.ratio - 1.0) < 0.01


def test_joint_sqp_rejects_silent_scalar_objective() -> None:
    problem = _joint_problem()
    problem = FullSpaceProblem(
        problem.initial, problem.boundaries, problem.theta, problem.segment_map,
        problem.layout, objective=lambda nodes, theta: nodes[-1].square().sum(),
        task_residual=problem.task_residual,
    )
    with pytest.raises(ValueError, match="task_residual"):
        solve_joint_sqp_step(problem, linear_solver="legacy-cg")


def test_checked_cg_stops_on_negative_curvature_without_huge_step() -> None:
    rhs = torch.ones(3, dtype=torch.float64)
    result = _cg_solve(lambda value: -value, rhs, iterations=8, tolerance=1.0e-10)
    assert result.breakdown
    assert not result.converged
    assert result.iterations == 0
    torch.testing.assert_close(result.value, torch.zeros_like(rhs))
