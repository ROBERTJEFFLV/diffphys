"""TRAIN-only, low-dimensional correction of short-window Actor directions.

The proposed span comes from physics/Critic gradients. Real continuous forward
flights determine the direction inside that span; no Actor optimizer transforms
it afterward. Finite differences are diagnostics, not a safety certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import torch


@dataclass(frozen=True)
class SubspaceConfig:
    fd_relative_step: float = 1.e-4
    parameter_relative_step: float = 1.e-4
    fd_relative_tolerance: float = .1
    fd_absolute_tolerance: float = 0.
    backtracks: int = 4

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in (
            self.fd_relative_step, self.parameter_relative_step, self.fd_relative_tolerance
        )) or self.backtracks < 1:
            raise ValueError("subspace radii/tolerance and bounded backtrack count must be positive")
        if not math.isfinite(self.fd_absolute_tolerance) or self.fd_absolute_tolerance < 0:
            raise ValueError("FD absolute tolerance must be finite and nonnegative")


@dataclass(frozen=True)
class CorrectedDirection:
    direction: torch.Tensor | None
    reason: str | None
    evidence: dict


def gradient_basis(gradients: torch.Tensor, *, rank_tolerance: float = 1.e-6) -> torch.Tensor:
    """Reorthogonalize the at-most-five gradient rows, independent of their units."""
    if gradients.ndim != 2 or not bool(torch.isfinite(gradients).all()):
        raise FloatingPointError("nonfinite or malformed objective gradients")
    columns = []
    for row in gradients.detach():
        norm = row.norm(dtype=torch.float64)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite objective gradient norm")
        if float(norm) == 0:
            continue
        candidate = row / norm
        for _ in range(2):
            for column in columns:
                candidate = candidate - torch.dot(candidate, column) * column
        length = candidate.norm()
        if float(length) > rank_tolerance:
            columns.append(candidate / length)
    return torch.stack(columns, 1) if columns else gradients.new_empty((gradients.shape[1], 0))


def projected_performance_direction(matrix: torch.Tensor, *, flat_tolerance: float = 1.e-10,
                                    feasibility_tolerance: float = 1.e-10) -> torch.Tensor | None:
    """Project -performance onto the cone risk @ q <= 0; boundary solutions count.

    Enumerate active subsets of at most five risk constraints. Their projection
    satisfies q = y - R_active.T @ multipliers, with nonnegative multipliers.
    Row normalization preserves the risk halfspaces. Normalize y for the small
    solve, then return the projection in the original performance-gradient units.
    """
    if matrix.ndim != 2 or not 1 <= matrix.shape[0] <= 6 or not 1 <= matrix.shape[1] <= 5:
        raise ValueError("projection needs one to six objective rows and at most five directions")
    if not bool(torch.isfinite(matrix).all()):
        raise FloatingPointError("nonfinite measured direction matrix")
    rows = matrix.detach().to(device="cpu", dtype=torch.float64)
    perf_norm = rows[0].norm()
    if not bool(torch.isfinite(perf_norm)):
        raise FloatingPointError("nonfinite performance-gradient norm")
    if float(perf_norm) <= flat_tolerance:
        return None
    y = -rows[0] / perf_norm
    risks = rows[1:]
    norms = risks.norm(dim=1)
    risks = risks[norms > 0] / norms[norms > 0, None]
    best, distance = torch.zeros_like(y), float(y.square().sum())
    for count in range(risks.shape[0] + 1):
        for indices in combinations(range(risks.shape[0]), count):
            if count:
                active = risks[list(indices)]
                multipliers = torch.linalg.pinv(active @ active.T) @ (active @ y)
                if bool((multipliers < -feasibility_tolerance).any()):
                    continue
                candidate = y - active.T @ multipliers
            else:
                candidate = y
            if bool((risks @ candidate > feasibility_tolerance).any()):
                continue
            error = float((candidate - y).square().sum())
            if error < distance:
                best, distance = candidate, error
    if float(best.norm()) <= flat_tolerance or float(y @ best) <= 0:
        return None
    return (best * perf_norm).to(matrix)


def check_direction_at_scales(full: torch.Tensor, half: torch.Tensor, q: torch.Tensor,
                              config: SubspaceConfig) -> tuple[bool, dict]:
    """Check the actual unit direction at both scales; near-zero risk is unknown.

    Directional thresholds use the same normalized slope units as row checks.
    A small sign change inside atol + rtol * max(|d_h|, |d_half|) is allowed to
    reach the real trajectory gate. Performance must be clearly negative twice.
    """
    slopes = torch.stack((full @ q, half @ q))
    if not bool(torch.isfinite(slopes).all()):
        raise FloatingPointError("nonfinite corrected directional slopes")
    tolerance = config.fd_absolute_tolerance * q.norm() + config.fd_relative_tolerance * slopes.abs().amax(0)
    perf_ok = bool((slopes[:, 0] < -tolerance[0]).all())
    increasing = (slopes[:, 1:] > tolerance[1:]).any(0)
    decreasing = (slopes[:, 1:] < -tolerance[1:]).all(0)
    labels = ["increasing" if bool(up) else "decreasing" if bool(down) else "unknown_or_near_zero"
              for up, down in zip(increasing, decreasing)]
    return perf_ok and not bool(increasing.any()), {
        "directional_slopes_full_step": slopes[0].tolist(),
        "directional_slopes_half_step": slopes[1].tolist(),
        "directional_slope_tolerances": tolerance.tolist(),
        "direction_performance_clear_descent": perf_ok,
        "direction_risk_classification": labels,
    }


@torch.no_grad()
def assign_parameters(policy, vector: torch.Tensor) -> None:
    """Copy a checked vector without replacing Parameter objects or their storage."""
    if not bool(torch.isfinite(vector).all()):
        raise FloatingPointError("nonfinite proposed parameter vector")
    parameters = list(policy.parameters())
    if vector.ndim != 1 or vector.numel() != sum(p.numel() for p in parameters):
        raise ValueError("parameter vector does not match the Actor")
    offset = 0
    for parameter in parameters:
        parameter.copy_(vector[offset:offset + parameter.numel()].view_as(parameter))
        offset += parameter.numel()


@torch.no_grad()
def correct_direction(policy, gradients, evaluate_train, config: SubspaceConfig, *, baseline=None):
    """Measure +/- perturbations from identical Actor, initial states and RNG.

    The callback must return the real TRAIN performance first, then risk
    components and optionally total-risk CVaR. It has no DEV inputs. Restore
    parameters and RNG on every exit, including a failed forward evaluation.
    """
    from response_training import capture_rng, restore_rng

    base = torch.nn.utils.parameters_to_vector(policy.parameters()).detach().clone()
    basis = gradient_basis(gradients)
    evidence = {"basis_rank": basis.shape[1], "probe_rollouts": 0,
                "objective_gradient_norms": gradients.norm(dim=1, dtype=torch.float64).tolist()}
    if not basis.shape[1]:
        return CorrectedDirection(None, "no_feasible_direction", evidence)
    rng = capture_rng()
    scale = max(float(base.norm(dtype=torch.float64)), 1.)
    epsilon = config.fd_relative_step * scale
    evidence.update(fd_parameter_step=epsilon, parameter_step=config.parameter_relative_step * scale)

    def measure():
        value = evaluate_train().detach().to(device="cpu", dtype=torch.float64)
        if value.ndim != 1 or not bool(torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite true TRAIN objective in direction probe")
        return value

    try:
        baseline = measure() if baseline is None else baseline.detach().to(device="cpu", dtype=torch.float64)
        if baseline.ndim != 1 or not bool(torch.isfinite(baseline).all()):
            raise FloatingPointError("nonfinite direction baseline")
        units = baseline.abs().clamp_min(1.e-6)
        matrices = []
        for radius in (epsilon, epsilon / 2):
            columns = []
            for direction in basis.T:
                pair = []
                for sign in (1, -1):
                    proposed = base + sign * radius * direction
                    # Detect a perturbation lost or distorted by parameter dtype.
                    realized = (proposed - base) / (sign * radius)
                    if float((realized - direction).norm()) > .05:
                        evidence["fd_failure"] = "parameter_roundoff"
                        return CorrectedDirection(None, "fd_unreliable", evidence)
                    assign_parameters(policy, proposed)
                    restore_rng(rng)
                    pair.append(measure())
                    evidence["probe_rollouts"] += 1
                columns.append((pair[0] - pair[1]) / (2 * radius) / units)
            matrices.append(torch.stack(columns, 1))
        measured = matrices[-1]
        evidence["measured_objective_slopes"] = measured.tolist()
        evidence["objective_units"] = units.tolist()
        evidence["flat_objective_rows"] = (measured.norm(dim=1) <= 1.e-10).nonzero().flatten().tolist()
        if not all(bool(torch.isfinite(matrix).all()) for matrix in matrices):
            raise FloatingPointError("nonfinite finite-difference slopes")
        diff = (matrices[0] - measured).norm(dim=1)
        size = torch.maximum(matrices[0].norm(dim=1), measured.norm(dim=1))
        tolerance = config.fd_absolute_tolerance + config.fd_relative_tolerance * size
        evidence["fd_absolute_error_by_objective"] = diff.tolist()
        evidence["fd_tolerance_by_objective"] = tolerance.tolist()
        evidence["fd_absolute_tolerance"] = config.fd_absolute_tolerance
        evidence["fd_relative_tolerance"] = config.fd_relative_tolerance
        if bool((diff > tolerance).any()):
            evidence["fd_failure"] = "row_disagreement"
            return CorrectedDirection(None, "fd_unreliable", evidence)
        q = projected_performance_direction(measured)
        if q is None:
            return CorrectedDirection(None, "no_feasible_direction", evidence)
        q = q / q.norm()
        reliable, directional = check_direction_at_scales(matrices[0], measured, q, config)
        evidence.update(directional)
        if not reliable:
            evidence["fd_failure"] = "direction_disagreement"
            return CorrectedDirection(None, "fd_unreliable", evidence)
        evidence["corrected_objective_slopes"] = (measured @ q).tolist()
        direction = basis @ q.to(basis)
        return CorrectedDirection(direction / direction.norm(), None, evidence)
    finally:
        assign_parameters(policy, base)
        restore_rng(rng)
