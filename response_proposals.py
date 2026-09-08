"""Bounded TRAIN rollout search in a short-window gradient subspace."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class SubspaceConfig:
    parameter_relative_step: float = 1.e-4

    def __post_init__(self):
        if not math.isfinite(self.parameter_relative_step) or self.parameter_relative_step <= 0:
            raise ValueError("candidate parameter radius must be finite and positive")


@dataclass(frozen=True)
class CandidateSearchResult:
    parameters: torch.Tensor | None
    metrics: dict | None
    reason: str | None
    evidence: dict


def gradient_basis(gradients: torch.Tensor, *, rank_tolerance: float = 1.e-6) -> torch.Tensor:
    """Reorthogonalize the at-most-five gradient rows, independent of their units."""
    if gradients.ndim != 2 or not bool(torch.isfinite(gradients).all()):
        raise FloatingPointError("nonfinite or malformed objective gradients")
    if not 1 <= gradients.shape[0] <= 5:
        raise ValueError("candidate search supports at most five objective gradients")
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
def search_candidates(policy, gradients, evaluate_train, config: SubspaceConfig, *, baseline,
                      accept_candidate) -> CandidateSearchResult:
    """Test +/- each basis vector at rho and rho/4; return the best safe TRAIN point.

    All trials start from the same parameters and RNG. DEV is unavailable here.
    Restore the original Actor even on success: the caller explicitly applies
    the selected parameters once before its independent DEV check.
    """
    from response_training import capture_rng, restore_rng

    base = torch.nn.utils.parameters_to_vector(policy.parameters()).detach().clone()
    if not bool(torch.isfinite(base).all()):
        raise FloatingPointError("nonfinite baseline Actor")
    basis = gradient_basis(gradients)
    if basis.shape[0] != base.numel():
        raise ValueError("objective gradients do not match the Actor")
    evidence = {"basis_rank": basis.shape[1], "candidate_rollouts": 0, "candidates": [],
                "objective_gradient_norms": gradients.norm(dim=1, dtype=torch.float64).tolist()}
    if not basis.shape[1]:
        return CandidateSearchResult(None, None, "empty_subspace", evidence)
    radius = config.parameter_relative_step * max(float(base.norm(dtype=torch.float64)), 1.)
    evidence.update(parameter_step=radius, step_fractions=[1., .25])
    best_parameters, best_metrics, best_loss = None, None, float(baseline["task_objective"])
    if not math.isfinite(best_loss):
        raise FloatingPointError("nonfinite baseline performance")
    rng = capture_rng()
    try:
        for index, direction in enumerate(basis.T):
            for fraction in (1., .25):
                for sign in (1, -1):
                    parameters = base + (sign * fraction * radius) * direction
                    trial = {"basis_index": index, "sign": sign, "fraction": fraction}
                    if torch.equal(parameters, base):
                        trial["rejection_reason"] = "unchanged_parameters"
                        evidence["candidates"].append(trial)
                        continue
                    if not bool(torch.isfinite(parameters).all()):
                        trial["rejection_reason"] = "candidate_nonfinite"
                        evidence["candidates"].append(trial)
                        continue
                    assign_parameters(policy, parameters)
                    restore_rng(rng)
                    evidence["candidate_rollouts"] += 1
                    try:
                        metrics = evaluate_train()
                        reason = accept_candidate(metrics)
                    except FloatingPointError:
                        metrics, reason = {"finite": False}, "candidate_nonfinite"
                    trial["rejection_reason"] = reason
                    if metrics.get("finite", False):
                        trial.update(performance=metrics["task_objective"],
                                     hard_risk_components=metrics["hard_risk_components"],
                                     bounds_violated=metrics["hard_risk_bounds_violated"])
                    evidence["candidates"].append(trial)
                    if reason is None and float(metrics["task_objective"]) < best_loss:
                        best_parameters, best_metrics = parameters.clone(), metrics
                        best_loss = float(metrics["task_objective"])
                        evidence["selected_candidate"] = len(evidence["candidates"]) - 1
        return CandidateSearchResult(best_parameters, best_metrics,
            None if best_parameters is not None else "no_acceptable_candidate", evidence)
    finally:
        assign_parameters(policy, base)
        restore_rng(rng)
