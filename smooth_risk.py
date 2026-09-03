"""Differentiable tail-risk primitives for fixed-theta microbatch training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F


Tensor = torch.Tensor


def solve_smooth_cvar_eta(
    values: Tensor,
    *,
    alpha: float = 0.8,
    beta: float = 1.0e-2,
    iterations: int = 48,
) -> Tensor:
    """Solve the scalar smooth RU threshold by monotone bisection.

    At the optimum, ``mean(sigmoid((L-eta)/beta)) = 1-alpha``.  The returned
    eta is detached deliberately: the envelope theorem gives the objective
    derivative with respect to losses while avoiding a spurious quantile or
    root-solver differentiation branch.
    """

    if values.numel() == 0 or not 0.0 <= alpha < 1.0 or beta <= 0.0:
        raise ValueError("invalid values, alpha, or beta")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    flat = values.detach().reshape(-1)
    lower = flat.min() - 40.0 * float(beta)
    upper = flat.max() + 40.0 * float(beta)
    target = 1.0 - float(alpha)
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        tail_mass = torch.sigmoid((flat - midpoint) / float(beta)).mean()
        # tail_mass decreases monotonically with eta.
        lower = torch.where(tail_mass > target, midpoint, lower)
        upper = torch.where(tail_mass > target, upper, midpoint)
    return (0.5 * (lower + upper)).detach()


def smooth_positive(value: Tensor, *, beta: float = 1.0e-2, mode: str = "softplus") -> Tensor:
    """Smooth approximation of ``relu(value)`` with a continuous derivative."""

    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if mode == "softplus":
        return float(beta) * F.softplus(value / float(beta))
    if mode == "huber":
        positive = value.clamp_min(0.0)
        return torch.where(
            positive < float(beta),
            positive.square() / (2.0 * float(beta)),
            positive - 0.5 * float(beta),
        )
    raise ValueError("mode must be 'softplus' or 'huber'")


@dataclass(frozen=True)
class RiskSummary:
    value: Tensor
    eta: Tensor
    tail_mean: Tensor
    sample_count: int


def rockafellar_uryasev_cvar(
    values: Tensor,
    *,
    alpha: float = 0.8,
    eta: Tensor | float | None = None,
    beta: float = 1.0e-2,
    mode: str = "softplus",
    reduction: str = "mean",
    return_summary: bool = False,
) -> Tensor | RiskSummary:
    """Empirical Rockafellar--Uryasev CVaR surrogate.

    ``alpha`` is the confidence level (0.8 means the worst 20%).  Supplying
    ``eta`` is recommended when accumulating microbatches: the same threshold
    is then used for every microbatch and the result equals one call on their
    concatenation.  If omitted, the optimum of the smooth empirical objective
    is solved by detached monotone bisection.  By the envelope theorem this
    keeps the correct loss derivative without differentiating through the root
    solver.
    """

    if values.numel() == 0:
        raise ValueError("values must be non-empty")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0,1)")
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    flat = values.reshape(-1)
    threshold = (
        solve_smooth_cvar_eta(flat, alpha=alpha, beta=beta)
        if eta is None
        else torch.as_tensor(eta, dtype=flat.dtype, device=flat.device)
    )
    excess = smooth_positive(flat - threshold, beta=beta, mode=mode)
    tail_mean = excess.mean() if reduction == "mean" else excess.sum() / float(flat.numel())
    result = threshold + tail_mean / max(1.0e-12, 1.0 - float(alpha))
    if return_summary:
        return RiskSummary(result, threshold, tail_mean, int(flat.numel()))
    return result


# Short alias used by callers that do not need the historical name.
smooth_cvar = rockafellar_uryasev_cvar


class FixedThetaMicrobatchCVaR:
    """Accumulate RU CVaR terms over microbatches with one fixed threshold.

    This class intentionally does not retain computation graphs.  Each
    ``update`` may receive values from a different forward pass at the same
    theta; ``finalize`` is the exact mean-tail RU expression for the union of
    those samples, and its returned scalar remains differentiable through all
    supplied values if they remain live in the caller.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.8,
        eta: Tensor | float | None = None,
        beta: float = 1.0e-2,
        mode: str = "softplus",
    ) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0,1)")
        self.alpha = float(alpha)
        self.eta = eta
        self.beta = float(beta)
        self.mode = mode
        self._values: list[Tensor] = []
        self._sum = None
        self._count = 0

    def update(self, values: Tensor) -> None:
        if values.numel() == 0:
            raise ValueError("values must be non-empty")
        flat = values.reshape(-1)
        if self.eta is None:
            # Keep values only for a detached threshold fit at finalize; this
            # path is intended for convenience, not streaming-scale training.
            self._values.append(flat.detach())
        else:
            threshold = torch.as_tensor(self.eta, dtype=flat.dtype, device=flat.device)
            tail = smooth_positive(flat - threshold, beta=self.beta, mode=self.mode).sum()
            self._sum = tail if self._sum is None else self._sum + tail
        self._count += int(flat.numel())

    def finalize(self) -> RiskSummary:
        if self._count == 0:
            raise RuntimeError("cannot finalize an empty accumulator")
        if self.eta is None:
            all_values = torch.cat(self._values)
            threshold = solve_smooth_cvar_eta(
                all_values, alpha=self.alpha, beta=self.beta
            )
            # Convenience mode has no live graph by design.
            tail_sum = smooth_positive(all_values - threshold, beta=self.beta, mode=self.mode).sum()
        else:
            reference = self._sum
            assert reference is not None
            threshold = torch.as_tensor(self.eta, dtype=reference.dtype, device=reference.device)
            tail_sum = reference
        tail_mean = tail_sum / float(self._count)
        value = threshold + tail_mean / max(1.0e-12, 1.0 - self.alpha)
        return RiskSummary(value, threshold, tail_mean, self._count)


__all__ = [
    "FixedThetaMicrobatchCVaR",
    "RiskSummary",
    "rockafellar_uryasev_cvar",
    "smooth_cvar",
    "smooth_positive",
    "solve_smooth_cvar_eta",
]
