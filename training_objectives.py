from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


def accumulated_episode_objective(
    segment_loss: torch.Tensor,
    episode_event_loss: torch.Tensor,
    *,
    segments_per_episode: int,
    correct_episode_event_weighting: bool,
) -> torch.Tensor:
    """Return the per-segment backward objective before gradient averaging.

    The training loop averages accumulated gradients over the episode's
    segments.  Scaling event losses by N here makes the final gradient equal
    to ``mean(segment losses) + sum(event losses)`` rather than shrinking an
    H250/H500 event by ``1/N``.
    """
    if segments_per_episode < 1:
        raise ValueError("segments_per_episode must be positive")
    scale = float(segments_per_episode) if correct_episode_event_weighting else 1.0
    return segment_loss + scale * episode_event_loss


def time_normalized_segment_sum(
    value_sum: torch.Tensor,
    *,
    total_episode_count: int,
    segments_per_episode: int,
) -> torch.Tensor:
    """Scale one segment sum for a later mean-over-segment gradient reduction.

    The outer loop divides accumulated gradients by ``segments_per_episode``.
    Multiplication by the same value here makes all segment numerators add up
    to one exact episode-time mean, including a burn-in-shortened first segment.
    """

    if total_episode_count < 0 or segments_per_episode < 1:
        raise ValueError("episode count must be non-negative and segment count positive")
    if total_episode_count == 0:
        return value_sum * 0.0
    return value_sum * (float(segments_per_episode) / float(total_episode_count))


@dataclass(frozen=True)
class ThresholdTailResult:
    loss: torch.Tensor
    position_tail: torch.Tensor
    omega_tail: torch.Tensor
    per_sample_position: torch.Tensor
    per_sample_omega: torch.Tensor
    per_sample_margin: torch.Tensor
    selected_indices: torch.Tensor
    selected_mask: torch.Tensor

    @property
    def selected_fraction(self) -> float:
        return float(self.selected_mask.float().mean().detach().item())


@dataclass(frozen=True)
class IndependentTailResult:
    loss: torch.Tensor
    position_cvar: torch.Tensor
    omega_cvar: torch.Tensor
    per_sample_position: torch.Tensor
    per_sample_omega: torch.Tensor
    position_selected_indices: torch.Tensor
    omega_selected_indices: torch.Tensor
    position_selected_mask: torch.Tensor
    omega_selected_mask: torch.Tensor

    @property
    def position_selected_fraction(self) -> float:
        return float(self.position_selected_mask.float().mean().detach().item())

    @property
    def omega_selected_fraction(self) -> float:
        return float(self.omega_selected_mask.float().mean().detach().item())

    @property
    def selected_overlap_fraction(self) -> float:
        overlap = self.position_selected_mask & self.omega_selected_mask
        denominator = max(int(self.position_selected_mask.sum().item()), 1)
        return float(overlap.sum().detach().item()) / float(denominator)


@dataclass(frozen=True)
class OmegaDecayResult:
    loss: torch.Tensor
    horizon_losses: tuple[torch.Tensor, ...]
    mean_ratios: tuple[torch.Tensor, ...]
    active_fractions: tuple[float, ...]


def multistep_omega_decay_loss(
    omega_history: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    horizons: tuple[int, ...] = (5, 10, 25),
    beta: tuple[float, ...] = (0.2, 0.3, 0.5),
    rho: tuple[float, ...] = (0.90, 0.75, 0.50),
    success_omega: float = 0.20,
    eps: float = 1.0e-6,
) -> OmegaDecayResult:
    """Penalize insufficient future angular-rate contraction on selected samples.

    ``omega_history`` includes the state before the first action.  The starting
    norm is detached so the controller cannot reduce the ratio by increasing
    the denominator; gradients train only the future response.
    """

    if omega_history.ndim != 3 or omega_history.shape[-1] != 3:
        raise ValueError("omega_history must have shape [time,batch,3]")
    if sample_mask.shape != omega_history.shape[1:2] or sample_mask.dtype != torch.bool:
        raise ValueError("sample_mask must be boolean with shape [batch]")
    if not horizons or len(horizons) != len(beta) or len(horizons) != len(rho):
        raise ValueError("horizons, beta and rho must be non-empty and have equal length")
    if any(k <= 0 or k >= omega_history.shape[0] for k in horizons):
        raise ValueError("each decay horizon must be in [1, time-1]")
    if any(weight < 0.0 for weight in beta):
        raise ValueError("decay beta values must be non-negative")
    if any(target < 0.0 for target in rho):
        raise ValueError("decay rho values must be non-negative")
    if success_omega <= 0.0 or eps <= 0.0:
        raise ValueError("success_omega and eps must be positive")

    weighted_losses: list[torch.Tensor] = []
    horizon_losses: list[torch.Tensor] = []
    mean_ratios: list[torch.Tensor] = []
    active_fractions: list[float] = []
    for k, weight, target in zip(horizons, beta, rho):
        start_norm = torch.linalg.vector_norm(omega_history[:-k], dim=-1)
        future_norm = torch.linalg.vector_norm(omega_history[k:], dim=-1)
        active = sample_mask.unsqueeze(0) & (start_norm.detach() >= float(success_omega))
        ratio = future_norm / (start_norm.detach() + float(eps))
        penalty = F.relu(ratio - float(target)).square()
        if bool(active.any().item()):
            horizon_loss = penalty[active].mean()
            mean_ratio = ratio[active].mean()
        else:
            horizon_loss = future_norm.sum() * 0.0
            mean_ratio = ratio.detach().sum() * 0.0
        horizon_losses.append(horizon_loss)
        mean_ratios.append(mean_ratio)
        active_fractions.append(float(active.float().mean().detach().item()))
        weighted_losses.append(float(weight) * horizon_loss)
    loss = torch.stack(weighted_losses).sum()
    return OmegaDecayResult(
        loss=loss,
        horizon_losses=tuple(horizon_losses),
        mean_ratios=tuple(mean_ratios),
        active_fractions=tuple(active_fractions),
    )


def _tail_margins(
    position_history: torch.Tensor,
    omega_history: torch.Tensor,
    *,
    position_threshold: float,
    omega_threshold: float,
    window_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_history.ndim != 3 or position_history.shape[-1] != 3:
        raise ValueError("position_history must have shape [time, batch, 3]")
    if omega_history.shape != position_history.shape:
        raise ValueError("omega_history must match position_history")
    if position_history.shape[0] == 0 or position_history.shape[1] == 0:
        raise ValueError("tail histories must be non-empty")
    if window_steps <= 0 or window_steps > position_history.shape[0]:
        raise ValueError("window_steps must be in [1, recorded time]")
    if position_threshold <= 0.0 or omega_threshold <= 0.0:
        raise ValueError("tail thresholds must be positive")
    position = position_history[-window_steps:]
    omega = omega_history[-window_steps:]
    per_sample_position = F.relu(
        torch.linalg.vector_norm(position, dim=-1) / float(position_threshold) - 1.0
    ).square().mean(dim=0)
    per_sample_omega = F.relu(
        torch.linalg.vector_norm(omega, dim=-1) / float(omega_threshold) - 1.0
    ).square().mean(dim=0)
    return per_sample_position, per_sample_omega


def _topk_selection(
    values: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("CVaR fraction must be in (0, 1]")
    selected_count = max(1, int(math.ceil(float(fraction) * values.shape[0])))
    indices = torch.topk(values, k=selected_count, largest=True, sorted=False).indices
    mask = torch.zeros(values.shape[0], device=values.device, dtype=torch.bool)
    mask[indices] = True
    return indices, mask


def independent_cvar_tail_loss(
    position_history: torch.Tensor,
    omega_history: torch.Tensor,
    *,
    position_threshold: float = 0.05,
    omega_threshold: float = 0.20,
    q_position: float = 0.20,
    q_omega: float = 0.20,
    w_position_cvar: float = 1.0,
    w_omega_cvar: float = 1.0,
    window_steps: int = 100,
) -> IndependentTailResult:
    """Select position and angular-rate tails independently."""
    if w_position_cvar < 0.0 or w_omega_cvar < 0.0:
        raise ValueError("CVaR weights must be non-negative")
    per_sample_position, per_sample_omega = _tail_margins(
        position_history,
        omega_history,
        position_threshold=position_threshold,
        omega_threshold=omega_threshold,
        window_steps=window_steps,
    )
    position_indices, position_mask = _topk_selection(per_sample_position, q_position)
    omega_indices, omega_mask = _topk_selection(per_sample_omega, q_omega)
    position_cvar = per_sample_position[position_indices].mean()
    omega_cvar = per_sample_omega[omega_indices].mean()
    loss = float(w_position_cvar) * position_cvar + float(w_omega_cvar) * omega_cvar
    return IndependentTailResult(
        loss=loss,
        position_cvar=position_cvar,
        omega_cvar=omega_cvar,
        per_sample_position=per_sample_position,
        per_sample_omega=per_sample_omega,
        position_selected_indices=position_indices,
        omega_selected_indices=omega_indices,
        position_selected_mask=position_mask,
        omega_selected_mask=omega_mask,
    )


def threshold_cvar_tail_loss(
    position_history: torch.Tensor,
    omega_history: torch.Tensor,
    *,
    position_threshold: float = 0.05,
    omega_threshold: float = 0.20,
    lambda_tail_omega: float = 1.0,
    cvar_fraction: float = 0.20,
    window_steps: int = 100,
) -> ThresholdTailResult:
    """Threshold-aligned position/angular-rate CVaR over an episode tail.

    Histories have shape ``[time, batch, 3]``.  Top-k selection is performed
    with tensor indices, so gradients from selected margins remain connected to
    the simulator trajectory.
    """
    if lambda_tail_omega < 0.0:
        raise ValueError("lambda_tail_omega must be non-negative")
    per_sample_position, per_sample_omega = _tail_margins(
        position_history,
        omega_history,
        position_threshold=position_threshold,
        omega_threshold=omega_threshold,
        window_steps=window_steps,
    )
    per_sample_margin = per_sample_position + float(lambda_tail_omega) * per_sample_omega
    selected_indices, selected_mask = _topk_selection(per_sample_margin, cvar_fraction)
    position_tail = per_sample_position[selected_indices].mean()
    omega_tail = per_sample_omega[selected_indices].mean()
    loss = position_tail + float(lambda_tail_omega) * omega_tail
    return ThresholdTailResult(
        loss=loss,
        position_tail=position_tail,
        omega_tail=omega_tail,
        per_sample_position=per_sample_position,
        per_sample_omega=per_sample_omega,
        per_sample_margin=per_sample_margin,
        selected_indices=selected_indices,
        selected_mask=selected_mask,
    )


def retain_action_mse(
    candidate_action: torch.Tensor,
    baseline_action: torch.Tensor,
    retain_mask: torch.Tensor,
) -> torch.Tensor:
    if candidate_action.shape != baseline_action.shape:
        raise ValueError("candidate and baseline actions must have identical shapes")
    if candidate_action.ndim != 2 or candidate_action.shape[-1] != 4:
        raise ValueError("actions must have shape [batch, 4]")
    if retain_mask.shape != candidate_action.shape[:1] or retain_mask.dtype != torch.bool:
        raise ValueError("retain_mask must be a boolean [batch] tensor")
    if not bool(retain_mask.any().item()):
        return candidate_action.sum() * 0.0
    target = baseline_action.detach()
    return (candidate_action[retain_mask] - target[retain_mask]).square().mean()
