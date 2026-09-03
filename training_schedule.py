from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


FIXED_H500 = "fixed-h500"
MIXED_HORIZON_CURRICULUM = "mixed-h500-h1000-h2000"
COMPRESSED_T2_10PCT = "compressed-t2-10pct"

MIXED_70_30_PATTERN = (500, 500, 1000, 500, 500, 1000, 500, 500, 1000, 500)
# Exact 10% replay of the reset-horizon counts observed in T2@96M: scale the
# 400-episode H500 warmup to 40 episodes, then take the first 27 episodes of
# the deterministic 70/30 phase.  This gives 59xH500 + 8xH1000 = 75 H500
# equivalent blocks without introducing an H2000 episode.
COMPRESSED_T2_10PCT_SEQUENCE = (
    (500,) * 40
    + MIXED_70_30_PATTERN * 2
    + MIXED_70_30_PATTERN[:7]
)


@dataclass(frozen=True)
class EpisodeHorizonDecision:
    horizon_steps: int
    nominal_horizon_steps: int
    phase: str


def _next_boundary(
    completed_physical_steps: int,
    physical_step_budget: int,
    checkpoint_physical_steps: Iterable[int],
    *,
    include_phase_boundaries: bool,
) -> int:
    phase_boundaries = (
        (
            physical_step_budget * 20 // 100,
            physical_step_budget * 50 // 100,
            physical_step_budget,
        )
        if include_phase_boundaries
        else (physical_step_budget,)
    )
    candidates = [
        value
        for value in (*phase_boundaries, *checkpoint_physical_steps)
        if value > completed_physical_steps
    ]
    if not candidates:
        return physical_step_budget
    return min(candidates)


def select_episode_horizon(
    *,
    completed_physical_steps: int,
    physical_step_budget: int,
    checkpoint_physical_steps: Iterable[int],
    batch_size: int,
    segment_steps: int,
    episode_index: int,
    mode: str,
) -> EpisodeHorizonDecision:
    """Select a deterministic episode horizon without consuming an RNG stream.

    Curriculum percentages describe episode choices.  Phase transitions are
    driven by accumulated simulator work, and the selected horizon is shortened
    only when needed to land exactly on a phase/checkpoint/budget boundary.
    """

    if completed_physical_steps < 0 or physical_step_budget <= 0:
        raise ValueError("physical-step progress and budget must be non-negative/positive")
    if completed_physical_steps >= physical_step_budget:
        raise ValueError("cannot select an episode after the physical-step budget")
    if batch_size <= 0 or segment_steps <= 0 or episode_index < 0:
        raise ValueError("batch size/segment steps must be positive and episode index non-negative")
    if mode not in {FIXED_H500, MIXED_HORIZON_CURRICULUM, COMPRESSED_T2_10PCT}:
        raise ValueError(f"unsupported episode-horizon schedule: {mode}")

    if mode == COMPRESSED_T2_10PCT:
        expected_budget = batch_size * sum(COMPRESSED_T2_10PCT_SEQUENCE)
        if physical_step_budget != expected_budget:
            raise ValueError(
                "compressed-t2-10pct requires a physical-step budget equal to "
                f"batch_size * {sum(COMPRESSED_T2_10PCT_SEQUENCE)} "
                f"({expected_budget} for batch_size={batch_size}), got "
                f"{physical_step_budget}"
            )
        if episode_index >= len(COMPRESSED_T2_10PCT_SEQUENCE):
            raise ValueError("compressed-t2-10pct reset sequence is exhausted")

    fraction = completed_physical_steps / float(physical_step_budget)
    if fraction < 0.20:
        phase = "warmup_h500"
        pattern = (500,)
    elif fraction < 0.50:
        phase = "mixed_70_30"
        pattern = MIXED_70_30_PATTERN
    else:
        phase = "mixed_50_30_20"
        pattern = (500, 1000, 500, 500, 2000, 1000, 500, 500, 1000, 2000)

    if mode == FIXED_H500:
        nominal = 500
        phase = "fixed_h500"
    elif mode == COMPRESSED_T2_10PCT:
        nominal = COMPRESSED_T2_10PCT_SEQUENCE[episode_index]
        phase = "compressed_t2_10pct"
    else:
        nominal = pattern[episode_index % len(pattern)]

    allowed = tuple(value for value in (500, 1000, 2000) if value % segment_steps == 0)
    if 500 not in allowed:
        raise ValueError("segment length must divide the minimum H500 episode")

    next_boundary = _next_boundary(
        completed_physical_steps,
        physical_step_budget,
        checkpoint_physical_steps,
        include_phase_boundaries=mode == MIXED_HORIZON_CURRICULUM,
    )
    remaining_physical = next_boundary - completed_physical_steps
    if remaining_physical % batch_size != 0:
        raise ValueError("physical-step boundaries must be divisible by batch size")
    remaining_horizon = remaining_physical // batch_size
    feasible = [value for value in allowed if value <= remaining_horizon]
    if not feasible:
        raise ValueError(
            "next physical-step boundary is closer than one complete H500 episode; "
            "choose aligned phase/checkpoint/budget values"
        )
    if mode == COMPRESSED_T2_10PCT and nominal > remaining_horizon:
        raise ValueError(
            "a phase/checkpoint/budget boundary would truncate the preregistered "
            f"compressed reset episode H{nominal} to H{max(feasible)}"
        )
    selected = nominal if nominal <= remaining_horizon else max(feasible)
    return EpisodeHorizonDecision(
        horizon_steps=selected,
        nominal_horizon_steps=nominal,
        phase=phase,
    )


def normalized_episode_value(
    segment_values: Iterable[float],
    episode_event_value: float,
) -> float:
    """Reference scalar form: mean segment density plus full episode event."""

    values = tuple(float(value) for value in segment_values)
    if not values:
        raise ValueError("at least one segment value is required")
    return sum(values) / len(values) + float(episode_event_value)
