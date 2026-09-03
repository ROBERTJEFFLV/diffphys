from __future__ import annotations

import torch


def dense_tracking_time_weights(
    *,
    episode_steps: int,
    segment_steps: int,
    tail_steps: int,
    lambda_tail: float,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return exact per-step tracking-potential weights after segment averaging.

    This models the current training loop's ``mean(segment tracking)`` plus the
    last-``tail_steps`` mean in every segment.  It deliberately excludes CLF,
    outward, smoothness and episode CVaR terms because those are different
    mathematical quantities.
    """

    if episode_steps <= 0 or segment_steps <= 0 or tail_steps <= 0:
        raise ValueError("episode, segment and tail lengths must be positive")
    if episode_steps % segment_steps != 0:
        raise ValueError("episode_steps must be divisible by segment_steps")
    if tail_steps > segment_steps:
        raise ValueError("tail_steps cannot exceed segment_steps")
    if lambda_tail < 0.0:
        raise ValueError("lambda_tail must be non-negative")
    segment_count = episode_steps // segment_steps
    weights = torch.full(
        (episode_steps,),
        1.0 / float(segment_count * segment_steps),
        dtype=dtype,
    )
    tail_extra = float(lambda_tail) / float(segment_count * tail_steps)
    for segment in range(segment_count):
        end = (segment + 1) * segment_steps
        weights[end - tail_steps : end] += tail_extra
    return weights
