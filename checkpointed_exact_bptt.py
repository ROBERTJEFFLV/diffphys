"""Canonical name for the legacy segmented exact-BPTT diagnostic.

The implementation remains in :mod:`strict_multiple_shooting` for checkpoint
and script compatibility.  It passes every segment end directly into the next
segment, so it is not the full-space solver used by new structured training.
"""

from strict_multiple_shooting import (  # noqa: F401
    RecurrentSystemState,
    StrictObjectiveResult,
    StrictShootingResult,
    StrictShootingConfig,
    recurrent_tensors,
    strict_multiple_shooting_objective,
    rollout_strict_multiple_shooting,
)


__all__ = [
    "RecurrentSystemState",
    "StrictObjectiveResult",
    "StrictShootingResult",
    "StrictShootingConfig",
    "recurrent_tensors",
    "strict_multiple_shooting_objective",
    "rollout_strict_multiple_shooting",
]
