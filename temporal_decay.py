from __future__ import annotations

import math


CURRENT_DECAY_MODE = "current"
NMI_DECAY_MODE = "nmi"
GRADIENT_DECAY_MODES = (CURRENT_DECAY_MODE, NMI_DECAY_MODE)


def current_decay_equivalent_alpha(base: float) -> float:
    """Return alpha such that ``base**dt == exp(-alpha*dt)``."""

    if not 0.0 < base <= 1.0:
        raise ValueError("current gradient-decay base must be in (0, 1]")
    return -math.log(float(base))


def resolve_step_gradient_decay(
    *,
    mode: str,
    dt: float,
    current_base: float,
    alpha: float | None,
) -> float:
    """Resolve the backward-only multiplier applied at every simulator step.

    ``current`` exactly preserves the repository's historical behavior,
    ``current_base**dt``.  ``nmi`` exposes the equivalent continuous-rate form
    ``exp(-alpha*dt)`` directly.
    """

    if dt <= 0.0:
        raise ValueError("gradient-decay dt must be positive")
    if mode == CURRENT_DECAY_MODE:
        if alpha is not None:
            raise ValueError("gradient-decay alpha is valid only in nmi mode")
        if not 0.0 < current_base <= 1.0:
            raise ValueError("current gradient-decay base must be in (0, 1]")
        return float(current_base) ** float(dt)
    if mode == NMI_DECAY_MODE:
        if alpha is None:
            raise ValueError("nmi gradient decay requires alpha")
        if alpha < 0.0:
            raise ValueError("gradient-decay alpha must be non-negative")
        return math.exp(-float(alpha) * float(dt))
    raise ValueError(f"unsupported gradient-decay mode: {mode}")
