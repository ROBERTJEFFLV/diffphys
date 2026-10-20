"""Per-scene PHR constraints and Gaussian-smoothed HARD-program gradients.

Adapted from the independently tested event_waypoint_rebuild/gradient_tools.py.
No straight-through switch, critic, learned dynamics, or extra external force.
"""
from __future__ import annotations

import math
import torch


def phr(time_cost, constraints, multipliers, rho):
    """All residuals are g<=0, normalized by physical task tolerances.

    Apply PHR per scene and per constraint BEFORE the mean across scenes.
    Inactive inequalities can have a negative constant AL contribution.
    """
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive")
    if (constraints.ndim != 2 or multipliers.shape != constraints.shape
            or time_cost.shape != constraints.shape[:1]):
        raise ValueError("expected time[N], constraints[N,C], multipliers[N,C]")
    if multipliers.requires_grad or bool((multipliers < 0).any()):
        raise ValueError("multipliers must be detached and nonnegative")
    if not bool(torch.isfinite(constraints).all() & torch.isfinite(multipliers).all()):
        raise FloatingPointError("nonfinite AL input")
    return time_cost + (((multipliers + rho*constraints).clamp_min(0).square()
                        - multipliers.square())/(2*rho)).sum(-1)


@torch.no_grad()
def update_duals(constraints, multipliers, rho):
    if constraints.shape != multipliers.shape:
        raise ValueError("dual shape mismatch")
    result = (multipliers + rho*constraints.detach()).clamp_min(0)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("nonfinite multiplier update")
    return result


@torch.no_grad()
def corrected_gradient(theta, center_gradient, evaluate, *, sigma, directions, seed):
    """Control-variate estimate of grad E[J(theta+sigma*z)], z~N(0,I).

    evaluate returns one scalar from the SAME hard-switch program/initial pool,
    noise tapes and frozen duals. g_p is fixed independently of every direction.
    Do not normalize z. This routine does not clip or rescale its output.
    The caller must restore mutated parameters in a finally block.
    """
    if theta.ndim != 1 or center_gradient.shape != theta.shape:
        raise ValueError("parameter and gradient vectors must match")
    if theta.device != center_gradient.device or theta.dtype != center_gradient.dtype:
        raise ValueError("gradient vector dtype/device mismatch")
    if not math.isfinite(sigma) or sigma <= 0 or type(directions) is not int or directions < 1:
        raise ValueError("positive sigma and direction count required")
    generator = torch.Generator(device=theta.device).manual_seed(int(seed))
    total = theta.new_zeros(theta.shape)
    squared = theta.new_zeros(theta.shape)
    contrasts = []
    for _ in range(directions):
        z = torch.randn(theta.shape, dtype=theta.dtype, device=theta.device, generator=generator)
        plus, minus = evaluate(theta + sigma*z), evaluate(theta - sigma*z)
        if plus.numel() != 1 or minus.numel() != 1 or not bool(torch.isfinite(plus+minus)):
            raise FloatingPointError("paired hard rollouts must return finite scalars")
        slope = (plus-minus)/(2*sigma)
        residual = slope - torch.dot(center_gradient, z)
        sample = residual*z
        total += sample
        squared += sample.square()
        contrasts.append(float(slope))
    result = center_gradient + total/directions
    variance = ((squared-total.square()/directions)/(directions-1)).clamp_min(0) if directions > 1 else None
    return result, {
        "quantity": "Gaussian-smoothed hard-event augmented Lagrangian gradient",
        "sigma": sigma, "directions": directions,
        "extra_forward_rollouts": 2*directions,
        "paired_slopes": contrasts,
        "standard_error_l2": None if variance is None else float((variance.sum()/directions).sqrt()),
        "raw_gradient_l2": float(result.norm()),
        "center_branch_gradient_l2": float(center_gradient.norm()),
        "correction_l2": float((result-center_gradient).norm()),
    }
