"""Small differentiable box-QP allocator for four-motor policies.

The active set is selected discretely, but the solution on the selected face
is an ordinary differentiable linear solve.  This is the standard piecewise
smooth semantics of a strictly convex box-constrained quadratic program.  The
implementation enumerates only ``3**4 == 81`` faces, which is inexpensive for
the fixed four-motor L2F plant and avoids hiding constraint violations behind
a post-hoc clamp.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Optional

import torch
from torch import nn


Tensor = torch.Tensor
ACTION_DIM = 4


@dataclass(frozen=True)
class BoxQPAllocatorDiagnostics:
    condition_number: Tensor
    wrench_residual: Tensor
    saturation: Tensor
    rate_limited: Tensor
    lower_headroom: Tensor
    upper_headroom: Tensor
    headroom_violation: Tensor
    minimum_headroom: Tensor
    trust_limited: Tensor
    primal_violation: Tensor
    kkt_residual: Tensor
    active_lower: Tensor
    active_upper: Tensor
    objective: Tensor


def _as_batch_column(
    value: Tensor | float,
    reference: Tensor,
    *,
    name: str,
    nonnegative: bool = True,
) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        result = result.expand(reference.shape[0], 1)
    elif result.ndim == 1:
        result = result[:, None]
    if result.shape != (reference.shape[0], 1):
        raise ValueError(f"{name} must be scalar or shape [batch]")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    if nonnegative and bool((result < 0.0).any()):
        raise ValueError(f"{name} must be non-negative")
    return result


class ActiveSetBoxQPAllocator(nn.Module):
    """Solve a strictly convex four-variable allocation QP exactly.

    For each batch row the layer solves

    ``min_u .5 ||W (B u - w)||^2 + .5*damping*||u-trim||^2``

    subject to the motor box, optional action-delta trust box and optional
    command-rate box.  Bounds are part of the solve; no output clamp is needed.
    """

    def __init__(self, damping: float = 1.0e-3,
                 wrench_weights: tuple[float, float, float, float] = (1, 1, 1, 1),
                 feasibility_tolerance: float = 1.0e-6) -> None:
        super().__init__()
        if damping <= 0.0 or feasibility_tolerance <= 0.0:
            raise ValueError("damping and feasibility_tolerance must be positive")
        weights = torch.tensor(wrench_weights, dtype=torch.float32)
        if weights.shape != (ACTION_DIM,) or bool((weights <= 0.0).any()):
            raise ValueError("wrench_weights must contain four positive values")
        self.register_buffer("wrench_weights", weights)
        self.damping = float(damping)
        self.feasibility_tolerance = float(feasibility_tolerance)
        patterns = tuple(itertools.product((-1, 0, 1), repeat=ACTION_DIM))
        self.register_buffer("active_patterns", torch.tensor(patterns, dtype=torch.int8))

    def _bounds(
        self,
        reference: Tensor,
        *,
        trim: Tensor,
        previous_action: Optional[Tensor],
        dt: float,
        action_delta_cap: Tensor | float | None,
        rate_limit: Tensor | float | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        lower = torch.full_like(reference, -1.0)
        upper = torch.full_like(reference, 1.0)
        # Keep each optional candidate around through the complete
        # intersection.  A constraint is a source of a final bound only when
        # its candidate is equal to that final bound; merely having supplied
        # the constraint is not enough (the physical box may be tighter).
        trust_lower = torch.full_like(reference, -torch.inf)
        trust_upper = torch.full_like(reference, torch.inf)
        rate_lower = torch.full_like(reference, -torch.inf)
        rate_upper = torch.full_like(reference, torch.inf)
        if action_delta_cap is not None:
            cap = _as_batch_column(action_delta_cap, reference, name="action_delta_cap")
            if bool((cap <= 0.0).any()):
                raise ValueError("action_delta_cap must be positive")
            trust_lower, trust_upper = trim - cap, trim + cap
            lower, upper = torch.maximum(lower, trust_lower), torch.minimum(upper, trust_upper)
        if rate_limit is not None:
            rate = _as_batch_column(rate_limit, reference, name="rate_limit")
            if previous_action is None and bool((rate > 0.0).any()):
                raise ValueError("positive rate_limit requires previous_action")
            if previous_action is not None:
                radius = rate * float(dt)
                has_rate = rate > 0.0
                candidate_lower = previous_action - radius
                candidate_upper = previous_action + radius
                rate_lower = torch.where(has_rate, candidate_lower, rate_lower)
                rate_upper = torch.where(has_rate, candidate_upper, rate_upper)
                lower = torch.where(has_rate, torch.maximum(lower, candidate_lower), lower)
                upper = torch.where(has_rate, torch.minimum(upper, candidate_upper), upper)
        if bool((lower > upper).any()):
            raise RuntimeError("allocator constraints have an empty intersection")
        trust_lower_source = torch.isfinite(trust_lower) & (trust_lower == lower)
        trust_upper_source = torch.isfinite(trust_upper) & (trust_upper == upper)
        rate_lower_source = torch.isfinite(rate_lower) & (rate_lower == lower)
        rate_upper_source = torch.isfinite(rate_upper) & (rate_upper == upper)
        return (
            lower,
            upper,
            trust_lower_source,
            trust_upper_source,
            rate_lower_source,
            rate_upper_source,
        )

    def forward(
        self,
        desired_wrench: Tensor,
        previous_action: Optional[Tensor] = None,
        *,
        dt: float = 0.01,
        mixer: Tensor,
        trim: Optional[Tensor] = None,
        action_delta_cap: Tensor | float | None = None,
        rate_limit: Tensor | float | None = None,
    ) -> tuple[Tensor, BoxQPAllocatorDiagnostics]:
        if desired_wrench.ndim != 2 or desired_wrench.shape[-1] != ACTION_DIM:
            raise ValueError("desired_wrench must have shape [batch,4]")
        batch = desired_wrench.shape[0]
        if mixer.shape != (batch, ACTION_DIM, ACTION_DIM):
            raise ValueError("mixer must have shape [batch,4,4]")
        if previous_action is not None and previous_action.shape != desired_wrench.shape:
            raise ValueError("previous_action must match desired_wrench")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if trim is None:
            trim = torch.zeros_like(desired_wrench)
        if trim.shape != desired_wrench.shape:
            raise ValueError("trim must match desired_wrench")

        (
            lower,
            upper,
            trust_lower_source,
            trust_upper_source,
            rate_lower_source,
            rate_upper_source,
        ) = self._bounds(
            desired_wrench, trim=trim, previous_action=previous_action, dt=dt,
            action_delta_cap=action_delta_cap, rate_limit=rate_limit,
        )
        weights = self.wrench_weights.to(desired_wrench)
        weighted_mixer = weights[None, :, None] * mixer
        weighted_wrench = weights[None, :] * desired_wrench
        identity = torch.eye(ACTION_DIM, device=mixer.device, dtype=mixer.dtype)
        hessian = weighted_mixer.transpose(1, 2) @ weighted_mixer
        hessian = hessian + self.damping * identity.unsqueeze(0)
        rhs = (
            weighted_mixer.transpose(1, 2) @ weighted_wrench.unsqueeze(-1)
        ).squeeze(-1) + self.damping * trim

        tolerance = self.feasibility_tolerance
        # Evaluate the same 81 faces in one batched solve. Active coordinates
        # get identity rows; the free block is exactly the original H_ff.
        patterns = self.active_patterns.to(device=mixer.device)
        free = (patterns == 0).to(mixer.dtype)[None]
        active = 1.0 - free
        fixed = torch.where(patterns[None] < 0, lower[:, None], upper[:, None]) * active
        face_hessian = (hessian[:, None] * free[..., :, None] * free[..., None, :]
                        + torch.diag_embed(active))
        face_rhs = free * (rhs[:, None] - (hessian[:, None] @ fixed[..., None]).squeeze(-1)) + fixed
        candidate_stack = torch.linalg.solve(face_hessian, face_rhs[..., None]).squeeze(-1)
        feasible = ((candidate_stack >= lower[:, None] - tolerance)
                    & (candidate_stack <= upper[:, None] + tolerance)
                    & torch.isfinite(candidate_stack)).all(-1)
        wrench_error = ((mixer[:, None] @ candidate_stack[..., None]).squeeze(-1)
                        - desired_wrench[:, None]) * weights
        objective_stack = (0.5 * wrench_error.square().sum(-1)
                           + 0.5 * self.damping * (candidate_stack - trim[:, None]).square().sum(-1))
        objective_stack = torch.where(feasible, objective_stack, torch.full_like(objective_stack, torch.inf))
        selected = objective_stack.argmin(dim=1)
        action = candidate_stack.gather(
            1, selected[:, None, None].expand(batch, 1, ACTION_DIM)
        ).squeeze(1)
        objective = objective_stack.gather(1, selected[:, None]).squeeze(1)
        if not bool(torch.isfinite(objective).all()):
            raise RuntimeError("box-QP allocator found no finite feasible face")

        wrench_residual_vector = (
            torch.bmm(mixer, action.unsqueeze(-1)).squeeze(-1) - desired_wrench
        )
        gradient = (
            torch.bmm(hessian, action.unsqueeze(-1)).squeeze(-1) - rhs
        )
        # Projected-gradient residual is zero exactly at the box-QP KKT point.
        projected = torch.maximum(lower, torch.minimum(upper, action - gradient))
        kkt_residual = torch.linalg.vector_norm(action - projected, dim=-1)
        primal_violation = torch.maximum(
            torch.relu(lower - action), torch.relu(action - upper)
        ).amax(dim=-1)
        active_lower = (action - lower).abs() <= 10.0 * tolerance
        active_upper = (action - upper).abs() <= 10.0 * tolerance
        box_lower_headroom = action + 1.0
        box_upper_headroom = 1.0 - action
        minimum_headroom = torch.minimum(action - lower, upper - action).amin(dim=-1)
        trust_limited = (
            ((active_lower & trust_lower_source) | (active_upper & trust_upper_source)).any(dim=-1)
        ).to(action.dtype)
        rate_limited = (
            ((active_lower & rate_lower_source) | (active_upper & rate_upper_source)).any(dim=-1)
        ).to(action.dtype)
        diagnostics = BoxQPAllocatorDiagnostics(
            condition_number=torch.linalg.cond(weighted_mixer).to(action),
            wrench_residual=torch.linalg.vector_norm(wrench_residual_vector, dim=-1),
            saturation=action.abs().mean(dim=-1),
            rate_limited=rate_limited,
            lower_headroom=box_lower_headroom.amin(dim=-1),
            upper_headroom=box_upper_headroom.amin(dim=-1),
            headroom_violation=primal_violation,
            minimum_headroom=minimum_headroom,
            trust_limited=trust_limited,
            primal_violation=primal_violation,
            kkt_residual=kkt_residual,
            active_lower=active_lower.sum(dim=-1),
            active_upper=active_upper.sum(dim=-1),
            objective=objective,
        )
        return action, diagnostics


__all__ = ["ActiveSetBoxQPAllocator", "BoxQPAllocatorDiagnostics"]
