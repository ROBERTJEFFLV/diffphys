"""Matrix-free algebraic MVP for full-space multiple shooting.

Unlike :mod:`checkpointed_exact_bptt`, this module keeps every segment endpoint
as an independent decision variable and computes a joint policy/state KKT
direction.  The implementation deliberately remains labelled an MVP: nested
CG is not yet block-preconditioned and the trust step is a backtracked damped
Gauss--Newton ray rather than a production sparse SQP package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


Tensor = torch.Tensor
SegmentMap = Callable[[Tensor, Tensor], Tensor]
Objective = Callable[[Tensor, Tensor], Tensor]
TaskResidual = Callable[[Tensor, Tensor, Tensor], Tensor]


def so3_local_residual(predicted: Tensor, actual: Tensor) -> Tensor:
    """Return the local rotation residual ``log(predicted.T @ actual)``.

    The small-angle vector formula is stable around the identity and avoids
    treating the nine matrix entries as independent Euclidean coordinates.
    """

    relative = predicted.transpose(-1, -2) @ actual
    vector = 0.5 * torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) - 1.0)
    angle = torch.atan2(sine, cosine.clamp(-1.0, 1.0))
    # angle*vector/clamp(sine) has ZERO derivative at identity.  The log
    # derivative there must be the identity on the tangent space.
    scale = torch.where(cosine > 0.9999,
                        1.0 + sine.square() / 6.0 + 3.0 * sine.pow(4) / 40.0,
                        angle / sine.clamp_min(1.0e-12))
    result = scale * vector
    # At pi the skew part vanishes: recover the axis from R + I.  Choosing
    # the longest column avoids division by a small axis component.  The
    # logarithm's cut is inherently non-smooth; no smoothness claim is made.
    identity = torch.eye(3, device=relative.device, dtype=relative.dtype)
    symmetric = relative + relative.transpose(-1, -2) + 2.0 * identity
    column = symmetric.diagonal(dim1=-2, dim2=-1).argmax(-1)
    axis = symmetric.gather(-1, column[..., None, None].expand(*column.shape, 3, 1)).squeeze(-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = torch.where((axis * vector).sum(-1, keepdim=True) < 0, -axis, axis)
    return torch.where(cosine < -0.9999, angle * axis, result)


def so3_exp(tangent: Tensor) -> Tensor:
    """Exponential map from a 3-vector tangent coordinate to SO(3)."""

    if tangent.shape[-1] != 3:
        raise ValueError("SO(3) tangent must have final dimension 3")
    angle = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    x = angle.clamp_min(1.0e-12)
    a = torch.where(angle < 1.0e-4, 1.0 - angle.square() / 6.0, torch.sin(angle) / x)
    b = torch.where(
        angle < 1.0e-4,
        0.5 - angle.square() / 24.0,
        (1.0 - torch.cos(angle)) / x.square(),
    )
    wx = torch.zeros(*tangent.shape[:-1], 3, 3, dtype=tangent.dtype, device=tangent.device)
    wx[..., 0, 1] = -tangent[..., 2]
    wx[..., 0, 2] = tangent[..., 1]
    wx[..., 1, 0] = tangent[..., 2]
    wx[..., 1, 2] = -tangent[..., 0]
    wx[..., 2, 0] = -tangent[..., 1]
    wx[..., 2, 1] = tangent[..., 0]
    identity = torch.eye(3, dtype=tangent.dtype, device=tangent.device).expand_as(wx)
    return identity + a.unsqueeze(-1) * wx + b.unsqueeze(-1) * (wx @ wx)


def so3_retract(base_rotation: Tensor, tangent: Tensor) -> Tensor:
    """Retract a 3D local boundary coordinate while preserving SO(3)."""

    if base_rotation.shape[-2:] != (3, 3):
        raise ValueError("base_rotation must end in [3,3]")
    return base_rotation @ so3_exp(tangent)


@dataclass(frozen=True)
class BoundaryLayout:
    """Flattened boundary layout.

    ``rotation_slice`` identifies three local tangent entries for an SO(3)
    orientation.  The actual 3x3 rotation is reconstructed by
    :func:`so3_retract`; storing nine Euclidean entries is intentionally not
    supported because it introduces six null directions and leaves SO(3).
    """

    state_dim: int
    rotation_slice: slice | None = None
    rotation_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.state_dim < 1:
            raise ValueError("state_dim must be positive")
        if self.rotation_slice is not None and self.rotation_slice.stop - self.rotation_slice.start != 3:
            raise ValueError("rotation_slice must contain exactly three SO(3) tangent entries")
        if self.rotation_scale <= 0.0:
            raise ValueError("rotation_scale must be positive")

    @property
    def residual_dim(self) -> int:
        return self.state_dim

    def residual(self, predicted: Tensor, actual: Tensor) -> Tensor:
        if predicted.shape != actual.shape or predicted.shape[-1] != self.state_dim:
            raise ValueError("boundary tensors do not match BoundaryLayout")
        if self.rotation_slice is None:
            return predicted - actual
        sl = self.rotation_slice
        before = predicted[..., :sl.start] - actual[..., :sl.start]
        after = predicted[..., sl.stop:] - actual[..., sl.stop:]
        # Boundary vectors may be normalized.  Apply Exp in physical radians,
        # then map the group residual back to normalized tangent coordinates.
        predicted_rotation = so3_exp(predicted[..., sl] * self.rotation_scale)
        actual_rotation = so3_exp(actual[..., sl] * self.rotation_scale)
        orientation = (
            so3_local_residual(predicted_rotation, actual_rotation)
            / self.rotation_scale
        )
        return torch.cat((before, orientation, after), dim=-1)


@dataclass(frozen=True)
class TrustRegionDiagnostics:
    parameter_norm: float
    action_norm: float
    parameter_radius: float
    action_radius: float
    within_parameter_radius: bool
    within_action_radius: bool
    accepted: bool


@dataclass(frozen=True)
class FullSpaceStep:
    boundaries: Tensor
    defects_before: float
    defects_after: float
    merit_before: float
    merit_after: float
    predicted_reduction: float
    actual_reduction: float
    ratio: float
    damping: float
    step_norm: float
    accepted: bool
    backtracks: int


@dataclass(frozen=True)
class JointFullSpaceStep:
    theta: Tensor
    boundaries: Tensor
    defects_before: float
    defects_after: float
    task_norm_before: float
    task_norm_after: float
    merit_before: float
    merit_after: float
    predicted_reduction: float
    actual_reduction: float
    ratio: float
    damping: float
    linearized_constraint_residual: float
    linearized_constraint_relative: float
    kkt_stationarity_residual: float
    kkt_stationarity_relative: float
    parameter_step_norm: float
    action_step_norm: float
    action_step_max: float
    linear_solver_converged: bool
    linear_solver_breakdown: bool
    linear_solver_iterations: int
    linear_solver_residual_max: float
    accepted: bool
    backtracks: int


@dataclass(frozen=True)
class LinearSolveResult:
    value: Tensor
    iterations: int
    residual_norm: float
    converged: bool
    breakdown: bool


@dataclass(frozen=True)
class FullSpaceProblem:
    """Independent-boundary deterministic shooting problem.

    ``boundaries`` has shape ``[segments, ..., state_dim]``: one independent
    endpoint node per segment, including the terminal endpoint.  The initial
    node is fixed separately in ``initial``.  Batch and other leading
    dimensions are allowed; the solver treats the final axis as the state axis.
    """

    initial: Tensor
    boundaries: Tensor
    theta: Tensor
    segment_map: SegmentMap
    layout: BoundaryLayout
    objective: Objective | None = None
    task_residual: TaskResidual | None = None
    fixed_boundary_mask: Tensor | None = None

    def _all_boundaries(self, boundaries: Tensor) -> Tensor:
        if boundaries.ndim < 2 or boundaries.shape[-1] != self.layout.state_dim:
            raise ValueError("boundaries must end in BoundaryLayout.state_dim")
        if self.fixed_boundary_mask is not None:
            boundaries = torch.where(self.fixed_boundary_mask, self.boundaries.detach(), boundaries)
        return torch.cat((self.initial.unsqueeze(0), boundaries), dim=0)

    def segment_ends(self, boundaries: Tensor | None = None, theta: Tensor | None = None) -> Tensor:
        """Return each segment's predicted endpoint before continuity comparison."""

        b = self.boundaries if boundaries is None else boundaries
        t = self.theta if theta is None else theta
        all_b = self._all_boundaries(b)
        return torch.stack(
            tuple(self.segment_map(all_b[index], t) for index in range(all_b.shape[0] - 1)),
            dim=0,
        )

    def defects(self, boundaries: Tensor | None = None, theta: Tensor | None = None) -> Tensor:
        b = self.boundaries if boundaries is None else boundaries
        t = self.theta if theta is None else theta
        all_b = self._all_boundaries(b)
        predicted_ends = self.segment_ends(b, t)
        residuals = [
            self.layout.residual(predicted_ends[index], all_b[index + 1])
            for index in range(all_b.shape[0] - 1)
        ]
        return torch.stack(residuals, dim=0)

    def value(self, boundaries: Tensor | None = None, theta: Tensor | None = None) -> tuple[Tensor, Tensor]:
        b = self.boundaries if boundaries is None else boundaries
        t = self.theta if theta is None else theta
        defect = self.defects(b, t)
        objective = torch.zeros((), dtype=defect.dtype, device=defect.device)
        if self.objective is not None:
            objective = self.objective(self._all_boundaries(b), t)
        return objective, defect

    def task_values(self, boundaries: Tensor | None = None, theta: Tensor | None = None) -> Tensor:
        """Return a vector task residual, kept separate from scalar merit."""

        b = self.boundaries if boundaries is None else boundaries
        t = self.theta if theta is None else theta
        all_b = self._all_boundaries(b)
        if self.task_residual is None:
            return torch.zeros(0, dtype=all_b.dtype, device=all_b.device)
        values = self.task_residual(all_b[:-1], self.segment_ends(b, t), t)
        if values.ndim == 0:
            values = values.reshape(1)
        return values.reshape(-1)

    def merit(self, boundaries: Tensor, theta: Tensor, penalty: float) -> tuple[Tensor, Tensor, Tensor]:
        objective = torch.zeros((), dtype=boundaries.dtype, device=boundaries.device)
        if self.objective is not None:
            objective = self.objective(self._all_boundaries(boundaries), theta)
        defects = self.defects(boundaries, theta)
        task = self.task_values(boundaries, theta)
        return objective + 0.5 * float(penalty) * defects.square().sum() + 0.5 * task.square().sum(), defects, task


def boundary_jvp(problem: FullSpaceProblem, direction: Tensor, boundaries: Tensor | None = None) -> Tensor:
    """Matrix-free Jacobian-vector product for the defect map."""

    b = problem.boundaries if boundaries is None else boundaries
    if direction.shape != b.shape:
        raise ValueError("boundary direction has the wrong shape")
    _, result = torch.autograd.functional.jvp(
        lambda x: problem.defects(x), (b,), (direction,), create_graph=True, strict=False
    )
    return result


def boundary_vjp(problem: FullSpaceProblem, cotangent: Tensor, boundaries: Tensor | None = None) -> Tensor:
    """Matrix-free vector-Jacobian product for the defect map."""

    b = (problem.boundaries if boundaries is None else boundaries).detach().requires_grad_(True)
    residual = problem.defects(b)
    if residual.shape != cotangent.shape:
        raise ValueError("defect cotangent has the wrong shape")
    return torch.autograd.grad((residual * cotangent).sum(), b, allow_unused=False)[0]


def _joint_residual(problem: FullSpaceProblem, theta: Tensor, boundaries: Tensor, kind: str) -> Tensor:
    if kind == "constraint":
        return problem.defects(boundaries, theta)
    if kind == "task":
        return problem.task_values(boundaries, theta)
    raise ValueError("kind must be 'constraint' or 'task'")


def joint_jvp(
    problem: FullSpaceProblem,
    theta_direction: Tensor,
    boundary_direction: Tensor,
    *,
    kind: str = "constraint",
) -> Tensor:
    """Matrix-free JVP with respect to both policy and independent boundaries."""

    theta, boundaries = problem.theta, problem.boundaries
    if theta_direction.shape != theta.shape or boundary_direction.shape != boundaries.shape:
        raise ValueError("joint direction has the wrong shape")
    _, result = torch.autograd.functional.jvp(
        lambda t, b: _joint_residual(problem, t, b, kind),
        (theta, boundaries),
        (theta_direction, boundary_direction),
        create_graph=True,
        strict=False,
    )
    return result


def joint_vjp(
    problem: FullSpaceProblem,
    cotangent: Tensor,
    *,
    kind: str = "constraint",
) -> tuple[Tensor, Tensor]:
    """Matrix-free VJP returning policy and boundary cotangents."""

    theta = problem.theta.detach().requires_grad_(True)
    boundaries = problem.boundaries.detach().requires_grad_(True)
    residual = _joint_residual(problem, theta, boundaries, kind)
    if residual.shape != cotangent.shape:
        raise ValueError("joint cotangent has the wrong shape")
    gradients = torch.autograd.grad((residual * cotangent).sum(), (theta, boundaries), allow_unused=True)
    theta_gradient = torch.zeros_like(theta) if gradients[0] is None else gradients[0]
    boundary_gradient = torch.zeros_like(boundaries) if gradients[1] is None else gradients[1]
    return theta_gradient, boundary_gradient


def _pack_joint(theta: Tensor, boundaries: Tensor) -> Tensor:
    return torch.cat((theta.reshape(-1), boundaries.reshape(-1)))


def _unpack_joint(problem: FullSpaceProblem, vector: Tensor) -> tuple[Tensor, Tensor]:
    theta_count = problem.theta.numel()
    return vector[:theta_count].reshape_as(problem.theta), vector[theta_count:].reshape_as(problem.boundaries)


def _joint_vjp_flat(problem: FullSpaceProblem, cotangent: Tensor, *, kind: str) -> Tensor:
    theta, boundaries = joint_vjp(problem, cotangent, kind=kind)
    return _pack_joint(theta, boundaries)


def _joint_jvp_flat(problem: FullSpaceProblem, direction: Tensor, *, kind: str) -> Tensor:
    theta, boundaries = _unpack_joint(problem, direction)
    return joint_jvp(problem, theta, boundaries, kind=kind)


def _cg_solve(
    operator: Callable[[Tensor], Tensor],
    rhs: Tensor,
    *,
    iterations: int,
    tolerance: float,
) -> LinearSolveResult:
    """Conjugate gradient with explicit non-finite/curvature failure detection.

    All callers add positive LM damping, so a non-positive ``p.T A p`` is a
    numerical/modeling failure.  Clamping that value to a tiny positive number
    creates an enormous, apparently valid step; returning the last finite
    iterate lets the outer trust-region gate reject it cleanly instead.
    """

    x = torch.zeros_like(rhs)
    r = rhs - operator(x)
    if not bool(torch.isfinite(r).all()):
        return LinearSolveResult(x, 0, float("inf"), False, True)
    p = r.clone()
    rr = (r * r).sum()
    completed = 0
    converged = float(torch.sqrt(rr).detach()) <= tolerance
    breakdown = False
    for iteration in range(max(1, iterations)):
        if converged:
            break
        ap = operator(p)
        denominator = (p * ap).sum()
        scale = torch.linalg.vector_norm(p) * torch.linalg.vector_norm(ap)
        curvature_floor = 1.0e-14 * scale.clamp_min(1.0)
        if (
            not bool(torch.isfinite(ap).all())
            or not bool(torch.isfinite(denominator))
            or float(denominator.detach()) <= float(curvature_floor.detach())
        ):
            breakdown = True
            break
        alpha = rr / denominator
        trial_x = x + alpha * p
        trial_r = r - alpha * ap
        if not bool(torch.isfinite(trial_x).all() and torch.isfinite(trial_r).all()):
            breakdown = True
            break
        x, r = trial_x, trial_r
        completed = iteration + 1
        new_rr = (r * r).sum()
        converged = float(torch.sqrt(new_rr).detach()) <= tolerance
        p = r + (new_rr / rr.clamp_min(1.0e-20)) * p
        rr = new_rr
    return LinearSolveResult(
        value=x,
        iterations=completed,
        residual_norm=float(torch.sqrt(rr).detach()),
        converged=bool(converged),
        breakdown=bool(breakdown),
    )


def _cg(operator: Callable[[Tensor], Tensor], rhs: Tensor, *, iterations: int, tolerance: float) -> Tensor:
    """Compatibility wrapper returning only the checked CG iterate."""

    return _cg_solve(
        operator, rhs, iterations=iterations, tolerance=tolerance
    ).value


def _merit(problem: FullSpaceProblem, boundaries: Tensor, penalty: float) -> tuple[Tensor, Tensor, Tensor]:
    """Compatibility merit for the boundary-only solver."""

    objective, defects = problem.value(boundaries)
    task = problem.task_values(boundaries)
    merit = objective + 0.5 * float(penalty) * defects.square().sum() + 0.5 * task.square().sum()
    return merit, objective, defects


def solve_boundary_lm(
    problem: FullSpaceProblem,
    *,
    damping: float = 1.0e-3,
    penalty: float = 1.0,
    cg_iterations: int = 64,
    cg_tolerance: float = 1.0e-8,
    trust_radius: float = float("inf"),
    max_backtracks: int = 12,
    sufficient_decrease: float = 1.0e-4,
) -> FullSpaceStep:
    """Take one damped Gauss--Newton boundary step without forming a KKT matrix."""

    boundaries = problem.boundaries.detach().requires_grad_(True)
    merit, objective, defects = _merit(problem, boundaries, penalty)
    # J^T r is obtained by a single reverse product.  JVP/VJP remain matrix-free.
    task = problem.task_values(boundaries)
    grad_terms = 0.5 * float(penalty) * defects.square().sum() + 0.5 * task.square().sum()
    if problem.objective is not None:
        grad_terms = grad_terms + objective
    grad = torch.autograd.grad(grad_terms, boundaries)[0]

    def normal(direction: Tensor) -> Tensor:
        jv = boundary_jvp(problem, direction, boundaries)
        return boundary_vjp(problem, jv, boundaries).detach() + float(damping) * direction

    step = _cg(normal, -grad.detach(), iterations=cg_iterations, tolerance=cg_tolerance)
    step_norm = float(torch.linalg.vector_norm(step).detach())
    if step_norm > trust_radius:
        step = step * (float(trust_radius) / (step_norm + 1.0e-12))
        step_norm = float(torch.linalg.vector_norm(step).detach())

    linearized = defects.detach() + boundary_jvp(problem, step, boundaries).detach()
    predicted = 0.5 * float(penalty) * (defects.detach().square().sum() - linearized.square().sum())
    predicted = predicted - 0.5 * float(damping) * step.square().sum()
    predicted_value = float(predicted)

    accepted = False
    backtracks = 0
    candidate = boundaries.detach()
    candidate_merit = merit.detach()
    candidate_defects = defects.detach()
    for backtracks in range(max_backtracks + 1):
        scale = 0.5**backtracks
        trial = boundaries.detach() + scale * step
        trial_merit, _, trial_defects = _merit(problem, trial, penalty)
        actual = float((merit.detach() - trial_merit.detach()))
        # Merit decrease is the primary filter; a defect decrease also passes
        # when the user supplied objective is flat or absent.
        defect_ok = float(trial_defects.square().sum()) < float(defects.detach().square().sum())
        merit_ok = actual > 0.0 and (predicted_value <= 0.0 or actual >= sufficient_decrease * predicted_value * scale)
        if merit_ok or (defect_ok and problem.objective is None):
            candidate, candidate_merit, candidate_defects = trial, trial_merit.detach(), trial_defects.detach()
            accepted = True
            break
    actual_value = float(merit.detach() - candidate_merit)
    ratio = actual_value / predicted_value if predicted_value > 0.0 else 0.0
    return FullSpaceStep(
        boundaries=candidate,
        defects_before=float(torch.linalg.vector_norm(defects.detach())),
        defects_after=float(torch.linalg.vector_norm(candidate_defects)),
        merit_before=float(merit.detach()),
        merit_after=float(candidate_merit),
        predicted_reduction=predicted_value,
        actual_reduction=actual_value,
        ratio=ratio,
        damping=float(damping),
        step_norm=float(torch.linalg.vector_norm(candidate - boundaries.detach())),
        accepted=accepted,
        backtracks=backtracks,
    )


def trust_region_diagnostics(
    theta_before: Tensor,
    theta_after: Tensor,
    *,
    action_before: Tensor | None = None,
    action_after: Tensor | None = None,
    parameter_radius: float = float("inf"),
    action_radius: float = float("inf"),
    accepted: bool = True,
) -> TrustRegionDiagnostics:
    """Report parameter/action trust-region usage; does not mutate either input."""

    parameter_norm = float(torch.linalg.vector_norm(theta_after - theta_before).detach())
    if action_before is None or action_after is None:
        action_norm = 0.0
    else:
        if action_before.shape != action_after.shape:
            raise ValueError("action tensors must have equal shapes")
        action_norm = float(torch.sqrt((action_after - action_before).square().mean()).detach())
    p_ok = parameter_norm <= float(parameter_radius)
    a_ok = action_norm <= float(action_radius)
    return TrustRegionDiagnostics(
        parameter_norm=parameter_norm,
        action_norm=action_norm,
        parameter_radius=float(parameter_radius),
        action_radius=float(action_radius),
        within_parameter_radius=p_ok,
        within_action_radius=a_ok,
        accepted=bool(accepted and p_ok and a_ok),
    )


def solve_joint_sqp_step(
    problem: FullSpaceProblem,
    *,
    damping: float = 1.0e-3,
    penalty: float = 1.0,
    cg_iterations: int = 64,
    cg_tolerance: float = 1.0e-8,
    parameter_radius: float = float("inf"),
    parameter_scale: Tensor | None = None,
    action_radius: float = float("inf"),
    action_max_radius: float = float("inf"),
    action_evaluator: Callable[[Tensor, Tensor], Tensor] | None = None,
    max_backtracks: int = 12,
    sufficient_decrease: float = 1.0e-4,
    minimum_ratio: float = 0.0,
) -> JointFullSpaceStep:
    """Take a joint equality-constrained Gauss--Newton/LM SQP step.

    The linearized subproblem is ``min ||r + J_r d||`` subject to
    ``J_c d = -c``.  A Schur complement is solved with nested matrix-free CG:
    no Jacobian, normal matrix, or KKT matrix is materialized.
    """

    if problem.objective is not None:
        raise ValueError(
            "joint SQP requires objective=None; put differentiable terms in task_residual"
        )
    theta = problem.theta.detach().requires_grad_(True)
    boundaries = problem.boundaries.detach().requires_grad_(True)
    local = FullSpaceProblem(
        problem.initial, boundaries, theta, problem.segment_map, problem.layout,
        problem.objective, problem.task_residual, problem.fixed_boundary_mask,
    )
    merit, defects, task = local.merit(boundaries, theta, penalty)
    c = defects.detach()
    r = task.detach()
    total_size = theta.numel() + boundaries.numel()
    linear_solves: list[LinearSolveResult] = []

    def solve(operator: Callable[[Tensor], Tensor], rhs: Tensor) -> Tensor:
        result = _cg_solve(
            operator, rhs, iterations=cg_iterations, tolerance=cg_tolerance
        )
        linear_solves.append(result)
        return result.value

    def hessian(direction: Tensor) -> Tensor:
        jv = _joint_jvp_flat(local, direction, kind="task")
        return _joint_vjp_flat(local, jv, kind="task").detach() + float(damping) * direction

    task_grad = _joint_vjp_flat(local, r, kind="task").detach()
    # First solve H y = -J_r^T r.
    y = solve(hessian, -task_grad)

    def schur(cotangent: Tensor) -> Tensor:
        lifted = _joint_vjp_flat(local, cotangent.reshape_as(c), kind="constraint")
        solved = solve(hessian, lifted)
        return _joint_jvp_flat(local, solved, kind="constraint").detach().reshape(-1)

    rhs = -c.reshape(-1) - _joint_jvp_flat(local, y, kind="constraint").detach().reshape(-1)
    # Constraint residual is flattened to make CG independent of batch layout.
    multipliers = solve(schur, rhs)
    correction_rhs = _joint_vjp_flat(local, multipliers.reshape_as(c), kind="constraint")
    correction = solve(hessian, correction_rhs)
    direction = y + correction
    dtheta, dboundary = _unpack_joint(local, direction)
    dtheta = dtheta.detach()
    dboundary = dboundary.detach()

    if parameter_scale is not None and parameter_scale.shape != theta.shape:
        raise ValueError("parameter_scale must match theta")

    def scaled_parameter_norm(value: Tensor) -> float:
        if parameter_scale is None:
            return float(torch.linalg.vector_norm(value))
        return float(torch.sqrt((value / parameter_scale.clamp_min(1.0e-12)).square().mean()))

    parameter_norm = scaled_parameter_norm(dtheta)
    action_before = action_evaluator(theta.detach(), boundaries.detach()) if action_evaluator is not None else None
    action_before = None if action_before is None else action_before.detach()

    linear_constraint = c + joint_jvp(local, dtheta, dboundary, kind="constraint").detach()
    linear_task = r + joint_jvp(local, dtheta, dboundary, kind="task").detach()
    predicted = 0.5 * (r.square().sum() - linear_task.square().sum()) - 0.5 * float(damping) * direction.square().sum()
    predicted = predicted + 0.5 * float(penalty) * (c.square().sum() - linear_constraint.square().sum())
    predicted_value = float(predicted)

    candidate_theta = theta.detach()
    candidate_boundaries = boundaries.detach()
    candidate_merit = merit.detach()
    candidate_defects = defects.detach()
    candidate_task = task.detach()
    accepted = False
    backtracks = 0
    accepted_predicted = 0.0
    attempted_action_norm = 0.0
    attempted_action_max = 0.0
    for backtracks in range(max_backtracks + 1):
        scale = 0.5**backtracks
        trial_theta = theta.detach() + scale * dtheta
        trial_boundaries = boundaries.detach() + scale * dboundary
        trial_parameter_norm = scaled_parameter_norm(scale * dtheta)
        if trial_parameter_norm > parameter_radius:
            continue
        if action_evaluator is not None:
            trial_action = action_evaluator(trial_theta, trial_boundaries)
            action_delta = float(torch.sqrt((trial_action - action_before).square().mean()))
            action_max = float((trial_action - action_before).abs().max())
            attempted_action_norm = max(attempted_action_norm, action_delta)
            attempted_action_max = max(attempted_action_max, action_max)
            if action_delta > action_radius or action_max > action_max_radius:
                continue
        trial_merit, trial_defects, trial_task = problem.merit(trial_boundaries, trial_theta, penalty)
        actual = float(merit.detach() - trial_merit.detach())
        trial_linear_task = r + scale * (linear_task - r)
        trial_linear_constraint = c + scale * (linear_constraint - c)
        trial_predicted = 0.5 * (r.square().sum() - trial_linear_task.square().sum())
        trial_predicted = trial_predicted + 0.5 * float(penalty) * (c.square().sum() - trial_linear_constraint.square().sum())
        trial_predicted = trial_predicted - 0.5 * float(damping) * (scale * direction).square().sum()
        trial_predicted_value = float(trial_predicted)
        defect_ok = float(trial_defects.square().sum()) < float(defects.detach().square().sum())
        task_ok = float(trial_task.square().sum()) <= float(task.detach().square().sum())
        trial_ratio = actual / trial_predicted_value if trial_predicted_value > 0.0 else 0.0
        merit_ok = (
            actual > 0.0
            and (trial_predicted_value <= 0.0 or actual >= sufficient_decrease * trial_predicted_value)
            and (trial_predicted_value <= 0.0 or trial_ratio >= float(minimum_ratio))
        )
        # Filter rule: accept a strict merit decrease, or simultaneous task and
        # defect progress when the scalar objective is not decreasing.
        filter_ok = (
            defect_ok
            and task_ok
            and problem.objective is None
            and (trial_predicted_value <= 0.0 or trial_ratio >= float(minimum_ratio))
        )
        if merit_ok or filter_ok:
            candidate_theta, candidate_boundaries = trial_theta, trial_boundaries
            candidate_merit, candidate_defects, candidate_task = trial_merit.detach(), trial_defects.detach(), trial_task.detach()
            accepted_predicted = trial_predicted_value
            accepted = True
            break

    actual_value = float(merit.detach() - candidate_merit)
    if not accepted:
        accepted_predicted = 0.0
    ratio = actual_value / accepted_predicted if accepted_predicted > 0.0 else 0.0
    action_after = action_evaluator(candidate_theta, candidate_boundaries) if action_evaluator is not None else None
    action_norm = (
        float(torch.sqrt((action_after.detach() - action_before).square().mean()))
        if action_after is not None else 0.0
    )
    action_max = (
        float((action_after.detach() - action_before).abs().max())
        if action_after is not None else 0.0
    )
    if not accepted:
        action_norm = attempted_action_norm
        action_max = attempted_action_max
    accepted_direction = _pack_joint(candidate_theta - theta.detach(), candidate_boundaries - boundaries.detach())
    linearized_constraint = c.reshape(-1) + _joint_jvp_flat(local, accepted_direction, kind="constraint").detach().reshape(-1)
    linearized_task = r + _joint_jvp_flat(local, accepted_direction, kind="task").detach()
    stationarity = (
        _joint_vjp_flat(local, linearized_task, kind="task")
        + float(damping) * accepted_direction
    )
    if accepted:
        # Backtracking changes the accepted primal step, so the full-step
        # multiplier is no longer its KKT multiplier.  Refit a least-squares
        # multiplier for the accepted scale before reporting stationarity.
        def constraint_normal(cotangent: Tensor) -> Tensor:
            lifted = _joint_vjp_flat(
                local, cotangent.reshape_as(c), kind="constraint"
            )
            projected = _joint_jvp_flat(
                local, lifted, kind="constraint"
            ).reshape(-1)
            return projected + 1.0e-8 * cotangent

        multiplier_rhs = -_joint_jvp_flat(
            local, stationarity, kind="constraint"
        ).detach().reshape(-1)
        accepted_multipliers = solve(constraint_normal, multiplier_rhs)
        stationarity = stationarity + _joint_vjp_flat(
            local, accepted_multipliers.reshape_as(c), kind="constraint"
        )
    constraint_norm = torch.linalg.vector_norm(c.reshape(-1))
    task_gradient_norm = torch.linalg.vector_norm(task_grad)
    return JointFullSpaceStep(
        theta=candidate_theta,
        boundaries=candidate_boundaries,
        defects_before=float(torch.linalg.vector_norm(defects.detach())),
        defects_after=float(torch.linalg.vector_norm(candidate_defects)),
        task_norm_before=float(torch.linalg.vector_norm(task.detach())),
        task_norm_after=float(torch.linalg.vector_norm(candidate_task)),
        merit_before=float(merit.detach()),
        merit_after=float(candidate_merit),
        predicted_reduction=accepted_predicted,
        actual_reduction=actual_value,
        ratio=ratio,
        damping=float(damping),
        linearized_constraint_residual=float(torch.linalg.vector_norm(linearized_constraint)),
        linearized_constraint_relative=float(
            torch.linalg.vector_norm(linearized_constraint)
            / constraint_norm.clamp_min(1.0e-12)
        ) if float(constraint_norm) > 1.0e-12 else float(torch.linalg.vector_norm(linearized_constraint)),
        kkt_stationarity_residual=float(torch.linalg.vector_norm(stationarity)),
        kkt_stationarity_relative=float(
            torch.linalg.vector_norm(stationarity)
            / task_gradient_norm.clamp_min(1.0e-12)
        ) if float(task_gradient_norm) > 1.0e-12 else float(torch.linalg.vector_norm(stationarity)),
        parameter_step_norm=scaled_parameter_norm(candidate_theta - theta.detach()),
        action_step_norm=action_norm,
        action_step_max=action_max,
        linear_solver_converged=bool(
            linear_solves and all(result.converged for result in linear_solves)
        ),
        linear_solver_breakdown=bool(
            not linear_solves or any(result.breakdown for result in linear_solves)
        ),
        linear_solver_iterations=sum(result.iterations for result in linear_solves),
        linear_solver_residual_max=max(
            (result.residual_norm for result in linear_solves), default=float("inf")
        ),
        accepted=accepted,
        backtracks=backtracks,
    )


__all__ = [
    "BoundaryLayout",
    "FullSpaceProblem",
    "FullSpaceStep",
    "JointFullSpaceStep",
    "LinearSolveResult",
    "TrustRegionDiagnostics",
    "boundary_jvp",
    "boundary_vjp",
    "joint_jvp",
    "joint_vjp",
    "so3_local_residual",
    "so3_exp",
    "so3_retract",
    "solve_boundary_lm",
    "solve_joint_sqp_step",
    "trust_region_diagnostics",
]
