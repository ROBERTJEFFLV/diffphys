"""Reduced, symmetrically scaled KKT solves using PETSc MINRES.

PETSc owns Krylov vectors; PyTorch owns physics/autograd. All vector exchange
uses DLPack, including on CPU. CUDA callbacks deliberately synchronize at the
runtime boundary: shared storage does not imply shared stream ordering.
PETSc is an optional, lazily imported dependency; it must use the same real
scalar precision as the problem. No CPU or legacy-solver fallback is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from full_space_shooting import FullSpaceProblem, joint_jvp, joint_vjp

Tensor = torch.Tensor


@dataclass(frozen=True)
class PetscKKTResult:
    dtheta: Tensor
    dboundary: Tensor
    multipliers: Tensor
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    scaled_relative_residual: float
    reason: int
    finite: bool
    preconditioner: str = "block-diagonal"
    curvature_probes: int = 0


def _positive_scale(value: Tensor | None, reference: Tensor, name: str) -> Tensor:
    if value is None:
        return torch.ones_like(reference)
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must share the problem dtype and device")
    try:
        scale = torch.broadcast_to(value.detach(), reference.shape).clone()
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to {tuple(reference.shape)}") from exc
    if not bool(torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError(f"{name} must be finite and strictly positive")
    return scale


@dataclass(frozen=True)
class ReducedKKTLayout:
    """Pack free policy/boundary variables and dynamic continuity rows.

The fixed mask identifies deterministic coordinates such as the response
clock. Their continuity rows are removed together with their primal columns.
The solver checks excluded constraints again on the computed direction.
"""
    theta_template: Tensor
    boundary_template: Tensor
    free_boundary_mask: Tensor
    free_constraint_mask: Tensor
    primal_scale: Tensor
    constraint_scale: Tensor

    @classmethod
    def from_problem(
        cls, problem: FullSpaceProblem, *, parameter_scale: Tensor | None = None,
        boundary_scale: Tensor | None = None, constraint_scale: Tensor | None = None,
    ) -> ReducedKKTLayout:
        theta, boundary = problem.theta.detach(), problem.boundaries.detach()
        if theta.dtype != boundary.dtype or theta.device != boundary.device:
            raise ValueError("policy and boundaries must share dtype and device")
        if theta.dtype not in (torch.float32, torch.float64):
            raise ValueError("PETSc KKT requires real float32 or float64 tensors")
        fixed = problem.fixed_boundary_mask
        if fixed is None:
            free = torch.ones_like(boundary, dtype=torch.bool)
        else:
            if fixed.dtype != torch.bool or fixed.device != boundary.device:
                raise ValueError("fixed_boundary_mask must be boolean on the problem device")
            free = ~torch.broadcast_to(fixed, boundary.shape).detach().clone()
        theta_scale = _positive_scale(
            theta.abs().clamp_min(1.0e-2) if parameter_scale is None else parameter_scale,
            theta, "parameter_scale",
        )
        node_scale = _positive_scale(boundary_scale, boundary, "boundary_scale")
        row_scale = _positive_scale(constraint_scale, boundary, "constraint_scale")
        return cls(theta, boundary, free, free.clone(),
                   torch.cat((theta_scale.reshape(-1), node_scale[free])), row_scale[free])

    @property
    def n_primal(self) -> int:
        return self.primal_scale.numel()

    @property
    def n_constraint(self) -> int:
        return self.constraint_scale.numel()

    @property
    def size(self) -> int:
        return self.n_primal + self.n_constraint

    def pack_primal(self, theta: Tensor, boundary: Tensor) -> Tensor:
        return torch.cat((theta.reshape(-1), boundary[self.free_boundary_mask]))

    def unpack_primal(self, vector: Tensor) -> tuple[Tensor, Tensor]:
        if vector.ndim != 1 or vector.numel() != self.n_primal:
            raise ValueError("incorrect reduced primal size")
        count = self.theta_template.numel()
        boundary = torch.zeros_like(self.boundary_template)
        boundary[self.free_boundary_mask] = vector[count:]
        return vector[:count].reshape_as(self.theta_template), boundary

    def pack_constraint(self, value: Tensor) -> Tensor:
        if value.shape != self.boundary_template.shape:
            raise ValueError("continuity residual must have the boundary shape")
        return value[self.free_constraint_mask]

    def unpack_constraint(self, value: Tensor) -> Tensor:
        if value.ndim != 1 or value.numel() != self.n_constraint:
            raise ValueError("incorrect reduced constraint size")
        result = torch.zeros_like(self.boundary_template)
        result[self.free_constraint_mask] = value
        return result

    def recover(self, solution: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dtheta, dboundary = self.unpack_primal(solution[:self.n_primal] * self.primal_scale)
        # lambda = S_c^{-T} lambda_hat, not S_c lambda_hat.
        multipliers = self.unpack_constraint(solution[self.n_primal:] / self.constraint_scale)
        return dtheta.detach(), dboundary.detach(), multipliers.detach()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _vec_tensor(vector, device: torch.device, dtype: torch.dtype) -> Tensor:
    result = torch.utils.dlpack.from_dlpack(vector.toDLPack(mode="r"))
    if result.device != device or result.dtype != dtype:
        raise RuntimeError("PETSc vector device/dtype does not match PyTorch; no copying fallback")
    return result


class TorchKKTContext:
    """Frozen linearization and MatPython callback for the full KKT operator."""
    def __init__(self, problem: FullSpaceProblem, damping: float, layout: ReducedKKTLayout) -> None:
        if not math.isfinite(damping) or damping <= 0:
            raise ValueError("damping must be finite and positive")
        if problem.objective is not None:
            raise ValueError("KKT requires objective=None; use task_residual")
        self.problem, self.damping, self.layout = problem, float(damping), layout
        self.device, self.dtype = problem.theta.device, problem.theta.dtype
        with torch.enable_grad():
            self.task = problem.task_values().detach()
            self.defects = problem.defects().detach()
            self.gradient = layout.pack_primal(*joint_vjp(problem, self.task, kind="task")).detach()
        self.constraint = layout.pack_constraint(self.defects)
        self.raw_rhs = -torch.cat((self.gradient, self.constraint))
        self.scaling = torch.cat((layout.primal_scale, layout.constraint_scale.reciprocal()))
        self.rhs = (self.scaling * self.raw_rhs).contiguous()

    def raw_hessian(self, direction: Tensor) -> Tensor:
        dtheta, dboundary = self.layout.unpack_primal(direction)
        with torch.enable_grad():
            task_jvp = joint_jvp(self.problem, dtheta, dboundary, kind="task").detach()
            task_vjp = joint_vjp(self.problem, task_jvp, kind="task")
        return self.layout.pack_primal(*task_vjp).detach() + self.damping * direction

    def raw_matvec(self, vector: Tensor) -> Tensor:
        layout = self.layout
        direction = vector[:layout.n_primal]
        dtheta, dboundary = layout.unpack_primal(direction)
        multiplier = layout.unpack_constraint(vector[layout.n_primal:])
        primal = self.raw_hessian(direction)
        with torch.enable_grad():
            constraint_vjp = joint_vjp(self.problem, multiplier, kind="constraint")
            constraint_jvp = joint_jvp(self.problem, dtheta, dboundary, kind="constraint").detach()
        primal = primal + layout.pack_primal(*constraint_vjp).detach()
        return torch.cat((primal, layout.pack_constraint(constraint_jvp))).detach()

    def matvec(self, vector: Tensor) -> Tensor:
        # Congruence D K D preserves symmetry; D = diag(S_d, S_c^{-1}).
        return self.scaling * self.raw_matvec(self.scaling * vector)

    def mult(self, matrix, x, y) -> None:
        _synchronize(self.device)
        xt = _vec_tensor(x, self.device, self.dtype)
        yt = torch.utils.dlpack.from_dlpack(y.toDLPack(mode="w"))
        if yt.device != self.device or yt.dtype != self.dtype:
            raise RuntimeError("PETSc output vector has incorrect device/dtype")
        yt.copy_(self.matvec(xt))
        _synchronize(self.device)

    def multTranspose(self, matrix, x, y) -> None:
        self.mult(matrix, x, y)


class FixedSPDPreconditioner:
    """Fixed diagonal P^{-1}; copied once so caller mutations cannot alter PC."""
    def __init__(self, inverse_diag: Tensor) -> None:
        if inverse_diag.ndim != 1 or not bool(
            torch.isfinite(inverse_diag).all() and (inverse_diag > 0).all()
        ):
            raise ValueError("preconditioner diagonal must be finite and strictly positive")
        self.inverse_diag = inverse_diag.detach().clone().contiguous()

    def apply(self, pc, x, y) -> None:
        diagonal = self.inverse_diag
        _synchronize(diagonal.device)
        xt = _vec_tensor(x, diagonal.device, diagonal.dtype)
        yt = torch.utils.dlpack.from_dlpack(y.toDLPack(mode="w"))
        if yt.device != diagonal.device or yt.dtype != diagonal.dtype:
            raise RuntimeError("PETSc preconditioner output has incorrect device/dtype")
        yt.copy_(xt * diagonal)
        _synchronize(diagonal.device)

    def applyTranspose(self, pc, x, y) -> None:
        self.apply(pc, x, y)


def curvature_diagonal(context: TorchKKTContext, probes: int = 8) -> Tensor:
    """Fixed-seed Hutchinson diagonal of S_d H S_d, computed once per solve."""
    if not isinstance(probes, int) or probes < 1:
        raise ValueError("curvature_probes must be a positive integer")
    scale = context.layout.primal_scale
    generator = torch.Generator(device=scale.device).manual_seed(0)
    vectors = torch.randint(0, 2, (probes, scale.numel()), generator=generator,
                            device=scale.device).to(scale.dtype).mul_(2).sub_(1)
    diagonal = torch.zeros_like(scale)
    for vector in vectors.unbind(0):
        diagonal.add_(vector * scale * context.raw_hessian(scale * vector))
    # Bounds and abs make this SPD even when a finite-sample estimate is
    # negative. The PC's copy then remains unchanged throughout MINRES.
    return (diagonal / probes).abs().clamp(min=1.0e-4, max=1.0e4)


def require_petsc(reference: Tensor):
    """Fail before exporting storage if PETSc would reinterpret its precision."""
    try:
        from petsc4py import PETSc
    except ImportError as exc:
        raise RuntimeError(
            "petsc-minres requires petsc4py and PETSc with DLPack support. "
            "Use a matching real precision build (CUDA enabled for CUDA tensors), "
            "or explicitly select linear_solver='legacy-cg' for debugging."
        ) from exc
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(PETSc.ScalarType.__name__)
    if dtype is None or reference.dtype != dtype:
        raise ValueError(
            f"PETSc ScalarType={PETSc.ScalarType.__name__} does not match {reference.dtype}; "
            "use matching problem precision, without implicit tensor conversion"
        )
    if not all(hasattr(PETSc.Vec, name) for name in ("createWithDLPack", "toDLPack")):
        raise RuntimeError("petsc4py must support Vec.createWithDLPack and Vec.toDLPack")
    if reference.device.type == "cuda" and not PETSc.Sys.hasExternalPackage("cuda"):
        raise RuntimeError("CUDA tensors require a CUDA-enabled PETSc build")
    if reference.device.type not in ("cpu", "cuda"):
        raise ValueError("KKT bridge supports CPU and CUDA only")
    return PETSc


def solve_petsc_minres(
    problem: FullSpaceProblem, *, damping: float = 1.0e-3,
    rtol: float = 1.0e-6, atol: float = 1.0e-10, max_iterations: int = 200,
    parameter_scale: Tensor | None = None, boundary_scale: Tensor | None = None,
    constraint_scale: Tensor | None = None, monitor: bool = False,
    preconditioner: str = "curvature-diagonal", curvature_probes: int = 8,
) -> PetscKKTResult:
    """Solve one reduced KKT system and independently certify its residual.

Both scaled and original reduced residuals must satisfy ``rtol``, in addition
to a positive PETSc reason and finite values. ``atol`` controls PETSc stopping
only; it does not weaken the independent relative-residual gate.
"""
    if not (math.isfinite(rtol) and 0 < rtol < 1 and math.isfinite(atol) and atol >= 0):
        raise ValueError("rtol must be in (0,1) and atol finite/nonnegative")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if preconditioner not in ("block-diagonal", "curvature-diagonal"):
        raise ValueError("preconditioner must be block-diagonal or curvature-diagonal")
    PETSc = require_petsc(problem.theta)
    layout = ReducedKKTLayout.from_problem(
        problem, parameter_scale=parameter_scale,
        boundary_scale=boundary_scale, constraint_scale=constraint_scale,
    )
    context = TorchKKTContext(problem, damping, layout)
    if not bool(torch.isfinite(context.rhs).all() and torch.isfinite(context.defects).all()):
        raise ValueError("non-finite KKT right-hand side")
    excluded = context.defects[~layout.free_constraint_mask]
    if excluded.numel() and bool((excluded.abs() > 1.0e-12).any()):
        raise ValueError("cannot remove fixed-coordinate constraint rows with nonzero residual")
    # V1 is available for comparison. The curvature variant was introduced
    # after V1 failed the real tiny-problem oracle at 2000 iterations.
    primal_diag = (curvature_diagonal(context, curvature_probes)
                   if preconditioner == "curvature-diagonal"
                   else torch.full_like(layout.primal_scale, 1.0 + damping))
    inverse_diag = torch.cat((
        primal_diag.reciprocal(),
        torch.ones_like(layout.constraint_scale),
    ))
    pc_context = FixedSPDPreconditioner(inverse_diag)
    # A tolerance in a preconditioned/scaled norm is not the tolerance in
    # original coordinates. Use conservative diagonal norm bounds for both
    # PETSc MINRES residual conventions (P^-1 r and P^-1/2 r). The final
    # independent matvec remains authoritative, including recurrence drift.
    norm_floor = 1.0e-12
    pc_weight_lower = torch.minimum(inverse_diag, inverse_diag.sqrt())
    initial_norm_upper = max(float((context.rhs * inverse_diag).norm()),
                             float((context.rhs * inverse_diag.sqrt()).norm()), norm_floor)
    residual_target = 0.5 * rtol * min(
        float(context.raw_rhs.norm().clamp_min(norm_floor)) * float((context.scaling * pc_weight_lower).min()),
        float(context.rhs.norm().clamp_min(norm_floor)) * float(pc_weight_lower.min()),
    )
    internal_rtol = min(rtol, residual_target / initial_norm_upper)
    internal_atol = min(atol, residual_target)
    solution = torch.zeros_like(context.rhs)
    matrix = rhs_vec = solution_vec = ksp = None
    try:
        _synchronize(context.device)
        rhs_vec = PETSc.Vec().createWithDLPack(context.rhs, comm=PETSc.COMM_SELF)
        solution_vec = PETSc.Vec().createWithDLPack(solution, comm=PETSc.COMM_SELF)
        matrix = PETSc.Mat().createPython([layout.size, layout.size], context=context, comm=PETSc.COMM_SELF)
        matrix.setVecType("cuda" if context.device.type == "cuda" else "seq")
        matrix.setUp()
        matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(matrix)
        ksp.setType(PETSc.KSP.Type.MINRES)
        ksp.setPCSide(PETSc.PC.Side.LEFT)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(pc_context)
        ksp.setTolerances(rtol=internal_rtol, atol=internal_atol, max_it=max_iterations)
        ksp.setInitialGuessNonzero(False)
        if monitor:
            ksp.setMonitor(lambda solver, it, norm: print(
                f"PETSc MINRES iteration={it} residual={norm:.12e}", flush=True
            ))
        # Deliberately no setFromOptions: an ambient option must not replace
        # MINRES, left PC, or the fixed SPD context behind the API's back.
        ksp.solve(rhs_vec, solution_vec)
        _synchronize(context.device)
        reason, iterations = int(ksp.getConvergedReason()), int(ksp.getIterationNumber())
    finally:
        _synchronize(context.device)
        for handle in (ksp, matrix, solution_vec, rhs_vec):
            if handle is not None:
                handle.destroy()
    raw_solution = context.scaling * solution
    raw_residual = context.raw_matvec(raw_solution) - context.raw_rhs
    scaled_residual = context.scaling * raw_residual
    residual_norm = float(raw_residual.norm())
    relative = float(raw_residual.norm() / context.raw_rhs.norm().clamp_min(1.0e-12))
    scaled_relative = float(scaled_residual.norm() / context.rhs.norm().clamp_min(1.0e-12))
    dtheta, dboundary, multipliers = layout.recover(solution)
    with torch.enable_grad():
        full_constraint = context.defects + joint_jvp(problem, dtheta, dboundary, kind="constraint").detach()
    excluded_after = full_constraint[~layout.free_constraint_mask]
    excluded_ok = not excluded_after.numel() or bool((excluded_after.abs() <= 1.0e-12).all())
    finite = bool(torch.isfinite(solution).all() and torch.isfinite(raw_residual).all()
                  and torch.isfinite(full_constraint).all())
    converged = bool(reason > 0 and finite and excluded_ok and relative <= rtol and scaled_relative <= rtol)
    return PetscKKTResult(dtheta, dboundary, multipliers, converged, iterations,
                          residual_norm, relative, scaled_relative, reason, finite,
                          preconditioner, curvature_probes if preconditioner == "curvature-diagonal" else 0)
