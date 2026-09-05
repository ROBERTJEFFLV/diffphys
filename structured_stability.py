from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch.nn import functional as F


Tensor = torch.Tensor
StepMap = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class LyapunovMetric:
    """Candidate quadratic metric derived from a stable desired linear model."""

    matrix: Tensor
    state_cost: Tensor
    linear_model_retention: float
    equation_residual: float


@dataclass(frozen=True)
class LocalStabilityReport:
    """A local diagnostic, not a nonlinear or distribution-wide certificate."""

    jacobian: Tensor
    eigenvalues: Tensor
    spectral_radius: float
    finite_time_gain: float
    lyapunov_max_eigenvalue: Optional[float]
    locally_schur_stable: bool
    finite: bool


@dataclass(frozen=True)
class AugmentedStabilityConfig:
    """Conservative, matrix-free settings for the deployed state diagnostic.

    The supplied callback is one complete closed-loop transition;
    ``horizon_steps`` composes it into the Poincare map and ``cadences`` are
    powers of that map (1, 2 and 4 windows).  The
    output is explicitly sampled evidence: it is not a nonlinear stability
    proof or a certificate.
    """

    horizon_steps: int = 25
    cadences: tuple[int, ...] = (1, 2, 4)
    krylov_dim: int = 8
    arnoldi_tolerance: float = 1.0e-7
    residual_tolerance: float = 5.0e-3
    fixed_point_residual_threshold: float = 5.0e-3
    fixed_point_max_iterations: int = 25
    fixed_point_relaxation: float = 1.0
    jvp_fd_epsilon: float = 1.0e-5
    # A one-percent check is conservative for the saturated SO(3)/allocator
    # rollout while allowing the finite-difference smoke probe to expose real
    # implementation mismatches.
    jvp_fd_tolerance: float = 1.0e-2
    spectral_radius_threshold: float = 0.995
    finite_gain_thresholds: tuple[float, ...] = (1.05, 1.10, 1.20)
    seed: int = 1729


@dataclass(frozen=True)
class MatrixFreeStabilityReport:
    """Serializable summary of an augmented closed-loop probe.

    Arnoldi residuals and finite-time gains are lower-bound estimates when the
    operator is not explicitly materialized.  Discrete latch coordinates can
    be supplied through ``discrete_mask``; their tangent is held at zero.
    """

    spectral_radius: float
    spectral_margin: float
    finite_time_gains: Mapping[str, float]
    arnoldi_dimension: int
    arnoldi_converged: bool
    arnoldi_residual: float
    arnoldi_breakdown: bool
    power_norm_estimate: float
    fixed_point_residual: float
    fixed_point_residual_threshold: float
    fixed_point_iterations: int
    fixed_point_converged: bool
    fixed_point_anchor_hash: str
    fixed_point_anchor_passed: bool
    finite_gain_subspaces_converged: bool
    jvp_central_relative_error: float
    jvp_validation_passed: bool
    finite: bool
    gate_passed: bool
    evidence_kind: str = "sampled_matrix_free_evidence_not_a_proof_or_certificate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "spectral_radius_non_yaw_estimate": self.spectral_radius,
            "spectral_margin_non_yaw": self.spectral_margin,
            "finite_time_gain_by_cadence": dict(self.finite_time_gains),
            "arnoldi_dimension": self.arnoldi_dimension,
            "arnoldi_converged": self.arnoldi_converged,
            "arnoldi_residual": self.arnoldi_residual,
            "arnoldi_breakdown": self.arnoldi_breakdown,
            "power_norm_estimate": self.power_norm_estimate,
            "fixed_point_residual": self.fixed_point_residual,
            "fixed_point_residual_threshold": self.fixed_point_residual_threshold,
            "fixed_point_iterations": self.fixed_point_iterations,
            "fixed_point_converged": self.fixed_point_converged,
            "fixed_point_anchor_hash": self.fixed_point_anchor_hash,
            "fixed_point_anchor_passed": self.fixed_point_anchor_passed,
            "finite_gain_subspaces_converged": self.finite_gain_subspaces_converged,
            "jvp_central_relative_error": self.jvp_central_relative_error,
            "jvp_validation_passed": self.jvp_validation_passed,
            "finite": self.finite,
            "gate_passed": self.gate_passed,
            "evidence_kind": self.evidence_kind,
        }


def default_phase_space_metric(
    *,
    dt: float = 0.01,
    sample_steps: int = 25,
    position_natural_frequency: float = 1.5,
    position_damping_ratio: float = 0.9,
    tilt_natural_frequency: float = 5.0,
    tilt_damping_ratio: float = 1.0,
    tilt_scale: float = 0.10,
    omega_scale: float = 0.50,
    yaw_rate: float = 3.0,
    motor_rate: float = 8.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> LyapunovMetric:
    """Construct a 15D candidate metric from explicit desired poles.

    State order matches ``endpoint_control_error``: p3, v3, tilt2, omega3,
    motor4.  This is a design metric, not a certificate for the learned
    saturated augmented closed loop.
    """

    if dt <= 0.0 or sample_steps < 1:
        raise ValueError("dt and sample_steps must be positive")
    if min(position_natural_frequency, tilt_natural_frequency,
           yaw_rate, motor_rate) <= 0.0:
        raise ValueError("desired rates must be positive")
    if position_damping_ratio <= 0.0 or tilt_damping_ratio <= 0.0:
        raise ValueError("damping ratios must be positive")
    if tilt_scale <= 0.0 or omega_scale <= 0.0:
        raise ValueError("tilt_scale and omega_scale must be positive")
    continuous = torch.zeros(15, 15, device=device, dtype=dtype)

    def second_order(position: int, velocity: int, wn: float, zeta: float,
                     velocity_sign: float = 1.0) -> None:
        continuous[position, velocity] = float(velocity_sign)
        continuous[velocity, position] = -float(velocity_sign) * float(wn) ** 2
        continuous[velocity, velocity] = -2.0 * float(zeta) * float(wn)

    for axis in range(3):
        second_order(axis, 3 + axis, position_natural_frequency,
                     position_damping_ratio)
    # The policy/endpoint chart uses q=(R^T z_d)[:2].  In that chart the
    # physical linearization is q_x_dot=-omega_y and q_y_dot=+omega_x.
    second_order(6, 9, tilt_natural_frequency, tilt_damping_ratio,
                 velocity_sign=-1.0)
    second_order(7, 8, tilt_natural_frequency, tilt_damping_ratio,
                 velocity_sign=1.0)
    continuous[10, 10] = -float(yaw_rate)
    for index in range(11, 15):
        continuous[index, index] = -float(motor_rate)
    # Endpoint errors are normalized as p/v/tilt by 0.1 and omega by 0.5;
    # omega therefore has the required five-times physical scale of tilt.
    scales = torch.tensor(
        (0.10,) * 6 + (float(tilt_scale),) * 2 + (float(omega_scale),) * 3
        + (0.10,) * 4,
        device=device, dtype=dtype,
    )
    similarity = torch.diag(scales.reciprocal()) @ continuous @ torch.diag(scales)
    desired = torch.matrix_exp(similarity * (float(dt) * int(sample_steps)))
    state_cost = torch.eye(15, device=device, dtype=dtype)
    return solve_discrete_lyapunov(desired, state_cost)


def _symmetric(value: Tensor) -> Tensor:
    return 0.5 * (value + value.transpose(-1, -2))


def solve_discrete_lyapunov(
    desired_dynamics: Tensor,
    state_cost: Tensor,
    *,
    max_iterations: int = 10000,
    tolerance: float = 1.0e-11,
) -> LyapunovMetric:
    """Solve ``A.T P A - P = -Q`` by its stable fixed-point series.

    The returned matrix is only a candidate metric for a learned nonlinear
    controller.  ``desired_dynamics`` must itself be Schur stable.
    """

    if desired_dynamics.ndim != 2 or desired_dynamics.shape[0] != desired_dynamics.shape[1]:
        raise ValueError("desired_dynamics must be square")
    if state_cost.shape != desired_dynamics.shape:
        raise ValueError("state_cost must match desired_dynamics")
    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive")
    eigenvalues = torch.linalg.eigvals(desired_dynamics)
    spectral_radius = torch.abs(eigenvalues).max()
    if not bool(torch.isfinite(spectral_radius).item()) or float(spectral_radius) >= 1.0:
        raise ValueError("desired_dynamics must be finite and Schur stable")
    q = _symmetric(state_cost)
    if float(torch.linalg.eigvalsh(q).min()) <= 0.0:
        raise ValueError("state_cost must be positive definite")

    p = torch.zeros_like(q)
    for _ in range(max_iterations):
        updated = q + desired_dynamics.transpose(-1, -2) @ p @ desired_dynamics
        relative = torch.linalg.matrix_norm(updated - p) / torch.linalg.matrix_norm(updated).clamp_min(
            torch.finfo(updated.dtype).eps
        )
        p = updated
        if float(relative.detach()) <= tolerance:
            break
    else:
        raise RuntimeError("discrete Lyapunov iteration did not converge")

    p = _symmetric(p)
    residual = desired_dynamics.transpose(-1, -2) @ p @ desired_dynamics - p + q
    chol = torch.linalg.cholesky(p)
    # P^{-1/2} Q P^{-1/2}; triangular solves avoid an explicit inverse.
    left = torch.linalg.solve_triangular(chol, q, upper=False)
    normalized_q = torch.linalg.solve_triangular(
        chol, left.transpose(-1, -2), upper=False
    ).transpose(-1, -2)
    normalized_q = _symmetric(normalized_q)
    retention = 1.0 - float(torch.linalg.eigvalsh(normalized_q).min().detach())
    return LyapunovMetric(
        matrix=p,
        state_cost=q,
        linear_model_retention=max(0.0, retention),
        equation_residual=float(torch.linalg.matrix_norm(residual).detach()),
    )


def quadratic_metric(error: Tensor, metric: Tensor) -> Tensor:
    if error.shape[-1] != metric.shape[-1] or metric.shape[-2] != metric.shape[-1]:
        raise ValueError("metric shape must match the final error dimension")
    return torch.einsum("...i,ij,...j->...", error, metric, error)


def smooth_contraction_objective(
    energy: Tensor,
    *,
    retention: float,
    epsilon: float = 0.0,
    softness: float = 1.0e-3,
) -> Tensor:
    """Smooth sampled contraction surrogate without claiming a certificate."""

    if energy.ndim < 1 or energy.shape[0] < 2:
        raise ValueError("energy must include at least two sample times")
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must be in [0,1]")
    if softness <= 0.0:
        raise ValueError("softness must be positive")
    margin = energy[1:] - float(retention) * energy[:-1] - float(epsilon)
    positive = float(softness) * F.softplus(margin / float(softness))
    return positive.square()


def _matrix_free_jvp(step_map: StepMap, state: Tensor, vector: Tensor) -> Tensor:
    """JVP with a compatibility fallback for older torch releases."""

    try:
        return torch.func.jvp(step_map, (state,), (vector,))[1]
    except (AttributeError, NotImplementedError, RuntimeError):
        value = state.detach().requires_grad_(True)
        return torch.autograd.functional.jvp(
            step_map, value, vector, create_graph=False, strict=False
        )[1]


def _orthonormalize_basis(basis: Optional[Tensor], dimension: int, dtype: torch.dtype,
                          device: torch.device) -> Optional[Tensor]:
    if basis is None:
        return None
    value = basis.reshape(dimension, -1).to(device=device, dtype=dtype)
    if value.numel() == 0:
        return None
    q, r = torch.linalg.qr(value, mode="reduced")
    keep = torch.abs(torch.diagonal(r)) > 1.0e-10
    return q[:, keep]


def _remove_gauge(vector: Tensor, basis: Optional[Tensor]) -> Tensor:
    if basis is None:
        return vector
    # The gauge basis is real.  ``transpose`` (rather than conjugate
    # transpose) keeps this operation valid for complex Arnoldi Ritz vectors.
    return vector - basis @ (basis.transpose(0, 1) @ vector)


def _sanitize_tangent(vector: Tensor, discrete_mask: Optional[Tensor]) -> Tensor:
    if discrete_mask is None:
        return vector
    mask = discrete_mask.to(device=vector.device, dtype=torch.bool).reshape(-1)
    return torch.where(mask, torch.zeros_like(vector), vector)


def _tensor_sha256(value: Tensor) -> str:
    """Hash a detached anchor without making device/dtype conversions implicit."""

    cpu = value.detach().to(device="cpu").contiguous()
    header = f"{cpu.dtype}:{tuple(cpu.shape)}:".encode("ascii")
    return hashlib.sha256(header + cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _arnoldi_matrix_free(operator: Callable[[Tensor], Tensor], state: Tensor,
                          *, basis: Optional[Tensor], discrete_mask: Optional[Tensor],
                          krylov_dim: int, tolerance: float, seed: int) -> dict[str, Any]:
    """Arnoldi on an operator callback; no Jacobian is ever materialized."""

    dimension = state.numel()
    count = min(int(krylov_dim), dimension)
    if count < 1:
        raise ValueError("krylov_dim must be positive")
    generator = torch.Generator(device=state.device)
    generator.manual_seed(int(seed))
    vector = torch.randn(dimension, dtype=state.dtype, device=state.device,
                         generator=generator)
    vector = _sanitize_tangent(_remove_gauge(vector, basis), discrete_mask)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(tolerance)
    q = torch.zeros((dimension, count + 1), dtype=state.dtype, device=state.device)
    h = torch.zeros((count + 1, count), dtype=state.dtype, device=state.device)
    q[:, 0] = vector
    actual = count
    breakdown = False
    for column in range(count):
        value = operator(q[:, column])
        value = _sanitize_tangent(_remove_gauge(value, basis), discrete_mask)
        for row in range(column + 1):
            coefficient = torch.dot(q[:, row], value)
            h[row, column] += coefficient
            value = value - coefficient * q[:, row]
        # A second pass makes the residual useful for mildly nonnormal maps.
        for row in range(column + 1):
            coefficient = torch.dot(q[:, row], value)
            h[row, column] += coefficient
            value = value - coefficient * q[:, row]
        norm = torch.linalg.vector_norm(value)
        h[column + 1, column] = norm
        if float(norm.detach()) <= tolerance:
            actual = column + 1
            breakdown = True
            break
        if column + 1 < count:
            q[:, column + 1] = value / norm

    projected_h = h[:actual, :actual]
    eigenvalues = torch.linalg.eigvals(projected_h)
    radius = torch.abs(eigenvalues).max() if eigenvalues.numel() else state.new_tensor(float("nan"))
    residuals: list[Tensor] = []
    if eigenvalues.numel():
        _, eigenvectors = torch.linalg.eig(projected_h)
        for index in range(eigenvalues.numel()):
            # The Arnoldi residual is h[m,m-1] times the final Ritz-vector
            # coordinate.  It remains valid when the last column broke down.
            coefficient = (h[actual, actual - 1] if actual < h.shape[0] else h.new_zeros(()))
            residuals.append(torch.abs(coefficient * eigenvectors[-1, index]))
    residual = torch.stack(residuals).max() if residuals else state.new_tensor(float("inf"))
    # A projected Hessenberg is the standard matrix-free lower-bound estimate
    # of the induced gain.  Include the Arnoldi power norm as a conservative
    # randomized cross-check for short horizons.
    gain = torch.linalg.svdvals(projected_h).max() if projected_h.numel() else state.new_tensor(float("nan"))
    power = state.new_tensor(0.0)
    for index in range(actual):
        power = torch.maximum(power, torch.linalg.vector_norm(h[:actual, index]))
    return {
        "radius": float(radius.detach()),
        "residual": float(residual.detach()),
        "dimension": actual,
        "breakdown": breakdown,
        "converged": bool(torch.isfinite(residual).item()
                           and float(residual.detach()) <= tolerance),
        "gain": float(torch.maximum(gain, power).detach()),
        "eigenvalues": eigenvalues.detach(),
    }


def matrix_free_augmented_stability_report(
    step_map: StepMap,
    equilibrium: Tensor,
    *,
    config: AugmentedStabilityConfig = AugmentedStabilityConfig(),
    yaw_basis: Optional[Tensor] = None,
    discrete_mask: Optional[Tensor] = None,
) -> MatrixFreeStabilityReport:
    """Validate a Poincare map with JVP/Arnoldi, quotienting only yaw.

    ``step_map`` is expected to be one complete closed-loop transition.  The
    function composes it into a 25-transition Poincare map and its requested
    cadence powers, and never calls a dense
    Jacobian routine.  ``yaw_basis`` is deliberately explicit: callers must
    declare the one global-yaw tangent they want removed; no other state mode
    is projected automatically.
    """

    if equilibrium.ndim != 1:
        raise ValueError("equilibrium must be a flat vector")
    if config.horizon_steps < 1 or tuple(config.cadences) != (1, 2, 4):
        raise ValueError("augmented stability requires 25-step cadences (1,2,4)")
    if len(config.finite_gain_thresholds) != len(config.cadences):
        raise ValueError("finite_gain_thresholds must match cadences")
    if (config.krylov_dim < 1 or config.arnoldi_tolerance <= 0.0
            or config.residual_tolerance <= 0.0
            or config.fixed_point_residual_threshold <= 0.0
            or config.fixed_point_max_iterations < 1
            or not 0.0 < config.fixed_point_relaxation <= 1.0):
        raise ValueError("invalid Arnoldi settings")
    if config.spectral_radius_threshold <= 0.0:
        raise ValueError("spectral radius threshold must be positive")
    dimension = equilibrium.numel()
    mask = None if discrete_mask is None else discrete_mask.reshape(-1)
    if mask is not None and mask.numel() != dimension:
        raise ValueError("discrete_mask must match equilibrium")
    basis = _orthonormalize_basis(yaw_basis, dimension, equilibrium.dtype, equilibrium.device)
    if basis is not None and mask is not None:
        # A declared gauge tangent must not consume a discrete latch
        # coordinate.  Re-orthogonalize after removing those coordinates so
        # the residual normalization is genuinely non-yaw/non-discrete.
        basis = _orthonormalize_basis(
            _sanitize_tangent(basis, mask), dimension,
            equilibrium.dtype, equilibrium.device,
        )

    def compose(value: Tensor, count: int) -> Tensor:
        result = value
        for _ in range(count):
            result = step_map(result)
        return result

    if basis is not None:
        for column in range(basis.shape[1]):
            tangent = basis[:, column]
            image = _matrix_free_jvp(step_map, equilibrium, tangent)
            if not bool(torch.isfinite(image).all()) or float((image - tangent).norm()) > 1e-5:
                raise ValueError("declared yaw gauge is not a neutral symmetry of this closed-loop map")

    def hold_discrete(value: Tensor, reference: Tensor) -> Tensor:
        """Keep v2 latch coordinates on the Poincare section."""

        if mask is None:
            return value
        return torch.where(mask, reference, value)

    def operator(count: int) -> Callable[[Tensor], Tensor]:
        def apply(vector: Tensor) -> Tensor:
            vector = _sanitize_tangent(_remove_gauge(vector, basis), mask)
            nominal = anchor
            for _ in range(count):
                vector = _matrix_free_jvp(step_map, nominal, vector)
                # The nominal rollout is only a linearization anchor; do not
                # retain its autograd graph across a 100-transition probe.
                nominal = step_map(nominal).detach()
                vector = _sanitize_tangent(_remove_gauge(vector, basis), mask)
            return vector
        return apply

    # Restore a genuine fixed point of the 25-transition Poincare map before
    # linearizing.  Only non-yaw, non-discrete coordinates are corrected.  In
    # particular this does not silently call the post-call75 trajectory state
    # an equilibrium merely because one residual happened to be small.
    active_dimension = dimension
    if mask is not None:
        active_dimension -= int(mask.to(dtype=torch.bool).sum().item())
    if basis is not None:
        active_dimension -= basis.shape[1]
    active_dimension = max(1, active_dimension)

    def fixed_point_step(anchor: Tensor) -> tuple[Tensor, float]:
        with torch.no_grad():
            mapped = compose(anchor, config.horizon_steps)
            delta = _sanitize_tangent(
                _remove_gauge(mapped - anchor, basis), mask
            )
            residual = torch.linalg.vector_norm(delta) / math.sqrt(active_dimension)
            # A yaw drift is a declared gauge and is not allowed to drag the
            # anchor through the orientation chart during restoration.
            updated = hold_discrete(
                anchor + float(config.fixed_point_relaxation) * delta, anchor
            )
        return updated.detach(), float(residual.detach())

    anchor = equilibrium.detach().clone()
    fixed_point_residual = float("inf")
    fixed_point_iterations = 0
    fixed_point_converged = False
    for iteration in range(config.fixed_point_max_iterations):
        anchor, fixed_point_residual = fixed_point_step(anchor)
        fixed_point_iterations = iteration + 1
        # Verify at the state that will actually be linearized.  This avoids
        # reporting the pre-update residual as a fixed-point claim.
        if math.isfinite(fixed_point_residual):
            _, verified_residual = fixed_point_step(anchor)
            fixed_point_residual = verified_residual
            if fixed_point_residual <= config.fixed_point_residual_threshold:
                fixed_point_converged = True
                break
    fixed_point_ok = bool(
        fixed_point_converged
        and math.isfinite(fixed_point_residual)
        and fixed_point_residual <= config.fixed_point_residual_threshold
    )
    fixed_point_anchor_hash = _tensor_sha256(anchor)

    # The JVP operator follows the same restored trajectory and is
    # intentionally matrix-free.
    tangent = torch.randn_like(equilibrium)
    tangent = _sanitize_tangent(_remove_gauge(tangent, basis), mask)
    tangent = tangent / torch.linalg.vector_norm(tangent).clamp_min(config.arnoldi_tolerance)
    nominal = anchor
    analytic = tangent
    for _ in range(config.horizon_steps):
        analytic = _matrix_free_jvp(step_map, nominal, analytic)
        nominal = step_map(nominal).detach()
        analytic = _sanitize_tangent(_remove_gauge(analytic, basis), mask)
    eps = float(config.jvp_fd_epsilon)
    with torch.no_grad():
        central = (compose(anchor + eps * tangent, config.horizon_steps)
                   - compose(anchor - eps * tangent, config.horizon_steps)) / (2.0 * eps)
    central = _sanitize_tangent(_remove_gauge(central, basis), mask)
    jvp_error = float((torch.linalg.vector_norm(analytic - central)
                       / torch.linalg.vector_norm(central).clamp_min(1.0e-12)).detach())

    arnoldi = _arnoldi_matrix_free(
        operator(config.horizon_steps), anchor, basis=basis,
        discrete_mask=mask, krylov_dim=config.krylov_dim,
        tolerance=config.arnoldi_tolerance, seed=config.seed,
    )
    gains: dict[str, float] = {}
    gain_records: dict[int, dict[str, Any]] = {}
    for index, cadence in enumerate(config.cadences):
        record = _arnoldi_matrix_free(
            operator(cadence * config.horizon_steps), anchor,
            basis=basis, discrete_mask=mask, krylov_dim=config.krylov_dim,
            tolerance=config.arnoldi_tolerance, seed=config.seed + index + 1,
        )
        gains[str(cadence)] = record["gain"]
        gain_records[int(cadence)] = record
    finite = bool(torch.isfinite(anchor).all().item()
                  and torch.isfinite(tangent).all().item()
                  and torch.isfinite(analytic).all().item()
                  and torch.isfinite(central).all().item()
                  and all(math.isfinite(value) for value in gains.values())
                  and math.isfinite(arnoldi["radius"]))
    jvp_ok = bool(finite and jvp_error <= config.jvp_fd_tolerance)
    gains_ok = all(gains[str(cadence)] <= limit
                   for cadence, limit in zip(config.cadences, config.finite_gain_thresholds))
    arnoldi_ok = bool(
        math.isfinite(arnoldi["residual"])
        and arnoldi["residual"] <= config.residual_tolerance
    )
    gain_subspaces_ok = bool(all(
        math.isfinite(record["residual"])
        and record["residual"] <= config.residual_tolerance
        for record in gain_records.values()
    ))
    gate = bool(finite and fixed_point_ok and jvp_ok and arnoldi_ok
                and gain_subspaces_ok
                and arnoldi["radius"] <= config.spectral_radius_threshold
                and gains_ok)
    return MatrixFreeStabilityReport(
        spectral_radius=arnoldi["radius"],
        spectral_margin=float(1.0 - arnoldi["radius"]),
        finite_time_gains=gains,
        arnoldi_dimension=arnoldi["dimension"],
        arnoldi_converged=arnoldi_ok,
        arnoldi_residual=arnoldi["residual"],
        arnoldi_breakdown=arnoldi["breakdown"],
        power_norm_estimate=max(gains.values()) if gains else float("nan"),
        fixed_point_residual=fixed_point_residual,
        fixed_point_residual_threshold=config.fixed_point_residual_threshold,
        fixed_point_iterations=fixed_point_iterations,
        fixed_point_converged=fixed_point_converged,
        fixed_point_anchor_hash=fixed_point_anchor_hash,
        fixed_point_anchor_passed=fixed_point_ok,
        finite_gain_subspaces_converged=gain_subspaces_ok,
        jvp_central_relative_error=jvp_error,
        jvp_validation_passed=jvp_ok,
        finite=finite,
        gate_passed=gate,
    )


# Friendly aliases for downstream tools and external validation scripts.
augmented_closed_loop_stability_report = matrix_free_augmented_stability_report
matrix_free_stability_diagnostic = matrix_free_augmented_stability_report


def dense_step_jacobian(step_map: StepMap, equilibrium: Tensor) -> Tensor:
    """Return a dense local Jacobian for small validation problems only."""

    if equilibrium.ndim != 1:
        raise ValueError("equilibrium must be a flat vector")
    try:
        return torch.func.jacrev(step_map)(equilibrium)
    except (AttributeError, NotImplementedError, RuntimeError):
        return torch.autograd.functional.jacobian(
            step_map,
            equilibrium.detach().requires_grad_(True),
            create_graph=False,
            strict=False,
        )


def local_stability_report(
    step_map: StepMap,
    equilibrium: Tensor,
    *,
    horizon_steps: int = 25,
    metric: Optional[Tensor] = None,
    tolerance: float = 1.0e-7,
) -> LocalStabilityReport:
    """Inspect the augmented one-step linearization at one equilibrium."""

    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    jacobian = dense_step_jacobian(step_map, equilibrium)
    eigenvalues = torch.linalg.eigvals(jacobian)
    spectral_radius_tensor = torch.abs(eigenvalues).max()
    transition = torch.linalg.matrix_power(jacobian, horizon_steps)
    finite_time_gain_tensor = torch.linalg.svdvals(transition).max()
    values = (jacobian, eigenvalues, spectral_radius_tensor, finite_time_gain_tensor)
    finite = all(bool(torch.isfinite(value).all().item()) for value in values)

    lyapunov_max = None
    if metric is not None:
        if metric.shape != jacobian.shape:
            raise ValueError("metric must match the step Jacobian")
        decrease = jacobian.transpose(-1, -2) @ metric @ jacobian - metric
        lyapunov_max = float(torch.linalg.eigvalsh(_symmetric(decrease)).max().detach())
    spectral_radius = float(spectral_radius_tensor.detach())
    return LocalStabilityReport(
        jacobian=jacobian.detach(),
        eigenvalues=eigenvalues.detach(),
        spectral_radius=spectral_radius,
        finite_time_gain=float(finite_time_gain_tensor.detach()),
        lyapunov_max_eigenvalue=lyapunov_max,
        locally_schur_stable=finite and spectral_radius < 1.0 - float(tolerance),
        finite=finite,
    )
