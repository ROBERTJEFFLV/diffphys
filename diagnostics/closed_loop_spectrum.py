from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

Tensor = torch.Tensor
StepMap = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class ArnoldiConfig:
    krylov_dim: int = 32
    num_eigenvalues: int = 10
    tolerance: float = 1.0e-8
    reorthogonalize: bool = True
    seed: int = 1007
    finite_difference_epsilon: float = 1.0e-5
    residual_tolerance: float = 1.0e-5


@dataclass
class SpectrumResult:
    eigenvalues: np.ndarray
    residuals: np.ndarray
    ritz_vectors: np.ndarray
    spectral_radius: float
    converged: bool
    arnoldi_dimension: int
    jvp_validation_central_error: float
    jvp_validation_forward_error: float
    jvp_validation_backward_error: float
    nonsmooth: bool
    residual_tolerance: float


def _jvp(step_map: StepMap, state: Tensor, vector: Tensor) -> Tensor:
    try:
        return torch.func.jvp(step_map, (state,), (vector,))[1]
    except (AttributeError, NotImplementedError, RuntimeError):
        value = state.detach().requires_grad_(True)
        return torch.autograd.functional.jvp(step_map, value, vector, create_graph=False, strict=False)[1]


def orthogonal_projector_from_basis(basis: Tensor, eps: float = 1.0e-10) -> Tensor:
    if basis.ndim == 1:
        basis = basis[:, None]
    if basis.ndim != 2:
        raise ValueError("basis must contain basis vectors in columns")
    q, r = torch.linalg.qr(basis, mode="reduced")
    q = q[:, torch.abs(torch.diagonal(r)) > eps]
    identity = torch.eye(basis.shape[0], dtype=basis.dtype, device=basis.device)
    return identity - q @ q.T


def _project(projector: Tensor | None, vector: Tensor) -> Tensor:
    return vector if projector is None else projector @ vector


def make_projected_operator(step_map: StepMap, state: Tensor, projector: Tensor | None) -> Callable[[Tensor], Tensor]:
    if projector is not None and projector.shape != (state.numel(), state.numel()):
        raise ValueError("projector shape must match flattened state dimension")
    return lambda vector: _project(projector, _jvp(step_map, state, _project(projector, vector)))


def _finite_difference(
    step_map: StepMap,
    state: Tensor,
    vector: Tensor,
    *,
    epsilon: float,
    direction: str,
    projector: Tensor | None,
) -> Tensor:
    tangent = _project(projector, vector)
    with torch.no_grad():
        baseline = step_map(state)
        if direction == "central":
            result = (step_map(state + epsilon * tangent) - step_map(state - epsilon * tangent)) / (2 * epsilon)
        elif direction == "forward":
            result = (step_map(state + epsilon * tangent) - baseline) / epsilon
        elif direction == "backward":
            result = (baseline - step_map(state - epsilon * tangent)) / epsilon
        else:
            raise ValueError(direction)
    return _project(projector, result)


def validate_jvp(
    step_map: StepMap,
    state: Tensor,
    vector: Tensor,
    *,
    epsilon: float,
    projector: Tensor | None,
) -> tuple[float, float, float]:
    vector = _project(projector, vector)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)
    analytic = _project(projector, _jvp(step_map, state, vector))
    errors = []
    for direction in ("central", "forward", "backward"):
        finite = _finite_difference(
            step_map, state, vector, epsilon=epsilon, direction=direction, projector=projector
        )
        errors.append(float(
            (torch.linalg.vector_norm(analytic - finite) / torch.linalg.vector_norm(finite).clamp_min(1e-12))
            .detach().cpu()
        ))
    return tuple(errors)  # type: ignore[return-value]


def arnoldi_spectrum(
    *,
    step_map: StepMap,
    state: Tensor,
    projector: Tensor | None = None,
    config: ArnoldiConfig = ArnoldiConfig(),
    nonsmooth: bool = False,
) -> SpectrumResult:
    if state.ndim != 1:
        raise ValueError("state must be a flattened vector")
    dimension = state.numel()
    krylov = min(config.krylov_dim, dimension)
    if krylov < 2:
        raise ValueError("state dimension must be at least two")
    operator = make_projected_operator(step_map, state, projector)
    generator = torch.Generator(device=state.device)
    generator.manual_seed(config.seed)
    initial = _project(projector, torch.randn(dimension, dtype=state.dtype, device=state.device, generator=generator))
    initial = initial / torch.linalg.vector_norm(initial).clamp_min(config.tolerance)
    q = torch.zeros((dimension, krylov + 1), dtype=state.dtype, device=state.device)
    h = torch.zeros((krylov + 1, krylov), dtype=state.dtype, device=state.device)
    q[:, 0] = initial
    actual_dimension = krylov
    for column in range(krylov):
        vector = operator(q[:, column])
        for row in range(column + 1):
            coefficient = torch.dot(q[:, row], vector)
            h[row, column] = coefficient
            vector -= coefficient * q[:, row]
        if config.reorthogonalize:
            for row in range(column + 1):
                correction = torch.dot(q[:, row], vector)
                h[row, column] += correction
                vector -= correction * q[:, row]
        norm = torch.linalg.vector_norm(vector)
        h[column + 1, column] = norm
        if float(norm.detach().cpu()) <= config.tolerance:
            actual_dimension = column + 1
            break
        q[:, column + 1] = vector / norm

    eigenvalues, eigenvectors_h = torch.linalg.eig(h[:actual_dimension, :actual_dimension])
    keep = torch.argsort(torch.abs(eigenvalues), descending=True)[: min(config.num_eigenvalues, actual_dimension)]
    eigenvalues = eigenvalues[keep]
    eigenvectors_h = eigenvectors_h[:, keep]
    complex_dtype = torch.complex128 if state.dtype == torch.float64 else torch.complex64
    ritz_vectors = q[:, :actual_dimension].to(complex_dtype) @ eigenvectors_h
    ritz_vectors /= torch.linalg.vector_norm(ritz_vectors, dim=0).clamp_min(config.tolerance)
    residuals = []
    for index in range(eigenvalues.numel()):
        vector = ritz_vectors[:, index]
        applied = torch.complex(operator(vector.real), operator(vector.imag))
        residuals.append(float(torch.linalg.vector_norm(applied - eigenvalues[index] * vector).detach().cpu()))
    residual_array = np.asarray(residuals, dtype=np.float64)

    validation_vector = torch.randn(dimension, dtype=state.dtype, device=state.device, generator=generator)
    validation = validate_jvp(
        step_map,
        state,
        validation_vector,
        epsilon=config.finite_difference_epsilon,
        projector=projector,
    )
    eigenvalues_np = eigenvalues.detach().cpu().numpy()
    threshold = config.residual_tolerance
    return SpectrumResult(
        eigenvalues=eigenvalues_np,
        residuals=residual_array,
        ritz_vectors=ritz_vectors.detach().cpu().numpy(),
        spectral_radius=float(np.max(np.abs(eigenvalues_np))) if eigenvalues_np.size else float("nan"),
        converged=bool(residual_array.size and np.all(residual_array <= threshold)),
        arnoldi_dimension=actual_dimension,
        jvp_validation_central_error=validation[0],
        jvp_validation_forward_error=validation[1],
        jvp_validation_backward_error=validation[2],
        nonsmooth=nonsmooth,
        residual_tolerance=config.residual_tolerance,
    )


def mode_energy_fractions(
    ritz_vectors: np.ndarray,
    layout: Mapping[str, slice | Sequence[int]],
) -> list[dict[str, float]]:
    if ritz_vectors.ndim != 2:
        raise ValueError("ritz_vectors must have shape [state_dim,num_modes]")
    records = []
    for mode in range(ritz_vectors.shape[1]):
        vector = ritz_vectors[:, mode]
        total = float(np.vdot(vector, vector).real)
        record: dict[str, float] = {"mode_index": float(mode)}
        for name, selector in layout.items():
            component = vector[selector]
            record[f"energy_fraction_{name}"] = float(np.vdot(component, component).real) / max(total, 1e-18)
        records.append(record)
    return records


def spectrum_to_frames(
    result: SpectrumResult,
    *,
    scenario_uid: str,
    checkpoint: str,
    spectrum_kind: str,
    dt: float,
    layout: Mapping[str, slice | Sequence[int]] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = dict(metadata or {})
    eigenvalue_records = []
    for index, (eigenvalue, residual) in enumerate(zip(result.eigenvalues, result.residuals)):
        magnitude = float(abs(eigenvalue))
        eigenvalue_records.append({
            "scenario_uid": scenario_uid,
            "checkpoint": checkpoint,
            "spectrum_kind": spectrum_kind,
            "mode_index": index,
            "eigenvalue_real": float(eigenvalue.real),
            "eigenvalue_imaginary": float(eigenvalue.imag),
            "magnitude": magnitude,
            "frequency_hz": abs(float(np.angle(eigenvalue))) / (2 * np.pi * dt),
            "continuous_decay_rate_per_s": float(np.log(max(magnitude, 1e-18)) / dt),
            "residual": float(residual),
            "spectral_radius": result.spectral_radius,
            "arnoldi_dimension": result.arnoldi_dimension,
            "arnoldi_converged": result.converged,
            "mode_converged": bool(float(residual) <= result.residual_tolerance),
            "nonsmooth": result.nonsmooth,
            **metadata,
        })
    spectrum = pd.DataFrame(eigenvalue_records)
    modes = pd.DataFrame(mode_energy_fractions(result.ritz_vectors, layout)) if layout else pd.DataFrame()
    if not modes.empty:
        for key, value in reversed((
            ("scenario_uid", scenario_uid), ("checkpoint", checkpoint), ("spectrum_kind", spectrum_kind)
        )):
            modes.insert(0, key, value)
        for key, value in metadata.items():
            modes[key] = value
    validation = pd.DataFrame([{
        "scenario_uid": scenario_uid,
        "checkpoint": checkpoint,
        "spectrum_kind": spectrum_kind,
        "jvp_central_relative_error": result.jvp_validation_central_error,
        "jvp_forward_relative_error": result.jvp_validation_forward_error,
        "jvp_backward_relative_error": result.jvp_validation_backward_error,
        "nonsmooth": result.nonsmooth,
        "pass": bool(
            (not result.nonsmooth and result.jvp_validation_central_error < 1e-3)
            or (result.nonsmooth and min(result.jvp_validation_forward_error, result.jvp_validation_backward_error) < 1e-3)
        ),
        **metadata,
    }])
    return spectrum, modes, validation
