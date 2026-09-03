from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

Tensor = torch.Tensor
BatchWrenchFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class IntegralSensitivityConfig:
    eps: float = 1.0e-9
    finite_difference_epsilon: float = 1.0e-4
    weak_sensitivity_relative_threshold: float = 1.0e-3
    cancellation_threshold: float = 0.5
    wrong_direction_cosine_threshold: float = 0.0


def _validate_shapes(integral: Tensor, wrench: Tensor) -> None:
    if integral.ndim != 2 or integral.shape[-1] != 3:
        raise ValueError(f"Integral must have shape [batch,3], got {tuple(integral.shape)}")
    if wrench.shape != (integral.shape[0], 4):
        raise ValueError(f"Wrench must have shape [batch,4], got {tuple(wrench.shape)}")


def batched_independent_jacobian_jvp(function: BatchWrenchFn, integral: Tensor) -> tuple[Tensor, Tensor]:
    """Recover each independent 4x3 sample Jacobian with three batched JVPs."""
    value = integral.detach().requires_grad_(True)
    baseline = function(value)
    _validate_shapes(value, baseline)
    columns: list[Tensor] = []
    for axis in range(3):
        tangent = torch.zeros_like(value)
        tangent[:, axis] = 1.0
        try:
            output_tangent = torch.func.jvp(function, (value,), (tangent,))[1]
        except (AttributeError, NotImplementedError, RuntimeError):
            output_tangent = torch.autograd.functional.jvp(
                function, value, tangent, create_graph=False, strict=False
            )[1]
        columns.append(output_tangent)
    return baseline.detach(), torch.stack(columns, dim=-1).detach()


def batched_independent_jacobian_finite_difference(
    function: BatchWrenchFn,
    integral: Tensor,
    *,
    epsilon: float,
    direction: str = "central",
) -> Tensor:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    with torch.no_grad():
        baseline = function(integral)
        _validate_shapes(integral, baseline)
        columns = []
        for axis in range(3):
            delta = torch.zeros_like(integral)
            delta[:, axis] = epsilon
            if direction == "central":
                derivative = (function(integral + delta) - function(integral - delta)) / (2 * epsilon)
            elif direction == "forward":
                derivative = (function(integral + delta) - baseline) / epsilon
            elif direction == "backward":
                derivative = (baseline - function(integral - delta)) / epsilon
            else:
                raise ValueError(f"unknown finite-difference direction: {direction}")
            columns.append(derivative)
    return torch.stack(columns, dim=-1).detach()


def _norm(value: Tensor, dim: int | tuple[int, ...], eps: float = 0.0) -> Tensor:
    result = torch.linalg.vector_norm(value, dim=dim)
    return result.clamp_min(eps) if eps else result


def _relative_error(reference: Tensor, estimate: Tensor, eps: float) -> Tensor:
    return _norm(reference - estimate, (1, 2)) / _norm(estimate, (1, 2), eps)


def compute_integral_sensitivity(
    *,
    integral: Tensor,
    main_wrench_fn: BatchWrenchFn,
    main_plus_integral_wrench_fn: BatchWrenchFn,
    full_wrench_fn: BatchWrenchFn,
    desired_wrench_direction: Tensor | None = None,
    integral_update_direction: Tensor | None = None,
    actuator_wrench_scale: Tensor | None = None,
    scenario_uid: Sequence[str] | None = None,
    metadata: Mapping[str, Sequence[object] | Tensor | np.ndarray] | None = None,
    config: IntegralSensitivityConfig = IntegralSensitivityConfig(),
    validate_finite_difference: bool = True,
    nonsmooth_mask: Tensor | None = None,
) -> pd.DataFrame:
    """Measure main, explicit-integral, and total physical-wrench sensitivity.

    Callbacks must include logits, tanh, one motor-lag step, thrust polynomial,
    and mixer. State, previous hidden, and true current motor state are frozen at
    the diagnostic snapshot; true motor state is never supplied to the actor.
    """
    _, jac_main = batched_independent_jacobian_jvp(main_wrench_fn, integral)
    _, jac_main_integral = batched_independent_jacobian_jvp(main_plus_integral_wrench_fn, integral)
    _, jac_full = batched_independent_jacobian_jvp(full_wrench_fn, integral)
    jac_explicit = jac_main_integral - jac_main
    jac_damping_dependence = jac_full - jac_main_integral
    eps = config.eps

    norm_main = _norm(jac_main, (1, 2))
    norm_explicit = _norm(jac_explicit, (1, 2))
    norm_main_integral = _norm(jac_main_integral, (1, 2))
    norm_total = _norm(jac_full, (1, 2))
    cancellation_ratio = norm_main_integral / (norm_main + norm_explicit).clamp_min(eps)
    cancellation_fraction = (1.0 - cancellation_ratio).clamp(0.0, 1.0)

    singular_values = torch.linalg.svdvals(jac_full)
    largest = singular_values[:, 0]
    smallest = singular_values[:, -1]
    condition = torch.where(smallest > eps, largest / smallest, torch.full_like(largest, float("inf")))
    rank = (singular_values > eps).sum(dim=-1)

    if actuator_wrench_scale is None:
        actuator_wrench_scale = torch.ones((integral.shape[0], 4), dtype=integral.dtype, device=integral.device)
    if actuator_wrench_scale.shape != (integral.shape[0], 4):
        raise ValueError("actuator_wrench_scale must have shape [batch,4]")
    normalized_jac = jac_full / actuator_wrench_scale.abs().clamp_min(eps).unsqueeze(-1)

    direction_sensitivity = torch.full_like(largest, float("nan"))
    direction_cosine = torch.full_like(largest, float("nan"))
    directional_fd_sensitivity = torch.full_like(largest, float("nan"))
    directional_fd_cosine = torch.full_like(largest, float("nan"))
    directional_fd_cancellation = torch.full_like(largest, float("nan"))
    directional_fd_normalized_sensitivity = torch.full_like(largest, float("nan"))
    dominant_alignment_magnitude = torch.full_like(largest, float("nan"))
    if desired_wrench_direction is not None:
        if desired_wrench_direction.shape != (integral.shape[0], 4):
            raise ValueError("desired_wrench_direction must have shape [batch,4]")
        desired = desired_wrench_direction / _norm(desired_wrench_direction, 1, eps).unsqueeze(-1)
        dominant_input = torch.linalg.svd(jac_full, full_matrices=False).Vh[:, 0]
        dominant_wrench = torch.einsum("boi,bi->bo", jac_full, dominant_input)
        dominant_alignment_magnitude = torch.abs(torch.einsum("bo,bo->b", dominant_wrench, desired)) / _norm(
            dominant_wrench, 1, eps
        )
        if integral_update_direction is not None:
            if integral_update_direction.shape != (integral.shape[0], 3):
                raise ValueError("integral_update_direction must have shape [batch,3]")
            update = integral_update_direction / _norm(integral_update_direction, 1, eps).unsqueeze(-1)
            induced = torch.einsum("boi,bi->bo", jac_full, update)
            direction_sensitivity = _norm(induced, 1)
            direction_cosine = torch.einsum("bo,bo->b", induced, desired) / _norm(induced, 1, eps)
            with torch.no_grad():
                delta = config.finite_difference_epsilon * update
                full_base = full_wrench_fn(integral)
                main_base = main_wrench_fn(integral)
                main_integral_base = main_plus_integral_wrench_fn(integral)
                full_direction = (full_wrench_fn(integral + delta) - full_base) / config.finite_difference_epsilon
                main_direction = (main_wrench_fn(integral + delta) - main_base) / config.finite_difference_epsilon
                main_integral_direction = (
                    main_plus_integral_wrench_fn(integral + delta) - main_integral_base
                ) / config.finite_difference_epsilon
                explicit_direction = main_integral_direction - main_direction
            directional_fd_sensitivity = _norm(full_direction, 1)
            directional_fd_cosine = torch.einsum("bo,bo->b", full_direction, desired) / _norm(
                full_direction, 1, eps
            )
            directional_fd_cancellation = (
                1.0 - _norm(main_integral_direction, 1)
                / (_norm(main_direction, 1) + _norm(explicit_direction, 1)).clamp_min(eps)
            ).clamp(0.0, 1.0)
            directional_fd_normalized_sensitivity = _norm(
                full_direction / actuator_wrench_scale.abs().clamp_min(eps), 1
            )

    nonsmooth = (
        torch.zeros(integral.shape[0], dtype=torch.bool, device=integral.device)
        if nonsmooth_mask is None
        else nonsmooth_mask.to(device=integral.device, dtype=torch.bool)
    )
    central = forward = backward = torch.full_like(jac_full, float("nan"))
    fd_central = fd_forward = fd_backward = torch.full_like(largest, float("nan"))
    if validate_finite_difference:
        central = batched_independent_jacobian_finite_difference(
            full_wrench_fn, integral, epsilon=config.finite_difference_epsilon, direction="central"
        )
        forward = batched_independent_jacobian_finite_difference(
            full_wrench_fn, integral, epsilon=config.finite_difference_epsilon, direction="forward"
        )
        backward = batched_independent_jacobian_finite_difference(
            full_wrench_fn, integral, epsilon=config.finite_difference_epsilon, direction="backward"
        )
        fd_central = _relative_error(jac_full, central, eps)
        fd_forward = _relative_error(jac_full, forward, eps)
        fd_backward = _relative_error(jac_full, backward, eps)

    batch = integral.shape[0]
    result: dict[str, object] = {
        "scenario_uid": list(scenario_uid) if scenario_uid is not None else [str(i) for i in range(batch)],
        "integral_x": integral[:, 0].detach().cpu().numpy(),
        "integral_y": integral[:, 1].detach().cpu().numpy(),
        "integral_z": integral[:, 2].detach().cpu().numpy(),
        "main_jacobian_norm": norm_main.cpu().numpy(),
        "explicit_jacobian_norm": norm_explicit.cpu().numpy(),
        "main_plus_integral_jacobian_norm": norm_main_integral.cpu().numpy(),
        "total_jacobian_norm": norm_total.cpu().numpy(),
        "damping_integral_dependence_norm": _norm(jac_damping_dependence, (1, 2)).cpu().numpy(),
        "cancellation_ratio": cancellation_ratio.cpu().numpy(),
        "cancellation_fraction": cancellation_fraction.cpu().numpy(),
        "largest_singular_value": largest.cpu().numpy(),
        "smallest_singular_value": smallest.cpu().numpy(),
        "condition_number": condition.cpu().numpy(),
        "jacobian_rank": rank.cpu().numpy(),
        "collective_sensitivity": _norm(jac_full[:, 0], 1).cpu().numpy(),
        "torque_sensitivity": _norm(jac_full[:, 1:], (1, 2)).cpu().numpy(),
        "explicit_collective_sensitivity": _norm(jac_explicit[:, 0], 1).cpu().numpy(),
        "explicit_torque_sensitivity": _norm(jac_explicit[:, 1:], (1, 2)).cpu().numpy(),
        "normalized_total_sensitivity": _norm(normalized_jac, (1, 2)).cpu().numpy(),
        "actual_update_direction_sensitivity": direction_sensitivity.cpu().numpy(),
        "actual_update_wrench_cosine": direction_cosine.cpu().numpy(),
        "actual_update_fd_direction_sensitivity": directional_fd_sensitivity.cpu().numpy(),
        "actual_update_fd_wrench_cosine": directional_fd_cosine.cpu().numpy(),
        "actual_update_fd_cancellation_fraction": directional_fd_cancellation.cpu().numpy(),
        "actual_update_fd_normalized_sensitivity": directional_fd_normalized_sensitivity.cpu().numpy(),
        "dominant_wrench_alignment_magnitude": dominant_alignment_magnitude.cpu().numpy(),
        "finite_difference_central_relative_error": fd_central.cpu().numpy(),
        "finite_difference_forward_relative_error": fd_forward.cpu().numpy(),
        "finite_difference_backward_relative_error": fd_backward.cpu().numpy(),
        "finite_difference_selected_relative_error": torch.where(
            nonsmooth, torch.minimum(fd_forward, fd_backward), fd_central
        ).cpu().numpy(),
        "jacobian_validation_pass": torch.where(
            nonsmooth, torch.minimum(fd_forward, fd_backward), fd_central
        ).lt(1e-3).cpu().numpy(),
        "nonsmooth": nonsmooth.cpu().numpy(),
    }
    result["effective_actual_update_wrench_cosine"] = torch.where(
        nonsmooth, directional_fd_cosine, direction_cosine
    ).cpu().numpy()
    result["effective_cancellation_fraction"] = torch.where(
        nonsmooth, directional_fd_cancellation, cancellation_fraction
    ).cpu().numpy()
    result["effective_normalized_sensitivity"] = torch.where(
        nonsmooth, directional_fd_normalized_sensitivity, _norm(normalized_jac, (1, 2))
    ).cpu().numpy()
    if desired_wrench_direction is not None and integral_update_direction is not None:
        for component, name in enumerate(("collective", "tau_x", "tau_y", "tau_z")):
            result[f"actual_update_fd_{name}"] = full_direction[:, component].cpu().numpy()
    for output_axis, output_name in enumerate(("collective", "tau_x", "tau_y", "tau_z")):
        for input_axis, input_name in enumerate(("ix", "iy", "iz")):
            result[f"d_{output_name}_d_{input_name}_main"] = jac_main[:, output_axis, input_axis].cpu().numpy()
            result[f"d_{output_name}_d_{input_name}_explicit"] = jac_explicit[:, output_axis, input_axis].cpu().numpy()
            result[f"d_{output_name}_d_{input_name}_total"] = jac_full[:, output_axis, input_axis].cpu().numpy()
            result[f"d_{output_name}_d_{input_name}_fd_forward"] = forward[:, output_axis, input_axis].cpu().numpy()
            result[f"d_{output_name}_d_{input_name}_fd_backward"] = backward[:, output_axis, input_axis].cpu().numpy()
    if metadata:
        for key, values in metadata.items():
            if isinstance(values, Tensor):
                values = values.detach().cpu().numpy()
            array = np.asarray(values)
            if array.shape[0] != batch:
                raise ValueError(f"metadata {key!r} has length {array.shape[0]}, expected {batch}")
            result[key] = array
    return pd.DataFrame(result)


def classify_integral_mechanism(
    frame: pd.DataFrame,
    *,
    config: IntegralSensitivityConfig = IntegralSensitivityConfig(),
) -> pd.Series:
    labels = []
    for _, row in frame.iterrows():
        clamped = bool(row.get("clamped", False))
        headroom = float(row.get("actuator_headroom_fraction", float("nan")))
        sensitivity = float(row.get("effective_normalized_sensitivity", float("nan")))
        direction = float(row.get("effective_actual_update_wrench_cosine", float("nan")))
        cancellation = float(row.get("effective_cancellation_fraction", float("nan")))
        if bool(row.get("analytically_infeasible", False)):
            label = "analytically_infeasible"
        elif clamped and np.isfinite(headroom) and headroom <= 0:
            label = "clamped_no_actuator_headroom"
        elif np.isfinite(sensitivity) and sensitivity < config.weak_sensitivity_relative_threshold:
            label = "weak_integral_to_wrench_sensitivity"
        elif np.isfinite(cancellation) and cancellation >= config.cancellation_threshold:
            label = "main_explicit_cancellation"
        elif np.isfinite(direction) and direction < config.wrong_direction_cosine_threshold:
            label = "wrong_instantaneous_wrench_projection"
        elif clamped:
            label = "clamped_with_headroom"
        else:
            label = "unclassified_or_healthy"
        labels.append(label)
    return pd.Series(labels, index=frame.index, name="integral_mechanism")
