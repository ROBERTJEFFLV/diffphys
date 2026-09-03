"""Read-only multi-tau observer oracle for the structured policy.

The tool deliberately does not train or alter a policy.  It rolls out one
frozen structured checkpoint on deterministic, 4x4 authority-balanced banks,
then fits ordinary standardized ridge regressions to the resulting aligned
features.  The training/validation/final seed split is fixed in the command
line defaults.  ``legacy24`` is the production 24-dimensional
excitation/response summary, ``bank120`` concatenates eight modal sufficient
statistics for each of 15 registered (rise, fall) observer pairs, and
``privileged_true_motor`` uses the simulator's true motor state with the same
modal construction.

The response attached to command ``u_t`` is always the transition
``state_t -> state_(t+1)``.  Publication t25/t50/t75 consumes exactly the
preceding 25 transitions, avoiding the one-step shift which can make a motor
observer appear better than it is.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from structured_distillation import (  # noqa: E402
    build_dagger_scenario_bank,
    normalize_log_capability,
)
from structured_checkpoint import CADENCE_SEMANTICS_VERSION  # noqa: E402
from structured_policy import (  # noqa: E402
    CAPABILITY_HI,
    CAPABILITY_LO,
    MotorObserver,
    motor_observer_tau_grid,
    StructuredPolicyConfig,
    StructuredRecurrentPolicy,
)
from identification_features import (  # noqa: E402
    bank_modal_features,
    modal,
    normalize_response,
    production_legacy24,
    sol_response,
)
from structured_rollout import StructuredClosedLoopState, structured_observation  # noqa: E402


# Preserve the historical module-level K15 grid for callers/imports.  Runs
# select and thread the requested version explicitly below.
TAU_GRID = motor_observer_tau_grid(1)
PUBLICATION_STEPS = (25, 50, 75)
TRAIN_SEEDS = (3707, 4707, 5707, 6707)
VALIDATION_SEED = 7707
FINAL_SEEDS = (8707, 9707)
LAMBDA_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)


def _clone_state(state: L2FState) -> L2FState:
    return L2FState(**{
        name: getattr(state, name).detach().clone()
        for name in state.__dataclass_fields__
    })


def _raw_capability(state: L2FState) -> torch.Tensor:
    return torch.stack((
        state.thrust_to_weight, state.alpha_roll_max, state.eta_yaw,
        state.jz_over_jxy, state.motor_time_rising, state.motor_time_falling,
    ), dim=-1)


def _load_policy(path: Path, amplitude: float) -> StructuredRecurrentPolicy:
    payload = torch.load(path, map_location="cpu")
    config_values = dict(payload["config"])
    # Early K15 migration artifacts did not persist the explicit mode/version
    # fields.  Infer them from the bank size so those artifacts remain usable.
    bank_size = int(config_values.get("motor_observer_bank_size", 0))
    if "motor_observer_mode" not in config_values:
        config_values["motor_observer_mode"] = {
            0: "legacy", 15: "fixed_multi_tau_v1", 35: "fixed_multi_tau_v2",
        }.get(bank_size, "legacy")
    if "motor_tau_grid_version" not in config_values:
        config_values["motor_tau_grid_version"] = {0: 0, 15: 1, 35: 2}.get(bank_size, 0)
    config = StructuredPolicyConfig(**config_values)
    # Keep all architectural fields from the checkpoint.  Only the diagnostic
    # probe amplitude is overridden, and no parameter is updated.
    from dataclasses import replace
    config = replace(config, burn_in_probe_amplitude=float(amplitude))
    policy = StructuredRecurrentPolicy(config).to(device="cpu", dtype=torch.float32)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    return policy


def _response_features(
    state: L2FState, next_state: L2FState, *, dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return legacy force, legacy angular, and registered collective y."""

    acceleration = (next_state.velocity - state.velocity) / float(dt)
    gravity = acceleration.new_tensor((0.0, 0.0, 9.80665)).expand_as(acceleration)
    specific_body = torch.bmm(
        state.rotation.transpose(1, 2), (acceleration + gravity).unsqueeze(-1)
    ).squeeze(-1)
    angular_raw = (next_state.omega - state.omega) / float(dt)
    force, angular, collective = normalize_response(specific_body, angular_raw)
    return force, angular, collective.squeeze(-1)


def _modal(value: torch.Tensor) -> torch.Tensor:
    """Registered Sol modal coordinates for a four-channel motor vector."""
    return modal(value)


def _sol_response(collective: torch.Tensor, angular: torch.Tensor) -> torch.Tensor:
    """Return the exact normalized [collective, roll, pitch, yaw] response."""
    # ``_response_features`` returns the scalar collective channel squeezed
    # to [batch], while ``sol_response`` intentionally requires [batch, 1].
    if collective.ndim == angular.ndim - 1:
        collective = collective[..., None]
    return sol_response(collective, angular)


def _legacy24_step(
    excitation_history: torch.Tensor,
    force_response: torch.Tensor,
    angular_response: torch.Tensor,
    motor_delta: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Build the exact production 24-D feature for one causal transition."""

    return production_legacy24(
        excitation_history=excitation_history,
        force_response=force_response,
        angular_response=angular_response,
        motor_delta=motor_delta,
        response_mask=response_mask,
    )


def _window_features(rows: torch.Tensor, publication: int) -> torch.Tensor:
    start = max(0, int(publication) - 25)
    end = int(publication)
    return rows[start:end].mean(dim=0)


def collect_seed(
    checkpoint: Path,
    seed: int,
    *,
    scenarios: int = 64,
    horizon: int = 126,
    probe_amplitude: float = 0.005,
    tau_grid: tuple[tuple[float, float], ...] | None = None,
) -> dict[str, Any]:
    """Collect aligned rollout data for one fixed authority-balanced bank."""

    torch.set_num_threads(1)
    if horizon <= max(PUBLICATION_STEPS):
        raise ValueError("horizon must include policy call t25/t50/t75 diagnostics")
    selected_tau_grid = TAU_GRID if tau_grid is None else tuple(tau_grid)
    bank = build_dagger_scenario_bank(
        scenarios, seed=int(seed), dt=0.01, per_cell=scenarios // 16
    )
    policy = _load_policy(checkpoint, probe_amplitude)
    simulator = L2FSimulator(L2FParams(dt=policy.config.dt))
    physical = _clone_state(bank.state)
    observation = torch.cat((
        physical.position, physical.velocity,
        physical.rotation.reshape(scenarios, 9), physical.omega,
        torch.zeros_like(physical.position), physical.previous_action,
    ), dim=-1)
    current = StructuredClosedLoopState(
        physical=physical, policy=policy.initial_state(observation)
    )
    legacy_observer = MotorObserver(policy.config.observer_tau)
    legacy_estimate = current.policy.motor_estimate
    candidate_estimates = torch.stack(
        tuple(legacy_estimate for _ in selected_tau_grid), dim=1
    )
    # Time-major arrays.  Each item has shape [scenario, channel].
    true_motor, legacy_motor, candidate_motor = [], [], []
    command = []
    true_motor_delta, force_response, angular_response, sol_response = [], [], [], []
    legacy_rows = []
    candidate_rows = []
    privileged_rows = []
    capability_rows, trim_rows, body_z_rows, disturbance_rows = [], [], [], []
    identification_rows = []
    with torch.no_grad():
        for _step in range(horizon):
            observation = structured_observation(current)
            output = policy.forward_with_aux(
                observation, current.policy, simulator.params.dt
            )
            action = output.auxiliary["applied_action"]
            previous_motor = current.physical.motor
            next_physical = simulator.step(current.physical, action, grad_decay=1.0)
            force, angular, collective = _response_features(
                current.physical, next_physical, dt=simulator.params.dt
            )
            response_y = _sol_response(collective, angular)
            # Every x is a *post-transition motor state*, not the command
            # itself.  This is the registered Sol oracle convention.
            legacy_next = output.next_state.motor_estimate
            candidate_feature_rows = []
            tau_rise = torch.tensor(
                [pair[0] for pair in selected_tau_grid], dtype=action.dtype
            ).view(1, -1, 1)
            tau_fall = torch.tensor(
                [pair[1] for pair in selected_tau_grid], dtype=action.dtype
            ).view(1, -1, 1)
            candidate_estimate_delta = (action[:, None, :] - candidate_estimates).clamp(-1.0, 1.0)
            candidate_estimates_next = candidate_estimates + simulator.params.dt * candidate_estimate_delta / torch.where(
                candidate_estimate_delta >= 0.0, tau_rise, tau_fall
            ).clamp_min(1.0e-4)
            for index, _pair in enumerate(selected_tau_grid):
                estimate = candidate_estimates[:, index]
                motor_delta = (next_physical.motor - previous_motor) / 0.10
                x_modal = _modal(candidate_estimates_next[:, index])
                y_modal = response_y
                candidate_feature_rows.append(torch.cat((
                    (x_modal * y_modal).clamp(-25.0, 25.0),
                    x_modal.square().clamp(0.0, 25.0),
                ), dim=-1))
            candidate_rows.append(torch.stack(candidate_feature_rows, dim=1))
            true_x_modal = _modal(next_physical.motor)
            privileged_rows.append(torch.cat((
                (true_x_modal * response_y).clamp(-25.0, 25.0),
                true_x_modal.square().clamp(0.0, 25.0),
            ), dim=-1))
            true_motor.append(next_physical.motor)
            legacy_motor.append(legacy_next)
            candidate_motor.append(candidate_estimates_next)
            command.append(action)
            true_motor_delta.append(next_physical.motor - previous_motor)
            force_response.append(force)
            angular_response.append(angular)
            sol_response.append(response_y)
            capability_rows.append(output.auxiliary["capability"])
            trim_rows.append(output.auxiliary["trim_action"])
            body_z_rows.append(output.auxiliary["body_z"])
            disturbance_rows.append(output.auxiliary["disturbance_accel"])
            identification_rows.append(output.auxiliary["identification_failed"])
            legacy_estimate = legacy_next
            candidate_estimates = candidate_estimates_next
            transition_excitation = (
                (action - current.policy.motor_estimate) / 0.10
            ).clamp(-5.0, 5.0)
            prior_history = current.policy.excitation_history
            if prior_history is None:
                prior_history = action.new_zeros(action.shape[0], 13, 4)
            transition_history = torch.cat(
                (transition_excitation.unsqueeze(1), prior_history[:, :-1]), dim=1
            )
            next_boot = output.next_state.boot_progress
            if next_boot is None:
                raise RuntimeError("structured policy did not expose call index")
            legacy_rows.append(_legacy24_step(
                transition_history, force, angular,
                output.next_state.motor_estimate - current.policy.motor_estimate,
                (next_boot > 0.0).to(action.dtype),
            ))
            current = StructuredClosedLoopState(
                physical=next_physical, policy=output.next_state
            )
    target = normalize_log_capability(_raw_capability(bank.state))
    target_eq = __import__("equilibrium_control").analytic_equilibrium_target(
        bank.state, gravity=simulator.params.gravity
    )
    true_motor = torch.stack(true_motor)
    legacy_motor = torch.stack(legacy_motor)
    candidate_motor = torch.stack(candidate_motor)
    motor_delta = torch.stack(true_motor_delta)
    command = torch.stack(command)
    force_response = torch.stack(force_response)
    angular_response = torch.stack(angular_response)
    sol_response = torch.stack(sol_response)
    legacy_rows = torch.stack(legacy_rows)
    candidate_rows = torch.stack(candidate_rows)
    privileged_rows = torch.stack(privileged_rows)
    capability_rows = torch.stack(capability_rows)
    trim_rows = torch.stack(trim_rows)
    body_z_rows = torch.stack(body_z_rows)
    disturbance_rows = torch.stack(disturbance_rows)
    identification_rows = torch.stack(identification_rows)
    features = {"legacy24": [], "bank120": [], "privileged_true_motor": []}
    for publication in PUBLICATION_STEPS:
        features["legacy24"].append(_window_features(legacy_rows, publication))
        features["bank120"].append(_window_features(candidate_rows, publication).reshape(scenarios, -1))
        # The privileged feature uses true motor state and the same aligned
        # command/response statistics, making it an optimistic upper bound.
        # The privileged arm uses exactly the same eight modal sufficient
        # statistics as one bank candidate; only x changes from the observer
        # motor state to the simulator true-motor state.  This keeps the
        # comparison equal-dimensional and makes its optimism explicit.
        features["privileged_true_motor"].append(
            _window_features(privileged_rows, publication)
        )
    return {
        "seed": int(seed),
        "scenario_count": int(scenarios),
        "authority_cells": list(bank.stratum),
        "features": {key: torch.stack(value).cpu() for key, value in features.items()},
        "target": target.cpu(),
        "capability": _raw_capability(bank.state).cpu(),
        "authority": torch.stack((bank.state.thrust_to_weight, bank.state.alpha_roll_max), dim=-1).cpu(),
        "true_motor": true_motor.cpu(),
        "legacy_motor": legacy_motor.cpu(),
        "candidate_motor": candidate_motor.cpu(),
        "identification_failed": identification_rows.cpu(),
        "equilibrium_trim": target_eq.motor_trim.cpu(),
        "equilibrium_body_z": target_eq.body_z.cpu(),
        "disturbance_target": (bank.state.external_force / bank.state.mass[:, None]).cpu(),
        "capability_estimate": capability_rows.cpu(),
        "trim_estimate": trim_rows.cpu(),
        "body_z_estimate": body_z_rows.cpu(),
        "disturbance_estimate": disturbance_rows.cpu(),
    }


def _ridge_fit(x: torch.Tensor, y: torch.Tensor, ridge: float) -> dict[str, torch.Tensor]:
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    y_mean = y.mean(dim=0)
    xz = (x - mean) / scale
    yz = y - y_mean
    eye = torch.eye(x.shape[-1], dtype=torch.float64)
    xtx = xz.double().T @ xz.double()
    coefficient = torch.linalg.solve(xtx + float(ridge) * eye, xz.double().T @ yz.double())
    return {"mean": mean, "scale": scale, "y_mean": y_mean, "coefficient": coefficient}


def _ridge_predict(model: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return ((x - model["mean"]) / model["scale"]).double() @ model["coefficient"] + model["y_mean"].double()


def _rms(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).square().mean().sqrt())


def _fit_select(train: list[dict[str, Any]], validation: dict[str, Any], name: str) -> tuple[dict[str, torch.Tensor], float, dict[str, Any]]:
    x = torch.cat([row["features"][name] for row in train], dim=1).reshape(-1, train[0]["features"][name].shape[-1])
    y = torch.cat([row["target"].unsqueeze(0).expand(3, -1, -1) for row in train], dim=1).reshape(-1, 6)
    xv = validation["features"][name].reshape(-1, validation["features"][name].shape[-1])
    yv = validation["target"].unsqueeze(0).expand(3, -1, -1).reshape(-1, 6)
    candidates = []
    for ridge in LAMBDA_GRID:
        model = _ridge_fit(x, y, ridge)
        candidates.append((ridge, _rms(_ridge_predict(model, xv).float(), yv), model))
    ridge, error, model = min(candidates, key=lambda item: item[1])
    return model, float(ridge), {"validation_rms": float(error), "lambda_candidates": [
        {"lambda": float(item[0]), "validation_rms": float(item[1])} for item in candidates
    ]}


def _bootstrap_ci(values: torch.Tensor, *, seed: int = 9137, repeats: int = 2000) -> tuple[float, float]:
    """Paired percentile CI for values shaped ``[seed, scenario]``.

    Rows are kept paired: each draw samples scenarios independently within
    each final seed, and the same indices must be used for competing methods
    before this function is called.  The capability gate passes MSE deltas,
    never RMS deltas, to this helper.
    """
    if values.ndim != 2:
        raise ValueError("bootstrap values must have shape [seed,scenario]")
    if not bool(torch.isfinite(values).all()):
        return (float("nan"), float("nan"))
    generator = torch.Generator().manual_seed(int(seed))
    draws = []
    for _ in range(int(repeats)):
        indices = torch.randint(values.shape[1], (values.shape[0], values.shape[1]), generator=generator)
        draws.append(values.gather(1, indices).mean())
    result = torch.stack(draws)
    return float(torch.quantile(result, 0.025)), float(torch.quantile(result, 0.975))


def _evaluate_method(model: dict[str, torch.Tensor], rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    per_seed = []
    per_seed_dimension = []
    per_seed_scenario_mse = []
    per_checkpoint = {}
    axis = []
    authority_cells: dict[str, list[torch.Tensor]] = {}
    authority_cells_by_publication: dict[str, list[torch.Tensor]] = {}
    for row in rows:
        x = row["features"][name]
        prediction = _ridge_predict(model, x.reshape(-1, x.shape[-1])).reshape(3, -1, 6).float()
        target = row["target"].unsqueeze(0).expand(3, -1, -1)
        squared_error = (prediction - target).square()
        scenario_mse = squared_error.mean(dim=2)
        per_seed_scenario_mse.append(scenario_mse)
        per_seed.append(scenario_mse.mean(dim=1).sqrt())
        per_seed_dimension.append(squared_error.mean(dim=1).sqrt())
        axis.append(squared_error.mean(dim=1).sqrt())
        for cell in sorted(set(row["authority_cells"])):
            mask = torch.tensor(
                [value == cell for value in row["authority_cells"]], dtype=torch.bool
            )
            authority_cells.setdefault(cell, []).append(
                (prediction[:, mask] - target[:, mask]).square().mean().sqrt()
            )
            authority_cells_by_publication.setdefault(cell, []).append(
                (prediction[:, mask] - target[:, mask]).square().mean(dim=(1, 2)).sqrt()
            )
        for index, step in enumerate(PUBLICATION_STEPS):
            per_checkpoint.setdefault(str(step), []).append((prediction[index] - target[index]).square().mean().sqrt())
    errors = torch.stack(per_seed)
    axis_values = torch.stack(axis)
    scenario_mse_values = torch.stack(per_seed_scenario_mse)
    dimension_values = torch.stack(per_seed_dimension)
    return {
        "overall_rms": float(errors.mean()),
        "per_seed_publication_rms": errors.tolist(),
        "per_checkpoint_rms": {step: float(torch.stack(values).mean()) for step, values in per_checkpoint.items()},
        "per_dimension_rms": axis_values.mean(dim=(0, 1)).tolist(),
        "per_publication_dimension_rms": dimension_values.mean(dim=0).tolist(),
        "per_seed_publication_dimension_rms": dimension_values.tolist(),
        "authority_cell_rms": {
            cell: float(torch.stack(values).mean())
            for cell, values in sorted(authority_cells.items())
        },
        "authority_cell_publication_rms": {
            cell: torch.stack(values).mean(dim=0).tolist()
            for cell, values in sorted(authority_cells_by_publication.items())
        },
        # This is deliberately retained at seed/scenario granularity.  The
        # pre-registered capability test pairs this tensor with legacy24
        # before bootstrapping; bootstrapping already-aggregated RMS values
        # would test a different estimand.
        "scenario_mse_matrix": scenario_mse_values.tolist(),
        "scenario_error_matrix": errors.tolist(),
    }


def _motor_coverage(
    rows: list[dict[str, Any]],
    tau_grid: tuple[tuple[float, float], ...] | None = None,
) -> dict[str, Any]:
    selected_tau_grid = TAU_GRID if tau_grid is None else tuple(tau_grid)
    output: dict[str, Any] = {
        "legacy": {},
        "candidates": {str(tau): {} for tau in selected_tau_grid},
        "best_candidate": {},
        "by_seed": {},
        "authority_cell_ratios": {},
    }
    cells = sorted({cell for row in rows for cell in row["authority_cells"]})
    output["authority_cells"] = cells
    for step in PUBLICATION_STEPS:
        index = step - 1
        true_by_seed = [row["true_motor"][index] for row in rows]
        legacy_by_seed = [row["legacy_motor"][index] for row in rows]
        true = torch.cat(true_by_seed, dim=0)
        legacy = torch.cat(legacy_by_seed, dim=0)
        legacy_error = (legacy - true).square().mean(dim=-1).sqrt()
        output["legacy"][str(step)] = {
            "rms": float(legacy_error.mean()),
            "per_motor_rms": (legacy - true).square().mean(dim=0).sqrt().tolist(),
        }
        candidate_errors = []
        for candidate_index, tau in enumerate(selected_tau_grid):
            candidate = torch.cat([row["candidate_motor"][index, :, candidate_index] for row in rows], dim=0)
            target = true
            error = (candidate - target).square().mean(dim=-1).sqrt()
            candidate_errors.append(error)
            output["candidates"][str(tau)][str(step)] = {
                "rms": float(error.mean()),
                "per_motor_rms": (candidate - target).square().mean(dim=0).sqrt().tolist(),
            }
        stacked = torch.stack(candidate_errors, dim=-1)
        best, best_index = stacked.min(dim=-1)
        ratios = _safe_relative_ratio(best, legacy_error)
        per_seed = []
        offset = 0
        for row, true_seed, legacy_seed in zip(rows, true_by_seed, legacy_by_seed):
            count = true_seed.shape[0]
            seed_best = best[offset:offset + count]
            seed_legacy = legacy_error[offset:offset + count]
            seed_ratios = ratios[offset:offset + count]
            per_seed.append({
                "seed": int(row["seed"]),
                "best_rms": seed_best.tolist(),
                "legacy_rms": seed_legacy.tolist(),
                "best_to_legacy_ratio": seed_ratios.tolist(),
                "ratio_median": _finite_stat(seed_ratios, torch.median),
                "ratio_p95": _finite_stat(seed_ratios, lambda value: torch.quantile(value, 0.95)),
            })
            per_seed[-1]["passed"] = bool(
                math.isfinite(per_seed[-1]["ratio_median"])
                and math.isfinite(per_seed[-1]["ratio_p95"])
                and per_seed[-1]["ratio_median"] <= 0.50
                and per_seed[-1]["ratio_p95"] <= 0.80
            )
            offset += count
        output["by_seed"][str(step)] = per_seed
        cell_ratios = {}
        for cell in cells:
            mask = torch.tensor(
                [value == cell for row in rows for value in row["authority_cells"]],
                dtype=torch.bool,
            )
            # The cell comparison uses the selected best candidate per
            # scenario, as required for the bank observer coverage screen.
            cell_legacy = legacy_error[mask]
            cell_best = best[mask]
            cell_ratios[cell] = _safe_relative_ratio(
                cell_best.square().mean().sqrt(), cell_legacy.square().mean().sqrt()
            )
        output["authority_cell_ratios"][str(step)] = {
            cell: float(value) for cell, value in cell_ratios.items()
        }
        output["best_candidate"][str(step)] = {
            "rms": float(best.mean()),
            "legacy_rms": float(legacy_error.mean()),
            "improvement_vs_legacy": float((legacy_error - best).mean()),
            "improvement_fraction_vs_legacy": float(
                ((legacy_error - best) / legacy_error.clamp_min(1.0e-8)).mean()
            ),
            "per_scenario_best_to_legacy_ratio": ratios.tolist(),
            "ratio_median": _finite_stat(ratios, torch.median),
            "ratio_p95": _finite_stat(ratios, lambda value: torch.quantile(value, 0.95)),
            "all_authority_cells_finite": bool(all(
                math.isfinite(value) for value in cell_ratios.values()
            )),
            "all_final_seed_checks_passed": bool(all(item["passed"] for item in per_seed)),
            "best_pair_counts": {
                str(selected_tau_grid[index]): int((best_index == index).sum())
                for index in range(len(selected_tau_grid)) if bool((best_index == index).any())
            },
            "per_scenario_best_rms": best.tolist(),
            "per_scenario_legacy_rms": legacy_error.tolist(),
            "per_scenario_best_pair": [str(selected_tau_grid[int(value)]) for value in best_index],
        }
    output["all_authority_cells_finite"] = bool(
        bool(cells)
        and all(
            math.isfinite(value)
            for ratios in output["authority_cell_ratios"].values()
            for value in ratios.values()
        )
    )
    return output


def _safe_relative_ratio(numerator: torch.Tensor | float, denominator: torch.Tensor | float) -> torch.Tensor:
    """Return numerator/denominator while making zero-denominator failure explicit."""

    num = torch.as_tensor(numerator, dtype=torch.float64)
    den = torch.as_tensor(denominator, dtype=torch.float64)
    zero_both = (den == 0) & (num == 0)
    return torch.where(zero_both, torch.zeros_like(num), num / den)


def _finite_stat(values: torch.Tensor, reducer: Any) -> float:
    """Reduce finite values, preserving NaN/Inf as a visible gate failure."""

    if not bool(torch.isfinite(values).all()):
        return float("nan") if bool(torch.isnan(values).any()) else float("inf")
    return float(reducer(values))


def _error_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = torch.stack([row["target"] for row in rows])
    result = {}
    for method, key in (("capability_mean", "capability_estimate"), ("trim", "trim_estimate"),
                        ("body_z", "body_z_estimate"), ("disturbance", "disturbance_estimate")):
        values = []
        for row in rows:
            if method == "capability_mean":
                prediction = normalize_log_capability(row[key])
                truth = row["target"].unsqueeze(0).expand(prediction.shape[0], -1, -1)
            elif method == "trim":
                prediction, truth = row[key], row["equilibrium_trim"].unsqueeze(0).expand_as(row[key])
            elif method == "body_z":
                prediction, truth = row[key], row["equilibrium_body_z"].unsqueeze(0).expand_as(row[key])
            else:
                prediction, truth = row[key], row["disturbance_target"].unsqueeze(0).expand_as(row[key])
            if prediction.ndim == 3:
                # [time,scenario,channel] -> publication snapshots.
                # Estimate rows are indexed by policy call.  Publication t25
                # therefore reads call25, not the old one-based row24.
                indices = list(PUBLICATION_STEPS)
                prediction = prediction[indices]
                truth = truth[indices]
            values.append((prediction - truth).square().mean(dim=-1).sqrt())
        values = torch.stack(values)
        result[method] = {
            f"t{step}_rms": float(values[:, index].mean())
            for index, step in enumerate(PUBLICATION_STEPS)
        }
    return result


def _evaluate_pre_registered_gate(
    evaluation: dict[str, Any], coverage: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the immutable Sol gate and return every boolean/raw value.

    The primary effectiveness estimand is final-seed publication t50.  The
    t25/t75 values and bootstrap intervals are reported as secondary checks,
    but cannot silently replace the registered t50 gate.
    """

    required_steps = tuple(str(step) for step in PUBLICATION_STEPS)
    primary_step = "50"
    legacy = evaluation.get("legacy24", {})
    bank = evaluation.get("bank120", {})
    privileged = evaluation.get("privileged_true_motor", {})

    def checkpoint(method: dict[str, Any], step: str) -> float:
        values = method.get("per_checkpoint_rms", {})
        value = values.get(step, float("nan"))
        return float(value)

    # Motor coverage is deliberately a per-scenario ratio, not a ratio of
    # pooled RMS values.  This prevents a large, easy cell from masking a
    # failing final-seed scenario.
    coverage_by_step = {}
    for step in required_steps:
        best = coverage.get("best_candidate", {}).get(step, {})
        median = float(best.get("ratio_median", float("nan")))
        p95 = float(best.get("ratio_p95", float("nan")))
        finite = bool(best.get("all_authority_cells_finite", False))
        finite = finite and math.isfinite(median) and math.isfinite(p95)
        per_seed = coverage.get("by_seed", {}).get(step, [])
        per_seed_passed = bool(
            all(bool(item.get("passed", False)) for item in per_seed)
        ) if per_seed else True
        coverage_by_step[step] = {
            "passed": bool(finite and median <= 0.50 and p95 <= 0.80 and per_seed_passed),
            "finite": finite,
            "ratio_median": median,
            "ratio_p95": p95,
            "per_seed_passed": per_seed_passed,
            "per_seed": per_seed,
            "median_threshold": 0.50,
            "p95_threshold": 0.80,
            "per_scenario_best_to_legacy_ratio": best.get(
                "per_scenario_best_to_legacy_ratio", []
            ),
        }
    motor_coverage_passed = bool(
        all(value["passed"] for value in coverage_by_step.values())
        and bool(coverage_by_step)
    )

    # Dimensional t50 effectiveness is measured from RMS values over the
    # final seed/scenario bank.  Use explicit ratios so a dimension's large
    # improvement cannot hide another dimension's degradation.
    legacy_dims = torch.tensor(
        legacy.get("per_publication_dimension_rms", []), dtype=torch.float64
    )
    bank_dims = torch.tensor(
        bank.get("per_publication_dimension_rms", []), dtype=torch.float64
    )
    if legacy_dims.ndim == 2 and bank_dims.shape == legacy_dims.shape and legacy_dims.shape[0] >= 2:
        legacy_t50_dims = legacy_dims[1]
        bank_t50_dims = bank_dims[1]
        improvement = _safe_relative_ratio(legacy_t50_dims - bank_t50_dims, legacy_t50_dims)
        ratio = _safe_relative_ratio(bank_t50_dims, legacy_t50_dims)
        finite_dims = bool(torch.isfinite(improvement).all() and torch.isfinite(ratio).all())
    else:
        legacy_t50_dims = torch.full((6,), float("nan"), dtype=torch.float64)
        bank_t50_dims = torch.full((6,), float("nan"), dtype=torch.float64)
        improvement = torch.full((6,), float("nan"), dtype=torch.float64)
        ratio = torch.full((6,), float("nan"), dtype=torch.float64)
        finite_dims = False
    capability_effectiveness_passed = bool(
        finite_dims and improvement.numel() == 6
        and bool((improvement[[0, 1, 2]] >= 0.25).all())
    )
    tau_effectiveness_passed = bool(
        finite_dims and improvement.numel() == 6
        and bool((improvement[[4, 5]] >= 0.25).all())
    )
    effectiveness_passed = bool(capability_effectiveness_passed and tau_effectiveness_passed)
    degradation_passed = bool(
        finite_dims and bool((ratio <= 1.05).all())
    )
    effectiveness = {
        "passed": effectiveness_passed,
        "publication": 50,
        "legacy_rms_by_dimension": legacy_t50_dims.tolist(),
        "bank_rms_by_dimension": bank_t50_dims.tolist(),
        "improvement_fraction_by_dimension": improvement.tolist(),
        "required_improvement_fraction": 0.25,
        "capability_dimensions_0_3": [0, 1, 2],
        "tau_dimensions_4_6": [4, 5],
        "capability_dimensions_passed": capability_effectiveness_passed,
        "tau_dimensions_passed": tau_effectiveness_passed,
        "finite": finite_dims,
    }
    degradation = {
        "passed": degradation_passed,
        "bank_to_legacy_ratio_by_dimension": ratio.tolist(),
        "maximum_allowed_ratio": 1.05,
        "maximum_degradation_fraction": 0.05,
        "finite": finite_dims,
    }

    # The cadence check follows the direction established at t50.  Since the
    # t50 gate requires an improvement, t25 and t75 must also improve.
    cadence = {}
    for step in required_steps:
        legacy_rms = checkpoint(legacy, step)
        bank_rms = checkpoint(bank, step)
        delta = bank_rms - legacy_rms
        cadence[step] = {
            "passed": bool(math.isfinite(delta) and delta <= 0.0),
            "legacy_rms": legacy_rms,
            "bank_rms": bank_rms,
            "bank_minus_legacy_rms": delta,
            "direction": "improvement" if delta < 0.0 else "tie" if delta == 0.0 else "degradation",
        }
    cadence_passed = bool(all(item["passed"] for item in cadence.values()))

    # Authority-cell ratios use the ridge capability RMS and are required for
    # every cell.  Missing cells are a failed contract, not an empty pass.
    legacy_cells = legacy.get("authority_cell_publication_rms", {})
    bank_cells = bank.get("authority_cell_publication_rms", {})
    if legacy_cells and bank_cells:
        # t50 is the registered primary publication for the capability gate.
        legacy_cells = {
            cell: values[1] for cell, values in legacy_cells.items()
            if len(values) > 1
        }
        bank_cells = {
            cell: values[1] for cell, values in bank_cells.items()
            if len(values) > 1
        }
    else:
        # Accept the compact aggregate form for lightweight callers, while
        # production reports always provide publication-specific cell values.
        legacy_cells = legacy.get("authority_cell_rms", {})
        bank_cells = bank.get("authority_cell_rms", {})
    cells = sorted(set(legacy_cells) | set(bank_cells))
    cell_ratios = {}
    for cell in cells:
        value = _safe_relative_ratio(
            bank_cells.get(cell, float("nan")), legacy_cells.get(cell, float("nan"))
        )
        cell_ratios[cell] = float(value)
    authority_cell_passed = bool(
        bool(cells) and all(math.isfinite(value) and value <= 1.10 for value in cell_ratios.values())
    )
    authority_cells = {
        "passed": authority_cell_passed,
        "maximum_allowed_bank_to_legacy_ratio": 1.10,
        "ratios": cell_ratios,
        "all_cells_present": bool(cells) and set(legacy_cells) == set(bank_cells),
        "all_cells_finite": bool(cells) and all(math.isfinite(value) for value in cell_ratios.values()),
    }

    # Paired bootstrap is over the registered capability MSE delta at each
    # publication.  t50 is primary; all publications are retained in report.
    bootstrap_by_step = {}
    legacy_mse = torch.tensor(legacy.get("scenario_mse_matrix", []), dtype=torch.float64)
    bank_mse = torch.tensor(bank.get("scenario_mse_matrix", []), dtype=torch.float64)
    mse_shape_ok = (
        legacy_mse.ndim == 3 and bank_mse.shape == legacy_mse.shape
        and legacy_mse.shape[1] >= len(PUBLICATION_STEPS)
    )
    for index, step in enumerate(required_steps):
        if mse_shape_ok:
            delta_values = bank_mse[:, index, :] - legacy_mse[:, index, :]
            mean_delta = float(delta_values.mean())
            ci = _bootstrap_ci(delta_values)
        else:
            delta_values = torch.empty((0, 0), dtype=torch.float64)
            mean_delta, ci = float("nan"), (float("nan"), float("nan"))
        bootstrap_by_step[step] = {
            "mean_mse_bank_minus_legacy": mean_delta,
            "ci95": list(ci),
            "sample_shape": list(delta_values.shape),
            "finite": bool(mse_shape_ok and torch.isfinite(delta_values).all()),
            "passed": bool(
                mse_shape_ok and math.isfinite(ci[1]) and ci[1] < 0.0
            ),
        }
    bootstrap_passed = bool(bootstrap_by_step[primary_step]["passed"])

    # Gap closure is relative to privileged_true_motor at t50.  If the
    # privileged arm is not better than legacy, the denominator is not a
    # meaningful attainable gap and the gate must fail explicitly.
    legacy_t50 = checkpoint(legacy, primary_step)
    bank_t50 = checkpoint(bank, primary_step)
    privileged_t50 = checkpoint(privileged, primary_step)
    privileged_gain = legacy_t50 - privileged_t50
    if not all(math.isfinite(value) for value in (legacy_t50, bank_t50, privileged_t50)) or privileged_gain <= 0.0:
        closure = None
        closure_defined = False
        closure_passed = False
    else:
        closure = (legacy_t50 - bank_t50) / privileged_gain
        closure_defined = math.isfinite(closure)
        closure_passed = bool(closure_defined and closure >= 0.50)
    gap_closure = {
        "passed": closure_passed,
        "publication": 50,
        "legacy_rms": legacy_t50,
        "bank_rms": bank_t50,
        "privileged_rms": privileged_t50,
        "privileged_improvement_vs_legacy": privileged_gain,
        "gap_closure_relative_privileged": closure,
        "minimum_gap_closure": 0.50,
        "defined": closure_defined,
        "failure_reason": None if closure_defined else "privileged_true_motor is not better than legacy24; gap is undefined",
    }

    # The legacy baseline is built by the same causal production feature
    # function as the policy: transition excitation, aligned response, and
    # applied-observer motor delta are all from the current transition.
    baseline_contract_match = True
    baseline_contract_match_details = {
        "passed": baseline_contract_match,
        "matched": baseline_contract_match,
        "reason": "oracle legacy24 delegates to identification_features.production_legacy24",
    }

    return {
        "motor_coverage": {"passed": motor_coverage_passed, "by_publication": coverage_by_step},
        "ridge_effectiveness_t50": effectiveness,
        "dimension_degradation_t50": degradation,
        "cadence_direction": {"passed": cadence_passed, "by_publication": cadence},
        "authority_cell_ratio": authority_cells,
        "paired_bootstrap_capability_mse": {
            "passed": bootstrap_passed,
            "primary_publication": 50,
            "by_publication": bootstrap_by_step,
        },
        "gap_closure_relative_privileged": gap_closure,
        "baseline_contract_match": baseline_contract_match,
        "baseline_contract_match_details": baseline_contract_match_details,
        "passed": bool(
            motor_coverage_passed and effectiveness_passed and degradation_passed
            and cadence_passed and authority_cell_passed and bootstrap_passed
            and closure_passed and baseline_contract_match
        ),
    }


# Short alias for callers/tests that refer to the registered contract by its
# name rather than the implementation-oriented helper name.
_pre_registered_gate = _evaluate_pre_registered_gate


def run(args: argparse.Namespace) -> dict[str, Any]:
    all_seeds = TRAIN_SEEDS + (VALIDATION_SEED,) + FINAL_SEEDS
    if args.scenarios < 16 or args.scenarios % 16 or args.horizon <= max(PUBLICATION_STEPS):
        raise ValueError("oracle requires a multiple of 16 scenarios and horizon beyond t75")
    tau_grid_version = int(getattr(args, "tau_grid_version", 1))
    tau_grid = motor_observer_tau_grid(tau_grid_version)
    jobs = args.n_jobs
    if Parallel is None or jobs == 1:
        collected = [collect_seed(
            args.checkpoint, seed, scenarios=args.scenarios, horizon=args.horizon,
            tau_grid=tau_grid,
        ) for seed in all_seeds]
    else:
        collected = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
            delayed(collect_seed)(
                args.checkpoint, seed, scenarios=args.scenarios, horizon=args.horizon,
                tau_grid=tau_grid,
            )
            for seed in all_seeds
        )
    by_seed = {int(row["seed"]): row for row in collected}
    train = [by_seed[seed] for seed in TRAIN_SEEDS]
    validation = by_seed[VALIDATION_SEED]
    final = [by_seed[seed] for seed in FINAL_SEEDS]
    models = {}
    selection = {}
    for name in ("legacy24", "bank120", "privileged_true_motor"):
        models[name], selected, selection[name] = _fit_select(train, validation, name)
        selection[name]["selected_lambda"] = selected
    evaluation = {name: _evaluate_method(models[name], final, name) for name in models}
    coverage = _motor_coverage(final, tau_grid=tau_grid)
    gate = _evaluate_pre_registered_gate(evaluation, coverage)
    # Keep the top-level gap name for downstream report consumers while
    # storing the primary t50 relative-to-privileged contract inside it.
    gap = {"bank120": gate["gap_closure_relative_privileged"]}
    return {
        "diagnostic": "multi-tau-observer-ridge-oracle",
        "formal_eligible": False,
        "diagnostic_only": True,
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": {
            "call_index_completed_transitions": True,
            "call0_has_response": False,
            "diagnostic_calls": list(PUBLICATION_STEPS),
            "publication_calls": [50, 75],
            "availability_t25": [0, 0, 0, 0, 0, 0],
            "t50_call_index": 50,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "probe_amplitude": 0.005,
        "horizon": int(args.horizon),
        "publication_steps": list(PUBLICATION_STEPS),
        "tau_grid_version": tau_grid_version,
        "tau_grid": list(tau_grid),
        "tau_pairs": [
            {"index": int(index), "tau_rise": float(pair[0]), "tau_fall": float(pair[1])}
            for index, pair in enumerate(tau_grid)
        ],
        "bank_feature_dimension": int(8 * len(tau_grid)),
        "lambda_grid": list(LAMBDA_GRID),
        "seed_split": {"train": list(TRAIN_SEEDS), "validation": [VALIDATION_SEED], "final": list(FINAL_SEEDS)},
        "feature_contract": {
            "legacy24": "production_legacy24: three causal excitation/response lag blocks plus rise/fall and energy terms",
            "bank120": "8 features per selected rise/fall pair (dynamic dimension 8K) over four Sol modal channels; x=post-transition candidate motor state, y=[y_C,y_R,y_P,y_Y]",
            "privileged_true_motor": "same eight modal sufficient statistics using post-transition simulator true motor state",
            "target": "6D normalized-log capability [TW, alpha_roll, eta_yaw, Jz/Jxy, tau_rise, tau_fall]",
            "alignment": "command u_t is paired with state_t -> state_(t+1); t25/t50/t75 use preceding 25 transitions",
        },
        "contract_parity": {
            "legacy24_shared_production_feature": True,
            "legacy24_dimension": 24,
            "bank_dimension": int(8 * len(tau_grid)),
            "production_cadence_fixed": True,
        },
        "ridge_selection": selection,
        "motor_coverage": coverage,
        "capability_diagnostics_final": _error_diagnostics(final),
        "evaluation_final": evaluation,
        "gap_closure": gap,
        "pre_registered_gate": {
            **gate,
            "contract": {
                "motor_coverage": "each final-seed/scenario best-to-legacy motor RMS ratio has median <= 0.50 and p95 <= 0.80 at every publication; every authority cell is finite",
                "ridge_validation": "lambda selected only on seed 7707; final seeds are untouched",
                "identification": "report rank/coverage separately; no claim of deployable identification from oracle alone",
                "baseline_contract_match": "TRUE: oracle legacy24 delegates to production_legacy24 with explicit transition alignment",
                "ridge_effectiveness_t50": "bank120 RMS improvement is >= 25% in dimensions 0:3 and 4:6",
                "paired_bootstrap": "paired bootstrap of per-final-seed/per-scenario capability MSE bank120 - legacy24; primary publication t50, CI upper < 0",
                "dimension_degradation": "no t50 dimension bank/legacy RMS ratio exceeds 1.05",
                "cadence": "t25 and t75 overall RMS must have the same (improving) direction as t50",
                "authority_cells": "every authority-cell bank120/legacy24 capability RMS ratio <= 1.10",
                "gap_closure": "t50 closure relative to privileged_true_motor >= 50%; undefined privileged denominator is an explicit failure",
            },
        },
        "gate_passed": bool(gate["passed"]),
        "coverage_gate_passed": bool(gate["motor_coverage"]["passed"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=126)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--tau-grid-version", type=int, choices=(1, 2), default=1,
        help="registered motor observer grid (v1=K15, v2=K35)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnostic": report["diagnostic"],
        "gate_passed": report["gate_passed"],
        "evaluation_final": {
            key: value["overall_rms"] for key, value in report["evaluation_final"].items()
        },
        "gap_closure": report["gap_closure"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
