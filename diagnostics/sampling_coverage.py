from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from env_l2f import L2FParams, L2FState, sample_physical_broad_episode_dynamics


@dataclass(frozen=True)
class CoverageData:
    external_force: np.ndarray
    thrust_to_weight: np.ndarray
    roll_authority: np.ndarray
    yaw_authority: np.ndarray
    rise_time: np.ndarray
    fall_time: np.ndarray
    required_trim_ratio: np.ndarray
    required_tilt_deg: np.ndarray
    analytically_feasible: np.ndarray

    def mapping(self) -> Mapping[str, np.ndarray]:
        return {
            "external_force": self.external_force,
            "thrust_to_weight": self.thrust_to_weight,
            "roll_authority": self.roll_authority,
            "yaw_authority": self.yaw_authority,
            "rise_time": self.rise_time,
            "fall_time": self.fall_time,
            "required_trim_ratio": self.required_trim_ratio,
            "required_tilt_deg": self.required_tilt_deg,
        }


def _derived(
    mass: torch.Tensor,
    external_force: torch.Tensor,
    thrust_to_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gravity = mass * 9.80665
    vector = -external_force.clone()
    vector[:, 2] += gravity
    required = torch.linalg.vector_norm(vector, dim=-1)
    trim = required / (thrust_to_weight * gravity).clamp_min(1.0e-12)
    tilt = torch.rad2deg(torch.atan2(torch.linalg.vector_norm(vector[:, :2], dim=-1), vector[:, 2].clamp_min(1.0e-12)))
    feasible = trim <= 1.0 + 1.0e-6
    return trim, tilt, feasible


def sample_physical_coverage(
    sample_count: int,
    *,
    seed: int,
    chunk_size: int = 65_536,
) -> CoverageData:
    if sample_count <= 0 or chunk_size <= 0:
        raise ValueError("sample count and chunk size must be positive")
    torch.manual_seed(seed)
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in CoverageData.__dataclass_fields__}
    remaining = sample_count
    params = L2FParams()
    while remaining:
        count = min(remaining, chunk_size)
        dynamics = sample_physical_broad_episode_dynamics(
            params, count, device=torch.device("cpu"), dtype=torch.float32
        )
        trim, tilt, feasible = _derived(dynamics.mass, dynamics.external_force, dynamics.thrust_to_weight)
        values = {
            "external_force": torch.linalg.vector_norm(dynamics.external_force, dim=-1),
            "thrust_to_weight": dynamics.thrust_to_weight,
            "roll_authority": dynamics.alpha_roll_max,
            "yaw_authority": dynamics.alpha_yaw_max,
            "rise_time": dynamics.motor_time_rising,
            "fall_time": dynamics.motor_time_falling,
            "required_trim_ratio": trim,
            "required_tilt_deg": tilt,
            "analytically_feasible": feasible,
        }
        for name, value in values.items():
            chunks[name].append(value.numpy().copy())
        remaining -= count
    return CoverageData(**{name: np.concatenate(parts) for name, parts in chunks.items()})


def state_coverage(state: L2FState) -> CoverageData:
    trim, tilt, feasible = _derived(state.mass, state.external_force, state.thrust_to_weight)
    cpu = lambda value: value.detach().cpu().numpy().copy()
    return CoverageData(
        external_force=cpu(torch.linalg.vector_norm(state.external_force, dim=-1)),
        thrust_to_weight=cpu(state.thrust_to_weight),
        roll_authority=cpu(state.alpha_roll_max),
        yaw_authority=cpu(state.alpha_yaw_max),
        rise_time=cpu(state.motor_time_rising),
        fall_time=cpu(state.motor_time_falling),
        required_trim_ratio=cpu(trim),
        required_tilt_deg=cpu(tilt),
        analytically_feasible=cpu(feasible),
    )


def extreme_thresholds(data: CoverageData) -> dict[str, tuple[float, float]]:
    return {
        name: (float(np.quantile(value, 0.20)), float(np.quantile(value, 0.80)))
        for name, value in data.mapping().items()
    }


def extreme_masks(
    data: CoverageData,
    thresholds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    values = data.mapping()
    thresholds = thresholds or extreme_thresholds(data)
    low = {name: value <= thresholds[name][0] for name, value in values.items()}
    high = {name: value >= thresholds[name][1] for name, value in values.items()}
    return {
        "low_roll+low_yaw": low["roll_authority"] & low["yaw_authority"],
        "low_yaw+slow_motor": low["yaw_authority"] & high["fall_time"],
        "high_force+low_thrust": high["external_force"] & low["thrust_to_weight"],
        "high_force+low_roll": high["external_force"] & low["roll_authority"],
        "high_force+low_yaw": high["external_force"] & low["yaw_authority"],
        "high_trim+slow_fall": high["required_trim_ratio"] & high["fall_time"],
        "high_force+low_authority+slow_motor": high["external_force"] & low["roll_authority"] & low["yaw_authority"] & high["fall_time"],
    }
