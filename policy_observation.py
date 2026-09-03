from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from env_l2f import L2FState


LEGACY_OBSERVATION_MODE = "legacy40"
COMPACT_OBSERVATION_MODE = "compact22"
INTEGRAL_OBSERVATION_MODE = "integral25"
OBSERVATION_MODES = (
    LEGACY_OBSERVATION_MODE,
    COMPACT_OBSERVATION_MODE,
    INTEGRAL_OBSERVATION_MODE,
)
OBSERVATION_DIMS = {
    LEGACY_OBSERVATION_MODE: 40,
    COMPACT_OBSERVATION_MODE: 22,
    INTEGRAL_OBSERVATION_MODE: 25,
}
PHYSICAL_OBSERVATION_DIM = 18
PREVIOUS_ACTION_DIM = 4
INTEGRAL_POSITION_DIM = 3
INTEGRAL_INPUT_FRAMES = ("world", "body")
LEGACY_BOX_INTEGRAL_CLAMP_MODE = "box"
CYLINDRICAL_INTEGRAL_CLAMP_MODE = "cylindrical"
INTEGRAL_CLAMP_MODES = (
    LEGACY_BOX_INTEGRAL_CLAMP_MODE,
    CYLINDRICAL_INTEGRAL_CLAMP_MODE,
)


@dataclass
class PolicyObservationState:
    integral_position: torch.Tensor

    def detach(self) -> "PolicyObservationState":
        return PolicyObservationState(self.integral_position.detach())

    def clone(self) -> "PolicyObservationState":
        return PolicyObservationState(self.integral_position.clone())


def observation_dim(mode: str) -> int:
    try:
        return OBSERVATION_DIMS[mode]
    except KeyError as exc:
        raise ValueError(f"observation mode must be one of {OBSERVATION_MODES}, got {mode!r}") from exc


def mode_from_observation_dim(width: int) -> str:
    for mode, dimension in OBSERVATION_DIMS.items():
        if dimension == width:
            return mode
    raise ValueError(f"unsupported observation width: {width}")


def validate_integral_clamp_mode(mode: str) -> str:
    if mode not in INTEGRAL_CLAMP_MODES:
        raise ValueError(
            f"integral_clamp_mode must be one of {INTEGRAL_CLAMP_MODES}, got {mode!r}"
        )
    return mode


def cylindrical_integral_radial_limit(integral_limit: float) -> float:
    """Return the circle radius with the legacy horizontal box's area."""

    if integral_limit <= 0.0:
        raise ValueError("integral_limit must be positive")
    return 2.0 * float(integral_limit) / math.sqrt(math.pi)


def integral_clamp_active_mask(
    integral: torch.Tensor,
    *,
    integral_limit: float,
    integral_clamp_mode: str = LEGACY_BOX_INTEGRAL_CLAMP_MODE,
    tolerance: float = 0.0,
) -> torch.Tensor:
    """Return a per-axis mask for values on the selected clamp boundary.

    A cylindrical horizontal clamp is one rotationally symmetric constraint, so
    both horizontal axes are marked active when its radial boundary is active.
    """

    if integral.ndim != 2 or integral.shape[-1] != INTEGRAL_POSITION_DIM:
        raise ValueError("integral must have shape [batch, 3]")
    if integral_limit <= 0.0:
        raise ValueError("integral_limit must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    mode = validate_integral_clamp_mode(integral_clamp_mode)
    threshold = float(integral_limit) - float(tolerance)
    if mode == LEGACY_BOX_INTEGRAL_CLAMP_MODE:
        return integral.abs() >= threshold

    radial_threshold = cylindrical_integral_radial_limit(integral_limit) - float(tolerance)
    radial_active = torch.linalg.vector_norm(integral[:, :2], dim=-1) >= radial_threshold
    vertical_active = integral[:, 2].abs() >= threshold
    return torch.stack((radial_active, radial_active, vertical_active), dim=-1)


def initial_observation_state(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PolicyObservationState:
    return PolicyObservationState(
        torch.zeros(batch_size, INTEGRAL_POSITION_DIM, device=device, dtype=dtype)
    )


def reset_observation_state(
    observation_state: PolicyObservationState,
    reset_mask: torch.Tensor,
) -> None:
    if reset_mask.dtype != torch.bool or reset_mask.shape != observation_state.integral_position.shape[:1]:
        raise ValueError("reset_mask must be a boolean [batch] tensor")
    observation_state.integral_position[reset_mask] = 0.0


def physical_observation(state: "L2FState") -> torch.Tensor:
    return torch.cat(
        (
            state.position,
            state.velocity,
            state.rotation.reshape(state.position.shape[0], 9),
            state.omega,
        ),
        dim=-1,
    )


def sample_deployable_physical_observation(
    state: "L2FState",
    *,
    noise_max: float = 0.0,
) -> torch.Tensor:
    observed = physical_observation(state)
    if noise_max < 0.0:
        raise ValueError("observation noise must be non-negative")
    if noise_max > 0.0:
        observed = observed + torch.empty_like(observed).uniform_(-noise_max, noise_max)
    return observed


def build_policy_observation(
    state: "L2FState",
    observation_state: PolicyObservationState,
    *,
    mode: str,
    noise_max: float = 0.0,
    integral_input_frame: str = "world",
    integral_input_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one deployable observation and return its observed position.

    Physical state noise is sampled exactly once.  ``legacy40`` deterministically
    derives its historical affine duplicate from that same sample; it never
    creates a second error/noise branch.
    """
    observed_physical = sample_deployable_physical_observation(state, noise_max=noise_max)
    previous_action = state.previous_action
    if noise_max > 0.0:
        previous_action = previous_action + torch.empty_like(previous_action).uniform_(
            -noise_max, noise_max
        )

    if mode == LEGACY_OBSERVATION_MODE:
        affine_duplicate = observed_physical.clone()
        affine_duplicate[:, (6, 10, 14)] -= 1.0
        observation = torch.cat((observed_physical, affine_duplicate, previous_action), dim=-1)
    elif mode == COMPACT_OBSERVATION_MODE:
        observation = torch.cat((observed_physical, previous_action), dim=-1)
    elif mode == INTEGRAL_OBSERVATION_MODE:
        if integral_input_frame not in INTEGRAL_INPUT_FRAMES:
            raise ValueError(
                f"integral_input_frame must be one of {INTEGRAL_INPUT_FRAMES}"
            )
        integral_input = observation_state.integral_position
        if integral_input_frame == "body":
            observed_rotation = observed_physical[:, 6:15].reshape(-1, 3, 3)
            integral_input = world_integral_to_body(
                observation_state.integral_position,
                observed_rotation,
            )
        if integral_input_multiplier < 0.0:
            raise ValueError("integral_input_multiplier must be non-negative")
        integral_input = float(integral_input_multiplier) * integral_input
        observation = torch.cat(
            (observed_physical, integral_input, previous_action),
            dim=-1,
        )
    else:
        observation_dim(mode)
        raise AssertionError("unreachable")
    return observation, observed_physical[:, :3]


def world_integral_to_body(
    integral_world: torch.Tensor,
    rotation_body_to_world: torch.Tensor,
) -> torch.Tensor:
    """Rotate a deployable world-frame position integral into body axes."""

    if integral_world.ndim != 2 or integral_world.shape[-1] != 3:
        raise ValueError("integral_world must have shape [batch,3]")
    if rotation_body_to_world.shape != (integral_world.shape[0], 3, 3):
        raise ValueError("rotation_body_to_world must have shape [batch,3,3]")
    return torch.bmm(
        rotation_body_to_world.transpose(1, 2),
        integral_world.unsqueeze(-1),
    ).squeeze(-1)


def update_position_integral(
    observation_state: PolicyObservationState,
    observed_position: torch.Tensor,
    *,
    dt: float,
    integral_limit: float,
    integral_leak: float = 0.0,
    integral_clamp_mode: str = LEGACY_BOX_INTEGRAL_CLAMP_MODE,
) -> PolicyObservationState:
    if observed_position.shape != observation_state.integral_position.shape:
        raise ValueError("observed_position must have shape [batch, 3]")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if integral_limit <= 0.0:
        raise ValueError("integral_limit must be positive")
    if integral_leak < 0.0:
        raise ValueError("integral_leak must be non-negative")
    clamp_mode = validate_integral_clamp_mode(integral_clamp_mode)
    retention = max(0.0, 1.0 - float(integral_leak) * float(dt))
    integral = retention * observation_state.integral_position + float(dt) * observed_position
    if clamp_mode == LEGACY_BOX_INTEGRAL_CLAMP_MODE:
        # Keep the historical operation exactly intact for checkpoint replay and
        # all training paths that do not opt into a counterfactual clamp.
        integral = integral.clamp(-float(integral_limit), float(integral_limit))
    else:
        radial_limit = cylindrical_integral_radial_limit(integral_limit)
        horizontal = integral[:, :2]
        radial_scale = (
            radial_limit
            / torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True).clamp_min(1.0e-12)
        ).clamp_max(1.0)
        integral = torch.cat(
            (
                horizontal * radial_scale,
                integral[:, 2:3].clamp(-float(integral_limit), float(integral_limit)),
            ),
            dim=-1,
        )
    return PolicyObservationState(integral)
