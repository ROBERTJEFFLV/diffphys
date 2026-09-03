"""L2F rollout and full recurrent boundary utilities for structured control."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn

from env_l2f import L2FSimulator, L2FState
from full_space_shooting import BoundaryLayout, so3_exp, so3_local_residual
from policy_observation import PolicyObservationState, build_policy_observation
from structured_policy import (
    StructuredPolicyConfig,
    StructuredPolicyState,
    StructuredRecurrentPolicy,
)
from structured_stability import quadratic_metric
from smooth_risk import solve_smooth_cvar_eta
from structured_checkpoint import require_current_cadence_semantics


Tensor = torch.Tensor
BOUNDARY_CODEC_VERSION = 2
LEGACY_BOUNDARY_CODEC_VERSION = 1
STRUCTURED_BOUNDARY_CODEC_VERSION = BOUNDARY_CODEC_VERSION


def load_structured_policy(
    checkpoint_path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[StructuredRecurrentPolicy, dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    require_current_cadence_semantics(payload, context="structured rollout")
    config = StructuredPolicyConfig(**payload["config"])
    policy = StructuredRecurrentPolicy(config).to(device=device, dtype=dtype)
    policy.load_state_dict(payload["model"], strict=True)
    if bool(payload.get("fast_feedback_verified", False)):
        if not policy.fast_feedback.verify():
            raise RuntimeError("checkpoint labels an invalid fast gain as verified")
    return policy, payload


@dataclass
class StructuredClosedLoopState:
    physical: L2FState
    policy: StructuredPolicyState


@dataclass(frozen=True)
class ActionProbeBank:
    states: Tensor
    slow_counters: Tuple[int, ...]


@dataclass(frozen=True)
class ParameterVectorSpec:
    names: Tuple[str, ...]
    shapes: Tuple[torch.Size, ...]
    numels: Tuple[int, ...]

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        trainable_only: bool = True,
        allow_prefixes: Optional[Iterable[str]] = None,
    ) -> "ParameterVectorSpec":
        prefixes = None if allow_prefixes is None else tuple(allow_prefixes)
        entries = tuple(
            (name, value)
            for name, value in module.named_parameters()
            if (not trainable_only or value.requires_grad)
            and (prefixes is None or any(name.startswith(prefix) for prefix in prefixes))
        )
        if not entries:
            raise ValueError("parameter selection is empty")
        return cls(
            names=tuple(name for name, _ in entries),
            shapes=tuple(value.shape for _, value in entries),
            numels=tuple(value.numel() for _, value in entries),
        )

    def flatten(self, module: nn.Module) -> Tensor:
        values = dict(module.named_parameters())
        return torch.cat(tuple(values[name].reshape(-1) for name in self.names))

    def unflatten(self, vector: Tensor) -> Dict[str, Tensor]:
        if vector.ndim != 1 or vector.numel() != sum(self.numels):
            raise ValueError("parameter vector has the wrong size")
        result: Dict[str, Tensor] = {}
        offset = 0
        for name, shape, count in zip(self.names, self.shapes, self.numels):
            result[name] = vector[offset:offset + count].reshape(shape)
            offset += count
        return result

    def trust_scale(self, module: nn.Module, *, floor: float = 1.0e-2) -> Tensor:
        """Return one robust scale per selected parameter coordinate."""

        if floor <= 0.0:
            raise ValueError("trust scale floor must be positive")
        values = dict(module.named_parameters())
        blocks = []
        for name in self.names:
            value = values[name].detach()
            scale = value.square().mean().sqrt().clamp_min(float(floor))
            blocks.append(torch.ones_like(value).reshape(-1) * scale)
        return torch.cat(blocks)

    @torch.no_grad()
    def assign_(self, module: nn.Module, vector: Tensor) -> None:
        values = self.unflatten(vector)
        for name, parameter in module.named_parameters():
            if name in values:
                parameter.copy_(values[name])


def clone_l2f_state(state: L2FState) -> L2FState:
    return L2FState(
        **{
            field.name: getattr(state, field.name).detach().clone()
            for field in fields(L2FState)
        }
    )


def clone_policy_state(state: StructuredPolicyState) -> StructuredPolicyState:
    return StructuredPolicyState(
        hidden=state.hidden.detach().clone(),
        identifier=state.identifier.detach().clone(),
        motor_estimate=state.motor_estimate.detach().clone(),
        integral=state.integral.detach().clone(),
        motor_bank=None if state.motor_bank is None else state.motor_bank.detach().clone(),
        slow_trim=None if state.slow_trim is None else state.slow_trim.detach().clone(),
        slow_body_z=None if state.slow_body_z is None else state.slow_body_z.detach().clone(),
        capability=None if state.capability is None else state.capability.detach().clone(),
        slow_counter=int(state.slow_counter),
        prev_velocity=None if state.prev_velocity is None else state.prev_velocity.detach().clone(),
        prev_omega=None if state.prev_omega is None else state.prev_omega.detach().clone(),
        context_sum=None if state.context_sum is None else state.context_sum.detach().clone(),
        capability_log_mean=(None if state.capability_log_mean is None
                             else state.capability_log_mean.detach().clone()),
        capability_log_scale=(None if state.capability_log_scale is None
                              else state.capability_log_scale.detach().clone()),
        capability_ucb=(None if state.capability_ucb is None
                        else state.capability_ucb.detach().clone()),
        capability_ucb_target=(None if state.capability_ucb_target is None
                               else state.capability_ucb_target.detach().clone()),
        contextual_gain=(None if state.contextual_gain is None
                         else state.contextual_gain.detach().clone()),
        contextual_gain_target=(None if state.contextual_gain_target is None
                                else state.contextual_gain_target.detach().clone()),
        contextual_blend=(None if state.contextual_blend is None
                          else state.contextual_blend.detach().clone()),
        boot_progress=(None if state.boot_progress is None
                       else state.boot_progress.detach().clone()),
        identification_failed=(None if state.identification_failed is None
                               else state.identification_failed.detach().clone()),
        disturbance_accel=(None if state.disturbance_accel is None
                           else state.disturbance_accel.detach().clone()),
        disturbance_residual_sum=(
            None if state.disturbance_residual_sum is None
            else state.disturbance_residual_sum.detach().clone()
        ),
        disturbance_thrust_sum=(
            None if state.disturbance_thrust_sum is None
            else state.disturbance_thrust_sum.detach().clone()
        ),
        previous_executed_action=(
            None if state.previous_executed_action is None
            else state.previous_executed_action.detach().clone()
        ),
        disturbance_response_count=(
            None if state.disturbance_response_count is None
            else state.disturbance_response_count.detach().clone()
        ),
        excitation_history=(
            None if state.excitation_history is None
            else state.excitation_history.detach().clone()
        ),
        previous_motor_estimate=(
            None if state.previous_motor_estimate is None
            else state.previous_motor_estimate.detach().clone()
        ),
    )


def clone_closed_loop(state: StructuredClosedLoopState) -> StructuredClosedLoopState:
    return StructuredClosedLoopState(
        physical=clone_l2f_state(state.physical),
        policy=clone_policy_state(state.policy),
    )


def _rotation_log(rotation: Tensor) -> Tensor:
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    identity = identity.expand(*rotation.shape[:-2], 3, 3)
    return so3_local_residual(identity, rotation)


def _sphere2_coordinates(direction: Tensor) -> Tensor:
    """Stereographic two-coordinate chart for a near-hover unit direction."""

    unit = torch.nn.functional.normalize(direction, dim=-1)
    return unit[:, :2] / (1.0 + unit[:, 2:3]).clamp_min(1.0e-6)


def _sphere2_direction(coordinates: Tensor) -> Tensor:
    radius2 = coordinates.square().sum(dim=-1, keepdim=True)
    denominator = 1.0 + radius2
    return torch.cat(
        (2.0 * coordinates / denominator, (1.0 - radius2) / denominator),
        dim=-1,
    )


class StructuredBoundaryCodec:
    """Pack the complete deployable recurrent state in normalized coordinates.

    ``codec_version=2`` is the current layout and includes cadence latches
    (response count, identification failure, and slow phase).  Passing
    ``codec_version=1`` is an explicit compatibility mode for legacy boundary
    artifacts; it retains their historical zeroed latches and phase override.

    Orientations use three absolute local SO(3) coordinates.  This chart is
    suitable for the near-hover regions used by the MVP and is not claimed to
    cover rotations at the log-map singularity.
    """

    def __init__(self, template: L2FState, policy: StructuredRecurrentPolicy,
                 *, boot_completed: bool = False,
                 codec_version: int = BOUNDARY_CODEC_VERSION) -> None:
        if codec_version not in (LEGACY_BOUNDARY_CODEC_VERSION, BOUNDARY_CODEC_VERSION):
            raise ValueError("codec_version must be 1 (legacy) or 2 (complete state)")
        self.template = template
        self.policy = policy
        self.boot_completed = bool(boot_completed)
        self.codec_version = int(codec_version)
        hidden_dim = policy.config.hidden_dim
        identifier_dim = policy.config.identifier_dim
        widths = [
            ("position", 3),
            ("velocity", 3),
            ("orientation", 3),
            ("omega", 3),
            ("motor", 4),
            ("previous_action", 4),
            ("fast_hidden", hidden_dim),
            ("slow_hidden", identifier_dim),
            ("motor_estimate", 4),
        ]
        if policy.config.motor_observer_bank_size:
            widths.append(("motor_bank", 4 * policy.config.motor_observer_bank_size))
        widths.extend([
            ("integral", 3),
            ("disturbance_accel", 3),
            ("disturbance_residual_sum", 3),
            ("disturbance_thrust_sum", 3),
            ("previous_executed_action", 4),
            ("slow_trim", 4),
            # Desired thrust direction is an S2 variable, not three free
            # Euclidean coordinates.  A two-dimensional stereographic chart
            # removes the radial null direction from the shooting KKT system.
            ("slow_body_z", 2),
            ("capability", 6),
            ("prev_velocity", 3),
            ("prev_omega", 3),
            ("context_sum", 24),
            ("excitation_history", 13 * 4),
            ("previous_motor_estimate", 4),
            ("capability_log_mean", 6),
            ("capability_log_scale", 6),
            ("capability_ucb", 6),
            ("capability_ucb_target", 6),
            ("contextual_gain", 4 * 15),
            ("contextual_gain_target", 4 * 15),
            ("contextual_blend", 1),
        ])
        if self.codec_version >= BOUNDARY_CODEC_VERSION:
            widths.extend([
                ("disturbance_response_count", 1),
                ("identification_failed", 1),
                ("slow_counter", 1),
            ])
        if not self.boot_completed:
            widths.append(("boot_progress", 1))
        self.slices: Dict[str, slice] = {}
        offset = 0
        for name, width in widths:
            self.slices[name] = slice(offset, offset + width)
            offset += width
        self.state_dim = offset
        scales = torch.ones(self.state_dim, dtype=template.position.dtype, device=template.position.device)
        scale_values = {
            "position": (0.10,) * 3,
            "velocity": (0.10,) * 3,
            "orientation": (0.10,) * 3,
            "omega": (0.50,) * 3,
            "motor": (0.10,) * 4,
            "previous_action": (0.10,) * 4,
            "motor_estimate": (0.10,) * 4,
            "integral": (0.10,) * 3,
            "disturbance_accel": (1.0,) * 3,
            "disturbance_residual_sum": (1.0,) * 3,
            "disturbance_thrust_sum": (1.0,) * 3,
            "previous_executed_action": (0.10,) * 4,
            "excitation_history": (1.0,) * (13 * 4),
            "previous_motor_estimate": (0.10,) * 4,
            "slow_trim": (0.10,) * 4,
            "slow_body_z": (0.10,) * 2,
            "capability": (1.0, 500.0, 0.20, 0.20, 0.10, 0.10),
            "prev_velocity": (0.10,) * 3,
            "prev_omega": (0.50,) * 3,
            "capability_log_mean": (1.0,) * 6,
            "capability_log_scale": (1.0,) * 6,
            "capability_ucb": (1.0, 500.0, 0.20, 0.20, 0.10, 0.10),
            "capability_ucb_target": (1.0, 500.0, 0.20, 0.20, 0.10, 0.10),
        }
        if policy.config.motor_observer_bank_size:
            scale_values["motor_bank"] = (0.10,) * (4 * policy.config.motor_observer_bank_size)
        if self.codec_version >= BOUNDARY_CODEC_VERSION:
            scale_values.update({
                "disturbance_response_count": (1.0,),
                "identification_failed": (1.0,),
                "slow_counter": (1.0,),
            })
        for name, value in scale_values.items():
            scales[self.slices[name]] = scales.new_tensor(value)
        self.scales = scales
        self.layout = BoundaryLayout(
            self.state_dim,
            rotation_slice=self.slices["orientation"],
            rotation_scale=float(self.scales[self.slices["orientation"]][0]),
        )

    def _required(self, value: Optional[Tensor], name: str) -> Tensor:
        if value is None:
            raise ValueError("structured policy state is missing %s" % name)
        return value

    def pack(self, state: StructuredClosedLoopState) -> Tensor:
        policy = state.policy
        values = [
            state.physical.position,
            state.physical.velocity,
            _rotation_log(state.physical.rotation),
            state.physical.omega,
            state.physical.motor,
            state.physical.previous_action,
            policy.hidden,
            policy.identifier,
            policy.motor_estimate,
        ]
        if self.policy.config.motor_observer_bank_size:
            values.append(self._required(policy.motor_bank, "motor_bank").reshape(
                policy.hidden.shape[0], -1
            ))
        values.extend([
            policy.integral,
            self._required(policy.disturbance_accel, "disturbance_accel"),
            self._required(policy.disturbance_residual_sum, "disturbance_residual_sum"),
            self._required(policy.disturbance_thrust_sum, "disturbance_thrust_sum"),
            self._required(policy.previous_executed_action, "previous_executed_action"),
            self._required(policy.slow_trim, "slow_trim"),
            _sphere2_coordinates(
                self._required(policy.slow_body_z, "slow_body_z")
            ),
            self._required(policy.capability, "capability"),
            self._required(policy.prev_velocity, "prev_velocity"),
            self._required(policy.prev_omega, "prev_omega"),
            self._required(policy.context_sum, "context_sum"),
            self._required(policy.excitation_history, "excitation_history").reshape(
                policy.hidden.shape[0], -1
            ),
            self._required(policy.previous_motor_estimate, "previous_motor_estimate"),
            self._required(policy.capability_log_mean, "capability_log_mean"),
            self._required(policy.capability_log_scale, "capability_log_scale"),
            self._required(policy.capability_ucb, "capability_ucb"),
            self._required(policy.capability_ucb_target, "capability_ucb_target"),
            self._required(policy.contextual_gain, "contextual_gain").reshape(
                policy.hidden.shape[0], -1
            ),
            self._required(policy.contextual_gain_target, "contextual_gain_target").reshape(
                policy.hidden.shape[0], -1
            ),
            self._required(policy.contextual_blend, "contextual_blend"),
        ])
        if self.codec_version >= BOUNDARY_CODEC_VERSION:
            response_count = self._required(
                policy.disturbance_response_count, "disturbance_response_count"
            ).reshape(-1, 1)
            failed = self._required(policy.identification_failed, "identification_failed")
            values.extend([
                response_count,
                failed.to(dtype=policy.hidden.dtype).reshape(-1, 1),
                policy.hidden.new_full((policy.hidden.shape[0], 1), float(policy.slow_counter)),
            ])
        if self.boot_completed:
            progress = self._required(policy.boot_progress, "boot_progress")
            required = float(
                self.policy.config.burn_in_steps
                + self.policy.config.contextual_blend_steps
            )
            if bool((progress < required - 1.0e-6).any().item()):
                raise ValueError("boot-completed codec received an unfinished policy state")
        else:
            values.append(self._required(policy.boot_progress, "boot_progress"))
        flat = torch.cat(values, dim=-1)
        return flat / self.scales.to(flat)

    def unpack(self, flat: Tensor, *, slow_counter: int | None = None) -> StructuredClosedLoopState:
        if flat.ndim != 2 or flat.shape[-1] != self.state_dim:
            raise ValueError("boundary must have shape [batch,state_dim]")
        raw = flat * self.scales.to(flat)
        dynamic = {
            "position": raw[:, self.slices["position"]],
            "velocity": raw[:, self.slices["velocity"]],
            "rotation": so3_exp(raw[:, self.slices["orientation"]]),
            "omega": raw[:, self.slices["omega"]],
            "motor": raw[:, self.slices["motor"]],
            "previous_action": raw[:, self.slices["previous_action"]],
        }
        physical = L2FState(
            **{
                field.name: dynamic.get(field.name, getattr(self.template, field.name))
                for field in fields(L2FState)
            }
        )
        encoded_slow_counter = None
        encoded_identification_failed = None
        encoded_response_count = raw.new_zeros((raw.shape[0], 1))
        if self.codec_version >= BOUNDARY_CODEC_VERSION:
            encoded_response_count = raw[:, self.slices["disturbance_response_count"]]
            encoded_identification_failed = (
                raw[:, self.slices["identification_failed"]] >= 0.5
            ).squeeze(-1)
            encoded = raw[:, self.slices["slow_counter"]].squeeze(-1)
            rounded = encoded.round()
            if bool((encoded - rounded).abs().max().item() > 1.0e-5):
                raise ValueError("boundary slow_counter is not an integer")
            if bool((rounded != rounded[0]).any().item()):
                raise ValueError("boundary batch has inconsistent slow_counter values")
            encoded_slow_counter = int(rounded[0].item())
        effective_slow_counter = (
            int(slow_counter) if slow_counter is not None
            else (encoded_slow_counter if encoded_slow_counter is not None else 0)
        )
        policy = StructuredPolicyState(
            hidden=raw[:, self.slices["fast_hidden"]],
            identifier=raw[:, self.slices["slow_hidden"]],
            motor_estimate=raw[:, self.slices["motor_estimate"]],
            integral=raw[:, self.slices["integral"]],
            motor_bank=(
                raw[:, self.slices["motor_bank"]].reshape(
                    -1, self.policy.config.motor_observer_bank_size, 4
                )
                if self.policy.config.motor_observer_bank_size else None
            ),
            disturbance_accel=raw[:, self.slices["disturbance_accel"]],
            disturbance_residual_sum=raw[:, self.slices["disturbance_residual_sum"]],
            disturbance_thrust_sum=raw[:, self.slices["disturbance_thrust_sum"]],
            previous_executed_action=raw[:, self.slices["previous_executed_action"]],
            disturbance_response_count=encoded_response_count,
            slow_trim=raw[:, self.slices["slow_trim"]],
            slow_body_z=_sphere2_direction(raw[:, self.slices["slow_body_z"]]),
            capability=raw[:, self.slices["capability"]],
            slow_counter=effective_slow_counter,
            prev_velocity=raw[:, self.slices["prev_velocity"]],
            prev_omega=raw[:, self.slices["prev_omega"]],
            context_sum=raw[:, self.slices["context_sum"]],
            excitation_history=raw[:, self.slices["excitation_history"]].reshape(
                -1, 13, 4
            ),
            previous_motor_estimate=raw[:, self.slices["previous_motor_estimate"]],
            capability_log_mean=raw[:, self.slices["capability_log_mean"]],
            capability_log_scale=raw[:, self.slices["capability_log_scale"]],
            capability_ucb=raw[:, self.slices["capability_ucb"]],
            capability_ucb_target=raw[:, self.slices["capability_ucb_target"]],
            contextual_gain=raw[:, self.slices["contextual_gain"]].reshape(-1, 4, 15),
            contextual_gain_target=raw[:, self.slices["contextual_gain_target"]].reshape(-1, 4, 15),
            contextual_blend=raw[:, self.slices["contextual_blend"]],
            boot_progress=(
                raw.new_full(
                    (raw.shape[0], 1),
                    float(self.policy.config.burn_in_steps
                          + self.policy.config.contextual_blend_steps),
                )
                if self.boot_completed
                else raw[:, self.slices["boot_progress"]]
            ),
            identification_failed=encoded_identification_failed,
        )
        return StructuredClosedLoopState(physical=physical, policy=policy)


def make_structured_step_map(
    policy: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    codec: StructuredBoundaryCodec,
):
    """Return the differentiable one-transition map on the complete v2 state.

    v2 contains discrete cadence/failure latches.  They are part of the
    deployable state and are carried through the map, but have no meaningful
    derivative.  During a tangent evaluation their input coordinates are
    therefore held at the primal value; this keeps JVPs well-defined without
    silently dropping the latches from the packed state.
    """

    latch_slices = tuple(codec.slices[name] for name in
                         ("identification_failed", "slow_counter")
                         if name in codec.slices)

    def step_map(value: Tensor) -> Tensor:
        one_dimensional = value.ndim == 1
        if one_dimensional:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[-1] != codec.state_dim:
            raise ValueError("structured step-map input must be [state_dim] or [1,state_dim]")
        # Detaching only the discrete slices removes data-dependent integer /
        # bool operations from the JVP trace.  Every continuous field remains
        # a true derivative coordinate of the complete v2 codec.
        primal = value.detach()
        for selected in latch_slices:
            mask = torch.zeros(codec.state_dim, dtype=torch.bool, device=value.device)
            mask[selected] = True
            value = torch.where(mask.unsqueeze(0), primal, value)
        current = codec.unpack(value)
        end, _ = rollout_structured_segment(
            policy, simulator, current, steps=1, collect=False
        )
        packed = codec.pack(end)
        return packed[0] if one_dimensional else packed

    return step_map


def structured_global_yaw_basis(
    codec: StructuredBoundaryCodec,
    packed_state: Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> Tensor:
    """Return the *declared* global-yaw tangent in packed coordinates.

    Only absolute orientation is quotiented.  Position, velocity, integral,
    recurrent and actuator coordinates are intentionally left untouched; this
    avoids turning unrelated rotational or recurrent modes into a gauge.
    """

    if packed_state.ndim != 1 or packed_state.numel() != codec.state_dim:
        raise ValueError("packed_state must be a flat complete codec state")
    raw = packed_state * codec.scales.to(packed_state)
    orientation = raw[codec.slices["orientation"]]
    rotation = so3_exp(orientation.unsqueeze(0))[0]
    z = raw.new_zeros(3)
    z[2] = 1.0
    skew = torch.zeros((3, 3), dtype=raw.dtype, device=raw.device)
    skew[0, 1], skew[1, 0] = -z[2], z[2]
    exp_plus = torch.matrix_exp(skew * float(epsilon))
    exp_minus = torch.matrix_exp(-skew * float(epsilon))
    plus = _rotation_log(torch.matmul(exp_plus, rotation).unsqueeze(0))[0]
    minus = _rotation_log(torch.matmul(exp_minus, rotation).unsqueeze(0))[0]
    basis = raw.new_zeros(codec.state_dim)
    basis[codec.slices["orientation"]] = (plus - minus) / (2.0 * float(epsilon))
    return basis / codec.scales.to(raw)


def structured_observation(state: StructuredClosedLoopState) -> Tensor:
    observation, _ = build_policy_observation(
        state.physical,
        PolicyObservationState(state.policy.integral),
        mode="integral25",
        noise_max=0.0,
        integral_input_frame="body",
        integral_input_multiplier=1.0,
    )
    return observation


def _functional_policy(
    policy: StructuredRecurrentPolicy,
    parameters: Mapping[str, Tensor],
    observation: Tensor,
    state: StructuredPolicyState,
    dt: float,
):
    try:
        return torch.func.functional_call(
            policy,
            parameters,
            (observation, state, dt),
        )
    except AttributeError:
        from torch.nn.utils.stateless import functional_call

        return functional_call(policy, parameters, (observation, state, dt))


def functional_node_actions(
    policy: StructuredRecurrentPolicy,
    codec: "StructuredBoundaryCodec",
    parameter_spec: ParameterVectorSpec,
    theta: Tensor,
    initial: Tensor,
    endpoints: Tensor,
) -> ActionProbeBank:
    """Evaluate actions at every segment start for an action trust region."""

    parameters = parameter_spec.unflatten(theta)
    starts = torch.cat((initial.unsqueeze(0), endpoints[:-1]), dim=0)
    actions = []
    for start in starts:
        closed = codec.unpack(start)
        observation = structured_observation(closed)
        action, _ = _functional_policy(
            policy,
            parameters,
            observation,
            closed.policy,
            policy.config.dt,
        )
        actions.append(action)
    return torch.stack(actions)


def functional_trajectory_actions(
    policy: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    codec: "StructuredBoundaryCodec",
    parameter_spec: ParameterVectorSpec,
    theta: Tensor,
    initial: Tensor,
    endpoints: Tensor,
    *,
    steps: int,
) -> Tensor:
    """Evaluate every action on all independently started shooting segments.

    The action trust region is a function-space constraint.  Checking only the
    first action at each shooting node can miss a large policy change later in
    an H250 segment, so the formal trainer uses this dense trajectory value.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    parameters = parameter_spec.unflatten(theta)
    starts = torch.cat((initial.unsqueeze(0), endpoints[:-1]), dim=0)
    segment_actions = []
    for start in starts:
        current = codec.unpack(start)
        actions = []
        for _ in range(steps):
            observation = structured_observation(current)
            action, policy_state = _functional_policy(
                policy,
                parameters,
                observation,
                current.policy,
                policy.config.dt,
            )
            physical = simulator.step(current.physical, action, grad_decay=1.0)
            current = StructuredClosedLoopState(physical=physical, policy=policy_state)
            actions.append(action)
        segment_actions.append(torch.stack(actions))
    return torch.stack(segment_actions)


def functional_trajectory_diagnostics(
    policy: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    codec: "StructuredBoundaryCodec",
    parameter_spec: ParameterVectorSpec,
    theta: Tensor,
    initial: Tensor,
    endpoints: Tensor,
    *,
    steps: int,
) -> tuple[Tensor, Tensor]:
    """Return actions and identifier latches from one functional rollout.

    Nonlinear MS acceptance needs both values.  Collecting them together avoids
    repeating every H250 segment solely to read a boolean safety latch.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    parameters = parameter_spec.unflatten(theta)
    starts = torch.cat((initial.unsqueeze(0), endpoints[:-1]), dim=0)
    segment_actions = []
    segment_failures = []
    for start in starts:
        current = codec.unpack(start)
        actions = []
        failures = []
        for _ in range(steps):
            observation = structured_observation(current)
            action, policy_state = _functional_policy(
                policy, parameters, observation, current.policy, policy.config.dt
            )
            flag = policy_state.identification_failed
            actions.append(action)
            failures.append(
                torch.zeros(action.shape[0], dtype=torch.bool, device=action.device)
                if flag is None else flag
            )
            physical = simulator.step(current.physical, action, grad_decay=1.0)
            current = StructuredClosedLoopState(physical=physical, policy=policy_state)
        segment_actions.append(torch.stack(actions))
        segment_failures.append(torch.stack(failures))
    return torch.stack(segment_actions), torch.stack(segment_failures)


def functional_trajectory_identification_failures(
    policy: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    codec: "StructuredBoundaryCodec",
    parameter_spec: ParameterVectorSpec,
    theta: Tensor,
    initial: Tensor,
    endpoints: Tensor,
    *,
    steps: int,
) -> Tensor:
    """Return the non-differentiable identifier safety flag on every segment.

    The boolean latch is intentionally not an optimization coordinate.  It is
    recomputed on the restored nonlinear rollout and acts only as an acceptance
    gate, so an SQP update cannot hide loss of capability confidence at a
    shooting boundary.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    parameters = parameter_spec.unflatten(theta)
    starts = torch.cat((initial.unsqueeze(0), endpoints[:-1]), dim=0)
    segment_failures = []
    for start in starts:
        current = codec.unpack(start)
        failures = []
        for _ in range(steps):
            observation = structured_observation(current)
            action, policy_state = _functional_policy(
                policy, parameters, observation, current.policy, policy.config.dt
            )
            flag = policy_state.identification_failed
            failures.append(
                torch.zeros(
                    action.shape[0], dtype=torch.bool, device=action.device
                ) if flag is None else flag
            )
            physical = simulator.step(current.physical, action, grad_decay=1.0)
            current = StructuredClosedLoopState(physical=physical, policy=policy_state)
        segment_failures.append(torch.stack(failures))
    return torch.stack(segment_failures)


def build_action_probe_bank(
    policy: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    codec: "StructuredBoundaryCodec",
    parameter_spec: ParameterVectorSpec,
    theta: Tensor,
    initial: Tensor,
    endpoints: Tensor,
    *,
    steps: int,
    stride: int = 1,
) -> Tensor:
    """Freeze recurrent/physical states used by the policy action trust region."""

    if steps < 1 or stride < 1:
        raise ValueError("steps and stride must be positive")
    parameters = parameter_spec.unflatten(theta)
    starts = torch.cat((initial.unsqueeze(0), endpoints[:-1]), dim=0)
    probes = []
    counters = []
    with torch.no_grad():
        for start in starts:
            # Action probes must use the exact deployment cadence phase stored
            # in the v2 boundary.  Resetting every segment to phase zero is
            # only harmless for one special segment length and invalidates a
            # function-space trust region as soon as segment/cadence choices
            # change.
            current = codec.unpack(start)
            for index in range(steps):
                if index % stride == 0:
                    probes.append(codec.pack(current))
                    counters.append(int(current.policy.slow_counter))
                observation = structured_observation(current)
                action, policy_state = _functional_policy(
                    policy, parameters, observation, current.policy, policy.config.dt
                )
                physical = simulator.step(current.physical, action, grad_decay=1.0)
                current = StructuredClosedLoopState(physical=physical, policy=policy_state)
    return ActionProbeBank(torch.stack(probes), tuple(counters))


def functional_probe_actions(
    policy: StructuredRecurrentPolicy,
    codec: "StructuredBoundaryCodec",
    parameter_spec: ParameterVectorSpec,
    theta: Tensor,
    probes: ActionProbeBank,
) -> Tensor:
    """Evaluate theta at fixed probe states without policy-induced state drift."""

    if probes.states.ndim != 3 or probes.states.shape[-1] != codec.state_dim:
        raise ValueError("probes must have shape [probe,batch,state_dim]")
    if len(probes.slow_counters) != probes.states.shape[0]:
        raise ValueError("probe state/counter lengths differ")
    parameters = parameter_spec.unflatten(theta)
    actions = []
    for value, counter in zip(probes.states, probes.slow_counters):
        closed = codec.unpack(value, slow_counter=counter)
        observation = structured_observation(closed)
        action, _ = _functional_policy(
            policy, parameters, observation, closed.policy, policy.config.dt
        )
        actions.append(action)
    return torch.stack(actions)


def rollout_structured_segment(
    policy: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    initial: StructuredClosedLoopState,
    *,
    steps: int,
    parameters: Optional[Mapping[str, Tensor]] = None,
    collect: bool = False,
) -> Tuple[StructuredClosedLoopState, Dict[str, Tensor]]:
    if steps < 1:
        raise ValueError("steps must be positive")
    current = initial
    actions = []
    positions = []
    velocities = []
    omegas = []
    conditions = []
    headrooms = []
    wrench_errors = []
    identification_failures = []
    identification_publications = []
    identification_publication_availability = []
    confidence_widths = []
    for _ in range(steps):
        observation = structured_observation(current)
        if parameters is None:
            output = policy.forward_with_aux(
                observation, current.policy, simulator.params.dt
            )
            action = output.action
            next_policy_state = output.next_state
        else:
            action, next_policy_state = _functional_policy(
                policy,
                parameters,
                observation,
                current.policy,
                simulator.params.dt,
            )
            output = None
        physical = simulator.step(current.physical, action, grad_decay=1.0)
        current = StructuredClosedLoopState(physical=physical, policy=next_policy_state)
        if collect:
            if output is None:
                raise ValueError("functional parameter rollout does not collect auxiliary diagnostics")
            allocation = output.auxiliary["allocator"]
            actions.append(action)
            positions.append(physical.position)
            velocities.append(physical.velocity)
            omegas.append(physical.omega)
            conditions.append(allocation.condition_number)
            headrooms.append(allocation.minimum_headroom)
            wrench_errors.append(allocation.wrench_residual)
            identification_failures.append(
                output.auxiliary["identification_failed"]
            )
            identification_publications.append(
                output.auxiliary["identification_published"]
            )
            identification_publication_availability.append(
                output.auxiliary["identification_publication_available"]
            )
            confidence_widths.append(
                output.auxiliary["effectiveness_log_interval_width"]
            )
    if not collect:
        return current, {}
    return current, {
        "action": torch.stack(actions),
        "position": torch.stack(positions),
        "velocity": torch.stack(velocities),
        "omega": torch.stack(omegas),
        "allocator_condition": torch.stack(conditions),
        "minimum_headroom": torch.stack(headrooms),
        "wrench_residual": torch.stack(wrench_errors),
        "identification_failed": torch.stack(identification_failures),
        "identification_published": torch.stack(identification_publications),
        "identification_publication_available": torch.stack(
            identification_publication_availability
        ),
        "effectiveness_log_interval_width": torch.stack(confidence_widths),
    }


def make_segment_map(
    policy: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    codec: StructuredBoundaryCodec,
    parameter_spec: ParameterVectorSpec,
    *,
    steps: int,
):
    if steps % policy.config.slow_cadence != 0:
        raise ValueError("segment steps must align with slow context cadence")

    def segment_map(start: Tensor, theta: Tensor) -> Tensor:
        parameters = parameter_spec.unflatten(theta)
        initial = codec.unpack(start)
        end, _ = rollout_structured_segment(
            policy,
            simulator,
            initial,
            steps=steps,
            parameters=parameters,
            collect=False,
        )
        return codec.pack(end)

    return segment_map


def initialize_exact_endpoints(segment_map, initial: Tensor, theta: Tensor, segments: int) -> Tensor:
    if segments < 1:
        raise ValueError("segments must be positive")
    endpoints = []
    current = initial
    with torch.no_grad():
        for _ in range(segments):
            current = segment_map(current, theta)
            endpoints.append(current)
    return torch.stack(endpoints)


def endpoint_control_error(codec: StructuredBoundaryCodec, endpoint: Tensor) -> Tensor:
    """Return the 15D normalized physical/control error at packed endpoints."""

    state = codec.unpack(endpoint)
    p = state.physical.position / 0.10
    v = state.physical.velocity / 0.10
    body_z = state.physical.rotation[:, :, 2]
    desired = state.policy.slow_body_z
    assert desired is not None
    desired_body = torch.bmm(
        state.physical.rotation.transpose(1, 2), desired.unsqueeze(-1)
    ).squeeze(-1)
    tilt = desired_body[:, :2] / 0.10
    omega = state.physical.omega / 0.50
    trim = state.policy.slow_trim
    assert trim is not None
    motor = (state.policy.motor_estimate - trim) / 0.10
    return torch.cat((p, v, tilt, omega, motor), dim=-1)


def terminal_risk_residual(
    codec: StructuredBoundaryCodec,
    predicted_ends: Tensor,
    *,
    alpha: float = 0.8,
    beta: float = 1.0e-2,
) -> Tensor:
    """Least-squares residual whose square contains a smooth terminal CVaR term."""

    if not 0.5 <= alpha < 1.0 or beta <= 0.0:
        raise ValueError("terminal residual requires alpha in [0.5,1) and beta > 0")
    terminal_error = endpoint_control_error(codec, predicted_ends[-1])
    terminal_energy = terminal_error.square().mean(dim=-1)
    eta = solve_smooth_cvar_eta(terminal_energy, alpha=alpha, beta=beta)
    excess = float(beta) * torch.nn.functional.softplus(
        (terminal_energy - eta) / float(beta)
    )
    # The least-squares contribution is exactly the smooth empirical
    # Rockafellar--Uryasev expression: eta + mean(excess)/(1-alpha).  Keeping
    # eta in the residual is essential for line-search merit comparisons even
    # though its quantile selection is intentionally detached.
    eta_residual = torch.sqrt(2.0 * eta.clamp_min(0.0) + 1.0e-12).reshape(1)
    tail_residual = torch.sqrt(
        2.0 * excess
        / (max(1.0e-12, 1.0 - float(alpha)) * float(terminal_energy.numel()))
        + 1.0e-12
    )
    return torch.cat(
        (0.10 * terminal_error.reshape(-1), eta_residual, tail_residual.reshape(-1))
    )


def phase_space_contraction_risk_residual(
    codec: StructuredBoundaryCodec,
    starts: Tensor,
    predicted_ends: Tensor,
    *,
    metric: Tensor,
    retention: float,
    alpha: float = 0.8,
    beta: float = 1.0e-2,
    contraction_softness: float = 1.0e-2,
) -> Tensor:
    """Boundary-local contraction plus final phase-space smooth CVaR.

    The metric comes from explicit desired poles.  This sampled objective is a
    candidate Lyapunov condition.  Separate augmented closed-loop checks add
    sampled local evidence, but do not by themselves constitute a certificate.
    """

    if starts.shape != predicted_ends.shape:
        raise ValueError("shooting starts and predicted ends must match")
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must be in [0,1]")
    if contraction_softness <= 0.0:
        raise ValueError("contraction softness must be positive")
    start_error = torch.stack([endpoint_control_error(codec, value) for value in starts])
    end_error = torch.stack([
        endpoint_control_error(codec, value) for value in predicted_ends
    ])
    metric = metric.to(end_error)
    start_energy = quadratic_metric(start_error, metric) / float(metric.shape[0])
    end_energy = quadratic_metric(end_error, metric) / float(metric.shape[0])
    margin = end_energy - float(retention) * start_energy
    violation = float(contraction_softness) * torch.nn.functional.softplus(
        margin / float(contraction_softness)
    )
    # 0.5 * ||sqrt(2)*violation||^2 is the mean squared contraction violation.
    contraction_residual = (
        (2.0 / float(violation.numel())) ** 0.5 * violation.reshape(-1)
    )

    terminal_energy = end_energy[-1]
    if not 0.5 <= alpha < 1.0 or beta <= 0.0:
        raise ValueError("terminal residual requires alpha in [0.5,1) and beta > 0")
    eta = solve_smooth_cvar_eta(terminal_energy, alpha=alpha, beta=beta)
    excess = float(beta) * torch.nn.functional.softplus(
        (terminal_energy - eta) / float(beta)
    )
    eta_residual = torch.sqrt(2.0 * eta.clamp_min(0.0) + 1.0e-12).reshape(1)
    tail_residual = torch.sqrt(
        2.0 * excess
        / (max(1.0e-12, 1.0 - float(alpha)) * float(terminal_energy.numel()))
        + 1.0e-12
    )
    return torch.cat((contraction_residual, eta_residual, tail_residual))


__all__ = [
    "BOUNDARY_CODEC_VERSION",
    "LEGACY_BOUNDARY_CODEC_VERSION",
    "STRUCTURED_BOUNDARY_CODEC_VERSION",
    "ParameterVectorSpec",
    "ActionProbeBank",
    "StructuredBoundaryCodec",
    "StructuredClosedLoopState",
    "clone_closed_loop",
    "clone_l2f_state",
    "build_action_probe_bank",
    "endpoint_control_error",
    "functional_node_actions",
    "functional_probe_actions",
    "functional_trajectory_actions",
    "functional_trajectory_diagnostics",
    "functional_trajectory_identification_failures",
    "initialize_exact_endpoints",
    "make_segment_map",
    "make_structured_step_map",
    "phase_space_contraction_risk_residual",
    "load_structured_policy",
    "rollout_structured_segment",
    "structured_observation",
    "structured_global_yaw_basis",
    "terminal_risk_residual",
]
