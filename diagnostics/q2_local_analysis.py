from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from diagnostics.closed_loop_spectrum import orthogonal_projector_from_basis
from diagnostics.formal_rollout import clone_state
from diagnostics.integral_sensitivity import classify_integral_mechanism, compute_integral_sensitivity
from diagnostics.physics import branch_wrench_decomposition, motor_to_thrust
from env_l2f import L2FParams, L2FSimulator, L2FState, _so3_exp
from model import MotorGRUPolicy
from policy_observation import (
    PolicyObservationState,
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)


@dataclass
class CapturedSnapshot:
    scenario_uid: str
    step: int
    state: L2FState
    hidden: torch.Tensor
    integral: torch.Tensor
    metadata: dict[str, object]


def _select_state_row(state: L2FState, index: int) -> L2FState:
    return L2FState(**{
        name: getattr(state, name)[index : index + 1].detach().cpu().clone()
        for name in L2FState.__dataclass_fields__
    })


def state_to(state: L2FState, *, device: torch.device | str, dtype: torch.dtype) -> L2FState:
    return L2FState(**{
        name: getattr(state, name).to(device=device, dtype=dtype)
        for name in L2FState.__dataclass_fields__
    })


def stack_states(states: Sequence[L2FState], *, device: torch.device | str, dtype: torch.dtype) -> L2FState:
    return L2FState(**{
        name: torch.cat([getattr(state, name).to(device=device, dtype=dtype) for state in states], dim=0)
        for name in L2FState.__dataclass_fields__
    })


@torch.no_grad()
def capture_q2_snapshots(
    policy: MotorGRUPolicy,
    initial_state: L2FState,
    scenario_uids: Sequence[str],
    plan: pd.DataFrame,
    *,
    backend: str = "cuda",
) -> list[CapturedSnapshot]:
    required = {"scenario_uid", "step", "selection_reason", "failure_group"}
    if missing := required.difference(plan.columns):
        raise KeyError(f"snapshot plan missing {sorted(missing)}")
    if len(scenario_uids) != initial_state.position.shape[0]:
        raise ValueError("scenario UID count does not match state batch")
    plan = plan.copy()
    plan["scenario_uid"] = plan["scenario_uid"].astype(str)
    requested: dict[tuple[str, int], list[dict[str, object]]] = {}
    for record in plan.to_dict(orient="records"):
        requested.setdefault((str(record["scenario_uid"]), int(record["step"])), []).append(record)
    missing_uids = set(plan["scenario_uid"]).difference(scenario_uids)
    if missing_uids:
        raise KeyError(f"plan UIDs absent from initial state: {sorted(missing_uids)[:3]}")

    sim = L2FSimulator(L2FParams(dt=0.01))
    state = clone_state(initial_state)
    batch = state.position.shape[0]
    hidden = policy.initial_hidden(batch, device=state.position.device, dtype=state.position.dtype)
    observation_state = initial_observation_state(batch, device=state.position.device, dtype=state.position.dtype)
    uid_to_index = {uid: index for index, uid in enumerate(scenario_uids)}
    maximum_step = int(plan["step"].max())
    captured: list[CapturedSnapshot] = []
    if backend == "cuda":
        from l2f_cuda_backend import cuda_step, load_extension

        load_extension()
        step_function = lambda current, command: cuda_step(
            current, command, L2FParams(dt=0.01), grad_decay=1.0
        )
    elif backend == "torch":
        step_function = lambda current, command: sim.step(current, command, grad_decay=1.0)
    else:
        raise ValueError(f"unknown backend: {backend}")
    for step in range(1, maximum_step + 1):
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode="integral25",
            noise_max=0.0,
            integral_input_frame="body",
            integral_input_multiplier=1.0,
        )
        action, hidden = policy(observation, hidden)
        observation_state = update_position_integral(
            observation_state, observed_position, dt=0.01, integral_limit=0.5, integral_leak=0.0
        )
        state = step_function(state, action)
        for uid in scenario_uids:
            records = requested.get((uid, step))
            if not records:
                continue
            index = uid_to_index[uid]
            for record in records:
                metadata = {key: value for key, value in record.items() if key not in {"scenario_uid", "step"}}
                captured.append(CapturedSnapshot(
                    scenario_uid=uid,
                    step=step,
                    state=_select_state_row(state, index),
                    hidden=hidden[index : index + 1].detach().cpu().clone(),
                    integral=observation_state.integral_position[index : index + 1].detach().cpu().clone(),
                    metadata=metadata,
                ))
    if len(captured) != plan.shape[0]:
        raise RuntimeError(f"captured {len(captured)} snapshots for a {plan.shape[0]}-row plan")
    return captured


def snapshot_payload(snapshots: Sequence[CapturedSnapshot]) -> list[dict[str, object]]:
    return [{
        "scenario_uid": snapshot.scenario_uid,
        "step": snapshot.step,
        "state": {name: getattr(snapshot.state, name) for name in L2FState.__dataclass_fields__},
        "hidden": snapshot.hidden,
        "integral": snapshot.integral,
        "metadata": snapshot.metadata,
    } for snapshot in snapshots]


def snapshots_from_payload(payload: Sequence[Mapping[str, Any]]) -> list[CapturedSnapshot]:
    return [CapturedSnapshot(
        scenario_uid=str(record["scenario_uid"]),
        step=int(record["step"]),
        state=L2FState(**record["state"]),
        hidden=record["hidden"],
        integral=record["integral"],
        metadata=dict(record["metadata"]),
    ) for record in payload]


def run_integral_sensitivity(
    policy: MotorGRUPolicy,
    snapshots: Sequence[CapturedSnapshot],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> pd.DataFrame:
    state = stack_states([snapshot.state for snapshot in snapshots], device=device, dtype=dtype)
    hidden = torch.cat([snapshot.hidden.to(device=device, dtype=dtype) for snapshot in snapshots])
    integral = torch.cat([snapshot.integral.to(device=device, dtype=dtype) for snapshot in snapshots])

    def decomposition(value: torch.Tensor):
        observation, _ = build_policy_observation(
            state,
            PolicyObservationState(value),
            mode="integral25",
            noise_max=0.0,
            integral_input_frame="body",
            integral_input_multiplier=1.0,
        )
        _, _, auxiliary = policy.forward_with_aux(observation, hidden)
        return branch_wrench_decomposition(state, auxiliary, dt=0.01)

    baseline = decomposition(integral)
    minimum_thrust = motor_to_thrust(state, torch.full_like(state.motor, -1.0))
    maximum_thrust = motor_to_thrust(state, torch.full_like(state.motor, 1.0))
    thrust_range = maximum_thrust - minimum_thrust
    actuator_scale = torch.stack((
        thrust_range.sum(dim=-1),
        state.arm_length * thrust_range[:, 0],
        state.arm_length * thrust_range[:, 0],
        2 * state.rotor_torque_constant * thrust_range[:, 0],
    ), dim=-1)
    position_norm = torch.linalg.vector_norm(state.position, dim=-1, keepdim=True)
    position_unit = state.position / position_norm.clamp_min(1e-12)
    body_z = state.rotation[:, :, 2]
    desired_collective_sign = (-position_unit * body_z).sum(dim=-1)
    desired_wrench = torch.zeros((len(snapshots), 4), device=device, dtype=dtype)
    desired_wrench[:, 0] = desired_collective_sign
    motor_switch_margin = torch.abs(baseline.counterfactuals.full - state.motor).amin(dim=-1)
    motor_switch_nonsmooth = motor_switch_margin < 1e-5
    metadata: dict[str, Sequence[object] | torch.Tensor] = {
        "step": [snapshot.step for snapshot in snapshots],
        "snapshot_id": [
            f"{snapshot.scenario_uid}:{snapshot.step}:{snapshot.metadata.get('selection_reason', '')}"
            for snapshot in snapshots
        ],
        "failure_group": [snapshot.metadata.get("failure_group", "") for snapshot in snapshots],
        "selection_reason": [snapshot.metadata.get("selection_reason", "") for snapshot in snapshots],
        "clamped": (integral.abs() >= 0.5 - 1e-7).any(dim=-1),
        "actuator_headroom_fraction": (1.0 - baseline.counterfactuals.full.abs()).amin(dim=-1),
        "motor_switch_margin": motor_switch_margin,
        "motor_switch_nonsmooth": motor_switch_nonsmooth,
        "position_norm": position_norm.squeeze(-1),
    }
    frame = compute_integral_sensitivity(
        integral=integral,
        main_wrench_fn=lambda value: decomposition(value).main_only,
        main_plus_integral_wrench_fn=lambda value: decomposition(value).main_integral,
        full_wrench_fn=lambda value: decomposition(value).full,
        desired_wrench_direction=desired_wrench,
        integral_update_direction=state.position,
        actuator_wrench_scale=actuator_scale,
        scenario_uid=[snapshot.scenario_uid for snapshot in snapshots],
        metadata=metadata,
        nonsmooth_mask=motor_switch_nonsmooth,
    )
    instantaneous_alignment = []
    for index, row in frame.iterrows():
        direction = position_unit[index].detach().cpu().numpy()
        if bool(row["nonsmooth"]):
            induced_collective = float(row["actual_update_fd_collective"])
        else:
            d_collective = np.array([
                row["d_collective_d_ix_total"], row["d_collective_d_iy_total"], row["d_collective_d_iz_total"]
            ])
            induced_collective = float(d_collective @ direction)
        induced_world_force = induced_collective * body_z[index].detach().cpu().numpy()
        denominator = np.linalg.norm(induced_world_force)
        instantaneous_alignment.append(
            float(induced_world_force @ (-direction) / denominator) if denominator > 1e-12 else float("nan")
        )
    frame["instantaneous_world_force_cosine"] = instantaneous_alignment
    frame["integral_mechanism"] = classify_integral_mechanism(frame)
    return frame


class Q2ClosedLoopMap:
    """One-step Q2 map in a 215D local SO(3)-tangent state."""

    def __init__(self, policy: MotorGRUPolicy, snapshot: CapturedSnapshot, *, dtype: torch.dtype = torch.float64):
        self.policy = policy
        self.state = state_to(snapshot.state, device="cpu", dtype=dtype)
        self.hidden = snapshot.hidden.to(dtype=dtype)
        self.integral = snapshot.integral.to(dtype=dtype)
        hidden_dim = self.hidden.shape[-1]
        start = 0
        self.layout: dict[str, slice] = {}
        for name, size in (
            ("position", 3), ("velocity", 3), ("attitude", 3), ("omega", 3),
            ("motor", 4), ("previous_action", 4), ("hidden", hidden_dim), ("integral", 3),
        ):
            self.layout[name] = slice(start, start + size)
            start += size
        self.dimension = start
        self.simulator = L2FSimulator(L2FParams(dt=0.01))
        self.base_vector = torch.cat((
            self.state.position[0], self.state.velocity[0], torch.zeros(3, dtype=dtype),
            self.state.omega[0], self.state.motor[0], self.state.previous_action[0],
            self.hidden[0], self.integral[0],
        )).detach()
        with torch.no_grad():
            decoded = self._decode(self.base_vector)
            next_state, _, _ = self._advance(*decoded)
            self.reference_next_rotation = next_state.rotation.detach()
            observation, _ = build_policy_observation(
                self.state, PolicyObservationState(self.integral), mode="integral25",
                noise_max=0.0, integral_input_frame="body", integral_input_multiplier=1.0,
            )
            action = self.policy(observation, self.hidden)[0]
            next_unclamped = self.integral + 0.01 * self.state.position
            self.nonsmooth = bool(
                (torch.abs(torch.abs(next_unclamped) - 0.5) < 1e-6).any()
                or (torch.abs(action - self.state.motor) < 1e-6).any()
            )

    def _decode(self, vector: torch.Tensor) -> tuple[L2FState, torch.Tensor, torch.Tensor]:
        theta = vector[self.layout["attitude"]][None]
        rotation = _so3_exp(theta) @ self.state.rotation
        replacements = {
            "position": vector[self.layout["position"]][None],
            "velocity": vector[self.layout["velocity"]][None],
            "rotation": rotation,
            "omega": vector[self.layout["omega"]][None],
            "motor": vector[self.layout["motor"]][None],
            "previous_action": vector[self.layout["previous_action"]][None],
        }
        state = L2FState(**{
            name: replacements.get(name, getattr(self.state, name))
            for name in L2FState.__dataclass_fields__
        })
        return state, vector[self.layout["hidden"]][None], vector[self.layout["integral"]][None]

    def _advance(
        self, state: L2FState, hidden: torch.Tensor, integral: torch.Tensor
    ) -> tuple[L2FState, torch.Tensor, torch.Tensor]:
        observation, observed_position = build_policy_observation(
            state, PolicyObservationState(integral), mode="integral25", noise_max=0.0,
            integral_input_frame="body", integral_input_multiplier=1.0,
        )
        action, next_hidden = self.policy(observation, hidden)
        next_integral = update_position_integral(
            PolicyObservationState(integral), observed_position, dt=0.01, integral_limit=0.5, integral_leak=0.0
        ).integral_position
        return self.simulator.step(state, action, grad_decay=1.0), next_hidden, next_integral

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        next_state, next_hidden, next_integral = self._advance(*self._decode(vector))
        relative = next_state.rotation @ self.reference_next_rotation.transpose(1, 2)
        attitude = 0.5 * torch.stack((
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ), dim=-1)
        return torch.cat((
            next_state.position[0], next_state.velocity[0], attitude[0], next_state.omega[0],
            next_state.motor[0], next_state.previous_action[0], next_hidden[0], next_integral[0],
        ))

    def yaw_projector(self) -> torch.Tensor:
        gauge = torch.zeros(self.dimension, dtype=self.base_vector.dtype)
        gauge[self.layout["attitude"]] = self.state.rotation[0, :, 2]
        return orthogonal_projector_from_basis(gauge)
