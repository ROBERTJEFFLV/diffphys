from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import clone_state, load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from equilibrium_control import analytic_equilibrium_target, body_z_error_body_frame  # noqa: E402
from policy_observation import (  # noqa: E402
    PolicyObservationState,
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)
from structured_policy import (  # noqa: E402
    StructuredPolicyConfig,
    StructuredPolicyState,
    StructuredRecurrentPolicy,
    effective_wrench_mixer,
    reference_fast_gain,
)
from structured_rollout import (  # noqa: E402
    StructuredClosedLoopState,
    rollout_structured_segment,
    structured_observation,
)
from structured_checkpoint import CADENCE_SEMANTICS_VERSION  # noqa: E402


DEFAULT_Q2 = (
    ROOT
    / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
)


@dataclass
class TeacherReplay:
    observations: torch.Tensor
    actions: torch.Tensor
    motors: torch.Tensor
    features: torch.Tensor
    capability: torch.Tensor
    trim: torch.Tensor
    body_z: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill the complete Q2 behavior into the structured policy.")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/structured_q2_distilled.pt")
    parser.add_argument("--report", type=Path, default=ROOT / "runs/structured_q2_distillation.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--identifier-dim", type=int, default=64)
    parser.add_argument("--slow-cadence", type=int, default=25)
    parser.add_argument("--detach-cadence", type=int, default=25)
    parser.add_argument("--jacobian-samples", type=int, default=2)
    parser.add_argument("--gain-init", choices=("reference-lqr", "teacher-fit"), default="reference-lqr")
    parser.add_argument(
        "--allocator-solver", choices=("smooth_dls", "box_qp"), default="smooth_dls",
    )
    parser.add_argument("--allocator-rate-limit", type=float, default=0.0)
    parser.add_argument(
        "--residual-scale", type=float, default=0.0,
        help="structured residual scale; nonzero is intended only for explicit smoke/distillation runs",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _raw_capability(state: L2FState) -> torch.Tensor:
    return torch.stack(
        (
            state.thrust_to_weight,
            state.alpha_roll_max,
            state.eta_yaw,
            state.jz_over_jxy,
            state.motor_time_rising,
            state.motor_time_falling,
        ),
        dim=-1,
    )


@torch.no_grad()
def collect_teacher_replay(
    teacher,
    simulator: L2FSimulator,
    initial: L2FState,
    *,
    horizon: int,
    error_scales: torch.Tensor,
) -> TeacherReplay:
    state = clone_state(initial)
    batch = state.position.shape[0]
    hidden = teacher.initial_hidden(batch, device=state.position.device, dtype=state.position.dtype)
    observation_state = initial_observation_state(
        batch, device=state.position.device, dtype=state.position.dtype
    )
    target = analytic_equilibrium_target(state, gravity=simulator.params.gravity)
    capability = _raw_capability(state)
    mixer = effective_wrench_mixer(capability)
    observations: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    motors: List[torch.Tensor] = []
    features: List[torch.Tensor] = []
    for _ in range(horizon):
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode="integral25",
            noise_max=0.0,
            integral_input_frame="body",
            integral_input_multiplier=1.0,
        )
        action, hidden = teacher(observation, hidden)
        tilt = body_z_error_body_frame(state.rotation, target.body_z)[:, :2]
        error = torch.cat(
            (
                state.position,
                state.velocity,
                tilt,
                state.omega,
                state.motor - target.motor_trim,
            ),
            dim=-1,
        )
        observations.append(observation)
        actions.append(action)
        motors.append(state.motor)
        features.append(torch.tanh(error / error_scales.to(error)))
        observation_state = update_position_integral(
            observation_state,
            observed_position,
            dt=simulator.params.dt,
            integral_limit=0.5,
            integral_leak=0.0,
        )
        state = simulator.step(state, action, grad_decay=1.0)
    return TeacherReplay(
        observations=torch.stack(observations),
        actions=torch.stack(actions),
        motors=torch.stack(motors),
        features=torch.stack(features),
        capability=capability,
        trim=target.motor_trim,
        body_z=target.body_z,
    )


def fit_teacher_wrench_gain(replay: TeacherReplay, ridge: float = 1.0e-3) -> torch.Tensor:
    mixer = effective_wrench_mixer(replay.capability)
    action_delta = replay.actions - replay.trim.unsqueeze(0)
    wrench = torch.einsum("bij,tbj->tbi", mixer, action_delta)
    features = replay.features.reshape(-1, replay.features.shape[-1]).double()
    targets = wrench.reshape(-1, wrench.shape[-1]).double()
    gram = features.T @ features
    rhs = features.T @ targets
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    solution = torch.linalg.solve(gram + float(ridge) * identity, rhs)
    return solution.T.to(replay.features).clamp(-100.0, 100.0)


def _log_capability_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(
        prediction.clamp_min(1.0e-6).log(),
        target.clamp_min(1.0e-6).log(),
    )


def replay_student_loss(
    student: StructuredRecurrentPolicy,
    replay: TeacherReplay,
    *,
    detach_cadence: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    recurrent: StructuredPolicyState = student.initial_state(replay.observations[0])
    action_loss = replay.actions.sum() * 0.0
    motor_loss = action_loss
    capability_loss = action_loss
    trim_loss = action_loss
    direction_loss = action_loss
    for step in range(replay.observations.shape[0]):
        output = student.forward_with_aux(replay.observations[step], recurrent)
        action_loss = action_loss + F.smooth_l1_loss(output.action, replay.actions[step])
        motor_loss = motor_loss + F.smooth_l1_loss(
            output.auxiliary["motor_estimate"], replay.motors[step]
        )
        capability_loss = capability_loss + _log_capability_loss(
            output.auxiliary["capability"], replay.capability
        )
        trim_loss = trim_loss + F.smooth_l1_loss(
            output.auxiliary["trim_action"], replay.trim
        )
        direction_loss = direction_loss + F.smooth_l1_loss(
            output.auxiliary["body_z"], replay.body_z
        )
        recurrent = output.next_state
        if detach_cadence > 0 and (step + 1) % detach_cadence == 0:
            recurrent = recurrent.detach()
    count = float(replay.observations.shape[0])
    action_loss = action_loss / count
    motor_loss = motor_loss / count
    capability_loss = capability_loss / count
    trim_loss = trim_loss / count
    direction_loss = direction_loss / count
    total = (
        action_loss
        + 0.10 * motor_loss
        + 0.02 * capability_loss
        + 0.10 * trim_loss
        + 0.05 * direction_loss
    )
    return total, {
        "action": float(action_loss.detach()),
        "motor": float(motor_loss.detach()),
        "capability": float(capability_loss.detach()),
        "trim": float(trim_loss.detach()),
        "direction": float(direction_loss.detach()),
    }


@torch.no_grad()
def replay_action_parity(student: StructuredRecurrentPolicy, replay: TeacherReplay) -> Dict[str, float]:
    recurrent = student.initial_state(replay.observations[0])
    differences = []
    for step in range(replay.observations.shape[0]):
        action, recurrent = student(replay.observations[step], recurrent)
        differences.append(action - replay.actions[step])
    difference = torch.stack(differences)
    return {
        "action_rms": float(torch.sqrt(difference.square().mean())),
        "action_max_abs": float(difference.abs().max()),
    }


@torch.no_grad()
def _closed_loop_metrics(
    simulator: L2FSimulator,
    policy,
    initial: L2FState,
    *,
    horizon: int,
    structured: bool,
) -> Dict[str, float]:
    if structured:
        observation = torch.cat(
            (
                initial.position,
                initial.velocity,
                initial.rotation.reshape(initial.position.shape[0], 9),
                initial.omega,
                torch.zeros_like(initial.position),
                initial.previous_action,
            ),
            dim=-1,
        )
        closed = StructuredClosedLoopState(clone_state(initial), policy.initial_state(observation))
        end, trace = rollout_structured_segment(policy, simulator, closed, steps=horizon, collect=True)
        physical = end.physical
        positions = torch.linalg.vector_norm(trace["position"], dim=-1)
        velocities = torch.linalg.vector_norm(trace["velocity"], dim=-1)
        omegas = torch.linalg.vector_norm(trace["omega"], dim=-1)
    else:
        state = clone_state(initial)
        hidden = policy.initial_hidden(
            state.position.shape[0], device=state.position.device, dtype=state.position.dtype
        )
        observation_state = initial_observation_state(
            state.position.shape[0], device=state.position.device, dtype=state.position.dtype
        )
        p_values, v_values, w_values = [], [], []
        for _ in range(horizon):
            observation, observed_position = build_policy_observation(
                state, observation_state, mode="integral25", integral_input_frame="body"
            )
            action, hidden = policy(observation, hidden)
            observation_state = update_position_integral(
                observation_state, observed_position, dt=simulator.params.dt,
                integral_limit=0.5, integral_leak=0.0,
            )
            state = simulator.step(state, action)
            p_values.append(torch.linalg.vector_norm(state.position, dim=-1))
            v_values.append(torch.linalg.vector_norm(state.velocity, dim=-1))
            w_values.append(torch.linalg.vector_norm(state.omega, dim=-1))
        physical = state
        positions, velocities, omegas = map(torch.stack, (p_values, v_values, w_values))
    tail = min(100, horizon)
    return {
        "tail_position": float(positions[-tail:].mean()),
        "tail_velocity": float(velocities[-tail:].mean()),
        "tail_omega": float(omegas[-tail:].mean()),
        "final_position": float(torch.linalg.vector_norm(physical.position, dim=-1).mean()),
    }


def _jacobian_parity(teacher, student, replay: TeacherReplay, samples: int) -> Dict[str, float]:
    if samples <= 0:
        return {"jacobian_relative_mean": float("nan")}
    indices = torch.linspace(
        0, replay.observations.shape[0] - 1, samples,
        device=replay.observations.device,
    ).long()
    teacher_hidden = teacher.initial_hidden(
        replay.observations.shape[1],
        device=replay.observations.device,
        dtype=replay.observations.dtype,
    )
    student_state = student.initial_state(replay.observations[0])
    selected = set(int(value) for value in indices.tolist())
    errors = []
    for step in range(replay.observations.shape[0]):
        observation = replay.observations[step]
        if step in selected:
            # Use one representative scenario to keep this migration diagnostic cheap.
            obs = observation[:1].detach().requires_grad_(True)
            teacher_h = teacher_hidden[:1].detach()
            student_s = student_state.detach()
            # Structured recurrent state evolves as the architecture grows.
            # Slice every tensor field generically so this legacy smoke
            # diagnostic cannot retain a stale batch-sized observer field.
            for field in fields(student_s):
                name = field.name
                value = getattr(student_s, name, None)
                if torch.is_tensor(value):
                    setattr(student_s, name, value[:1])
            teacher_j = torch.autograd.functional.jacobian(
                lambda value: teacher(value, teacher_h)[0], obs, create_graph=False
            ).reshape(4, 25)
            student_j = torch.autograd.functional.jacobian(
                lambda value: student(value, student_s)[0], obs, create_graph=False
            ).reshape(4, 25)
            errors.append(
                torch.linalg.vector_norm(student_j - teacher_j)
                / torch.linalg.vector_norm(teacher_j).clamp_min(1.0e-8)
            )
        with torch.no_grad():
            _, teacher_hidden = teacher(observation, teacher_hidden)
            _, student_state = student(observation, student_state)
    values = torch.stack(errors)
    return {
        "jacobian_relative_mean": float(values.mean()),
        "jacobian_relative_max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    if args.horizon < 1 or args.updates < 0 or args.batch_size < 1:
        raise ValueError("horizon/batch must be positive and updates non-negative")
    if args.residual_scale < 0.0:
        raise ValueError("residual scale must be non-negative")
    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    teacher, checkpoint_args = load_q2_policy(
        args.source_checkpoint, device=device, dtype=torch.float32
    )
    simulator = L2FSimulator(L2FParams(dt=float(checkpoint_args.get("dt", 0.01))))
    initial = simulator.reset(
        args.batch_size,
        device=device,
        dtype=torch.float32,
        sample_dynamics=True,
        sampled_dynamics_level="broad",
        broad_sampler="physical-fit",
        balanced_dynamics_sampling=args.batch_size % 256 == 0,
        sample_external_force=True,
    )
    config = StructuredPolicyConfig(
        hidden_dim=args.hidden_dim,
        identifier_dim=args.identifier_dim,
        dt=simulator.params.dt,
        slow_cadence=args.slow_cadence,
        allocator_solver=args.allocator_solver,
        allocator_rate_limit=args.allocator_rate_limit,
        residual_scale=args.residual_scale,
        residual_trainable=args.residual_scale > 0.0,
    )
    student = StructuredRecurrentPolicy(config).to(device)
    replay = collect_teacher_replay(
        teacher, simulator, initial, horizon=args.horizon, error_scales=student.error_scales
    )
    fitted_gain = fit_teacher_wrench_gain(replay)
    installed_gain = (
        fitted_gain
        if args.gain_init == "teacher-fit"
        else reference_fast_gain(device=device, dtype=torch.float32)
    )
    if not student.fast_feedback.install_verified_gain(installed_gain):
        raise RuntimeError("the initial fast gain failed the finite/nonzero migration gate")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1.0e-5)
    history = []
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, components = replay_student_loss(
            student, replay, detach_cadence=args.detach_cadence
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("non-finite distillation gradient")
        optimizer.step()
        history.append(
            {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                **components,
            }
        )

    parity = replay_action_parity(student, replay)
    jacobian = _jacobian_parity(teacher, student, replay, args.jacobian_samples)
    evaluation_horizon = min(500, max(args.horizon, 1))
    q2_metrics = _closed_loop_metrics(
        simulator, teacher, initial, horizon=evaluation_horizon, structured=False
    )
    student_metrics = _closed_loop_metrics(
        simulator, student, initial, horizon=evaluation_horizon, structured=True
    )
    relative = {
        name: (student_metrics[name] / max(q2_metrics[name], 1.0e-12) - 1.0)
        for name in q2_metrics
    }
    gates = {
        "fast_gain_nonzero_verified": student.fast_feedback.verified,
        "action_rms_le_5e-4": parity["action_rms"] <= 5.0e-4,
        "action_max_le_2e-3": parity["action_max_abs"] <= 2.0e-3,
        "jacobian_mean_le_2pct": jacobian["jacobian_relative_mean"] <= 0.02,
        "rollout_metrics_within_2pct": all(value <= 0.02 for value in relative.values()),
    }
    payload = {
        "phase": "q2-recurrent-distillation",
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "updates": args.updates,
        "config": asdict(config),
        "gain_initialization": args.gain_init,
        "installed_gain_norm": float(torch.linalg.vector_norm(installed_gain)),
        "fitted_gain_norm": float(torch.linalg.vector_norm(fitted_gain)),
        "history": history,
        "replay_parity": parity,
        "jacobian_parity": jacobian,
        "q2_metrics": q2_metrics,
        "student_metrics": student_metrics,
        "student_relative_to_q2": relative,
        "gates": gates,
        "migration_gate_passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "structured-recurrent-motor-policy",
            "model": student.state_dict(),
            "config": asdict(config),
            "fast_feedback_verified": student.fast_feedback.verified,
            "report": payload,
        },
        args.output,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "report": str(args.report), **payload["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
