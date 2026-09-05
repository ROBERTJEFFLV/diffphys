"""Joint task-driven full-space MS; nodes are not deployment trajectories."""
from __future__ import annotations

from dataclasses import fields, replace
import math

import torch

from env_l2f import L2FState
from full_space_shooting import FullSpaceProblem, solve_joint_sqp_step, so3_exp, so3_local_residual
from response_policy import ResponseMotorPolicy, ResponsePolicyState
from response_task import (
    ResponseClosedLoopState, TaskLossConfig, concatenate, initialize,
    prediction_residual, rollout, task_loss, task_residual,
)


class ResponseBoundaryCodec:
    """Full physical/recurrent boundary with SO(3) coordinates for both R's.

    Episode parameters/force stay in the simulator template, outside optimized
    nodes and outside policy inputs. Executed action history IS dynamic; it is
    not a frozen offline log. The clock is the only fixed boundary column.
    """
    def __init__(self, template: L2FState, memory_dim: int) -> None:
        self.template = template
        self.memory_dim = memory_dim
        self.state_dim = 37 + memory_dim
        self.clock_index = self.state_dim - 1
        self.rotation_slices = (slice(6, 9), slice(29 + memory_dim, 32 + memory_dim))

    @staticmethod
    def _log(rotation):
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand_as(rotation)
        return so3_local_residual(identity, rotation)

    def pack(self, closed: ResponseClosedLoopState) -> torch.Tensor:
        p, s = closed.physical, closed.policy
        return torch.cat((
            p.position, p.velocity, self._log(p.rotation), p.omega,
            p.motor, p.previous_action, s.memory, s.integral,
            s.previous_velocity, s.previous_omega, self._log(s.previous_rotation),
            s.older_action, s.calls,
        ), -1)

    def unpack(self, value: torch.Tensor) -> ResponseClosedLoopState:
        if value.shape[-1] != self.state_dim:
            raise ValueError("incorrect response-policy boundary width")
        m = self.memory_dim
        physical = replace(
            self.template, position=value[:, 0:3], velocity=value[:, 3:6],
            rotation=so3_exp(value[:, 6:9]), omega=value[:, 9:12],
            motor=value[:, 12:16], previous_action=value[:, 16:20],
        )
        policy = ResponsePolicyState(
            memory=value[:, 20:20+m], integral=value[:, 20+m:23+m],
            previous_velocity=value[:, 23+m:26+m], previous_omega=value[:, 26+m:29+m],
            previous_rotation=so3_exp(value[:, 29+m:32+m]),
            last_action=physical.previous_action,
            older_action=value[:, 32+m:36+m], calls=value[:, 36+m:37+m],
        )
        return ResponseClosedLoopState(physical, policy)

    def residual(self, predicted, actual):
        pieces, start = [], 0
        for sl in self.rotation_slices:
            pieces.append(predicted[..., start:sl.start] - actual[..., start:sl.start])
            pieces.append(so3_local_residual(so3_exp(predicted[..., sl]), so3_exp(actual[..., sl])))
            start = sl.stop
        pieces.append(predicted[..., start:] - actual[..., start:])
        return torch.cat(pieces, -1)


class PolicyVector:
    """ALL policy parameters, including memory and first-order motor feedback."""
    def __init__(self, policy: ResponseMotorPolicy) -> None:
        self.names = tuple(name for name, _ in policy.named_parameters())
        self.shapes = tuple(p.shape for p in policy.parameters())
        self.sizes = tuple(p.numel() for p in policy.parameters())

    def flatten(self, policy):
        return torch.cat([p.reshape(-1) for p in policy.parameters()])

    def mapping(self, theta):
        return {name: value.reshape(shape) for name, shape, value in zip(
            self.names, self.shapes, theta.split(self.sizes)
        )}

    @torch.no_grad()
    def install(self, policy, theta):
        for name, value in self.mapping(theta).items():
            dict(policy.named_parameters())[name].copy_(value)


def make_problem(policy, simulator, initial, loss_config, *, segment_steps, segments):
    codec = ResponseBoundaryCodec(initial, policy.config.memory_dim)
    vector = PolicyVector(policy)
    theta = vector.flatten(policy).detach().requires_grad_(True)
    # Only initialize independent node guesses without a graph. Every segment
    # map below, including segment0/call0, is recomputed differentiably.
    with torch.no_grad():
        closed = initialize(policy, initial)
        initial_node = codec.pack(closed)
        endpoints, records = [], []
        for _ in range(segments):
            trace = rollout(policy, simulator, closed, segment_steps)
            closed = trace.end
            endpoints.append(codec.pack(closed))
            records.append(trace)
        record = concatenate(records)
    boundaries = torch.stack(endpoints).detach().requires_grad_(True)

    def segment_map(node, parameters):
        trace = rollout(policy, simulator, codec.unpack(node), segment_steps,
                        parameters=vector.mapping(parameters))
        return codec.pack(trace.end)

    def residual(starts, ends, parameters):
        traces = [
            rollout(policy, simulator, codec.unpack(node), segment_steps,
                    parameters=vector.mapping(parameters))
            for node in starts.unbind(0)
        ]
        values = [task_residual(concatenate(traces), loss_config)]
        if loss_config.prediction_weight:
            values.append(math.sqrt(2 * loss_config.prediction_weight) * prediction_residual(
                policy, record.observations, record.actions, parameters=vector.mapping(parameters)
            ))
        return torch.cat(values)

    fixed_mask = torch.zeros_like(boundaries, dtype=torch.bool)
    fixed_mask[..., codec.clock_index] = True
    problem = FullSpaceProblem(initial_node, boundaries, theta, segment_map, codec,
                               task_residual=residual, fixed_boundary_mask=fixed_mask)
    return problem, codec, vector


def task_shooting_step(
    policy, simulator, initial, development_initial, loss_config, *,
    segment_steps=250, segments=2, damping=100.0, cg_iterations=16,
    parameter_radius=0.05, action_radius=0.01, action_max_radius=0.05, debug_solver=False,
):
    problem, codec, vector = make_problem(
        policy, simulator, initial, loss_config,
        segment_steps=segment_steps, segments=segments,
    )
    horizon = segment_steps * segments
    theta_before = problem.theta.detach()

    def continuous(parameters, state):
        # ALWAYS reconstruct recurrent state from the episode's initial
        # observation under these parameters, never splice in old hidden state.
        return rollout(policy, simulator, state, horizon, parameters=vector.mapping(parameters))

    with torch.no_grad():
        before = continuous(theta_before, initial)
        dev_before = continuous(theta_before, development_initial)
        before_value = float(task_loss(before, loss_config))
        dev_before_value = float(task_loss(dev_before, loss_config))

    step = solve_joint_sqp_step(
        problem, damping=damping, penalty=10.0, cg_iterations=cg_iterations,
        parameter_radius=parameter_radius,
        parameter_scale=theta_before.abs().clamp_min(1.0e-2),
        action_radius=action_radius, action_max_radius=action_max_radius,
        action_evaluator=lambda parameters, nodes: continuous(parameters, initial).actions,
        max_backtracks=8, minimum_ratio=0.25,
    )
    with torch.no_grad():
        after = continuous(step.theta, initial)
        dev_after = continuous(step.theta, development_initial)
        after_value = float(task_loss(after, loss_config))
        dev_after_value = float(task_loss(dev_after, loss_config))
        restored, closed = [], initialize(policy, initial)
        for _ in range(segments):
            part = rollout(policy, simulator, closed, segment_steps,
                           parameters=vector.mapping(step.theta))
            closed = part.end
            restored.append(codec.pack(closed))
        restored = torch.stack(restored)
        continuity = float(problem.defects(restored, step.theta).abs().max())
    accepted = bool(
        torch.isfinite(step.theta).all()
        and math.isfinite(after_value) and math.isfinite(dev_after_value)
        and math.isfinite(continuity)
        and after_value < before_value and dev_after_value <= dev_before_value
    )
    if accepted:
        vector.install(policy, step.theta)
    # Solver telemetry is debugging information, not a second performance gate.
    evidence = {
        "continuous_loss_before": before_value, "continuous_loss_after": after_value,
        "heldout_loss_before": dev_before_value, "heldout_loss_after": dev_after_value,
        "continuity_defect_after_restoration": continuity, "accepted": accepted,
    }
    if debug_solver:
        evidence["solver_debug"] = {
            field.name: getattr(step, field.name)
            for field in fields(step) if field.name not in ("theta", "boundaries")
        }
    state = {
        "damping": min(1.0e12, max(1.0e-6, damping * (0.7 if accepted else 2.0))),
        "theta": (step.theta if accepted else theta_before).detach(),
        "restored_boundaries": restored.detach() if accepted else problem.boundaries.detach(),
        "initial_state": {f.name: getattr(initial, f.name).detach() for f in fields(initial)},
        "last_evidence": evidence,
    }
    return evidence, state
