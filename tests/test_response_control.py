from __future__ import annotations

from dataclasses import fields, replace
import torch

from env_l2f import L2FParams, L2FSimulator
from full_space_shooting import BoundaryLayout, FullSpaceProblem, solve_joint_sqp_step, so3_exp
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import (
    TaskLossConfig, concatenate, initialize, observation,
    rollout, sample_scenarios, task_loss,
)


class PolicyVector:
    """Test-only functional parameter packing; no training solver dependency."""
    def __init__(self, policy):
        self.items = [(name, p.shape, p.numel()) for name, p in policy.named_parameters()]

    def flatten(self, policy):
        return torch.nn.utils.parameters_to_vector(policy.parameters())

    def mapping(self, vector):
        result, start = {}, 0
        for name, shape, count in self.items:
            result[name] = vector[start:start+count].reshape(shape)
            start += count
        return result

    def install(self, policy, vector):
        torch.nn.utils.vector_to_parameters(vector, policy.parameters())


def fixture(dtype=torch.float64):
    torch.manual_seed(7)
    simulator = L2FSimulator(L2FParams())
    physical = simulator.reset(2, device=torch.device("cpu"), dtype=dtype)
    physical = replace(
        physical, position=torch.tensor(((.3, -.2, .4), (-.4, .1, .2)), dtype=dtype),
        velocity=torch.tensor(((.1, .2, -.1), (-.2, .1, .2)), dtype=dtype),
        rotation=torch.eye(3, dtype=dtype).repeat(2, 1, 1),
        omega=torch.tensor(((.2, -.1, .3), (-.1, .2, -.2)), dtype=dtype),
        motor=torch.zeros(2, 4, dtype=dtype), previous_action=torch.zeros(2, 4, dtype=dtype),
    )
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4, hidden_dim=8)).to(dtype=dtype)
    return policy, simulator, physical


def test_task_loss_updates_memory_and_controller_without_teacher():
    policy, simulator, physical = fixture()
    trace = rollout(policy, simulator, physical, 8)
    loss = task_loss(trace, TaskLossConfig())
    loss.backward()
    for prefix in ("response_encoder.", "response_memory.", "controller."):
        values = [p.grad for name, p in policy.named_parameters() if name.startswith(prefix)]
        assert all(value is not None and torch.isfinite(value).all() for value in values)
        assert sum(float(value.square().sum()) for value in values) > 1.0e-20


def test_short_rollout_task_directional_gradient_matches_finite_difference():
    policy, simulator, physical = fixture()
    vector = PolicyVector(policy)
    theta = vector.flatten(policy).detach().requires_grad_(True)
    direction = torch.linspace(-1, 1, theta.numel(), dtype=theta.dtype)
    direction = direction / direction.norm()
    config = TaskLossConfig()
    def value(parameters):
        return task_loss(rollout(policy, simulator, physical, 5,
                                 parameters=vector.mapping(parameters)), config)
    gradient = torch.autograd.grad(value(theta), theta)[0]
    analytic = (gradient * direction).sum()
    epsilon = 1.0e-5
    numeric = (value(theta + epsilon * direction) - value(theta - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close(analytic, numeric, rtol=2.0e-3, atol=1.0e-7)


def test_early_executed_action_affects_later_response_memory_and_task():
    policy, simulator, physical = fixture()
    outputs = []
    handle = policy.register_forward_hook(lambda module, inputs, output: outputs.append(output))
    trace = rollout(policy, simulator, physical, 6)
    handle.remove()
    early = outputs[0].action
    memory_gradient = torch.autograd.grad(trace.end.policy.memory.square().sum(), early,
                                          retain_graph=True)[0]
    terminal_gradient = torch.autograd.grad(trace.positions[-1].square().sum(), early)[0]
    assert float(memory_gradient.abs().max()) > 1.0e-12
    assert float(terminal_gradient.abs().max()) > 1.0e-12
    assert float(trace.end.policy.calls.min()) == 6


def test_actual_action_advance_uses_same_input_and_consumes_one_call():
    policy, simulator, physical = fixture()
    closed = initialize(policy, physical)
    obs = observation(physical, closed.policy.integral)
    proposed = policy(obs, closed.policy)
    actual = torch.full_like(proposed.action, .2)
    executed = policy(obs, closed.policy, applied_action=actual)
    assert float(closed.policy.calls.max()) == 0
    torch.testing.assert_close(executed.next_state.calls, proposed.next_state.calls)
    torch.testing.assert_close(executed.next_state.last_action, actual)
    next_physical = simulator.step(physical, actual, grad_decay=1.)
    next_obs = observation(next_physical, executed.next_state.integral)
    next_output = policy(next_obs, executed.next_state)
    assert float(next_output.next_state.calls.min()) == 2
    torch.testing.assert_close(next_output.next_state.older_action, actual)


def test_deployment_step_loop_and_training_rollout_are_identical():
    policy, simulator, physical = fixture(torch.float32)
    trace = rollout(policy, simulator, physical, 8)
    closed = initialize(policy, physical)
    actions = []
    for _ in range(8):
        obs = observation(closed.physical, closed.policy.integral)
        output = policy(obs, closed.policy)
        physical_next = simulator.step(closed.physical, output.action, grad_decay=1.)
        closed = type(closed)(physical_next, output.next_state)
        actions.append(output.action)
    assert torch.equal(torch.stack(actions), trace.actions)
    for field in fields(closed.policy):
        assert torch.equal(getattr(closed.policy, field.name), getattr(trace.end.policy, field.name))
    assert torch.equal(closed.physical.position, trace.end.physical.position)


def test_split_rollout_preserves_complete_state_and_parameter_update_restarts():
    policy, simulator, physical = fixture()
    first = rollout(policy, simulator, physical, 4)
    second = rollout(policy, simulator, first.end, 4)
    continuous = rollout(policy, simulator, physical, 8)
    assert torch.equal(concatenate([first, second]).observations, continuous.observations)
    vector = PolicyVector(policy)
    changed = vector.flatten(policy).detach() + 1.0e-4
    functional = rollout(policy, simulator, physical, 8, parameters=vector.mapping(changed))
    vector.install(policy, changed)
    restarted = rollout(policy, simulator, physical, 8)
    assert torch.equal(functional.actions, restarted.actions)
    assert float(restarted.end.policy.calls.min()) == 8


def test_privileged_truth_is_not_a_policy_input():
    policy, simulator, physical = fixture()
    changed_truth = replace(
        physical, mass=physical.mass * 2, motor=torch.ones_like(physical.motor),
        external_force=torch.ones_like(physical.external_force) * 99,
        motor_time_rising=physical.motor_time_rising * 2,
    )
    first = observation(physical, torch.zeros_like(physical.position))
    second = observation(changed_truth, torch.zeros_like(physical.position))
    assert torch.equal(first, second)
    assert torch.equal(policy(first).action, policy(second).action)


def test_task_has_no_attitude_yaw_target_but_penalizes_spinning():
    policy, simulator, physical = fixture()
    trace = rollout(policy, simulator, physical, 4)
    obs = trace.observations.clone()
    obs[:, :, 6:15] = so3_exp(torch.tensor((.3, -.2, 1.4), dtype=obs.dtype)).reshape(1, 1, 9)
    rotated = replace(trace, observations=obs)
    config = TaskLossConfig()
    torch.testing.assert_close(task_loss(trace, config), task_loss(rotated, config))
    spinning = replace(trace, omegas=trace.omegas + 20.)
    assert float(task_loss(spinning, config)) > float(task_loss(trace, config))


def test_action_box_rate_and_memory_ablation():
    policy, simulator, physical = fixture()
    trace = rollout(policy, simulator, physical, 6)
    assert bool((trace.actions.abs() <= 1).all())
    assert bool((trace.action_deltas.abs() <= policy.config.dt * policy.config.action_rate + 1e-12).all())
    ablated = rollout(policy, simulator, physical, 6, memory_enabled=False)
    assert not bool(ablated.end.policy.memory.any())


def test_sampling_is_stratified_and_does_not_change_model_rng():
    torch.manual_seed(909)
    before = torch.get_rng_state().clone()
    _, cells = sample_scenarios(16, seed=31_000_007)
    assert torch.equal(before, torch.get_rng_state())
    assert len(set(map(tuple, cells.tolist()))) == 16


def test_joint_solver_keeps_fixed_clock_in_its_internal_problem():
    initial = torch.zeros(1, 2, dtype=torch.float64)
    theta = torch.tensor([.1], dtype=torch.float64)
    boundaries = torch.tensor([[[.1, 1.]]], dtype=torch.float64)
    def segment(node, parameters):
        return torch.cat((node[..., :1] + parameters[0] * (1 + node[..., 1:]),
                          node[..., 1:] + 1), -1)
    def residual(starts, ends, parameters):
        return (ends[..., :1] - 1).reshape(-1)
    problem = FullSpaceProblem(
        initial, boundaries, theta, segment, BoundaryLayout(2), task_residual=residual,
        fixed_boundary_mask=torch.tensor([[[False, True]]]),
    )
    result = solve_joint_sqp_step(problem, linear_solver="legacy-cg", damping=1., cg_iterations=8, max_backtracks=2)
    assert torch.equal(result.boundaries[..., 1], boundaries[..., 1])
