"""The agile objective uses the original full physical/recurrent graph."""
from dataclasses import replace
import copy

import pytest
import torch

from diffphys.agile_mission import AgileMissionConfig, sample_agile_mission
from diffphys.agile_objective import AGILE_COMPONENTS, AgileLossConfig, agile_terminal
from diffphys.env_raptor import RaptorSimulator
from diffphys.response_adjoints import collect_rollout, backward_actor
from diffphys.response_groups import GroupBalanceConfig
from diffphys.response_noise import DisturbanceConfig
from diffphys.response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from diffphys.response_task import initialize, rollout, _decay_closed_state, _select_rows


def setup_case(horizon=61, count=4):
    torch.manual_seed(412)
    simulator = RaptorSimulator()
    raw = simulator.reset(count, seed=17, horizon=horizon, dtype=torch.float64,
                          disturbances=DisturbanceConfig.clean())
    initial, mission = sample_agile_mission(
        raw, AgileMissionConfig(hold_seconds=.02, hover_fraction=1.),
        seed=12, horizon=horizon)
    # A controlled geometric reference exercises a real RK4 rollout; actuator
    # feasibility of sampled missions has separate numerical tests.
    direction = torch.zeros_like(initial.position)
    direction[:, 0] = 1.
    initial = replace(initial, position=-.15*direction, velocity=direction.clone(),
                      previous_velocity=direction[:, None].expand(-1, 3, -1).clone())
    duration = max(2, 2*((horizon-2)//2))*.01
    mission = replace(mission, start=initial.position.clone(),
                      waypoint=torch.zeros_like(direction), goal=.15*direction,
                      direction=direction, velocity=(315*.15/(64*duration))*direction,
                      center=torch.zeros_like(direction), bend=torch.zeros_like(direction),
                      duration=torch.full((count,), duration, dtype=torch.float64),
                      is_hover=torch.zeros(count, dtype=torch.bool),
                      feasible=torch.ones(count, dtype=torch.bool),
                      half_span=torch.full((count,), .15, dtype=torch.float64),
                      rotation=initial.rotation.clone())
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8, goal_conditioning=True)).double()
    with torch.no_grad():
        policy.readout.weight.mul_(.01)
    return policy, simulator, initial, mission, AgileLossConfig()


def raw_loss(trace, mission, config):
    return (sum(trace.agile_components.values()).sum(0)
            + sum(agile_terminal(trace.end.physical, mission, config).values()))


@pytest.mark.parametrize("decay", [0., .2])
def test_chunked_costs_and_all_parameter_gradients_equal_one_continuous_rollout(decay):
    policy, simulator, initial, mission, config = setup_case(count=32)
    reference = copy.deepcopy(policy)
    trace = rollout(reference, simulator, initial, 61, mission=mission,
                    agile_config=config, horizon=61, time_decay=decay)
    assert trace.end.elapsed_steps == 61
    record = collect_rollout(policy, simulator, initial, config, horizon=61,
                             mission=mission, time_decay=decay,
                             group_config=GroupBalanceConfig(backward_backend="vjp"))
    expected = raw_loss(trace, mission, config)
    torch.testing.assert_close(record.costs, expected, rtol=1e-12, atol=1e-12)
    record.costs.mean().backward()
    expected.mean().backward()
    for actual, target in zip(policy.parameters(), reference.parameters()):
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, target.grad, rtol=1e-9, atol=1e-9)
    assert set(record.metrics["task_components"]) == set(AGILE_COMPONENTS)
    assert sum(record.metrics["task_components"].values()) == pytest.approx(
        record.metrics["task_objective"], rel=1e-12)
    assert record.metrics["scenario_count"] == len(initial.position)


def test_planned_clock_survives_decay_and_resume_without_a_state_based_phase():
    policy, simulator, initial, mission, config = setup_case(horizon=9)
    first = rollout(policy, simulator, initial, 5, mission=mission,
                    agile_config=config, horizon=9)
    assert first.end.elapsed_steps == 5
    decayed = _decay_closed_state(first.end, .9)
    assert decayed.elapsed_steps == 5
    second = rollout(policy, simulator, decayed, 4, mission=mission,
                     agile_config=config, horizon=9)
    assert second.end.elapsed_steps == 9
    assert second.observations.shape == (5, 4, 22)
    assert not hasattr(second.end, "agile")


def test_agile_raw_gradient_matches_finite_difference_through_physics():
    policy, simulator, initial, mission, config = setup_case(horizon=8, count=1)

    def cost():
        trace = rollout(policy, simulator, initial, 8, mission=mission,
                        agile_config=config, horizon=8, time_decay=0.)
        return raw_loss(trace, mission, config).mean()

    value = cost()
    analytic = torch.autograd.grad(value, policy.readout.bias)[0][0]
    delta = 1e-6
    with torch.no_grad():
        original = policy.readout.bias[0].item()
        policy.readout.bias[0] = original+delta
        plus = cost().item()
        policy.readout.bias[0] = original-delta
        minus = cost().item()
        policy.readout.bias[0] = original
    assert abs(analytic) > 1e-8
    assert analytic.item() == pytest.approx((plus-minus)/(2*delta), rel=2e-5, abs=2e-6)


@pytest.mark.parametrize("backend", ["probe", "vjp"])
def test_failure_compaction_keeps_all_final_residuals_and_backpropagates(backend):
    policy, simulator, initial, mission, config = setup_case(horizon=8, count=32)
    initial = replace(initial, position=initial.position.clone())
    initial.position[0, 0] = initial.position_limit[0]-.001
    record = collect_rollout(policy, simulator, initial, config, horizon=8,
                             mission=mission, time_decay=.2,
                             group_config=GroupBalanceConfig(backward_backend=backend))
    assert record.costs.shape == (32,)
    assert record.metrics["boundary_failure_rate"] >= 1/32
    assert record.metrics["physical_transitions"] < 256
    assert record.metrics["task_components"]["final_position"] > 0
    backward_actor(policy, simulator, record, config)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in policy.parameters())
    assert policy.response_memory.weight_ih.grad[:, 16:].norm() > 0


def test_goal_policy_requires_mission_and_whole_episode_deadline():
    policy, simulator, initial, mission, config = setup_case(horizon=8)
    with pytest.raises(ValueError, match="mission"):
        initialize(policy, initial)
    with pytest.raises(ValueError, match="horizon"):
        rollout(policy, simulator, initial, 3, mission=mission, agile_config=config)
    first = rollout(policy, simulator, initial, 5, mission=mission,
                    agile_config=config, horizon=8)
    with pytest.raises(ValueError, match="horizon"):
        rollout(policy, simulator, first.end, 4, mission=mission,
                agile_config=config, horizon=8)


@pytest.mark.parametrize("decay", [0., .2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_group_probe_matches_explicit_vjp_for_the_complete_agile_loss(decay, dtype):
    policy, simulator, initial, mission, config = setup_case(horizon=12, count=64)
    policy = policy.to(dtype=dtype)
    initial, mission = initial.to("cpu", dtype), mission.to("cpu", dtype)
    reference = copy.deepcopy(policy)
    outputs = []
    for actor, backend in ((policy, "probe"), (reference, "vjp")):
        record = collect_rollout(actor, simulator, initial, config, horizon=12,
                                 mission=mission, time_decay=decay,
                                 group_config=GroupBalanceConfig(backward_backend=backend))
        outputs.append((record.costs.detach(), record.metrics))
        backward_actor(actor, simulator, record, config)
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1] == outputs[1][1]
    tolerance = dict(rtol=3e-4, atol=2e-6) if dtype == torch.float32 else dict(rtol=3e-10, atol=1e-11)
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
    assert not policy._forward_hooks and not policy.response_memory._forward_hooks


def test_no_grad_evaluation_accepts_a_small_bank_without_training_group_quotas():
    policy, simulator, initial, mission, config = setup_case(horizon=8, count=1)
    with torch.no_grad():
        record = collect_rollout(policy, simulator, initial, config, horizon=8, mission=mission)
    assert record.probe is None
    assert record.metrics["scenario_count"] == 1
    assert record.metrics["finite"]
    assert not record.costs.requires_grad


@pytest.mark.parametrize("decay", [0., .2])
def test_failed_row_tail_gradient_does_not_depend_on_another_aircraft_surviving(decay):
    policy, simulator, original, mission, config = setup_case(count=2)
    results = []
    for count in (1, 2):
        indices = torch.arange(count)
        selected = _select_rows(original, indices)
        position = selected.position.detach().clone()
        position[0, 0] = selected.position_limit[0]-.001
        position.requires_grad_()
        initial = replace(selected, position=position)
        task = mission.select(indices)
        trace = rollout(policy, simulator, initial, 61, mission=task,
                        agile_config=config, horizon=61, time_decay=decay)
        assert trace.valid[:, 0].sum() == 1
        cost = raw_loss(trace, task, config)[0]
        gradient = torch.autograd.grad(cost, position)[0][0]
        results.append((cost.detach(), gradient))
    for single, mixed in zip(results[0], results[1]):
        torch.testing.assert_close(single, mixed, rtol=1e-11, atol=1e-11)
