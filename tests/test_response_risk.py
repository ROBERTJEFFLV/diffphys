from __future__ import annotations

import copy
from dataclasses import fields, replace
import math

import pytest
import torch

import response_critic as critic
import response_task as task
from test_response_control import fixture


def risk_api():
    assert hasattr(task, "RiskConfig"), "missing physical risk objective"
    assert hasattr(critic, "RiskToGoCritic"), "Critic still predicts task cost"


def test_risk_is_limit_relative_and_reference_relative_with_finite_gradient():
    risk_api()
    policy, simulator, initial = fixture()
    trace = task.rollout(policy, simulator, initial, 1)
    p = torch.tensor([[[5., 0., 0.], [10., 0., 0.]]], dtype=torch.float64, requires_grad=True)
    omega = torch.full_like(p, 20.)
    trace = replace(trace, positions=p, velocities=torch.zeros_like(p), omegas=omega,
                    actions=torch.zeros_like(trace.actions))
    config = task.RiskConfig(position_limit=5., sharpness=10.)
    risks = task.risk_components(trace, config, omega_reference=omega)
    torch.testing.assert_close(risks["position"], p.new_tensor([[math.log(2)/10, math.log1p(math.exp(10))/10]]))
    torch.testing.assert_close(risks["omega"], p.new_full((1, 2), math.log1p(math.exp(-10))/10))
    gradient = torch.autograd.grad(risks["position"].sum(), p)[0]
    torch.testing.assert_close(gradient[0, 0], p.new_tensor([.1, 0., 0.]))
    assert torch.isfinite(gradient).all() and gradient[0, 1, 0] > gradient[0, 0, 0]
    scaled = task.risk_components(replace(trace, positions=2*p), replace(config, position_limit=10.))
    torch.testing.assert_close(scaled["position"], risks["position"])


def test_exact_suffix_risk_has_zero_terminal_and_collection_uses_risk_not_performance():
    risk_api()
    torch.testing.assert_close(task.suffix_risks(torch.tensor([[1., 4.], [2., 5.], [3., 6.]])),
                               torch.tensor([[6., 15.], [5., 11.], [3., 6.], [0., 0.]]))
    policy, simulator, initial = fixture()
    config = task.RiskConfig()
    record = critic.collect_trajectory(policy, simulator, initial, 6, task.TaskLossConfig(), risk_config=config)
    full = task.rollout(policy, simulator, initial, 6)
    assert torch.equal(record.trajectory.actions, full.actions)
    torch.testing.assert_close(record.returns[0].sum(-1), task.step_risks(full, config).sum(0))
    assert not torch.allclose(record.returns[0].sum(-1), task.scenario_costs(full, task.TaskLossConfig()))
    assert not record.inputs.requires_grad and not record.returns.requires_grad
    assert (record.returns[-1] == 0).all()


def test_critic_normalizes_truth_and_histories_with_fixed_scales_and_is_nonnegative():
    risk_api()
    policy, _, initial = fixture()
    closed = task.initialize(policy, initial)
    original = critic.critic_features(closed, 2, 6)
    changed = replace(closed, physical=replace(initial, position=initial.position + 5.,
                                               mass=initial.mass + .05),
                      policy=replace(closed.policy, previous_omega=closed.policy.previous_omega + 10.))
    difference = critic.critic_features(changed, 2, 6) - original
    nonzero = difference[difference.abs() > 1e-12]
    torch.testing.assert_close(nonzero, torch.ones_like(nonzero))
    net = critic.RiskToGoCritic(original.shape[-1]).double()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.network[-2].bias.fill_(-20.)
    assert (net(original) >= 0).all()
    assert torch.isfinite(net(original)).all()


def test_ranking_loss_pushes_true_order_ignores_ties_and_detaches_truth():
    risk_api()
    assert hasattr(critic, "direction_ranking_loss"), "missing direction supervision"
    plus = torch.tensor([0., 1., 5.], dtype=torch.float64, requires_grad=True)
    minus = torch.tensor([1., 0., -5.], dtype=torch.float64, requires_grad=True)
    truth_plus = torch.tensor([2., 0., 1.], dtype=torch.float64, requires_grad=True)
    truth_minus = torch.tensor([0., 2., 1.], dtype=torch.float64, requires_grad=True)
    loss = critic.direction_ranking_loss(plus, minus, truth_plus, truth_minus, temperature=.5)
    loss.backward()
    assert plus.grad[0] < 0 and plus.grad[1] > 0 and plus.grad[2] == 0
    assert minus.grad[0] > 0 and minus.grad[1] < 0 and minus.grad[2] == 0
    assert truth_plus.grad is None and truth_minus.grad is None
    tied = critic.direction_ranking_loss(plus, minus, truth_plus, truth_plus, temperature=.5)
    assert tied.item() == 0 and torch.isfinite(tied)


def test_motor_direction_states_are_reachable_and_preserve_executed_action_history():
    risk_api()
    policy, simulator, initial = fixture()
    closed = task.rollout(policy, simulator, initial, 2).end
    obs = task.observation(closed.physical, closed.policy.integral)
    base = policy(obs, closed.policy).action
    delta = torch.zeros_like(base)
    delta[:, 2] = .01
    plus, minus = critic.motor_perturbation_states(policy, simulator, closed, delta)
    for state, sign in [(plus, 1), (minus, -1)]:
        executed = base + sign * delta
        expected = simulator.step(closed.physical, executed, grad_decay=1.)
        for field in fields(expected):
            torch.testing.assert_close(getattr(state.physical, field.name), getattr(expected, field.name))
        torch.testing.assert_close(state.policy.last_action, executed)
        torch.testing.assert_close(state.policy.older_action, closed.physical.previous_action)
        torch.testing.assert_close(state.policy.calls, closed.policy.calls + 1)
        assert not state.physical.position.requires_grad and not state.policy.memory.requires_grad
    # Requests larger than actuator/rate headroom remain symmetric and legal.
    plus, minus = critic.motor_perturbation_states(policy, simulator, closed, torch.full_like(base, 4.))
    torch.testing.assert_close(plus.physical.previous_action + minus.physical.previous_action, 2*base)
    for state in (plus, minus):
        assert (state.physical.previous_action.abs() <= 1).all()
        assert ((state.physical.previous_action - closed.physical.previous_action).abs() <=
                policy.config.action_rate * policy.config.dt + 1e-12).all()


def test_direction_labels_continue_to_original_horizon_without_prefix_risk():
    risk_api()
    policy, simulator, initial = fixture()
    closed = task.rollout(policy, simulator, initial, 2).end
    risk = task.RiskConfig()
    torch.manual_seed(12)
    samples = critic.collect_direction_samples(policy, simulator, ((2, closed),), 6, risk, .01)
    # Replay the same physical pair; the perturbed transition advances time to 3.
    plus, minus = critic.motor_perturbation_states(policy, simulator, closed, samples.perturbations)
    for state, inputs, targets in ((plus, samples.plus_inputs, samples.plus_returns),
                                   (minus, samples.minus_inputs, samples.minus_returns)):
        torch.testing.assert_close(inputs, critic.critic_features(state, 3, 6))
        future = task.rollout(policy, simulator, state, 3)
        torch.testing.assert_close(targets, torch.stack(tuple(task.risk_components(future, risk).values()), -1).sum(0))
        assert not inputs.requires_grad and not targets.requires_grad


@pytest.mark.parametrize("bank", ["train", "dev"])
def test_true_risk_gate_rejects_component_worsening_despite_better_performance_and_total_risk(bank):
    risk_api()
    before = {"task_objective": 10., "risk_objective": 8.,
              "risk_components": {"position": 4., "velocity": 1., "omega": 2., "saturation": 1.}}
    improved = {"task_objective": 9., "risk_objective": 7.,
                "risk_components": {"position": 3., "velocity": 1., "omega": 2., "saturation": 1.}}
    unsafe = {"task_objective": 8., "risk_objective": 6.,
              "risk_components": {"position": 1., "velocity": 1., "omega": 3., "saturation": 1.}}
    reason = critic.acceptance_rejection(before, unsafe if bank == "train" else improved,
                                         [before, before], [improved, unsafe] if bank == "dev" else [improved, improved],
                                         dev_relative_tolerance=.002)
    assert reason == ("train_risk_deteriorated" if bank == "train" else "development_risk_deteriorated")
    assert critic.acceptance_rejection(before, improved, [before, before], [improved, improved],
                                      dev_relative_tolerance=.002) is None


def test_actor_local_risk_gradient_and_terminal_risk_match_detached_oracle():
    risk_api()
    policy, simulator, initial = fixture()
    loss_config = task.TaskLossConfig(steady_steps=2)
    risk = task.RiskConfig()
    record = critic.collect_trajectory(policy, simulator, initial, 6, loss_config)
    target = critic.RiskToGoCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    critic.accumulate_actor_gradients(policy, target, simulator, initial, 6, 2, loss_config,
                                      record.weights, risk_config=risk, risk_weight=.7, baseline_returns=record.returns)
    actual = [None if p.grad is None else p.grad.clone() for p in policy.parameters()]
    policy.zero_grad(set_to_none=True)
    closed = task.initialize(policy, initial)
    for start in (0, 2, 4):
        trace = task.rollout(policy, simulator, critic.detach_closed_state(closed), 2)
        value = task.step_costs(trace, loss_config, start=start, horizon=6).sum(0)
        danger = torch.stack(tuple(task.risk_components(trace, risk).values()), -1).sum(0)
        danger = danger + record.returns[0] - record.returns[start]
        if start < 4:
            danger = danger + target(critic.critic_features(trace.end, start + 2, 6))
        cw = torch.stack([task.risk_weights(record.returns[0,:,j],loss_config) for j in range(4)],-1)
        ratio = (cw*danger).sum(0)/((cw*record.returns[0]).sum(0)+1.e-12)
        ((record.weights * value).sum()/3 + .7*torch.logsumexp(10*ratio,0)/30).backward()
        closed = trace.end
    for p, expected in zip(policy.parameters(), actual):
        if expected is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, expected, rtol=1e-11, atol=1e-12)
    assert all(p.grad is None for p in target.parameters())


def test_risk_checkpoint_rejects_cost_critic_and_cli_binds_risk_settings():
    risk_api()
    from tools.train_response_control import parse_args
    from response_training import critic_configuration
    policy, _, initial = fixture()
    state = critic.CriticTrainer(policy, task.initialize(policy, initial), 6,
                                 critic.CriticConfig(window_steps=2))
    saved = copy.deepcopy(state.state_dict())
    saved.pop("objective")
    with pytest.raises(ValueError, match="risk"):
        state.load_state_dict(saved)
    args = parse_args(["--risk-omega-limit", "12", "--risk-weight", ".7", "--critic-direction-samples", "2"])
    cfg = critic_configuration(args)
    assert cfg.risk.omega_limit == 12 and cfg.risk_weight == .7 and cfg.direction_samples == 2
    with pytest.raises(SystemExit):
        parse_args(["--risk-omega-limit", "0"])


def test_physical_performance_and_risk_improvement_commits_one_actor_update():
    policy, simulator, initial = fixture()
    with torch.no_grad():
        policy.controller[-1].weight.zero_()
        policy.controller[-1].bias.fill_(.2)
    initial = replace(initial, position=initial.position.new_tensor([[0., 0., .3]]).repeat(2, 1),
                      velocity=initial.velocity.new_tensor([[0., 0., .2]]).repeat(2, 1),
                      omega=torch.zeros_like(initial.omega))
    state = critic.CriticTrainer(policy, task.initialize(policy, initial), 6,
                                 critic.CriticConfig(window_steps=2, batch_size=32))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.)
    original = optimizer.step
    def reduce_collective(*args, **kwargs):
        original(*args, **kwargs)
        with torch.no_grad():
            policy.controller[-1].bias.sub_(.001)
    optimizer.step = reduce_collective
    report = state.guarded_step(policy, optimizer, simulator, initial, 6, task.TaskLossConfig(),
                                development_initials=(initial, initial), gradient_clip=10.)
    assert report["accepted"], report
    assert report["continuous_loss_after"] < report["continuous_loss_before"]
    assert report["continuous_risk_after"] <= report["continuous_risk_before"]
    assert state.completed_fits == 1 and report["critic_direction_after"]["pairs"] == 4
    assert all(float(value["step"]) == 1 for value in optimizer.state.values())
    assert all(p.grad is None for p in state.target.parameters())


def test_development_report_persists_true_risk_configuration_and_components(tmp_path):
    from response_training import evaluate
    policy, _, _ = fixture()
    risk = task.RiskConfig(omega_limit=3.)
    report = evaluate(policy, task.TaskLossConfig(), seeds=(32000007,), horizons=(4,),
                      scenarios=16, risk_config=risk, output=tmp_path / "development.json")
    assert report["risk_config"]["omega_limit"] == 3.
    row = report["records"][0]["policy"]
    assert row["risk_objective"] > 0
    assert set(row["risk_components"]) == {"position", "velocity", "omega", "saturation"}
    assert report["risk_objective"] == row["risk_objective"]


def test_direction_fit_uses_reachable_pair_labels_and_improves_their_ranking():
    policy, simulator, initial = fixture()
    config = critic.CriticConfig(window_steps=2, batch_size=32, direction_weight=10., direction_min_gap=1.e-12)
    state = critic.CriticTrainer(policy, task.initialize(policy, initial), 6, config)
    record = critic.collect_trajectory(policy, simulator, initial, 6, task.TaskLossConfig(),
                                       window_steps=2, direction_samples=4)
    initial_actor = copy.deepcopy(policy.state_dict())
    # Same initialization and MC regression, with/without the ranking term.
    value_only = critic.CriticTrainer(policy, task.initialize(policy, initial), 6,
                                      replace(config, direction_weight=0.))
    value_only.load_state_dict(copy.deepcopy(state.state_dict()))
    rng = torch.get_rng_state()
    metrics = state.fit(record)
    torch.set_rng_state(rng)
    baseline = value_only.fit(record)
    assert metrics["critic_direction_after"]["valid_pairs"] > 0
    assert metrics["critic_direction_after"]["loss"] < baseline["critic_direction_after"]["loss"]
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in initial_actor.items())


def test_sparse_direction_collection_keeps_main_flight_and_full_performance_tail_fixed():
    policy, simulator, initial = fixture()
    config = task.TaskLossConfig()
    before = copy.deepcopy(policy.state_dict())
    plain = critic.collect_trajectory(policy, simulator, initial, 6, config)
    sampled = critic.collect_trajectory(policy, simulator, initial, 6, config,
                                        window_steps=2, direction_samples=3)
    assert sampled.directions.plus_inputs.shape[0] == 3
    assert torch.equal(sampled.trajectory.actions, plain.trajectory.actions)
    assert torch.equal(sampled.returns, plain.returns)
    assert torch.equal(sampled.weights, plain.weights)
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_risk_at_zero_errors_and_full_motor_command_has_finite_gradients(dtype):
    policy, simulator, initial = fixture(dtype=dtype)
    trace = task.rollout(policy, simulator, initial, 1)
    p = torch.zeros_like(trace.positions, requires_grad=True)
    actions = torch.ones_like(trace.actions, requires_grad=True)
    loss = task.step_risks(replace(trace, positions=p, velocities=p, omegas=p, actions=actions),
                           task.RiskConfig()).sum()
    dp, du = torch.autograd.grad(loss, (p, actions))
    assert torch.isfinite(dp).all() and torch.isfinite(du).all()
    assert (dp == 0).all() and (du > 0).all()
