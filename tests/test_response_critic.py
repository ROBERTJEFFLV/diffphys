from __future__ import annotations

from dataclasses import replace
import copy
import importlib.util
import torch
import pytest

import response_task as task
from test_response_control import fixture


def critic_module():
    assert importlib.util.find_spec("response_critic") is not None, "missing training-only Critic"
    import response_critic
    return response_critic


def test_huber_limits_large_physical_error_gradient_without_changing_small_error():
    policy, simulator, initial = fixture()
    trace = task.rollout(policy, simulator, initial, 1)
    position = torch.tensor([[[.2, 3., 0.], [.2, 3., 0.]]], dtype=torch.float64,
                            requires_grad=True)
    trace = replace(trace, positions=position, velocities=torch.zeros_like(trace.velocities),
                    omegas=torch.zeros_like(trace.omegas), actions=torch.zeros_like(trace.actions),
                    action_deltas=torch.zeros_like(trace.action_deltas),
                    omega_deltas=torch.zeros_like(trace.omega_deltas))
    loss = task.task_loss(trace, task.TaskLossConfig(steady_weight=0, tail_weight=0))
    # 2 * Huber(delta=1): .2² + (2*3 - 1), mean across two scenes.
    torch.testing.assert_close(loss, loss.new_tensor(5.04))
    torch.testing.assert_close(torch.autograd.grad(loss, position)[0],
                               position.new_tensor([[[.2, 1., 0.], [.2, 1., 0.]]]))


def test_window_costs_use_global_time_and_fixed_full_trajectory_tail():
    assert hasattr(task, "step_costs"), "missing additive globally timed physical costs"
    assert hasattr(task, "risk_weights"), "missing frozen whole-trajectory CVaR weights"
    policy, simulator, initial = fixture()
    config = task.TaskLossConfig(steady_steps=2, tail_fraction=.5, tail_weight=.5)
    full = task.rollout(policy, simulator, initial, 6)
    first = task.rollout(policy, simulator, initial, 3)
    second = task.rollout(policy, simulator, first.end, 3)
    costs = task.step_costs(full, config)
    split = torch.cat([task.step_costs(first, config, start=0, horizon=6),
                       task.step_costs(second, config, start=3, horizon=6)])
    torch.testing.assert_close(split, costs)
    weights = task.risk_weights(costs.sum(0), config)
    torch.testing.assert_close((costs * weights).sum(), task.task_loss(full, config))
    torch.testing.assert_close(task.risk_weights(torch.tensor([1., 3., 2., 4.]), config),
                               torch.tensor([.25, .5, .25, .5]))


def test_exact_returns_and_complete_state_teacher_are_detached_and_continuous():
    critic = critic_module()
    costs = torch.tensor([[1., 4.], [2., 5.], [3., 6.]])
    torch.testing.assert_close(task.suffix_risks(costs),
                               torch.tensor([[6., 15.], [5., 11.], [3., 6.], [0., 0.]]))
    policy, simulator, initial = fixture()
    config = task.TaskLossConfig()
    record = critic.collect_trajectory(policy, simulator, initial, 6, config)
    direct = task.rollout(policy, simulator, initial, 6)
    assert torch.equal(record.trajectory.actions, direct.actions)
    assert not record.inputs.requires_grad and not record.returns.requires_grad
    torch.testing.assert_close(record.returns[0].sum(-1), task.step_risks(direct, task.RiskConfig()).sum(0))
    assert bool((record.returns[-1] == 0).all())
    # Motor truth, recurrent history and mass each change Critic input only.
    closed = task.initialize(policy, initial)
    original = critic.critic_features(closed, 0, 6)
    for field in ("mass", "motor", "inertia_x", "external_force"):
        changed = replace(closed, physical=replace(initial, **{field: getattr(initial, field) + .01}))
        assert not torch.equal(original, critic.critic_features(changed, 0, 6))
    changed = replace(closed, policy=replace(closed.policy, older_action=closed.policy.older_action + .1))
    assert not torch.equal(original, critic.critic_features(changed, 0, 6))


def test_frozen_value_retains_state_gradient_and_window_boundary_cuts_graph():
    critic = critic_module()
    policy, simulator, initial = fixture()
    closed = task.initialize(policy, initial)
    net = critic.RiskToGoCritic(critic.critic_features(closed, 0, 6).shape[-1]).double()
    net.requires_grad_(False)
    first = task.rollout(policy, simulator, closed, 2)
    value = net(critic.critic_features(first.end, 2, 6)).sum()
    gradients = torch.autograd.grad(value, list(policy.controller.parameters()), retain_graph=True)
    assert sum(float(g.abs().sum()) for g in gradients) > 1e-12
    boundary = critic.detach_closed_state(first.end)
    later = task.rollout(policy, simulator, boundary, 2)
    assert torch.autograd.grad(later.positions.sum(), first.end.physical.position,
                               allow_unused=True)[0] is None
    assert all(p.grad is None for p in net.parameters())


def test_window_gradients_match_explicit_detached_oracle_without_actor_update():
    critic = critic_module()
    policy, simulator, initial = fixture()
    config = task.TaskLossConfig(steady_steps=2)
    record = critic.collect_trajectory(policy, simulator, initial, 6, config)
    target = critic.RiskToGoCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    before = copy.deepcopy(policy.state_dict())
    measured = critic.accumulate_actor_gradients(policy, target, simulator, initial,
                                                6, 2, config, record.weights, baseline_returns=record.returns)
    actual = {name: None if p.grad is None else p.grad.clone() for name, p in policy.named_parameters()}
    assert all(torch.equal(before[name], p) for name, p in policy.state_dict().items())
    assert torch.equal(measured["end"].physical.position, record.trajectory.end.physical.position)
    policy.zero_grad(set_to_none=True)
    # Independent oracle restarts each window from saved numerical closed state;
    # terminal risk contributes only at t=2 and t=4, never at t=6.
    closed = task.initialize(policy, initial)
    for start in (0, 2, 4):
        closed = critic.detach_closed_state(closed)
        trace = task.rollout(policy, simulator, closed, 2)
        local = task.step_costs(trace, config, start=start, horizon=6).sum(0)
        risk = torch.stack(tuple(task.risk_components(trace, task.RiskConfig()).values()), -1).sum(0)
        risk = risk + record.returns[0] - record.returns[start]
        if start < 4:
            risk = risk + target(critic.critic_features(trace.end, start + 2, 6))
        cw = torch.stack([task.risk_weights(record.returns[0,:,j],config) for j in range(4)],-1)
        ratio = (cw*risk).sum(0)/((cw*record.returns[0]).sum(0)+1.e-12)
        ((record.weights * local).sum()/3 + torch.logsumexp(10*ratio,0)/30).backward()
        closed = trace.end
    for name, p in policy.named_parameters():
        if actual[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(actual[name], p.grad, rtol=1e-11, atol=1e-12)
    assert all(p.grad is None for p in target.parameters())


def trainer_fixture():
    critic = critic_module()
    assert hasattr(critic, "CriticTrainer"), "missing transactional Critic/Actor proposal"
    policy, simulator, initial = fixture()
    state = critic.CriticTrainer(policy, task.initialize(policy, initial), 6,
                                critic.CriticConfig(window_steps=2, batch_size=32, proposal="smoothmax-adam"))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.)
    return critic, policy, simulator, initial, state, optimizer


def test_monte_carlo_fit_learns_returns_and_target_is_a_frozen_copy():
    critic, policy, simulator, initial, state, optimizer = trainer_fixture()
    record = critic.collect_trajectory(policy, simulator, initial, 6, task.TaskLossConfig())
    # Constant inputs make the regression identifiable without physics noise.
    record = replace(record, inputs=torch.zeros_like(record.inputs),
                     returns=torch.ones_like(record.returns))
    before = copy.deepcopy(policy.state_dict())
    for _ in range(5):
        metrics = state.fit(record)
    assert metrics["critic_loss_after"] < metrics["critic_loss_before"]
    assert all(torch.equal(before[n], p) for n, p in policy.state_dict().items())
    for p, q in zip(state.critic.parameters(), state.target.parameters()):
        assert torch.equal(p, q) and not q.requires_grad and q.grad is None


def observe_completed_fit(state):
    """Record the real completed fit, so tests can detect later rollback of it."""
    from response_training import capture_rng
    captured = {}
    original = state.fit
    def fit(record):
        result = original(record)
        captured.update(training=copy.deepcopy(state.state_dict()), rng=capture_rng())
        return result
    state.fit = fit
    return captured


def test_actor_rejection_retains_critic_fit_target_optimizer_and_post_fit_rng():
    from response_training import capture_rng
    from test_response_guarded_updates import assert_nested_equal
    _, policy, simulator, initial, state, optimizer = trainer_fixture()
    saved_actor = copy.deepcopy(policy.state_dict())
    saved_critic = copy.deepcopy(state.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    fitted = observe_completed_fit(state)
    report = state.guarded_step(policy, optimizer, simulator, initial, 6, task.TaskLossConfig(),
                                development_initials=(initial,), gradient_clip=10.)
    assert not report["accepted"] and report["rejection_reason"] == "train_not_improved"
    assert_nested_equal(saved_actor, policy.state_dict())
    assert_nested_equal(fitted["training"], state.state_dict())
    assert any(not torch.equal(saved_critic["critic"][k], v) for k, v in state.critic.state_dict().items())
    assert_nested_equal(saved_optimizer, optimizer.state_dict())
    assert_nested_equal(fitted["rng"], capture_rng())
    assert report["critic_update_retained"] and report["critic_fits"] == 1


def test_exception_after_actor_step_preserves_completed_critic_fit():
    from response_training import capture_rng
    from test_response_guarded_updates import assert_nested_equal
    _, policy, simulator, initial, state, optimizer = trainer_fixture()
    saved = copy.deepcopy((policy.state_dict(), state.state_dict(), optimizer.state_dict()))
    fitted = observe_completed_fit(state)
    original = optimizer.step
    def failing_step(*args, **kwargs):
        original(*args, **kwargs)
        torch.rand(3)  # Only randomness after the committed fit should rewind.
        raise RuntimeError("injected optimizer failure after mutation")
    optimizer.step = failing_step
    with pytest.raises(RuntimeError, match="injected"):
        state.guarded_step(policy, optimizer, simulator, initial, 6, task.TaskLossConfig(),
                           development_initials=(initial,), gradient_clip=10.)
    assert_nested_equal(saved[0], policy.state_dict())
    assert_nested_equal(saved[2], optimizer.state_dict())
    assert_nested_equal(fitted["training"], state.state_dict())
    assert_nested_equal(fitted["rng"], capture_rng())


def test_primary_cli_uses_h500_ten_windows_two_banks_and_no_auxiliary():
    from tools.train_response_control import parse_args
    from response_training import training_batch_count
    args = parse_args([])
    assert args.optimizer == "short-window"
    assert args.horizon == 500 and args.window_steps == 50
    assert training_batch_count(args) == 2 and args.scenarios == 64
    assert args.prediction_weight == 0 and args.huber_delta == 1


def test_short_window_checkpoint_exact_resume_includes_critic_target_and_both_optimizers(tmp_path, monkeypatch):
    import response_training as training
    from tools.train_response_control import parse_args
    from response_policy import ResponsePolicyConfig
    from test_response_guarded_updates import assert_nested_equal
    assert hasattr(training, "critic_configuration"), "missing Critic trainer integration"
    monkeypatch.setattr(training, "FINAL_CLAIM", tmp_path / "unused-claim.json")
    def run(path, budget):
        args = parse_args(["--optimizer", "short-window", "--actor-proposal", "smoothmax-adam", "--work-dir", str(path),
                           "--updates", str(budget), "--horizon", "4", "--window-steps", "2",
                           "--scenarios", "16", "--memory-dim", "4", "--hidden-dim", "8",
                           "--checkpoint-every", "1", "--development-every", "2",
                           "--minimum-updates", "1", "--steady-steps", "4", "--lr", "1e-6"])
        training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8),
                       task.TaskLossConfig(prediction_weight=0, steady_steps=4))
    run(tmp_path / "full", 2)
    run(tmp_path / "split", 1)
    run(tmp_path / "split", 2)
    first = torch.load(tmp_path / "full/latest.training.pt", map_location="cpu")
    second = torch.load(tmp_path / "split/latest.training.pt", map_location="cpu")
    for key in ("model", "optimizer", "critic_training", "rng"):
        assert_nested_equal(first[key], second[key])
    assert first["critic_training"] is not None
    assert first["progress"]["training_seeds"] == [31000007, 31000008, 31000009, 31000010]
    assert first["progress"]["attempts"] == second["progress"]["attempts"] == 2


@pytest.mark.parametrize("motor,delta,reason", [(2, .001, None), (3, .01, "development_deteriorated")])
def test_real_physics_proposal_uses_hard_risk_and_dev_performance(motor, delta, reason):
    from response_training import capture_rng
    from test_response_guarded_updates import assert_nested_equal
    _, policy, simulator, initial, state, optimizer = trainer_fixture()
    dev = replace(initial, position=-initial.position, velocity=-initial.velocity, omega=-initial.omega)
    saved = copy.deepcopy((policy.state_dict(), state.state_dict(), optimizer.state_dict()))
    fitted = observe_completed_fit(state)
    steps = []
    # A known physical direction isolates the acceptance gate from Critic quality.
    original = optimizer.step
    def controlled_step(*args, **kwargs):
        steps.append(1)
        original(*args, **kwargs)
        with torch.no_grad():
            policy.controller[-1].bias[motor] += delta
    optimizer.step = controlled_step
    report = state.guarded_step(policy, optimizer, simulator, initial, 6, task.TaskLossConfig(),
                                development_initials=(initial, dev), gradient_clip=10.)
    assert len(steps) == 1
    assert report["continuous_loss_after"] < report["continuous_loss_before"]
    assert report["accepted"] == (reason is None)
    assert report["rejection_reason"] == reason
    # These real proposals used to be vetoed by small soft-risk tradeoffs.
    assert any(report["continuous_risk_components_after"][key] > value
               for key, value in report["continuous_risk_components_before"].items())
    assert report["continuous_hard_risk_after"] == {"omega": 0., "saturation": 0.}
    if reason is not None:
        assert_nested_equal(saved[0], policy.state_dict())
        assert_nested_equal(saved[2], optimizer.state_dict())
    assert_nested_equal(fitted["training"], state.state_dict())
    assert_nested_equal(fitted["rng"], capture_rng())


def test_profile_executes_actual_short_window_proposal(tmp_path):
    from tools.train_response_control import parse_args
    from response_training import profile
    from response_policy import ResponsePolicyConfig
    args = parse_args(["--mode", "profile", "--work-dir", str(tmp_path), "--horizon", "4",
                       "--window-steps", "2", "--scenarios", "16"])
    report = profile(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8),
                     task.TaskLossConfig(prediction_weight=0))
    assert report.get("optimizer") == "short-window"
    assert report["guard"]["windows"] == 2 and "critic_loss_after" in report["guard"]
    assert not list(tmp_path.glob("*.pt"))


def test_default_contract_matches_primary_training_objective(tmp_path):
    from tools.check_response_training_contract import run_contract
    from response_training import atomic_json, validate_training_contract
    from response_policy import ResponsePolicyConfig
    report = run_contract(device="cpu")
    assert report["loss_config"]["prediction_weight"] == 0
    path = tmp_path / "contract.json"
    atomic_json(path, report)
    validate_training_contract(path, ResponsePolicyConfig(), task.TaskLossConfig(prediction_weight=0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_short_window_cuda_gradients_match_cpu():
    critic = critic_module()
    policy, simulator, initial = fixture()
    config = task.TaskLossConfig(prediction_weight=0)
    record = critic.collect_trajectory(policy, simulator, initial, 6, config)
    target = critic.RiskToGoCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    cuda_policy = copy.deepcopy(policy).cuda()
    cuda_target = copy.deepcopy(target).cuda()
    from dataclasses import fields
    cuda_initial = replace(initial, **{f.name: getattr(initial, f.name).cuda() for f in fields(initial)})
    critic.accumulate_actor_gradients(policy, target, simulator, initial, 6, 2, config, record.weights, baseline_returns=record.returns)
    critic.accumulate_actor_gradients(cuda_policy, cuda_target, simulator, cuda_initial,
                                      6, 2, config, record.weights.cuda(), baseline_returns=record.returns.cuda())
    for p, q in zip(policy.parameters(), cuda_policy.parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad.cpu(), atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("stage", ["critic", "actor"])
def test_nonfinite_optimizer_mutation_only_rolls_back_uncommitted_training(stage):
    from response_training import capture_rng
    from test_response_guarded_updates import assert_nested_equal
    _, policy, simulator, initial, state, optimizer = trainer_fixture()
    saved = copy.deepcopy((policy.state_dict(), state.state_dict(), optimizer.state_dict()))
    rng = capture_rng()
    fitted = observe_completed_fit(state)
    mutated_optimizer = state.optimizer if stage == "critic" else optimizer
    mutated_parameter = next(state.critic.parameters()) if stage == "critic" else next(policy.parameters())
    original = mutated_optimizer.step
    def nonfinite_step(*args, **kwargs):
        original(*args, **kwargs)
        with torch.no_grad():
            mutated_parameter.fill_(float("nan"))
    mutated_optimizer.step = nonfinite_step
    report = state.guarded_step(policy, optimizer, simulator, initial, 6, task.TaskLossConfig(),
                                development_initials=(initial,), gradient_clip=10.)
    assert not report["accepted"] and not report["proposal_finite"]
    assert_nested_equal(saved[0], policy.state_dict())
    assert_nested_equal(saved[2], optimizer.state_dict())
    if stage == "critic":
        assert_nested_equal(saved[1], state.state_dict())
        assert_nested_equal(rng, capture_rng())
    else:
        assert_nested_equal(fitted["training"], state.state_dict())
        assert_nested_equal(fitted["rng"], capture_rng())


def test_profile_honors_weights_only_initialization(tmp_path):
    from dataclasses import asdict
    from tools.train_response_control import parse_args
    from response_training import profile, model_hash, PROTOCOL_VERSION
    from response_policy import ARCHITECTURE
    policy, _, _ = fixture(dtype=torch.float32)
    with torch.no_grad():
        policy.controller[-1].bias.fill_(.2)
    checkpoint = tmp_path / "actor.pt"
    torch.save({"schema": PROTOCOL_VERSION, "architecture": ARCHITECTURE,
                "policy_config": asdict(policy.config), "model": policy.state_dict()}, checkpoint)
    args = parse_args(["--mode", "profile", "--horizon", "4", "--window-steps", "2",
                       "--scenarios", "16", "--work-dir", str(tmp_path / "profile"),
                       "--initialize-from", str(checkpoint)])
    report = profile(args, policy.config, task.TaskLossConfig(prediction_weight=0))
    assert report.get("initial_model_sha256") == model_hash(policy)


def test_critic_learning_accumulates_across_rejected_actors_and_resume():
    from response_training import capture_rng, restore_rng
    from test_response_guarded_updates import assert_nested_equal
    _, policy, simulator, initial, state, optimizer = trainer_fixture()
    def attempt():
        return state.guarded_step(policy, optimizer, simulator, initial, 6,
                                  task.TaskLossConfig(), development_initials=(initial,), gradient_clip=10.)
    first = attempt()
    assert not first["accepted"]
    checkpoint = copy.deepcopy(state.state_dict())
    rng = capture_rng()
    second = attempt()
    assert not second["accepted"]
    assert first["critic_fits"] == 1 and second["critic_fits"] == 2
    # Same Actor and same flight: the second fit must start at the first fit's loss.
    assert second["critic_loss_before"] == first["critic_loss_after"]
    expected, expected_rng = copy.deepcopy(state.state_dict()), capture_rng()
    state.load_state_dict(checkpoint)
    restore_rng(rng)
    resumed = attempt()
    assert resumed == second
    assert_nested_equal(expected, state.state_dict())
    assert_nested_equal(expected_rng, capture_rng())


@pytest.mark.parametrize("failure", ["catastrophe", "exception"])
def test_periodic_dev_rollback_preserves_accumulated_critic_and_rng(tmp_path, monkeypatch, failure):
    import response_training as training
    from tools.train_response_control import parse_args
    from response_policy import ResponsePolicyConfig
    from test_response_guarded_updates import assert_nested_equal
    monkeypatch.setattr(training, "FINAL_CLAIM", tmp_path / "unused-claim.json")
    evaluations = []
    def development(*args, **kwargs):
        evaluations.append(1)
        if len(evaluations) > 1 and failure == "exception":
            raise RuntimeError("injected DEV failure")
        return {"score": 10. if len(evaluations) == 1 else 21., "finite": True,
                "position_rms": 1., "velocity_rms": 1., "omega_rms": 1.,
                "steady_success_rate": 0., "motor_saturation_fraction": 0.,
                "risk_objective": 1., "risk_components": {"position": 1., "velocity": 0., "omega": 0., "saturation": 0.},
                "hard_risk_components": {"omega": 0., "saturation": 0.}, "hard_risk_bounds_violated": []}
    monkeypatch.setattr(training, "evaluate", development)
    # Exercise the periodic rollback after an Actor change. An unchanged,
    # rejected Actor now correctly reuses its existing development report.
    module = critic_module()
    guarded = module.CriticTrainer.guarded_step
    def changed_candidate(self, policy, *args, **kwargs):
        record = guarded(self, policy, *args, **kwargs)
        with torch.no_grad():
            policy.controller[-1].bias.add_(1.e-5)
        return {**record, "accepted": True, "rejection_reason": None}
    monkeypatch.setattr(module.CriticTrainer, "guarded_step", changed_candidate)
    args = parse_args(["--optimizer", "short-window", "--actor-proposal", "smoothmax-adam", "--work-dir", str(tmp_path),
                       "--updates", "2", "--horizon", "4", "--window-steps", "2",
                       "--scenarios", "16", "--minimum-updates", "1",
                       "--checkpoint-every", "1", "--development-every", "2", "--lr", "1e-6"])
    def run():
        return training.train(args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8),
                               task.TaskLossConfig(prediction_weight=0))
    if failure == "exception":
        with pytest.raises(RuntimeError, match="injected DEV failure"):
            run()
    else:
        assert run()["status"] == "adam_development_rollback"
    latest = torch.load(tmp_path / "latest.training.pt", map_location="cpu")
    best = torch.load(tmp_path / "best.training.pt", map_location="cpu")
    rejected = torch.load(tmp_path / "rejected_development/0000002.training.pt", map_location="cpu")
    assert_nested_equal(latest["model"], best["model"])
    assert_nested_equal(latest["optimizer"], best["optimizer"])
    assert latest["critic_training"]["optimizer"]["state"]
    assert_nested_equal(latest["critic_training"], rejected["critic_training"])
    assert_nested_equal(latest["rng"], rejected["rng"])
