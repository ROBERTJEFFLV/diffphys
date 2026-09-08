from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env_l2f import L2FParams, L2FSimulator, _so3_exp
from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig, sample_scenarios
import response_training as training
from tools.train_response_control import parse_args


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first.cpu(), second.cpu())
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_so3_zero_has_correct_finite_tangent(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    phi = torch.zeros(1, 3, device=device, dtype=dtype, requires_grad=True)
    rotation = _so3_exp(phi)
    gradient = torch.autograd.grad(rotation[0, 1, 2], phi)[0]
    assert torch.equal(rotation, torch.eye(3, device=device, dtype=dtype)[None])
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, phi.new_tensor([[-1., 0., 0.]]), rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0., 1.e-7, 1.e-3])
def test_so3_zero_and_small_angle_gradcheck(scale):
    phi = (torch.tensor([[1., -2., 3.]], dtype=torch.float64) * scale).requires_grad_()
    assert torch.autograd.gradcheck(_so3_exp, (phi,), eps=1.e-6, atol=1.e-5, rtol=1.e-4)


@dataclass
class InitialMarker:
    marker: torch.Tensor


def guarded_fixture(monkeypatch, outcome):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    model.weight.grad = torch.ones_like(model.weight)
    initial = InitialMarker(torch.ones(1))
    before = {"task_objective": 10., "omega_rms": 1., "finite": True}
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())

    def replay(policy, simulator, actual_initial, horizon):
        assert actual_initial is initial
        assert not torch.is_grad_enabled()
        assert not torch.equal(policy.weight, before_model["weight"])
        random.random()
        np.random.rand()
        torch.rand(1)
        if outcome == "exception":
            raise RuntimeError("synthetic replay exception")
        values = {name: torch.zeros(2, 1, 3) for name in (
            "observations", "actions", "positions", "velocities", "omegas"
        )}
        return SimpleNamespace(**values)

    monkeypatch.setattr(training, "rollout", replay)
    monkeypatch.setattr(training, "trajectory_metrics", lambda *args: {
        "finite": True, "task_objective": 21. if outcome == "loss" else 12.,
        "omega_rms": 2.1 if outcome == "omega" else 1.2,
    })
    return model, optimizer, initial, before, before_model, before_optimizer


@pytest.mark.parametrize("outcome", ["loss", "omega", "exception"])
def test_rejected_proposal_restores_model_adam_and_all_rng(tmp_path, monkeypatch, outcome):
    model, optimizer, initial, before, old_model, old_optimizer = guarded_fixture(monkeypatch, outcome)
    old_rng = training.capture_rng()
    kwargs = dict(max_loss_ratio=2., max_omega_ratio=2.,
                  rejection_path=tmp_path / "rejected.pt")
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="synthetic replay"):
            training.guarded_adam_step(model, optimizer, None, initial, 2, None, before, **kwargs)
    else:
        result = training.guarded_adam_step(model, optimizer, None, initial, 2, None, before, **kwargs)
        assert not result["accepted"]
        assert result["proposal_finite"]
        assert result["rejection_reason"] == (
            "continuous_task_loss_catastrophe" if outcome == "loss" else "continuous_omega_catastrophe"
        )
        artifact = torch.load(tmp_path / "rejected.pt", map_location="cpu")
        assert not artifact["guard"]["accepted"]
        assert not torch.equal(artifact["candidate_model"]["weight"], old_model["weight"])
    assert_nested_equal(model.state_dict(), old_model)
    assert_nested_equal(optimizer.state_dict(), old_optimizer)
    assert_nested_equal(training.capture_rng(), old_rng)


def test_guard_allows_finite_nonmonotone_proposal(monkeypatch):
    model, optimizer, initial, before, old_model, old_optimizer = guarded_fixture(monkeypatch, "small")
    result = training.guarded_adam_step(
        model, optimizer, None, initial, 2, None, before,
        max_loss_ratio=2., max_omega_ratio=2.,
    )
    assert result["accepted"]
    assert result["continuous_loss_after"] > result["continuous_loss_before"]
    assert not torch.equal(model.weight, old_model["weight"])
    assert optimizer.state_dict()["state"][0]["step"] == old_optimizer["state"][0]["step"] + 1


def test_nonfinite_optimizer_proposal_is_rolled_back(monkeypatch):
    model, optimizer, initial, before, old_model, old_optimizer = guarded_fixture(monkeypatch, "small")
    original_step = optimizer.step

    def corrupt(*args, **kwargs):
        original_step(*args, **kwargs)
        with torch.no_grad():
            model.weight.fill_(float("nan"))

    monkeypatch.setattr(optimizer, "step", corrupt)
    old_rng = training.capture_rng()
    result = training.guarded_adam_step(
        model, optimizer, None, initial, 2, None, before,
        max_loss_ratio=2., max_omega_ratio=2.,
    )
    assert result["rejection_reason"] == "nonfinite_model_or_optimizer"
    assert not result["proposal_finite"]
    assert_nested_equal(model.state_dict(), old_model)
    assert_nested_equal(optimizer.state_dict(), old_optimizer)
    assert_nested_equal(training.capture_rng(), old_rng)


@pytest.mark.parametrize("outcome", ["finite_catastrophe", "evaluation_exception"])
def test_periodic_dev_failure_restores_best_and_keeps_failed_history(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(training, "FINAL_CLAIM", tmp_path / "unconsumed.json")
    calls = []

    def development(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1 and outcome == "evaluation_exception":
            raise RuntimeError("synthetic DEV failure")
        return {"score": 10. if len(calls) == 1 else 21., "finite": True,
                "position_rms": 1., "velocity_rms": 1., "omega_rms": 1.,
                "steady_success_rate": 0., "motor_saturation_fraction": 0.}

    monkeypatch.setattr(training, "evaluate", development)
    args = parse_args([
        "--optimizer", "adam", "--adam-train-batches", "1",
        "--device", "cpu", "--work-dir", str(tmp_path / "run"), "--updates", "2",
        "--horizon", "4", "--scenarios", "16", "--memory-dim", "4", "--hidden-dim", "8",
        "--minimum-updates", "1", "--checkpoint-every", "1", "--development-every", "2",
        "--prediction-weight", "0", "--steady-steps", "4",
    ])
    config = ResponsePolicyConfig(memory_dim=4, hidden_dim=8)
    loss = TaskLossConfig(prediction_weight=0, steady_steps=4)
    if outcome == "evaluation_exception":
        with pytest.raises(RuntimeError, match="synthetic DEV"):
            training.train(args, config, loss)
    else:
        result = training.train(args, config, loss)
        assert result["status"] == "adam_development_rollback"
    latest = torch.load(args.work_dir / "latest.training.pt", map_location="cpu")
    best = torch.load(args.work_dir / "best.training.pt", map_location="cpu")
    assert latest["progress"]["attempts"] == 2
    assert latest["progress"]["updates"] == 0
    assert len(latest["progress"]["training_seeds"]) == 2
    assert all(row["rolled_back"] and not row["accepted"] for row in latest["progress"]["history"])
    assert_nested_equal(latest["model"], best["model"])
    assert_nested_equal(latest["optimizer"], best["optimizer"])
    assert_nested_equal(latest["rng"], best["rng"])
    rejected = torch.load(args.work_dir / "rejected_development" / "0000002.training.pt", map_location="cpu")
    assert rejected["progress"]["updates"] == 2


@pytest.mark.parametrize("horizon", [8, 16])
@pytest.mark.parametrize("vary_actions", [False, True])
def test_cuda_vjp_matches_torch_with_zero_rotation_startup(horizon, vary_actions):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from l2f_cuda_backend import cuda_step
    params = L2FParams()
    simulator = L2FSimulator(params)
    initial, _ = sample_scenarios(16, seed=31_000_007, device=torch.device("cuda"), dtype=torch.float32)
    initial = replace(
        initial, rotation=torch.eye(3, device="cuda").expand(16, 3, 3).clone(),
        omega=torch.zeros_like(initial.omega), motor=torch.zeros_like(initial.motor),
        previous_action=torch.zeros_like(initial.previous_action),
    )
    values = torch.zeros(horizon, 16, 4, device="cuda")
    if vary_actions:
        values = values + torch.arange(horizon, device="cuda").sin()[:, None, None] * values.new_tensor([.002, -.001, .001, -.002])
    results = []
    for use_cuda in (False, True):
        actions = values.clone().requires_grad_()
        state = initial
        for step in range(horizon):
            state = cuda_step(state, actions[step], params, grad_decay=1.) if use_cuda else simulator.step(state, actions[step], grad_decay=1.)
        loss = state.rotation[:, 1, 2].mean() + .1 * state.position.square().mean() + .03 * state.omega.square().mean()
        gradient = torch.autograd.grad(loss, actions)[0]
        assert torch.isfinite(gradient).all()
        results.append((state, loss.detach(), gradient))
    a, b = results
    for name in ("position", "velocity", "rotation", "omega", "motor", "previous_action"):
        torch.testing.assert_close(getattr(a[0], name), getattr(b[0], name), rtol=2.e-4, atol=2.e-5)
    torch.testing.assert_close(a[1], b[1], rtol=2.e-4, atol=2.e-5)
    torch.testing.assert_close(a[2], b[2], rtol=2.e-3, atol=2.e-4)
