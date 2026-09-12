from dataclasses import fields, replace
import copy

import pytest
import torch

from env_l2f import L2FParams, L2FSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, rollout, risk_weights, step_costs, trajectory_metrics
import response_adjoints as adjoints
import response_task as task


def test_warning_observations_preserve_limits_without_risk_label_pipeline():
    actor, sim, initial = fixture()
    trace = task.rollout(actor, sim, initial, 2)
    trace = replace(trace, omegas=torch.tensor([[[0.,0.,0.],[20.,0.,0.]]]*2, dtype=torch.float64),
                    actions=torch.tensor([[[.1]*4,[.975]*4]]*2, dtype=torch.float64))
    risks = task.warning_risk_steps(trace)
    torch.testing.assert_close(risks['omega'], torch.tensor([[0.,1.]]*2, dtype=torch.float64))
    torch.testing.assert_close(risks['saturation'], torch.tensor([[0.,.25]]*2, dtype=torch.float64))
    assert not hasattr(task, 'suffix_risks')


def test_fixed_airframe_changes_only_initial_kinematics_and_preserves_rng():
    rng = torch.get_rng_state().clone()
    a, cells = task.sample_scenarios(
        32, seed=31000007, scenario_mode="fixed-airframe", dtype=torch.float64
    )
    b, _ = task.sample_scenarios(
        32, seed=32000007, scenario_mode="fixed-airframe", dtype=torch.float64
    )
    assert torch.equal(torch.get_rng_state(), rng)
    with torch.random.fork_rng(devices=[]):
        nominal = L2FSimulator(L2FParams()).reset(
            32,
            device="cpu",
            dtype=torch.float32,
            sample_dynamics=False,
            sample_external_force=False,
        )
    varying = {"position", "velocity", "rotation", "omega"}
    for f in fields(a):
        if f.name in varying:
            assert not torch.equal(getattr(a, f.name), getattr(b, f.name))
        else:
            torch.testing.assert_close(
                getattr(a, f.name), getattr(nominal, f.name).double(), rtol=0, atol=0
            )
            assert torch.equal(getattr(a, f.name), getattr(b, f.name))
    assert not a.external_force.any()
    assert (cells == -1).all()  # No fictitious dynamics strata.


def fixture(dtype=torch.float64):
    torch.manual_seed(7)
    sim = L2FSimulator(L2FParams())
    initial = sim.reset(
        2, device="cpu", dtype=dtype, sample_dynamics=False, sample_external_force=False
    )
    actor = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4, hidden_dim=8)).to(dtype)
    return actor, sim, initial


def flat_grad(actor):
    return torch.cat(
        [
            torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten()
            for p in actor.parameters()
        ]
    )


def test_control_network_has_no_predictor_and_keeps_task_memory_gradient():
    actor, sim, initial = fixture()
    assert not hasattr(actor, "response_predictor"), "unused prediction head still runs"
    trace = rollout(actor, sim, initial, 6)
    step_costs(trace, TaskLossConfig()).sum().backward()
    for prefix in ("response_encoder.", "response_memory.", "controller."):
        grads = [p.grad for n, p in actor.named_parameters() if n.startswith(prefix)]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads) > 0


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("window", [2, 3, 6])
def test_reverse_actor_matches_full_bptt_with_fixed_scale_and_pooled_cvar(
    dtype, window, monkeypatch
):
    assert hasattr(adjoints, "collect_boundary_rollout"), "missing lean boundary collection"
    actor, sim, initial = fixture(dtype)
    actor.register_parameter(
        "unused_diagnostic_parameter", torch.nn.Parameter(torch.ones(1, dtype=dtype))
    )
    config = TaskLossConfig(steady_steps=2, tail_fraction=0.5)
    record = adjoints.collect_boundary_rollout(
        actor, sim, initial, config, horizon=6, window_steps=window
    )
    full = rollout(actor, sim, initial, 6)
    costs = step_costs(full, config).sum(0)
    torch.testing.assert_close(record.costs, costs, rtol=0, atol=0)
    torch.testing.assert_close(record.weights, risk_weights(costs, config), rtol=0, atol=0)
    (0.1 * (record.weights * costs).sum()).backward()
    expected = flat_grad(actor).clone()
    actor.zero_grad(set_to_none=True)
    calls = []
    original = torch.autograd.grad

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted)
    report = adjoints.backward_actor(actor, sim, record, config, gradient_scale=0.1)
    assert len(calls) == 6 // window  # one state+parameter VJP per window
    assert report["reverse_starts"] == list(reversed(range(0, 6, window)))
    assert all(r["exact"] for r in report["boundaries"])
    assert actor.unused_diagnostic_parameter.grad is None
    torch.testing.assert_close(
        flat_grad(actor),
        expected,
        rtol=3e-5 if dtype == torch.float32 else 1e-9,
        atol=1e-7 if dtype == torch.float32 else 1e-11,
    )
    assert not hasattr(record, "trajectory") and not hasattr(record, "returns")
    metrics = trajectory_metrics(full, config)
    for name in (
        "position_rms",
        "velocity_rms",
        "omega_rms",
        "steady_success_rate",
        "motor_saturation_fraction",
    ):
        assert record.metrics[name] == pytest.approx(metrics[name], rel=3e-6, abs=1e-10)


def test_failed_boundary_recompute_does_not_publish_partial_actor_gradients():
    assert hasattr(adjoints, "collect_boundary_rollout"), "missing lean boundary collection"
    actor, sim, initial = fixture()
    config = TaskLossConfig(steady_steps=2)
    record = adjoints.collect_boundary_rollout(
        actor, sim, initial, config, horizon=6, window_steps=2
    )
    with torch.no_grad():
        actor.controller[-1].bias.add_(0.01)
    with pytest.raises(RuntimeError, match="boundary"):
        adjoints.backward_actor(actor, sim, record, config)
    assert all(p.grad is None for p in actor.parameters())
