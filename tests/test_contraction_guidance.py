"""Numerical regressions for physical conditioning and auxiliary guidance."""
from dataclasses import fields, replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_l2f import L2FParams, L2FSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import initialize, _select_rows, rollout
from response_contraction import (
    ContractionConfig, ContractionMetric, StateGeometry, contraction_loss,
    sample_boundaries, _direction_penalties, _reduce_penalties,
)


def fixture(n=3):
    torch.manual_seed(7)
    actor = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8, hidden_dim=8)).double()
    sim = L2FSimulator()
    closed = initialize(actor, sim.reset(n, seed=42, horizon=20, dtype=torch.float64))
    return actor, sim, closed


@pytest.mark.parametrize('indices', [[2, 0, 1], [2, 2, 0]])
def test_full_length_row_selection_obeys_order_and_duplicates(indices):
    _, _, closed = fixture()
    indices = torch.tensor(indices)
    selected = _select_rows(closed.physical, indices)
    for field in fields(selected):
        torch.testing.assert_close(getattr(selected, field.name),
                                   getattr(closed.physical, field.name)[indices], rtol=0, atol=0)


def test_legacy_context_cannot_distinguish_different_yaw_authority():
    _, sim, closed = fixture()
    changed = replace(closed, physical=replace(
        closed.physical, rotor_torque_constant=2 * closed.physical.rotor_torque_constant))
    torch.testing.assert_close(StateGeometry.context(closed), StateGeometry.context(changed), rtol=0, atol=0)
    thrust = closed.physical.motor.new_tensor([1., 2., 1., 2.]).expand(3, -1)
    torque = sim.body_torque(closed.physical, thrust)
    altered = sim.body_torque(changed.physical, thrust)
    assert bool((torque[:, 2] != altered[:, 2]).all())


@pytest.mark.parametrize('name', ['rotor_torque_constant', 'rotor_positions', 'thrust_coefficients', 'noise_std'])
def test_physics_context_exposes_independent_plant_and_sensor_configuration(name):
    _, _, closed = fixture()
    p = closed.physical
    changed = replace(closed, physical=replace(p, **{name: getattr(p, name) + .01}))
    original_context = StateGeometry.context(closed, 'physics')
    changed_context = StateGeometry.context(changed, 'physics')
    assert original_context.shape == (3, 67)
    assert bool(torch.isfinite(original_context).all())
    assert not torch.equal(original_context, changed_context)
    torch.testing.assert_close(original_context[:, :20], StateGeometry.context(closed), rtol=0, atol=0)


@pytest.mark.parametrize('protocol', ['l2f', 'raptor'])
def test_normalized_motor_polynomial_reconstructs_actual_thrust(protocol):
    actor, _, _ = fixture()
    sim = L2FSimulator(L2FParams(protocol=protocol))
    closed = initialize(actor, sim.reset(3, seed=42, dtype=torch.float64))
    context = StateGeometry.context(closed, 'physics')
    # Serialized context: 20 old + arm(1) + geometry(12) + yaw(4), then 4x3 polynomial.
    polynomial = context[:, 37:49].reshape(3, 4, 3)
    normalized = torch.tensor([0., .3, .7, 1.], dtype=torch.float64).expand(3, -1)
    actual = sim.thrust(closed.physical, sim.motor_command(closed.physical, 2*normalized-1))
    predicted = (polynomial[..., 0] + polynomial[..., 1]*normalized
                 + polynomial[..., 2]*normalized.square()) * closed.physical.mass[:, None]*9.81
    torch.testing.assert_close(actual, predicted, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('context_mode', ['legacy', 'physics'])
@pytest.mark.parametrize('fusion', ['concat', 'film_gated'])
def test_metric_bounds_batched_prefixes_and_gradients_for_all_conditioning_modes(context_mode, fusion):
    actor, _, closed = fixture()
    config = ContractionConfig(context=context_mode, fusion=fusion, hidden_dim=8, rank=2)
    metric = ContractionMetric(actor.config.memory_dim, config).double()
    geometry = StateGeometry(closed, actor.config.integral_limit)
    x = geometry.pack(closed).expand(2, -1, -1).clone().requires_grad_()
    context = geometry.context(closed, context_mode).expand(2, -1, -1).clone().requires_grad_()
    tangent = torch.randn_like(x)
    matrix = metric.matrix(x, context)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert float(eigenvalues.detach().min()) >= config.metric_min
    assert float(eigenvalues.detach().max()) <= config.metric_max
    energy = metric.energy(x, context, tangent)
    expected = (tangent*(matrix@tangent[..., None]).squeeze(-1)).sum(-1)
    torch.testing.assert_close(energy, expected)
    energy.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in metric.parameters())
    assert context.grad.norm() > 0 and x.grad.norm() > 0


def test_transient_penalty_detects_growth_hidden_by_contracting_endpoint():
    config = ContractionConfig(prefix_weight=.5, prefix_max_gain_squared=4.)
    gains = torch.tensor([[1.], [9.], [.5], [1e9]], dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True], [True], [False]])  # Last value is frozen padding.
    loss, endpoint, prefix, peak = _direction_penalties(torch.tensor([.5]), gains, valid, config)
    assert endpoint.item() == 0 and prefix.item() > 0 and peak.item() == 9
    loss.sum().backward()
    assert gains.grad[1].item() > 0
    assert gains.grad[0].item() == gains.grad[2].item() == gains.grad[3].item() == 0
    baseline, *_ = _direction_penalties(torch.tensor([.5]), gains, valid, replace(config, prefix_weight=0))
    assert baseline.item() == 0


def test_tail_reduction_does_not_dilute_worst_sample():
    values = torch.tensor([0., 0., 1., 3.], requires_grad=True)
    assert _reduce_penalties(values, 1.).item() == 1.
    loss = _reduce_penalties(values, .25)
    assert loss.item() == 3.
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([0., 0., 0., 1.]))


@pytest.mark.parametrize('task', [[3., 4.], [0., 0.]])
def test_auxiliary_norm_cap_does_not_scale_metric_optimizer(task):
    from response_training import _merge_contraction_gradients
    actor = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    metric = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    actor.grad = torch.tensor(task, dtype=torch.float64)
    old = actor.grad.clone()
    gradients = [torch.tensor([100., 0.], dtype=torch.float64), torch.tensor([20.], dtype=torch.float64)]
    config = ContractionConfig(weight=.1, actor_max_ratio=.1, metric_gradient_scale=1.)
    report = _merge_contraction_gradients([actor], [metric], gradients, config, .1)
    assert (actor.grad-old).norm() <= .1*old.norm() + 1e-12
    assert metric.grad.item() == 20.
    assert report['actor_gradient_capped']
    if old.norm() > 0:
        assert report['task_auxiliary_cosine'] == pytest.approx(.6)
        assert report['auxiliary_task_norm_ratio'] == pytest.approx(.1)
    else:
        assert report['task_auxiliary_cosine'] is None


def test_default_auxiliary_merge_preserves_original_scaling():
    from response_training import _merge_contraction_gradients
    actor = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    metric = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    actor.grad = torch.tensor([3., 4.], dtype=torch.float64)
    gradients = [torch.tensor([100., 0.], dtype=torch.float64), torch.tensor([20.], dtype=torch.float64)]
    _merge_contraction_gradients([actor], [metric], gradients, ContractionConfig(weight=.1), .1)
    torch.testing.assert_close(actor.grad, torch.tensor([4., 4.], dtype=torch.float64))
    assert metric.grad.item() == pytest.approx(.2)


def test_mass_quantile_probes_cover_batch_without_global_rng_changes():
    from types import SimpleNamespace
    _, _, closed = fixture(n=8)
    record = SimpleNamespace(boundaries={0: closed}, horizon=4, valid=torch.ones(4, 8, dtype=torch.bool))
    rng = torch.get_rng_state().clone()
    selected, report = sample_boundaries(record, 4, 2, 111, sampling='mass_quantiles')
    order = closed.physical.mass.argsort().tolist()
    indices = [entry['scene'] for entry in report['boundaries']]
    assert sorted(order.index(index)//2 for index in indices) == [0, 1, 2, 3]
    torch.testing.assert_close(selected.physical.mass, closed.physical.mass[indices])
    assert torch.equal(torch.get_rng_state(), rng)
    again, metadata = sample_boundaries(record, 4, 2, 111, sampling='mass_quantiles')
    assert metadata == report
    torch.testing.assert_close(again.physical.mass, selected.physical.mass, rtol=0, atol=0)


@pytest.mark.parametrize('protocol', ['l2f', 'raptor'])
def test_guided_true_jvp_actor_gradient_matches_finite_difference(protocol):
    actor, _, _ = fixture(n=1)
    sim = L2FSimulator(L2FParams(protocol=protocol))
    closed = initialize(actor, sim.reset(1, seed=42, horizon=20, dtype=torch.float64))
    closed = rollout(actor, sim, closed, 2).end
    config = ContractionConfig(weight=.1, steps=3, samples=1, directions=2, hidden_dim=8, rank=2,
                               context='physics', fusion='film_gated', prefix_weight=.5,
                               prefix_max_gain_squared=1., tail_fraction=.5)
    metric = ContractionMetric(actor.config.memory_dim, config).double()
    loss, report = contraction_loss(actor, sim, metric, closed, config, seed=18)
    parameter = actor.controller[-1].bias
    gradient = torch.autograd.grad(loss, parameter)[0]
    index = int(gradient.abs().argmax())
    old = parameter[index].detach().clone()
    epsilon = 1e-6
    values = []
    try:
        for sign in (1, -1):
            with torch.no_grad():
                parameter[index].copy_(old + sign*epsilon)
            score, _ = contraction_loss(actor, sim, metric, closed, config, seed=18, differentiable=False)
            values.append(float(score))
    finally:
        with torch.no_grad():
            parameter[index].copy_(old)
    assert gradient[index].abs() > 0
    assert float(gradient[index]) == pytest.approx((values[0]-values[1])/(2*epsilon), rel=2e-4, abs=1e-5)
    assert report['max_prefix_euclidean_gain_squared'] >= report['max_euclidean_gain_squared']


@pytest.mark.parametrize('options', [dict(context='bad'), dict(fusion='bad'), dict(sampling='bad'),
    dict(prefix_weight=-1), dict(prefix_max_gain_squared=.5), dict(tail_fraction=0),
    dict(tail_fraction=1.1), dict(actor_max_ratio=-1), dict(metric_gradient_scale=float('nan'))])
def test_guidance_configuration_is_validated(options):
    with pytest.raises(ValueError):
        ContractionConfig(**options)


@pytest.mark.parametrize('protocol', ['l2f', 'raptor'])
def test_float32_default_width_guidance_has_finite_actor_and_metric_gradients(protocol):
    torch.manual_seed(7)
    actor = ResponseMotorPolicy().float()
    sim = L2FSimulator(L2FParams(protocol=protocol))
    closed = initialize(actor, sim.reset(2, seed=42, horizon=20, dtype=torch.float32))
    config = ContractionConfig(weight=.1, steps=3, samples=2, context='physics',
                               fusion='film_gated', prefix_weight=.1, tail_fraction=.5)
    metric = ContractionMetric(actor.config.memory_dim, config).float()
    loss, report = contraction_loss(actor, sim, metric, closed, config, seed=18)
    loss.backward()
    for module in (actor, metric):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.norm()) for g in gradients) > 0
    assert report['evaluated_directions'] == 4
