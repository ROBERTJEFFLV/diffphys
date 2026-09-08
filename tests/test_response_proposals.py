from __future__ import annotations

import copy
import importlib.util

import pytest
import torch

from test_response_guarded_updates import assert_nested_equal


def api():
    assert importlib.util.find_spec('response_proposals_debug'), 'missing TRAIN-only direction corrector'
    import response_proposals_debug
    return response_proposals_debug


def test_basis_drops_zero_and_dependent_gradients_preserving_their_span():
    proposals = api()
    gradients = torch.tensor([[2., 0., 0.], [0., 3., 0.], [4., 3., 0.], [0., 0., 0.]], dtype=torch.float64)
    basis = proposals.gradient_basis(gradients)
    assert basis.shape == (3, 2)
    torch.testing.assert_close(basis.T @ basis, torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close((gradients @ basis) @ basis.T, gradients)


def test_large_finite_float32_gradients_do_not_disappear_during_normalization():
    proposals = api()
    gradients = torch.tensor([[1.e25, 0.], [0., 1.e25]], dtype=torch.float32)
    basis = proposals.gradient_basis(gradients)
    assert basis.shape == (2, 2)
    torch.testing.assert_close(basis.T @ basis, torch.eye(2))


def test_projection_minimizes_distance_to_negative_performance_gradient_on_risk_boundary():
    proposals = api()
    assert hasattr(proposals, 'projected_performance_direction')
    # Risk requires y >= 0. Projection of (-2,-3) is (-2,0), not a strict risk descent.
    matrix = torch.tensor([[2., 3.], [0., -1.]], dtype=torch.float64)
    q = proposals.projected_performance_direction(matrix)
    assert q is not None
    torch.testing.assert_close(q, matrix.new_tensor([-2., 0.]), atol=1.e-12, rtol=1.e-12)
    assert proposals.projected_performance_direction(torch.tensor([[1.], [-1.]], dtype=torch.float64)) is None
    assert proposals.projected_performance_direction(torch.tensor([[0.], [1.]], dtype=torch.float64)) is None


def test_opposing_risk_rows_allow_performance_descent_tangent_to_both():
    proposals = api()
    assert hasattr(proposals, 'projected_performance_direction')
    matrix = torch.tensor([[1., 2.], [0., 1.], [0., -10.], [0., 2.], [0., 0.]], dtype=torch.float64)
    q = proposals.projected_performance_direction(matrix)
    torch.testing.assert_close(q, matrix.new_tensor([-1., 0.]), atol=1.e-12, rtol=1.e-12)


def test_real_forward_corrector_finds_common_direction_and_restores_probe_state_rng():
    proposals = api()
    from response_training import capture_rng
    policy = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        policy.weight.zero_()
    before = copy.deepcopy(policy.state_dict())
    rng = capture_rng()
    observations = []
    def evaluate_train():
        theta = policy.weight.detach().flatten()
        observations.append(theta.clone())
        torch.rand(2)  # Probe RNG must not leak into fitting or actual acceptance.
        return torch.stack((10 + theta[0] + 2*theta[1], 3 + theta[0] - .5*theta[1], theta.new_zeros(())))
    result = proposals.correct_direction(policy, torch.eye(2, dtype=torch.float64),
                                          evaluate_train, proposals.FiniteDifferenceConfig())
    assert result.reason is None
    assert result.evidence['probe_rollouts'] == 8  # 2 directions, +/- at two radii.
    assert result.evidence['basis_rank'] == 2
    assert_nested_equal(policy.state_dict(), before)
    assert_nested_equal(capture_rng(), rng)
    actual_slopes = torch.tensor([[1., 2.], [1., -.5]], dtype=torch.float64) @ result.direction
    assert actual_slopes[0] < 0 and actual_slopes[1] <= 1.e-8
    assert len(observations) == 9  # Including a baseline; no DEV callback exists.


def test_inconsistent_finite_differences_do_not_produce_an_update():
    proposals = api()
    policy = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        policy.weight.zero_()
    def evaluate_train():
        x = policy.weight.detach().flatten()[0]
        return (1 + x**3).view(1)
    config = proposals.FiniteDifferenceConfig(fd_relative_step=.1, fd_relative_tolerance=.05)
    result = proposals.correct_direction(policy, torch.ones(1, 1, dtype=torch.float64), evaluate_train, config)
    assert result.direction is None and result.reason == 'fd_unreliable'
    assert policy.weight.item() == 0


def test_probe_exception_restores_parameters_and_rng():
    proposals = api()
    from response_training import capture_rng
    policy = torch.nn.Linear(2, 1, bias=False).double()
    before, rng = copy.deepcopy(policy.state_dict()), capture_rng()
    calls = []
    def evaluate_train():
        calls.append(1)
        if len(calls) == 3:
            with torch.no_grad():
                policy.weight.fill_(100.)
            torch.rand(4)
            raise RuntimeError('probe failed')
        return policy.weight.detach().flatten().clone()
    with pytest.raises(RuntimeError, match='probe failed'):
        proposals.correct_direction(policy, torch.eye(2, dtype=torch.float64), evaluate_train,
                                    proposals.FiniteDifferenceConfig())
    assert_nested_equal(policy.state_dict(), before)
    assert_nested_equal(capture_rng(), rng)


def test_fd_absolute_tolerance_accepts_near_zero_row_but_not_large_absolute_error():
    proposals = api()
    assert 'fd_absolute_tolerance' in proposals.FiniteDifferenceConfig.__dataclass_fields__
    policy = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        policy.weight.zero_()
    def evaluate_train():
        x = policy.weight.detach().flatten()[0]
        return torch.stack((1+x, 1+1.e-3*x**3))
    config = proposals.FiniteDifferenceConfig(fd_relative_step=.1, fd_absolute_tolerance=1.e-5)
    result = proposals.correct_direction(policy, torch.ones(1, 1, dtype=torch.float64), evaluate_train, config)
    assert result.direction is not None and result.reason is None
    assert result.evidence['direction_risk_classification'] == ['unknown_or_near_zero']
    strict = proposals.correct_direction(policy, torch.ones(1, 1, dtype=torch.float64), evaluate_train,
        proposals.FiniteDifferenceConfig(fd_relative_step=.1, fd_absolute_tolerance=1.e-8))
    assert strict.direction is None and strict.reason == 'fd_unreliable'


@pytest.mark.parametrize('full_risk, half_risk, expected', [(1.e-6, -2.e-6, True), (.01, 0., False)])
def test_final_direction_check_distinguishes_unknown_sign_from_real_risk_increase(full_risk, half_risk, expected):
    proposals = api()
    assert hasattr(proposals, 'check_direction_at_scales')
    full = torch.tensor([[-1., 0.], [full_risk, 1.]], dtype=torch.float64)
    half = torch.tensor([[-1., 0.], [half_risk, 1.]], dtype=torch.float64)
    q = torch.tensor([1., 0.], dtype=torch.float64)
    valid, evidence = proposals.check_direction_at_scales(full, half, q,
        proposals.FiniteDifferenceConfig(fd_absolute_tolerance=1.e-5))
    assert valid == expected
    assert evidence['direction_risk_classification'] == ['unknown_or_near_zero' if expected else 'increasing']


def test_final_direction_requires_clear_performance_descent_at_both_scales():
    proposals = api()
    assert hasattr(proposals, 'check_direction_at_scales')
    full = torch.tensor([[-1.e-6, 1.], [0., 1.]], dtype=torch.float64)
    half = torch.tensor([[-2.e-6, 1.], [0., 1.]], dtype=torch.float64)
    valid, _ = proposals.check_direction_at_scales(full, half, full.new_tensor([1., 0.]),
                                                proposals.FiniteDifferenceConfig(fd_absolute_tolerance=1.e-5))
    assert not valid


@pytest.mark.parametrize('linear, cubic, usable', [(-3.e-6, 4.e-4, True), (-1/300, 4/3, False)])
def test_corrector_checks_projected_direction_even_when_full_rows_agree(linear, cubic, usable):
    proposals = api()
    policy = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        policy.weight.zero_()
    def evaluate_train():
        x, y = policy.weight.detach().flatten()
        return torch.stack((1-x, 1+linear*x+cubic*x**3+y))
    result = proposals.correct_direction(policy, torch.eye(2, dtype=torch.float64), evaluate_train,
        proposals.FiniteDifferenceConfig(fd_relative_step=.1, fd_absolute_tolerance=1.e-5))
    assert (result.direction is not None) == usable
    if usable:
        assert result.evidence['direction_risk_classification'] == ['unknown_or_near_zero']
    else:
        assert result.reason == 'fd_unreliable'
        assert result.evidence['fd_failure'] == 'direction_disagreement'
    assert torch.equal(policy.weight, torch.zeros_like(policy.weight))


def test_projection_handles_five_rotated_risk_constraints_with_different_units():
    proposals = api()
    generator = torch.Generator().manual_seed(17)
    rotation, _ = torch.linalg.qr(torch.randn(5, 5, generator=generator, dtype=torch.float64))
    perf = torch.tensor([2., -3., 4., -5., 6.], dtype=torch.float64)
    units = torch.tensor([1.e-8, 10., 1.e4, .02, 3.], dtype=torch.float64)
    matrix = torch.cat(((perf @ rotation).view(1, 5), units[:, None]*rotation), 0)
    expected = torch.tensor([-2., 0., -4., 0., -6.], dtype=torch.float64) @ rotation
    torch.testing.assert_close(proposals.projected_performance_direction(matrix), expected, rtol=1.e-9, atol=1.e-10)
