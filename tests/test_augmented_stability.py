from __future__ import annotations

import torch
import pytest

from env_l2f import L2FParams, L2FSimulator
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy
from structured_rollout import (
    StructuredBoundaryCodec,
    StructuredClosedLoopState,
    make_structured_step_map,
    structured_global_yaw_basis,
    structured_observation,
)
from structured_stability import (
    AugmentedStabilityConfig,
    matrix_free_augmented_stability_report,
)
from structured_distillation import build_dagger_scenario_bank
from tools.postcalibration_local_evidence import _force_bins


def _config() -> AugmentedStabilityConfig:
    return AugmentedStabilityConfig(krylov_dim=6, jvp_fd_epsilon=1.0e-4)


def test_force_rank_bins_cover_each_authority_cell_exactly_once() -> None:
    bank = build_dagger_scenario_bank(64, seed=7, per_cell=4)
    force = _force_bins(bank)
    for tw in range(4):
        for alpha in range(4):
            selected = force[(bank.tw_bin == tw) & (bank.log_alpha_bin == alpha)]
            assert sorted(int(value) for value in selected.tolist()) == [0, 1, 2, 3]


def test_matrix_free_augmented_probe_detects_stable_and_unstable_linear_maps() -> None:
    stable = matrix_free_augmented_stability_report(
        lambda value: 0.8 * value, torch.zeros(5), config=_config()
    )
    unstable = matrix_free_augmented_stability_report(
        lambda value: 1.01 * value, torch.zeros(5), config=_config()
    )
    assert stable.gate_passed
    assert stable.spectral_radius < 0.995
    assert stable.arnoldi_converged
    assert stable.fixed_point_anchor_passed
    assert stable.fixed_point_converged
    assert stable.fixed_point_iterations >= 1
    assert len(stable.fixed_point_anchor_hash) == 64
    assert stable.jvp_validation_passed
    assert not unstable.gate_passed
    assert unstable.spectral_radius > 1.0
    assert unstable.finite_time_gains["1"] > 1.0


def test_matrix_free_probe_projects_only_declared_yaw_basis() -> None:
    # The first coordinate is a pure gauge mode; the second is a real unstable
    # mode and must remain visible after the explicit quotient.
    matrix = torch.diag(torch.tensor((1.0, 1.01, 0.7), dtype=torch.float64))
    result = matrix_free_augmented_stability_report(
        lambda value: matrix @ value,
        torch.zeros(3, dtype=torch.float64),
        config=AugmentedStabilityConfig(krylov_dim=3, jvp_fd_epsilon=1.0e-5),
        yaw_basis=torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64),
    )
    assert result.spectral_radius > 1.0
    assert not result.gate_passed


def test_nonstationary_anchor_cannot_be_called_local_stability() -> None:
    result = matrix_free_augmented_stability_report(
        # A translation has no fixed point.  A one-shot residual check would
        # incorrectly accept it when the threshold is made permissive.
        lambda value: value + 0.1,
        torch.zeros(5),
        config=AugmentedStabilityConfig(
            krylov_dim=6, fixed_point_max_iterations=4,
            fixed_point_residual_threshold=1.0e-3,
        ),
    )
    assert not result.fixed_point_anchor_passed
    assert not result.fixed_point_converged
    assert result.fixed_point_iterations == 4
    assert not result.gate_passed


def test_stable_nonzero_map_is_restored_before_linearization() -> None:
    result = matrix_free_augmented_stability_report(
        lambda value: 0.8 * value + 0.1,
        torch.zeros(5, dtype=torch.float64),
        config=AugmentedStabilityConfig(
            krylov_dim=5, fixed_point_max_iterations=20,
            fixed_point_residual_threshold=1.0e-4,
            jvp_fd_epsilon=1.0e-4,
        ),
    )
    assert result.fixed_point_converged
    assert result.fixed_point_anchor_passed
    assert result.fixed_point_residual <= 1.0e-4
    assert result.spectral_radius < 0.995
    assert result.gate_passed


def test_tiny_real_v2_rollout_pack_and_jvp_smoke() -> None:
    simulator = L2FSimulator(L2FParams(dt=0.01, max_initial_position=0.001,
                                        max_initial_velocity=0.001,
                                        max_initial_angle=0.001,
                                        max_initial_omega=0.001))
    physical = simulator.reset(1, device="cpu", dtype=torch.float64)
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(
        hidden_dim=3, identifier_dim=3, slow_cadence=25,
    )).double().eval()
    initial_observation = torch.cat((
        physical.position, physical.velocity, physical.rotation.reshape(1, 9),
        physical.omega, torch.zeros(1, 3, dtype=torch.float64),
        physical.previous_action,
    ), dim=-1)
    closed = StructuredClosedLoopState(
        physical, policy.initial_state(initial_observation)
    )
    for _ in range(125):
        output = policy.forward_with_aux(
            structured_observation(closed), closed.policy, simulator.params.dt
        )
        closed = StructuredClosedLoopState(
            simulator.step(closed.physical, output.action), output.next_state
        )
    codec = StructuredBoundaryCodec(closed.physical, policy, boot_completed=True)
    packed = codec.pack(closed)[0]
    step_map = make_structured_step_map(policy, simulator, codec)
    restored = codec.unpack(step_map(packed).unsqueeze(0))
    assert restored.policy.slow_counter == closed.policy.slow_counter + 1
    # The diagnostic's compatibility JVP path falls back to autograd for
    # policy code that uses ``Tensor.new_tensor`` (unsupported by torch.func
    # transforms on older torch versions).
    discrete = torch.zeros(codec.state_dim, dtype=torch.bool)
    discrete[codec.slices["identification_failed"]] = True
    discrete[codec.slices["slow_counter"]] = True
    discrete |= codec.fixed_boundary_mask
    result = matrix_free_augmented_stability_report(
        step_map, packed, config=AugmentedStabilityConfig(
                horizon_steps=1, krylov_dim=2, jvp_fd_epsilon=1.0e-3,
            spectral_radius_threshold=100.0,
            finite_gain_thresholds=(100.0, 100.0, 100.0),
        ), yaw_basis=None,
        discrete_mask=discrete,
    )
    assert result.finite
    assert torch.isfinite(torch.tensor(result.jvp_central_relative_error))
    assert result.jvp_central_relative_error < 0.1
