from __future__ import annotations

import pytest
import torch

from identification_features import (
    bank_modal_features,
    modal,
    normalize_response,
    production_legacy24,
    sol_response,
)
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy, motor_observer_tau_grid
from structured_rollout import StructuredBoundaryCodec, StructuredClosedLoopState, structured_observation
from env_l2f import L2FParams, L2FSimulator
from structured_checkpoint import (
    CADENCE_SEMANTICS_VERSION,
    require_current_cadence_semantics,
)


def test_versioned_tau_grids_have_registered_endpoints_and_shapes() -> None:
    v1 = motor_observer_tau_grid(1)
    v2 = motor_observer_tau_grid(2)
    assert len(v1) == 15
    assert len(v2) == 35
    assert min(r for r, _ in v2) == pytest.approx(.025)
    assert max(r for r, _ in v2) == pytest.approx(.18)
    assert min(f for _, f in v2) >= .03
    assert max(f for _, f in v2) <= .35
    assert len({r for r, _ in v2}) == 7
    assert len({f for _, f in v2}) >= 7


def test_stale_structured_publication_semantics_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="is stale"):
        require_current_cadence_semantics(
            {"cadence_semantics_version": "call_index_completed_transitions_v2"},
            context="test",
        )
    require_current_cadence_semantics(
        {"report": {"cadence_semantics_version": CADENCE_SEMANTICS_VERSION}},
        context="test",
    )


def test_observer_mode_and_grid_version_fail_fast() -> None:
    with pytest.raises(ValueError, match="must match"):
        StructuredRecurrentPolicy(StructuredPolicyConfig(motor_observer_bank_size=15))
    with pytest.raises(ValueError, match="must match"):
        StructuredRecurrentPolicy(StructuredPolicyConfig(
            motor_observer_mode="fixed_multi_tau_v2",
            motor_observer_bank_size=15,
            motor_tau_grid_version=2,
        ))
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(
        motor_observer_mode="fixed_multi_tau_v2",
        motor_observer_bank_size=35,
        motor_tau_grid_version=2,
        hidden_dim=4,
        identifier_dim=4,
    ))
    assert policy.motor_observer_bank is not None
    assert policy.motor_observer_bank.modes == 35


def test_shared_feature_shapes_and_causal_legacy_parity() -> None:
    torch.manual_seed(1707)
    history = torch.randn(3, 13, 4)
    force = torch.randn(3, 3)
    angular = torch.randn(3, 3)
    delta = torch.randn(3, 4)
    mask = torch.tensor([1., 0., 1.])
    legacy = production_legacy24(history, force, angular, delta, mask)
    assert legacy.shape == (3, 24)
    # A call without a measured transition cannot leak a GRU-bias or an
    # unpaired response product.  Excitation energy is the only non-response
    # diagnostic and is zero at the real call0 because its history is empty.
    assert torch.equal(legacy[1, :20], torch.zeros(20))
    state = torch.randn(3, 35, 4)
    response = torch.randn(3, 4)
    bank = bank_modal_features(state, response)
    assert bank.shape == (3, 35, 8)
    torch.testing.assert_close(modal(state[:, 0]), modal(state[:, 0]))


def test_k0_action_and_k35_boundary_state() -> None:
    torch.manual_seed(3)
    simulator = L2FSimulator(L2FParams(dt=.01))
    physical = simulator.reset(2, device="cpu", dtype=torch.float64)
    observation = torch.cat((
        physical.position, physical.velocity, physical.rotation.reshape(2, 9),
        physical.omega, torch.zeros(2, 3, dtype=torch.float64),
        physical.previous_action,
    ), dim=-1)
    legacy = StructuredRecurrentPolicy(StructuredPolicyConfig(
        hidden_dim=4, identifier_dim=4,
    )).double()
    k35 = StructuredRecurrentPolicy(StructuredPolicyConfig(
        hidden_dim=4, identifier_dim=4, motor_observer_mode="fixed_multi_tau_v2",
        motor_observer_bank_size=35, motor_tau_grid_version=2,
    )).double()
    state = StructuredClosedLoopState(physical, k35.initial_state(observation))
    codec = StructuredBoundaryCodec(physical, k35)
    restored = codec.unpack(codec.pack(state))
    assert codec.slices["motor_bank"].stop - codec.slices["motor_bank"].start == 35 * 4
    torch.testing.assert_close(restored.policy.motor_bank, state.policy.motor_bank)
    # A zero bank adapter is the explicit K0 compatibility mechanism.
    k35.load_state_dict(legacy.state_dict(), strict=False)
    assert torch.count_nonzero(k35.bank_adapter.weight) == 0


def test_call0_exposes_zero_shared_features_and_keeps_identifier_exact() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(
        hidden_dim=4,
        identifier_dim=4,
        motor_observer_mode="fixed_multi_tau_v2",
        motor_observer_bank_size=35,
        motor_tau_grid_version=2,
    ))
    observation = torch.zeros(2, 25)
    observation[:, (6, 10, 14)] = 1.0
    state = policy.initial_state(observation)
    output = policy.forward_with_aux(observation, state)
    torch.testing.assert_close(output.next_state.identifier, state.identifier)
    assert torch.count_nonzero(
        output.auxiliary["identification_legacy_context"]
    ) == 0
    assert torch.count_nonzero(
        output.auxiliary["identification_bank_features"]
    ) == 0


def test_first_response_oracle_features_equal_deployed_policy_hooks() -> None:
    """Prove call1 uses the same aligned transition as the offline oracle."""

    torch.manual_seed(19)
    simulator = L2FSimulator(L2FParams(dt=.01))
    physical0 = simulator.reset(3, device="cpu", dtype=torch.float64)
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(
        hidden_dim=4,
        identifier_dim=4,
        motor_observer_mode="fixed_multi_tau_v2",
        motor_observer_bank_size=35,
        motor_tau_grid_version=2,
    )).double()
    observation0 = torch.cat((
        physical0.position, physical0.velocity, physical0.rotation.reshape(3, 9),
        physical0.omega, torch.zeros(3, 3, dtype=torch.float64),
        physical0.previous_action,
    ), dim=-1)
    policy_state0 = policy.initial_state(observation0)
    output0 = policy.forward_with_aux(observation0, policy_state0)
    physical1 = simulator.step(physical0, output0.action, grad_decay=1.0)
    current1 = StructuredClosedLoopState(physical1, output0.next_state)
    output1 = policy.forward_with_aux(
        structured_observation(current1), current1.policy
    )

    acceleration = (physical1.velocity - physical0.velocity) / simulator.params.dt
    gravity = acceleration.new_tensor((0.0, 0.0, simulator.params.gravity))
    specific = torch.bmm(
        physical0.rotation.transpose(1, 2),
        (acceleration + gravity).unsqueeze(-1),
    ).squeeze(-1)
    angular_acceleration = (physical1.omega - physical0.omega) / simulator.params.dt
    force, angular, collective = normalize_response(specific, angular_acceleration)
    expected_legacy = production_legacy24(
        output0.next_state.excitation_history,
        force,
        angular,
        output0.next_state.motor_estimate - policy_state0.motor_estimate,
        torch.ones(3, dtype=torch.float64),
    )
    expected_bank = bank_modal_features(
        output0.next_state.motor_bank,
        sol_response(collective, angular),
    ).reshape(3, -1)
    torch.testing.assert_close(
        output1.auxiliary["identification_legacy_context"], expected_legacy,
        atol=1.0e-10, rtol=1.0e-10,
    )
    torch.testing.assert_close(
        output1.auxiliary["identification_bank_features"], expected_bank,
        atol=1.0e-10, rtol=1.0e-10,
    )
