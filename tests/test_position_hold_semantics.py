from __future__ import annotations

import math
import unittest

import torch

from env_l2f import L2FSimulator
from model import MotorGRUPolicy
from policy_observation import (
    INTEGRAL_OBSERVATION_MODE,
    build_policy_observation,
    initial_observation_state,
)
from train import (
    _position_hold_checkpoint,
    _position_hold_histories,
    _position_hold_step_success,
)


def _rotation_x(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)),
        dtype=torch.float32,
    )


def _rotation_z(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
    )


class PositionHoldSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9)
        self.sim = L2FSimulator()

    def test_exactly_ninety_five_of_one_hundred_steps_is_steady(self) -> None:
        hold = torch.zeros(100, 2, dtype=torch.bool)
        hold[:94, 0] = True
        hold[:95, 1] = True
        survival = torch.ones_like(hold)

        fractions, steady = _position_hold_histories(
            hold,
            survival,
            window_steps=100,
            required_fraction=0.95,
        )
        summary = _position_hold_checkpoint(
            hold,
            survival,
            fractions,
            steady,
            completed_steps=100,
            dt=0.01,
        )

        torch.testing.assert_close(
            summary["final_window_fraction"],
            torch.tensor((0.94, 0.95)),
        )
        self.assertEqual(summary["steady"].tolist(), [False, True])

    def test_omega_z_above_threshold_fails_position_hold(self) -> None:
        position = torch.zeros(2, 3)
        velocity = torch.zeros(2, 3)
        omega = torch.tensor(((0.0, 0.0, 0.19), (0.0, 0.0, 0.21)))
        step_success = (
            (position.norm(dim=-1) < 0.05)
            & (velocity.norm(dim=-1) < 0.10)
            & (omega.norm(dim=-1) < 0.20)
        )
        hold = step_success.expand(100, -1).clone()
        survival = torch.ones_like(hold)

        fractions, steady = _position_hold_histories(
            hold,
            survival,
            window_steps=100,
            required_fraction=0.95,
        )

        self.assertEqual(steady[-1].tolist(), [True, False])

    def test_success_is_independent_of_arbitrary_rotation_and_yaw(self) -> None:
        # Rotation is deliberately absent from the production success API.
        rotations = torch.stack(
            (
                _rotation_z(0.0),
                _rotation_z(2.4),
                _rotation_z(-1.2) @ _rotation_x(1.1),
            )
        )
        position = torch.zeros(3, 3)
        velocity = torch.zeros(3, 3)
        omega = torch.zeros(3, 3)
        success = _position_hold_step_success(
            position,
            velocity,
            omega,
            success_position=0.05,
            success_velocity=0.10,
            success_omega=0.20,
        )

        self.assertEqual(rotations.shape, (3, 3, 3))
        self.assertEqual(success.tolist(), [True, True, True])

    def test_arbitrary_yaw_with_zero_rate_succeeds_but_yaw_spin_fails(self) -> None:
        position = torch.zeros(2, 3)
        velocity = torch.zeros(2, 3)
        omega = torch.tensor(((0.0, 0.0, 0.0), (0.0, 0.0, 0.21)))
        success = _position_hold_step_success(
            position,
            velocity,
            omega,
            success_position=0.05,
            success_velocity=0.10,
            success_omega=0.20,
        )

        self.assertEqual(success.tolist(), [True, False])

    def test_single_final_frame_success_is_not_steady(self) -> None:
        hold = torch.zeros(100, 1, dtype=torch.bool)
        hold[-1] = True
        survival = torch.ones_like(hold)
        fractions, steady = _position_hold_histories(
            hold,
            survival,
            window_steps=100,
            required_fraction=0.95,
        )

        self.assertAlmostEqual(float(fractions[-1, 0]), 0.01, places=6)
        self.assertFalse(bool(steady[-1, 0]))

    def test_artificial_trajectory_matches_matlab_parity_fixture(self) -> None:
        # Mirrors matlab_l2f/tests/test_position_hold_semantics.m exactly.
        hold = torch.tensor((True, True, False, True, False, True))[:, None]
        survival = torch.tensor((True, True, True, True, False, False))[:, None]
        fractions, steady = _position_hold_histories(
            hold,
            survival,
            window_steps=4,
            required_fraction=0.75,
        )
        summary = _position_hold_checkpoint(
            hold,
            survival,
            fractions,
            steady,
            completed_steps=6,
            dt=0.01,
        )

        self.assertAlmostEqual(float(summary["final_window_fraction"]), 0.5)
        self.assertAlmostEqual(float(summary["settling_time"]), 0.04)
        self.assertAlmostEqual(float(summary["stay"]), 0.5)
        self.assertFalse(bool(summary["steady"]))
        self.assertFalse(bool(summary["survival"]))

    def test_survival_is_independent_of_position_hold_success(self) -> None:
        hold = torch.zeros(100, 1, dtype=torch.bool)
        survival = torch.ones_like(hold)
        fractions, steady = _position_hold_histories(
            hold,
            survival,
            window_steps=100,
            required_fraction=0.95,
        )
        summary = _position_hold_checkpoint(
            hold,
            survival,
            fractions,
            steady,
            completed_steps=100,
            dt=0.01,
        )

        self.assertFalse(bool(summary["steady"][0]))
        self.assertTrue(bool(summary["survival"][0]))

    def test_stay_is_step_success_fraction_after_first_steady(self) -> None:
        hold = torch.ones(105, 2, dtype=torch.bool)
        hold[102, 0] = False
        survival = torch.ones_like(hold)
        fractions, steady = _position_hold_histories(
            hold,
            survival,
            window_steps=100,
            required_fraction=0.95,
        )
        summary = _position_hold_checkpoint(
            hold,
            survival,
            fractions,
            steady,
            completed_steps=105,
            dt=0.01,
        )

        torch.testing.assert_close(summary["stay"], torch.tensor((0.8, 1.0)))

        final_only = torch.ones(100, 1, dtype=torch.bool)
        final_fractions, final_steady = _position_hold_histories(
            final_only,
            final_only,
            window_steps=100,
            required_fraction=0.95,
        )
        final_summary = _position_hold_checkpoint(
            final_only,
            final_only,
            final_fractions,
            final_steady,
            completed_steps=100,
            dt=0.01,
        )
        self.assertTrue(math.isnan(float(final_summary["stay"][0])))

    def test_deployable_observation_is_twenty_five_dimensional_and_action_is_four_dimensional(self) -> None:
        state = self.sim.reset(4, device="cpu", sample_dynamics=False)
        observation_state = initial_observation_state(4, device="cpu", dtype=state.position.dtype)
        deployable_observation, _ = build_policy_observation(
            state, observation_state, mode=INTEGRAL_OBSERVATION_MODE
        )
        policy = MotorGRUPolicy(encoder_dim=16, hidden_dim=12)

        action, _ = policy(deployable_observation)

        self.assertEqual(deployable_observation.shape, (4, 25))
        self.assertEqual(action.shape, (4, 4))

    def test_external_force_is_not_part_of_deployable_observation(self) -> None:
        state = self.sim.reset(3, device="cpu", sample_dynamics=False)
        observation_state = initial_observation_state(3, device="cpu", dtype=state.position.dtype)
        observation_before, _ = build_policy_observation(
            state, observation_state, mode=INTEGRAL_OBSERVATION_MODE
        )
        state.external_force.copy_(torch.randn_like(state.external_force) * 100.0)
        observation_after, _ = build_policy_observation(
            state, observation_state, mode=INTEGRAL_OBSERVATION_MODE
        )
        torch.testing.assert_close(observation_after, observation_before)


if __name__ == "__main__":
    unittest.main()
