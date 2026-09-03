from __future__ import annotations

import unittest

import torch

from model import MotorGRUPolicy, compensate_integral_input_scale_


class ResidualControlTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(73)
        self.observation = torch.randn(7, 25)
        self.hidden = torch.randn(7, 15)

    def test_integral_residual_zero_initialization_preserves_p4b(self) -> None:
        p4b = MotorGRUPolicy(
            observation_dim=25, encoder_dim=20, hidden_dim=15
        ).eval()
        q1 = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=20,
            hidden_dim=15,
            enable_integral_residual=True,
        ).eval()
        missing, unexpected = q1.load_compatible_state_dict(p4b.state_dict())
        self.assertFalse(unexpected)
        self.assertTrue(any(key.startswith("integral_residual_head.") for key in missing))
        with torch.no_grad():
            expected_action, expected_hidden = p4b(self.observation, self.hidden)
            actual_action, actual_hidden, details = q1.forward_with_aux(
                self.observation, self.hidden
            )
        torch.testing.assert_close(actual_action, expected_action, atol=0.0, rtol=0.0)
        torch.testing.assert_close(actual_hidden, expected_hidden, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            details["integral_action_contribution"], torch.zeros_like(actual_action)
        )

    def test_damping_residual_zero_initialization_preserves_q1(self) -> None:
        q1 = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=20,
            hidden_dim=15,
            enable_integral_residual=True,
        ).eval()
        q2 = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=20,
            hidden_dim=15,
            enable_integral_residual=True,
            enable_damping_residual=True,
        ).eval()
        missing, unexpected = q2.load_compatible_state_dict(q1.state_dict())
        self.assertFalse(unexpected)
        self.assertTrue(any(key.startswith("damping_residual_head.") for key in missing))
        with torch.no_grad():
            expected_action, expected_hidden = q1(self.observation, self.hidden)
            actual_action, actual_hidden, details = q2.forward_with_aux(
                self.observation, self.hidden
            )
        torch.testing.assert_close(actual_action, expected_action, atol=0.0, rtol=0.0)
        torch.testing.assert_close(actual_hidden, expected_hidden, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            details["damping_action_contribution"], torch.zeros_like(actual_action)
        )

    def test_damping_deployment_path_is_deterministic_from_observation(self) -> None:
        policy = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=20,
            hidden_dim=15,
            enable_damping_residual=True,
        ).eval()
        with torch.no_grad():
            policy.damping_residual_head[2].weight.fill_(0.02)
            first_action, first_hidden = policy(self.observation, self.hidden)
            second_action, second_hidden = policy(self.observation.clone(), self.hidden)
        torch.testing.assert_close(second_action, first_action)
        torch.testing.assert_close(second_hidden, first_hidden)

    def test_inverse_weight_scaling_preserves_nonzero_integral_action_hidden(self) -> None:
        base = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=20,
            hidden_dim=15,
            enable_integral_residual=True,
            enable_damping_residual=True,
        ).eval()
        with torch.no_grad():
            base.integral_residual_head[2].weight.normal_(0.0, 0.03)
            base.damping_residual_head[2].weight.normal_(0.0, 0.02)
        for multiplier in (0.5, 2.0, 4.0):
            scaled = MotorGRUPolicy(
                observation_dim=25,
                encoder_dim=20,
                hidden_dim=15,
                enable_integral_residual=True,
                enable_damping_residual=True,
            ).eval()
            scaled.load_state_dict(base.state_dict())
            compensate_integral_input_scale_(scaled, multiplier)
            scaled_observation = self.observation.clone()
            scaled_observation[:, 18:21].mul_(multiplier)
            with torch.no_grad():
                expected_action, expected_hidden = base(self.observation, self.hidden)
                actual_action, actual_hidden = scaled(scaled_observation, self.hidden)
            torch.testing.assert_close(actual_action, expected_action, atol=0.0, rtol=0.0)
            torch.testing.assert_close(actual_hidden, expected_hidden, atol=0.0, rtol=0.0)

    def test_inverse_weight_scaling_rejects_zero(self) -> None:
        policy = MotorGRUPolicy(observation_dim=25)
        with self.assertRaisesRegex(ValueError, "multiplier > 0"):
            compensate_integral_input_scale_(policy, 0.0)


if __name__ == "__main__":
    unittest.main()
