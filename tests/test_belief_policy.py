from __future__ import annotations

import unittest

import torch

from env_l2f import CAPABILITY_BOUNDS, L2FSimulator, normalized_capability_target
from model import AUXILIARY_STATE_PREFIXES, MotorGRUPolicy


class BeliefPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.policy = MotorGRUPolicy(encoder_dim=16, hidden_dim=12)
        self.observation = torch.randn(3, 25)
        self.hidden = self.policy.initial_hidden(3, device="cpu")

    def test_deployment_and_auxiliary_interfaces_share_action_path(self) -> None:
        action, hidden = self.policy(
            self.observation,
            self.hidden.clone(),
        )
        aux_action, aux_hidden, auxiliary = self.policy.forward_with_aux(
            self.observation,
            self.hidden.clone(),
        )

        torch.testing.assert_close(aux_action, action)
        torch.testing.assert_close(aux_hidden, hidden)
        self.assertEqual(auxiliary["motor_state"].shape, (3, 4))
        self.assertEqual(auxiliary["capability"].shape, (3, 6))
        self.assertEqual(auxiliary["response"].shape, (3, 6))
        self.assertTrue((auxiliary["motor_state"].abs() <= 1.0).all().item())

    def test_auxiliary_gradient_shapes_belief_but_not_action_head(self) -> None:
        _, _, auxiliary = self.policy.forward_with_aux(
            self.observation,
        )
        motor_target = torch.randn_like(auxiliary["motor_state"]).tanh().detach()
        capability_target = torch.randn_like(auxiliary["capability"]).detach()
        loss = torch.nn.functional.smooth_l1_loss(auxiliary["motor_state"], motor_target)
        loss = loss + torch.nn.functional.smooth_l1_loss(
            auxiliary["capability"],
            capability_target,
        )
        response_target = torch.randn_like(auxiliary["response"]).detach()
        loss = loss + torch.nn.functional.smooth_l1_loss(
            auxiliary["response"],
            response_target,
        )
        loss.backward()

        self.assertIsNotNone(self.policy.encoder[0].weight.grad)
        self.assertIsNotNone(self.policy.gru.weight_ih.grad)
        self.assertIsNotNone(self.policy.motor_state_head.weight.grad)
        self.assertIsNotNone(self.policy.capability_head.weight.grad)
        self.assertIsNotNone(self.policy.response_head.weight.grad)
        self.assertIsNone(self.policy.motor_head.weight.grad)

    def test_action_only_checkpoint_is_accepted_with_strict_allowlist(self) -> None:
        action_only = {
            key: value.detach().clone()
            for key, value in self.policy.state_dict().items()
            if not key.startswith(AUXILIARY_STATE_PREFIXES)
        }
        restored = MotorGRUPolicy(encoder_dim=16, hidden_dim=12)
        missing, unexpected = restored.load_compatible_state_dict(action_only)

        self.assertEqual(unexpected, [])
        self.assertEqual(
            set(missing),
            {
                "motor_state_head.weight",
                "motor_state_head.bias",
                "capability_head.weight",
                "capability_head.bias",
                "response_head.weight",
                "response_head.bias",
            },
        )
        expected, _ = self.policy(self.observation)
        actual, _ = restored(self.observation)
        torch.testing.assert_close(actual, expected)

    def test_non_auxiliary_checkpoint_mismatch_is_rejected(self) -> None:
        malformed = dict(self.policy.state_dict())
        del malformed["motor_head.bias"]
        with self.assertRaisesRegex(RuntimeError, "motor_head.bias"):
            self.policy.load_compatible_state_dict(malformed)

    def test_damping_checkpoint_requires_its_deployed_motor_state_head(self) -> None:
        source = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=16,
            hidden_dim=12,
            enable_damping_residual=True,
        )
        malformed = dict(source.state_dict())
        del malformed["motor_state_head.weight"]
        del malformed["motor_state_head.bias"]
        restored = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=16,
            hidden_dim=12,
            enable_damping_residual=True,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "incomplete damping deployment checkpoint.*motor_state_head.weight",
        ):
            restored.load_compatible_state_dict(malformed)

    def test_old_action_only_checkpoint_can_initialize_zeroed_damping_branch(self) -> None:
        action_only = {
            key: value.detach().clone()
            for key, value in self.policy.state_dict().items()
            if not key.startswith(AUXILIARY_STATE_PREFIXES)
        }
        restored = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=16,
            hidden_dim=12,
            enable_damping_residual=True,
        )

        missing, unexpected = restored.load_compatible_state_dict(action_only)
        expected_action, expected_hidden = self.policy(
            self.observation,
            self.hidden.clone(),
        )
        actual_action, actual_hidden = restored(
            self.observation,
            self.hidden.clone(),
        )

        self.assertIn("motor_state_head.weight", missing)
        self.assertIn("damping_residual_head.0.weight", missing)
        self.assertEqual(unexpected, [])
        torch.testing.assert_close(actual_action, expected_action, atol=0.0, rtol=0.0)
        torch.testing.assert_close(actual_hidden, expected_hidden, atol=0.0, rtol=0.0)

    def test_partial_integral_checkpoint_is_rejected(self) -> None:
        source = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=16,
            hidden_dim=12,
            enable_integral_residual=True,
        )
        malformed = dict(source.state_dict())
        del malformed["integral_residual_head.2.bias"]
        restored = MotorGRUPolicy(
            observation_dim=25,
            encoder_dim=16,
            hidden_dim=12,
            enable_integral_residual=True,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "incomplete integral deployment checkpoint.*integral_residual_head.2.bias",
        ):
            restored.load_compatible_state_dict(malformed)

    def test_capability_bounds_map_to_log_space_endpoints(self) -> None:
        state = L2FSimulator().reset(2, device="cpu")
        bounds = torch.tensor(CAPABILITY_BOUNDS)
        state.thrust_to_weight = bounds[:, 0:1].new_tensor([bounds[0, 0], bounds[0, 1]])
        state.alpha_roll_max = bounds[:, 0:1].new_tensor([bounds[1, 0], bounds[1, 1]])
        state.eta_yaw = bounds[:, 0:1].new_tensor([bounds[2, 0], bounds[2, 1]])
        state.jz_over_jxy = bounds[:, 0:1].new_tensor([bounds[3, 0], bounds[3, 1]])
        state.motor_time_rising = bounds[:, 0:1].new_tensor([bounds[4, 0], bounds[4, 1]])
        state.motor_time_falling = bounds[:, 0:1].new_tensor([bounds[5, 0], bounds[5, 1]])

        target = normalized_capability_target(state)
        torch.testing.assert_close(target[0], -torch.ones(6), atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(target[1], torch.ones(6), atol=1.0e-6, rtol=0.0)

    def test_physical_fit_sampling_can_isolate_vehicle_dynamics(self) -> None:
        state = L2FSimulator().reset(
            8,
            device="cpu",
            sample_dynamics=True,
            sampled_dynamics_level="broad",
            broad_sampler="physical-fit",
            balanced_dynamics_sampling=True,
            sample_external_force=False,
        )

        torch.testing.assert_close(state.external_force, torch.zeros_like(state.external_force))
        self.assertGreater(torch.unique(state.thrust_to_weight).numel(), 1)
        self.assertTrue(torch.isfinite(normalized_capability_target(state)).all().item())

    def test_unimplemented_sampling_level_is_not_silently_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not implemented"):
            L2FSimulator().reset(
                2,
                device="cpu",
                sample_dynamics=True,
                sampled_dynamics_level="small",
            )


if __name__ == "__main__":
    unittest.main()
