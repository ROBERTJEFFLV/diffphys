from __future__ import annotations

import math
import unittest
from argparse import Namespace

import torch

from env_l2f import L2FSimulator
from model import MotorGRUPolicy
from policy_observation import (
    COMPACT_OBSERVATION_MODE,
    CYLINDRICAL_INTEGRAL_CLAMP_MODE,
    INTEGRAL_OBSERVATION_MODE,
    LEGACY_BOX_INTEGRAL_CLAMP_MODE,
    LEGACY_OBSERVATION_MODE,
    PolicyObservationState,
    build_policy_observation,
    cylindrical_integral_radial_limit,
    initial_observation_state,
    reset_observation_state,
    update_position_integral,
    world_integral_to_body,
)
from train import _validate_cuda_full_observation_mode, _validate_cuda_full_residual_support


class PolicyObservationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.state = L2FSimulator().reset(4, device="cpu", sample_dynamics=False)
        self.obs_state = initial_observation_state(4, device="cpu", dtype=self.state.position.dtype)

    def test_legacy_state_error_is_an_affine_duplicate_of_one_sample(self) -> None:
        observation, _ = build_policy_observation(
            self.state, self.obs_state, mode=LEGACY_OBSERVATION_MODE, noise_max=0.1
        )
        physical = observation[:, :18]
        duplicate = observation[:, 18:36].clone()
        duplicate[:, (6, 10, 14)] += 1.0
        torch.testing.assert_close(duplicate, physical)

    def test_shared_python_matlab_integral25_fixture(self) -> None:
        self.state.position[0] = torch.tensor([1.0, 2.0, 3.0])
        self.state.velocity[0] = torch.tensor([0.1, 0.2, 0.3])
        self.state.rotation[0] = torch.eye(3)
        self.state.omega[0] = torch.tensor([0.4, 0.5, 0.6])
        self.state.previous_action[0] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.obs_state.integral_position[0] = torch.tensor([0.1, -0.2, 0.3])
        observation, _ = build_policy_observation(
            self.state, self.obs_state, mode=INTEGRAL_OBSERVATION_MODE
        )
        expected = torch.tensor([
            1.0, 2.0, 3.0, 0.1, 0.2, 0.3,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.4, 0.5, 0.6, 0.1, -0.2, 0.3, 0.1, 0.2, 0.3, 0.4,
        ])
        torch.testing.assert_close(observation[0], expected)

    def test_world_integral_is_rotated_to_body_axes(self) -> None:
        rotation = torch.tensor(
            [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
        )
        integral_world = torch.tensor([[1.0, 0.0, 0.0]])
        expected = torch.tensor([[0.0, -1.0, 0.0]])
        torch.testing.assert_close(
            world_integral_to_body(integral_world, rotation), expected
        )

    def test_yaw_changes_body_integral_through_rotation_transpose(self) -> None:
        self.obs_state.integral_position[:] = torch.tensor([1.0, 0.0, 0.0])
        self.state.rotation[:] = torch.eye(3)
        identity, _ = build_policy_observation(
            self.state,
            self.obs_state,
            mode=INTEGRAL_OBSERVATION_MODE,
            integral_input_frame="body",
        )
        self.state.rotation[:] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        yawed, _ = build_policy_observation(
            self.state,
            self.obs_state,
            mode=INTEGRAL_OBSERVATION_MODE,
            integral_input_frame="body",
        )
        torch.testing.assert_close(
            identity[:, 18:21], torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
        )
        torch.testing.assert_close(
            yawed[:, 18:21], torch.tensor([[0.0, -1.0, 0.0]]).repeat(4, 1)
        )

    def test_integral_input_multiplier_only_scales_network_feature(self) -> None:
        self.obs_state.integral_position[:] = torch.tensor([0.25, -0.5, 0.125])
        unscaled, _ = build_policy_observation(
            self.state,
            self.obs_state,
            mode=INTEGRAL_OBSERVATION_MODE,
            integral_input_frame="body",
        )
        scaled, _ = build_policy_observation(
            self.state,
            self.obs_state,
            mode=INTEGRAL_OBSERVATION_MODE,
            integral_input_frame="body",
            integral_input_multiplier=4.0,
        )
        torch.testing.assert_close(scaled[:, :18], unscaled[:, :18])
        torch.testing.assert_close(scaled[:, 18:21], 4.0 * unscaled[:, 18:21])
        torch.testing.assert_close(scaled[:, 21:25], unscaled[:, 21:25])
        torch.testing.assert_close(
            self.obs_state.integral_position,
            torch.tensor([[0.25, -0.5, 0.125]]).repeat(4, 1),
        )

    def test_40d_to_25d_fold_preserves_action_and_hidden_at_zero_integral(self) -> None:
        old = MotorGRUPolicy(observation_dim=40, encoder_dim=24, hidden_dim=19).eval()
        new = MotorGRUPolicy(observation_dim=25, encoder_dim=24, hidden_dim=19).eval()
        new.load_compatible_state_dict(old.state_dict())
        old_observation, _ = build_policy_observation(
            self.state, self.obs_state, mode=LEGACY_OBSERVATION_MODE
        )
        new_observation, _ = build_policy_observation(
            self.state, self.obs_state, mode=INTEGRAL_OBSERVATION_MODE
        )
        hidden = torch.randn(4, 19)
        with torch.no_grad():
            old_action, old_hidden = old(old_observation, hidden)
            new_action, new_hidden = new(new_observation, hidden)
        torch.testing.assert_close(new_action, old_action, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(new_hidden, old_hidden, atol=3e-7, rtol=2e-6)

    def test_integral_persists_detaches_clamps_and_resets(self) -> None:
        position = torch.tensor([[2.0, -3.0, 0.5]]).repeat(4, 1).requires_grad_()
        updated = update_position_integral(
            self.obs_state, position, dt=0.1, integral_limit=0.15, integral_leak=0.0
        )
        torch.testing.assert_close(
            updated.integral_position[0], torch.tensor([0.15, -0.15, 0.05])
        )
        detached = updated.detach()
        self.assertFalse(detached.integral_position.requires_grad)
        reset_observation_state(detached, torch.tensor([False, True, False, True]))
        torch.testing.assert_close(detached.integral_position[1], torch.zeros(3))
        torch.testing.assert_close(detached.integral_position[0], updated.integral_position.detach()[0])

    def test_default_integral_clamp_is_bitwise_identical_to_explicit_legacy_box(self) -> None:
        prior = torch.randn(128, 3, dtype=torch.float64)
        position = torch.randn(128, 3, dtype=torch.float64)
        default = update_position_integral(
            PolicyObservationState(prior),
            position,
            dt=0.01,
            integral_limit=0.5,
            integral_leak=0.03,
        )
        explicit = update_position_integral(
            PolicyObservationState(prior),
            position,
            dt=0.01,
            integral_limit=0.5,
            integral_leak=0.03,
            integral_clamp_mode=LEGACY_BOX_INTEGRAL_CLAMP_MODE,
        )
        self.assertTrue(torch.equal(default.integral_position, explicit.integral_position))

    def test_cylindrical_integral_clamp_is_yaw_equivariant(self) -> None:
        dtype = torch.float64
        yaw = torch.tensor([0.37, -1.2, 2.4], dtype=dtype)
        cosine, sine = yaw.cos(), yaw.sin()
        rotation = torch.zeros(3, 3, 3, dtype=dtype)
        rotation[:, 0, 0] = cosine
        rotation[:, 0, 1] = -sine
        rotation[:, 1, 0] = sine
        rotation[:, 1, 1] = cosine
        rotation[:, 2, 2] = 1.0
        raw = torch.tensor(
            [[2.0, -0.7, 0.8], [-1.3, 1.8, -0.9], [0.2, 0.3, 0.1]],
            dtype=dtype,
        )
        rotated_raw = torch.bmm(rotation, raw.unsqueeze(-1)).squeeze(-1)
        zero = torch.zeros_like(raw)
        direct = update_position_integral(
            PolicyObservationState(zero),
            raw,
            dt=1.0,
            integral_limit=0.5,
            integral_clamp_mode=CYLINDRICAL_INTEGRAL_CLAMP_MODE,
        ).integral_position
        rotated = update_position_integral(
            PolicyObservationState(zero),
            rotated_raw,
            dt=1.0,
            integral_limit=0.5,
            integral_clamp_mode=CYLINDRICAL_INTEGRAL_CLAMP_MODE,
        ).integral_position
        expected = torch.bmm(rotation, direct.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(rotated, expected, atol=2.0e-15, rtol=2.0e-15)

    def test_cylindrical_integral_clamp_has_exact_radial_and_vertical_bounds(self) -> None:
        limit = 0.5
        raw = torch.tensor(
            [[3.0, 4.0, 2.0], [-4.0, 3.0, -2.0]], dtype=torch.float64
        )
        clamped = update_position_integral(
            PolicyObservationState(torch.zeros_like(raw)),
            raw,
            dt=1.0,
            integral_limit=limit,
            integral_clamp_mode=CYLINDRICAL_INTEGRAL_CLAMP_MODE,
        ).integral_position
        expected_radius = 2.0 * limit / math.sqrt(math.pi)
        self.assertEqual(cylindrical_integral_radial_limit(limit), expected_radius)
        torch.testing.assert_close(
            torch.linalg.vector_norm(clamped[:, :2], dim=-1),
            torch.full((2,), expected_radius, dtype=torch.float64),
            atol=1.0e-15,
            rtol=1.0e-15,
        )
        torch.testing.assert_close(clamped[:, 2], torch.tensor([limit, -limit], dtype=torch.float64))

    def test_unknown_integral_clamp_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integral_clamp_mode"):
            update_position_integral(
                self.obs_state,
                torch.zeros_like(self.obs_state.integral_position),
                dt=0.01,
                integral_limit=0.5,
                integral_clamp_mode="octagon",
            )

    def test_integral_leak_uses_observed_position_and_compact_has_no_integral(self) -> None:
        self.obs_state.integral_position.fill_(1.0)
        observed = torch.zeros(4, 3)
        updated = update_position_integral(
            self.obs_state, observed, dt=0.1, integral_limit=2.0, integral_leak=0.5
        )
        torch.testing.assert_close(updated.integral_position, torch.full((4, 3), 0.95))
        compact, _ = build_policy_observation(
            self.state, updated, mode=COMPACT_OBSERVATION_MODE
        )
        integral, _ = build_policy_observation(
            self.state, updated, mode=INTEGRAL_OBSERVATION_MODE
        )
        self.assertEqual(compact.shape[-1], 22)
        self.assertEqual(integral.shape[-1], 25)
        torch.testing.assert_close(integral[:, 18:21], updated.integral_position)

    def test_privileged_motor_capability_and_force_are_not_actor_inputs(self) -> None:
        before, _ = build_policy_observation(
            self.state, self.obs_state, mode=INTEGRAL_OBSERVATION_MODE
        )
        self.state.motor.add_(torch.randn_like(self.state.motor))
        self.state.external_force.add_(torch.randn_like(self.state.external_force))
        self.state.alpha_roll_max.mul_(1.7)
        self.state.alpha_yaw_max.mul_(0.4)
        self.state.motor_time_falling.mul_(2.0)
        after, _ = build_policy_observation(
            self.state, self.obs_state, mode=INTEGRAL_OBSERVATION_MODE
        )
        torch.testing.assert_close(after, before)

    def test_cuda_full_rejects_compact_observations_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy 40D"):
            _validate_cuda_full_observation_mode("cuda-full", INTEGRAL_OBSERVATION_MODE)
        _validate_cuda_full_observation_mode("cuda-full", LEGACY_OBSERVATION_MODE)
        options = Namespace(
            enable_integral_residual=True,
            enable_rate_damping_residual=False,
            w_omega_decay=0.0,
        )
        with self.assertRaisesRegex(ValueError, "does not implement"):
            _validate_cuda_full_residual_support("cuda-full", options)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_cuda_observation_parity_without_noise(self) -> None:
        self.obs_state.integral_position.normal_()
        cpu, _ = build_policy_observation(
            self.state,
            self.obs_state,
            mode=INTEGRAL_OBSERVATION_MODE,
            integral_input_frame="body",
        )
        cuda_state = type(self.state)(**{
            field: value.cuda() for field, value in vars(self.state).items()
        })
        cuda_obs_state = initial_observation_state(4, device="cuda", dtype=self.state.position.dtype)
        cuda_obs_state.integral_position.copy_(self.obs_state.integral_position.cuda())
        gpu, _ = build_policy_observation(
            cuda_state,
            cuda_obs_state,
            mode=INTEGRAL_OBSERVATION_MODE,
            integral_input_frame="body",
        )
        torch.testing.assert_close(gpu.cpu(), cpu)


if __name__ == "__main__":
    unittest.main()
