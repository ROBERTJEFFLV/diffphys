from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io
import torch

from model import MotorGRUPolicy
from tools.export_motor_gru_to_mat import ACTION_STATE_KEYS, export_checkpoint_to_mat


def _leaky_relu(value: np.ndarray, negative_slope: float) -> np.ndarray:
    return np.maximum(value, 0.0) + negative_slope * np.minimum(value, 0.0)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _matlab_action_forward(
    weights: dict[str, np.ndarray],
    observation: np.ndarray,
    hidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    negative_slope = float(weights["negative_slope"].item())
    encoded = observation @ weights["encoder_0_weight"].T + weights["encoder_0_bias"]
    encoded = _leaky_relu(encoded, negative_slope)
    encoded = encoded @ weights["encoder_2_weight"].T + weights["encoder_2_bias"]
    encoded = _leaky_relu(encoded, negative_slope)

    ih = encoded @ weights["gru_weight_ih"].T + weights["gru_bias_ih"]
    hh = hidden @ weights["gru_weight_hh"].T + weights["gru_bias_hh"]
    hidden_dim = hidden.shape[1]
    reset_gate = _sigmoid(ih[:, :hidden_dim] + hh[:, :hidden_dim])
    update_gate = _sigmoid(
        ih[:, hidden_dim : 2 * hidden_dim] + hh[:, hidden_dim : 2 * hidden_dim]
    )
    new_gate = np.tanh(
        ih[:, 2 * hidden_dim :] + reset_gate * hh[:, 2 * hidden_dim :]
    )
    next_hidden = (1.0 - update_gate) * new_gate + update_gate * hidden
    head_input = _leaky_relu(next_hidden, negative_slope)
    main_logits = head_input @ weights["motor_head_weight"].T + weights["motor_head_bias"]
    integral_logits = np.zeros_like(main_logits)
    if bool(float(weights.get("enable_integral_residual", np.array([[0.0]])).item())):
        integral_hidden = (
            observation[:, 18:21] @ weights["integral_residual_0_weight"].T
            + weights["integral_residual_0_bias"]
        )
        integral_hidden = _leaky_relu(integral_hidden, negative_slope)
        integral_logits = float(weights["integral_residual_scale"].item()) * (
            integral_hidden @ weights["integral_residual_2_weight"].T
            + weights["integral_residual_2_bias"]
        )
    damping_logits = np.zeros_like(main_logits)
    if bool(float(weights.get("enable_rate_damping_residual", np.array([[0.0]])).item())):
        motor_state = np.tanh(
            head_input @ weights["motor_state_head_weight"].T
            + weights["motor_state_head_bias"]
        )
        damping_input = np.concatenate(
            (head_input, observation[:, 15:18], observation[:, 21:25], motor_state),
            axis=-1,
        )
        damping_hidden = (
            damping_input @ weights["damping_residual_0_weight"].T
            + weights["damping_residual_0_bias"]
        )
        damping_hidden = _leaky_relu(damping_hidden, negative_slope)
        damping_logits = float(weights["damping_residual_scale"].item()) * (
            damping_hidden @ weights["damping_residual_2_weight"].T
            + weights["damping_residual_2_bias"]
        )
    action = np.tanh(main_logits + integral_logits + damping_logits)
    return action, next_hidden


class MotorGRUMatlabExportTest(unittest.TestCase):
    def test_residual_action_branches_export_with_forward_parity(self) -> None:
        torch.manual_seed(91)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for integral_enabled, damping_enabled in ((True, False), (True, True)):
                policy = MotorGRUPolicy(
                    observation_dim=25,
                    encoder_dim=12,
                    hidden_dim=9,
                    enable_integral_residual=integral_enabled,
                    enable_damping_residual=damping_enabled,
                ).eval()
                with torch.no_grad():
                    policy.integral_residual_head[2].weight.normal_(0.0, 0.03)
                    if damping_enabled:
                        policy.damping_residual_head[2].weight.normal_(0.0, 0.02)
                observation = torch.randn(5, 25)
                hidden = torch.randn(5, 9)
                with torch.no_grad():
                    expected_action, expected_hidden = policy(observation, hidden)
                checkpoint = {
                    "model": policy.state_dict(),
                    "args": {
                        "observation_mode": "integral25",
                        "integral_input_frame": "body",
                        "enable_integral_residual": integral_enabled,
                        "enable_rate_damping_residual": damping_enabled,
                        "integral_residual_scale": 0.7,
                        "damping_residual_scale": 0.8,
                    },
                }
                # Match the explicit export scales in the PyTorch policy.
                policy.integral_residual_scale = 0.7
                policy.damping_residual_scale = 0.8
                with torch.no_grad():
                    expected_action, expected_hidden = policy(observation, hidden)
                checkpoint_path = root / f"residual_{damping_enabled}.pt"
                output_path = root / f"residual_{damping_enabled}.mat"
                torch.save(checkpoint, checkpoint_path)
                export_checkpoint_to_mat(checkpoint_path, output_path)
                exported = scipy.io.loadmat(output_path)
                actual_action, actual_hidden = _matlab_action_forward(
                    exported,
                    observation.numpy().astype(np.float64),
                    hidden.numpy().astype(np.float64),
                )
                np.testing.assert_allclose(actual_action, expected_action.numpy(), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(actual_hidden, expected_hidden.numpy(), rtol=2e-5, atol=2e-6)
                self.assertEqual(str(exported["integral_input_frame"].item()).strip(), "body")
                self.assertAlmostEqual(float(exported["integral_input_multiplier"].item()), 1.0)

                override_path = root / f"residual_{damping_enabled}_scale4.mat"
                export_checkpoint_to_mat(
                    checkpoint_path,
                    override_path,
                    integral_input_multiplier=4.0,
                )
                overridden = scipy.io.loadmat(override_path)
                self.assertAlmostEqual(
                    float(overridden["integral_input_multiplier"].item()),
                    4.0,
                )

    def test_legacy40_and_integral25_checkpoints_export_the_action_path(self) -> None:
        torch.manual_seed(41)
        negative_slope = 0.2
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for input_dim, mode, action_only in (
                (40, "legacy40", True),
                (25, "integral25", False),
            ):
                policy = MotorGRUPolicy(
                    observation_dim=input_dim,
                    encoder_dim=12,
                    hidden_dim=9,
                    encoder_depth=2,
                    negative_slope=negative_slope,
                ).eval()
                state_dict = policy.state_dict()
                if action_only:
                    state_dict = {
                        key: state_dict[key].detach().clone() for key in ACTION_STATE_KEYS
                    }
                observation = torch.randn(5, input_dim)
                hidden = torch.randn(5, 9)
                with torch.no_grad():
                    expected_action, expected_hidden = policy(observation, hidden)
                checkpoint = {
                    "model": state_dict,
                    "args": {
                        "negative_slope": negative_slope,
                        "observation_mode": mode,
                        "integral_limit": 0.75,
                        "integral_leak": 0.1,
                    },
                }
                name = mode
                with self.subTest(checkpoint=name):
                    checkpoint_path = root / f"{name}.pt"
                    output_path = root / f"{name}.mat"
                    torch.save(checkpoint, checkpoint_path)
                    export_checkpoint_to_mat(checkpoint_path, output_path)
                    exported = scipy.io.loadmat(output_path)

                    actual_action, actual_hidden = _matlab_action_forward(
                        exported,
                        observation.numpy().astype(np.float64),
                        hidden.numpy().astype(np.float64),
                    )
                    np.testing.assert_allclose(
                        actual_action,
                        expected_action.numpy(),
                        rtol=2.0e-5,
                        atol=2.0e-6,
                    )
                    np.testing.assert_allclose(
                        actual_hidden,
                        expected_hidden.numpy(),
                        rtol=2.0e-5,
                        atol=2.0e-6,
                    )
                    self.assertEqual(int(exported["hidden_dim"].item()), 9)
                    self.assertEqual(int(exported["input_dim"].item()), input_dim)
                    self.assertEqual(str(exported["observation_mode"].item()).strip(), mode)
                    self.assertAlmostEqual(float(exported["integral_limit"].item()), 0.75)
                    self.assertAlmostEqual(
                        float(exported["negative_slope"].item()),
                        negative_slope,
                    )
                    self.assertNotIn("motor_state_head_weight", exported)
                    self.assertNotIn("capability_head_weight", exported)
                    self.assertNotIn("response_head_weight", exported)


if __name__ == "__main__":
    unittest.main()
