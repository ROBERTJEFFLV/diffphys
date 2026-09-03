from __future__ import annotations

import math
import unittest

import torch

from env_l2f import L2FSimulator
from l2f_cuda_backend import cuda_step, load_extension
from model import MotorGRUPolicy
from train import _reset_detach_hidden, _tail_axis_diagnostics
from training_objectives import (
    accumulated_episode_objective,
    independent_cvar_tail_loss,
    multistep_omega_decay_loss,
    retain_action_mse,
    threshold_cvar_tail_loss,
)


class ThresholdTailLossTest(unittest.TestCase):
    def test_multistep_omega_decay_matches_configured_ratios(self) -> None:
        omega = torch.zeros(26, 2, 3)
        omega[..., 0] = 1.0
        omega.requires_grad_()
        result = multistep_omega_decay_loss(
            omega,
            torch.tensor([True, False]),
            horizons=(5, 10, 25),
            beta=(0.2, 0.3, 0.5),
            rho=(0.90, 0.75, 0.50),
            eps=1.0e-12,
        )
        expected = 0.2 * 0.10**2 + 0.3 * 0.25**2 + 0.5 * 0.50**2
        self.assertAlmostEqual(float(result.loss.item()), expected, places=6)
        result.loss.backward()
        self.assertGreater(float(omega.grad.abs().sum().item()), 0.0)

    def test_multistep_decay_does_not_press_already_stable_rates(self) -> None:
        omega = torch.full((26, 3, 3), 0.05, requires_grad=True)
        result = multistep_omega_decay_loss(
            omega,
            torch.ones(3, dtype=torch.bool),
            horizons=(5, 10, 25),
        )
        self.assertEqual(float(result.loss.item()), 0.0)
        result.loss.backward()
        torch.testing.assert_close(omega.grad, torch.zeros_like(omega))

    def test_independent_cvar_selects_different_position_and_omega_samples(self) -> None:
        batch = 10
        position = torch.zeros(2, batch, 3, requires_grad=True)
        omega = torch.zeros(2, batch, 3, requires_grad=True)
        position.data[:, 8:, 0] = torch.tensor([0.09, 0.10])
        omega.data[:, :2, 2] = torch.tensor([0.35, 0.40])
        result = independent_cvar_tail_loss(
            position, omega, q_position=0.2, q_omega=0.2, window_steps=2
        )
        self.assertEqual(set(result.position_selected_indices.tolist()), {8, 9})
        self.assertEqual(set(result.omega_selected_indices.tolist()), {0, 1})
        self.assertEqual(result.selected_overlap_fraction, 0.0)
        result.loss.backward()
        self.assertTrue(torch.all(position.grad[:, 8:].abs().sum(dim=(0, 2)) > 0))
        self.assertTrue(torch.all(omega.grad[:, :2].abs().sum(dim=(0, 2)) > 0))

    def test_selects_only_worst_twenty_percent_and_backpropagates(self) -> None:
        batch = 10
        position = torch.zeros(4, batch, 3, requires_grad=True)
        omega = torch.zeros(4, batch, 3, requires_grad=True)
        position.data[:, :, 0] = torch.linspace(0.01, 0.10, batch)
        omega.data[:, :, 1] = torch.linspace(0.01, 0.30, batch)

        result = threshold_cvar_tail_loss(
            position,
            omega,
            cvar_fraction=0.20,
            lambda_tail_omega=1.0,
            window_steps=4,
        )

        self.assertEqual(result.selected_mask.sum().item(), math.ceil(0.20 * batch))
        self.assertEqual(set(result.selected_indices.tolist()), {8, 9})
        result.loss.backward()
        grad_per_sample = position.grad.abs().sum(dim=(0, 2)) + omega.grad.abs().sum(dim=(0, 2))
        self.assertTrue(torch.all(grad_per_sample[result.selected_mask] > 0))
        self.assertTrue(torch.all(grad_per_sample[~result.selected_mask] == 0))

    def test_margins_are_zero_at_task_thresholds(self) -> None:
        position = torch.tensor([[[0.05, 0.0, 0.0]]], requires_grad=True)
        omega = torch.tensor([[[0.0, 0.20, 0.0]]], requires_grad=True)
        result = threshold_cvar_tail_loss(
            position,
            omega,
            cvar_fraction=1.0,
            lambda_tail_omega=3.0,
            window_steps=1,
        )
        torch.testing.assert_close(result.per_sample_position, torch.zeros(1))
        torch.testing.assert_close(result.per_sample_omega, torch.zeros(1))
        torch.testing.assert_close(result.loss, torch.zeros(()))

    def test_velocity_is_not_an_input_to_tail_objective(self) -> None:
        parameter_names = threshold_cvar_tail_loss.__annotations__
        self.assertNotIn("velocity_history", parameter_names)

    def test_h250_and_h500_tail_events_reach_their_own_trajectory(self) -> None:
        first = torch.zeros(250, 2, 3, requires_grad=True)
        second = torch.zeros(250, 2, 3, requires_grad=True)
        first.data[:, 0, 0] = 0.08
        second.data[:, 1, 0] = 0.09
        omega_first = first * 0.0
        omega_second = second * 0.0
        early = independent_cvar_tail_loss(
            first, omega_first, q_position=0.5, q_omega=0.5, window_steps=100
        ).position_cvar
        final_position = torch.cat((first.detach(), second), dim=0)
        final_omega = torch.cat((omega_first.detach(), omega_second), dim=0)
        final = independent_cvar_tail_loss(
            final_position, final_omega, q_position=0.5, q_omega=0.5, window_steps=100
        ).position_cvar
        (0.25 * early + final).backward()
        self.assertGreater(float(first.grad.abs().sum()), 0.0)
        self.assertGreater(float(second.grad.abs().sum()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_pytorch_and_compact_cuda_tail_loss_match(self) -> None:
        load_extension()
        sim = L2FSimulator()
        initial = sim.reset(4, device="cuda", sample_dynamics=False)
        torch_state = type(initial)(**{
            name: value.detach().clone() for name, value in vars(initial).items()
        })
        cuda_state = type(initial)(**{
            name: value.detach().clone() for name, value in vars(initial).items()
        })
        generator = torch.Generator(device="cuda").manual_seed(91)
        actions = [torch.rand(4, 4, device="cuda", generator=generator) * 0.4 - 0.2 for _ in range(4)]
        torch_positions, torch_omegas = [], []
        cuda_positions, cuda_omegas = [], []
        for action in actions:
            torch_state = sim.step(torch_state, action, grad_decay=1.0)
            cuda_state = cuda_step(cuda_state, action, sim.params, grad_decay=1.0)
            torch_positions.append(torch_state.position)
            torch_omegas.append(torch_state.omega)
            cuda_positions.append(cuda_state.position)
            cuda_omegas.append(cuda_state.omega)
        torch_loss = independent_cvar_tail_loss(
            torch.stack(torch_positions), torch.stack(torch_omegas), window_steps=4
        ).loss
        cuda_loss = independent_cvar_tail_loss(
            torch.stack(cuda_positions), torch.stack(cuda_omegas), window_steps=4
        ).loss
        torch.testing.assert_close(cuda_loss, torch_loss, atol=2e-4, rtol=2e-4)


class BaselineRetainTest(unittest.TestCase):
    def test_baseline_parameters_receive_no_gradient(self) -> None:
        torch.manual_seed(4)
        candidate = MotorGRUPolicy(encoder_dim=16, hidden_dim=12)
        baseline = MotorGRUPolicy(encoder_dim=16, hidden_dim=12)
        baseline.load_state_dict(candidate.state_dict())
        for parameter in baseline.parameters():
            parameter.requires_grad_(False)

        observation = torch.randn(5, 25)
        candidate_action, _ = candidate(observation)
        with torch.no_grad():
            baseline_action, _ = baseline(observation)
        loss = retain_action_mse(
            candidate_action,
            baseline_action,
            torch.tensor([True, True, False, False, False]),
        )
        loss.backward()

        self.assertTrue(any(parameter.grad is not None for parameter in candidate.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in baseline.parameters()))

    def test_candidate_and_baseline_hidden_are_independent_reset_and_detach(self) -> None:
        candidate_source = torch.arange(12.0, requires_grad=True).reshape(3, 4)
        baseline_source = (torch.arange(12.0) + 100.0).requires_grad_().reshape(3, 4)
        reset_mask = torch.tensor([False, True, False])
        candidate_hidden = _reset_detach_hidden(candidate_source, reset_mask)
        baseline_hidden = _reset_detach_hidden(baseline_source, reset_mask)

        self.assertEqual(candidate_hidden.data_ptr() == baseline_hidden.data_ptr(), False)
        self.assertFalse(candidate_hidden.requires_grad)
        self.assertFalse(baseline_hidden.requires_grad)
        torch.testing.assert_close(candidate_hidden[1], torch.zeros(4))
        torch.testing.assert_close(baseline_hidden[1], torch.zeros(4))
        torch.testing.assert_close(candidate_hidden[0], candidate_source.detach()[0])
        torch.testing.assert_close(baseline_hidden[0], baseline_source.detach()[0])


class EpisodeBoundaryWeightingTest(unittest.TestCase):
    def test_h250_h500_event_weight_is_not_halved(self) -> None:
        parameter = torch.tensor(1.0, requires_grad=True)
        for event in (torch.zeros(()), 0.001 * parameter):
            objective = accumulated_episode_objective(
                0.0 * parameter,
                event,
                segments_per_episode=2,
                correct_episode_event_weighting=True,
            )
            objective.backward(retain_graph=True)
        parameter.grad.div_(2.0)
        self.assertAlmostEqual(float(parameter.grad), 0.001, places=9)


class TailAxisDiagnosticsTest(unittest.TestCase):
    def test_axis_rms_peak_and_bounded_motion_fixture(self) -> None:
        time = torch.arange(100, dtype=torch.float64) * 0.01
        omega = torch.zeros(100, 1, 3, dtype=torch.float64)
        omega[:, 0, 0] = torch.sin(2.0 * math.pi * 5.0 * time)
        omega[:, 0, 1] = 0.25
        action = torch.zeros(100, 1, 4, dtype=torch.float64)
        action[:, 0, 0] = torch.sin(2.0 * math.pi * 2.0 * time)
        diagnostics = _tail_axis_diagnostics(
            torch.full((100, 1), 0.04, dtype=torch.float64),
            torch.full((100, 1), 0.09, dtype=torch.float64),
            omega,
            action,
            dt=0.01,
        )

        self.assertAlmostEqual(float(diagnostics["omega_rms_axis"][0, 0]), math.sqrt(0.5), places=10)
        self.assertAlmostEqual(float(diagnostics["omega_rms_axis"][0, 1]), 0.25, places=10)
        self.assertAlmostEqual(float(diagnostics["omega_peak_hz_axis"][0, 0]), 5.0, places=10)
        self.assertAlmostEqual(float(diagnostics["omega_peak_hz_axis"][0, 1]), 0.0, places=10)
        self.assertTrue(bool(diagnostics["strict_bounded_angular_motion"][0]))
        self.assertTrue(bool(diagnostics["loose_bounded_angular_motion"][0]))


if __name__ == "__main__":
    unittest.main()
