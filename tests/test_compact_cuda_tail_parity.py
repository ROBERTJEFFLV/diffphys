from __future__ import annotations

import unittest

import torch

from env_l2f import L2FSimulator, L2FState
from l2f_cuda_backend import cuda_step, load_extension
from training_objectives import multistep_omega_decay_loss, threshold_cvar_tail_loss


def clone_state(state: L2FState) -> L2FState:
    return L2FState(
        **{
            name: getattr(state, name).detach().clone()
            for name in L2FState.__dataclass_fields__
        }
    )


@unittest.skipUnless(torch.cuda.is_available(), "compact CUDA parity requires CUDA")
class CompactCudaTailParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_extension()

    def test_threshold_tail_value_and_action_gradient_match_torch(self) -> None:
        torch.manual_seed(19)
        torch.cuda.manual_seed_all(19)
        sim = L2FSimulator()
        initial = sim.reset(10, device="cuda", sample_dynamics=False)
        torch_state = clone_state(initial)
        cuda_state = clone_state(initial)
        actions_torch = (torch.randn(4, 10, 4, device="cuda") * 0.05).requires_grad_()
        actions_cuda = actions_torch.detach().clone().requires_grad_()
        torch_positions: list[torch.Tensor] = []
        torch_omegas: list[torch.Tensor] = []
        cuda_positions: list[torch.Tensor] = []
        cuda_omegas: list[torch.Tensor] = []
        for step in range(4):
            torch_state = sim.step(torch_state, actions_torch[step], grad_decay=1.0)
            cuda_state = cuda_step(cuda_state, actions_cuda[step], sim.params, grad_decay=1.0)
            torch_positions.append(torch_state.position)
            torch_omegas.append(torch_state.omega)
            cuda_positions.append(cuda_state.position)
            cuda_omegas.append(cuda_state.omega)
        torch_result = threshold_cvar_tail_loss(
            torch.stack(torch_positions),
            torch.stack(torch_omegas),
            cvar_fraction=0.20,
            lambda_tail_omega=1.0,
            window_steps=4,
        )
        cuda_result = threshold_cvar_tail_loss(
            torch.stack(cuda_positions),
            torch.stack(cuda_omegas),
            cvar_fraction=0.20,
            lambda_tail_omega=1.0,
            window_steps=4,
        )
        torch_result.loss.backward()
        cuda_result.loss.backward()

        torch.testing.assert_close(cuda_result.loss, torch_result.loss, rtol=2.0e-5, atol=2.0e-5)
        self.assertEqual(
            set(cuda_result.selected_indices.tolist()),
            set(torch_result.selected_indices.tolist()),
        )
        torch.testing.assert_close(actions_cuda.grad, actions_torch.grad, rtol=3.0e-4, atol=3.0e-5)

    def test_multistep_decay_value_and_action_gradient_match_torch(self) -> None:
        torch.manual_seed(29)
        torch.cuda.manual_seed_all(29)
        sim = L2FSimulator()
        initial = sim.reset(4, device="cuda", sample_dynamics=False)
        initial.omega[:, 0] = torch.tensor([0.8, 0.7, 0.6, 0.5], device="cuda")
        torch_state = clone_state(initial)
        cuda_state = clone_state(initial)
        actions_torch = (torch.randn(25, 4, 4, device="cuda") * 0.03).requires_grad_()
        actions_cuda = actions_torch.detach().clone().requires_grad_()
        torch_omegas = [torch_state.omega]
        cuda_omegas = [cuda_state.omega]
        for step in range(25):
            torch_state = sim.step(torch_state, actions_torch[step], grad_decay=1.0)
            cuda_state = cuda_step(cuda_state, actions_cuda[step], sim.params, grad_decay=1.0)
            torch_omegas.append(torch_state.omega)
            cuda_omegas.append(cuda_state.omega)
        mask = torch.ones(4, device="cuda", dtype=torch.bool)
        torch_result = multistep_omega_decay_loss(
            torch.stack(torch_omegas), mask, horizons=(5, 10, 25)
        )
        cuda_result = multistep_omega_decay_loss(
            torch.stack(cuda_omegas), mask, horizons=(5, 10, 25)
        )
        torch_result.loss.backward()
        cuda_result.loss.backward()
        torch.testing.assert_close(cuda_result.loss, torch_result.loss, rtol=5.0e-4, atol=5.0e-5)
        torch.testing.assert_close(actions_cuda.grad, actions_torch.grad, rtol=2.0e-3, atol=1.0e-4)


if __name__ == "__main__":
    unittest.main()
