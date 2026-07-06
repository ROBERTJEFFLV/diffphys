from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FParams, L2FSimulator, L2FState
from l2f_cuda_backend import cuda_step, load_extension


def clone_state_for_grad(state: L2FState) -> L2FState:
    return L2FState(
        position=state.position.detach().clone().requires_grad_(True),
        velocity=state.velocity.detach().clone().requires_grad_(True),
        rotation=state.rotation.detach().clone().requires_grad_(True),
        omega=state.omega.detach().clone().requires_grad_(True),
        motor=state.motor.detach().clone().requires_grad_(True),
        previous_action=state.previous_action.detach().clone().requires_grad_(True),
        external_force=state.external_force.detach().clone(),
        mass=state.mass.detach().clone(),
        thrust_coeff_c0=state.thrust_coeff_c0.detach().clone(),
        thrust_coeff_c1=state.thrust_coeff_c1.detach().clone(),
        thrust_coeff_c2=state.thrust_coeff_c2.detach().clone(),
        thrust_to_weight=state.thrust_to_weight.detach().clone(),
        torque_to_inertia=state.torque_to_inertia.detach().clone(),
        rotor_distance_factor=state.rotor_distance_factor.detach().clone(),
        inertia_factor=state.inertia_factor.detach().clone(),
        motor_time_rising=state.motor_time_rising.detach().clone(),
        motor_time_falling=state.motor_time_falling.detach().clone(),
        rotor_torque_constant=state.rotor_torque_constant.detach().clone(),
        cbrt_mass=state.cbrt_mass.detach().clone(),
        force_std=state.force_std.detach().clone(),
        arm_length=state.arm_length.detach().clone(),
        inertia_x=state.inertia_x.detach().clone(),
        inertia_y=state.inertia_y.detach().clone(),
        inertia_z=state.inertia_z.detach().clone(),
    )


def state_tensors(state: L2FState) -> tuple[torch.Tensor, ...]:
    return (
        state.position,
        state.velocity,
        state.rotation,
        state.omega,
        state.motor,
        state.previous_action,
    )


def max_abs(values: list[torch.Tensor]) -> float:
    return max(float(value.detach().abs().max().item()) for value in values)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping CUDA VJP validation.")
        return
    load_extension()

    torch.manual_seed(11)
    params = L2FParams()
    sim = L2FSimulator(params)
    batch_size = 8
    base_state = sim.reset(batch_size, device="cuda")
    base_action = torch.empty(batch_size, 4, device="cuda").uniform_(-0.5, 0.5)
    grad_decay = 0.73

    torch_state = clone_state_for_grad(base_state)
    cuda_state = clone_state_for_grad(base_state)
    torch_action = base_action.detach().clone().requires_grad_(True)
    cuda_action = base_action.detach().clone().requires_grad_(True)

    torch_next = sim.step(torch_state, torch_action, grad_decay=grad_decay)
    cuda_next = cuda_step(cuda_state, cuda_action, params, grad_decay=grad_decay)
    forward_errors = [
        (a - b).abs().max()
        for a, b in zip(state_tensors(torch_next), state_tensors(cuda_next))
    ]
    print(f"max forward abs error: {max_abs(forward_errors):.6e}")

    output_grads = tuple(torch.randn_like(value) for value in state_tensors(torch_next))
    torch.autograd.backward(state_tensors(torch_next), output_grads)
    torch.autograd.backward(state_tensors(cuda_next), output_grads)

    torch_grads = [value.grad for value in (*state_tensors(torch_state)[:5], torch_action)]
    cuda_grads = [value.grad for value in (*state_tensors(cuda_state)[:5], cuda_action)]
    backward_errors = [
        (a - b).abs().max()
        for a, b in zip(torch_grads, cuda_grads)
    ]
    print(f"max backward abs error: {max_abs(backward_errors):.6e}")

    if max_abs(forward_errors) > 2.0e-5 or max_abs(backward_errors) > 2.0e-4:
        raise SystemExit("CUDA VJP validation failed")
    print("CUDA VJP validation passed.")


if __name__ == "__main__":
    main()
