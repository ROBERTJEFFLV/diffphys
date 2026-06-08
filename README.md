# L2F-in-seconds

This folder contains a compact CUDA-tensor training chain derived from the current
DiffPhys L2F CUDA work and reshaped to match the `diffphysDrone` recurrent control
flow.

## Layout

- `model.py`: motor policy network:
  `state/error/previous_action -> Linear/MLP encoder -> LeakyReLU -> GRU -> Linear motor head -> 4 motors`.
- `env_l2f.py`: differentiable L2F-style Euler quadrotor simulator. It runs on
  CUDA when tensors are on a CUDA device.
- `env_cuda.py`: compatibility export for the simulator, matching the source
  project's naming style.
- `l2f_cuda_backend.py` and `cuda_ext/`: optional native CUDA Euler transition
  backend with an analytic VJP for the rigid-body step.
- `train.py` and `main_cuda.py`: direct horizon training entry points. The four
  motor outputs are fed into the simulator at every step.
- `configs/`: runnable argument files for smoke and H500 training.
- `reference/current_diffphys_cuda_chain/`: copied current RLtools CUDA L2F
  simulator and differential pre-training source files.
- `reference/diffphysDrone_flow/`: copied files showing the source recurrent
  workflow from `~/diffphysDrone`.

## Architecture

At each simulated step, the trainer builds three policy inputs:

- `state`: position, velocity, rotation matrix, body rates, and motor state.
- `error`: position/velocity/attitude/body-rate error to the origin hover target.
- `previous_action`: previous 4-motor command.

The policy emits four normalized motor commands in `[-1, 1]`. Zero command is
hover thrust; positive and negative values increase or decrease each rotor around
hover.

Training now uses a dense H500 recovery objective rather than a short-horizon
auxiliary loss. Each rollout accumulates robust full-state tracking terms,
CLF-style contraction, outward-velocity recovery, tail stability, command
smoothness, command acceleration, and soft saturation penalties. Backward-only
gradient decay is applied to simulator state and GRU hidden chains; the forward
rollout and loss values remain full-horizon. The CUDA backend mirrors the
reference repository's Euler style with `R_next = R * Exp(dt * skew(omega_next))`
and a custom transition VJP.

## Commands

Run a short validation:

```bash
python3 train.py $(cat configs/smoke.args)
```

Run the H500 CUDA training path:

```bash
python3 train.py $(cat configs/direct_h500.args)
```

Validate the CUDA transition VJP against PyTorch autograd on a CUDA machine with
`nvcc` available:

```bash
python3 tools/validate_cuda_vjp.py
```

Outputs are written under `runs/` and `checkpoints/`.
