# Repository Guidelines

## Project Structure & Module Organization

This repository contains a compact PyTorch/CUDA training chain for an L2F-style quadrotor motor policy. Main Python modules live at the repository root:

- `train.py`: primary training entry point and CLI argument parser.
- `main_cuda.py`: compatibility wrapper that calls `train.main()`.
- `model.py`: `MotorGRUPolicy`, the state/error/previous-action GRU controller.
- `env_l2f.py`: differentiable Euler quadrotor simulator and loss terms.
- `env_cuda.py`: simulator compatibility export.

Use `configs/*.args` for reusable training configurations. Training logs go to `runs/`, and model checkpoints go to `checkpoints/`. Reference implementations are under `reference/`; treat them as source context unless a task explicitly asks to modify them.

## Build, Test, and Development Commands

Run a short validation pass:

```bash
python3 train.py $(cat configs/smoke.args)
```

Run the H500 CUDA configuration:

```bash
python3 train.py $(cat configs/direct_h500.args)
```

Use CLI overrides for local experiments:

```bash
python3 train.py --device cpu --steps 2 --horizon 8 --batch-size 4
```

There is no build step; the project runs directly with Python and PyTorch. CUDA is used with `--device cuda` or when `--device auto` detects a GPU.

## Coding Style & Naming Conventions

Use Python 3 type hints, `from __future__ import annotations`, and four-space indentation. Keep module constants in `UPPER_SNAKE_CASE`, classes in `PascalCase`, and functions, variables, and CLI flags in `snake_case` or kebab-case for command-line options. Prefer small, explicit functions and dataclasses for simulator state and parameters. Tensors should preserve caller device and dtype unless conversion is intentional.

## Testing Guidelines

No pytest suite is checked in. Validate changes with the smoke command before handoff. For simulator or model changes, also run a tiny CPU invocation to catch device-agnostic issues. If adding tests, place them under `tests/`, name files `test_*.py`, and keep fixtures small enough to run without a GPU.

## Commit & Pull Request Guidelines

This workspace does not include Git history, so no project-specific commit convention can be inferred. Use concise, imperative commit subjects such as `Add smoke config for CPU validation` or `Fix motor loss dtype handling`. Pull requests should describe the behavioral change, list commands run, mention whether CUDA was available, and note any changes to generated artifacts in `runs/` or `checkpoints/`.

## Security & Configuration Tips

Avoid committing large generated checkpoints or experiment logs unless they are required for reproducibility. Keep machine-specific paths out of configs, and prefer checked-in argument files for repeatable runs.
