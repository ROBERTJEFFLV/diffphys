# Repository Guidelines

## Project Structure & Module Organization

This repository wraps the RAPTOR foundation policy work around the `rl-tools/` submodule-style source tree. Core C++ headers live in `rl-tools/include/rl_tools/`; executable targets and experiments live under `rl-tools/src/`, especially `rl-tools/src/foundation_policy/`. Differential pre-training code is in `rl-tools/src/foundation_policy/diff_pre_training/`, with CUDA pieces in `cuda_main.cu`, `gpu_rollout.cu`, and `gpu_rollout.h`. Tests are under `rl-tools/tests/`. Root-level `reports/`, `logs/`, `checkpoints/`, `verify_rdac_sampler_full/`, and `build_stage96*` directories contain generated artifacts; avoid editing them unless updating a specific experiment result.

## Build, Test, and Development Commands

Initialize dependencies after a fresh clone:

```bash
git submodule update --init rl-tools data
cd rl-tools && git submodule update --init --recursive -- external/highfive external/json external/tensorboard
```

Configure a CPU release build:

```bash
cmake -S rl-tools -B /tmp/raptor_diff_build -DCMAKE_BUILD_TYPE=Release
```

Build the main differential physics targets:

```bash
cmake --build /tmp/raptor_diff_build --target foundation_policy_diff_pre_training foundation_policy_diff_physics_check -j$(nproc)
```

Run the project smoke/sampler validation:

```bash
BUILD_DIR=/tmp/raptor_diff_build JOBS=4 ./verify_rdac_sampler.sh
```

For CUDA work, configure with `-DRL_TOOLS_BACKEND_ENABLE_CUDA=ON` and build `foundation_policy_diff_pre_training_cuda`.

## Coding Style & Naming Conventions

Use C++17. Follow the existing template-heavy RLtools style: namespace aliases such as `namespace rlt = rl_tools;`, `UPPER_CASE` compile-time config fields, and snake_case filenames. Keep C++/CUDA indentation at 4 spaces and prefer explicit types for configuration and tensor specs. Python utility scripts should be small, standard-library first, and use snake_case names. No repository-wide formatter is configured; match nearby formatting before introducing new style.

## Testing Guidelines

GoogleTest-based tests live in `rl-tools/tests/src/` and are enabled with `-DRL_TOOLS_ENABLE_TESTS=ON`; run them from the build directory with `ctest --output-on-failure`. For foundation-policy changes, also run `./verify_rdac_sampler.sh` or a narrowed equivalent that builds both physics and training targets and checks sampler balance, finite outputs, and smoke CSVs. Name new tests after the behavior or component under test, e.g. `test_rl_components_replay_buffer`.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `Add H16 recovery completion audit` and `Align CUDA training with H16 origin recovery`. Keep subjects focused and under about 72 characters. Pull requests should describe the changed training/physics behavior, list exact build or validation commands run, and link or attach relevant report paths, CSVs, or plots when experiment outputs changed. Note CUDA/CPU parity expectations explicitly when touching GPU rollout or checkpoint code.
