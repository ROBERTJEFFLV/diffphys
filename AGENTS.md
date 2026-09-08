# Repository Guidelines

## Astra and optional Spark workflow

Astra (`gpt-6-astra`) is the parent/orchestrator for this repository and owns task decomposition, architecture-level reasoning, cross-module integration decisions, risk assessment, and final review.

The default workflow is Astra using project skills in the current single-agent thread. Do not spawn or delegate work to subagents unless the user explicitly requests it. A skill invocation does not authorize delegation or launching another Codex session.

The following Spark workflow applies only when the user explicitly requests delegation for the current task. Use Astra as parent plus Spark custom agents loaded from `.codex/agents/spark-*.toml` in a fresh Codex session. Delegate routine code searches, log analysis, focused tests, and bounded local edits to Spark; Astra owns global reasoning, coordination, integration, and final review.

- `spark_explorer`: read-only code-path exploration, symbol tracing, and call-chain mapping.
- `spark_log_analyzer`: log and metric inspection for explicit evidence.
- `spark_test_runner`: focused, task-scoped checks and command execution.
- `spark_implementation_worker`: local bounded edits to assigned files.

Spawn via `agent_type` using these fixed project roles, with `fork_turns = "none"` and no runtime model override. All four roles select `gpt-5.3-codex-spark` with `medium` reasoning in their TOML files. Tasks must be self-contained and include:

- the files assigned for ownership,
- expected evidence to return,
- any constraints or success criteria.

For explicitly requested delegation, start new work in a fresh Codex session so the project agent files are loaded. After configuration changes, or when the active chat cannot select these custom roles, the parent should launch a fresh local `codex exec -m gpt-6-astra` session that delegates the bounded task to the configured Spark role, waits, and returns its result. Use the permissions appropriate to the task. Reuse same-task child threads for follow-up work; do not start another orchestrator for each tiny subtask.

Parallelize independent read tasks. Give concurrent implementation workers disjoint file ownership and serialize overlapping edits. Astra synthesizes the findings and reviews changes before presenting the result.

Verify model routing from runtime records when first establishing a session path or investigating routing failures, never from the child's self-description. If Spark is unavailable or a role cannot be selected, report the limitation instead of silently substituting another model.

These orchestration instructions apply to the parent only. Spark children must not delegate further, spawn other Codex sessions, or run recursive orchestration. A worker that cannot write a protected configuration path should return a concrete patch for the authorized parent to review and apply within its permissions.

The existing rule still applies: run training/validation only when explicitly requested.

## Project Structure & Module Organization

This repository contains a compact PyTorch/CUDA training chain for an L2F-style quadrotor motor policy. Main Python modules live at the repository root:

- `tools/train_response_control.py`: primary response-conditioned task-learning entry point.
- `response_policy.py`: deployable response memory and conditioned motor controller.
- `response_task.py`: teacher-free physical rollouts and task/CVaR objectives.
- `response_training.py` and `response_shooting.py`: resumable task training and joint MS.
- `train.py`: earlier direct-training baseline and CLI argument parser.
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

Tests live under `tests/`. New response-control checks are in `test_response_control.py` and `test_response_training.py`. Keep fixtures small enough to run without a GPU. Run validation and training only when requested; distinguish code correctness, learned performance, and deployment safety. See `docs/response_control_v1.md` for the new protocol. The Q2 migration pipeline is historical and requires `--historical-q2-distillation`.

## Commit & Pull Request Guidelines

This workspace does not include Git history, so no project-specific commit convention can be inferred. Use concise, imperative commit subjects such as `Add smoke config for CPU validation` or `Fix motor loss dtype handling`. Pull requests should describe the behavioral change, list commands run, mention whether CUDA was available, and note any changes to generated artifacts in `runs/` or `checkpoints/`.

## Security & Configuration Tips

Avoid committing large generated checkpoints or experiment logs unless they are required for reproducibility. Keep machine-specific paths out of configs, and prefer checked-in argument files for repeatable runs.

## Skill Routing

Use installed skills automatically when their trigger conditions apply.
Use the minimum set of skills needed for the task.
Project execution boundaries apply even when a skill suggests delegation or expensive verification.

- `systematic-debugging`
  Use for bugs, unexpected behavior, numerical instability, training regressions,
  test failures, NaNs, incorrect gradients, solver failures, or unexplained
  performance changes. Establish the root cause before modifying production code.

- `test-driven-development`
  Use when fixing a confirmed bug or changing observable behavior.
  Prefer the smallest deterministic failing regression test before the implementation.
  Do not treat long training runs as TDD tests.

- `hypothesis-testing`
  Use when correctness must hold across a range of numerical inputs, vehicle
  parameters, states, dtypes, tensor shapes, or physical boundary conditions.
  Prefer ordinary pytest for a small number of fixed examples.

- `optimize-for-gpu`
  Use for CUDA/PyTorch performance, GPU memory, kernel launch, synchronization,
  host-device transfer, rollout throughput, or GPU bottleneck problems.
  Profile before optimization and verify numerical equivalence after optimization.

- `code-review-and-quality`
  Use after a non-trivial implementation is complete.
  Review the actual diff for correctness, readability, architecture, performance,
  and unintended scope expansion. Do not refactor unrelated code.

- `verification-before-completion`
  Use before claiming that a bug is fixed, a change is correct, tests pass,
  or a task is complete. Require fresh evidence appropriate to the claim.

## Skill Order

For debugging or unexpected behavior:

systematic-debugging
→ test-driven-development when a code change is needed
→ hypothesis-testing when correctness spans an input domain
→ implementation
→ code-review-and-quality
→ verification-before-completion

For GPU performance problems:

systematic-debugging when the slowdown is unexplained
→ optimize-for-gpu
→ implementation
→ code-review-and-quality
→ verification-before-completion

Do not invoke skills mechanically when their trigger does not apply.

## Execution Boundaries

Do not spawn or delegate work to subagents unless the user explicitly requests it.

Do not launch full training, long-running evaluation, multi-seed experiments,
or expensive GPU jobs solely for verification unless the user explicitly requests them.
Prefer deterministic unit tests, numerical checks, small probes, smoke tests,
and targeted diagnostics.

Distinguish:

- code correctness,
- numerical/gradient correctness,
- learned performance,
- deployment safety.

Evidence for one does not prove the others.
