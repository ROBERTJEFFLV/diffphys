# Teacher-Cost Pilot Bridge Report

Date: 2026-06-02

## Scope

The current priority is the controlled teacher-cost pilot, not further broad sampled-dynamics expansion. The bridge is intended to compare scratch post-training against diff-pretraining initialization under identical teacher counts, teacher IDs, seeds, and update budgets.

## Implemented

- Added `foundation_policy_post_training --init-actor-path PATH` runtime option.
- The option loads a diff-pretraining actor-only text checkpoint into the post-training student actor.
- Only actor weights are initialized from the text checkpoint; the optimizer is freshly initialized.
- Existing post-training HDF5/code checkpoint output remains unchanged.
- Added runtime options for teacher count, epochs, teacher-forcing epochs, run path, teacher index, dynamics parameter path, and teacher search root.
- Added `foundation_policy_actor_equivalence_check` to compare diff-pretraining actor output with post-training student output on the same observation batch.
- Added `post_training/scripts/run_teacher_cost_pilot.sh` for the minimal `teachers=0,8,32`, seeds `0,1,2`, scratch vs diff-init runner.

## Verification

Builds performed on the remote host:

```bash
cmake -S rl-tools -B /tmp/diffphys_teacher_cost_build \
  -DRL_TOOLS_ENABLE_TARGETS=ON \
  -DRL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS=ON
cmake --build /tmp/diffphys_teacher_cost_build --target foundation_policy_diff_pre_training -j2
cmake --build /tmp/diffphys_teacher_cost_build --target foundation_policy_actor_equivalence_check -j2

cmake -S rl-tools -B /tmp/diffphys_teacher_cost_build_hdf5 \
  -DRL_TOOLS_ENABLE_TARGETS=ON \
  -DRL_TOOLS_ENABLE_HDF5=ON \
  -DRL_TOOLS_ENABLE_JSON=ON \
  -DRL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS=ON
cmake --build /tmp/diffphys_teacher_cost_build_hdf5 --target foundation_policy_post_training -j2
cmake --build /tmp/diffphys_teacher_cost_build_hdf5 --target foundation_policy_actor_equivalence_check -j2
```

Actor equivalence smoke test using a generated compatible actor text checkpoint:

```text
actor_equivalence max_abs_error=0 mean_abs_error=0 count=612
PASS
```

Post-training init-only smoke test:

```bash
/tmp/diffphys_teacher_cost_build_hdf5/src/foundation_policy/foundation_policy_post_training \
  --init-actor-path /tmp/diffphys_actor_smoke.txt \
  --num-teachers 0 \
  --epochs 0 \
  --run-path /tmp/diffphys_post_init_smoke
```

Result:

```text
Loaded student actor initialization from /tmp/diffphys_actor_smoke.txt
No teachers requested; skipping teacher checkpoint loading. Use --epochs 0 for init-only export.
Checkpointing to: "/tmp/diffphys_post_init_smoke/checkpoint.h5"
Checkpointing to: "/tmp/diffphys_post_init_smoke/checkpoint.h"
```

## Current Blocker For P2

The source bridge is ready, but the real pilot cannot be run from the checked-out artifacts yet:

- No sampled-medium diff actor text checkpoint was found under `runs/diff_pre_training/sampled_medium_finetune_from_fixed_12000/checkpoints/`.
- No repository-local teacher HDF5 checkpoints were found via `find . -maxdepth 5 -type f -name checkpoint.h5`.
- Full submodule initialization is blocked by private GitHub credential prompts for data/media submodules, so teacher artifacts must be supplied separately or the correct artifact root must be mounted.

Only smoke checkpoints under `/tmp` were generated for bridge validation; they are not valid experimental initializations.

## Pilot Command Once Artifacts Exist

Provide the real sampled-medium actor text checkpoint and teacher artifact paths, then run:

```bash
DIFF_INIT_ACTOR_PATH=/path/to/sampled_medium_actor_checkpoint.txt \
TEACHER_SEARCH_ROOT=/path/to/1k-experiments \
TEACHER_INDEX_PATH=/path/to/checkpoints_2025-04-16_20-10-58.txt \
DYNAMICS_PARAMETERS_PATH=/path/to/dynamics_parameters_2025-04-16_20-10-58 \
rl-tools/src/foundation_policy/post_training/scripts/run_teacher_cost_pilot.sh
```

Defaults in the script:

- teacher counts: `0 8 32`
- train seeds: `0 1 2`
- zero-teacher epochs: `0`
- post-training epochs for nonzero teacher counts: `1000`
- equivalence check: enabled before training

## Success Criteria

Minimum signal:

- diff-init is clearly above scratch at 8 or 32 teachers.
- medium L2F success improves by at least 0.10.
- no NaN, invalid, skipped-update, or checkpoint-load instability.

Strong signal:

- diff-init plus 8 teachers approaches or exceeds scratch plus 32 teachers, or
- diff-init plus 16 teachers approaches or exceeds scratch plus 64 teachers in the expanded sweep.

Allowed claim after strong evidence only:

`Differentiable pretraining reduces teacher cost in a controlled few-teacher pilot.`
