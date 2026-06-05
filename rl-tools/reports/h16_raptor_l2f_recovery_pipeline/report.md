# H16 RAPTOR/L2F Recovery-Style Tracking Pipeline

## Scope

This pass refactors the H16 moving-setpoint CUDA path toward a RAPTOR/L2F-style recovery-and-tracking setup while preserving the existing differentiable force/torque rigid-body rollout, CUDA VJP chain, GRU actor, and exactly four normalized motor-command outputs.

H16 is treated as the fixed BPTT window. No H64/H128 horizon curriculum or teacher/student experiment was run from this pipeline.

## Implemented

- Fixed H16 BPTT training/eval path with moving time-indexed references.
- H16 training windows can now be sampled from a longer reference episode with `--training-episode-steps`; the default is `500` steps, i.e. 5 seconds at 100 Hz.
- Terminal loss defaults to off for moving-setpoint training (`terminal_loss_scale=0`).
- Recovery-style initial state defaults:
  - position axes approximately within `[-0.5, 0.5]` m
  - velocity axes approximately within `[-1, 1]` m/s
  - attitude perturbation up to about 90 degrees
  - angular velocity axes approximately within `[-1, 1]` rad/s
  - near-zero guidance probability default `0.10`
- Per-step horizon-normalized tracking/control losses remain the main training objective.
- Optional temporal gradient decay for propagated state adjoints:
  - `--temporal-gradient-decay-alpha`
  - default `0`
  - tested with `0.5`
- Optional velocity observation perturbations:
  - `--velocity-observation-noise`
  - `--velocity-observation-delay-steps`
- RAPTOR/L2F-style eval metrics:
  - `settling_fraction_position`
  - per-episode mean error mean for position, angle, linear velocity, angular velocity, angular acceleration, action, and action relative
  - per-episode max error mean/std for the same quantities
  - RMSE and strict throughout statistics kept as supplemental diagnostics
  - saturation and NaN/Inf safety diagnostics
- Current-reference-only deployment contract is preserved: external setpoints provide `p_ref(t), v_ref(t)` every control step; the adapter feeds relative position/velocity offsets to the same GRU actor. There is no trajectory lookahead.

The long-episode window sampler is reference-window sampling, not full physical episode backpropagation through 500 steps: each sample chooses a start step inside the 500-step reference episode, fills the `[H+1,B,3]` reference segment from that phase, and samples the recovery initial state around the selected window's first reference state. The differentiable physics BPTT window remains exactly H16.

## Validation Commands

Build:

```bash
cmake --build /tmp/raptor_stage96_cuda_build --target foundation_policy_diff_pre_training_cuda -j2
```

CPU/CUDA moving-reference parity with terminal loss disabled:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 64 \
  --horizon 16 \
  --seed 9401 \
  --trajectory-mode circle \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --terminal-loss-scale 0 \
  --w-linear-acceleration 0.001 \
  --w-angular-acceleration 0.0001
```

CPU/CUDA parity with temporal gradient decay:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 64 \
  --horizon 16 \
  --seed 9402 \
  --trajectory-mode figure8 \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --terminal-loss-scale 0 \
  --temporal-gradient-decay-alpha 0.5 \
  --w-linear-acceleration 0.001 \
  --w-angular-acceleration 0.0001
```

CPU/CUDA parity with 500-step reference episode window sampling:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 64 \
  --horizon 16 \
  --training-episode-steps 500 \
  --seed 9501 \
  --trajectory-mode mixed \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --terminal-loss-scale 0 \
  --temporal-gradient-decay-alpha 0.5 \
  --w-linear-acceleration 0.001 \
  --w-angular-acceleration 0.0001
```

Observation validation with delayed/noisy velocity channels:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-observation \
  --batch-size 64 \
  --horizon 16 \
  --seed 9403 \
  --trajectory-mode mixed \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --validation-step 8 \
  --velocity-observation-noise 0.01 \
  --velocity-observation-delay-steps 2
```

Smoke training:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler \
  --gpu-rollout \
  --steps 20 \
  --horizon 16 \
  --batch-size 2048 \
  --fixed-dynamics \
  --trajectory-mode fixed \
  --terminal-loss-scale 0 \
  --w-linear-acceleration 0.001 \
  --w-angular-acceleration 0.0001 \
  --temporal-gradient-decay-alpha 0.5 \
  --save-optimizer \
  --log-path reports/h16_raptor_l2f_recovery_pipeline/train_fixed_fixed_smoke_20.csv \
  --save-path reports/h16_raptor_l2f_recovery_pipeline/checkpoint_fixed_fixed_smoke_20.ckpt
```

## Pass/Fail Table

| Gate | Result | Evidence |
| --- | --- | --- |
| CUDA build | PASS | target linked successfully |
| actor output dimension | PASS | startup reports `actor_output_dim=4` |
| terminal loss default off | PASS | CLI/default path reports `terminal_loss_scale=0`; smoke CSV terminal ratio is 0 |
| recovery initial-state defaults | PASS | startup reports recovery-scale defaults for position, velocity, attitude, angular velocity, and near-zero guidance |
| moving-reference CPU/CUDA parity | PASS | circle validation passed with terminal loss disabled |
| temporal-gradient-decay CPU/CUDA parity | PASS | figure-eight validation passed with `alpha=0.5` |
| 500-step reference-window CPU/CUDA parity | PASS | mixed trajectory validation passed with `--training-episode-steps 500` and `alpha=0.5` |
| noisy/delayed velocity observation validation | PASS | mixed trajectory step-8 observation validation passed |
| fixed H16 smoke training | PASS | finite 20-step run, checkpoint saved, NaN/Inf count 0 |
| H16 windows sampled from 500-step episode | PASS | training CSV includes `training_episode_steps=500`; smoke window-start means were nonzero (`227.688`, `239.696`, `241.297`) |
| fixed-dynamics task-curriculum smoke | PASS | fixed, step, circle, figure-eight, and mixed 20-step phases all completed finite with per-mode eval CSVs |
| broad correlated mixed compatibility smoke | PASS | 5-step broad+balanced+correlated mixed run completed finite with action saturation 0 |
| longer fixed-dynamics task-curriculum logs | PASS | fixed, step, circle, figure-eight, and mixed 1000-step phases completed finite with 500-step eval CSVs |
| broad correlated staged log | PASS | 200-step broad+balanced+correlated mixed continuation completed finite and produced eval CSV |
| RAPTOR/L2F eval metric logging | PASS | fixed/step/circle/figure8/mixed eval CSVs include settling, mean-error, max-error mean/std, RMSE, saturation, and NaN/Inf fields |
| strict tracking performance | FAIL | smoke checkpoint is not a solved controller |
| broad correlated convergence training | NOT RUN | only short compatibility/staged runs have run |
| H64/H128 or teacher/student experiment | NOT RUN | explicitly blocked until H16 recovery tracking is stable |

## Smoke Training Result

| Metric | Value |
| --- | ---: |
| final loss | 0.0544715 |
| final grad norm | 0.0244202 |
| finite | true |
| terminal loss mean | 0 |
| terminal loss ratio | 0 |
| final action saturation | 0 |

## 500-Step Eval Smoke Summary

The checkpoint below was trained for only 20 fixed-dynamics smoke steps. These results validate the eval path and metrics, not policy quality.

| Trajectory | Settling fraction p | Position mean error mean | Position max error mean | Angle mean error mean | Angular velocity mean error mean | Action saturation | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 0.118302 | 34.2476 | 113.218 | 1.90468 | 231.827 | 0.167803 | 0 |
| step | 0.118302 | 34.2474 | 113.217 | 1.90497 | 231.513 | 0.167795 | 0 |
| circle | 0.118287 | 34.2488 | 113.238 | 1.90501 | 230.789 | 0.167207 | 0 |
| figure8 | 0.118209 | 34.2553 | 113.274 | 1.90609 | 231.833 | 0.167916 | 0 |
| mixed | 0.118263 | 34.2526 | 113.243 | 1.90399 | 231.779 | 0.167635 | 0 |

## Fixed-Dynamics Task-Curriculum Smoke

This smoke uses fixed H16 BPTT, `--training-episode-steps 500`, terminal loss off, recovery initial states, and a chained fixed-dynamics task curriculum. Each phase ran only 20 optimizer steps; it is evidence for the training/eval interface and metrics, not policy quality.

| Phase | Train steps | Window start mean | Eval horizon | Settling fraction p | Position mean error mean | Position max error mean | Action saturation | Invalid/NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 20 | 239.696 | 500 | 0.118209 | 34.2551 | 112.083 | 0.191717 | 0 |
| step | 20 | 239.696 | 500 | 0.122605 | 31.2491 | 103.327 | 0.130136 | 0.0078125 |
| circle | 20 | 239.696 | 500 | 0.122864 | 30.1387 | 98.848 | 0.0980898 | 0 |
| figure8 | 20 | 239.696 | 500 | 0.122918 | 29.1663 | 94.6836 | 0.0841484 | 0 |
| mixed | 20 | 239.696 | 500 | 0.123534 | 28.6787 | 93.6287 | 0.095123 | 0 |

## Broad Correlated Compatibility Smoke

The broad correlated sampler remains compatible with the H16 reference-window path:

| Run | Steps | Batch | Dynamics | Trajectory | Episode steps | Window start mean | Final loss | Final action saturation | NaN/Inf |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| broad correlated mixed | 5 | 512 | broad balanced + correlated size-mass | mixed | 500 | 241.297 | 0.0549195 | 0 | 0 |

## Longer Fixed-Dynamics Task Curriculum

This run uses fixed H16 BPTT, `--training-episode-steps 500`, terminal loss off, recovery initial states, stronger velocity/control penalties, and chained checkpoints across the fixed-dynamics task curriculum. Each phase ran 1000 optimizer steps with batch size 2048. The run remained finite and improved the training loss, but the 500-step long-control eval is still weak.

Training tail:

| Phase | Steps | Window start mean | Final loss | Grad norm | Train saturation | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 1000 | 239.583 | 0.0231278 | 0.0126013 | 0 | 0 |
| step | 1000 | 239.583 | 0.0104752 | 0.00798309 | 0.00567627 | 0 |
| circle | 1000 | 239.583 | 0.00733026 | 0.00281998 | 0.017395 | 0 |
| figure8 | 1000 | 239.583 | 0.00653596 | 0.00113341 | 0.0286331 | 0 |
| mixed | 1000 | 239.583 | 0.00623544 | 0.000504628 | 0.0359497 | 0 |

500-step fixed-dynamics eval:

| Phase | Eval episodes | Settling fraction p | Position mean error mean | Position max error mean | Action saturation | Invalid/NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 512 | 0.138574 | 25.5424 | 77.1851 | 0.0674102 | 0 |
| step | 512 | 0.174545 | 25.7823 | 78.0441 | 0.0931641 | 0 |
| circle | 512 | 0.175867 | 25.5277 | 74.6442 | 0.065749 | 0 |
| figure8 | 512 | 0.176405 | 22.9548 | 64.4341 | 0.0853223 | 0 |
| mixed | 512 | 0.159603 | 22.1705 | 60.8432 | 0.0860166 | 0 |

Broad correlated staged continuation from the mixed checkpoint:

| Run | Steps | Batch | Dynamics | Trajectory | Window start mean | Final loss | Grad norm | Train saturation | Eval settling p | Eval position mean error mean | Eval position max error mean | Eval saturation | Invalid/NaN |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| broad correlated mixed | 200 | 1024 | broad balanced + correlated size-mass | mixed | 243.127 | 0.00666089 | 0.103719 | 0.0419464 | 0.161388 | 22.3834 | 61.6083 | 0.0614893 | 0 |

## Recommended Task Curriculum

Use fixed H16 BPTT throughout:

1. fixed dynamics + fixed target
2. fixed dynamics + step target
3. fixed dynamics + circle target
4. fixed dynamics + figure-eight target
5. fixed dynamics + mixed target
6. small dynamics randomization
7. broad dynamics randomization
8. broad dynamics randomization with correlated size-mass sampling

Run temporal-gradient-decay ablations at `0`, `0.25`, `0.5`, `1`, and `2`. Treat strict t=0 throughout success as a diagnostic only for this recovery-style setup; the primary long-control metrics should be settling fraction, mean/max error statistics, saturation, and NaN/Inf rate.

## Current Blockers

- The current smoke checkpoint is not a usable tracking controller.
- Longer fixed-dynamics staged H16 recovery training has run, but long-control eval remains weak and action saturation is still nontrivial.
- Broad correlated dynamics has only short staged evidence, not convergence training.
- `sampled-dynamics-level small` is still not a separate implemented GPU sampling domain in this path.
- Current implementation samples reference windows from a long episode; it does not yet roll a persistent physical 500-step episode state and then backpropagate H16 slices from that physical trajectory.
- H64/H128 and RAPTOR teacher/student experiments remain blocked until H16 recovery tracking is stable.

## Artifacts

```text
reports/h16_raptor_l2f_recovery_pipeline/train_fixed_fixed_smoke_20.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_fixed_fixed_smoke_500.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_step_smoke_500.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_circle_smoke_500.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_figure8_smoke_500.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_mixed_smoke_500.csv
reports/h16_raptor_l2f_recovery_pipeline/train_window_smoke_3.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/train_fixed_20.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/train_step_20.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/train_circle_20.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/train_figure8_20.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/train_mixed_20.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/eval_fixed_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/eval_step_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/eval_circle_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/eval_figure8_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/eval_mixed_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/train_broad_correlated_mixed_5.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_fixed_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_step_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_circle_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_figure8_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_mixed_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_fixed_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_step_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_circle_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_figure8_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_mixed_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_broad_correlated_mixed_200.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_broad_correlated_mixed_500.csv
```

The smoke checkpoint is intentionally kept as a runtime artifact and should not be committed:

```text
reports/h16_raptor_l2f_recovery_pipeline/checkpoint_fixed_fixed_smoke_20.ckpt
reports/h16_raptor_l2f_recovery_pipeline/checkpoint_window_smoke_3.ckpt
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_smoke/*.ckpt
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/*.ckpt
```
