# H16 RAPTOR/L2F Recovery-Style Tracking Pipeline

## Scope

This pass refactors the H16 moving-setpoint CUDA path toward a RAPTOR/L2F-style recovery-and-tracking setup while preserving the existing differentiable force/torque rigid-body rollout, CUDA VJP chain, GRU actor, and exactly four normalized motor-command outputs.

H16 is treated as the fixed BPTT window. No H64/H128 horizon curriculum or teacher/student experiment was run from this pipeline.

## Implemented

- Fixed H16 BPTT training/eval path with moving time-indexed references.
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
| noisy/delayed velocity observation validation | PASS | mixed trajectory step-8 observation validation passed |
| fixed H16 smoke training | PASS | finite 20-step run, checkpoint saved, NaN/Inf count 0 |
| RAPTOR/L2F eval metric logging | PASS | fixed/step/circle/figure8/mixed eval CSVs include settling, mean-error, max-error mean/std, RMSE, saturation, and NaN/Inf fields |
| strict tracking performance | FAIL | smoke checkpoint is not a solved controller |
| broad correlated long training | NOT RUN | held until recovery-style interface and metrics are committed |
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
- Long staged H16 recovery training has not been run.
- Broad correlated dynamics training has not been run under the recovery-style task curriculum.
- True long-episode H16-window sampling from multi-second episodes is not implemented yet.
- H64/H128 and RAPTOR teacher/student experiments remain blocked until H16 recovery tracking is stable.

## Artifacts

```text
reports/h16_raptor_l2f_recovery_pipeline/train_fixed_fixed_smoke_20.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_fixed_fixed_smoke_500.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_step_smoke_500.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_circle_smoke_500.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_figure8_smoke_500.csv
reports/h16_raptor_l2f_recovery_pipeline/eval_mixed_smoke_500.csv
```

The smoke checkpoint is intentionally kept as a runtime artifact and should not be committed:

```text
reports/h16_raptor_l2f_recovery_pipeline/checkpoint_fixed_fixed_smoke_20.ckpt
```
