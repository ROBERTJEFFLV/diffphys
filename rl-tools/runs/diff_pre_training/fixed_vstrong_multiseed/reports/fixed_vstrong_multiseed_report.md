# Fixed SO(3) Multi-Seed Repeatability

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_multiseed`

## Verdict

PASS: fixed SO(3) differentiable pretraining appears repeatable.

This experiment only tests fixed-dynamics repeatability. It does not prove teacher-cost reduction, RAPTOR replacement, sampled-dynamics success, Sim2Real transfer, or full L2F dynamics equivalence.

## Required Artifact Completeness

Missing artifacts:

None.

## Training Seed Table

| train seed | first loss | final loss | loss reduction | final mean p/v/w | applied steps | skipped steps | invalid rollout count | NaN/Inf flag | checkpoint |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | 1056.45 | 132.695 | 923.755 | 1.19133 / 0.771525 / 0.587346 | 6000 | 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_multiseed\checkpoints\fixed_vstrong_seed0_policy.bin` |
| 1 | 1513.71 | 6296.54 | -4782.83 | 3.5796 / 5.95773 / 9.69578 | 6000 | 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_multiseed\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 2 | 1614.79 | 204.475 | 1410.32 | 1.16132 / 1.05014 / 1.42149 | 6000 | 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_multiseed\checkpoints\fixed_vstrong_seed2_policy.bin` |

## Evaluation By Training Seed

| train seed | eval model | mean success | std success | min/max success | mean final p/v/w | mean invalid_or_nan_rate |
| --- | --- | ---: | ---: | --- | --- | ---: |
| 0 | euler | 0.433333 | 0.00942809 | 0.42 / 0.44 | 1.23295 / 0.727639 / 0.994368 | 0 |
| 0 | l2f | 0.403333 | 0.0169967 | 0.38 / 0.42 | 1.31597 / 0.859992 / 1.23737 | 0 |
| 1 | euler | 0 | 0 | 0 / 0 | 5.58572 / 8.83271 / 5.77791 | 0 |
| 1 | l2f | 0 | 0 | 0 / 0 | 5.68248 / 9.18101 / 5.95081 | 0 |
| 2 | euler | 0.17 | 0.0216025 | 0.15 / 0.2 | 1.677 / 1.40924 / 1.49802 | 0 |
| 2 | l2f | 0.14 | 0.0408248 | 0.09 / 0.19 | 1.78164 / 1.65758 / 1.94389 | 0 |

## Overall Aggregate

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across training seeds | 0.201111 / 0.17827 |
| L2F fixed mean/std success across training seeds | 0.181111 / 0.167207 |
| Train seeds beating old Euler success baseline > 0.11 | 2 / 3 |
| Train seeds beating old L2F success baseline > 0.09 | 2 / 3 |
| Train seeds beating both success baselines | 2 / 3 |
| Total skipped actor steps | 0 |
| Any training NaN/Inf | false |

Old first-order fixed v-strong baselines:

| Eval model | success | mean final p/v/w |
| --- | ---: | --- |
| Euler fixed | 0.11 | 2.22573 / 2.26618 / 0.939211 |
| L2F fixed | 0.09 | 2.44163 / 2.65331 / 1.02248 |
