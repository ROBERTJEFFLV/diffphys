# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_terminal_ramp_focus`

This audit only evaluates fixed-dynamics repeatability of SO(3) differentiable pretraining. It does not prove teacher-cost reduction, replacement of RAPTOR teachers, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Completeness

| item | value |
| --- | ---: |
| expected train seeds | 7 |
| completed train seeds | 7 |
| expected eval files | 42 |
| completed eval files | 42 |

Missing artifacts:

None.

Diagnostic fields unavailable without broader model/optimizer instrumentation: actor_grad_max_abs, actor_parameter_norm, actor_update_norm, adam_m_norm, adam_v_norm, gru_hidden_norm_max, gru_hidden_norm_mean

## Training Seed Table

| seed | first loss | final loss | loss reduction | final p/v/w | applied/skipped steps | invalid rollouts | NaN/Inf | checkpoint |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 1 | 1513.71 | 1745.09 | -231.38 | 0.933049 / 1.36997 / 7.20735 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_terminal_ramp_focus\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 2 | 1614.79 | 791.665 | 823.125 | 2.19969 / 3.03274 / 1.89154 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_terminal_ramp_focus\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 10908.5 | 233.265 | 10675.2 | 0.838276 / 0.411153 / 2.45978 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_terminal_ramp_focus\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 4 | 2817.89 | 3524.43 | -706.54 | 3.22072 / 5.89307 / 5.97259 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_terminal_ramp_focus\checkpoints\fixed_vstrong_seed4_policy.bin` |
| 6 | 5999.68 | 3851.92 | 2147.76 | 3.66473 / 5.89749 / 5.32183 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_terminal_ramp_focus\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3030.59 | 795.622 | 2234.97 | 2.46028 / 2.51373 / 2.02462 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_terminal_ramp_focus\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 9 | 5486.16 | 567.795 | 4918.36 | 2.19561 / 1.80668 / 0.793538 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_terminal_ramp_focus\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 0.356667 / 0.0169967 | 0.206667 / 0.0188562 | 1.18599 / 1.06407 / 3.91458 | 1.45362 / 1.73867 / 5.57136 | true | PASS_WEAK |
| 2 | 0 / 0 | 0 / 0 | 2.73329 / 3.33246 / 1.72055 | 2.84784 / 3.56264 / 2.24441 | false | TYPE_G_OBJECTIVE_MISMATCH |
| 3 | 0.56 / 0.0326599 | 0.36 / 0.0282843 | 0.989727 / 0.758577 / 2.89983 | 1.15659 / 1.07045 / 4.15104 | true | PASS_STRONG |
| 4 | 0.1 / 0.00816497 | 0.05 / 0.0141421 | 2.84951 / 5.12723 / 6.70942 | 3.21771 / 5.84834 / 7.26438 | false | TYPE_D_ANGULAR_INSTABILITY |
| 6 | 0.06 / 0.00816497 | 0.0366667 / 0.00471405 | 2.0807 / 3.41477 / 7.95113 | 2.58622 / 4.72112 / 8.63636 | false | TYPE_D_ANGULAR_INSTABILITY |
| 7 | 0.116667 / 0.00942809 | 0.0733333 / 0.0124722 | 2.14015 / 2.48792 / 2.07074 | 2.40944 / 2.8782 / 2.23736 | false | TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH |
| 9 | 0.3 / 0.0216025 | 0.226667 / 0.0169967 | 1.57832 / 1.28623 / 1.08742 | 1.74587 / 1.58068 / 1.58328 | true | PASS_WEAK |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 1 |
| PASS_WEAK | 2 |
| TYPE_D_ANGULAR_INSTABILITY | 2 |
| TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH | 1 |
| TYPE_G_OBJECTIVE_MISMATCH | 1 |

Failed seeds: 2, 4, 6, 7.

Most likely failure explanation: TYPE_D_ANGULAR_INSTABILITY is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.213333 / 0.18495 |
| L2F fixed mean/std success across train seeds | 0.13619 / 0.121347 |
| seeds beating Euler baseline > 0.11 | 4 / 7 |
| seeds beating L2F baseline > 0.09 | 3 / 7 |
| seeds beating both baselines | 3 / 7 |
| catastrophic zero-success seeds | 1 |
| total skipped actor steps | 0 |
| any NaN/Inf | false |
| mean invalid_or_nan_rate | 0 |

Old first-order fixed v-strong baselines:

| Eval model | success | mean final p/v/w |
| --- | ---: | --- |
| Euler fixed | 0.11 | 2.22573 / 2.26618 / 0.939211 |
| L2F fixed | 0.09 | 2.44163 / 2.65331 / 1.02248 |

## Decision

UNSTABLE_NEEDS_FIX

## Recommended Next Action

Do not automatically modify training from this audit alone; run a targeted ablation next: reduce w-terminal-w and possibly w-w.
