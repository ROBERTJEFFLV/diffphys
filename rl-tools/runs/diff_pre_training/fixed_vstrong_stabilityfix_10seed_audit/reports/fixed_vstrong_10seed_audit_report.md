# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit`

This audit only evaluates fixed-dynamics repeatability of SO(3) differentiable pretraining. It does not prove teacher-cost reduction, replacement of RAPTOR teachers, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Completeness

| item | value |
| --- | ---: |
| expected train seeds | 10 |
| completed train seeds | 10 |
| expected eval files | 60 |
| completed eval files | 60 |

Missing artifacts:

None.

Diagnostic fields unavailable without broader model/optimizer instrumentation: actor_grad_max_abs, actor_parameter_norm, actor_update_norm, adam_m_norm, adam_v_norm, gru_hidden_norm_max, gru_hidden_norm_mean

## Training Seed Table

| seed | first loss | final loss | loss reduction | final p/v/w | applied/skipped steps | invalid rollouts | NaN/Inf | checkpoint |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 0 | 1056.45 | 402.186 | 654.264 | 1.59091 / 1.33648 / 1.6026 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed0_policy.bin` |
| 1 | 1513.71 | 119.71 | 1394 | 0.832233 / 0.355081 / 1.4287 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 2 | 1614.79 | 1154.56 | 460.23 | 2.80822 / 3.90686 / 1.2779 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 10908.5 | 196.954 | 10711.5 | 0.888229 / 0.48937 / 1.83096 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 4 | 2817.89 | 2772.93 | 44.96 | 2.86551 / 5.2877 / 5.0576 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed4_policy.bin` |
| 5 | 2785.71 | 1245.16 | 1540.55 | 1.59089 / 1.60751 / 4.1923 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed5_policy.bin` |
| 6 | 5999.68 | 958.354 | 5041.33 | 1.34655 / 1.6855 / 4.39345 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3030.59 | 889.849 | 2140.74 | 2.60556 / 2.73126 / 2.30518 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 8 | 14880.7 | 1245.01 | 13635.7 | 1.36344 / 2.60289 / 4.36517 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed8_policy.bin` |
| 9 | 5486.16 | 1203.13 | 4283.03 | 2.96404 / 3.16402 / 1.38819 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 0.283333 / 0.0339935 | 0.233333 / 0.020548 | 1.71549 / 1.58377 / 2.03633 | 1.9198 / 1.90261 / 2.19517 | true | PASS_WEAK |
| 1 | 0.523333 / 0.0235702 | 0.35 / 0.0141421 | 1.00804 / 0.633199 / 2.36491 | 1.12115 / 1.03343 / 4.63023 | true | PASS_STRONG |
| 2 | 0 / 0 | 0 / 0 | 3.07659 / 3.93588 / 1.47965 | 3.22519 / 4.20695 / 1.76942 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 3 | 0.52 / 0.0374166 | 0.363333 / 0.0309121 | 1.0951 / 0.821564 / 2.59627 | 1.29177 / 1.18877 / 3.83859 | true | PASS_STRONG |
| 4 | 0.09 / 0.00816497 | 0.0566667 / 0.0124722 | 2.756 / 4.8223 / 5.73305 | 3.07651 / 5.38074 / 6.37715 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 5 | 0.233333 / 0.0169967 | 0.12 / 0.0374166 | 1.32489 / 1.89772 / 6.13926 | 1.66267 / 2.94151 / 7.06945 | true | PASS_WEAK |
| 6 | 0.216667 / 0.020548 | 0.0966667 / 0.020548 | 1.0247 / 1.32949 / 6.13414 | 1.20313 / 1.796 / 7.51311 | true | PASS_WEAK |
| 7 | 0.11 / 0.0141421 | 0.08 / 0.0163299 | 2.23168 / 2.66252 / 2.75256 | 2.50405 / 3.09924 / 3.0072 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 8 | 0.203333 / 0.0169967 | 0.103333 / 0.0385861 | 1.85683 / 3.24811 / 5.48863 | 2.26605 / 4.37666 / 6.13669 | true | PASS_WEAK |
| 9 | 0.156667 / 0.020548 | 0.1 / 0.0163299 | 1.97734 / 2.04784 / 1.41196 | 2.20466 / 2.44896 / 1.80198 | true | PASS_WEAK |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 2 |
| PASS_WEAK | 5 |
| TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE | 3 |

Failed seeds: 2, 4, 7.

Most likely failure explanation: TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.233667 / 0.163105 |
| L2F fixed mean/std success across train seeds | 0.150333 / 0.117042 |
| seeds beating Euler baseline > 0.11 | 7 / 10 |
| seeds beating L2F baseline > 0.09 | 7 / 10 |
| seeds beating both baselines | 7 / 10 |
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

ACCEPTABLE_BUT_UNSTABLE

## Recommended Next Action

Do not automatically modify training from this audit alone; run a targeted ablation next: slow down or freeze horizon/state curriculum.
