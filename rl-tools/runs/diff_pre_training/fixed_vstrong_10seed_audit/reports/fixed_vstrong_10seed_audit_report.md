# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit`

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

Diagnostic fields unavailable without broader model/optimizer instrumentation: actor_grad_max_abs, actor_update_norm, gru_hidden_norm_max, gru_hidden_norm_mean

## Training Seed Table

| seed | first loss | final loss | loss reduction | final p/v/w | applied/skipped steps | invalid rollouts | NaN/Inf | checkpoint |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 0 | 1056.45 | 132.695 | 923.755 | 1.19133 / 0.771525 / 0.587346 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed0_policy.bin` |
| 1 | 1513.71 | 6296.54 | -4782.83 | 3.5796 / 5.95773 / 9.69578 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 2 | 1614.79 | 204.475 | 1410.32 | 1.16132 / 1.05014 / 1.42149 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 10908.5 | 401.023 | 10507.5 | 1.57278 / 1.81206 / 1.59987 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 4 | 2817.89 | 634.162 | 2183.73 | 1.92424 / 1.86049 / 2.04233 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed4_policy.bin` |
| 5 | 2785.71 | 3907.89 | -1122.18 | 3.10179 / 5.58385 / 2.72591 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed5_policy.bin` |
| 6 | 5999.68 | 1903.29 | 4096.39 | 2.30108 / 3.43933 / 4.43495 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3030.59 | 785.489 | 2245.1 | 2.65674 / 2.18657 / 1.81596 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 8 | 14880.7 | 388.061 | 14492.6 | 1.5723 / 1.80625 / 1.32124 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed8_policy.bin` |
| 9 | 5486.16 | 2491.39 | 2994.77 | 3.13953 / 5.34741 / 3.8269 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_10seed_audit\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 0.433333 / 0.00942809 | 0.403333 / 0.0169967 | 1.23295 / 0.727639 / 0.994368 | 1.31597 / 0.859992 / 1.23737 | true | PASS_STRONG |
| 1 | 0 / 0 | 0 / 0 | 5.58572 / 8.83271 / 5.77791 | 5.68248 / 9.18101 / 5.95081 | false | TYPE_A_EARLY_DIVERGENCE |
| 2 | 0.17 / 0.0216025 | 0.14 / 0.0408248 | 1.677 / 1.40924 / 1.49802 | 1.78164 / 1.65758 / 1.94389 | true | PASS_WEAK |
| 3 | 0.05 / 0.0141421 | 0.04 / 0.00816497 | 2.33442 / 2.63474 / 1.76997 | 2.5191 / 2.9927 / 2.142 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 4 | 0.213333 / 0.0402768 | 0.18 / 0.0374166 | 1.91283 / 1.76157 / 1.93578 | 2.09375 / 2.10734 / 2.27727 | true | PASS_WEAK |
| 5 | 0.163333 / 0.00942809 | 0.116667 / 0.0124722 | 3.22866 / 4.60059 / 3.75345 | 3.54785 / 5.25322 / 4.17711 | true | PASS_WEAK |
| 6 | 0.0366667 / 0.0169967 | 0.0166667 / 0.00471405 | 2.25449 / 3.80538 / 7.51984 | 2.72825 / 4.93721 / 7.36192 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 7 | 0.146667 / 0.0309121 | 0.136667 / 0.0169967 | 2.21087 / 2.06726 / 1.96843 | 2.39739 / 2.38555 / 2.17085 | true | PASS_WEAK |
| 8 | 0.243333 / 0.020548 | 0.226667 / 0.0169967 | 2.04663 / 1.96455 / 1.57759 | 2.21901 / 2.188 / 1.71467 | true | PASS_WEAK |
| 9 | 0.0233333 / 0.0124722 | 0.0166667 / 0.00471405 | 3.52501 / 6.14409 / 3.80093 | 3.81854 / 6.6374 / 3.79476 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 1 |
| PASS_WEAK | 5 |
| TYPE_A_EARLY_DIVERGENCE | 1 |
| TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE | 3 |

Failed seeds: 1, 3, 6, 9.

Most likely failure explanation: TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.148 / 0.124474 |
| L2F fixed mean/std success across train seeds | 0.127667 / 0.117266 |
| seeds beating Euler baseline > 0.11 | 6 / 10 |
| seeds beating L2F baseline > 0.09 | 6 / 10 |
| seeds beating both baselines | 6 / 10 |
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

Do not automatically modify training from this audit alone; run a targeted ablation next: slow down or freeze horizon/state curriculum.
