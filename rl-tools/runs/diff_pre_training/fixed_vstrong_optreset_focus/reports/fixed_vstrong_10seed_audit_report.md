# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_optreset_focus`

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
| 1 | 1513.71 | 5519.54 | -4005.83 | 4.17008 / 9.77583 / 4.59084 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_optreset_focus\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 2 | 1614.79 | 1886.09 | -271.3 | 2.00983 / 4.71609 / 4.57102 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_optreset_focus\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 10908.5 | 202.857 | 10705.6 | 0.903873 / 0.401882 / 2.20236 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_optreset_focus\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 4 | 2817.89 | 3433.63 | -615.74 | 3.05177 / 5.15676 / 5.26149 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_optreset_focus\checkpoints\fixed_vstrong_seed4_policy.bin` |
| 6 | 5999.68 | 913.325 | 5086.36 | 1.01011 / 0.682495 / 5.00371 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_optreset_focus\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3030.59 | 965.765 | 2064.83 | 2.22841 / 2.64039 / 3.20395 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_optreset_focus\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 9 | 5486.16 | 865.187 | 4620.97 | 2.81465 / 2.57226 / 1.80676 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_optreset_focus\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 0 / 0 | 0 / 0 | 3.98445 / 7.7921 / 6.14249 | 4.49106 / 8.74426 / 5.86435 | false | TYPE_A_EARLY_DIVERGENCE |
| 2 | 0.01 / 0.00816497 | 0.00333333 / 0.00471405 | 3.06781 / 5.02452 / 4.87712 | 3.39144 / 5.66534 / 5.28963 | false | TYPE_A_EARLY_DIVERGENCE |
| 3 | 0.53 / 0.0374166 | 0.41 / 0.0216025 | 1.05526 / 0.669656 / 2.36769 | 1.18227 / 0.989283 / 3.7792 | true | PASS_STRONG |
| 4 | 0.146667 / 0.0124722 | 0.0933333 / 0.00471405 | 2.45872 / 3.9803 / 5.8149 | 2.82707 / 4.8254 / 6.56244 | true | PASS_WEAK |
| 6 | 0.43 / 0.0374166 | 0.21 / 0.00816497 | 0.89387 / 0.8103 / 4.24298 | 0.979405 / 1.02906 / 6.07342 | true | PASS_WEAK |
| 7 | 0.1 / 0 | 0.0633333 / 0.00942809 | 2.68685 / 3.98278 / 3.54655 | 3.14371 / 4.89305 / 4.04929 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 9 | 0.06 / 0.0216025 | 0.0633333 / 0.020548 | 2.21876 / 2.47387 / 1.56708 | 2.40666 / 2.78487 / 1.91785 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 1 |
| PASS_WEAK | 2 |
| TYPE_A_EARLY_DIVERGENCE | 2 |
| TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE | 2 |

Failed seeds: 1, 2, 7, 9.

Most likely failure explanation: TYPE_A_EARLY_DIVERGENCE is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.182381 / 0.19573 |
| L2F fixed mean/std success across train seeds | 0.120476 / 0.134877 |
| seeds beating Euler baseline > 0.11 | 3 / 7 |
| seeds beating L2F baseline > 0.09 | 3 / 7 |
| seeds beating both baselines | 3 / 7 |
| catastrophic zero-success seeds | 2 |
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

Do not automatically modify training from this audit alone; run a targeted ablation next: try lower actor learning rate and lower actor-grad-clip.
