# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actionclip10_s23679`

This audit only evaluates fixed-dynamics repeatability of SO(3) differentiable pretraining. It does not prove teacher-cost reduction, replacement of RAPTOR teachers, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Completeness

| item | value |
| --- | ---: |
| expected train seeds | 5 |
| completed train seeds | 5 |
| expected eval files | 30 |
| completed eval files | 30 |

Missing artifacts:

None.

Diagnostic fields unavailable without broader model/optimizer instrumentation: actor_grad_max_abs, actor_param_max_abs, actor_param_norm, actor_parameter_norm, actor_update_norm, actor_update_to_param_norm_ratio, adam_m_norm, adam_v_norm, gru_hidden_abs_max, gru_hidden_abs_mean, gru_hidden_norm_max, gru_hidden_norm_mean

## Training Seed Table

| seed | first loss | final loss | loss reduction | final p/v/w | applied/skipped steps | invalid rollouts | NaN/Inf | checkpoint |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 2 | 1614.79 | 114.437 | 1500.35 | 0.873717 / 0.821642 / 0.967152 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actionclip10_s23679\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 10908.5 | 815.946 | 10092.6 | 1.26936 / 2.25908 / 3.40876 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actionclip10_s23679\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 6 | 5999.68 | 911.557 | 5088.12 | 2.23204 / 2.1449 / 2.41173 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actionclip10_s23679\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3030.59 | 852.964 | 2177.63 | 0.665394 / 1.19435 / 5.42054 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actionclip10_s23679\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 9 | 5486.16 | 274.518 | 5211.64 | 1.5129 / 0.982139 / 1.30563 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actionclip10_s23679\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 2 | 0.493333 / 0.0169967 | 0.466667 / 0.020548 | 1.11862 / 0.871011 / 1.28149 | 1.22492 / 1.02784 / 1.52343 | true | PASS_STRONG |
| 3 | 0.306667 / 0.054365 | 0.196667 / 0.0329983 | 1.57443 / 1.77646 / 3.7243 | 1.8142 / 2.35267 / 4.89147 | true | PASS_WEAK |
| 6 | 0.0633333 / 0.0169967 | 0.0366667 / 0.020548 | 2.08355 / 2.65468 / 2.9818 | 2.20824 / 2.75963 / 3.00158 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 7 | 0.293333 / 0.0249444 | 0.116667 / 0.0124722 | 1.5447 / 2.21477 / 4.50599 | 2.15421 / 3.91962 / 5.85171 | true | PASS_WEAK |
| 9 | 0.276667 / 0.0339935 | 0.226667 / 0.0235702 | 1.57769 / 1.08017 / 1.29899 | 1.70137 / 1.26968 / 1.49115 | true | PASS_WEAK |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 1 |
| PASS_WEAK | 3 |
| TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE | 1 |

Failed seeds: 6.

Most likely failure explanation: TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.286667 / 0.13648 |
| L2F fixed mean/std success across train seeds | 0.208667 / 0.144969 |
| seeds beating Euler baseline > 0.11 | 4 / 5 |
| seeds beating L2F baseline > 0.09 | 4 / 5 |
| seeds beating both baselines | 4 / 5 |
| catastrophic zero-success seeds | 0 |
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
