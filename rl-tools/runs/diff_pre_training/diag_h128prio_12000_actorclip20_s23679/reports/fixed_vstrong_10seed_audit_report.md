# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actorclip20_s23679`

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
| 2 | 1614.79 | 117.461 | 1497.33 | 0.943596 / 0.807185 / 0.82467 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actorclip20_s23679\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 10908.5 | 2880.39 | 8028.11 | 2.67317 / 4.76485 / 6.92604 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actorclip20_s23679\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 6 | 5999.68 | 1261.32 | 4738.36 | 3.19169 / 2.13409 / 1.75873 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actorclip20_s23679\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3030.59 | 137.231 | 2893.36 | 0.900499 / 0.860907 / 1.13265 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actorclip20_s23679\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 9 | 5486.16 | 1475.06 | 4011.1 | 2.66489 / 4.1445 / 2.49027 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_12000_actorclip20_s23679\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 2 | 0.506667 / 0.00471405 | 0.47 / 0.0163299 | 1.12476 / 0.874931 / 1.1344 | 1.22685 / 1.02593 / 1.41836 | true | PASS_STRONG |
| 3 | 0.1 / 0.0163299 | 0.04 / 0.00816497 | 1.89018 / 2.69244 / 5.07446 | 2.25016 / 3.4928 / 5.74316 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 6 | 0.0266667 / 0.0124722 | 0.0266667 / 0.020548 | 2.52265 / 2.71365 / 2.34103 | 2.68439 / 2.89748 / 2.46914 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 7 | 0.516667 / 0.0188562 | 0.49 / 0.0141421 | 1.05667 / 0.621899 / 0.64281 | 1.09325 / 0.619199 / 0.755887 | true | PASS_STRONG |
| 9 | 0.00666667 / 0.00942809 | 0.00333333 / 0.00471405 | 2.28977 / 3.88087 / 3.50375 | 2.48462 / 4.13743 / 3.44815 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 2 |
| TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE | 3 |

Failed seeds: 3, 6, 9.

Most likely failure explanation: TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.231333 / 0.231013 |
| L2F fixed mean/std success across train seeds | 0.206 / 0.224117 |
| seeds beating Euler baseline > 0.11 | 2 / 5 |
| seeds beating L2F baseline > 0.09 | 2 / 5 |
| seeds beating both baselines | 2 / 5 |
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
