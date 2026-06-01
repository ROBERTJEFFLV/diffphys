# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_sg_t020_8000`

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
| 1 | 1464.14 | 265.849 | 1198.29 | 1.04016 / 0.754263 / 2.0597 | 8000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_sg_t020_8000\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 2 | 1508.53 | 435.639 | 1072.89 | 1.90773 / 1.72508 / 1.48425 | 8000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_sg_t020_8000\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 11322.1 | 286.568 | 11035.5 | 0.823869 / 0.81485 / 2.5765 | 8000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_sg_t020_8000\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 4 | 2797.77 | 364.937 | 2432.83 | 1.60049 / 1.04766 / 1.50474 | 8000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_sg_t020_8000\checkpoints\fixed_vstrong_seed4_policy.bin` |
| 6 | 6006.98 | 7342.73 | -1335.75 | 4.71386 / 8.41419 / 10.1902 | 8000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_sg_t020_8000\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3126.8 | 4912.84 | -1786.04 | 4.14576 / 6.71123 / 5.06183 | 8000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_sg_t020_8000\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 9 | 5448.25 | 2829.37 | 2618.88 | 2.55743 / 6.01756 / 4.94257 | 8000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_sg_t020_8000\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 0.38 / 0.0163299 | 0.323333 / 0.0169967 | 1.26874 / 0.813862 / 1.64711 | 1.30049 / 0.885818 / 2.91783 | true | PASS_STRONG |
| 2 | 0.233333 / 0.0262467 | 0.216667 / 0.00942809 | 1.64391 / 1.36678 / 0.987143 | 1.72227 / 1.52283 / 1.26412 | true | PASS_WEAK |
| 3 | 0.38 / 0.0141421 | 0.3 / 0.0216025 | 1.32391 / 1.14803 / 2.09708 | 1.49685 / 1.65734 / 3.13194 | true | PASS_STRONG |
| 4 | 0.336667 / 0.020548 | 0.31 / 0.0216025 | 1.52914 / 1.39751 / 2.06463 | 1.68439 / 1.67934 / 2.43475 | true | PASS_STRONG |
| 6 | 0 / 0 | 0 / 0 | 3.66044 / 7.44422 / 8.77144 | 4.03839 / 7.98994 / 8.49459 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 7 | 0.0466667 / 0.00471405 | 0.05 / 0.0216025 | 3.29563 / 5.63748 / 3.81845 | 3.54349 / 6.01467 / 4.05812 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 9 | 0.01 / 0.00816497 | 0.0133333 / 0.0124722 | 2.88272 / 5.69802 / 6.36649 | 3.24276 / 6.31405 / 6.81022 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 3 |
| PASS_WEAK | 1 |
| TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE | 3 |

Failed seeds: 6, 7, 9.

Most likely failure explanation: TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.198095 / 0.162206 |
| L2F fixed mean/std success across train seeds | 0.173333 / 0.136254 |
| seeds beating Euler baseline > 0.11 | 4 / 7 |
| seeds beating L2F baseline > 0.09 | 4 / 7 |
| seeds beating both baselines | 4 / 7 |
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

Do not automatically modify training from this audit alone; run a targeted ablation next: slow down or freeze horizon/state curriculum.
