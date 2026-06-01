# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_baseline_12000_s23679`

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
| 2 | 1614.79 | 137.185 | 1477.61 | 0.996408 / 0.85892 / 0.879405 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_baseline_12000_s23679\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 10908.5 | 166.418 | 10742.1 | 0.844834 / 0.748245 / 1.80183 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_baseline_12000_s23679\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 6 | 5999.68 | 720.87 | 5278.81 | 1.98262 / 1.72767 / 2.27642 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_baseline_12000_s23679\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3030.59 | 70.0395 | 2960.55 | 0.722452 / 0.413347 / 0.626836 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_baseline_12000_s23679\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 9 | 5486.16 | 1258.18 | 4227.98 | 2.62143 / 3.17234 / 3.20231 | 12000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_baseline_12000_s23679\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 2 | 0.51 / 0.00816497 | 0.463333 / 0.020548 | 1.13471 / 0.901785 / 1.21135 | 1.22398 / 1.0451 / 1.46497 | true | PASS_STRONG |
| 3 | 0.563333 / 0.0262467 | 0.506667 / 0.020548 | 0.997903 / 0.642011 / 1.4914 | 1.11167 / 0.798045 / 2.66651 | true | PASS_STRONG |
| 6 | 0.04 / 0.0163299 | 0.0566667 / 0.0188562 | 2.23485 / 2.95693 / 2.82589 | 2.3893 / 3.10353 / 2.96163 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 7 | 0.583333 / 0.00942809 | 0.553333 / 0.0124722 | 0.966344 / 0.444165 / 0.677757 | 1.00529 / 0.466842 / 0.995929 | true | PASS_STRONG |
| 9 | 0.0133333 / 0.00471405 | 0.00333333 / 0.00471405 | 2.28892 / 3.57085 / 5.0818 | 2.55165 / 4.07565 / 5.13475 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 3 |
| TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE | 2 |

Failed seeds: 6, 9.

Most likely failure explanation: TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.342 / 0.25872 |
| L2F fixed mean/std success across train seeds | 0.316667 / 0.23639 |
| seeds beating Euler baseline > 0.11 | 3 / 5 |
| seeds beating L2F baseline > 0.09 | 3 / 5 |
| seeds beating both baselines | 3 / 5 |
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
