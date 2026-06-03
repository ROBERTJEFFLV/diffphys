# Fixed SO(3) 10-Seed Stability Audit

Run root: `/home/lvmingyang/raptor_diff_20260531_215150/diffphys/rl-tools/runs/diff_pre_training/teacher_cost_fixed_balanced_16000_seed0`

This audit only evaluates fixed-dynamics repeatability of SO(3) differentiable pretraining. It does not prove teacher-cost reduction, replacement of RAPTOR teachers, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Completeness

| item | value |
| --- | ---: |
| expected train seeds | 1 |
| completed train seeds | 1 |
| expected eval files | 6 |
| completed eval files | 6 |

Missing artifacts:

None.

Diagnostic fields unavailable without broader model/optimizer instrumentation: actor_grad_max_abs, actor_param_max_abs, actor_param_norm, actor_parameter_norm, actor_update_norm, actor_update_to_param_norm_ratio, adam_m_norm, adam_v_norm, gru_hidden_abs_max, gru_hidden_abs_mean, gru_hidden_norm_max, gru_hidden_norm_mean

## Training Seed Table

| seed | first loss | final loss | loss reduction | final p/v/w | applied/skipped steps | invalid rollouts | NaN/Inf | checkpoint |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 0 | 1056.45 | 303.871 | 752.579 | 1.64621 / 0.983908 / 1.36272 | 16000 / 0 | 0 | false | `/home/lvmingyang/raptor_diff_20260531_215150/diffphys/rl-tools/runs/diff_pre_training/teacher_cost_fixed_balanced_16000_seed0/checkpoints/fixed_vstrong_seed0_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 0.353333 / 0.046428 | 0.35 / 0.0374166 | 1.37041 / 1.3897 / 1.52198 | 1.43028 / 1.41321 / 1.60735 | true | PASS_STRONG |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 1 |

Failed seeds: None.

Most likely failure explanation: No non-pass failure type was observed.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.353333 / 0 |
| L2F fixed mean/std success across train seeds | 0.35 / 0 |
| seeds beating Euler baseline > 0.11 | 1 / 1 |
| seeds beating L2F baseline > 0.09 | 1 / 1 |
| seeds beating both baselines | 1 / 1 |
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

Do not automatically modify training from this audit alone; run a targeted ablation next: inspect failed seed logs manually before changing training.
