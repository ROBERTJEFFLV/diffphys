# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_successgate_t020_8000`

This audit only evaluates fixed-dynamics repeatability of SO(3) differentiable pretraining. It does not prove teacher-cost reduction, replacement of RAPTOR teachers, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Completeness

| item | value |
| --- | ---: |
| expected train seeds | 2 |
| completed train seeds | 2 |
| expected eval files | 2 |
| completed eval files | 2 |

Missing artifacts:

None.

Diagnostic fields unavailable without broader model/optimizer instrumentation: actor_grad_max_abs, actor_parameter_norm, actor_update_norm, adam_m_norm, adam_v_norm, gru_hidden_norm_max, gru_hidden_norm_mean

## Training Seed Table

| seed | first loss | final loss | loss reduction | final p/v/w | applied/skipped steps | invalid rollouts | NaN/Inf | checkpoint |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 1 | 1464.14 | 265.849 | 1198.29 | 1.04016 / 0.754263 / 2.0597 | 8000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_successgate_t020_8000\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 2 | 1508.53 | 435.639 | 1072.89 | 1.90773 / 1.72508 / 1.48425 | 8000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_focus_successgate_t020_8000\checkpoints\fixed_vstrong_seed2_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 0.4 / 0 | not available / not available | 1.38906 / 0.902144 / 1.81925 | not available / not available / not available | false | TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH |
| 2 | 0.2 / 0 | not available / not available | 1.93151 / 1.78901 / 1.32294 | not available / not available / not available | false | TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH | 2 |

Failed seeds: 1, 2.

Most likely failure explanation: TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.3 / 0.1 |
| L2F fixed mean/std success across train seeds | not available / not available |
| seeds beating Euler baseline > 0.11 | 2 / 2 |
| seeds beating L2F baseline > 0.09 | 0 / 2 |
| seeds beating both baselines | 0 / 2 |
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

Do not automatically modify training from this audit alone; run a targeted ablation next: investigate Euler-to-L2F model mismatch.
