# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_slowcurr_failedseed_validation`

This audit only evaluates fixed-dynamics repeatability of SO(3) differentiable pretraining. It does not prove teacher-cost reduction, replacement of RAPTOR teachers, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Completeness

| item | value |
| --- | ---: |
| expected train seeds | 4 |
| completed train seeds | 4 |
| expected eval files | 24 |
| completed eval files | 24 |

Missing artifacts:

None.

Diagnostic fields unavailable without broader model/optimizer instrumentation: actor_grad_max_abs, actor_update_norm, gru_hidden_norm_max, gru_hidden_norm_mean

## Training Seed Table

| seed | first loss | final loss | loss reduction | final p/v/w | applied/skipped steps | invalid rollouts | NaN/Inf | checkpoint |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 1 | 1513.71 | 119.71 | 1394 | 0.832233 / 0.355081 / 1.4287 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_slowcurr_failedseed_validation\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 3 | 10908.5 | 196.954 | 10711.5 | 0.888229 / 0.48937 / 1.83096 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_slowcurr_failedseed_validation\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 6 | 5999.68 | 958.354 | 5041.33 | 1.34655 / 1.6855 / 4.39345 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_slowcurr_failedseed_validation\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 9 | 5486.16 | 1203.13 | 4283.03 | 2.96404 / 3.16402 / 1.38819 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_slowcurr_failedseed_validation\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 0.523333 / 0.0235702 | 0.35 / 0.0141421 | 1.00804 / 0.633199 / 2.36491 | 1.12115 / 1.03343 / 4.63023 | true | PASS_STRONG |
| 3 | 0.52 / 0.0374166 | 0.363333 / 0.0309121 | 1.0951 / 0.821564 / 2.59627 | 1.29177 / 1.18877 / 3.83859 | true | PASS_STRONG |
| 6 | 0.216667 / 0.020548 | 0.0966667 / 0.020548 | 1.0247 / 1.32949 / 6.13414 | 1.20313 / 1.796 / 7.51311 | true | PASS_WEAK |
| 9 | 0.156667 / 0.020548 | 0.1 / 0.0163299 | 1.97734 / 2.04784 / 1.41196 | 2.20466 / 2.44896 / 1.80198 | true | PASS_WEAK |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 2 |
| PASS_WEAK | 2 |

Failed seeds: None.

Most likely failure explanation: No non-pass failure type was observed.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.354167 / 0.168842 |
| L2F fixed mean/std success across train seeds | 0.2275 / 0.129258 |
| seeds beating Euler baseline > 0.11 | 4 / 4 |
| seeds beating L2F baseline > 0.09 | 4 / 4 |
| seeds beating both baselines | 4 / 4 |
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
