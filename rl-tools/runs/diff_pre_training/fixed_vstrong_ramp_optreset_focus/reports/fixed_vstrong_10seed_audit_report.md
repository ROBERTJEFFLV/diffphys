# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_ramp_optreset_focus`

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
| 1 | 1513.71 | 280.534 | 1233.18 | 0.802157 / 0.57375 / 2.71325 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_ramp_optreset_focus\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 2 | 1614.79 | 763.155 | 851.635 | 2.22597 / 2.97118 / 1.51907 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_ramp_optreset_focus\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 10908.5 | 301.275 | 10607.2 | 1.02046 / 0.475823 / 2.62171 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_ramp_optreset_focus\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 4 | 2817.89 | 3136.94 | -319.05 | 3.04838 / 5.19665 / 5.33746 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_ramp_optreset_focus\checkpoints\fixed_vstrong_seed4_policy.bin` |
| 6 | 5999.68 | 1905.85 | 4093.83 | 1.5316 / 2.25275 / 6.99423 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_ramp_optreset_focus\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3030.59 | 940.167 | 2090.42 | 2.41905 / 2.97611 / 2.3049 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_ramp_optreset_focus\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 9 | 5486.16 | 1427.74 | 4058.42 | 3.36623 / 3.43224 / 2.56562 | 6000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_ramp_optreset_focus\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 0.503333 / 0.0329983 | 0.353333 / 0.0235702 | 1.05669 / 0.688035 / 1.93536 | 1.1374 / 0.961125 / 4.11084 | true | PASS_STRONG |
| 2 | 0.00666667 / 0.00471405 | 0.00333333 / 0.00471405 | 2.68298 / 3.29095 / 2.07957 | 2.81201 / 3.58864 / 2.69588 | false | TYPE_G_OBJECTIVE_MISMATCH |
| 3 | 0.323333 / 0.0524934 | 0.14 / 0.0141421 | 1.43745 / 1.35689 / 3.85874 | 1.74049 / 2.11662 / 5.27905 | true | PASS_WEAK |
| 4 | 0.113333 / 0.00471405 | 0.0766667 / 0.0124722 | 2.67285 / 4.66795 / 6.96226 | 3.09917 / 5.6323 / 8.14958 | false | TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH |
| 6 | 0.113333 / 0.0169967 | 0.0333333 / 0.0169967 | 1.19002 / 1.85078 / 7.35592 | 1.45084 / 2.51011 / 8.27854 | false | TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH |
| 7 | 0.0266667 / 0.020548 | 0.00666667 / 0.00471405 | 2.92521 / 4.46872 / 3.55097 | 3.34686 / 5.2523 / 3.81457 | false | TYPE_G_OBJECTIVE_MISMATCH |
| 9 | 0.06 / 0.0216025 | 0.0433333 / 0.00942809 | 2.49091 / 3.17929 / 2.11322 | 2.70454 / 3.53172 / 2.36103 | false | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 1 |
| PASS_WEAK | 1 |
| TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE | 1 |
| TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH | 2 |
| TYPE_G_OBJECTIVE_MISMATCH | 2 |

Failed seeds: 2, 4, 6, 7, 9.

Most likely failure explanation: TYPE_G_OBJECTIVE_MISMATCH is the most common non-pass label.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.16381 / 0.169072 |
| L2F fixed mean/std success across train seeds | 0.0938095 / 0.114496 |
| seeds beating Euler baseline > 0.11 | 4 / 7 |
| seeds beating L2F baseline > 0.09 | 2 / 7 |
| seeds beating both baselines | 2 / 7 |
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

Do not automatically modify training from this audit alone; run a targeted ablation next: revise objective/success alignment.
