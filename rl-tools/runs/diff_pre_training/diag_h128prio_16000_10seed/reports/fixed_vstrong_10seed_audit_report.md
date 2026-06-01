# Fixed SO(3) 10-Seed Stability Audit

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed`

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

Diagnostic fields unavailable without broader model/optimizer instrumentation: actor_grad_max_abs, actor_param_max_abs, actor_param_norm, actor_parameter_norm, actor_update_norm, actor_update_to_param_norm_ratio, adam_m_norm, adam_v_norm, gru_hidden_abs_max, gru_hidden_abs_mean, gru_hidden_norm_max, gru_hidden_norm_mean

## Training Seed Table

| seed | first loss | final loss | loss reduction | final p/v/w | applied/skipped steps | invalid rollouts | NaN/Inf | checkpoint |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| 0 | 1056.45 | 328.729 | 727.721 | 1.54806 / 1.15451 / 1.78463 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed0_policy.bin` |
| 1 | 1513.71 | 1269.96 | 243.75 | 2.03638 / 2.40398 / 4.00371 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed1_policy.bin` |
| 2 | 1614.79 | 163.598 | 1451.19 | 1.10572 / 0.850712 / 1.13674 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed2_policy.bin` |
| 3 | 10908.5 | 118.027 | 10790.5 | 1.19614 / 0.593637 / 0.585823 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed3_policy.bin` |
| 4 | 2817.89 | 855.247 | 1962.64 | 2.00647 / 2.33682 / 3.46853 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed4_policy.bin` |
| 5 | 2785.71 | 224.101 | 2561.61 | 1.23706 / 0.792659 / 1.16237 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed5_policy.bin` |
| 6 | 5999.68 | 305.531 | 5694.15 | 1.63678 / 1.11268 / 1.03602 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed6_policy.bin` |
| 7 | 3030.59 | 81.7954 | 2948.79 | 1.0691 / 0.409536 / 0.419191 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed7_policy.bin` |
| 8 | 14880.7 | 85.3609 | 14795.3 | 0.608654 / 0.73802 / 0.872705 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed8_policy.bin` |
| 9 | 5486.16 | 166.443 | 5319.72 | 1.10511 / 0.651759 / 1.26722 | 16000 / 0 | 0 | false | `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\diag_h128prio_16000_10seed\checkpoints\fixed_vstrong_seed9_policy.bin` |

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 0.376667 / 0.0492161 | 0.336667 / 0.0309121 | 1.35805 / 1.28306 / 1.62143 | 1.43411 / 1.30211 / 1.83027 | true | PASS_STRONG |
| 1 | 0.346667 / 0.0262467 | 0.26 / 0.0141421 | 1.5128 / 1.53133 / 1.99572 | 1.62642 / 1.78267 / 2.61511 | true | PASS_STRONG |
| 2 | 0.42 / 0.0141421 | 0.39 / 0.0216025 | 1.18171 / 0.89321 / 1.23545 | 1.24556 / 0.949922 / 1.35783 | true | PASS_STRONG |
| 3 | 0.55 / 0.0141421 | 0.503333 / 0.0329983 | 1.05434 / 0.626378 / 0.788562 | 1.10321 / 0.629174 / 1.038 | true | PASS_STRONG |
| 4 | 0.166667 / 0.0188562 | 0.153333 / 0.0286744 | 1.96503 / 2.13707 / 2.62275 | 2.16518 / 2.49279 / 2.80924 | true | PASS_WEAK |
| 5 | 0.636667 / 0.0410961 | 0.556667 / 0.0385861 | 0.943097 / 0.767772 / 1.23828 | 1.03156 / 0.799821 / 1.5372 | true | PASS_STRONG |
| 6 | 0.373333 / 0.0124722 | 0.336667 / 0.00942809 | 1.39923 / 1.43424 / 1.33208 | 1.53689 / 1.50234 / 1.39897 | true | PASS_STRONG |
| 7 | 0.55 / 0.0374166 | 0.526667 / 0.0402768 | 1.00298 / 0.641412 / 0.637403 | 1.03778 / 0.668072 / 0.827412 | true | PASS_STRONG |
| 8 | 0.543333 / 0.0771722 | 0.49 / 0.0496655 | 1.12605 / 0.879455 / 1.55798 | 1.25053 / 0.942366 / 2.1738 | true | PASS_STRONG |
| 9 | 0.466667 / 0.0309121 | 0.45 / 0.0141421 | 1.07648 / 0.790859 / 1.5722 | 1.14512 / 0.896309 / 1.99464 | true | PASS_STRONG |

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
| PASS_STRONG | 9 |
| PASS_WEAK | 1 |

Failed seeds: None.

Most likely failure explanation: No non-pass failure type was observed.

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | 0.443 / 0.129017 |
| L2F fixed mean/std success across train seeds | 0.400333 / 0.122651 |
| seeds beating Euler baseline > 0.11 | 10 / 10 |
| seeds beating L2F baseline > 0.09 | 10 / 10 |
| seeds beating both baselines | 10 / 10 |
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

STRONG_STABILITY

## Recommended Next Action

Do not automatically modify training from this audit alone; run a targeted ablation next: inspect failed seed logs manually before changing training.
