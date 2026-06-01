# Sampled-Dynamics Audit Summary

Run root: `runs\diff_pre_training\sampled_narrow_finetune_from_fixed_8000`

This report only evaluates sampled-dynamics training/evaluation for the differentiable pretraining executable. It does not prove teacher-cost reduction, RAPTOR replacement, sampled dynamics solved in general, Sim2Real transfer, or full L2F dynamics equivalence.

## Metadata

| item | value |
| --- | --- |
| run_name | sampled_narrow_finetune_from_fixed_8000 |
| train_seeds | 0 1 2 3 4 |
| steps | 8000 |
| sampled_dynamics_level | narrow |
| init_actor_path | runs/diff_pre_training/diag_h128prio_16000_10seed/checkpoints |
| zero_shot | 0 |
| h128_schedule | balanced_16000 |

## Aggregate

| metric | value |
| --- | ---: |
| Sampled Euler success mean/std | 0.416 / 0.262876 |
| Sampled L2F success mean/std | 0.375333 / 0.247347 |
| Fixed Euler success mean/std after run | 0.331333 / 0.184338 |
| Fixed L2F success mean/std after run | 0.285333 / 0.164986 |
| Seeds with nonzero sampled success on both eval models | 4 / 5 |
| Seeds beating old fixed baselines under sampled eval | 4 / 5 |
| Mean sampled invalid_or_nan_rate | 0 |
| Mean dynamics rejection rate | 0 |

## Per Seed

| seed | init | sampled Euler/L2F success | fixed Euler/L2F success | sampled p/v/w Euler | sampled p/v/w L2F | rejection | invalid | sampled nonzero both |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | fixed-pretrained | 0.67 / 0.646667 | 0.48 / 0.433333 | 0.869679 / 0.494519 / 0.514415 | 0.916122 / 0.519036 / 0.854951 | 0 | 0 | true |
| 1 | fixed-pretrained | 0.00333333 / 0 | 0 / 0 | 3.45622 / 5.04738 / 3.19171 | 3.69018 / 5.40502 / 3.35876 | 0 | 0 | false |
| 2 | fixed-pretrained | 0.643333 / 0.59 | 0.476667 / 0.443333 | 0.910518 / 0.641218 / 1.15214 | 0.96043 / 0.692083 / 1.36338 | 0 | 0 | true |
| 3 | fixed-pretrained | 0.55 / 0.46 | 0.44 / 0.336667 | 1.00465 / 0.615928 / 0.875119 | 1.16092 / 0.737708 / 1.60693 | 0 | 0 | true |
| 4 | fixed-pretrained | 0.213333 / 0.18 | 0.26 / 0.213333 | 1.6846 / 1.98999 / 1.43882 | 1.94351 / 2.41782 / 1.77081 | 0 | 0 | true |
