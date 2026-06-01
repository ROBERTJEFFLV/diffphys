# Sampled-Dynamics Audit Summary

Run root: `runs\diff_pre_training\sampled_medium_scratch_12000`

This report only evaluates sampled-dynamics training/evaluation for the differentiable pretraining executable. It does not prove teacher-cost reduction, RAPTOR replacement, sampled dynamics solved in general, Sim2Real transfer, or full L2F dynamics equivalence.

## Metadata

| item | value |
| --- | --- |
| run_name | sampled_medium_scratch_12000 |
| train_seeds | 0 1 2 3 4 |
| steps | 12000 |
| sampled_dynamics_level | medium |
| init_actor_path | none |
| zero_shot | 0 |
| h128_schedule | balanced_16000 |

## Aggregate

| metric | value |
| --- | ---: |
| Sampled Euler success mean/std | 0.406667 / 0.0981609 |
| Sampled L2F success mean/std | 0.346667 / 0.103966 |
| Fixed Euler success mean/std after run | 0.208 / 0.0297844 |
| Fixed L2F success mean/std after run | 0.168667 / 0.0309552 |
| Seeds with nonzero sampled success on both eval models | 5 / 5 |
| Seeds beating old fixed baselines under sampled eval | 5 / 5 |
| Mean sampled invalid_or_nan_rate | 0 |
| Mean dynamics rejection rate | 0 |

## Per Seed

| seed | init | sampled Euler/L2F success | fixed Euler/L2F success | sampled p/v/w Euler | sampled p/v/w L2F | rejection | invalid | sampled nonzero both |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | scratch | 0.31 / 0.256667 | 0.23 / 0.206667 | 1.38818 / 1.37905 / 1.76209 | 1.48512 / 1.47145 / 1.92539 | 0 | 0 | true |
| 1 | scratch | 0.403333 / 0.343333 | 0.176667 / 0.153333 | 1.32632 / 1.12232 / 1.47192 | 1.46507 / 1.25524 / 1.61695 | 0 | 0 | true |
| 2 | scratch | 0.463333 / 0.403333 | 0.21 / 0.16 | 1.1833 / 1.19149 / 1.75833 | 1.27152 / 1.37503 / 2.10534 | 0 | 0 | true |
| 3 | scratch | 0.56 / 0.51 | 0.25 / 0.2 | 1.02282 / 0.921184 / 1.28809 | 1.10886 / 1.03174 / 2.10576 | 0 | 0 | true |
| 4 | scratch | 0.296667 / 0.22 | 0.173333 / 0.123333 | 1.52167 / 1.48224 / 2.31382 | 1.69848 / 1.75347 / 2.84451 | 0 | 0 | true |
