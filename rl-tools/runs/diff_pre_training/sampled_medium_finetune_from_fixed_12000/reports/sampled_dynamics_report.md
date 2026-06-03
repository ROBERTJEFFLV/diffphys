# Sampled-Dynamics Audit Summary

Run root: `runs/diff_pre_training/sampled_medium_finetune_from_fixed_12000`

This report only evaluates sampled-dynamics training/evaluation for the differentiable pretraining executable. It does not prove teacher-cost reduction, RAPTOR replacement, sampled dynamics solved in general, Sim2Real transfer, or full L2F dynamics equivalence.

## Metadata

| item | value |
| --- | --- |
| run_name | sampled_medium_finetune_from_fixed_12000 |
| train_seeds | 0 |
| steps | 12000 |
| sampled_dynamics_level | medium |
| init_actor_path | runs/diff_pre_training/teacher_cost_fixed_balanced_16000_seed0/checkpoints |
| zero_shot | 0 |
| h128_schedule | balanced_16000 |

## Aggregate

| metric | value |
| --- | ---: |
| Sampled Euler success mean/std | 0.626667 / 0 |
| Sampled L2F success mean/std | 0.52 / 0 |
| Fixed Euler success mean/std after run | 0.21 / 0 |
| Fixed L2F success mean/std after run | 0.18 / 0 |
| Seeds with nonzero sampled success on both eval models | 1 / 1 |
| Seeds beating old fixed baselines under sampled eval | 1 / 1 |
| Mean sampled invalid_or_nan_rate | 0 |
| Mean dynamics rejection rate | 0 |

## Per Seed

| seed | init | sampled Euler/L2F success | fixed Euler/L2F success | sampled p/v/w Euler | sampled p/v/w L2F | rejection | invalid | sampled nonzero both |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | fixed-pretrained | 0.626667 / 0.52 | 0.21 / 0.18 | 0.878641 / 0.643091 / 1.27172 | 1.005 / 0.849975 / 2.36357 | 0 | 0 | true |
