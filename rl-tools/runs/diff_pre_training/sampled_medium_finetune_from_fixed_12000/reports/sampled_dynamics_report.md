# Sampled-Dynamics Audit Summary

Run root: `runs\diff_pre_training\sampled_medium_finetune_from_fixed_12000`

This report only evaluates sampled-dynamics training/evaluation for the differentiable pretraining executable. It does not prove teacher-cost reduction, RAPTOR replacement, sampled dynamics solved in general, Sim2Real transfer, or full L2F dynamics equivalence.

## Metadata

| item | value |
| --- | --- |
| run_name | sampled_medium_finetune_from_fixed_12000 |
| train_seeds | 0 1 2 3 4 |
| steps | 12000 |
| sampled_dynamics_level | medium |
| init_actor_path | runs/diff_pre_training/diag_h128prio_16000_10seed/checkpoints |
| zero_shot | 0 |
| h128_schedule | balanced_16000 |

## Aggregate

| metric | value |
| --- | ---: |
| Sampled Euler success mean/std | 0.556 / 0.0997419 |
| Sampled L2F success mean/std | 0.519333 / 0.113498 |
| Fixed Euler success mean/std after run | 0.284 / 0.0223507 |
| Fixed L2F success mean/std after run | 0.234667 / 0.0154344 |
| Seeds with nonzero sampled success on both eval models | 5 / 5 |
| Seeds beating old fixed baselines under sampled eval | 5 / 5 |
| Mean sampled invalid_or_nan_rate | 0 |
| Mean dynamics rejection rate | 0 |

## Per Seed

| seed | init | sampled Euler/L2F success | fixed Euler/L2F success | sampled p/v/w Euler | sampled p/v/w L2F | rejection | invalid | sampled nonzero both |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | fixed-pretrained | 0.703333 / 0.7 | 0.263333 / 0.256667 | 0.826487 / 0.576945 / 0.615255 | 0.846254 / 0.614304 / 1.01996 | 0 | 0 | true |
| 1 | fixed-pretrained | 0.623333 / 0.59 | 0.256667 / 0.23 | 0.893497 / 0.805819 / 1.30833 | 0.956911 / 0.897248 / 1.92506 | 0 | 0 | true |
| 2 | fixed-pretrained | 0.493333 / 0.44 | 0.283333 / 0.21 | 1.02723 / 1.47901 / 0.963067 | 1.12168 / 1.58711 / 1.12058 | 0 | 0 | true |
| 3 | fixed-pretrained | 0.543333 / 0.486667 | 0.316667 / 0.243333 | 1.03832 / 1.05586 / 0.71265 | 1.11002 / 1.07351 / 0.793741 | 0 | 0 | true |
| 4 | fixed-pretrained | 0.416667 / 0.38 | 0.3 / 0.233333 | 1.19609 / 1.68141 / 1.42081 | 1.39803 / 1.94665 / 1.80657 | 0 | 0 | true |
