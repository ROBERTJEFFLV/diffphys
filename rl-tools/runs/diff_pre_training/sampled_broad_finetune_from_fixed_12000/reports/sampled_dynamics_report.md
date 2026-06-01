# Sampled-Dynamics Audit Summary

Run root: `runs\diff_pre_training\sampled_broad_finetune_from_fixed_12000`

This report only evaluates sampled-dynamics training/evaluation for the differentiable pretraining executable. It does not prove teacher-cost reduction, RAPTOR replacement, sampled dynamics solved in general, Sim2Real transfer, or full L2F dynamics equivalence.

## Metadata

| item | value |
| --- | --- |
| run_name | sampled_broad_finetune_from_fixed_12000 |
| train_seeds | 0 1 2 3 4 |
| steps | 12000 |
| sampled_dynamics_level | broad |
| init_actor_path | runs/diff_pre_training/diag_h128prio_16000_10seed/checkpoints |
| zero_shot | 0 |
| h128_schedule | balanced_16000 |

## Aggregate

| metric | value |
| --- | ---: |
| Sampled Euler success mean/std | 0.276667 / 0.0630696 |
| Sampled L2F success mean/std | 0.226667 / 0.0543241 |
| Fixed Euler success mean/std after run | 0 / 0 |
| Fixed L2F success mean/std after run | 0 / 0 |
| Seeds with nonzero sampled success on both eval models | 5 / 5 |
| Seeds beating old fixed baselines under sampled eval | 5 / 5 |
| Mean sampled invalid_or_nan_rate | 0 |
| Mean dynamics rejection rate | 0.724003 |

## Per Seed

| seed | init | sampled Euler/L2F success | fixed Euler/L2F success | sampled p/v/w Euler | sampled p/v/w L2F | rejection | invalid | sampled nonzero both |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | fixed-pretrained | 0.33 / 0.263333 | 0 / 0 | 1.47491 / 1.37258 / 1.53543 | 1.66648 / 1.70964 / 1.98912 | 0.809524 | 0 | true |
| 1 | fixed-pretrained | 0.34 / 0.296667 | 0 / 0 | 1.29016 / 1.18238 / 1.5139 | 1.44161 / 1.4777 / 2.10221 | 0.8 | 0 | true |
| 2 | fixed-pretrained | 0.22 / 0.17 | 0 / 0 | 1.68034 / 1.90081 / 2.24191 | 1.86843 / 2.28401 / 2.68357 | 0.5 | 0 | true |
| 3 | fixed-pretrained | 0.31 / 0.246667 | 0 / 0 | 1.42114 / 1.43035 / 1.64111 | 1.58854 / 1.68776 / 2.08186 | 0.818182 | 0 | true |
| 4 | fixed-pretrained | 0.183333 / 0.156667 | 0 / 0 | 1.88864 / 2.01722 / 2.09647 | 2.04683 / 2.32835 / 2.47154 | 0.692308 | 0 | true |
