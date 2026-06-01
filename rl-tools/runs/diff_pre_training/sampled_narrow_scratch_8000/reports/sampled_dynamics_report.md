# Sampled-Dynamics Audit Summary

Run root: `runs\diff_pre_training\sampled_narrow_scratch_8000`

This report only evaluates sampled-dynamics training/evaluation for the differentiable pretraining executable. It does not prove teacher-cost reduction, RAPTOR replacement, sampled dynamics solved in general, Sim2Real transfer, or full L2F dynamics equivalence.

## Metadata

| item | value |
| --- | --- |
| run_name | sampled_narrow_scratch_8000 |
| train_seeds | 0 1 2 3 4 |
| steps | 8000 |
| sampled_dynamics_level | narrow |
| init_actor_path | none |
| zero_shot | 0 |
| h128_schedule | balanced_16000 |

## Aggregate

| metric | value |
| --- | ---: |
| Sampled Euler success mean/std | 0.364 / 0.180707 |
| Sampled L2F success mean/std | 0.308667 / 0.170263 |
| Fixed Euler success mean/std after run | 0.26 / 0.139124 |
| Fixed L2F success mean/std after run | 0.207333 / 0.127181 |
| Seeds with nonzero sampled success on both eval models | 5 / 5 |
| Seeds beating old fixed baselines under sampled eval | 4 / 5 |
| Mean sampled invalid_or_nan_rate | 0 |
| Mean dynamics rejection rate | 0 |

## Per Seed

| seed | init | sampled Euler/L2F success | fixed Euler/L2F success | sampled p/v/w Euler | sampled p/v/w L2F | rejection | invalid | sampled nonzero both |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | scratch | 0.6 / 0.536667 | 0.456667 / 0.393333 | 0.996262 / 0.723876 / 1.52698 | 1.08186 / 0.864329 / 1.80094 | 0 | 0 | true |
| 1 | scratch | 0.0566667 / 0.03 | 0.0466667 / 0.0233333 | 3.07412 / 5.07353 / 5.05338 | 3.35901 / 5.73181 / 5.84244 | 0 | 0 | true |
| 2 | scratch | 0.473333 / 0.41 | 0.35 / 0.296667 | 1.21696 / 1.1019 / 1.79488 | 1.35658 / 1.41809 / 2.28422 | 0 | 0 | true |
| 3 | scratch | 0.33 / 0.24 | 0.253333 / 0.173333 | 1.3048 / 1.25372 / 3.01209 | 1.41011 / 1.39486 / 3.31237 | 0 | 0 | true |
| 4 | scratch | 0.36 / 0.326667 | 0.193333 / 0.15 | 1.59129 / 1.33932 / 2.01475 | 1.74314 / 1.56005 / 2.29524 | 0 | 0 | true |
