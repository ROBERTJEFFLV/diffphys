# Sampled-Dynamics Audit Summary

Run root: `runs\diff_pre_training\sampled_zeroshot_fixedpretrain_narrow`

This report only evaluates sampled-dynamics training/evaluation for the differentiable pretraining executable. It does not prove teacher-cost reduction, RAPTOR replacement, sampled dynamics solved in general, Sim2Real transfer, or full L2F dynamics equivalence.

## Metadata

| item | value |
| --- | --- |
| run_name | sampled_zeroshot_fixedpretrain_narrow |
| train_seeds | 0 1 2 3 4 |
| steps | 8000 |
| sampled_dynamics_level | narrow |
| init_actor_path | runs/diff_pre_training/diag_h128prio_16000_10seed/checkpoints |
| zero_shot | 1 |
| h128_schedule | balanced_16000 |

## Aggregate

| metric | value |
| --- | ---: |
| Sampled Euler success mean/std | 0.348667 / 0.139564 |
| Sampled L2F success mean/std | 0.327333 / 0.136941 |
| Fixed Euler success mean/std after run | nan / nan |
| Fixed L2F success mean/std after run | nan / nan |
| Seeds with nonzero sampled success on both eval models | 5 / 5 |
| Seeds beating old fixed baselines under sampled eval | 5 / 5 |
| Mean sampled invalid_or_nan_rate | 0 |
| Mean dynamics rejection rate | nan |

## Per Seed

| seed | init | sampled Euler/L2F success | fixed Euler/L2F success | sampled p/v/w Euler | sampled p/v/w L2F | rejection | invalid | sampled nonzero both |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | fixed-pretrained-zero-shot | 0.223333 / 0.25 | nan / nan | 1.3545 / 1.64282 / 1.88765 | 1.42908 / 1.60847 / 2.05259 | nan | 0 | true |
| 1 | fixed-pretrained-zero-shot | 0.36 / 0.276667 | nan / nan | 1.38123 / 1.74719 / 2.46286 | 1.50115 / 1.98873 / 2.84108 | nan | 0 | true |
| 2 | fixed-pretrained-zero-shot | 0.436667 / 0.41 | nan / nan | 1.14963 / 1.39186 / 1.58752 | 1.2035 / 1.37655 / 1.76006 | nan | 0 | true |
| 3 | fixed-pretrained-zero-shot | 0.553333 / 0.546667 | nan / nan | 0.978394 / 1.08232 / 1.01662 | 1.00249 / 1.02957 / 1.2727 | nan | 0 | true |
| 4 | fixed-pretrained-zero-shot | 0.17 / 0.153333 | nan / nan | 1.84898 / 2.29969 / 2.94702 | 2.01491 / 2.57846 / 3.09559 | nan | 0 | true |
