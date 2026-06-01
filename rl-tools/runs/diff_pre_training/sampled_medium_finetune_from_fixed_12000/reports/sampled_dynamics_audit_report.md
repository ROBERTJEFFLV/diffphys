# Sampled-Dynamics Audit After Fixed H128 Stability

Run root: `runs/diff_pre_training/sampled_medium_finetune_from_fixed_12000`

This audit tests whether the fixed-dynamics SO(3) differentiable-pretraining checkpoint is useful under sampled dynamics. It does not modify SO(3) equations, VJP/transition math, motor delay, thrust polynomial, torque model, actor observation, original L2F reward, SAC/TD3, RAPTOR teacher code, or HDF5 paths. It does not prove teacher-cost reduction, RAPTOR replacement, sampled-dynamics success in general, Sim2Real transfer, or full L2F dynamics equivalence.

## Physics Gate And Fixed Checkpoint

Physics gate:

```powershell
build_diff\bin\Release\foundation_policy_diff_physics_check.exe --diff-model euler
```

Result: `overall=STRICT_PASS`.

Fixed checkpoint source:

```text
runs/diff_pre_training/diag_h128prio_16000_10seed/checkpoints/
```

Manifest:

```text
runs/diff_pre_training/diag_h128prio_16000_10seed/reports/best_fixed_checkpoint_manifest.csv
```

The fixed H128-prioritized `balanced_16000` audit reached Euler fixed success `0.443 +/- 0.129017`, L2F fixed success `0.400333 +/- 0.122651`, with 10/10 seeds beating both old fixed first-order baselines, zero skipped actor steps, zero invalid/NaN rate, and no NaN/Inf flags.

## Sampled Levels

| level | definition used in this audit |
| --- | --- |
| `fixed` | nominal fixed dynamics, no sampling |
| `narrow` | domain randomization around nominal parameters with variation `0.10`, disturbance force max `0.05`, and the existing sampled-parameter safety filter |
| `medium` | domain randomization around nominal parameters with variation `0.25`, disturbance force max `0.15`, and the existing sampled-parameter safety filter |
| `broad` | existing broad L2F domain randomization; broad exploratory run used `--dynamics-curriculum` safety filtering |

All levels preserve the implicit-identification actor observation and do not expose mass, inertia, thrust coefficients, torque constants, or motor time constants to the actor.

## Commands

Common launcher:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_sampled_dynamics_audit.sh
```

Zero-shot narrow sampled evaluation:

```powershell
$env:RUN_NAME='sampled_zeroshot_fixedpretrain_narrow'
$env:TRAIN_SEEDS='0 1 2 3 4'
$env:ZERO_SHOT='1'
$env:INIT_ACTOR_PATH='runs/diff_pre_training/diag_h128prio_16000_10seed/checkpoints'
$env:SAMPLED_DYNAMICS_LEVEL='narrow'
$env:EVAL_SEEDS='10000 10001 10002'
$env:EVAL_MODELS='euler l2f'
$env:EVAL_DYNAMICS_MODES='sampled'
$env:EVAL_EPISODES='100'
$env:EVAL_HORIZON='128'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:PYTHON='python'
```

Narrow fixed-init fine-tune:

```powershell
$env:RUN_NAME='sampled_narrow_finetune_from_fixed_8000'
$env:TRAIN_SEEDS='0 1 2 3 4'
$env:STEPS='8000'
$env:ZERO_SHOT='0'
$env:INIT_ACTOR_PATH='runs/diff_pre_training/diag_h128prio_16000_10seed/checkpoints'
$env:SAMPLED_DYNAMICS_LEVEL='narrow'
$env:EVAL_DYNAMICS_MODES='sampled fixed'
$env:H128_PRIORITIZED_CURRICULUM='1'
$env:H128_SCHEDULE='balanced_16000'
$env:DIAGNOSTIC_LOG_DETAIL='1'
$env:DIAGNOSTIC_LOG_EVERY='20'
$env:DIAGNOSTIC_LOG_FIRST_STEPS='200'
```

Narrow scratch:

```powershell
$env:RUN_NAME='sampled_narrow_scratch_8000'
$env:STEPS='8000'
$env:INIT_ACTOR_PATH=''
$env:SAMPLED_DYNAMICS_LEVEL='narrow'
```

Medium fixed-init fine-tune:

```powershell
$env:RUN_NAME='sampled_medium_finetune_from_fixed_12000'
$env:STEPS='12000'
$env:INIT_ACTOR_PATH='runs/diff_pre_training/diag_h128prio_16000_10seed/checkpoints'
$env:SAMPLED_DYNAMICS_LEVEL='medium'
```

Medium scratch:

```powershell
$env:RUN_NAME='sampled_medium_scratch_12000'
$env:STEPS='12000'
$env:INIT_ACTOR_PATH=''
$env:SAMPLED_DYNAMICS_LEVEL='medium'
```

Broad exploratory fixed-init:

```powershell
$env:RUN_NAME='sampled_broad_finetune_from_fixed_12000'
$env:STEPS='12000'
$env:INIT_ACTOR_PATH='runs/diff_pre_training/diag_h128prio_16000_10seed/checkpoints'
$env:SAMPLED_DYNAMICS_LEVEL='broad'
$env:DYNAMICS_CURRICULUM='1'
```

Unless overridden above, sampled training/evaluation commands used train seeds `0 1 2 3 4`, eval seeds `10000 10001 10002`, eval models `euler l2f`, eval episodes `100`, eval horizon `128`, `--h128-prioritized-curriculum`, `--h128-schedule balanced_16000`, actor grad clip `50`, and v-strong weights `w_v=1.5`, `w_w=1.0`, `w_terminal_v=10`, `w_terminal_w=6`.

## Results

| run | sampled level | init | sampled Euler success | sampled L2F success | fixed Euler after run | fixed L2F after run | nonzero sampled both | invalid | rejection |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero-shot fixed checkpoint | narrow | fixed-pretrained | 0.348667 +/- 0.139564 | 0.327333 +/- 0.136941 | not run | not run | 5 / 5 | 0 | n/a |
| narrow fine-tune | narrow | fixed-pretrained | 0.416000 +/- 0.262876 | 0.375333 +/- 0.247347 | 0.331333 +/- 0.184338 | 0.285333 +/- 0.164986 | 4 / 5 | 0 | 0 |
| narrow scratch | narrow | scratch | 0.364000 +/- 0.180707 | 0.308667 +/- 0.170263 | 0.260000 +/- 0.139124 | 0.207333 +/- 0.127181 | 5 / 5 | 0 | 0 |
| medium fine-tune | medium | fixed-pretrained | 0.556000 +/- 0.099742 | 0.519333 +/- 0.113498 | 0.284000 +/- 0.022351 | 0.234667 +/- 0.015434 | 5 / 5 | 0 | 0 |
| medium scratch | medium | scratch | 0.406667 +/- 0.098161 | 0.346667 +/- 0.103966 | 0.208000 +/- 0.029784 | 0.168667 +/- 0.030955 | 5 / 5 | 0 | 0 |
| broad fine-tune | broad | fixed-pretrained | 0.276667 +/- 0.063070 | 0.226667 +/- 0.054324 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | 5 / 5 | 0 | 0.724003 |

Narrow fixed-init improved aggregate sampled success over narrow scratch by `+0.052` Euler and `+0.066666` L2F, but one fixed-init seed collapsed to zero L2F success.

Medium fixed-init improved aggregate sampled success over medium scratch by `+0.149333` Euler and `+0.172666` L2F. It also retained higher fixed-dynamics success after sampled fine-tuning. Medium fixed-init is the strongest current result.

Broad fixed-init remained nonzero on sampled evaluation, but it had high dynamics rejection and collapsed to zero fixed-dynamics success after fine-tuning. Broad sampled dynamics is not solved.

## Dynamics And Validity Stats

Training invalid rollouts, skipped actor steps, and NaN/Inf flags were zero/false for all narrow, medium, and broad sampled training runs.

Medium fixed-init dynamics proxies by seed:

| seed | mass | thrust-to-weight | torque-to-inertia | motor tau mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0299386 | 1.76522 | 470.467 | 0.177907 |
| 1 | 0.0286436 | 1.74392 | 500.648 | 0.155281 |
| 2 | 0.0307312 | 1.72101 | 397.047 | 0.141599 |
| 3 | 0.0301255 | 1.72383 | 370.549 | 0.156397 |
| 4 | 0.0335934 | 1.68227 | 438.715 | 0.143385 |

Broad fixed-init rejection rates by seed were `0.809524`, `0.8`, `0.5`, `0.818182`, and `0.692308`, with mean `0.724003`.

## Bridge Notes

Bridge notes for a future few-teacher comparison are in:

```text
runs/diff_pre_training/sampled_medium_finetune_from_fixed_12000/reports/teacher_cost_bridge_notes.md
```

The current gap is checkpoint format and post-training initialization wiring, not actor shape. The recommended next step is a controlled few-teacher comparison between scratch and diff-initialized post-training runs using identical teacher IDs, seeds, and update budgets.

## Decision

`MEDIUM_PROMISING`

The fixed checkpoint transfers to narrow zero-shot sampled evaluation, fixed-init sampled fine-tuning outperforms scratch in aggregate, and the medium fixed-init run is clearly stronger than medium scratch with zero invalid rollouts and zero rejection. This supports continuing to a bridge/few-teacher experiment.

The broad sampled domain remains unresolved because it requires heavy rejection filtering and destroys fixed-dynamics retention in this run.

