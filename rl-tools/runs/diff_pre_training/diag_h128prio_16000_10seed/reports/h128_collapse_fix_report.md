# Fixed SO(3) H128 Collapse Diagnostics and Fix

Run root: `runs/diff_pre_training/diag_h128prio_16000_10seed`

This report only covers fixed-dynamics SO(3) Euler differentiable pretraining stability. It does not prove teacher-cost reduction, RAPTOR teacher replacement, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Scope

The goal was to diagnose the remaining H128 collapse for train seeds 6 and 9 under the fixed-dynamics H128-prioritized curriculum, while keeping physics, VJP, motor delay, thrust/torque, actor observation, original L2F reward, sampled dynamics, SAC/TD3, teacher code, and HDF5 paths unchanged.

Old fixed first-order success baselines:

| Eval model | Success baseline |
| --- | ---: |
| Euler fixed | 0.11 |
| L2F fixed | 0.09 |

## Diagnostics Added

Training CSV logging now records:

- H128 schedule state: `h128_prioritized_curriculum_enabled`, `h128_schedule_name`, `curriculum_phase_name`, `phase_index`, `steps_in_current_phase`
- actor gradient finite flag: `actor_grad_nan_or_inf_flag`
- action loss terms: `loss_action_magnitude`, `loss_action_smoothness`, `loss_action_saturation`
- action gradient aliases and clipping status: `action_grad_norm_pre_clip`, `action_grad_norm_post_clip`, `action_grad_clip_scale`, `action_grad_clip_active_flag`
- action statistics: `action_abs_mean`, `action_delta_mean`, `action_delta_max`
- per-step batch training success estimate: `training_success_rate`
- unavailable low-level model/optimizer fields as `nan`: actor parameter/update/Adam norms and GRU hidden norms/abs values

GRU hidden diagnostics remain unavailable because the rollout GRU hidden tensor is not exposed through the current logging path without broader model instrumentation. Actor parameter/update and Adam state diagnostics also remain unavailable because the optimizer/model traversal does not currently expose those aggregate statistics.

The focused analysis script `src/foundation_policy/diff_pre_training/scripts/analyze_h128_failure_modes.py` writes:

- `reports/h128_failure_analysis.csv`
- `reports/h128_failure_analysis.md`

It compares seeds 6/9 against controls 2/3/7 by `H16_easy`, `H32_easy`, `H64_easy`, `H128_easy`, `H128_medium`, and `H128_full`.

## Physics Gate

The Euler physics gate was run before the baseline diagnostic and reported `overall=STRICT_PASS`.

Log:

```text
runs/diff_pre_training/diag_h128prio_baseline_12000_s23679/logs/physics_gate_euler.log
```

## Commands

Build:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build_diff -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS=ON
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build_diff --config Release --target foundation_policy_diff_physics_check foundation_policy_diff_pre_training -j2
```

Baseline H128-prioritized diagnostic:

```powershell
$env:RUN_NAME='diag_h128prio_baseline_12000_s23679'
$env:TRAIN_SEEDS='2 3 6 7 9'
$env:STEPS='12000'
$env:EVAL_SEEDS='10000 10001 10002'
$env:EVAL_MODELS='euler l2f'
$env:EVAL_EPISODES='100'
$env:EVAL_HORIZON='128'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='0'
$env:H128_PRIORITIZED_CURRICULUM='1'
$env:H128_SCHEDULE='short_warmup_12000'
$env:SUCCESS_GATED_CURRICULUM='0'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='0'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='0'
$env:DIAGNOSTIC_LOG_DETAIL='1'
$env:DIAGNOSTIC_LOG_EVERY='20'
$env:DIAGNOSTIC_LOG_FIRST_STEPS='200'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
python src\foundation_policy\diff_pre_training\scripts\analyze_h128_failure_modes.py --run-root runs\diff_pre_training\diag_h128prio_baseline_12000_s23679
```

Longer H128 schedule:

```powershell
$env:RUN_NAME='diag_h128prio_16000_s23679'
$env:TRAIN_SEEDS='2 3 6 7 9'
$env:STEPS='16000'
$env:EVAL_SEEDS='10000 10001 10002'
$env:EVAL_MODELS='euler l2f'
$env:EVAL_EPISODES='100'
$env:EVAL_HORIZON='128'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:H128_PRIORITIZED_CURRICULUM='1'
$env:H128_SCHEDULE='balanced_16000'
$env:SUCCESS_GATED_CURRICULUM='0'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='0'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='0'
$env:DIAGNOSTIC_LOG_DETAIL='1'
$env:DIAGNOSTIC_LOG_EVERY='20'
$env:DIAGNOSTIC_LOG_FIRST_STEPS='200'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
python src\foundation_policy\diff_pre_training\scripts\analyze_h128_failure_modes.py --run-root runs\diff_pre_training\diag_h128prio_16000_s23679
```

Actor-gradient clip 20:

```powershell
$env:RUN_NAME='diag_h128prio_12000_actorclip20_s23679'
$env:TRAIN_SEEDS='2 3 6 7 9'
$env:STEPS='12000'
$env:EVAL_SEEDS='10000 10001 10002'
$env:EVAL_MODELS='euler l2f'
$env:EVAL_EPISODES='100'
$env:EVAL_HORIZON='128'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:H128_PRIORITIZED_CURRICULUM='1'
$env:H128_SCHEDULE='short_warmup_12000'
$env:SUCCESS_GATED_CURRICULUM='0'
$env:ACTOR_GRAD_CLIP='20'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='0'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='0'
$env:DIAGNOSTIC_LOG_DETAIL='1'
$env:DIAGNOSTIC_LOG_EVERY='20'
$env:DIAGNOSTIC_LOG_FIRST_STEPS='200'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
python src\foundation_policy\diff_pre_training\scripts\analyze_h128_failure_modes.py --run-root runs\diff_pre_training\diag_h128prio_12000_actorclip20_s23679
```

Action-gradient clip 10:

```powershell
$env:RUN_NAME='diag_h128prio_12000_actionclip10_s23679'
$env:TRAIN_SEEDS='2 3 6 7 9'
$env:STEPS='12000'
$env:EVAL_SEEDS='10000 10001 10002'
$env:EVAL_MODELS='euler l2f'
$env:EVAL_EPISODES='100'
$env:EVAL_HORIZON='128'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:H128_PRIORITIZED_CURRICULUM='1'
$env:H128_SCHEDULE='short_warmup_12000'
$env:SUCCESS_GATED_CURRICULUM='0'
$env:ACTION_GRAD_CLIP='10'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='0'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='0'
$env:DIAGNOSTIC_LOG_DETAIL='1'
$env:DIAGNOSTIC_LOG_EVERY='20'
$env:DIAGNOSTIC_LOG_FIRST_STEPS='200'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
python src\foundation_policy\diff_pre_training\scripts\analyze_h128_failure_modes.py --run-root runs\diff_pre_training\diag_h128prio_12000_actionclip10_s23679
```

Full focused validation:

```powershell
$env:RUN_NAME='diag_h128prio_16000_s1234679'
$env:TRAIN_SEEDS='1 2 3 4 6 7 9'
$env:STEPS='16000'
$env:EVAL_SEEDS='10000 10001 10002'
$env:EVAL_MODELS='euler l2f'
$env:EVAL_EPISODES='100'
$env:EVAL_HORIZON='128'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:H128_PRIORITIZED_CURRICULUM='1'
$env:H128_SCHEDULE='balanced_16000'
$env:SUCCESS_GATED_CURRICULUM='0'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='0'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='0'
$env:DIAGNOSTIC_LOG_DETAIL='1'
$env:DIAGNOSTIC_LOG_EVERY='20'
$env:DIAGNOSTIC_LOG_FIRST_STEPS='200'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
python src\foundation_policy\diff_pre_training\scripts\analyze_h128_failure_modes.py --run-root runs\diff_pre_training\diag_h128prio_16000_s1234679
```

Full 10-seed audit:

```powershell
$env:RUN_NAME='diag_h128prio_16000_10seed'
$env:TRAIN_SEEDS='0 1 2 3 4 5 6 7 8 9'
$env:STEPS='16000'
$env:EVAL_SEEDS='10000 10001 10002'
$env:EVAL_MODELS='euler l2f'
$env:EVAL_EPISODES='100'
$env:EVAL_HORIZON='128'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:H128_PRIORITIZED_CURRICULUM='1'
$env:H128_SCHEDULE='balanced_16000'
$env:SUCCESS_GATED_CURRICULUM='0'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='0'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='0'
$env:DIAGNOSTIC_LOG_DETAIL='1'
$env:DIAGNOSTIC_LOG_EVERY='20'
$env:DIAGNOSTIC_LOG_FIRST_STEPS='200'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
python src\foundation_policy\diff_pre_training\scripts\analyze_h128_failure_modes.py --run-root runs\diff_pre_training\diag_h128prio_16000_10seed
```

## Focused Results

| Run | Seeds | Euler mean/std success | L2F mean/std success | Seeds beating both baselines | Catastrophic seeds | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `diag_h128prio_baseline_12000_s23679` | 2 3 6 7 9 | 0.342 / 0.258720 | 0.316667 / 0.236390 | 3 / 5 | 1 | seeds 6/9 failed |
| `diag_h128prio_16000_s23679` | 2 3 6 7 9 | 0.472 / 0.070193 | 0.441333 / 0.070415 | 5 / 5 | 0 | pass |
| `diag_h128prio_12000_actorclip20_s23679` | 2 3 6 7 9 | 0.231333 / 0.231013 | 0.206 / 0.224117 | 2 / 5 | 1 | rejected |
| `diag_h128prio_12000_actionclip10_s23679` | 2 3 6 7 9 | 0.286667 / 0.136480 | 0.208667 / 0.144969 | 4 / 5 | 0 | partial, rejected as best fix |

Actor-grad clip 10 was not run because clip 20 regressed controls and did not rescue seeds 6/9. Action-grad clip 5 was not run because clip 10 remained below the 16000 schedule and still failed seed 6. Actor LR scale and initialization changes were not run because the longer schedule solved the focused failure set without optimizer or initialization changes.

## Baseline Failure Analysis

In the 12000-step H128-prioritized baseline, seeds 6/9 did not diverge before H128. Their median final pre-H128 phase loss was 208.473, while their median final H128 phase loss was 1550.03.

H128 diagnostics:

| Metric | Seeds 6/9 | Controls 2/3/7 |
| --- | ---: | ---: |
| Median actor grad pre-clip | 221652 | 123306 |
| Actor clip fraction | 1 | 1 |
| Action clip fraction | 0 | 0 |
| Median action abs / saturation | 0.603686 / 0 | 0.543888 / 0 |
| Terminal velocity / position loss fraction | 0.354049 / 0.295067 | 0.343854 / 0.355481 |
| H128_full loss slope | -0.624470 | n/a |
| H128_full final L2F success median | 0.03 | n/a |

The failure was concentrated after H128. Both failed and control seeds had actor gradients clipped essentially every step, so actor clipping alone did not distinguish pass/fail. Action-gradient clipping was disabled in the baseline. Seed 6 also showed high late H128 action saturation.

## Seed 6/9 H128 Full Diagnostics

| Run | Seed | Final H128 loss | Loss slope | Final p/v/w | Actor grad median | Action grad median | Action clip frac | Action abs | Action sat | Eval Euler/L2F |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline 12000 | 6 | 720.87 | -0.951735 | 1.98262 / 1.72767 / 2.27642 | 454218 | 22686.8 | 0 | 0.886073 | 0.228027 | 0.04 / 0.056667 |
| baseline 12000 | 9 | 1258.18 | -0.297205 | 2.62143 / 3.17234 / 3.20231 | 222034 | 10074.4 | 0 | 0.684314 | 0 | 0.013333 / 0.003333 |
| balanced 16000 | 6 | 305.531 | -0.060573 | 1.63678 / 1.11268 / 1.03602 | 171135 | 8621.67 | 0 | 0.732659 | 0 | 0.373333 / 0.336667 |
| balanced 16000 | 9 | 166.443 | -0.015083 | 1.10511 / 0.651759 / 1.26722 | 135331 | 6070.81 | 0 | 0.705763 | 0 | 0.466667 / 0.45 |
| actor clip 20 | 6 | 1261.32 | -0.909000 | 3.19169 / 2.13409 / 1.75873 | 405115 | 20588.9 | 0 | 0.908551 | 0.351562 | 0.026667 / 0.026667 |
| actor clip 20 | 9 | 1475.06 | -0.273687 | 2.66489 / 4.1445 / 2.49027 | 225902 | 10162.3 | 0 | 0.684611 | 0 | 0.006667 / 0.003333 |
| action clip 10 | 6 | 911.557 | -0.965753 | 2.23204 / 2.1449 / 2.41173 | 207.931 | 22285.6 | 1 | 0.899826 | 0.254883 | 0.063333 / 0.036667 |
| action clip 10 | 9 | 274.518 | -0.372274 | 1.5129 / 0.982139 / 1.30563 | 227.242 | 11809.5 | 1 | 0.684753 | 0 | 0.276667 / 0.226667 |

The balanced 16000 schedule reduced seed 6/9 late H128 loss and recovered both seeds above both baselines. Actor clip 20 worsened the failed seeds and regressed seed 3. Action clip 10 rescued seed 9 but not seed 6 and reduced control strength, so it was not selected.

## Validation

Full focused set with the selected `balanced_16000` schedule:

| Seed set | Euler mean/std success | L2F mean/std success | Seeds beating both baselines | Catastrophic seeds |
| --- | ---: | ---: | ---: | ---: |
| 1 2 3 4 6 7 9 | 0.410476 / 0.123679 | 0.374286 / 0.124871 | 7 / 7 | 0 |

Full 10-seed audit with the selected `balanced_16000` schedule:

| Metric | Value |
| --- | ---: |
| Completed train seeds | 10 / 10 |
| Completed eval files | 60 / 60 |
| Euler fixed mean/std success | 0.443 / 0.129017 |
| L2F fixed mean/std success | 0.400333 / 0.122651 |
| Seeds beating Euler baseline > 0.11 | 10 / 10 |
| Seeds beating L2F baseline > 0.09 | 10 / 10 |
| Seeds beating both baselines | 10 / 10 |
| Catastrophic zero-success seeds | 0 |
| Total skipped actor steps | 0 |
| Any training NaN/Inf | false |
| Mean invalid_or_nan_rate | 0 |

Full 10-seed per-seed success:

| Seed | Euler success | L2F success | Result |
| ---: | ---: | ---: | --- |
| 0 | 0.376667 | 0.336667 | PASS_STRONG |
| 1 | 0.346667 | 0.26 | PASS_STRONG |
| 2 | 0.42 | 0.39 | PASS_STRONG |
| 3 | 0.55 | 0.503333 | PASS_STRONG |
| 4 | 0.166667 | 0.153333 | PASS_WEAK |
| 5 | 0.636667 | 0.556667 | PASS_STRONG |
| 6 | 0.373333 | 0.336667 | PASS_STRONG |
| 7 | 0.55 | 0.526667 | PASS_STRONG |
| 8 | 0.543333 | 0.49 | PASS_STRONG |
| 9 | 0.466667 | 0.45 | PASS_STRONG |

## Best Fix

Selected fix: use the H128-prioritized `balanced_16000` schedule:

```text
0-999:      H16_easy
1000-1999:  H32_easy
2000-2999:  H64_easy
3000-5999:  H128_easy
6000-8999:  H128_medium
9000-15999: H128_full
```

Reason: it was the only tested intervention that recovered both seeds 6 and 9 while preserving strong control-seed performance, removing catastrophic zero-success cases, keeping invalid/NaN/skipped instability at zero, and passing both the 7-seed and full 10-seed validations.

Most likely explanation: the previous 12000-step schedule was not giving the hard seeds enough useful H128 adaptation time. The longer balanced schedule lowers late-H128 losses and gradient magnitudes for seeds 6/9 without changing the physics, loss, actor observation, or legacy RL paths.

## Final Decision

`FIX_CONFIRMED`

The full 10-seed audit reached 10/10 seeds beating both old fixed first-order success baselines, with zero catastrophic seeds, zero skipped actor steps, zero invalid rollouts, and no NaN/Inf training flags.

## Next Recommended Action

Use `--h128-prioritized-curriculum --h128-schedule balanced_16000` as the fixed-dynamics stability candidate for the next fixed-only experiments. Further work should test whether this stability carries into separate teacher-cost or teacher-baseline comparisons, but those claims are not established by this audit.
