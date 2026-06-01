# Fixed SO(3) Stability Fix Follow-Up

Run root: `C:\Users\A\Desktop\rl-tools\diffphys\rl-tools\runs\diff_pre_training\fixed_vstrong_stabilityfix_10seed_audit`

This experiment only evaluates fixed-dynamics SO(3) differentiable pretraining stability. It does not prove teacher-cost reduction, RAPTOR teacher replacement, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Scope

The follow-up tested two targeted hypotheses for the remaining slow-curriculum failures:

- abrupt terminal velocity/angular-velocity loss pressure after horizon-curriculum transitions
- stale Adam state across horizon/state curriculum stage changes

No physics, SO(3) VJP, transition dynamics, rollout VJP, motor delay, thrust/torque model, actor observation, sampled dynamics, original L2F reward, SAC/TD3 core, RAPTOR teacher code, or HDF5 path was changed.

## Implemented Instrumentation And Knobs

Added training CSV diagnostics:

- `horizon_changed_flag`
- `state_curriculum_changed_flag`
- `steps_since_horizon_change`
- `steps_since_state_change`
- `effective_terminal_velocity_weight`
- `effective_terminal_angular_velocity_weight`
- `terminal_ramp_multiplier`
- `optimizer_reset_flag`

Added opt-in CLI knobs:

- `--terminal-ramp-after-horizon-change`
- `--terminal-ramp-min VALUE`
- `--terminal-ramp-steps N`
- `--terminal-ramp-terminal-only`
- `--reset-optimizer-on-curriculum-transition`

The following requested diagnostics are still logged as unavailable because the current model/optimizer containers do not expose them through a narrow local API: `actor_grad_max_abs`, `actor_update_norm`, `actor_parameter_norm`, `adam_m_norm`, `adam_v_norm`, `gru_hidden_norm_mean`, `gru_hidden_norm_max`.

## Build And Physics Gate

Build command:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build_diff -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS=ON
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build_diff --config Release --target foundation_policy_diff_physics_check foundation_policy_diff_pre_training -j2
```

Physics gate:

```powershell
build_diff\bin\Release\foundation_policy_diff_physics_check.exe --diff-model euler
```

Result: `overall=STRICT_PASS`.

## Focused Validation Commands

Common focused seed set: `1 2 3 4 6 7 9`.

Slow-curriculum baseline:

```powershell
$env:RUN_NAME='fixed_vstrong_slowcurr_focus_baseline'
$env:TRAIN_SEEDS='1 2 3 4 6 7 9'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='0'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
```

Terminal ramp:

```powershell
$env:RUN_NAME='fixed_vstrong_terminal_ramp_focus'
$env:TRAIN_SEEDS='1 2 3 4 6 7 9'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='1'
$env:TERMINAL_RAMP_MIN='0.25'
$env:TERMINAL_RAMP_STEPS='1000'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='0'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
```

Optimizer reset:

```powershell
$env:RUN_NAME='fixed_vstrong_optreset_focus'
$env:TRAIN_SEEDS='1 2 3 4 6 7 9'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='0'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='1'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
```

Ramp plus optimizer reset:

```powershell
$env:RUN_NAME='fixed_vstrong_ramp_optreset_focus'
$env:TRAIN_SEEDS='1 2 3 4 6 7 9'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='1'
$env:TERMINAL_RAMP_MIN='0.25'
$env:TERMINAL_RAMP_STEPS='1000'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='1'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
```

## Focused Results

| Config | Euler success mean/std | L2F success mean/std | Seeds beating Euler baseline | Seeds beating L2F baseline | Seeds beating both | Catastrophic seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| slow curriculum baseline | 0.230952 / 0.193684 | 0.149524 / 0.134677 | 4 / 7 | 4 / 7 | 4 / 7 | 1 |
| terminal v/w ramp | 0.213333 / 0.184950 | 0.136190 / 0.121347 | 4 / 7 | 3 / 7 | 3 / 7 | 1 |
| optimizer reset | 0.182381 / 0.195730 | 0.120476 / 0.134877 | 3 / 7 | 3 / 7 | 3 / 7 | 2 |
| ramp + reset | 0.163810 / 0.169072 | 0.093810 / 0.114496 | 4 / 7 | 2 / 7 | 2 / 7 | 1 |

Focused result: the new interventions did not improve the focused failure set. The slow-curriculum baseline remained the best focused configuration.

## Full Selected Audit Command

Because no new intervention beat the slow-curriculum baseline, the selected full audit reran the best focused configuration with no ramp and no optimizer reset, under the requested stability-fix run name:

```powershell
$env:RUN_NAME='fixed_vstrong_stabilityfix_10seed_audit'
$env:TRAIN_SEEDS='0 1 2 3 4 5 6 7 8 9'
$env:FORCE='1'
$env:SKIP_BUILD='1'
$env:SKIP_PHYSICS_GATE='1'
$env:TERMINAL_RAMP_AFTER_HORIZON_CHANGE='0'
$env:RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION='0'
$env:PYTHON='python'
& 'C:\Program Files\Git\bin\bash.exe' src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
```

## Full Before/After

| Metric | Original 500-step curriculum | 1000-step slow curriculum | Selected stability-fix audit |
| --- | ---: | ---: | ---: |
| Euler fixed mean/std success | 0.148 / 0.124474 | 0.233667 / 0.163105 | 0.233667 / 0.163105 |
| L2F fixed mean/std success | 0.127667 / 0.117266 | 0.150333 / 0.117042 | 0.150333 / 0.117042 |
| Seeds beating Euler baseline > 0.11 | 6 / 10 | 7 / 10 | 7 / 10 |
| Seeds beating L2F baseline > 0.09 | 6 / 10 | 7 / 10 | 7 / 10 |
| Seeds beating both baselines | 6 / 10 | 7 / 10 | 7 / 10 |
| Catastrophic zero-success seeds | 1 | 1 | 1 |
| Total skipped actor steps | 0 | 0 | 0 |
| Any training NaN/Inf | false | false | false |
| Total invalid rollouts | 0 | 0 | 0 |
| Audit decision | UNSTABLE_NEEDS_FIX | ACCEPTABLE_BUT_UNSTABLE | ACCEPTABLE_BUT_UNSTABLE |

## Selected Full Seed Table

| Seed | Euler success | L2F success | Euler mean p/v/w | L2F mean p/v/w | Classification |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | 0.283333 | 0.233333 | 1.71549 / 1.58377 / 2.03633 | 1.9198 / 1.90261 / 2.19517 | PASS_WEAK |
| 1 | 0.523333 | 0.35 | 1.00804 / 0.633199 / 2.36491 | 1.12115 / 1.03343 / 4.63023 | PASS_STRONG |
| 2 | 0 | 0 | 3.07659 / 3.93588 / 1.47965 | 3.22519 / 4.20695 / 1.76942 | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 3 | 0.52 | 0.363333 | 1.0951 / 0.821564 / 2.59627 | 1.29177 / 1.18877 / 3.83859 | PASS_STRONG |
| 4 | 0.09 | 0.056667 | 2.756 / 4.8223 / 5.73305 | 3.07651 / 5.38074 / 6.37715 | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 5 | 0.233333 | 0.12 | 1.32489 / 1.89772 / 6.13926 | 1.66267 / 2.94151 / 7.06945 | PASS_WEAK |
| 6 | 0.216667 | 0.096667 | 1.0247 / 1.32949 / 6.13414 | 1.20313 / 1.796 / 7.51311 | PASS_WEAK |
| 7 | 0.11 | 0.08 | 2.23168 / 2.66252 / 2.75256 | 2.50405 / 3.09924 / 3.0072 | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 8 | 0.203333 | 0.103333 | 1.85683 / 3.24811 / 5.48863 | 2.26605 / 4.37666 / 6.13669 | PASS_WEAK |
| 9 | 0.156667 | 0.10 | 1.97734 / 2.04784 / 1.41196 | 2.20466 / 2.44896 / 1.80198 | PASS_WEAK |

## Diagnostic Readout

The selected full run, with ramp/reset disabled, logged curriculum transitions at steps 1000, 2000, and 3000. For seed 2, the H=128 transition row was:

| step | horizon | horizon changed | state changed | steps since horizon change | steps since state change | terminal ramp multiplier | optimizer reset | effective terminal v/w |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 3000 | 128 | true | true | 0 | 0 | 1 | false | 10 / 6 |

In the terminal-ramp diagnostic run, the ramp correctly reduced terminal v/w weights to `2.5 / 1.5` at horizon changes and ramped back to `10 / 6` over 1000 steps. That behavior did not translate into better focused evaluation success.

## Decision

`NO_FIX_CONFIRMED` for terminal ramp and optimizer reset.

The current supported state remains the prior `PARTIAL_FIX`: the 1000-step slow curriculum improves the original fixed-dynamics audit from 6/10 to 7/10 seeds beating both old fixed first-order baselines, but it does not fully solve repeatability. The remaining failures are still most consistent with curriculum stability or objective/update-scale stress, not with a physics/VJP validity failure.

Recommended next targeted experiment: do not continue down the terminal-ramp or optimizer-reset path. Test a cleaner curriculum control next, such as freezing H=64 longer, extending the state curriculum further, or adding explicit stability-gated curriculum advancement.
