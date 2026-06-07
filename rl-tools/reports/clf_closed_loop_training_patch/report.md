# CLF Closed-Loop Training Plumbing Patch

## Scope

This patch refines the active CUDA EULER differentiable-physics objective plumbing for fixed-H16 origin-recovery training. It does not add a residual allocator, geometric controller, physical-parameter actor inputs, or long training campaign.

## Implemented

- Normalized H16 window-CLF objective:
  - `delta_raw = V_H - rho^H * V_0`
  - `delta_norm = delta_raw / (eps + V_0)`
  - `L = w_window_clf * Huber(ReLU(delta_norm))`
- Conservative default `clf_alpha = 0.2` for CLI/GPU runtime defaults. Existing behavior remains unchanged when CLF weights are zero.
- Lyapunov energy cross terms:
  - translational `c_pv * e_p^T e_v` with `--clf-pv-beta`
  - rotational `c_Rw * e_R^T omega` with `--clf-rw-beta`
- Velocity, angular-velocity, and attitude barrier losses.
- CUDA/CPU EULER rollout loss parity for the new objective terms.
- Per-sample/per-rotor hover-relative action magnitude in the active CUDA loss path, using `initial_previous_action[b,a]` instead of a scalar copied from the first sample.
- Saturation diagnostics now use `weights.saturation_start` instead of a hard-coded `0.95` in the active CUDA diagnostics and parity path.
- H1000 forward-only gate and checkpoint rollback plumbing:
  - saves a candidate checkpoint at gate intervals
  - evaluates H500/H1000-style forward stability with configurable dynamics mode
  - saves the best passing checkpoint
  - rolls back to the best checkpoint after gate failure when one is available
- Minimal active failure-state replay plumbing:
  - gate eval can collect failure-preceding snapshots from forward trajectories
  - snapshots include physical state, RPM, previous action, GRU hidden, references, and dynamics parameters
  - training reset can sample replay snapshots by `--failure-replay-ratio` and run them as one H16 segment
  - replay collection uses configurable position, velocity, attitude, and angular-velocity failure thresholds
- Training and eval CSVs include:
  - `velocity_barrier_loss`
  - `angular_velocity_barrier_loss`
  - `attitude_barrier_loss`
- Training CSV additionally logs failure replay buffer size, added count, used count, and whether the current segment came from replay.

## Modified Files

- `include/rl_tools/rl/environments/l2f/diff_euler_rollout.h`
- `src/foundation_policy/diff_pre_training/cli_options.h`
- `src/foundation_policy/diff_pre_training/cuda_main.cu`
- `src/foundation_policy/diff_pre_training/eval_utils.h`
- `src/foundation_policy/diff_pre_training/gpu_rollout.cu`
- `src/foundation_policy/diff_pre_training/gpu_rollout.h`
- `src/foundation_policy/diff_pre_training/logging_utils.h`
- `src/foundation_policy/diff_pre_training/main.cpp`

## Validation

Build passed:

```bash
cmake --build /tmp/raptor_stage96_cuda_build \
  --target foundation_policy_diff_physics_check foundation_policy_diff_pre_training_cuda foundation_policy_diff_pre_training -j2
```

CPU/CUDA parity passed with zero new weights:

```text
reports/clf_closed_loop_training_patch/validate_zero.stdout
forward_close=true
loss_close=true
action_gradient_close=true
```

CPU/CUDA parity passed with normalized window CLF, cross terms, barriers, hover-relative action, and `--action-saturation-start 0.85`:

```text
reports/clf_closed_loop_training_patch/validate_nonzero.stdout
forward_close=true
loss_close=true
action_gradient_close=true
```

One-step CUDA fixed-H16 training smoke passed:

```text
reports/clf_closed_loop_training_patch/train_smoke_1.stdout
gpu_full_training_passed=true
gpu_full_training_finite=true
gpu_full_training_nan_inf_count=0
```

H16 fixed eval smoke passed:

```text
reports/clf_closed_loop_training_patch/eval_smoke_h16.stdout
gpu_eval_finite=true
gpu_eval_nan_inf_count=0
gpu_eval_action_saturation_rate=0
```

CSV headers were checked and include the new barrier loss columns in both training and eval outputs.

H1000 gate smoke passed with lenient H16 gate settings:

```text
reports/h1000_gate_smoke/train.stdout
gpu_full_training_h1000_gate_enabled=true
gpu_full_training_h1000_gate_eval_count=2
gpu_full_training_h1000_gate_pass_count=2
gpu_full_training_h1000_gate_rollback_count=0
```

The gate CSV recorded two passing gate evaluations:

```text
reports/h1000_gate_smoke/gate.csv
step 1: passed=true, rolled_back=false
step 2: passed=true, rolled_back=false
```

Failure replay smoke passed with deliberately strict gate/failure thresholds:

```text
reports/failure_replay_smoke/train.stdout
gpu_full_training_passed=true
gpu_full_training_finite=true
gpu_full_training_nan_inf_count=0
gpu_full_training_h1000_gate_eval_count=3
gpu_full_training_h1000_gate_fail_count=3
gpu_full_training_failure_replay_enabled=true
gpu_full_training_failure_replay_buffer_size=48
gpu_full_training_failure_replay_added_count=48
gpu_full_training_failure_replay_used_count=2
```

The training CSV confirms replay sampling:

```text
reports/failure_replay_smoke/train.csv
failure_replay_last_episode=1 on rows 1 and 2
```

## Follow-Up: Gate Failure Time And Segment Replay

The H1000 gate now records first-failure timing:

```text
mean_first_failure_time_s
gpu_full_training_h1000_gate_last_mean_first_failure_time_s
```

This was added because the closed-loop training goal needs to show whether failure replay
pushes long-horizon failures later in time, not only whether the final H1000 gate passes.

The failure replay lifecycle was also corrected. Replay sampling had been triggered only
at persistent episode resets, so a nominal 70% replay ratio could be diluted by normal
500-step persistent episodes. Replay snapshots are now injected at H16 segment granularity:

```text
normal segment -> optionally force next segment from replay
replay segment -> exactly one H16 update -> reset
```

The segment-ratio smoke used strict failure thresholds and `--failure-replay-ratio 0.70`.
It collected 160 replay snapshots and used 54 replay segments in 100 updates, confirming
that replay now operates at the intended H16 segment scale:

```text
reports/clf_closed_loop_training_patch/failure_replay_segment_ratio_smoke/train.stdout
gpu_full_training_failure_replay_buffer_size=160
gpu_full_training_failure_replay_added_count=160
gpu_full_training_failure_replay_used_count=54
```

## Fixed-Local Diagnostics

All runs below used fixed dynamics, local initial conditions, H16 training, H1000
forward-only gates every 100 updates, terminal loss scale 0, batch size 8192, and no
H32/H64/H128 curriculum.

| Run | Replay ratio | Key change | Last failure time s | Last mean max omega | Last max omega | Last saturation | NaN/Inf | Replay used |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_local_replay_1000` | 0.30 episode-level | baseline closed-loop plumbing | not logged | 701739 | 3.5226e+07 | 0.157142 | 150 | 15 |
| `fixed_local_replay_rate_guard_600` | 0.70 episode-level | stronger rate/barriers | 0.900781 | 2558.97 | 6155.43 | 0.169002 | 0 | 26 |
| `fixed_local_segment_replay_rate_guard_600` | 0.70 segment-level | corrected replay sampling | 1.15305 | 2357.56 | 2826.35 | 0.704053 | 0 | 353 |
| `fixed_local_segment_replay_early_satguard_600` | 0.30 segment-level | earlier replay, stronger saturation guard | 0.953984 | 5855.78 | 7690.66 | 0.407635 | 0 | 152 |
| `fixed_local_segment_replay_low_satclip_600` | 0.10 segment-level | action-gradient clip and strongest saturation guard | 0.777025 | 3392.17 | 349170 | 0.053405 | 14 | 45 |
| `fixed_local_segment_replay_high_satguard_600` | 0.70 segment-level | high replay with strongest saturation guard | 1.34281 | 406.867 | 1706.96 | 0.368770 | 0 | 358 |
| `fixed_local_segment_replay_mid_satguard_600` | 0.50 segment-level | medium replay with strongest saturation guard | 1.07457 | 242905 | 7.69276e+06 | 0.277478 | 164 | 156 |

Current interpretation:

- Segment-level replay is now strong enough to move the failure time later.
- High replay plus stronger saturation guard gives the best angular-rate containment so far:
  `mean_first_failure_time_s=1.34281`, `mean_max_omega=406.867`, and no NaN/Inf.
- The same high-replay run still fails the action gate badly with saturation `0.368770`.
- Clipping action gradients and reducing replay can lower saturation to roughly the 5% guard,
  but it makes failures earlier and reintroduces NaN/Inf.
- Fixed-local H1000 is therefore not solved yet. The next pass should keep segment-level
  replay, but search for a saturation/rate tradeoff rather than proceeding to recovery or
  sampled dynamics.

## Limits

- Failure replay is an in-memory training-loop buffer only; it is not serialized across runs.
- Replay segments are single H16 training windows sampled from failure-preceding snapshots. They intentionally do not implement a persistent multi-segment replay episode.
- The smoke gate used short/lenient settings to validate plumbing. It is not evidence that a production H1000 recovery policy is solved.
- Rollback has compile-time and pass-path smoke coverage; a forced fail-after-best runtime rollback scenario should still be exercised before a long production run.
