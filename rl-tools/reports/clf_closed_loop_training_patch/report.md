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
- Per-sample/per-rotor hover-relative action magnitude in the active CUDA loss path now uses an independent `action_hover_center[b,a]`.
  `initial_previous_action[b,a]` remains observation/history only and is no longer used as the action-magnitude center.
- Replay batch serialization now stores `action_hover_center` explicitly in schema 4.
  Older schema 3 replay batches are still readable; their hover centers are recomputed
  from mass, gravity, thrust curves, and action limits instead of falling back to
  `initial_previous_action`.
- Saturation diagnostics now use `weights.saturation_start` instead of a hard-coded `0.95` in the active CUDA diagnostics and parity path.
- H1000 forward-only gate and checkpoint rollback plumbing:
  - saves a candidate checkpoint at gate intervals
  - evaluates H500/H1000-style forward stability with configurable dynamics mode
  - saves the best passing checkpoint
  - rolls back to the best checkpoint after gate failure when one is available
- Minimal active failure-state replay plumbing:
  - gate eval can collect failure-preceding snapshots from forward trajectories
  - snapshots include physical state, RPM, previous action, true hover action center, GRU hidden, references, and dynamics parameters
  - training uses `--failure-replay-ratio` as a batch-slot mixed replay ratio: the first ratio of batch slots are initialized from replay while the rest continue normal persistent rollout
  - replay collection uses configurable position, velocity, attitude, and angular-velocity failure thresholds
  - optional response replay samples can now be collected at or after the trigger step
    instead of only before it:
    `--failure-replay-response-sampling --failure-replay-response-min N --failure-replay-response-max N`
- Velocity, angular-velocity, and attitude barriers are normalized Huber barriers with optional componentwise barrier-adjoint cap:
  - `excess = ReLU(norm_sq / safe_sq - 1)`
  - `L = w * Huber(excess, barrier_huber_delta)`
  - `--barrier-gradient-cap` caps only the direct barrier adjoint contribution before VJP.
- H1000 gate now saves a shadow-best checkpoint even when the strict gate fails, using:
  `10 * saturation + 0.002 * mean_max_omega + 0.0001 * max_omega - 0.5 * mean_first_failure_time_s`
  plus finite/NaN penalties.
- Replay-late H16 position-progress loss for multi-segment replay lanes:
  - `delta = (||p_H - p_ref||^2 - ||p_0 - p_ref||^2 + margin) / (eps + ||p_0 - p_ref||^2)`
  - `L = w_replay_recovery_progress * Huber(ReLU(delta))`
  - gradients are injected only into the current H16 segment endpoints, then propagated
    through the existing H16 VJP; segment boundaries remain detached.
- Training and eval CSVs include:
  - `velocity_barrier_loss`
  - `angular_velocity_barrier_loss`
  - `attitude_barrier_loss`
  - `replay_recovery_progress_loss`
- H1000 gate/eval diagnostics now record first-failure cause counts:
  - position
  - velocity
  - attitude
  - angular velocity
  - nonfinite
- Replay-local linear-velocity recovery loss:
  - `--w-replay-recovery-velocity`
  - active only on replay-late H16 segments
  - contributes to `replay_recovery_loss`
- Replay-local velocity-envelope barrier:
  - `--w-replay-recovery-velocity-barrier`
  - `--replay-recovery-velocity-safe`
  - active only on replay-late H16 segments
  - uses the same normalized Huber/capped-gradient barrier form as the global velocity
    barrier
  - logs `replay_recovery_velocity_barrier_loss`
- Training CSV additionally logs failure replay buffer size, added count, used count, and whether the current segment came from replay.

## Modified Files

- `include/rl_tools/rl/environments/l2f/diff_euler_rollout.h`
- `include/rl_tools/rl/environments/l2f/diff_euler_model.h`
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

The failure replay lifecycle was first corrected from episode-reset replay to segment replay.
The current implementation goes one step further and uses batch-mixed replay instead of
whole-batch replay. Replay sampling had been triggered only at persistent episode resets,
so a nominal 70% replay ratio could be diluted by normal 500-step persistent episodes.
Whole-batch segment replay fixed the ratio but over-weighted failure-state gradients.
The current code now injects replay at H16 segment granularity and only into a fixed
fraction of batch slots:

```text
normal persistent slots -> continue carried state/hidden
replay slots -> initialized from failure replay snapshot for this H16 segment
all slots -> one H16 update, then carry forward
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

The replay-chain coverage log was made explicit after a later audit showed that stdout
only reported the current row's segment index. Training CSV/stdout now also report:

```text
failure_replay_completed_chain_count
failure_replay_observed_max_segment_index
```

A short H16 fixed-local smoke with `--failure-replay-segments 8` verifies that the chain
does execute through the final replay segment while still detaching at each H16 boundary:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/chain_coverage_smoke/
```

| Metric | Value |
| --- | ---: |
| replay chain length | 8 |
| completed chain count | 307 |
| observed max segment index | 7 |
| carry position error | 0 |
| carry velocity error | 0 |
| carry rotation error | 0 |
| carry angular velocity error | 0 |
| carry RPM error | 0 |
| carry previous-action error | 0 |
| carry hidden error | 0 |

This confirms the intended replay structure:

```text
snapshot -> H16 update/detach -> carry -> ... -> segment 7
```

The last printed `failure_replay_max_segment_index` remains a current-row field and can be
less than `7` depending on the chain phase at program exit. The cumulative
`failure_replay_observed_max_segment_index` is the authoritative coverage field.

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

## Hover-Center, Huber-Barrier, Mixed-Replay Follow-Up

This follow-up addresses three suspected saturation/rate failure sources:

- `hover_relative_action_magnitude` no longer centers action magnitude around replay
  `previous_action`. The CUDA batch now carries `action_hover_center[B,4]`; replay
  snapshots preserve it; observations still use `previous_action`.
- Barriers are normalized Huber barriers with optional `--barrier-gradient-cap`.
- Failure replay is batch-mixed. `--failure-replay-ratio 0.30` means 30% of batch slots
  are replay initialized each H16 segment, not 30% of updates as whole-batch replay.

Validation evidence:

```text
reports/clf_closed_loop_training_patch/hover_huber_mixed/build_recheck.stdout
reports/clf_closed_loop_training_patch/hover_huber_mixed/build_cpu_recheck2.stdout
reports/clf_closed_loop_training_patch/hover_huber_mixed/validate_hover_huber_recheck.stdout
```

CPU/CUDA parity passed with `--hover-relative-action-magnitude`, Huber barrier weights,
and `--barrier-gradient-cap 1.0`:

```text
gpu_validation_passed=true
loss_close=true
action_gradient_close=true
hover_relative_action_magnitude_center=per_sample_action_hover_center
```

After the replay schema 4 fix, the same active parity path was re-run:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/validate_hover_huber_schema4.stdout
gpu_validation_passed=true
loss_close=true
action_gradient_close=true
hover_relative_action_magnitude_center=per_sample_action_hover_center
```

The schema write/read smoke also successfully wrote and loaded a replay file:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/schema4_replay_check/debug.stdout
stage9_replay_written=true
stage9_replay_loaded=true
```

The legacy stage9 CPU/GPU close gate remains false in that smoke because of the known
stage9 action-gradient mismatch; it is not a replay file read failure.

Fixed-local A-F matrix:

```text
reports/clf_closed_loop_training_patch/hover_huber_mixed/summary.csv
```

All runs used fixed dynamics, local initial conditions, H16 training, H1000 fixed gate
every 100 updates, batch size 8192, terminal loss scale 0, true hover action center,
batch-mixed replay, and shadow-best rollback.

| Run | Replay | Barrier mode | Failure time s | Mean max omega | Max omega | Saturation | NaN/Inf | Shadow score |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.30 | quadratic-like normalized | 0.669102 | 62.3167 | 88.2275 | 0.238945 | 0 | 2.18836 |
| B | 0.50 | quadratic-like normalized | 0.660859 | 55.8905 | 77.9976 | 0.263145 | 0 | 2.42060 |
| C | 0.30 | Huber + cap | 0.669570 | 61.3500 | 86.1900 | 0.237397 | 0 | 2.17051 |
| D | 0.50 | Huber + cap | 0.860742 | 182.528 | 231.807 | 0.289586 | 0 | 2.85372 |
| E | 0.50 | Huber + cap, lower omega weight | 0.664961 | 58.4990 | 73.7037 | 0.239036 | 0 | 2.18225 |
| F | 0.50 | Huber + cap, stronger smoothness | 0.860742 | 182.528 | 231.807 | 0.289587 | 0 | 2.85373 |

Interpretation:

- The true hover center reduces the severe old high-replay saturation signature
  (`0.368770`) to roughly `0.237-0.290`, but it does not yet reach the 5% guard.
- Huber barriers at the tested weights do not by themselves improve the final fixed-local
  shadow score versus quadratic-like normalized barriers.
- Increasing mixed replay from 30% to 50% can move failure time later in D/F, but the
  tradeoff is higher angular-rate growth and saturation.
- Lowering omega weight in E recovers lower saturation and max omega, but loses the later
  failure time.
- Stronger smoothness in F did not separate from D in this short fixed-local matrix.

Current recommendation: keep true hover action center and batch-mixed replay. Do not
increase replay ratio or barrier weight yet. The next fixed-local patch should target
lower saturation directly, likely with per-sample hover centers for sampled dynamics and
a more explicit cap on the barrier-to-action gradient path if saturation remains above
5% in fixed dynamics.
- Fixed-local H1000 is therefore not solved yet. The next pass should keep segment-level
  replay, but search for a saturation/rate tradeoff rather than proceeding to recovery or
  sampled dynamics.

## Multi-Segment H16 Replay Follow-Up

The replay path was extended from single-H16 replay to multi-segment replay lanes:

```text
failure replay snapshot
-> H16 forward/backward/update
-> detach
-> carry p/v/R/omega/rpm/previous_action/GRU hidden
-> repeat for --failure-replay-segments
-> reset lane from a new replay snapshot
```

Runtime option:

```text
--failure-replay-segments N
```

The default is `1`, preserving the previous single-segment replay behavior. Values such
as `4` and `8` keep replay slots active across consecutive H16 segments. Normal H16
training remains active in the non-replay batch slots, and gradients are still truncated
to the current H16 segment only.

Training logs now include:

```text
failure_replay_chain_length
failure_replay_episode_count
failure_replay_segment_count
failure_replay_mean_segment_index
failure_replay_max_segment_index
failure_replay_state_carry_enabled
failure_replay_hidden_carry_enabled
failure_replay_action_saturation_rate
failure_replay_carry_check_slots
failure_replay_carry_position_error
failure_replay_carry_velocity_error
failure_replay_carry_rotation_error
failure_replay_carry_angular_velocity_error
failure_replay_carry_rpm_error
failure_replay_carry_previous_action_error
failure_replay_carry_hidden_error
```

Validation evidence:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/
```

| Check | Result | Evidence |
| --- | --- | --- |
| build | PASS | CUDA target rebuilt successfully |
| CPU/CUDA parity | PASS | `validate_hover_huber_after_carry_check.stdout`: `loss_close=true`, `action_gradient_close=true` |
| single-segment compatibility | PASS | `smoke_segments_1`: `new_slots=active_slots=512`, segment index always 0 |
| multi-segment execution | PASS | `smoke_segments_4_carry_check`: active slots stay 512 while new slots are 0 for segment indices 1, 2, and 3 |
| carry continuity | PASS | carry errors for p/v/R/omega/rpm/previous_action/hidden are all 0 in the multi-segment smoke |

Fixed-local H1000 diagnostic:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/summary.csv
```

All runs used fixed dynamics, local initial conditions, H16 training, H1000 fixed
forward-only gates, batch size 8192, terminal loss scale 0, true hover action center,
Huber barriers, mixed replay, and no H32/H64/H128 curriculum.

| Run | Segments | Replay | Key change | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf | Carry errors |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `segments_4` | 4 | 0.30 | baseline multi-segment | 1.26547 | 6.12764e+06 | 8.9997e+08 | 0.316035 | 0 | 0 |
| `segments_8` | 8 | 0.30 | longer replay chain | 0.636172 | 42174.4 | 218209 | 0.504553 | 0 | 0 |
| `segments4_r0p10` | 4 | 0.10 | lower replay mix | 0.8208 | 8.1218e+10 | 1.19103e+13 | 0.495941 | 0 | 0 |
| `segments4_r0p30_sat2` | 4 | 0.30 | stronger saturation guard | 0.775455 | 1.0635e+12 | 3.50946e+13 | 0.53453 | 0 | 0 |

Gate first/last trend:

| Run | First failure time | Last failure time | First saturation | Last saturation | First mean max omega | Last mean max omega |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `segments_4` | 1.25379 | 1.26547 | 0.276636 | 0.316035 | 8525.33 | 6.12764e+06 |
| `segments_8` | 1.15719 | 0.636172 | 0.555718 | 0.504553 | 13577.8 | 42174.4 |
| `segments4_r0p10` | 0.894645 | 0.8208 | 0.540687 | 0.495941 | 5.74474e+12 | 8.1218e+10 |
| `segments4_r0p30_sat2` | 0.895028 | 0.775455 | 0.549652 | 0.53453 | 7.05748e+14 | 1.0635e+12 |

Interpretation:

- Multi-segment replay is now wired correctly: consecutive replay segments execute from
  the same lane, and p/v/R/omega/rpm/previous_action/hidden carry continuity is exact in
  the smoke checks.
- This patch keeps BPTT at H16. H1000 remains forward-only evaluation.
- The current fixed-local objective still does not pass the H1000 gate. `segments_4`
  gives the best failure time among these runs, but it increases H1000 angular-rate
  growth and saturation. `segments_8` lowers max omega relative to `segments_4`, but
  failure time gets worse and saturation rises above 50%.
- Lower replay ratio and stronger saturation guard did not fix the tradeoff. Both made
  long-horizon saturation and omega worse in this short matrix.
- Do not proceed to fixed-recovery, small dynamics, broad dynamics, or long production
  training from this result. The next patch should target the objective/saturation
  interaction inside fixed-local multi-segment replay.

### Multi-Segment Follow-Up Diagnostics

Additional fixed-local diagnostics were run to isolate whether the remaining failure is
caused by replay timing, barrier scale, action-gradient scale, or actor update scale.

Replay timing:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/summary_extended.csv
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/close_backtrack_summary.csv
```

| Run | Backtrack | Segments | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `segments_4` | 32-128 | 4 | 1.26547 | 6.12764e+06 | 8.9997e+08 | 0.316035 | 0 | best failure-time baseline, but omega explodes |
| `segments4_bt64_192` | 64-192 | 4 | 0.895854 | 6.42297e+08 | 2.54836e+10 | 0.46561 | 0 | earlier snapshots are worse |
| `segments4_bt128_256` | 128-256 | 4 | 0.67 | 1.62438e+12 | 1.62438e+12 | 0.5075 | 0 | much earlier snapshots are worse |
| `segments4_actorclip5_bt0_64` | 0-64 | 4 | 0.726484 | 9331.92 | 14734.6 | 0.704437 | 0 | closer snapshots over-drive actions |
| `segments4_actorclip5_bt0_32` | 0-32 | 4 | 0.723008 | 13155.2 | 14524.4 | 0.701665 | 0 | closest snapshots also over-drive actions |

Single versus multi-segment:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/single_vs_multi_summary.csv
```

| Run | Chain | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `segments_1_samecfg` | 1 | 0.737115 | 88992.9 | 2.20367e+06 | 0.364005 | 0 |
| `segments_4` | 4 | 1.26547 | 6.12764e+06 | 8.9997e+08 | 0.316035 | 0 |
| `segments_8` | 8 | 0.636172 | 42174.4 | 218209 | 0.504553 | 0 |

The single-segment replay baseline is not better. Four-segment replay improves failure
time and saturation relative to single-segment replay, but creates a larger long-horizon
angular-rate tail. Eight-segment replay reduces max omega relative to four segments but
fails earlier and saturates more.

Barrier/action-scale tradeoff:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/saturation_tradeoff_summary.csv
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/action_saturation_tradeoff_summary.csv
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/update_scale_summary.csv
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/actor_clip_summary.csv
```

| Run | Key change | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `segments4_omega0p01_cap0p1` | omega barrier 0.01, barrier cap 0.1 | 1.00579 | 8756.41 | 309399 | 0.3063 | 0 |
| `segments4_lowbarrier_actorclip5` | plus actor grad clip 5 | 0.924844 | 185.319 | 201.783 | 0.29303 | 0 |
| `segments4_lowbarrier_actorclip1` | actor grad clip 1 | 0.908437 | 339.214 | 1803.46 | 0.263406 | 0 |
| `segments4_lowbarrier_clip1` | action-gradient clip 1 | 0.789844 | 8212.72 | 46938.9 | 0.392271 | 0 |
| `segments4_lowbarrier_sat_early` | stronger/earlier saturation penalty | 0.950476 | 145011 | 1.43688e+06 | 0.483167 | 0 |
| `segments4_lowbarrier_lr1e5` | lower learning rate | 1.04676 | 2547.13 | 5613.57 | 0.538218 | 0 |

Best observed shadow checkpoint in this follow-up:

| Run | Step | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf | Shadow score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `segments4_lowbarrier_actorclip5` | 200 | 1.05926 | 212.791 | 275.242 | 0.225258 | 0 | 2.17606 |

Interpretation:

- Barrier-gradient cap is important. Reducing the cap from `1.0` to `0.1` and lowering
  the omega barrier weight reduces the omega blow-up by orders of magnitude.
- Actor-gradient clipping is the useful update-scale knob. `--actor-grad-clip 5` reduces
  H1000 omega substantially while keeping training finite.
- Action-gradient clipping does not address the current failure because raw injected
  action gradients are already tiny in these runs.
- Stronger or earlier saturation penalties make the H1000 saturation/failure tradeoff
  worse. Lower learning rate also worsens saturation.
- The best current fixed-local result is finite and much safer in angular rate, but it
  still fails the fixed-local gate: H1000 saturation remains far above 5%, and mean/max
  omega remain above the gate.
- Fixed-local is therefore still incomplete. Do not advance to fixed-recovery or sampled
  dynamics.

Action-triggered replay collection:

The first action-triggered attempt only changed `--failure-replay-action-abs 0.85`.
That did not change behavior because state thresholds still fired first. The replay
trigger was then made explicit:

```text
--failure-replay-trigger any|state|action
```

Default remains `any`, preserving old behavior. `action` mode uses action threshold
crossing as the replay trigger while H1000 gate metrics remain unchanged. Gate and
training logs now include trigger-source counts:

```text
failure_replay_state_triggers
failure_replay_action_triggers
failure_replay_nonfinite_triggers
failure_replay_last_state_trigger_count
failure_replay_last_action_trigger_count
failure_replay_last_nonfinite_trigger_count
```

The source counts report which conditions were true at the chosen replay trigger step.
In action mode, state conditions can still be true at that same step, but they no longer
decide whether the sample is collected.

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/action_triggered_replay_summary.csv
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/action_only_replay_tradeoff_summary.csv
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/formal_action_trigger_segments4/
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/formal_action_trigger_segments8/
```

| Run | Replay trigger | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf | Shadow score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `segments4_lowbarrier_actorclip5` | state thresholds | 0.924844 | 185.319 | 201.783 | 0.29303 | 0 | 2.8587 |
| `segments4_actorclip5_action_only_replay` | action-only | 1.08719 | 228.978 | 298.624 | 0.206317 | 0 | 2.0074 |
| best shadow: `segments4_actorclip5_action_only_replay` | action-only | 1.09008 | 241.683 | 328.43 | 0.194038 | 0 | 1.91155 |
| formal best: `formal_action_trigger_segments4` | `--failure-replay-trigger action` | 1.09008 | 241.682 | 328.433 | 0.194039 | 0 | 1.91156 |
| formal best: `formal_action_trigger_segments8` | `--failure-replay-trigger action` | 1.08531 | 250.58 | 335.699 | 0.18892 | 0 | 1.88127 |

Formal action-trigger source evidence:

| Run | Segment chain | Gate action triggers | Gate state true at trigger | Gate nonfinite triggers | Carry check |
| --- | ---: | ---: | ---: | ---: | --- |
| `formal_action_trigger_segments4` | 4 | 256 | 256 | 0 | p/v/R/omega/rpm/previous-action/hidden carry errors 0 |
| `formal_action_trigger_segments8` | 8 | 256 | 256 | 0 | carry slots present; no carry error regression observed |

Follow-up variants around action-only replay:

| Run | Key change | Best failure time s | Best mean max omega | Best max omega | Best saturation | Last saturation | NaN/Inf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `segments4_actorclip5_action_only_replay` | baseline action-only | 1.09008 | 241.683 | 328.43 | 0.194038 | 0.206317 | 0 |
| `actiononly_actorclip1` | actor grad clip 1 | 1.18133 | 128.197 | 329.733 | 0.293667 | 0.325622 | 0 |
| `actiononly_replay0p1` | replay ratio 0.10 | 1.08379 | 229.575 | 305.225 | 0.206107 | 0.224819 | 0 |
| `actiononly_sat2_start0p8` | stronger/earlier saturation penalty | 1.09008 | 241.683 | 328.428 | 0.201094 | 0.21348 | 0 |

Action-only replay is the first change in this series that improves both failure time
and H1000 saturation without introducing NaN/Inf. It still fails the fixed-local gate:
best saturation remains about 19.4%, not below 5%, and omega remains above the target.

Action-trigger threshold and replay-ratio diagnostics:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/action_trigger_threshold_summary.csv
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/action_trigger_ratio_summary.csv
```

| Run | Threshold | Replay | Best failure time s | Best mean max omega | Best max omega | Best saturation | Last saturation | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `formal_action_trigger_segments4` | 0.85 | 0.30 | 1.09008 | 241.682 | 328.433 | 0.194039 | 0.206318 | 0 |
| `formal_action_trigger_segments4_abs0p75` | 0.75 | 0.30 | 1.09043 | 242.724 | 330.645 | 0.192849 | 0.204234 | 0 |
| `formal_action_trigger_segments4_abs0p65` | 0.65 | 0.30 | 1.09031 | 245.406 | 334.49 | 0.190679 | 0.207352 | 0 |
| `formal_action_trigger_segments4_abs0p55` | 0.55 | 0.30 | 1.09285 | 238.689 | 330.074 | 0.194842 | 0.210326 | 0 |
| `formal_action_abs0p65_r0p10` | 0.65 | 0.10 | 1.08719 | 235.647 | 318.137 | 0.199638 | 0.229667 | 0 |
| `formal_action_abs0p65_r0p05` | 0.65 | 0.05 | 1.08266 | 222.432 | 294.402 | 0.212607 | 0.256107 | 0 |

Lowering the action trigger from `0.85` to `0.65` gives only a small saturation
improvement. Reducing replay ratio to `0.10` or `0.05` lowers omega somewhat, but it
worsens saturation and failure time. The desired property that replay ratio can be
reduced without losing stability is not met yet.

Outward/window strength diagnostics:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/formal_action_abs0p65_out0p1_window1/
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/formal_action_abs0p65_out0p1_window0p5/
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/formal_action_abs0p65_out0p5_window1/
```

All three keep H16 BPTT, four replay segments, action-trigger replay at `0.65`, and
fixed-local H1000 forward-only gates.

| Run | Window CLF | Outward | Last failure time s | Last mean max omega | Last max omega | Last saturation | H1000 NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `formal_action_trigger_segments4_abs0p65` | 1.0 | 0.25 | 1.09871 | 221.618 | 302.204 | 0.207352 | 0 |
| `formal_action_abs0p65_out0p1_window1` | 1.0 | 0.10 | 0.766667 | 220366 | 226438 | 0.505000 | 506 |
| `formal_action_abs0p65_out0p1_window0p5` | 0.5 | 0.10 | 0.787273 | 407357 | 1.22097e+06 | 0.500614 | 468 |
| `formal_action_abs0p65_out0p5_window1` | 1.0 | 0.50 | 0.857028 | 4.43342e+06 | 6.45598e+08 | 0.376776 | 88 |

This rejects a simple one-dimensional explanation. Reducing outward/window strength
removes too much long-horizon stabilization and reintroduces NaN/Inf. Increasing outward
strength also worsens angular-rate blow-up and saturation. The remaining saturation
problem should not be approached by only sweeping outward/window weights.

Hover-relative action magnitude was also increased:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_diag/action_magnitude_tradeoff_summary.csv
```

| Run | `--w-action-magnitude` | Best saturation | Last saturation | Last train action cost | Last hover-relative mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `segments4_actorclip5_action_only_replay` | 0.005 | 0.194038 | 0.206317 | 0.00139647 | 0.180678 |
| `actiononly_wmag0p05` | 0.05 | 0.194042 | 0.206325 | 0.0128649 | 0.180676 |
| `actiononly_wmag0p1` | 0.1 | 0.194046 | 0.206328 | 0.0256104 | 0.180675 |

Increasing hover-relative action magnitude cost by 10x or 20x does not materially reduce
H1000 saturation in this setup. The local H16 action-cost path is therefore not the
dominant remaining control knob.

Current best fixed-local candidate from this pass:

```text
--failure-replay-segments 4
--failure-replay-ratio 0.30
--failure-replay-action-abs 0.65
--failure-replay-trigger action
--failure-replay-position-norm 100000
--failure-replay-velocity-norm 100000
--failure-replay-omega-norm 100000
--failure-replay-attitude-error-deg 360
--barrier-gradient-cap 0.1
--w-omega-barrier 0.01
--actor-grad-clip 5
```

This is not a passing configuration. It is the best diagnostic candidate so far because
it is finite, uses true multi-segment replay, improves failure time, and reduces H1000
saturation compared with state-triggered replay. Action-triggered replay is now a
first-class logged mode with failure-source counts. The next useful patch should target
the remaining saturation with a closed-loop low-action recovery objective rather than
only increasing local H16 action penalties or sweeping outward/window strength.

## Replay-Recovery Late-Segment Objective

The active CUDA loss now has an optional replay-only recovery regularizer:

```text
--w-replay-recovery-action
--w-replay-recovery-saturation
--w-replay-recovery-angular-velocity
--replay-recovery-start-segment
```

It is applied only to batch slots that are currently inside a failure-replay lane and
whose zero-based replay segment index is at least `replay_recovery_start_segment`.
The default start segment is `1`, so the first H16 replay segment is still a direct
failure-state response segment, while the second and later H16 replay segments get
extra low-action, low-saturation, and low-angular-rate pressure. The objective still
backpropagates only through the current H16 segment; state, motor RPM, previous action,
true hover action center, and GRU hidden are carried into the next segment and detached.

The new loss component is logged as:

```text
replay_recovery_loss
```

Build and zero-new-weight parity:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/validate_zero_replay_recovery.stdout
```

```text
gpu_validation_passed=true
forward_close=true
loss_close=true
action_gradient_close=true
hover_relative_action_magnitude_center=per_sample_action_hover_center
```

Replay-recovery smoke:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/replay_recovery_smoke_interval3/
```

The smoke used `--failure-replay-segments 3`, `--failure-replay-ratio 0.50`, and
`--replay-recovery-start-segment 1`. It confirms that replay lanes can run as continuous
multi-segment H16 chains without H500 BPTT:

| Step | Active slots | New slots | Mean segment | Max segment | Replay-recovery loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 256 | 256 | 0 | 0 | 0 |
| 4 | 256 | 0 | 1 | 1 | 0.025059 |
| 5 | 256 | 0 | 2 | 2 | 0.059220 |
| 6 | 256 | 256 | 0 | 0 | 0 |
| 7 | 256 | 0 | 1 | 1 | 0.021879 |

Carry checks in the same smoke report zero error for position, velocity, rotation,
angular velocity, motor RPM, previous action, and hidden state.

Two short fixed-local 300-step diagnostics were then run from the current action-trigger
segments=4 baseline with true hover action centers and H1000 fixed forward-only gates:

| Run | Replay recovery weights | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf | Max segment | Nonzero loss rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_local_replay_recovery_300` | action `0.2`, saturation `3.0`, omega `0.5` | 1.01262 | 198.279 | 218.423 | 0.234208 | 0 | 3 | 148 |
| `fixed_local_replay_recovery_action_strong_300` | action `1.0`, saturation `10.0`, omega `0.2` | 1.01262 | 198.947 | 219.784 | 0.234899 | 0 | 3 | 148 |

Interpretation:

- The replay-recovery objective is wired and active on later replay segments.
- Continuous replay lanes carry state and hidden correctly across H16 segments.
- The new regularizer lowers angular-rate metrics relative to the earlier best, but it
  does not reduce H1000 saturation; both diagnostics remain far above the 5% guard.
- The fixed-local gate is still not solved. The next saturation-focused patch should
  alter the closed-loop action tradeoff more directly, not merely increase this
  late-segment action penalty.

## No-Rollback Gate Diagnostic

The H1000 gate now has an optional rollback switch:

```text
--disable-h1000-gate-rollback
```

Default behavior is unchanged: failed gates still roll back to the shadow-best checkpoint
when rollback is enabled. With rollback disabled, the gate still saves candidate and
shadow-best checkpoints, logs H1000 forward-only metrics, and collects failure replay
snapshots, but it does not restore actor/optimizer state or reset active replay chains
after every failed gate. This tests whether always rolling back on a failed-but-finite
gate freezes exploration before the policy can learn lower-rate, lower-saturation
closed-loop behavior.

Backtrack-window diagnostics were first run with action-triggered 4-segment replay:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/backtrack_window_diag/
```

| Run | Backtrack window | Best failure time s | Best mean max omega | Best max omega | Best saturation | H1000 NaN/Inf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `default_32_128` | 32-128 | 1.03260 | 348166 | 9.55734e+06 | 0.229775 | 412 |
| `early_64_192` | 64-192 | 1.02889 | 2.29466e+06 | 1.09581e+08 | 0.230611 | 404 |
| `early_128_256` | 128-256 | 1.03094 | 784937 | 3.05258e+07 | 0.234594 | 406 |

Earlier snapshots alone did not reduce saturation and reintroduced H1000 NaN/Inf in this
matrix. Carry checks still reported zero position/hidden error, so this is not evidence
of broken carry.

Replay-late action/saturation scaling was then increased:

| Run | Replay chain | Replay-recovery weights | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_local_replay_recovery_300` | 4 | action `0.2`, saturation `3.0`, omega `0.5` | 1.01262 | 198.279 | 218.423 | 0.234208 | 0 |
| `fixed_local_replay_recovery_action_strong_300` | 4 | action `1.0`, saturation `10.0`, omega `0.2` | 1.01262 | 198.947 | 219.784 | 0.234899 | 0 |
| `fixed_local_replay_recovery_very_strong_action_300` | 4 | action `10.0`, saturation `100.0`, omega `0.2` | 1.01262 | 198.995 | 219.915 | 0.234783 | 0 |
| `fixed_local_segments8_replay_recovery_300` | 8 | action `1.0`, saturation `10.0`, omega `0.2` | 1.02156 | 208.681 | 223.809 | 0.235479 | 0 |

The stronger replay-recovery terms lower angular-rate metrics relative to older
state-triggered replay, but they do not lower H1000 saturation. The training rows for
the 4-segment variants had zero replay action saturation, so the local replay action
regularizer is not seeing the same saturated closed-loop behavior that appears in H1000
eval.

No-rollback diagnostics:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_segments8_no_rollback_600/
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_segments8_no_rollback_1000/
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_segments8_no_rollback_sat2_start08_1000/
```

| Run | Rollback | Best step | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_local_segments8_norecovery_600` | enabled | 400 | 1.01980 | 182.664 | 214.001 | 0.240707 | 0 |
| `fixed_local_segments8_no_rollback_600` | disabled | 600 | 0.991289 | 115.340 | 127.724 | 0.218023 | 0 |
| `fixed_local_segments8_no_rollback_1000` | disabled | 800 | 0.850898 | 20.6626 | 21.5749 | 0.190481 | 0 |
| `fixed_local_segments8_no_rollback_sat2_start08_1000` | disabled | 800 | 0.851094 | 20.5370 | 21.4812 | 0.199108 | 0 |

No-rollback training confirms a useful separation:

- continuing past failed gates can drive mean/max omega down by an order of magnitude
  while remaining finite;
- the best no-rollback H1000 gate has no NaN/Inf and max omega near `21.6`, but it still
  misses the mean-omega gate and misses the action-saturation gate badly;
- failure time decreases from about `1.02s` to `0.85s`, and then `0.77s` by the last
  1000-step gate, so the policy is reducing angular rate partly by failing earlier;
- strengthening global saturation pressure to `--w-sat 2.0 --action-saturation-start 0.80`
  worsens saturation rather than improving it.

This means the fixed-local objective still has an unresolved recovery/saturation tradeoff.
The current best direction is no-rollback, 8-segment replay for rate containment, but it
is not a passing recovery configuration. The next change should make later replay
segments train on post-failure-time closed-loop states where saturation is actually
present, or add an explicit first-failure-time/progress term, rather than only increasing
local H16 action or saturation penalties.

## Replay-Late Progress Diagnostic

An explicit replay-late H16 progress objective was added to test whether later replay
segments need a direct "move closer to origin over this local segment" signal:

```text
--w-replay-recovery-progress W
--replay-recovery-progress-margin M
--replay-recovery-progress-eps EPS
--replay-recovery-progress-huber-delta D
```

Defaults are zero/no-op. The term activates only for replay lanes whose zero-based
segment index is at least `--replay-recovery-start-segment`. It does not join gradients
across replay segments; it only changes the current H16 segment endpoint adjoints.

Validation:

| Check | Result | Evidence |
| --- | --- | --- |
| build | PASS | `multi_segment_replay/replay_progress_patch/` |
| CPU/CUDA parity with no active replay | PASS | `validate_progress_no_replay.stdout`: `forward_close=true`, `loss_close=true`, `action_gradient_close=true` |
| replay-progress smoke | PASS | `smoke/train.csv` includes `replay_recovery_progress_loss`; segment 1 and 2 rows are nonzero |

Smoke evidence:

| Step | Replay segment | `replay_recovery_progress_loss` |
| ---: | ---: | ---: |
| 3 | 0 | 0 |
| 4 | 1 | 0.511309 |
| 5 | 2 | 1.76627 |
| 6 | 0 | 0 |
| 7 | 1 | 0.537246 |

This confirms that progress loss is connected to replay-late segments while segment 0
remains unaffected.

Fixed-local no-rollback diagnostics were repeated with 8 replay segments, 30% mixed
replay, action-triggered replay, and H1000 forward-only fixed gates:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_segments8_progress_600/
reports/clf_closed_loop_training_patch/multi_segment_replay/fixed_local_segments8_progress100_600/
```

| Run | Progress weight | Failure time s | Mean max omega | Max omega | H1000 saturation | NaN/Inf | Max progress loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_local_segments8_progress_600` | 1 | 0.991289 | 115.340 | 127.724 | 0.218023 | 0 | 0.00606178 |
| `fixed_local_segments8_progress100_600` | 100 | 0.991289 | 115.339 | 127.722 | 0.218024 | 0 | 0.606178 |

The simple endpoint position-progress objective is correctly wired, but it does not move
the H1000 gate metrics in this setup, even at `w=100`. It should therefore not be treated
as the missing recovery mechanism. The current evidence points back to the data
distribution: replay-late training segments still report zero replay action saturation,
while H1000 eval saturation remains around `0.22`. A future patch should collect or
construct replay states after the first high-saturation response, or use a forward-only
failure-time objective, instead of further scaling this endpoint progress term.

## Response-State Replay Timing

The action-triggered replay path previously used only backtracked samples:

```text
replay_step = failure_step - sampled_backtrack
```

With the default backtrack window this means high-action H1000 failures can fill the
replay buffer with states 32-128 steps before the high-saturation response. This matched
the observed failure mode: replay training rows often had zero replay action saturation
even while H1000 eval saturation was around `0.20`.

An optional response-sampling mode was added:

```text
--failure-replay-response-sampling
--failure-replay-response-probability P
--failure-replay-response-min N
--failure-replay-response-max N
```

When enabled:

```text
replay_step = min(horizon, failure_step + sampled_response_offset)
```

`--failure-replay-response-sampling` is pure response sampling. A nonzero
`--failure-replay-response-probability` mixes response samples with the original
backtracked samples inside the same replay buffer.

This keeps the gradient horizon at H16. It changes only which forward-eval state is saved
into the replay buffer.

Fixed-local 600-step diagnostics, all with 8-segment replay, 30% mixed replay,
no H1000 rollback, action-triggered replay, fixed dynamics, and H1000 forward-only gates:

| Run | Replay timing | Failure time s | Mean max omega | Max omega | H1000 saturation | Replay action saturation | NaN/Inf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_local_segments8_no_rollback_600` | backtrack 32-128 | 0.991289 | 115.340 | 127.724 | 0.218023 | 0 | 0 |
| `fixed_local_segments8_response0_600_fg` | trigger step | 0.995586 | 54.8284 | 64.8643 | 0.206092 | 0.250407 | 0 |
| `fixed_local_segments8_response0_replaysat_600` | trigger step + replay action/sat regularizer | 0.995117 | 54.6798 | 64.7419 | 0.206060 | 0.250305 | 0 |
| `fixed_local_segments8_response0_strong_replaysat_600` | trigger step + strong replay action/sat regularizer | 1.00074 | 56.5558 | 65.2450 | 0.204760 | 0.250674 | 0 |
| `fixed_local_segments8_response16_64_600` | trigger + 16-64 steps | 1.09816 | 61.5216 | 230.486 | 0.236729 | 0.315729 | 0 |
| `fixed_local_segments8_mixed_timing_600` | 50% backtrack 32-128, 50% trigger step | 0.923867 | 33.6889 | 42.7290 | 0.216927 | 0.034759 | 0 |

Interpretation:

- The replay timing hypothesis is confirmed: response replay makes replay training lanes
  see nonzero action saturation, while backtracked replay does not.
- Trigger-step response replay is a better local tradeoff than the old 32-128 backtrack
  window: it roughly halves H1000 mean/max omega and slightly lowers H1000 saturation.
- Adding replay-local action/saturation regularization does not materially improve the
  trigger-step response run; the resulting H1000 metrics are effectively unchanged.
- Very strong replay-local action/saturation regularization also does not solve the
  gate: saturation improves only slightly and angular-rate metrics worsen slightly.
- Sampling 16-64 steps after the trigger improves first-failure time, but it worsens
  saturation and max omega, so post-trigger offsets are useful diagnostically but not yet
  a passing fixed-local configuration.
- Mixed timing sharply lowers mean/max omega, but it shortens first-failure time and keeps
  H1000 saturation near `0.217`, so timing mix alone is not sufficient.

Fixed-local remains unresolved. The next useful direction is no longer just changing
replay timing; the replay buffer now covers both earlier states and high-saturation
response states, but the policy still trades lower angular rate for early failure or high
H1000 saturation. A forward-only failure-time objective or explicit low-saturation
closed-loop recovery target is likely needed. Directly scaling replay-local
saturation/action terms is not enough.

## First-Failure Cause And Velocity Recovery Diagnostic

The H1000 gate now records which state dimension trips the first local-window failure:

```text
first_failure_position_count
first_failure_velocity_count
first_failure_attitude_count
first_failure_angular_velocity_count
first_failure_nonfinite_count
```

A short smoke confirmed these fields are present in stdout and the H1000 gate CSV:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/first_failure_cause_smoke/
```

In that smoke, all 64 first failures were linear-velocity failures.

The same diagnostic was then applied to existing fixed-local H1000 checkpoints:

```text
reports/clf_closed_loop_training_patch/multi_segment_replay/first_failure_cause_eval/
```

| Checkpoint | Mean max position | Mean max velocity | Mean max omega | H1000 saturation | First-failure cause |
| --- | ---: | ---: | ---: | ---: | --- |
| `no_rollback_600` | 361.799 | 83.2327 | 115.558 | 0.199596 | velocity 256 / 256 |
| `response0_600` | 368.558 | 84.3466 | 54.6233 | 0.198496 | velocity 256 / 256 |
| `mixed_timing_600` | 362.408 | 81.6174 | 33.4217 | 0.189451 | velocity 256 / 256 |

This narrows the early-failure issue: angular-rate containment improved substantially, but
the local H1000 gate is now dominated by linear velocity leaving the local envelope.

Two velocity-focused diagnostics were run:

| Run | Key change | Failure time s | Mean max velocity | Mean max omega | H1000 saturation | First-failure cause | NaN/Inf |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `fixed_local_mixed_timing_vx5_600` | global `--w-v 4.0 --w-clf-velocity 4.0` | 0.871211 | 86.1568 | 27.2803 | 0.200439 | velocity 256 / 256 | 0 |
| `fixed_local_mixed_timing_replay_v_600` | replay-late `--w-replay-recovery-velocity 1.0` | 0.917187 | 83.1082 | 29.9042 | 0.214575 | velocity 256 / 256 | 0 |
| `fixed_local_velocity_trigger_replay_v_600` | velocity-triggered replay plus replay-late velocity loss | 0.833867 | not recorded | 226.354 | 0.271241 | velocity 256 / 256 | 0 |
| `fixed_local_mixed_timing_lower_rate_guard_600` | lower rate/attitude/barrier guard | 0.860156 | not recorded | 69.5297 | 0.241115 | velocity 256 / 256 | 0 |
| `fixed_local_mixed_timing_velocity_barrier_600` | global normalized velocity barrier, safe speed 2.0 | 0.919609 | 82.8882 | 30.6661 | 0.215736 | velocity 256 / 256 | 0 |
| `fixed_local_mixed_timing_replay_velocity_barrier_600` | replay-late velocity barrier, weight 0.05, safe speed 2.0 | 0.924336 | 82.3341 | 32.5270 | 0.220377 | velocity 256 / 256 | 0 |
| `fixed_local_mixed_timing_replay_velocity_barrier_w0p5_600` | replay-late velocity barrier, weight 0.5, safe speed 2.0 | 0.924062 | 82.3148 | 32.6110 | 0.220662 | velocity 256 / 256 | 0 |

Interpretation:

- Increasing global velocity weight lowers angular rate further but makes first failure
  earlier and does not reduce mean max velocity enough.
- Replay-local linear-velocity damping is connected (`replay_recovery_loss` becomes large)
  but also does not solve the velocity gate.
- Velocity-triggered replay makes the policy worse in this setup: it loses the high-action
  response distribution needed for rate recovery and pushes H1000 saturation upward.
- Lowering the rate/attitude/barrier guard also makes the fixed-local tradeoff worse:
  failure comes earlier, angular rate rises, and saturation increases.
- A global velocity barrier is numerically safe and slightly lowers angular rate versus
  mixed timing, but it does not move the velocity first-failure cause or the H1000
  saturation gate.
- The replay-late velocity-envelope barrier is wired and active: final
  `replay_recovery_velocity_barrier_loss` is `6.12067` for weight `0.05` and `61.1819`
  for weight `0.5`. Despite that, the H1000 metrics are essentially unchanged, so this
  local barrier is not a sufficient recovery signal by itself.
- The fixed-local blocker is no longer primarily NaN/Inf or angular-rate explosion. It is
  a low-saturation, low-linear-velocity closed-loop recovery problem.
- The next patch should target the closed-loop failure-time/velocity envelope directly,
  likely by collecting replay states around the velocity threshold crossing or by adding
  a forward-only gate-derived objective/priority, rather than continuing to scale local
  velocity or action penalties.

## Limits

- Failure replay is an in-memory training-loop buffer only; it is not serialized across runs.
- Stage/replay batch files now serialize true hover action centers in schema 4, but the
  online failure replay buffer itself remains in-memory only.
- Replay segments are sampled from failure-preceding snapshots and can now run as
  multi-segment H16 replay lanes. They intentionally still detach at every H16 boundary.
- The smoke gate used short/lenient settings to validate plumbing. It is not evidence that a production H1000 recovery policy is solved.
- Rollback has compile-time and pass-path smoke coverage; a forced fail-after-best runtime rollback scenario should still be exercised before a long production run.
