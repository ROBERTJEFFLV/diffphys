# Persistent H16 Long-Horizon Recovery Diagnosis

## Scope

This report covers the fixed-H16 CUDA RDAC origin-recovery path. Physics BPTT remains fixed at H16. No H32/H64/H128 training or horizon curriculum was run. The GRU actor still outputs exactly four normalized motor commands, non-fixed trajectories remain eval/deployment setpoint shifting only, and metrics remain RAPTOR-style costs/returns in the active train/eval outputs.

## Code Changes

Persistent H16 truncated BPTT is implemented in:

- `src/foundation_policy/diff_pre_training/gpu_rollout.h`
- `src/foundation_policy/diff_pre_training/gpu_rollout.cu`
- `src/foundation_policy/diff_pre_training/cuda_main.cu`

The full-training path now supports persistent 500-step episodes:

```text
reset episode
roll out latest H16 segment
compute H16 loss/VJP
inject dL/du_t into actor outputs
run GRU BPTT only over latest H16
Adam update
detach
carry p/v/R/omega/rpm/previous_action/GRU hidden into next segment
reset only after reaching 500 steps or invalid/NaN
```

The default active mode is persistent. The CLI also supports:

```text
--persistent-episode-training
--disable-persistent-episode-training
--fresh-h16-batches
```

GRU hidden persistence is implemented through a detached per-batch `initial_hidden` buffer:

- `reset_segment_initial_hidden_kernel`
- `store_segment_final_hidden_kernel`
- `rdac_actor_forward_step_kernel(..., use_persistent_initial_hidden=true)`
- `rdac_actor_backward_step_kernel(..., use_persistent_initial_hidden=true)`

Physical state persistence is implemented through:

- `carry_segment_final_state_kernel`

The carried state includes position, velocity, rotation, angular velocity, motor rpm, and previous action.

## Logging

Training CSV now includes persistent metadata:

```text
persistent_episode_training
persistent_episode_step
segment_start
segment_end
episode_reset_count
```

Diagnostic columns were added for action conservatism and gradient/update scale:

```text
action_saturation_rate
action_abs_mean
action_abs_max
action_delta_mean
action_delta_max
hover_relative_action_mean
hover_relative_action_max
action_gradient_norm
raw_action_gradient_norm_pre_clip
raw_action_gradient_norm_post_clip
actor_gradient_norm_pre_clip
actor_update_norm
```

Evaluation CSV metadata now distinguishes fixed and sampled dynamics:

```text
sample_dynamics
eval_dynamics_mode
sampled_dynamics_level
correlated_size_mass_sampling
mass_min/mass_mean/mass_max
thrust_to_weight_min/thrust_to_weight_mean/thrust_to_weight_max
motor_delay_min/motor_delay_mean/motor_delay_max
```

## Validation Commands

Build:

```bash
cmake --build /tmp/raptor_stage96_cuda_build \
  --target foundation_policy_diff_pre_training_cuda -j2
```

Persistent smoke:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler \
  --gpu-rollout \
  --steps 100 \
  --horizon 16 \
  --batch-size 8192 \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --terminal-loss-scale 0 \
  --seed 45001 \
  --load-path reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_5000_b65536_eval100/checkpoint_1500.ckpt \
  --save-optimizer \
  --log-path reports/h16_raptor_l2f_recovery_pipeline/persistent_h16_smoke_verify2/train_persistent_100.csv \
  --save-path reports/h16_raptor_l2f_recovery_pipeline/persistent_h16_smoke_verify2/checkpoint_persistent_100.ckpt
```

Horizon/dynamics eval matrix:

```bash
for mode in fixed small broad broad_correlated; do
  for horizon in 16 50 100 250 500; do
    foundation_policy_diff_pre_training_cuda --eval-only --gpu-rollout \
      --load-path reports/h16_raptor_l2f_recovery_pipeline/persistent_h16_smoke_verify2/checkpoint_persistent_100.ckpt \
      --eval-horizon ${horizon} \
      --eval-episodes 256 \
      <mode flags> \
      --terminal-loss-scale 0 \
      --seed 46001 \
      --log-path reports/h16_raptor_l2f_recovery_pipeline/persistent_h16_smoke_verify2/eval_matrix/eval_${mode}_h${horizon}.csv
  done
done
```

Loss/scale ablation:

```bash
foundation_policy_diff_pre_training_cuda \
  --diff-model euler \
  --gpu-rollout \
  --steps 250 \
  --horizon 16 \
  --batch-size 8192 \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --terminal-loss-scale 0 \
  --seed 48001 \
  --load-path reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_5000_b65536_eval100/checkpoint_1500.ckpt \
  <single ablation flag>
```

## Persistent Smoke Evidence

`reports/h16_raptor_l2f_recovery_pipeline/persistent_h16_smoke_verify2/train_persistent_100.csv`

| Row | persistent step | segment | reset count | finite | weighted cost | action saturation | actor grad pre | update norm |
| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 0-16 | 0 | true | 8.45274 | 0 | 0.0110525 | 0.00342882 |
| 1 | 16 | 16-32 | 0 | true | 10.8368 | 0 | 0.218863 | 0.00806546 |
| 30 | 480 | 480-496 | 0 | true | 7683.72 | 0.0349197 | 1.52912 | 0.0150454 |
| 31 | 496 | 496-512 | 0 | true | 7815.58 | 0.0291309 | 1.41215 | 0.0147041 |
| 32 | 0 | 0-16 | 1 | true | 8.5005 | 0 | 0.283035 | 0.0129958 |
| 63 | 496 | 496-512 | 1 | true | 6123.31 | 0.00235176 | 0.499446 | 0.00662364 |
| 64 | 0 | 0-16 | 2 | true | 8.3864 | 0 | 0.197375 | 0.00618077 |
| 99 | 48 | 48-64 | 3 | true | 28.3463 | 0 | 0.24194 | 0.0040062 |

Stdout:

```text
gpu_full_training_passed=true
gpu_full_training_finite=true
gpu_full_training_nan_inf_count=0
gpu_full_training_persistent_episode_training=true
gpu_full_training_persistent_episode_step=64
gpu_full_training_segment_start=48
gpu_full_training_segment_end=64
gpu_full_training_episode_reset_count=3
```

This proves that states are not resampled every optimizer step. The first episode keeps the same batch while `persistent_episode_step` advances from 0 to 496, then resets. GRU hidden also persists through the persistent hidden buffer and is detached at each H16 boundary.

## Horizon And Dynamics Matrix

Checkpoint:

```text
reports/h16_raptor_l2f_recovery_pipeline/persistent_h16_smoke_verify2/checkpoint_persistent_100.ckpt
```

| Mode | H | weighted cost | position cost | velocity cost | action cost | returns mean | terminated | invalid/NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 16 | 7.70455 | 1.8177 | 0.935423 | 0.000194171 | -7.70455 | 0 | 0 |
| fixed | 50 | 10.0325 | 2.95444 | 2.68951 | 0.000164243 | -10.0325 | 0 | 0 |
| fixed | 100 | 26.1972 | 14.8504 | 7.73908 | 0.000262834 | -26.1972 | 0 | 0 |
| fixed | 250 | 327.482 | 301.203 | 24.2978 | 0.000412134 | -327.482 | 0 | 0 |
| fixed | 500 | 1804.95 | 1776.21 | 27.4929 | 0.000366077 | -1804.95 | 0 | 0 |
| small | 16 | 8.44426 | 1.90346 | 0.976142 | 0.000235398 | -8.44426 | 0 | 0 |
| small | 100 | 30.7153 | 17.7172 | 8.91841 | 0.000259082 | -30.7153 | 0 | 0 |
| small | 500 | 2023.7 | 1992.23 | 30.048 | 0.000457581 | -2023.7 | 0 | 0 |
| broad | 16 | 8.26277 | 1.99121 | 1.094 | 0.000664596 | -8.26277 | 0 | 0 |
| broad | 100 | 39.3402 | 23.0925 | 12.1741 | 0.000648117 | -39.3402 | 0 | 0 |
| broad | 500 | 2394.02 | 2351.16 | 40.5944 | 0.00210922 | -2394.02 | 0 | 0 |
| broad correlated | 16 | 8.72144 | 1.89809 | 1.11447 | 0.000659522 | -8.72144 | 0 | 0 |
| broad correlated | 100 | 40.108 | 23.3169 | 12.6742 | 0.000640627 | -40.108 | 0 | 0 |
| broad correlated | 500 | 2432.55 | 2394.45 | 36.5069 | 0.00203978 | -2432.55 | 0 | 0 |

Full CSV:

```text
reports/h16_raptor_l2f_recovery_pipeline/persistent_h16_smoke_verify2/eval_matrix/horizon_dynamics_summary.csv
```

Eval metadata confirms the flags are no longer ambiguous:

- fixed: `sample_dynamics=false`, nominal mass `0.05`, thrust-to-weight `3`, motor delay `0.065`.
- small: `sample_dynamics=true`, `sampled_dynamics_level=small`, mass mean `0.054651`.
- broad: `sample_dynamics=true`, `sampled_dynamics_level=broad`, mass mean `0.135897`.
- broad correlated: `sample_dynamics=true`, `sampled_dynamics_level=broad`, `correlated_size_mass_sampling=true`, mass mean `0.110666`.

## Loss And Scaling Ablation

All runs start from the same 1500-step checkpoint and use H16 persistent broad correlated dynamics for 250 updates.

| Variant | train finite | train saturation | train action mean | train actor grad | H16 weighted | H100 weighted | H500 weighted | H500 position | H500 velocity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | true | 0.00260925 | 0.198742 | 0.156073 | 8.28117 | 34.2551 | 2829.28 | 2784.66 | 43.2751 |
| diff_x5 | true | 0.00185966 | 0.200625 | 0.178538 | 8.28061 | 34.8265 | 2496.38 | 2458.12 | 36.9331 |
| diff_x10 | true | 0.00172424 | 0.200787 | 0.173803 | 8.27523 | 35.3792 | 2864.16 | 2818.46 | 44.3628 |
| smooth_x0.5 | true | 0.00261116 | 0.198742 | 0.156087 | 8.28092 | 34.2551 | 2829.18 | 2784.56 | 43.2725 |
| action_mag_x0.5 | true | 0.00261307 | 0.198649 | 0.156101 | 8.28108 | 34.2594 | 2834.12 | 2789.41 | 43.3592 |
| progress_w1 | true | 0.00246048 | 0.198468 | 0.15882 | 8.28223 | 34.2769 | 2559.39 | 2518.47 | 37.6517 |
| outward_w1 | true | 0.00946617 | 0.230395 | 0.149284 | 8.48285 | 64.9911 | 407.125 | 232.189 | 6.69475 |
| vref_k0.5 | true | 0.00412941 | 0.188284 | 0.239466 | 8.37483 | 39.7477 | 2072.28 | 1944.34 | 126.289 |

Full CSV:

```text
reports/h16_raptor_l2f_recovery_pipeline/loss_design_ablation/ablation_summary.csv
```

The strongest result is `outward_w1`: it worsens the H100 weighted cost because the outward penalty itself is active, but it sharply reduces H500 position and velocity drift. This supports the diagnosis that the old H16 objective can remain locally stable while allowing outward long-horizon drift.

## Hypothesis Table

| Hypothesis | Evidence | Conclusion | Next action |
| --- | --- | --- | --- |
| Fresh independent H16 windows caused a train/eval distribution gap. | Persistent episode logging now advances 0,16,...,496 before reset and carries state/hidden. H500 is still poor, though better than the old 6075 broad-correlated reference. | Persistent BPTT is necessary for diagnosis but not sufficient by itself. | Keep persistent episodes as default; train longer only after loss fixes. |
| Eval flags were wrong because fixed and sampled looked identical. | Eval metadata now separates fixed/small/broad/broad-correlated mass, thrust-to-weight, motor delay, and `sample_dynamics`. Fixed is nominal; sampled modes differ. | The earlier matching was a logging/metadata ambiguity, not proof that eval ignored dynamics flags. | Keep metadata columns in eval CSV. |
| Broad correlated dynamics are the only reason H500 fails. | Fixed H500 position cost is already 1776.21, small is 1992.23, broad correlated is 2394.45. | Dynamics randomization worsens the problem, but fixed dynamics also fails long horizon. | Do not add FiLM/latent yet; fix objective first. |
| The controller is too conservative. | Baseline H500 action cost is only 0.00215 and train action mean is about 0.20; H500 position dominates. | Action penalties are not the main limiter; merely halving smoothness/action magnitude does not help. | Do not tune action penalties first. |
| Physics-gradient scale is too small. | diff_x5 improves H500 position from 2784.66 to 2458.12; diff_x10 loses the improvement. | Some scale increase helps, but it is not the main fix and can become too strong. | Use at most diff_x5 as a secondary knob. |
| Short-horizon zero-velocity loss discourages moving inward. | `outward_w1` cuts H500 position from 2784.66 to 232.189 and velocity from 43.2751 to 6.69475. | This is the strongest evidence: the active loss needs an inward-progress or outward-velocity term for long closed-loop recovery. | Next formal run should train persistent H16 with a tuned outward penalty, then eval H16/H50/H100/H250/H500. |
| A velocity reference toward origin solves the issue. | `vref_k0.5` improves H500 position to 1944.34 but increases H500 velocity to 126.289. | Simple proportional velocity reference is unstable or too aggressive at this gain. | If retried, use smaller gain and combine with outward penalty, not alone. |

## Current Recommendation

Use persistent H16 truncated BPTT as the active training path. Do not return to fresh independent H16 batches for origin recovery.

For the next training run, keep H16 and broad correlated dynamics, and try a conservative outward-velocity penalty. Suggested starting point:

```text
--w-outward-velocity 0.5
--diff-rollout-loss-weight 0.001 to 0.0025
```

Do not add FiLM, latent dynamics conditioning, H64/H128, or teacher/student experiments yet. The current evidence points to objective design, not insufficient architecture or dynamics conditioning.
