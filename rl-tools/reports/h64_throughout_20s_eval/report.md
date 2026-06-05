# H64 20s Throughout Stability Check

## Purpose

The previous CUDA eval success metric checked the final state only. This pass adds stricter throughout-window metrics for long continuous control:

- `throughout_success_rate`: every timestep from `t=0` through `t=H` must satisfy the same position, velocity, and angular-velocity thresholds as final-state success.
- `mean_time_inside_fraction`: mean fraction of timesteps inside all thresholds.
- `mean_first_failure_time_s`: mean first threshold-crossing time among failed episodes.
- max-over-time position, velocity, and angular-velocity norms.
- max absolute action over the full rollout.

For these checks, `H=2000`, `dt=0.01`, so each episode is 20 seconds.

## Code Changes

Files:

```text
src/foundation_policy/diff_pre_training/gpu_rollout.h
src/foundation_policy/diff_pre_training/gpu_rollout.cu
src/foundation_policy/diff_pre_training/cuda_main.cu
```

The GPU eval path now copies the full eval-only `p`, `v`, `omega`, and action trajectories back to host after rollout and computes the stricter throughout metrics. Training forward/backward is unchanged.

## Build

```bash
cmake --build /tmp/raptor_stage96_cuda_build \
  --target foundation_policy_diff_pre_training_cuda -j2
```

Result: PASS.

## Commands

H64 checkpoint:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --eval-only \
  --gpu-rollout \
  --fixed-dynamics \
  --load-path reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_3_h64.ckpt \
  --eval-horizon 2000 \
  --eval-episodes 1000 \
  --success-velocity-threshold 0.5 \
  --log-path reports/h64_throughout_20s_eval/eval_h64_h2000_e1000.csv
```

H128 checkpoint comparison:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --eval-only \
  --gpu-rollout \
  --fixed-dynamics \
  --load-path reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_8_h128.ckpt \
  --eval-horizon 2000 \
  --eval-episodes 1000 \
  --success-velocity-threshold 0.5 \
  --log-path reports/h64_throughout_20s_eval/eval_h128_h2000_e1000.csv
```

## Results

| Checkpoint | Final success | Throughout success | Mean inside fraction | Mean first failure time | Mean max p | Mean max v | Mean max w | Max action abs | Saturation | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H64 chunk 3 | 0.987 | 0.939 | 0.969842 | 0.904098s | 0.507714 | 0.22543 | 0.135689 | 0.146045 | 0 | 0 |
| H128 chunk 8 | 0 | 0 | 0.247452 | 4.49171s | 1484.24 | 233.77 | 11.3004 | 1 | 0.593947 | 0 |

## Interpretation

H64 is substantially more stable than H128 for 20s fixed-dynamics continuous control. Its final-state success remains high and action saturation stays at zero. However, it is not a strict 100% throughout-stable controller: 61 of 1000 episodes crossed at least one threshold at some point during the 20s rollout.

The H128 checkpoint is not suitable for 20s continuous control. It leaves the valid envelope after roughly 4.5s on average, then diverges and uses saturated actions heavily.

## Conclusion

Allowed claim:

- The H64 checkpoint is the stronger long-continuous-control checkpoint among the preserved fixed-dynamics checkpoints.
- Under fixed nominal dynamics, it reaches 0.987 final success and 0.939 strict throughout success over 1000 20s episodes without action saturation.

Not allowed:

- Claiming H64 is fully stable for every 20s episode under the current thresholds.
- Claiming the H128 checkpoint is a 20s controller.

