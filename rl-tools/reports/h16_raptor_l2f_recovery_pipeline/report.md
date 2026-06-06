# H16 Origin Recovery RAPTOR Metrics

## Scope

This report tracks the active CUDA diffphys objective after aligning its public metrics with the original RAPTOR/L2F reporting style.

The active task remains fixed-H16 origin recovery:

- GRU actor output remains exactly four normalized motor commands.
- The differentiable force/torque rigid-body rollout, CUDA VJP chain, motor delay, previous-action input, and optimizer path are unchanged.
- Training recovers to the origin: zero position offset, zero linear velocity, identity attitude, and zero angular velocity.
- Initial position, velocity, attitude, and angular velocity are randomized with recovery-scale defaults.
- Terminal loss remains disabled by default.
- Non-origin setpoints are eval/deployment-time setpoint shifting only; they are not an active training curriculum.
- H64/H128 and stage-style curriculum entry points remain rejected by the active CUDA interface.

## Metric Interface

Active training and eval now report RAPTOR-style aggregate metrics:

- `returns_mean`
- `returns_std`
- `episode_length_mean`
- `episode_length_std`
- `num_terminated`
- `share_terminated`

Reward component reporting uses the diffphys weighted-cost decomposition:

- `position_cost`
- `orientation_cost`
- `linear_velocity_cost`
- `angular_velocity_cost`
- `action_cost`
- `weighted_cost`
- `reward`

For this differentiable origin-recovery path, `weighted_cost` is the horizon-normalized rollout cost and `reward = -weighted_cost`. Episode length is fixed by the rollout horizon unless the rollout is non-finite, in which case it is counted as terminated.

The previous local tracking classification and window-diagnostic fields are no longer emitted by the active train/eval stdout or CSV files.

## Validation Commands

Build:

```bash
cmake --build /tmp/raptor_stage96_cuda_build --target foundation_policy_diff_pre_training_cuda -j2
```

H16 origin-recovery smoke train:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler \
  --gpu-rollout \
  --steps 5 \
  --horizon 16 \
  --batch-size 256 \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --terminal-loss-scale 0 \
  --save-optimizer \
  --seed 9701 \
  --log-path reports/h16_raptor_l2f_recovery_pipeline/raptor_metric_alignment/train_smoke.csv \
  --save-path reports/h16_raptor_l2f_recovery_pipeline/raptor_metric_alignment/train_smoke.ckpt
```

500-step fixed-dynamics eval:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --eval-only \
  --gpu-rollout \
  --load-path reports/h16_raptor_l2f_recovery_pipeline/raptor_metric_alignment/train_smoke.ckpt \
  --eval-horizon 500 \
  --eval-episodes 128 \
  --fixed-dynamics \
  --sampled-dynamics-level broad \
  --trajectory-mode fixed \
  --seed 9702 \
  --log-path reports/h16_raptor_l2f_recovery_pipeline/raptor_metric_alignment/eval_fixed_500.csv
```

## Validation Results

| Check | Result | Evidence |
| --- | --- | --- |
| CUDA build | PASS | Target rebuilt and linked. |
| actor output dimension | PASS | Startup reports `actor_output_dim=4`. |
| active H16 invariant | PASS | Active training still requires `--horizon 16`. |
| origin training invariant | PASS | Non-fixed trajectory modes remain eval/deployment-only for active training. |
| train stdout metric vocabulary | PASS | Smoke stdout contains no removed classification wording. |
| eval stdout metric vocabulary | PASS | Fixed eval stdout contains no removed classification wording. |
| train CSV schema | PASS | Header is RAPTOR-style aggregate/cost fields; no classification-rate fields. |
| eval CSV schema | PASS | Header contains only returns, episode length, terminated share, and reward-cost fields. |
| smoke train finite | PASS | 5-step H16 broad correlated smoke completed with `finite=true`. |
| fixed eval finite | PASS | 500-step fixed eval completed with `num_terminated=0`. |

## Smoke Train Metrics

| Metric | Value |
| --- | ---: |
| returns_mean | -9.66902 |
| returns_std | 6.4727 |
| episode_length_mean | 16 |
| episode_length_std | 0 |
| num_terminated | 0 |
| share_terminated | 0 |
| position_cost | 1.94749 |
| orientation_cost | 5.41736 |
| linear_velocity_cost | 1.08864 |
| angular_velocity_cost | 1.21487 |
| action_cost | 0.000660842 |
| weighted_cost | 9.66902 |
| reward | -9.66902 |

## Fixed Eval Metrics

| Metric | Value |
| --- | ---: |
| returns_mean | -37490.4 |
| returns_std | 28094 |
| episode_length_mean | 500 |
| episode_length_std | 0 |
| num_terminated | 0 |
| share_terminated | 0 |
| position_cost | 17058.7 |
| orientation_cost | 18.0308 |
| linear_velocity_cost | 508.87 |
| angular_velocity_cost | 19904.8 |
| action_cost | 0.0062305 |
| weighted_cost | 37490.4 |
| reward | -37490.4 |

## Artifacts

Tracked report:

```text
reports/h16_raptor_l2f_recovery_pipeline/report.md
reports/h16_raptor_l2f_recovery_pipeline/completion_audit.md
```

Runtime artifacts, not intended for commit:

```text
reports/h16_raptor_l2f_recovery_pipeline/raptor_metric_alignment/*.ckpt
reports/h16_raptor_l2f_recovery_pipeline/raptor_metric_alignment/*.stdout
```
