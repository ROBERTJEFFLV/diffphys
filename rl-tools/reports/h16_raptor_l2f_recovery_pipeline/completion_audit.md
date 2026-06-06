# H16 Origin Recovery Metric Alignment Audit

This audit covers the active CUDA objective: fixed-H16 origin recovery as a RAPTOR/L2F-style motor-level position controller. It only audits the public metric interface and smoke validation; it does not claim controller quality.

## Requirement Audit

| Requirement | Status | Evidence |
| --- | --- | --- |
| Preserve rigid-body force/torque rollout, CUDA VJP, motor delay, previous-action input, GRU actor, and 4-motor output | PASS | The path still uses `forward_step_kernel`, `backward_step_kernel`, motor delay dynamics, and startup reports `actor_output_dim=4`. |
| Keep active training fixed at H16 origin recovery | PASS | Active training still rejects non-H16 training and non-fixed training setpoints. |
| Keep non-origin setpoints as eval/deployment shifting only | PASS | The active training invariant rejects non-fixed trajectory modes. |
| Remove non-RAPTOR classification metrics from active train/eval stdout | PASS | Smoke train stdout and fixed eval stdout contain no removed classification wording. |
| Remove non-RAPTOR classification metrics from active training CSV | PASS | `train_smoke.csv` header uses returns, episode length, terminated share, and reward-cost columns. |
| Preserve RAPTOR-style eval aggregate summary | PASS | `eval_fixed_500.csv` header is `returns_mean,returns_std,episode_length_mean,episode_length_std,num_terminated,share_terminated,...`. |
| Preserve reward component cost reporting | PASS | Train and eval CSVs include position, orientation, linear velocity, angular velocity, action, weighted cost, and reward. |
| Preserve finite/NaN accounting through terminated share | PASS | Non-finite rollout accounting is exposed through `num_terminated` and `share_terminated`. |
| Keep terminal loss disabled by default | PASS | Smoke command used `--terminal-loss-scale 0`. |
| Keep broad correlated dynamics smoke path | PASS | Smoke training used broad dynamics, balanced sampling, and correlated size-mass sampling. |

## Validation Evidence

```text
CUDA build passed
H16 broad correlated origin-recovery smoke train passed
500-step fixed-dynamics eval passed
train stdout vocabulary check passed
eval stdout vocabulary check passed
train CSV header check passed
eval CSV header check passed
```

## Current Public Metrics

Training and eval report:

```text
returns_mean
returns_std
episode_length_mean
episode_length_std
num_terminated
share_terminated
position_cost
orientation_cost
linear_velocity_cost
angular_velocity_cost
action_cost
weighted_cost
reward
```

Training also keeps optimizer/debug fields needed for engineering health:

```text
critic_loss_raw
critic_loss_weight
critic_loss_scaled
critic_output_mean
critic_target_mean
critic_error_mean
critic_error_norm
grad_norm
finite
```

## Smoke Values

| Source | returns_mean | returns_std | episode_length_mean | num_terminated | share_terminated | weighted_cost | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H16 train smoke | -9.66902 | 6.4727 | 16 | 0 | 0 | 9.66902 | -9.66902 |
| 500-step fixed eval | -37490.4 | 28094 | 500 | 0 | 0 | 37490.4 | -37490.4 |

## Non-Claims

- The smoke checkpoint is not a trained controller.
- This audit does not claim broad-domain convergence.
- This audit does not start H64/H128 training.
- This audit does not run teacher/student imitation; supervised MSE reporting remains a separate post-training path when teacher-action data is available.
