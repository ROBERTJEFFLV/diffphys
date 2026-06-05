# H16 Origin Recovery Completion Audit

This audit is for the active CUDA training objective: fixed-H16 origin recovery as a RAPTOR/L2F-style motor-level position controller. It does not claim that the learned policy is high quality.

## Requirement Audit

| Requirement | Status | Evidence |
| --- | --- | --- |
| Preserve rigid-body force/torque rollout, CUDA VJP, motor delay, previous-action input, GRU actor, and 4-motor output | PASS | The modified path still uses `run_full_gpu_training`, `forward_step_kernel`, `backward_step_kernel`, motor delay dynamics, and startup reports `actor_output_dim=4`. |
| Active training task is fixed-H16 origin recovery | PASS | Training with `--horizon 16 --trajectory-mode fixed` ran finite; `make_full_training_options` forces H16 and fixed origin reference. |
| Remove H64/H128 and curriculum logic from active CUDA training | PASS | `--horizon-curriculum 16,32,64,128`, `--horizon 32`, and `--stage11 --stage11-steps 1` all returned nonzero before training. |
| Train to origin with zero linear velocity, identity attitude, and zero angular velocity | PASS | Fixed reference mode fills `reference_p_traj/reference_v_traj` with zero, and loss terms remain position, velocity, attitude, and angular velocity errors. |
| Keep randomized recovery initial states | PASS | Recovery defaults remain approximately +/-0.5 m position, +/-1 m/s velocity, up to about 90 degrees attitude, +/-1 rad/s angular velocity, and near-zero guidance probability 0.10. |
| Terminal loss disabled by default | PASS | `terminal_loss_scale` default remains 0. |
| Keep action magnitude/smoothness/saturation penalties | PASS | Loss weights and CSV component logging remain in the active training path. |
| Keep finite/NaN diagnostics | PASS | Training and eval summaries still report finite status and NaN/Inf counts. |
| Keep 500-step long recovery evaluation | PASS | Fixed-dynamics 500-step origin-recovery eval from the smoke checkpoint ran finite and wrote metrics CSV. |
| Waypoint/trajectory tracking only as deployment-time setpoint shifting | PASS | Non-fixed trajectory training was rejected; circle deployment adapter validation passed with max observation error `2.14204e-06`. |
| Reports use origin recovery, position controller, and setpoint shifting terminology | PASS | `report.md` and this audit use the new terms. |
| Remove obsolete future-planning language from active reports | PASS | Current report text describes origin recovery, position controller behavior, and deployment/eval setpoint shifting only. |

## Validation Evidence

```text
CUDA build passed
origin-recovery CPU/CUDA parity passed
removed curriculum flag rejection passed
removed H32 training rejection passed
removed stage entry point rejection passed
non-fixed trajectory training rejection passed
H16 broad correlated origin-recovery smoke training finite
500-step fixed-dynamics origin-recovery eval finite
deployment setpoint-shifting adapter validation passed
```

## Non-Claims

- This is not a solved flight controller.
- The report does not claim broad-domain convergence.
- Non-origin trajectories are not part of the active training objective.
