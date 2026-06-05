# H16 Recovery-Style Tracking Completion Audit

This audit checks the requested refactor/deliverable, not whether the learned policy is a good controller. The current policy quality is still weak; the question here is whether the pipeline now follows the requested RAPTOR/L2F-style H16 recovery-and-tracking setup and produces the required evidence artifacts.

## Summary

| Requirement | Status | Evidence |
| --- | --- | --- |
| Preserve rigid-body force/torque rollout, CUDA VJP, GRU actor, and 4-motor output | PASS | CUDA target still reports `actor_output_dim=4`; changes are in task/reference/sampler/options/metrics paths, not a replacement controller interface. |
| Fixed H16 BPTT, no H32/H64/H128 curriculum for this pipeline | PASS | All committed recovery-training CSVs use `horizon=16`; report commands use fixed H16 and no `--horizon-curriculum`. |
| Long episode reference windows | PASS | Training CSVs include `training_episode_steps=500` and nonzero `window_start_mean` values, e.g. `239.583`, `244.242`. |
| Strict gate removed from main training objective | PASS | Training loss logs use per-step loss components; strict throughout/final success metrics are eval diagnostics only. |
| RAPTOR/L2F recovery initial state defaults | PASS | Defaults map to approximately +/-0.5 m position, +/-1 m/s velocity, up to ~90 deg attitude, +/-1 rad/s angular velocity, and near-zero guidance probability `0.10`. |
| Terminal loss disabled by default | PASS | `terminal_loss_scale=0` default; training CSV terminal loss/ratio columns are present and zero in the reported runs. |
| Horizon-normalized per-step tracking/control loss over `t=1...H` | PASS | `loss_and_action_kernel` uses `state_step = step_i + 1` and `normalizer = 1/H`; CPU parity path mirrors this. |
| Position/velocity moving-reference tracking | PASS | Observation and loss use `reference_p_traj/reference_v_traj`; moving-reference CPU/CUDA parity passed. |
| Attitude, angular velocity, action magnitude, smoothness, saturation losses | PASS | Component columns exist in training CSVs and are logged per run. |
| Linear/angular acceleration losses exposed and logged | PASS | `--w-linear-acceleration`, `--w-angular-acceleration`; training CSV includes acceleration component and ratio columns. |
| Temporal gradient decay support | PASS | `--temporal-gradient-decay-alpha`; CPU/CUDA parity passed with alpha `0.5`; implementation accepts arbitrary values including `0`, `0.25`, `0.5`, `1`, and `2`. |
| Task curriculum, not horizon curriculum | PASS | Committed curriculum logs cover fixed, step, circle, figure-eight, mixed, small dynamics, and broad correlated stages, all fixed H16. |
| Separate small and broad dynamics domains | PASS | `--sampled-dynamics-level small|broad` selects separate GPU sampler ranges; eval CSV metadata records the selected level. |
| Broad + correlated size-mass sampling compatibility | PASS | Broad correlated mixed staged run completed finite and produced train/eval CSV. |
| RAPTOR/L2F-style metrics | PASS | Eval CSVs include settling fraction, mean-error mean, max-error mean/std for position, angle, linear velocity, angular velocity, angular acceleration, action, and action relative. |
| RMSE supplemental metrics | PASS | Eval CSVs include `position_rmse`, `velocity_rmse`, `attitude_rmse`, and `angular_velocity_rmse`. |
| Saturation and NaN/Inf diagnostics | PASS | Eval CSVs include action saturation, invalid/NaN rate, and stability diagnostics; training/eval stdout also reports finite status. |
| Velocity observation noise/delay options | PASS | `--velocity-observation-noise` and `--velocity-observation-delay-steps`; observation parity validation passed with nonzero noise/delay. |
| No trajectory lookahead | PASS | Report documents current-reference-only tracking; actor consumes current relative `p_ref(t), v_ref(t)` offsets only. |
| Final report and CSV logs for fixed/step/circle/figure8/mixed | PASS | `report.md` and committed CSV logs under `task_curriculum_long_1000_b2048/`. |

## Primary Evidence Artifacts

```text
reports/h16_raptor_l2f_recovery_pipeline/report.md
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_fixed_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_step_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_circle_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_figure8_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_mixed_1000.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_fixed_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_step_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_circle_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_figure8_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_mixed_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_small_mixed_200.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_small_mixed_500.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/train_broad_correlated_mixed_200.csv
reports/h16_raptor_l2f_recovery_pipeline/task_curriculum_long_1000_b2048/eval_broad_correlated_mixed_500.csv
```

## Important Non-Claims

- The current policy is not a good or deployable controller.
- The broad correlated domain is not solved.
- The H16 recovery setup does not prove teacher-cost reduction.
- H64/H128 and teacher/student experiments remain separate future work.
- Current long-episode support samples reference windows from a 500-step trajectory; it does not claim full 500-step differentiable backpropagation.

## Residual Research Risks

- Long-control position errors remain large in the committed eval logs.
- Some short from-scratch sampled-domain evals show invalid/NaN episodes; staged small-domain eval from the mixed checkpoint is finite.
- Further tuning is needed before using the policy as a controller or comparing against RAPTOR teachers.
