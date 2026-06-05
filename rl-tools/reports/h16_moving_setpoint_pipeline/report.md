# H16 Moving-Setpoint Trajectory Tracking Pipeline

## Scope

This pass upgrades the CUDA differentiable motor-policy path from fixed-reference stabilization toward time-indexed moving-setpoint tracking while preserving the existing force/torque rigid-body rollout, CUDA VJP chain, GRU actor, and exactly four normalized motor-command outputs.

The implementation is validated through smoke tests, CPU/CUDA parity gates, local initial-condition checks, and an initial fixed-dynamics/fixed-trajectory H16 local training run. The strict H16 moving-setpoint throughout gate under broad correlated dynamics is not yet solved, so no H64/H128 or teacher/student experiments were run from this pipeline.

## Implemented

- Added time-indexed `reference_p_traj` and `reference_v_traj` with shape `[H+1, B, 3]` to the CUDA batch, device arrays, copies, and replay schema.
- Preserved fixed-reference compatibility by filling all trajectory steps with the fixed reference in `fixed` mode and keeping legacy `reference_p/reference_v` metadata.
- Changed CUDA observation construction to use `p_t - reference_p_traj[t]` and `v_t - reference_v_traj[t]`.
- Changed CUDA rollout loss, critic target construction, diagnostics, and eval metrics to use time-indexed references.
- Added trajectory modes: `fixed`, `step`, `circle`, `figure8`, and `mixed`.
- Added CLI options:
  - `--trajectory-mode fixed|step|circle|figure8|mixed`
  - `--trajectory-amplitude`
  - `--trajectory-frequency-hz`
  - `--gpu-validate-trajectory-sampler`
- Kept broad dynamics randomization compatible with `--correlated-size-mass-sampling`.
- Added eval RMSE/statistics for position, velocity, attitude, angular velocity, final success, throughout success, max errors, action saturation, and invalid/NaN rate.
- Added a deployment observation adapter that converts external `p_ref(t), v_ref(t)` setpoints into relative offsets for the same GRU actor. The actor still outputs exactly four motor commands.
- Added moving-reference CPU/CUDA rollout loss and action-gradient parity. The CPU validation path now uses the same `reference_p_traj/reference_v_traj` as CUDA instead of the legacy fixed `TrackingReference`.
- Added CUDA training CSV component logging for position, velocity, attitude, angular velocity, action magnitude, action smoothness, saturation, terminal loss, terminal subterms, and component ratios.
- Added deterministic deployment-adapter validation against the CUDA training observation path.
- Added optional linear/angular acceleration loss terms with CPU/CUDA parity coverage:
  - `--w-linear-acceleration`
  - `--w-angular-acceleration`
- Added eval metrics for linear acceleration error RMSE, angular acceleration error RMSE, global max tracking errors, action magnitude, and action smoothness.
- Added local initial attitude perturbation sampling for strict local-window H16 checks.
- Local tracking initial states are now sampled around `reference_p_traj[0]` and `reference_v_traj[0]`, not around world zero, so moving-reference local eval starts near the current trajectory setpoint.
- Added `--gpu-validate-local-initial-conditions` to verify local samples satisfy the strict tube at t=0.
- Added local/recovery gate plumbing:
  - `--tracking-gate-mode local|recovery`
  - `--throughout-gate-start-step`
  - `--initial-position-scale-local`
  - `--initial-velocity-scale-local`
  - `--initial-attitude-scale-local`
  - `--initial-angular-velocity-scale-local`
  - eval reports both full-window `throughout_success_rate` and `post_burnin_throughout_success_rate`.

## Validation Commands

```bash
cmake --build /tmp/raptor_stage96_cuda_build --target foundation_policy_diff_pre_training_cuda -j2

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-trajectory-sampler \
  --batch-size 64 \
  --horizon 16 \
  --seed 9001 \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-correlated-size-mass-sampler \
  --batch-size 512 \
  --horizon 16 \
  --seed 9002

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-observation \
  --gpu-validate-against-cpu \
  --batch-size 64 \
  --horizon 16 \
  --seed 9003 \
  --trajectory-mode fixed

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-observation \
  --batch-size 64 \
  --horizon 16 \
  --seed 9004 \
  --trajectory-mode circle \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 64 \
  --horizon 16 \
  --seed 9101 \
  --trajectory-mode circle \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --terminal-loss-scale 0.2 \
  --action-saturation-start 0.85

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-observation \
  --batch-size 64 \
  --horizon 16 \
  --seed 9102 \
  --trajectory-mode figure8 \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --validation-step 8

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-deployment-adapter \
  --batch-size 64 \
  --horizon 16 \
  --seed 9103 \
  --trajectory-mode mixed \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --validation-step 8

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 64 \
  --horizon 16 \
  --seed 9201 \
  --trajectory-mode circle \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --terminal-loss-scale 0.2 \
  --action-saturation-start 0.85 \
  --w-linear-acceleration 0.001 \
  --w-angular-acceleration 0.0001

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-local-initial-conditions \
  --tracking-gate-mode local \
  --batch-size 4096 \
  --horizon 16 \
  --seed 9301 \
  --trajectory-mode mixed \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --correlated-size-mass-sampling
```

Smoke training:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler \
  --gpu-rollout \
  --steps 20 \
  --horizon 16 \
  --batch-size 2048 \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --trajectory-mode mixed \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --initial-position-scale 0.2 \
  --strict-stability-thresholds \
  --w-linear-acceleration 0.001 \
  --w-angular-acceleration 0.0001 \
  --w-v 6.0 \
  --w-terminal-v 40 \
  --terminal-loss-scale 0.2 \
  --w-u 0.3 \
  --w-sat 2.0 \
  --action-saturation-start 0.85 \
  --actor-grad-clip 1.0 \
  --save-optimizer \
  --log-path reports/h16_moving_setpoint_pipeline/train_smoke_20.csv \
  --save-path reports/h16_moving_setpoint_pipeline/checkpoint_smoke_20.ckpt
```

Eval was run separately for `fixed`, `step`, `circle`, and `figure8` with `--eval-horizon 16`, `--eval-episodes 256`, broad dynamics, balanced sampling, correlated size-mass sampling, and strict stability thresholds.

## Pass/Fail Table

| Gate | Result | Evidence |
| --- | --- | --- |
| CUDA build | PASS | target linked successfully |
| actor output dimension | PASS | startup reports `actor_output_dim=4` |
| trajectory sampler deterministic | PASS | max deterministic error 0 |
| fixed-reference compatibility | PASS | fixed reference max abs error 0 |
| mixed sampler coverage | PASS | coverage mask 15 covers fixed/step/circle/figure8 |
| correlated size-mass sampler | PASS | formula mismatch count 0, NaN/Inf count 0 |
| fixed-reference CPU/CUDA parity | PASS | observation, forward, loss, and action-gradient parity passed |
| moving-reference observation parity | PASS | circle observation max/mean error 0 |
| H16 broad correlated smoke training | PASS | finite, checkpoint saved, NaN/Inf count 0 |
| action saturation safety in smoke/eval | PASS | saturation rate 0 |
| moving-reference CPU/CUDA loss/VJP parity | PASS | circle trajectory, terminal scale 0.2, saturation start 0.85, loss/action-gradient parity passed |
| nonzero-step moving observation parity | PASS | figure8 trajectory at step 8, max error 1.82353e-06 |
| deployment adapter parity | PASS | mixed trajectory at step 8, max error 2.93925e-06 |
| CUDA component loss ratios in training CSV | PASS | component columns and raw-loss ratios written in `train_component_check_1.csv` |
| local/recovery gate CLI plumbing | PASS | local mode applies local initial scales; eval reports post-burn-in throughout success |
| acceleration loss CPU/CUDA parity | PASS | nonzero linear/angular acceleration weights passed moving-reference loss/action-gradient parity |
| acceleration/action eval metrics | PASS | `eval_local_fixed_metrics_check.csv` writes acceleration RMSE, global max errors, action magnitude, and action smoothness |
| local initial attitude sampling | PASS | local smoke ran with `initial_attitude_scale=1` and finite rollout/gradients |
| local initial-condition strict tube | PASS | 4096 mixed samples: max p/v/attitude/w = 0.0456943/0.0416643/0.0828115/0.0336794, all below 0.05/0.1/0.0872665/0.2 |
| fixed-dynamics fixed-trajectory local training | PARTIAL | best 4096-episode eval throughout success 0.90625, saturation 0, invalid/NaN 0 |
| H16 strict throughout tracking gate | FAIL | throughout success 0 in trajectory eval |

## Smoke Training Result

| Metric | Value |
| --- | ---: |
| finite | true |
| final loss | 0.0515648 |
| final grad norm | 0.254843 |
| final success rate | 0.0356445 |
| final action saturation | 0 |
| final position error mean | 0.0416385 |
| final velocity error mean | 0.184312 |
| final attitude error mean | 0.0462043 |
| final angular velocity norm mean | 0.582534 |
| NaN/Inf count | 0 |

## H16 Evaluation Summary

| Trajectory | Final success | Throughout success | Position RMSE | Velocity RMSE | Attitude RMSE | Angular velocity RMSE | Saturation | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 0.000000 | 0.000000 | 0.200950 | 0.153228 | 0.027150 | 0.410480 | 0 | 0 |
| step | 0.000000 | 0.000000 | 0.201596 | 0.153228 | 0.027155 | 0.410619 | 0 | 0 |
| circle | 0.000000 | 0.000000 | 0.200978 | 0.158613 | 0.026541 | 0.401760 | 0 | 0 |
| figure8 | 0.000000 | 0.000000 | 0.200914 | 0.160968 | 0.027411 | 0.414351 | 0 | 0 |

## Fixed Local Fixed-Trajectory Progress

The first staged training step was run only on fixed nominal dynamics with the fixed trajectory mode. This is not the final broad-correlated moving-trajectory result, but it verifies that the corrected local initialization and strict H16 local gate are trainable without action saturation.

| Run | Eval episodes | Final success | Throughout success | Position RMSE | Velocity RMSE | Attitude RMSE | Angular velocity RMSE | Max p | Max v | Max attitude | Max w | Saturation | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed local checkpoint 2000 | 4096 | 0.885254 | 0.885254 | 0.028453 | 0.0434195 | 0.0469639 | 0.0246006 | 0.0528287 | 0.15424 | 0.0833874 | 0.0710718 | 0 | 0 |
| fixed local checkpoint 5000 lr1e-4 | 4096 | 0.906494 | 0.906250 | 0.0284419 | 0.0421142 | 0.045596 | 0.0443425 | 0.052958 | 0.149351 | 0.0831466 | 0.207445 | 0 | 0 |

The remaining failures in the best fixed-local run are tail failures against the strict max thresholds, mainly max position slightly above 0.05 m and max angular velocity slightly above 0.2 rad/s. The interrupted `rate2` continuation was stopped and is not used as evidence.

## Current Blockers

- Strict H16 throughout tracking under broad correlated dynamics is not achieved yet.
- Fixed-dynamics fixed-trajectory local tracking is not yet at full strict-window success; best current throughout success is 0.90625.
- Step, circle, figure-eight, mixed, and broad correlated-size-mass stages still need staged training after the fixed-local gate is solved.
- No H64/H128 or RAPTOR teacher/student experiment is allowed from this moving-setpoint pipeline until the strict H16 gate passes.

## Artifacts

```text
reports/h16_moving_setpoint_pipeline/train_smoke_20.csv
reports/h16_moving_setpoint_pipeline/eval_fixed_h16.csv
reports/h16_moving_setpoint_pipeline/eval_step_h16.csv
reports/h16_moving_setpoint_pipeline/eval_circle_h16.csv
reports/h16_moving_setpoint_pipeline/eval_figure8_h16.csv
reports/h16_moving_setpoint_pipeline/train_component_check_1.csv
reports/h16_moving_setpoint_pipeline/train_accel_component_check_1.csv
reports/h16_moving_setpoint_pipeline/eval_local_fixed_metrics_check.csv
reports/h16_moving_setpoint_pipeline/fixed_local_fixedtraj/train_2000.csv
reports/h16_moving_setpoint_pipeline/fixed_local_fixedtraj/eval_2000_fixed_h16.csv
reports/h16_moving_setpoint_pipeline/fixed_local_fixedtraj/train_2000_to_5000_lr1e4.csv
reports/h16_moving_setpoint_pipeline/fixed_local_fixedtraj/eval_5000_lr1e4_fixed_h16.csv
```

The smoke checkpoint is intentionally kept as a runtime artifact and should not be committed:

```text
reports/h16_moving_setpoint_pipeline/checkpoint_smoke_20.ckpt
```
