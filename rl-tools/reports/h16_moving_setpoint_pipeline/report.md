# H16 Moving-Setpoint Trajectory Tracking Pipeline

## Scope

This pass upgrades the CUDA differentiable motor-policy path from fixed-reference stabilization toward time-indexed moving-setpoint tracking while preserving the existing force/torque rigid-body rollout, CUDA VJP chain, GRU actor, and exactly four normalized motor-command outputs.

The implementation is smoke-validated only. The strict H16 moving-setpoint throughout gate is not yet solved, so no H64/H128 or teacher/student experiments were run from this pipeline.

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
| H16 strict throughout tracking gate | FAIL | throughout success 0 in trajectory eval |
| moving-reference full CPU/CUDA loss parity | FAIL | CPU validation path still uses fixed `TrackingReference` |
| CUDA component loss ratios in training CSV | FAIL | total/critic/final metrics logged; per-component ratios still missing |

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

## Current Blockers

- Strict H16 throughout tracking under broad correlated dynamics is not achieved yet.
- Moving-reference CPU/CUDA full loss/VJP parity needs a CPU moving-trajectory reference path, not only fixed-reference parity and moving observation parity.
- CUDA training CSV still needs per-component loss terms and ratios for position, velocity, attitude, rate, action magnitude, smoothness, saturation, and terminal terms.
- No long H16 training was started from this smoke checkpoint.
- No H64/H128 or RAPTOR teacher/student experiment is allowed from this moving-setpoint pipeline until the strict H16 gate passes.

## Artifacts

```text
reports/h16_moving_setpoint_pipeline/train_smoke_20.csv
reports/h16_moving_setpoint_pipeline/eval_fixed_h16.csv
reports/h16_moving_setpoint_pipeline/eval_step_h16.csv
reports/h16_moving_setpoint_pipeline/eval_circle_h16.csv
reports/h16_moving_setpoint_pipeline/eval_figure8_h16.csv
```

The smoke checkpoint is intentionally kept as a runtime artifact and should not be committed:

```text
reports/h16_moving_setpoint_pipeline/checkpoint_smoke_20.ckpt
```
