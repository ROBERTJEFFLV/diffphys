# H16 Origin Recovery Position Controller

## Scope

This report tracks the active CUDA diffphys training objective after aligning it with the original RAPTOR/L2F position-controller formulation.

The active training task is fixed-H16 origin recovery:

- The GRU actor still outputs exactly four normalized motor commands.
- The force/torque rigid-body differentiable rollout, CUDA VJP chain, motor delay, previous-action input, and action magnitude/smoothness/saturation penalties are preserved.
- The training reference is the origin for every step: `p_ref = 0`, `v_ref = 0`, identity attitude, and zero angular velocity.
- Initial position, velocity, attitude, and angular velocity are randomized with the recovery-scale defaults.
- Terminal loss remains disabled by default.
- Loss terms are per-step and horizon-normalized over the H16 BPTT window.
- Non-origin waypoint or trajectory behavior is deployment/eval-time setpoint shifting: an external setpoint provider supplies `p_ref(t), v_ref(t)`, and the adapter converts the current state to relative position/velocity offsets before feeding the same GRU actor.

There is no active CUDA training curriculum in this pipeline. H64/H128 training paths and stage-style curriculum entry points are rejected by the active CUDA training interface.

## Active Interface

Training command shape:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler \
  --gpu-rollout \
  --steps N \
  --horizon 16 \
  --batch-size B \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --terminal-loss-scale 0 \
  --log-path reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_active/origin_recovery_train.csv \
  --save-path reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_active/origin_recovery_checkpoint.ckpt
```

The executable now rejects:

- `--stage11`
- `--horizon-curriculum`
- curriculum scale/reset flags
- training with `--horizon` other than `16`
- training with non-fixed `--trajectory-mode`

Evaluation and deployment setpoint shifting may still use `--trajectory-mode fixed|step|circle|figure8|mixed`, because that path is not a training curriculum.

## Metrics

Training CSV logs include:

- finite/NaN diagnostics
- final position, velocity, attitude, and angular velocity diagnostics
- action saturation rate
- action magnitude, smoothness, and saturation loss components
- linear and angular acceleration loss components
- terminal loss components, which should remain zero unless explicitly enabled
- critic loss diagnostics
- gradient norm diagnostics

500-step evaluation logs include:

- `settling_fraction_position`
- mean-error means for position, angle, linear velocity, angular velocity, angular acceleration, action, and action relative
- max-error mean/std for the same quantities
- RMSE diagnostics
- action saturation rate
- invalid/NaN rate

Strict throughout success is retained as a diagnostic only. It is not the active training objective.

## Validation Results

Build:

```bash
cmake --build /tmp/raptor_stage96_cuda_build --target foundation_policy_diff_pre_training_cuda -j2
```

Origin-recovery CPU/CUDA parity:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 64 \
  --horizon 16 \
  --seed 9601 \
  --trajectory-mode fixed \
  --terminal-loss-scale 0 \
  --w-linear-acceleration 0.001 \
  --w-angular-acceleration 0.0001
```

Reject removed H64/H128 curriculum entry point:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --steps 1 \
  --horizon-curriculum 16,32,64,128
```

Reject non-origin training setpoint:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --steps 1 \
  --horizon 16 \
  --trajectory-mode circle
```

Origin-recovery smoke training:

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
  --log-path reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_smoke_train.csv \
  --save-path reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_smoke.ckpt
```

500-step origin-recovery evaluation:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --eval-only \
  --gpu-rollout \
  --load-path reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_smoke.ckpt \
  --eval-horizon 500 \
  --eval-episodes 128 \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --trajectory-mode fixed \
  --log-path reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_smoke_eval_500.csv
```

Deployment setpoint-shifting adapter validation:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-deployment-adapter \
  --batch-size 64 \
  --horizon 16 \
  --seed 9602 \
  --trajectory-mode circle \
  --trajectory-amplitude 0.02 \
  --trajectory-frequency-hz 0.4 \
  --validation-step 8
```

| Check | Result | Evidence |
| --- | --- | --- |
| CUDA build | PASS | `foundation_policy_diff_pre_training_cuda` rebuilt successfully. |
| actor output dimension | PASS | Startup reports `actor_output_dim=4`. |
| origin-recovery CPU/CUDA parity | PASS | `gpu_validation_passed=true`; action-gradient L2 relative error `1.40141e-06`. |
| removed curriculum flag rejection | PASS | `--steps 1 --horizon-curriculum 16,32,64,128` returned nonzero with the fixed-H16 origin-recovery error message. |
| removed H32/H64/H128 training rejection | PASS | `--steps 1 --horizon 32` returned nonzero with the fixed-H16 origin-recovery error message. |
| removed stage entry point rejection | PASS | `--stage11 --stage11-steps 1` returned nonzero with the removed-stage error message. |
| non-origin training setpoint rejection | PASS | `--steps 1 --horizon 16 --trajectory-mode circle` returned nonzero and says non-fixed setpoints are eval/deployment setpoint shifting only. |
| H16 broad correlated origin-recovery smoke training | PASS | 5-step smoke training passed finite, saved checkpoint, NaN/Inf count `0`, final action saturation `0`. |
| 500-step origin-recovery eval diagnostics | PASS | Fixed-dynamics 500-step eval from the smoke checkpoint passed finite and wrote RAPTOR/L2F-style metrics. |
| deployment setpoint-shifting adapter | PASS | Circle setpoint adapter validation passed; max observation error `2.14204e-06`. |

The 5-step smoke checkpoint is not a useful controller. Its broad-correlated 500-step eval intentionally exposed the safety diagnostics: `invalid_or_nan_rate=0.398438` and action saturation `0.342487`. Fixed-dynamics 500-step eval from the same checkpoint remained finite with `invalid_or_nan_rate=0`.

## Smoke Metrics

| Metric | Value |
| --- | ---: |
| train steps | 5 |
| train batch | 256 |
| train horizon | 16 |
| train dynamics | broad + balanced + correlated size-mass |
| final loss | 0.0550651 |
| final grad norm | 0.0479047 |
| final batch success diagnostic | 0.796875 |
| final action saturation | 0 |
| NaN/Inf count | 0 |

500-step fixed-dynamics origin-recovery eval from the smoke checkpoint:

| Metric | Value |
| --- | ---: |
| eval episodes | 128 |
| eval horizon | 500 |
| finite | true |
| invalid/NaN rate | 0 |
| settling fraction position | 0.115644 |
| position mean error mean | 35.4815 |
| position max error mean | 114.467 |
| action saturation rate | 0.249824 |

## Artifacts

```text
reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_active/origin_recovery_smoke_train.csv
reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_active/origin_recovery_smoke_eval_fixed_500.csv
```

Runtime-only artifacts that should not be committed:

```text
reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_active/*.ckpt
reports/h16_raptor_l2f_recovery_pipeline/origin_recovery_active/*.stdout
```
