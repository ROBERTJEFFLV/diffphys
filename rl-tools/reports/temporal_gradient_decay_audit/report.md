# Temporal Gradient Decay Audit

## Scope

This audit checks the active fixed-H16 CUDA differentiable-physics RDAC origin-recovery path. It does not enable Temporal Gradient Decay by default, does not run H32/H64/H128, does not change the GRU actor interface, and does not add physical-parameter inputs.

The active controller remains a GRU actor with exactly four normalized motor commands. The rollout remains the Euler rigid-body force/torque motor model with motor delay and CUDA VJP.

## Implementation Audit

`--temporal-gradient-decay-alpha` is connected to the active CUDA rollout/VJP path.

Relevant source locations:

- `src/foundation_policy/diff_pre_training/cuda_main.cu`
  - `Options::temporal_gradient_decay_alpha` stores the CLI value.
  - `--temporal-gradient-decay-alpha A` parses into the option.
  - `make_full_training_options` passes the value into `FullGpuTrainingOptions`.
  - validation `EulerGpuRunOptions` receives the same value.
- `src/foundation_policy/diff_pre_training/gpu_rollout.h`
  - `EulerGpuRunOptions::temporal_gradient_decay_alpha`
  - `FullGpuTrainingOptions::temporal_gradient_decay_alpha`
- `src/foundation_policy/diff_pre_training/gpu_rollout.cu`
  - `backward_step_kernel(DeviceArrays, batch_size, step_i, temporal_decay_factor)`
  - `run_cpu_reference(...)`
  - `run_euler_gpu_rollout(...)`
  - active full-GPU training VJP loop
  - actor-gradient validation path

The older Stage9 replay debug path still calls `backward_step_kernel(..., 1.0f)`, so it is not an alpha-ablation path. The active full-GPU training and `--gpu-validate-against-cpu` paths are alpha-aware.

## Corrected Scale

The option was already used, but the previous factor was:

```text
exp(-alpha)
```

That is a per-step scale. The requested mathematical form is:

```text
exp(-alpha * dt)
```

The implementation was corrected to use:

```text
exp(-max(0, alpha) * DT)
```

With `alpha=0`, the decay factor is exactly `1.0`, so the default behavior remains unchanged.

## Gradient-Path Audit

The direct loss kernel writes:

- per-step state-loss adjoints into `lambda_p`, `lambda_v`, `lambda_R`, and `lambda_omega`;
- direct per-step action regularization gradients into `action_gradients`.

Temporal decay is applied when the next-state adjoint is read by the transition VJP:

```text
lambda_next_used = exp(-alpha * dt) * lambda_x_next
```

This affects cross-time state-adjoint propagation and the transition-induced action gradient:

```text
lambda_x_t += exp(-alpha * dt) * (df/dx_t)^T lambda_x_{t+1}
dL/du_t    += exp(-alpha * dt) * (df/du_t)^T lambda_x_{t+1}
```

It does not decay same-step action magnitude, smoothness, or saturation gradients, because those are accumulated directly in `loss_and_action_kernel`.

## Validation Commands

Build:

```bash
cmake --build /tmp/raptor_stage96_cuda_build --target foundation_policy_diff_pre_training_cuda -j2
```

CPU/CUDA rollout and action-gradient parity:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 64 \
  --horizon 16 \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --terminal-loss-scale 0 \
  --seed 31001 \
  --temporal-gradient-decay-alpha 0.0

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 64 \
  --horizon 16 \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --terminal-loss-scale 0 \
  --seed 31001 \
  --temporal-gradient-decay-alpha 0.5
```

Short H16 ablation changed only `--temporal-gradient-decay-alpha`:

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
  --seed 32001 \
  --temporal-gradient-decay-alpha <alpha> \
  --save-optimizer \
  --log-path reports/temporal_gradient_decay_audit/dt_scaled/train_alpha_<alpha>.csv \
  --save-path reports/temporal_gradient_decay_audit/dt_scaled/checkpoint_alpha_<alpha>.ckpt
```

H16 eval:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --eval-only \
  --gpu-rollout \
  --load-path reports/temporal_gradient_decay_audit/dt_scaled/checkpoint_alpha_<alpha>.ckpt \
  --eval-horizon 16 \
  --eval-episodes 512 \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --terminal-loss-scale 0 \
  --seed 33001 \
  --temporal-gradient-decay-alpha <alpha> \
  --log-path reports/temporal_gradient_decay_audit/dt_scaled/eval_h16_alpha_<alpha>.csv
```

## CPU/CUDA Parity

| alpha | CPU/CUDA rollout parity | max loss abs err | action grad L2 rel err | NaN/Inf |
| ---: | --- | ---: | ---: | ---: |
| 0.0 | PASS | 9.53674e-06 | 5.32388e-06 | 0 |
| 0.5 | PASS | 9.53674e-06 | 5.30592e-06 | 0 |

## H16 Ablation Results

| alpha | train finite | train NaN/Inf | train weighted cost | train pos | train vel | train att | train omega | action cost | actor grad | eval weighted | eval pos | eval vel | eval omega | eval NaN/Inf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | true | 0 | 8.69781 | 1.88936 | 1.05679 | 5.2344 | 0.516853 | 0.00039675 | 0.0238594 | 8.77287 | 1.89352 | 1.12104 | 0.538684 | 0 |
| 0.25 | true | 0 | 8.69786 | 1.88936 | 1.0568 | 5.23448 | 0.516819 | 0.000396723 | 0.0238532 | 8.77294 | 1.89352 | 1.12106 | 0.538652 | 0 |
| 0.5 | true | 0 | 8.69791 | 1.88936 | 1.05682 | 5.23455 | 0.516785 | 0.000396696 | 0.0238473 | 8.773 | 1.89352 | 1.12107 | 0.538622 | 0 |
| 1.0 | true | 0 | 8.69802 | 1.88936 | 1.05684 | 5.23469 | 0.516733 | 0.000396629 | 0.0238364 | 8.77313 | 1.89352 | 1.1211 | 0.538569 | 0 |
| 2.0 | true | 0 | 8.69823 | 1.88936 | 1.05688 | 5.23499 | 0.516603 | 0.000396513 | 0.0238169 | 8.77339 | 1.89353 | 1.12116 | 0.538451 | 0 |

## Metric Availability

The active training/eval CSV remains RAPTOR-style by design. It does not export success-rate fields, final norm fields, or action saturation rate in the active CSV. For this audit:

- finite and NaN/Inf are reported from stdout;
- actor gradient is `grad_norm` from the active training CSV after actor-gradient clipping;
- action saturation is not exported; `action_cost` stayed near zero for every alpha and no invalid values were observed;
- batch success diagnostics are intentionally not used because success-rate terminology was removed from the active RAPTOR-compatible interface;
- per-bin failure concentration was not available from the active alpha-ablation logs.

## Interpretation

After correcting the scale to `exp(-alpha * dt)`, alpha values from `0.25` to `2.0` have only a very small effect over H16. They slightly reduce the actor gradient norm, but do not reduce invalid values, do not improve weighted cost, and do not improve the position or velocity cost. The eval weighted cost becomes marginally worse as alpha increases.

This supports the hypothesis that the current H16 origin-recovery path is not strongly dependent on Temporal Gradient Decay. The actor receives explicit per-step `dL/du_t` injection, and the H16 window is short enough that the state-adjoint path is not producing observed gradient spikes in this diagnostic.

## Recommendation

Keep Temporal Gradient Decay disabled by default:

```text
--temporal-gradient-decay-alpha 0.0
```

Retain it as an optional diagnostic tool for future cases where H16 shows actual gradient spikes, NaN/Inf, or action saturation. Do not enable it for the current fixed-H16 origin-recovery baseline based on these results, because the observed effect is mostly mild gradient attenuation without recovery improvement.
