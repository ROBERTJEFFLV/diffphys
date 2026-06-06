# CLF Euler H16 Objective Patch

## Scope

This patch adds framework support for an H16 CLF-style local energy-decrease objective in the EULER differentiable physics path. It does not change the actor architecture, actor output parameterization, motor command interface, broad dynamics sampler, or CUDA VJP chain.

No residual allocator, mixer logic, geometric controller, desired body-axis construction, yaw reference, or large experiment campaign is introduced.

## Files Modified

```text
src/foundation_policy/diff_pre_training/config.h
src/foundation_policy/diff_pre_training/cli_options.h
src/foundation_policy/diff_pre_training/eval_utils.h
src/foundation_policy/diff_pre_training/main.cpp
src/foundation_policy/diff_pre_training/logging_utils.h
src/foundation_policy/diff_pre_training/gpu_rollout.h
src/foundation_policy/diff_pre_training/gpu_rollout.cu
src/foundation_policy/diff_pre_training/cuda_main.cu
src/foundation_policy/diff_pre_training/physics_check.cpp
include/rl_tools/rl/environments/l2f/diff_euler_rollout.h
include/rl_tools/rl/environments/l2f/diff_euler_check.h
```

## Runtime Horizon Behavior

`DIFF_TRAINING_SEQUENCE_LENGTH` remains `128`, and `DIFF_TRAINING_HORIZON` remains tied to that compile-time maximum.

The runtime default horizon is now separate:

```cpp
constexpr TI DIFF_TRAINING_DEFAULT_HORIZON = 16;
```

`RuntimeOptions::horizon` now defaults to `DIFF_TRAINING_DEFAULT_HORIZON`. Command-line `--horizon` overrides still work, and compile-time tensor sizes are unchanged.

## New Euler Loss Terms

`EulerLossWeights<T>` now includes:

```cpp
clf
window_clf
clf_alpha
clf_position
clf_velocity
clf_attitude
clf_angular_velocity
outward_velocity
attitude_control
attitude_control_k_R
attitude_control_k_omega
```

`EulerLossTerms<T>` now includes:

```cpp
clf
window_clf
outward_velocity
attitude_control
```

`total()`, `add()`, and `scale()` include the new fields.

## CLF Injection Point

The CLF objective is implemented in:

```text
include/rl_tools/rl/environments/l2f/diff_euler_rollout.h
```

inside `rollout_loss_and_gradients(...)`, after the forward rollout and before reverse VJP. It applies a per-transition objective:

```text
V_prev = V(x_t)
V_next = V(x_{t+1})
target_next = (1 - clf_alpha * dt) * V_prev
delta = V_next - target_next

if delta > 0:
    L_clf = clf * delta^2
```

Adjoints are accumulated into both `lambda[step_i]` and `lambda[step_i + 1]`. The Lyapunov energy uses origin/identity-attitude stabilization with Frobenius-style identity rotation error, matching the existing state-loss convention.

An optional window-level CLF objective is also wired through the same EULER path:

```text
V_initial = V(x_0)
V_terminal = V(x_H)
target_terminal = (1 - clf_alpha * dt)^H * V_initial
delta_window = V_terminal - target_terminal

if delta_window > 0:
    L_window_clf = window_clf * delta_window^2
```

This is controlled separately from per-transition CLF through `--w-window-clf`. It defaults to zero, so it does not change existing behavior unless explicitly enabled.

## Other Local Objectives

The outward-velocity penalty is per transition:

```text
radial = (p_next - p_ref) dot (v_next - v_ref)
if radial > 0:
    L_out = outward_velocity * radial^2
```

The attitude-control objective uses:

```text
e_R = 0.5 * vee(R - R^T)
omega_target_next = omega_t + dt * (-k_R * e_R - k_omega * omega_t)
L_attitude_control = attitude_control * ||omega_next - omega_target_next||^2
```

This first implementation is target-detached: gradients are applied only to `omega_next`.

## CUDA Wiring

The active CUDA EULER loss path now has matching fields in:

```text
EulerGpuLossWeights
LossComponentMeans
loss_and_action_kernel
run_cpu_reference
```

The CUDA implementation keeps the motor-policy interface unchanged:

```text
actor -> exactly 4 normalized motor commands
```

The GPU/CPU validation bridge maps the new CUDA weights into `diff::EulerLossWeights`.

## CLI Options

The following options were added to CPU and CUDA entry points:

```text
--w-clf
--w-window-clf
--clf-alpha
--w-clf-position
--w-clf-velocity
--w-clf-attitude
--w-clf-angular-velocity
--w-outward-velocity
--w-attitude-control
--attitude-control-k-r
--attitude-control-k-omega
```

All new loss weights default to zero. The gain defaults are nonzero but inactive unless their corresponding loss weight is nonzero.

## Logging

Training/evaluation loss component logging now includes:

```text
clf_loss
window_clf_loss
outward_velocity_loss
attitude_control_loss
```

Existing logging fields are preserved.

## Validation/Diagnostic Hooks

The physics check target now accepts:

```text
--clf-validation
```

The CUDA target now accepts:

```text
--gpu-diagnose-objective-gradient-conflicts
```

The first validates local finite-difference action gradients for the new EULER objectives and runs a bounded motor-action oracle feasibility diagnostic for the strict per-transition CLF decrease condition. The second reports physical-diff versus critic shared-parameter gradient cosine for the active CUDA objective path. Both are diagnostic-only and do not modify training.

## Build And Smoke Checks

Builds:

```bash
cmake --build /tmp/raptor_stage96_cuda_build --target foundation_policy_diff_pre_training_cuda -j2
cmake --build /tmp/raptor_stage96_cuda_build --target foundation_policy_diff_pre_training -j2
```

CPU default-horizon smoke:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --steps 1 \
  --fixed-dynamics \
  --batch-size 2 \
  --log-path /tmp/clf_cpu_default_horizon_smoke.csv
```

Observed:

```text
horizon=16
mean_clf_loss=0
mean_outward_velocity_loss=0
mean_attitude_control_loss=0
```

CUDA nonzero-loss CPU parity smoke:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 32 \
  --horizon 16 \
  --fixed-dynamics \
  --terminal-loss-scale 0 \
  --w-clf 0.1 \
  --clf-alpha 1.0 \
  --w-outward-velocity 0.1 \
  --w-attitude-control 0.1 \
  --seed 71001
```

Observed:

```text
gpu_validation_passed=true
loss_close=true
action_gradient_close=true
max_loss_abs_error=5.72205e-06
action_gradient_l2_rel_error=2.43203e-09
```

CUDA zero-new-weight CPU parity smoke:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 32 \
  --horizon 16 \
  --fixed-dynamics \
  --terminal-loss-scale 0 \
  --seed 71002
```

CUDA nonzero window-CLF CPU parity smoke:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-against-cpu \
  --batch-size 32 \
  --horizon 16 \
  --fixed-dynamics \
  --terminal-loss-scale 0 \
  --w-window-clf 1.0 \
  --clf-alpha 1.0 \
  --seed 64002
```

Observed:

```text
window_clf_weight=1
gpu_validation_passed=true
```

Observed:

```text
clf_weight=0
outward_velocity_weight=0
attitude_control_weight=0
gpu_validation_passed=true
loss_close=true
action_gradient_close=true
max_loss_abs_error=3.8147e-06
action_gradient_l2_rel_error=7.46818e-09
```

## Assumptions And Limitations

- The CLF reference is origin-centered with identity attitude.
- Rotation gradients use the same Frobenius identity-target convention as the existing attitude loss.
- The attitude-control term is target-detached by design.
- The window-CLF term is an optional diagnostic/training objective and remains disabled by default.
- No residual allocator or geometric controller is included.
- No large training campaign was run in this patch.
- Zero new weights preserve the previous objective behavior except for added zero-valued logging columns and the runtime default horizon change to H16.
