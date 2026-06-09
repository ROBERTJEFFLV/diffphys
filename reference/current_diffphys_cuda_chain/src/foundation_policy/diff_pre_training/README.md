# Recurrent Differentiable-Physics Actor-Critic Pre-Training

This module is the differentiable-rollout pre-training path for **RDAC / Recurrent DiffPhys AC**: a single recurrent adaptive controller trained across many dynamics with differentiable rigid-body gradients.

The GRU hidden state is the only adaptive dynamics memory. It receives the current observation, previous motor action, and previous physical response-error features through the recurrent input stream. There is no explicit equivalent-dynamics latent policy output and no latent consistency target.

Training may use sampled simulator parameters `theta_j` internally for stable differentiable rigid-body gradients through the rollout model. Equivalent-dynamics calculations may remain as sampled-dynamics diagnostics, but they are not policy outputs and are not training targets.

The actor deployment input must not include raw physical parameters.

The intended closed loop is:

```text
balanced dynamics sampler
-> vectorized L2F/Euler interaction
-> GRU-based adaptive actor
-> differentiable rigid-body rollout
-> actor-critic + differentiable physics joint update
```

The current executable implements the RDAC model skeleton directly: an encoder/GRU trunk, a 4D motor-command `actor_head` that consumes `[o_t, h_t]`, and a scalar `critic_head` trained from rollout returns over `[o_t, h_t]`. It does not train 1000 SAC teachers and does not meta-imitate teacher policies.

## Architecture Target

```text
many dynamics sampler
        |
balanced vectorized L2F rollout as interaction data
        |
Recurrent DiffPhys Actor-Critic
        |
actor outputs motor commands / RPM setpoints
        |
differentiable rigid-body + motor model
        |
K-step differentiable rollout loss
        |
single adaptive controller
```

Critic training can be made asymmetric later by adding `theta_j` to the critic path during training only. The current critic is already separated from the deployed actor path, and it uses `[o_t, h_t]` rather than raw simulator parameters.

Current RDAC head layout:

```text
encoder(o_t, a_{t-1}, e_{t-1}) -> u_t
GRU(u_t, h_{t-1}) -> h_t
actor_head([o_t, h_t]) -> a_t[0:4] motor commands / normalized RPM setpoints
critic_head([o_t, h_t]) -> V_t
```

The actor output is exactly four motor actions. The recurrent hidden state carries all online adaptation information.

## Models

`--diff-model euler` is the default. It uses a simplified semi-implicit Euler quadrotor model with an analytic reverse-mode VJP. The forward and backward paths are intentionally self-consistent and are checked against finite differences for H=1, H=4, and H=16.

`--diff-model l2f_approx` keeps the previous approximate gradient path against the original L2F RK4 forward step. It is useful as a reference, but it is not the default training model because its rollout-level gradient is incomplete.

The original L2F RK4 simulator is still kept for evaluation and later refinement. This MVP does not attempt to implement the full RK4 adjoint/Jacobian.

`theta_j` sampled by the simulator is used by the differentiable model during training. It is not exposed to the actor.

## Euler Model

State:

- `p`: world position
- `v`: world velocity
- `R`: body-to-world rotation matrix
- `omega`: body angular velocity
- `rpm`: internal motor-speed units used by L2F action limits

Action scaling:

```text
setpoint = normalized_action * (action_limit.max - action_limit.min) / 2
         + action_limit.min
         + (action_limit.max - action_limit.min) / 2
```

The setpoint is clamped to `action_limit`. The motor delay is:

```text
tau = rising_tau if setpoint >= rpm else falling_tau
alpha = exp(-dt / tau)
rpm_next = alpha * rpm + (1 - alpha) * setpoint
```

The VJP ignores the derivative through the rising/falling tau switch boundary. Gradient checks choose actions away from that boundary.

The force/torque model uses the same internal L2F dynamics parameters where possible:

- mass
- `J`, `J_inv`
- rotor positions
- rotor thrust directions
- rotor torque directions
- thrust coefficients
- torque constants
- motor time constants
- action limits
- gravity

World acceleration is:

```text
a_world = R * thrust_body / mass + gravity
```

Angular acceleration is:

```text
omega_dot = J_inv * (torque_body - omega x (J * omega))
```

The forward path includes the gyroscopic/Coriolis term. The VJP includes its derivative with respect to `omega`.

Integration:

```text
omega_next = omega + dt * omega_dot
v_next     = v + dt * a_world
p_next     = p + dt * v_next
R_next     = R * Exp(dt * skew(omega_next))
```

The Euler training path uses an SO(3) exponential-map rotation update. `Exp(.)` is orthogonal up to numerical precision, so no re-orthonormalization is needed. The local VJP through the exponential map uses an analytic Rodrigues-form derivative and is validated by unit, transition, and rollout-level finite-difference checks. The SO(3) VJP unit check uses a double-precision finite-difference reference plus absolute-error criteria for near-zero trace-like gradients because relative error is unreliable when finite-difference scalar changes are close to float precision, especially under MSVC. Transition VJP and prefix-horizon rollout checks remain strict and are the main training-safety gates. The previous first-order `R * (I + dt * skew)` update caused orthogonality drift for long horizons or large angular velocity.

## Loss

The RDAC objective is:

```text
L = L_AC
  + lambda_diff * L_K_step_differentiable_rollout
  + lambda_state * L_transition_consistency
  + lambda_smooth * L_action_smoothness
```

`L_AC` is the normal single recurrent actor-critic signal for long-horizon return and exploration. It is not a bank of SAC teachers.

`L_diff` is the K-step differentiable rollout loss. Starting from an interaction state `x_t`, the current actor produces motor commands, the differentiable rigid-body/motor model rolls forward, and position, attitude, velocity, angular-velocity, and action costs backpropagate through:

```text
action -> motor response -> thrust/torque -> attitude -> position -> cost
```

`L_transition_consistency` is the optional one-step interaction-model consistency term:

```text
diff_model(x_t, a_t, theta_j) ~= x_{t+1}^{interaction}
```

`L_action_smoothness` and other small mechanical stabilizers should stay minimal. The main policy gradient comes from the K-step differentiable physics rollout.

The smooth stabilization loss is:

```text
L_t =
  w_p   * ||p_t||^2
+ w_v   * ||v_t||^2
+ w_R   * ||R_t - I||^2
+ w_w   * ||omega_t||^2
+ w_mag * ||u_t||^2
+ w_u   * ||u_t - u_{t-1}||^2
+ w_sat * saturation_penalty(u_t)
+ w_terminal * terminal_loss_T
```

The terminal loss is:

```text
terminal_loss_T =
  w_Tp * ||p_T||^2
+ w_Tv * ||v_T||^2
+ w_TR * ||R_T - I||^2
+ w_Tw * ||omega_T||^2
```

Weights live in `src/foundation_policy/diff_pre_training/config.h`.

## Curriculum And Safety Options

Runtime options added for the stabilization MVP:

- `--horizon-curriculum`: uses 16 -> 32 -> 64 -> user `--horizon`, capped by the build horizon.
- `--state-curriculum`: starts from smaller initial position, velocity, attitude, and angular-velocity perturbations, then ramps to the normal sampler.
- `--dynamics-curriculum`: starts sampled-dynamics training from nominal dynamics, then filters broad samples using basic mass, inertia, motor-time-constant, and thrust-to-weight checks.
- `--balanced-dynamics-sampling` / `--disable-balanced-dynamics-sampling`: balanced sampling is enabled by default. Each batch is assigned bins over size/mass scale, thrust-to-weight, torque-to-inertia, and motor-delay root variables before final simulator parameters are generated. This is preferred over independently uniform final-parameter sampling.
- `--actor-grad-clip VALUE` / `--disable-actor-grad-clip`: scales actor parameter gradients dL/dtheta by global norm after BPTT. This is the recommended default stabilizer.
- `--action-grad-clip VALUE` / `--disable-action-grad-clip`: clips rollout-output/action gradients dL/du before BPTT. Optional, not the recommended default.
- `--grad-clip VALUE` / `--disable-grad-clip`: deprecated aliases for `--action-grad-clip` / `--disable-action-grad-clip`. Prints a warning; use the explicit names instead.
- `--action-bound VALUE` / `--disable-action-bound`: clamps actor outputs before normalized action-to-motor-setpoint mapping. The clamp derivative is zero outside the bound for BPTT.
- `--w-v`, `--w-w`, `--w-terminal-v`, `--w-terminal-w`: runtime velocity and angular-velocity weight overrides for stabilization tuning.
- `--terminal-ramp-after-horizon-change`, `--terminal-ramp-min VALUE`, `--terminal-ramp-steps N`: experimental fixed-dynamics diagnostic knobs that ramp terminal velocity/angular-velocity weights after a horizon-curriculum transition. Defaults are backward-compatible unless the flag is enabled.
- `--terminal-ramp-terminal-only`: with terminal ramp enabled, also ramps terminal position and attitude weights.
- `--reset-optimizer-on-curriculum-transition`: experimental fixed-dynamics diagnostic knob that resets Adam state when the horizon or state-curriculum stage bucket changes.

Action convention:

```text
raw actor output
-> optional clamp, default [-1, 1]
-> normalized action u
-> setpoint = u * half_range + action_limit.min + half_range
-> setpoint clamp to L2F action_limit
-> first-order motor delay
```

## Build

```bash
cmake -S . -B /tmp/raptor_diff_check_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRL_TOOLS_ENABLE_TARGETS=ON \
  -DRL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS=ON

cmake --build /tmp/raptor_diff_check_build --target foundation_policy_diff_physics_check -j2
cmake --build /tmp/raptor_diff_check_build --target foundation_policy_diff_pre_training -j2
cmake --build /tmp/raptor_diff_check_build --target rl_zoo_l2f_sac_diff -j2
```

## Checks

Euler model:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_physics_check --diff-model euler
```

Expected: local motor/thrust/torque/acceleration checks pass, and rollout loss gradients pass for H=1, H=4, and H=16.

Approximate L2F model:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_physics_check --diff-model l2f_approx
```

Expected: local physical derivatives pass, but rollout-level gradient checks may fail because the full original L2F RK4 `dx_next/dx` Jacobian is not implemented.

## Training Smoke Tests

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --diff-model euler \
  --steps 10 \
  --fixed-dynamics \
  --horizon 16

/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --diff-model euler \
  --steps 50 \
  --fixed-dynamics \
  --horizon 32 \
  --log-path /tmp/diff_euler_fixed.csv
```

The smoke test checks that loss remains finite, actor gradient norm is nonzero, and the rollout does not explode. It is not evidence of teacher-cost reduction by itself.

## Experiment Commands

Fixed-dynamics curriculum training:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --diff-model euler \
  --steps 4000 \
  --fixed-dynamics \
  --horizon 128 \
  --horizon-curriculum \
  --state-curriculum \
  --actor-grad-clip 100 \
  --log-path /tmp/diff_euler_fixed_curriculum.csv \
  --save-path /tmp/diff_euler_fixed_curriculum_policy.bin
```

Fixed-dynamics velocity-focused run used in the current best result:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --diff-model euler \
  --steps 6000 \
  --fixed-dynamics \
  --horizon 128 \
  --horizon-curriculum \
  --state-curriculum \
  --actor-grad-clip 50 \
  --w-v 1.5 \
  --w-w 1.0 \
  --w-terminal-v 10 \
  --w-terminal-w 6 \
  --log-path /tmp/diff_euler_fixed_curriculum_vstrong.csv \
  --save-path /tmp/diff_euler_fixed_curriculum_vstrong_policy.bin
```

Evaluate on Euler:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --eval-only \
  --load-path /tmp/diff_euler_fixed_curriculum_vstrong_policy.bin \
  --eval-model euler \
  --fixed-dynamics \
  --eval-episodes 100 \
  --eval-horizon 128 \
  --log-path /tmp/eval_fixed_curriculum_vstrong_euler.csv
```

Evaluate on original L2F:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --eval-only \
  --load-path /tmp/diff_euler_fixed_curriculum_vstrong_policy.bin \
  --eval-model l2f \
  --fixed-dynamics \
  --eval-episodes 100 \
  --eval-horizon 128 \
  --log-path /tmp/eval_fixed_curriculum_vstrong_l2f.csv
```

No-physics-gradient baseline:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --diff-model euler \
  --steps 1000 \
  --fixed-dynamics \
  --horizon 64 \
  --horizon-curriculum \
  --state-curriculum \
  --disable-physics-gradient \
  --log-path /tmp/no_physics_fixed_curriculum.csv \
  --save-path /tmp/no_physics_fixed_curriculum_policy.bin
```

Hidden-state reset ablation:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --diff-model euler \
  --steps 1000 \
  --fixed-dynamics \
  --horizon 64 \
  --horizon-curriculum \
  --state-curriculum \
  --reset-hidden-each-step \
  --actor-grad-clip 100 \
  --log-path /tmp/reset_hidden_fixed_curriculum.csv \
  --save-path /tmp/reset_hidden_fixed_curriculum_policy.bin
```

Sampled-dynamics curriculum training:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --diff-model euler \
  --steps 4000 \
  --sample-dynamics \
  --horizon 128 \
  --horizon-curriculum \
  --state-curriculum \
  --dynamics-curriculum \
  --actor-grad-clip 100 \
  --log-path /tmp/diff_euler_sampled_curriculum.csv \
  --save-path /tmp/diff_euler_sampled_curriculum_policy.bin
```

Sampled-dynamics evaluation:

```bash
/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --eval-only \
  --load-path /tmp/diff_euler_sampled_curriculum_policy.bin \
  --eval-model euler \
  --sample-dynamics \
  --eval-episodes 100 \
  --eval-horizon 128 \
  --log-path /tmp/eval_sampled_curriculum_euler.csv

/tmp/raptor_diff_check_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --eval-only \
  --load-path /tmp/diff_euler_sampled_curriculum_policy.bin \
  --eval-model l2f \
  --sample-dynamics \
  --eval-episodes 100 \
  --eval-horizon 128 \
  --log-path /tmp/eval_sampled_curriculum_l2f.csv
```

## Logging

Training CSV rows include model, dynamics mode, curriculum state, transition flags, steps since curriculum transitions, effective terminal ramp weights, optimizer-reset flags, physics-gradient enabled/disabled, hidden-reset mode, running and terminal loss terms, action magnitude/smoothness/saturation losses, rollout-output gradient norm before/after clipping, actor gradient norm, initial/final state norms, action norms, action clamp rate, rejected dynamics samples, valid/invalid rollout counts, and `nan_or_inf_flag`.

Evaluation CSV rows include eval model, episodes, horizon, strict success rate, position-only and position-velocity near-success rates, mean total loss, mean/median/p90 final state norms, action norm, and invalid/NaN rate.

The current success thresholds live in `config.h`:

- final position norm < `SUCCESS_POSITION_THRESHOLD`
- final velocity norm < `SUCCESS_VELOCITY_THRESHOLD`
- final angular velocity norm < `SUCCESS_ANGULAR_VELOCITY_THRESHOLD`

## Gradient Stabilization

* **Action-gradient clipping** (`--action-grad-clip`, deprecated alias `--grad-clip`): clips rollout-output gradients dL/du before BPTT. This is optional and not the recommended default.
* **Actor-gradient clipping** (`--actor-grad-clip`): scales actor parameter gradients dL/dtheta by global norm after BPTT. This is the recommended default stabilization mechanism.
* `--grad-clip` is a deprecated alias for `--action-grad-clip`, not `--actor-grad-clip`. Use the explicit names.
* **Hard actor-gradient skip** (`--actor-grad-skip-norm`, default 1e12): skips the optimizer step when raw actor gradient norm exceeds a threshold. This is only a last-resort NaN/Inf/extreme-norm safety guard, not the main stabilizer.

## Stability Guards

The training executable supports optional rollout-output/action-gradient clipping, but the recommended stabilizer is global actor parameter-gradient scaling via `--actor-grad-clip`. Optimizer steps are skipped only for nonfinite losses/gradients or last-resort extreme gradient safety (default skip norm 1e12).

In sampled-dynamics curriculum mode, parameter samples outside simple safety checks are rejected before rollout. Invalid sampled rollouts are logged with `nan_or_inf_flag=true`, skipped for optimizer update, and training continues.

## Current Results

All results below are 100-episode, 128-step evaluations unless stated otherwise. The prior fixed-policy reference was Euler final p/v/w = 4.85451/7.1413/1.76763 and L2F final p/v/w = 4.84627/7.17501/1.82732, both with success 0.

| Policy | Eval model | Mode | Success | Near p | Near p+v | Mean final p | Mean final v | Mean final w | Invalid |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed curriculum | euler | fixed | 0.00 | 0.00 | 0.00 | 2.99911 | 3.82312 | 1.14743 | 0.00 |
| fixed curriculum | l2f | fixed | 0.00 | 0.00 | 0.00 | 3.03741 | 3.92261 | 2.24831 | 0.00 |
| fixed curriculum v-strong | euler | fixed | 0.11 | 0.11 | 0.11 | 2.22573 | 2.26618 | 0.939211 | 0.00 |
| fixed curriculum v-strong | l2f | fixed | 0.09 | 0.10 | 0.09 | 2.44163 | 2.65331 | 1.02248 | 0.00 |
| reset-hidden ablation | euler | fixed | 0.00 | 0.00 | 0.00 | 5.87459 | 9.69469 | 2.3738 | 0.00 |
| sampled curriculum | euler | sampled | 0.06 | 0.07 | 0.06 | 3.37966 | 4.38511 | 1.54187 | 0.00 |
| sampled curriculum | l2f | sampled | 0.02 | 0.03 | 0.02 | 3.56591 | 4.65724 | 2.3701 | 0.00 |

Training notes:

- Fixed curriculum v-strong ran 6000 steps, horizon 16 -> 128, loss 1056.83 -> 517.016, final p/v/w 0.17437/1.03776/5.93396 -> 1.97252/2.37937/0.622829, invalid rollouts 0.
- No-physics-gradient sanity baseline ran 1000 steps with actor gradient norm 0 throughout; it is not a fair RL baseline.
- Sampled curriculum ran 4000 steps, invalid rollouts 0, rejected sampled dynamics 60524. The filtering makes training finite but also shows that the current broad sampled domain is still too aggressive.

## Fixed SO(3) Multi-Seed Repeatability

The fixed-dynamics SO(3) Euler repeatability run used train seeds 0, 1, and 2, evaluation seeds 10000, 10001, and 10002, 6000 actor steps, horizon 16 -> 128 curriculum, state curriculum, actor gradient clip 50, and fixed v-strong weights `w_v=1.5`, `w_w=1.0`, `w_terminal_v=10`, `w_terminal_w=6`.

Artifacts are under `runs/diff_pre_training/fixed_vstrong_multiseed/`. The Euler physics gate reported `overall=STRICT_PASS` before training.

| Train seed | Eval model | Mean success | Std success | Min/max success | Mean final p/v/w | Invalid |
| --- | --- | ---: | ---: | --- | --- | ---: |
| 0 | euler | 0.433333 | 0.00942809 | 0.42 / 0.44 | 1.23295 / 0.727639 / 0.994368 | 0 |
| 0 | l2f | 0.403333 | 0.0169967 | 0.38 / 0.42 | 1.31597 / 0.859992 / 1.23737 | 0 |
| 1 | euler | 0 | 0 | 0 / 0 | 5.58572 / 8.83271 / 5.77791 | 0 |
| 1 | l2f | 0 | 0 | 0 / 0 | 5.68248 / 9.18101 / 5.95081 | 0 |
| 2 | euler | 0.17 | 0.0216025 | 0.15 / 0.20 | 1.677 / 1.40924 / 1.49802 | 0 |
| 2 | l2f | 0.14 | 0.0408248 | 0.09 / 0.19 | 1.78164 / 1.65758 / 1.94389 | 0 |

Overall fixed success was Euler 0.201111 +/- 0.17827 and L2F 0.181111 +/- 0.167207 across training seeds. Two of three seeds beat both previous first-order fixed v-strong success baselines: Euler 0.11 and L2F 0.09. Training had zero invalid rollouts, zero skipped actor steps, and no NaN/Inf flags.

Decision: `PASS` for fixed-dynamics repeatability, but not a strong pass because seed 1 failed. This result only supports fixed-dynamics repeatability. It does not prove teacher-cost reduction, RAPTOR teacher replacement, sampled-dynamics success, Sim2Real transfer, or full L2F dynamics equivalence.

## Fixed SO(3) 10-Seed Stability Audit

The 10-seed audit tests only fixed-dynamics repeatability of the SO(3) Euler differentiable pretraining path. It uses the same fixed v-strong configuration as the 3-seed run, train seeds 0 through 9, evaluation seeds 10000 through 10002, and both Euler and original L2F fixed-dynamics evaluation.

Run command:

```bash
bash src/foundation_policy/diff_pre_training/scripts/run_fixed_10seed_stability_audit.sh
```

Artifacts are under `runs/diff_pre_training/fixed_vstrong_10seed_audit/`. The Euler physics gate reported `overall=STRICT_PASS` before training.

| Metric | Value |
| --- | ---: |
| Completed train seeds | 10 / 10 |
| Completed eval files | 60 / 60 |
| Euler fixed mean/std success | 0.148 / 0.124474 |
| L2F fixed mean/std success | 0.127667 / 0.117266 |
| Seeds beating Euler baseline > 0.11 | 6 / 10 |
| Seeds beating L2F baseline > 0.09 | 6 / 10 |
| Seeds beating both baselines | 6 / 10 |
| Catastrophic zero-success seeds | 1 |
| Total skipped actor steps | 0 |
| Any training NaN/Inf | false |
| Mean invalid_or_nan_rate | 0 |

Failure classification:

| Class | Count |
| --- | ---: |
| PASS_STRONG | 1 |
| PASS_WEAK | 5 |
| TYPE_A_EARLY_DIVERGENCE | 1 |
| TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE | 3 |

Decision: `UNSTABLE_NEEDS_FIX`. Six of ten seeds beat both old success baselines, below the 7/10 acceptable-stability threshold, and one seed had catastrophic zero success. The most common non-pass label is curriculum-triggered collapse, so the recommended next diagnostic is to slow down or freeze horizon/state curriculum before changing physics or the objective.

This audit does not prove teacher-cost reduction, RAPTOR teacher replacement, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Fixed SO(3) Curriculum Stability Fix

The 10-seed audit showed that failed seeds had no NaN/Inf, no invalid rollouts, no skipped actor steps, and little action saturation. The common stress point was curriculum progression into long horizons, where terminal velocity/angular losses and actor gradient norms jumped sharply after horizon increases, especially at H=128.

The smallest supported fix is a slower curriculum:

| Setting | Before | After |
| --- | ---: | ---: |
| `HORIZON_STAGE_STEPS` | 500 | 1000 |
| `STATE_CURRICULUM_STAGE_STEPS` | 500 | 1000 |

Focused validation on the originally failed seeds 1, 3, 6, and 9 improved all four to beating both old fixed first-order baselines:

| Seed | Before Euler/L2F success | After Euler/L2F success | Result |
| ---: | --- | --- | --- |
| 1 | 0 / 0 | 0.523333 / 0.35 | PASS_STRONG |
| 3 | 0.05 / 0.04 | 0.52 / 0.363333 | PASS_STRONG |
| 6 | 0.036667 / 0.016667 | 0.216667 / 0.096667 | PASS_WEAK |
| 9 | 0.023333 / 0.016667 | 0.156667 / 0.10 | PASS_WEAK |

Full 10-seed rerun with the slower curriculum:

| Metric | 500-step curriculum | 1000-step curriculum |
| --- | ---: | ---: |
| Euler fixed mean/std success | 0.148 / 0.124474 | 0.233667 / 0.163105 |
| L2F fixed mean/std success | 0.127667 / 0.117266 | 0.150333 / 0.117042 |
| Seeds beating both baselines | 6 / 10 | 7 / 10 |
| Catastrophic zero-success seeds | 1 | 1 |
| Audit decision | UNSTABLE_NEEDS_FIX | ACCEPTABLE_BUT_UNSTABLE |

Decision: `PARTIAL_FIX`. The slower curriculum meets the minimum stability target but still leaves one catastrophic zero-success seed and three seeds below both baselines. The next targeted experiment should stay on curriculum stability, such as freezing H=64 longer, extending the state curriculum further, or adding a stability-gated curriculum advance. This is still only fixed-dynamics pretraining stability evidence.

## Fixed SO(3) Stability Fix

A follow-up fixed-dynamics stability experiment tested whether the remaining slow-curriculum failures were caused by abrupt terminal velocity/angular-velocity loss pressure after horizon transitions or stale Adam state across curriculum stages. It added diagnostics and two opt-in training knobs, without changing physics, rollout VJP, motor delay, thrust/torque, actor observation, sampled dynamics, original L2F reward, SAC/TD3, teacher code, or HDF5 paths.

Added diagnostics:

- horizon and state-curriculum transition flags
- steps since horizon/state transition
- effective terminal velocity/angular-velocity weights
- terminal ramp multiplier
- optimizer-reset flag

The requested low-level optimizer/model diagnostics `actor_grad_max_abs`, `actor_update_norm`, `actor_parameter_norm`, `adam_m_norm`, `adam_v_norm`, `gru_hidden_norm_mean`, and `gru_hidden_norm_max` remain logged as unavailable because the current model/optimizer traversal does not expose them without broader instrumentation.

Focused seven-seed validation used train seeds 1, 2, 3, 4, 6, 7, and 9:

| Config | Euler success mean/std | L2F success mean/std | Seeds beating both baselines | Catastrophic seeds |
| --- | ---: | ---: | ---: | ---: |
| slow curriculum baseline | 0.230952 / 0.193684 | 0.149524 / 0.134677 | 4 / 7 | 1 |
| terminal v/w ramp | 0.213333 / 0.184950 | 0.136190 / 0.121347 | 3 / 7 | 1 |
| optimizer reset | 0.182381 / 0.195730 | 0.120476 / 0.134877 | 3 / 7 | 2 |
| ramp + reset | 0.163810 / 0.169072 | 0.093810 / 0.114496 | 2 / 7 | 1 |

The terminal ramp and optimizer reset interventions did not improve the focused failure set. The selected full audit therefore reran the best focused configuration, the existing 1000-step slow curriculum, under `runs/diff_pre_training/fixed_vstrong_stabilityfix_10seed_audit/` with the new diagnostics enabled in the logs.

| Metric | 500-step curriculum | 1000-step slow curriculum | selected stability-fix audit |
| --- | ---: | ---: | ---: |
| Euler fixed mean/std success | 0.148 / 0.124474 | 0.233667 / 0.163105 | 0.233667 / 0.163105 |
| L2F fixed mean/std success | 0.127667 / 0.117266 | 0.150333 / 0.117042 | 0.150333 / 0.117042 |
| Seeds beating both baselines | 6 / 10 | 7 / 10 | 7 / 10 |
| Catastrophic zero-success seeds | 1 | 1 | 1 |
| Skipped actor steps | 0 | 0 | 0 |
| Any training NaN/Inf | false | false | false |
| Audit decision | UNSTABLE_NEEDS_FIX | ACCEPTABLE_BUT_UNSTABLE | ACCEPTABLE_BUT_UNSTABLE |

Decision: `NO_FIX_CONFIRMED` for terminal ramp and optimizer reset. The supported state remains the earlier `PARTIAL_FIX`: the slower curriculum improves the original audit from 6/10 to 7/10 seeds beating both fixed first-order baselines, but it does not fully solve repeatability. The next targeted experiment should avoid the rejected ramp/reset path and instead test a cleaner curriculum control, such as freezing H=64 longer, extending the state curriculum further, or implementing stability-gated curriculum advancement.

## Fixed SO(3) H128 Collapse Diagnostics and Fix

A targeted fixed-dynamics audit diagnosed the remaining H128-prioritized failures for seeds 6 and 9. It added phase-aware H128 logging, action-gradient/action-stat diagnostics, unavailable-model-field placeholders, and the analysis script `scripts/analyze_h128_failure_modes.py`. GRU hidden and actor parameter/update diagnostics are still logged as unavailable because the current training path does not expose those aggregate tensors without broader instrumentation.

The baseline H128-prioritized schedule was:

```text
0-499:      H16_easy
500-999:    H32_easy
1000-1499:  H64_easy
1500-3999:  H128_easy
4000-5999:  H128_medium
6000-11999: H128_full
```

Focused baseline results for seeds 2, 3, 6, 7, and 9 were Euler 0.342 +/- 0.258720 and L2F 0.316667 +/- 0.236390, with only 3/5 seeds beating both old fixed first-order baselines and one catastrophic seed. The failure was concentrated after H128: seeds 6/9 had median final pre-H128 phase loss 208.473 versus median final H128 phase loss 1550.03. Actor gradients were clipped on essentially every H128 step for both failed and control seeds, and action-gradient clipping was disabled in the baseline.

Tested interventions:

| Config | Seeds | Euler mean/std success | L2F mean/std success | Seeds beating both baselines | Catastrophic seeds | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline 12000 | 2 3 6 7 9 | 0.342 / 0.258720 | 0.316667 / 0.236390 | 3 / 5 | 1 | rejected |
| balanced 16000 | 2 3 6 7 9 | 0.472 / 0.070193 | 0.441333 / 0.070415 | 5 / 5 | 0 | selected |
| actor grad clip 20 | 2 3 6 7 9 | 0.231333 / 0.231013 | 0.206 / 0.224117 | 2 / 5 | 1 | rejected |
| action grad clip 10 | 2 3 6 7 9 | 0.286667 / 0.136480 | 0.208667 / 0.144969 | 4 / 5 | 0 | partial, rejected |

The selected fix is the H128-prioritized `balanced_16000` schedule:

```text
0-999:      H16_easy
1000-1999:  H32_easy
2000-2999:  H64_easy
3000-5999:  H128_easy
6000-8999:  H128_medium
9000-15999: H128_full
```

It recovered both formerly failing seeds 6 and 9 without changing physics, loss, actor observation, sampled dynamics, original L2F reward, SAC/TD3, teacher code, or HDF5 paths. Full focused validation on seeds 1, 2, 3, 4, 6, 7, and 9 reached 7/7 seeds beating both baselines, with Euler 0.410476 +/- 0.123679 and L2F 0.374286 +/- 0.124871.

The full 10-seed audit under `runs/diff_pre_training/diag_h128prio_16000_10seed/` reached:

| Metric | Value |
| --- | ---: |
| Euler fixed mean/std success | 0.443 / 0.129017 |
| L2F fixed mean/std success | 0.400333 / 0.122651 |
| Seeds beating both baselines | 10 / 10 |
| Catastrophic zero-success seeds | 0 |
| Skipped actor steps | 0 |
| Any training NaN/Inf | false |
| Mean invalid_or_nan_rate | 0 |

Decision: `FIX_CONFIRMED` for fixed-dynamics pretraining stability. This result does not prove teacher-cost reduction, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Sampled-Dynamics Audit After Fixed H128 Stability

The sampled-dynamics audit starts from the fixed H128-prioritized `balanced_16000` checkpoint set under `runs/diff_pre_training/diag_h128prio_16000_10seed/`. The fixed checkpoint manifest is:

```text
runs/diff_pre_training/diag_h128prio_16000_10seed/reports/best_fixed_checkpoint_manifest.csv
```

The audit added sampled-dynamics levels without changing SO(3) equations, transition/VJP code, motor delay, thrust polynomial, torque model, actor observation, original L2F reward, SAC/TD3, RAPTOR teacher code, or HDF5 paths:

| Level | Meaning |
| --- | --- |
| `fixed` | nominal fixed dynamics, no sampling |
| `narrow` | nominal-centered sampled dynamics with variation `0.10`, disturbance force max `0.05`, and the sampled-parameter safety filter |
| `medium` | nominal-centered sampled dynamics with variation `0.25`, disturbance force max `0.15`, and the sampled-parameter safety filter |
| `broad` | existing broad sampled domain; the broad exploratory run used `--dynamics-curriculum` filtering |

The audit launcher and summarizer are:

```text
src/foundation_policy/diff_pre_training/scripts/run_sampled_dynamics_audit.sh
src/foundation_policy/diff_pre_training/scripts/summarize_sampled_dynamics_audit.py
```

All rows below use train seeds 0 through 4, evaluation seeds 10000 through 10002, 100-episode evaluations, horizon 128, Euler and original L2F evaluation, and the fixed v-strong weights `w_v=1.5`, `w_w=1.0`, `w_terminal_v=10`, `w_terminal_w=6`.

| Run | Level | Init | Sampled Euler success | Sampled L2F success | Fixed Euler after run | Fixed L2F after run | Invalid | Rejection |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero-shot fixed checkpoint | narrow | fixed-pretrained | 0.348667 +/- 0.139564 | 0.327333 +/- 0.136941 | not run | not run | 0 | n/a |
| narrow fine-tune | narrow | fixed-pretrained | 0.416000 +/- 0.262876 | 0.375333 +/- 0.247347 | 0.331333 +/- 0.184338 | 0.285333 +/- 0.164986 | 0 | 0 |
| narrow scratch | narrow | scratch | 0.364000 +/- 0.180707 | 0.308667 +/- 0.170263 | 0.260000 +/- 0.139124 | 0.207333 +/- 0.127181 | 0 | 0 |
| medium fine-tune | medium | fixed-pretrained | 0.556000 +/- 0.099742 | 0.519333 +/- 0.113498 | 0.284000 +/- 0.022351 | 0.234667 +/- 0.015434 | 0 | 0 |
| medium scratch | medium | scratch | 0.406667 +/- 0.098161 | 0.346667 +/- 0.103966 | 0.208000 +/- 0.029784 | 0.168667 +/- 0.030955 | 0 | 0 |
| broad fine-tune | broad | fixed-pretrained | 0.276667 +/- 0.063070 | 0.226667 +/- 0.054324 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | 0 | 0.724003 |

Decision: `MEDIUM_PROMISING`. The fixed checkpoint transfers to narrow zero-shot sampled evaluation, and fixed-initialized medium sampled fine-tuning outperforms medium scratch by `+0.149333` Euler success and `+0.172666` L2F success with zero invalid rollouts and zero rejection. The broad sampled domain remains unresolved because it required heavy rejection filtering and collapsed fixed-dynamics retention to zero.

Audit details and bridge notes are:

```text
runs/diff_pre_training/sampled_medium_finetune_from_fixed_12000/reports/sampled_dynamics_audit_report.md
runs/diff_pre_training/sampled_medium_finetune_from_fixed_12000/reports/teacher_cost_bridge_notes.md
```

The current bridge gap for a teacher-cost experiment is checkpoint format and post-training initialization wiring, not actor shape. A future few-teacher comparison should use identical teacher IDs, seeds, and update budgets for scratch versus diff-initialized post-training runs.

## Current Validity

- `euler`: gradient-consistent simplified differentiable model.
- `l2f_approx`: incomplete rollout gradient against original L2F RK4, retained for reference only.
- Original L2F reward and SAC/TD3 core paths are not modified.
- The existing SAC diff prototype remains separate.
- Original L2F evaluation is implemented through `--eval-model l2f`.
- Checkpoint save/load uses a small text format for the RDAC trunk, actor head, and critic head weights, not optimizer state.
- Horizon/state curriculum, terminal loss, action bounding, rollout-output gradient clipping, and sampled-dynamics filtering are implemented for the Euler differentiable pre-training executable.
- Balanced root-dynamics sampling, group-wise rollout weighting, equivalent-dynamics diagnostics, and the recurrent deterministic actor-critic update are implemented in the Euler differentiable pre-training executable. The deployed actor still receives no raw physical parameters.

Allowed claim:

- A gradient-consistent Euler differentiable physics pre-training path exists for the RAPTOR GRU foundation policy.
- In the current fixed-dynamics tests, curriculum plus stronger velocity/angular-velocity weighting improves closed-loop evaluation and reaches nonzero success on both Euler and original L2F fixed-dynamics evaluation.
- In the current medium sampled-dynamics audit, fixed-initialized sampled fine-tuning outperforms scratch on sampled Euler and original L2F evaluation with zero invalid rollouts and zero rejection.
- In the current broad sampled-dynamics exploratory run, sampled evaluation remains nonzero but requires heavy rejection filtering and loses fixed-dynamics retention.
- The architecture direction is now implemented as RDAC in the pre-training executable: one recurrent adaptive controller trained with differentiable rollout gradients, equivalent-dynamics adaptation targets, and a single critic head.

Not allowed yet:

- Teacher-cost reduction is proven.
- A full SAC-style stochastic actor-critic or asymmetric critic with `theta_j` input is complete.
- The RAPTOR 1000-teacher pipeline is fully replaced in all legacy post-training code outside this RDAC pre-training path.
- Sim2real or cross-platform transfer is achieved.
- The simplified Euler policy is a complete controller for original L2F.
- The broad sampled-dynamics domain is solved without filtering.

## Next Steps

1. Add transition-consistency supervision that compares `diff_model(x_t, a_t, theta_j)` against the sampled interaction transition.
2. Add the optional asymmetric critic input path for `theta_j` during training only.
3. Use vectorized original L2F interaction rollouts as the primary data source for `L_AC`, `L_diff`, and `L_state`.
4. Reduce broad sampled-dynamics rejection by improving balanced root-domain sampling and group-wise normalization.
5. Later implement full L2F RK4 adjoint/Jacobian if needed.
