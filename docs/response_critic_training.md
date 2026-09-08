# Short-window physical performance + Risk-to-Go

The deployable `ResponseMotorPolicy` is unchanged: action/response → GRU memory
→ controller → four motor commands. The training-only Critic is one
256 → SiLU → 256 → SiLU → 1 → Softplus MLP. Its complete closed-loop state,
GRU/history tensors, simulator dynamics truth and t/H never enter the Actor.
No teacher, identification head, latent redesign, replay, TD or MS is used here.

## Objectives and units

The original local performance loss retains its weights, action normalization,
2×Huber physical errors, whole-H time averaging and final steady-region weighting.
Only its learned terminal term changes: the Critic now predicts future **risk**.

For each real transition Z_t → Z_(t+1), compute four barriers:

```
B(q, limit) = softplus(kappa * (q / limit - 1)) / kappa
b_t = B(||position error||, p_limit)
    + B(||velocity error||, v_limit)
    + B(||angular velocity error||, omega_limit)
    + mean_over_motors B(|executed command|, saturation_limit)
```

State errors are measured after that transition; saturation uses its executed
normalized command. Hover references are zero. `risk_components` accepts p/v/omega
references, so risk semantics are task-relative. A future tracking task must wire
its references consistently into collection, continuation, Actor loss and gates;
this change does not implement flips or other tracking tasks.

Engineering starting defaults (not experimentally calibrated safety limits):

| Setting | Default |
| --- | --- |
| position error limit | 5 m |
| velocity error limit | 5 m/s |
| angular velocity error limit | 10 rad/s |
| normalized command magnitude limit | 0.95 |
| barrier sharpness kappa | 10 |
| Actor risk weight lambda_r | 1 |
| direction pairs per pooled proposal | at most 4 |
| single-motor perturbation magnitude | at most 0.02 |
| direction loss weight lambda_dir | 0.1 |
| ranking temperature tau | 0.1 |
| minimum true risk gap for a usable pair | 1e-6 |

The barrier is near zero inside limits, smooth at the limit and approximately
linear above it; its large-error slope approaches 1/limit. Softplus is not a hard
safety constraint or a binary failure classifier. Risk uses raw undiscounted
transition sums, with **no division by H, steady-window multiplier or discount**.
Consequently risk magnitudes depend on the flight horizon. Lambda_r controls its
scale relative to the existing time-normalized performance objective.

## One proposal

1. Pool two independent 64-scene TRAIN banks with the existing 4×4
   thrust-to-weight × log-roll-authority stratification. Fix the Actor for the
   entire proposal collection and all gradient windows.
2. Fly continuous H500 without autograd, preserving all physical, memory,
   integral, motor and action-history values. Record Z_0 ... Z_500. Exact
   supervision is R_t = sum(b[t:500]), with R_500 = 0. Select the worst 20% of the
   pooled full-flight **performance** costs once; freeze these CVaR scene weights
   for all ten windows, as in the preceding short-window implementation.
3. Uniformly sample up to four (internal H50 boundary, TRAIN scene) pairs without
   replacement. For each, choose one motor and a random sign. Execute symmetric
   +/- commands around the current Actor command, intersecting amplitude and
   action slew limits. Use the existing `applied_action` hook to record the actual
   executed command and advance memory/history exactly once. Dynamics and
   disturbances are identical across each pair. Clamped motors may yield zero
   perturbations; tied labels do not produce ranking gradients.
4. From the resulting states Z_(t+1)^+/- continue the unchanged Actor without
   gradients to the **original H500**, not a new 500-step horizon. Pair labels
   exclude the perturbation transition itself: they match risk-to-go at Z_(t+1).
   Only those few selected states are branched; the main trajectory is unchanged.
5. Fit the Critic on all current (Z_t, R_t) records with Smooth-L1, plus
   lambda_dir * softplus(-sign(R_true+ - R_true-) * (R_pred+ - R_pred-) / tau)
   averaged over valid pairs. Reuse this small current pair set in each minibatch;
   never store a replay buffer. Targets/signs are detached. Hard-copy the finite
   completed fit into the frozen target Critic.
6. Re-fly the same numerical continuous TRAIN trajectory with ten H50 graphs.
   Each window's per-scene loss is
   `local performance + lambda_r * (local risk + target future risk)`.
   Target parameters are frozen while dR/dZ remains differentiable. The last
   window has exactly zero terminal risk, bypassing the MLP at t=500. Multiply
   every term by the fixed performance-CVaR scene weights and divide each window
   loss by ten. Backpropagate immediately and detach the full state between
   windows, without resetting any values or changing the Actor mid-flight.
7. Clip the accumulated Actor gradient once, then perform exactly one AdamW step.
8. Re-fly continuous H500 from identical TRAIN and fixed DEV initial states.
   TRAIN performance must strictly improve. Each DEV bank's performance may
   increase by at most the existing 0.2% relative tolerance. In TRAIN and each DEV
   bank, true total risk **and each of its four components must not increase**.
   Risk is recomputed from the real candidate trajectory, never from the Critic.
9. Retain a finite completed Critic/target/Adam fit even if the Actor is rejected.
   Actor rejection restores Actor/AdamW and post-fit RNG. A failed/incomplete
   Critic fit restores pre-fit Critic state and RNG. Periodic DEV Actor rollback
   also preserves accumulated Critic learning. Exact resume retains both
   optimizers, target, completed-fit count and Python/NumPy/Torch/CUDA RNG.

## Acceptance aggregation and evidence

Each real flight risk component and their sum are aggregated separately as
`mean + tail_weight * CVaR(top 20%)` across all scenes of the relevant bank.
Their tails are recomputed for acceptance, independently of the frozen Actor
window weights. Component-wise non-deterioration is deliberate: a lower position
barrier cannot compensate for a higher angular-velocity or saturation barrier.
There is no allowed physical-risk deterioration tolerance; deterministic replay
with unchanged Actor compares equally. Existing periodic catastrophic DEV guards
remain additional checks after the per-proposal gates.

Proposal logs preserve `continuous_loss_before/after`,
`development_loss_before/after`, total risk and the four risk component dictionaries
for TRAIN and each DEV bank. `critic_direction_before/after` contain requested
pairs actually collected, valid pairs, ranking loss and sign accuracy on those
**same fitted pairs**. A null accuracy means no usable non-tied pairs.
`critic_loss_before/after` now mean **risk regression** loss, not historical
remaining-performance-cost regression. These metrics must not be compared as
though the labels were unchanged.

`critic_update_retained` and `critic_fits` continue tracking independent Critic
commitment. Value fit, in-sample direction accuracy and finite Actor gradients do
not establish learned performance, gradient accuracy on unseen directions or
safe deployment. The new risk limits, weights and direction sampling budget
still need an explicitly authorized CUDA profile and bounded experiment.

## Fixed input normalization

`PHYSICAL_SCALES` and `POLICY_SCALES` in `response_critic.py` specify a fixed
positive divisor for **every** input field. There is no running RMS or clipping.
Examples: p/5, v/5, omega/10, force/0.5, mass/0.05, arm/0.05,
inertia_x/y/1e-5, inertia_z/2e-5, motor time/0.1, roll authority/500,
yaw authority/100, integral/0.5; rotations, commands, motors and memory use 1.
Both the policy call counter and explicit clock use H. These feature divisors are
fixed independently of configurable risk thresholds. Changing them changes the
source binding, so old Critic weights are never silently reinterpreted.

## Entry points and checkpoints

Use `configs/response_risk_critic_seed7.args` for H500/H50/two-bank CUDA settings.
It intentionally has no update budget: starting an experiment requires an explicit
`--updates` cap. Its work directory is separate from historical cost-Critic runs.

```bash
# Parse and inspect only; no trajectory or optimizer update.
python3 tools/train_response_control.py \
  $(cat configs/response_risk_critic_seed7.args) --dry-run

# Deterministic correctness checks with tiny fixture horizons.
python3 -m pytest tests/test_response_risk.py tests/test_response_critic.py \
  tests/test_response_control.py tests/test_response_training.py \
  tests/test_response_guarded_updates.py -q
```

Risk settings are available as `--risk-*` flags and direction settings as
`--critic-direction-*`; all are bound into checkpoints/configuration and profile
reports. `--mode profile` executes a real proposal; it requires experiment
permission, unlike `--dry-run`. No profile or H500 experiment was run for this
implementation change.

Risk Critic state declares objective `risk-to-go-v1-fixed-physical-scales`.
Old cost-Critic training state cannot be resumed as risk training. To reuse a
compatible historical Actor, explicitly use `--initialize-from <checkpoint>` in
a fresh run directory; both optimizers and the Risk Critic then start fresh.
Omit it for random seed7 Actor initialization. Deployment/evaluation instantiate
only the unchanged Actor. Historical reports and checkpoints remain untouched.
