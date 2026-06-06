# CLF Euler H16 Objective Validation

## Scope

This report validates the H16 CLF-style local objective plumbing in the EULER differentiable physics path.

No residual allocator, geometric controller, actor output change, H32/H64/H128 experiment training, broad sampled long training, or teacher/student experiment was run in this validation pass. A one-step `--horizon 128` smoke was run only to verify runtime override support.

## Horizon And Loss Normalization

- `DIFF_TRAINING_SEQUENCE_LENGTH` remains `128`.
- `DIFF_TRAINING_HORIZON` remains the compile-time maximum alias to `DIFF_TRAINING_SEQUENCE_LENGTH`.
- `DIFF_TRAINING_DEFAULT_HORIZON = 16` is the runtime default.
- `RuntimeOptions::horizon` now defaults to `DIFF_TRAINING_DEFAULT_HORIZON`.
- Command-line horizon override remains supported: CPU runtime, CUDA rollout validation,
  and CUDA full-training smoke all accepted `--horizon 128`.

The EULER rollout loss uses horizon normalization:

```text
normalizer = 1 / horizon
loss = mean_t(loss_t) + optional terminal loss
```

There is no default time-decay weighting in this CLF loss path. Temporal gradient decay is controlled separately by `--temporal-gradient-decay-alpha`, defaults to `0`, and is applied only in CUDA state-adjoint propagation when enabled.

## Wiring

New loss terms are implemented in:

```text
include/rl_tools/rl/environments/l2f/diff_euler_rollout.h
```

The terms are injected inside `rollout_loss_and_gradients(...)` after forward rollout and before reverse VJP:

```text
for step_i in [0, horizon):
    add_state_loss(...)
    add_clf_transition_loss(...)
add_terminal_loss(...)
add_action_loss(...)
reverse VJP through step_vjp(...)
```

New runtime fields and CLI options are wired through CPU/CUDA training options. CUDA training/eval CSVs include:

```text
clf_loss
outward_velocity_loss
attitude_control_loss
```

## Validation Commands

Build:

```bash
cmake --build /tmp/raptor_stage96_cuda_build \
  --target foundation_policy_diff_physics_check foundation_policy_diff_pre_training_cuda foundation_policy_diff_pre_training -j2
```

CLF gradient validation:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_physics_check \
  --diff-model euler \
  --clf-validation
```

Current recheck:

```text
reports/clf_euler_h16_validation/recheck_build_physics.stdout
reports/clf_euler_h16_validation/recheck_physics_check_clf_validation.stdout
```

The recheck build passed. `--clf-validation` intentionally exits nonzero now because finite-difference, local-direction, and attitude sign checks pass, but the bounded motor-action oracle feasibility gates fail.

Default and override horizon smoke:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --steps 1 --fixed-dynamics --batch-size 2

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training \
  --steps 1 --fixed-dynamics --batch-size 2 --horizon 128

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler --gpu-rollout --fixed-dynamics --steps 1 --batch-size 2 \
  --terminal-loss-scale 0

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler --gpu-rollout --fixed-dynamics --steps 1 --horizon 128 \
  --batch-size 2 --terminal-loss-scale 0
```

Current smoke evidence:

```text
reports/clf_euler_h16_validation/current_horizon_smoke/
```

CPU default reports `horizon=16`; CPU override reports `horizon=128`. CUDA full
training reports `active_training_default_horizon=16`; the default run writes
CSV `horizon=16`, and the override run writes CSV `horizon=128`.

CPU/CUDA parity:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout --gpu-validate-against-cpu \
  --batch-size 32 --horizon 16 --fixed-dynamics --terminal-loss-scale 0

/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout --gpu-validate-against-cpu \
  --batch-size 32 --horizon 16 --fixed-dynamics --terminal-loss-scale 0 \
  --w-clf 1.0 --clf-alpha 1.0 --w-outward-velocity 0.25 --w-attitude-control 0.05
```

Objective gradient conflict diagnostic:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-diagnose-objective-gradient-conflicts \
  --batch-size 64 \
  --horizon 16 \
  --fixed-dynamics \
  --terminal-loss-scale 0 \
  --w-clf 0.5 \
  --w-outward-velocity 0.1 \
  --w-attitude-control 0 \
  --diff-rollout-loss-weight 0.00025
```

## Gradient And Local-Direction Validation

All objective action gradients were compared against finite differences. For attitude-control, the finite-difference target was frozen to match the intended target-detached training semantics. The checker now reports two separate gates:

- `fd_status`: analytic `dL/du` agrees with finite difference and actions remain unsaturated.
- `local_direction_status`: a small step along `u - eta * normalized(dL/du)` decreases the scalar loss and does not worsen positive CLF delta or summed `V_next`.

| Objective | FD status | Local-direction status | Safe eta | Cosine | Relative error | Sign matches | Local loss descent | CLF delta descent | V_next descent |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| CLF | PASS | PASS | 0.01 | 0.999947 | 0.010346 | 64/64 | true | true | true |
| outward velocity | PASS | PASS | 1e-6 | 1.000000 | 0.000537 | 64/64 | true | true | true |
| attitude-control | PASS | PASS | 0.01 | 0.999999 | 0.001189 | 64/64 | true | true | true |
| combined | PASS | PASS | 0.01 | 0.999971 | 0.007635 | 64/64 | true | true | true |

Aggregate output:

```text
PASS euler_clf_objective_finite_difference_validation
PASS euler_clf_objective_local_direction_validation
PASS euler_clf_objective_gradient_validation
FAIL euler_clf_objective_validation
```

Interpretation: the new terms have correct action gradients and a local safe negative-gradient direction exists for each objective. The outward-only objective is scale-sensitive: the lowest scalar-loss step at `eta=0.01` increases positive CLF delta and summed next-state Lyapunov energy, but a much smaller step at `eta=1e-6` lowers outward scalar loss without worsening CLF delta or `V_next`. This is a step-size/objective-scale risk, not an adjoint or finite-difference wiring failure.

## Bounded Motor-Action Oracle Feasibility

A diagnostic bounded-action oracle was added to `--clf-validation`. It is diagnostic only and does not introduce a controller into training. At each H16 step it first enumerates normalized motor commands around hover in a bounded range, then refines the best grid action with coordinate-descent search. The selected action minimizes the next-step CLF delta found by this diagnostic search:

```text
delta = V_next - (1 - clf_alpha * dt) * V_prev
```

Result on the local feasibility case:

| Metric | Value |
| --- | ---: |
| feasible transitions | 1 / 16 |
| max CLF delta | 0.0011864491 |
| initial V | 0.057258762 |
| final V | 0.062567189 |
| H16 decay target V | 0.048753433 |
| final V - initial V | 0.0053084269 |
| final V - decay target V | 0.013813756 |
| window decreased | false |
| window decay target met | false |
| max abs action | 0.94999999 |
| action unsaturated | false |
| feasibility status | FAIL |

The refined greedy action search still failed the strict per-transition CLF decrease gate and used actions at the configured bound. It also failed to decrease the Lyapunov energy across the full H16 window: final `V` was above both the initial `V` and the ideal H16 decay target.

A stronger H16 sequence-action oracle was then added. It seeds the sequence with the refined greedy actions and runs bounded coordinate search over the full `H * 4` action sequence to minimize final `V`.

| Sequence Oracle Metric | Value |
| --- | ---: |
| hover-sequence final V | 0.088124558 |
| greedy-seed final V | 0.062567189 |
| optimized final V | 0.055492569 |
| initial V | 0.057258762 |
| H16 decay target V | 0.048753433 |
| optimized final V - initial V | -0.0017661937 |
| optimized final V - decay target V | 0.0067391358 |
| effective H16 window alpha | 0.19563437 |
| alpha margin to requested alpha=1 | -0.80436563 |
| feasible transitions | 12 / 16 |
| accepted coordinate updates | 239 |
| max abs action | 0.94999999 |
| action unsaturated | false |
| window decreased | true |
| window decay target met | false |
| sequence oracle status | FAIL |

For this optimized sequence, the H16 window target is met only if the decay rate is softened. The validation binary now reports the effective H16 window alpha directly:

```text
sequence_oracle_effective_window_alpha=0.19563437
sequence_oracle_requested_alpha=1
sequence_oracle_alpha_margin_to_requested=-0.80436563
```

| Window alpha | Target V | Final V - Target V | Target met |
| ---: | ---: | ---: | --- |
| 0.00 | 0.057258762 | -0.001766193 | true |
| 0.05 | 0.056802406 | -0.001309837 | true |
| 0.10 | 0.056349461 | -0.000856892 | true |
| 0.15 | 0.055899904 | -0.000407335 | true |
| 0.20 | 0.055453711 | 0.000038858 | false |
| 0.25 | 0.055010859 | 0.000481710 | false |
| 0.50 | 0.052845894 | 0.002646675 | false |
| 1.00 | 0.048753418 | 0.006739151 | false |

The sequence oracle shows that an H16 bounded sequence can reduce final Lyapunov energy, but only with actions reaching the configured bound and only for a much softer window target than `clf_alpha=1`. This does not contradict the finite-difference gradient checks; it means the requested one-step CLF condition is not currently proven physically feasible under the Euler motor-delay model and bounded motor commands, and a softened/windowed CLF target should be validated before more training.

Decision: treat CLF as a local regularizer only. Do not claim that the per-transition CLF decrease condition is enforceable across the local state used here. Do not proceed to broad dynamics or long training based on this CLF objective until either:

- the CLF condition is relaxed to a multi-step/window decrease condition, or
- a better bounded oracle/PD feasibility diagnostic proves per-transition feasibility without action saturation.

## Softened Alpha Fixed-Dynamics Pair

Because the sequence oracle reports an effective H16 window alpha near `0.1956`, a paired fixed-dynamics diagnostic compared the original per-transition CLF setting against a softened per-transition alpha. Both runs used the same seed, H16 persistent training, fixed dynamics, 1000 updates, and the same CLF/outward/attitude weights:

```text
--w-clf 1.0
--w-outward-velocity 0.25
--w-attitude-control 0.05
```

Only `--clf-alpha` differed.

Training result:

| Variant | alpha | train passed | train finite | train NaN/Inf | final weighted cost | final position cost | final velocity cost | final angular velocity cost |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| alpha1_pair | 1.0 | false | true | 112 | 3638.35 | 816.697 | 49.7296 | 1.75924 |
| alpha0p2_pair | 0.2 | false | true | 112 | 822.911 | 239.851 | 37.9539 | 4.56622 |

Evaluation result:

| Variant | H | finite | weighted cost | position cost | velocity cost | angular velocity cost | mean final position | mean final velocity | mean final angular velocity | action saturation | NaN/Inf |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alpha1_pair | 16 | true | 8.91449 | 1.93564 | 1.08088 | 0.824479 | 0.469943 | 1.33711 | 1.01506 | 0 | 0 |
| alpha1_pair | 100 | true | 37.8019 | 21.3941 | 10.3005 | 1.82914 | 2.87206 | 4.72524 | 1.32256 | 0 | 0 |
| alpha1_pair | 500 | true | 2682.13 | 2606.1 | 61.3518 | 3.00196 | 31.0431 | 9.4526 | 2.04834 | 0.00570898 | 0 |
| alpha1_pair | 1000 | true | 20093 | 19863.8 | 203.776 | 10.6222 | 89.5813 | 26.5007 | 6.15915 | 0.0751055 | 0 |
| alpha0p2_pair | 16 | true | 8.98261 | 1.93585 | 1.07758 | 0.906783 | 0.469977 | 1.3308 | 1.10146 | 0 | 0 |
| alpha0p2_pair | 100 | true | 37.314 | 20.1926 | 9.88123 | 2.61748 | 2.77547 | 4.77995 | 1.69568 | 0 | 0 |
| alpha0p2_pair | 500 | true | 6830.01 | 6538.09 | 227.693 | 50.6676 | 59.8405 | 27.6739 | 15.5238 | 0.00165039 | 0 |
| alpha0p2_pair | 1000 | true | 143056 | 141563 | 1203.2 | 273.241 | 274.138 | 64.8942 | 21.9784 | 0.241722 | 0 |

This negative result matters. Although `alpha=0.2` matches the sequence-oracle window-feasibility scale better than `alpha=1`, simply reducing the per-transition CLF alpha does not solve the training objective. It improves the final training-row cost and slightly improves H100 eval, but it badly worsens H500/H1000 position, velocity, angular velocity, and H1000 saturation. The evidence points away from "lower alpha in the same per-step CLF penalty" and toward an explicit multi-step/window decrease objective if CLF is kept.

## Explicit Window-CLF Patch And Fixed-Dynamics Check

An optional window-level CLF objective was added to test the next hypothesis without changing the actor, motor interface, or physics rollout:

```text
--w-window-clf W
```

The default is `0`, so existing behavior is unchanged. The objective uses the same Lyapunov energy weights and `--clf-alpha`, but applies one H16-window penalty:

```text
delta_window = V_H - (1 - clf_alpha * dt)^H * V_0
L_window_clf = w_window_clf * ReLU(delta_window)^2
```

The term is implemented in the EULER path and mirrored in CUDA:

```text
include/rl_tools/rl/environments/l2f/diff_euler_rollout.h
src/foundation_policy/diff_pre_training/gpu_rollout.h
src/foundation_policy/diff_pre_training/gpu_rollout.cu
src/foundation_policy/diff_pre_training/cuda_main.cu
src/foundation_policy/diff_pre_training/cli_options.h
src/foundation_policy/diff_pre_training/main.cpp
src/foundation_policy/diff_pre_training/eval_utils.h
```

Validation:

| Check | Result | Evidence |
| --- | --- | --- |
| build | PASS | `window_clf_patch/build_recheck.stdout` |
| zero-weight CPU/CUDA parity | PASS | `window_clf_patch/validate_zero_window.stdout` |
| nonzero window-CLF CPU/CUDA parity | PASS | `window_clf_patch/validate_window_nonzero.stdout` |
| training CSV field | PASS | `window_clf_patch/log_check/train.csv` includes `window_clf_loss`; one-step run writes nonzero `window_clf_loss=4.26088`. |

Important correction: the CUDA training CLI currently defaults to `sample_dynamics=true`.
Any run that does not explicitly pass `--fixed-dynamics` is not a strict fixed-dynamics
run, even if the trajectory mode is `fixed`. The following window-CLF diagnostics were
kept as sampled-default/window diagnostics; they should not be used as proof of the
fixed-dynamics gate.

Sampled-default H16 persistent training, 1000 updates:

```text
--w-window-clf 1.0
--clf-alpha 1.0
--w-outward-velocity 0.25
--w-attitude-control 0.05
```

Training did not pass because NaN/Inf resets still occurred:

| Variant | train passed | train finite | train NaN/Inf | final weighted cost | final position cost | final velocity cost | final angular velocity cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| window_outward_attitude | false | true | 14 | 24.7514 | 2.66146 | 3.02192 | 1.46892 |
| window_outward_attitude_lr5e5 | false | true | 20 | 25.6549 | 2.65521 | 2.80741 | 1.41816 |
| window0p25_outward_attitude | false | true | 14 | 13155.8 | 884.656 | 42.3245 | 0.665227 |

Evaluation for `window_outward_attitude`:

| H | finite | weighted cost | position cost | velocity cost | angular velocity cost | mean final position | mean final velocity | mean final angular velocity | action saturation | NaN/Inf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | true | 9.20358 | 1.84157 | 1.07593 | 0.837848 | 0.457543 | 1.30866 | 1.02611 | 0 | 0 |
| 100 | true | 31.9821 | 18.4428 | 8.13691 | 1.41416 | 2.53818 | 3.82048 | 0.811487 | 0 | 0 |
| 500 | true | 1212.97 | 1182.84 | 25.1002 | 0.729506 | 18.6165 | 6.19258 | 0.532844 | 0.011998 | 0 |
| 1000 | true | 7444.01 | 7376.16 | 63.0064 | 0.564089 | 45.7088 | 10.0705 | 0.563277 | 0.0160469 | 0 |

This is the best sampled-default/window-CLF long-horizon eval result among these variants so far: H1000 is finite, action saturation stays below 5%, and angular velocity remains contained. It is still not an accepted solution because the training run itself reports NaN/Inf resets, and it is not strict fixed-dynamics evidence because `--fixed-dynamics` was not passed. Lowering the learning rate to `5e-5` does not fix that and worsens H500/H1000. Lowering the window weight to `0.25` also does not fix NaN/Inf and worsens H1000 relative to `w=1.0`.

The current conclusion is narrower but useful: explicit H16 window-CLF is a better direction than merely lowering per-transition alpha, but it still needs a stability fix before any broad dynamics staging claim.

Action-gradient clipping was tested on the same window/outward/attitude setup with
`--action-grad-clip 1.0`. This clipped the scaled injected action-gradient norm from
large spikes down to `1.0` and reduced mean action saturation from about `0.0883` to
`0.00389`, but did not remove NaN/Inf resets. The 250-step diagnostic still had 7
non-finite rows and `window_clf_loss` spikes up to `1.90028e+29`. This points to the
window objective becoming too hard on long persistent segments, not merely to a missing
action-gradient clip.

## True Fixed-Dynamics Minimal Re-Run

Because the CUDA default is sampled dynamics, the fixed-dynamics usefulness gate was
re-run with explicit `--fixed-dynamics`. All runs used H16 persistent training, 1000
updates, batch size 8192, terminal loss scale 0, and H16/H100/H500/H1000 fixed eval:

```text
reports/clf_euler_h16_validation/true_fixed_minimal/
```

Training summary:

| Variant | New loss flags | Train RC | Bad rows | Final weighted | Final position | Final velocity | Final angular velocity | Final saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | none | 0 | 0 | 160.791 | 132.078 | 23.6947 | 0.46205 | 0 |
| clf_outward | `--w-clf 1.0 --w-outward-velocity 0.25` | 0 | 0 | 285.017 | 103.844 | 16.96 | 1.05524 | 0 |
| clf_outward_attitude | `--w-clf 1.0 --w-outward-velocity 0.25 --w-attitude-control 0.05` | 0 | 0 | 284.166 | 104.503 | 16.5007 | 0.888434 | 0 |

Fixed eval summary:

| Variant | H | finite | weighted cost | position cost | velocity cost | angular velocity cost | mean final position | mean final velocity | mean final angular velocity | saturation | NaN/Inf |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 16 | true | 8.74012 | 1.78148 | 1.02202 | 0.665814 | 0.440962 | 1.20642 | 0.789307 | 0 | 0 |
| baseline | 100 | true | 32.4358 | 18.2941 | 8.7946 | 0.771194 | 2.48686 | 4.04014 | 0.739028 | 0 | 0 |
| baseline | 500 | true | 2139.47 | 2085.99 | 44.9841 | 0.743494 | 25.6491 | 8.19461 | 1.4322 | 0 | 0 |
| baseline | 1000 | true | 1.67497e+06 | 20378.8 | 172.452 | 1.65441e+06 | 75.6042 | 18.3614 | 1678.13 | 0.00546094 | 0 |
| clf_outward | 16 | true | 8.73457 | 1.78191 | 1.04022 | 0.682158 | 0.441053 | 1.23317 | 0.864183 | 0 | 0 |
| clf_outward | 100 | true | 30.2414 | 16.7448 | 7.4589 | 1.41393 | 2.37738 | 3.6482 | 1.03563 | 0 | 0 |
| clf_outward | 500 | true | 1045 | 1002.7 | 25.3725 | 2.02141 | 17.9665 | 7.2085 | 2.54132 | 0 | 0 |
| clf_outward | 1000 | true | 5.11189e+07 | 8157.72 | 98.6044 | 5.11106e+07 | 63.4542 | 19.5306 | 10211.1 | 0.0035 | 0 |
| clf_outward_attitude | 16 | true | 8.73003 | 1.78197 | 1.03934 | 0.674532 | 0.441077 | 1.23157 | 0.857122 | 0 | 0 |
| clf_outward_attitude | 100 | true | 30.2449 | 16.8775 | 7.51887 | 1.27735 | 2.39258 | 3.64245 | 0.967977 | 0 | 0 |
| clf_outward_attitude | 500 | true | 1019.44 | 978.459 | 25.4382 | 1.45437 | 17.7049 | 7.49532 | 1.92468 | 0 | 0 |
| clf_outward_attitude | 1000 | false | 195488 | nan | nan | nan | 65.6138 | 21.6463 | 383.275 | 0.00334252 | 4 |

Interpretation:

- The CLF+outward variants improve fixed H500 position and velocity versus baseline.
- At H1000, CLF+outward still improves position and velocity costs, but angular velocity
  becomes much worse than baseline.
- Adding attitude-control reduces the H500 angular-velocity cost and lowers the H1000
  mean final angular velocity relative to baseline, but introduces 4 invalid/NaN eval
  samples at H1000.
- Therefore the true fixed-dynamics architecture-usefulness gate is still not passed:
  the objective is connected and useful in the H500 direction, but H1000 rate stability
  and invalid/NaN safety remain unresolved.

## Boundary-Corrected True Fixed Re-Run

After fixing the persistent episode boundary condition, the required fixed-dynamics
three-run matrix was repeated:

```text
reports/clf_euler_h16_validation/true_fixed_minimal_boundary_v2/
```

All three runs used:

```text
--diff-model euler
--gpu-rollout
--fixed-dynamics
--steps 1000
--horizon 16
--batch-size 8192
--terminal-loss-scale 0
```

Training summary:

| Variant | Bad rows | Max segment end | Crossed episode boundary | Final weighted | Final position | Final velocity | Final angular velocity | Final saturation |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 496 | false | 130.848 | 107.134 | 17.9453 | 0.567823 | 0 |
| clf_outward | 0 | 496 | false | 225.761 | 83.6543 | 16.5264 | 1.62347 | 0 |
| clf_outward_attitude | 0 | 496 | false | 278.756 | 87.9841 | 19.752 | 2.03525 | 0 |

Fixed eval summary:

| Variant | H | finite | weighted cost | position cost | velocity cost | angular velocity cost | mean final position | mean final velocity | mean final angular velocity | saturation | NaN/Inf |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 16 | true | 8.87055 | 1.81421 | 1.07524 | 0.715395 | 0.454749 | 1.27373 | 0.871042 | 0 | 0 |
| baseline | 100 | true | 28.9599 | 16.2918 | 7.08096 | 1.09862 | 2.37241 | 3.58747 | 0.837884 | 0 | 0 |
| baseline | 500 | true | 2909.51 | 2819.84 | 80.3817 | 1.38297 | 34.412 | 14.1668 | 2.24851 | 0 | 0 |
| baseline | 1000 | false | 1.56901e+17 | nan | nan | nan | 127.851 | 33.2096 | 8.87293e8 | 0.0176224 | 26 |
| clf_outward | 16 | true | 8.82342 | 1.81292 | 1.06314 | 0.720527 | 0.454169 | 1.25769 | 0.935388 | 0 | 0 |
| clf_outward | 100 | true | 26.0712 | 13.6992 | 5.92925 | 1.57898 | 2.12951 | 3.42461 | 1.10485 | 0 | 0 |
| clf_outward | 500 | true | 2715.75 | 2613.24 | 82.6627 | 6.36807 | 33.4094 | 16.9848 | 4.87787 | 0.000861328 | 0 |
| clf_outward | 1000 | false | 1.58024e35 | nan | nan | nan | 181.211 | 53.6282 | 9.54085e17 | 0.0535933 | 78 |
| clf_outward_attitude | 16 | true | 8.84156 | 1.81296 | 1.06954 | 0.740076 | 0.454151 | 1.26367 | 0.96356 | 0 | 0 |
| clf_outward_attitude | 100 | true | 27.3215 | 14.0997 | 6.3714 | 1.8102 | 2.16457 | 3.6513 | 0.999169 | 0 | 0 |
| clf_outward_attitude | 500 | true | 2457.82 | 2332.28 | 101.204 | 11.2192 | 33.9667 | 19.2631 | 6.54227 | 0.00360742 | 0 |
| clf_outward_attitude | 1000 | false | 6.28009e9 | nan | nan | nan | 235.902 | 75.7037 | 198130 | 0.109897 | 38 |

Interpretation after the boundary fix:

- The three required minimal variants all train without non-finite rows and no longer cross
  the 500-step episode boundary.
- CLF+outward and CLF+outward+attitude still improve H500 position relative to baseline,
  but they increase H500 velocity/angular-velocity costs.
- All three fail H1000 fixed eval. CLF variants reduce neither the H1000 NaN/Inf count nor
  the H1000 saturation relative to baseline. Therefore the minimal architecture-usefulness
  gate remains failed after the corrected persistent boundary.

## Attitude-Control Sign Diagnostic

The first sign diagnostic used a fixed `eta=1e-3`, which was too coarse for roll and pitch. The diagnostic now scans candidate step sizes and reports whether any negative-gradient step lowers the attitude-control loss without worsening final attitude error or final angular-velocity norm.

| Axis | Status | Best safe eta | Base attitude error | Safe-step attitude error | Best-loss attitude error | Base omega norm | Safe-step omega norm | Best-loss omega norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| roll | PASS | 3e-5 | 0.138386 | 0.138391 | 0.149875 | 1.206149 | 1.205787 | 0.848226 |
| pitch | PASS | 3e-5 | 0.111725 | 0.111729 | 0.125603 | 1.206149 | 1.205788 | 0.848375 |
| yaw | PASS | 0.03 | 0.174085 | 0.150588 | 0.150588 | 1.206149 | 0.848422 | 0.848422 |

Decision: the attitude-control adjoint is no longer blocked by a sign failure. It is still scale-sensitive: for roll/pitch, the step that most reduces the attitude-control scalar loss worsens final attitude error, while only a smaller step is metric-safe. Any training run that enables `--w-attitude-control` should use conservative weighting/update scale and compare against the CLF+outward run.

## CPU/CUDA Parity

| Check | Result |
| --- | --- |
| zero new weights CPU/CUDA parity | PASS |
| `--w-clf 1.0 --w-outward-velocity 0.25 --w-attitude-control 0.05` CPU/CUDA parity | PASS |
| log fields present | PASS |

## Fixed-Dynamics Minimal Training

Configuration:

```text
--diff-model euler
--gpu-rollout
--steps 1000
--horizon 16
--batch-size 8192
--fixed-dynamics
--terminal-loss-scale 0
```

The original CLF+outward run disabled `attitude_control` because the earlier fixed-eta sign diagnostic failed. After eta-scan revalidation passed, the missing third CLF+outward+attitude run was added with the same fixed-dynamics H16 setup.

| Variant | Train weighted cost | Train CLF | Train outward | Train attitude-control | Train saturation | Train grad norm | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 177.565 | 0 | 0 | 0 | 0 | 0.222729 | 0 |
| CLF+outward | 682.514 | 26.9262 | 419.217 | 0 | 0 | 0.118975 | 0 |
| CLF+outward+attitude | 174.271 | 4.39185 | 66.9631 | 8.37e-05 | 0 | 0.120773 | 0 |

Fixed-dynamics evaluation:

| Variant | H | Weighted cost | Position cost | Velocity cost | Angular velocity cost | Terminated share | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 16 | 8.73071 | 1.99730 | 1.00698 | 0.737398 | 0 | finite |
| baseline | 100 | 31.2346 | 15.1264 | 7.45041 | 2.75407 | 0 | finite |
| baseline | 500 | 7611.81 | 7311.51 | 278.261 | 7.22343 | 0 | finite |
| baseline | 1000 | 1.70858e22 | NaN | NaN | NaN | 0.445312 | invalid/NaN present |
| CLF+outward | 16 | 8.84140 | 1.99672 | 1.02789 | 0.943514 | 0 | finite |
| CLF+outward | 100 | 36.9977 | 15.6032 | 9.76857 | 3.89345 | 0 | finite |
| CLF+outward | 500 | 7622.21 | 7391.21 | 205.312 | 9.76046 | 0 | finite |
| CLF+outward | 1000 | 8.35045e8 | NaN | NaN | NaN | 0.078125 | invalid/NaN present |
| CLF+outward+attitude | 16 | 9.98157 | 2.02006 | 1.04661 | 0.747368 | 0 | finite |
| CLF+outward+attitude | 100 | 28.7132 | 15.2099 | 6.26151 | 2.11096 | 0 | finite |
| CLF+outward+attitude | 500 | 2452.2 | 2368.81 | 67.3999 | 5.78902 | 0 | finite |
| CLF+outward+attitude | 1000 | 1.77021e9 | NaN | NaN | NaN | 0.109375 | invalid/NaN present |

## Interpretation

- CLF/outward/attitude-control are correctly wired and have correct action gradients.
- Plain CLF+outward does not show a clean H500 position improvement and leaves H1000 invalid.
- Adding the small attitude-control term improves H500 position substantially versus both baseline and CLF+outward, while keeping H500 velocity and angular velocity below baseline and saturation at zero.
- In this initial minimal set, the H1000 eval is still invalid/NaN, so the architecture is not yet validated as a long-horizon solution.
- The CLF+outward+attitude run is useful evidence for fixed-dynamics H500 behavior, but it fails the full usefulness gate before the later rate-containment ablation.

## H1000 Failure Onset

The CLF+outward+attitude checkpoint was re-evaluated on CUDA Euler fixed dynamics with horizons between H500 and H1000:

```text
reports/clf_euler_h16_validation/fixed_minimal/clf_outward_attitude/horizon_failure_sweep/summary.csv
```

| H | Weighted | Position | Velocity | Angular velocity | Terminated |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 550 | 3474.51 | 3372.56 | 82.048 | 9.069 | 0 |
| 600 | 4807.08 | 4682.25 | 99.472 | 13.965 | 0 |
| 650 | 6532.11 | 6378.27 | 121.044 | 20.922 | 0 |
| 700 | 8766.66 | 8571.85 | 147.846 | 34.618 | 0 |
| 750 | 12142 | 11402.4 | 180.471 | 546.437 | 0 |
| 800 | 72275.6 | 15041.1 | 219.614 | 57001.8 | 0 |
| 850 | 5.53122e6 | 19693.4 | 265.688 | 5.51125e6 | 0 |
| 900 | 6.34974e6 | NaN | NaN | NaN | 0.03125 |
| 950 | 3.16796e17 | NaN | NaN | NaN | 0.0546875 |
| 1000 | 1.77021e9 | NaN | NaN | NaN | 0.109375 |

Interpretation: the failure starts as an angular-rate blow-up before the first NaN. H500 is finite, H750 already has a large angular-velocity cost, H800 is dominated by angular-velocity cost, and NaNs appear by H900. This narrows the next fixed-dynamics debugging target to long-horizon attitude/rate containment rather than immediate H16 gradient wiring.

## Eval Action Saturation Diagnostics

CUDA eval logging now appends action diagnostics:

```text
mean_action_magnitude
max_action_magnitude
mean_action_smoothness
max_action_smoothness
max_action_abs
action_saturation_rate
nan_inf_count
finite
```

The CLF+outward+attitude checkpoint was re-evaluated at H500/H800/H1000:

```text
reports/clf_euler_h16_validation/fixed_minimal/clf_outward_attitude/eval_action_metrics/
```

| H | Weighted | Position | Angular velocity | Terminated | Saturation | Max abs action | Mean action magnitude | Max action magnitude | NaN/Inf | Finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 500 | 2452.2 | 2368.81 | 5.78902 | 0 | 0 | 0.916738 | 0.295148 | 1.17993 | 0 | true |
| 800 | 72275.6 | 15041.1 | 57001.8 | 0 | 0.00450439 | 1 | 0.396282 | 2 | 0 | true |
| 1000 | 1.77021e9 | NaN | NaN | 0.109375 | 0.0377128 | 1 | 0.504162 | 2 | 28 | false |

Interpretation: action saturation stays below the 5% safety threshold even at H1000. H1000 failure is therefore not explained by persistent high action saturation. The policy reaches action bounds occasionally as the state diverges, but the dominant failure signature remains angular-rate/state blow-up.

## Eval State-Norm Diagnostics

CUDA eval logging also appends direct state-norm diagnostics:

```text
mean_final_position_norm
mean_final_velocity_norm
mean_final_attitude_error
mean_final_angular_velocity_norm
mean_max_position_norm
mean_max_velocity_norm
mean_max_attitude_error
mean_max_angular_velocity_norm
max_position_norm
max_velocity_norm
max_attitude_error
max_angular_velocity_norm
```

The CLF+outward+attitude checkpoint was re-evaluated at H500/H800/H1000:

```text
reports/clf_euler_h16_validation/fixed_minimal/clf_outward_attitude/eval_state_metrics/
```

| H | Mean final position | Mean final velocity | Mean final angular velocity | Mean max position | Mean max velocity | Mean max angular velocity | Max position | Max velocity | Max angular velocity | Saturation | NaN/Inf | Finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 500 | 32.7249 | 12.5984 | 5.74785 | 32.7411 | 13.792 | 5.98102 | 123.405 | 55.6637 | 15.0988 | 0 | 0 | true |
| 800 | 76.9495 | 26.608 | 239.907 | 77.5255 | 26.7059 | 240.223 | 361.047 | 108.801 | 34885.7 | 0.00450439 | 0 | true |
| 1000 | 120.982 | 34.9972 | 151592 | 121.021 | 36.5267 | 151593 | 597.74 | 159.711 | 1.54072e7 | 0.0377128 | 28 | false |

Interpretation: the H1000 failure is dominated by angular-rate growth. Position and velocity drift are large, but the angular velocity grows by orders of magnitude between H500 and H1000. This supports targeting long-horizon rate/attitude containment before changing dynamics randomization.

## Rate-Containment Ablation

Fixed-dynamics H16 1000-step ablations were run from the same CLF+outward+attitude setup to target the H800-H1000 angular-rate blow-up:

```text
reports/clf_euler_h16_validation/rate_containment_ablation/summary.csv
```

The most relevant variants:

| Variant | Key change |
| --- | --- |
| `rate_x5` | `--w-angular-v 4.0 --w-clf-angular-velocity 4.0` |
| `attctrl_x4` | `--w-attitude-control 0.2 --attitude-control-k-omega 2.0` |
| `rate_x5_attctrl_x4` | both rate and attitude-control increases |
| `rate_x5_attctrl_x4_sat_guard` | combined plus `--w-sat 0.5 --action-saturation-start 0.85` |
| `rate_x5_attctrl_x4_sat_stronger` | combined plus `--w-sat 1.0 --action-saturation-start 0.85` |
| `rate_x3_attctrl_x2_sat_guard` | intermediate tradeoff: `--w-angular-v 2.4 --w-attitude-control 0.1 --w-sat 0.5 --action-saturation-start 0.85` |

| Variant | H | Position | Velocity | Angular velocity | Mean final angular velocity | Max angular velocity | Saturation | NaN/Inf | Finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rate_x5` | 1000 | NaN | NaN | NaN | 1558.33 | 236360 | 0.0533676 | 6 | false |
| `attctrl_x4` | 1000 | NaN | NaN | NaN | 9185.23 | 1.34794e6 | 0.0139321 | 4 | false |
| `rate_x5_attctrl_x4` | 1000 | 136247 | 1201.55 | 12.59 | 5.59842 | 20.6836 | 0.246061 | 0 | true |
| `rate_x5_attctrl_x4_sat_guard` | 1000 | 85133.9 | 799.094 | 29.8103 | 11.8828 | 54.3118 | 0.0581094 | 0 | true |
| `rate_x5_attctrl_x4_sat_stronger` | 1000 | 47912.4 | 393.391 | 30.2536 | 11.568 | 602.332 | 0.00213184 | 0 | true |
| `rate_x3_attctrl_x2_sat_guard` | 1000 | NaN | NaN | NaN | 1.46382e6 | not recorded | 0.00917073 | 20 | false |

Training evidence for the strongest guarded variant:

```text
--w-angular-v 4.0
--w-clf-angular-velocity 4.0
--w-attitude-control 0.2
--attitude-control-k-omega 2.0
--w-sat 1.0
--action-saturation-start 0.85
```

```text
gpu_full_training_passed=true
gpu_full_training_finite=true
gpu_full_training_weighted_cost=485.338
gpu_full_training_nan_inf_count=0
```

Interpretation:

- Increasing rate loss alone or attitude-control alone does not solve H1000 NaN/Inf.
- Combining stronger rate loss and stronger attitude-control makes H1000 finite, but causes excessive action saturation.
- Adding a stronger saturation penalty keeps H1000 finite and reduces saturation to 0.213%, below the 5% gate.
- The intermediate tradeoff `rate_x3_attctrl_x2_sat_guard` is not enough: training still has 2 non-finite rows, H1000 eval has 20 NaN/Inf samples, and angular velocity still blows up.
- The fixed-dynamics NaN/Inf and saturation blockers are resolved by `rate_x5_attctrl_x4_sat_stronger`.
- H1000 position/velocity remain large, so this is not a high-quality long-horizon recovery policy yet. It is a stabilized fixed-dynamics diagnostic candidate, not a sampled-dynamics-ready result.

The strongest guarded candidate was re-run after the persistent episode boundary fix:

```text
reports/clf_euler_h16_validation/rate_containment_boundary_v2/rate_x5_attctrl_x4_sat_stronger/
```

Training stayed finite and respected the corrected 500-step episode boundary:

| Metric | Value |
| --- | ---: |
| bad rows | 0 |
| max segment end | 496 |
| crossed episode boundary | false |
| final weighted cost | 295.577 |
| final position cost | 107.428 |
| final velocity cost | 18.2637 |
| final angular velocity cost | 1.57252 |
| final action saturation | 0 |

Boundary-corrected fixed eval:

| H | finite | weighted cost | position cost | velocity cost | angular velocity cost | mean final position | mean final velocity | mean final angular velocity | saturation | NaN/Inf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | true | 8.88885 | 1.78398 | 1.06451 | 0.598855 | 0.44573 | 1.25543 | 0.785656 | 0 | 0 |
| 100 | true | 28.1767 | 15.8377 | 6.9814 | 0.757329 | 2.34513 | 3.65966 | 0.592173 | 0 | 0 |
| 500 | true | 1862.49 | 1809.77 | 41.1734 | 0.672111 | 28.7808 | 10.8511 | 1.51525 | 0 | 0 |
| 1000 | true | 22514.2 | 22297.6 | 181.909 | 20.8635 | 108.371 | 26.0281 | 11.3707 | 0.000978516 | 0 |

This confirms that the rate/saturation guarded CLF+outward+attitude objective can make
the corrected fixed-dynamics H1000 rollout finite with low saturation and no NaN/Inf.
It still does not solve recovery quality: H1000 position and velocity remain large, so
the result is a fixed-dynamics safety candidate rather than a solved controller.

Boundary-corrected dynamics staging eval for the same checkpoint:

```text
reports/clf_euler_h16_validation/rate_containment_boundary_v2/rate_x5_attctrl_x4_sat_stronger/dynamics_staging_eval/summary.csv
```

| Mode | finite | weighted cost | position cost | velocity cost | angular velocity cost | mean final position | mean final velocity | mean final angular velocity | saturation | NaN/Inf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | true | 120615 | 22910.1 | 177.891 | 97512.7 | 109.961 | 26.386 | 301.112 | 0.00131836 | 0 |
| small | true | 29117.9 | 28189 | 218.611 | 696.018 | 118.505 | 28.5683 | 38.077 | 0.00574121 | 0 |
| broad | false | 2.25734e7 | NaN | NaN | NaN | 212.656 | 51.5057 | 5595.98 | 0.0369692 | 8 |
| broad_correlated | false | 1.16533e24 | NaN | NaN | NaN | 194.057 | 46.3948 | 2.43204e12 | 0.0327713 | 20 |

This preserves the intended staging interpretation: the guarded checkpoint is finite in
fixed and small dynamics, but broad and broad-correlated dynamics reintroduce NaN/Inf and
large angular-rate blow-up. Direct broad production training is therefore still not
justified from this CLF objective.

## Eval Model Comparison For CLF+Outward+Attitude

The same checkpoint was evaluated through CPU `--eval-model euler` and CPU `--eval-model l2f`:

```text
reports/clf_euler_h16_validation/fixed_minimal/clf_outward_attitude/eval_model_compare/summary.csv
```

| Eval model | H | Mean total loss | Mean final position | Mean final velocity | Mean final angular velocity | Invalid/NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| euler | 500 | NaN | NaN | NaN | NaN | 0.914062 |
| euler | 800 | NaN | NaN | NaN | NaN | 1 |
| euler | 900 | NaN | NaN | NaN | NaN | 1 |
| euler | 1000 | NaN | NaN | NaN | NaN | 1 |
| l2f | 500 | 62148.5 | 122.546 | 61.179 | 336.103 | 0 |
| l2f | 800 | 479402 | 565.74 | 296.759 | 484.497 | 0 |
| l2f | 900 | 1.22881e6 | 998.303 | 605.89 | 550.01 | 0 |
| l2f | 1000 | 3.72192e6 | 1797.7 | 1002.08 | 567.514 | 0 |

This comparison is diagnostic but not a clean CPU/CUDA parity claim: CPU Euler eval is already invalid at H500 while CUDA Euler H500 is finite, so the CPU and CUDA eval wrappers are not interchangeable for this checkpoint. Still, CPU L2F shows finite but severe long-horizon drift, which means the H1000 failure is not only an Euler NaN artifact; the closed-loop policy is also not producing a stable long-horizon recovery.

## Objective Conflict Diagnostic

An action-gradient cosine diagnostic was added to `--clf-validation`. It compares physical action gradients from:

- baseline state loss
- action penalty
- CLF loss
- outward velocity loss
- CLF+outward

Result:

| Row/column | state | action penalty | CLF | outward | CLF+outward |
| --- | ---: | ---: | ---: | ---: | ---: |
| state | 1.000 | 0.041 | 0.942 | -0.832 | 0.940 |
| action penalty | 0.041 | 1.000 | 0.045 | 0.124 | 0.047 |
| CLF | 0.942 | 0.045 | 1.000 | -0.688 | 1.000 |
| outward | -0.832 | 0.124 | -0.688 | 1.000 | -0.681 |
| CLF+outward | 0.940 | 0.047 | 1.000 | -0.681 | 1.000 |

Interpretation:

- CLF and the original state loss are strongly aligned in action-gradient space.
- The outward-velocity penalty is strongly opposed to both state loss and CLF in the local H16 window.
- Action penalty is nearly orthogonal and is not the main source of conflict.
- This explains why large outward weights can improve long-horizon drift while worsening short-window training cost.

## Actor/Critic Parameter-Gradient Diagnostic

A CUDA CLI diagnostic was added:

```text
--gpu-diagnose-objective-gradient-conflicts
```

It uses the active CUDA objective structure but computes the actor/GRU parameter-gradient decomposition on the deterministic CPU reference path:

- physical diff loss enters through `dL/du_t` and the actor head;
- critic loss enters through critic output gradients and the shared encoder/GRU trunk;
- transition-consistency is reported as absent from the active CUDA training path because that objective is CPU-main-only in the inspected code.

Result for `--w-clf 0.5 --w-outward-velocity 0.1 --w-attitude-control 0 --diff-rollout-loss-weight 0.00025`:

| Metric | Value |
| --- | ---: |
| diagnostic passed | true |
| finite | true |
| active CUDA transition consistency present | false |
| scaled diff loss | 0.000987158 |
| scaled critic loss | 0.0499925 |
| physics shared-trunk grad norm | 0.00254088 |
| critic shared-trunk grad norm | 0.000238222 |
| combined shared-trunk grad norm | 0.00265386 |
| physics actor-head grad norm | 0.11409 |
| critic actor-head grad norm | 0 |
| critic-head grad norm | 0.00620752 |
| shared-trunk cosine | 0.43794 |
| encoder cosine | 0.556363 |
| GRU input cosine | 0.566434 |
| GRU hidden cosine | 0.559651 |
| GRU h0 cosine | -0.578031 |
| NaN/Inf count | 0 |

The same diagnostic with all new CLF/outward/attitude-control weights set to zero produced the same values on this deterministic validation batch. Interpretation:

- the active CUDA critic gradient is not strongly opposed to the physical diff gradient in the shared encoder/GRU trunk;
- `gru_h0` alone is directionally opposed, but the aggregate shared trunk is positive cosine;
- transition-consistency should not be cited as an active CUDA conflict source unless it is later ported into the CUDA training update.

## Lower-Weight Fixed-Dynamics Ablation

Because the initial `--w-clf 1.0 --w-outward-velocity 0.25` did not improve H500 position, lower fixed-dynamics ablations were run. All use H16, fixed dynamics, terminal loss scale 0, batch size 8192, and 1000 training steps.

| Variant | Train pass | Train weighted | Train CLF | Train outward | Train NaN/Inf | H500 weighted | H500 position | H500 velocity | H500 angular velocity | H1000 terminated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | PASS | 177.565 | 0 | 0 | 0 | 7611.81 | 7311.51 | 278.261 | 7.223 | 0.445 |
| CLF 1.0 + outward 0.25 | PASS | 682.514 | 26.926 | 419.217 | 0 | 7622.21 | 7391.21 | 205.312 | 9.760 | 0.078 |
| outward 0.25 | FAIL | 453.140 | 0 | 270.184 | 8 | not run | not run | not run | not run | not run |
| CLF 0.1 + outward 0.05 | FAIL | 153.897 | 0.843 | 22.913 | 6 | 4662.25 | 4501.82 | 130.011 | 15.720 | 0.008 |
| CLF 0.25 + outward 0.05 | FAIL | 107.157 | 0.934 | 10.905 | 8 | 1235.82 | 1191.17 | 28.681 | 4.296 | 0.008 |
| CLF 0.5 + outward 0.1 | FAIL | 137.905 | 2.432 | 31.644 | 6 | 1099.76 | 1067.33 | 23.022 | 1.728 | 0.008 |

Interpretation:

- Lower weights are substantially better than the first CLF/outward attempt.
- `CLF 0.5 + outward 0.1` is the best H500 fixed-dynamics result in this batch.
- However, all lower-weight CLF/outward variants still had nonzero training NaN/Inf counts, so none passes the safety gate yet.
- H1000 remains invalid/NaN for all lower-weight variants in this table, although termination share is much lower than the baseline.

## Dynamics Staging Evaluation

The best lower-weight checkpoint, `CLF 0.5 + outward 0.1`, was evaluated across fixed, small, broad, and broad-correlated dynamics.

| Dynamics | H16 weighted | H100 weighted | H500 weighted | H500 position | H500 velocity | H500 angular velocity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 8.37685 | 24.5611 | 1093.71 | 1059.91 | 24.512 | 1.857 |
| small | 8.84680 | 25.3726 | 1241.00 | 1202.15 | 28.863 | 2.079 |
| broad | 8.61870 | 41.4776 | 4.75378e8 | 4887.14 | 116.261 | 4.75374e8 |
| broad correlated | 8.99541 | 34.4838 | 4365.75 | 4246.06 | 100.229 | 8.888 |

Interpretation:

- Fixed and small are close enough to justify staging.
- Broad and broad-correlated remain much harder, especially over H500.
- Broad should not be used yet to judge whether the CLF objective is useful.

Updated staging for the rate/saturation guarded candidate:

```text
reports/clf_euler_h16_validation/rate_containment_ablation/rate_x5_attctrl_x4_sat_stronger/dynamics_staging_eval/summary.csv
```

| Dynamics | H | Weighted | Position | Velocity | Angular velocity | Mean final angular velocity | Max angular velocity | Saturation | NaN/Inf | Finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed | 500 | 3916.82 | 3803.18 | 98.6978 | 1.6022 | 2.72444 | 5.48379 | 0 | 0 | true |
| fixed | 1000 | 48351.7 | 47912.4 | 393.391 | 30.2536 | 11.568 | 602.332 | 0.00213184 | 0 | true |
| small | 500 | 4264.91 | 4139.83 | 109.382 | 2.14198 | 3.15447 | 10.807 | 0 | 0 | true |
| small | 1000 | 55150.8 | 54645 | 461.301 | 28.5614 | 10.8568 | 148.499 | 0.00312793 | 0 | true |
| broad | 500 | 7726.01 | 7485.47 | 222.125 | 3.50105 | 3.03653 | 17.1744 | 0 | 0 | true |
| broad | 1000 | 125036 | NaN | NaN | NaN | 35.3901 | 5887.94 | 0.015394 | 12 | false |
| broad correlated | 500 | 6767.5 | 6567.79 | 178.563 | 4.88749 | 4.12799 | 16.1195 | 0 | 0 | true |
| broad correlated | 1000 | 2.43183e11 | NaN | NaN | NaN | 1.0545e6 | 2.66777e8 | 0.00766996 | 6 | false |

Updated interpretation:

- The guarded candidate transfers from fixed to small dynamics at H1000 without NaN/Inf and keeps saturation below 5%.
- Broad and broad-correlated H1000 still fail with NaN/Inf and angular-rate explosion.
- The next staged step should be small-dynamics training or small-dynamics fine-tuning from the guarded checkpoint, not direct broad training.

## Stabilization Guard Run

To test whether the nonzero training NaN/Inf counts were caused by update scale, the best low-weight candidate was rerun with conservative update settings:

```text
--w-clf 0.5
--w-outward-velocity 0.1
--learning-rate 0.00005
--diff-rollout-loss-weight 0.00025
--actor-grad-clip 50
```

Result:

| Metric | Value |
| --- | ---: |
| training passed | true |
| training finite | true |
| training NaN/Inf count | 0 |
| final weighted cost | 138.032 |
| final position cost | 83.5204 |
| final velocity cost | 12.4418 |
| final angular velocity cost | 0.660014 |
| action saturation rate | 0 |
| final grad norm | 0.154139 |

Evaluation:

| H | Weighted | Position | Velocity | Angular velocity | Terminated | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 8.68050 | 1.95125 | 1.04092 | 0.676458 | 0 | finite |
| 100 | 26.2692 | 14.7599 | 5.92298 | 1.41219 | 0 | finite |
| 500 | 1999.25 | 1925.27 | 62.1924 | 2.30785 | 0 | finite |
| 1000 | 1.57103e15 | NaN | NaN | NaN | 0.03125 | invalid/NaN present |

Interpretation:

- Conservative update scale removes training NaN/Inf and passes the fixed-H16 training safety gate.
- H500 remains much better than baseline, but worse than the unguarded `CLF 0.5 + outward 0.1` checkpoint.
- H1000 still fails, so this is not a complete long-horizon solution.

## Euler vs L2F Evaluation

The same `CLF 0.5 + outward 0.1` checkpoint was evaluated with the CPU evaluator on fixed dynamics.

| Eval model | H16 mean total | H100 mean total | H500 mean total | H500 invalid/NaN |
| --- | ---: | ---: | ---: | ---: |
| euler | 18.7622 | 486.616 | NaN | 0.984375 |
| l2f | 18.4927 | 468.45 | 41582.5 | 0 |

Interpretation:

- Euler and L2F are close at H16/H100.
- Long-horizon instability is present in both, but Euler eval becomes numerically invalid by H500 while L2F remains finite and drifts badly.
- This supports continuing to debug long closed-loop behavior before broad randomization or teacher/student experiments.

## Training Structure And Safety Checks

Current active CUDA training is still origin recovery/stabilization:

- `trajectory_mode=fixed`
- `training_reference_mode=origin`
- default training horizon is H16; explicit horizon override is still available
- actor output dimension remains 4 normalized motor commands
- no physical parameters are fed to the actor
- no residual allocator or geometric controller is used

The active CUDA training path is long rollout with short BPTT:

```text
carry p/v/R/omega/rpm/previous_action/hidden
roll out latest H16 segment
backpropagate through latest H16 only
detach and carry final state/hidden
reset after training_episode_steps = 500 or invalid episode reset
```

Persistent boundary correction:

- Before the correction, a 500-step episode could still train a final segment
  `496 -> 512`, because reset happened only after `persistent_episode_step >= 500`.
- The full-training loop now marks reset when the next H16 segment would cross the
  episode boundary:

```text
persistent_episode_step + horizon > training_episode_steps
```

- Smoke evidence:

```text
reports/clf_euler_h16_validation/persistent_boundary_check/train.csv
```

The 40-step fixed-dynamics smoke has `max_segment_end=496`, late segments only
`464 -> 480` and `480 -> 496`, and `has_cross_episode_segment=false`.

Evidence from the lower-weight runs:

- `persistent_episode_training=true`
- final segment `112 -> 128`
- `episode_reset_count=31`
- `terminal_loss_scale=0`, so terminal loss is not dominating these tests
- `action_saturation_rate=0`
- action cost remains tiny relative to position/velocity costs
- action-gradient clipping is disabled in these runs; actor-grad clipping remains enabled

The hover-relative action fields are logged, and action penalty does not appear to be the limiting factor in this fixed-dynamics evidence.

## Hover-Relative Action Magnitude

The original action-magnitude loss penalized absolute normalized motor command:

```text
L_u = w_u_mag * ||u||^2
```

That does penalize the nonzero motor command required to hover. A backward-compatible optional center was added:

```text
L_u = w_u_mag * ||u - action_magnitude_center||^2
```

Runtime options:

```text
--action-magnitude-center VALUE
--hover-relative-action-magnitude
```

The default center remains `0`, so old commands keep the old behavior. In fixed-dynamics CUDA validation, `--hover-relative-action-magnitude` sets the center from the generated batch hover/previous-action value. For broad sampled dynamics, a true per-sample hover center is still future work; the scalar center should only be used as a fixed/nominal diagnostic or training option.

Physics-check evidence:

| Metric | Value |
| --- | ---: |
| hover action normalized | 0.4522779 |
| absolute action-magnitude loss at hover | 0.81822175 |
| absolute action-magnitude grad norm at hover | 0.45227793 |
| centered action-magnitude loss at hover | 0 |
| centered action-magnitude grad norm at hover | 0 |
| hover-relative check | PASS |

CUDA CPU/CUDA parity evidence:

| Run | Result |
| --- | --- |
| default `action_magnitude_center=0` | PASS |
| manual `--action-magnitude-center 0.4522779` | PASS |
| `--hover-relative-action-magnitude` fixed batch center | PASS |

CPU smoke evidence:

```text
reports/clf_euler_h16_validation/cpu_hover_relative_smoke.stdout
reports/clf_euler_h16_validation/cpu_hover_relative_smoke.csv
```

The CPU target parsed `--hover-relative-action-magnitude`, reported `hover_relative_action_magnitude=true`, applied `action_magnitude_center=0.612249`, ran one finite fixed-dynamics H16 step, and wrote the expected action/CLF loss columns.

Additional audit notes:

| Check | Result | Evidence |
| --- | --- | --- |
| active CUDA transition-consistency objective | absent | `transition_consistency_loss` and response-error recurrence are present in CPU `main.cpp`; active CUDA full training uses diff action gradients plus critic output gradients. |
| train/eval response-error consistency | not applicable to active CUDA | The inspected CUDA path does not use CPU response-error channels in the active update. |
| terminal loss domination | ruled out for reported runs | All CLF validation/training/eval commands here used `--terminal-loss-scale 0`. |
| action-gradient clipping zeroing useful `dL/du` | ruled out for diagnostics | The objective-gradient conflict diagnostics used `action_grad_clip_enabled=false`; training logs expose raw action-gradient norm before/after clipping when enabled. CUDA eval action metrics now show H500/H800/H1000 saturation below 5%, so long-horizon failure is not dominated by persistent saturation. |
| action penalty versus hover | fixed-dynamics option added | Default remains absolute action magnitude for backward compatibility. `--action-magnitude-center` and `--hover-relative-action-magnitude` make hover action zero-cost/zero-gradient in fixed/nominal validation; broad sampled dynamics still needs per-sample hover centers before using this as a full sampled-dynamics training claim. |
| CLF feasibility under bounded motor actions | failed diagnostic | The one-step refined oracle fails strict per-transition decrease with only 1 / 16 feasible transitions. The H16 sequence oracle can reduce final `V`, but only with actions reaching the configured bound and only for softened window targets around `alpha <= 0.1956`, not the requested `alpha=1` decay target. Local finite-difference checks prove gradient correctness, not feasibility of the strict one-step CLF decrease condition. |

## Small-Dynamics Fine-Tune Follow-Up

The fixed-dynamics rate/saturation guarded candidate was fine-tuned under small sampled dynamics before any broad run. This was diagnostic only; it does not satisfy the recovery-quality or saturation gates.

Base guarded fixed-dynamics candidate:

```text
reports/clf_euler_h16_validation/rate_containment_ablation/rate_x5_attctrl_x4_sat_stronger/checkpoint.ckpt
```

Small-dynamics fine-tune command shape:

```text
--sample-dynamics
--sampled-dynamics-level small
--balanced-dynamics-sampling
--learning-rate 0.00005
--w-clf 1.0
--w-outward-velocity 0.25
--w-attitude-control 0.2
--attitude-control-k-omega 2.0
--w-angular-v 4.0
--w-clf-angular-velocity 4.0
```

First small-dynamics fine-tune:

| Mode | H1000 finite | H1000 NaN/Inf | H1000 saturation | H1000 position cost |
| --- | --- | ---: | ---: | ---: |
| fixed | true | 0 | 0.0661621 | 28202.8 |
| small | true | 0 | 0.0888828 | 36324.8 |
| broad | true | 0 | 0.160828 | 63206.1 |
| broad correlated | true | 0 | 0.0902822 | 41233.2 |

Stronger saturation version:

```text
--w-sat 2.0
--action-saturation-start 0.80
```

| Mode | H1000 finite | H1000 NaN/Inf | H1000 saturation | H1000 position cost |
| --- | --- | ---: | ---: | ---: |
| fixed | true | 0 | 0.0644287 | 28224.3 |
| small | true | 0 | 0.0799189 | 34241.7 |
| broad | true | 0 | 0.156977 | 60441.6 |
| broad correlated | true | 0 | 0.0888691 | 41387.9 |

This is a meaningful stability improvement over the pre-finetune staging result: broad and broad-correlated H1000 no longer terminate with NaN/Inf. It is not a solved controller. H1000 saturation remains above the 5% guard in every mode, and long-horizon position costs remain very large. Stronger saturation weight helps small/broad position slightly, but does not bring H1000 saturation below the guard.

## Requirement Audit

| Requirement | Status | Evidence |
| --- | --- | --- |
| Runtime default horizon is H16 while compile-time max remains 128 | complete | `DIFF_TRAINING_DEFAULT_HORIZON=16`, `DIFF_TRAINING_SEQUENCE_LENGTH=128`; current CPU and CUDA default smokes report H16, and current CPU/CUDA `--horizon 128` smokes report H128. |
| New objectives are in the EULER differentiable rollout path | complete | `include/rl_tools/rl/environments/l2f/diff_euler_rollout.h` owns CLF/outward/attitude-control loss and adjoint accumulation. |
| New loss fields appear in logging with nonzero CLI weights | complete | `train_log_field_check.csv` includes `clf_loss`, `outward_velocity_loss`, and `attitude_control_loss`. |
| Zero-weight parity | complete | `cuda_zero_weight_parity.stdout` passes CPU/CUDA loss and action-gradient parity with new weights at zero. |
| Nonzero CPU/CUDA parity | complete | `cuda_nonzero_clf_out_att_parity.stdout` passes for `--w-clf 1.0 --w-outward-velocity 0.25 --w-attitude-control 0.05`. |
| Action-gradient finite-difference checks | complete | `--clf-validation` reports finite-difference PASS for CLF, outward velocity, target-detached attitude-control, and combined objectives. |
| Local negative-gradient direction checks | complete with scale warning | CLF, outward velocity, target-detached attitude-control, and combined objectives pass local scalar-loss and CLF-energy direction checks. Outward-only requires a very small safe step (`best_local_eta=1e-6`); larger scalar-loss-improving steps can worsen CLF delta and `V_next`. |
| Attitude-control frame/sign check | complete with scale warning | Eta-scan sign diagnostic passes for roll, pitch, and yaw. Roll/pitch require small safe steps (`best_local_eta=3e-5`) because larger loss-improving steps worsen attitude error. |
| Fixed-dynamics minimal baseline and CLF+outward training | complete after boundary fix | `true_fixed_minimal_boundary_v2/baseline` and `true_fixed_minimal_boundary_v2/clf_outward` ran H16 training and H16/H100/H500/H1000 eval with `max_segment_end=496`. |
| Fixed-dynamics CLF+outward+attitude training | complete after boundary fix | `true_fixed_minimal_boundary_v2/clf_outward_attitude` ran H16 training and H16/H100/H500/H1000 eval. H500 position improves, but H500 velocity/rate costs worsen and H1000 remains invalid. |
| Fixed-dynamics architecture usefulness gate | failed for the required minimal three-run matrix; guarded candidate is safety-only | The boundary-corrected minimal CLF variants fail H1000. The boundary-corrected rate/saturation guarded variant `rate_containment_boundary_v2/rate_x5_attctrl_x4_sat_stronger` makes H1000 finite with NaN/Inf 0 and saturation 0.0979%, but H1000 position/velocity remain large, so it is a stabilized diagnostic candidate rather than a solved recovery policy. |
| Lower-weight and outward-only failure localization | complete | `fixed_weight_ablation` covers outward-only and reduced CLF/outward weights; several runs fail NaN/Inf, guarded scale run is finite but H1000 still fails. |
| Actor/critic/action objective conflict audit | complete | `objective_gradient_conflict_cuda*.stdout` and physics-check cosine diagnostics are recorded. |
| Active task remains origin recovery/stabilization | complete | CUDA training logs and options report fixed trajectory/origin reference; actor output remains four normalized motor commands. |
| Long rollout + short H16 BPTT structure | complete | Persistent training logs carry state/hidden and detach each H16 segment. The boundary fix prevents a final `496 -> 512` segment from crossing the 500-step episode boundary; smoke now has `max_segment_end=496`. |
| Bounded-action CLF feasibility | failed diagnostic | Refined one-step oracle fails per-transition feasibility. H16 sequence oracle can decrease final `V` only with saturated actions and only meets softened window targets around `alpha <= 0.1956`, not the `alpha=1` decay target. |
| Softened per-transition alpha diagnostic | complete negative result | A paired fixed-dynamics run with `--clf-alpha 0.2` improves H100 slightly but worsens H500/H1000 position, velocity, angular velocity, and saturation relative to the same-seed `alpha=1` run. |
| Explicit window-CLF diagnostic | partially passed, training stability incomplete | `--w-window-clf 1.0` passes CPU/CUDA parity and gives a finite sampled-default/window H1000 eval with saturation 1.60% and contained angular velocity. The 1000-step training run still reports NaN/Inf resets, and it was not a strict fixed-dynamics run because `--fixed-dynamics` was not passed. |
| Hover/action penalty and clipping audit | complete for fixed dynamics | Hover-relative action magnitude option passes physics-check and CUDA CPU/CUDA parity; broad sampled dynamics still needs per-sample hover centers before making a full sampled claim. |
| Euler vs L2F comparison | complete with caveat | `fixed_minimal/clf_outward_attitude/eval_model_compare` evaluates the same checkpoint through CPU Euler and CPU L2F. CPU Euler is not interchangeable with CUDA Euler for this checkpoint, but L2F also shows severe finite drift through H1000. |
| Terminal loss domination | complete | Reported validation/training/eval commands use `--terminal-loss-scale 0`. |
| Dynamics staging | complete for diagnosis | Boundary-corrected guarded checkpoint was evaluated fixed -> small -> broad -> broad-correlated. Fixed and small are finite; broad and broad-correlated H1000 fail with NaN/Inf and angular-rate blow-up. Older small-dynamics fine-tuning can make broad finite but still violates saturation/recovery-quality gates. |

## Recommendation

Do not proceed to broad sampled dynamics or long training yet.

The strongest empirical fixed-dynamics safety evidence is now the boundary-corrected rate/saturation guarded CLF+outward+attitude run:

```text
--w-clf 1.0
--w-outward-velocity 0.25
--w-attitude-control 0.2
--attitude-control-k-omega 2.0
--w-angular-v 4.0
--w-clf-angular-velocity 4.0
--w-sat 1.0
--action-saturation-start 0.85
```

It is finite in training, respects the corrected 500-step episode boundary (`max_segment_end=496`), has H1000 NaN/Inf count 0, and keeps H1000 action saturation below 5%. However, H1000 position and velocity remain large, so it should be treated as a fixed-dynamics safety candidate, not as a solved recovery controller. The stricter local-direction check shows that outward-only is CLF-safe only at a very small step size in the validation case: a tiny safe step exists, but larger scalar-loss-improving steps increase positive CLF delta and summed `V_next`.

The bounded motor-action oracle also fails the strict per-transition CLF decrease gate on the local feasibility case. The stronger sequence oracle finds a saturated sequence that lowers final `V`, but it only satisfies a softened window target near `alpha <= 0.1956`, and still does not satisfy the requested `alpha=1` exponential decay target or action-unsaturated gate. A paired `--clf-alpha 0.2` fixed-dynamics training run confirms that simply lowering the per-transition alpha is not enough: it worsens H500/H1000 long-horizon behavior and saturation. The explicit `--w-window-clf 1.0` variant is more promising in fixed eval, but it still reports NaN/Inf resets during training. This means the current CLF implementation is gradient-correct and window-CLF improves the direction of travel, but the objective still has not passed the full fixed-dynamics architecture-usefulness gate.

Next steps should be:

1. Continue with explicit multi-step/window CLF rather than strict per-transition CLF, but first eliminate the fixed-dynamics training NaN/Inf resets. Do not rely on merely lowering `--clf-alpha` in the existing per-transition penalty.
2. If keeping per-transition CLF, improve the feasibility oracle and require it to pass without action saturation before claiming physical feasibility.
3. Continue small-dynamics staging only as a diagnostic refinement path; broad and broad-correlated H1000 can now be made finite after small fine-tune, but they still violate the saturation guard and have poor recovery quality.
4. Do not start direct broad production training until H1000 saturation is below 5% and long-horizon position/velocity costs improve materially.

The attitude-control sign behavior is now revalidated by eta scan, and the required fixed-dynamics CLF+outward+attitude run has been added. The term remains scale-sensitive and should stay in fixed-dynamics refinement until H1000 is finite.
