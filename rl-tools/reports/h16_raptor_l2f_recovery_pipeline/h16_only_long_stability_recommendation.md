# H16-Only Long-Horizon Stability Recommendation

## Scope

This report answers the active question: under the current CUDA RDAC differentiable-physics architecture, what is the most reasonable path toward long-horizon stable origin recovery without increasing the differentiable physics window beyond H16 and without changing the GRU actor architecture.

The active invariants are unchanged:

- Physics BPTT horizon stays fixed at H16.
- The GRU actor outputs exactly four normalized motor commands.
- No H32/H64/H128 curriculum is used.
- No explicit dynamics parameters, FiLM layer, 9D latent, or other architecture change is introduced.
- Non-fixed setpoints remain deployment/evaluation adapters, not active training targets.

## RAPTOR-Compatible Guidance Used

The original RAPTOR interface is compatible with this active task shape:

- Observations are position, rotation matrix, linear velocity, angular velocity, and previous action.
- Position and linear velocity can be expressed as relative offsets to a target.
- The policy is trained for 100 Hz control, so `dt = 0.01`.
- The actor output is four normalized motor commands.

The original RAPTOR training pipeline still relies on many teacher policies and post-training. This report does not claim to replace that pipeline. The only training principle borrowed here is to keep a recurrent motor-level policy and use long rollout cost/return evaluation, while avoiding architecture changes.

## Current Backpropagation Structure

The active persistent H16 training loop is:

```text
reset 500-step episode
roll out latest H16 segment
compute mean H16 loss
run differentiable physics VJP over only that H16 segment
inject dL/du_t into actor outputs
run GRU BPTT over only that H16 segment
Adam update
detach
carry p/v/R/omega/rpm/previous_action/GRU hidden to next segment
reset after 500 steps or invalid state
```

At 100 Hz, H16 covers 0.16 seconds. A 500-step episode covers 5 seconds. Therefore each optimizer update sees 16 / 500 = 3.2% of the full 5-second episode through differentiable physics BPTT. Across an episode, every 0.16-second segment receives an update, but credit assignment never crosses a segment boundary.

The scalar loss is horizon-mean, not horizon-sum:

```text
L = (L_1 + ... + L_16) / 16
```

`--temporal-gradient-decay-alpha` is not a loss time-weight. It only attenuates cross-step state adjoints inside the VJP. The default `alpha = 0` leaves this path unchanged.

## Experimental Evidence

### Persistent H16 Smoke And Dynamics Matrix

Persistent H16 state/hidden carry was verified with:

```text
reports/h16_raptor_l2f_recovery_pipeline/persistent_h16_smoke_verify2/train_persistent_100.csv
```

The episode step advanced through 0, 16, ..., 496 before reset, proving that physical state and detached GRU hidden state were carried across H16 segments.

Evaluation of the persistent smoke checkpoint showed that long-horizon drift exists even in fixed dynamics:

| Mode | H | Weighted cost | Position cost | Velocity cost | Action cost | Invalid/NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 16 | 7.70455 | 1.8177 | 0.935423 | 0.000194 | 0 |
| fixed | 500 | 1804.95 | 1776.21 | 27.4929 | 0.000366 | 0 |
| broad correlated | 16 | 8.72144 | 1.89809 | 1.11447 | 0.000660 | 0 |
| broad correlated | 500 | 2432.55 | 2394.45 | 36.5069 | 0.002040 | 0 |

Conclusion: broad correlated dynamics worsens the problem, but it is not the root cause. Fixed dynamics already drifts at H500.

### 250-Update Loss Design Ablation

All variants started from the same 1500-step checkpoint, used persistent H16 broad correlated dynamics, batch 8192, terminal loss disabled, and were evaluated under the same H16/H100/H500 matrix.

| Variant | H500 weighted | H500 position | H500 velocity | Interpretation |
| --- | ---: | ---: | ---: | --- |
| baseline | 2829.28 | 2784.66 | 43.2751 | Long drift remains. |
| diff_x5 | 2496.38 | 2458.12 | 36.9331 | Some gradient-scale help, not decisive. |
| diff_x10 | 2864.16 | 2818.46 | 44.3628 | Too much scale loses the gain. |
| smooth_x0.5 | 2829.18 | 2784.56 | 43.2725 | Action smoothness was not the blocker. |
| action_mag_x0.5 | 2834.12 | 2789.41 | 43.3592 | Action magnitude was not the blocker. |
| progress_w1 | 2559.39 | 2518.47 | 37.6517 | Small improvement. |
| outward_w1 | 407.125 | 232.189 | 6.69475 | Strongest short diagnostic. |
| vref_k0.5 | 2072.28 | 1944.34 | 126.289 | Position improves but velocity becomes unsafe. |

Conclusion: the main missing signal is not action authority or saturation. The H16 objective lacks a long-horizon directional incentive to keep moving inward. A penalty on outward radial velocity, `max(0, p dot v)^2`, is the strongest local fix among tested options.

### 1000-Update Candidate Training

All variants were trained for 1000 persistent H16 updates from the same checkpoint, then evaluated under the base objective with no outward/progress eval flags.

| Variant | H250 weighted | H250 position | H250 velocity | H500 weighted | H500 position | H500 velocity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 612.715 | 564.347 | 46.0789 | 3320.99 | 3271.94 | 47.5386 |
| outward0p25 | 336.244 | 315.203 | 18.1608 | 604.045 | 564.841 | 36.4076 |
| outward0p5 | 267.916 | 236.177 | 25.4518 | 2804.62 | 2668.07 | 128.782 |
| outward1p0 | 251.368 | 222.390 | 22.3280 | 1426.69 | 1328.94 | 88.7227 |
| outward0p5_diff2 | 301.993 | 264.379 | 30.1199 | 2916.40 | 2732.31 | 171.893 |
| outward0p5_diff5 | 242.026 | 221.254 | 16.6990 | 713.168 | 665.857 | 42.5115 |
| outward0p5_progress0p5 | 545.848 | 463.627 | 66.6961 | 9520.54 | 9099.89 | 396.542 |

Conclusion: `outward0p25` is the best balanced 5-second candidate. Larger outward weights improve H250 but can overdrive H500 velocity or position.

### H500/H1000 Check

The most important 10-second check was run on baseline, `outward0p25`, and `outward0p5_diff5`.

| Variant | Mode | H500 weighted | H500 position | H500 velocity | H1000 weighted | H1000 position | H1000 velocity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | fixed | 2570.46 | 2531.24 | 37.8356 | 5743.36 | 5713.75 | 28.7855 |
| baseline | broad correlated | 3312.38 | 3264.22 | 46.5906 | 7687.32 | 7621.48 | 64.8615 |
| outward0p25 | fixed | 297.861 | 281.170 | 14.6314 | 309.881 | 291.885 | 16.3285 |
| outward0p25 | broad correlated | 661.132 | 618.646 | 39.6120 | 11704.9 | 11507.8 | 193.878 |
| outward0p5_diff5 | fixed | 259.650 | 234.781 | 20.9152 | 1679.71 | 1638.86 | 36.3183 |
| outward0p5_diff5 | broad correlated | 581.506 | 539.537 | 37.3934 | 13144.6 | 12962.8 | 176.800 |

Conclusion: `outward0p25` makes fixed-dynamics H1000 much better and improves broad-correlated H500. It does not solve broad-correlated H1000. That means the active architecture can learn useful long-horizon behavior from H16 segments, but broad correlated robustness still needs staged training/evaluation, not a larger BPTT window.

### 3000-Update Longer Candidates

Two longer runs were tested after the 1000-update evidence.

| Variant | Mode | H500 weighted | H500 position | H500 velocity | H1000 weighted | H1000 position | H1000 velocity | Invalid/termination note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| outward0p25 | fixed | 2032.79 | 1941.03 | 70.8969 | 72610.4 | NaN | NaN | H1000 non-finite/terminated |
| outward0p25 | broad correlated | 2952.30 | 2829.41 | 91.3357 | 4.96949e6 | NaN | NaN | H1000 non-finite/terminated |
| outward0p25_diff2 | fixed | 1767.60 | 1693.17 | 58.1336 | 20732.6 | 20407.0 | 283.284 | finite but bad |
| outward0p25_diff2 | broad correlated | 1244.97 | 1193.19 | 39.4600 | 11293.0 | 11061.9 | 197.929 | finite but bad |

Conclusion: simply running `outward0p25` longer is not safe. The 1000-update candidate was better than the 3000-update candidate. This points to over-optimization of the shaped short-window objective and supports early stopping plus staged dynamics, not unlimited long training on the same configuration.

## Recommendation

The most reasonable H16-only path is:

1. Keep persistent H16 truncated BPTT as the active training structure.
2. Use a conservative outward-velocity penalty as the primary objective fix:

```text
--w-outward-velocity 0.25
--terminal-loss-scale 0
```

3. Keep the default diff rollout scale first:

```text
--diff-rollout-loss-weight 0.0005
```

Use larger diff scale only as a secondary ablation. It helped short H500 in some runs, but did not solve broad-correlated H1000 and can overdrive longer runs.

4. Do not train a single broad-correlated outward run indefinitely. Use early stopping and evaluate every 250-500 updates under the base objective with no outward/progress eval terms.

5. Use sampler-level dynamics staging, not horizon curriculum:

```text
fixed or small dynamics -> broad non-correlated -> broad correlated
```

Advance only when H500 and H1000 base-objective evaluations do not regress. This is not H32/H64/H128 curriculum; H16 stays fixed.

6. Treat broad-correlated H1000 as the active hard gate. H16/H100/H250 can look good while H1000 fails.

7. Do not enable temporal gradient decay by default. It is not a loss decay; it only attenuates state-adjoint propagation. Existing evidence does not show it is the primary fix.

8. Do not use simple proportional velocity reference alone at `k=0.5`. It reduced position drift but produced excessive velocity.

9. Do not reduce action penalties as the first fix. The ablation showed action magnitude and smoothness weights were not the main limiter.

10. Do not add explicit dynamics conditioning, FiLM, latent variables, or teacher/student experiments yet. The current evidence points first to objective design and dynamics staging.

## Concrete Next Training Recipe

Recommended next run:

```bash
/tmp/raptor_stage96_cuda_build/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler \
  --gpu-rollout \
  --steps 1000 \
  --horizon 16 \
  --batch-size 8192 \
  --sample-dynamics \
  --sampled-dynamics-level small \
  --balanced-dynamics-sampling \
  --correlated-size-mass-sampling \
  --terminal-loss-scale 0 \
  --w-outward-velocity 0.25 \
  --seed <fixed-seed> \
  --load-path <current-best-checkpoint> \
  --save-optimizer \
  --log-path <run-dir>/train.csv \
  --save-path <run-dir>/checkpoint.ckpt
```

Evaluate every 250-500 updates under the base objective:

```bash
--eval-horizon 16
--eval-horizon 100
--eval-horizon 250
--eval-horizon 500
--eval-horizon 1000
```

Required modes:

```text
fixed
small
broad
broad correlated
```

Promotion rule:

- Promote from small to broad only if fixed/small H1000 do not regress.
- Promote from broad to broad-correlated only if broad H500 and H1000 remain finite and improve.
- Stop or roll back when H500/H1000 weighted cost regresses, velocity explodes, or NaN/Inf appears.

## Bottom Line

The current architecture does not fail because the actor lacks motor authority or because H16 training is intrinsically useless. The experiments show it can learn much better 5-second behavior from H16 persistent segments when the objective includes an inward-progress signal.

The main failure mode is that uniform H16 origin recovery loss can be locally satisfied while allowing long-horizon radial drift. The best tested fix is a conservative outward-velocity penalty, combined with persistent episodes, base-objective H500/H1000 evaluation, early stopping, and sampler-level dynamics staging.

The current best candidate direction is:

```text
persistent H16 + w_outward_velocity=0.25 + base diff scale + staged dynamics + H500/H1000 early stopping
```

This is the most defensible next path under the no-architecture-change and no-longer-BPTT constraints.
