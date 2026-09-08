# Short-window physics with TRAIN-corrected Actor directions

`ResponseMotorPolicy` is unchanged: action/response → GRU memory → controller →
four motor commands. The Critic is training-only: one 256/256 SiLU MLP with four
Softplus outputs. It sees the complete physical/recurrent state, dynamics truth
and t/H. No privileged input is added to the deployable Actor.

## Risk semantics

Position, velocity and angular-velocity errors retain the reference-relative
softplus barrier `softplus(kappa * (error / limit - 1)) / kappa`. Hover references
are zero. A future tracking task must supply its references consistently in
collection, continuation, gradients and acceptance; flips are not implemented.

Saturation now means approaching the command limit:

```
b_sat = mean_motors(relu((abs(executed_command) - warning) / (1 - warning))**2)
```

The default warning is 0.95 and the normalized hard command limit is 1. The
barrier and its gradient are exactly zero below the warning, including ordinary
hover compensation such as command 0.0741. The per-motor penalty reaches 1 at the
hard limit. Command effort and action changes remain soft performance costs with
the existing weights. They are not individually required to decrease by a safety
gate. Position/velocity/omega limits remain 5 m, 5 m/s and 10 rad/s, respectively;
these are engineering settings, not certified limits.

Risk labels are the four exact undiscounted suffix sums of real transition
barriers, including a zero target at Z_H. They are not performance cost-to-go.
The original physical performance objective retains its weights, Huber physical
errors, horizon normalization and steady-window weighting.

## Fixed TRAIN supervision units

On the first successful fit, all current TRAIN suffix labels set four scales:
`scale[j] = max(mean_over_time_and_scenes(R_true[j]), return_scale_floor)`.
The default floor is 0.001 risk units, including an all-zero component. The scales
remain fixed across proposals and are saved with the Critic and target. Neither
DEV nor later TRAIN batches recalibrate them.

The MLP predicts normalized nonnegative risks; its public output is multiplied
by these scales. Actor terminal values and dR/dZ therefore retain physical risk
units. Regression uses `mean(Huber((prediction - truth) / scale))`. Ranking uses
both predicted and true differences divided by the same scales; its temperature
and minimum non-tie gap are in normalized units. Reports contain both physical MAE
and MAE/scale per component, and per-component direction accuracy/valid counts.
Input normalization still uses the fixed `PHYSICAL_SCALES` / `POLICY_SCALES`
divisors, without running RMS or clipping.

Direction samples are still a few legal +/- motor perturbations followed by
real no-gradient continuation. Immediately after that motor transition the two
memories can be identical: their subsequent responses have not yet entered the
GRU. Accuracy on these fitted pairs is not evidence that all memory gradients
are correct. No extra memory perturbation scheme is added here.

## One proposal

1. Pool two independent 64-scene TRAIN banks using the existing 4×4
   thrust-to-weight × roll-authority stratification. Fly a continuous H500 with
   the current Actor and no autograd. Record every closed-loop state and exact
   component suffix risk. Fit and retain the Critic using only this TRAIN data
   and the small current set of reachable motor-direction labels.
2. Freeze the target Critic parameters while preserving derivatives through its
   inputs. Re-fly the same numerical trajectory as ten H50 graphs. Detach all
   physical/recurrent state at boundaries; reset no values and update no Actor
   parameters during the flight. The last window has no terminal Critic value.
3. Accumulate five separate parameter gradients, averaged across the windows:
   local performance and each local-plus-terminal risk component. Each objective
   freezes its own full-H500 CVaR scene weights; no window chooses a new tail.
4. Reorthogonalize these gradients to an at-most-five-dimensional parameter
   subspace. In that subspace measure central finite differences using real
   continuous H500 **TRAIN** flights from the identical initial conditions.
   The measured rows are performance, four component risks, and total-risk CVaR,
   so the extra total-risk gate is not omitted from direction construction.
5. Project the negative performance gradient onto the cone where all five
   risk rows satisfy `A_risk @ q <= 0`. Enumerate the at-most-32 active risk
   subsets to minimize `||q + a_performance||^2 / 2`; equality constraints are
   valid, so risks may stay unchanged. A nearly zero projection returns
   `no_feasible_direction` for this proposal. Normalize a usable projection
   and verify its performance/risk slopes at both finite-difference scales.
   First-order risk equality still requires the real finite-step trajectory gate.
6. Apply the checked direction directly, without Adam, momentum or weight decay.
   Test at most four decreasing step lengths on TRAIN. Only after TRAIN improves
   and passes every risk gate evaluate the candidate on the two fixed DEV banks.
   A DEV rejection ends the proposal; DEV does not choose the direction, probe
   radius or a different step size.
7. Accept only real continuous TRAIN performance improvement and risk
   non-deterioration. Each DEV bank must meet the existing performance tolerance
   (0.2%) and the total/per-component risk gate. All risk rows use real
   `mean + tail_weight * top-20%-CVaR`, recomputed on each candidate flight.
   The default relative risk tolerance remains zero.
8. On Actor rejection or exception, restore Actor and post-fit RNG. Keep every
   finite completed Critic/target/optimizer fit and its fixed scales. A failed
   Critic fit restores its own pre-fit state instead.

The debug backend `--actor-proposal smoothmax-adam` retains scalarized gradient
proposals. Smooth-max alone does not enforce component-wise non-deterioration;
it is not the primary proposal backend.

## Cost, diagnostics and stopping

The default probe radius and initial parameter step are each
`1e-4 * max(||theta||, 1)`. Both full and half-radius probes are required:
a full-rank correction uses twenty extra H500 TRAIN forwards. Row consistency is

```
||a_h - a_half|| <= fd_atol + fd_rtol * max(||a_h||, ||a_half||)
```

`fd_rtol` remains 0.1. `--subspace-fd-atol` is measured in the normalized slope
units, independently of the real risk gate tolerance. It defaults to zero until
a TRAIN-only CUDA numerical probe supplies an appropriate absolute noise scale;
zero retains conservative near-zero checks and is not a calibrated noise estimate.
The configured value is bound into the experiment/checkpoint.

For the final unit direction q, compute d_h and d_half for every row. Use
`fd_atol + fd_rtol * max(abs(d_h), abs(d_half))` as the directional sign threshold.
Performance must be below the negative threshold at both scales. Risk above the
positive threshold at either scale rejects the proposal as `fd_unreliable`.
Small, mixed-sign risk slopes are recorded as `unknown_or_near_zero` and allowed
to reach the true H500 line search. They are not treated as proved non-increase.

Probe roundoff that distorts the requested parameter direction by more than 5%
also yields `fd_unreliable`. Parameters and Python/NumPy/Torch/CUDA RNG are
restored on every probe exit, including failures. No Actor Adam transforms the
checked direction afterward.

Logs include separate objective-gradient norms, basis rank, actual probe count,
per-row finite-difference errors/tolerances, directional slopes at both scales,
per-risk sign classifications and bounded TRAIN line-search outcomes. `development_evaluated=false` means no candidate DEV
was run; missing DEV fields are not fabricated results. Baselines are cached
only for identical Actor weights, physical configuration, horizon, loss/risk
settings and exact DEV state tensors. A Critic-only update does not invalidate
this cache; an Actor or scene change does.

`no_feasible_direction`, `fd_unreliable`, and `line_search_exhausted` reject only
the current proposal. Restore Actor, retain the completed Critic fit, increment
`consecutive_proposal_rejections`, and sample the next TRAIN batch. A successful
Actor update resets the counter to zero. True TRAIN/DEV gate rejections count
in the same consecutive-rejection budget.

At `--maximum-proposal-rejections` (default 3; the old
`--maximum-adam-rejections` spelling remains an alias), save and stop normally
with `status=proposal_plateau` and `exit_class=business_stop`. Exact or automatic
resume cannot re-enter that plateau. The local supervisor scripts use this
shared business-stop classification. A non-finite proposal or corrupted state
stops as an error: the guard rolls back unchecked state, the outer loop saves
`status=failed` / `exit_class=crash`, and the exception produces a nonzero exit.

Business stops do not append an extra DEV flight; unchanged-Actor periodic DEV
reports are reused. Interrupted runs and explicit time/proposal budgets remain
separate from both rejection plateaus and errors.

## New experiments and checkpoints

The objective is `component-risk-v3-warning-saturation-train-scaled`. Old scalar
or component Critic training states are rejected, including old lossless scalar
splitting. Reuse an Actor through `--initialize-from <checkpoint>` in a **new**
work directory. No old Actor Adam history or incompatible Critic state is loaded.
The direct backend stores no Actor optimizer. Exact resume within the same
experiment preserves fixed scales, Critic/target/optimizer, completed fits and
RNG; ordinary source/configuration bindings still apply.

`configs/response_risk_subspace.args` defines the complete H500/H50, 128-TRAIN,
two-64-scene-DEV CUDA profile. It starts a new run directory and defaults to
profile mode; training requires an explicit proposal budget and mode override.
First inspect it without execution:

```bash
python3 tools/train_response_control.py \
  $(cat configs/response_risk_subspace.args) --dry-run
```

Once a profile is requested, use the same configuration with an optional
`--initialize-from` Actor checkpoint. Measure time per proposal, forward count,
finite gradients, samples/s and memory; allocating a fixed fraction of GPU memory
is not a throughput criterion. This implementation has not rerun the historical
2000th proposal, demonstrated H500 training improvement, or benchmarked CUDA
throughput. Tiny deterministic tests establish software/numerical behavior only.
No FINAL, hidden reset, acrobatic task or full-budget training is part of this
change.
