# Short-window training with real-rollout candidate selection

The deployable `ResponseMotorPolicy` remains unchanged: action response → GRU
memory → controller → four motors. The training-only Risk-to-Go Critic is the
same four-output 256/256 SiLU MLP. Its architecture, fixed physical input scales,
TRAIN-calibrated output units and supervised remaining-risk labels are unchanged.

The normal `physics-subspace` backend uses short-window gradients to propose a
small search space. Complete continuous physical rollouts choose the update.
It does not estimate an H500 Jacobian or require a local descent certificate.

## One proposal

1. Hold the current Actor fixed and collect a continuous H500 trajectory without
   autograd, pooling two independent 64-scene TRAIN banks. Keep the existing 4×4
   thrust-to-weight / roll-authority stratification and full-trajectory CVaR.
2. Fit Critic remaining-risk values and the existing small direction sample set.
   Commit a finite completed fit, target copy, optimizer and RNG progression.
3. Process the same flight as 10 H50 windows. Preserve all numerical physical
   state and memory across windows, detach only the graph at each boundary,
   freeze target Critic parameters while retaining its state derivatives, and
   accumulate five gradient rows: performance and four risk guidance components.
   There is no Actor update between windows; the final window has no terminal V.
4. Orthonormalize those rows into at most five basis vectors. Starting from the
   original Actor, test each sign of each vector at `rho` and `rho/4`, where
   `rho = subspace_parameter_relative_step * max(||theta||, 1)`.
5. Every candidate executes a fresh, complete continuous H500 TRAIN flight from
   identical initial states and RNG. Filter candidates using the TRAIN gate
   below, then choose the lowest true performance cost among all valid trials.
   Do not stop at the first improvement. Full rank requires at most 20 candidate
   flights. No candidate is passed through Adam or weight decay.
6. Evaluate the selected candidate on both fixed DEV banks. DEV never ranks
   candidates and never chooses a fallback after rejecting the TRAIN winner.
   Accept the selected Actor or restore the old Actor. Completed Critic learning
   survives either outcome.

All four Critic components remain useful guidance. Position/velocity guidance
and the original soft risk sum are diagnostic/training quantities, not hard
acceptance constraints.

## Two acceptance stages

TRAIN requires strictly lower real performance cost. Both DEV banks require
performance no worse than the current Actor baseline beyond the existing 0.2%
DEV tolerance. Performance retains the physical Huber position/velocity/angular
tracking and control-effort/smoothness terms with their existing weights.

Both TRAIN and DEV additionally require:

- Finite physical/recurrent trajectory and finite metrics.
- No configured flight-envelope violation or action outside the hard `[-1, 1]`
  command range.
- Neither of the two true danger exposure components exceeds its declared
  budget relative to the current Actor.

The hard exposure components are computed directly from the full trajectory:

```
omega warning:     max(||omega - omega_reference|| / omega_limit - 1, 0)^2
motor warning:     mean_motor(max((|u| - saturation_limit) / (1 - saturation_limit), 0)^2)
```

The default hover warning thresholds are 10 rad/s and normalized motor command
0.95. Both exposures are exactly zero below their warning region. Each is summed
across the full flight and aggregated with the existing scene mean+CVaR rule.
There is no position/velocity component gate and no gate on the old total risk.

For each hard exposure component, the budget is:

```
new <= old * (1 + hard_risk_relative_tolerance) + hard_risk_absolute_tolerance
```

Defaults are 0.002 relative and 1e-8 absolute in accumulated dimensionless risk
units. These are explicit configurable acceptance budgets, not measured CUDA
noise tolerances or safety certificates. Equality is allowed. A normal action
increase below the warning region consumes no saturation-risk budget.

`--hard-position-bound` optionally sets a physical flight radius in metres;
`--hard-velocity-bound` optionally sets a speed envelope in m/s. Bounds are
checked over all scenarios/times, including startup, and are not fitted from
DEV. No physical envelope was supplied, so these bounds default to disabled.
The Critic's position/velocity normalization and tracking scales are not used
as implicit bounds. Future angular tracking tasks must supply an appropriate
angular reference; the current task is hover with zero reference.

The normal direct-search backend's periodic DEV report selects the best saved
checkpoint and reports progress; it adds no separate best-history Actor veto.

## Rejection, retention and reproducibility

`empty_subspace` or `no_acceptable_candidate` rejects one proposal. An unsafe or
non-finite trial is rejected and the remaining candidate budget is still tested.
A rejected selected candidate on DEV restores the Actor without trying the
next-best TRAIN candidate. Every failure retains a completed finite Critic fit.

Each finite proposal rejection increments `consecutive_proposal_rejections`;
an accepted Actor update resets it. At `--maximum-proposal-rejections` (default
3), save and stop normally with `proposal_plateau` / `business_stop`. Automatic
resume cannot re-enter that plateau. An invalid baseline, non-finite gradient,
corrupted optimizer or unexpected simulator exception remains an error stop.
A failed Critic fit restores its pre-fit state; completed fits survive Actor
errors/rejections.

Search restores Actor parameters and RNG even on an exception. Candidate
rollouts never accumulate parameter increments. DEV baselines are cached only
for identical Actor weights, physical state, horizon, task/risk definition and
hard-risk configuration. TRAIN rejection skips candidate DEV work.

Logs include all attempted candidates with basis index, sign, radius fraction,
true performance, hard-risk exposure, bound violations and rejection reason.
They identify the selected candidate and record the subsequent TRAIN/DEV
metrics. A `development_evaluated=false` record has no candidate DEV result.

## Historical diagnostics and checkpoints

`response_proposals_debug.py` contains the old two-scale finite differences,
row/direction reliability checks and cone projection. Normal training does not
import or invoke that module. Its `FiniteDifferenceConfig` is a diagnostic API;
FD tolerances and backtracking flags are absent from the primary training CLI.
`--actor-proposal smoothmax-adam` remains a historical regression backend.
MS/PETSc is unchanged and remains separate from this training path.

Exact resume binds source, proposal configuration, hard-risk definition and
supervision scales. Changing the proposal/gate configuration requires a new
experiment; use `--initialize-from <checkpoint>` to reuse Actor weights in a
new work directory. Do not edit checkpoint hashes or inherit stale Actor Adam
history into the direct-search backend.

`configs/response_risk_subspace.args` specifies the full H500/H50, 128-TRAIN,
two-DEV-bank CUDA profile. Its only subspace search setting is the parameter
radius; the two radius fractions and maximum 20 trials are fixed.

```bash
python3 tools/train_response_control.py \
  $(cat configs/response_risk_subspace.args) --dry-run
```

Training/profile jobs run only when requested. Unit and numerical correctness
checks do not establish learned control quality or deployment safety. A Critic
loss decrease or finite window gradient is not an accepted Actor improvement.
