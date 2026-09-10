# Phase 1: task value and complete boundary feedback

The deployable `ResponseMotorPolicy`, physics, nominal dynamics sampling, loss
weights, H500/H50 timing and evaluation criteria are unchanged. The task-value
schema is `task-value-v3-full-boundary-adjoints`. This revision establishes a full
feedback reference and bounded Critic readiness checks; it does not claim that
the learned Critic now provides accurate feedback or that the Actor can hover.

## One task and one continuous episode

`response_task.step_costs()` supplies both value labels and Actor costs, including
its existing full-flight and final steady-interval weights. Each window passes
absolute `start=b, horizon=H`. `training_step_costs()` and risk scalarization are
not used. Risk remains an observation, not an Actor veto.

Two banks of 32 nominal initial states are pooled. A fixed Actor generates one
continuous H500 trajectory and per-scene remaining task costs. Full-flight
mean+CVaR weights are selected once and reused in every window. The scalar
256/256 SiLU Critic still predicts cumulative task cost, with a signed linear
output and the existing dimensionless state/capability features. It predicts
neither batch CVaR nor optimal-policy cost. No running normalization, output
transformation, additional observations or remaining-time multiplier is added.

Actor weights stay fixed throughout sampling and all ten windows. Physical state,
GRU memory, integral and action history remain numerically continuous; only the
computation graph is detached at each H50 boundary. Ten parameter gradients are
averaged before clipping and, when ready, one persistent Adam update. Evaluation
is periodic observation/best-checkpoint selection; it never rolls back an update.
No candidate search, FD, MS, PETSc or per-update EVAL approval enters this path.

## Exact full-state reference

`response_adjoints.collect_boundary_adjoints()` recomputes the ten windows in
reverse order with frozen Actor parameters. At each start, all dynamic state
fields become independent leaves. With lambda[H]=0, differentiate

```
Q = sum(step_costs(window, start, horizon)) + lambda[end] dot Z_end
```

with respect to those start leaves. The linear terminal injection preserves the
recorded numerical remaining value while providing exactly the specified
covector. Labels are per-scene and **unweighted by CVaR**. They include physical
and recurrent history aliases; they are bound to Actor hash, initial-state hash,
loss and time configuration. Boundary mismatch or nonfinite values stop the run.

Only one H50 graph lives at a time. The propagated adjoint is nevertheless
mathematically the full long-horizon derivative, so it can exhibit real exploding
gradients. This oracle is a diagnostic/short-baseline facility, not a claim that
truncation has eliminated long-horizon conditioning.

`accumulate_task_gradients()` supports explicit terminal modes:

- `critic` (default): local cost plus target value, after readiness below.
- `oracle_full_state`: local cost plus the complete true boundary covector;
  explicit diagnostic, no learned-Critic readiness requirement.
- `none`: local-only diagnostic, no terminal value.

For H500/H50, **10 times** the oracle's averaged Actor gradient must match full
H500 BPTT under the same fixed pooled CVaR weights. Neither oracle nor Critic
feedback detaches any physical branch to hide untrained derivatives.

## Supervision coverage and coordinates

The default derivative pool now covers all nine nonterminal boundaries and all
64 TRAIN scenes. A single scene split holds out 16 scenes at every boundary:
432 training states and 144 derivative-held-out states. Value regression still
uses all scenes, so this is derivative holdout within TRAIN, not independent
EVAL/generalization evidence. Each 1024-value minibatch draws 32 derivative states
with replacement. Labels are reused within the bounded frozen-Actor fit.

All twelve differentiable state groups used by `critic_features()` are supervised:
position, velocity, rotation, omega, motor, previous action, memory, integral,
previous velocity, previous omega, previous rotation and older action. The full
oracle additionally retains `policy.last_action` and `policy.calls`; they are
absent or differentially inactive in the Critic mapping. Their omission from
regression does not omit them from the oracle.

For Euclidean x/s, both true and predicted gradients use s*dV/dx. The fixed scales
are recorded in `response_adjoints.STATE_SCALES`; no statistics are estimated.
Rotations use tangent perturbations R exp(skew(delta)), in radians, not a loss on
nine independent rotation entries. The oracle retains ambient covectors for
exact Actor VJPs. Tangent supervision is checked against the actual Actor gradient
below, rather than assumed to constrain every ambient neural-network direction.

Direction and magnitude losses are computed per state group and then averaged:

```
L = Huber(value, remaining_task_cost)
  + 0.5 * wd * mean_group(1 - cosine(g_pred, g_true))
  + wm * mean_group(Huber(log(norm(g_pred)+eps) - log(norm(g_true)+eps)))
```

Zero true gradients contribute only magnitude loss. Float64 reductions stabilize
cosine/lognorm computation; `create_graph=True` carries state-derivative losses
back into Critic parameters. The three parameter-gradient norms are remeasured
before every minibatch update. Ratios to the value gradient set wd and wm; a
near-zero denominator disables that term and is logged, rather than magnifying
it with an epsilon denominator. `fixed` remains an explicit ablation option.
This norm-ratio heuristic is **not** the full GradNorm algorithm and does not
establish that the losses can be jointly learned.

Critic clip=10 and target EMA tau=0.6 are unchanged. Checkpoints record the schema,
state groups, boundaries, scene IDs, scales, balance mode and current weights.
Every minibatch logs raw and weighted parameter-gradient contributions. Online
and target derivatives are reported separately, per group and per boundary,
including cosine, negative-cosine fraction, norm-ratio median/p90 and log error.

## Bounded readiness and checkpoint transactions

On fixed Actor/data, each fit is followed by actual Actor-gradient comparisons:

```
g_critic = average_window_gradient(local cost + target terminal)
g_oracle = average_window_gradient(local cost + full true terminal)
```

The log records their cosine and relative vector error for all TRAIN scenes and
the derivative-held-out scenes; online-Critic/all-scene results are separate.
Held-out checks retain the **original forward batch size** and mask the original
CVaR weights without renormalizing. Slicing a CUDA batch can change the numerical
trajectory and make a long-horizon gradient comparison misleading.

Default readiness requires both target comparisons to reach cosine >=0.9 and
relative error <=0.5 within 8 fits/120 seconds. These are explicit, configurable
**engineering criteria**, not experimentally established convergence or deployment
thresholds. The time budget starts at trajectory collection and is checked at
fit boundaries; a fit plus its audits completes atomically and may exceed the
wall-clock limit. A successful readiness check does not certify an Adam step or
finite-step task improvement. The actual parameter step and unaveraged
`g_oracle dot delta_theta` are logged for learned-Critic updates.

If not ready, the result is `critic_not_ready`: retain completed Critic/target/
Critic-Adam work, keep Actor/Actor-Adam unchanged, checkpoint and stop normally.
The runner refuses automatic resume of this business stop. `--value-critic-only`
never commits an Actor update, including when ready (`critic_only_ready`).
Numerical failures instead save their stage/inputs and stop as failures. Failed
Critic fit transactions roll back only that fit, not previously completed fits.

Old v1/v2 or risk-Critic training state is incompatible and cannot be silently
resumed. `--initialize-from` explicitly loads **Actor weights only** into a new
experiment and creates fresh Actor/Critic optimizers. It does not claim resetting
Actor moments is better. The separate shadow diagnostic preserves saved Actor
Adam moments to compare identical optimizer transformations. Exact new-schema
resume otherwise preserves optimizer states and RNG.

## Bounded diagnostic commands

Frozen-Actor readiness, one attempt, no online training:

```bash
python3 tools/train_response_control.py $(cat configs/response_phase1_critic_only.args)
```

Use `--initialize-from PATH` for explicit Actor-only initialization and a fresh
`--work-dir`. The independent config uses CUDA float32, seed 7, H500/H50 and 64
pooled TRAIN scenes. A fixed EVAL baseline is recorded; it does not approve fits.

Saved-checkpoint oracle and shadow checks (no Actor update is retained):

```bash
python3 tools/check_response_boundary_adjoints.py \
  --snapshots reports/phase1_derivative_profile_20m/snapshots \
  --fits 1 6 --device cuda --chunk-size 64 --shadow-steps \
  --max-seconds 180 --output reports/phase1_boundary_check
```

`--frozen-fits N` additionally compares memory-only and full-state supervision on
the first saved Actor, with the same expanded pool, initial scalar MLP weights,
minibatch RNG and fresh Critic Adam. Historical Critic weights are read for this
explicit diagnostic; incompatible optimizer/schema state is not resumed.
Use the original batch size for the full BPTT reference when memory permits.
Smaller chunks are a memory fallback, **not** a guarantee of the same CUDA
numerics; the tool records scene-cost discrepancy and stops on oracle error.

The existing `response_phase1_single_airframe.args` remains the opt-in online
configuration, now subject to bounded readiness. Long training still requires
explicit authorization and a fresh source-bound training contract. Numerical
correctness, Critic feedback quality, learned control and deployment safety are
separate claims.
