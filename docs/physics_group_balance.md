# Physical-group score balancing: one backward, not per-scene clipping

This replaces the expensive bounded-contribution experiment on
`codex/time-decay-bounded-influence-20260925`, parent `17cfc625`.
No other repository/branch is changed. The public parent does not include the
reported local batched-VJP implementation or position-only termination patch.
Reconcile uncommitted local changes explicitly; do not overwrite them.

## What is grouped

The fixed physical features are thrust-to-weight, torque-to-inertia, mean
motor rising time, and mean motor falling time. These distinguish translational
control authority, rotational authority, and actuator response. Rotor times are
identical across a vehicle's four rotors in the current sampler. Mass/arm length
alone do not determine group membership; no privileged features enter the Actor.

Take the logarithm of each positive feature and divide by its reference
log-span: log(5/1.5), log(1200/40), log(0.10/0.03), log(0.30/0.03).
This prevents a feature's units/range from arbitrarily controlling the split.
These are coordinate scales, not clipping or altered sampling ranges.

Use balanced k-d splits: for each current node find its widest normalized
physical feature and split at the median rank. All nodes at a depth are sorted
and split together using device tensors. Stable sorts resolve ties without RNG.
Stop at the configured maximum group count or before any child could contain
fewer than 32 scenes. If a current node has identical physical features, stop
that depth rather than create meaningless labels for identical aircraft.

Defaults: maximum 16 groups, minimum 32 unique initial scenes per group.
For the normal randomized 512-scene bank this gives 16 groups of 32. A 128-scene
bank gives at most 4 groups of 32; 100 scenes give at most 2 groups of 50.
A bank smaller than 32 is rejected when enabled. Larger minimums are permitted,
but values below 32 are not. No resampling, duplicate aircraft, discarded rows,
or survivor-only filter is used. Early failed scenes still belong to their
initial group. Groups can be fewer for physically identical/degenerate banks.

The labels are batch-relative clusters, NOT fixed physical bins across time.
Log each group's feature minimum/maximum for interpretation. Because membership
can change across batches, do not share a running EMA under a group ID whose
physical meaning has changed. This implementation uses detached current-batch
RMS instead of cross-batch running normalization.

## Exact objective and CVaR compatibility

Let C_i be the unchanged complete trajectory cost, including the original
terminal/dead costs, and N the number of sampled scenes. Original pooled weights
are w_i = 1/N + tail_weight/K * 1[i belongs to pooled top-K],
K = ceil(tail_fraction*N). They are already detached and include the batch mean.

Define a_i = N*w_i and, for group g with n_g scenes,

    s_g = max(sqrt(mean_{i in g}(C_i^2)), group_scale_floor)
    J_opt = (1/G) * sum_g mean_{i in g}(a_i*C_i / stopgrad(s_g))
    w_opt_i = N*w_i / (G*n_g*stopgrad(s_g))
    gradient = gradient_scale * D_theta(sum_i w_opt_i*C_i).

Use scale_mode=none to set s_g=1. With tail_weight=0 this is precisely the
selected group-mean/scale formula. With the existing CVaR enabled, the original
pooled top-K membership and per-scene amplification are retained, not selected
again inside groups. Equal counts and s_g=1 exactly recover w_i: no double mean.

Groups have equal outer averaging coefficients, NOT identical gradient norms.
CVaR intentionally adds weight to tail scenes and can affect groups differently.
RMS is based on C_i before the CVaR amplification, preserving relative tail
priority. The default scale floor 1 prevents tiny/no-variance costs from creating
huge inverse scales. It is configurable, not an experimentally optimal value.
Use a maximum-scaled RMS to avoid squaring very large FP32 scores directly.

Group membership, RMS, and weights are calculated under torch.no_grad. Do not
subtract the current group mean and average it again: the objective would cancel.
Do not differentiate through RMS or train an extra scale network. Time Decay
still makes D a configured surrogate derivative when alpha>0. Score scaling
adds another explicit, frozen per-update weighting; it is not a stability proof.

## Parallel computation and cost

One normal full-batch forward, then a tiny [groups, members, features] table.
Batched sorts/reductions run on the scene tensor's device (CUDA when training
on CUDA). One vector of N coefficients feeds ONE ordinary autograd.grad in
full mode. There is no [scene, parameter] Jacobian, per-group/per-scene backward,
batched VJP, vmap fallback, repeated forward, or time truncation. At most four
split-depth scalar checks at defaults and normal finite checks remain; the
small report table is transferred once after the update. H500 time steps still
follow their causal order; scenarios within a step are parallel.

The optional original windowed recomputation path reuses the same frozen
weights across windows. It is not group-wise differentiation and is not the
single-call full mode used in the default H500 configuration.

Original raw task metrics, physical termination, failure scoring, EVAL and
best.pt selection remain unchanged. Log optimization_objective separately from
task_objective. A smaller normalized optimization score is NOT by itself a
control-performance improvement. The existing final global gradient clip,
finite checks and Adam failure rollback remain; no new clipping level is added.
Finite but harmful updates are still possible, as before.

## Files / obsolete implementation

New production module: response_groups.py (group layout, score statistics,
frozen coefficients). It is included in the training source hash. The existing
seven-file chain is otherwise retained: response_adjoints.py uses its weights,
response_training.py binds/logs configuration, and the CLI exposes switches.
response_policy.py, response_task.py, env_l2f.py and response_execution.py are
unchanged from the parent. No MLP, PPO, new physics model or action interface.

Remove the old _bounded_contribution_gradients, contribution CLI options,
bounded-influence smoke, and expensive verification script. They remain in Git
history at 17cfc625, not in the production import path. Old per-scene regression
tests are replaced with group-layout, weight, gradient and one-VJP tests.
Old --contribution-* commands fail rather than silently select another method.

## Usage and checkpoints

The normal RAPTOR args file now enables:

    --group-balance
    --group-max-groups 16
    --group-min-scenarios 32
    --group-scale-mode rms
    --group-scale-floor 1

It uses a new runs/raptor_group_score/seed7 directory. Bare CLI remains opt-in
for small existing smoke tests; --no-group-balance selects the original pooled
baseline. The unchanged L2F args file stays ungrouped. Use --mode profile and a
new work-dir for one update; never launch a long run merely to test grouping.

Strict resume still requires matching source and bound settings. No source/hash
check was bypassed. --init-checkpoint remains weights-only (new Adam); it is not
a restart with historical Adam. This module has no running normalization state,
so there is no unrecorded EMA to restore. Report tables are not Actor inputs.

    PYTHONPATH=. python -m pytest -q tests
    python tools/verify_group_balance.py --output group-verification.json
    # On the actual training GPU, add --device cuda.

## Validation scope

Tests check all four physical features, minimum counts and coverage, odd-sized
banks, identical aircraft, preserved sampling/RNG/forward masks/costs, CVaR and
non-double-mean identities, detached RMS gradients, numerical edge cases,
exact new-run resume and nonfinite rollback. They count a single full-mode VJP.
A counterexample explicitly shows equal finite costs can still have enormous
gradients: this is score balancing, not a replacement for a sensitivity bound.

The bounded benchmark compares ordinary pooled versus group-weighted backwards
on the same inputs. It includes a random H500-cap bank with early terminations,
and a separately constructed 512-scene hover fixture that really executes all
500 steps while varying rotational authority and motor time constants. The
fixture is not a learned controller or a sample of the unmodified training
initialization distribution. Neither benchmark replays the missing mature
17065 checkpoint. CPU results cannot predict exact RTX 4060 Ti timings.

Relevant PyTorch operations: torch.argsort(stable=True), torch.no_grad,
and torch.autograd.grad. No new runtime dependency or compiler is required.
