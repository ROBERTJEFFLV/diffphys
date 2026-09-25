# Physics-group GRADIENT normalization

This supersedes cost-RMS normalization at `3ea9fe8` on the SAME experimental
branch, `codex/time-decay-bounded-influence-20260925`. Cost normalization and
its two CLI flags are removed, not left enabled beside gradient normalization.
The original Actor, simulator, termination, sampling, task costs, pooled CVaR,
Time Decay and raw EVAL metrics are unchanged.

## Exact update and zero convention

For N initial scenes, let C_i be the original full-flight cost and w_i the
existing detached mean-plus-pooled-CVaR coefficient (already includes 1/N).
Let s be the existing gradient_scale and D denote the configured H500 VJP;
D is a surrogate derivative when Time Decay is nonzero. For physical group g:

```
L_g = (N/n_g) * sum_{i in g} w_i*C_i
h_g = s * D L_g
r_g = ||h_g||_2                         # ALL Actor parameters together
m   = lower_median({r_g : r_g > 0})      # 0 if every group is zero
q_g = m / max(r_g, epsilon)             # 0 when r_g is zero
g   = (1/G) * sum_g q_g*h_g
```

For r_g >= epsilon, the normalized group gradient has norm m. Groups below
epsilon contribute less; zeros remain zero and retain their 1/G slot. They
are excluded from the median so that a majority of zeros does not erase the
remaining gradient. Epsilon defaults to 1e-12. With even positive-group counts,
the lower middle value is used, not the mean of the two middle values. No
cost statistic, running cost scale, per-layer norm, or individual scene norm
is used to select q_g. Small nonzero groups are scaled UP as well as large
groups down: this is normalization, not one-sided gradient clipping.

Without normalization, averaging h_g equals the old pooled gradient for equal
n_g. The factor N removes the preexisting 1/N before group averaging. Original
CVaR factors remain inside h_g exactly once. We do NOT compute mean(w_i*C_i)
then average again, and do NOT differentiate through m or q_g. There is no
single original loss whose exact gradient is claimed after this aggregation;
therefore the old normalized `optimization_objective` is removed from logs.

This balances group gradient magnitudes, NOT group performance or Adam's
per-group parameter displacement. A single scene can still dominate its own
group direction. Other group directions can cancel. A data-dependent median
is not a fixed hard bound, and if most groups explode it may also be large.
Normalizing gradients does not repair a physically oscillatory controller.
Existing final global clipping and Adam are retained, with ONE Adam step after
aggregation. No TRAIN/DEV candidate-acceptance gate is added in this experiment.

## Group membership unchanged

The batched physical-space partition is byte-for-byte unchanged. Features are
TWR, torque-to-inertia, and mean rising/falling motor time constants, with log
spans from the reference sampler. Default B512 => 16 groups of 32 initial
scenes when physical diversity permits. At least 32 UNIQUE initial scenes
per group; fewer groups for smaller batches or identical physical parameters.
Early failures are not dropped; no scene is resampled and flight order is not
changed. These are group gradients, not the historical per-scene gradients.

## Parallel computation and cost

The existing full-batch forward graph is retained once. A detached [G,N]
coefficient matrix supplies G VJPs; no [N,N] identity/per-scene Jacobian is
constructed. Default `--group-vjp-chunk-size 16` processes all 16 GROUP signals
in one batched autograd call for B512. This is NOT one ordinary backward's work:
it still carries 16 reverse signals. Chunk=1 provides a serial reference;
chunk=4 gives four batched calls for 16 groups. Every chunk uses the same graph,
so there is no isolated-scene rerun or changed compaction. The final chunk frees
it. There is no silent retry of partially consumed graphs on OOM/operator errors.

Only the small G-by-P parameter-gradient table is promoted to FP64 for robust
norm/reduction; the physical graph and batched state adjoints remain in the
configured dtype. Stable scaled norms avoid overflow from squaring large raw
entries. Group rows are checked for NaN/Inf before normalization and publishing
`.grad`. Invalid updates use the existing Actor/Adam/RNG rollback. Diagnostics
stay on the device until the trainer logs the small tables.

Grouped mode currently requires `--backprop-mode full --agc 0`; unsupported
windowed/AGC combinations fail explicitly rather than falling back to costs.
The ungrouped full/windowed paths are unchanged. H500/Time Decay are not truncated.
`torch.autograd.grad(..., is_grads_batched=True)` can still have vmap fallbacks;
CUDA time and memory must be measured, not inferred from the call count.

## Configuration / checkpoint compatibility

The standard RAPTOR config enables:

```
--group-balance
--group-max-groups 16
--group-min-scenarios 32
--group-gradient-epsilon 1e-12
--group-vjp-chunk-size 16
```

Its output directory is `runs/raptor_group_gradient/seed7`. The bare CLI keeps
`--no-group-balance` as its baseline. Old `--group-scale-mode` and
`--group-scale-floor` are rejected. Per-scene `--contribution-*` flags remain
absent. Config, algorithm version and source hash are bound into checkpoints.
Old exact resume is deliberately rejected; do not alter checkpoint hashes.
Compatible weights-only initialization resets Adam and is NOT an exact incident
replay. The user's local position-only termination patch and mature 17206
incident checkpoint are not on this public base and were not overwritten.

`group_balance` logs only membership counts and physical ranges.
`group_gradient` logs raw/normalized norm, multiplier and target for EVERY group,
plus group_count, vjp_chunk_size and vjp_calls. `task_objective` remains the
original raw cost. Never interpret finite parameters or equal group norms as
proof that an accepted Adam step improves control.

## Verification

```
PYTHONPATH=. python -m pytest -q tests
python tools/verify_group_balance.py --repeats 1 --output group-gradient-verification.json
# Same bounded probe on the user's GPU:
python tools/verify_group_balance.py --device cuda --repeats 1 --output group-gradient-cuda.json
```

Tests cover constant-score gradient outliers, upscaling vs clipping, zero/tiny/
huge/nonfinite rows, exact group-weight/CVaR accounting, unequal group sizes,
serial/chunked group VJP equivalence, unchanged forward/RNG, one final Adam,
complete rollback including existing moments, exact NEW-version resume and
rejection of old flags. Disabled mode is compared to a pinned parent for two
updates. A separate test runs CUDA only when available; skips are reported.

The bounded H500 benchmark distinguishes random cold-start H500-CAP samples
(early termination) from the constructed hover fixture that executes 512*500
transitions. The latter tests long computation graphs, not learned flight.
Timings compare original pooled vs GROUP-gradient normalization, not cost RMS
or the earlier 512-scene VJP implementation. No mature incident replay,
long-run performance, GPU speed or closed-loop stability claim follows from
CPU tests. The same benchmark can be run locally for chunk=1/4/8/16.

Primary API documentation:
https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad
https://docs.pytorch.org/docs/stable/generated/torch.median.html
