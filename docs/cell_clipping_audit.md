# Fixed physical-cell clipping and replayable update evidence

## Scope and exact update rule

Production changes only group membership, group-gradient scaling, and evidence
retention. Actor, physics, sensors/disturbances, sampler formulas and frozen yaw
thresholds, H500, equal-time task loss, pooled CVaR, Time Decay, global clip and
Adam equations are retained. No finite-update rejection or difficult-scene
filtering is added. No TBPTT/shorter physical rollout is introduced.

At 2048 TRAIN scenes the existing coverage sampler produces 128 cells x 16 rows.
The SAME `response_sampling.physics_cell_ids` now partition the initial rows for
backpropagation. Membership includes early-terminal scenes, is independent of
cost/survival, and is never recomputed from the live compacted pool. Fixed IDs
include TTI x rising x falling x conditional yaw; no new calibration is fitted.
Unequal/missing quotas fail explicitly, rather than falling back to adaptive
16-group splits. The random sampler's legacy adaptive partition remains available
for small regression/API callers, but it also uses shrink-only clipping.

Let C_i be the existing full-flight scene cost and w_i its existing detached
pooled mean+CVaR coefficient. For cell g, with N=2048, n_g=16, G=128:

    L_g = (N/n_g) * sum_{i in g} w_i C_i
    h_g = gradient_scale * grad_theta(L_g)
    q_g = min(1, group_clip_norm / ||h_g||)  # q=1 for zero gradient
    gradient = mean_g(q_g*h_g)
    original global clip -> original Adam

No extra /16 or /128 is applied elsewhere. Without clipping, equal-sized groups
recover the original pooled task gradient (up to floating reduction order).
CVaR is not recomputed within groups. The fixed cap acts AFTER gradient_scale
(default .1), not on the unscaled gradient or the final Adam parameter step.
Zero groups retain their share of the average. Small gradients remain unchanged.
Norms use stable FP64 reductions of only G x P parameter data; no H500 graph is
promoted to FP64. Nonfinite group derivatives abort and retain failure evidence.

`--group-clip-norm 1.0` is a **provisional configurable default, not a calibrated
optimum**. The user's stable-window 128-group gradient data were not available.
Old 16-group statistics are not interchangeable with new 16-scene group norms.
The cap never follows this batch's median. After clipping, each row and its
mean have norm <= cap up to floating rounding. This does NOT bound Adam's
parameter step, remove within-cell directional dominance, or repair unstable
forward feedback. In particular, a raw 1e8 gradient can still occur inside BPTT.

`--group-vjp-chunk-size 16` gives eight VJP calls on one retained forward graph.
The final call frees the graph. We do not split forward batches, detach GRU or
physics state, use 2048 per-scene VJPs, or silently retry a different backend.
The total backward computation increases versus 16 groups; chunking bounds
concurrent adjoint width, not total runtime. CUDA throughput remains unverified.

## Configuration and compatibility

Production args: `--scenarios 512 --train-sampling coverage128
--group-max-groups 128 --group-min-scenarios 16 --group-clip-norm 1.0`.
Scenarios remain PER BANK (four banks in one pool). Smaller coverage smoke tests
must explicitly lower `--group-min-scenarios`, keeping all 128 cells nonempty.
Old commands explicitly requesting 16 groups with coverage128 fail clearly.
Bare random CLI calls default to the legacy adaptive 16-group partition, with
fixed clipping instead of median normalization. No Actor interface is changed.

Group algorithm version, layout, cap, chunk size, source and audit configuration
are checkpoint-bound. Strict resume stays strict: a new algorithm is not an
exact continuation of old source. Use a separate worktree and new work directory
with the existing `--init-checkpoint` for an explicit weights-only fork. It
resets Adam/update numbering/sampling as before; do not call this an Adam-
preserving continuation or rewrite old checkpoint hashes. Compatible historical
checkpoints retain their saved evaluation loss and physical metrics.

## Evidence retained by default

Each completed update writes `audit/updates/u<update>_<unique>.pt` containing:

- Actor tensors, complete named Adam state and Python/NumPy/Torch/CUDA RNG before
  and after the actual update, and the TRAIN sampling index/seeds/report;
- group IDs and scene membership, original/clipped norms and multipliers;
- named gradients entering Adam, plus global/per-layer actual parameter changes;
- policy settings and full source/physics/loss/sampling/optimizer binding.

This is not merely a gradient-norm log. Failed attempts retain the candidate
state BEFORE rollback (including NaNs); `failure.pt` still holds the restored
valid Actor/Adam/RNG. The audit never turns a finite large-gradient scene into a
skipped training example. Source files are archived once in `audit/source.zip`.
The recorder uses separate filenames and OS randomness, not training RNG.

Recent capsules are a bounded ring: default 64, automatically at least one EVAL
interval + 1. Alerts pin a context window using hard links where supported.
At most 8 event windows are kept by default; evictions are recorded explicitly
in `audit/events.jsonl`. This is NOT unlimited retention of every training step.
Copy an event elsewhere before retention expires if it must be kept permanently.
Each EVAL additionally saves its own full checkpoint in `checkpoints/`;
these EVAL checkpoints are not automatically pruned by the event recorder.

Default evidence-only alerts: raw group >100*fixed cap; actual parameter-step
norm >5*its preceding EMA; fixed EVAL objective >1.5*previous best; numerical
failure. Raw/step alerts have one EVAL interval cooldown; EVAL regression/failure
always pins context. These are logging thresholds, not proven stability margins.
The full update ring also captures subthreshold changes preceding a regression.
No EVAL data affects gradients, and finite updates are not rolled back by alerts.

`history.jsonl` retains old norm column names for GUI compatibility:
`raw_gradient_norm` is still AFTER group processing and BEFORE global clip.
`group_gradient.values` uses columns `raw_gradient_norm`,
`normalized_gradient_norm` (now the clipped norm), `multiplier`, `target_norm`
(now the fixed cap). New `parameter_changes` gives actual global/per-layer
parameter changes; `gradient_entering_adam_norm` records the final gradient.
Small relative layer norms can be huge for zero-initialized weights: they are
reported, not used as an automatic control/safety threshold.

## Read-only replay

Use the matching source, PyTorch version, device/backend and batch shape:

    python tools/replay_response_update.py RUN/audit/updates/u...pt
    python tools/replay_response_update.py RUN/audit/updates/u...pt --mode adam

`full` regenerates the SAME TRAIN pool, full rollout, 128 clipped gradients,
global clipping and Adam update, then compares stored gradients/Actor/moments/RNG.
`adam` uses the recorded gradient to isolate the optimizer step. Neither modifies
the original checkpoint or run. A CUDA capsule does not silently replay on CPU.
Failed attempts remain forensic evidence, not accepted-step replay fixtures.
Bitwise replay across different hardware or numerical kernels is not promised.

## Verification and interpretation

Tests cover membership/quota completeness, pooled CVaR scaling, analytic 128-group
VJPs with eight calls, the non-clipped GRU gradient vs original pooled BPTT,
zero/small/huge/nonfinite gradients, forward-kernel preservation, accepted-step
Actor/Adam/RNG replay, same-source split resume, bounded/pinned evidence, and
failure capture before rollback. Only the two intentionally changed protected
group kernels receive new hashes with reasons; remaining protected hashes stay.

CPU tests and synthetic outlier probes establish implementation/numerical
properties. They do not prove high-TTI flight convergence, improved forgetting,
CUDA speed/memory, or deployment safety. The user's best8250/11000 checkpoint
and full stable-window logs were unavailable here. Keep deployment unauthorized.
