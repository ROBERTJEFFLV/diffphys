# Bounded contribution experiment (Time Decay only)

Base: `time-decay-only-20260917` at
`1564d25a210c139a94431fc6ae59d10cc0430ffe`.
Only three production files change. Actor, physics, sampling, original loss,
pooled CVaR, failure accounting, evaluation, and all existing launch configs stay
unchanged. No Metric MLP or PPO is introduced.

## Mathematical audit and corrected update

Let `D_i` be the parameter VJP of the full scenario cost `C_i` under the current
backward rule. With alpha > 0 it is a **surrogate** derivative, not the exact
derivative of the unchanged forward objective. H500 connection is preserved.

For N scenes and K = ceil(tail_fraction*N), upstream uses detached coefficients

```
w_i = 1/N + tail_weight/K * 1[i in the pooled top-K costs]
g_old = s * sum_i w_i D_i,       s = gradient_scale.
```

The previous discussion incorrectly averaged `w_i D_i` again. That introduces
an additional 1/N (or an unintended physical-group scaling). Correct de-averaged
votes and the implemented SINGLE pre-merge clipping operation are

```
v_i = N*s*w_i*D_i
P_tau(v) = v                         if ||v||_2 <= tau
         = (tau/||v||_2)*v           otherwise
g_new = (1/N)*sum_i P_tau(v_i).
```

Zero vectors are returned unchanged; no division by zero or epsilon-dependent
shrinkage is needed. `tau` must be finite and strictly positive when enabled.
The Euclidean norm is over **all Actor parameters together**, not independently
per tensor/layer. Norms and accumulation use FP64; final gradients retain the
parameter dtype. A nonfinite per-unit derivative aborts BEFORE clipping/Adam;
Inf cannot be made into a meaningful direction by multiplying it by zero.

Consequences (in exact arithmetic; implementation assertions use tolerances):

* No active clipping: `g_new == g_old`, with CVaR and s included exactly once.
* Each scene's contributed vector has norm <= tau/N; `||g_new|| <= tau`.
* With fixed weights/other vote vectors/tau, replacing one vote changes the
  aggregate by at most 2*tau/N. Changing CVaR membership or a data-dependent tau
  changes the premises, so this is not a blanket data-replacement guarantee.
* Equal coefficients are NOT equal gradient norms or equal performance. One
  bounded vote may still dominate if the others are zero or cancel.
* If every scene vector is bounded by tau, a group's mean is already bounded by
  tau (triangle inequality). Another group clip at that same tau is redundant.
* Global clipping only scales a pooled direction. Adam subsequently applies its
  nonlinear, history-dependent transformation: a gradient-input bound is NOT an
  independent bound on a scene's influence on the eventual parameter displacement.
* Clipping is a deliberately biased aggregation rule. It does not bound the
  internal H500 Jacobian products, guarantee loss descent, or certify flight.

`tau = kappa*median(norms)` is a possible calibration heuristic, not a proof of
an absolute influence bound. With a zero median it may erase all useful votes;
if most votes grow it need not detect anything. The earlier log-MAD formula
`exp(m + 3*1.4826*MAD)` is algebraically meaningful with a positive log offset,
but 1.4826 is a Gaussian scale calibration, NOT an established confidence rule
for these gradient norms. Neither adaptive rule is silently installed here.
Use the logged vote norms from TRAIN to calibrate a fixed tau; 0.1/5 in the
probes are explicit test values, not validated optimal training hyperparameters.
Do not copy the total-gradient threshold 10 blindly into per-scene units.

Cosine with the pooled gradient can be close to one just because an outlier
already dominates it. Cosine with the remainder is more diagnostic, but even
alignment cannot prove a useful Adam direction or rule out long-horizon curvature.

## Exact scenes versus explicitly approximate blocks

`--contribution-unit-size 1` computes the exact scene-level rule above, with one
VJP per scene on the same retained full-batch graph. It never reruns an isolated
scene, changes the forward batch shape, samples new noise, or cuts time edges.
This serial correctness-first implementation can be expensive for N=512.

Larger units are an OPTIONAL lower-cost approximation, not physical size groups.
For a consecutive index block U containing n_U scenes:

```
v_U = (N/n_U)*s*sum_{i in U} w_i D_i
g_block = sum_U (n_U/N)*P_tau(v_U).
```

Actual block sizes are used, including a partial last block. This preserves
`g_old` when clipping is inactive. A block contributes at most `n_U*tau/N`.
No individual guarantee survives within a multi-scene block: opposing huge
scene vectors can cancel invisibly, and clipping an outlier block also shrinks
its normal scenes. The test suite deliberately demonstrates this limitation.
Logs say `unit_kind=scene` or `block`; the two cannot be reported interchangeably.
With N=512 and unit-size=64 there are eight VJPs, not 512.

## Line-level change map against the pinned base

| Base location | Minimal change |
| --- | --- |
| `response_adjoints.py:131-152`, `backward_actor` | Add an opt-in same-graph VJP aggregator. Leave the original full/windowed paths unchanged when disabled. Publish parameter gradients only after all units pass finite checks. |
| `response_training.py:272-309`, `binding` | Bind both flags and label the enabled algorithm; do not relax resume. |
| `response_training.py:523-525`, backward call | Forward the two flags, with one Adam step AFTER all units, never one Adam per block. |
| `response_training.py:546-567`, log row | Add raw pooled norm, bounded norm, per-unit norms/scales, unit sizes and cap metadata. Existing raw_gradient_norm is after aggregation; use nested pooled_gradient_norm to see the original spike. |
| `tools/train_response_control.py:56-70,100-114` | Add flags and reject negative/nonfinite caps, nonpositive unit sizes, enabled windowed mode, and AGC mixing. |

Production diff: see the commit diff for exact current line numbers; the table
uses the unmodified base. The source hash already covers all three changed files.

## Excluded changes / conflicts

No physical-group reweighting: equal group weights would change the desired
training distribution, and the evidence concerns one anomalous trajectory.
No per-group gradient clip, PCGrad/MGDA, loss normalization, PPO, new critic,
MLP, resampling, temporal truncation, or physics/termination changes.
AGC is rejected when this experiment is enabled to avoid an additional
layer-wise transformation. Existing final global clipping is retained for
baseline compatibility (and is normally inactive when tau < 10).

The base's RAPTOR termination STILL checks position, velocity and omega. The
reported later local position-only patch is NOT on this public base. It must
be reconciled separately; neither the 140 local tests nor the mature 17065
checkpoint are present here. Do not substitute this branch over uncommitted
local fixes or claim a replay of that event.

A finite-loss TRAIN/DEV acceptance gate is also NOT included in this isolation
experiment. Upstream nonfinite failure rollback remains and is tested for Actor,
Adam and RNG, before publishing an update index. Finite but harmful updates are
still possible. Adding DEV acceptance would be another optimization policy and
would turn that set into model-selection data, not independent TEST.

## Usage and compatibility

Default `--contribution-clip 0` executes the unchanged algorithm. To bound each
scene use `--contribution-clip <calibrated tau> --contribution-unit-size 1` with
`--backprop-mode full --agc 0`. Values >1 for unit size are approximate blocks.
This is independent of `--window-steps 50`: the H500 adjoint chain is not cut.

The new `configs/response_raptor_bounded_influence_smoke.args` is ONLY an 8-scene,
one-update H500-cap GPU smoke, not a replacement for the 512-scene training
configuration. Its tau=0.1 is a test setting. `--mode profile` limits the update
budget but finishes the current update; it is not a hard wall-time timeout.

Old exact resume still rejects changed source/config. Compatible
`--init-checkpoint` is weights-only and starts fresh Adam; do not compare that
to a baseline restored with historical Adam. New-run resume restores the complete
bound configuration, Actor, Adam and RNG exactly. Deployment ignores the
aggregator and keeps the same Actor architecture and inputs.

## Reproducible validation and limits

```
PYTHONPATH=. python -m pytest -q tests
python tools/verify_bounded_influence.py --output verification-results.json
# Optional parent comparison: pass --baseline-root <checkout of 1564d25>.
# On the actual GPU: append --device cuda.
```

The CI workflow runs the tests and bounded probes on CPU and uploads the exact
source/report artifacts. It does not run long training. Tests cover weighted
sum equivalence, single-outlier caps, zero/unused gradients, finite costs with
nonfinite derivatives, FP32 large finite norms, partial blocks, within-block
cancellation, unchanged forward state/RNG, exact new-run resume, source/config
binding, and pre-Adam rollback. Two additional CUDA checks skip without a GPU.

Reports distinguish randomized H500-CAP runs (with early failures) from a separate
constructed hover fixture that actually executes all 500 steps. The fixture
uses a sampled airframe duplicated with zero disturbances, equilibrium rotor
initialization and a constant-output Actor; it tests a long computation graph,
NOT learned control quality or randomized coverage. Injected backward multipliers
are synthetic sensitivity faults; they are NOT measured physical instabilities.
A bounded tanh recurrence also shows a zero forward state and explosive H500
Time Decay gradient, then verifies pre-merge clipping.

Do not infer convergence, a speedup, 17065 recovery, all-scene robustness, or
real-flight safety from these numerical tests. Single-scene separation costs
one VJP per scene; measured timings must be considered before long training.

Primary implementation references:
- PyTorch `torch.autograd.grad`: retained-graph vector-Jacobian products.
  https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
- PyTorch `clip_grad_norm_`: whole-parameter-vector norm semantics.
  https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
