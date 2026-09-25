# Scoped native GRU VJP candidates and rejected broad compaction

Base: `0417ad3bd5b546e7e9ed3bd813e19fdf19236f6d`, same experimental branch.
No Actor, physical step, forward rollout, position-only termination, task loss,
CVaR coefficients, group layout, normalization, Time Decay or Adam formula changes.
The original modern-vmap backend remains DEFAULT. CUDA candidates are opt-in
until the user's ordinary AND high-sensitivity checkpoints pass the CUDA audit.
CPU checks of a stand-in are explicitly NOT CUDA kernel validation.

## Source diagnosis

`response_policy.py` uses native `nn.GRUCell`. The CUDA implementation first
computes the two affine gate inputs, then calls `_thnn_fused_gru_cell`; its
backward uses `_thnn_fused_gru_cell_backward`. A missing FuncTorchBatched rule
invokes the native operator separately for each group. Merely changing the
number of Python vmap calls does not remove those operator-level loops.

The pinned PyTorch 2.2 native CUDA kernel consumes a saved `[B,5H]` workspace
and a `[B,H]` cotangent. Gate derivatives are elementwise/row-local. Its two
bias derivatives are the sums of gate derivatives over B, not over groups.
We reuse this ORIGINAL native kernel, rather than rewriting/unfusing the GRU
or recomputing its forward gates. No Triton/CUDA extension, TF32 or AMP change.

## Candidate 1: native

Pack `[G,B,H]` into `[G*B,H]`, repeat the saved workspace, invoke the native
kernel ONCE with `has_bias=False`, and restore the G dimension. Compute each
bias sum at its original `[B,3H]` shape. This removes G native kernel launches
but keeps G*B arithmetic; workspace expansion and bias launches still cost time.
No speed claim follows from the invocation count alone.

## Candidate 2: sparse (narrow, not whole-graph reorganization)

In this simulator scenes do not interact, and each scene belongs to one group.
At a row-local GRU output, at most one group's cotangent is nonzero. Choose that
cotangent per scene, call the native kernel on B rows, and scatter its output
back to the ORIGINAL `[G,B,...]` shape. All downstream matrix products, bias
reductions, temporal accumulation, and group normalization keep their old shapes.
Thus the GRU's group-outside rows need not repeat gate arithmetic or workspace
loads. This does NOT remove all group-outside work in the physical backward.

The disjoint-support and finite-workspace assumptions are checked on-device for
every call, then tested before any Actor `.grad` is published. A violation aborts;
it never silently drops a second active group, ignores tiny gradients, changes
a task weight, or retries a partially consumed graph. No threshold defines zero.

## Integration and safety

The private helpers in `response_groups.py` supply a temporary `FuncTorchBatched` registration only
around the grouped VJP. The newer public register_vmap API is unavailable on
some target 2.2 installations, so the small private dispatcher adapter is isolated
here. Never override an upstream batching rule. Registration/hooks are removed
on exceptions too. A cost-tensor hook tags the autograd graph task (CUDA may use
a worker thread); unrelated graph tasks retain separate native calls. Nested
vmap is unsupported in this opt-in backend. No permanent global monkeypatch.

Flags: `--group-gru-vmap-mode fallback|native|verify|sparse|sparse-verify`.
`fallback` is unchanged production behavior. `verify` and `sparse-verify` compare
EVERY native GRU invocation against separate original calls before returning;
they are intentionally slower diagnostics. `native`/`sparse` are the candidates
to benchmark only after verification. Group-gradient logs count optimized calls,
logical rows and rows actually sent to the native kernel; zero calls is not an
accelerated validation. Shape arithmetic and raw scene order are unchanged.

The new mode and the already-bound `response_groups.py` source are checkpoint-bound. Old exact resume still
rejects source changes. Do not bypass checkpoint hashes or use weights-only
initialization as evidence of unchanged historical Adam. The audit restores
saved named Adam moments into shadow optimizers and does not write a checkpoint.

## Why broad sparse backward was NOT adopted

A separate `tools/probe_group_capture.py` experiments with one physical adjoint,
then reconstructs group parameter gradients from native affine-node cotangents.
It does not replace production code. It runs a private copy of rollout with a
row-ID metadata assignment; original outputs must compare equal. No group or
trajectory is dropped. Its local node API is research-only.

Measured locally, CPU float32, B512, H500 CAP, 16 groups, 39,931 valid transitions:

| Measurement | Dense modern vmap | Single-adjoint reconstruction |
| --- | ---: | ---: |
| Backward time (single sample) | 8.660 s | 3.503 s |
| Forward equality | reference | yes |
| Raw group gradient bit equality | reference | NO |
| Final Adam weights/moments equality | reference | NO |

Max absolute raw-gradient difference was 2.98e-8; relative L2 difference 3.01e-8.
Small is not identical, particularly at the reported sensitive checkpoint.
This candidate was rejected for default/production adoption. Changes in backward
organization and floating-point accumulation require stronger validation, not
an assertion that a small local error cannot affect later control.
The CAP fixture is not 512 learned aircraft all surviving to H500.

## Validation and target-GPU acceptance

Local CPU baseline: 171 passed, 4 CUDA skipped. After changes: 193 passed,
8 CUDA skipped (four new dtype/backend combinations). Stand-in tests check
native-call counts, bias grouping, disjoint support, undefined bias outputs,
noncontiguous inputs, and cleanup; no CUDA time is inferred. Two enabled-group
training updates match base Actor, named Adam state and RNG, with mode=fallback.
The CI repeats the existing H500 fixtures and records all skips.

Run the real CUDA tests:

```
PYTHONPATH=. python -m pytest -q tests/test_gru_vmap_backend.py
python tools/verify_vjp_acceleration.py --device cuda \
  --checkpoint /path/to/ordinary.pt \
  --checkpoint /path/to/high_sensitivity.pt \
  --output vjp-acceleration-audit.json
```

The same original graph, raw G-by-P gradients, normalized gradients, boundaries,
RNG, and one shadow Adam step (parameters AND moments) are compared. Full-group
and Adam checks use byte comparisons, including signed zero. Checkpoint mode
uses its own horizon, sampling index, task configuration and existing optimizer
history. An explicit `--attempt-index` can select an incident batch. Do not
mix a new optimizer with a restored reference optimizer. No source hash is forged.

A native-operator or final equality failure exits nonzero and prevents approval.
The GPU audit records separate untimed verification and timed fast passes, and
requires nonzero optimized-kernel counts. It is not a multi-seed long-training
or stability certificate. This environment has no CUDA device or the user's
mature checkpoint files; actual GPU feasibility/speed remains unverified here.
Keep `fallback` until those checks pass, then benchmark wall time and peak memory.

Primary sources inspected:
- PyTorch v2.2.0 `aten/src/ATen/native/RNN.cpp`, native GRUCell CUDA path.
- PyTorch v2.2.0 `aten/src/ATen/native/cuda/RNN.cu`, GRU backward and bias sums.
- https://docs.pytorch.org/docs/stable/library.html
- https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
