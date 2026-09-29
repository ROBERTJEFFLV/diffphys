# One native backward traversal for exact physical-group gradients

## What changes, and what does not

`--group-backward probe` is the production default. `response_grad_probe.py`
observes the existing native `nn.GRUCell` and `nn.Linear` outputs and computes
their local parameter contributions grouped by the existing physical-cell IDs.
One scalar vector-Jacobian product traverses the original full H500 graph.
No separate group VJP traversals or second flight rollout are performed.

The deployment Actor (including `response_policy.py`) is unchanged byte for byte.
Physics, sensors, pulses, 128-cell sampler/calibration, H500, Time Decay, CVaR,
fixed shrink-only cap, global clipping and Adam are retained. `response_task`
only gains an optional training observer: it passes the current original live
row indices to the observer immediately before the unchanged Actor invocation.
No group ID, physical parameter or privileged state enters the Actor.

The previous eight-chunk VJP implementation is retained **only as an explicit
reference backend** (`--group-backward vjp`). It does not run or silently replace
a failed probe. `--group-vjp-chunk-size` affects that reference only; changing it
does not accelerate or alter the probe.

## Why one scalar backward is sufficient here

Let C_i denote each independent scene cost, and let the existing coefficient
matrix A contain its original pooled mean+CVaR weights scaled by N/n_g, one
nonzero group per scene. The required pre-clipping group derivative is

    h_g = gradient_scale * sum_i A[g,i] * dC_i/dtheta.

The single native VJP uses

    s_i = gradient_scale * sum_g A[g,i].

Because scenes do not interact, the gradient at a per-scene activation belongs
only to that scene's group, even though all scenes share Actor parameters.
Hooks observe that adjoint BEFORE the native parameter gradient sums lose
scene identity. They never change the adjoint returned to native autograd.

For Linear, with y = W x + b:

    local dW_i = delta_i x_i^T
    local db_i = delta_i.

For the native PyTorch GRU, in r,z,n order:

    r = sigmoid(i_r + h_r)
    z = sigmoid(i_z + h_z)
    n = tanh(i_n + r*h_n)
    output = (1-z)*n + z*previous_hidden.

The probe recomputes these local gate values from detached captured inputs and
the unchanged weights, then evaluates the partial derivatives for W_ih, W_hh,
b_ih and b_hh. The recurrent candidate bias is multiplied by the reset gate,
just as PyTorch specifies. It does not substitute a custom GRU forward,
dispatcher or state-backward kernel.

The observed output adjoint already contains the effect of FUTURE physics,
actions, memories and delayed-velocity history through native BPTT, including
the existing Time Decay gates. Therefore local parameter contributions must
be summed over ALL uses in time BEFORE calculating each group's norm. Squaring
or clipping each timestep and then summing would be a different algorithm.

After the one native traversal, the same `normalize_group_rows` shrink-only
routine clips each complete group derivative, averages all groups and publishes
the resulting parameter gradients. Normal global clip and Adam follow. The
aggregate derivative returned by native `autograd.grad` is not published; it
has already lost group identity. No extra division by 16 or 128 is introduced.

This is an analytic calculation of group gradients, **not a norm estimator**.
It assumes separable scenes and the current native GRUCell + Linear Actor.
BatchNorm, inter-scene attention/collisions, cross-scene differentiable rewards,
new parameterized layers, higher-order differentiation and mixed precision
require a new derivation/validation. The current probe validates the six
trainable Actor parameter tensors and refuses unsupported module replacements.

## Storage and numerical behavior

Original scene IDs map to unique padded group slots. After compaction, only the
current live rows contribute to those original slots. Zero padding contributes
nothing; a terminated scene never joins a different group. The packing uses
unique `index_copy_` writes, not floating atomic scatter sums. Grouped matrix
multiplications reduce the 16 scene members directly. There is no full NxP
per-scene derivative tensor, and no HxNxP derivative tape.

The persistent accumulators are GxP in float64: 128 x 16068 x 8 bytes = about
15.7 MiB for the production Actor, plus normal graph/activation storage and
small temporary packed tensors. Float32 local gate calculations/group matmuls
remain float32; float64 accumulation does not promote the whole physics graph.
Saved input aliases are detached only for the observer, not for native BPTT,
and are released as hooks execute or on error. Forward hooks are removed after
rollout; output hooks are single-use and removed after backward. EVAL and
deployment do not install the observer.

Native CUDA GRU fusion and alternative matrix reduction orders can introduce
roundoff differences between the probe and reference VJPs. We require matched
forward trajectories and numerical gradient agreement, NOT bitwise equality
between different backends. An ill-conditioned closed loop may amplify small
weight differences during later training; this is not removed by the probe.
Same-backend recorded updates remain subject to the existing strict
source/PyTorch/device checks and read-only replay tests.

## Evidence and profiling

Existing Actor/Adam/RNG before-and-after capsules, group norms and coefficients,
actual parameter-step reports, alert windows and EVAL checkpoints are retained.
The backend choice is stored in `binding.group_balance.backward_backend` and
the new implementation is included in `source_sha256`/`audit/source.zip`.

Group reports now include `backend`, `graph_traversals`, observed GRU/readout
hook counts and accumulator bytes. `vjp_calls=1` for probe; the reference keeps
its actual chunk count. `vjp_chunk_size=null` for probe is intentional.
TRAIN additionally reports `backward_seconds`, `optimizer_seconds` and
`audit_seconds`, alongside existing `forward_seconds` and `update_seconds`.
`forward_seconds` retains its existing inclusion of sampling/metrics; the
new audit phase measures capsule creation and retention, not just file bytes.

Read-only numerical/timing comparison (no optimizer update):

    python tools/benchmark_group_backward.py --device cuda --scenes 2048 \
        --horizon 500 --memory-dim 64 --repeats 3 \
        --report /tmp/group_probe_benchmark.json

The tool alternates backend order and includes a warmup. It reports forward,
backward and combined gradient-phase time, CUDA peak allocation when available,
forward-cost identity, group-norm/multiplier agreement and clipped-gradient
agreement. The timings EXCLUDE Adam and capsule I/O, so they must not be labeled
as complete training-update latency. Do not extrapolate CPU speedups to CUDA.

Run focused regressions:

    python -m pytest -q tests/test_group_grad_probe.py tests/test_update_audit.py
    python -m pytest -q tests

Tests compare local GRU derivatives (including reset-gated recurrent bias),
all grouped rows against independent VJPs, float32/64, unequal adaptive groups,
fixed 128 cells, compaction, time decay, delayed observations, pulses across
metric chunks, long horizons, one-traversal execution, cleanup on failure,
nonpublication of partial gradients and strict recorded Actor/Adam/RNG replay.
CUDA-specific fused-GRU checks skip explicitly without a GPU.

## Compatibility and limits

This is a new source/backend and is not an exact resume of old source.
Preserve the old checkout/run. The existing `--init-checkpoint` path imports
the same Actor interface into a new directory and starts fresh Adam/sampling;
no checkpoint hashes are rewritten. Compatible historical EVAL still uses
its saved loss and unchanged physical/sensing model. The provisional cap 1.0
is not retuned here.

One graph traversal still includes native backward, gate recomputation, hooks,
group matmuls and evidence I/O. Reduced traversals are not proof of a particular
GPU update time, high-TTI convergence or safe real flight. No learned-performance
claim or deployment authorization is introduced.

Primary implementation references:
- PyTorch GRUCell gate equations (`torch.nn.GRUCell`, also in the installed docstring).
- PyTorch `Tensor.register_hook` (read-only hooks return None).
- Opacus guide to grad samplers (activation/adjoint layer-local parameter derivatives).
No Opacus dependency, DP noise or privacy claim is added.
