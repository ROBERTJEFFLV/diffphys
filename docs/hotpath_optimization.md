# Verified hot-path optimization

Base: `83a75e272787524489925244f0aa6e1d1a078ce9` (`master`, 2026-09-29).
The baseline archive's Git tree was independently reconstructed as
`349845529a99f83efc63333690f31e7b877347d0` before edits.
See [machine-readable evidence](hotpath_validation.json).

## Implemented, enabled by default

- TRAIN no longer constructs the unused observation tape. Observations used by
  the Actor are still acquired after the same Time Decay gates, with the same
  acquisition noise, differentiable velocity history and survivor indices.
  `rollout()` retains observation recording by default for EVAL/other callers.
- TRAIN computes squared Huber features once per metric chunk; diagnostic
  statistics reuse detached values instead of recomputing the same features.
  The loss/reduction order and full cross-chunk BPTT are unchanged.
- The group probe casts directly when adding local derivatives into its FP64
  accumulators, avoiding six explicit FP64 temporary conversions per live step.
  Group matmuls, FP64 accumulation, clipping and averaging are unchanged.

A counted H8 fixture now invokes observation acquisition 9 rather than 18 times
(including initialization), and constructs the task features once rather than
twice. The changes do not modify native Actor/physics, the 128-cell sampler,
noise/pulses, time horizon, loss weights, Time Decay, clip caps, Adam, or the
synchronous audit durability/rollback policy. Protected core hashes are unchanged.

## Optional RK4-step compiler

`--physics-backend compile` compiles only `RaptorSimulator.step` with Inductor,
`fullgraph=True`, and dynamic batch dimensions. The original function remains the
single source of the RK4 equations. The compiled callable is constructed once
per run and reused. The default remains `--physics-backend eager`.

Native GRU/Linear hooks and survivor compaction stay outside the compiled region.
This is not H25/H50/H500 whole-rollout compilation. CUDA Graphs, TF32/AMP,
handwritten adjoints and fallback to eager on compiler failure are not enabled.
Compilation failures propagate. Changing batch sizes and grad mode can still
cause specialization/compile cost; `dynamic=True` is not a no-recompile promise.

The selected backend and compiler settings are checkpoint-bound and included in
source/audit evidence. Evaluation and update replay use the saved backend.
Existing exact-resume source/config checks are not weakened.

## Validation completed

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, one CPU thread; CUDA unavailable.

| Check | Result |
| --- | --- |
| Pristine baseline suite | 307 passed, 10 skipped |
| Final default suite | 324 passed, 16 skipped |
| Real Inductor suite, explicitly enabled | 6 passed |
| Default eager vs pristine base, three fixed-input cases | Bitwise identical costs, raw group vectors, group norms/scales, clipped gradient, metrics, Actor/Adam update and RNG |
| Eager vs compiled H500 preflight, cold + three warm comparisons | All numerical and physical-transition-count checks passed |

Default-suite skips include the five opt-in compiler tests and unavailable
CUDA/GUI-dependent checks. The six compiler tests were also run separately using
real Inductor, not a mocked compiler or `backend="eager"`. They cover FP32/FP64
RK4 state/action VJPs, changing batch size, H55 across metric-chunk boundaries,
Time Decay both on/off, delayed sensing, pulses, full raw group derivatives, and
an actual one-update Actor/Adam/RNG replay. Source checks rejected an in-progress
source edit during an earlier replay attempt; the final suite was rerun after
freezing the source, without relaxing those checks.

Commands from repository root:

```bash
python -m pytest -q tests
DIFFPHYS_TEST_COMPILE=1 python -m pytest -q tests/test_compiled_physics.py
```

The optional compiler tests may take minutes on a cold compiler cache. No long
training run was used as a correctness test.

## CPU timing evidence, not a GPU speed claim

For default eager changes, the pristine and optimized sources used identical
fixed inputs, one cold iteration and seven warm iterations each, run serially.
These are warm medians for rollout plus gradient computation, excluding sampling,
Adam, audit storage and periodic EVAL/checkpoint overhead.

| Scenes / horizon / memory | Valid transitions | Base (s) | Optimized eager (s) | Speedup |
| --- | --- | --- | --- | --- |
| 2048 / 8 / 64 | 16384 / 16384 | 0.32434 | 0.30916 | 1.049x |
| 128 / 50 / 8 | 6321 / 6400 | 0.51441 | 0.47167 | 1.091x |
| 128 / 500 / 8 | 11590 / 64000 | 1.77930 | 1.51989 | 1.171x |

A separate same-source, alternating-order N128/H500/GRU8 compiler preflight
measured 1.83501 s eager versus 0.95713 s compiled, or 1.917x warm speedup.
Its first compiled rollout+backward took 120.01 s and was excluded from the warm
median. That fixture executed 11494 of 64000 possible physical transitions.

Both experiments use a fresh random Actor with many early terminations, NOT a
stable trained policy. The two tables/runs use different fixed sampling attempts;
do not multiply their speedups. These CPU numbers do not establish CUDA latency,
CUDA memory use, or the user's full N2048/H500/GRU64 6-second update performance.

## Target-GPU admission check

Use an idle target GPU; profiling and latency measurements should be separate.
Start with a bounded shape before the full-shape test. This preflight is read-only:
no Adam, checkpoints, running training service or existing run directory changes.

```bash
python tools/benchmark_hotpath.py --device cuda --scenes 128 \
  --horizon 50 --memory-dim 64 --repeats 3 \
  --report /tmp/diffphys-preflight-small.json

python tools/benchmark_hotpath.py --device cuda --scenes 2048 \
  --horizon 500 --memory-dim 64 --repeats 5 \
  --report /tmp/diffphys-preflight-h500.json
```

The command fails on numerical disagreement or missing CUDA. It compares the raw
128 group vectors, not only the already-clipped gradient. It reports compilation
warmups separately and checks every comparison including warmups. A run with
`--backends eager` is valid for profiling but reports `passed: null` because no
backend equivalence comparison took place.

Passing a random-policy preflight is not sufficient to establish throughput for
a mostly-surviving trained policy. Measure representative checkpoints, including
high-TTI/pulse conditions and long-lived rollouts, then record full trainer
`forward_seconds`, `backward_seconds`, `optimizer_seconds`, `audit_seconds` and
wall-clock throughput. Compiler and eager floating-point differences can change
long training trajectories even when local numerical tests pass.

## Running this branch safely

The checked-in config automatically uses the verified default eager improvements.
Use a NEW work directory; do not overwrite or exact-resume a run from old source.

```bash
python tools/train_response_control.py \
  @configs/response_raptor_multi_airframe.args \
  --work-dir runs/hotpath_eager/seed7
```

Only after target-device validation, select `--physics-backend compile` and a
separate new directory. Cold compilation counts toward the trainer's elapsed
budget. `--init-checkpoint` is an explicit weights-only initialization with fresh
Adam/sampling, not an exact continuation. Retain the old checkout and run for
historical replay; do not rewrite source/checkpoint/calibration hashes.

## Intentionally deferred

Fixed-shape alive masking needs safe inactive inputs and exact first-failure,
zero-adjoint and frozen-history validation; a final `where` alone does not make
invalid intermediate arithmetic safe. Whole-rollout compile/CUDA Graphs need
separate hook, dynamic-shape, graph-lifetime and memory validation. A fused GRU,
physics custom adjoint or persistent GPU kernel needs exact discrete RK4,
Time Decay/history and group-clipping derivatives, not a generic adjoint formula.
Those changes, precision changes, asynchronous audit and sampler pipelines were
not silently introduced. Deployment remains unauthorized.
