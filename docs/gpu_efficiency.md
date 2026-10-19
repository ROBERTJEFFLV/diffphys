# Production GPU execution: Compiled + dynamic compaction + GC + prefetch

P is now the production training implementation, entered only through
`tools/train_response_control.py`. Its checked-in argument file is
`configs/response_raptor_multi_airframe.args`; it uses a new run directory
`runs/compiled_prefetch_3072/fresh_seed7` to avoid mixing with source-bound older runs.
It selects 3072 TRAIN scenes (`--scenarios 768`), with 24 scenes in each of
the same 128 physical/gradient groups. The local RTX 4060 Ti 8 GiB H500 sweep
measured 1.896/1.925 seconds per complete steady update at N2048/N3072;
scene throughput increased 47.3%, while N4096 exhausted memory. This is a
short throughput qualification, not a convergence comparison. Raw evidence is
in `reports/batch_sweet_spot_20261019/result.json`. A batch-size change starts a
new run through weights-only initialization; old Adam/bindings are not rewritten.

## Runtime

- `diffphys/compiled_physics.py` selects strict full-step compilation on CUDA;
  CPU explicitly uses the native `RaptorSimulator` as numerical reference.
- The captured function is the original bound `RaptorSimulator.step`. Inductor
  compiles it with `fullgraph=True, dynamic=True`. H500 control flow and the
  native GRU Actor remain outside compilation; AOTAutograd differentiates the
  original physics region.
- `diffphys/strict_inductor.py` preserves native sum/cross/vector_norm/bmm
  boundaries in forward/backward and fixes precision/division-rounding controls.
  CUDA Graphs remain disabled. The backend is pinned to verified PyTorch 2.10;
  errors propagate without switching to a looser numerical implementation.
- `diffphys/response_execution.py` defers automatic cyclic GC only during
  forward/backward, detaches completed costs after backward and closes Probe
  payloads before restoring the original GC setting. Final collection and
  outstanding worker shutdown remain inside run elapsed time.
- `diffphys/scene_prefetch.py` runs one persistent CPU sampler using `spawn`,
  with at most one queued/running/ready next batch. It reuses the original
  random/coverage128 sampler, local seed generators, dtype, disturbance config
  and configured CPU thread count. It creates no Actor, Adam or CUDA context.
- Main takes batch k by index, uploads using the original blocking
  `RaptorState.to`, then submits k+1 before GPU training k. The next forward
  uses the newly updated weights. Successful commit alone advances progress.
- Original dynamic scene compaction, stable scene IDs and shared immutable
  noise/pulse tapes remain unchanged. Fixed Batch/Mask and bool-fill experiment
  switches are absent from production.

The only production CUDA physics path is strict Compiled; the rotation-only
selector has been retired. Native CPU physics and explicit grouped VJP remain
numerical references. Actor/loss/RK4/Time Decay/group clipping/Adam kernels are
unchanged and still checked by `tests/core_contract.json`.

## Checkpoint and failure behavior

Binding includes new production source files, physics backend, PyTorch/CUDA/cuDNN
versions, GPU identity, TF32 policy,
spawn sampling, thread configuration, GC policy and dynamic compaction. Exact
resume requires the same binding. Source promotion creates a new binding;
old checkpoints require their archived source for exact resume. The existing
`--init-checkpoint` continues to import compatible Actor weights with fresh Adam.
It does not transplant old learning state or rewrite a source digest.

Unused prefetched batches are not checkpoint state. Shutdown cancels queued
work when possible and joins running work; resume regenerates the saved next
index. Forward/backward/audit failures preserve their evidence, roll back
Actor/Adam/main RNG, close Hook payloads, restore GC and join the worker.
Normal SIGINT/SIGTERM stops at a committed update boundary.

The retired rotation wrapper was removed from `env_raptor.py`; protected physics
ASTs and calibration thresholds are unchanged. `configs/physics_coverage.json`
records the new generator file digest for that source-only edit. This is a new
source/config binding, not a modification of any existing checkpoint.

## Measured evidence before promotion

All timings below are historical qualifications on RTX 4060 Ti 8 GiB,
PyTorch 2.10.0+cu126/CUDA12.6, FP32/TF32 off, N2048/H500. They are independent
stage comparisons and must not be multiplied into a claimed end-to-end gain.

| Comparison | Baseline | Adopted | Scope |
| --- | ---: | ---: | --- |
| Rotation-only vs strict step | 3.496 s | 2.062 s | Frozen core computation |
| Automatic GC vs C | 470.247 s | 432.711 s | 200 true continuous updates, 7.98% lower training total |
| C vs P | 244.440 s | 216.016 s | 100 updates, complete process wall, 11.63% lower |
| C vs P training interval | 222.419 s | 194.167 s | Includes worker start/first wait/exit and final I/O/GC |

For C/P, CPU generation totals are 25.442/25.144 s; main batch wait is
25.442/1.732 s including first startup; delivery/upload is 1.377/1.550 s.
P's remaining 99 batch waits sum to 0.498 ms. All ten matched ten-update blocks
improve, within one C-then-P process pair. Forward/backward timings also vary,
so the full reduction is not attributed entirely to removed sampling wait.

C passed A200/C200/C100+100 acceptance (600 actual updates). P passed
C100/P100/P50+50 acceptance (300 actual updates). All per-update Actor, Adam,
main RNG, sampling/group reports and final gradients matched raw bytes; key
H500 trajectory/raw group-gradient replays and independent-process resume
matched too. Newly triggered learning-time compilations were zero. P has a
bounded additional CPU process/cache/batch footprint in the 100-update window;
post-release CUDA live storage remained 19,339,264 bytes at every boundary.

The unchanged source snapshots and all raw report/checkpoint artifacts remain
in `reports/`. See [archive index](archive/README.md) and the detailed
[historical P report](archive/cpu_prefetch.md). These measurements establish
numerical continuity and bounded-run throughput on the qualified stack, not
cross-version performance, convergence or indefinite memory stability.

## Why unconstrained compilation was rejected

Compiler fusion changed polynomial rounding, four-term reduction order and
other sensitive arithmetic. These local differences entered RK4 stages and
accumulated through the closed loop. Retaining the four native boundaries plus
rounding controls restored byte equality. The validated boundary set is
sufficient; a smaller necessary set has not been established. See the archived
[RK4 drift diagnosis](archive/gpu_efficiency.md).

## Current checks

```bash
python -m pytest -q tests
python -m pytest -q tests/test_compiled_physics.py  # CUDA H500 check when available
```

The bounded CUDA test compares complete H500 trajectories, raw group gradients,
final gradients and a disposable Adam update with the native reference. It
performs no sustained training or saved checkpoint modification.

For read-only timings on a fixed Actor and tapes:

```bash
python tools/profile_response_training.py --device cuda --scenes 2048 --horizon 500 \
  --checkpoint /path/to/native_B.pt --warmup 1 --repeats 5 \
  --epsilon-p .01 --epsilon-a .01 --lambda-R 25 --report /tmp/profile.json
```

This profiles current forward/backward only. It does not call Adam or represent
complete production elapsed time with sampling overlap, EVAL or I/O.
