> Archived pre-promotion experiment. Runtime commands require the source in
> [optimization_sources_20261019.zip](optimization_sources_20261019.zip).
> See [current production execution](../gpu_efficiency.md) for the adopted P path.

# Continuous Compiled candidate acceptance, 2026-10-19

This study promotes the [C scheduling experiment](dynamic_scheduling.md) to a
continuous-learning acceptance candidate: strict Compiled physics, original dynamic
compaction and automatic cyclic-GC deferral through forward/backward. Bool-fill
is a separately bound, disabled option; fixed Batch/Mask is not advanced.
Production remains unchanged. CPU prefetch is gated on these results.

**Result:** C passes continuous-learning and fresh-process resume checks on the
tested RTX 4060 Ti / PyTorch 2.10.0+cu126 configuration. All 600 updates preserve
the exact state chain, and all selected H500 replays match at the byte level.
Including normal I/O and final cleanup, uninterrupted C takes **432.711 s**
versus A's **470.247 s**, a **7.98% total-time reduction**. This result does not
authorize deployment or establish learned control quality.

## Scope and initialization

The authorized GPU budget is A200, C200 and C100 + a fresh-process resume100:
600 actual optimizer updates. Each process performs one frozen startup
forward/backward for compiler stabilization, without calling Adam. Startup cost
is reported separately; any compilation triggered during learning is inside
the training clock. Offline correctness checks do not contribute to performance
claims and do not call Adam.

Every learning update carries the previous model and named Adam state. The
sampler uses the successful update index, 0..199, and fresh scenes/noise/pulses
under the original coverage128 protocol. This is environment resampling, not
resetting learning state. Total scenes are 2048, H500, groups128, FP32, native B,
Time Decay=1, `.01/.01/25` loss, group cap1, gradient scale0.1 and global cap10.

The old local `best.pt` is an explicit **model + named Adam experiment donor**.
Its original file/bindings are retained. New experiment sampling progress starts
at zero, and donor Adam step 78,950 advances to 79,150 after 200 updates. This is
neither production's weights-only `--init-checkpoint` nor exact resume of old
source. The donor SHA256 is
`5ddbef550e1bd5a15701caa32d0244e1512b38b6a212b13cc5c3665cd2764851`.

New candidate checkpoints have schema `continuous-compiled-candidate-v1`. Their
binding includes actual config, production/experimental source hashes, donor
provenance, objective/group/disturbance settings and numerical runtime/backend.
`--resume` checks this binding and sampling progress. No old checkpoint hashes
are rewritten. Separate run directories preserve all 200 audit capsules for
each learning chain, including Actor/Adam/RNG before and after every update.

## Execution

```bash
python tools/train_compiled_candidate.py --scheme A --donor best.pt --work-dir /tmp/accept-A
python tools/train_compiled_candidate.py --scheme C --donor best.pt --work-dir /tmp/accept-C
python tools/train_compiled_candidate.py --scheme C --donor best.pt \
  --work-dir /tmp/accept-split --stop-after 100
python tools/train_compiled_candidate.py --scheme C --resume /tmp/accept-split/latest.pt \
  --work-dir /tmp/accept-split
```

The checked-in experiment config is `configs/compiled_candidate_acceptance.json`.
`--stop-after` controls a segment boundary without changing the bound 200-update
target. `--bool-index-fill` is independent, default off, and changes the binding;
it is not enabled in these comparisons. New training directories must be empty.
These commands spend the stated 600-update budget; do not rerun without a new
training budget.

Correctness runs separately:

```bash
python tools/verify_compiled_continuity.py --root reports/continuous_compiled_20261019
```

This checks every audit state chain, then frozen H500 replays of updates
1/50/100/150/200 for trajectory/termination and raw group-gradient comparisons.
Actual audit gradients are also checked against the frozen replay. It performs
no optimizer updates.

## Real learning-state continuity and exact resume

All 200 audit capsules in each of A, C and C_split were compared offline.
For every numbered update:

- Actor/Adam/RNG before the update exactly match that run's preceding output.
- A/C/C_split have identical input and output Actor/Adam/RNG bytes, sampling
  seeds/reports, clipping/group reports, gradients entering Adam and parameter
  change metrics.
- Adam steps increment once; every run has 200 nonzero actual parameter changes.
  The optimizer is not reinitialized between updates.
- Sampling indices are exactly 0..199, with 200 distinct four-bank seed tuples.
  The resumed process begins at index 100 and performs updates 101..200.

The original midpoint checkpoint is preserved as `C_split_100.pt`, with its
hash in the evidence manifest. Its model/Adam/RNG are also checked against the
uninterrupted C update-100 audit output. Final C and C_split agree on Actor,
Adam, RNG, `next_update`, sampling progress and all algorithmic progress fields.
Elapsed wall-clock fields are excluded from state equality.

Final Actor SHA256 for all three learning chains:
`2d863df04994c8b4847d9e6c36d909fb45aac71efbfc6fdd95b426ae47b199a9`.

Selected updates 1/50/100/150/200 were replayed separately from their actual
audit input weights and RNG. These checks compare all captured step inputs and
outputs, physical/history state, memory, active/termination and returned H500
trajectory fields, raw 128-group gradients, clipping and gradients entering Adam.
The replayed final gradients also equal the corresponding actual update's
audit gradients. **All compared tensor bytes match**, including signed-zero
representations; no tolerance relaxation is used. Replays call backward but
never Adam and are outside the performance runs.

## Performance, compilation and cleanup costs

| Learning chain | Actual updates | Training total incl. final I/O/GC | Startup warmup | Cycle median | p90 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A uninterrupted | 200 | 470.247 s | 20.087 s | 2.3553 s | 2.4279 s | 2.4689 s | 2.4755 s |
| C uninterrupted | 200 | **432.711 s** | 18.953 s | **2.1494 s** | **2.2031 s** | **2.2370 s** | **2.3296 s** |
| C 100 + fresh-process resume100 | 200 | 439.565 s | 37.766 s, two starts | 2.1874 s | 2.2321 s | 2.2872 s | 2.3600 s |

Cycle timing includes snapshot, fresh sampling and its unchanged transfer,
forward/statistics, native backward/group clipping, global clipping, persistent
Adam, finite checks, real update-audit CPU copy/fsync, history fsync, graph/probe
release, telemetry, GC-event logging and periodic checkpoint saving. Normal
checkpoints occur every 50 successful updates. The segment total additionally
charges remaining timing-log/progress-output overhead, final GC and final
checkpoint/GC-log I/O, so cycle sums do not equal segment totals. There is no
per-update correctness comparison or restored-checkpoint reset inside these
performance runs. No EVAL was scheduled for this update/resume acceptance study.

Startup is a frozen full forward/backward compilation warmup with no Adam;
the saved RNG is restored afterward. Each of the four processes has four
unique Dynamo graphs after startup and at its end. **New graph compilations
during all 600 learning updates: zero.** Had one occurred, its wall time would
have remained inside the normal training timer. Source creation/import and
process-launch time are additionally preserved in `processes.json`; startup
timing is not presented as a free production operation. Split-run overhead
includes another process start and another final cleanup; the two learning
segments were not assumed to have identical operating-system timing.

Observed GC provides context for the speed difference:

| Observed collection | A200 | C200 | C100+100 |
| --- | ---: | ---: | ---: |
| Generation-2 events during forward | 154 | 0 | 0 |
| Their total duration | 24.778 s | 0 | 0 |
| Generation-2 events during audit | 0 | 1 | 2 |
| Audit generation-2 duration | 0 | 0.162 s | 0.306 s |
| Explicit final generation-2 drains | 1 | 1 | 2 |
| Final drain duration | 0.150 s | 0.156 s | 0.309 s |

C also changes when finished-graph destruction occurs. Its benefit is assessed
from total elapsed time, not by subtracting GC callbacks or assuming every
pause is removable. No full GC is forced every update. One final drain per
process is charged to that process's training time. GC callbacks are enabled
for both schemes and flushed to disk each cycle rather than retained unboundedly.

## Memory, Hooks and exceptional exits

All 600 update boundaries have exactly **19,339,264 bytes (18.443 MiB)** of
allocated CUDA storage after graph/result release and gradient clearing. There
is no live tensor-memory trend. C's RSS rises during allocator warming and then
fluctuates rather than growing monotonically: 20-update block medians are
1989/2035/2022/2029/2018/2001/2015/2025/2021/2013 MiB. Its last 50 measurements
range from 1981 to 2041 MiB; final post-GC RSS is 2127982592 bytes. The first-to-last
RSS difference alone is not interpreted as a leak or as constant memory. This
is bounded 200-update evidence, not a guarantee for indefinite training.

Native Probe.close remains the source of Hook/payload cleanup. Each actual
update verifies it is closed, its payloads/handles are empty, and weak probe and
record references die immediately after releasing the result, without forcing GC.
Final cleanup runs after result/gradient release. All four actual process exits
restore GC's original enabled state and report no retained module forward hooks.

The candidate GC context records `gc.isenabled()` and restores that exact state
in `finally`, including originally disabled entry. Tests cover normal completion,
real OS SIGINT at an update boundary, forward and backward exceptions, and direct
KeyboardInterrupt. Failed updates preserve their audit evidence and roll back
Actor/Adam/RNG/sampling; Hook payloads are cleared before cleanup. Forward/backward
failure injections and SIGINT are deterministic CPU tests of the shared runtime;
the large CUDA experiment covers normal and saved-boundary fresh-process exits.
There is no claim of surviving an arbitrary kill during an optimizer kernel.

## CPU preparation wait and the next-stage gate

The original blocking sampler is timed without changing its sampling or transfer
operations. A scoped timer observes the original `RaptorState.to` calls; this
does not add pinning, streams, asynchronous transfer or CPU workers.

| Preparation in uninterrupted C | Median per update | Sum over 200 |
| --- | ---: | ---: |
| CPU scene/physics/noise/pulse preparation | **0.2560 s** | **51.357 s** |
| Original transfer to CUDA, including its host overhead/wait | 0.0132 s | 2.677 s |

CPU preparation occupies **11.87%** of C's measured training total. Therefore
perfectly covering this CPU work could save at most about 11.9% of elapsed time
on this workload (about 1.135x throughput); this is an ideal bound, **not a
measured prefetch speedup**. GIL/CPU contention, handoff and the unchanged blocking
transfer can reduce or eliminate the benefit.

The sampling wait is large enough to justify a later one-task, one-prepared-batch
experiment. No prefetch implementation or performance claim is included here.
The next experiment must prepare only index n+1's CPU initial state/parameters/
tapes, consume by index, and still execute its forward using updated n weights.
Unused prepared data must be discarded/regenerated from saved sampling progress
on resume. C remains the accepted candidate baseline for that comparison.

## Evidence and limits

Local ignored evidence is under `reports/continuous_compiled_20261019/`:
three run directories with 600 real update-audit capsules/source archives,
all actual checkpoints/history/timing/GC logs, the preserved midpoint checkpoint,
process commands/wall times, `state_chain_comparison.json`,
`key_h500_comparison.json`, `performance_summary.json`, fresh test logs and
`evidence_manifest.json` / `source_snapshot.zip`. The manifest verifies original
donor/protected-kernel/production-source identity and all measured source hashes.

The production source digest is still
`0965252f0b69a6d9f2a755d9ad2dafae2f2ac20e9503b58736f903414fe8af61`.
Numerical identity is qualified only on the tested RTX 4060 Ti, PyTorch 2.10,
CUDA 12.6 and matching software/configuration. It is not generalized to another
GPU/compiler release. Production defaults, nonzero compaction, the RK4/Actor/
Time Decay/group/optimizer implementation and old checkpoint hashes are unchanged.

Fresh checks: **520 CPU tests passed, 24 skipped**; **35 focused CUDA tests
passed**. The actual GPU training driver and offline verifier both exited zero.


## Subsequent source version

The [single-worker CPU prefetch study](cpu_prefetch.md) extends the independent
runner to C/P and uses new v2 source/config bindings and fresh run directories.
The v1 measurements and exact midpoint checkpoint above remain associated with
`reports/continuous_compiled_20261019/source_snapshot.zip`. Reproducing or
resuming that historical v1 experiment requires its archived source; the current
v2 runner does not relax its binding to accept a v1 checkpoint.
