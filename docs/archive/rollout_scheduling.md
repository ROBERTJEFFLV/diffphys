> Archived pre-promotion experiment. Runtime commands require the source in
> [optimization_sources_20261019.zip](optimization_sources_20261019.zip).
> See [current production execution](../gpu_efficiency.md) for the adopted P path.

# H500 scheduling and complete-update benchmark, 2026-10-19

Follow-up: [dynamic compaction, bool writes and GC](dynamic_scheduling.md) retains
compaction, eliminates only the extra bool-write sync, and compares A/B/C/D over
20 timed updates each, including deferred cleanup in total elapsed time.

The independent strict Compiled backend completes an update in **2.173 s**,
versus **3.614 s** for production's rotation-only compilation: **1.66x** throughput
on this fixture. Fixed Batch + Mask removes all 1,000 synchronization calls
inside the 500 steps, but its complete-update median is **2.206 s**. This experiment
does not establish an additional net speedup from masks. Production is unchanged.

## Fixture and authorized budget

- RTX 4060 Ti, 8 GiB; PyTorch 2.10.0+cu126, CUDA 12.6, cuDNN 91002;
  Python 3.11, FP32, one CPU thread, TF32 disabled.
- Local `best.pt`: native hidden-only B, update 78,950, original named Adam
  moments and step counters. SHA256:
  `5ddbef550e1bd5a15701caa32d0244e1512b38b6a212b13cc5c3665cd2764851`.
- 2,048 TRAIN scenes, coverage128, 128 fixed groups, H500, dt=0.01 s.
  Identical initial state and immutable noise/pulse tapes in every trial:
  `f24f69dc63db49974b95cc97eaf37bc54183e27a8931c2516c0c90fa6ec0673d`.
  There are **1,018,203 valid physical transitions**. Mask schedules 1,024,000
  physics rows, only 0.57% more than compaction on this mature-policy fixture.
- Unchanged loss `.01/.01/25`, Time Decay=1, full BPTT, native group probe,
  gradient scale=0.1, shrink-only group cap=1, global cap=10, saved persistent
  Adam (lr=3e-4). No Actor compilation, new physics fusion or altered RK4 order.
- **One warmup + three timed updates per scheme**, three schemes, **12 updates
  total**. Every update restores the same weights, named Adam state and RNG.
  Three timed rounds rotate scheme order. The warmup also captures numerical
  evidence; timed updates contain no trajectory capture hooks or host snapshots.
- Experiment sampling index is deliberately fixed at 0. Experiment audit update
  numbers start at 1; restored Adam advances from 78,950 to 78,951 each time.
  This is a controlled update benchmark, not exact training continuation. The
  checkpoint and its bindings are never rewritten or overwritten.

The production source digest remains identical to the earlier strict compiler
qualification: `0965252f0b69a6d9f2a755d9ad2dafae2f2ac20e9503b58736f903414fe8af61`.
The measured experimental files also match their recorded SHA256s. The compiler
fix and its earlier bitwise qualification are described in
[gpu_efficiency.md](gpu_efficiency.md#independent-compiled-backend-2026-10-19).

## Complete training updates

All values below are **unprofiled medians of three updates**, in seconds.

| Scheme | Forward + statistics | Backward + group clipping | Complete update | Peak allocation |
| --- | ---: | ---: | ---: | ---: |
| Rotation compile + dynamic compaction | 1.523 | 1.865 | 3.614 | 4.196 GiB |
| Strict Compiled + dynamic compaction | 0.961 | 1.004 | **2.173** | **4.037 GiB** |
| Strict Compiled + fixed Batch + Mask | **0.944** | 1.060 | 2.206 | 4.259 GiB |

| Complete-update component | Rotation + dynamic | Compiled + dynamic | Compiled + Mask |
| --- | ---: | ---: | ---: |
| Actor/Adam/RNG backup | 0.00060 | 0.00059 | 0.00058 |
| Sampling, host interval | 0.19654 | 0.19371 | 0.19162 |
| Forward/statistics | 1.52260 | 0.96079 | 0.94446 |
| Backward/native group clipping | 1.86504 | 1.00404 | 1.05961 |
| Global clip, Adam, finite checks, parameter changes | 0.00151 | 0.00141 | 0.00146 |
| Actual audit capsule, including CPU copy and atomic fsync | 0.02076 | 0.01407 | 0.01473 |
| Remaining metrics, JSON history and fsync | 0.00856 | 0.00896 | 0.00875 |

Independent component medians need not sum to the median total. Forward and
backward boundaries synchronize CUDA. Sampling's host interval can include
asynchronously submitted work charged to the following forward boundary.
Complete time runs from before backup through the real history fsync. Trial
reset, compiler cold start and post-timing numerical inspection are excluded;
periodic EVAL and checkpoint saving are excluded because they are not performed
on every update. Each scheme wrote four real update-audit capsules and four
history rows to a separate experiment directory.

Complete-update samples:

| Scheme | Three samples (s) |
| --- | --- |
| Rotation + dynamic | 3.72939, 3.61352, 3.52983 |
| Compiled + dynamic | 2.03631, 2.18846, 2.17256 |
| Compiled + Mask | 2.20615, 2.23502, 2.10477 |

Mask's forward median improves 1.7%, backward worsens 5.5%, and complete median
is 1.5% slower than Compiled + dynamic. The ranges overlap and n=3 is small;
these measurements demonstrate neither stable net acceleration nor a statistically
established regression. Mask uses 0.222 GiB more peak allocated memory.

## Full H500 numerical and gradient comparison

The actual warmup update captures all 500 step inputs, valid step outputs, Actor
memory inputs/outputs, all returned trajectory tensors, every 50-step chunk end,
physical/history state, validity, costs, all 128 unaggregated native group rows,
group norms/multipliers, gradients entering Adam, and updated Actor/Adam tensors.
Stable scene IDs reconstruct frozen rows when comparing compacted and fixed batches.
Immutable tapes remain shared by identity. The other three updates in each scheme
reproduce that scheme's captured costs/gradients/Actor/Adam **bitwise**.

Original gates remain **state/cost rtol=1e-6, atol=1e-6**, and **derivative/update
rtol=1e-5, atol=1e-6**. Exact equality is reported separately. No gate was relaxed.

**Compiled + dynamic versus rotation + dynamic: every compared value is bitwise
equal**, including complete H500 trajectories, native group rows, clipping,
final gradients and the real persistent Adam update.

**Compiled + Mask versus Compiled + dynamic:** all forward states, delayed history,
memory, validity and costs are bitwise equal. All derivative/update gates pass,
but gradients and the resulting Adam update are **not bitwise equal**:

| Compared quantity | Maximum absolute difference | Relative L2 difference |
| --- | ---: | ---: |
| All forward fields and costs | 0 | 0 |
| Raw 128-group gradients | 5.51518e-9 | 5.08665e-8 |
| Group norms and shrink multipliers | 1.87489e-8 | 1.72251e-9 |
| Gradient entering Adam | 4.65661e-10 | 1.74217e-8 |
| Pre-global-clip norm | 1.10378e-10 | 2.18315e-9 |
| Actor after Adam | 2.98023e-8 | 5.48128e-10 |
| Adam first moment | 5.82077e-11 | 2.87595e-8 |
| Adam second moment | 1.42109e-14 | 1.71899e-11 |
| Adam step counters | 0 | 0 |

This retains the equations and optimizer operations, but the masked backward
graph and batch shapes differ from compaction. The results are consistent with
floating-point accumulation differences; no specific backward operator was
isolated as their sole cause. Mask is an experiment passing the existing
tolerances, **not a zero-error replacement**. The earlier strict Compiled
dynamic backend retains its bitwise result.

Deterministic tests additionally exercise initially terminal scenes, first/second
step failure, all-dead chunks, Time Decay on/off, immutable tape identity, all
physical/history/initial-memory gradients, and deliberately active group clipping.
Those tests cover cases the mature H500 fixture rarely reaches. They do not
establish mask equivalence for arbitrary nonfinite inactive candidate computations.

## Inside the 500 steps: two synchronization sources

Nsight Systems/CUPTI plus NVTX scopes cover the original per-step body; AST
instrumentation adds scopes without changing its expressions or order. A scoped
wrapper marks `nonzero` and host scalar reads. No profiler modifies production.

| Inside 500 step bodies | Dynamic compaction | Fixed Batch + Mask |
| --- | ---: | ---: |
| `nonzero` calls | 500 | **0** |
| Waits within `nonzero` | 500 | **0** |
| Other CUDA synchronization calls | 500 | **0** |
| Total step synchronization calls | **1,000** | **0** |
| CUDA synchronization API wall time, GC-instrumented capture | 9.584 ms | 0 ms |

There are another 10 chunk-entry `nonzero` calls in dynamic compaction, making
510 in the full forward pass. The corresponding inside-step `nonzero` CPU scope
totals 23.078 ms, including 5.022 ms of child synchronization waits. These nested
durations must not be added. A separate earlier dynamic capture measured
27.449 ms for the same 1,000 step waits: the wait duration is variable.

The second 500 waits come from **`active[indices] = True`**. Each assignment
stages a one-byte CPU bool to CUDA and synchronizes. Evidence includes:

1. All 500 waits outside the `nonzero` scopes in the first H500 timeline are
   immediately preceded by a one-byte Host-to-Device copy.
2. A separate H8 PyTorch operator trace attributes eight such waits through
   `aten::index_put_ -> _index_put_impl_ -> to -> _to_copy -> copy_`.

Mask keeps activity on the device and eliminates both causes. Chunk-boundary
tape/finite checks and statistics remain: the final captures contain 1,086 versus
86 forward synchronization APIs in total, and seven in each backward phase.
Removing 1,000 short waits alone does not remove the remaining host launch cost.

## GPU execution gaps, launch pressure and Python GC

The following are **instrumented timeline measurements**, not benchmark timings.
GPU activity is the interval union of this process's kernels, copies and memsets.
The complement means no traced work from this process; it is **neither SM
occupancy nor whole-device utilization**. Overlapping durations are never summed.

| GC-instrumented capture | Compiled + dynamic | Compiled + Mask |
| --- | ---: | ---: |
| Forward wall / GPU activity union | 1064.49 / 176.58 ms | 1165.21 / 166.91 ms |
| Forward no-traced-work fraction | 83.41% | 85.68% |
| Backward wall / GPU activity union | 1400.78 / 433.78 ms | 1489.02 / 452.64 ms |
| Backward no-traced-work fraction | 69.03% | 69.60% |
| Forward kernel launch API count | 88,199 | 90,290 |
| Backward kernel launch API count | 156,423 | **181,767** |
| Forward host-late submission, conservative lower bound | 550.08 ms | 680.32 ms |
| Backward host-late submission, conservative lower bound | 500.98 ms | 513.11 ms |

Host-late time is the portion of each execution gap before the next GPU
activity's correlated CUDA API even starts. It identifies host-side submission
delay without pretending every idle interval is caused by an explicit wait.
In the ten GPU windows belonging to the step bodies, dynamic has 166.66 ms of
GPU activity in 1016.43 ms, and Mask 157.96 ms in 1128.44 ms. GPU windows include
queued work past CPU enqueue scopes; clipping to CPU scopes would miscount it.
Mask's backward launches increase **16.2%**. Launch pressure and masked backward
work remain substantial even after eliminating every inner-loop synchronization.

Python cyclic GC is another observed source of long host pauses:

- A frozen untraced dynamic run contains a **159.76 ms generation-2 collection
  during forward**; a second annotated untraced run contains 169.66 ms.
- The final Mask CUDA trace directly encloses a **131.69 ms generation-2 GC
  range inside a step**, with a 131.95 ms GPU execution gap. All GC generations
  together overlap **147.99 ms of no-GPU-work intervals** in that forward pass.
  The dynamic capture happens to contain no generation-2 collection and only
  16.02 ms of such overlap. GC cadence varies between runs.
- The first dynamic timeline had a 180.71 ms gap with approximately 180.45 ms
  before any next CUDA API entry. It had no GC markers, so that particular gap
  is **not proven to be GC**. Later GC-marked captures supply the direct evidence.

GC remains enabled, its thresholds are unchanged, and no collection is moved.
Lightweight GC callbacks themselves can affect collection cadence. Nsight also
inflates wall time: the final dynamic frozen run is 2.008 s bare versus 2.484 s
captured (+23.7%), and Mask 1.856 versus 2.673 s (+44.0%, including different GC
cadence). These single runs and timeline idle fractions cannot establish an
unprofiled utilization or speedup claim. The complete-update table is the timing
result; the traces explain where scheduling work occurs.

The measured next targets are fewer host submissions/kernel launches and
generation-2 GC placement. A narrowly scoped device scalar assignment replacing
the bool `index_put_` is also worth testing in isolation. Neither that change,
GC rescheduling, CUDA Graph capture nor additional compiler fusion was applied
to production or claimed as a measured improvement here.

## Implementation, reproduction and evidence

New code is confined to independent experiment modules/tools/tests:

- `diffphys/experimental/masked_rollout.py`: fixed batch, frozen failed rows,
  parallel closed/live Time Decay edges, unchanged 50-step collector and probe.
- `diffphys/experimental/training_benchmark.py`: successful update measurement
  using original sampling, backward, clipping, Adam, audit and history operations.
- `tools/benchmark_rollout_scheduling.py`: bounded restored-checkpoint updates,
  actual H500 captures and comparisons; it refuses a nonempty output directory.
- `tools/profile_rollout_scheduling.py`: frozen profiling and SQLite interval
  analysis. It does not call Adam or change weights.

The exact benchmark invocation was:

```bash
python tools/benchmark_rollout_scheduling.py --device cuda --scenes 2048 \
  --horizon 500 --repeats 3 --measure-rejected \
  --output reports/rollout_scheduling_20261019/update-benchmark
```

`--measure-rejected` permits measuring a rejected experiment, with explicit
qualification flags; all three actual schemes passed the unchanged tolerance
gates. It does not turn Mask's non-bitwise result into exact equivalence.
Do not rerun into this directory. More training updates require a new budget.

For frozen capture (use a fresh output prefix):

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=false --output=/tmp/mask-h500 \
  python tools/profile_rollout_scheduling.py --schedule mask \
    --scenes 2048 --horizon 500 --report /tmp/mask-wall.json
nsys export --type sqlite --output=/tmp/mask-h500.sqlite /tmp/mask-h500.nsys-rep
python tools/profile_rollout_scheduling.py --analyze /tmp/mask-h500.sqlite \
  --report /tmp/mask-analysis.json
```

Use `--schedule dynamic` for the other schedule. Actual Python was
`/tmp/diffphys-cuda-venv/bin/python`, and Nsight Systems 2024.6.2 was installed at
`/usr/local/cuda-12.8/bin/nsys`; these are local runtime paths, not config changes.

Local evidence under `reports/rollout_scheduling_20261019/` is ignored by Git and
preserved: `update-benchmark/report.json` contains all per-field H500 error
metrics and trials; `warmup_update_values.pt` preserves raw group gradients and
update tensors; per-scheme audit capsules contain restored and updated Adam plus
source archives; `nsys-{dynamic,mask}-gc-h500.{nsys-rep,sqlite}` preserve complete
timelines; `*-gc-analysis-final.json` and `active-mask-sync-attribution.json`
contain interval analysis and operator attribution. Earlier captures and red/green
test logs are retained. `evidence_manifest.json` and `source_snapshot.zip` bind
the final experimental sources, runtime and evidence files.

Fresh validation: **498 passed, 22 skipped** in the complete CPU suite;
**11 passed** in the focused CUDA experiment suite. No long training, new EVAL,
production backend switch or deployment validation was performed.
