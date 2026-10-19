> Historical implementation plan; see [archive notes](../README.md).

# H500 scheduling and complete update benchmark

**Goal:** Measure full updates using the independent compiled physics backend,
attribute H500 host/device synchronization and GPU execution gaps, and compare
dynamic compaction with a fixed batch and masks without changing production.

**Budget:** User approved one warmup plus three timed updates per scheme. Each
trial restores the same best.pt Actor and named Adam state. Frozen numerical
and profiler diagnostics are separate from the timed training updates.

**Architecture:** Keep production modules byte-identical. Add an experimental
masked rollout and collector using the original observation, Actor, physics,
Time Decay, loss, 50-step reductions, native group probe and clipping. Freeze
first-failure rows with device-side masks; preserve original immutable tapes
and original scene/group IDs. Compare against actual production compaction.

- [x] Profile the existing compiled/dynamic H500 before implementation; extract
  nonzero counts, runtime synchronization waits, kernel activity and execution
  gaps, separating the inner chunks from chunk-boundary metrics and backward.
- [x] Write failing deterministic tests for masked first-failure semantics,
  frozen histories/step indices/tapes, Time Decay and per-group derivatives.
- [x] Implement only the independent fixed-batch rollout/collector and scoped
  measurement helpers. Never compile the Actor or change the RK4 backend.
- [x] Compare all H500 physical/policy state fields, valid masks, costs, native
  group rows, group clipping, final gradients and shadow persistent Adam on the
  identical mature-policy fixture. Report exact and original tolerance gates.
- [x] Measure each scheme's complete successful update: snapshot, sampling,
  forward/statistics, backward/group clipping, global clipping, Adam, finite
  checks/parameter changes, real audit capsule and history I/O. Report periodic
  checkpoint/EVAL separately; neither occurs on every update.
- [x] Collect comparable CUDA timelines for both schedules; analyze overlapping
  intervals using unions, do not sum overlapping kernel/runtime durations or
  confuse kernel-active time with SM occupancy or NVML utilization.
- [x] Run focused CUDA checks and deterministic CPU regressions, review the
  diff, preserve source/runtime/artifact hashes, and document results/limits.

**Numerical gates:** Retain FP32 state/cost rtol=1e-6 atol=1e-6 and derivative
rtol=1e-5 atol=1e-6. Also report exact equality. Never hide a failed mask result
behind averaged gradients or report unqualified throughput as a valid speedup.
No checkpoint hash rewrite, production/default switch or long training run.

**Measured outcome:** Complete-update medians rotation/dynamic 3.614 s,
compiled/dynamic 2.173 s, compiled/mask 2.206 s. All 12 authorized updates are
complete. Compiled/dynamic is bitwise equal to rotation/dynamic; mask forward is
bitwise equal and gradients pass the unchanged gates, but backward/Adam are not
bitwise equal. Mask removes 1,000 inner-step synchronization calls but adds 16.2%
backward launch APIs and has no demonstrated stable net speedup. GC-marked CUDA
timelines also identify a 131.69 ms generation-2 pause inside a step. Production
source digest is unchanged. CPU suite: 498 passed/22 skipped; focused CUDA:
11 passed. See [full report](../rollout_scheduling.md), with raw evidence under
`reports/rollout_scheduling_20261019/`.
