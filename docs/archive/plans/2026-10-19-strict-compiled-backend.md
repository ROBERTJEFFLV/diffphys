> Historical implementation plan; see [archive notes](../README.md).

# Numerically faithful compiled physics

**Goal:** Package the verified native arithmetic boundaries into an isolated
compiled backend, then measure whether its unchanged H500 forward/backward is
faster than eager and the existing rotation-only compiler.

**Architecture:** Compile the original bound simulator step with fullgraph
Inductor and original AOTAutograd. Preserve native sum/cross/norm/bmm boundaries
only inside this graph; do not mutate global lowering or decomposition tables.
Give the compiler pass a source-derived cache identity. Pin the tested PyTorch
2.10 stack and propagate unsupported-runtime/compilation errors.

**Execution:** Single agent, no delegation or training. Fixed best.pt, original
loss, noise tapes, Time Decay, fixed group clipping and checkpoint Adam moments.

- [x] Write failing tests for isolated exact arithmetic, state/action VJPs,
  pulse stages, history/compaction and lack of global compiler side effects.
- [x] Implement the smallest scoped Inductor backend and expose it through the
  experimental simulator and numerical diagnostic CLI.
- [x] Verify CUDA single-step/VJPs and 2048-scene H500 trajectories, raw groups,
  clipping, and persistent shadow Adam before timing. Require bitwise trajectory
  and rollout results for the requested zero-drift claim.
- [x] Measure synchronized warm frozen forward/backward with alternating order,
  at least five paired samples, equal valid transitions and peak memory. Compare
  with both eager and the existing production rotation-only backend.
- [x] If preserving every boundary is slow, profile the bounded workload before
  narrowing any boundary; revalidate numerical/gradient behavior after changes.
- [x] Review the diff, run deterministic regressions, and document exact scope,
  cold compile cost, warm timings and limitations. No production-default switch
  before the measured backend passes the same numerical gates.

## Verified result

The scoped backend passes bitwise H500 state/VJP/group/clipping/Adam gates on
PyTorch 2.10.0+cu126, RTX 4060 Ti, best.pt, 2048 scenes/128 groups. The clean
six-repeat medians are eager 3.935 s, rotation-only 3.496 s, compiled 2.062 s.
No boundary narrowing was necessary: the conservative set already accelerates
the workload, so further optimization/profiling was not pursued. Total cold
compile cost is not isolated; first-use/specialization overhead is explicitly
excluded from the warm claim. CPU regressions: 489 passed, 20 skipped. CUDA
compiled-backend checks: 10 passed. No training or production-default switch.
