# Compile the AFT training path and prefetch CPU scenes

**Goal:** Compile supported large tensor blocks, including their AOTAutograd backward graphs, and prepare exactly one next immutable TRAIN bank on a persistent CPU worker.

**Scope:** Single agent. Native physics equations, noise, Actor parameter layout, minimum-speed objective, fixed physical groups, clipping and Adam semantics are retained. No stale-weight CPU rollout. Source/config changes require a fresh run or weights-only import; historical checkpoints and source archives remain intact.

## Work

- [x] Extend the existing read-only profiler to the actual GRU69 task and measure synchronized native forward/backward before production edits.
- [x] Add failing numerical/integration tests for full-step compilation, Actor hooks, tensor loss compilation, group partials and CPU prefetch identity/resume/cleanup.
- [x] Reuse the strict full-step compiler and bounded spawn sampler from the sibling project, adapting only flat imports and the AFT entry. CPU remains the numerical reference and compiler errors propagate.
- [x] Compile pure Actor/goal/loss/group-gradient tensor regions that preserve native module hooks and validated arithmetic. Keep sampling, event bookkeeping, validation and I/O outside captured graphs. Record supported and rejected compiler boundaries with evidence.
- [x] Integrate source/backend checkpoint bindings, replay/evaluation and worker lifecycle; retain transactional failure rollback and deterministic committed-update seeds.
- [x] Verify CPU regressions, bounded CUDA forward/gradient/Adam comparisons, H500 memory/throughput at the requested scene count, and sampler CUDA isolation. Select production defaults from measured results and document exact limitations.

## Verification constraints

- One GPU experiment at a time, compiler threads limited to one on this 8 GB device.
- Same weights, physical scenes, immutable sensor tapes and task settings for paired measurements.
- Warm-up/compilation time reported separately from steady-state timings.
- Compilation is not a convergence or flight-performance result.

## Result

Verified final compiler boundary v2, CPU 608 passed / 25 skipped, CUDA targeted 8 passed, 3072 × H500 three-update training smoke, and bitwise full CUDA update replay. Worker joined; training stopped at its three-update budget. Full evidence: `reports/compiled_prefetch_20261023_192850/verification.json`.
