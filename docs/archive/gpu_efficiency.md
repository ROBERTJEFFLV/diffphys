> Archived pre-promotion experiment. Runtime commands require the source in
> [optimization_sources_20261019.zip](optimization_sources_20261019.zip).
> See [current production execution](../gpu_efficiency.md) for the adopted P path.

# CUDA efficiency audit, 2026-10-05

The subsequent [H500 scheduling and complete-update benchmark](rollout_scheduling.md)
reports 2.173 s complete updates for the independent strict Compiled backend,
versus 3.614 s for rotation-only compilation. It traces 1,000 inner-step waits,
kernel submission gaps and GC, and qualifies an independent fixed Batch + Mask
experiment without changing production.

The Actor remains native B: GRUCell(16,64), hidden-only Linear(64,4), tanh.
These changes accelerate the existing H500 computation; they do not change
the controller, loss, physical sampler, group clipping or Adam.

## Applied changes

- Cache the read-only gravity and rotor-spin vectors by device/dtype. Values
  and RK4 arithmetic order are unchanged.
- Omit unused observation storage during training. The noisy Actor observation
  is still calculated at its original Time Decay location. EVAL and replay
  record complete observations by default.
- Convert local probe contributions directly into the existing FP64 accumulators
  with `add_`, avoiding separate FP64 temporaries. Local float32 matmuls/sums
  remain float32; this does not cure overflow.
- Compile only the original quaternion-to-rotation expression inside physics
  RK4. Actor/GRU, sensing, other physics, compaction and probe remain eager.

The main config explicitly uses `--rotation-backend compile`. Bare CLI and
simulator defaults remain `eager`. There is no automatic compiler fallback.
The backend/version/options are checkpoint-bound and reproduced in audit replay.
Precision-cast and division-rounding emulation are enabled, CUDA Graphs disabled.
This was verified with PyTorch 2.10.0+cu126, not arbitrary compiler releases.

## Measured on RTX 4060 Ti, 8 GiB

Float32, mature B weights, 2048 TRAIN scenes, 128 groups, H500, Time Decay=1.
The frozen batch executes 936,700 valid transitions. First samples are excluded
from warm medians; these are not cheap random-Actor early-failure timings.

| Comparison | Before | After | Scope |
| --- | ---: | ---: | --- |
| Actual update | 5.35 s | 4.29 s | Sampling through Adam/audit, 3 paired updates |
| Eager cleanup | 5.13 s | 4.75 s | Frozen forward/backward, 5 paired samples |
| Rotation compile vs optimized eager | 4.64 s | 4.03 s | Frozen forward/backward, 5 paired samples |
| NVML busy-time mean | 33.9% | 28.2% | Rotation comparison; not SM occupancy |

Actual update latency fell about 20%, approximately 25% more updates per unit
time. NVML busy-time did **not** rise. It measures sampled time with at least
one executing kernel, not the fraction of compute capacity used. Fusion can
complete work faster with less GPU active time. See [NVIDIA's definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
and [PyTorch's profiling guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html).

The 3 paired updates produced bitwise-identical Actor, Adam and RNG states.
A separate 256-scene H500 EVAL had identical complete trajectories and costs.
Full and Adam-only update replay passed. The full regression run passed 414
tests, including actual CPU/CUDA checks. This is bounded numerical/performance
evidence, not proof of better control or prevention of gradient explosions.

Whole-RK4 compilation failed H500 cost/gradient tolerances and is not integrated.
Probe-gate compilation gave no additional benefit over rotation fusion and is
not integrated. A 3072-scene eager batch improved valid physical transitions/s
by about 44%, with 5.97 GiB allocation, but changes batch statistics; the default
remains 2048. 4096 scenes were not tested.

## Reproduce without updating weights

```bash
python tools/profile_response_training.py --device cuda --scenes 2048 \
  --horizon 500 --rotation-backend eager --warmup 1 --repeats 5 \
  --checkpoint /path/to/native_B.pt \
  --epsilon-p .01 --epsilon-a .01 --lambda-R .2 --report /tmp/eager.json

python tools/profile_response_training.py --device cuda --scenes 2048 \
  --horizon 500 --rotation-backend compile --warmup 1 --repeats 5 \
  --checkpoint /path/to/native_B.pt \
  --epsilon-p .01 --epsilon-a .01 --lambda-R .2 --report /tmp/compile.json

python -m pytest -q tests
```

Profiler `--scenes` means total scenes, while training `--scenarios` means per
bank. The profiler never calls Adam or writes a model; sampling and warm-up
are separate from compute timings. Cold compiler caches increase initial cost.
CPU compilation additionally needs a C++ toolchain and `setuptools`.

Exact resume still requires identical source/config. Do not rewrite old run
hashes: preserve original source/checkpoints and use a new directory with
explicit weights-only `--init-checkpoint` when initializing from old weights.
That creates fresh Adam/sampling, not exact continuation. Eager and compile
have different backend bindings. Local raw results and the performance-only
patch are under `reports/gpu_efficiency_20261005/`, ignored by Git.

## RK4 investigation, 2026-10-19

The Oct 5 raw GPU report/fixture is not available in this checkout. The local
`best.pt` is a native hidden-only B Actor at update 78,950, with matching model
digest and the current `.01/.01/25` loss. Its file SHA256 is
`5ddbef550e1bd5a15701caa32d0244e1512b38b6a212b13cc5c3665cd2764851`.
The following is a current-source reproduction, not an exact replay of the
historical CUDA failure or a new training run.

### Experimental boundary and gates

`diffphys/experimental/compiled_physics.py` compiles the **original bound**
`RaptorSimulator.step` with `backend="inductor", fullgraph=True, dynamic=True`.
There is no second RK4 implementation. The captured boundary includes the
existing pulse lookup, four stages, clamps, normalization and state/history
replacement. Actor, measurements, Python H500 loop, survivor compaction, true
task loss, Time Decay, native group probe, clipping and Adam remain outside it.
The production CLI, configs, kernel contract and checkpoint rules are unchanged.
Compilation errors propagate; there is no eager retry.

The diagnostic ladder uses `backend="eager"` for graph capture,
`backend="aot_eager"` for capture plus AOTAutograd, and `backend="inductor"`
for generated kernels. These distinctions and the single-graph requirement
follow the [PyTorch compiler documentation](https://docs.pytorch.org/docs/2.10/generated/torch.compile.html)
and [compiler troubleshooting guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html).

`tools/diagnose_rk4_compile.py` checks a real pulse transition's state/action
VJPs, every dynamic trajectory field, delayed histories, per-scene costs, raw
native-probe rows, group norms/multipliers, final gradients and one disposable
shadow Adam update. With a checkpoint, Adam uses its named moments and original
hyperparameters. Without one it uses fresh Adam. It never writes a checkpoint.
Initial state/tape hashes, source hashes, runtime/options, checkpoint digest,
loss, seeds, group settings and fixed tolerances are recorded in JSON.

FP32 state/cost gates use the existing short-rotation limits `rtol=1e-6,
atol=1e-6`; derivative gates use `rtol=1e-5, atol=1e-6`. FP64 uses the historical
whole-step limits `rtol=3e-9, atol=2e-10`. These limits have not been relaxed.
The missing Oct 5 H500 raw thresholds are not reconstructed or claimed. A
failure writes `passed: false`, exits 1 and skips optional timing. Timing, when
eligible, excludes one warmup and sampling, synchronizes CUDA, and alternates
the order of identical frozen forward/backward workloads.

### CUDA 2.10 reproduction and causal attribution

The supported **PyTorch 2.10.0+cu126** stack was installed separately from the
pre-existing 2.11 environment. RTX 4060 Ti, FP32, `best.pt`, seed 7, 2048
coverage128 scenes, 128 groups, H500, `.01/.01/25`, and the checkpoint's named
Adam moments execute **1,018,203 identical valid transitions** in each case.
The initial state/tape digest is
`f24f69dc63db49974b95cc97eaf37bc54183e27a8931c2516c0c90fa6ec0673d`.
The historical fixture executed 936,700 transitions and used a different loss
configuration; this is a new, fully specified failure, not its exact replay.

Graph capture plus `aot_eager` matches every trajectory, cost, raw group row,
clipping result and shadow Adam tensor **bitwise**. Default whole-step Inductor
first differs at step 1; omega exceeds the fixed state tolerance at step 3.
Raw group gradients fail: maximum absolute difference `4.5387285e-6`, maximum
scaled error `4.0121`, relative L2 `3.4113e-5`. Group norms/multipliers also fail.
Costs, the combined clipped gradient and persistent shadow Adam still pass;
their averaging must not hide the failed per-group check. The largest H500
omega difference is `2.5510788e-4`; orientation is `3.3259392e-5` and known
commands `1.6558170e-4`. These numbers describe numerical drift, not a measured
training failure or deployment outcome.

`reports/rk4_compile_20261019/causal_probe.py` performs two **diagnostic-only**
value/VJP interventions. It asserts that every forward trajectory tensor is
bitwise identical to the intended eager or compiled trajectory before interpreting
the gradients. Neither intervention is a valid training backend or a proposed
gradient change.

| Forward values | Step VJP implementation | Raw group gradient error L2 vs ordinary eager | Fraction of default Inductor error |
| --- | --- | ---: | ---: |
| Eager | Eager | 0 | 0 |
| Compiled | Compiled | `2.68853e-5` | 100% |
| Eager | Compiled | `1.84971e-7` | **0.688%** |
| Compiled | Eager | `2.69036e-5` | **100.068%** |

The last intervention's error has cosine `0.9999736` with the original error.
Thus the dominant error in this fixture follows **the forward values**, rather
than a large independent AOTAutograd backward error. This is a counterfactual
attribution, not an additive percentage decomposition: nonlinear interactions
prevent adding these fractions. A 500-step same-input scan also rules out the
suspected motor rising/falling branch Jacobian jump in this fixture: no
`dm_next/dm` differences above `1e-3`, maximum difference `1.1920929e-7`.

### First RHS sources on CUDA

Same-input ablations in `reports/rk4_compile_20261019/rhs_probe.py` identify
two concrete changes in arithmetic implementation **before integration**:

1. Compiled thrust polynomial contraction changes rounding: maximum FP32
   difference `9.5367432e-7`. Setting only `emulate_precision_casts=True`
   makes this isolated polynomial bitwise equal to eager.
2. The four-term reduction order changes. For the actual first-stage roll
   torque terms `x = rotor_positions[..., 1] * thrust`, independent FP32
   parenthesization oracles match every result:

   - Native CUDA sum: `(x0 + x2) + (x1 + x3)`.
   - Compiled sum: `((x0 + x1) + x2) + x3`.

   **1016 of 2048** sums differ, with maximum absolute difference
   `2.3841858e-7`. Precision-cast controls, exact division controls, or their
   combination leave this isolated reduction difference unchanged. Consequently
   disabling FMA and requesting precise division cannot restore the eager
   operation order by themselves.

For example, scene 918 has roll-torque terms
`[-0.0186368003, -0.0169282351, 0.00247121230, 0.0134493476]`.
The native-tree FP32 oracle yields `-0.01964447647`; the serial oracle yields
`-0.01964447275`. Dividing this `3.72529e-9` torque difference by the scene's
`Jx = 8.4432708e-5` already gives an angular-acceleration difference of
`4.41214e-5`. This scalar calculation isolates reduction order; it is not a
claim that every full-kernel error comes from this one term. The full isolated
thrust/torque/angular-acceleration chain differs by up to `6.1035156e-5` in
angular acceleration; returning all original RK4 stage values observes errors
already in k1, up to `9.1552734e-5` in angular acceleration. Extra outputs can
change fusion, so the uninstrumented original step remains the acceptance gate.

These local differences enter all four RK4 evaluations and recur throughout
the closed loop, changing the states, Actor commands and inputs at which
subsequent derivatives are evaluated. In the observed grouped gradient error,
the highest torque-to-inertia quartile contributes **80.16% of squared error**;
groups 114 and 102 together contribute 68.85%. This distribution supports the
high angular-acceleration sensitivity of the fixture, without proving that one
operator alone dominates every trajectory. Replacing only selected step output
fields with their native values does not monotonically reduce the H500 error;
those diagnostic interventions are retained in `field-interventions-cuda210.json`.
They rule out treating a single-output correction as a validated remedy.

An explicit diagnostic Inductor lowering that retains native `aten.sum`
still fails H500. Adding the precision/division controls reduces the maximum
raw gradient difference to `1.9841638e-6`, but still fails both trajectory and
group checks. Other changed boundaries therefore matter too.

The decisive restoration keeps native `aten.sum.dim_IntList`,
`aten.linalg_cross.default`, `aten.linalg_vector_norm.default` and
`aten.bmm.default` boundaries together with the precision/division controls.
The **unchanged original bound step** still runs through fullgraph Inductor,
with these operations emitted as explicit native external calls. On the same
2048-scene H500 fixture, **every dynamic trajectory field, per-scene cost, raw
group gradient, group norm/multiplier, final gradient, Actor after the shadow
update and Adam moment/step becomes bitwise identical** to eager. This is a
positive causal restoration, not merely failure of several compiler settings.
The boundary set is a demonstrated sufficient combination; separate necessity
of each member and the smallest successful boundary set remain unproven.

These interventions use private compiler internals only in the ignored report
directory; they are neither an automatic fallback nor production code. They
have no performance qualification or generalization claim. No loss, Time Decay,
clipping, optimizer or tolerance was changed to obtain these results. Default
whole-RK4 Inductor remains rejected; any supported acceleration must preserve
the established numerical boundaries and pass the original gates.

Primary evidence: `cuda210-{aot,inductor}-mature-n2048-h500.json`,
`causal-hybrids-cuda210.json`, `local-motor-jacobians-cuda210.json`,
`rhs_probe.json`, `reduction-oracle-example.json`, and
`native-sum-{default,division-casts}-cuda210.json`, and
`native-all-division-casts-cuda210.json` under
`reports/rk4_compile_20261019/`.

The full CPU regression run passed 485 tests (19 skipped), and a fresh focused
run of `tests/test_compiled_physics.py`, `tests/test_core_contract.py` and
`tests/test_package_layout.py` passed 8 tests. CUDA evidence above comes from
bounded physical/gradient diagnostics, not a full CUDA pytest run or training.

### Separate CPU numerical mechanism

For the 128-scene mature-Actor H500 CPU fixture, 63,535 valid transitions execute.
PyTorch **2.10.0+cpu** with `aot_eager` matches eager trajectories, costs and raw
group derivatives bitwise. Default Inductor first differs in orientation at
step 1 (maximum `1.1920929e-7`); action history first exceeds the fixed trajectory
gate at step 8. The combined group-norm/multiplier report also fails its gate. This
separates the drift from graph capture, Time Decay and native group bookkeeping.

Additional observed-stage diagnostics localize a concrete source to **cross
product rounding in the very first RHS evaluation**, before quaternion
normalization. On the same FP32 inputs:

| Isolated CPU operation | Default Inductor vs eager |
| --- | --- |
| `(q_vector * omega).sum(...)` | Bitwise equal |
| `torch.linalg.cross(q_vector, omega)` | 104 of 384 elements differ; max `1.1920929e-7` |
| Quaternion derivative scalar component | Bitwise equal |
| Quaternion derivative vector component | Max `1.1920929e-7` difference |

An independent FP64 calculation rounded to FP32 matches native cross exactly
when it emulates one fused multiply/subtract. The separately rounded multiply
then subtract differs. Switching **only the isolated compiled cross** from
`cpp.enable_floating_point_contract_flag="off"` to `"fast"` changes it from
matching the unfused reference to matching eager bitwise. This is a causal
rounding check, not a proposed global RK4 fix: enabling contraction everywhere
can alter other eager operation boundaries. Native CPU/CUDA cross kernels can
be inspected in the [PyTorch CPU source](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/CrossKernel.cpp)
and [CUDA source](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/CrossKernel.cu).

Isolated normalization on eager's pre-normalization quaternion is bitwise equal
on CPU. Observing all four RHS stages and the unnormalized quaternion still
shows drift. Extra graph outputs can alter fusion, so these observations
localize the mechanism; the uninstrumented original step remains the acceptance
gate. They do not establish cross-product contraction as the only source of
all CUDA differences.

Precision-cast/division-rounding controls were varied separately in small
probes. Their combination does not rescue the CPU H500 gate. In PyTorch 2.10,
the relevant names are `emulate_precision_casts` and the misspelled
`emulate_divison_rounding`; 2.11 renames the latter to
`eager_numerics.division_rounding`. The tool records the actual selected key;
it does not silently ignore an unsupported option.

### Supplemental CUDA 12.8 / PyTorch 2.11 checks

The available pre-existing GPU environment is **2.11.0+cu128**, which differs
from the historical 2.10 CUDA stack. On the RTX 4060 Ti, the same 128-scene
mature-Actor fixture gives:

| Single-step backend inside Python H500 | Numerical gate | First trajectory failure |
| --- | --- | --- |
| `aot_eager` | Pass; trajectories bitwise equal | None |
| Default Inductor | Fail | Omega, step 4 |
| Inductor, division + precision controls | Fail; raw group gradients also fail | Omega, step 5 |

Only the passing control enters timing. With one warmup excluded and three
alternating paired repeats, eager's median is 4.006 s and `aot_eager` is
3.973 s (1.008x). Individual eager samples span 3.840-4.077 s; this does not
establish a meaningful acceleration. Both execute 63,535 valid transitions.
These are synchronized frozen forward/backward timings with no simulation
recording, Adam update, sampling or checkpoint I/O in the timed region.
The rejected Inductor implementations have no qualified speedup result.
These supplemental measurements must not be labeled PyTorch 2.10 evidence.

### Reproduce

```bash
python tools/diagnose_rk4_compile.py --device cpu --backend aot_eager \
  --checkpoint best.pt --scenes 128 --horizon 500 \
  --epsilon-p .01 --epsilon-a .01 --lambda-R 25 \
  --report reports/rk4_compile_20261019/cpu-aot.json

python tools/diagnose_rk4_compile.py --device cuda --backend inductor \
  --checkpoint best.pt --scenes 128 --horizon 500 --timing-repeats 3 \
  --epsilon-p .01 --epsilon-a .01 --lambda-R 25 \
  --report reports/rk4_compile_20261019/cuda-inductor.json
```

Replace `--backend inductor` with `--backend aot_eager` to isolate code
generation, or with `--backend rotation` to gate/time the existing production
rotation-only compiler on its supported PyTorch 2.10 stack. Add
`--rounding division`, `casts`, then `division-casts` for
controlled rounding comparisons. Use a separate output file per case.
`--scenes` is the total, so this small fixture has one scene per coverage cell;
it is not the 2048-scene production throughput workload.

The current experiment remains rejected for production and whole-rollout
compilation. A smaller/faster trace is not a speedup if numerical gates fail.
No training, checkpoint-hash rewrite, loss change or tolerance relaxation is
part of this investigation. Raw local evidence is retained under
`reports/rk4_compile_20261019/` (ignored by Git).

## Independent compiled backend, 2026-10-19

`CompiledRaptorSimulator()` now selects `backend="compiled"`. The original
bound `RaptorSimulator.step` is still the only RK4 implementation. Fullgraph
Inductor fuses its pointwise work while keeping native sum, cross product,
vector norm and batched-matrix-multiply boundaries. AOTAutograd differentiates
the original ATen graph before the backend selects these boundaries in both
the forward and backward graphs. There is no hand-written backward or gradient
intervention in this backend.

`diffphys/experimental/strict_inductor.py` implements the isolated compiler
pass. It copies the decomposition table and rewrites only nodes in the graphs
being compiled. Unlike the earlier diagnostic intervention, it never replaces
global lowering/decomposition entries or adds operators to the global fallback
set. Its native external calls are explicit arithmetic boundaries; compiler
errors still propagate. The pass's cache identity hashes its source file, so
its graphs cannot reuse the unconstrained compiler's cache entries. PyTorch
versions outside 2.10 raise an error; the measured stack is 2.10.0+cu126.

The independent simulator API is:

```python
from diffphys.experimental.compiled_physics import CompiledRaptorSimulator

simulator = CompiledRaptorSimulator()  # Original RaptorParams(dt=.01).
next_state = simulator.step(state, action)
```

The production training CLI, backend defaults and checkpoint bindings have
not been switched. The other explicit experimental selectors (`inductor`,
`eager`, `aot_eager`) remain available for the diagnostic ladder. Numerical
options for `compiled` are fixed and cannot be overridden by callers.

For qualification, `tools/diagnose_rk4_compile.py --backend compiled` automatically
requires **bitwise equality**, including the single-step state/action VJPs,
every dynamic trajectory tensor, raw per-group gradients, group clipping,
final gradients and persistent shadow Adam. The existing tolerance gates and
finite checks still run. A mismatch exits 1 and skips timing. Optional
`--compare-rotation` subjects the existing production rotation-only simulator
to the same rollout gates and includes it in the timings. One warmup is
excluded; six repeats balance all three workload positions and alternate their
order. CUDA is synchronized at forward/backward boundaries. Sampling,
trajectory recording, checkpoint I/O and the disposable shadow Adam update
are excluded from the measured forward/backward region.

Reproduce on the supported CUDA stack:

```bash
python tools/diagnose_rk4_compile.py --device cuda --backend compiled \
  --checkpoint best.pt --scenes 2048 --horizon 500 --compare-rotation \
  --timing-repeats 6 --epsilon-p .01 --epsilon-a .01 --lambda-R 25 \
  --report reports/strict_compiled_20261019/compiled-n2048-h500-clean.json
```

This is a frozen forward/backward benchmark with numerical shadow-update
checks, not a training run or end-to-end training-update timing. First use also
incurs compilation; warm timings exclude that cost. Runtime, source hashes,
compiler contract, checkpoint digest, fixed loss, group settings, identical
state/tape digest and every individual timing are recorded in the JSON report.

### Qualified CUDA measurements

RTX 4060 Ti 8 GiB, PyTorch **2.10.0+cu126**, FP32, TF32 disabled,
`best.pt` at update 78,950, `.01/.01/25`, coverage128 with 2048 scenes and
128 physical groups, Time Decay=1. All backends execute **1,018,203 valid
transitions** over H500. The independent backend matches ordinary eager
single-step states/VJPs, every trajectory tensor, per-scene costs, raw group
gradients, clipping and persistent shadow Adam **bitwise**. Original checkpoint
moments are loaded and the disposable update reaches step 78,951. The
rotation-only reference also matches all trajectory/cost/group/clipping/shadow
Adam tensors bitwise in the same run.

After numerical qualification and one excluded warmup, six synchronized
repeats with balanced workload positions give:

| Backend | Median forward + backward | Observed range | Peak allocated memory |
| --- | ---: | ---: | ---: |
| Ordinary eager | 3.935 s | 3.874–4.088 s | 4.194 GiB |
| Existing rotation-only compiler | 3.496 s | 3.245–3.580 s | 4.194 GiB |
| Independent `compiled` | **2.062 s** | 1.873–2.140 s | **4.036 GiB** |

The independent backend is **1.909x** faster than ordinary eager (47.6% less
latency) and **1.696x** faster than the existing rotation-only backend (41.0%
less latency) on this frozen workload. These are ratios of measured warm
medians, not projections for complete training updates. Compilation is excluded:
the numerical recording pass takes 16.912 s for the candidate including
remaining specialization/compilation and 4.306 s for eager, after an earlier
single-step compile. That is not a measurement of total cold compile cost.

The primary report is
`reports/strict_compiled_20261019/compiled-n2048-h500-clean.json`; no
task-owned CPU tests or other GPU compute jobs ran concurrently with its
measurements. A preliminary run that overlapped CPU regression tests is retained
as `compiled-n2048-h500.json`; its medians were 3.928/3.272/2.177 s for
eager/rotation/compiled. It is not used for the table above.

Fresh checks: **489 CPU tests pass, 20 skip**; **10 compiled-backend tests pass**
in the CUDA environment, including actual CUDA step/VJP/compaction checks.
The gate regression confirms that a tiny, tolerance-acceptable unconstrained
compiler drift is rejected and never timed when exact equality is required.
Independent compiler test cases reset Dynamo's code cache; this avoids exhausting
its eight-variant limit by mixing devices, dtypes and diagnostic backends in one
pytest process. The backend itself does not reset caches or raise that limit.

The production default remains unchanged. This provides an explicit independent
backend with qualified numerical and warm-performance evidence on the specified
stack/fixture; longer training, learned performance and other compiler releases
have not been qualified here.
