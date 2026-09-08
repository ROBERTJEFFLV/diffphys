# PETSc KKT backend

Response full-space MS defaults to `--linear-solver petsc-minres`. Physics,
policy, task residuals, autograd, trust radii, backtracking, continuous TRAIN
improvement and acceptance DEV non-deterioration remain in PyTorch. The
historical structured trainer explicitly retains `legacy-cg`.

The staged implementation run is complete: 53 focused checks pass and both
CPU/CUDA A/B dense oracles pass. C completed exactly three 200-iteration
proposals, all rejected by linear certification, with zero updates. See the
[result report](../reports/petsc_kkt_implementation/SUMMARY_ZH.md). Full-size
convergence remains unresolved at this budget and preconditioner setting.

`petsc_kkt_solver.py` imports PETSc only when selected. A missing dependency,
incompatible scalar precision or CPU-only PETSc with CUDA input raises an
error before exporting tensor storage. There is no automatic backend or
device fallback. Use `--linear-solver legacy-cg` for the previous regression
path; `--cg-iterations` applies only to that path.

## Linear system and diagnostics

`ReducedKKTLayout` removes deterministic fixed boundary coordinates and their
continuity rows (the response clock), preserving zero increments and zero
reported multipliers in those slots. Excluded rows must have zero residual;
their linearized residual is checked again after solving. A generic fixed
coordinate with a nonzero continuity residual cannot be discarded this way.

For physical increments `d = S_d d_hat` and multipliers
`lambda = S_c^{-1} lambda_hat`, the operator is the congruence
`D K D`, where `D = diag(S_d, S_c^{-1})` and
`K = [[J_r.T J_r + damping I, C.T], [C, 0]]`. The right-hand side is
`-D [J_r.T r, c]`. Damping remains in original coordinates. Policy scales
are `max(abs(theta), .01)`. Response boundary/constraint scales are 1 except
angular velocities 10, integral .5 and previous velocity 3. Both orientation
blocks retain their existing SO(3) representation, with scale 1 radian.

The V1 `--kkt-preconditioner block-diagonal` applies a fixed positive inverse
diagonal: primal entries `1/(1+damping)` and dual entries 1, in normalized
coordinates. It failed B on CPU after 2000 iterations (true residual .00538).
The matvec matched the explicit Jacobian KKT to 2.2e-16, so the failure was
retained as evidence for upgrading the preconditioner.

The default is now `--kkt-preconditioner curvature-diagonal`: 8 fixed-seed
Rademacher probes estimate `diag(S_d H S_d)` once per linearization, then
`abs().clamp(1e-4, 1e4)` makes the primal diagonal strictly positive. Dual
entries remain 1. `--kkt-curvature-probes` controls the probe count. A local
generator preserves the caller's random state. This adds no explicit
Jacobian, and the resulting PC is frozen throughout MINRES. MINRES uses left
preconditioning. There are no Schur inner iterations, adaptive PC updates,
multiplier refits, or QLP mode in this path.

The default controls are `--kkt-rtol 1e-6 --kkt-atol 1e-10
--kkt-max-iterations 200`. `--kkt-monitor` prints PETSc iteration residuals.
Ambient PETSc options cannot override the solver or PC: the implementation
intentionally does not call `setFromOptions()`.
Internally, diagonal norm bounds tighten PETSc's stopping tolerances to allow
for the variable/constraint scaling and the preconditioned residual norm.

After the solve, Torch recomputes the original reduced residual and its
scaled equivalent. Both relative residuals must be at most `kkt_rtol`, with
RHS norms floored at `1e-12`, and the solution and full linearized constraints
must be finite. A positive PETSc reason is also required. PETSc's absolute
stopping tolerance does not bypass these relative checks. An uncertified
direction is rejected before backtracking and the policy stays unchanged.

Every response proposal records backend, reason, iteration count and both
relative residuals. `--ms-debug` additionally records trust/merit diagnostics.
`kkt_stationarity_*` describes the returned primal step using the raw
full-step multiplier (`stationarity_multiplier_kind=raw-full-step`); it can
be nonzero after backtracking. `linear_solver_*` describes the original
linear solve. These quantities must not be conflated.

## Installation

The tested dependency pair is PETSc/petsc4py 3.25.5, real double precision,
CUDA enabled, serial `COMM_SELF`. A single PETSc build has one scalar type;
this build requires `--dtype float64`. Build a real single-precision variant
separately if FP32 is needed; shared storage cannot reinterpret FP32 as FP64.

Use a separate Python environment. One local build can reuse an existing
PyTorch installation with `python3 -m venv --system-site-packages <venv>`.
Inside downloaded PETSc 3.25.5 sources, configure and build, replacing CUDA
path and architecture with those of the target machine:

```bash
python3 configure PETSC_ARCH=arch-cuda-real-double \
  --with-mpi=0 --with-fc=0 --with-cc=gcc --with-cxx=g++ \
  --with-cuda=1 --with-cuda-dir=<cuda-toolkit> --with-cuda-arch=<sm> \
  --with-precision=double --with-scalar-type=real \
  --with-debugging=0 --with-shared-libraries=1
make PETSC_ARCH=arch-cuda-real-double -j8 all
```

Install Cython >=3, setuptools, wheel and NumPy in the isolated environment.
Set `PETSC_DIR` to the source root and `PETSC_ARCH=arch-cuda-real-double`, then
install the matching petsc4py source with `python -m pip install
--no-build-isolation --no-deps <petsc4py-3.25.5-source>`.

RHS, solution, callbacks and PETSc work vectors use DLPack. CUDA storage is
shared without a host vector copy. Explicit device synchronizations protect
ordering across the two runtimes. This is not a claim of zero synchronization
or of measured throughput improvement.

Primary references: [Vec DLPack interfaces](https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Vec.html),
[MINRES requirements](https://petsc.org/release/manualpages/KSP/KSPMINRES/),
[Python matrix and PC contexts](https://petsc.org/release/petsc4py/petsc_python_types.html),
[CUDA configuration](https://petsc.org/release/install/install/).

## Staged verification

Run tests with the PETSc environment's Python. The ordinary tests skip
PETSc-specific checks when the optional dependency is absent; the A/B tool
fails if PETSc is absent, so a skipped test cannot authorize stage C.

```bash
python -m pytest -q tests/test_petsc_kkt_solver.py
python tools/validate_petsc_kkt.py --device cpu --output reports/petsc_kkt_ab_cpu.json
python tools/validate_petsc_kkt.py --device cuda --output reports/petsc_kkt_ab_cuda.json
```

A is an artificial 108-dimensional symmetric indefinite KKT with nonuniform
scaling and a fixed clock. B uses all parameters of a small real response
policy, two scenarios and two H2 segments in FP64. Both explicitly form
`J_r` and `C` in original coordinates and compare with `torch.linalg.solve`.
The reported acceptance limits are dense relative residual <1e-8 and solution
relative error <1e-6, in addition to the solver's independent certification.

Only after both stages pass should a separately authorized bounded experiment
use the learned 825 checkpoint with 64 scenarios, 2xH125 and 3 proposals. The
checked-in `configs/response_petsc_probe_825.args` encodes that experiment.
It uses weights-only initialization and a fresh output directory; it never
rewrites the original checkpoint. Algebraic correctness, learned performance
and deployment safety remain separate evidence.
