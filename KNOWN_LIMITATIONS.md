# Known limitations of this revision

This revision contains the complete experimental implementation of the
physics-structured recurrent controller and the matrix-free full-space
multiple-shooting training path.  It must not be interpreted as a promoted
replacement for the Q2 controller.

## 1. V5 training-bank checks pass; formal structured training is still unpromoted

The pre-registered v4 motor-identification probe did not pass its paired Q2
safety gate.  The best candidate increased H125 mean angular-rate error to
`1.05661x` the zero-probe Q2 value, while the frozen acceptance limit was
`1.05x`.  Every registered candidate failed at least one training seed.

The new v5 protocol uses Q2's natural recovery commands before considering
additional excitation. The four registered training banks pass paired safety
with exact Q2 parity, and a separate 4x128 training audit passes the K35 and
privileged physics/continuous-observer ceilings. See
[the v5 design](docs/passive_identification_v5.md) and
[verification](docs/passive_identification_v5_verification.md).

These are training-only results, not a v4 pass or a formal freeze. No validation
seed7707 or blind bank was consumed in this repair. Production identifier
pretraining, A1/A2/B/C and formal H500/H1000 MS have not been run. The formal
runner requires a new v5 freeze and all downstream gates. The 5% threshold is
unchanged, and the experimental active collective probe is not promotable.

## 2. The structured checkpoints in smoke runs are not deployable models

The small structured-policy and full-space checkpoints produced by smoke tests
use tiny batches/horizons and explicit override flags.  Their reports contain
`migration_gate_passed=false`; the full-space smokes contain
`accepted_steps=0`.  They validate code paths only.

The checkpoint shipped with this repository is therefore the last completed
Q2 baseline:

```text
reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt
SHA-256: b401dc6f02beadf51d1b55b24b9056f0b00f17a7a8a5d94d3675554fdb15370d
```

It is the teacher/baseline for migration, not a trained instance of the new
structured architecture.

## 3. Full-space multiple shooting remains an MVP solver

The solver uses independent shooting nodes, SO(3) boundary residuals,
matrix-free JVP/VJP products, a joint Gauss-Newton/Levenberg-Marquardt KKT
direction, action/parameter trust regions, nonlinear trajectory restoration,
and a disjoint held-out acceptance bank.  It no longer uses the old exact
reduced H1000 gradient.

However, its nested conjugate-gradient solve is not block-preconditioned and
is not a production sparse SQP implementation.  Smoke runs can finish without
meeting the strict formal linear-solver tolerances.  Scaling and conditioning
must be revalidated at 2xH250 and 4xH250 after a structured controller passes
the migration gate.

## 4. Stability diagnostics are sampled evidence, not a proof

The fixed-point restoration, augmented-state JVP/Arnoldi estimates,
finite-time gains, and candidate Lyapunov metric are local sampled checks.
They are not a nonlinear stability certificate over a continuous state and
parameter region, especially when the allocator saturates or changes active
faces.

## 5. Long-horizon control improvement has not been demonstrated

Earlier penalty multiple-shooting experiments showed nonzero cross-segment
credit and improvements on repeatedly trained scenarios, but those gains did
not establish unseen-scenario generalization.  The new structured controller
was introduced to address this failure mode, but its formal training is
blocked before a fair Q2-versus-structured comparison can be made.

Therefore this revision supports claims about implementation and fail-closed
validation only.  It does not support a claim that the structured controller
outperforms Q2, improves H5000 control, or is ready for deployment.

## 6. Identifiability, observer and allocator scope

The startup information ledger is a necessary excitation screen, not a noise-
aware Fisher matrix or a posterior confidence certificate. The physical ceiling
uses privileged motor states and external acceleration; it does not establish
identifiability from noisy deployable observations. The nearest K35 mode's
absolute error is reported separately. Formal A1 gates the actual continuous
observer at p95 motor RMS <=0.001 and max <=0.005 as well as capability means.

Startup information and action history are frozen under the simulator's
stationary-parameter episode assumption. Time-varying payloads or actuator
faults require change detection and a new identification/fallback design.

The allocator exactly solves its four-variable box QP, but its wrench mixer is
a nominal local control model, not an exact inversion of all motor lag branches
and zero-thrust knees. Observer/feedback compensation and continuous rollout
checks remain essential. Conservative UCB gains alone do not prove robust
closed-loop safety over the entire capability box.

Yaw projection now requires a verified neutral JVP direction. An arbitrary
absolute-attitude MLP is not automatically yaw-equivariant; a failed symmetry
check must not be bypassed by removing its eigenmode. Runtime/code hashes
invalidate stale calibration evidence when inference/state semantics change.

The validation claim prevents accidental re-use in one workspace. It is not a
cross-machine evaluation authority, and published deterministic seeds are not
truly secret blind data. Independent final evaluation still needs a controlled
external process.
