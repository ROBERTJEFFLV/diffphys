# Known limitations of this revision

This revision contains the complete experimental implementation of the
physics-structured recurrent controller and the matrix-free full-space
multiple-shooting training path.  It must not be interpreted as a promoted
replacement for the Q2 controller.

## 1. Formal structured training is currently blocked

The pre-registered v4 motor-identification probe did not pass its paired Q2
safety gate.  The best candidate increased H125 mean angular-rate error to
`1.05661x` the zero-probe Q2 value, while the frozen acceptance limit was
`1.05x`.  Every registered candidate failed at least one training seed.

Consequently:

- no v4 waveform was formally frozen;
- the K35 causal-identifier collection stage was not authorized;
- production identifier pretraining and phases A1/A2/B/C were not run;
- no formal H500/H1000 multiple-shooting update was run;
- no blind seed was consumed.

The runner is deliberately fail-closed.  A new probe or identification
protocol must be pre-registered and independently checked; the existing 5%
safety threshold must not be relaxed after seeing this failure.

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
