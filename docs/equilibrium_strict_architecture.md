# Structured recurrent control and full-space shooting

This is an experimental branch beside the Q2/TBPTT baseline.  It does not
change `MotorGRUPolicy`, the L2F dynamics, the Q2 losses, or the baseline
training configurations.  A structured checkpoint is not a promoted
replacement for Q2 until every migration gate described below passes.

## Why the first prototype was replaced

The old `strict_multiple_shooting.py` passed each H250 endpoint directly into
the next segment.  Its gradient was exactly the checkpointed H1000
single-shooting/full-BPTT gradient.  That validates checkpoint recomputation,
but it does not provide the numerical conditioning of full-space multiple
shooting.  The file remains only as a compatibility diagnostic through
`checkpointed_exact_bptt.py`.

The first equilibrium controller also copied the complete Q2 motor head into a
head named `trim`, while initializing the new feedback to zero.  The copied
head still read position, velocity, attitude and angular rate every step.  It
therefore was neither a slow equilibrium estimator nor a behavior-preserving
decomposition of Q2.

## Controller

`structured_policy.py` implements one physical action path:

```text
deployable observation
        |-----------------------------|
        v                             v
fast differentiable motor observer   causal response identifier
        |                             |
        |                    capability / disturbance context
        |                             |
        +---------- equilibrium trim and body-z target
                                      |
current state ---- physical equilibrium error e
                                      |
             reference feedback + bounded contextual gain
                                      + O(||e||^2) neural residual
                                      |
                           desired specific wrench
                                      |
                    damped constrained local allocator
                                      |
                              four motor commands
```

The hidden variables have two time scales.  The deployable motor observer
tracks the fast actuator state from executed actions and estimated rise/fall
time constants.  The observer bank is explicitly versioned: v1 is the original
15-candidate (five by three) diagnostic, while v2 is the preregistered
35-candidate (seven by five) coverage experiment.  Every candidate is advanced only by the
actually executed command and never reads the capability prediction.  Its
modal response statistics pass through a zero-initialized `8K`-to-24 adapter,
so enabling the bank can preserve an old controller exactly while exposing
motor-lag hypotheses to Phase A1.  The causal identifier consumes a
24-dimensional response-only sequence at every physical step, while its
capability/trim outputs are published and held only every 25 steps.  Its inputs
are lagged (0/4/12-step)
command-innovation/force and roll-pitch-yaw/angular-acceleration cross-products,
rise/fall motor-response products, and excitation energy.  This preserves the
order information needed to distinguish actuator lag from authority; the old
design averaged commands and responses before one recurrent update and was
empirically unable to identify even a fixed training bank.  Absolute position,
velocity, attitude error, angular-rate level and position integral do not enter
this path.  It predicts the six control capabilities

```text
(thrust/weight, roll authority, yaw ratio, inertia ratio, tau_up, tau_down).
```

During the first 50 steps a deterministic `.005` motor-coordinate probe is
mapped through the same wrench mixer and constrained allocator.  It uses two
nonperiodic 25-step blocks; each block is exactly zero-sum in every motor
channel, and the requested lag-0/4/12 design is full rank.  Four independent
authority-balanced H125 banks passed finite/recovery checks, but the actually
executed innovation Gramian remained ill-conditioned (worst condition about
1402 after the corrected cadence, versus the preregistered `<30` gate).  Consequently this probe is treated
as a safe excitation candidate, not evidence that the legacy representation is
identifiable.  Call25 therefore only updates persistent identifier memory;
capability and trim remain unavailable until the pre-registered call50
publication, with call75 used as the non-drift check.  K15 subsequently failed the strict tail-coverage/oracle gate;
it remains archived evidence.  K35 plus the exact causal-sequence/analytic
ceiling diagnostic is the next release gate, and Phase A1 remains disabled
until that gate passes.

Persistent external acceleration is a separate explicit world-frame observer,
updated from the force-balance residual.  No neural trim or body-z head exists.
The response window stores force-balance sufficient statistics that are affine
in thrust-to-weight, then evaluates the whole window with the capability value
produced at that cadence boundary.  This avoids turning a one-window-stale TW
estimate into a fictitious external force.  Velocity increments are paired
with the reconstructed previous orientation (the orientation that actually
generated that increment), not the current orientation.
For the symmetric linear-thrust physical-fit family, body-z and equal-motor
trim are solved analytically from estimated thrust-to-weight and external
acceleration.  Scenarios outside the observer/actuator feasibility region are
reported rather than silently clamped out of a metric.

Positive capabilities use a normalized log-coordinate posterior.  Before
formal calibration the upper-confidence allocator is explicitly smoke-only.
Formal checkpoints need independent split-conformal metadata.  The one-sided
effectiveness bound is used only for thrust/roll/yaw allocation; time constants
are not treated as monotone effectiveness variables.

The fast error contains position, velocity, yaw-free body-z error, angular
velocity, and estimated-motor-minus-trim.  The fixed reference gain is never
trained by a long-horizon objective.  A capability-conditioned gain correction
is projected so that its allocator-induced action Jacobian remains a bounded
fraction of the reference Jacobian.  Capability uncertainty gates this
correction.  At the end of burn-in, an insufficiently narrow interval marks
the scenario as an identification failure; it is not silently counted as a
successful rollout and never falls back to a teacher.

The optional nonlinear residual is

\[
r(e,\chi)=\frac{\lVert e\rVert^2}{1+\lVert e\rVert^2}N(e,\chi).
\]

It is exactly zero and has zero error Jacobian at equilibrium when context is
fixed.  That statement applies only to this residual, not to the complete
observer/controller/actuator closed loop.  The residual is frozen until the
pure structured policy passes the H250 migration gate.

The allocator maps a four-dimensional specific wrench (collective
acceleration and three angular accelerations) to four motors with damped least
squares, smooth saturation, action/rate limits, and diagnostics for condition,
headroom and wrench residual.  It is a local equilibrium allocator, not a
global inverse-dynamics proof.

## Q2 migration

Q2 is an offline teacher, never a runtime dependency.  DAgger collection runs
teacher and student recurrent states on the same actual observation history,
and the motor observer is advanced with the action actually sent to L2F.
Teacher execution is scheduled per scenario and is exactly zero in the final
rounds.  Train, conformal-calibration, validation, and final-evaluation scenario
banks use different seeds.

The slow equilibrium path never learns a Q2 action intercept.  It is supervised
only by simulator truth: capability, persistent external acceleration, analytic
trim/body-z and one-step equilibrium invariance.  A same-latent Q2 intercept is
retained only as a diagnostic because Q2's newly updated hidden state already
contains the current fast observation.  Treating that intercept as trim was the
source of an earlier false 28--40x oracle improvement.

The distinction is empirically material.  On 64 fixed physical-fit scenarios,
Q2 retains about 0.063 action RMS bias even after 251 steps of an analytic
equilibrium input (`reports/q2_equilibrium_bias_seed1707.json`).  The structured
controller intentionally rejects that bias: its reference/context gain is fit
only to the teacher's first-order action JVP around an independently burned-in
equilibrium recurrent state, and the O(||e||^2) recurrent residual handles
away-from-equilibrium nonlinear/history behavior.  It is therefore impossible
and undesirable to require exact Q2 action parity at physical equilibrium.

The migration gate is deliberately separate from training.  It requires an
independent pure-student L2F evaluation, no teacher/shield execution, valid
calibration metadata, finite trajectories, per-state mean/tail parity, explicit
velocity checks, high-authority angular-rate limits, and allocator diagnostics.
Failed gates stop the pipeline instead of being overridden with clipping or a
teacher safety chain.

## Candidate phase-space metric

`structured_stability.py` builds a discrete quadratic metric from explicit
second-order target dynamics.  The translational and normalized tilt/rate
coordinates include their real kinematic scale and sign.  Segment objectives
penalize sampled violations

\[
V(z_{k+1}) \le \gamma^2 V(z_k)
\]

and a smooth Rockafellar--Uryasev CVaR of final energy.  The CVaR threshold is
the stationary solution of the smooth empirical objective, not a detached raw
quantile.  At least eight effective tail scenarios are required.

This is a candidate Lyapunov metric and a sampled training condition.  The
augmented closed-loop Jacobian (physical state, real and observed motors, slow
state and integral), finite-time gain, saturation behavior, and held-out
rollouts provide sampled local evidence only.  Passing those finite checks does
not by itself constitute a stability certificate over a continuous region.

## Full-space multiple shooting

`full_space_shooting.py` keeps H250 endpoint nodes independent and solves the
linearized joint problem

\[
\min_{\Delta\theta,\Delta z}
\frac12\lVert r+J_\theta\Delta\theta+J_z\Delta z\rVert^2
+\frac\mu2\lVert(\Delta\theta,\Delta z)\rVert^2
\]

subject to

\[
c+A\Delta z+B\Delta\theta=0,
\qquad c_k=z_{k+1}-\Phi_{250}(z_k,\theta).
\]

Physical orientation uses a three-dimensional SO(3) local residual.  The
boundary contains the complete deployable recurrent state, including physical
and motor state, previous action, integral, identifier, motor observer, held
capability posterior and held gain/allocator transition state.  Formal segment
starts occur after burn-in and on a cadence boundary.

JVP/VJP products and a Schur-complement solve avoid a dense KKT matrix.  A
fixed-reference-state action trust region constrains RMS and per-scenario
maximum motor changes.  After every tentative step the driver restores exact
continuous endpoints and recomputes task, merit, all trajectory actions, and
predicted/actual reduction; only the restored rollout can accept the policy.
The LM damping and radii adapt after acceptance/rejection.

The present implementation is an algebraic matrix-free MVP.  It has genuine
independent nodes and a joint KKT direction, but still uses nested unpreconditioned
CG and backtracking along one damped Gauss--Newton direction.  It must not be
described as a production-scalable SQP solver until block preconditioning,
large-batch convergence and held-out post-step guards are demonstrated.

## Promotion order

1. Pass structural gates: response-only slow input, no learned trim/body-z
   head, residual value/Jacobian zero at equilibrium, and record Q2's
   equilibrium bias.
2. Phase A1 trains the capability mean explicitly at the held t25/t50/t75 publication
   instants while the uncertainty scale is frozen and no conformal score is
   installed; validate the explicit disturbance
   observer, analytic equilibrium and one-step invariance.  Heteroscedastic
   NLL is not allowed to replace mean accuracy by a large sigma, and no Q2
   action loss enters this phase.
3. Phase A2 is allowed to start only after the A1 physical-mean gate.  It
   freezes the identifier and mean head, fits only uncertainty scale, then
   performs independent calibration/width checks.
4. With residual disabled, fit the capability-conditioned gain only on
   equilibrium-history local JVPs.  Actual-flight-history JVPs are shift
   diagnostics, not targets.
5. Freeze the equilibrium and local-gain parts, then DAgger-train only the
   O(||e||^2) recurrent residual.  Action parity applies only on a preregistered
   away-from-equilibrium distribution and uses a held-out structured-residual
   oracle threshold.
6. Freeze all weights and obtain a self-consistent formal capability quantile.
   Under that exact deployment hash, rerun the equilibrium, local-JVP,
   residual/action, allocator and H250 evidence; pre-calibration reports cannot
   be inherited as final evidence.
7. Pass independent pure-student H250 and H500 L2F gates on natural and
   authority-balanced banks.
8. Run two H250 segments with the full-space solver, then four only after
   continuity, KKT, action-trust, restored-merit, and held-out gates pass.
9. Every accepted policy change invalidates calibration and migration; repeat
   steps 6--7 before promotion.  The checked-in runner includes these
   post-MS calibration, postcheck and paired-migration stages explicitly.

No failed phase is promoted merely because aggregate loss decreases.
