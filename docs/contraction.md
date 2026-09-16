# Actor + training-only contraction metric

## Scope

This is a **sampled differential dissipativity regularizer**, not a certified
controller, an action filter, a Critic, or a learned optimizer. The deployed
`ResponseMotorPolicy`, original task score, time decay, RK4, actuator semantics,
initialization, termination thresholds, and pooled task CVaR are unchanged.
The new MLP exists only in training/checkpoint diagnostics.

### Why not demand strict contraction in every coordinate?

The task does not specify an absolute heading. Requiring every pair of states
to converge to one full state would impose a new heading objective and may be
infeasible for neutral internal modes. We do NOT delete yaw from the Jacobian.
Instead, a bounded positive metric measures the **entire dynamic state**, while
a semidefinite task term asks for dissipation of position, velocity, thrust-axis
and angular-velocity perturbations. This is weaker than full-state exponential
contraction and is reported as such.

## Exact criterion used by this implementation

Let `s(Z)` be the fixed, dimensionless dynamic embedding and `v_t` its true
closed-loop directional derivative. Let `y(Z)` contain normalized position,
velocity, world thrust axis `R e_z`, and angular velocity. For an interval of
actual duration `T` (including the first terminal transition), define

```
V_start = v_start' M(s_start, constants) v_start
V_end   = v_end'   M(s_end, constants)   v_end
r = (V_end + rate * T * ||delta_y_start||^2) / V_start
L_aux = mean(relu(log(r))^2)
```

The desired inequality is `r <= 1`. Its unnormalized matrix form is
`A' M_end A - M_start + rate*T*E_start'E_start <= 0`, in legal tangent
coordinates. For neutral task directions, this asks for non-expansion rather
than strict decay. If the inequality held for ALL relevant states, legal
perturbations and shared input sequences in a forward-invariant domain, bounded
metrics would bound the products of interval Jacobians by their endpoint norm
ratios. Full-state exponential decay would require additional conditions.

**Training samples do not establish those assumptions.** We use two sampled
directions by default. The log is a scale-independent penalty, not a proof, and
`rate=0.1` is an initial experimental setting, not a certified convergence rate.
No hard projection constrains the Actor update. Finite Adam updates can still
violate the condition. Time decay remains in task BPTT while this is evaluated.

## Geometry and complete feedback

With 64 memory units the legal tangent has 100 dimensions; its rotation-matrix
embedding has 112 dimensions. Both current orientation and cached previous
orientation use legal three-dimensional right-rotation retractions. Matrix
entries are NOT perturbed independently. The previous measured matrix is
`R_true + noise`: the same cached noise is retained during retraction.

Included dynamic quantities: position, velocity, current rotation, omega,
physical motor state, previous executed command, GRU memory, integral,
previous measured velocity/omega/rotation, and older action. The unused
`last_action` copy and discrete counters are not tangent variables. Calls and
time indices retain their actual values. Interior hidden/history states are
sampled from real rollouts and are never reset for a contraction probe.

Fixed scales are per-airframe position limit, 2 m/s, 10 rad/s, the configured
integral bound, motor half-range, and unit normalized command/hidden scale.
Rotation matrices use Frobenius scale 1/sqrt(2), so their infinitesimal rotation
norm equals the three-dimensional angular tangent norm. RPM and normalized
rotor states are converted only in the diagnostic embedding, NOT in control.

Metric context contains mass, inertia, rising/falling time constants,
thrust-to-weight, torque-to-inertia, and external force/torque. These are
fixed across a pair and enter ONLY the metric. All state-induced feedback
through the Actor and GRU remains differentiable.

## Bounded metric -- no zero-ruler solution

The MLP has two SiLU hidden layers (default width 64), and outputs diagonal
parameters `d` and a rank-4 matrix `B`. It represents

```
M = lo I + (hi-lo)/2 * (diag(sigmoid(d)) + B'B/(1+||B||_F^2))
```

Both bracket terms have eigenvalues in [0,1], so `lo I <= M <= hi I` for
all finite outputs. Defaults are lo=0.5, hi=2, condition bound 4. The low-rank
term permits cross-state coupling. This restricted metric class can fail to
represent an otherwise valid certificate; optimization failure is not a proof
that the plant/controller cannot be stabilized. No direct inverse/eigendecomposition
is needed for the training loss. `worst_direction` does use a generalized
eigenproblem for a bounded **single-state** audit, never a region certificate.

## Training and evaluation

For each update, uniformly choose up to 8 scene IDs, then one live saved
boundary for each scene, with room for the configured local interval.
Early-failing scenes are NOT removed from this selection. Each scene follows
its own Actor and physics; the same external parameters and noise tape are used.
`local_flow` reuses production `rollout(..., time_decay=0)` for each step.
Every true step is differentiated with `torch.func.jvp`; mixed derivatives
train both Actor and metric. There is no `detach` inside that local flow. The auxiliary policy is a shallow
parameter-sharing view: only the GRU fused primitive is expanded into the same
PyTorch GRU equations. This avoids the documented CUDA forward-AD limitation
(pytorch/pytorch issue #174355) without changing Actor parameter shapes or the
deployment implementation. CPU native/unfused values and derivatives are tested;
roundoff equality is tolerance-based, not bitwise. Every auxiliary probe checks
its endpoint against the native Actor and requires identical termination masks.
CUDA tests remain required on the user's GPU.

Time decay only affects the original task-gradient branch. It is NEVER used
to claim physical contraction. No likelihood-ratio/terminal-time derivative
has been introduced. Both regularizer and task preserve the true terminal
transition and stop failed rows. Padded rows are excluded; terminal intervals
remain explicitly reported failures, not contraction successes.

Directions alternate between full-state probes and plant/command-only probes
to avoid the 64-dimensional memory drowning out physical perturbations. Probes
are reproducible from the update index without altering training RNG. Their
actual additional forward transitions and duration are logged.

The update combines `gradient_scale * (g_task + weight*g_aux)` for Actor and
`gradient_scale * weight*g_aux_metric` for the metric, before the existing
finite checks and clip. Separate Adam states share the configured learning
rate. Parameters, both optimizers and RNG are committed/rolled back together.

Periodic fixed EVAL preserves the original task score/best-model selection.
A small fixed EVAL subset supplies independent contraction diagnostics without
training on them. Reports include directional violations, terminal/complete
interval counts, metric gain, **unweighted Euclidean tangent gain**, internal
prefix gain, sample coverage, and `certified=false`. Endpoints improving does
not bound the interior; the maximum real-prefix gain is separately reported.

## Configuration and compatibility

The two reference launch configs now explicitly enable:

```
--contraction-weight 0.1
--contraction-steps 10
--contraction-samples 8
--contraction-directions 2
```

Their work directories are new `runs/*_contraction/seed7` paths. Bare CLI defaults
to weight=0 for an explicit baseline; `--contraction-weight 0` disables all metric
initialization, gradients and metric EVAL. Other options have the
`--contraction-` prefix and are validated and bound in checkpoints. Local interval
length is independent of the task's H50 recomputation chunks.

Actor parameter names/shapes and environment/action semantics are unchanged.
A compatible native-motor v3 checkpoint can initialize Actor weights through
`--init-checkpoint`, with a fresh metric and fresh optimizers in a NEW directory.
Old runs cannot strictly resume after source/algorithm changes. New exact
resume restores Actor, metric, both Adam states, RNG and sampling progress.
Actor-only export/load does not require or use metric outputs.

## Verification and limits

Tests exercise metric bounds, legal rotations/noise, heading-neutral task
geometry, true JVPs versus central differences, mixed Actor gradients versus
finite differences, no use of decay in the auxiliary path, first-terminal-step
handling, local worst-direction witnesses, joint updates, exact resume, atomic
failure rollback, and deployment isolation. CUDA has a dedicated test that skips
when no GPU is available. CUDA performance/operator coverage must be tested on
the actual training GPU. Use bounded checks before a long run.

This branch has no formal domain verifier, no saturation-region certificate,
no guarantee of convergence to an optimal controller, and no guarantee that a
stochastic penalty/Adam update preserves the inequality. Reaching the task
boundary remains failure. The physical distribution is not narrowed to make
these checks pass. No long-run or real-flight claim follows from unit tests.

## Mathematical/code references

- Sun, Jha and Fan, *Learning Certified Control Using Contraction Metric*,
  CoRL 2020 / PMLR 155: https://proceedings.mlr.press/v155/sun21b.html
- Author code: https://github.com/sundw2014/C3M at
  `5275d32761c06807982b9d0318100ce06b52df4e`. Its continuous-time control-affine
  equations and dual metric W are NOT pasted into the RK4/GRU implementation.
- PyTorch JVP reference: https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html

This module is independently implemented for the existing discrete closed loop.
It borrows the positive-metric and closed-loop differential-constraint approach,
not C3M's controller, continuous-time certificate, or published performance.
