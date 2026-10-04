# Structured geometric feedback and recurrent residual

This branch implements the requested Actor-only experiment, starting from
`e45739e1b290df67592ccbcc3799cdaed4764846`. It is not a certified controller.
The physics, sensing, task objective, reset/sampling distributions, full-horizon
BPTT, backward Time Decay, group clipping and Adam rules are outside the scope.

## Intended deployment contract

Keep the 22-value deployable observation, 16 response features, native
`GRUCell(16, memory_dim)`, hidden-only recurrent state, and absolute normalized
FR/BR/BL/FL motor-command interface. Do not expose physical parameters, motor
truth, disturbance truth or velocity-delay metadata to the Actor. Do not add a
parameter-identification head, external controller, critic, free gate or action
slew limiter.

Replace the free `[W_c | W_h]` readout with two explicit paths:

1. Current measured motion -> bounded position/velocity feedback -> desired
   thrust direction -> nonlinear reduced-attitude feedback and angular damping.
   Feedback coefficients have declared signs and finite parameterized ranges.
2. The unchanged causal response features -> native GRU -> constrained recurrent
   command residual, without an explicit physical-parameter prediction target.

Combine the paths in collective/roll/pitch/yaw *command* coordinates, mix into
four motors, then apply the final command saturation. These coordinates are not
newton/newton-metre wrench commands. A fixed mixer knows only the existing X-frame
motor ordering and signs, not the sampled airframe's thrust curve or inertia.

## Boundaries that the implementation must preserve

- The existing task has no commanded heading. Reduced attitude controls the
  thrust direction; yaw-rate damping must not introduce a new yaw target/loss.
- Geometry must work beyond small-angle approximations. Near antipodal thrust
  directions, a recovery-axis convention is necessary; it is not a globally
  smooth stability proof and must be tested and documented explicitly.
- Gravity is known, but the hover motor command is not known for each airframe.
  A common trainable operating-point bias and a recurrent collective residual
  must not be described as an oracle hover thrust.
- Bounding residual amplitude is different from bounding input sensitivity.
  The residual head's incremental bound and the full GRU/input-to-output path
  must both be considered; neither alone certifies the motor/rigid-body loop.
- Final saturation couples the channels and can reduce control authority.
  Audit the actual mixed and saturated output, not only pre-saturation heads.
- Initial hidden state is zero and updates on the first observation. No warm-up
  may use hidden physics or assume that dynamics have already been identified.
- The one-traversal group-gradient probe must cover every new trainable parameter
  and agree with independent grouped VJPs, including compacted live scenes.
- New architecture metadata must reject old Actor/Adam checkpoints instead of
  silently interpreting the old affine weights as geometric gains.

## Verification scope

Use deterministic unit tests, finite-difference derivatives, grouped-gradient
comparisons and bounded smoke checks. Do not launch a training campaign or claim
improved flight performance from these tests. CUDA throughput and closed-loop
robust stability require separate evidence. Deployment remains unauthorized.

Structural ranges and residual limits are experimental settings, not verified
stability margins. Whole-loop checks must include the GRU, motors, rigid body,
known-command history, sensing delays and final saturation, using the real
forward response rather than Time-Decay-modified derivatives.

## Primary references and applicability

- Lee, Leok and McClamroch, *Geometric tracking control of a quadrotor UAV on
  SE(3)*, CDC 2010, DOI: 10.1109/CDC.2010.5717652. Geometric feedback motivates
  avoiding Euler-angle subtraction; its model-based stability result is not
  claimed for this normalized-command, uncertain-airframe residual policy.
- Song, Slotine and Pham, *Stable modular control via contraction theory for
  reinforcement learning*, L4DC 2024, PMLR 242:1136-1148. Structural derivative
  constraints motivate the design, but its assumptions are not automatically
  satisfied by this discrete-time recurrent motor-control system.

Implementation details and verification scope are recorded below.

## Implemented forward path

`ResponseMotorPolicy` now owns a native GRU, an
`IncrementalResidualReadout(64,4)`, and a `GeometricFeedback` module. It has
16,013 trainable scalars at hidden width 64. The external observation, state and
output dataclasses remain compatible in shape; the checkpoint architecture ID
changes to `geometric-gru-bounded-residual-absolute-motor-policy-v5`.

The direct path uses measured world position/velocity, measured rotation and
body rates. Positive position/velocity feedback coefficients form a desired
specific-force vector. Horizontal acceleration is radially saturated to 6 m/s^2;
vertical desired specific force stays between 0.2g and 1.8g by a smooth mapping.
This avoids normalizing a zero desired force or dividing collective thrust by
cos(tilt) around 90 degrees. These limits constrain the desired force used in the
feedback geometry; they do not constrain the actual vehicle acceleration.

The desired direction is expressed in the measured body frame. A reduced-attitude
rotation vector aligns body +z with that direction; a separate negative body-rate
term provides the intended damping contribution. No desired yaw is introduced.
At nearly antipodal directions (`cos < 0` and squared cross-product magnitude
below `1e-8` by default), the implementation selects +body-x with angle pi. This
small switching cap is not smooth at its boundary, is noise-sensitive near
inversion, and is **not** a global stability guarantee. Finite derivatives away
from this boundary and finite exact-inversion behavior are tested separately.

The nine direct-path coefficients are sigmoid-parameterized in finite intervals:

| Coefficient | Initial | Lower | Upper |
|---|---:|---:|---:|
| Horizontal position gain | 1 | 0.2 | 4 |
| Vertical position gain | 2 | 0.2 | 6 |
| Horizontal velocity gain | 1.5 | 0.2 | 4 |
| Vertical velocity gain | 2 | 0.2 | 5 |
| Tilt-vector gain | 0.06 | 0.01 | 0.25 |
| Roll/pitch rate damping | 0.035 | 0.002 | 0.15 |
| Yaw-rate damping | 0.025 | 0.002 | 0.10 |
| Collective projection gain | 0.3 | 0.05 | 1 |
| Common hover logit | 0 | -1 | 1 |

These are provisional engineering settings, not fitted or certified margins
for TTI 40-1200 or any other airframe range. A shared hover bias is not exact
hover trim for every aircraft. The recurrent collective residual may supply a
persistent correction; there is no requirement that it always be smaller than
the base command. Its constant bias can also be wrong, so bounded sensitivity
alone does not ensure correct equilibrium or tracking.

The combined channel coordinates are **motor logits**: `[T,r,p,y]` maps to
`[T-r-p-y, T-r+p+y, T+r+p-y, T+r-p+y]` in FR/BR/BL/FL order, followed by the
original final motor-wise tanh. They are not force/torque units and are not
independently preserved after tanh or the motor's nonlinear thrust curve.
`policy.components(...)` exposes both contributions and the mixed logits for
read-only diagnostics. It is not a new hidden-state field.

## One-step residual sensitivity bound

Let `h_new=GRU(c,h_old)`, with `h_old` in the reachable cube [-1,1]^H. Zero
initialization and the native GRU convex gate update preserve this cube. External
injected hidden states outside the cube are outside the claimed bound's domain.
For PyTorch's r,z,n gate order, define

```
B = max_row(sum(abs(W_hn), row) + abs(b_hn))
L_c = ||W_in||_F + B ||W_ir||_F / 4 + ||W_iz||_F / 2
L_h = 1 + ||W_hn||_F + B ||W_hr||_F / 4 + ||W_hz||_F / 2
L = L_c + L_h
```

These conservative derivative bounds follow from sigmoid slope <=1/4, tanh
slope <=1, `abs(h_old-n)<=2`, and spectral norm <= Frobenius norm. They bound a
controller step; `L_h` is not assumed below one. For each readout row:

```
s_i = 1 + (A_i/G_i) * ||w_i||_2 * L
w_eff_i = w_i/s_i
residual_i = A_i * tanh(w_eff_i @ h_new + b_i)
```

The residual has amplitude <=A_i and Euclidean derivative norm with respect to
`[c,h_old]` <=G_i. By the mean value inequality this also bounds finite differences
between two such inputs in the convex hidden cube. This is in the **existing
normalized feature/hidden coordinates**, not metres/newtons or raw-observation
coordinates; it is not a bound on an arbitrarily long history-to-action map.

Default amplitudes are `(0.6,0.12,0.12,0.06)` and derivative budgets are
`(0.15,0.06,0.06,0.03)` in collective/roll/pitch/yaw order. These conservative
bounds can reduce adaptation authority and require later ablation/calibration.
A_i=0 is an explicit residual-channel ablation. The scale depends on BOTH input
and recurrent GRU weights and the candidate recurrent bias, not only W_h.
Neither this bound nor positive base coefficients prevents all destructive
interference, biased trim, saturation, delay-induced instability or transient
physical amplification.

The normalization is differentiated exactly with respect to parameters. The
one-pass probe collects geometric coefficient partials before their across-scene
sum, collects effective readout partials, and analytically pulls them back through
this normalization, including the additional GRU-parameter terms. It still uses
one `torch.autograd.grad` traversal of the physical rollout and no per-scene
parameter-gradient storage. The original group clipping/averaging is unchanged.

## Training and diagnostic commands

Use a new directory. Old v4 affine Actor/Adam checkpoints are intentionally
rejected; no silent or partial weight migration is added. The retained training
config is byte-identical, including its physical settings. Override its old work
directory explicitly:

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args \
    --work-dir runs/geometric_feedback/seed7
```

The existing budget flags still govern any training run. To alter residual
limits, use four values in channel order, for example:

```bash
--residual-amplitude 0.6 0.12 0.12 0.06 --residual-gain 0.15 0.06 0.06 0.03
```

After obtaining a new-architecture checkpoint, the independent finite-difference
audit compares actual paired forward trajectories, with shared noise/pulse tapes:

```bash
python tools/audit_geometric_feedback.py \
    --checkpoint runs/geometric_feedback/seed7/best.pt \
    --output runs/geometric_feedback/audit_full.json
python tools/audit_geometric_feedback.py \
    --checkpoint runs/geometric_feedback/seed7/best.pt --base-only \
    --output runs/geometric_feedback/audit_base.json
```

The augmented perturbation includes position, velocity, tangent attitude,
angular velocity, actual motor state (audit only), issued-command history, all
three delayed-velocity slots and GRU memory. It records terminal and internal
peak gains using an explicitly scaled metric, and also final motor-command gain.
`--warmup` tests nonzero recurrent history; zero tests cold start. Repeat with
multiple perturbation sizes, directions, seeds and horizons before interpreting
sensitivity. Post-termination motion is marked out-of-task rather than silently
frozen. The tool does not update weights, filter samples, change training or
select checkpoints; its report always says `certified: false`.

Full robust closed-loop conditions and feasible operating regions remain to be
established. In particular, there is deliberately no claim of recovering every
120-degree reset, supporting every possible unbounded Gaussian disturbance, or
preserving base feedback under every saturation. Do not interpret a regularizer,
a local derivative bound, a finite test set or numerical training stability as
proof of flight safety.

## Local verification (2026-10-04)

`python -m pytest -q tests --junitxml=full-tests.xml` completed on Python 3.13,
PyTorch 2.10.0+cpu: **344 passed, 10 skipped, 93.41 seconds**. CUDA and Pygame
were unavailable in that environment. Existing CI supplies the separate CPU
Pygame/headless GUI checks; its result is reported on the pull request rather
than inferred here. No training campaign or real-flight test was performed.

Tests include large-angle geometric signs, the declared antipodal convention,
first-step recurrence, channel mixing/saturation, residual input/hidden Jacobian
bounds, finite differences through the parameter-dependent normalization,
independent grouped VJP agreement with and without Time Decay, compacted scenes,
an H500 delayed-measurement graph fixture, checkpoint/resume and read-only audit.
These are numerical/software regressions, not evidence of learned recovery.

Physics, sensing, sampling, task loss, group layout/clipping, Time Decay,
optimizer update rules and all existing argument files are preserved. Only the
Actor constructor hash changes in the protected-kernel contract, with the old
hash and rationale retained. The training module changes only checkpoint
architecture/configuration validation; the CLI adds the new Actor options.
