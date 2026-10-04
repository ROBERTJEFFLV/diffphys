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

Implementation details and fresh test evidence will be recorded after the
Actor and its necessary interfaces have been updated.
