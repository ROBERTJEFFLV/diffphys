# AFT: master + fixed PASS/STOP goals + differentiable completion-time proxy

## Scope and provenance

This AFT revision rebuilds the source tree from `master@82e7634ac82e8f719f33ee028768a73a82d33191`.
The prior AFT implementation remains in Git history (`85bf954`); its additional
`diffphys/` package, AL trainer, bank multipliers, paired parameter probes, event
root solver, reference-trajectory generator and non-deceleration constraints are
not in this tree. `master` itself is not modified.

There is still one training entry and one optimizer: `tools/train_response_control.py`
and `response_training.py`. The opt-in `pass_stop.py` task supplies goals and a
full-episode cost to the existing native grouped backward/Adam pipeline. No MPC,
critic, teacher, additional readout, learnable metric, curriculum or line search
is added. No long training was run for this change.

Protected master files are unchanged: `env_raptor.py`, `response_noise.py`,
`response_policy.py`, `response_task.py`, `response_groups.py`,
`response_sampling.py`, `response_audit.py`, `response_execution.py` and the
physical-coverage configuration. Their original Git blob hashes are recorded in
`pass_stop_validation.json`.

**Compiler scope follows master, not the old AFT:** `--rotation-backend compile`
compiles the quaternion-to-rotation expression. It does not compile the whole
RK4 step, whole backward, or Actor. The previous AFT whole-step compilation and
CPU-prefetch subsystem are removed along with that baseline. Native physical-cell
gradient clipping and the one-traversal GRU/Linear parameter probe remain active.

## Run

```bash
python tools/train_response_control.py @configs/agile_flight.args \
  --init-checkpoint /absolute/path/hover.pt \
  --work-dir runs/pass_stop/from_hover
```

`configs/agile_flight.args` includes the original master argument file and overrides
only the mission, `time-decay=0`, pulse generation and run directory. H500, 100 Hz,
4 x 512 TRAIN scenes, coverage128, native group probe and Adam remain. The inherited
50-update budget is a smoke/screening budget, not a claim of sufficient training.

`configs/pass_stop.json` defines the public task. Default positions are fractions
of each airframe's original world-origin boundary B:

- Point1: `(-0.7B,0,0)`, spawn; horizontal attitude and balanced motor state.
- Point2: `(0,0.38B,0)`, PASS; first sampled entry into the 0.1 m ball must have the
  specified full attitude (default 105-degree roll, one-degree tolerance).
- Point3: `(0.7B,0,0)`, STOP; position, velocity, full attitude and angular velocity
  tolerances. The terminal position target remains forever.

Set `relative_to_boundary=false` to use positions in metres. Invalid/out-of-bound
or overlapping goals raise an error; neither tilt nor geometry is silently reduced
for weak airframes. This checks geometry, not complete dynamic feasibility.

PASS has **no speed target, speed floor, non-deceleration condition or zero-rate
condition**. A time-minimizing policy may slow down when that helps complete the
whole task. STOP includes a zero-velocity and zero-angular-velocity requirement.

The initial velocity sample from master is retained, including the original 10%
zero-speed guidance rows. This is an initial-state deviation, not an initial dwell
phase. Constant external force and force-at-point pulse tapes are zeroed only by
task preparation; sensor noise and 10–30 ms velocity delay are preserved. Physical
parameters and random coverage sampling are unchanged.

## Command semantics and smallest network change

The original 22 measured values are unchanged. The native 16-feature GRU becomes
26 features, while its hidden size, Linear(64,4), tanh and six parameter tensors
remain structurally unchanged. The first 16 features keep their meanings except
that position is relative to the current fixed target. The extra ten numbers are:

1. End-point displacement in body coordinates (3).
2. First two columns of `R_measured.T @ R_target - I` (6); no Euler-angle wrapping.
3. STOP target flag (1).

The controller targets point2 immediately after spawning at point1. The position
switch uses the acquired/measured position and does not wait for attitude or
velocity success. On first entry, the next motor command uses point3 and its
level target attitude. GRU memory is not reset. Losses retain the true position
and attitude **at that first entry**, so switching does not forgive a bad PASS.

This specification is radius-entry passage, not an exact crossing-plane or gate
collision certificate. It does not claim the desired attitude at the exact centre
or over a surrounding flight band. At 100 Hz a brief between-sample entry can be
missed; the control and evaluation use the same sampled command convention.

## Two task objectives

### 1. Simultaneous PASS pose and terminal STOP

Let `r_p(t)=|p(t)-p2|/0.1` and

```
r_R(t) = |R(t)-R2|_F / (2 sqrt(2) sin(epsilon_R/2)).
```

For unit rotations, `r_R<=1` means the full relative rotation is within tolerance;
quaternion signs have no effect. At the first measured radius entry `j`, score

```
L_PASS = w_point * (relu(r_p(j)-1)^2 + relu(r_R(j)-1)^2).
```

When no entry occurs, use the single post-action live state minimizing the **sum
of these two violations**, and add a missing-entry penalty. Position and attitude
are never minimized at different times. The fixed spawn is excluded as a fallback
when a live post-action state exists, to avoid a witness with no control derivative.
No distance-dependent multiplier reduces the attitude penalty when moving away.
For a fixed witness/attitude, the direct radial derivative is nonnegative.

STOP ratios are position, speed, full attitude and angular speed divided by their
explicit tolerances. At H, their squared positive violations are scored unconditionally,
including when PASS has not happened. No stage transition turns on an otherwise
waived final stopping obligation. Default `w_point=10`; these remain finite strong
penalties, **not solver-enforced hard constraints or convergence guarantees**.

Master's `25*(1-cos(R_t.T R_(t+1)))` is an attitude-motion penalty, not a target
attitude penalty. It is not used by PASS/STOP. Original hover still uses it.
A weak original action-change regularizer and original failure accounting remain;
a differentiable boundary violation supplements the unchanged hard termination.

### 2. Completion-time proxy through ordinary physical BPTT

Define joint ratios at each actual state (no moving reference):

```
r_PASS(t) = max(position_ratio, attitude_ratio)
r_STOP(t) = max(position_ratio, velocity_ratio, attitude_ratio, angular_speed_ratio)
q(r) = 1 / (1 + relu(r-1))
```

`q` is one within all task tolerances and decreases outside. Its rational tail
avoids an exponential underflow and yields `dq/dr=-1/r^2` for `r>1`.

```
P(t) = max_{i<=t} q(r_PASS(i))            # best simultaneous past pose
S(t) = min_{i>=t} q(r_STOP(i))            # worst future stop quality through H
C(t) = min(P(t), S(t))
T_proxy = dt * sum_{t=0}^{H-1} (1-C(t))
```

Once a sampled entry exists, `P` is capped by that first entry's true joint pose
quality, so an incorrect first PASS cannot be erased by a later good revisit.
Dead/frozen padding has zero quality. Repeated weak near-passes are not summed,
so waiting does not accumulate completion evidence. Taking `min` rather than a
product also avoids multiplying two tiny quality gradients. Its tradeoff is
sparse bottleneck gradients; it does not provide a nonzero gradient to every node.

In the binary-quality limit, this clock equals the later of valid passage and
**last** departure from STOP, not the first momentary zero velocity. For fixed
event selection, increasing any ratio cannot lower the proxy; all its direct
ratio partials are nonnegative wherever differentiable. Delaying an otherwise
identical completion increases the measured proxy in the timing audit.

This is a continuous, **piecewise differentiable surrogate**, not the ordinary
derivative of the hard sampled completion time. `cummax/cummin`, ties and hinge
boundaries have nonsmooth points. Position-only command switches and first-entry
indices remain discrete: no saltation claim, straight-through switch gradient,
paired stochastic probe, or hidden detach-based correction is introduced. At a
changed hard branch, ordinary finite-difference equality is not promised.

There is no global ranking guarantee between arbitrary trajectories: nearly-correct
but technically failed passages can have a low proxy, and hard failure produces
zero time-proxy gradients. The separate PASS/STOP penalties, boundary residual and
independent hard metrics are therefore necessary. A finite horizon cannot certify
infinite-time hovering; reaching STOP only at H is not evidence of later stability.

The total scalar is

```
L = w_time * T_proxy/(H*dt) + L_PASS + L_STOP
    + weak_original_action_change + original_failure_cost + boundary_violation.
```

Only the task terms change. A native full-horizon graph yields the same per-scene
scalar derivatives, then master's physical-group shrink-only clipping, averaging,
global clipping and Adam apply. Those postprocessings are preserved, but the final
optimizer step is not labelled an exact raw scalar gradient. `time-decay=0` is
required for this task. No detachment at the old 50-step statistics boundaries.

## Checkpoints and deployment interface

Compatible master hover weights initialize the first 16 GRU input columns; new
columns are zero. All recurrent/readout weights are copied and **all** weights
remain trainable. Different matrix widths can introduce FP32 reduction rounding,
so structural initialization equivalence is not claimed bit-for-bit against the
old 16-input network. The original source checkpoint digest is checked before
expansion. Output contains the complete 26-input Actor, task specification, Adam
and RNG state, with a different architecture/objective identifier.

```bash
# Exact resume: same source and settings, no original hover file required.
python tools/train_response_control.py @configs/agile_flight.args \
  --resume runs/pass_stop/from_hover/latest.pt \
  --work-dir runs/pass_stop/from_hover --updates 200

# Evaluate the checkpoint's saved task, not a newly supplied mission.
python tools/train_response_control.py --mode evaluate --device cuda \
  --checkpoint runs/pass_stop/from_hover/latest.pt \
  --work-dir runs/pass_stop/eval --epsilon-p .01 --epsilon-a .01 --lambda-R 25

# Existing update-capsule replay includes the typed mission and native probe.
python tools/replay_response_update.py <audit/updates/u00000001_....pt>
```

Old AFT 36-input/event-AL checkpoints are not silently reinterpreted as this task.
Use a compatible master hover checkpoint or a new same-architecture checkpoint.
Original hover rendering tools are not generic PASS/STOP players; use
`pass_stop.task_rollout` with the saved mission for trajectory inspection.

For a controller, load with `response_training.load_policy_checkpoint`, then create
`policy.task_state(observation22, via, stop, via_rotation, stop_rotation, radius)`
from public goals and call `policy(observation22, state)` each control step. The
returned next state contains recurrent memory and the permanently latched STOP
flag. It uses no hidden motor/force/airframe state. Keep goal tensors on the same
device and dtype. A complete `.pt` still needs its matching model code and this
goal wrapper; it does not require a second weight file or MPC runtime.

## Verification and limitations

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_pass_stop.py tests/test_pass_stop_integration.py
python tools/audit_pass_stop_time.py --report /tmp/time_gradient_checks.json
```

The recorded run uses Python 3.13.5 / PyTorch 2.10.0+cpu. 68 focused tests passed.
They cover geometry/velocity separation, first-entry scoring, no accumulated weak
success, late departures, finite failure padding, native full H500 forward/backward,
a fixed-branch H500 finite difference, probe-vs-VJP, unchanged hover collection,
new-process standalone load, checkpoint/Adam/RNG continuation and update replay.
A 24-case temporal shift audit performs ten bounded gradient steps per case and
finite-difference checks. Its inputs are synthetic residual histories, **not a
quadrotor planner or learned flight benchmark**. Randomized derivative signs are
also checked; their nonzero counts are reported to expose gradient sparsity.

This environment could not clone the full repository. Original runtime dependencies
were retrieved through GitHub and individually verified by Git blob hash, and the
focused tests ran on that dependency snapshot. Existing master tests were retained
in the published tree, not replaced or counted as run. This is not a full-checkout
regression, CUDA compilation/throughput qualification, trained-hover import test,
105-degree learned-policy result, or hardware-flight authorization.

General reference for finite-difference limitations at nonsmooth points:
https://docs.pytorch.org/docs/main/generated/torch.autograd.gradcheck.gradcheck.html
