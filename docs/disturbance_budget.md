# Gaussian disturbances, force-at-point pulses and recovery initialization

This filename is retained for existing links. There is no shared 10% budget.
The current protocol extends the Gaussian protocol merged in b134dff with
user-requested recurrent force pulses and harder initial kinematics. TRAIN and
the single fixed EVAL use the same distribution. This is not an exact RAPTOR
native benchmark and no stability, recoverability or deployment guarantee is made.

## Task-specific external force setting

The v3 agile task uses the user's requested **no external force, retained initial
linear velocity** setting. With `--task agile`, constant external force and every
force-at-point pulse are disabled by default, including the corresponding pulse
torque. Gravity, rotor thrust/torque and motor dynamics remain unchanged.
Measurement Gaussians and 10-30 ms velocity latency remain active. The ordinary
hover task retains the force-enabled protocol documented below.

`--disable-external-forces` makes the force-only choice explicit for either task;
`--enable-external-forces` opts into the original force protocol. These choices
are mutually exclusive and are stored in the checkpoint binding, used by fixed
EVAL, exact resume and update replay. The force switch does not alter the sampled
aircraft, initial physical draws, sensor noise tapes or delay values for a fixed seed.
`--disable-pulses` only suppresses pulses; `--disable-disturbances` suppresses
all disturbances, including sensor noise and latency, and is therefore different
from the agile default.

Agile moves the sampled aircraft to point 1, sets angular velocity to zero and
uses static-equilibrium orientation/motors, while retaining each maneuver row's
originally nonzero world linear velocity. Originally zero-velocity guidance rows
receive a new independent seeded per-axis `Uniform[-2.5, 2.5] m/s` draw, matching
the original non-guidance range. This only fills initial velocity; aircraft,
physical parameters and sensor noise are not resampled. The mission report
includes `initial_zero_velocity_replaced_count`. All three previous-velocity
slots are initialized to the actual starting velocity. Only intentional
hover rows use a rest reset. Failed attempted maneuvers keep their initial
velocity even when their reference falls back to a stationary point. No initial
velocity is repeatedly injected during the rollout.

The nominal S9 reference still begins with zero velocity, so the retained initial
velocity is an intentional initial tracking error for the continuous loss. It
does not introduce another waypoint, recovery stage or success threshold. The
velocity need not point toward point 2 and can initially point out of the arena;
the planner's continuous reference containment does not guarantee recovery from
every such state. Original position termination remains unchanged, and rows are
not filtered or resampled according to recoverability. See
[the three-point mission guide](agile_flight.md) for geometry and time planning.

## Constant force and pulse scale

The pinned RAPTOR numerical law is retained without inserting an extra g:

```
r ~ Uniform(0, 0.3 * max(TWR - 1, 0))
sigma_F = r * TWR * mass / 3
F_const ~ Normal(0, sigma_F^2 I_3)
F_pulse[k] ~ Normal(0, sigma_F^2 I_3)
```

The scale and constant world-frame force are sampled at reset and held for the
episode. Each pulse independently resamples its three force components using the
same airframe sigma, NOT the realized constant force. Conditional on the sampled
sigma, event forces and the constant force are independent. Neither is clipped.
The Euclidean force magnitude is not itself Gaussian; the three signed components
are Gaussian. The scalar scale has a support ceiling of 10 in the source formula
at mass=5 kg, TWR=5 and the endpoint r=1.2; actual Gaussian force has no hard bound.
The division by three is a scale convention, not truncation at three sigma.

During a pulse the forces ADD: F_total = F_const + F_pulse. Between pulses,
F_total = F_const. This does not replace the persistent disturbance or reduce its
amplitude. There is no added independently sampled torque or motor-command error.

## Schedule, application point and units

The simulation uses dt=0.01 s. Every episode/scene has its own independent schedule:

- First onset: integer step Uniform{0,...,99}, i.e. 0.00 through 0.99 s.
- Subsequent onset-to-onset intervals: integer Uniform{80,...,120}, i.e. 0.8-1.2 s.
- Each rectangular pulse: exactly 10 transitions, i.e. 0.1 s.
- Every event samples a new force vector and a new application point.

Intervals are measured between pulse ONSETS, not from the end of a pulse. Thus
quiet gaps are 0.7-1.1 s. Events do not overlap. Pulse edges are on control-step
boundaries, and the force is held across all four RK4 stages of the transition.
A pulse beginning shortly before the horizon or a first-failure termination is
cut off by episode end. It is not moved earlier, lengthened or rescaled. H is an
observation-only sentinel, not an extra executed transition. No velocity or pose
is instantaneously overwritten: the finite force changes acceleration for 0.1 s.

The repo has rotor positions, not a fuselage collision mesh. The explicit minimal
body geometry is therefore four thin arms connecting the center of mass to the
rotors: choose one arm j uniformly, sample lambda ~ Uniform(0,1), and set
r_body = lambda * rotor_positions[j]. This is uniform along the four equal-length
arms, not uniform over a guessed volume or a square containing empty space. It
scales with each airframe without introducing a new length parameter. The body
point is fixed during each pulse; the force vector is fixed in world coordinates.

For the true body-to-world rotation R(q), each RK4 stage evaluates

```
F_world = F_const + F_pulse
F_body_pulse = R(q).T @ F_pulse
tau_body_pulse = cross(r_body, F_body_pulse)
v_dot = existing_rotor_acceleration + gravity + F_world / mass
omega_dot = (rotor_torque + external_torque + tau_body_pulse
             - cross(omega, J * omega)) / J
```

The legacy external_torque field stays zero in sampled scenes. Pulse torque is
computed from the force and lever, including yaw, rather than written into that
constant field. It is recomputed at EVERY RK4 stage with the true intermediate
attitude. It is never computed with the noisy Actor attitude, frozen at pulse
onset, or added a second time. Differentiation through R(q) is retained.

## Original hover recovery initialization

For the 90% non-guidance population:

| Quantity | Distribution |
|---|---|
| Position | Each axis Uniform[-10*arm_length, +10*arm_length], unchanged |
| Orientation | Uniform random axis; rotation angle Uniform[0,120 degrees] |
| World linear velocity | Each axis Uniform[-2.5,+2.5] m/s |
| Body angular velocity | Each axis Uniform[-2.2,+2.2] rad/s |
| Motor coordinate | Each rotor Uniform[0,0.5], unchanged |

The limits on velocity and angular velocity are PER AXIS, not vector norms.
The 10% guidance draw still zeros position, velocities and attitude, but not
motor state. Initial previous_action is a fixed zero placeholder, never a
re-encoding of true motor state. All three prior velocity slots equal reset
velocity; their original measurement noise is reused at startup.

Position-only first failure is strict per-axis exceedance of 30*arm_length.
These hover initial conditions and loss settings are unchanged by the agile
force switch; agile's position/orientation/motor reset is described above. Some initial
conditions/pulses may be unrecoverable. Numerical/gradient tests
cannot establish trained survival under this distribution.

## Measurement protocol retained

Position, world velocity, attitude rotation vector and body angular velocity
have independent zero-mean acquired-sample Gaussians with per-axis sigma
0.001 m, 0.002 m/s, 0.001 rad and 0.002 rad/s, respectively.
R_observed = R_true @ exp(skew(delta_theta)) remains on SO(3).
The scales are L2F-inspired modeling choices; rotation-vector sigma is NOT the
same covariance as independently corrupting nine rotation-matrix entries.

Velocity latency is sampled once per episode from Uniform[0.010,0.030] seconds.
Current plus three previous world-frame velocities, and each sample's ORIGINAL
noise, support causal fractional interpolation. Pre-reset history holds the first
measurement. Only velocity is delayed. Current measured attitude converts the
result to body features. Other sensor values and known command remain current.
These fixed latency bounds are a modeling choice motivated by RAPTOR's 10-30 ms
sensitivity diagnosis, not an identified distribution of the user's hardware.

## Actor isolation and differentiation

Actor inputs remain exactly the same 22 values: measured position, delayed
measured world velocity, measured rotation matrix, measured angular velocity,
and last known command. It never receives pulse force, application point,
on/off mask, schedule, pulse countdown, sigma_F, force truth, current undelayed
velocity as a replacement, physical parameters, or RNG state. GRU memory starts
at zero and is driven only by these permitted measurements and commands.
Observing motion AFTER a pulse is legitimate feedback, not privileged leakage.

Both pulse tapes are immutable original-pool tensors, shared under stable
noise_row IDs when live rows compact. No random numbers are drawn by step or
observe. Separate CPU RNG streams leave constant force, sensors, latency and
initial-state draws unchanged when only pulses are disabled. Disabling all
external forces also preserves sensor, latency and initial-state draws. Pulse sampling is
independent of Actor quality, reward or survival. Exogenous tape values need no
Actor gradient; physical, recurrent and measurement-history paths remain
connected, including the posture-dependent pulse torque and 50-step boundaries.
Task loss, termination and reported flight metrics use physical truth.

TRAIN draws new banks at each update; EVAL reuses one fixed bank. Checkpoint
source/config/environment/noise bindings cover pulses, their implementation and
the new reset version. Existing checkpoint schema is retained because its format
is unchanged; environment/noise versions are bumped. Old environment checkpoints
cannot resume or be silently rescored here. Compatible weights-only import is
explicit and starts fresh Adam. Use a new work directory.
`--disable-pulses` exists for regression, not as a second evaluator; it leaves
harder resets and other noise unchanged. `--disable-disturbances` turns off all
noise/pulses for physical unit tests but does not undo harder initialization.
The separate force-only setting permits agile's requested training distribution
without using this sensor-clean regression mode.

## Source provenance and evidence

RAPTOR pins rl-tools at e43ae4bcda4556321a63f4eb5dcc826cd637aa39. Its
`src/foundation_policy/pre_training/sample_dynamics_parameters.cpp` sets 0.3;
`include/rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h`
computes sigma; `30_sample_initial_state.h` draws the Gaussian constant force;
`70_post_integration.h` copies it unchanged. Its Langevin process changes the
reference trajectory, NOT the external force. The pulses here are a requested
DiffPhys extension, not a rediscovered RAPTOR training feature.

The intentionally changed RK4 right-hand side has an explicit old/new hash entry
in tests/core_contract.json; other protected mathematical kernel hashes are not
regenerated. Independent NumPy RK4, analytic center-of-mass impulse, off-center
force/attitude/action finite differences, pulse timing, original-row selection,
Actor noninterference, pre-pulse causality, first-failure freeze, grouped BPTT,
metric-boundary continuity and checkpoint replay are regression requirements.
No automatic safety authorization, full-training result or GPU-speed claim follows
from these tests. Checkpoints remain deployment_authorized=false.
