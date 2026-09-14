# Reference environment and motor contract (v3)

This is the current environment contract. It supersedes the scene, motor and
checkpoint semantics in `response_control_v1.md`. No historical runs are relabeled.

## Source pins and precedence

- RAPTOR paper: Eschmann et al., *RAPTOR: A foundation policy for quadrotor control*,
  Science Robotics 11, eaec1481 (2026), Materials and Methods, printed page 11.
  DOI: https://doi.org/10.1126/scirobotics.aec1481
- RAPTOR repository: https://github.com/rl-tools/raptor/tree/2c789dfcf16cc96fe697704492b3bf79dd2cc5a0
- Its **pinned** implementation: https://github.com/rl-tools/rl-tools/tree/e43ae4bcda4556321a63f4eb5dcc826cd637aa39
- 2024 L2F baseline: https://github.com/arplaboratory/learning-to-fly/tree/d07592d5c5dea3c90954d2be6f04cfa68581ebe8

Use the RAPTOR **paper** where initialization conflicts with current source.
The source's `sample_orientation(limit,...)` ignores `limit`; v3 explicitly uses
pi/2 instead of copying the accidental 1-radian limit. The posttraining helper
restores a legacy +/-0.5 m initialization; v3 instead uses the paper's +/-10*l_arm.
There is no parameter whose value is merely logged but ignored.

The paper states 10% target-state guidance. In the RAPTOR profile this zeroes
position, velocity and body rate and sets the quaternion to identity. The paper
does not give a universal motor target value; rotor states and action-history
initialization follow the source separately, not a manufactured hover command.

## Profiles

### `raptor`: multi-airframe

Initialization, per axis: position U[-10*l_arm,10*l_arm], velocity U[-1,1] m/s,
body rate U[-1,1] rad/s. The rotation axis is uniform on the sphere; the rotation
angle is uniform in [0,pi/2]. Kinematic guidance probability is 0.1. The center-to-
rotor distance is `l_arm`, NOT the x/y coordinate of a rotor in an X frame.

Failure limits: position +/-20*l_arm, velocity +/-2 m/s, body rate +/-35 rad/s;
strict `>` comparisons. No fixed 1 m box is applied to every airframe.

The source distribution is translated from
`src/foundation_policy/pre_training/sample_dynamics_parameters.cpp` and
`include/rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h`:

- Sample cube-root mass uniformly between cube-root(0.02) and cube-root(5.0),
  then cube it. Sample thrust-to-weight uniformly in [1.5,5.0].
- Scale the source Crazyflie coefficients `(0.00352526,0.01437313,0.09223048)`
  together so that max total thrust equals TWR*m*g. Motor states are in [0,1].
- Sample the upstream torque-to-inertia root in [40,1200], preserve the base
  inertia ratios, and apply the source mass/size deviation construction.
  The source uses Normal(mean=-0.1,std=0.1) followed by a reciprocal mapping;
  this is preserved, not replaced by a bounded uniform deviation.
- Independently sample rising motor delay in [0.03,0.10] s and falling delay
  in [0.03,0.30] s. Falling may be faster than rising. All four motors on one
  sampled airframe share these constants, as in the default generator.
- Sample torque coefficient in [0.005,0.05]. No old alpha/eta clipping or
  4x4 authority stratification changes the resulting distribution.
- Force standard deviation is `U(0,0.3*(TWR-1))*TWR*m/3`, followed by independent
  zero-mean Gaussian forces per world axis. This is the source formula, including
  its absence of an extra gravity factor. Force remains constant per episode.
- Source-default observation noise and external-torque standard deviations are
  zero. A configuration boolean named OBSERVATION_NOISE is not evidence of a
  nonzero numerical standard deviation. Historical archived JSON files could
  differ; this port targets the checked source generator, not an unverified archive.
- Rotor state initialization is independently uniform in [0,0.5], equivalent to
  normalized action range [-1,0]. Initial action history is mapped from those
  rotor states, as in `30_sample_initial_state.h`.

The source nominal mass is 0.0306 kg, rotor coordinates are (+/-.028,+/-.028,0),
and principal inertia is `(9.416556729130406e-6, 9.644051701582312e-6,
1.745951732253285e-5)` kg*m^2. The base body frame and axis inertia ratios are
retained while scaling. The sampled torque/inertia root is not the same numeric
quantity as a two-rotor maximum roll acceleration; do not silently equate them.

### `l2f`: single Crazyflie baseline

This profile uses the 2024 source defaults, **not** the newer RAPTOR nominal model:

- Mass 0.027 kg; X rotor coordinates (+/-.028,+/-.028,0) m.
- J=(3.85e-6,3.85e-6,5.9675e-6) kg*m^2; torque coefficient 0.005964552.
- Physical RPM in [0,21702], thrust `3.16e-10*RPM^2`, time constant 0.15 s.
- Initial RPM 10851 on all rotors; normalized action history zero. This produces
  about 56% of body weight, NOT hover. No equilibrium recentering is applied.
- Initial position each axis +/-0.2 m; velocity and body rate each axis +/-1.
  Uniform quaternions are rejected outside a 90-degree rotation ball. Exact pi/2
  is used for 90 degrees rather than the source's rounded `3.14/2`.
  The original 10% guidance branch zeroes position/attitude only; velocity and
  body rate remain randomized, unlike the RAPTOR paper protocol.
- Failure boundaries: +/-0.6 m, +/-1000 m/s, +/-1000 rad/s.
- Independent Gaussian observation noise: position .001 m, velocity .002 m/s,
  **each of the nine rotation-matrix entries** .001, body rate .002 rad/s.
  Rotation-entry noise is not an angle in radians and is never applied to truth.
- Constant episode force std `.027*9.81/20` N and torque std `.027*9.81/10000` Nm
  on each axis; world force and body torque. No extra environment action noise.

Source: `src/config/parameters.h`, `simulator/parameters/init/default.h`,
`simulator/parameters/dynamics/crazy_flie.h`, and `simulator/operations_generic.h`
under `include/learning_to_fly/` where applicable.

## Native motor and integration semantics

Actor output `a` is clipped to [-1,1], then mapped to the source motor input:
`u = u_min + (a+1)/2*(u_max-u_min)`.
The lagged motor state obeys `dm/dt=(u-m)/tau`, with rising/falling tau selected
at each RK4 stage. Thrust is `c0+c1*m+c2*m^2`. No per-airframe inverse-thrust map,
hover offset, mass-dependent command centering or torque-authority normalization
is inserted between Actor output and the motors.

FLU X ordering is `[front-right,back-right,back-left,front-left]` with spin signs
`[-,+,-,+]`. Body torque is computed from `r x F` and the source yaw coefficients.
The state is advanced by joint RK4 of position, velocity, Hamilton quaternion,
body rate and motors, with quaternion normalization and motor-range clipping
only after the RK4 step. RAPTOR numerical p/v/omega clamps +/-100000 are distinct
from its much smaller episode boundaries. The source quaternion double-cross
rotation and Newton-Euler gyroscopic term are retained. No drag/map/collision
physics has been added under the reference label.

## Noise, adjoints and evaluation

A local CPU generator creates each bank without changing optimizer RNG. Noise
is pre-sampled for H+1 observations; true physical states and loss targets remain
noise-free. The tape is indexed by the integer physical step, so re-observation
of a window boundary uses exactly the same sample. Reverse-window recomputation
has no fresh random draws and shares the immutable tape instead of cloning it
at every boundary. Noise/force samples are fixed in paired EVAL, independent of
TRAIN batch size. Float32/float64 and device placement are preserved.

Only the matching reference metric prefix is published. `l2f_*` cannot be
computed on a RAPTOR bank with the wrong box. A bank starting outside its own
reference limits is rejected. Reference episode length counts transitions up
to and including the first failure. TRAIN and EVAL stop each aircraft there;
other aircraft continue to their own first failure or the horizon cap. No failed
row is reset or passed back into Actor/RK4. Frozen storage padding is marked by
`TaskTrajectory.valid` and excluded from costs and statistics. BPTT remains
complete on each executed prefix; reverse windows verify the same validity mask.
L2F's 200 mm settling statistic retains its source final-distance convention.

## Checkpoints and remaining experimental differences

Schema `response-actor-only-reference-v3` and architecture
`response-conditioned-absolute-motor-policy-v3` intentionally reject older
checkpoints at **all** public load paths, including weights-only initialization.
The environment contract and source hash are bound separately, so metric-only
rescoring cannot quietly swap a different physical environment. Exact resume
still requires full source/config identity, named Adam state and RNG. EVAL count
is independently bound through `--eval-scenarios`.

This revision implements the five requested environment/motor changes. It does
not replace Actor-only BPTT with teacher distillation or silently change the
Huber/CVaR terms or weights. Only post-termination padding is excluded; the
original fixed-horizon normalization and steady window remain. Failure penalties
and other loss changes are deferred for discussion, so short failures may still
look artificially cheap under this interim objective. No long training is
validated by this termination-only change. Absolute-yaw objectives, Langevin reference tracking,
1000 frozen published teacher airframes and the seven held-out real models are
not reproduced. The current control features are still the original response-
conditioned features, not RAPTOR's network. Matching these environment conditions
therefore does not by itself establish full task/experiment equivalence or equal
adaptation performance. In particular, yaw recovery requires separate objective
and observation work before a zero-yaw performance claim.

The Python and C++ RNG streams are not byte-identical; distribution equality is
not equality of the original 1000 sampled parameter files. Fresh TRAIN airframes
are drawn per update, while fixed EVAL seeds are held out. No archived datasets
or real flight results have been relabeled, and no deployment is authorized.

## Verification

`tests/test_reference_environment.py` checks initial-state support and guidance,
per-airframe limits, X torque signs, non-universal hover, parameter relationships,
noise statistics and RNG isolation, independent NumPy RK4 agreement, finite-
difference action Jacobians, full/windowed BPTT and Adam agreement, and metric
failure semantics. `tests/test_reference_training.py` checks bounded updates,
exact resume, separate EVAL count, stored-protocol evaluation, legacy-checkpoint
rejection and failure rollback. These are correctness checks, not performance
experiments. CUDA throughput and real-flight adaptation require separate tests.
