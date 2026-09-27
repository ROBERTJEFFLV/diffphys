# Physics provenance and deliberate differences

Upstream references retained for reproducibility:

- `arplaboratory/learning-to-fly` at `d07592d5c5dea3c90954d2be6f04cfa68581ebe8`.
- `rl-tools/rl-tools` at `e43ae4bcda4556321a63f4eb5dcc826cd637aa39`, pinned by
  `rl-tools/raptor` at `2c789dfcf16cc96fe697704492b3bf79dd2cc5a0`.

Relevant upstream files are `src/foundation_policy/pre_training/sample_dynamics_parameters.cpp`,
`include/rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h`,
`30_sample_initial_state.h`, `40_observe.h`, `60_dynamics.h`,
`70_post_integration.h`, `parameters/default.h`, and the Crazyflie dynamics
registry. License attribution remains in `THIRD_PARTY_NOTICES.md`.

The sole runtime simulator is `env_raptor.py`. The L2F fixed-airframe profile,
profile selector and training entry remain removed. References to L2F document
inherited physics; they do not expose another training mode.

The multi-airframe distribution samples mass by cubing a uniform sample between
cbrt(.02) and cbrt(5) kg^(1/3), thrust/weight uniformly in [1.5,5], torque/inertia
in [40,1200], rotor moment coefficient in [.005,.05], rise time in [.03,.10] s and
fall time in [.03,.30] s. The four rotors share the sampled moment coefficient and
rise/fall times. Thrust curves, geometry and inertia are coupled by the upstream
sampling formulas. The geometry multiplier uses the source Normal(-.1,.1)
reciprocal transform.

The base normalized-motor thrust polynomial is (.00352526,.01437313,.09223048).
Its scale matches sampled thrust/weight. Joint RK4 integrates position, world
velocity, Hamilton-wxyz quaternion, body angular velocity and actual motor states,
with asymmetric first-order motor time constants inside all four stages.
Quaternion normalization, numerical state bounds and motor saturation are retained.
Actions are absolute [-1,1] commands, not hover-centered residuals. Axis convention
is FLU, X-frame FR/BR/BL/FL.

Pre-existing deliberate differences from the pinned upstream source remain:
random-axis attitude angles are uniform up to pi/2 (paper-first), rather than
copying the sampling bug that ignores the angle limit; termination is position-only
per-axis strict exceedance, not the paper's additional velocity/omega thresholds.
Initial position bounds are 10 times rotor radius and termination bounds 20 times
radius. Initial velocities/omegas are sampled per axis in [-1,1]; 10% guidance
scenes zero the kinematics. Initial motor coordinates are uniform in [0,.5],
independent of the guidance override.

The Gaussian protocol restores RAPTOR's conditional external-force law:
`r~U[0,.3*(TWR-1)]`, `sigma_F=r*TWR*m/3`, independent normal force components,
constant within an episode. There is deliberately no extra gravity factor or
force clipping. This is a numerical source convention, not a physical-unit
reinterpretation or a guarantee of recoverable initial states.

Independent Gaussian position/velocity/attitude/gyro measurements and uniform
10-30 ms velocity latency are **our extensions**, not RAPTOR-native settings.
The attitude perturbation is on SO(3); L2F's same numeric orientation-noise scale
uses matrix entries and is not distributionally identical. Random torque and
additive motor-command error are disabled. The former shared bounded budget and
its certificate are removed. See `disturbance_budget.md` for full definitions.

Initial command history is now a fixed zero placeholder: encoding random actual
motor state as `2*motor-1` leaked privileged information. Later history contains
only issued commands. This intentionally changes the reset observation and first
action-change reference, without changing the Actor or task formula. No force,
latency, noise scale, motor truth or airframe parameters are exposed to the Actor.

No training-time transient force schedule was found in the pinned RAPTOR reset,
RHS, post-integration and posttraining data-gathering path. Its Langevin process
changes reference trajectories. Tool hits/fan demonstrations are real-world tests,
not evidence of an implemented training pulse distribution. No pulse mechanism
is added here.

The Actor, rigid-body kernels, physical-group normalization, task/Huber/CVaR loss,
Time Decay rule, Adam and failure handling remain. `tests/core_contract.json` is
unchanged; independent numerical tests cover new history and sensing interfaces.

Independent RNG streams keep airframe/initial-kinematic samples stable when noise
settings change. These new runs are internally deterministic, not bitwise matches
to upstream C++ or old disturbance draws. Old checkpoint resume/evaluation require
their original source and are rejected by this protocol. Explicit weights-only
`--init-checkpoint` may reuse an interface-compatible Actor with fresh Adam and
the new environment, but it is not from-scratch training or exact continuation.

Historical `reference/`, `物理配置/` and `docs/images/` assets are retained as
provenance, not imported or invoked by training. Former runtime code and
documentation remain in Git history.
