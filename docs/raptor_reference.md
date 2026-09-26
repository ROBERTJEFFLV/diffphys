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

The sole runtime simulator is now `env_raptor.py`. The L2F fixed-airframe profile,
profile selector and training entry were removed, not merely disabled. References
to L2F here document inherited physics; they do not expose another training mode.

The retained multi-airframe distribution samples mass by cubing a uniform sample
between cbrt(.02) and cbrt(5) kg^(1/3), thrust/weight uniformly in [1.5,5],
torque/inertia in [40,1200], rotor moment coefficient in [.005,.05], rise time in
[.03,.10] s and fall time in [.03,.30] s. The four rotors share the sampled moment
coefficient and rise/fall times. Thrust curves, geometry and inertia are coupled
by the upstream sampling formulas, not independently randomized without units.
The geometry multiplier uses the source Normal(-.1,.1) reciprocal transform.

The base normalized-motor thrust polynomial is (.00352526,.01437313,.09223048).
Its scale is chosen to match sampled thrust/weight. Joint RK4 integrates position,
world velocity, Hamilton-wxyz quaternion, body angular velocity and actual motor
states, using asymmetric first-order motor time constants inside all four stages.
The quaternion is normalized after the step; source numerical state bounds and
motor saturation are retained. Actions are absolute [-1,1] commands, not
hover-centered residuals. Axis convention is FLU, X-frame FR/BR/BL/FL.

Deliberate differences already present in parent source c15ca418 are preserved:
random-axis attitude angles are uniform up to pi/2 (paper-first), rather than
copying the pinned upstream sampling bug that ignores the angle limit; termination
is position-only per-axis strict exceedance, not the original paper's additional
velocity/omega thresholds. Initial position bounds are 10 times rotor radius and
termination bounds 20 times radius. Initial velocities/omegas are sampled in
[-1,1]; 10% guidance scenes zero the kinematics. Initial motor coordinates are
uniform in [0,.5], independent of the guidance override.

The current disturbance distribution is an intentional change: source unbounded
constant force is replaced, four requested uncertainty mechanisms are enabled,
and all share a bounded budget. See `disturbance_budget.md` for the equations and
the explicit limits of the certificate. Sensor attitude is now an SO(3) error,
not additive independent rotation-matrix elements. Hidden execution error does
not overwrite the known previous command. The Actor, true dynamics equations,
physical-group normalization, task/Huber/CVaR loss, Time Decay rule, Adam and
failure handling are retained. `tests/core_contract.json` checks unchanged
mathematical kernels against parent source, while independent numerical tests
cover the new interfaces.

Removing source-specific random draws changes the old sampler's exact RNG
sequence. New same-seed runs are internally deterministic and noise-independent
for airframe/initial-state sampling, but are not claimed to reproduce old
same-seed banks bit for bit. Old checkpoint resume/evaluation require their original source and
are rejected by this protocol rather than silently migrated. The explicit
weights-only `--init-checkpoint` can reuse an interface-compatible Actor, with
fresh Adam and the new environment; it restores no legacy runtime.

Historical material in `reference/`, `物理配置/` and `docs/images/` is retained as
provenance, not imported or invoked by the training chain. Other Git branches are
untouched. Former runtime code and documentation remain in Git history.
