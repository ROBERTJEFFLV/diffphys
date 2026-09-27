# Physics provenance and deliberate differences

Upstream references retained for reproducibility:

- `arplaboratory/learning-to-fly` at `d07592d5c5dea3c90954d2be6f04cfa68581ebe8`.
- `rl-tools/rl-tools` at `e43ae4bcda4556321a63f4eb5dcc826cd637aa39`, pinned by
  `rl-tools/raptor` at `2c789dfcf16cc96fe697704492b3bf79dd2cc5a0`.

Relevant files are `src/foundation_policy/pre_training/sample_dynamics_parameters.cpp`,
`include/rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h`,
`30_sample_initial_state.h`, `40_observe.h`, `60_dynamics.h`, `70_post_integration.h`,
`parameters/default.h`, and the Crazyflie dynamics registry. Attribution remains
in THIRD_PARTY_NOTICES.md. References do not expose another runtime simulator.

## Retained dynamics

Mass is sampled by cubing a uniform draw between cbrt(.02) and cbrt(5) kg^(1/3);
TWR is uniform in [1.5,5], torque/inertia in [40,1200], rotor moment coefficient
in [.005,.05], rise time in [.03,.10] s and fall time in [.03,.30] s. All four
rotors share the sampled coefficient and rise/fall times. Geometry, thrust curves
and inertia are physically coupled using the upstream formulas. The size
multiplier uses the source Normal(-.1,.1) reciprocal transform.

The base normalized-motor thrust polynomial is (.00352526,.01437313,.09223048).
It is scaled to the sampled TWR. Joint RK4 integrates position, world velocity,
Hamilton-wxyz quaternion, body angular velocity and actual motor states. Motor
time constants are asymmetric within all RK4 stages; quaternion normalization,
numerical state bounds and motor saturation are retained. Actions are absolute
[-1,1], not hover residuals. Axis convention is FLU, X-frame FR/BR/BL/FL.

## Constant force restored in b134dff

The numerical RAPTOR law is `r~U(0,.3*max(TWR-1,0))`, then
`sigma_F=r*TWR*mass/3`, then independent Gaussian world force components.
There is no extra g factor or clipping; division by 3 is not a hard bound.
The force is sampled at reset and remains constant. This replaced the earlier
shared bounded disturbance budget. Position/velocity/SO(3)-attitude/omega noise
with sigmas .001/.002/.001/.002 and Uniform[10,30] ms velocity measurement latency
are DiffPhys additions. They are not a literal RAPTOR observation protocol.
Reset command history is zero, not derived from hidden motor state.

## Current requested recovery extension

The original RAPTOR paper's 90-degree initialization is now deliberately expanded
to a uniform random-axis angle in [0,120] degrees. Per-axis linear velocity is
[-2.5,2.5] m/s and angular velocity [-2.2,2.2] rad/s. Initial positions remain
within 10 times rotor radius per axis; 10% guidance scenes zero the kinematics.
Motor coordinates stay uniform in [0,.5], independent of guidance. This still
uses the random-axis sampling mechanism rather than reproducing the pinned
upstream angle-limit bug. No initial-state filtering is introduced.

There are now 0.1-second rectangular force pulses with a per-scene random first
phase in steps 0..99 and later onset intervals of 80..120 steps at 100 Hz. A new
Gaussian vector with the SAME airframe sigma_F and a new point along a randomly
chosen COM-to-rotor arm are drawn for each event. A pulse adds world force and
body torque `r_body x (R_true.T @ F_world)` at every RK4 stage. The constant force
remains present. The body point and world force are held during the event, not
the resulting torque. These pulses are a DiffPhys extension requested after the
RAPTOR audit, NOT part of the audited upstream training method. The upstream
Langevin reference trajectory is not an external-force process.

Position-only per-axis strict first-failure termination is retained, at 20 times
rotor radius. It differs from the original paper's additional velocity/omega
thresholds. Task/Huber/CVaR losses, Time Decay, physical-group normalization and
Adam are retained. Only the protected dynamics RHS hash is intentionally updated,
with provenance; independent force-at-point numerical/gradient tests cover it.

## Scope and compatibility

The only production simulator is env_raptor.py. No single-airframe L2F mode,
critic, teacher, alternate optimizer, independent torque-noise process or additive
command noise is introduced. Omitting these uncertainties does not mean dynamics
randomization mathematically subsumes them. Existing motor dynamics are unchanged.

Noise/pulse sampling is separate from initial-state RNG. Within the new version,
enabling pulses does not change the other random draws for matched seeds. New
reset velocity/angle values intentionally differ from the former version. C++
RAPTOR and Python are not claimed to produce bitwise-identical seeded samples.
Pulse/sensor tapes use stable original-pool row IDs, stay immutable, and never
become Actor inputs. Costs and boundaries use truth. The Actor architecture and
its permitted observation schema are unchanged. See disturbance_budget.md for
timestamp semantics, exact schedule, geometry approximation and limitations.

Environment/noise versions and source hashes reject old exact resume/evaluation.
An explicit interface-compatible --init-checkpoint imports only Actor weights,
with fresh Adam and new sampling. Neither tests nor a mathematical noise scale
prove convergence, recovery from all samples, or real-flight safety.
Historical reference material, physics configuration, licenses and images are
retained; removed code remains in Git history. Other branches are not rewritten.
