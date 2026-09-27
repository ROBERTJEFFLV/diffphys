# Gaussian measurement and RAPTOR force protocol

The filename is retained for existing links. The former joint <=10% uniform
budget is **removed**. This document describes the current, single TRAIN/EVAL
protocol, not a claim about a learned region of attraction.

## Final configuration and time semantics

At 100 Hz, independent acquired-sample Gaussian standard deviations are:

| Signal | sigma per scalar axis | Coordinates |
|---|---:|---|
| Position | 0.001 m | World |
| Linear velocity | 0.002 m/s | World, then delayed |
| Attitude | 0.001 rad | Three-dimensional rotation vector |
| Angular velocity | 0.002 rad/s | Body |

Each episode also samples a constant external world-force vector using the pinned
RAPTOR formula and a constant velocity latency from Uniform(0.010,0.030) seconds.
No external-torque noise, motor-command noise or transient-force schedule is enabled.
The existing external_torque state field stays zero to preserve the rigid-body kernel.

These sigmas are **discrete 100 Hz measurement values**, not noise densities.
The L2F code reference is `arplaboratory/learning-to-fly` at
`d07592d5c5dea3c90954d2be6f04cfa68581ebe8`, `src/config/parameters.h`:
position/orientation/linear_velocity/angular_velocity scales are
0.001/0.001/0.002/0.002. Its `OrientationRotationMatrix` observation adds noise to
nine matrix entries. Our SO(3) model is an intentional modeling choice, not an
exact reproduction of that covariance or an empirically calibrated sensor model.

RAPTOR reports that 10-30 ms delayed linear velocity reproduces certain z-axis
oscillations. That diagnosis does not establish a uniform latency law or prove
that training on it is stable. The uniform range is an explicit experiment design.
No 7-sigma/10%-reserve safety interpretation remains.

## External force: source-faithful numerical law

Let m be the sampled mass in kg and TWR its thrust-to-weight ratio. Draw

```
r ~ Uniform(0, 0.3 * max(TWR - 1, 0))
sigma_F = r * TWR * m / 3
F_ext,j ~ Normal(0, sigma_F^2), independently for j=x,y,z.
```

Keep F_ext constant during the episode and add F_ext/m to the world acceleration.
The source formula does not multiply by gravity; this implementation deliberately
copies its numerical convention instead of silently correcting or reinterpreting it.
For m<=5 and TWR<=5, sigma_F<=10 in the simulator's force units. This bounds sigma,
**not samples**. There is no clipping, rejection of large finite forces, or guarantee
that every sampled combination is recoverable. Division by 3 does not truncate a
Gaussian. Source-faithful distribution does not mean identical C++/PyTorch RNG bits.

Official pin: `rl-tools/rl-tools@e43ae4bcda4556321a63f4eb5dcc826cd637aa39`, referenced
by the `rl-tools/raptor` submodule. Source locations:

- `src/foundation_policy/pre_training/sample_dynamics_parameters.cpp`: sets
  `disturbance_force_max=0.3` and the airframe support.
- `include/rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h`:
  samples the multiplier and computes `multiplier*TWR*mass/3`.
- `.../30_sample_initial_state.h`: samples three Gaussian force components.
- `.../60_dynamics.h`: applies `state.force / mass`.
- `.../70_post_integration.h`: copies each force component to the next state.

RAPTOR first creates a finite teacher/airframe population. DiffPhys continues to
resample multi-airframe training banks, so matching the conditional force law is
not a claim to reproduce RAPTOR's teacher population, training algorithm or logs.

## Causal measurement model

For each acquired control-time sample:

```
p_m(t) = p(t) + eps_p(t)
v_m(t) = v(t) + eps_v(t)
R_m(t) = R(t) @ exp(skew(delta_theta(t)))
omega_m(t) = omega(t) + eps_omega(t)
```

Only the velocity measurement is delayed. For d in [0,0.030] seconds, dt=0.010,
let q=d/dt, k=floor(q), j=min(k+1,3), lambda=q-k:

```
v_obs(t) = (1-lambda)*v_m(t-k) + lambda*v_m(t-j)
```

The samples are current plus three previous acquired world-frame velocities.
Exactly 30 ms uses slot 3; it does not access a fifth sample. All negative-time
history slots hold the first acquired measurement, including the same first
noise sample. Old noise is never redrawn. Current noisy attitude converts the
delayed world vector to body coordinates only inside the unchanged Actor features.

For fractional latency, interpolation also filters the white measurement noise:
when the two samples are distinct, variance is
`((1-lambda)^2 + lambda^2) * sigma_v^2`. Do not add compensating fresh noise;
0.002 m/s is the acquisition sigma, not a promised post-interpolation sigma.
The held pre-reset samples are correlated by construction. This discrete model
is not a complete sensor/firmware simulation or an ideal continuous delay.

The state field `previous_velocity` stores true world velocities with shape
[N,3,3], ordered t-1,t-2,t-3; original acquisition noise is looked up from the
immutable tape. Their sum is the stored-measurement equivalent. All history
transitions stay differentiable and use the existing per-control-step Time Decay.
No detach occurs at 50-step metric boundaries or when selecting live rows.

## Actor information boundary

The observation is assembled by an explicit whitelist, not by flattening state:

```
[p_measured(3), v_delayed_measured(3), R_measured(9),
 omega_measured(3), previous_known_command(4)]
```

The Actor never receives motor truth, force, airframe parameters, noise samples,
noise standard deviations, latency, group IDs, boundary limits or future tape
entries as features. It may infer dynamics from legitimate past observations and
its own commands. Gradients through the simulator during training do not create
a privileged input at deployment.

A reset leak is fixed: `previous_action=2*motor-1` exposed sampled motor truth.
Now the initial known-command field is a fixed zero placeholder independent of
motor state. It is not the actual motor speed or a computed hover command. After
the first action, it stores the command actually issued by the Actor. Deployment
must initialize the same placeholder when no pre-activation command is available.
The first action-change penalty intentionally uses that placeholder too.

`response_policy.py` is unchanged. Loss, terminal masks, risk metrics and failure
costs still use true state. Physical grouping uses simulator parameters **only in
the training gradient aggregation**, not in the policy observation.

## Reproducibility and training integration

Force, sensor and delay draws use separate CPU generators; airframe/initial-state
sampling has its own stream. Changing a sensor sigma does not change sampled
force, latency or airframe. The source RNG sequence differs from the previous
uniform-budget protocol. Tapes are sampled at reset and shared with stable row
IDs, including across first-failure compaction. No random draw occurs in observe
or step. Terminated states, history and noise indices freeze.

TRAIN and the one fixed EVAL call the same sampler and DisturbanceConfig. TRAIN
bank seeds change by update; EVAL seeds and all tapes stay fixed. EVAL uses saved
checkpoint settings. The zero-noise switch is for numerical regressions, not a
second evaluation benchmark. New environment/noise versions reject old exact
resume/evaluation. Explicit weights-only import is permitted for interface-compatible
Actors but restores no optimizer, simulator or old sampler.

Gaussian tails and delayed feedback can still cause nonfinite gradients or failed
recoveries. Existing finite checks, global clipping, physical-group normalization
and transactional Adam remain; they are numerical protections, not flight guarantees.
No new objective, curriculum, hidden safety controller or loss-based update veto is added.

## Audit of transient forces (no additional mechanism added)

The official pinned pretraining environment uses `StateRandomForce`; force is
sampled at reset and copied unchanged by `70_post_integration.h`. The posttraining
`helper.h` gathers ordinary simulator trajectories via `evaluate` and labels their
observations; it does not inject time-scheduled impulses. The Langevin state in
`70_post_integration.h` is a moving **reference**, not a force process. No pulse
schedule was found in this audited official training path. A caller can externally
modify state.force, but that capability alone is not evidence of training with it.

RAPTOR's real-world tool hits, fan disturbances and payload demonstrations test
its learned policy. We do not translate these demos into an invented training
pulse distribution. Transient disturbances are deliberately left for a separate
design discussion, as requested.

This protocol also does not model bias/drift, packet loss, time-varying wind,
unequal motor faults or full firmware latency. Omitting torque and command noise
is a scope decision, not a claim they are equivalent to dynamics randomization.
