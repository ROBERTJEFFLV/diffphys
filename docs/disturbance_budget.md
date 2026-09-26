# Joint bounded disturbance protocol

This is the implemented protocol, not an experimental roadmap. All equations
below describe `response_noise.py` and the sole multi-airframe training path.
The normalizer is **two-sided static hover actuator reserve plus an explicitly
specified reference error model**. No region of attraction of the learned
recurrent Actor has been established. Consequently “10%” must not be described
as 10% of a proven closed-loop recovery basin or a deployment safety guarantee.

## Classification and percentage sampling

| Physics-dependent per aircraft | Common across aircraft in the same pool |
|---|---|
| Episode-constant world force | Position measurement error, metres |
| Episode-constant body torque, three distinct axes | World-velocity measurement error, m/s |
| Normalized motor-command execution error | SO(3) attitude measurement error, radians |
| | Body angular-velocity measurement error, rad/s |
| | Velocity observation delay, seconds |

Command normalization does not remove differing thrust-curve derivatives and
actuator margins. A fixed action-noise amplitude cannot certify the same relative
physical load for all aircraft. This corrects the earlier proposed fixed action
sigma. Likewise, a single torque-to-inertia scalar does not certify yaw authority.

The pool has five weights for severity levels `0, .25, .50, .75, 1`. The weights
are normalized by `torch.multinomial`. For each episode, sample a level `s`, set
`b = s * disturbance_budget`, and sample a simplex `w` over the eight components
(force, torque, action, position, velocity, attitude, omega, delay). Implement the
simplex by independent exponential samples `-log(U)` divided by their sum.
Component fractions are `rho_k = b*w_k`; hence `sum(rho_k) <= .10` deterministically.
A small floating-point safety margin is applied before casting to FP32.
Default equal pool weights yield 20% fully clean episodes and 20% at each other
**total** budget. Pool membership is not a physical group and is not exposed to
the policy; every physical group sees the same sampling law.

Both classes use the same function:

`sample_bounded(bound, generator, steps=None) = bound * Uniform[-1,1]`.

The bound is an SI or command-space capacity multiplied by its normalized
fraction. Force and torque have one vector per episode. Measurement and action
errors have one bounded vector per timestep. Delay is a nonnegative fraction of
its capacity and stays fixed during an episode. There is no Gaussian tail, jitter,
dropout, hidden bias process, different simulator, or additional training branch.
Airframe randomness and nuisance randomness use independent local CPU generators.
For reproducible comparisons, the same bank seeds preserve airframe/initial-state
samples when the disturbance configuration changes.

## Exact actuator reserve

For rotor i, write thrust as `f_i(m) = c0_i + c1_i*m + c2_i*m^2`, where the retained
RAPTOR motor coordinate is [0,1]. Nonzero minimum thrust c0 is included. Let B map
rotor thrust to collective force and body torque:

```
B = [  1       1       1       1    ]
    [  y0      y1      y2      y3   ]
    [ -x0     -x1     -x2     -x3   ]
    [ -km0    +km1    -km2    +km3  ]
Q = inverse(B)
f_hover = Q @ [m*g, 0, 0, 0]
h_i = min(f_hover_i - f_min_i, f_max_i - f_hover_i)
```

Every sampled aircraft must have finite, positive h_i. Otherwise sampling fails
explicitly; it never silently creates an impossible hover task. All capacity
calculations use FP64 at reset. The existing float32/float64 rollout is unchanged.

### External force

Let `C_F = min_i h_i / abs(Q_i0)`. Sample each world-force component with bound
`rho_F*C_F/sqrt(3)`, so `||F_ext|| <= rho_F*C_F`.
For static position holding, the required collective is
`T_req = ||m*g*e_z - F_ext||`. The reverse triangle inequality gives
`abs(T_req-m*g) <= ||F_ext||`, so this load uses at most `rho_F*h_i` of every
rotor's reserve. Required tilt is implicit in this static trim, not an assumption
that the aircraft can counter a horizontal force without tilting.

The prior episode-constant Gaussian force is replaced by this bounded draw.
Keeping that unbounded force while constraining only the four newly requested
mechanisms would not bound the overall training disturbance.

### External torque

For body axis j, define

`C_tau_j = min_i h_i / (3*abs(Q_i,j+1))`, ignoring zero denominator entries.

Sample `tau_ext_j` in `[-rho_tau*C_tau_j, +rho_tau*C_tau_j]`. Then

`abs(sum_j Q_i,j+1*tau_ext_j) <= rho_tau*h_i`.

The factor 3 explicitly covers simultaneous roll, pitch and yaw box corners.
The torque is added to the true rotational dynamics, never to gyro observations.
It stays fixed in body coordinates throughout the episode and is hidden from
the Actor. Yaw uses its actual km coefficients rather than roll/pitch TTI.

### Action execution error

For normalized command u in [-1,1], `m(u) = m_min + (u+1)*(m_max-m_min)/2`.
Its maximum thrust sensitivity over the valid motor range is

`L_i = (m_max-m_min)/2 * (c1_i + 2*c2_i*m_max)`.

Define `C_u_i = h_i/L_i` and sample
`delta_u_i in [-rho_u*C_u_i, +rho_u*C_u_i]`.
Monotonicity, the mean-value theorem and the non-expansiveness of clipping imply
`abs(f(m(clip(u+delta_u)))-f(m(u))) <= rho_u*h_i` for any command in [-1,1].
The bound is for the commanded thrust map; it is not an assertion that a
transient motor state or arbitrary aggressive maneuver is at static equilibrium.

Execution order is known command -> bounded error -> clip [-1,1] -> motor mapping
-> existing asymmetric first-order motor response -> joint RK4.
`previous_action` and action smoothness costs contain the **known command**, not
the corrupted execution or motor truth. Otherwise unobservable actuator error
would leak into the deployable observation. Noise is not TD3/SAC exploration.

Combined force, torque and commanded thrust error use no more than
`(rho_F + rho_tau + rho_u)*h_i`. This is a joint bound, including vector corners,
not independent per-axis maximum-thrust percentages.

## Common sensor envelope and reference error model

Sensors do not intrinsically become less accurate because an aircraft is heavier.
Their sampling law is therefore independent of individual aircraft identity.
To enforce a conservative common allowance, calculate capacities for every
vehicle, take the most restrictive one over the **entire pooled bank**, and then
sample percentages with the same law for every aircraft. These common capacities
can vary across differently sampled banks; they are not per-aircraft scalings or
a claim to a global bound over all possible vehicles. Given a fixed pooled bank,
permutation of vehicle identities does not change the common envelope.

Converting sensor errors to a bounded actuator-load surrogate requires a
controller/error model. The explicit model used here has

`T = .30 s`, `k_p = 1/T^2`, `k_v = 2/T`.

T is the largest motor-response time constant in the retained parameter support.
These reference gains define the normalizer. They are **not identified gains of
the learned Actor**, not new gains in its deployment path, and not a stability
proof. Let J contain the three principal inertias.

Position box capacity:

`C_p = min_aircraft C_F / (m*k_p*sqrt(3))`.

Velocity box capacity:

`C_v = min_aircraft C_F / (m*k_v*sqrt(3))`.

Gyro box capacity:

`C_omega = 1 / max_aircraft,i sum_j abs(Q_i,j+1)*J_j*k_v/h_i`.

Attitude box capacity:

```
A_theta = max_i sum_j abs(Q_i,j+1)*J_j*k_p/h_i
          + sqrt(3)*F_max_total/C_F
C_theta = 1 / max_aircraft A_theta
```

The attitude expression budgets both reference angular correction and thrust
orientation error. A rotation-vector box has norm at most sqrt(3) times its
per-axis bound. `R_measured = R_true @ exp(skew(delta_theta))` remains on SO(3).
This deliberately replaces the old off-manifold matrix-element corruption;
it is not claimed to reproduce the original L2F measurement-noise distribution.
Exp matrices are precomputed once on CPU, not at each physics timestep.

Each sensor's bound is its common capacity times rho. In the reference model,
the position/velocity virtual forces, gyro/attitude virtual torques and thrust
orientation error together consume at most the sum of their rho values in the
same rotor-reserve norm. No loss or termination threshold uses noisy measurements.
A learned network can amplify sensor errors more than this reference model;
parameter bounds alone do not exclude that behavior.

## Fractional velocity delay

A fixed 10-30 ms delay is **not** automatically “10% harmless”. Instead use the
retained physical model to bound the stale-velocity error. With

`A_max = g + F_max_total/m + disturbance_budget*C_F/m`,

world acceleration is bounded in the ideal continuous model. A delay d has
`||v_t-v_(t-d)|| <= A_max*d`. Its reference correction load is at most
`m*k_v*A_max*d/C_F`.

The common delay capacity is

`C_d = min(.03 s, min_aircraft C_F/(m*k_v*A_max))`.

The .03 s ceiling is the fastest motor time constant in the retained support.
Sample `d = rho_d*C_d`. Because rho_d <= .10, **d <= 3 ms**, not 1-3 control
steps. The actual shared budget generally makes it smaller. At dt=10 ms, the
implemented delay is the causal fractional-delay approximation

`v_obs_t = (1-lambda)*(v_t+eps_t) + lambda*(v_(t-1)+eps_(t-1))`,

where `lambda=d/dt <= .3`. It interpolates adjacent acquired noisy **world-frame**
velocity measurements. It is not a full simulation of continuous sensor/firmware
latency; linear interpolation is the stated discrete model. There is no new noise
draw when an old measurement is reused. The interpolation's measurement-noise
bound remains within the original box because its weights are nonnegative and
sum to one. At reset, the previous sample is held at the first measurement.
Only velocity is delayed; position, attitude, gyro and known command remain current.
The current attitude is used to convert the delayed world velocity to body axes.

The stored previous velocity is differentiable and included in the Time Decay
state edges. It is never detached at a 50-step metrics chunk. When a scene fails,
its state, memory, measurement history and time index freeze.

## Sum bound, implementation and interpretation

In the defined static allocation/reference-error model,

`total normalized additional load <= sum_k rho_k <= disturbance_budget <= .10`.

This is meaningful, unit-consistent and testable. It does not imply that an
untrained policy survives, that a saturated aggressive state has 90% spare
control authority, or that every sampled initial condition lies in a recoverable
region. Actual recurrent-policy recoverability depends on the checkpoint,
hidden state, timing and initial condition. No checkpoint in this repository is
automatically authorized for flight on the basis of these tests.

All random tapes are immutable. Stable original-bank row IDs allow live-scene
compaction without copying the full tapes. Pool before sampling the common
sensor envelope, then transfer the bank once. Full BPTT/group VJPs reuse the
same random realization. TRAIN changes bank seeds each update; EVAL keeps the
same seeds/tapes. Noise config, implementation hashes and version are checkpointed.
Old protocol resume/evaluation are rejected. Explicit compatible Actor-weight
import restores neither the old environment nor its optimizer. There is no new Critic, estimator or safety-controller
network in the deployable policy.

## Reproducible calculation example

For the retained config's first TRAIN pool (seeds 31000007..31000010, 512 scenes,
H500, FP32), the sampled mass range is 0.0200485..4.98567 kg. The unscaled common
sensor capacities are 0.256210 m, 0.427017 m/s, 0.0699168 rad per attitude-vector
axis, 0.323507 rad/s; common delay capacity is 0.0207091 s. These are **100%-unit
capacities**, multiplied by each component's allocated fraction, not default
noise standard deviations. All draws use bounded uniform distributions.

The resulting largest allocated per-axis bounds in this pool are 0.0147822 m,
0.0237802 m/s, 0.00381827 rad and 0.0133386 rad/s. Maximum sampled delay is
1.41263 ms; maximum allocated command bound is 0.0181205 in [-1,1] coordinates.
Maximum total fraction is 0.099999629 after numerical headroom. Other seeds yield
other actual ranges but satisfy the same mathematical inequalities. These
calculations are not trained-policy performance results.
