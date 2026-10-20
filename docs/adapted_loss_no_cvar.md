# Smooth-vector task objective with relative attitude changes

The adapted loss was introduced on the GRU16/direct-Linear commit
`e45739e1b290df67592ccbcc3799cdaed4764846`. The current local branch
`feat/gru-only-no-wc` adopts B: GRUCell(16,64) plus hidden-only Linear(64,4),
with no Wc or geometric feedback. The attitude-motion revision changes only the
loss and its necessary trajectory/logging interfaces.
Physical integration, motor law,
observations, noise/delay/pulse tapes, sampling, termination, groups, BPTT,
Time Decay, clipping and Adam retain their original implementation.

## Definition and timing

For a vector x, with fixed epsilon > 0:

    rho(x) = sqrt(sum(x*x) + epsilon*epsilon) - epsilon
           = sum(x*x) / (sqrt(sum(x*x) + epsilon*epsilon) + epsilon).

One scene is scored as:

    L_i = sum_valid_t [rho_epsilon_p(p_t)
                      + lambda_R * (1 - cos(theta(R_t.T @ R_(t+1))))
                      + rho_epsilon_a(a_t - a_(t-1))] / H
          + failed_i * [3*(H - T_i) + 200] / H.
    J = mean_i L_i, over the initial N scenes.

`TaskTrajectory.pre_positions` and `pre_orientations` store true pre-command
states. `post_orientations` stores each executed transition's true endpoint.
For unit wxyz quaternions the attitude cost is twice the squared vector part of
`conj(q_t) * q_(t+1)`. This equals `(3-trace(R_t.T @ R_(t+1)))/2` and
`||R_(t+1)-R_t||_F^2/4`. The implementation uses quaternion differences to avoid
cancellation at tiny rotations; it needs neither acos nor a new smoothing scale.
The original `positions`, velocities and omegas remain post-transition records
for evaluation, replay and failure detection. No noisy observation or delayed
measurement is used as task truth.

A constant attitude, including the tilt needed to balance a horizontal force,
costs zero. Roll, pitch and yaw changes all count, without a world-upright or
absolute-heading target. Normal acceleration/braking/recovery also rotates:
this is a soft motion regularizer, not an instability test. For small angles
the term is approximately `lambda_R*theta^2/2`; a relative half-turn is a
stationary point. This is no stability or arbitrary-attitude recovery guarantee.

The first position cost comes from the initial state; there is no extra terminal
position cost. Each attitude cost compares the actual pre/post endpoints,
including the last executed or boundary-crossing transition. It is not reset at
statistics chunk boundaries. Actions are the final tanh
commands in [-1,1]. The four-command delta is one vector smooth L2 norm,
without the old factor .5, squaring, division by dt or per-motor averaging.
Its zero initial history and differentiable cross-chunk history are retained.

H is the planned horizon. T_i includes the crossing transition. Frozen padding
is masked before nonlinear calculations and contributes no cost. A failure on
transition H still pays 200/H, while completing H without a crossing pays zero.
Valid nonfinite data or computed costs raise an error rather than being erased.

## Fixed parameters and interfaces

The user-selected experiment parameters are recorded in the argument file:

| Parameter | Fixed value | Meaning |
|---|---:|---|
| `epsilon_p` | 0.01 m | Whole three-dimensional position-vector smooth norm |
| `epsilon_a` | 0.01 | Whole four-command difference-vector smooth norm in raw [-1,1] units |
| `lambda_R` | 25.0 | Multiplier of per-control-step relative rotation `1 - cos(theta)` |

These finite positive constants are experimental starting settings, not validated
stability bounds. They are frozen configuration values, not network parameters;
they do not change with updates, scenes or gradient statistics. New checkpoint
bindings serialize all three values with the objective version. Bare CLI calls
still require explicit `--epsilon-p`, `--epsilon-a`, `--lambda-R`; the checked-in
argument file supplies them. The test fixture values .5/.25/.4 remain independent
arithmetic examples, not recommended training settings.

No division by dt or dt squared is applied. At 100 Hz, a constant yaw rate of
1 rad/s produces `25*(1-cos(0.01))`, approximately 0.00125. The previous
coefficient 0.2 gave approximately 0.00001 for the same motion. Equal relative
rotation angles about roll, pitch or yaw have equal cost; normal recovery and
disturbance rejection can require roll/pitch changes, so this is not evidence
that every attitude change is undesirable. The larger fixed coefficient is an
experimental setting and does not establish that learned yaw spin will disappear.
Position/action smoothing, learning rate and sampling are unchanged.

The checked-in argument file preserves 2048 TRAIN scenes, coverage128, H500,
Time Decay 1, gradient_scale .1, group cap 1, global clip 10 and Adam lr 3e-4.
It uses a fresh `runs/attitude_delta_lambda25_gru_only_no_cvar/seed7` directory and supplies the three
confirmed constants above. Training remains an explicit CLI operation, separately authorized by the user.

`step_cost_components()` returns position, attitude_delta, action_delta, dead, terminal
directly. Their sum defines `step_costs()`. TRAIN streaming statistics, fixed
EVAL, independent checkpoint evaluation and best-score selection consume this
same definition. Component sums agree with the total up to floating reduction
roundoff. The old residual/feature-square and old loss-field interfaces are
removed, not silently accepted or converted into square-root costs.

Scene weights are exactly 1/N. The group coefficient remains N/n_g times that
scene weight, applied once; group clipping, group mean, global clipping and Adam
follow in their original order. A Time Decay/group-clipped update is still a
processed proxy gradient, not the exact gradient of unmodified J.

Failure costs remain detached bookkeeping. They change evaluation and best.pt
selection, but give no direct avoidance gradient. Without CVaR they also cannot
redirect scene weights. The retained omega/saturation warning reports keep their
prior read-only aggregation (mean + .5*top-20%-mean); this does not enter J,
group cotangents, clipping, or update approval.

## Checkpoints and verification

Binding stores `task_objective=pre-position-so3-transition-action-delta-equal-scenes-v2`
and the fixed loss configuration. Resume still requires exact source and full
configuration. An old-objective checkpoint is rejected for resume/current
scoring. A same-architecture checkpoint may explicitly initialize weights using
`--init-checkpoint`; Adam, sampling index and best score start afresh. This is
fine-tuning, not exact continuation. No old checkpoint binding or score is edited.

Intentional loss updates and the later Wc removal are separately recorded in
`tests/core_contract.json`, with previous hashes and reasons preserved.
Physics, feature construction, sampling, group clipping, Time Decay and Adam
contracts remain unchanged by Wc removal.

The deterministic regressions cover independent hand values, analytic/finite
differences, zero gradients, quaternion sign equivalence, arbitrary fixed tilt,
three-axis rotation and relative half-turn, pre/post timing,
initial commands and 50/51 continuity, first/middle/final failure, invalid padding,
uniform scene weights, grouped versus direct VJPs, proxy Time Decay, failure-cost
gradient independence, streamed/EVAL component agreement, exact resume and
explicit old-objective weight import. CUDA tests run when available and otherwise
report their explicit skip reason. These checks do not establish learned control
quality or stability, and old/new task scores are not directly comparable.
