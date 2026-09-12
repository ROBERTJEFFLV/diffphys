# L2F-in-seconds

## Current production response path

`tools/train_response_control.py` trains `ResponseMotorPolicy` using exact H500
backpropagation with H50 reverse-window rematerialization and one Adam update.
There is no Critic, auxiliary prediction head or per-update candidate search.
Use `configs/response_phase1_single_airframe.args`; see
[the training and checkpoint contract](docs/response_control_v1.md).
The baseline architectures and earlier experiments below are historical context.

This folder contains a compact CUDA-tensor training chain derived from the current
DiffPhys L2F CUDA work and reshaped to match the `diffphysDrone` recurrent control
flow.

## Layout

- `model.py`: motor policy network:
  `25D observation -> Linear/MLP encoder -> LeakyReLU -> GRU -> Linear motor head -> 4 motors`.
- `env_l2f.py`: differentiable L2F-style Euler quadrotor simulator. It runs on
  CUDA when tensors are on a CUDA device.
- `env_cuda.py`: compatibility export for the simulator, matching the source
  project's naming style.
- `l2f_cuda_backend.py` and `cuda_ext/`: optional native CUDA Euler transition
  backend with an analytic VJP for the rigid-body step.
- `train.py` and `main_cuda.py`: direct horizon training entry points. The four
  motor outputs are fed into the simulator at every step.
- `matlab_l2f/`: MATLAB/Simulink-oriented mirror of the L2F PyTorch micro
  quadrotor environment, with realtime plotting and extensible environment
  force/torque hooks.
- `configs/`: runnable argument files for smoke and H500 training.
- `reference/current_diffphys_cuda_chain/`: copied current RLtools CUDA L2F
  simulator and differential pre-training source files.
- `reference/diffphysDrone_flow/`: copied files showing the source recurrent
  workflow from `~/diffphysDrone`.

The experimental replacement for the failed checkpointed-H1000 prototype is
documented in [`docs/equilibrium_strict_architecture.md`](docs/equilibrium_strict_architecture.md).
It uses a fast motor observer, slow capability identifier, equilibrium-centered
specific-wrench feedback, a constrained allocator, and independent-node
full-space shooting.  It remains gated beside Q2; a lower training loss alone
does not promote it.

## Architecture

The current deployable actor input has 25 values:

```text
[position(3), velocity(3), flatten(R)(9), omega(3),
 position_integral(3), previous_action(4)]
```

The integral state is accumulated in world coordinates from the same deployable
position sample as the actor, persists
across H250 truncated-BPTT segments, detaches at segment boundaries, resets at
episode reset, and is clamped by `--integral-limit`. Its leak defaults to zero
and is configurable with `--integral-leak`. P4b feeds that world-frame value
directly to the actor; Q1/Q2 rotate it with the observed `R` as
`I_body = R^T I_world`. Physical observation noise is
sampled once; there is no second `error` branch or independent duplicate noise.
Motor state, external force, and vehicle dynamics parameters are never actor inputs.

For paired migration tests, `compact22` omits the integral and `legacy40`
reconstructs the historical affine duplicate from the same physical sample.
Old 40D checkpoints are folded losslessly into 22D/25D first-layer weights;
when the integral is zero, action and hidden outputs are numerically equivalent.

The policy emits four normalized motor commands in `[-1, 1]`. Zero command is
hover thrust; positive and negative values increase or decrease each rotor around
hover.

The task is position holding under an unknown, persistent external force.
Roll, pitch, and yaw are not terminal targets. Rotation matrix `R` remains an
actor observation because the controller must know its thrust direction, and
the complete angular-rate norm must converge, so continuous yaw spin is not
successful.

The primary success metric is a steady window: position error below 0.05 m,
velocity below 0.10 m/s, and full angular-rate norm below 0.20 rad/s for at
least 95% of the last 100 physical steps. A one-frame snapshot is diagnostic
only. No attitude, yaw, body-z, or required-tilt threshold participates in
success, termination, stay, survival, checkpoint selection, or model ranking.

Training now uses a dense H500 recovery objective rather than a short-horizon
auxiliary loss. Each rollout accumulates robust position/velocity/angular-rate terms,
CLF-style contraction, outward-velocity recovery, tail stability, command
smoothness, command acceleration, and soft saturation penalties. Backward-only
gradient decay is applied to simulator state and GRU hidden chains; the forward
rollout and loss values remain full-horizon. The CUDA backend mirrors the
reference repository's Euler style with implicit-midpoint angular velocity,
`R_next = R * Exp(dt * skew(0.5 * (omega + omega_next)))`, and a custom
transition VJP.

### Training-only belief supervision

`MotorGRUPolicy.forward()` retains the P4b deployment path
`encoder -> GRU -> motor_head` unless an explicitly exported Q residual branch
is enabled. Q1 adds a small integral-to-motor-logit residual; Q2 also adds a
rate-damping residual based only on the GRU latent, observed omega, previous
action, and the network's predicted motor state. Both residual output layers
are zero-initialized, so loading P4b preserves the zero-integral initial action
and hidden state. `forward_with_aux()` additionally predicts the
current pre-action motor state, six aggregate capability quantities, and the
normalized next-step velocity/angular-velocity response from the same GRU
latent. Detached simulator values supervise auxiliary Huber losses. Response
uses the current action as a stop-gradient explanatory input; no privileged
auxiliary gradient enters the action head.

Old action-only checkpoints can initialize this model. Only the new auxiliary
or residual parameters may be missing, and their old optimizer state is
intentionally not restored. MATLAB export includes only modules needed for the
deployed action path; capability/response teacher heads remain excluded.

For a motor-only, single-variable first experiment, use:

```bash
python3 train.py $(cat configs/belief_motor_aux.args)
```

This configuration carries hidden state across 250-step truncated-BPTT
segments, updates only at the configured training-episode boundary, and uses current-time
motor targets after burn-in. Stage 1 deliberately retains the existing
`physical` sampler and disturbance distribution for single-variable attribution.
The compact CUDA step consumes per-sample randomized dynamics and supplies an
analytic VJP; CUDA-full still has neither auxiliary-head gradients nor
`hidden0`/`final_hidden` support. The trainer raises an error instead of silently
using auxiliary or persistent-hidden training with CUDA-full.

On Windows, the CUDA loader discovers Visual Studio Build Tools through
`vswhere` and imports the `vcvars64` environment automatically, so training does
not require launching a separate Developer Command Prompt.

There is no attitude, tilt, or yaw objective in the training loss, success
metric, or termination path. `R` remains a deployable observation because the
controller must know the current thrust direction. The complete angular-rate
norm must converge, so a fixed arbitrary yaw is allowed but continuous spin is
not.

## Commands

Run a short validation:

```bash
python3 train.py $(cat configs/smoke.args)
```

Run the H500 CUDA training path:

```bash
python3 train.py $(cat configs/direct_h500.args)
```

Validate the CUDA transition VJP against PyTorch autograd on a CUDA machine with
`nvcc` available:

```bash
python3 tools/validate_cuda_vjp.py
```

Outputs are written under `runs/` and `checkpoints/`.

## Paired belief auxiliary ablation

The formal A-E comparison uses compact CUDA and one shared baseline:

```text
A  control
B  motor
C  motor + capability
D  motor + response
E  motor + capability + response
```

Run the five training seeds, verify exact reset-sample pairing, and then run
the fixed-checkpoint H500/H10000 evaluations:

```bash
python tools/run_belief_ablation.py --phase train
python tools/run_belief_ablation.py --phase audit
python tools/run_belief_ablation.py --phase eval
python tools/run_belief_ablation.py --phase summarize \
  --hard-samples-path reports/dynamic_hard_43.csv
```

The optional hard-sample manifest must contain exactly 43 rows with
`eval_seed,sample_index` columns. The runner rejects skipped/invalid updates,
checks that all five groups saw byte-identical reset samples for each training
seed, records per-sample evaluation outcomes, and reports retention on samples
that the original baseline solved. Auxiliary training and persistent hidden
remain intentionally unsupported by `cuda-full`.

The current compact-input/tail comparison uses paired `P0`--`P4b` configs. All
groups use the same H500 episode, baseline initialization, retain-bank hard
replay, reset stream, optimizer-update checkpoints, learning rate and seeds.

```text
P0   current R1 control: legacy40 + combined position/omega top-k
P1   P0 + independently selected position and omega CVaR tails
P2   P1 + corrected full episode-event weighting
P3   P2 + an early H250 tail (weight 0.25; H500 weight 1.0)
P4a  P3 + compact22 input, without integral
P4b  P3 + integral25 input and clamped position integral
```

The Q0--Q2 residual comparison continues from the fixed P4b update-2000
checkpoint:

```text
Q0  unchanged P4b control (world-frame integral)
Q1  Q0 + body-frame integral + explicit integral motor-logit residual
Q2  Q1 + deployable rate-damping residual + low-authority multi-step omega decay
```

The multi-step decay uses simulated future angular rates only as training
supervision at 5/10/25 steps. It is restricted to low-authority hard-replay
samples whose starting angular rate is at least the success threshold; no
future state or privileged value enters deployment. Run and audit the paired
training with:

```bash
python tools/run_q_residual_ablation.py --phase all --max-parallel 2
```

`cuda-full` does not implement 25D observations, persistent hidden state,
residual action branches, or this decay loss and is rejected explicitly.

The current CVaR objective computes position and angular-rate margins
separately, selects each worst 20% independently, and then adds the two means.
Velocity is deliberately absent. H250 and H500 events are scaled before
episode gradient averaging, so a configured weight of 0.001 remains 0.001
instead of becoming 0.0005 in a two-segment episode.

The retain bank is generated from a fixed H10000 evaluation of the initialization
baseline. A scenario enters the bank only if the baseline is steady-successful
at H10000 and lies in the lowest 20% of roll authority, lowest 20% of yaw
authority, or highest 20% of fall time. Every group receives the same 25% bank
sampling stream. In P0--P4b this is hard replay only: action-retain MSE is off,
as are capability and response auxiliary losses. These are distinct concepts:
retain-bank sampling changes which hard vehicles are trained on; action retain
is an anti-forgetting regularizer and is not enabled in this experiment.

Historical `physical` retain banks remain valid only for historical
reproduction. A `physical-fit` run with nonzero retain sampling requires bank
metadata declaring `broad_sampler=physical-fit` plus the exact current
`env_l2f.py` SHA-256. The training-entry guard checks every floating state
field for finiteness, critical physical fields for strict positivity,
nonnegative disturbance scale, `tau_fall >= tau_rise`, and the rigid-body
inertia inequalities. This prevents a corrected sampler from being silently
contaminated by legacy or unverifiable resets; it intentionally requires a
bank rebuild after any sampler-source change.

All schedules are counted in accepted optimizer updates. Formal configs run
2000 updates, ramp auxiliary weights over 75 updates, and save updates
250/500/750/1000/1500/2000.

The follow-up integral-scale/time-horizon experiment keeps Q2's control,
CVaR, damping, hard-replay, learning-rate, batch and dynamics settings fixed.
`--integral-input-multiplier` scales only the three deployable integral
features immediately before the network. Formal scale comparisons use
`--compensate-integral-input-scale-on-load`, which divides every first-layer
integral column by the same multiplier so action and GRU hidden are exactly
unchanged at initialization.

Long-horizon training uses a batched physical-step budget rather than an
optimizer-update budget. `physical_steps = batch_size * simulated_steps`; at
batch 256 an H500 episode therefore consumes 128,000 physical steps. The
mixed deterministic curriculum is H500-only for the first 20%, 70/30
H500/H1000 for the next 30%, and 50/30/20 H500/H1000/H2000 for the final 50%.
Every episode remains H250 truncated BPTT: hidden and position integral persist
across segments and detach at each boundary, with one optimizer step only at
the complete episode boundary. Dense and motor-auxiliary losses are averaged
over episode time; final CVaR is not divided by the segment count. Auxiliary
ramping and history checkpoints are addressed by accumulated physical steps.

```bash
python tools/run_time_horizon_training.py --workers 2
python tools/run_time_horizon_checkpoint_eval.py --workers 2
```

The four configs are `time_horizon_T0.args` through `time_horizon_T3.args`:
T0/T1 use fixed H500 with integral multipliers 1.0/selected-scale, while T2/T3
use the same two scales with the mixed horizon curriculum.

### Continuity/cadence causal follow-up

`--optimization-block-horizon` controls optimizer commits inside longer reset
episodes. The default-off `--tail-supervision-block-horizon` can independently
place complete position/omega CVaR blocks inside one optimizer block. A custom
tail cadence is restricted to independent CVaR, must contain at least two H250
segments, and must divide both the optimizer and every scheduled reset horizon.
Zero preserves the historical optimizer-coupled behavior.

`configs/continuity_cadence_D.args` keeps the compressed T2 reset sequence and
67 reset-boundary commits while producing 75 complete H500 CVaR blocks. This
changes both CVaR event placement and total event mass at fixed commit count;
it is not a pure cadence intervention. It is a preregistered experiment, not a
default. See
`reports/PREREGISTRATION_ARM_D_20260806.md`.

### Physical coverage audits

`tools/audit_physical_fit_sampler.py` checks 65,536 balanced current-source
samples. The listed necessary/source-consistency constraints pass, including
all 4^4 root cells and feasible principal inertias, but this is not a real-fleet
validation: no sampled rise time is at most 35 ms, `Jx=Jy` and linear identical
motor curves are universal assumptions, and several derived quantities pile up
at caps. `physical-fit` therefore remains opt-in.

`tools/audit_size_causality.py` constructs scale-equivalent trajectories across
0.02--5 kg under matched normalized capability and branch-covering commands.
It is a counterexample to mandatory absolute-size identification, not a general
identifiability theorem. `tools/analyze_physical_failure_axes.py` separately
shows that the historical formal failures stratify much more strongly by
required tilt, thrust-to-weight, disturbance scale, and angular authority than
by mass alone. These marginal strata are exploratory because the parameters
are correlated.

```bash
python tools/build_retain_bank.py
python tools/run_compact_cvar_ablation.py --phase train --max-parallel 2
python tools/run_compact_cvar_ablation.py --phase audit
python tools/run_matlab_position_hold_eval.py \
  --experiment-root reports/compact_cvar_ablation_h500_u2000_gpu2 \
  --output-root reports/compact_cvar_ablation_h500_u2000_matlab \
  --groups P0,P1,P2,P3,P4a,P4b --checkpoint-update 2000 --include-baseline
```

Evaluation CSVs include tail RMS, absolute maximum, and dominant frequency for
each angular-rate axis, per-motor action/action-delta RMS, and strict/loose
bounded-angular-motion flags. The independent reachability diagnostic is:

```bash
python tools/eval_privileged_oracle.py --updates 2000 --restarts 3
```

Each scenario/restart owns a separate privileged teacher trained from scratch;
there is no baseline residual. Full H500 trajectories participate in gradient
optimization, with H250 and H500 position/omega tails. Teacher failures receive
a differentiable direct-shooting attempt. Final labels use MATLAB H10000 steady
validation: success is `oracle_reachable`; every optimization failure remains
`unproven`, never `physically_infeasible`.

## Controller Comparison & Visualization

Use `compare_controllers.py` to compare learned policy and baseline controllers
under one `L2FSimulator` rollout with identical random initial states and
parameters.

This script exports:

- position, attitude, angular velocity and control trajectories,
- per-sample dynamics parameters in `dynamics.csv`,
- stability/recovery/overshoot metrics, and
- plotting outputs compatible with the Bidirectional reference styles:
  `draw_real.png`, `draw_q.png`, `draw_w.png`, `draw_u.png`.

Example:

```bash
python3 compare_controllers.py \
  --checkpoint-path checkpoints/cpu_tiny.pt \
  --controllers learned,pid \
  --horizon 120 \
  --batch-size 8 \
  --seed 7 \
  --eval-seed-count 1 \
  --dt 0.01 \
  --save-dir reports/compare_pid_vs_learned \
  --no-plot
```

You can keep plots enabled by omitting `--no-plot`; plots are saved in
`<save-dir>/<save-prefix>_plots`.

Use `--dynamics-profile physical-broad` or `--dynamics-profile raptor-broad` to
reuse the training-time randomized physical parameter coverage from
`env_l2f.py`. Use fixed mode with nominal overrides to evaluate a custom
quadrotor:

```bash
python3 compare_controllers.py \
  --controllers pid \
  --mass 0.08 \
  --arm-length 0.055 \
  --inertia-x 2.0e-5 \
  --inertia-y 2.1e-5 \
  --inertia-z 3.0e-5 \
  --motor-authority 1.5 \
  --save-dir reports/custom_quad
```

## MATLAB/Simulink Environment

`matlab_l2f/` contains a MATLAB-side implementation of the current L2F PyTorch
micro-quadrotor dynamics and environment randomization. It now separates
simulation numerics from UAV physical config, uses generic per-rotor motor
geometry, and includes wind/drag/ground-effect/battery/payload/motor-fault hooks
for model validation.

```matlab
addpath('matlab_l2f');
params = l2f_default_params('dynamics_profile', 'physical-broad');
world = l2f_make_world('empty');
sensors = l2f_make_sensors('ideal');
reward_cfg = l2f_default_reward_cfg();
sim_cfg = l2f_default_sim_cfg('horizon', 500, 'batch_size', 8);

logs = l2f_rollout(params, world, sensors, [], reward_cfg, sim_cfg);
metrics = l2f_metrics(logs, world, reward_cfg);
l2f_replay_3d(logs, world, 'vehicle_id', 1);
```

Use `uav_cfg` for vehicle physics:

```matlab
uav_cfg = l2f_uav_cfg_library('small150g');
sim_cfg = l2f_default_sim_cfg('horizon', 500, 'batch_size', 8, 'uav_cfg', uav_cfg);
logs = l2f_rollout(params, l2f_make_world('wind_drag'), sensors, [], reward_cfg, sim_cfg);
```

Model-validation tasks can be run as:

```matlab
task = l2f_task_config_generalization('horizon', 300, 'batch_size', 16);
logs = l2f_rollout(task, [], l2f_default_sim_cfg('horizon', 300, 'batch_size', 16));
```

The batch log layout is standardized as `(T+1) x N x state_dim` for state
history and `T x N x action_dim` for per-step command history, with
`logs.done` and `logs.reward` shaped as `T x N`. `run_l2f_matlab_demo` is now a
compatibility wrapper around this API.

The Simulink scaffold can be generated from MATLAB with:

```matlab
addpath('matlab_l2f');
build_l2f_simulink_model;
```
