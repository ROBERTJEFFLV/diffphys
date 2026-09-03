# MATLAB L2F Quadrotor Environment

This folder mirrors the current PyTorch `env_l2f.py` micro-quadrotor dynamics in
MATLAB-oriented form. It is intended as a MATLAB/Simulink simulation environment
for monitoring trajectories, attitude, body rates, motor commands, and future
environment modules such as wind or gusts.

The model is still the L2F PyTorch micro quadrotor model:

- normalized motor command in `[-1, 1]`,
- first-order motor response with separate rising/falling time constants,
- thrust polynomial `c0 + c1 * motor + c2 * motor^2`,
- generic motor geometry with per-rotor position, axis, spin direction, health,
  delay, thrust scale, and torque coefficient,
- body-frame motor wrench, gravity, external force, aerodynamic drag, ground
  effect, battery sag, payload shift, and motor fault hooks,
- implicit-midpoint angular velocity integration,
- SO(3) exponential rotation update.

## Standard MATLAB Rollout API

From the repository root:

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

`l2f_rollout` is the long-term entry point. `run_l2f_matlab_demo` remains as a
backward-compatible wrapper for quick manual demos.

The main log tensors keep an explicit batch dimension:

```matlab
logs.position  % (T+1) x N x 3
logs.velocity  % (T+1) x N x 3
logs.rotation  % (T+1) x N x 3 x 3
logs.omega     % (T+1) x N x 3
logs.action    % T x N x 4
logs.motor     % T x N x 4
logs.done      % T x N
logs.reward    % T x N
```

Use a custom quadrotor through `uav_cfg`, not by growing `params`:

```matlab
uav_cfg = l2f_uav_cfg_library('small150g');
uav_cfg.motor_health = [1.0 1.0 0.7 1.0];
sim_cfg = l2f_default_sim_cfg( ...
    'horizon', 500, ...
    'batch_size', 8, ...
    'uav_cfg', uav_cfg);
logs = l2f_rollout(params, world, sensors, [], reward_cfg, sim_cfg);
```

`params` is kept for legacy L2F numeric settings and compatibility. New vehicle
physics should live in `uav_cfg`; reset expands it into per-batch state dynamics.

Named vehicle configs currently include `nominal_50g`, `small150g`, and `x500`.
Each config carries:

```matlab
uav_cfg.mass
uav_cfg.inertia
uav_cfg.rotor_pos_body
uav_cfg.rotor_axis_body
uav_cfg.spin_dir
uav_cfg.thrust_coeff_c0
uav_cfg.thrust_coeff_c1
uav_cfg.thrust_coeff_c2
uav_cfg.motor_tau_rise
uav_cfg.motor_tau_fall
uav_cfg.motor_deadzone
uav_cfg.motor_delay_steps
uav_cfg.rotor_torque_constant
uav_cfg.drag_linear
uav_cfg.drag_quadratic
uav_cfg.collision_radius
uav_cfg.collision_height
uav_cfg.battery
```

Enable wind and aerodynamic drag:

```matlab
world = l2f_make_world('wind_drag');
world.wind.velocity = [2 0 0];
logs = l2f_rollout(params, world, sensors, [], reward_cfg, sim_cfg);
```

For a quick single-command smoke run, the old demo wrapper still works:

```matlab
logs = run_l2f_matlab_demo('horizon', 500, 'dynamics_profile', 'physical-broad');
```

## Controller Interface

The runtime accepts any controller function with this signature:

```matlab
action = controller(obs, state, params, world, t);
```

`action` must be `batch_size x 4` and is clamped to `[-1, 1]` by the simulator.
The default controller is `l2f_pid_action`, which is only a reference baseline.
The trained policy can be connected later through ONNX, MATLAB Engine for Python,
or a Simulink block, as long as it returns the same normalized four-motor action.

For compatibility, function handles with `(state, params, t)`, `(obs, t)`, or
`(obs)` are also accepted.

## Task Layer

The default task is persistent-disturbance position hold. Main success depends
only on position, linear velocity, and the complete angular-rate norm. Roll,
pitch, and yaw orientation are free; a fixed arbitrary yaw can succeed, while
continuous yaw spin cannot. A complete final window of 100 executed steps must
contain at least 95% successful steps. Snapshot success is diagnostic only.

`R` remains in the MotorGRU observation so the controller knows the current
thrust direction. There is no attitude, tilt, or yaw reward, success gate, or
termination condition.

Current exported policies use `integral25`:
`[p, v, flatten(R), omega, integral_position, previous_action]`. The integral
state is always accumulated in world coordinates. Exports declare whether the
actor receives it directly (`world`, P4b/Q0) or receives
`R^T*integral_position` (`body`, Q1/Q2). The integral
persists through the rollout, uses the same observed position sample as the
actor, and applies the exported clamp/leak settings. `compact22` omits the
integral. Old 40-column exports remain supported as `legacy40`; their affine
duplicate is constructed from the same physical sample, not from a separately
noised error branch. The exported `observation_mode` and `input_dim` must agree,
otherwise MATLAB fails explicitly. Q1/Q2 exports also carry the optional
zero-initialized integral and rate-damping motor-logit residual branches. The
damping branch uses the network-predicted motor state, never simulator motor
state, capability, or external force.

Streaming and full-log evaluation both report the final-window RMS, absolute
maximum, and dominant frequency for `omega_x/y/z`, per-motor action and action
delta RMS, and strict/loose bounded-angular-motion flags. The same diagnostics
are also captured at each requested horizon (for example H500 and H10000), so
MATLAB and Python failure analysis use the same final-100-step convention.

Tasks are available both in `matlab_l2f/tasks/` and mirrored at the folder root
so this machine can run them after `cd('matlab_l2f')` even when `addpath` is
broken.

```matlab
task = l2f_task_config_generalization( ...
    'uav_family', {'micro50g', 'small150g', 'x500'}, ...
    'horizon', 300, ...
    'batch_size', 16);

logs = l2f_rollout(task, [], l2f_default_sim_cfg('horizon', 300, 'batch_size', 16));
metrics = l2f_metrics(logs, logs.world, logs.reward_cfg);
```

Current task entry points:

- `l2f_task_recovery`
- `l2f_task_config_generalization`
- `l2f_task_wind_rejection`
- `l2f_task_payload_shift`
- `l2f_task_motor_fault`
- `l2f_task_trajectory_tracking`
- `l2f_task_sensor_delay`

`l2f_config_sweep` performs a mass / motor time constant / thrust-to-weight
sweep and returns rows with success and failure-boundary metrics.

## Simulink Scaffold

`build_l2f_simulink_model.m` creates a one-vehicle Simulink scaffold with a
MATLAB Function block calling `l2f_step_flat`. It uses flat vectors instead of
Simulink buses so that later environment modules can be wired in cleanly.

```matlab
addpath(genpath('matlab_l2f'));
build_l2f_simulink_model;
```

The scaffold is a starting point for Simulink integration. The pure MATLAB demo
is the reference implementation for numerical behavior.


## Fast H500-H10000 Policy Evaluation

`run_motor_gru_policy_eval` now uses a streaming metrics-only rollout by default.
It simulates one continuous H10000 trajectory and records snapshots at H500,
H1000, H2000, H5000 and H10000. Physical state, GRU hidden state, `dt`, the
implicit-midpoint rigid-body step, SO(3) update and double-precision values are
unchanged; the speed and memory gain comes from not storing every state, wrench,
Euler angle and reward term for all vehicles and all time steps.

```matlab
addpath('matlab_l2f');
metrics = run_motor_gru_policy_eval( ...
    'weights_path', 'model_step_4000.mat', ...
    'output_path', 'summary.csv', ...
    'sample_output_path', 'samples.csv', ...
    'mat_output_path', 'metrics.mat', ...
    'batch_size', 1024, ...
    'seed', 7, ...
    'horizon', 10000, ...
    'long_horizons', [500 1000 2000 5000 10000], ...
    'dynamics_profile', 'physical-broad', ...
    'broad_sampler', 'physical', ...
    'eval_mode', 'streaming');
```

Use the legacy full-log path only when trajectory replay or complete logs are
needed:

```matlab
run_motor_gru_policy_eval( ...
    'weights_path', 'model_step_4000.mat', ...
    'horizon', 500, ...
    'batch_size', 16, ...
    'eval_mode', 'full', ...
    'save_logs', true, ...
    'mat_output_path', 'full_logs.mat');
```

The sample CSV reports position-hold snapshot/steady-window outcomes,
final-window fraction, settling time, stay, and physical survival. External
force and required-equilibrium tilt remain diagnostic columns only; no
identity-attitude feasibility gate is exported or used for ranking.

Run the equivalence test in MATLAB before first production use:

```matlab
addpath('matlab_l2f/tests');
results = run_fast_eval_tests;
```
