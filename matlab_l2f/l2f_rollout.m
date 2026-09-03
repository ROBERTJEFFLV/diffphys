function logs = l2f_rollout(params, world, sensors, controller, reward_cfg, sim_cfg)
%L2F_ROLLOUT Standard batch rollout entry point for the MATLAB L2F environment.
%
% logs.position: (T+1) x N x 3
% logs.velocity: (T+1) x N x 3
% logs.rotation: (T+1) x N x 3 x 3
% logs.omega:    (T+1) x N x 3
% logs.action:   T x N x 4
% logs.motor:    T x N x 4
% logs.done:     T x N
% logs.reward:   T x N

if nargin >= 1 && isstruct(params) && isfield(params, 'is_l2f_task') && params.is_l2f_task
    if nargin < 2
        world = [];
    end
    if nargin < 3
        sensors = [];
    end
    logs = l2f_rollout_task(params, world, sensors);
    return;
end

if nargin < 1 || isempty(params)
    params = l2f_default_params();
end
if nargin < 2 || isempty(world)
    world = l2f_make_world('empty');
end
if nargin < 3 || isempty(sensors)
    sensors = l2f_make_sensors('ideal');
end
if nargin < 4 || isempty(controller)
    controller = @(obs, state, params, world, t) l2f_pid_action(state, params, []);
end
if nargin < 5 || isempty(reward_cfg)
    reward_cfg = l2f_default_reward_cfg();
end
if nargin < 6 || isempty(sim_cfg)
    sim_cfg = l2f_default_sim_cfg();
end

params = sync_params_from_sim_cfg(params, sim_cfg);
if sim_cfg.terminate_on_success
    error(['terminate_on_success is unsupported for steady position-hold success; ' ...
        'evaluate the complete steady window instead.']);
end
horizon = sim_cfg.horizon;
batch_size = sim_cfg.batch_size;
uav_cfg = resolve_uav_cfg(sim_cfg);
if ~isempty(sim_cfg.initial_state)
    state = sim_cfg.initial_state;
    batch_size = size(state.position, 1);
else
    state = l2f_reset(batch_size, params, sim_cfg.seed, uav_cfg, sim_cfg);
end

logs = initialize_logs(horizon, batch_size, params, world, sensors, reward_cfg, sim_cfg, state);
logs = write_state_log(logs, state, 1);

done_accum = false(1, batch_size);
monitor = [];
sensor_buffer = {};
if sim_cfg.live_plot
    monitor = l2f_live_plot('init', [], [], 0);
end

for step = 1:horizon
    t = logs.time(step);
    previous_state = state;
    [obs, sensor_buffer] = l2f_apply_sensors(state, sensors, t, sensor_buffer);
    [action, controller_info] = l2f_controller_action(controller, obs, state, params, world, t);
    action = ensure_action_shape(action, batch_size);
    logs = write_controller_log(logs, controller_info, step, batch_size);
    if sim_cfg.freeze_done
        action(done_accum, :) = 0.0;
    end

    [candidate_state, aux] = l2f_step(state, action, params, world, t);
    if sim_cfg.freeze_done && any(done_accum)
        candidate_state = select_state(previous_state, candidate_state, ~done_accum);
        aux.command(done_accum, :) = 0.0;
        aux.motor(done_accum, :) = previous_state.motor(done_accum, :);
        aux.thrust(done_accum, :) = 0.0;
        aux.torque(done_accum, :) = 0.0;
    end

    [reward, reward_terms] = l2f_reward(candidate_state, aux.command, previous_state.previous_action, reward_cfg, world, t);
    done_step = l2f_termination(candidate_state, reward_cfg, sim_cfg, world, t);
    done_accum = done_accum | done_step;

    logs.action(step, :, :) = reshape(aux.command, 1, batch_size, 4);
    logs.motor(step, :, :) = reshape(aux.motor, 1, batch_size, 4);
    logs.thrust(step, :, :) = reshape(aux.thrust, 1, batch_size, 4);
    logs.torque(step, :, :) = reshape(aux.torque, 1, batch_size, 3);
    logs.env_force(step, :, :) = reshape(aux.env_force, 1, batch_size, 3);
    logs.env_torque(step, :, :) = reshape(aux.env_torque, 1, batch_size, 3);
    logs.reward(step, :) = reward;
    logs.done(step, :) = done_accum;
    logs.reward_terms.position(step, :) = reward_terms.position;
    logs.reward_terms.velocity(step, :) = reward_terms.velocity;
    logs.reward_terms.omega(step, :) = reward_terms.omega;
    logs.reward_terms.action(step, :) = reward_terms.action;
    logs.reward_terms.smooth(step, :) = reward_terms.smooth;

    state = candidate_state;
    logs = write_state_log(logs, state, step + 1);

    if sim_cfg.live_plot && (mod(step, max(sim_cfg.live_plot_every, 1)) == 0 || step == horizon)
        monitor = l2f_live_plot('update', monitor, logs, step);
    end

    if sim_cfg.stop_when_all_done && all(done_accum)
        logs = fill_tail_logs(logs, state, step, horizon);
        break;
    end
end
end

function logs = l2f_rollout_task(task, controller, sim_cfg)
params = l2f_get_field_or(task, 'params', l2f_default_params());
world = l2f_get_field_or(task, 'world', l2f_make_world('empty'));
sensors = l2f_get_field_or(task, 'sensors', l2f_make_sensors('ideal'));
reward_cfg = l2f_get_field_or(task, 'reward_cfg', l2f_default_reward_cfg());
task_sim_cfg = l2f_get_field_or(task, 'sim_cfg', l2f_default_sim_cfg());
if nargin >= 3 && ~isempty(sim_cfg)
    task_sim_cfg = merge_struct(task_sim_cfg, sim_cfg);
end
if isfield(task, 'uav_cfg')
    task_sim_cfg.uav_cfg = task.uav_cfg;
end
task_sim_cfg.task = task;
if nargin < 2 || isempty(controller)
    controller = [];
end
logs = l2f_rollout(params, world, sensors, controller, reward_cfg, task_sim_cfg);
logs.task = task;
end

function out = merge_struct(a, b)
out = a;
fields = fieldnames(b);
for i = 1:numel(fields)
    out.(fields{i}) = b.(fields{i});
end
end

function params = sync_params_from_sim_cfg(params, sim_cfg)
if isfield(sim_cfg, 'dt') && ~isempty(sim_cfg.dt)
    params.dt = sim_cfg.dt;
end
if isfield(sim_cfg, 'gravity') && ~isempty(sim_cfg.gravity)
    params.gravity = sim_cfg.gravity;
end
end

function uav_cfg = resolve_uav_cfg(sim_cfg)
uav_cfg = [];
if isfield(sim_cfg, 'uav_cfg') && ~isempty(sim_cfg.uav_cfg)
    uav_cfg = sim_cfg.uav_cfg;
elseif isfield(sim_cfg, 'uav_cfg_name') && ~isempty(sim_cfg.uav_cfg_name)
    uav_cfg = l2f_uav_cfg_library(sim_cfg.uav_cfg_name);
end
end

function logs = initialize_logs(horizon, batch_size, params, world, sensors, reward_cfg, sim_cfg, state)
logs = struct();
logs.time = (0:horizon).' * params.dt;
logs.position = zeros(horizon + 1, batch_size, 3);
logs.velocity = zeros(horizon + 1, batch_size, 3);
logs.rotation = zeros(horizon + 1, batch_size, 3, 3);
logs.euler_deg = zeros(horizon + 1, batch_size, 3);
logs.omega = zeros(horizon + 1, batch_size, 3);
logs.action = zeros(horizon, batch_size, 4);
logs.motor = zeros(horizon, batch_size, 4);
logs.thrust = zeros(horizon, batch_size, 4);
logs.torque = zeros(horizon, batch_size, 3);
logs.env_force = zeros(horizon, batch_size, 3);
logs.env_torque = zeros(horizon, batch_size, 3);
logs.controller_hidden_norm = nan(horizon, batch_size);
logs.controller_hidden_abs_max = nan(horizon, batch_size);
logs.done = false(horizon, batch_size);
logs.reward = zeros(horizon, batch_size);
logs.reward_terms = struct();
logs.reward_terms.position = zeros(horizon, batch_size);
logs.reward_terms.velocity = zeros(horizon, batch_size);
logs.reward_terms.omega = zeros(horizon, batch_size);
logs.reward_terms.action = zeros(horizon, batch_size);
logs.reward_terms.smooth = zeros(horizon, batch_size);
dyn0 = l2f_pack_dynamics(state, 1);
logs.dynamics = zeros(batch_size, numel(dyn0));
for i = 1:batch_size
    logs.dynamics(i, :) = l2f_pack_dynamics(state, i).';
end
logs.params = params;
logs.world = world;
logs.sensors = sensors;
logs.reward_cfg = reward_cfg;
logs.sim_cfg = sim_cfg;
end

function logs = write_state_log(logs, state, index)
batch_size = size(state.position, 1);
logs.position(index, :, :) = reshape(state.position, 1, batch_size, 3);
logs.velocity(index, :, :) = reshape(state.velocity, 1, batch_size, 3);
logs.omega(index, :, :) = reshape(state.omega, 1, batch_size, 3);
for i = 1:batch_size
    logs.rotation(index, i, :, :) = state.rotation(:, :, i);
end
logs.euler_deg(index, :, :) = reshape(l2f_matrix_to_euler_zyx(state.rotation) * 180 / pi, 1, batch_size, 3);
end

function logs = write_controller_log(logs, info, step, batch_size)
if ~isstruct(info)
    return;
end
if isfield(info, 'hidden_norm')
    value = reshape(info.hidden_norm, 1, []);
    if numel(value) == batch_size
        logs.controller_hidden_norm(step, :) = value;
    end
end
if isfield(info, 'hidden_abs_max')
    value = reshape(info.hidden_abs_max, 1, []);
    if numel(value) == batch_size
        logs.controller_hidden_abs_max(step, :) = value;
    end
end
end

function [obs, sensor_buffer] = l2f_apply_sensors(state, sensors, t, sensor_buffer)
current_obs = l2f_sensor_model(state, sensors, t);
delay_steps = 0;
if isfield(sensors, 'delay_steps')
    delay_steps = max(round(sensors.delay_steps), 0);
end
if delay_steps == 0
    obs = current_obs;
    sensor_buffer = {};
    return;
end
if isempty(sensor_buffer)
    sensor_buffer = cell(1, delay_steps + 1);
    for i = 1:(delay_steps + 1)
        sensor_buffer{i} = current_obs;
    end
else
    sensor_buffer = [{current_obs}, sensor_buffer(1:end-1)];
end
obs = sensor_buffer{delay_steps + 1};
end

function [action, info] = l2f_controller_action(controller, obs, state, params, world, t)
info = struct();
if isa(controller, 'function_handle')
    try
        n = nargin(controller);
    catch
        n = 5;
    end
    try
        out_count = nargout(controller);
    catch
        out_count = 1;
    end
    want_info = out_count >= 2 || out_count < 0;
    if n == 5 || n < 0
        if want_info
            [action, info] = controller(obs, state, params, world, t);
        else
            action = controller(obs, state, params, world, t);
        end
    elseif n == 3
        if want_info
            [action, info] = controller(state, params, t);
        else
            action = controller(state, params, t);
        end
    elseif n == 2
        if want_info
            [action, info] = controller(obs, t);
        else
            action = controller(obs, t);
        end
    elseif n == 1
        if want_info
            [action, info] = controller(obs);
        else
            action = controller(obs);
        end
    else
        error('Controller function must accept 1, 2, 3, or 5 inputs.');
    end
elseif isstruct(controller) && isfield(controller, 'fn')
    [action, info] = l2f_controller_action(controller.fn, obs, state, params, world, t);
else
    error('controller must be a function handle or a struct with field fn.');
end
end

function action = ensure_action_shape(action, batch_size)
if isvector(action) && numel(action) == 4
    action = reshape(action, 1, 4);
end
if size(action, 1) == 1 && batch_size > 1
    action = repmat(action, batch_size, 1);
end
if size(action, 1) ~= batch_size || size(action, 2) ~= 4
    error('Controller action must be N x 4.');
end
action = min(max(action, -1.0), 1.0);
end

function [reward, terms] = l2f_reward(state, action, previous_action, cfg, world, t)
ref = l2f_reference(world, t, size(state.position, 1));
position_error = state.position - ref.position;
velocity_error = state.velocity - ref.velocity;
omega_error = state.omega - ref.omega;
position_cost = cfg.w_position * sum((position_error / cfg.p_scale) .* (position_error / cfg.p_scale), 2).';
velocity_cost = cfg.w_velocity * sum((velocity_error / cfg.v_scale) .* (velocity_error / cfg.v_scale), 2).';
omega_cost = cfg.w_omega * sum((omega_error / cfg.omega_scale) .* (omega_error / cfg.omega_scale), 2).';
action_cost = cfg.w_action * (sum(action .* action, 2).' / size(action, 2));
delta = action - previous_action;
smooth_cost = cfg.w_smooth * (sum(delta .* delta, 2).' / size(delta, 2));
reward = -(position_cost + velocity_cost + omega_cost + action_cost + smooth_cost);
terms = struct();
terms.position = position_cost;
terms.velocity = velocity_cost;
terms.omega = omega_cost;
terms.action = action_cost;
terms.smooth = smooth_cost;
end

function done = l2f_termination(state, ~, sim_cfg, world, t)
batch_size = size(state.position, 1);
done = false(1, batch_size);
ref = l2f_reference(world, t, batch_size);
position_error = state.position - ref.position;
velocity_error = state.velocity - ref.velocity;
omega_error = state.omega - ref.omega;
position_norm = sqrt(sum(position_error .* position_error, 2)).';
velocity_norm = sqrt(sum(velocity_error .* velocity_error, 2)).';
omega_norm = sqrt(sum(omega_error .* omega_error, 2)).';
if sim_cfg.terminate_on_bounds
    done = done | position_norm > sim_cfg.max_position_norm;
    done = done | velocity_norm > sim_cfg.max_velocity_norm;
    done = done | omega_norm > sim_cfg.max_omega_norm;
end
if sim_cfg.terminate_on_nonfinite
    done = done | ~all(isfinite(state.position), 2).';
    done = done | ~all(isfinite(state.velocity), 2).';
    done = done | ~all(isfinite(state.omega), 2).';
    done = done | ~reshape(all(all(isfinite(state.rotation), 1), 2), 1, batch_size);
end
end

function out = select_state(old_state, new_state, use_new)
out = new_state;
mask = ~use_new(:);
if ~any(mask)
    return;
end
out.position(mask, :) = old_state.position(mask, :);
out.velocity(mask, :) = old_state.velocity(mask, :);
out.omega(mask, :) = old_state.omega(mask, :);
out.motor(mask, :) = old_state.motor(mask, :);
out.previous_action(mask, :) = old_state.previous_action(mask, :);
for i = find(mask).'
    out.rotation(:, :, i) = old_state.rotation(:, :, i);
end
end

function logs = fill_tail_logs(logs, state, step, horizon)
if step >= horizon
    return;
end
for i = (step + 2):(horizon + 1)
    logs = write_state_log(logs, state, i);
end
for i = (step + 1):horizon
    logs.done(i, :) = true;
end
end
