function export_q3_phase1_consistency(weights_path, output_path, scenario_ids, horizon)
%EXPORT_Q3_PHASE1_CONSISTENCY Export a short exact-state trajectory.
%
% Diagnostics only: this uses the same observation, policy forward, integral
% update, and l2f_step functions as the formal streaming evaluator.

if nargin < 3 || isempty(scenario_ids)
    error('scenario_ids are required');
end
if nargin < 4 || isempty(horizon)
    horizon = 100;
end
scenario_ids = reshape(round(double(scenario_ids)), [], 1);
params = l2f_default_params('dynamics_profile', 'physical-broad', 'broad_sampler', 'physical');
world = l2f_make_world('empty');
sim_cfg = l2f_default_sim_cfg('batch_size', 1024, 'seed', 1007, ...
    'terminate_on_bounds', false, 'freeze_done', false);
full_state = l2f_reset(1024, params, 1007, [], sim_cfg);
state = select_scenarios(full_state, scenario_ids, 1024);
weights = l2f_prepare_motor_gru_weights(load(char(weights_path)));
batch_size = numel(scenario_ids);
hidden = zeros(batch_size, double(weights.hidden_dim), 'like', state.position);
integral_position = zeros(batch_size, 3, 'like', state.position);

position = zeros(horizon + 1, batch_size, 3, 'like', state.position);
velocity = zeros(horizon + 1, batch_size, 3, 'like', state.position);
rotation = zeros(horizon + 1, batch_size, 9, 'like', state.position);
omega = zeros(horizon + 1, batch_size, 3, 'like', state.position);
motor = zeros(horizon + 1, batch_size, 4, 'like', state.position);
action = zeros(horizon, batch_size, 4, 'like', state.position);
integral = zeros(horizon + 1, batch_size, 3, 'like', state.position);
hidden_log = zeros(horizon + 1, batch_size, double(weights.hidden_dim), 'like', state.position);
[position(1, :, :), velocity(1, :, :), rotation(1, :, :), omega(1, :, :), motor(1, :, :)] = pack_state(state);

for step = 1:horizon
    t = (step - 1) * params.dt;
    [observation, observed_position] = l2f_observation( ...
        state, weights.observation_mode, integral_position, ...
        weights.integral_input_frame, weights.integral_input_multiplier);
    [command, hidden] = l2f_motor_gru_forward(weights, observation, hidden);
    integral_position = l2f_update_position_integral( ...
        integral_position, observed_position, params.dt, ...
        weights.integral_limit, weights.integral_leak);
    [state, aux] = l2f_step(state, command, params, world, t);
    action(step, :, :) = reshape(aux.command, 1, batch_size, 4);
    integral(step + 1, :, :) = reshape(integral_position, 1, batch_size, 3);
    hidden_log(step + 1, :, :) = reshape(hidden, 1, batch_size, double(weights.hidden_dim));
    [position(step + 1, :, :), velocity(step + 1, :, :), rotation(step + 1, :, :), ...
        omega(step + 1, :, :), motor(step + 1, :, :)] = pack_state(state);
end

parent = fileparts(char(output_path));
if ~isempty(parent) && ~isfolder(parent)
    mkdir(parent);
end
save(char(output_path), 'scenario_ids', 'position', 'velocity', 'rotation', ...
    'omega', 'motor', 'action', 'integral', 'hidden_log', '-v7');
fprintf('saved Phase 1 consistency trace: %s\n', char(output_path));
end

function state = select_scenarios(state, ids, original_batch)
names = fieldnames(state);
for index = 1:numel(names)
    name = names{index};
    value = state.(name);
    if strcmp(name, 'rotation') && ndims(value) == 3 && size(value, 3) == original_batch
        state.(name) = value(:, :, ids);
    elseif ~isscalar(value) && size(value, 1) == original_batch
        state.(name) = value(ids, :, :);
    end
end
end

function [position, velocity, rotation, omega, motor] = pack_state(state)
batch_size = size(state.position, 1);
position = reshape(state.position, 1, batch_size, 3);
velocity = reshape(state.velocity, 1, batch_size, 3);
omega = reshape(state.omega, 1, batch_size, 3);
motor = reshape(state.motor, 1, batch_size, 4);
rotation = zeros(1, batch_size, 9, 'like', state.position);
for sample = 1:batch_size
    rotation(1, sample, :) = reshape(state.rotation(:, :, sample).', 1, 1, 9);
end
end
