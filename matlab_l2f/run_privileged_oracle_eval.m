function run_privileged_oracle_eval(artifact_path, scenario_csv, output_csv, horizon)
%RUN_PRIVILEGED_ORACLE_EVAL Validate privileged teachers/direct shooting in MATLAB.

artifact = load(artifact_path);
scenarios = readtable(scenario_csv, 'VariableNamingRule', 'preserve');
if nargin < 4 || isempty(horizon)
    horizon = 10000;
end
params = l2f_default_params('dynamics_profile', 'physical-broad', 'broad_sampler', 'physical');
world = l2f_make_world('empty');
rows = table();

teacher_ids = round(artifact.teacher_scenario_id(:));
if ~isempty(teacher_ids)
    state = scenario_state(scenarios, teacher_ids);
    result = evaluate_teacher(artifact, state, params, world, horizon);
    rows = result_table(teacher_ids, 'teacher', result);
end

shooting_ids = round(artifact.shooting_scenario_id(:));
if ~isempty(shooting_ids)
    state = scenario_state(scenarios, shooting_ids);
    result = evaluate_shooting(artifact, state, params, world, horizon);
    shooting_rows = result_table(shooting_ids, 'direct_shooting', result);
    rows = [rows; shooting_rows]; %#ok<AGROW>
end

parent = fileparts(output_csv);
if ~isempty(parent) && ~isfolder(parent)
    mkdir(parent);
end
writetable(rows, output_csv);
fprintf('saved MATLAB privileged-oracle validation: %s (%d rows)\n', output_csv, height(rows));
end

function result = evaluate_teacher(a, state, params, world, horizon)
count = size(state.position, 1);
capability = a.teacher_capability;
if size(capability, 1) ~= count
    error('teacher capability row count does not match teacher state count.');
end
controller = @(s, step) teacher_action(a, s, capability); %#ok<NASGU>
result = evaluate_controller(state, params, world, horizon, ...
    @(s, step) teacher_action(a, s, capability));
end

function result = evaluate_shooting(a, state, params, world, horizon)
actions = a.shooting_actions;
result = evaluate_controller(state, params, world, horizon, ...
    @(s, step) shooting_action(actions, step)); %#ok<INUSD>
end

function result = evaluate_controller(state, params, world, horizon, controller)
count = size(state.position, 1);
window_steps = 100;
required_fraction = 0.95;
buffer = false(window_steps, count);
alive = true(1, count);
fraction_h500 = zeros(count, 1);
steady_h500 = false(count, 1);
for step = 1:horizon
    action = controller(state, step);
    [state, ~] = l2f_step(state, action, params, world, (step - 1) * params.dt);
    finite = all(isfinite(state.position), 2) & all(isfinite(state.velocity), 2) ...
        & all(isfinite(state.omega), 2) & all(isfinite(state.motor), 2);
    alive = alive & finite.';
    success = vecnorm(state.position, 2, 2) < 0.05 ...
        & vecnorm(state.velocity, 2, 2) < 0.10 ...
        & vecnorm(state.omega, 2, 2) < 0.20 ...
        & alive.';
    slot = mod(step - 1, window_steps) + 1;
    buffer(slot, :) = success.';
    if step == 500
        fraction_h500 = mean(buffer, 1).';
        steady_h500 = fraction_h500 >= required_fraction & alive.';
    end
end
fraction_final = mean(buffer, 1).';
result = struct();
result.fraction_h500 = fraction_h500;
result.steady_h500 = steady_h500;
result.fraction_final = fraction_final;
result.steady_final = fraction_final >= required_fraction & alive.';
result.survival = alive.';
result.position_final = vecnorm(state.position, 2, 2);
result.velocity_final = vecnorm(state.velocity, 2, 2);
result.omega_final = vecnorm(state.omega, 2, 2);
end

function action = teacher_action(a, state, capability)
batch = size(state.position, 1);
rotation_flat = reshape(permute(state.rotation, [2, 1, 3]), 9, batch).';
physical = [state.position, state.velocity, rotation_flat, state.omega];
features = [physical, state.motor, state.previous_action, capability, state.external_force];
hidden = batched_linear(a.teacher_w1, features, a.teacher_b1);
hidden = leaky_relu(hidden, 0.05);
hidden = batched_linear(a.teacher_w2, hidden, a.teacher_b2);
hidden = leaky_relu(hidden, 0.05);
action = tanh(batched_linear(a.teacher_w_out, hidden, a.teacher_b_out));
end

function action = shooting_action(actions, step)
action_step = min(step, size(actions, 2));
action = reshape(actions(:, action_step, :), size(actions, 1), 4);
end

function y = batched_linear(weight, x, bias)
% weight is [batch,out,in], x is [batch,in], bias is [batch,out].
y = sum(weight .* reshape(x, size(x, 1), 1, size(x, 2)), 3) + bias;
end

function y = leaky_relu(x, slope)
y = max(x, 0.0) + slope .* min(x, 0.0);
end

function output = result_table(ids, controller_type, result)
restart = zeros(numel(ids), 1);
for i = 1:numel(ids)
    restart(i) = sum(ids(1:i) == ids(i)) - 1;
end
output = table( ...
    ids, repmat(string(controller_type), numel(ids), 1), restart, ...
    double(result.steady_h500), result.fraction_h500, ...
    double(result.steady_final), result.fraction_final, double(result.survival), ...
    result.position_final, result.velocity_final, result.omega_final, ...
    'VariableNames', {'scenario_id', 'controller_type', 'restart', ...
    'steady_H500', 'window_fraction_H500', 'steady_H10000', ...
    'window_fraction_H10000', 'survival', 'position_final', ...
    'velocity_final', 'omega_final'});
end

function state = scenario_state(source, ids)
source_ids = round(source.scenario_id);
indices = zeros(numel(ids), 1);
for i = 1:numel(ids)
    match = find(source_ids == ids(i), 1);
    if isempty(match)
        error('Scenario %d is absent from scenario CSV.', ids(i));
    end
    indices(i) = match;
end
t = source(indices, :);
count = height(t);
state = struct();
state.position = columns(t, 'position', 3);
state.velocity = columns(t, 'velocity', 3);
state.rotation = zeros(3, 3, count);
for r = 1:3
    for c = 1:3
        state.rotation(r, c, :) = reshape(t.(sprintf('rotation_%d%d', r - 1, c - 1)), 1, 1, count);
    end
end
state.omega = columns(t, 'omega', 3);
state.motor = columns(t, 'motor', 4);
state.previous_action = columns(t, 'previous_action', 4);
state.external_force = columns(t, 'external_force', 3);
state.thrust_coeff_c0 = columns(t, 'thrust_coeff_c0', 4);
state.thrust_coeff_c1 = columns(t, 'thrust_coeff_c1', 4);
state.thrust_coeff_c2 = columns(t, 'thrust_coeff_c2', 4);
scalar_names = { ...
    'mass', 'thrust_to_weight', 'torque_to_inertia', 'rotor_distance_factor', ...
    'inertia_factor', 'motor_time_rising', 'motor_time_falling', ...
    'rotor_torque_constant', 'cbrt_mass', 'force_std', 'arm_length', ...
    'inertia_x', 'inertia_y', 'inertia_z', 'alpha_roll_max', ...
    'alpha_pitch_max', 'alpha_yaw_max', 'eta_yaw', 'jz_over_jxy', ...
    'dt_alpha_roll_max', 'dt_alpha_yaw_max'};
for i = 1:numel(scalar_names)
    name = scalar_names{i};
    state.(name) = t.(name);
end
end

function values = columns(t, prefix, width)
values = zeros(height(t), width);
for i = 1:width
    values(:, i) = t.(sprintf('%s_%d', prefix, i - 1));
end
end
