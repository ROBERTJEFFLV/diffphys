function export_dynamic_hard_scenarios(output_path, scenario_ids, seed, batch_size)
%EXPORT_DYNAMIC_HARD_SCENARIOS Export exact MATLAB physical-broad reset states.

if nargin < 2 || isempty(scenario_ids)
    scenario_ids = [53 69 97 147 161 237 351 355 396 467 514 569 ...
        608 636 650 677 682 732 768 834 864 904];
end
if nargin < 3 || isempty(seed)
    seed = 1007;
end
if nargin < 4 || isempty(batch_size)
    batch_size = 1024;
end
scenario_ids = reshape(round(double(scenario_ids)), [], 1);
if any(scenario_ids < 1 | scenario_ids > batch_size)
    error('scenario IDs must be in [1, batch_size].');
end

params = l2f_default_params( ...
    'dynamics_profile', 'physical-broad', ...
    'broad_sampler', 'physical');
sim_cfg = l2f_default_sim_cfg( ...
    'batch_size', batch_size, ...
    'seed', seed, ...
    'terminate_on_bounds', false, ...
    'freeze_done', false);
state = l2f_reset(batch_size, params, seed, [], sim_cfg);
i = scenario_ids;
data = struct();
data.scenario_id = i;
data.position_0 = state.position(i, 1);
data.position_1 = state.position(i, 2);
data.position_2 = state.position(i, 3);
data.velocity_0 = state.velocity(i, 1);
data.velocity_1 = state.velocity(i, 2);
data.velocity_2 = state.velocity(i, 3);
for row = 1:3
    for column = 1:3
        values = reshape(state.rotation(row, column, i), [], 1);
        data.(sprintf('rotation_%d%d', row - 1, column - 1)) = values;
    end
end
data.omega_0 = state.omega(i, 1);
data.omega_1 = state.omega(i, 2);
data.omega_2 = state.omega(i, 3);
for motor_index = 1:4
    suffix = sprintf('%d', motor_index - 1);
    data.(['motor_' suffix]) = state.motor(i, motor_index);
    data.(['previous_action_' suffix]) = state.previous_action(i, motor_index);
    data.(['thrust_coeff_c0_' suffix]) = state.thrust_coeff_c0(i, motor_index);
    data.(['thrust_coeff_c1_' suffix]) = state.thrust_coeff_c1(i, motor_index);
    data.(['thrust_coeff_c2_' suffix]) = state.thrust_coeff_c2(i, motor_index);
end
data.external_force_0 = state.external_force(i, 1);
data.external_force_1 = state.external_force(i, 2);
data.external_force_2 = state.external_force(i, 3);
scalar_names = { ...
    'mass', 'thrust_to_weight', 'torque_to_inertia', ...
    'rotor_distance_factor', 'inertia_factor', 'cbrt_mass', 'force_std', ...
    'arm_length', 'inertia_x', 'inertia_y', 'inertia_z', ...
    'alpha_roll_max', 'alpha_pitch_max', 'alpha_yaw_max', 'eta_yaw', ...
    'jz_over_jxy', 'dt_alpha_roll_max', 'dt_alpha_yaw_max'};
for name_index = 1:numel(scalar_names)
    name = scalar_names{name_index};
    data.(name) = state.(name)(i, 1);
end
data.motor_time_rising = state.motor_time_rising(i, 1);
data.motor_time_falling = state.motor_time_falling(i, 1);
data.rotor_torque_constant = state.rotor_torque_constant(i, 1);

output_path = char(output_path);
parent = fileparts(output_path);
if ~isempty(parent) && ~isfolder(parent)
    mkdir(parent);
end
writetable(struct2table(data), output_path);
fprintf('saved %d exact dynamic-hard scenarios: %s\n', numel(i), output_path);
end
