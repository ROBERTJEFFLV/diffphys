function dyn = l2f_pack_dynamics(state, sample_index)
%L2F_PACK_DYNAMICS Flatten per-sample dynamics for logs and Simulink.

if nargin < 2
    sample_index = 1;
end
i = sample_index;
dyn = [
    state.mass(i);
    state.thrust_coeff_c0(i, :).';
    state.thrust_coeff_c1(i, :).';
    state.thrust_coeff_c2(i, :).';
    state.external_force(i, :).';
    state.thrust_to_weight(i);
    state.torque_to_inertia(i);
    state.rotor_distance_factor(i);
    state.inertia_factor(i);
    scalar_field(state, 'motor_time_rising', i, 0.06);
    scalar_field(state, 'motor_time_falling', i, 0.06);
    scalar_field(state, 'rotor_torque_constant', i, 0.02);
    state.cbrt_mass(i);
    state.force_std(i);
    state.arm_length(i);
    state.inertia_x(i);
    state.inertia_y(i);
    state.inertia_z(i);
    state.alpha_roll_max(i);
    state.alpha_pitch_max(i);
    state.alpha_yaw_max(i);
    state.eta_yaw(i);
    state.jz_over_jxy(i);
    state.dt_alpha_roll_max(i);
    state.dt_alpha_yaw_max(i);
    flatten_geom(get_geom(state, 'rotor_pos_body', i, default_rotor_pos(state, i)));
    flatten_geom(get_geom(state, 'rotor_axis_body', i, repmat([0 0 1], 4, 1)));
    row_field(state, 'spin_dir', i, 4, [1 -1 1 -1]).';
    row_field(state, 'rotor_torque_constant', i, 4, 0.02).';
    row_field(state, 'motor_time_rising', i, 4, 0.06).';
    row_field(state, 'motor_time_falling', i, 4, 0.06).';
    row_field(state, 'motor_deadzone', i, 4, 0.0).';
    row_field(state, 'motor_delay_steps', i, 4, 0.0).';
    row_field(state, 'motor_health', i, 4, 1.0).';
    row_field(state, 'thrust_scale', i, 4, 1.0).';
    row_field(state, 'drag_linear', i, 3, [0 0 0]).';
    row_field(state, 'drag_quadratic', i, 3, [0 0 0]).';
    row_field(state, 'angular_drag_linear', i, 3, [0 0 0]).';
    row_field(state, 'angular_drag_quadratic', i, 3, [0 0 0]).';
    scalar_field(state, 'collision_radius', i, 0.08);
    scalar_field(state, 'collision_height', i, 0.04);
    scalar_field(state, 'rotor_radius', i, 0.02);
    scalar_field(state, 'battery_enabled', i, 0.0);
    scalar_field(state, 'battery_voltage', i, 1.0);
    scalar_field(state, 'battery_nominal_voltage', i, 1.0);
    scalar_field(state, 'battery_internal_resistance', i, 0.0);
    scalar_field(state, 'battery_current_gain', i, 0.0);
    scalar_field(state, 'battery_capacity_gain', i, 0.0);
    scalar_field(state, 'battery_min_scale', i, 0.6);
    scalar_field(state, 'allow_negative_thrust', i, 0.0)
];
end

function value = scalar_field(state, name, i, default_value)
if ~isfield(state, name)
    value = default_value;
    return;
end
data = state.(name);
if isscalar(data)
    value = data;
elseif size(data, 1) >= i
    value = data(i, 1);
else
    value = default_value;
end
end

function row = row_field(state, name, i, width, default_value)
if isfield(state, name)
    data = state.(name);
else
    data = default_value;
end
if isscalar(data)
    row = repmat(data, 1, width);
elseif size(data, 1) >= i && size(data, 2) >= width
    row = data(i, 1:width);
elseif size(data, 1) >= i && size(data, 2) == 1
    row = repmat(data(i, 1), 1, width);
else
    row = reshape(data, 1, width);
end
end

function geom = get_geom(state, name, i, default_value)
if isfield(state, name)
    data = state.(name);
    geom = reshape(data(i, :, :), size(data, 2), size(data, 3));
else
    geom = default_value;
end
end

function flat = flatten_geom(geom)
flat = reshape(geom.', numel(geom), 1);
end

function pos = default_rotor_pos(state, i)
arm = scalar_field(state, 'arm_length', i, 0.046);
pos = [
     arm, 0, 0;
     0, arm, 0;
    -arm, 0, 0;
     0,-arm, 0
];
end
