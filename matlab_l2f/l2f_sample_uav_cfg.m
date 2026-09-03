function dynamics = l2f_sample_uav_cfg(uav_cfg, batch_size, sim_cfg)
%L2F_SAMPLE_UAV_CFG Expand a UAV config into per-batch dynamics fields.

if nargin < 1 || isempty(uav_cfg)
    uav_cfg = l2f_default_uav_cfg();
end
if nargin < 3 || isempty(sim_cfg)
    sim_cfg = l2f_default_sim_cfg();
end

if iscell(uav_cfg)
    dynamics = sample_from_family_list(uav_cfg, batch_size, sim_cfg);
    return;
end
if isfield(uav_cfg, 'uav_family')
    dynamics = sample_from_family_list(uav_cfg.uav_family, batch_size, sim_cfg, uav_cfg);
    return;
end

gravity = l2f_get_field_or(sim_cfg, 'gravity', 9.80665);
rand_cfg = l2f_get_field_or(uav_cfg, 'randomization', struct());

mass = repmat(uav_cfg.mass, batch_size, 1);
mass = mass .* sample_scale(rand_cfg, 'mass_scale', batch_size, 1);

base_inertia = uav_cfg.inertia;
inertia_x = repmat(base_inertia(1, 1), batch_size, 1);
inertia_y = repmat(base_inertia(2, 2), batch_size, 1);
inertia_z = repmat(base_inertia(3, 3), batch_size, 1);
inertia_scale = sample_vector_scale(rand_cfg, 'inertia_scale', batch_size, 3);
inertia_x = inertia_x .* inertia_scale(:, 1);
inertia_y = inertia_y .* inertia_scale(:, 2);
inertia_z = inertia_z .* inertia_scale(:, 3);

rotor_pos_body = rep_geom(uav_cfg.rotor_pos_body, batch_size);
arm_scale = sample_scale(rand_cfg, 'arm_scale', batch_size, 1);
for i = 1:batch_size
    rotor_pos_body(i, :, :) = rotor_pos_body(i, :, :) * arm_scale(i);
end
rotor_axis_body = rep_geom(uav_cfg.rotor_axis_body, batch_size);
spin_dir = rep_row(uav_cfg.spin_dir, batch_size, 4);

hover = mass * gravity / 4.0;
base_hover = max(uav_cfg.mass * gravity / 4.0, 1.0e-12);
base_authority = sum(uav_cfg.thrust_coeff_c1) / max(4.0 * base_hover, 1.0e-12);
thrust_coeff_c0 = repmat(hover, 1, 4);
thrust_coeff_c1 = repmat(hover * base_authority, 1, 4);
thrust_coeff_c2 = rep_row(uav_cfg.thrust_coeff_c2, batch_size, 4);

if isfield(rand_cfg, 'thrust_to_weight')
    thrust_to_weight = sample_range(rand_cfg.thrust_to_weight, batch_size, 1);
    max_per_rotor = thrust_to_weight .* mass * gravity / 4.0;
    thrust_coeff_c1 = repmat(max(max_per_rotor - hover, 1.0e-9), 1, 4);
else
    thrust_to_weight = sum(thrust_coeff_c0 + thrust_coeff_c1 + thrust_coeff_c2, 2) ./ max(mass * gravity, 1.0e-12);
end

motor_time_rising = rep_row(l2f_get_field_or(uav_cfg, 'motor_tau_rise', 0.06), batch_size, 4);
motor_time_falling = rep_row(l2f_get_field_or(uav_cfg, 'motor_tau_fall', 0.06), batch_size, 4);
if isfield(rand_cfg, 'motor_tau_rise')
    motor_time_rising = sample_range(rand_cfg.motor_tau_rise, batch_size, 4);
end
if isfield(rand_cfg, 'motor_tau_fall')
    motor_time_falling = sample_range(rand_cfg.motor_tau_fall, batch_size, 4);
end

rotor_torque_constant = rep_row(uav_cfg.rotor_torque_constant, batch_size, 4);
rotor_torque_constant = rotor_torque_constant .* sample_vector_scale(rand_cfg, 'rotor_torque_constant_scale', batch_size, 4);
if isfield(rand_cfg, 'rotor_torque_constant')
    rotor_torque_constant = sample_range(rand_cfg.rotor_torque_constant, batch_size, 4);
end

external_force_std = l2f_get_field_or(rand_cfg, 'external_force_std', 0.0);
external_force_std_row = sample_range(external_force_std, batch_size, 3);
external_force = randn(batch_size, 3) .* external_force_std_row;
drag_linear = sample_vector_or_cfg(rand_cfg, uav_cfg, 'drag_linear', batch_size, 3);
drag_quadratic = sample_vector_or_cfg(rand_cfg, uav_cfg, 'drag_quadratic', batch_size, 3);
angular_drag_linear = sample_vector_or_cfg(rand_cfg, uav_cfg, 'angular_drag_linear', batch_size, 3);
angular_drag_quadratic = sample_vector_or_cfg(rand_cfg, uav_cfg, 'angular_drag_quadratic', batch_size, 3);

motor_deadzone = rep_row(l2f_get_field_or(uav_cfg, 'motor_deadzone', zeros(1, 4)), batch_size, 4);
motor_delay_steps = round(rep_row(l2f_get_field_or(uav_cfg, 'motor_delay_steps', zeros(1, 4)), batch_size, 4));
motor_health = rep_row(l2f_get_field_or(uav_cfg, 'motor_health', ones(1, 4)), batch_size, 4);
thrust_scale = rep_row(l2f_get_field_or(uav_cfg, 'thrust_scale', ones(1, 4)), batch_size, 4);
if isfield(rand_cfg, 'motor_health')
    motor_health = sample_range(rand_cfg.motor_health, batch_size, 4);
end
if isfield(rand_cfg, 'thrust_scale')
    thrust_scale = sample_range(rand_cfg.thrust_scale, batch_size, 4);
end

arm_length = estimate_arm_length(rotor_pos_body);
thrust_delta = max(max(thrust_coeff_c0 + thrust_coeff_c1 + thrust_coeff_c2, [], 2) ...
    - min(max(thrust_coeff_c0 - thrust_coeff_c1 + thrust_coeff_c2, 0.0), [], 2), 1.0e-9);
rotor_torque_scalar = sum(rotor_torque_constant, 2) / size(rotor_torque_constant, 2);
alpha_roll_max = arm_length .* thrust_delta ./ max(inertia_x, 1.0e-12);
alpha_pitch_max = arm_length .* thrust_delta ./ max(inertia_y, 1.0e-12);
alpha_yaw_max = rotor_torque_scalar .* (2.0 * thrust_delta) ./ max(inertia_z, 1.0e-12);

dynamics = struct();
dynamics.mass = mass;
dynamics.thrust_coeff_c0 = thrust_coeff_c0;
dynamics.thrust_coeff_c1 = thrust_coeff_c1;
dynamics.thrust_coeff_c2 = thrust_coeff_c2;
dynamics.external_force = external_force;
dynamics.thrust_to_weight = thrust_to_weight;
dynamics.torque_to_inertia = alpha_roll_max;
dynamics.rotor_distance_factor = arm_scale;
dynamics.inertia_factor = inertia_scale(:, 1);
dynamics.motor_time_rising = motor_time_rising;
dynamics.motor_time_falling = motor_time_falling;
dynamics.rotor_torque_constant = rotor_torque_constant;
dynamics.cbrt_mass = exp(log(max(mass, 1.0e-12)) / 3.0);
dynamics.force_std = sqrt(sum(external_force_std_row .* external_force_std_row, 2));
dynamics.arm_length = arm_length;
dynamics.inertia_x = inertia_x;
dynamics.inertia_y = inertia_y;
dynamics.inertia_z = inertia_z;
dynamics.alpha_roll_max = alpha_roll_max;
dynamics.alpha_pitch_max = alpha_pitch_max;
dynamics.alpha_yaw_max = alpha_yaw_max;
dynamics.eta_yaw = alpha_yaw_max ./ max(alpha_roll_max, 1.0e-9);
dynamics.jz_over_jxy = inertia_z ./ max(inertia_x, 1.0e-12);
dynamics.dt_alpha_roll_max = alpha_roll_max * l2f_get_field_or(sim_cfg, 'dt', 0.01);
dynamics.dt_alpha_yaw_max = alpha_yaw_max * l2f_get_field_or(sim_cfg, 'dt', 0.01);
dynamics.rotor_pos_body = rotor_pos_body;
dynamics.rotor_axis_body = rotor_axis_body;
dynamics.spin_dir = spin_dir;
dynamics.motor_deadzone = motor_deadzone;
dynamics.motor_delay_steps = motor_delay_steps;
dynamics.motor_health = motor_health;
dynamics.thrust_scale = thrust_scale;
dynamics.drag_linear = drag_linear;
dynamics.drag_quadratic = drag_quadratic;
dynamics.angular_drag_linear = angular_drag_linear;
dynamics.angular_drag_quadratic = angular_drag_quadratic;
dynamics.collision_radius = repmat(l2f_get_field_or(uav_cfg, 'collision_radius', 0.08), batch_size, 1);
dynamics.collision_height = repmat(l2f_get_field_or(uav_cfg, 'collision_height', 0.04), batch_size, 1);
dynamics.rotor_radius = repmat(l2f_get_field_or(uav_cfg, 'rotor_radius', 0.02), batch_size, 1);
battery_cfg = l2f_get_field_or(uav_cfg, 'battery', struct());
dynamics.battery_enabled = repmat(double(l2f_get_field_or(battery_cfg, 'enabled', false)), batch_size, 1);
dynamics.battery_voltage = repmat(l2f_get_field_or(battery_cfg, 'voltage', 1.0), batch_size, 1);
dynamics.battery_nominal_voltage = repmat(l2f_get_field_or(battery_cfg, 'nominal_voltage', 1.0), batch_size, 1);
dynamics.battery_internal_resistance = repmat(l2f_get_field_or(battery_cfg, 'internal_resistance', 0.0), batch_size, 1);
dynamics.battery_current_gain = repmat(l2f_get_field_or(battery_cfg, 'current_gain', 0.0), batch_size, 1);
dynamics.battery_capacity_gain = repmat(l2f_get_field_or(battery_cfg, 'capacity_gain', 0.0), batch_size, 1);
dynamics.battery_min_scale = repmat(l2f_get_field_or(battery_cfg, 'min_scale', 0.6), batch_size, 1);
dynamics.allow_negative_thrust = repmat(double(l2f_get_field_or(uav_cfg, 'allow_negative_thrust', false)), batch_size, 1);
dynamics.uav_name = l2f_get_field_or(uav_cfg, 'name', 'custom');
end

function dynamics = sample_from_family_list(family, batch_size, sim_cfg, template)
if nargin < 4
    template = struct();
end
dynamics = [];
for i = 1:batch_size
    idx = floor(rand() * numel(family)) + 1;
    cfg = l2f_uav_cfg_library(family{idx});
    if isfield(template, 'randomization')
        cfg.randomization = template.randomization;
    end
    one = l2f_sample_uav_cfg(cfg, 1, sim_cfg);
    if i == 1
        dynamics = one;
    else
        dynamics = append_dynamics(dynamics, one);
    end
end
end

function out = append_dynamics(a, b)
out = a;
fields = fieldnames(b);
for i = 1:numel(fields)
    name = fields{i};
    if isnumeric(b.(name)) || islogical(b.(name))
        out.(name) = cat(1, out.(name), b.(name));
    end
end
end

function values = rep_row(value, batch_size, width)
if isempty(value)
    value = zeros(1, width);
end
if isscalar(value)
    values = repmat(value, batch_size, width);
elseif size(value, 1) == batch_size && size(value, 2) == width
    values = value;
else
    values = repmat(reshape(value, 1, width), batch_size, 1);
end
end

function geom = rep_geom(value, batch_size)
if ndims(value) == 3 && size(value, 1) == batch_size
    geom = value;
else
    geom = zeros(batch_size, 4, 3);
    for i = 1:batch_size
        geom(i, :, :) = value;
    end
end
end

function scale = sample_scale(rand_cfg, name, batch_size, width)
if isfield(rand_cfg, name)
    scale = sample_range(rand_cfg.(name), batch_size, width);
else
    scale = ones(batch_size, width);
end
end

function scale = sample_vector_scale(rand_cfg, name, batch_size, width)
scale = sample_scale(rand_cfg, name, batch_size, width);
if size(scale, 2) == 1 && width > 1
    scale = repmat(scale, 1, width);
end
end

function values = sample_vector_or_cfg(rand_cfg, uav_cfg, name, batch_size, width)
if isfield(rand_cfg, name)
    values = sample_range(rand_cfg.(name), batch_size, width);
else
    values = rep_row(l2f_get_field_or(uav_cfg, name, zeros(1, width)), batch_size, width);
end
end

function values = sample_range(range, batch_size, width)
if isscalar(range)
    values = repmat(range, batch_size, width);
elseif size(range, 2) == 2 && size(range, 1) == 1
    values = range(1) + rand(batch_size, width) * (range(2) - range(1));
elseif size(range, 2) == 2 && size(range, 1) == width
    values = zeros(batch_size, width);
    for j = 1:width
        values(:, j) = range(j, 1) + rand(batch_size, 1) * (range(j, 2) - range(j, 1));
    end
else
    values = rep_row(range, batch_size, width);
end
end

function arm_length = estimate_arm_length(rotor_pos_body)
batch_size = size(rotor_pos_body, 1);
arm_length = zeros(batch_size, 1);
for i = 1:batch_size
    total = 0.0;
    for m = 1:size(rotor_pos_body, 2)
        r = reshape(rotor_pos_body(i, m, :), 1, 3);
        total = total + sqrt(sum(r .* r));
    end
    arm_length(i) = total / size(rotor_pos_body, 2);
end
end
