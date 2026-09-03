function dynamics = l2f_sample_dynamics(params, batch_size)
%L2F_SAMPLE_DYNAMICS Match env_l2f fixed and broad randomization profiles.

profile = lower(params.dynamics_profile);
if strcmp(profile, 'fixed')
    dynamics = fixed_dynamics(params, batch_size);
elseif strcmp(profile, 'raptor-broad') || (strcmp(profile, 'broad') && strcmp(params.broad_sampler, 'legacy'))
    dynamics = raptor_broad_dynamics(params, batch_size);
elseif strcmp(profile, 'physical-broad') || (strcmp(profile, 'broad') && strcmp(params.broad_sampler, 'physical'))
    dynamics = physical_broad_dynamics(params, batch_size);
elseif strcmp(profile, 'physical-fit-broad') || strcmp(profile, 'physical-fit') || (strcmp(profile, 'broad') && strcmp(params.broad_sampler, 'physical-fit'))
    dynamics = physical_fit_broad_dynamics(params, batch_size);
else
    error('Unknown dynamics profile: %s', params.dynamics_profile);
end
end

function dynamics = fixed_dynamics(p, batch)
mass = repmat(p.mass, batch, 1);
c0 = mass * p.gravity / 4;
c1 = c0 * p.motor_authority;
c2 = zeros(batch, 1);
thrust_to_weight = repmat(p.motor_authority + 1.0, batch, 1);
torque_to_inertia = repmat(250.0, batch, 1);
rotor_distance_factor = ones(batch, 1);
inertia_factor = ones(batch, 1);
motor_time_rising = repmat(p.motor_tau, batch, 1);
motor_time_falling = repmat(p.motor_tau, batch, 1);
rotor_torque_constant = repmat(0.02, batch, 1);
cbrt_mass = cbrt_positive(mass);
force_std = zeros(batch, 1);
arm_length = repmat(p.arm_length, batch, 1);
inertia_x = repmat(p.inertia_x, batch, 1);
inertia_y = repmat(p.inertia_y, batch, 1);
inertia_z = repmat(p.inertia_z, batch, 1);
external_force = zeros(batch, 3);
max_thrust_per_rotor = c0 + c1 + c2;
min_thrust_per_rotor = max(c0 - c1 + c2, 0);
thrust_delta = max(max_thrust_per_rotor - min_thrust_per_rotor, 1.0e-9);
roll_torque_max = arm_length .* thrust_delta;
yaw_mix_thrust_max = 2.0 * thrust_delta;
alpha_roll_max = roll_torque_max ./ max(inertia_x, 1.0e-12);
alpha_pitch_max = roll_torque_max ./ max(inertia_y, 1.0e-12);
alpha_yaw_max = rotor_torque_constant .* yaw_mix_thrust_max ./ max(inertia_z, 1.0e-12);
eta_yaw = alpha_yaw_max ./ max(alpha_roll_max, 1.0e-9);
jz_over_jxy = inertia_z ./ max(inertia_x, 1.0e-12);
dynamics = make_dynamics(mass, c0, c1, c2, external_force, thrust_to_weight, torque_to_inertia, ...
    rotor_distance_factor, inertia_factor, motor_time_rising, motor_time_falling, rotor_torque_constant, ...
    cbrt_mass, force_std, arm_length, inertia_x, inertia_y, inertia_z, alpha_roll_max, alpha_pitch_max, ...
    alpha_yaw_max, eta_yaw, jz_over_jxy, p.dt);
end

function dynamics = raptor_broad_dynamics(p, batch)
nominal_mass = p.mass;
nominal_c0 = nominal_mass * p.gravity / 4;
nominal_c1 = nominal_c0 * p.motor_authority;
nominal_c2 = 0;
max_thrust_nominal = (nominal_c0 + nominal_c1 * p.action_max + nominal_c2 * p.action_max * p.action_max) * 4;
thrust_to_weight_nominal = max_thrust_nominal / (nominal_mass * p.gravity);

thrust_to_weight = rand(batch, 1) * (5.0 - 1.5) + 1.5;
factor_thrust_to_weight = thrust_to_weight / thrust_to_weight_nominal;
relative_size_min = 0.27144176165949;
relative_size_max = 1.70997594667670;
cbrt_mass = rand(batch, 1) * (relative_size_max - relative_size_min) + relative_size_min;
mass = max(cbrt_mass .* cbrt_mass .* cbrt_mass, 1.0e-12);
scale_relative = cbrt_positive(mass / nominal_mass);
rotor_distance_factor = scale_relative .* reciprocal_factor(rand(batch, 1), 0.1);
arm_length = p.arm_length * rotor_distance_factor;

thrust_factor = factor_thrust_to_weight .* (mass / nominal_mass);
c0 = nominal_c0 * thrust_factor;
c1 = nominal_c1 * thrust_factor;
c2 = zeros(batch, 1);

max_thrust_per_rotor = thrust_to_weight .* mass * p.gravity / 4;
max_torque = sqrt(2.0) * abs(p.arm_length) * max_thrust_per_rotor;
torque_to_inertia_nominal = max_torque / p.inertia_x;
torque_to_inertia = rand(batch, 1) * (1200.0 - 40.0) + 40.0;
torque_factor = torque_to_inertia ./ torque_to_inertia_nominal;
inertia_factor = torque_factor ./ max(rotor_distance_factor, 1.0e-6);
inertia_x = p.inertia_x ./ max(inertia_factor, 1.0e-6);
inertia_y = p.inertia_y ./ max(inertia_factor, 1.0e-6);
inertia_z = p.inertia_z ./ max(inertia_factor, 1.0e-6);

rotor_torque_constant = rand(batch, 1) * (0.05 - 0.005) + 0.005;
motor_time_rising = rand(batch, 1) * (0.10 - 0.03) + 0.03;
motor_time_falling = rand(batch, 1) * (0.30 - 0.03) + 0.03;

min_thrust_per_rotor = max(c0 - c1 + c2, 0);
thrust_delta = max(max_thrust_per_rotor - min_thrust_per_rotor, 1.0e-9);
roll_torque_max = arm_length .* thrust_delta;
yaw_mix_thrust_max = 2.0 * thrust_delta;
alpha_roll_max = roll_torque_max ./ max(inertia_x, 1.0e-12);
alpha_pitch_max = roll_torque_max ./ max(inertia_y, 1.0e-12);
alpha_yaw_max = rotor_torque_constant .* yaw_mix_thrust_max ./ max(inertia_z, 1.0e-12);
eta_yaw = alpha_yaw_max ./ max(alpha_roll_max, 1.0e-9);
jz_over_jxy = inertia_z ./ max(inertia_x, 1.0e-12);

surplus = max(thrust_to_weight - 1.0, 0);
multiple = rand(batch, 1) .* (surplus * 0.3);
force_std = multiple .* thrust_to_weight .* mass / 3.0;
external_force = randn(batch, 3) .* force_std;

dynamics = make_dynamics(mass, c0, c1, c2, external_force, thrust_to_weight, torque_to_inertia, ...
    rotor_distance_factor, inertia_factor, motor_time_rising, motor_time_falling, rotor_torque_constant, ...
    cbrt_mass, force_std, arm_length, inertia_x, inertia_y, inertia_z, alpha_roll_max, alpha_pitch_max, ...
    alpha_yaw_max, eta_yaw, jz_over_jxy, p.dt);
end

function dynamics = physical_broad_dynamics(p, batch)
relative_size_min = 0.27144176165949;
relative_size_max = 1.70997594667670;
cbrt_mass = rand(batch, 1) * (relative_size_max - relative_size_min) + relative_size_min;
mass = max(cbrt_mass .* cbrt_mass .* cbrt_mass, 1.0e-12);
scale_relative = cbrt_positive(mass / p.mass);
rotor_distance_factor = scale_relative .* reciprocal_factor(rand(batch, 1), 0.1);
arm_length = p.arm_length * rotor_distance_factor;

thrust_to_weight = rand(batch, 1) * (5.0 - 1.5) + 1.5;
hover_thrust = mass * p.gravity / 4;
max_thrust_per_rotor = thrust_to_weight .* mass * p.gravity / 4;
min_thrust_per_rotor = max(2.0 * hover_thrust - max_thrust_per_rotor, 0);
thrust_delta = max(max_thrust_per_rotor - min_thrust_per_rotor, 1.0e-9);
c0 = hover_thrust;
c1 = max_thrust_per_rotor - hover_thrust;
c2 = zeros(batch, 1);

alpha_roll_max = rand(batch, 1) * (600.0 - 40.0) + 40.0;
alpha_pitch_max = alpha_roll_max;
eta_yaw = rand(batch, 1) * (0.15 - 0.05) + 0.05;
jz_over_jxy = rand(batch, 1) * (2.8 - 1.5) + 1.5;
alpha_yaw_max = eta_yaw .* alpha_roll_max;

roll_torque_max = arm_length .* thrust_delta;
inertia_x = roll_torque_max ./ max(alpha_roll_max, 1.0e-9);
inertia_y = roll_torque_max ./ max(alpha_pitch_max, 1.0e-9);
inertia_z = jz_over_jxy .* inertia_x;
yaw_mix_thrust_max = 2.0 * thrust_delta;
rotor_torque_constant = alpha_yaw_max .* inertia_z ./ max(yaw_mix_thrust_max, 1.0e-9);
torque_to_inertia = alpha_roll_max;
inertia_factor = p.inertia_x ./ max(inertia_x, 1.0e-12);
motor_time_rising = rand(batch, 1) * (0.10 - 0.03) + 0.03;
motor_time_falling = rand(batch, 1) * (0.30 - 0.03) + 0.03;

surplus = max(thrust_to_weight - 1.0, 0);
multiple = rand(batch, 1) .* (surplus * 0.3);
force_std = multiple .* thrust_to_weight .* mass / 3.0;
external_force = randn(batch, 3) .* force_std;

dynamics = make_dynamics(mass, c0, c1, c2, external_force, thrust_to_weight, torque_to_inertia, ...
    rotor_distance_factor, inertia_factor, motor_time_rising, motor_time_falling, rotor_torque_constant, ...
    cbrt_mass, force_std, arm_length, inertia_x, inertia_y, inertia_z, alpha_roll_max, alpha_pitch_max, ...
    alpha_yaw_max, eta_yaw, jz_over_jxy, p.dt);
end

function dynamics = physical_fit_broad_dynamics(p, batch)
%PHYSICAL_FIT_BROAD_DYNAMICS Physically fitted broad sampler.
% Root variables are sampled and remaining quantities are derived from
% rigid-body equations, instead of sampling roll/yaw authority independently.

[u_mass, u_tw, u_k, u_tau] = joint_stratified_units(batch, 4, 4);
u_misc = rand(batch, 1);
u_misc2 = rand(batch, 1);
u_misc3 = rand(batch, 1);

mass_min = 0.02;
mass_max = 5.0;
log_mass = log(mass_min) + u_mass .* (log(mass_max) - log(mass_min));
mass = max(exp(log_mass), 1.0e-12);
cbrt_mass = cbrt_positive(mass);

arm_center = 0.21257 .* mass .^ 0.5028;
arm_length = arm_center .* reciprocal_factor(u_misc, 0.20);
arm_length = min(max(arm_length, 0.028), 0.50);
rotor_distance_factor = arm_length / p.arm_length;

tw_center = 3.19478 .* mass .^ 0.09285;
tw_low = min(max(tw_center / 1.55, 1.45), 5.50);
tw_high = min(max(tw_center * 1.70, 1.45), 5.50);
thrust_to_weight = tw_low + u_tw .* (tw_high - tw_low);
thrust_to_weight = min(max(thrust_to_weight, 1.45), 5.50);

hover_thrust = mass * p.gravity / 4;
max_thrust_per_rotor = thrust_to_weight .* mass * p.gravity / 4;
min_thrust_per_rotor = max(2.0 * hover_thrust - max_thrust_per_rotor, 0);
thrust_delta = max(max_thrust_per_rotor - min_thrust_per_rotor, 1.0e-9);
c0 = hover_thrust;
c1 = max_thrust_per_rotor - hover_thrust;
c2 = zeros(batch, 1);

kxy_center = 0.19013 .* mass .^ 0.23165;
kxy = kxy_center .* reciprocal_factor(u_k, 0.50);
kxy = min(max(kxy, 0.045), 0.50);
% Jx == Jy here, so the rigid-body triangle inequality requires Jz/Jxy <= 2.
% The conservative upper end covers all named vehicle anchors.
jz_over_jxy = 1.45 + u_misc2 .* (1.95 - 1.45);
inertia_x = max(kxy .* mass .* arm_length .* arm_length, 1.0e-12);
inertia_y = inertia_x;
inertia_z = jz_over_jxy .* inertia_x;

alpha_raw = arm_length .* thrust_delta ./ max(inertia_x, 1.0e-12);
alpha_roll_max = min(max(alpha_raw, 35.0), 2200.0);
inertia_scale = alpha_raw ./ max(alpha_roll_max, 1.0e-9);
inertia_x = inertia_x .* inertia_scale;
inertia_y = inertia_y .* inertia_scale;
inertia_z = inertia_z .* inertia_scale;
alpha_pitch_max = alpha_roll_max;

km_center = 0.01523 .* mass .^ (-0.09015);
rotor_torque_constant = km_center .* reciprocal_factor(u_misc3, 0.45);
rotor_torque_constant = min(max(rotor_torque_constant, 0.006), 0.035);
yaw_mix_thrust_max = 2.0 * thrust_delta;
alpha_yaw_raw = rotor_torque_constant .* yaw_mix_thrust_max ./ max(inertia_z, 1.0e-12);
eta_raw = alpha_yaw_raw ./ max(alpha_roll_max, 1.0e-9);
eta_yaw = min(max(eta_raw, 0.02), 1.00);
rotor_torque_constant = rotor_torque_constant .* eta_yaw ./ max(eta_raw, 1.0e-9);
alpha_yaw_max = eta_yaw .* alpha_roll_max;

tau_center = 0.10658 .* mass .^ 0.18987;
motor_time_rising = tau_center .* reciprocal_factor(u_tau, 0.35);
motor_time_rising = min(max(motor_time_rising, 0.025), 0.18);
fall_factor = 1.0 + rand(batch, 1) .* (2.6 - 1.0);
motor_time_falling = min(max(motor_time_rising .* fall_factor, 0.03), 0.35);

disturbance_acc_std = rand(batch, 1) * 0.35 .* max(thrust_to_weight - 1.0, 0.0);
force_std = mass .* disturbance_acc_std;
external_force = randn(batch, 3) .* force_std;

torque_to_inertia = alpha_roll_max;
inertia_factor = p.inertia_x ./ max(inertia_x, 1.0e-12);

dynamics = make_dynamics(mass, c0, c1, c2, external_force, thrust_to_weight, torque_to_inertia, ...
    rotor_distance_factor, inertia_factor, motor_time_rising, motor_time_falling, rotor_torque_constant, ...
    cbrt_mass, force_std, arm_length, inertia_x, inertia_y, inertia_z, alpha_roll_max, alpha_pitch_max, ...
    alpha_yaw_max, eta_yaw, jz_over_jxy, p.dt);
end

function u = stratified_unit(batch, bins, offset)
idx = (0:(batch - 1)).';
bin_id = mod(floor(idx / max(1, offset)), bins);
u = (double(bin_id) + rand(batch, 1)) / double(bins);
end

function varargout = joint_stratified_units(batch, bins, dimensions)
%JOINT_STRATIFIED_UNITS Cover the full Cartesian grid when it fits exactly.
cell_count = bins ^ dimensions;
if mod(batch, cell_count) ~= 0
    for d = 1:dimensions
        varargout{d} = stratified_unit(batch, bins, bins ^ (d - 1));
    end
    return;
end
cell_id = mod((0:(batch - 1)).', cell_count);
units = zeros(batch, dimensions);
for d = 1:dimensions
    bin_id = mod(floor(cell_id / (bins ^ (d - 1))), bins);
    units(:, d) = (double(bin_id) + rand(batch, 1)) / double(bins);
end
units = units(randperm(batch), :);
for d = 1:dimensions
    varargout{d} = units(:, d);
end
end

function factor = reciprocal_factor(unit, deviation)
upper = max(1.0 + deviation, 1.0);
lower = 1.0 / upper;
factor = lower + unit .* (upper - lower);
end

function y = cbrt_positive(x)
y = exp(log(max(x, 1.0e-12)) / 3.0);
end

function dynamics = make_dynamics(mass, c0, c1, c2, external_force, thrust_to_weight, torque_to_inertia, ...
    rotor_distance_factor, inertia_factor, motor_time_rising, motor_time_falling, rotor_torque_constant, ...
    cbrt_mass, force_std, arm_length, inertia_x, inertia_y, inertia_z, alpha_roll_max, alpha_pitch_max, ...
    alpha_yaw_max, eta_yaw, jz_over_jxy, dt)
batch = numel(mass);
dynamics = struct();
dynamics.mass = mass;
dynamics.thrust_coeff_c0 = repmat(c0, 1, 4);
dynamics.thrust_coeff_c1 = repmat(c1, 1, 4);
dynamics.thrust_coeff_c2 = repmat(c2, 1, 4);
dynamics.external_force = external_force;
dynamics.thrust_to_weight = thrust_to_weight;
dynamics.torque_to_inertia = torque_to_inertia;
dynamics.rotor_distance_factor = rotor_distance_factor;
dynamics.inertia_factor = inertia_factor;
dynamics.motor_time_rising = motor_time_rising;
dynamics.motor_time_falling = motor_time_falling;
dynamics.rotor_torque_constant = rotor_torque_constant;
dynamics.cbrt_mass = cbrt_mass;
dynamics.force_std = force_std;
dynamics.arm_length = arm_length;
dynamics.inertia_x = inertia_x;
dynamics.inertia_y = inertia_y;
dynamics.inertia_z = inertia_z;
dynamics.alpha_roll_max = alpha_roll_max;
dynamics.alpha_pitch_max = alpha_pitch_max;
dynamics.alpha_yaw_max = alpha_yaw_max;
dynamics.eta_yaw = eta_yaw;
dynamics.jz_over_jxy = jz_over_jxy;
dynamics.dt_alpha_roll_max = alpha_roll_max * dt;
dynamics.dt_alpha_yaw_max = alpha_yaw_max * dt;
dynamics.rotor_pos_body = zeros(batch, 4, 3);
dynamics.rotor_axis_body = zeros(batch, 4, 3);
for i = 1:batch
    dynamics.rotor_pos_body(i, :, :) = [
         arm_length(i), 0, 0;
         0, arm_length(i), 0;
        -arm_length(i), 0, 0;
         0,-arm_length(i), 0
    ];
    dynamics.rotor_axis_body(i, :, :) = repmat([0 0 1], 4, 1);
end
dynamics.spin_dir = repmat([1 -1 1 -1], batch, 1);
dynamics.rotor_torque_constant = repmat(rotor_torque_constant, 1, 4);
dynamics.motor_time_rising = repmat(motor_time_rising, 1, 4);
dynamics.motor_time_falling = repmat(motor_time_falling, 1, 4);
dynamics.motor_deadzone = zeros(batch, 4);
dynamics.motor_delay_steps = zeros(batch, 4);
dynamics.motor_health = ones(batch, 4);
dynamics.thrust_scale = ones(batch, 4);
dynamics.drag_linear = zeros(batch, 3);
dynamics.drag_quadratic = zeros(batch, 3);
dynamics.angular_drag_linear = zeros(batch, 3);
dynamics.angular_drag_quadratic = zeros(batch, 3);
dynamics.collision_radius = repmat(0.08, batch, 1);
dynamics.collision_height = repmat(0.04, batch, 1);
dynamics.rotor_radius = max(arm_length * 0.45, 1.0e-3);
dynamics.battery_enabled = zeros(batch, 1);
dynamics.battery_voltage = ones(batch, 1);
dynamics.battery_nominal_voltage = ones(batch, 1);
dynamics.battery_internal_resistance = zeros(batch, 1);
dynamics.battery_current_gain = zeros(batch, 1);
dynamics.battery_capacity_gain = zeros(batch, 1);
dynamics.battery_min_scale = repmat(0.6, batch, 1);
dynamics.allow_negative_thrust = zeros(batch, 1);
if batch == 0
    dynamics.external_force = zeros(0, 3);
end
end
