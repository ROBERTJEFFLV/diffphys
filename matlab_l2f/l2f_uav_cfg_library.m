function cfg = l2f_uav_cfg_library(name)
%L2F_UAV_CFG_LIBRARY Named UAV physical configurations.

if nargin < 1 || isempty(name)
    name = 'nominal_50g';
end

switch lower(name)
    case {'nominal_50g', 'micro50g', 'micro_50g'}
        cfg = make_plus_quad('nominal_50g', 0.05, 0.046, ...
            [1.4e-5 1.4e-5 2.17e-5], 1.35, 0.06, 0.02, ...
            [0 0 0], [0.01 0.01 0.02], 0.08, 0.04);
    case {'small150g', 'small_150g'}
        cfg = make_plus_quad('small150g', 0.15, 0.085, ...
            [8.0e-5 8.0e-5 1.45e-4], 1.80, 0.075, 0.018, ...
            [0.02 0.02 0.03], [0.025 0.025 0.05], 0.13, 0.06);
    case {'x500', 'x500_quad'}
        cfg = make_plus_quad('x500', 1.20, 0.23, ...
            [1.5e-2 1.5e-2 2.8e-2], 2.20, 0.11, 0.015, ...
            [0.12 0.12 0.18], [0.18 0.18 0.28], 0.36, 0.16);
    otherwise
        error('Unknown UAV config: %s', name);
end
end

function cfg = make_plus_quad(name, mass, arm_length, inertia_diag, motor_authority, tau, km, ...
    drag_linear, drag_quadratic, collision_radius, collision_height)
gravity = 9.80665;
hover = mass * gravity / 4.0;

cfg = struct();
cfg.name = name;
cfg.mass = mass;
cfg.inertia = diag(inertia_diag);
cfg.rotor_pos_body = [
     arm_length, 0, 0;
     0, arm_length, 0;
    -arm_length, 0, 0;
     0,-arm_length, 0
];
cfg.rotor_axis_body = repmat([0 0 1], 4, 1);
cfg.spin_dir = [1 -1 1 -1];
cfg.thrust_coeff_c0 = hover * ones(1, 4);
cfg.thrust_coeff_c1 = motor_authority * hover * ones(1, 4);
cfg.thrust_coeff_c2 = zeros(1, 4);
cfg.motor_tau_rise = tau * ones(1, 4);
cfg.motor_tau_fall = tau * ones(1, 4);
cfg.motor_deadzone = zeros(1, 4);
cfg.motor_delay_steps = zeros(1, 4);
cfg.rotor_torque_constant = km * ones(1, 4);
cfg.motor_health = ones(1, 4);
cfg.thrust_scale = ones(1, 4);
cfg.allow_negative_thrust = false;
cfg.drag_linear = drag_linear;
cfg.drag_quadratic = drag_quadratic;
cfg.angular_drag_linear = [0 0 0];
cfg.angular_drag_quadratic = [0 0 0];
cfg.collision_radius = collision_radius;
cfg.collision_height = collision_height;
cfg.rotor_radius = max(arm_length * 0.45, 1.0e-3);
cfg.battery = struct( ...
    'enabled', false, ...
    'voltage', 1.0, ...
    'nominal_voltage', 1.0, ...
    'internal_resistance', 0.0, ...
    'current_gain', 0.0, ...
    'capacity_gain', 0.0, ...
    'min_scale', 0.6);
cfg.randomization = struct();
end
