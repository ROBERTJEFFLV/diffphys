function state = l2f_unpack_state(x, dyn)
%L2F_UNPACK_STATE Rebuild one-vehicle state from flat Simulink vectors.

x = x(:);
dyn = dyn(:);
state = struct();
state.position = x(1:3).';
state.velocity = x(4:6).';
state.rotation = reshape(x(7:15), 3, 3).';
state.rotation = reshape(state.rotation, 3, 3, 1);
state.omega = x(16:18).';
state.motor = x(19:22).';
state.previous_action = x(23:26).';

k = 1;
state.mass = dyn(k); k = k + 1;
state.thrust_coeff_c0 = dyn(k:k + 3).'; k = k + 4;
state.thrust_coeff_c1 = dyn(k:k + 3).'; k = k + 4;
state.thrust_coeff_c2 = dyn(k:k + 3).'; k = k + 4;
state.external_force = dyn(k:k + 2).'; k = k + 3;
state.thrust_to_weight = dyn(k); k = k + 1;
state.torque_to_inertia = dyn(k); k = k + 1;
state.rotor_distance_factor = dyn(k); k = k + 1;
state.inertia_factor = dyn(k); k = k + 1;
motor_time_rising_scalar = dyn(k); k = k + 1;
motor_time_falling_scalar = dyn(k); k = k + 1;
rotor_torque_scalar = dyn(k); k = k + 1;
state.cbrt_mass = dyn(k); k = k + 1;
state.force_std = dyn(k); k = k + 1;
state.arm_length = dyn(k); k = k + 1;
state.inertia_x = dyn(k); k = k + 1;
state.inertia_y = dyn(k); k = k + 1;
state.inertia_z = dyn(k); k = k + 1;
state.alpha_roll_max = dyn(k); k = k + 1;
state.alpha_pitch_max = dyn(k); k = k + 1;
state.alpha_yaw_max = dyn(k); k = k + 1;
state.eta_yaw = dyn(k); k = k + 1;
state.jz_over_jxy = dyn(k); k = k + 1;
state.dt_alpha_roll_max = dyn(k); k = k + 1;
state.dt_alpha_yaw_max = dyn(k); k = k + 1;

state.rotor_pos_body = reshape(default_rotor_pos(state.arm_length), 1, 4, 3);
state.rotor_axis_body = reshape(repmat([0 0 1], 4, 1), 1, 4, 3);
state.spin_dir = [1 -1 1 -1];
state.rotor_torque_constant = repmat(rotor_torque_scalar, 1, 4);
state.motor_time_rising = repmat(motor_time_rising_scalar, 1, 4);
state.motor_time_falling = repmat(motor_time_falling_scalar, 1, 4);
state.motor_deadzone = zeros(1, 4);
state.motor_delay_steps = zeros(1, 4);
state.motor_health = ones(1, 4);
state.thrust_scale = ones(1, 4);
state.drag_linear = [0 0 0];
state.drag_quadratic = [0 0 0];
state.angular_drag_linear = [0 0 0];
state.angular_drag_quadratic = [0 0 0];
state.collision_radius = 0.08;
state.collision_height = 0.04;
state.rotor_radius = 0.02;
state.battery_enabled = 0.0;
state.battery_voltage = 1.0;
state.battery_nominal_voltage = 1.0;
state.battery_internal_resistance = 0.0;
state.battery_current_gain = 0.0;
state.battery_capacity_gain = 0.0;
state.battery_min_scale = 0.6;
state.allow_negative_thrust = 0.0;

if numel(dyn) >= k + 11
    state.rotor_pos_body = read_geom(dyn(k:k + 11)); k = k + 12;
end
if numel(dyn) >= k + 11
    state.rotor_axis_body = read_geom(dyn(k:k + 11)); k = k + 12;
end
if numel(dyn) >= k + 3
    state.spin_dir = dyn(k:k + 3).'; k = k + 4;
end
if numel(dyn) >= k + 3
    state.rotor_torque_constant = dyn(k:k + 3).'; k = k + 4;
end
if numel(dyn) >= k + 3
    state.motor_time_rising = dyn(k:k + 3).'; k = k + 4;
end
if numel(dyn) >= k + 3
    state.motor_time_falling = dyn(k:k + 3).'; k = k + 4;
end
if numel(dyn) >= k + 3
    state.motor_deadzone = dyn(k:k + 3).'; k = k + 4;
end
if numel(dyn) >= k + 3
    state.motor_delay_steps = dyn(k:k + 3).'; k = k + 4;
end
if numel(dyn) >= k + 3
    state.motor_health = dyn(k:k + 3).'; k = k + 4;
end
if numel(dyn) >= k + 3
    state.thrust_scale = dyn(k:k + 3).'; k = k + 4;
end
if numel(dyn) >= k + 2
    state.drag_linear = dyn(k:k + 2).'; k = k + 3;
end
if numel(dyn) >= k + 2
    state.drag_quadratic = dyn(k:k + 2).'; k = k + 3;
end
if numel(dyn) >= k + 2
    state.angular_drag_linear = dyn(k:k + 2).'; k = k + 3;
end
if numel(dyn) >= k + 2
    state.angular_drag_quadratic = dyn(k:k + 2).'; k = k + 3;
end
if numel(dyn) >= k
    state.collision_radius = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.collision_height = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.rotor_radius = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.battery_enabled = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.battery_voltage = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.battery_nominal_voltage = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.battery_internal_resistance = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.battery_current_gain = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.battery_capacity_gain = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.battery_min_scale = dyn(k); k = k + 1;
end
if numel(dyn) >= k
    state.allow_negative_thrust = dyn(k);
end
end

function geom = read_geom(values)
geom = reshape(reshape(values(:), 3, 4).', 1, 4, 3);
end

function pos = default_rotor_pos(arm)
pos = [
     arm, 0, 0;
     0, arm, 0;
    -arm, 0, 0;
     0,-arm, 0
];
end
