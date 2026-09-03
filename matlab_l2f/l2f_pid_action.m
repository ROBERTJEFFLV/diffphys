function action = l2f_pid_action(state, params, gains)
%L2F_PID_ACTION Simple reference controller for MATLAB demos.

if nargin < 3 || isempty(gains)
    gains = struct();
end
kp_xy = get_field_or(gains, 'kp_xy', 2.2);
kd_xy = get_field_or(gains, 'kd_xy', 1.0);
kp_z = get_field_or(gains, 'kp_z', 2.8);
kd_z = get_field_or(gains, 'kd_z', 1.0);
kp_att = get_field_or(gains, 'kp_att', 2.0);
kd_att = get_field_or(gains, 'kd_att', 0.3);
kd_yaw = get_field_or(gains, 'kd_yaw', 0.2);
max_tilt = get_field_or(gains, 'max_tilt', 0.45);

batch = size(state.position, 1);
action = zeros(batch, 4);
euler = l2f_matrix_to_euler_zyx(state.rotation);

for i = 1:batch
    pos_err = -state.position(i, :);
    vel_err = -state.velocity(i, :);
    z_acc = kp_z * pos_err(3) + kd_z * vel_err(3);
    x_acc = kp_xy * pos_err(1) + kd_xy * vel_err(1);
    y_acc = kp_xy * pos_err(2) + kd_xy * vel_err(2);
    total_thrust = state.mass(i) * (params.gravity + z_acc);

    roll_des = min(max(y_acc / params.gravity, -max_tilt), max_tilt);
    pitch_des = min(max(x_acc / params.gravity, -max_tilt), max_tilt);
    att_err = [
        wrap_to_pi(roll_des - euler(i, 1)), ...
        wrap_to_pi(pitch_des - euler(i, 2))
    ];
    target_alpha = [
        kp_att * att_err(1) - kd_att * state.omega(i, 1), ...
        kp_att * att_err(2) - kd_att * state.omega(i, 2), ...
        -kd_yaw * state.omega(i, 3)
    ];
    inertia = [state.inertia_x(i), state.inertia_y(i), state.inertia_z(i)];
    torque = inertia .* target_alpha - cross_row(state.omega(i, :), inertia .* state.omega(i, :));

    arm = max(state.arm_length(i), 1.0e-8);
    kt = max(row_average(state.rotor_torque_constant, i), 1.0e-8);
    bx = torque(1) / arm;
    by = torque(2) / arm;
    bz = torque(3) / kt;
    thrust = [
        (total_thrust + bz - 2 * by) / 4, ...
        (total_thrust + 2 * bx - bz) / 4, ...
        (total_thrust + bz + 2 * by) / 4, ...
        (total_thrust - 2 * bx - bz) / 4
    ];
    action(i, :) = l2f_thrust_to_action(thrust, state.thrust_coeff_c0(i, :), state.thrust_coeff_c1(i, :), state.thrust_coeff_c2(i, :));
end
end

function y = row_average(value, i)
if isscalar(value)
    y = value;
elseif size(value, 2) == 1
    y = value(i, 1);
else
    y = sum(value(i, :)) / size(value, 2);
end
end

function y = wrap_to_pi(x)
y = mod(x + pi, 2 * pi) - pi;
end

function c = cross_row(a, b)
c = [
    a(2) * b(3) - a(3) * b(2), ...
    a(3) * b(1) - a(1) * b(3), ...
    a(1) * b(2) - a(2) * b(1)
];
end

function value = get_field_or(s, name, default_value)
if isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end
