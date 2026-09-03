function [force_world, torque_body] = l2f_motor_wrench(state, thrust)
%L2F_MOTOR_WRENCH Compute motor force and torque from generic geometry.

batch = size(state.position, 1);
motor_count = size(thrust, 2);
force_world = zeros(batch, 3);
torque_body = zeros(batch, 3);

for b = 1:batch
    rotation = state.rotation(:, :, b);
    force_body = [0 0 0];
    torque = [0 0 0];

    for m = 1:motor_count
        r = get_rotor_vector(state, 'rotor_pos_body', b, m, default_rotor_pos(state, b, m));
        axis = get_rotor_vector(state, 'rotor_axis_body', b, m, [0 0 1]);
        axis_norm = sqrt(sum(axis .* axis));
        axis = axis / max(axis_norm, 1.0e-12);

        rotor_force = axis * thrust(b, m);
        km = get_motor_value(state, 'rotor_torque_constant', b, m, 0.02);
        spin = get_motor_value(state, 'spin_dir', b, m, spin_default(m));
        torque = torque + cross_row(r, rotor_force) + spin * km * thrust(b, m) * axis;
        force_body = force_body + rotor_force;
    end

    force_world(b, :) = (rotation * force_body.').';
    torque_body(b, :) = torque;
end
end

function r = default_rotor_pos(state, b, m)
arm = get_motor_value(state, 'arm_length', b, 1, 0.046);
if m == 1
    r = [arm 0 0];
elseif m == 2
    r = [0 arm 0];
elseif m == 3
    r = [-arm 0 0];
else
    r = [0 -arm 0];
end
end

function value = get_rotor_vector(state, name, b, m, default_value)
if isfield(state, name)
    data = state.(name);
    value = reshape(data(b, m, :), 1, 3);
else
    value = default_value;
end
end

function value = get_motor_value(state, name, b, m, default_value)
if ~isfield(state, name)
    value = default_value;
    return;
end
data = state.(name);
if isscalar(data)
    value = data;
elseif size(data, 1) >= b && size(data, 2) >= m
    value = data(b, m);
elseif size(data, 1) >= b
    value = data(b, 1);
else
    value = default_value;
end
end

function s = spin_default(m)
if m == 1 || m == 3
    s = 1;
else
    s = -1;
end
end

function c = cross_row(a, b)
c = [
    a(2) * b(3) - a(3) * b(2), ...
    a(3) * b(1) - a(1) * b(3), ...
    a(1) * b(2) - a(2) * b(1)
];
end
