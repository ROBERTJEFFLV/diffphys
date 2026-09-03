function [next_state, aux] = l2f_step(state, action, params, environment, t)
%L2F_STEP One L2F PyTorch-style simulator step in MATLAB.

if nargin < 4 || isempty(environment)
    environment = struct('type', 'none');
end
if nargin < 5
    t = 0;
end

dt = params.dt;
world = normalize_world(environment);
state = l2f_payload_shift_model(state, world, t);
command = min(max(action, -1.0), 1.0);
command = apply_motor_deadzone(command, state);
[effective_command, action_buffer] = apply_motor_delay(command, state);

motor_time_rising = expand_motor_field(state, 'motor_time_rising', 0.06);
motor_time_falling = expand_motor_field(state, 'motor_time_falling', 0.06);
motor_tau = motor_time_falling;
rise_mask = effective_command >= state.motor;
motor_tau(rise_mask) = motor_time_rising(rise_mask);
alpha = min(max(dt ./ max(motor_tau, 1.0e-6), 0.0), 1.0);
motor = state.motor + alpha .* (effective_command - state.motor);

thrust = state.thrust_coeff_c0 + state.thrust_coeff_c1 .* motor + state.thrust_coeff_c2 .* motor .* motor;
if ~any(get_column(state, 'allow_negative_thrust', 0.0) > 0.5)
    thrust = max(thrust, 0.0);
else
    allow_negative = get_column(state, 'allow_negative_thrust', 0.0);
    for i = 1:size(thrust, 1)
        if allow_negative(i) <= 0.5
            thrust(i, :) = max(thrust(i, :), 0.0);
        end
    end
end

batch = size(state.position, 1);
thrust = l2f_motor_fault_model(thrust, state, world, t);
thrust = l2f_ground_effect(thrust, state, world);
[thrust, battery] = l2f_battery_model(thrust, state, dt);
total_thrust = sum(thrust, 2);
[force_world_motor, motor_torque] = l2f_motor_wrench(state, thrust);
[env_force, env_torque] = l2f_environment_wrench(state, t, params, world.environment);
[aero_force, aero_torque] = l2f_aero_wrench(state, world, t);
gravity = [0, 0, -params.gravity];
acceleration = zeros(batch, 3);
torque = zeros(batch, 3);
for i = 1:batch
    acceleration(i, :) = force_world_motor(i, :) / state.mass(i) + gravity ...
        + (state.external_force(i, :) + env_force(i, :) + aero_force(i, :)) / state.mass(i);
    torque(i, :) = motor_torque(i, :) + env_torque(i, :) + aero_torque(i, :);
end

velocity = state.velocity + dt * acceleration;
position = state.position + dt * velocity;
inertia = [state.inertia_x, state.inertia_y, state.inertia_z];
omega = l2f_implicit_midpoint_omega(state.omega, torque, inertia, dt, 4);

rotation = zeros(size(state.rotation));
for i = 1:batch
    omega_mid = 0.5 * (state.omega(i, :) + omega(i, :));
    rotation(:, :, i) = state.rotation(:, :, i) * l2f_so3_exp(dt * omega_mid(:));
end

next_state = state;
next_state.position = position;
next_state.velocity = velocity;
next_state.rotation = rotation;
next_state.omega = omega;
next_state.motor = motor;
next_state.previous_action = command;
if ~isempty(action_buffer)
    next_state.action_buffer = action_buffer;
end
if isfield(battery, 'voltage')
    next_state.battery_voltage = battery.voltage;
end

aux = struct();
aux.command = command;
aux.effective_command = effective_command;
aux.motor = motor;
aux.thrust = thrust;
aux.total_thrust = total_thrust;
aux.acceleration = acceleration;
aux.torque = torque;
aux.motor_torque = motor_torque;
aux.force_world_motor = force_world_motor;
aux.env_force = env_force;
aux.env_torque = env_torque;
aux.aero_force = aero_force;
aux.aero_torque = aero_torque;
aux.battery_voltage = get_column(next_state, 'battery_voltage', 1.0);
end

function world = normalize_world(environment)
if nargin < 1 || isempty(environment)
    environment = struct('type', 'none');
end
if isstruct(environment) && isfield(environment, 'environment')
    world = environment;
else
    world = struct();
    world.type = 'custom';
    world.environment = environment;
end
if ~isfield(world, 'environment') || isempty(world.environment)
    world.environment = struct('type', 'none');
end
end

function command = apply_motor_deadzone(command, state)
deadzone = expand_motor_field(state, 'motor_deadzone', 0.0);
mask = abs(command) < deadzone;
command(mask) = 0.0;
end

function [effective_command, action_buffer] = apply_motor_delay(command, state)
effective_command = command;
action_buffer = [];
if ~isfield(state, 'motor_delay_steps')
    return;
end
delay_steps = round(expand_motor_field(state, 'motor_delay_steps', 0.0));
max_delay = max(max(delay_steps));
if max_delay <= 0
    return;
end
batch = size(command, 1);
motor_count = size(command, 2);
if isfield(state, 'action_buffer') && size(state.action_buffer, 3) >= max_delay + 1
    action_buffer = state.action_buffer;
else
    action_buffer = zeros(batch, motor_count, max_delay + 1);
end
if size(action_buffer, 3) < max_delay + 1
    expanded = zeros(batch, motor_count, max_delay + 1);
    expanded(:, :, 1:size(action_buffer, 3)) = action_buffer;
    action_buffer = expanded;
end
action_buffer(:, :, 2:end) = action_buffer(:, :, 1:end-1);
action_buffer(:, :, 1) = command;
for i = 1:batch
    for m = 1:motor_count
        d = delay_steps(i, m);
        effective_command(i, m) = action_buffer(i, m, d + 1);
    end
end
end

function values = expand_motor_field(state, name, default_value)
batch = size(state.position, 1);
motor_count = size(state.motor, 2);
if isfield(state, name)
    data = state.(name);
else
    data = default_value;
end
if isscalar(data)
    values = repmat(data, batch, motor_count);
elseif size(data, 1) == batch && size(data, 2) == motor_count
    values = data;
elseif size(data, 1) == batch && size(data, 2) == 1
    values = repmat(data, 1, motor_count);
else
    values = repmat(reshape(data, 1, motor_count), batch, 1);
end
end

function values = get_column(state, name, default_value)
batch = size(state.position, 1);
if isfield(state, name)
    data = state.(name);
    if isscalar(data)
        values = repmat(data, batch, 1);
    else
        values = data(:, 1);
    end
else
    values = repmat(default_value, batch, 1);
end
end
