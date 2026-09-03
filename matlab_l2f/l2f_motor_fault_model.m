function thrust = l2f_motor_fault_model(thrust, state, world, t)
%L2F_MOTOR_FAULT_MODEL Apply motor health and scheduled motor faults.

if nargin < 4
    t = 0;
end
batch = size(thrust, 1);
motor_count = size(thrust, 2);
health = get_state_row(state, 'motor_health', batch, motor_count, 1.0);
scale = get_state_row(state, 'thrust_scale', batch, motor_count, 1.0);

fault = struct('enabled', false);
if isstruct(world) && isfield(world, 'motor_fault')
    fault = world.motor_fault;
elseif isstruct(world) && isfield(world, 'environment') && isfield(world.environment, 'motor_fault')
    fault = world.environment.motor_fault;
end

if l2f_get_field_or(fault, 'enabled', false)
    start_t = l2f_get_field_or(fault, 'start_time', 0.0);
    end_t = l2f_get_field_or(fault, 'end_time', inf);
    if t >= start_t && t <= end_t
        fault_health = l2f_get_field_or(fault, 'health', []);
        motor_index = l2f_get_field_or(fault, 'motor_index', []);
        if isempty(fault_health)
            fault_health = ones(1, motor_count);
        end
        if isempty(motor_index)
            health = health .* repmat(reshape(fault_health, 1, motor_count), batch, 1);
        else
            for j = 1:numel(motor_index)
                m = motor_index(j);
                if m >= 1 && m <= motor_count
                    if numel(fault_health) == 1
                        health(:, m) = health(:, m) * fault_health;
                    else
                        health(:, m) = health(:, m) * fault_health(j);
                    end
                end
            end
        end
    end
end

thrust = thrust .* health .* scale;
end

function row = get_state_row(state, name, batch, width, default_value)
if isfield(state, name)
    value = state.(name);
else
    value = default_value;
end
if isscalar(value)
    row = repmat(value, batch, width);
elseif size(value, 1) == batch && size(value, 2) == width
    row = value;
elseif size(value, 1) == batch && size(value, 2) == 1
    row = repmat(value, 1, width);
else
    row = repmat(reshape(value, 1, width), batch, 1);
end
end
