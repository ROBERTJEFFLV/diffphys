function [force, torque] = l2f_environment_wrench(state, t, params, env)
%L2F_ENVIRONMENT_WRENCH Extensible force/torque module hook.
%
% Returned force is world-frame Nx3. Returned torque is body-frame Nx3.

batch = size(state.position, 1);
force = zeros(batch, 3);
torque = zeros(batch, 3);

if nargin < 4 || isempty(env) || ~isfield(env, 'type')
    return;
end

switch lower(env.type)
    case 'none'
        return;
    case 'external_wrench'
        if isfield(env, 'force')
            force = repmat(reshape(env.force, 1, 3), batch, 1);
        end
        if isfield(env, 'torque')
            torque = repmat(reshape(env.torque, 1, 3), batch, 1);
        end
    case 'constant_wind'
        wind_velocity = get_field_or(env, 'wind_velocity', [0 0 0]);
        wind_gain = get_field_or(env, 'wind_gain', 0.0);
        relative_wind = repmat(reshape(wind_velocity, 1, 3), batch, 1) - state.velocity;
        force = wind_gain * relative_wind;
    case 'sinusoidal_wind'
        direction = get_field_or(env, 'direction', [1 0 0]);
        direction = reshape(direction, 1, 3);
        direction = direction / max(norm(direction), 1.0e-12);
        amplitude = get_field_or(env, 'amplitude', 1.0);
        frequency = get_field_or(env, 'frequency', 0.5);
        phase = get_field_or(env, 'phase', 0.0);
        wind_gain = get_field_or(env, 'wind_gain', 1.0);
        force = repmat(wind_gain * amplitude * sin(2 * pi * frequency * t + phase) * direction, batch, 1);
    case 'gust'
        start_t = get_field_or(env, 'start_time', 1.0);
        end_t = get_field_or(env, 'end_time', 1.5);
        if t >= start_t && t <= end_t
            force = repmat(reshape(get_field_or(env, 'force', [0 0 0]), 1, 3), batch, 1);
            torque = repmat(reshape(get_field_or(env, 'torque', [0 0 0]), 1, 3), batch, 1);
        end
    otherwise
        error('Unknown environment module type: %s', env.type);
end

% Keep params in the signature so future modules can depend on physical scale.
if isempty(params)
    return;
end
end

function value = get_field_or(s, name, default_value)
if isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end
