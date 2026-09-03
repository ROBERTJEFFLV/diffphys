function wind = l2f_wind_field(position, world, t)
%L2F_WIND_FIELD Evaluate wind velocity in world frame.

if nargin < 3
    t = 0;
end
batch = size(position, 1);
wind = zeros(batch, 3);

if nargin < 2 || isempty(world)
    return;
end

env = world;
if isfield(world, 'environment')
    env = world.environment;
end
if isfield(world, 'wind')
    env = world.wind;
end
if ~isstruct(env) || ~isfield(env, 'type')
    return;
end

switch lower(env.type)
    case {'none', 'empty'}
        return;
    case {'constant', 'constant_wind'}
        velocity = l2f_get_field_or(env, 'velocity', l2f_get_field_or(env, 'wind_velocity', [0 0 0]));
        wind = repmat(reshape(velocity, 1, 3), batch, 1);
    case {'sinusoidal', 'sinusoidal_wind'}
        direction = reshape(l2f_get_field_or(env, 'direction', [1 0 0]), 1, 3);
        direction = direction / max(sqrt(sum(direction .* direction)), 1.0e-12);
        amplitude = l2f_get_field_or(env, 'amplitude', 1.0);
        frequency = l2f_get_field_or(env, 'frequency', 0.5);
        phase = l2f_get_field_or(env, 'phase', 0.0);
        wind = repmat(amplitude * sin(2 * pi * frequency * t + phase) * direction, batch, 1);
    case 'shear'
        base = reshape(l2f_get_field_or(env, 'base_velocity', [0 0 0]), 1, 3);
        gradient = reshape(l2f_get_field_or(env, 'gradient_z', [0 0 0]), 1, 3);
        for i = 1:batch
            wind(i, :) = base + gradient * position(i, 3);
        end
    case 'gust'
        start_t = l2f_get_field_or(env, 'start_time', 1.0);
        end_t = l2f_get_field_or(env, 'end_time', 1.5);
        if t >= start_t && t <= end_t
            velocity = l2f_get_field_or(env, 'velocity', l2f_get_field_or(env, 'wind_velocity', [0 0 0]));
            wind = repmat(reshape(velocity, 1, 3), batch, 1);
        end
end
end
