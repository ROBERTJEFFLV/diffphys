function world = l2f_make_world(kind, varargin)
%L2F_MAKE_WORLD Build scenario/environment configuration.

if nargin < 1 || isempty(kind)
    kind = 'empty';
end

world = struct();
world.type = lower(kind);
world.name = world.type;
world.environment = struct('type', 'none');
world.wind = struct('type', 'none');
world.aero = struct('rho', 1.225);
world.ground_effect = struct('enabled', false, 'gain', 0.15, 'gain_max', 1.30, 'ground_z', 0.0, 'z_min', 0.05);
world.motor_fault = struct('enabled', false);
world.bounds = struct('position_norm', inf);
world.target_position = [0 0 0];
world.target_velocity = [0 0 0];
world.target_rotation = eye(3);
world.target_omega = [0 0 0];
world.obstacles = [];

switch world.type
    case 'empty'
    case 'constant_wind'
        world.environment = struct('type', 'constant_wind', 'wind_velocity', [1 0 0], 'wind_gain', 0.02);
        world.wind = struct('type', 'constant', 'velocity', [1 0 0]);
    case 'sinusoidal_wind'
        world.environment = struct('type', 'sinusoidal_wind', 'direction', [1 0 0], ...
            'amplitude', 1.0, 'frequency', 0.5, 'phase', 0.0, 'wind_gain', 0.02);
        world.wind = struct('type', 'sinusoidal', 'direction', [1 0 0], ...
            'amplitude', 1.0, 'frequency', 0.5, 'phase', 0.0);
    case 'gust'
        world.environment = struct('type', 'gust', 'start_time', 1.0, 'end_time', 1.5, ...
            'force', [0.1 0 0], 'torque', [0 0 0]);
        world.wind = struct('type', 'gust', 'start_time', 1.0, 'end_time', 1.5, 'velocity', [0 0 0]);
    case 'external_wrench'
        world.environment = struct('type', 'external_wrench', 'force', [0 0 0], 'torque', [0 0 0]);
    case 'wind_drag'
        world.environment = struct('type', 'none');
        world.wind = struct('type', 'constant', 'velocity', [2 0 0]);
    case 'ground_effect'
        world.ground_effect.enabled = true;
    case 'motor_fault'
        world.motor_fault = struct('enabled', true, 'start_time', 1.0, 'end_time', inf, ...
            'motor_index', 3, 'health', 0.7);
    otherwise
        error('Unknown world type: %s', kind);
end

world = apply_name_values(world, varargin{:});
end
