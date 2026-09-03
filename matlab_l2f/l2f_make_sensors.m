function sensors = l2f_make_sensors(kind, varargin)
%L2F_MAKE_SENSORS Build a sensor/observation configuration.

if nargin < 1 || isempty(kind)
    kind = 'ideal';
end

sensors = struct();
sensors.type = lower(kind);
sensors.observation_noise_std = 0.0;
sensors.action_feedback_noise_std = 0.0;
sensors.delay_steps = 0;
sensors.dropout_prob = 0.0;

switch sensors.type
    case 'ideal'
    case 'noisy'
        sensors.observation_noise_std = 1.0e-3;
        sensors.action_feedback_noise_std = 1.0e-3;
    otherwise
        error('Unknown sensor type: %s', kind);
end

sensors = apply_name_values(sensors, varargin{:});
end
