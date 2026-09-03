function obs = l2f_sensor_model(state, sensors, t)
%L2F_SENSOR_MODEL Observation construction with noise/dropout hooks.

if nargin < 2 || isempty(sensors)
    sensors = l2f_make_sensors('ideal');
end
if nargin < 3
    t = 0;
end

[compact_observation, ~] = l2f_observation(state, 'compact22', []);
physical_features = compact_observation(:, 1:18);
previous_action = compact_observation(:, 19:22);
noise_std = l2f_get_field_or(sensors, 'observation_noise_std', 0.0);
action_noise_std = l2f_get_field_or(sensors, 'action_feedback_noise_std', 0.0);
dropout_prob = l2f_get_field_or(sensors, 'dropout_prob', 0.0);

if noise_std > 0.0
    physical_features = physical_features + randn(size(physical_features)) * noise_std;
end
if action_noise_std > 0.0
    previous_action = previous_action + randn(size(previous_action)) * action_noise_std;
end
if dropout_prob > 0.0
    physical_features(rand(size(physical_features)) < dropout_prob) = 0.0;
end

obs = struct();
obs.physical_features = physical_features;
obs.observed_position = physical_features(:, 1:3);
obs.previous_action = previous_action;
obs.config_features = l2f_config_observation(state);
obs.time = t;
end
