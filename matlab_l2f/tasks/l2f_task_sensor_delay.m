function task = l2f_task_sensor_delay(varargin)
%L2F_TASK_SENSOR_DELAY Observation noise/dropout task scaffold.

opt = struct();
opt.uav_cfg = l2f_default_uav_cfg();
opt.observation_noise_std = 0.02;
opt.action_feedback_noise_std = 0.01;
opt.dropout_prob = 0.0;
opt.delay_steps = 2;
opt.horizon = 500;
opt.batch_size = 8;
opt = apply_name_values(opt, varargin{:});

sensors = l2f_make_sensors('noisy', ...
    'observation_noise_std', opt.observation_noise_std, ...
    'action_feedback_noise_std', opt.action_feedback_noise_std);
sensors.dropout_prob = opt.dropout_prob;
sensors.delay_steps = opt.delay_steps;

world = l2f_make_world('empty');
world.reference = struct('type', 'hover', 'position', [0 0 0], 'velocity', [0 0 0]);

task = struct();
task.is_l2f_task = true;
task.name = 'sensor_delay_noise';
task.params = l2f_default_params();
task.world = world;
task.sensors = sensors;
task.reward_cfg = l2f_default_reward_cfg();
task.sim_cfg = l2f_default_sim_cfg('horizon', opt.horizon, 'batch_size', opt.batch_size);
task.uav_cfg = opt.uav_cfg;
task.metrics = @(logs) l2f_metrics(logs, world, task.reward_cfg);
end
