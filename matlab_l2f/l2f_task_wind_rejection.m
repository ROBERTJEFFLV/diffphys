function task = l2f_task_wind_rejection(varargin)
%L2F_TASK_WIND_REJECTION Wind field plus aerodynamic drag validation task.

opt = struct();
opt.uav_cfg = l2f_default_uav_cfg();
opt.wind_velocity = [2 0 0];
opt.horizon = 500;
opt.batch_size = 8;
opt = apply_name_values(opt, varargin{:});

world = l2f_make_world('wind_drag');
world.wind.velocity = opt.wind_velocity;
world.reference = struct('type', 'hover', 'position', [0 0 0], 'velocity', [0 0 0]);

task = struct();
task.is_l2f_task = true;
task.name = 'wind_rejection';
task.params = l2f_default_params();
task.world = world;
task.sensors = l2f_make_sensors('ideal');
task.reward_cfg = l2f_default_reward_cfg();
task.sim_cfg = l2f_default_sim_cfg('horizon', opt.horizon, 'batch_size', opt.batch_size);
task.uav_cfg = opt.uav_cfg;
task.metrics = @(logs) l2f_metrics(logs, world, task.reward_cfg);
end
