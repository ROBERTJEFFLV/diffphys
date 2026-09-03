function task = l2f_task_trajectory_tracking(varargin)
%L2F_TASK_TRAJECTORY_TRACKING Reference trajectory tracking task.

opt = struct();
opt.uav_cfg = l2f_default_uav_cfg();
opt.reference_type = 'circle';
opt.radius = 0.8;
opt.omega = 0.6;
opt.z = 0.0;
opt.horizon = 500;
opt.batch_size = 8;
opt = apply_name_values(opt, varargin{:});

world = l2f_make_world('empty');
world.reference = struct('type', opt.reference_type, 'radius', opt.radius, 'omega', opt.omega, 'z', opt.z);

task = struct();
task.is_l2f_task = true;
task.name = 'trajectory_tracking';
task.params = l2f_default_params();
task.world = world;
task.sensors = l2f_make_sensors('ideal');
task.reward_cfg = l2f_default_reward_cfg();
task.sim_cfg = l2f_default_sim_cfg('horizon', opt.horizon, 'batch_size', opt.batch_size);
task.uav_cfg = opt.uav_cfg;
task.metrics = @(logs) l2f_metrics(logs, world, task.reward_cfg);
end
