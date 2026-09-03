function task = l2f_task_recovery(varargin)
%L2F_TASK_RECOVERY Hover recovery task.

opt = struct();
opt.uav_cfg = [];
opt.world = l2f_make_world('empty');
opt.sensors = l2f_make_sensors('ideal');
opt.params = l2f_default_params();
opt.reward_cfg = l2f_default_reward_cfg();
opt.sim_cfg = l2f_default_sim_cfg();
opt = apply_name_values(opt, varargin{:});

task = base_task('recovery_multi_config', opt);
task.world.reference = struct('type', 'hover', 'position', [0 0 0], 'velocity', [0 0 0]);
end

function task = base_task(name, opt)
task = struct();
task.is_l2f_task = true;
task.name = name;
task.params = opt.params;
task.world = opt.world;
task.sensors = opt.sensors;
task.reward_cfg = opt.reward_cfg;
task.sim_cfg = opt.sim_cfg;
task.uav_cfg = opt.uav_cfg;
task.metrics = @(logs) l2f_metrics(logs, task.world, task.reward_cfg);
end
