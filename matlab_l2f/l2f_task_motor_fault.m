function task = l2f_task_motor_fault(varargin)
%L2F_TASK_MOTOR_FAULT Scheduled single-motor degradation task.

opt = struct();
opt.uav_cfg = l2f_default_uav_cfg();
opt.motor_index = 3;
opt.health = 0.7;
opt.start_time = 1.0;
opt.horizon = 500;
opt.batch_size = 8;
opt = apply_name_values(opt, varargin{:});

world = l2f_make_world('motor_fault');
world.motor_fault.motor_index = opt.motor_index;
world.motor_fault.health = opt.health;
world.motor_fault.start_time = opt.start_time;
world.reference = struct('type', 'hover', 'position', [0 0 0], 'velocity', [0 0 0]);

task = struct();
task.is_l2f_task = true;
task.name = 'motor_fault';
task.params = l2f_default_params();
task.world = world;
task.sensors = l2f_make_sensors('ideal');
task.reward_cfg = l2f_default_reward_cfg();
task.sim_cfg = l2f_default_sim_cfg('horizon', opt.horizon, 'batch_size', opt.batch_size);
task.uav_cfg = opt.uav_cfg;
task.metrics = @(logs) l2f_metrics(logs, world, task.reward_cfg);
end
