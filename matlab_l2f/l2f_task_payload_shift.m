function task = l2f_task_payload_shift(varargin)
%L2F_TASK_PAYLOAD_SHIFT Scheduled mass/inertia change task.

opt = struct();
opt.uav_cfg = l2f_default_uav_cfg();
opt.start_time = 2.0;
opt.mass_delta = 0.02;
opt.mass_scale = 1.0;
opt.inertia_scale = [1.2 1.2 1.15];
opt.horizon = 500;
opt.batch_size = 8;
opt = apply_name_values(opt, varargin{:});

world = l2f_make_world('empty');
world.payload_shift = struct('enabled', true, 'start_time', opt.start_time, ...
    'mass_delta', opt.mass_delta, 'mass_scale', opt.mass_scale, 'inertia_scale', opt.inertia_scale);
world.reference = struct('type', 'hover', 'position', [0 0 0], 'velocity', [0 0 0]);

task = struct();
task.is_l2f_task = true;
task.name = 'payload_shift';
task.params = l2f_default_params();
task.world = world;
task.sensors = l2f_make_sensors('ideal');
task.reward_cfg = l2f_default_reward_cfg();
task.sim_cfg = l2f_default_sim_cfg('horizon', opt.horizon, 'batch_size', opt.batch_size);
task.uav_cfg = opt.uav_cfg;
task.metrics = @(logs) l2f_metrics(logs, world, task.reward_cfg);
end
