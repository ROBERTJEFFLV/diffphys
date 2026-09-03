function task = l2f_task_config_generalization(varargin)
%L2F_TASK_CONFIG_GENERALIZATION Multi-UAV physical parameter validation task.

opt = struct();
opt.uav_family = {'micro50g', 'small150g', 'x500'};
opt.randomize_mass = true;
opt.randomize_inertia = true;
opt.randomize_motor_tau = true;
opt.randomize_thrust_to_weight = true;
opt.randomize_drag = true;
opt.horizon = 500;
opt.batch_size = 16;
opt = apply_name_values(opt, varargin{:});

uav_cfg = struct();
uav_cfg.uav_family = opt.uav_family;
uav_cfg.randomization = struct();
if opt.randomize_mass
    uav_cfg.randomization.mass_scale = [0.7 1.3];
    uav_cfg.randomization.arm_scale = [0.8 1.2];
end
if opt.randomize_inertia
    uav_cfg.randomization.inertia_scale = [0.6 1.8];
end
if opt.randomize_motor_tau
    uav_cfg.randomization.motor_tau_rise = [0.03 0.20];
    uav_cfg.randomization.motor_tau_fall = [0.03 0.30];
end
if opt.randomize_thrust_to_weight
    uav_cfg.randomization.thrust_to_weight = [1.3 5.0];
end
if opt.randomize_drag
    uav_cfg.randomization.drag_linear = [0.00 0.15];
    uav_cfg.randomization.drag_quadratic = [0.00 0.35];
end
uav_cfg.randomization.rotor_torque_constant_scale = [0.6 1.4];
uav_cfg.randomization.external_force_std = [0.0 0.03];

world = l2f_make_world('empty');
world.reference = struct('type', 'hover', 'position', [0 0 0], 'velocity', [0 0 0]);

task = struct();
task.is_l2f_task = true;
task.name = 'config_generalization';
task.params = l2f_default_params();
task.world = world;
task.sensors = l2f_make_sensors('ideal');
task.reward_cfg = l2f_default_reward_cfg();
task.sim_cfg = l2f_default_sim_cfg('horizon', opt.horizon, 'batch_size', opt.batch_size);
task.uav_cfg = uav_cfg;
task.parameter_sweep = @l2f_config_sweep;
task.metrics = @(logs) l2f_metrics(logs, world, task.reward_cfg);
end
