function sim_cfg = l2f_default_sim_cfg(varargin)
%L2F_DEFAULT_SIM_CFG Standard rollout configuration.

sim_cfg = struct();
sim_cfg.dt = 0.01;
sim_cfg.gravity = 9.80665;
sim_cfg.horizon = 500;
sim_cfg.batch_size = 1;
sim_cfg.seed = 7;
sim_cfg.live_plot = false;
sim_cfg.live_plot_every = 5;
sim_cfg.vehicle_id = 1;
sim_cfg.freeze_done = true;
sim_cfg.terminate_on_nonfinite = true;
sim_cfg.terminate_on_bounds = true;
sim_cfg.terminate_on_success = false;
sim_cfg.max_position_norm = 10.0;
sim_cfg.max_velocity_norm = 25.0;
sim_cfg.max_omega_norm = 200.0;
sim_cfg.stop_when_all_done = false;
sim_cfg.initial_state = [];
sim_cfg.uav_cfg = [];
sim_cfg.uav_cfg_name = '';

sim_cfg = apply_name_values(sim_cfg, varargin{:});
end
