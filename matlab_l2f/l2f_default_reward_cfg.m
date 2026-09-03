function reward_cfg = l2f_default_reward_cfg(varargin)
%L2F_DEFAULT_REWARD_CFG Reward and success thresholds for rollout/metrics.

reward_cfg = struct();
reward_cfg.p_scale = 2.0;
reward_cfg.v_scale = 3.0;
reward_cfg.omega_scale = 10.0;
reward_cfg.w_position = 1.0;
reward_cfg.w_velocity = 0.3;
reward_cfg.w_omega = 0.05;
reward_cfg.w_action = 0.01;
reward_cfg.w_smooth = 0.04;
reward_cfg.success_position_m = 0.05;
reward_cfg.success_velocity = 0.10;
reward_cfg.success_omega = 0.20;
reward_cfg.steady_window_steps = 100;
reward_cfg.steady_required_fraction = 0.95;

reward_cfg = apply_name_values(reward_cfg, varargin{:});
end
