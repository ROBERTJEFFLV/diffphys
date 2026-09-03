function table_out = l2f_config_sweep(controller, varargin)
%L2F_CONFIG_SWEEP Parameter sweep over mass, motor tau, and thrust-to-weight.

opt = struct();
opt.mass_list = l2f_linspace(0.03, 0.25, 10);
opt.tau_list = l2f_linspace(0.03, 0.20, 10);
opt.ttw_list = l2f_linspace(1.3, 5.0, 10);
opt.horizon = 300;
opt.seed = 7;
opt = apply_name_values(opt, varargin{:});

rows = numel(opt.mass_list) * numel(opt.tau_list) * numel(opt.ttw_list);
table_out = zeros(rows, 7);
row = 1;
for i = 1:numel(opt.mass_list)
    for j = 1:numel(opt.tau_list)
        for k = 1:numel(opt.ttw_list)
            cfg = l2f_uav_cfg_library('nominal_50g');
            cfg.mass = opt.mass_list(i);
            hover = cfg.mass * 9.80665 / 4.0;
            cfg.thrust_coeff_c0 = hover * ones(1, 4);
            cfg.thrust_coeff_c1 = (opt.ttw_list(k) - 1.0) * hover * ones(1, 4);
            cfg.motor_tau_rise = opt.tau_list(j) * ones(1, 4);
            cfg.motor_tau_fall = opt.tau_list(j) * ones(1, 4);
            sim_cfg = l2f_default_sim_cfg('horizon', opt.horizon, 'batch_size', 1, 'seed', opt.seed, 'uav_cfg', cfg);
            logs = l2f_rollout(l2f_default_params(), l2f_make_world('empty'), l2f_make_sensors('ideal'), controller, ...
                l2f_default_reward_cfg(), sim_cfg);
            metrics = l2f_metrics(logs, logs.world, logs.reward_cfg);
            table_out(row, :) = [
                opt.mass_list(i), opt.tau_list(j), opt.ttw_list(k), ...
                metrics.success_rate, metrics.max_position_error, metrics.max_omega, metrics.action_saturation_ratio
            ];
            row = row + 1;
        end
    end
end
end
