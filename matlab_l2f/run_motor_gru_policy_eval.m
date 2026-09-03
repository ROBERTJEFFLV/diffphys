
function metrics = run_motor_gru_policy_eval(varargin)
%RUN_MOTOR_GRU_POLICY_EVAL Evaluate an exported MotorGRUPolicy.
%
% Default: one streaming H10000 rollout with snapshots at
% H500/H1000/H2000/H5000/H10000. Use eval_mode='full' only when complete
% trajectory logs are required.

parser = inputParser;
addParameter(parser, 'weights_path', '');
addParameter(parser, 'output_path', '');
addParameter(parser, 'mat_output_path', '');
addParameter(parser, 'sample_output_path', '');
addParameter(parser, 'save_logs', false);
addParameter(parser, 'eval_mode', 'auto');
addParameter(parser, 'horizon', 10000);
addParameter(parser, 'long_horizons', [500 1000 2000 5000 10000]);
addParameter(parser, 'batch_size', 1024);
addParameter(parser, 'seed', 7);
addParameter(parser, 'dynamics_profile', 'physical-broad');
addParameter(parser, 'broad_sampler', 'physical');
parse(parser, varargin{:});
opt = parser.Results;

if isempty(opt.weights_path)
    error('weights_path is required.');
end
requested_horizons = normalize_horizons(opt.long_horizons);
simulation_horizon = max([round(double(opt.horizon)), requested_horizons]);
if isempty(simulation_horizon) || simulation_horizon < 1
    error('At least one positive evaluation horizon is required.');
end

params = l2f_default_params( ...
    'dynamics_profile', opt.dynamics_profile, ...
    'broad_sampler', opt.broad_sampler);
world = l2f_make_world('empty');
reward_cfg = l2f_default_reward_cfg();
sim_cfg = l2f_default_sim_cfg( ...
    'horizon', simulation_horizon, ...
    'batch_size', opt.batch_size, ...
    'seed', opt.seed, ...
    'live_plot', false, ...
    'freeze_done', false, ...
    'terminate_on_bounds', false, ...
    'terminate_on_success', false, ...
    'stop_when_all_done', false);

mode = resolve_eval_mode(opt.eval_mode, opt.save_logs);
logs = [];
tic_id = tic;
switch mode
    case 'streaming'
        result = l2f_motor_gru_eval_streaming( ...
            opt.weights_path, params, world, reward_cfg, sim_cfg, requested_horizons);
    case 'full'
        sensors = l2f_make_sensors('ideal');
        controller = l2f_make_motor_gru_controller(opt.weights_path);
        logs = l2f_rollout(params, world, sensors, controller, reward_cfg, sim_cfg);
        result = l2f_eval_from_logs(logs, world, reward_cfg, requested_horizons);
    otherwise
        error('Unsupported evaluation mode: %s', mode);
end
elapsed_s = toc(tic_id);

metrics = result.metrics;
long_metrics = result.long_metrics;
summary = result.summary;
summary.elapsed_s = elapsed_s;
summary.simulated_vehicle_seconds = double(metrics.count) * simulation_horizon * params.dt;
summary.vehicle_seconds_per_wall_second = ...
    summary.simulated_vehicle_seconds / max(elapsed_s, eps);

if ~isempty(opt.output_path)
    ensure_parent_directory(opt.output_path);
    writetable(struct2table(summary), opt.output_path);
end
if ~isempty(opt.sample_output_path)
    ensure_parent_directory(opt.sample_output_path);
    writetable(l2f_eval_sample_table( ...
        result.dynamics, metrics, long_metrics), opt.sample_output_path);
end
if ~isempty(opt.mat_output_path)
    ensure_parent_directory(opt.mat_output_path);
    if opt.save_logs
        if isempty(logs)
            error('save_logs=true requires eval_mode=''full'' or eval_mode=''auto''.');
        end
        save(opt.mat_output_path, 'metrics', 'summary', 'long_metrics', 'logs', '-v7.3');
    else
        save(opt.mat_output_path, 'metrics', 'summary', 'long_metrics', '-v7');
    end
end

fprintf('Evaluation mode: %s\n', mode);
disp(summary);
end

function horizons = normalize_horizons(values)
horizons = reshape(double(values), 1, []);
horizons = horizons(isfinite(horizons));
horizons = unique(round(horizons));
horizons = horizons(horizons >= 1);
end

function mode = resolve_eval_mode(value, save_logs)
mode = lower(char(string(value)));
if strcmp(mode, 'auto')
    if save_logs
        mode = 'full';
    else
        mode = 'streaming';
    end
end
if ~any(strcmp(mode, {'streaming', 'full'}))
    error('eval_mode must be ''auto'', ''streaming'', or ''full''.');
end
if save_logs && strcmp(mode, 'streaming')
    error('save_logs=true is incompatible with eval_mode=''streaming''.');
end
end

function ensure_parent_directory(path_value)
[out_dir, ~, ~] = fileparts(path_value);
if ~isempty(out_dir) && ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
end
