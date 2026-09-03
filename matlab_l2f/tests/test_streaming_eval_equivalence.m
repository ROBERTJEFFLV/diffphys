
function tests = test_streaming_eval_equivalence
%TEST_STREAMING_EVAL_EQUIVALENCE Compare streaming and full-log evaluators.
tests = functiontests(localfunctions);
end

function testSmallDeterministicRollout(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cleanup = onCleanup(@() rmpath(root)); %#ok<NASGU>

weights_path = fullfile(tempdir, 'l2f_test_motor_gru_weights.mat');
create_test_weights(weights_path, 12);
file_cleanup = onCleanup(@() delete_if_exists(weights_path)); %#ok<NASGU>

params = l2f_default_params('dynamics_profile', 'physical-broad', 'broad_sampler', 'physical');
world = l2f_make_world('empty');
reward_cfg = l2f_default_reward_cfg( ...
    'steady_window_steps', 5, 'steady_required_fraction', 0.8);
sim_cfg = l2f_default_sim_cfg( ...
    'horizon', 24, 'batch_size', 7, 'seed', 19, ...
    'live_plot', false, 'freeze_done', false, ...
    'terminate_on_bounds', false, 'terminate_on_success', false, ...
    'stop_when_all_done', false);

initial_state = l2f_reset(sim_cfg.batch_size, params, sim_cfg.seed, [], sim_cfg);
sim_cfg.initial_state = initial_state;
horizons = [6 12 24];

controller = l2f_make_motor_gru_controller(weights_path);
sensors = l2f_make_sensors('ideal');
logs = l2f_rollout(params, world, sensors, controller, reward_cfg, sim_cfg);
full_result = l2f_eval_from_logs(logs, world, reward_cfg, horizons);
stream_result = l2f_motor_gru_eval_streaming( ...
    weights_path, params, world, reward_cfg, sim_cfg, horizons);

verifyEqual(testCase, stream_result.final_state.position, ...
    reshape(logs.position(end, :, :), sim_cfg.batch_size, 3), 'AbsTol', 0);
verifyEqual(testCase, stream_result.final_state.velocity, ...
    reshape(logs.velocity(end, :, :), sim_cfg.batch_size, 3), 'AbsTol', 0);
verifyEqual(testCase, stream_result.final_state.rotation, ...
    reshape(permute(logs.rotation(end, :, :, :), [3, 4, 2, 1]), 3, 3, sim_cfg.batch_size), 'AbsTol', 0);
verifyEqual(testCase, stream_result.final_state.motor, ...
    reshape(logs.motor(end, :, :), sim_cfg.batch_size, 4), 'AbsTol', 0);
verifyEqual(testCase, stream_result.metrics.success, full_result.metrics.success);
verifyEqual(testCase, stream_result.metrics.position_final, ...
    full_result.metrics.position_final, 'AbsTol', 1.0e-12);
verifyEqual(testCase, stream_result.metrics.position_hold_snapshot, ...
    full_result.metrics.position_hold_snapshot);
verifyEqual(testCase, stream_result.metrics.final_window_success_fraction, ...
    full_result.metrics.final_window_success_fraction, 'AbsTol', 1.0e-12);
verifyEqual(testCase, stream_result.metrics.position_hold_settling_time_s, ...
    full_result.metrics.position_hold_settling_time_s, 'AbsTol', 1.0e-12);
verifyEqual(testCase, stream_result.metrics.position_hold_stay_fraction, ...
    full_result.metrics.position_hold_stay_fraction, 'AbsTol', 1.0e-12);
verifyEqual(testCase, stream_result.metrics.survival, full_result.metrics.survival);
verifyEqual(testCase, stream_result.metrics.control_energy, ...
    full_result.metrics.control_energy, 'AbsTol', 1.0e-12);
verifyEqual(testCase, stream_result.metrics.return, ...
    full_result.metrics.return, 'AbsTol', 1.0e-12);
verifyEqual(testCase, stream_result.invalid_fraction, ...
    full_result.invalid_fraction, 'AbsTol', 0);

full_names = fieldnames(full_result.long_metrics.summary);
for i = 1:numel(full_names)
    name = full_names{i};
    verifyTrue(testCase, isfield(stream_result.long_metrics.summary, name));
    verifyEqual(testCase, stream_result.long_metrics.summary.(name), ...
        full_result.long_metrics.summary.(name), 'AbsTol', 1.0e-12);
end
verifyFalse(testCase, isfield(stream_result.metrics, 'angle_final_rad'));
verifyFalse(testCase, isfield(stream_result.summary, 'angle_final_deg_mean'));

names = full_result.long_metrics.sample.Properties.VariableNames;
for i = 1:numel(names)
    verifyTrue(testCase, ismember(names{i}, ...
        stream_result.long_metrics.sample.Properties.VariableNames));
    verifyEqual(testCase, stream_result.long_metrics.sample.(names{i}), ...
        full_result.long_metrics.sample.(names{i}), 'AbsTol', 1.0e-12);
end
stream_names = stream_result.long_metrics.sample.Properties.VariableNames;
for name = {'integral_world_norm_mean', 'integral_body_x_mean', ...
        'integral_clamp_ratio', 'integral_residual_action_rms', ...
        'damping_residual_action_rms', 'steady_motor_bias_0'}
    verifyTrue(testCase, ismember(name{1}, stream_names));
end
end

function testIntegral25StreamingMatchesFullRollout(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cleanup = onCleanup(@() rmpath(root)); %#ok<NASGU>
weights_path = fullfile(tempdir, 'l2f_test_motor_gru_weights_25d.mat');
create_test_weights(weights_path, 10, 25, 'integral25');
file_cleanup = onCleanup(@() delete_if_exists(weights_path)); %#ok<NASGU>
params = l2f_default_params('dynamics_profile', 'physical-broad', 'broad_sampler', 'physical');
world = l2f_make_world('empty');
reward_cfg = l2f_default_reward_cfg('steady_window_steps', 5, 'steady_required_fraction', 0.8);
sim_cfg = l2f_default_sim_cfg( ...
    'horizon', 20, 'batch_size', 5, 'seed', 29, 'live_plot', false, ...
    'freeze_done', false, 'terminate_on_bounds', false, ...
    'terminate_on_success', false, 'stop_when_all_done', false);
sim_cfg.initial_state = l2f_reset(sim_cfg.batch_size, params, sim_cfg.seed, [], sim_cfg);
logs = l2f_rollout(params, world, l2f_make_sensors('ideal'), ...
    l2f_make_motor_gru_controller(weights_path), reward_cfg, sim_cfg);
stream = l2f_motor_gru_eval_streaming( ...
    weights_path, params, world, reward_cfg, sim_cfg, [10 20]);
verifyEqual(testCase, stream.final_state.position, ...
    reshape(logs.position(end, :, :), sim_cfg.batch_size, 3), 'AbsTol', 1.0e-12);
verifyEqual(testCase, stream.final_state.omega, ...
    reshape(logs.omega(end, :, :), sim_cfg.batch_size, 3), 'AbsTol', 1.0e-12);
end

function create_test_weights(path_value, hidden_dim, input_dim, observation_mode)
rng(123);
if nargin < 3
    input_dim = 40;
end
if nargin < 4
    observation_mode = 'legacy40';
end
encoder_dim = hidden_dim;
weights = struct();
weights.hidden_dim = hidden_dim;
weights.input_dim = input_dim;
weights.observation_mode = observation_mode;
weights.integral_limit = 0.5;
weights.integral_leak = 0.0;
weights.negative_slope = 0.01;
weights.encoder_0_weight = randn(encoder_dim, input_dim) * 0.02;
weights.encoder_0_bias = randn(1, encoder_dim) * 0.01;
weights.encoder_2_weight = randn(encoder_dim, encoder_dim) * 0.02;
weights.encoder_2_bias = randn(1, encoder_dim) * 0.01;
weights.gru_weight_ih = randn(3 * hidden_dim, encoder_dim) * 0.02;
weights.gru_bias_ih = randn(1, 3 * hidden_dim) * 0.01;
weights.gru_weight_hh = randn(3 * hidden_dim, hidden_dim) * 0.02;
weights.gru_bias_hh = randn(1, 3 * hidden_dim) * 0.01;
weights.motor_head_weight = randn(4, hidden_dim) * 0.02;
weights.motor_head_bias = randn(1, 4) * 0.01;
save(path_value, '-struct', 'weights');
end

function delete_if_exists(path_value)
if exist(path_value, 'file')
    delete(path_value);
end
end
