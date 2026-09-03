function tests = test_position_hold_semantics
%TEST_POSITION_HOLD_SEMANTICS Regression tests for the migrated task contract.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
testCase.TestData.root = root;
end

function teardownOnce(testCase)
rmpath(testCase.TestData.root);
end

function testDefaultContract(testCase)
cfg = l2f_default_reward_cfg();
verifyEqual(testCase, cfg.steady_window_steps, 100);
verifyEqual(testCase, cfg.steady_required_fraction, 0.95);
verifyFalse(testCase, isfield(cfg, 'w_attitude'));
verifyFalse(testCase, isfield(cfg, 'success_angle_rad'));

sim_cfg = l2f_default_sim_cfg();
verifyFalse(testCase, isfield(sim_cfg, 'max_angle_rad'));
end

function testWindowSettlingStayAndSurvival(testCase)
cfg = l2f_default_reward_cfg( ...
    'success_position_m', 0.5, ...
    'steady_window_steps', 4, ...
    'steady_required_fraction', 0.75);
% Executed-step success: [1 1 0 1 0 1]. The first full window settles at
% step 4, stay after entry is 1/2, and the final window fraction is 1/2.
logs = synthetic_logs([0, 0, 1, 0, 1, 0]);
logs.done(5, 1) = true;
[metrics, errors] = l2f_metrics(logs, logs.world, cfg);

verifyFalse(testCase, metrics.position_hold_snapshot);
verifyFalse(testCase, metrics.position_hold_steady);
verifyFalse(testCase, metrics.success);
verifyEqual(testCase, metrics.final_window_success_fraction, 0.5, 'AbsTol', 0);
verifyEqual(testCase, metrics.position_hold_settling_step, 4, 'AbsTol', 0);
verifyEqual(testCase, metrics.position_hold_settling_time_s, 0.04, 'AbsTol', 1.0e-12);
verifyEqual(testCase, metrics.position_hold_stay_fraction, 0.5, 'AbsTol', 0);
verifyFalse(testCase, metrics.survival);
verifyTrue(testCase, errors.survival(4, 1));
verifyFalse(testCase, errors.survival(5, 1));

% A numerically stable frozen state after a physical failure must not become
% a main success merely because its p/v/omega window still passes.
crashed_logs = synthetic_logs(zeros(1, 6));
crashed_logs.done(5, 1) = true;
[crashed_metrics, crashed_errors] = l2f_metrics( ...
    crashed_logs, crashed_logs.world, cfg);
verifyTrue(testCase, crashed_errors.steady_success(end, 1));
verifyFalse(testCase, crashed_metrics.position_hold_steady);
verifyFalse(testCase, crashed_metrics.success);
verifyFalse(testCase, crashed_metrics.survival);

long_metrics = l2f_long_horizon_metrics( ...
    logs, logs.world, cfg, [3, 4, 6], errors);
sample = long_metrics.sample;
verifyEqual(testCase, sample.position_hold_snapshot_H3, 0);
verifyEqual(testCase, sample.position_hold_steady_H3, 0);
verifyTrue(testCase, isnan(sample.final_window_success_fraction_H3));
verifyEqual(testCase, sample.position_hold_steady_H4, 1);
verifyEqual(testCase, sample.final_window_success_fraction_H4, 0.75, 'AbsTol', 0);
verifyEqual(testCase, sample.position_hold_steady_H6, 0);
verifyEqual(testCase, sample.survival_H4, 1);
verifyEqual(testCase, sample.survival_H6, 0);
end

function testPidDoesNotTrackYawOrientation(testCase)
params = l2f_default_params();
state = l2f_reset(1, params, 41, [], l2f_default_sim_cfg());
state.position(:) = 0;
state.velocity(:) = 0;
state.omega(:) = [0, 0, 0.3];
state.rotation(:, :, 1) = eye(3);
action_identity = l2f_pid_action(state, params, []);

theta = 1.1;
state.rotation(:, :, 1) = [cos(theta), -sin(theta), 0; ...
    sin(theta), cos(theta), 0; 0, 0, 1];
action_yawed = l2f_pid_action(state, params, []);
verifyEqual(testCase, action_yawed, action_identity, 'AbsTol', 1.0e-12);
end

function testTailAxisDiagnosticsParityFixture(testCase)
t = (0:99).' * 0.01;
position_norm = 0.04 * ones(100, 1);
velocity_norm = 0.09 * ones(100, 1);
omega = zeros(100, 1, 3);
omega(:, 1, 1) = sin(2.0 * pi * 5.0 * t);
omega(:, 1, 2) = 0.25;
action = zeros(100, 1, 4);
action(:, 1, 1) = sin(2.0 * pi * 2.0 * t);
[sample, summary] = l2f_tail_axis_diagnostics( ...
    position_norm, velocity_norm, omega, action, 0.01);

verifyEqual(testCase, sample.omega_x_tail_rms, sqrt(0.5), 'AbsTol', 1.0e-12);
verifyEqual(testCase, sample.omega_y_tail_rms, 0.25, 'AbsTol', 1.0e-12);
verifyEqual(testCase, sample.omega_x_tail_spectral_peak_hz, 5.0, 'AbsTol', 1.0e-12);
verifyEqual(testCase, sample.omega_y_tail_spectral_peak_hz, 0.0, 'AbsTol', 1.0e-12);
verifyEqual(testCase, sample.strict_bounded_angular_motion, 1.0);
verifyEqual(testCase, sample.loose_bounded_angular_motion, 1.0);
verifyEqual(testCase, summary.strict_bounded_angular_motion_rate, 1.0);
end

function logs = synthetic_logs(position_x)
horizon = numel(position_x);
count = 1;
logs = struct();
logs.time = (0:horizon).' * 0.01;
logs.position = zeros(horizon + 1, count, 3);
logs.position(2:end, 1, 1) = reshape(position_x, [], 1);
logs.velocity = zeros(horizon + 1, count, 3);
logs.omega = zeros(horizon + 1, count, 3);
logs.rotation = zeros(horizon + 1, count, 3, 3);
for t = 1:(horizon + 1)
    logs.rotation(t, 1, :, :) = eye(3);
end
logs.action = zeros(horizon, count, 4);
logs.motor = zeros(horizon, count, 4);
logs.reward = zeros(horizon, count);
logs.done = false(horizon, count);
logs.dynamics = zeros(count, 34);
logs.params = struct('gravity', 9.80665, 'dt', 0.01);
logs.sim_cfg = l2f_default_sim_cfg( ...
    'horizon', horizon, 'batch_size', count, ...
    'terminate_on_bounds', false, 'freeze_done', false);
logs.world = l2f_make_world('empty');
end
