function [metrics, errors] = l2f_metrics(logs, world, reward_cfg)
%L2F_METRICS Compute position-hold recovery and rollout-integrity metrics.

if nargin < 2 || isempty(world)
    world = l2f_make_world('empty');
end
if nargin < 3 || isempty(reward_cfg)
    if isfield(logs, 'reward_cfg')
        reward_cfg = logs.reward_cfg;
    else
        reward_cfg = l2f_default_reward_cfg();
    end
end

[position_norm, velocity_norm, omega_norm] = l2f_error_series(logs, world);
hold = l2f_position_hold_series( ...
    position_norm, velocity_norm, omega_norm, reward_cfg);
survival_by_step = l2f_survival_from_logs( ...
    logs, position_norm, velocity_norm, omega_norm);

horizon = size(logs.action, 1);
count = size(logs.position, 2);
episode_return = sum(logs.reward, 1);
final_position = position_norm(end, :);
final_velocity = velocity_norm(end, :);
final_omega = omega_norm(end, :);
initial_position = max(position_norm(1, :), 1.0e-6);
overshoot = max(position_norm, [], 1) ./ initial_position - 1.0;

settling_time_s = nan(1, count);
settled = isfinite(hold.settling_step);
if any(settled)
    settling_time_s(settled) = logs.time(hold.settling_step(settled) + 1);
end
if horizon >= 1
    survival = survival_by_step(end, :);
else
    survival = false(1, count);
end
snapshot_success = hold.snapshot_success & survival;
steady_success = hold.steady_success & survival;

metrics = struct();
metrics.world_type = world.type;
metrics.count = count;
metrics.horizon = horizon;
metrics.gravity = logs.params.gravity;
metrics.steady_window_steps = hold.window_steps;
metrics.steady_required_fraction = hold.required_fraction;
metrics.return = episode_return;
metrics.return_mean = avg(episode_return);

% The generic success API intentionally maps to final-window steady success.
metrics.success = steady_success;
metrics.success_rate = avg(double(steady_success));
metrics.position_hold_snapshot = snapshot_success;
metrics.position_hold_snapshot_rate = avg(double(snapshot_success));
metrics.position_hold_steady = steady_success;
metrics.position_hold_steady_rate = avg(double(steady_success));
metrics.final_window_success_fraction = hold.final_window_success_fraction;
metrics.final_window_success_fraction_mean = avg_finite( ...
    hold.final_window_success_fraction);
metrics.position_hold_settling_step = hold.settling_step;
metrics.position_hold_settling_time_s = settling_time_s;
metrics.settling_time = settling_time_s;
metrics.position_hold_settled_rate = avg(double(settled));
metrics.position_hold_stay_fraction = hold.stay_fraction;
metrics.stay = hold.stay_fraction;
metrics.position_hold_stay_fraction_mean = avg_finite(hold.stay_fraction);
metrics.survival = survival;
metrics.survival_rate = avg(double(survival));
metrics.done_rate = avg(double(logs.done(end, :)));

metrics.position_final = final_position;
metrics.position_final_mean = avg(final_position);
metrics.position_max = max(position_norm, [], 1);
metrics.position_max_mean = avg(metrics.position_max);
metrics.max_position_error = max(metrics.position_max);
metrics.velocity_final = final_velocity;
metrics.velocity_final_mean = avg(final_velocity);
metrics.velocity_max = max(velocity_norm, [], 1);
metrics.velocity_max_mean = avg(metrics.velocity_max);
metrics.max_velocity = max(metrics.velocity_max);
metrics.omega_final = final_omega;
metrics.omega_final_mean = avg(final_omega);
metrics.omega_max = max(omega_norm, [], 1);
metrics.max_omega = max(metrics.omega_max);
metrics.omega_max_mean = avg(metrics.omega_max);
metrics.action_abs_mean = reshape(sum(sum(abs(logs.action), 3), 1) / ...
    max(size(logs.action, 1) * size(logs.action, 3), 1), 1, []);
metrics.motor_abs_mean = reshape(sum(sum(abs(logs.motor), 3), 1) / ...
    max(size(logs.motor, 1) * size(logs.motor, 3), 1), 1, []);
metrics.action_saturation_ratio = sum(abs(logs.action(:)) > 0.98) ...
    / max(numel(logs.action), 1);
metrics.control_energy = reshape(sum(sum(logs.action .* logs.action, 3), 1), 1, []);
metrics.control_energy_mean = avg(metrics.control_energy);
metrics.overshoot = overshoot;
metrics.overshoot_mean = avg(overshoot);
metrics.failure_by_config_bin = failure_by_mass_bin( ...
    logs.dynamics, steady_success);

errors = struct();
errors.position_norm = position_norm;
errors.velocity_norm = velocity_norm;
errors.omega_norm = omega_norm;
errors.step_success = hold.state_success;
errors.executed_step_success = hold.executed_success;
errors.window_success_fraction = hold.window_fraction;
errors.steady_success = hold.steady_success_by_step;
errors.survival = survival_by_step;
end

function value = avg(x)
value = sum(x(:)) / max(numel(x), 1);
end

function value = avg_finite(x)
x = x(isfinite(x));
if isempty(x)
    value = nan;
else
    value = sum(x(:)) / numel(x);
end
end

function bins = failure_by_mass_bin(dynamics, success)
bins = struct('edges', [], 'failure_rate', []);
if isempty(dynamics) || size(dynamics, 2) < 1 || size(dynamics, 1) < 2
    return;
end
mass = dynamics(:, 1).';
min_mass = min(mass);
max_mass = max(mass);
if max_mass <= min_mass
    return;
end
bin_count = min(5, numel(mass));
edges = l2f_linspace(min_mass, max_mass, bin_count + 1);
failure_rate = nan(1, bin_count);
for b = 1:bin_count
    if b == bin_count
        mask = mass >= edges(b) & mass <= edges(b + 1);
    else
        mask = mass >= edges(b) & mass < edges(b + 1);
    end
    if any(mask)
        failure_rate(b) = 1.0 - sum(success(mask)) / sum(mask);
    end
end
bins.edges = edges;
bins.failure_rate = failure_rate;
end
