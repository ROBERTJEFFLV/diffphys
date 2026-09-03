function out = l2f_long_horizon_metrics(logs, world, reward_cfg, requested_horizons, errors)
%L2F_LONG_HORIZON_METRICS Snapshot, steady, settling, stay, and survival metrics.

horizon = size(logs.action, 1);
count = size(logs.position, 2);
requested_horizons = reshape(double(requested_horizons), 1, []);
requested_horizons = requested_horizons(isfinite(requested_horizons));
requested_horizons = unique(round(requested_horizons));
horizons = requested_horizons(requested_horizons >= 1 & requested_horizons <= horizon);

if nargin < 5 || isempty(errors) ...
        || ~isfield(errors, 'step_success') || ~isfield(errors, 'survival')
    [~, errors] = l2f_metrics(logs, world, reward_cfg);
end
position_norm = errors.position_norm;
velocity_norm = errors.velocity_norm;
omega_norm = errors.omega_norm;

summary = struct();
sample = struct();
for j = 1:numel(horizons)
    h = horizons(j);
    suffix = sprintf('H%d', h);
    idx = h + 1;
    hold_h = l2f_position_hold_series( ...
        position_norm(1:idx, :), velocity_norm(1:idx, :), ...
        omega_norm(1:idx, :), reward_cfg);
    survival = errors.survival(h, :).';
    snapshot = hold_h.snapshot_success.' & survival;
    steady = hold_h.steady_success.' & survival;
    final_fraction = hold_h.final_window_success_fraction.';
    settling_time_s = hold_h.settling_step.' * logs.params.dt;

    sample.(['position_hold_snapshot_' suffix]) = double(snapshot);
    sample.(['position_hold_steady_' suffix]) = double(steady);
    sample.(['final_window_success_fraction_' suffix]) = final_fraction;
    sample.(['position_hold_settling_time_s_' suffix]) = settling_time_s;
    sample.(['position_hold_stay_fraction_' suffix]) = hold_h.stay_fraction.';
    sample.(['survival_' suffix]) = double(survival);
    sample.(['position_' suffix '_m']) = position_norm(idx, :).';
    sample.(['velocity_' suffix]) = velocity_norm(idx, :).';
    sample.(['omega_' suffix]) = omega_norm(idx, :).';

    summary.(['position_hold_snapshot_' suffix '_rate']) = avg(double(snapshot));
    summary.(['position_hold_steady_' suffix '_rate']) = avg(double(steady));
    summary.(['final_window_success_fraction_' suffix]) = avg_finite(final_fraction);
    summary.(['position_hold_settled_' suffix '_rate']) = ...
        avg(double(isfinite(settling_time_s)));
    summary.(['position_hold_settling_time_s_' suffix '_mean']) = ...
        avg_finite(settling_time_s);
    summary.(['position_hold_stay_fraction_' suffix '_mean']) = ...
        avg_finite(hold_h.stay_fraction);
    summary.(['survival_rate_' suffix]) = avg(double(survival));
    summary.(['position_final_mean_' suffix]) = avg(position_norm(idx, :));
    summary.(['velocity_final_mean_' suffix]) = avg(velocity_norm(idx, :));
    summary.(['omega_final_mean_' suffix]) = avg(omega_norm(idx, :));
    summary.(['invalid_fraction_' suffix]) = l2f_invalid_fraction_logs(logs, h);
    checkpoint_tail_steps = min(reward_cfg.steady_window_steps, h);
    checkpoint_state_tail = (h + 2 - checkpoint_tail_steps):(h + 1);
    checkpoint_action_tail = (h + 1 - checkpoint_tail_steps):h;
    [checkpoint_axis_sample, checkpoint_axis_summary] = l2f_tail_axis_diagnostics( ...
        position_norm(checkpoint_state_tail, :), ...
        velocity_norm(checkpoint_state_tail, :), ...
        logs.omega(checkpoint_state_tail, :, :), ...
        logs.action(checkpoint_action_tail, :, :), logs.params.dt);
    checkpoint_axis_names = fieldnames(checkpoint_axis_sample);
    for axis_field_index = 1:numel(checkpoint_axis_names)
        name = checkpoint_axis_names{axis_field_index};
        sample.([name '_' suffix]) = checkpoint_axis_sample.(name);
    end
    checkpoint_axis_summary_names = fieldnames(checkpoint_axis_summary);
    for axis_field_index = 1:numel(checkpoint_axis_summary_names)
        name = checkpoint_axis_summary_names{axis_field_index};
        summary.([name '_' suffix]) = checkpoint_axis_summary.(name);
    end
end

anchor_horizon = min(500, horizon);
after_range = (anchor_horizon + 1):(horizon + 1);
position_after = position_norm(after_range, :);
velocity_after = velocity_norm(after_range, :);
omega_after = omega_norm(after_range, :);
position_rms = rms_columns(position_after);
velocity_rms = rms_columns(velocity_after);
omega_rms = rms_columns(omega_after);
position_max_after = max(position_after, [], 1).';
velocity_max_after = max(velocity_after, [], 1).';
omega_max_after = max(omega_after, [], 1).';

sample.position_rms_after_H500 = position_rms;
sample.velocity_rms_after_H500 = velocity_rms;
sample.omega_rms_after_H500 = omega_rms;
sample.max_position_after_H500 = position_max_after;
sample.max_velocity_after_H500 = velocity_max_after;
sample.max_omega_after_H500 = omega_max_after;
summary.position_rms_after_H500_mean = avg_finite(position_rms);
summary.velocity_rms_after_H500_mean = avg_finite(velocity_rms);
summary.omega_rms_after_H500_mean = avg_finite(omega_rms);
summary.max_position_after_H500_mean = avg_finite(position_max_after);
summary.max_position_after_H500 = max_finite(position_max_after);
summary.max_velocity_after_H500_mean = avg_finite(velocity_max_after);
summary.max_velocity_after_H500 = max_finite(velocity_max_after);
summary.max_omega_after_H500_mean = avg_finite(omega_max_after);
summary.max_omega_after_H500 = max_finite(omega_max_after);

tail_steps = min(reward_cfg.steady_window_steps, horizon);
state_tail = (horizon + 2 - tail_steps):(horizon + 1);
action_tail = (horizon + 1 - tail_steps):horizon;
omega_tail = logs.omega(state_tail, :, :);
action_tail_values = logs.action(action_tail, :, :);
[axis_sample, axis_summary] = l2f_tail_axis_diagnostics( ...
    position_norm(state_tail, :), velocity_norm(state_tail, :), ...
    omega_tail, action_tail_values, logs.params.dt);
axis_sample_names = fieldnames(axis_sample);
for i = 1:numel(axis_sample_names)
    name = axis_sample_names{i};
    sample.(name) = axis_sample.(name);
end
axis_summary_names = fieldnames(axis_summary);
for i = 1:numel(axis_summary_names)
    name = axis_summary_names{i};
    summary.(name) = axis_summary.(name);
end

if horizon >= 2
    action_delta = diff(logs.action, 1, 1);
    action_delta_sq = sum(action_delta .* action_delta, 3);
    action_delta_rms = reshape(sqrt(mean(action_delta_sq, 1)), [], 1);
else
    action_delta_rms = zeros(count, 1);
end
sample.action_delta_rms = action_delta_rms;
summary.action_delta_rms_mean = avg_finite(action_delta_rms);
summary.action_delta_rms_max = max_finite(action_delta_rms);

if isfield(logs, 'controller_hidden_norm')
    hidden_norm_max = column_max_finite(logs.controller_hidden_norm);
    hidden_abs_max = column_max_finite(logs.controller_hidden_abs_max);
else
    hidden_norm_max = nan(count, 1);
    hidden_abs_max = nan(count, 1);
end
sample.hidden_state_norm_max = hidden_norm_max;
sample.hidden_state_abs_max = hidden_abs_max;
summary.hidden_state_norm_max_mean = avg_finite(hidden_norm_max);
summary.hidden_state_norm_max = max_finite(hidden_norm_max);
summary.hidden_state_abs_max_mean = avg_finite(hidden_abs_max);
summary.hidden_state_abs_max = max_finite(hidden_abs_max);

out = struct();
out.summary = summary;
if isempty(fieldnames(sample))
    out.sample = table();
else
    out.sample = struct2table(sample);
end
end

function out = rms_columns(x)
out = reshape(sqrt(mean(x .* x, 1)), [], 1);
end

function out = column_max_finite(x)
count = size(x, 2);
out = nan(count, 1);
for i = 1:count
    values = x(:, i);
    values = values(isfinite(values));
    if ~isempty(values)
        out(i) = max(values);
    end
end
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

function value = max_finite(x)
x = x(isfinite(x));
if isempty(x)
    value = nan;
else
    value = max(x(:));
end
end
