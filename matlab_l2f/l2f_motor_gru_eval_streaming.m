
function result = l2f_motor_gru_eval_streaming(weights_input, params, world, reward_cfg, sim_cfg, requested_horizons)
%L2F_MOTOR_GRU_EVAL_STREAMING Memory-bounded MotorGRU evaluation.
%
% The physical step, integration, dt, controller and numeric type are
% unchanged. Full trajectory logging is replaced by online accumulation.

if ischar(weights_input) || (isstring(weights_input) && isscalar(weights_input))
    weights = load(char(weights_input));
elseif isstruct(weights_input)
    weights = weights_input;
else
    error('weights_input must be a MAT path or a weights struct.');
end
weights = l2f_prepare_motor_gru_weights(weights);

if isfield(sim_cfg, 'dt') && ~isempty(sim_cfg.dt)
    params.dt = sim_cfg.dt;
end
if isfield(sim_cfg, 'gravity') && ~isempty(sim_cfg.gravity)
    params.gravity = sim_cfg.gravity;
end
horizon = round(double(sim_cfg.horizon));
if horizon < 1
    error('sim_cfg.horizon must be at least 1.');
end
if isfield(sim_cfg, 'live_plot') && sim_cfg.live_plot
    error('Streaming evaluation does not support live_plot.');
end
if isfield(sim_cfg, 'stop_when_all_done') && sim_cfg.stop_when_all_done
    error('Streaming evaluation requires stop_when_all_done=false.');
end
if isfield(sim_cfg, 'terminate_on_success') && sim_cfg.terminate_on_success
    error(['terminate_on_success is unsupported for steady position-hold success; ' ...
        'evaluate the complete steady window instead.']);
end

uav_cfg = resolve_uav_cfg(sim_cfg);
if isfield(sim_cfg, 'initial_state') && ~isempty(sim_cfg.initial_state)
    state = sim_cfg.initial_state;
    batch_size = size(state.position, 1);
else
    batch_size = sim_cfg.batch_size;
    state = l2f_reset(batch_size, params, sim_cfg.seed, uav_cfg, sim_cfg);
end

requested_horizons = reshape(double(requested_horizons), 1, []);
requested_horizons = requested_horizons(isfinite(requested_horizons));
requested_horizons = unique(round(requested_horizons));
horizons = requested_horizons(requested_horizons >= 1 & requested_horizons <= horizon);
horizon_count = numel(horizons);

hidden = zeros(batch_size, double(weights.hidden_dim), 'like', state.position);
integral_position = zeros(batch_size, 3, 'like', state.position);
reference_info = make_reference_info(world);
dynamics = pack_dynamics_batch(state);

[position_norm, velocity_norm, omega_norm] = ...
    state_error_snapshot(state, world, 0.0, reference_info);
initial_position = max(position_norm, 1.0e-6);
position_max = position_norm;
velocity_max = velocity_norm;
omega_max = omega_norm;

return_sum = zeros(1, batch_size);
action_abs_sum = zeros(1, batch_size);
motor_abs_sum = zeros(1, batch_size);
control_energy = zeros(1, batch_size);
action_saturation_count = 0;
action_value_count = 0;
action_delta_sq_sum = zeros(1, batch_size);
previous_logged_action = [];
integral_world_norm_sum = zeros(1, batch_size);
integral_body_sum = zeros(batch_size, 3, 'like', state.position);
integral_clamp_count = zeros(1, batch_size);
integral_residual_sq_sum = zeros(1, batch_size);
damping_residual_sq_sum = zeros(1, batch_size);

hidden_norm_max = -inf(1, batch_size);
hidden_abs_max = -inf(1, batch_size);
hidden_norm_seen = false(1, batch_size);
hidden_abs_seen = false(1, batch_size);

done_accum = false(1, batch_size);
[window_steps, required_fraction] = l2f_position_hold_settings(reward_cfg);
window_buffer = false(window_steps, batch_size);
window_sum = zeros(1, batch_size);
diagnostic_position_buffer = zeros(window_steps, batch_size, 'like', state.position);
diagnostic_velocity_buffer = zeros(window_steps, batch_size, 'like', state.position);
diagnostic_omega_buffer = zeros(window_steps, batch_size, 3, 'like', state.position);
diagnostic_action_buffer = zeros(window_steps, batch_size, 4, 'like', state.position);
window_fraction = nan(1, batch_size);
steady_success = false(1, batch_size);
settling_step = nan(1, batch_size);
stay_success_sum = zeros(1, batch_size);
stay_success_count = zeros(1, batch_size);
survival_accum = true(1, batch_size);

invalid_count = state_invalid_count(state);
value_count = 18 * batch_size;

snapshot_position = nan(batch_size, horizon_count);
snapshot_velocity = nan(batch_size, horizon_count);
snapshot_omega = nan(batch_size, horizon_count);
snapshot_success = false(batch_size, horizon_count);
snapshot_steady = false(batch_size, horizon_count);
snapshot_window_fraction = nan(batch_size, horizon_count);
snapshot_settling_step = nan(batch_size, horizon_count);
snapshot_stay_fraction = nan(batch_size, horizon_count);
snapshot_survival = false(batch_size, horizon_count);
snapshot_position_pass_count = zeros(batch_size, horizon_count);
snapshot_velocity_pass_count = zeros(batch_size, horizon_count);
snapshot_omega_pass_count = zeros(batch_size, horizon_count);
snapshot_invalid_fraction = nan(1, horizon_count);
snapshot_axis_sample = cell(1, horizon_count);
snapshot_axis_summary = cell(1, horizon_count);
snapshot_cursor = 1;

post_start_step = min(500, horizon);
post_position_sq_sum = zeros(1, batch_size);
post_velocity_sq_sum = zeros(1, batch_size);
post_omega_sq_sum = zeros(1, batch_size);
post_position_max = -inf(1, batch_size);
post_velocity_max = -inf(1, batch_size);
post_omega_max = -inf(1, batch_size);
post_count = 0;

for step = 1:horizon
    t = (step - 1) * params.dt;
    previous_state = state;

    [observation, observed_position] = l2f_observation( ...
        state, weights.observation_mode, integral_position, ...
        weights.integral_input_frame, weights.integral_input_multiplier);
    [action, hidden, action_details] = l2f_motor_gru_forward(weights, observation, hidden);
    integral_world_norm_sum = integral_world_norm_sum + ...
        sqrt(sum(integral_position .* integral_position, 2)).';
    if strcmp(weights.observation_mode, 'integral25')
        integral_body_sum = integral_body_sum + observation(:, 19:21);
    end
    integral_residual_sq_sum = integral_residual_sq_sum + ...
        mean(action_details.integral_action_contribution .^ 2, 2).';
    damping_residual_sq_sum = damping_residual_sq_sum + ...
        mean(action_details.damping_action_contribution .^ 2, 2).';
    integral_position = l2f_update_position_integral( ...
        integral_position, observed_position, params.dt, ...
        weights.integral_limit, weights.integral_leak);
    integral_clamp_count = integral_clamp_count + ...
        mean(abs(integral_position) >= double(weights.integral_limit) - 1.0e-7, 2).';
    action = ensure_action_shape(action, batch_size);

    hidden_norm = sqrt(sum(hidden .* hidden, 2)).';
    hidden_abs = max(abs(hidden), [], 2).';
    finite_mask = isfinite(hidden_norm);
    hidden_norm_max(finite_mask) = max(hidden_norm_max(finite_mask), hidden_norm(finite_mask));
    hidden_norm_seen = hidden_norm_seen | finite_mask;
    finite_mask = isfinite(hidden_abs);
    hidden_abs_max(finite_mask) = max(hidden_abs_max(finite_mask), hidden_abs(finite_mask));
    hidden_abs_seen = hidden_abs_seen | finite_mask;

    if sim_cfg.freeze_done
        action(done_accum, :) = 0.0;
    end

    [candidate_state, aux] = l2f_step(state, action, params, world, t);
    if sim_cfg.freeze_done && any(done_accum)
        candidate_state = select_state(previous_state, candidate_state, ~done_accum);
        aux.command(done_accum, :) = 0.0;
        aux.motor(done_accum, :) = previous_state.motor(done_accum, :);
        aux.thrust(done_accum, :) = 0.0;
        aux.torque(done_accum, :) = 0.0;
    end

    reward = reward_step(candidate_state, aux.command, ...
        previous_state.previous_action, reward_cfg, world, t, reference_info);
    done_step = termination_step(candidate_state, reward_cfg, sim_cfg, world, t, reference_info);
    done_accum = done_accum | done_step;

    return_sum = return_sum + reward;
    action_abs_sum = action_abs_sum + sum(abs(aux.command), 2).';
    motor_abs_sum = motor_abs_sum + sum(abs(aux.motor), 2).';
    control_energy = control_energy + sum(aux.command .* aux.command, 2).';
    action_saturation_count = action_saturation_count + nnz(abs(aux.command) > 0.98);
    action_value_count = action_value_count + numel(aux.command);
    if step > 1
        delta = aux.command - previous_logged_action;
        action_delta_sq_sum = action_delta_sq_sum + sum(delta .* delta, 2).';
    end
    previous_logged_action = aux.command;

    invalid_count = invalid_count + nnz(~isfinite(aux.command)) + nnz(~isfinite(aux.motor));
    value_count = value_count + 8 * batch_size;

    state = candidate_state;
    state_time = step * params.dt;
    [position_norm, velocity_norm, omega_norm] = ...
        state_error_snapshot(state, world, state_time, reference_info);
    position_max = max(position_max, position_norm);
    velocity_max = max(velocity_max, velocity_norm);
    omega_max = max(omega_max, omega_norm);

    invalid_count = invalid_count + state_invalid_count(state);
    value_count = value_count + 18 * batch_size;
    within = within_success(position_norm, velocity_norm, omega_norm, reward_cfg);

    % Stay is the point-success fraction strictly after the first complete
    % steady window, so update existing entrants before admitting new ones.
    already_settled = isfinite(settling_step);
    stay_success_sum(already_settled) = stay_success_sum(already_settled) ...
        + double(within(already_settled));
    stay_success_count(already_settled) = stay_success_count(already_settled) + 1;

    slot = mod(step - 1, window_steps) + 1;
    if step > window_steps
        window_sum = window_sum - double(window_buffer(slot, :));
    end
    window_buffer(slot, :) = within;
    window_sum = window_sum + double(within);
    diagnostic_position_buffer(slot, :) = position_norm;
    diagnostic_velocity_buffer(slot, :) = velocity_norm;
    diagnostic_omega_buffer(slot, :, :) = reshape(state.omega, 1, batch_size, 3);
    diagnostic_action_buffer(slot, :, :) = reshape(aux.command, 1, batch_size, 4);
    if step >= window_steps
        window_fraction = window_sum / window_steps;
        steady_success = window_fraction >= required_fraction;
        newly_settled = ~isfinite(settling_step) & steady_success;
        settling_step(newly_settled) = step;
    else
        window_fraction(:) = nan;
        steady_success(:) = false;
    end

    step_survived = state_survived( ...
        state, aux.command, aux.motor, done_step, sim_cfg, world, ...
        position_norm, velocity_norm, omega_norm);
    survival_accum = survival_accum & step_survived;

    if step >= post_start_step
        post_position_sq_sum = post_position_sq_sum + position_norm .* position_norm;
        post_velocity_sq_sum = post_velocity_sq_sum + velocity_norm .* velocity_norm;
        post_omega_sq_sum = post_omega_sq_sum + omega_norm .* omega_norm;
        post_position_max = max(post_position_max, position_norm);
        post_velocity_max = max(post_velocity_max, velocity_norm);
        post_omega_max = max(post_omega_max, omega_norm);
        post_count = post_count + 1;
    end

    if snapshot_cursor <= horizon_count && step == horizons(snapshot_cursor)
        snapshot_position(:, snapshot_cursor) = position_norm.';
        snapshot_velocity(:, snapshot_cursor) = velocity_norm.';
        snapshot_omega(:, snapshot_cursor) = omega_norm.';
        snapshot_success(:, snapshot_cursor) = (within & survival_accum).';
        snapshot_steady(:, snapshot_cursor) = (steady_success & survival_accum).';
        snapshot_window_fraction(:, snapshot_cursor) = window_fraction.';
        snapshot_settling_step(:, snapshot_cursor) = settling_step.';
        current_stay = nan(1, batch_size);
        has_stay_samples = stay_success_count > 0;
        current_stay(has_stay_samples) = stay_success_sum(has_stay_samples) ...
            ./ stay_success_count(has_stay_samples);
        snapshot_stay_fraction(:, snapshot_cursor) = current_stay.';
        snapshot_survival(:, snapshot_cursor) = survival_accum.';
        snapshot_invalid_fraction(snapshot_cursor) = invalid_count / max(value_count, 1);
        if step < window_steps
            current_diagnostic_order = 1:step;
        else
            current_oldest_slot = mod(step, window_steps) + 1;
            current_diagnostic_order = [current_oldest_slot:window_steps, ...
                1:(current_oldest_slot - 1)];
        end
        [snapshot_axis_sample{snapshot_cursor}, snapshot_axis_summary{snapshot_cursor}] = ...
            l2f_tail_axis_diagnostics( ...
                diagnostic_position_buffer(current_diagnostic_order, :), ...
                diagnostic_velocity_buffer(current_diagnostic_order, :), ...
                diagnostic_omega_buffer(current_diagnostic_order, :, :), ...
                diagnostic_action_buffer(current_diagnostic_order, :, :), params.dt);
        current_position = diagnostic_position_buffer(current_diagnostic_order, :);
        current_velocity = diagnostic_velocity_buffer(current_diagnostic_order, :);
        current_omega = diagnostic_omega_buffer(current_diagnostic_order, :, :);
        current_omega_norm = sqrt(sum(current_omega .* current_omega, 3));
        snapshot_position_pass_count(:, snapshot_cursor) = ...
            sum(current_position < reward_cfg.success_position_m, 1).';
        snapshot_velocity_pass_count(:, snapshot_cursor) = ...
            sum(current_velocity < reward_cfg.success_velocity, 1).';
        snapshot_omega_pass_count(:, snapshot_cursor) = ...
            sum(current_omega_norm < reward_cfg.success_omega, 1).';
        snapshot_cursor = snapshot_cursor + 1;
    end
end

if horizon < 500
    post_position_sq_sum = position_norm .* position_norm;
    post_velocity_sq_sum = velocity_norm .* velocity_norm;
    post_omega_sq_sum = omega_norm .* omega_norm;
    post_position_max = position_norm;
    post_velocity_max = velocity_norm;
    post_omega_max = omega_norm;
    post_count = 1;
end

snapshot_success_final = within_success( ...
    position_norm, velocity_norm, omega_norm, reward_cfg) & survival_accum;
success = steady_success & survival_accum;
settling_time_s = settling_step * params.dt;
stay_fraction = nan(1, batch_size);
has_stay_samples = stay_success_count > 0;
stay_fraction(has_stay_samples) = stay_success_sum(has_stay_samples) ...
    ./ stay_success_count(has_stay_samples);
overshoot = position_max ./ initial_position - 1.0;

metrics = struct();
metrics.world_type = world.type;
metrics.count = batch_size;
metrics.horizon = horizon;
metrics.gravity = params.gravity;
metrics.steady_window_steps = window_steps;
metrics.steady_required_fraction = required_fraction;
metrics.return = return_sum;
metrics.return_mean = avg(return_sum);
metrics.success = success;
metrics.success_rate = avg(double(success));
metrics.position_hold_snapshot = snapshot_success_final;
metrics.position_hold_snapshot_rate = avg(double(snapshot_success_final));
metrics.position_hold_steady = success;
metrics.position_hold_steady_rate = avg(double(success));
metrics.final_window_success_fraction = window_fraction;
metrics.final_window_success_fraction_mean = avg_finite(window_fraction);
metrics.position_hold_settling_step = settling_step;
metrics.position_hold_settling_time_s = settling_time_s;
metrics.settling_time = settling_time_s;
metrics.position_hold_settled_rate = avg(double(isfinite(settling_step)));
metrics.position_hold_stay_fraction = stay_fraction;
metrics.stay = stay_fraction;
metrics.position_hold_stay_fraction_mean = avg_finite(stay_fraction);
metrics.survival = survival_accum;
metrics.survival_rate = avg(double(survival_accum));
metrics.done_rate = avg(double(done_accum));
metrics.position_final = position_norm;
metrics.position_final_mean = avg(position_norm);
metrics.position_max = position_max;
metrics.position_max_mean = avg(position_max);
metrics.max_position_error = max(position_max);
metrics.velocity_final = velocity_norm;
metrics.velocity_final_mean = avg(velocity_norm);
metrics.velocity_max = velocity_max;
metrics.velocity_max_mean = avg(velocity_max);
metrics.max_velocity = max(velocity_max);
metrics.omega_final = omega_norm;
metrics.omega_final_mean = avg(omega_norm);
metrics.omega_max = omega_max;
metrics.max_omega = max(omega_max);
metrics.omega_max_mean = avg(omega_max);
metrics.action_abs_mean = action_abs_sum / (horizon * 4);
metrics.motor_abs_mean = motor_abs_sum / (horizon * 4);
metrics.action_saturation_ratio = action_saturation_count / max(action_value_count, 1);
metrics.control_energy = control_energy;
metrics.control_energy_mean = avg(control_energy);
metrics.overshoot = overshoot;
metrics.overshoot_mean = avg(overshoot);
metrics.failure_by_config_bin = failure_by_mass_bin(dynamics, success);

long_summary = struct();
long_sample = struct();
for j = 1:horizon_count
    h = horizons(j);
    suffix = sprintf('H%d', h);
    horizon_snapshot = snapshot_success(:, j);
    horizon_steady = snapshot_steady(:, j);
    horizon_fraction = snapshot_window_fraction(:, j);
    horizon_settling_time_s = snapshot_settling_step(:, j) * params.dt;
    horizon_stay_fraction = snapshot_stay_fraction(:, j);
    horizon_survival = snapshot_survival(:, j);
    long_sample.(['position_hold_snapshot_' suffix]) = double(horizon_snapshot);
    long_sample.(['position_hold_steady_' suffix]) = double(horizon_steady);
    long_sample.(['final_window_success_fraction_' suffix]) = horizon_fraction;
    long_sample.(['position_hold_settling_time_s_' suffix]) = horizon_settling_time_s;
    long_sample.(['position_hold_stay_fraction_' suffix]) = horizon_stay_fraction;
    long_sample.(['survival_' suffix]) = double(horizon_survival);
    % Diagnostics-only channel counts expose the exact 95/100 semantics.
    % They do not enter success, policy, state transition, or reward paths.
    long_sample.(['position_pass_count_' suffix]) = snapshot_position_pass_count(:, j);
    long_sample.(['velocity_pass_count_' suffix]) = snapshot_velocity_pass_count(:, j);
    long_sample.(['omega_pass_count_' suffix]) = snapshot_omega_pass_count(:, j);
    long_sample.(['position_' suffix '_m']) = snapshot_position(:, j);
    long_sample.(['velocity_' suffix]) = snapshot_velocity(:, j);
    long_sample.(['omega_' suffix]) = snapshot_omega(:, j);
    long_summary.(['position_hold_snapshot_' suffix '_rate']) = avg(double(horizon_snapshot));
    long_summary.(['position_hold_steady_' suffix '_rate']) = avg(double(horizon_steady));
    long_summary.(['final_window_success_fraction_' suffix]) = avg_finite(horizon_fraction);
    long_summary.(['position_hold_settled_' suffix '_rate']) = ...
        avg(double(isfinite(horizon_settling_time_s)));
    long_summary.(['position_hold_settling_time_s_' suffix '_mean']) = ...
        avg_finite(horizon_settling_time_s);
    long_summary.(['position_hold_stay_fraction_' suffix '_mean']) = ...
        avg_finite(horizon_stay_fraction);
    long_summary.(['survival_rate_' suffix]) = avg(double(horizon_survival));
    long_summary.(['position_final_mean_' suffix]) = avg(snapshot_position(:, j));
    long_summary.(['velocity_final_mean_' suffix]) = avg(snapshot_velocity(:, j));
    long_summary.(['omega_final_mean_' suffix]) = avg(snapshot_omega(:, j));
    long_summary.(['invalid_fraction_' suffix]) = snapshot_invalid_fraction(j);
    current_axis_sample = snapshot_axis_sample{j};
    current_axis_names = fieldnames(current_axis_sample);
    for axis_field_index = 1:numel(current_axis_names)
        name = current_axis_names{axis_field_index};
        long_sample.([name '_' suffix]) = current_axis_sample.(name);
    end
    current_axis_summary = snapshot_axis_summary{j};
    current_axis_summary_names = fieldnames(current_axis_summary);
    for axis_field_index = 1:numel(current_axis_summary_names)
        name = current_axis_summary_names{axis_field_index};
        long_summary.([name '_' suffix]) = current_axis_summary.(name);
    end
end

position_rms = reshape(sqrt(post_position_sq_sum / post_count), [], 1);
velocity_rms = reshape(sqrt(post_velocity_sq_sum / post_count), [], 1);
omega_rms = reshape(sqrt(post_omega_sq_sum / post_count), [], 1);
position_max_after = post_position_max.';
velocity_max_after = post_velocity_max.';
omega_max_after = post_omega_max.';
long_sample.position_rms_after_H500 = position_rms;
long_sample.velocity_rms_after_H500 = velocity_rms;
long_sample.omega_rms_after_H500 = omega_rms;
long_sample.max_position_after_H500 = position_max_after;
long_sample.max_velocity_after_H500 = velocity_max_after;
long_sample.max_omega_after_H500 = omega_max_after;
long_summary.position_rms_after_H500_mean = avg_finite(position_rms);
long_summary.velocity_rms_after_H500_mean = avg_finite(velocity_rms);
long_summary.omega_rms_after_H500_mean = avg_finite(omega_rms);
long_summary.max_position_after_H500_mean = avg_finite(position_max_after);
long_summary.max_position_after_H500 = max_finite(position_max_after);
long_summary.max_velocity_after_H500_mean = avg_finite(velocity_max_after);
long_summary.max_velocity_after_H500 = max_finite(velocity_max_after);
long_summary.max_omega_after_H500_mean = avg_finite(omega_max_after);
long_summary.max_omega_after_H500 = max_finite(omega_max_after);

if horizon < window_steps
    diagnostic_order = 1:horizon;
else
    oldest_slot = mod(horizon, window_steps) + 1;
    diagnostic_order = [oldest_slot:window_steps, 1:(oldest_slot - 1)];
end
[axis_sample, axis_summary] = l2f_tail_axis_diagnostics( ...
    diagnostic_position_buffer(diagnostic_order, :), ...
    diagnostic_velocity_buffer(diagnostic_order, :), ...
    diagnostic_omega_buffer(diagnostic_order, :, :), ...
    diagnostic_action_buffer(diagnostic_order, :, :), params.dt);
axis_sample_names = fieldnames(axis_sample);
for i = 1:numel(axis_sample_names)
    name = axis_sample_names{i};
    long_sample.(name) = axis_sample.(name);
end
axis_summary_names = fieldnames(axis_summary);
for i = 1:numel(axis_summary_names)
    name = axis_summary_names{i};
    long_summary.(name) = axis_summary.(name);
end

if horizon >= 2
    action_delta_rms = reshape(sqrt(action_delta_sq_sum / (horizon - 1)), [], 1);
else
    action_delta_rms = zeros(batch_size, 1);
end
long_sample.action_delta_rms = action_delta_rms;
long_summary.action_delta_rms_mean = avg_finite(action_delta_rms);
long_summary.action_delta_rms_max = max_finite(action_delta_rms);

integral_world_norm_mean = reshape(integral_world_norm_sum / horizon, [], 1);
integral_body_mean = integral_body_sum / horizon;
integral_clamp_ratio = reshape(integral_clamp_count / horizon, [], 1);
integral_residual_action_rms = reshape(sqrt(integral_residual_sq_sum / horizon), [], 1);
damping_residual_action_rms = reshape(sqrt(damping_residual_sq_sum / horizon), [], 1);
steady_motor_bias = squeeze(mean(diagnostic_action_buffer(diagnostic_order, :, :), 1));
if batch_size == 1
    steady_motor_bias = reshape(steady_motor_bias, 1, 4);
end
long_sample.integral_world_norm_mean = integral_world_norm_mean;
long_sample.integral_body_x_mean = integral_body_mean(:, 1);
long_sample.integral_body_y_mean = integral_body_mean(:, 2);
long_sample.integral_body_z_mean = integral_body_mean(:, 3);
long_sample.integral_clamp_ratio = integral_clamp_ratio;
long_sample.integral_residual_action_rms = integral_residual_action_rms;
long_sample.damping_residual_action_rms = damping_residual_action_rms;
for motor_index = 1:4
    long_sample.(sprintf('steady_motor_bias_%d', motor_index - 1)) = ...
        steady_motor_bias(:, motor_index);
end
long_summary.integral_world_norm_mean = avg_finite(integral_world_norm_mean);
long_summary.integral_body_x_mean = avg_finite(integral_body_mean(:, 1));
long_summary.integral_body_y_mean = avg_finite(integral_body_mean(:, 2));
long_summary.integral_body_z_mean = avg_finite(integral_body_mean(:, 3));
long_summary.integral_clamp_ratio = avg_finite(integral_clamp_ratio);
long_summary.integral_residual_action_rms = avg_finite(integral_residual_action_rms);
long_summary.damping_residual_action_rms = avg_finite(damping_residual_action_rms);
for motor_index = 1:4
    long_summary.(sprintf('steady_motor_bias_%d_mean', motor_index - 1)) = ...
        avg_finite(steady_motor_bias(:, motor_index));
end

hidden_norm_max(~hidden_norm_seen) = nan;
hidden_abs_max(~hidden_abs_seen) = nan;
hidden_norm_max_col = hidden_norm_max.';
hidden_abs_max_col = hidden_abs_max.';
long_sample.hidden_state_norm_max = hidden_norm_max_col;
long_sample.hidden_state_abs_max = hidden_abs_max_col;
long_summary.hidden_state_norm_max_mean = avg_finite(hidden_norm_max_col);
long_summary.hidden_state_norm_max = max_finite(hidden_norm_max_col);
long_summary.hidden_state_abs_max_mean = avg_finite(hidden_abs_max_col);
long_summary.hidden_state_abs_max = max_finite(hidden_abs_max_col);

long_metrics = struct();
long_metrics.summary = long_summary;
if isempty(fieldnames(long_sample))
    long_metrics.sample = table();
else
    long_metrics.sample = struct2table(long_sample);
end
invalid_fraction = invalid_count / max(value_count, 1);
summary = l2f_eval_summary(metrics, long_metrics, invalid_fraction);

result = struct();
result.metrics = metrics;
result.long_metrics = long_metrics;
result.summary = summary;
result.dynamics = dynamics;
result.final_state = state;
result.invalid_fraction = invalid_fraction;
result.horizons = horizons;
result.mode = 'streaming';
end

function uav_cfg = resolve_uav_cfg(sim_cfg)
uav_cfg = [];
if isfield(sim_cfg, 'uav_cfg') && ~isempty(sim_cfg.uav_cfg)
    uav_cfg = sim_cfg.uav_cfg;
elseif isfield(sim_cfg, 'uav_cfg_name') && ~isempty(sim_cfg.uav_cfg_name)
    uav_cfg = l2f_uav_cfg_library(sim_cfg.uav_cfg_name);
end
end

function dynamics = pack_dynamics_batch(state)
batch_size = size(state.position, 1);
dyn0 = l2f_pack_dynamics(state, 1);
dynamics = zeros(batch_size, numel(dyn0));
for i = 1:batch_size
    dynamics(i, :) = l2f_pack_dynamics(state, i).';
end
end

function info = make_reference_info(world)
info = struct('is_static', true, 'position', [0, 0, 0], 'velocity', [0, 0, 0]);
if nargin < 1 || isempty(world) || ~isstruct(world) || ~isfield(world, 'reference')
    return;
end
ref_cfg = world.reference;
kind = lower(l2f_get_field_or(ref_cfg, 'type', 'hover'));
if any(strcmp(kind, {'hover', 'point'}))
    info.position = reshape(l2f_get_field_or(ref_cfg, 'position', [0, 0, 0]), 1, 3);
    info.velocity = reshape(l2f_get_field_or(ref_cfg, 'velocity', [0, 0, 0]), 1, 3);
else
    info.is_static = false;
end
end

function [position_norm, velocity_norm, omega_norm] = ...
        state_error_snapshot(state, world, t, reference_info)
batch_size = size(state.position, 1);
if reference_info.is_static
    position_error = state.position - reference_info.position;
    velocity_error = state.velocity - reference_info.velocity;
    omega_error = state.omega;
else
    ref = l2f_reference(world, t, batch_size);
    position_error = state.position - ref.position;
    velocity_error = state.velocity - ref.velocity;
    omega_error = state.omega - ref.omega;
end
position_norm = sqrt(sum(position_error .* position_error, 2)).';
velocity_norm = sqrt(sum(velocity_error .* velocity_error, 2)).';
omega_norm = sqrt(sum(omega_error .* omega_error, 2)).';
end

function success = within_success(position_norm, velocity_norm, omega_norm, reward_cfg)
success = position_norm < reward_cfg.success_position_m ...
    & velocity_norm < reward_cfg.success_velocity ...
    & omega_norm < reward_cfg.success_omega;
end

function reward = reward_step(state, action, previous_action, cfg, world, t, reference_info)
batch_size = size(state.position, 1);
if reference_info.is_static
    position_error = state.position - reference_info.position;
    velocity_error = state.velocity - reference_info.velocity;
    omega_error = state.omega;
else
    ref = l2f_reference(world, t, batch_size);
    position_error = state.position - ref.position;
    velocity_error = state.velocity - ref.velocity;
    omega_error = state.omega - ref.omega;
end
position_cost = cfg.w_position * sum((position_error / cfg.p_scale) .* ...
    (position_error / cfg.p_scale), 2).';
velocity_cost = cfg.w_velocity * sum((velocity_error / cfg.v_scale) .* ...
    (velocity_error / cfg.v_scale), 2).';
omega_cost = cfg.w_omega * sum((omega_error / cfg.omega_scale) .* ...
    (omega_error / cfg.omega_scale), 2).';
action_cost = cfg.w_action * (sum(action .* action, 2).' / size(action, 2));
delta = action - previous_action;
smooth_cost = cfg.w_smooth * (sum(delta .* delta, 2).' / size(delta, 2));
reward = -(position_cost + velocity_cost + omega_cost + action_cost + smooth_cost);
end

function done = termination_step(state, ~, sim_cfg, world, t, reference_info)
batch_size = size(state.position, 1);
done = false(1, batch_size);
[position_norm, velocity_norm, omega_norm] = ...
    state_error_snapshot(state, world, t, reference_info);
if sim_cfg.terminate_on_bounds
    done = done | position_norm > sim_cfg.max_position_norm;
    done = done | velocity_norm > sim_cfg.max_velocity_norm;
    done = done | omega_norm > sim_cfg.max_omega_norm;
end
if sim_cfg.terminate_on_nonfinite
    done = done | ~all(isfinite(state.position), 2).';
    done = done | ~all(isfinite(state.velocity), 2).';
    done = done | ~all(isfinite(state.omega), 2).';
    done = done | ~reshape(all(all(isfinite(state.rotation), 1), 2), 1, batch_size);
end
end

function survived = state_survived(state, command, motor, done_step, sim_cfg, world, ...
        position_norm, velocity_norm, omega_norm)
batch_size = size(state.position, 1);
state_finite = all(isfinite(state.position), 2).' ...
    & all(isfinite(state.velocity), 2).' ...
    & all(isfinite(state.omega), 2).' ...
    & reshape(all(all(isfinite(state.rotation), 1), 2), 1, batch_size);
command_finite = all(isfinite(command), 2).' & all(isfinite(motor), 2).';
max_position = l2f_get_field_or(sim_cfg, 'max_position_norm', inf);
if isstruct(world) && isfield(world, 'bounds')
    max_position = min(max_position, ...
        l2f_get_field_or(world.bounds, 'position_norm', inf));
end
within_bounds = position_norm <= max_position ...
    & velocity_norm <= l2f_get_field_or(sim_cfg, 'max_velocity_norm', inf) ...
    & omega_norm <= l2f_get_field_or(sim_cfg, 'max_omega_norm', inf);
survived = state_finite & command_finite & within_bounds & ~done_step;
end

function count = state_invalid_count(state)
count = nnz(~isfinite(state.position)) ...
    + nnz(~isfinite(state.velocity)) ...
    + nnz(~isfinite(state.omega)) ...
    + nnz(~isfinite(state.rotation));
end

function action = ensure_action_shape(action, batch_size)
if isvector(action) && numel(action) == 4
    action = reshape(action, 1, 4);
end
if size(action, 1) == 1 && batch_size > 1
    action = repmat(action, batch_size, 1);
end
if size(action, 1) ~= batch_size || size(action, 2) ~= 4
    error('Controller action must be N x 4.');
end
action = min(max(action, -1.0), 1.0);
end

function out = select_state(old_state, new_state, use_new)
out = new_state;
mask = ~use_new(:);
if ~any(mask)
    return;
end
out.position(mask, :) = old_state.position(mask, :);
out.velocity(mask, :) = old_state.velocity(mask, :);
out.omega(mask, :) = old_state.omega(mask, :);
out.motor(mask, :) = old_state.motor(mask, :);
out.previous_action(mask, :) = old_state.previous_action(mask, :);
for i = find(mask).'
    out.rotation(:, :, i) = old_state.rotation(:, :, i);
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
