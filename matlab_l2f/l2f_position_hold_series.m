function hold = l2f_position_hold_series(position_norm, velocity_norm, omega_norm, reward_cfg)
%L2F_POSITION_HOLD_SERIES Position-hold success over executed-step states.
%
% Row 1 of each norm series is the reset state. Rolling windows contain only
% states after executed physical steps (rows 2:end). A horizon shorter than W
% has no complete steady window and therefore cannot be steady-successful.

if ~isequal(size(position_norm), size(velocity_norm), size(omega_norm))
    error('Position, velocity, and omega error series must have equal size.');
end
[window_steps, required_fraction] = l2f_position_hold_settings(reward_cfg);

state_success = position_norm < reward_cfg.success_position_m ...
    & velocity_norm < reward_cfg.success_velocity ...
    & omega_norm < reward_cfg.success_omega;
executed_success = state_success(2:end, :);
horizon = size(executed_success, 1);
count = size(executed_success, 2);

window_fraction = nan(horizon, count);
steady_success = false(horizon, count);
if horizon >= window_steps
    cumulative = [zeros(1, count); cumsum(double(executed_success), 1)];
    window_sum = cumulative((window_steps + 1):end, :) ...
        - cumulative(1:(end - window_steps), :);
    window_fraction(window_steps:end, :) = window_sum / window_steps;
    steady_success(window_steps:end, :) = ...
        window_fraction(window_steps:end, :) >= required_fraction;
end

settling_step = nan(1, count);
stay_fraction = nan(1, count);
for i = 1:count
    first_steady = find(steady_success(:, i), 1, 'first');
    if isempty(first_steady)
        continue;
    end
    settling_step(i) = first_steady;
    after = (first_steady + 1):horizon;
    if ~isempty(after)
        stay_fraction(i) = mean(double(executed_success(after, i)));
    end
end

hold = struct();
hold.state_success = state_success;
hold.executed_success = executed_success;
hold.window_fraction = window_fraction;
hold.steady_success_by_step = steady_success;
hold.snapshot_success = state_success(end, :);
if horizon >= 1
    hold.steady_success = steady_success(end, :);
    hold.final_window_success_fraction = window_fraction(end, :);
else
    hold.steady_success = false(1, count);
    hold.final_window_success_fraction = nan(1, count);
end
hold.settling_step = settling_step;
hold.stay_fraction = stay_fraction;
hold.window_steps = window_steps;
hold.required_fraction = required_fraction;
end
