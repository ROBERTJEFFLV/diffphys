
function summary = l2f_eval_summary(metrics, long_metrics, invalid_fraction)
%L2F_EVAL_SUMMARY Build the one-row evaluation summary.

summary = struct();
summary.count = metrics.count;
summary.horizon = metrics.horizon;
summary.steady_window_steps = metrics.steady_window_steps;
summary.steady_required_fraction = metrics.steady_required_fraction;
summary.success_rate = metrics.success_rate;
summary.position_hold_snapshot_rate = metrics.position_hold_snapshot_rate;
summary.position_hold_steady_rate = metrics.position_hold_steady_rate;
summary.final_window_success_fraction_mean = ...
    metrics.final_window_success_fraction_mean;
summary.position_hold_settled_rate = metrics.position_hold_settled_rate;
summary.position_hold_settling_time_s_mean = finite_mean( ...
    metrics.position_hold_settling_time_s);
summary.settling_time = summary.position_hold_settling_time_s_mean;
summary.position_hold_stay_fraction_mean = ...
    metrics.position_hold_stay_fraction_mean;
summary.stay = summary.position_hold_stay_fraction_mean;
summary.survival_rate = metrics.survival_rate;
summary.survival = metrics.survival_rate;
summary.done_rate = metrics.done_rate;
summary.invalid_fraction = invalid_fraction;
summary.return_mean = metrics.return_mean;
summary.position_final_mean = metrics.position_final_mean;
summary.position_max_mean = metrics.position_max_mean;
summary.max_position_error = metrics.max_position_error;
summary.velocity_final_mean = metrics.velocity_final_mean;
summary.velocity_max_mean = metrics.velocity_max_mean;
summary.max_velocity = metrics.max_velocity;
summary.omega_final_mean = metrics.omega_final_mean;
summary.omega_max_mean = metrics.omega_max_mean;
summary.max_omega = metrics.max_omega;
summary.action_saturation_ratio = metrics.action_saturation_ratio;
summary.control_energy_mean = metrics.control_energy_mean;
summary.overshoot_mean = metrics.overshoot_mean;
if nargin >= 2 && isstruct(long_metrics) && isfield(long_metrics, 'summary')
    names = fieldnames(long_metrics.summary);
    for i = 1:numel(names)
        summary.(names{i}) = long_metrics.summary.(names{i});
    end
end

function value = finite_mean(values)
values = values(isfinite(values));
if isempty(values)
    value = nan;
else
    value = sum(values(:)) / numel(values);
end
end
end
