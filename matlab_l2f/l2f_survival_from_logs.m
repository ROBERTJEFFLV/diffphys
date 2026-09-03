function survival = l2f_survival_from_logs(logs, position_norm, velocity_norm, omega_norm)
%L2F_SURVIVAL_FROM_LOGS Cumulative rollout integrity, independent of hold success.

horizon = size(logs.action, 1);
count = size(logs.action, 2);
if horizon < 1
    survival = false(0, count);
    return;
end

position = logs.position(2:(horizon + 1), :, :);
velocity = logs.velocity(2:(horizon + 1), :, :);
omega = logs.omega(2:(horizon + 1), :, :);
rotation = logs.rotation(2:(horizon + 1), :, :, :);
state_finite = reshape(all(isfinite(position), 3), horizon, count) ...
    & reshape(all(isfinite(velocity), 3), horizon, count) ...
    & reshape(all(isfinite(omega), 3), horizon, count) ...
    & reshape(all(all(isfinite(rotation), 4), 3), horizon, count);
command_finite = reshape(all(isfinite(logs.action), 3), horizon, count) ...
    & reshape(all(isfinite(logs.motor), 3), horizon, count);

sim_cfg = l2f_get_field_or(logs, 'sim_cfg', struct());
max_position = double(l2f_get_field_or(sim_cfg, 'max_position_norm', inf));
max_velocity = double(l2f_get_field_or(sim_cfg, 'max_velocity_norm', inf));
max_omega = double(l2f_get_field_or(sim_cfg, 'max_omega_norm', inf));
if isfield(logs, 'world') && isstruct(logs.world) && isfield(logs.world, 'bounds')
    max_position = min(max_position, double(l2f_get_field_or( ...
        logs.world.bounds, 'position_norm', inf)));
end
within_bounds = position_norm(2:end, :) <= max_position ...
    & velocity_norm(2:end, :) <= max_velocity ...
    & omega_norm(2:end, :) <= max_omega;

done = false(horizon, count);
if isfield(logs, 'done') && ~isempty(logs.done)
    done = logical(logs.done(1:horizon, :));
end
crash = false(horizon, count);
if isfield(logs, 'crash') && ~isempty(logs.crash)
    crash = logical(logs.crash(1:horizon, :));
elseif isfield(logs, 'collision') && ~isempty(logs.collision)
    crash = logical(logs.collision(1:horizon, :));
end

step_survived = state_finite & command_finite & within_bounds & ~done & ~crash;
survival = cumprod(double(step_survived), 1) > 0;
end
