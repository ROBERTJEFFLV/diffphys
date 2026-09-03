
function value = l2f_invalid_fraction_logs(logs, horizon)
%L2F_INVALID_FRACTION_LOGS Count non-finite values without concatenating logs.

full_horizon = size(logs.action, 1);
if nargin < 2 || isempty(horizon)
    horizon = full_horizon;
end
horizon = min(max(round(double(horizon)), 0), full_horizon);
state_end = horizon + 1;

invalid_count = 0;
value_count = 0;
[invalid_count, value_count] = add_count(invalid_count, value_count, logs.position(1:state_end, :, :));
[invalid_count, value_count] = add_count(invalid_count, value_count, logs.velocity(1:state_end, :, :));
[invalid_count, value_count] = add_count(invalid_count, value_count, logs.omega(1:state_end, :, :));
[invalid_count, value_count] = add_count(invalid_count, value_count, logs.rotation(1:state_end, :, :, :));
if horizon > 0
    [invalid_count, value_count] = add_count(invalid_count, value_count, logs.action(1:horizon, :, :));
    [invalid_count, value_count] = add_count(invalid_count, value_count, logs.motor(1:horizon, :, :));
end
value = invalid_count / max(value_count, 1);
end

function [invalid_count, value_count] = add_count(invalid_count, value_count, values)
invalid_count = invalid_count + nnz(~isfinite(values));
value_count = value_count + numel(values);
end
