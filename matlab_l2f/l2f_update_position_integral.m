function integral_position = l2f_update_position_integral(integral_position, observed_position, dt, integral_limit, integral_leak)
%L2F_UPDATE_POSITION_INTEGRAL Leaky, clamped deployable position integral.

if nargin < 5 || isempty(integral_leak)
    integral_leak = 0.0;
end
retention = max(0.0, 1.0 - double(integral_leak) * double(dt));
integral_position = retention .* integral_position + double(dt) .* observed_position;
integral_position = min(max(integral_position, -double(integral_limit)), double(integral_limit));
end
