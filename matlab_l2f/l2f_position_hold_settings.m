function [window_steps, required_fraction] = l2f_position_hold_settings(reward_cfg)
%L2F_POSITION_HOLD_SETTINGS Validate the shared steady-success definition.

window_steps = round(double(l2f_get_field_or( ...
    reward_cfg, 'steady_window_steps', 100)));
required_fraction = double(l2f_get_field_or( ...
    reward_cfg, 'steady_required_fraction', 0.95));
if ~isscalar(window_steps) || ~isfinite(window_steps) || window_steps < 1
    error('steady_window_steps must be a positive integer.');
end
if ~isscalar(required_fraction) || ~isfinite(required_fraction) ...
        || required_fraction <= 0.0 || required_fraction > 1.0
    error('steady_required_fraction must be in (0, 1].');
end
end
