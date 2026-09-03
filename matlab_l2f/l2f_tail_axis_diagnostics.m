function [sample, summary] = l2f_tail_axis_diagnostics(position_norm, velocity_norm, omega, action, dt)
%L2F_TAIL_AXIS_DIAGNOSTICS Per-axis diagnostics over one chronological tail.

if ndims(omega) ~= 3 || size(omega, 3) ~= 3
    error('omega must be time x batch x 3.');
end
if ndims(action) ~= 3 || size(action, 3) ~= 4
    error('action must be time x batch x 4.');
end
if size(position_norm, 1) ~= size(omega, 1) ...
        || size(velocity_norm, 1) ~= size(omega, 1) ...
        || size(action, 1) ~= size(omega, 1)
    error('tail diagnostics inputs must share the time dimension.');
end
if size(position_norm, 2) ~= size(omega, 2) ...
        || size(velocity_norm, 2) ~= size(omega, 2) ...
        || size(action, 2) ~= size(omega, 2)
    error('tail diagnostics inputs must share the batch dimension.');
end

position_rms = reshape(sqrt(mean(position_norm .* position_norm, 1)), [], 1);
velocity_rms = reshape(sqrt(mean(velocity_norm .* velocity_norm, 1)), [], 1);
omega_norm_sq = sum(omega .* omega, 3);
omega_rms = reshape(sqrt(mean(omega_norm_sq, 1)), [], 1);
omega_rms_axis = reshape(sqrt(mean(omega .* omega, 1)), size(omega, 2), 3);
omega_max_axis = reshape(max(abs(omega), [], 1), size(omega, 2), 3);
omega_peak_hz_axis = dominant_frequency(omega, dt);
action_rms_axis = reshape(sqrt(mean(action .* action, 1)), size(action, 2), 4);
if size(action, 1) >= 2
    delta = diff(action, 1, 1);
    action_delta_rms_axis = reshape( ...
        sqrt(mean(delta .* delta, 1)), size(action, 2), 4);
else
    action_delta_rms_axis = zeros(size(action_rms_axis), 'like', action_rms_axis);
end

strict = position_rms < 0.05 & velocity_rms < 0.10 & omega_rms >= 0.20;
loose = position_rms < 0.10 & velocity_rms < 0.20 & omega_rms >= 0.20;
sample = struct();
sample.position_tail_rms = position_rms;
sample.velocity_tail_rms = velocity_rms;
sample.omega_tail_rms = omega_rms;
sample.strict_bounded_angular_motion = double(strict);
sample.loose_bounded_angular_motion = double(loose);
axes = {'x', 'y', 'z'};
for axis_index = 1:3
    axis_name = axes{axis_index};
    sample.(['omega_' axis_name '_tail_rms']) = omega_rms_axis(:, axis_index);
    sample.(['omega_' axis_name '_tail_max']) = omega_max_axis(:, axis_index);
    sample.(['omega_' axis_name '_tail_spectral_peak_hz']) = ...
        omega_peak_hz_axis(:, axis_index);
end
for motor_index = 1:4
    suffix = sprintf('%d', motor_index - 1);
    sample.(['action_' suffix '_tail_rms']) = action_rms_axis(:, motor_index);
    sample.(['action_' suffix '_delta_tail_rms']) = ...
        action_delta_rms_axis(:, motor_index);
end

summary = struct();
summary.strict_bounded_angular_motion_rate = mean(double(strict));
summary.loose_bounded_angular_motion_rate = mean(double(loose));
for axis_index = 1:3
    axis_name = axes{axis_index};
    summary.(['omega_' axis_name '_tail_rms_mean']) = mean(omega_rms_axis(:, axis_index));
    summary.(['omega_' axis_name '_tail_max_mean']) = mean(omega_max_axis(:, axis_index));
    summary.(['omega_' axis_name '_tail_spectral_peak_hz_mean']) = ...
        mean(omega_peak_hz_axis(:, axis_index));
end
for motor_index = 1:4
    suffix = sprintf('%d', motor_index - 1);
    summary.(['action_' suffix '_tail_rms_mean']) = mean(action_rms_axis(:, motor_index));
    summary.(['action_' suffix '_delta_tail_rms_mean']) = ...
        mean(action_delta_rms_axis(:, motor_index));
end
end

function peak_hz = dominant_frequency(values, dt)
time_count = size(values, 1);
batch_size = size(values, 2);
axis_count = size(values, 3);
peak_hz = zeros(batch_size, axis_count, 'like', values);
if time_count < 2
    return;
end
centered = values - mean(values, 1);
power = abs(fft(centered, [], 1)).^2;
positive_count = floor(time_count / 2) + 1;
power = power(1:positive_count, :, :);
power(1, :, :) = 0;
for batch_index = 1:batch_size
    for axis_index = 1:axis_count
        spectrum = power(:, batch_index, axis_index);
        [energy, index] = max(spectrum);
        if energy > eps(class(values))
            peak_hz(batch_index, axis_index) = (index - 1) / (time_count * dt);
        end
    end
end
end
