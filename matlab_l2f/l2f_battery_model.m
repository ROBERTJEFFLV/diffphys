function [thrust, battery] = l2f_battery_model(thrust, state, dt)
%L2F_BATTERY_MODEL Simple voltage sag and depletion model.

batch = size(thrust, 1);
battery = struct();
battery.voltage = get_col(state, 'battery_voltage', batch, 1.0);
battery.nominal_voltage = get_col(state, 'battery_nominal_voltage', batch, 1.0);
battery.enabled = get_col(state, 'battery_enabled', batch, 0.0);

if ~any(battery.enabled > 0.5)
    return;
end

current_gain = get_col(state, 'battery_current_gain', batch, 0.0);
capacity_gain = get_col(state, 'battery_capacity_gain', batch, 0.0);
internal_resistance = get_col(state, 'battery_internal_resistance', batch, 0.0);
min_scale = get_col(state, 'battery_min_scale', batch, 0.6);

positive_thrust = max(thrust, 0.0);
current = current_gain .* sum(positive_thrust, 2);
loaded_voltage = battery.voltage - internal_resistance .* current;
battery.voltage = battery.voltage - capacity_gain .* current * dt;
voltage_ratio = max(loaded_voltage ./ max(battery.nominal_voltage, 1.0e-12), min_scale);
scale = voltage_ratio .* voltage_ratio;
for i = 1:batch
    if battery.enabled(i) > 0.5
        thrust(i, :) = thrust(i, :) * scale(i);
    end
end
end

function value = get_col(state, name, batch, default_value)
if isfield(state, name)
    data = state.(name);
    if isscalar(data)
        value = repmat(data, batch, 1);
    else
        value = data(:, 1);
    end
else
    value = repmat(default_value, batch, 1);
end
end
