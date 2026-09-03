function thrust = l2f_ground_effect(thrust, state, world)
%L2F_GROUND_EFFECT Apply near-ground thrust amplification.

if nargin < 3 || isempty(world)
    return;
end
ground = struct('enabled', false);
if isstruct(world) && isfield(world, 'ground_effect')
    ground = world.ground_effect;
elseif isstruct(world) && isfield(world, 'environment') && isfield(world.environment, 'ground_effect')
    ground = world.environment.ground_effect;
end
if ~l2f_get_field_or(ground, 'enabled', false)
    return;
end

batch = size(state.position, 1);
k_ge = l2f_get_field_or(ground, 'gain', 0.15);
gain_max = l2f_get_field_or(ground, 'gain_max', 1.30);
ground_z = l2f_get_field_or(ground, 'ground_z', 0.0);
z_min = l2f_get_field_or(ground, 'z_min', 0.05);

for i = 1:batch
    rotor_radius = get_scalar(state, 'rotor_radius', i, 0.02);
    z = max(state.position(i, 3) - ground_z, z_min);
    ratio = z / max(rotor_radius, 1.0e-6);
    gain = 1.0 + k_ge / max(ratio * ratio, z_min);
    gain = min(gain, gain_max);
    thrust(i, :) = thrust(i, :) * gain;
end
end

function value = get_scalar(state, name, i, default_value)
if isfield(state, name)
    data = state.(name);
    if isscalar(data)
        value = data;
    else
        value = data(i, 1);
    end
else
    value = default_value;
end
end
