function cfg_obs = l2f_config_observation(state)
%L2F_CONFIG_OBSERVATION Flatten per-vehicle physical config features.

batch = size(state.position, 1);
rotor_pos = flatten_geom(l2f_get_state_geom(state, 'rotor_pos_body', batch, zeros(4, 3)));
rotor_axis = flatten_geom(l2f_get_state_geom(state, 'rotor_axis_body', batch, repmat([0 0 1], 4, 1)));
cfg_obs = [
    state.mass, ...
    state.inertia_x, state.inertia_y, state.inertia_z, ...
    rotor_pos, rotor_axis, ...
    l2f_get_state_row(state, 'spin_dir', batch, 4, [1 -1 1 -1]), ...
    state.thrust_coeff_c0, state.thrust_coeff_c1, state.thrust_coeff_c2, ...
    l2f_get_state_row(state, 'motor_time_rising', batch, 4, 0.06), ...
    l2f_get_state_row(state, 'motor_time_falling', batch, 4, 0.06), ...
    l2f_get_state_row(state, 'rotor_torque_constant', batch, 4, 0.02), ...
    l2f_get_state_row(state, 'drag_linear', batch, 3, [0 0 0]), ...
    l2f_get_state_row(state, 'drag_quadratic', batch, 3, [0 0 0]), ...
    l2f_get_state_row(state, 'motor_health', batch, 4, 1), ...
    l2f_get_state_row(state, 'thrust_scale', batch, 4, 1)
];
end

function flat = flatten_geom(geom)
batch = size(geom, 1);
flat = zeros(batch, size(geom, 2) * size(geom, 3));
for i = 1:batch
    k = 1;
    for m = 1:size(geom, 2)
        for j = 1:size(geom, 3)
            flat(i, k) = geom(i, m, j);
            k = k + 1;
        end
    end
end
end

function geom = l2f_get_state_geom(state, name, batch, default_value)
if isfield(state, name)
    geom = state.(name);
else
    geom = zeros(batch, size(default_value, 1), size(default_value, 2));
    for i = 1:batch
        geom(i, :, :) = reshape(default_value, 1, size(default_value, 1), size(default_value, 2));
    end
end
end

function row = l2f_get_state_row(state, name, batch, width, default_value)
if isfield(state, name)
    value = state.(name);
else
    value = default_value;
end
if isscalar(value)
    row = repmat(value, batch, width);
elseif size(value, 1) == batch && size(value, 2) == width
    row = value;
elseif size(value, 1) == batch && size(value, 2) == 1
    row = repmat(value, 1, width);
else
    row = repmat(reshape(value, 1, width), batch, 1);
end
end
