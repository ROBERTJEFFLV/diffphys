
function [position_norm, velocity_norm, omega_norm] = l2f_error_series(logs, world)
%L2F_ERROR_SERIES Per-timestep norm errors against the reference.
%
% The common static-hover case is vectorized over time and vehicles.

if nargin < 2 || isempty(world)
    if isfield(logs, 'world')
        world = logs.world;
    else
        world = l2f_make_world('empty');
    end
end

horizon_plus_one = size(logs.position, 1);
batch_size = size(logs.position, 2);
[is_static, target_position, target_velocity] = static_hover_reference(world);
if is_static
    position_error = logs.position - reshape(target_position, 1, 1, 3);
    velocity_error = logs.velocity - reshape(target_velocity, 1, 1, 3);
    omega_error = logs.omega;
    position_norm = sqrt(sum(position_error .* position_error, 3));
    velocity_norm = sqrt(sum(velocity_error .* velocity_error, 3));
    omega_norm = sqrt(sum(omega_error .* omega_error, 3));
else
    position_norm = zeros(horizon_plus_one, batch_size);
    velocity_norm = zeros(horizon_plus_one, batch_size);
    omega_norm = zeros(horizon_plus_one, batch_size);
    for t = 1:horizon_plus_one
        ref = l2f_reference(world, logs.time(t), batch_size);
        p = reshape(logs.position(t, :, :), batch_size, 3) - ref.position;
        v = reshape(logs.velocity(t, :, :), batch_size, 3) - ref.velocity;
        w = reshape(logs.omega(t, :, :), batch_size, 3) - ref.omega;
        position_norm(t, :) = sqrt(sum(p .* p, 2)).';
        velocity_norm(t, :) = sqrt(sum(v .* v, 2)).';
        omega_norm(t, :) = sqrt(sum(w .* w, 2)).';
    end
end

end

function [is_static, position, velocity] = static_hover_reference(world)
is_static = true;
position = [0, 0, 0];
velocity = [0, 0, 0];
if nargin < 1 || isempty(world) || ~isstruct(world) || ~isfield(world, 'reference')
    return;
end
ref_cfg = world.reference;
kind = lower(l2f_get_field_or(ref_cfg, 'type', 'hover'));
if ~any(strcmp(kind, {'hover', 'point'}))
    is_static = false;
    return;
end
position = reshape(l2f_get_field_or(ref_cfg, 'position', [0, 0, 0]), 1, 3);
velocity = reshape(l2f_get_field_or(ref_cfg, 'velocity', [0, 0, 0]), 1, 3);
end
