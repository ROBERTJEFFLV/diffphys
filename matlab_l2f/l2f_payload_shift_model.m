function state = l2f_payload_shift_model(state, world, t)
%L2F_PAYLOAD_SHIFT_MODEL Apply scheduled mass/inertia changes.

if nargin < 3
    t = 0;
end
payload = struct('enabled', false);
if isstruct(world) && isfield(world, 'payload_shift')
    payload = world.payload_shift;
elseif isstruct(world) && isfield(world, 'environment') && isfield(world.environment, 'payload_shift')
    payload = world.environment.payload_shift;
end
if ~l2f_get_field_or(payload, 'enabled', false)
    return;
end
start_t = l2f_get_field_or(payload, 'start_time', 2.0);
if t < start_t
    return;
end
if isfield(state, 'payload_shift_applied') && all(state.payload_shift_applied > 0.5)
    return;
end

batch = size(state.position, 1);
mass_delta = l2f_get_field_or(payload, 'mass_delta', 0.0);
mass_scale = l2f_get_field_or(payload, 'mass_scale', 1.0);
inertia_scale = reshape(l2f_get_field_or(payload, 'inertia_scale', [1 1 1]), 1, 3);

if ~isfield(state, 'payload_shift_applied')
    state.payload_shift_applied = zeros(batch, 1);
end
for i = 1:batch
    if state.payload_shift_applied(i) <= 0.5
        state.mass(i) = state.mass(i) * mass_scale + mass_delta;
        state.inertia_x(i) = state.inertia_x(i) * inertia_scale(1);
        state.inertia_y(i) = state.inertia_y(i) * inertia_scale(2);
        state.inertia_z(i) = state.inertia_z(i) * inertia_scale(3);
        state.payload_shift_applied(i) = 1.0;
    end
end
end
