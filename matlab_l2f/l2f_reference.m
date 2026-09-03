function ref = l2f_reference(world, t, batch_size)
%L2F_REFERENCE Reference trajectory evaluated at time t.

if nargin < 2
    t = 0;
end
if nargin < 3
    batch_size = 1;
end

ref_cfg = struct('type', 'hover', 'position', [0 0 0], 'velocity', [0 0 0]);
if nargin >= 1 && isstruct(world) && isfield(world, 'reference')
    ref_cfg = world.reference;
end

ref = struct();
ref.position = zeros(batch_size, 3);
ref.velocity = zeros(batch_size, 3);
ref.rotation = zeros(3, 3, batch_size);
ref.omega = zeros(batch_size, 3);
for i = 1:batch_size
    ref.rotation(:, :, i) = eye(3);
end

switch lower(l2f_get_field_or(ref_cfg, 'type', 'hover'))
    case {'hover', 'point'}
        ref.position = repmat(reshape(l2f_get_field_or(ref_cfg, 'position', [0 0 0]), 1, 3), batch_size, 1);
        ref.velocity = repmat(reshape(l2f_get_field_or(ref_cfg, 'velocity', [0 0 0]), 1, 3), batch_size, 1);
    case 'step'
        start_time = l2f_get_field_or(ref_cfg, 'start_time', 1.0);
        before = reshape(l2f_get_field_or(ref_cfg, 'before', [0 0 0]), 1, 3);
        after = reshape(l2f_get_field_or(ref_cfg, 'after', [1 0 0]), 1, 3);
        if t >= start_time
            ref.position = repmat(after, batch_size, 1);
        else
            ref.position = repmat(before, batch_size, 1);
        end
    case 'circle'
        radius = l2f_get_field_or(ref_cfg, 'radius', 1.0);
        omega = l2f_get_field_or(ref_cfg, 'omega', 0.5);
        z = l2f_get_field_or(ref_cfg, 'z', 0.0);
        center = reshape(l2f_get_field_or(ref_cfg, 'center', [0 0 0]), 1, 3);
        p = center + [radius * cos(omega * t), radius * sin(omega * t), z];
        v = [-radius * omega * sin(omega * t), radius * omega * cos(omega * t), 0];
        ref.position = repmat(p, batch_size, 1);
        ref.velocity = repmat(v, batch_size, 1);
    case {'figure_eight', 'figure-eight'}
        radius = l2f_get_field_or(ref_cfg, 'radius', 1.0);
        omega = l2f_get_field_or(ref_cfg, 'omega', 0.5);
        z = l2f_get_field_or(ref_cfg, 'z', 0.0);
        p = [radius * sin(omega * t), radius * sin(omega * t) * cos(omega * t), z];
        v = [radius * omega * cos(omega * t), radius * omega * cos(2 * omega * t), 0];
        ref.position = repmat(p, batch_size, 1);
        ref.velocity = repmat(v, batch_size, 1);
    otherwise
        error('Unknown reference type: %s', ref_cfg.type);
end
end
