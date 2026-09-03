function [force, torque] = l2f_aero_wrench(state, world, t)
%L2F_AERO_WRENCH Aerodynamic drag from relative wind and body-rate damping.

if nargin < 3
    t = 0;
end
batch = size(state.position, 1);
force = zeros(batch, 3);
torque = zeros(batch, 3);

wind = l2f_wind_field(state.position, world, t);
v_air = state.velocity - wind;

rho = 1.225;
if isstruct(world) && isfield(world, 'aero')
    rho = l2f_get_field_or(world.aero, 'rho', rho);
end
drag_linear = get_state_row(state, 'drag_linear', batch, 3, [0 0 0]);
drag_quadratic = get_state_row(state, 'drag_quadratic', batch, 3, [0.01 0.01 0.02]);

for i = 1:batch
    speed = sqrt(sum(v_air(i, :) .* v_air(i, :)));
    force(i, :) = -drag_linear(i, :) .* v_air(i, :) ...
        - 0.5 * rho * drag_quadratic(i, :) .* speed .* v_air(i, :);
end

angular_linear = get_state_row(state, 'angular_drag_linear', batch, 3, [0 0 0]);
angular_quadratic = get_state_row(state, 'angular_drag_quadratic', batch, 3, [0 0 0]);
torque = -angular_linear .* state.omega - angular_quadratic .* abs(state.omega) .* state.omega;
end

function row = get_state_row(state, name, batch, width, default_value)
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
