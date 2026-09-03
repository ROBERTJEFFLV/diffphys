function l2f_replay_3d(logs, world, varargin)
%L2F_REPLAY_3D Replay one vehicle trajectory with a simple arm-frame body.

if nargin < 2 || isempty(world)
    world = l2f_make_world('empty');
end

opt = struct();
opt.vehicle_id = 1;
opt.stride = 2;
opt.pause_s = 0.01;
opt.trail = true;
opt.axis_margin = 0.5;
opt = parse_options(opt, varargin{:});

vehicle_id = opt.vehicle_id;
positions = reshape(logs.position(:, vehicle_id, :), size(logs.position, 1), 3);

fig = figure('Name', 'L2F 3D Replay', 'Color', 'w');
ax = axes(fig);
hold(ax, 'on');
grid(ax, 'on');
xlabel(ax, 'x (m)');
ylabel(ax, 'y (m)');
zlabel(ax, 'z (m)');
title(ax, ['world: ' world.type]);
view(ax, 3);

mins = min(positions, [], 1) - opt.axis_margin;
maxs = max(positions, [], 1) + opt.axis_margin;
axis(ax, [mins(1) maxs(1) mins(2) maxs(2) mins(3) maxs(3)]);

if opt.trail
    trail_handle = plot3(ax, NaN, NaN, NaN, 'b-', 'LineWidth', 1.2);
end
arm_x = plot3(ax, NaN, NaN, NaN, 'r-', 'LineWidth', 2.0);
arm_y = plot3(ax, NaN, NaN, NaN, 'k-', 'LineWidth', 2.0);
body_handle = plot3(ax, NaN, NaN, NaN, 'ko', 'MarkerFaceColor', 'k');

arm_length = estimate_arm_length(logs, vehicle_id);
for step = 1:opt.stride:size(logs.position, 1)
    p = reshape(logs.position(step, vehicle_id, :), 1, 3);
    r = reshape(logs.rotation(step, vehicle_id, :, :), 3, 3);
    x_arm = r(:, 1).' * arm_length;
    y_arm = r(:, 2).' * arm_length;
    set(arm_x, 'XData', [p(1) - x_arm(1), p(1) + x_arm(1)], ...
        'YData', [p(2) - x_arm(2), p(2) + x_arm(2)], ...
        'ZData', [p(3) - x_arm(3), p(3) + x_arm(3)]);
    set(arm_y, 'XData', [p(1) - y_arm(1), p(1) + y_arm(1)], ...
        'YData', [p(2) - y_arm(2), p(2) + y_arm(2)], ...
        'ZData', [p(3) - y_arm(3), p(3) + y_arm(3)]);
    set(body_handle, 'XData', p(1), 'YData', p(2), 'ZData', p(3));
    if opt.trail
        set(trail_handle, 'XData', positions(1:step, 1), ...
            'YData', positions(1:step, 2), ...
            'ZData', positions(1:step, 3));
    end
    drawnow limitrate;
    pause(opt.pause_s);
end
end

function arm_length = estimate_arm_length(logs, vehicle_id)
arm_length = 0.05;
if isfield(logs, 'dynamics') && size(logs.dynamics, 1) >= vehicle_id && size(logs.dynamics, 2) >= 26
    arm_length = logs.dynamics(vehicle_id, 26);
end
arm_length = max(arm_length, 0.02);
end

function opt = parse_options(opt, varargin)
if mod(numel(varargin), 2) ~= 0
    error('Name-value arguments must come in pairs.');
end
for i = 1:2:numel(varargin)
    name = varargin{i};
    value = varargin{i + 1};
    if ~isfield(opt, name)
        error('Unknown replay option: %s', name);
    end
    opt.(name) = value;
end
end
