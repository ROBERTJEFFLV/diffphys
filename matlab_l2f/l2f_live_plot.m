function monitor = l2f_live_plot(mode, monitor, logs, step)
%L2F_LIVE_PLOT Realtime trajectory/state monitor.

switch lower(mode)
    case 'init'
        monitor = struct();
        monitor.figure = figure('Name', 'L2F MATLAB Quadrotor Monitor', 'Color', 'w');
        tiledlayout(monitor.figure, 3, 2, 'TileSpacing', 'compact');
        monitor.ax_traj = nexttile;
        monitor.h_traj = plot3(monitor.ax_traj, 0, 0, 0, 'LineWidth', 1.5);
        grid(monitor.ax_traj, 'on');
        xlabel(monitor.ax_traj, 'x (m)'); ylabel(monitor.ax_traj, 'y (m)'); zlabel(monitor.ax_traj, 'z (m)');
        title(monitor.ax_traj, 'trajectory');

        monitor.ax_pos = nexttile;
        monitor.h_pos = gobjects(1, 3);
        hold(monitor.ax_pos, 'on');
        for i = 1:3
            monitor.h_pos(i) = plot(monitor.ax_pos, NaN, NaN, 'LineWidth', 1.2);
        end
        hold(monitor.ax_pos, 'off');
        grid(monitor.ax_pos, 'on'); title(monitor.ax_pos, 'position'); xlabel(monitor.ax_pos, 'time (s)');
        ylabel(monitor.ax_pos, 'm'); legend(monitor.ax_pos, {'x', 'y', 'z'});

        monitor.ax_euler = nexttile;
        monitor.h_euler = gobjects(1, 3);
        hold(monitor.ax_euler, 'on');
        for i = 1:3
            monitor.h_euler(i) = plot(monitor.ax_euler, NaN, NaN, 'LineWidth', 1.2);
        end
        hold(monitor.ax_euler, 'off');
        grid(monitor.ax_euler, 'on'); title(monitor.ax_euler, 'attitude'); xlabel(monitor.ax_euler, 'time (s)');
        ylabel(monitor.ax_euler, 'deg'); legend(monitor.ax_euler, {'roll', 'pitch', 'yaw'});

        monitor.ax_omega = nexttile;
        monitor.h_omega = gobjects(1, 3);
        hold(monitor.ax_omega, 'on');
        for i = 1:3
            monitor.h_omega(i) = plot(monitor.ax_omega, NaN, NaN, 'LineWidth', 1.2);
        end
        hold(monitor.ax_omega, 'off');
        grid(monitor.ax_omega, 'on'); title(monitor.ax_omega, 'body rate'); xlabel(monitor.ax_omega, 'time (s)');
        ylabel(monitor.ax_omega, 'rad/s'); legend(monitor.ax_omega, {'wx', 'wy', 'wz'});

        monitor.ax_action = nexttile;
        monitor.h_action = gobjects(1, 4);
        hold(monitor.ax_action, 'on');
        for i = 1:4
            monitor.h_action(i) = plot(monitor.ax_action, NaN, NaN, 'LineWidth', 1.2);
        end
        hold(monitor.ax_action, 'off');
        grid(monitor.ax_action, 'on'); title(monitor.ax_action, 'action'); xlabel(monitor.ax_action, 'time (s)');
        ylabel(monitor.ax_action, 'normalized'); legend(monitor.ax_action, {'a0', 'a1', 'a2', 'a3'});

        monitor.ax_motor = nexttile;
        monitor.h_motor = gobjects(1, 4);
        hold(monitor.ax_motor, 'on');
        for i = 1:4
            monitor.h_motor(i) = plot(monitor.ax_motor, NaN, NaN, 'LineWidth', 1.2);
        end
        hold(monitor.ax_motor, 'off');
        grid(monitor.ax_motor, 'on'); title(monitor.ax_motor, 'motor state'); xlabel(monitor.ax_motor, 'time (s)');
        ylabel(monitor.ax_motor, 'normalized'); legend(monitor.ax_motor, {'m0', 'm1', 'm2', 'm3'});
    case 'update'
        idx = 1:(step + 1);
        t_state = logs.time(idx);
        position = select_series(logs.position, idx, 1);
        euler_deg = select_series(logs.euler_deg, idx, 1);
        omega = select_series(logs.omega, idx, 1);
        set(monitor.h_traj, 'XData', position(:, 1), 'YData', position(:, 2), 'ZData', position(:, 3));
        for i = 1:3
            set(monitor.h_pos(i), 'XData', t_state, 'YData', position(:, i));
            set(monitor.h_euler(i), 'XData', t_state, 'YData', euler_deg(:, i));
            set(monitor.h_omega(i), 'XData', t_state, 'YData', omega(:, i));
        end
        if step > 0
            idx_u = 1:step;
            t_action = logs.time(idx_u);
            action = select_series(logs.action, idx_u, 1);
            motor = select_series(logs.motor, idx_u, 1);
            for i = 1:4
                set(monitor.h_action(i), 'XData', t_action, 'YData', action(:, i));
                set(monitor.h_motor(i), 'XData', t_action, 'YData', motor(:, i));
            end
        end
        drawnow limitrate;
    otherwise
        error('Unknown live plot mode: %s', mode);
end
end

function values = select_series(array, idx, vehicle_id)
if ndims(array) == 3
    values = reshape(array(idx, vehicle_id, :), numel(idx), size(array, 3));
elseif ndims(array) == 2
    values = array(idx, :);
else
    error('Unsupported log array dimensions.');
end
if isvector(values)
    values = reshape(values, numel(idx), []);
end
end
