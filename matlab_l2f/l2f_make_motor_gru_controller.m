
function controller = l2f_make_motor_gru_controller(weights_path)
%L2F_MAKE_MOTOR_GRU_CONTROLLER Create a MATLAB controller from exported MotorGRUPolicy weights.

weights = l2f_prepare_motor_gru_weights(load(weights_path));
hidden = [];
integral_position = [];
controller = @controller_step;

    function [action, info] = controller_step(obs, state, params, world, t) %#ok<INUSD>
        batch_size = size(obs.physical_features, 1);
        hidden_dim = double(weights.hidden_dim);
        if isempty(hidden) || size(hidden, 1) ~= batch_size
            hidden = zeros(batch_size, hidden_dim, 'like', obs.physical_features);
            integral_position = zeros(batch_size, 3, 'like', obs.physical_features);
        end
        integral_world_input = integral_position;
        observation = l2f_build_policy_observation( ...
            obs.physical_features, obs.previous_action, integral_position, ...
            weights.observation_mode, weights.integral_input_frame, ...
            weights.integral_input_multiplier);
        [action, hidden, action_details] = l2f_motor_gru_forward(weights, observation, hidden);
        integral_position = l2f_update_position_integral( ...
            integral_position, obs.observed_position, params.dt, ...
            weights.integral_limit, weights.integral_leak);
        info = struct();
        info.hidden_norm = sqrt(sum(hidden .* hidden, 2)).';
        info.hidden_abs_max = max(abs(hidden), [], 2).';
        info.integral_world = integral_world_input;
        info.integral_body = observation(:, 19:21);
        info.integral_action_contribution = action_details.integral_action_contribution;
        info.damping_action_contribution = action_details.damping_action_contribution;
    end
end
