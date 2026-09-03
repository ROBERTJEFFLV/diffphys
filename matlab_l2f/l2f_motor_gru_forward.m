function [action, next_hidden, details] = l2f_motor_gru_forward(weights, observation, hidden)
%L2F_MOTOR_GRU_FORWARD MATLAB forward pass for model.MotorGRUPolicy.

if ~isfield(weights, 'encoder_0_weight_t')
    weights = l2f_prepare_motor_gru_weights(weights);
end
if size(observation, 2) ~= double(weights.input_dim)
    error('Observation width %d does not match exported policy width %d.', ...
        size(observation, 2), double(weights.input_dim));
end
negative_slope = double(weights.negative_slope);

encoded = l2f_linear(observation, weights.encoder_0_weight_t, weights.encoder_0_bias);
encoded = l2f_leaky_relu(encoded, negative_slope);
encoded = l2f_linear(encoded, weights.encoder_2_weight_t, weights.encoder_2_bias);
encoded = l2f_leaky_relu(encoded, negative_slope);

next_hidden = l2f_gru_cell(encoded, hidden, weights);
head_input = l2f_leaky_relu(next_hidden, negative_slope);
main_logits = l2f_linear(head_input, weights.motor_head_weight_t, weights.motor_head_bias);
integral_logits = zeros(size(main_logits), 'like', main_logits);
if weights.enable_integral_residual
    integral_hidden = l2f_linear(observation(:, 19:21), ...
        weights.integral_residual_0_weight_t, weights.integral_residual_0_bias);
    integral_hidden = l2f_leaky_relu(integral_hidden, negative_slope);
    integral_logits = double(weights.integral_residual_scale) .* l2f_linear( ...
        integral_hidden, weights.integral_residual_2_weight_t, weights.integral_residual_2_bias);
end
integral_action = tanh(main_logits + integral_logits);
damping_logits = zeros(size(main_logits), 'like', main_logits);
motor_state_hat = zeros(size(main_logits), 'like', main_logits);
if weights.enable_rate_damping_residual
    motor_state_hat = tanh(l2f_linear(head_input, ...
        weights.motor_state_head_weight_t, weights.motor_state_head_bias));
    damping_input = [head_input, observation(:, 16:18), observation(:, 22:25), motor_state_hat];
    damping_hidden = l2f_linear(damping_input, ...
        weights.damping_residual_0_weight_t, weights.damping_residual_0_bias);
    damping_hidden = l2f_leaky_relu(damping_hidden, negative_slope);
    damping_logits = double(weights.damping_residual_scale) .* l2f_linear( ...
        damping_hidden, weights.damping_residual_2_weight_t, weights.damping_residual_2_bias);
end
action = tanh(main_logits + integral_logits + damping_logits);
details = struct();
details.main_logits = main_logits;
details.integral_residual_logits = integral_logits;
details.damping_residual_logits = damping_logits;
details.integral_action_contribution = integral_action - tanh(main_logits);
details.damping_action_contribution = action - integral_action;
details.motor_state = motor_state_hat;
end

function y = l2f_linear(x, weight_t, bias)
y = x * weight_t + bias;
end

function y = l2f_leaky_relu(x, negative_slope)
y = max(x, 0.0) + negative_slope * min(x, 0.0);
end

function next_hidden = l2f_gru_cell(x, hidden, weights)
hidden_dim = size(hidden, 2);
ih = x * weights.gru_weight_ih_t + weights.gru_bias_ih;
hh = hidden * weights.gru_weight_hh_t + weights.gru_bias_hh;

i_r = ih(:, 1:hidden_dim);
i_z = ih(:, hidden_dim + 1:2 * hidden_dim);
i_n = ih(:, 2 * hidden_dim + 1:3 * hidden_dim);
h_r = hh(:, 1:hidden_dim);
h_z = hh(:, hidden_dim + 1:2 * hidden_dim);
h_n = hh(:, 2 * hidden_dim + 1:3 * hidden_dim);

reset_gate = sigmoid(i_r + h_r);
update_gate = sigmoid(i_z + h_z);
new_gate = tanh(i_n + reset_gate .* h_n);
next_hidden = (1.0 - update_gate) .* new_gate + update_gate .* hidden;
end

function y = sigmoid(x)
y = 1.0 ./ (1.0 + exp(-x));
end
