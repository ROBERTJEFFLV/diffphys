
function weights = l2f_prepare_motor_gru_weights(weights)
%L2F_PREPARE_MOTOR_GRU_WEIGHTS Precompute fixed transposes and bias shapes.
%
% This changes only the MATLAB-side representation. No value is cast and no
% learned parameter is modified.

required = { ...
    'encoder_0_weight', 'encoder_0_bias', ...
    'encoder_2_weight', 'encoder_2_bias', ...
    'gru_weight_ih', 'gru_bias_ih', ...
    'gru_weight_hh', 'gru_bias_hh', ...
    'motor_head_weight', 'motor_head_bias', ...
    'hidden_dim', 'negative_slope'};
for i = 1:numel(required)
    if ~isfield(weights, required{i})
        error('Missing MotorGRUPolicy weight field: %s', required{i});
    end
end

weights.encoder_0_weight_t = weights.encoder_0_weight.';
weights.encoder_2_weight_t = weights.encoder_2_weight.';
weights.gru_weight_ih_t = weights.gru_weight_ih.';
weights.gru_weight_hh_t = weights.gru_weight_hh.';
weights.motor_head_weight_t = weights.motor_head_weight.';

weights.encoder_0_bias = reshape(weights.encoder_0_bias, 1, []);
weights.encoder_2_bias = reshape(weights.encoder_2_bias, 1, []);
weights.gru_bias_ih = reshape(weights.gru_bias_ih, 1, []);
weights.gru_bias_hh = reshape(weights.gru_bias_hh, 1, []);
weights.motor_head_bias = reshape(weights.motor_head_bias, 1, []);

if ~isfield(weights, 'input_dim') || isempty(weights.input_dim)
    weights.input_dim = size(weights.encoder_0_weight, 2);
end
input_dim = double(weights.input_dim);
if ~isfield(weights, 'observation_mode') || isempty(weights.observation_mode)
    switch input_dim
        case 40
            weights.observation_mode = 'legacy40';
        case 22
            weights.observation_mode = 'compact22';
        case 25
            weights.observation_mode = 'integral25';
        otherwise
            error('Unsupported MotorGRU observation width: %d', input_dim);
    end
end
weights.observation_mode = l2f_normalize_observation_mode(weights.observation_mode);
expected_width = struct('legacy40', 40, 'compact22', 22, 'integral25', 25);
if input_dim ~= expected_width.(weights.observation_mode)
    error('Exported observation_mode disagrees with encoder input width.');
end
if ~isfield(weights, 'integral_limit') || isempty(weights.integral_limit)
    weights.integral_limit = 0.5;
end
if ~isfield(weights, 'integral_leak') || isempty(weights.integral_leak)
    weights.integral_leak = 0.0;
end
if ~isfield(weights, 'integral_input_frame') || isempty(weights.integral_input_frame)
    weights.integral_input_frame = 'world';
end
weights.integral_input_frame = lower(char(string(weights.integral_input_frame)));
if ~any(strcmp(weights.integral_input_frame, {'world', 'body'}))
    error('Exported integral_input_frame must be ''world'' or ''body''.');
end
if ~isfield(weights, 'integral_input_multiplier') || isempty(weights.integral_input_multiplier)
    weights.integral_input_multiplier = 1.0;
end
weights.integral_input_multiplier = double(weights.integral_input_multiplier);
if ~isscalar(weights.integral_input_multiplier) || ...
        ~isfinite(weights.integral_input_multiplier) || weights.integral_input_multiplier < 0
    error('Exported integral_input_multiplier must be finite and non-negative.');
end
if ~isfield(weights, 'enable_integral_residual') || isempty(weights.enable_integral_residual)
    weights.enable_integral_residual = false;
end
if ~isfield(weights, 'enable_rate_damping_residual') || isempty(weights.enable_rate_damping_residual)
    weights.enable_rate_damping_residual = false;
end
weights.enable_integral_residual = logical(double(weights.enable_integral_residual));
weights.enable_rate_damping_residual = logical(double(weights.enable_rate_damping_residual));
if ~isfield(weights, 'integral_residual_scale') || isempty(weights.integral_residual_scale)
    weights.integral_residual_scale = 1.0;
end
if ~isfield(weights, 'damping_residual_scale') || isempty(weights.damping_residual_scale)
    weights.damping_residual_scale = 1.0;
end
if (weights.enable_integral_residual || weights.enable_rate_damping_residual) && input_dim ~= 25
    error('Residual control branches require the 25D observation.');
end
if weights.enable_integral_residual
    required_integral = {'integral_residual_0_weight', 'integral_residual_0_bias', ...
        'integral_residual_2_weight', 'integral_residual_2_bias'};
    require_fields(weights, required_integral);
    weights.integral_residual_0_weight_t = weights.integral_residual_0_weight.';
    weights.integral_residual_2_weight_t = weights.integral_residual_2_weight.';
    weights.integral_residual_0_bias = reshape(weights.integral_residual_0_bias, 1, []);
    weights.integral_residual_2_bias = reshape(weights.integral_residual_2_bias, 1, []);
end
if weights.enable_rate_damping_residual
    required_damping = {'motor_state_head_weight', 'motor_state_head_bias', ...
        'damping_residual_0_weight', 'damping_residual_0_bias', ...
        'damping_residual_2_weight', 'damping_residual_2_bias'};
    require_fields(weights, required_damping);
    weights.motor_state_head_weight_t = weights.motor_state_head_weight.';
    weights.damping_residual_0_weight_t = weights.damping_residual_0_weight.';
    weights.damping_residual_2_weight_t = weights.damping_residual_2_weight.';
    weights.motor_state_head_bias = reshape(weights.motor_state_head_bias, 1, []);
    weights.damping_residual_0_bias = reshape(weights.damping_residual_0_bias, 1, []);
    weights.damping_residual_2_bias = reshape(weights.damping_residual_2_bias, 1, []);
end
end

function require_fields(values, names)
for index = 1:numel(names)
    if ~isfield(values, names{index})
        error('Missing residual deployment weight field: %s', names{index});
    end
end
end
