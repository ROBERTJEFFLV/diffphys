function observation = l2f_build_policy_observation(physical, previous_action, integral_position, observation_mode, integral_input_frame, integral_input_multiplier)
%L2F_BUILD_POLICY_OBSERVATION Assemble a 22D, 25D or legacy 40D input.

mode = l2f_normalize_observation_mode(observation_mode);
if nargin < 5 || isempty(integral_input_frame)
    integral_input_frame = 'world';
end
if nargin < 6 || isempty(integral_input_multiplier)
    integral_input_multiplier = 1.0;
end
if ~isscalar(integral_input_multiplier) || ~isfinite(integral_input_multiplier) || integral_input_multiplier < 0
    error('integral_input_multiplier must be a finite non-negative scalar.');
end
integral_input_frame = lower(char(string(integral_input_frame)));
if ~any(strcmp(integral_input_frame, {'world', 'body'}))
    error('integral_input_frame must be ''world'' or ''body''.');
end
if size(physical, 2) ~= 18
    error('physical observation must have 18 columns.');
end
if size(previous_action, 2) ~= 4
    error('previous_action must have 4 columns.');
end
switch mode
    case 'legacy40'
        duplicate = physical;
        duplicate(:, [7, 11, 15]) = duplicate(:, [7, 11, 15]) - 1.0;
        observation = [physical, duplicate, previous_action];
    case 'compact22'
        observation = [physical, previous_action];
    case 'integral25'
        if size(integral_position, 2) ~= 3 || size(integral_position, 1) ~= size(physical, 1)
            error('integral_position must have shape [batch,3].');
        end
        integral_input = integral_position;
        if strcmp(integral_input_frame, 'body')
            % physical(:,7:15) stores row-major body-to-world R.  The
            % deployed feature is R^T*I_world, using that same observation.
            integral_input = [ ...
                sum(integral_position .* physical(:, [7, 10, 13]), 2), ...
                sum(integral_position .* physical(:, [8, 11, 14]), 2), ...
                sum(integral_position .* physical(:, [9, 12, 15]), 2)];
        end
        integral_input = double(integral_input_multiplier) .* integral_input;
        observation = [physical, integral_input, previous_action];
    otherwise
        error('Unsupported observation mode: %s', mode);
end
end
