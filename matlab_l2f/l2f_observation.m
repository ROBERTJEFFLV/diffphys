function [observation, observed_position] = l2f_observation(state, observation_mode, integral_position, integral_input_frame, integral_input_multiplier)
%L2F_OBSERVATION Build the deployable MotorGRU observation.
%
% Physical state is sampled once. legacy40 derives its affine duplicate from
% the same values; compact22 removes it; integral25 inserts the position
% integral before previous_action.

if nargin < 2 || isempty(observation_mode)
    observation_mode = 'integral25';
end
batch_size = size(state.position, 1);
if nargin < 3 || isempty(integral_position)
    integral_position = zeros(batch_size, 3, 'like', state.position);
end
if nargin < 4 || isempty(integral_input_frame)
    integral_input_frame = 'world';
end
if nargin < 5 || isempty(integral_input_multiplier)
    integral_input_multiplier = 1.0;
end

rotation_flat = reshape(permute(state.rotation, [2, 1, 3]), 9, batch_size).';
physical = [state.position, state.velocity, rotation_flat, state.omega];
observed_position = physical(:, 1:3);
observation = l2f_build_policy_observation( ...
    physical, state.previous_action, integral_position, observation_mode, ...
    integral_input_frame, integral_input_multiplier);
end
