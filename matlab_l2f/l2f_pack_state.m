function x = l2f_pack_state(state, sample_index)
%L2F_PACK_STATE Flatten one vehicle state for Simulink.

if nargin < 2
    sample_index = 1;
end
i = sample_index;
r = state.rotation(:, :, i);
x = [
    state.position(i, :).';
    state.velocity(i, :).';
    reshape(r.', 9, 1);
    state.omega(i, :).';
    state.motor(i, :).';
    state.previous_action(i, :).'
];
end
