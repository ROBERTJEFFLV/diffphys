function [x_next, y] = l2f_step_flat(x, action, dyn, env_force, env_torque, params_vec)
%L2F_STEP_FLAT Simulink-friendly one-vehicle step wrapper.

params = l2f_default_params('dt', params_vec(1), 'gravity', params_vec(2));
state = l2f_unpack_state(x, dyn);
environment = struct('type', 'external_wrench', 'force', env_force(:).', 'torque', env_torque(:).');
[next_state, aux] = l2f_step(state, action(:).', params, environment, 0);
x_next = l2f_pack_state(next_state, 1);
euler = l2f_matrix_to_euler_zyx(next_state.rotation);
y = [
    next_state.position(1, :).';
    next_state.velocity(1, :).';
    euler(1, :).';
    next_state.omega(1, :).';
    next_state.motor(1, :).';
    aux.command(1, :).';
    aux.thrust(1, :).';
    aux.torque(1, :).'
];
end
