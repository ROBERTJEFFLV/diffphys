function state = l2f_reset(batch_size, params, seed, uav_cfg, sim_cfg)
%L2F_RESET Sample initial state and per-episode dynamics.

if nargin < 4
    uav_cfg = [];
end
if nargin < 5 || isempty(sim_cfg)
    sim_cfg = l2f_default_sim_cfg();
end

if nargin >= 3 && ~isempty(seed)
    try
        rng(seed);
    catch
        rand('seed', seed);
        randn('seed', seed);
    end
end

state = struct();
state.position = (2 * rand(batch_size, 3) - 1) * params.max_initial_position;
state.velocity = (2 * rand(batch_size, 3) - 1) * params.max_initial_velocity;
state.omega = (2 * rand(batch_size, 3) - 1) * params.max_initial_omega;
state.motor = zeros(batch_size, 4);
state.previous_action = zeros(batch_size, 4);

state.rotation = zeros(3, 3, batch_size);
for i = 1:batch_size
    axis = randn(3, 1);
    axis = axis / max(norm(axis), 1.0e-12);
    angle = (2 * rand() - 1) * params.max_initial_angle;
    state.rotation(:, :, i) = l2f_so3_exp(axis * angle);
end

if ~isempty(uav_cfg)
    dynamics = l2f_sample_uav_cfg(uav_cfg, batch_size, sim_cfg);
else
    dynamics = l2f_sample_dynamics(params, batch_size);
end
state = l2f_apply_uav_cfg_to_state(state, dynamics);

if isfield(state, 'motor_delay_steps')
    max_delay = max(max(state.motor_delay_steps));
    if max_delay > 0
        state.action_buffer = zeros(batch_size, 4, max_delay + 1);
    end
end
end
