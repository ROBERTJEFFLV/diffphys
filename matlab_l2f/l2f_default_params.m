function params = l2f_default_params(varargin)
%L2F_DEFAULT_PARAMS Nominal parameters matching env_l2f.L2FParams.

params = struct();
params.dt = 0.01;
params.action_max = 1.0;
params.mass = 0.05;
params.gravity = 9.80665;
params.arm_length = 0.046;
params.yaw_drag = 0.012;
params.motor_tau = 0.06;
params.motor_authority = 1.35;
params.inertia_x = 1.4e-5;
params.inertia_y = 1.4e-5;
params.inertia_z = 2.17e-5;
params.max_initial_position = 1.0;
params.max_initial_velocity = 0.6;
params.max_initial_angle = 0.45;
params.max_initial_omega = 1.0;
params.disturbance_force_max = 0.0;
params.external_force_ratio = 0.0;
params.dynamics_profile = 'fixed';
params.broad_sampler = 'legacy';
params.environment = struct('type', 'none');

if mod(numel(varargin), 2) ~= 0
    error('Name-value arguments must come in pairs.');
end
for i = 1:2:numel(varargin)
    name = varargin{i};
    value = varargin{i + 1};
    if ~isfield(params, name)
        error('Unknown L2F parameter: %s', name);
    end
    params.(name) = value;
end
end
