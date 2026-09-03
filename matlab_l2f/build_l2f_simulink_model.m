function build_l2f_simulink_model(model_name)
%BUILD_L2F_SIMULINK_MODEL Create a flat-vector Simulink scaffold.
%
% The generated model is intentionally simple: Unit Delay holds the flattened
% state, a MATLAB Function block calls l2f_step_flat, and constants provide the
% action, dynamics, environment wrench and scalar parameters.

if nargin < 1 || isempty(model_name)
    model_name = 'l2f_simulink_env';
end

required_functions = {'bdIsLoaded', 'new_system', 'open_system', 'add_block', 'add_line', 'set_param', 'save_system'};
for i = 1:numel(required_functions)
    if exist(required_functions{i}, 'file') == 0 && exist(required_functions{i}, 'builtin') == 0
        error(['Simulink function %s is not visible. ', ...
            'Check that Simulink is installed and MATLAB path initialization is healthy.'], required_functions{i});
    end
end

params = l2f_default_params();
state = l2f_reset(1, params, 7);
x0 = l2f_pack_state(state, 1);
dyn0 = l2f_pack_dynamics(state, 1);
params_vec = [params.dt; params.gravity];

assignin('base', 'l2f_x0', x0);
assignin('base', 'l2f_dyn0', dyn0);
assignin('base', 'l2f_params_vec', params_vec);
assignin('base', 'l2f_action0', zeros(4, 1));
assignin('base', 'l2f_env_force0', zeros(3, 1));
assignin('base', 'l2f_env_torque0', zeros(3, 1));

if bdIsLoaded(model_name)
    close_system(model_name, 0);
end
new_system(model_name);
open_system(model_name);

add_block('simulink/Sources/Constant', [model_name '/Action'], ...
    'Value', 'l2f_action0', 'Position', [40 80 130 110]);
add_block('simulink/Sources/Constant', [model_name '/Dynamics'], ...
    'Value', 'l2f_dyn0', 'Position', [40 140 130 170]);
add_block('simulink/Sources/Constant', [model_name '/Env Force'], ...
    'Value', 'l2f_env_force0', 'Position', [40 200 130 230]);
add_block('simulink/Sources/Constant', [model_name '/Env Torque'], ...
    'Value', 'l2f_env_torque0', 'Position', [40 260 130 290]);
add_block('simulink/Sources/Constant', [model_name '/Params'], ...
    'Value', 'l2f_params_vec', 'Position', [40 320 130 350]);

add_block('simulink/Discrete/Unit Delay', [model_name '/State Delay'], ...
    'InitialCondition', 'l2f_x0', 'SampleTime', num2str(params.dt), ...
    'Position', [40 20 130 50]);

add_block('simulink/User-Defined Functions/MATLAB Function', [model_name '/L2F Step'], ...
    'Position', [240 100 390 260]);

add_block('simulink/Sinks/To Workspace', [model_name '/State Log'], ...
    'VariableName', 'l2f_state_log', 'SaveFormat', 'Array', ...
    'Position', [510 80 620 110]);
add_block('simulink/Sinks/To Workspace', [model_name '/Output Log'], ...
    'VariableName', 'l2f_output_log', 'SaveFormat', 'Array', ...
    'Position', [510 170 620 200]);
add_block('simulink/Sinks/Scope', [model_name '/Realtime Scope'], ...
    'Position', [510 250 620 300]);

add_line(model_name, 'State Delay/1', 'L2F Step/1');
add_line(model_name, 'Action/1', 'L2F Step/2');
add_line(model_name, 'Dynamics/1', 'L2F Step/3');
add_line(model_name, 'Env Force/1', 'L2F Step/4');
add_line(model_name, 'Env Torque/1', 'L2F Step/5');
add_line(model_name, 'Params/1', 'L2F Step/6');
add_line(model_name, 'L2F Step/1', 'State Delay/1');
add_line(model_name, 'L2F Step/1', 'State Log/1');
add_line(model_name, 'L2F Step/2', 'Output Log/1');
add_line(model_name, 'L2F Step/2', 'Realtime Scope/1');

try
    root = sfroot;
    chart = root.find('-isa', 'Stateflow.EMChart', 'Path', [model_name '/L2F Step']);
    chart.Script = sprintf([ ...
        'function [x_next, y] = fcn(x, action, dyn, env_force, env_torque, params_vec)\n' ...
        '%%#codegen\n' ...
        '[x_next, y] = l2f_step_flat(x, action, dyn, env_force, env_torque, params_vec);\n' ...
        'end\n']);
catch err
    warning('Could not configure MATLAB Function block script: %s', err.message);
end

set_param(model_name, 'StopTime', '5');
save_system(model_name);
fprintf('Created Simulink scaffold: %s.slx\n', model_name);
end
