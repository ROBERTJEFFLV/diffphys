function run_motor_gru_eval_manifest(manifest_path, varargin)
%RUN_MOTOR_GRU_EVAL_MANIFEST Run resumable streaming evaluation jobs from CSV.

parser = inputParser;
addParameter(parser, 'force', false);
parse(parser, varargin{:});
force = logical(parser.Results.force);

jobs = readtable( ...
    manifest_path, ...
    'Delimiter', ',', ...
    'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
required = ["label", "weights_path", "output_path", "sample_output_path", ...
    "mat_output_path", "batch_size", "eval_seed", "horizon"];
if ~all(ismember(required, string(jobs.Properties.VariableNames)))
    error('Manifest is missing one or more required columns.');
end

[manifest_dir, manifest_name, ~] = fileparts(manifest_path);
status_path = fullfile(manifest_dir, manifest_name + "_status.csv");
if force && exist(status_path, 'file')
    delete(status_path);
end

for i = 1:height(jobs)
    output_path = char(jobs.output_path(i));
    sample_output_path = char(jobs.sample_output_path(i));
    mat_output_path = char(jobs.mat_output_path(i));
    complete = exist(output_path, 'file') && exist(sample_output_path, 'file') ...
        && exist(mat_output_path, 'file');
    if complete && ~force
        fprintf('[%d/%d] skip complete: %s\n', i, height(jobs), jobs.label(i));
        continue;
    end

    horizon = double(jobs.horizon(i));
    long_horizons = unique([500 1000 2000 5000 horizon]);
    long_horizons = long_horizons(long_horizons <= horizon);
    fprintf('[%d/%d] start: %s\n', i, height(jobs), jobs.label(i));
    started = tic;
    try
        run_motor_gru_policy_eval( ...
            'weights_path', char(jobs.weights_path(i)), ...
            'output_path', output_path, ...
            'sample_output_path', sample_output_path, ...
            'mat_output_path', mat_output_path, ...
            'batch_size', double(jobs.batch_size(i)), ...
            'seed', double(jobs.eval_seed(i)), ...
            'horizon', horizon, ...
            'long_horizons', long_horizons, ...
            'dynamics_profile', 'physical-broad', ...
            'broad_sampler', 'physical', ...
            'eval_mode', 'streaming');
        elapsed_s = toc(started);
        append_status(status_path, jobs.label(i), "complete", elapsed_s, "");
        fprintf('[%d/%d] complete in %.3f s: %s\n', ...
            i, height(jobs), elapsed_s, jobs.label(i));
    catch exception
        elapsed_s = toc(started);
        append_status(status_path, jobs.label(i), "failed", elapsed_s, ...
            string(exception.message));
        rethrow(exception);
    end
end
end

function append_status(path_value, label, status, elapsed_s, message)
new_file = ~exist(path_value, 'file');
handle = fopen(path_value, 'a');
if handle < 0
    error('Unable to open status path: %s', path_value);
end
cleanup = onCleanup(@() fclose(handle)); %#ok<NASGU>
if new_file
    fprintf(handle, 'label,status,elapsed_s,message\n');
end
safe_label = strrep(char(label), '"', '""');
safe_message = strrep(char(message), '"', '""');
fprintf(handle, '"%s","%s",%.9f,"%s"\n', ...
    safe_label, char(status), elapsed_s, safe_message);
end
