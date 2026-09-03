
function results = run_fast_eval_tests
%RUN_FAST_EVAL_TESTS Run all MATLAB tests for the streaming evaluator.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cleanup = onCleanup(@() rmpath(root)); %#ok<NASGU>
results = runtests(fullfile(root, 'tests'));
disp(table(results));
assert(all([results.Passed]), 'At least one fast-evaluation test failed.');
end
