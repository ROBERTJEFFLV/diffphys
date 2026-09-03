function export_q3_phase1_scenarios(output_path, seed, batch_size)
%EXPORT_Q3_PHASE1_SCENARIOS Export the exact formal reset stream for diagnostics.
%
% This is a diagnostics-only wrapper around the existing reset/export path.  It
% intentionally does not modify the simulator, controller, or evaluation
% semantics used by the Q2/T0 baseline.

if nargin < 2 || isempty(seed)
    seed = 1007;
end
if nargin < 3 || isempty(batch_size)
    batch_size = 1024;
end
export_dynamic_hard_scenarios(output_path, (1:batch_size).', seed, batch_size);
end
