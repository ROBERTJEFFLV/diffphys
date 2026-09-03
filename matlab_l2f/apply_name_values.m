function s = apply_name_values(s, varargin)
%APPLY_NAME_VALUES Small local helper for MATLAB structs.

if mod(numel(varargin), 2) ~= 0
    error('Name-value arguments must come in pairs.');
end
for i = 1:2:numel(varargin)
    name = varargin{i};
    value = varargin{i + 1};
    if ~isfield(s, name)
        error('Unknown field: %s', name);
    end
    s.(name) = value;
end
end
