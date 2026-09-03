function mode = l2f_normalize_observation_mode(value)
%L2F_NORMALIZE_OBSERVATION_MODE Convert MAT-loaded strings/cells to char.

while iscell(value) && isscalar(value)
    value = value{1};
end
if isstring(value)
    value = char(value);
end
mode = strtrim(char(value));
if ~ismember(mode, {'legacy40', 'compact22', 'integral25'})
    error('Unsupported observation mode: %s', mode);
end
end
