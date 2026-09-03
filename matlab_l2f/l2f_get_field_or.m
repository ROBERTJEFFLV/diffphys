function value = l2f_get_field_or(s, name, default_value)
%L2F_GET_FIELD_OR Read a struct field with a default fallback.

if isstruct(s) && isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end
