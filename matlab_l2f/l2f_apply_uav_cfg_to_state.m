function state = l2f_apply_uav_cfg_to_state(state, uav_dynamics)
%L2F_APPLY_UAV_CFG_TO_STATE Attach sampled UAV dynamics to a state struct.

fields = fieldnames(uav_dynamics);
for i = 1:numel(fields)
    state.(fields{i}) = uav_dynamics.(fields{i});
end
end
