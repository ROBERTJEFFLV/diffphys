function cfg = l2f_default_uav_cfg(varargin)
%L2F_DEFAULT_UAV_CFG Default vehicle configuration for MATLAB L2F rollouts.

cfg = l2f_uav_cfg_library('nominal_50g');
cfg = apply_name_values(cfg, varargin{:});
end
