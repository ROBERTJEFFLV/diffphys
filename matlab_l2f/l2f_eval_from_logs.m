
function result = l2f_eval_from_logs(logs, world, reward_cfg, requested_horizons)
%L2F_EVAL_FROM_LOGS Compute standard evaluation products from full logs.

[metrics, errors] = l2f_metrics(logs, world, reward_cfg);
long_metrics = l2f_long_horizon_metrics( ...
    logs, world, reward_cfg, requested_horizons, errors);
invalid_fraction = l2f_invalid_fraction_logs(logs);
summary = l2f_eval_summary(metrics, long_metrics, invalid_fraction);

result = struct();
result.metrics = metrics;
result.long_metrics = long_metrics;
result.summary = summary;
result.dynamics = logs.dynamics;
result.invalid_fraction = invalid_fraction;
result.mode = 'full';
end
