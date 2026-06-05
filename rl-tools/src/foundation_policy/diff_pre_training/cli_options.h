#pragma once

#include "config.h"

#include <cmath>
#include <iostream>
#include <string>

namespace rl_tools::foundation_policy::diff_pre_training{
    namespace rlt = rl_tools;

    enum class DiffModel{
        EULER,
        L2F_APPROX
    };

    enum class EvalModel{
        EULER,
        L2F
    };

    enum class SampledDynamicsLevel{
        FIXED,
        NARROW,
        MEDIUM,
        BROAD
    };

    struct RuntimeOptions{
        TI num_steps = DIFF_TRAINING_NUM_STEPS;
        TI horizon = DIFF_TRAINING_HORIZON;
        TI batch_size = DIFF_TRAINING_BATCH_SIZE;
        TI eval_episodes = 100;
        TI eval_horizon = 128;
        TI seed = 0;
        bool sample_dynamics = DIFF_TRAINING_SAMPLE_DYNAMICS;
        bool eval_only = false;
        bool disable_physics_gradient = false;
        bool reset_hidden_each_step = false;
        bool horizon_curriculum = HORIZON_CURRICULUM_ENABLED;
        bool state_curriculum = STATE_CURRICULUM_ENABLED;
        bool dynamics_curriculum = DYNAMICS_CURRICULUM_ENABLED;
        bool action_grad_clip_enabled = ACTION_GRAD_CLIP_ENABLED;
        bool actor_grad_clip_enabled = ACTOR_GRAD_CLIP_ENABLED;
        bool action_bound_enabled = ACTION_BOUND_ENABLED;
        bool terminal_ramp_after_horizon_change = false;
        bool terminal_ramp_terminal_only = false;
        bool reset_optimizer_on_curriculum_transition = false;
        TI horizon_stage_steps = HORIZON_STAGE_STEPS;
        TI state_curriculum_stage_steps = STATE_CURRICULUM_STAGE_STEPS;
        TI dynamics_curriculum_stage_steps = DYNAMICS_CURRICULUM_STAGE_STEPS;
        TI terminal_ramp_steps = 1000;
        T action_grad_clip_norm = ACTION_GRAD_CLIP_NORM;
        T actor_grad_clip_norm = ACTOR_GRAD_CLIP_NORM;
        T actor_grad_skip_norm = ACTOR_GRAD_SKIP_NORM;
        T actor_grad_eps = ACTOR_GRAD_EPS;
        T action_bound_value = ACTION_BOUND_VALUE;
        T terminal_ramp_min = (T)0.25;
        bool decoupled_curriculum = false;
        bool one_curriculum_axis_at_a_time = false;
        bool stability_gated_curriculum = false;
        TI state_curriculum_lag_steps = 0;
        TI hold_h64_extra_steps = 0;
        TI curriculum_stability_window = 100;
        TI curriculum_min_stage_steps = 1000;
        TI curriculum_max_stage_steps = 3000;
        T curriculum_loss_spike_ratio = (T)2.0;
        bool success_gated_curriculum = false;
        TI curriculum_success_window = 100;
        T curriculum_success_threshold = (T)0.25;
        TI curriculum_gate_check_interval = 50;
        bool curriculum_no_forced_advance = true;
        std::string curriculum_stage_plan = "alternating";
        bool diagnostic_log_detail = false;
        TI diagnostic_log_every = 10;
        TI diagnostic_log_first_steps = 200;
        T fixed_state_difficulty = (T)-1;  // -1 = disabled, use curriculum
        bool h128_prioritized_curriculum = false;
        bool sampled_dynamics_curriculum_levels = false;
        bool balanced_dynamics_sampling = BALANCED_DYNAMICS_SAMPLING_ENABLED;
        std::string h128_schedule = "short_warmup_12000";
        SampledDynamicsLevel sampled_dynamics_level = SampledDynamicsLevel::BROAD;
        T loss_velocity_weight = LOSS_VELOCITY_WEIGHT;
        T loss_angular_velocity_weight = LOSS_ANGULAR_VELOCITY_WEIGHT;
        T terminal_velocity_weight = TERMINAL_VELOCITY_WEIGHT;
        T terminal_angular_velocity_weight = TERMINAL_ANGULAR_VELOCITY_WEIGHT;
        DiffModel diff_model = DiffModel::EULER;
        EvalModel eval_model = EvalModel::EULER;
        std::string save_path;
        std::string load_path;
        std::string init_actor_path;
        std::string log_path;
        std::string sampler_dump_path;
        TI sampler_dump_samples = 0;
        bool failure_analysis = false;
        std::string failure_analysis_path;
        bool force_dynamics_bins = false;
        TI force_size_mass_bin = 0;
        TI force_thrust_to_weight_bin = 0;
        TI force_torque_to_inertia_bin = 0;
        TI force_motor_delay_bin = 0;
        TI force_curve_shape_bin = 0;
        bool gpu_rollout = false;
        TI gpu_device = 0;
        TI gpu_batch_size = 0;
        bool gpu_validate_against_cpu = false;
        bool gpu_benchmark = false;
        TI gpu_benchmark_iterations = 20;
        bool production_objective_trace = false;
        std::string objective_trace_path;
        bool stage9_6_objective_parity = false;
        bool stage9_6_sampler_parity = false;
        bool stage9_6_eval_parity = false;
        bool stage9_6_checkpoint_parity = false;
        TI stage9_6_steps = 1000;
        std::string stage9_6_replay_path;
        bool save_optimizer = false;
        bool load_optimizer = false;
        bool checkpoint_inspect = false;
        std::string checkpoint_inspect_path;
        bool checkpoint_convert_old = false;
        std::string checkpoint_convert_old_path;
    };

    inline std::string diff_model_name(DiffModel model){
        return model == DiffModel::EULER ? "euler" : "l2f_approx";
    }

    inline std::string eval_model_name(EvalModel model){
        return model == EvalModel::EULER ? "euler" : "l2f";
    }

    inline std::string sampled_dynamics_level_name(SampledDynamicsLevel level){
        switch(level){
            case SampledDynamicsLevel::FIXED: return "fixed";
            case SampledDynamicsLevel::NARROW: return "narrow";
            case SampledDynamicsLevel::MEDIUM: return "medium";
            case SampledDynamicsLevel::BROAD: return "broad";
        }
        return "unknown";
    }

    inline bool parse_sampled_dynamics_level(const std::string& value, SampledDynamicsLevel& level){
        if(value == "fixed"){
            level = SampledDynamicsLevel::FIXED;
            return true;
        }
        if(value == "narrow"){
            level = SampledDynamicsLevel::NARROW;
            return true;
        }
        if(value == "medium"){
            level = SampledDynamicsLevel::MEDIUM;
            return true;
        }
        if(value == "broad"){
            level = SampledDynamicsLevel::BROAD;
            return true;
        }
        return false;
    }

    inline void print_usage(){
        std::cout << "Usage: foundation_policy_diff_pre_training [--steps N] [--horizon N] [--batch-size N] [--seed N]\n"
                  << "    [--fixed-dynamics|--sample-dynamics] [--diff-model euler|l2f_approx]\n"
                  << "    [--sampled-dynamics-level fixed|narrow|medium|broad] [--sampled-dynamics-curriculum-levels]\n"
                  << "    [--balanced-dynamics-sampling|--disable-balanced-dynamics-sampling]\n"
                  << "    [--disable-physics-gradient] [--reset-hidden-each-step]\n"
                  << "    [--horizon-curriculum] [--state-curriculum] [--dynamics-curriculum]\n"
                  << "    [--action-grad-clip VALUE] [--disable-action-grad-clip]\n"
                  << "    [--actor-grad-clip VALUE] [--disable-actor-grad-clip] [--actor-grad-skip-norm VALUE]\n"
                  << "    [--grad-clip VALUE]  (deprecated alias for --action-grad-clip)\n"
                  << "    [--disable-grad-clip]  (deprecated alias for --disable-action-grad-clip)\n"
                  << "    [--action-bound VALUE]\n"
                  << "    [--terminal-ramp-after-horizon-change] [--terminal-ramp-min VALUE] [--terminal-ramp-steps N]\n"
                  << "    [--terminal-ramp-terminal-only] [--reset-optimizer-on-curriculum-transition]\n"
                  << "    [--decoupled-curriculum] [--state-curriculum-lag-steps N]\n"
                  << "    [--one-curriculum-axis-at-a-time] [--hold-h64-extra-steps N]\n"
                  << "    [--stability-gated-curriculum] [--curriculum-stability-window N]\n"
                  << "    [--curriculum-min-stage-steps N] [--curriculum-max-stage-steps N]\n"
                  << "    [--curriculum-loss-spike-ratio VALUE]\n"
                  << "    [--eval-only] [--eval-model euler|l2f] [--eval-episodes N] [--eval-horizon H]\n"
                  << "    [--failure-analysis] [--failure-analysis-path PATH] [--force-dynamics-bins S T Q D C]\n"
                  << "    [--gpu-rollout] [--gpu-device N] [--gpu-batch-size N]\n"
                  << "    [--gpu-validate-against-cpu] [--gpu-benchmark] [--gpu-benchmark-iterations N]\n"
                  << "    [--save-path PATH] [--load-path PATH] [--init-actor-path PATH] [--log-path PATH]\n"
                  << "    [--production-objective-trace] [--objective-trace-path PATH]\n"
                  << "    [--stage9-6-objective-parity] [--stage9-6-sampler-parity] [--stage9-6-eval-parity]\n"
                  << "    [--stage9-6-steps N] [--stage9-6-replay-path PATH]\n"
                  << "    [--sampler-dump-path PATH] [--sampler-dump-samples N]\n"
                  << "  Physics checks are available via the separate target foundation_policy_diff_physics_check.\n";
    }

    inline bool parse_options(int argc, char** argv, RuntimeOptions& options){
        for(int arg_i = 1; arg_i < argc; arg_i++){
            std::string arg = argv[arg_i];
            if(arg == "--steps" && arg_i + 1 < argc){
                options.num_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--horizon" && arg_i + 1 < argc){
                options.horizon = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--batch-size" && arg_i + 1 < argc){
                options.batch_size = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--eval-episodes" && arg_i + 1 < argc){
                options.eval_episodes = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--eval-horizon" && arg_i + 1 < argc){
                options.eval_horizon = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--seed" && arg_i + 1 < argc){
                options.seed = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--fixed-dynamics"){
                options.sample_dynamics = false;
                options.sampled_dynamics_level = SampledDynamicsLevel::FIXED;
            }
            else if(arg == "--sample-dynamics"){
                options.sample_dynamics = true;
                if(options.sampled_dynamics_level == SampledDynamicsLevel::FIXED){
                    options.sampled_dynamics_level = SampledDynamicsLevel::BROAD;
                }
            }
            else if(arg == "--sampled-dynamics-level" && arg_i + 1 < argc){
                std::string value = argv[++arg_i];
                if(!parse_sampled_dynamics_level(value, options.sampled_dynamics_level)){
                    std::cerr << "Unknown --sampled-dynamics-level value: " << value << "\n";
                    return false;
                }
                options.sample_dynamics = options.sampled_dynamics_level != SampledDynamicsLevel::FIXED;
            }
            else if(arg == "--sampled-dynamics-curriculum-levels"){
                options.sampled_dynamics_curriculum_levels = true;
                options.sample_dynamics = true;
            }
            else if(arg == "--balanced-dynamics-sampling"){
                options.balanced_dynamics_sampling = true;
            }
            else if(arg == "--disable-balanced-dynamics-sampling"){
                options.balanced_dynamics_sampling = false;
            }
            else if(arg == "--disable-physics-gradient"){
                options.disable_physics_gradient = true;
            }
            else if(arg == "--reset-hidden-each-step"){
                options.reset_hidden_each_step = true;
            }
            else if(arg == "--horizon-curriculum"){
                options.horizon_curriculum = true;
            }
            else if(arg == "--state-curriculum"){
                options.state_curriculum = true;
            }
            else if(arg == "--dynamics-curriculum"){
                options.dynamics_curriculum = true;
            }
            else if(arg == "--horizon-stage-steps" && arg_i + 1 < argc){
                options.horizon_stage_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--state-curriculum-stage-steps" && arg_i + 1 < argc){
                options.state_curriculum_stage_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--dynamics-curriculum-stage-steps" && arg_i + 1 < argc){
                options.dynamics_curriculum_stage_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--terminal-ramp-after-horizon-change"){
                options.terminal_ramp_after_horizon_change = true;
            }
            else if(arg == "--terminal-ramp-min" && arg_i + 1 < argc){
                options.terminal_ramp_min = std::stof(argv[++arg_i]);
            }
            else if(arg == "--terminal-ramp-steps" && arg_i + 1 < argc){
                options.terminal_ramp_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--terminal-ramp-terminal-only"){
                options.terminal_ramp_terminal_only = true;
            }
            else if(arg == "--reset-optimizer-on-curriculum-transition"){
                options.reset_optimizer_on_curriculum_transition = true;
            }
            else if(arg == "--action-grad-clip" && arg_i + 1 < argc){
                options.action_grad_clip_enabled = true;
                options.action_grad_clip_norm = std::stof(argv[++arg_i]);
            }
            else if(arg == "--disable-action-grad-clip"){
                options.action_grad_clip_enabled = false;
            }
            else if(arg == "--grad-clip" && arg_i + 1 < argc){
                std::cerr << "WARNING: --grad-clip is deprecated and maps to --action-grad-clip. "
                          << "Use --actor-grad-clip for actor parameter-gradient clipping.\n";
                options.action_grad_clip_enabled = true;
                options.action_grad_clip_norm = std::stof(argv[++arg_i]);
            }
            else if(arg == "--disable-grad-clip"){
                std::cerr << "WARNING: --disable-grad-clip is deprecated and maps to --disable-action-grad-clip.\n";
                options.action_grad_clip_enabled = false;
            }
            else if(arg == "--actor-grad-clip" && arg_i + 1 < argc){
                options.actor_grad_clip_enabled = true;
                options.actor_grad_clip_norm = std::stof(argv[++arg_i]);
            }
            else if(arg == "--disable-actor-grad-clip"){
                options.actor_grad_clip_enabled = false;
            }
            else if(arg == "--actor-grad-skip-norm" && arg_i + 1 < argc){
                options.actor_grad_skip_norm = std::stof(argv[++arg_i]);
            }
            else if(arg == "--disable-action-bound"){
                options.action_bound_enabled = false;
            }
            else if(arg == "--action-bound" && arg_i + 1 < argc){
                options.action_bound_enabled = true;
                options.action_bound_value = std::stof(argv[++arg_i]);
            }
            else if(arg == "--w-v" && arg_i + 1 < argc){
                options.loss_velocity_weight = std::stof(argv[++arg_i]);
            }
            else if(arg == "--w-w" && arg_i + 1 < argc){
                options.loss_angular_velocity_weight = std::stof(argv[++arg_i]);
            }
            else if(arg == "--w-terminal-v" && arg_i + 1 < argc){
                options.terminal_velocity_weight = std::stof(argv[++arg_i]);
            }
            else if(arg == "--w-terminal-w" && arg_i + 1 < argc){
                options.terminal_angular_velocity_weight = std::stof(argv[++arg_i]);
            }
            else if(arg == "--diff-model" && arg_i + 1 < argc){
                std::string value = argv[++arg_i];
                if(value == "euler"){
                    options.diff_model = DiffModel::EULER;
                }
                else if(value == "l2f_approx"){
                    options.diff_model = DiffModel::L2F_APPROX;
                }
                else{
                    std::cerr << "Unknown --diff-model value: " << value << "\n";
                    return false;
                }
            }
            else if(arg == "--eval-model" && arg_i + 1 < argc){
                std::string value = argv[++arg_i];
                if(value == "euler"){
                    options.eval_model = EvalModel::EULER;
                }
                else if(value == "l2f"){
                    options.eval_model = EvalModel::L2F;
                }
                else{
                    std::cerr << "Unknown --eval-model value: " << value << "\n";
                    return false;
                }
            }
            else if(arg == "--save-path" && arg_i + 1 < argc){
                options.save_path = argv[++arg_i];
            }
            else if(arg == "--load-path" && arg_i + 1 < argc){
                options.load_path = argv[++arg_i];
            }
            else if(arg == "--init-actor-path" && arg_i + 1 < argc){
                options.init_actor_path = argv[++arg_i];
            }
            else if(arg == "--eval-only"){
                options.eval_only = true;
            }
            else if(arg == "--failure-analysis"){
                options.failure_analysis = true;
            }
            else if(arg == "--failure-analysis-path" && arg_i + 1 < argc){
                options.failure_analysis_path = argv[++arg_i];
            }
            else if(arg == "--force-dynamics-bins" && arg_i + 5 < argc){
                options.force_dynamics_bins = true;
                options.force_size_mass_bin = std::stoll(argv[++arg_i]);
                options.force_thrust_to_weight_bin = std::stoll(argv[++arg_i]);
                options.force_torque_to_inertia_bin = std::stoll(argv[++arg_i]);
                options.force_motor_delay_bin = std::stoll(argv[++arg_i]);
                options.force_curve_shape_bin = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--gpu-rollout"){
                options.gpu_rollout = true;
            }
            else if(arg == "--gpu-device" && arg_i + 1 < argc){
                options.gpu_device = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--gpu-batch-size" && arg_i + 1 < argc){
                options.gpu_batch_size = std::stoll(argv[++arg_i]);
                options.gpu_rollout = true;
            }
            else if(arg == "--gpu-validate-against-cpu"){
                options.gpu_validate_against_cpu = true;
                options.gpu_rollout = true;
            }
            else if(arg == "--gpu-benchmark"){
                options.gpu_benchmark = true;
                options.gpu_rollout = true;
            }
            else if(arg == "--gpu-benchmark-iterations" && arg_i + 1 < argc){
                options.gpu_benchmark_iterations = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--log-path" && arg_i + 1 < argc){
                options.log_path = argv[++arg_i];
            }
            else if(arg == "--sampler-dump-path" && arg_i + 1 < argc){
                options.sampler_dump_path = argv[++arg_i];
            }
            else if(arg == "--sampler-dump-samples" && arg_i + 1 < argc){
                options.sampler_dump_samples = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--production-objective-trace"){
                options.production_objective_trace = true;
            }
            else if(arg == "--objective-trace-path" && arg_i + 1 < argc){
                options.objective_trace_path = argv[++arg_i];
                options.production_objective_trace = true;
            }
            else if(arg == "--stage9-6-objective-parity"){
                options.stage9_6_objective_parity = true;
                options.production_objective_trace = true;
            }
            else if(arg == "--stage9-6-sampler-parity"){
                options.stage9_6_sampler_parity = true;
            }
            else if(arg == "--stage9-6-eval-parity"){
                options.stage9_6_eval_parity = true;
            }
            else if(arg == "--stage9-6-checkpoint-parity"){
                options.stage9_6_checkpoint_parity = true;
            }
            else if(arg == "--stage9-6-steps" && arg_i + 1 < argc){
                options.stage9_6_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--stage9-6-replay-path" && arg_i + 1 < argc){
                options.stage9_6_replay_path = argv[++arg_i];
            }
            else if(arg == "--save-optimizer"){
                options.save_optimizer = true;
            }
            else if(arg == "--load-optimizer"){
                options.load_optimizer = true;
            }
            else if(arg == "--checkpoint-inspect"){
                options.checkpoint_inspect = true;
                if(arg_i + 1 < argc && argv[arg_i + 1][0] != '-'){
                    options.checkpoint_inspect_path = argv[++arg_i];
                }
            }
            else if(arg == "--checkpoint-convert-old"){
                options.checkpoint_convert_old = true;
                if(arg_i + 1 < argc && argv[arg_i + 1][0] != '-'){
                    options.checkpoint_convert_old_path = argv[++arg_i];
                }
            }
            else if(arg == "--decoupled-curriculum"){
                options.decoupled_curriculum = true;
            }
            else if(arg == "--state-curriculum-lag-steps" && arg_i + 1 < argc){
                options.state_curriculum_lag_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--one-curriculum-axis-at-a-time"){
                options.one_curriculum_axis_at_a_time = true;
            }
            else if(arg == "--hold-h64-extra-steps" && arg_i + 1 < argc){
                options.hold_h64_extra_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--stability-gated-curriculum"){
                options.stability_gated_curriculum = true;
            }
            else if(arg == "--curriculum-stability-window" && arg_i + 1 < argc){
                options.curriculum_stability_window = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--curriculum-min-stage-steps" && arg_i + 1 < argc){
                options.curriculum_min_stage_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--curriculum-max-stage-steps" && arg_i + 1 < argc){
                options.curriculum_max_stage_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--curriculum-loss-spike-ratio" && arg_i + 1 < argc){
                options.curriculum_loss_spike_ratio = std::stof(argv[++arg_i]);
            }
            else if(arg == "--success-gated-curriculum"){
                options.success_gated_curriculum = true;
            }
            else if(arg == "--curriculum-success-window" && arg_i + 1 < argc){
                options.curriculum_success_window = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--curriculum-success-threshold" && arg_i + 1 < argc){
                options.curriculum_success_threshold = std::stof(argv[++arg_i]);
            }
            else if(arg == "--curriculum-gate-check-interval" && arg_i + 1 < argc){
                options.curriculum_gate_check_interval = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--curriculum-no-forced-advance"){
                options.curriculum_no_forced_advance = true;
            }
            else if(arg == "--curriculum-stage-plan" && arg_i + 1 < argc){
                options.curriculum_stage_plan = argv[++arg_i];
            }
            else if(arg == "--diagnostic-log-detail"){
                options.diagnostic_log_detail = true;
            }
            else if(arg == "--diagnostic-log-every" && arg_i + 1 < argc){
                options.diagnostic_log_every = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--diagnostic-log-first-steps" && arg_i + 1 < argc){
                options.diagnostic_log_first_steps = std::stoll(argv[++arg_i]);
            }
            else if(arg == "--fixed-state-difficulty" && arg_i + 1 < argc){
                options.fixed_state_difficulty = std::stof(argv[++arg_i]);
            }
            else if(arg == "--h128-prioritized-curriculum"){
                options.h128_prioritized_curriculum = true;
            }
            else if(arg == "--h128-schedule" && arg_i + 1 < argc){
                options.h128_schedule = argv[++arg_i];
            }
            else if(arg == "--help"){
                print_usage();
                return false;
            }
            else{
                std::cerr << "Unknown argument: " << arg << "\n";
                return false;
            }
        }
        return true;
    }
}
