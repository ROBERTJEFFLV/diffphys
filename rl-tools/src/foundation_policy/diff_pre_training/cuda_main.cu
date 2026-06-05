#include "gpu_rollout.h"

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu = rl_tools::foundation_policy::diff_pre_training::gpu;

namespace{

struct Options{
    std::size_t batch_size = 64;
    std::size_t horizon = 16;
    int gpu_device = 0;
    std::size_t gpu_batch_size = 0;
    bool gpu_rollout = false;
    bool validate_observation = false;
    bool validate_actor_forward = false;
    bool validate_critic_backward = false;
    bool validate_closed_loop = false;
    bool validate_action_gradient_injection = false;
    bool validate_actor_backward = false;
    bool validate_adam_update = false;
    bool validate_correlated_size_mass_sampler = false;
    bool validate_trajectory_sampler = false;
    bool validate_deployment_adapter = false;
    bool validate_against_cpu = false;
    bool benchmark = false;
    bool stage9_debug = false;
    bool sample_dynamics = true;
    int benchmark_iterations = 20;
    unsigned seed = 0;
    std::size_t steps = 0;
    std::size_t validation_step = 0;
    std::size_t stage9_debug_steps = 10;
    float learning_rate = 1e-4f;
    float action_magnitude_weight = 0.005f;
    float action_smoothness_weight = 0.03f;
    float action_saturation_weight = 0.05f;
    float action_saturation_start = 0.95f;
    float loss_velocity_weight = 0.8f;
    float loss_angular_velocity_weight = 0.8f;
    float terminal_loss_scale = 1.0f;
    float terminal_velocity_weight = 4.0f;
    float terminal_angular_velocity_weight = 4.0f;
    float diff_rollout_loss_weight = 5e-4f;
    bool action_grad_clip_enabled = false;
    float action_grad_clip_norm = 100.0f;
    bool actor_grad_clip_enabled = true;
    float actor_grad_clip_norm = 100.0f;
    float actor_grad_skip_norm = 1e12f;
    float actor_grad_eps = 1e-6f;
    bool disable_physics_gradient = false;
    bool reset_hidden_each_step = false;
    bool load_optimizer_state = true;
    std::string diff_model = "euler";
    std::string log_path;
    std::string save_path;
    std::string load_path;
    std::string stage9_debug_replay_path;
    bool production_objective_trace = false;
    std::string objective_trace_path;
    bool stage9_6_objective_parity = false;
    bool stage9_6_sampler_parity = false;
    bool stage9_6_eval_parity = false;
    bool stage9_6_checkpoint_parity = false;
    std::size_t stage9_6_steps = 1000;
    std::string stage9_6_replay_path;
    bool stage11 = false;
    std::size_t stage11_steps = 200000;
    float stage11_success_threshold = 0.8f;
    float stage11_velocity_threshold = 0.5f;
    bool eval_only = false;
    bool failure_analysis = false;
    std::string eval_model = "euler";
    std::size_t eval_episodes = 100;
    std::size_t eval_horizon = 128;
    float success_position_threshold = 1.0f;
    float success_velocity_threshold = 2.0f;
    float success_attitude_threshold = 3.14159265358979323846f;
    float success_angular_velocity_threshold = 5.0f;
    float success_action_saturation_threshold = 1.0f;
    bool horizon_curriculum_enabled = false;
    std::vector<std::size_t> horizon_curriculum;
    std::vector<float> curriculum_success_thresholds;
    std::vector<std::size_t> curriculum_min_steps;
    std::vector<float> curriculum_learning_rate_scales;
    std::vector<float> curriculum_diff_loss_scales;
    std::vector<float> curriculum_terminal_loss_scales;
    std::vector<float> curriculum_actor_grad_clip_norms;
    bool reset_optimizer_on_curriculum_transition = false;
    std::size_t curriculum_gate_check_interval = 1000;
    gpu::ForcedDynamicsBins forced_bins;
    bool balanced_dynamics_sampling = false;
    std::string sampled_dynamics_level = "broad";
    bool correlated_size_mass_sampling = false;
    gpu::TrajectoryMode trajectory_mode = gpu::TrajectoryMode::FIXED;
    float trajectory_amplitude = 0.03f;
    float trajectory_frequency_hz = 0.5f;
    std::string tracking_gate_mode = "recovery";
    std::size_t throughout_gate_start_step = 0;
    float initial_position_scale = 1.0f;
    float initial_velocity_scale = 1.0f;
    float initial_angular_velocity_scale = 1.0f;
    float initial_position_scale_local = 0.25f;
    float initial_velocity_scale_local = 0.5f;
    float initial_attitude_scale_local = 1.0f;
    float initial_angular_velocity_scale_local = 1.0f;
    bool save_optimizer = false;
    bool load_optimizer = false;
    bool checkpoint_inspect = false;
    std::string checkpoint_inspect_path;
    bool checkpoint_convert_old = false;
    std::string checkpoint_convert_old_path;
};

void print_usage(){
    std::cout
        << "Usage: foundation_policy_diff_pre_training_cuda [--gpu-rollout]\n"
        << "    [--batch-size N] [--gpu-batch-size N] [--horizon N] [--seed N]\n"
        << "    [--gpu-device N] [--diff-model euler]\n"
        << "    [--gpu-validate-observation] [--gpu-validate-against-cpu]\n"
        << "    [--gpu-validate-actor-forward]\n"
        << "    [--gpu-validate-critic-backward]\n"
        << "    [--gpu-validate-closed-loop]\n"
        << "    [--gpu-validate-action-gradient-injection]\n"
        << "    [--gpu-validate-actor-backward]\n"
        << "    [--gpu-validate-adam-update]\n"
        << "    [--gpu-validate-correlated-size-mass-sampler]\n"
        << "    [--gpu-validate-trajectory-sampler]\n"
        << "    [--gpu-validate-deployment-adapter] [--validation-step N]\n"
        << "    [--gpu-stage9-debug] [--stage9-debug-steps N] [--stage9-debug-replay-path PATH]\n"
        << "    [--gpu-benchmark] [--gpu-benchmark-iterations N]\n"
        << "    [--steps N] [--learning-rate LR]\n"
        << "    [--diff-rollout-loss-weight W] [--disable-physics-gradient]\n"
        << "    [--action-grad-clip VALUE] [--disable-action-grad-clip]\n"
        << "    [--actor-grad-clip VALUE] [--disable-actor-grad-clip] [--actor-grad-skip-norm VALUE]\n"
        << "    [--reset-hidden-each-step]\n"
        << "    [--production-objective-trace] [--objective-trace-path PATH]\n"
        << "    [--stage9-6-objective-parity] [--stage9-6-sampler-parity] [--stage9-6-eval-parity]\n"
        << "    [--stage9-6-checkpoint-parity] [--save-optimizer] [--load-optimizer]\n"
        << "    [--checkpoint-inspect [PATH]] [--checkpoint-convert-old [PATH]]\n"
        << "    [--stage9-6-steps N] [--stage9-6-replay-path PATH]\n"
        << "    [--stage11] [--stage11-steps N] [--stage11-success-threshold VALUE]\n"
        << "    [--stage11-velocity-threshold VALUE]\n"
        << "    [--horizon-curriculum H1,H2,...] [--curriculum-success-threshold VALUE[,VALUE...]]\n"
        << "    [--curriculum-min-steps N[,N...]] [--curriculum-gate-check-interval N]\n"
        << "    [--eval-only] [--eval-episodes N] [--eval-horizon N]\n"
        << "    [--failure-analysis] [--force-dynamics-bins size_mass=3,thrust_to_weight=3,...]\n"
        << "    [--strict-stability-thresholds]\n"
        << "    [--success-position-threshold VALUE] [--success-velocity-threshold VALUE]\n"
        << "    [--success-attitude-threshold RAD] [--success-attitude-threshold-deg DEG]\n"
        << "    [--success-angular-velocity-threshold VALUE]\n"
        << "    [--success-action-saturation-threshold VALUE]\n"
        << "    [--w-action-magnitude W] [--w-u W] [--w-sat W]\n"
        << "    [--action-saturation-start VALUE] [--terminal-loss-scale VALUE]\n"
        << "    [--curriculum-learning-rate-scale S1,S2,...]\n"
        << "    [--curriculum-diff-loss-scale S1,S2,...]\n"
        << "    [--curriculum-terminal-loss-scale S1,S2,...]\n"
        << "    [--curriculum-actor-grad-clip-norm N1,N2,...]\n"
        << "    [--reset-optimizer-on-curriculum-transition] [--no-load-optimizer]\n"
        << "    [--correlated-size-mass-sampling]\n"
        << "    [--trajectory-mode fixed|step|circle|figure8|mixed]\n"
        << "    [--trajectory-amplitude M] [--trajectory-frequency-hz F]\n"
        << "    [--tracking-gate-mode local|recovery] [--throughout-gate-start-step N]\n"
        << "    [--initial-position-scale S] [--initial-velocity-scale S] [--initial-angular-velocity-scale S]\n"
        << "    [--initial-position-scale-local S] [--initial-velocity-scale-local S]\n"
        << "    [--initial-attitude-scale-local S] [--initial-angular-velocity-scale-local S]\n"
        << "    [--log-path PATH] [--save-path PATH] [--load-path PATH]\n"
        << "\n"
        << "This CUDA target implements batched differentiable Euler rollout, physics VJP,\n"
        << "RDAC actor/GRU/critic-head BPTT, and Adam update on GPU. Stage 9.6 CPU/CUDA\n"
        << "checkpoint, sampler, and eval parity are still explicit validation gates.\n";
}

std::string trim(std::string value){
    auto not_space = [](unsigned char c){ return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::vector<std::string> split_csv(const std::string& value){
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while(std::getline(stream, item, ',')){
        result.push_back(trim(item));
    }
    return result;
}

std::vector<std::size_t> parse_size_list(const std::string& value){
    std::vector<std::size_t> result;
    for(const auto& item: split_csv(value)){
        if(!item.empty()){
            result.push_back(static_cast<std::size_t>(std::stoull(item)));
        }
    }
    return result;
}

std::vector<float> parse_float_list(const std::string& value){
    std::vector<float> result;
    for(const auto& item: split_csv(value)){
        if(!item.empty()){
            result.push_back(std::stof(item));
        }
    }
    return result;
}

gpu::TrajectoryMode parse_trajectory_mode(const std::string& value){
    if(value == "fixed") return gpu::TrajectoryMode::FIXED;
    if(value == "step") return gpu::TrajectoryMode::STEP;
    if(value == "circle") return gpu::TrajectoryMode::CIRCLE;
    if(value == "figure8" || value == "figure-eight") return gpu::TrajectoryMode::FIGURE8;
    if(value == "mixed") return gpu::TrajectoryMode::MIXED;
    throw std::runtime_error("Unknown --trajectory-mode value: " + value);
}

const char* trajectory_mode_name(gpu::TrajectoryMode mode){
    switch(mode){
        case gpu::TrajectoryMode::FIXED: return "fixed";
        case gpu::TrajectoryMode::STEP: return "step";
        case gpu::TrajectoryMode::CIRCLE: return "circle";
        case gpu::TrajectoryMode::FIGURE8: return "figure8";
        case gpu::TrajectoryMode::MIXED: return "mixed";
    }
    return "unknown";
}

std::uint32_t parse_bin_value(const std::string& value){
    const auto parsed = static_cast<unsigned long>(std::stoul(value));
    if(parsed > 3ul){
        throw std::runtime_error("Forced dynamics bin values must be in [0, 3]");
    }
    return static_cast<std::uint32_t>(parsed);
}

bool parse_force_bins_spec(const std::string& value, gpu::ForcedDynamicsBins& bins){
    gpu::ForcedDynamicsBins parsed;
    parsed.enabled = true;
    for(const auto& item: split_csv(value)){
        if(item.empty()){
            continue;
        }
        const auto equal = item.find('=');
        if(equal == std::string::npos){
            return false;
        }
        const auto key = trim(item.substr(0, equal));
        const auto bin_value = parse_bin_value(trim(item.substr(equal + 1)));
        if(key == "size_mass"){
            parsed.size_mass = bin_value;
        }
        else if(key == "thrust_to_weight"){
            parsed.thrust_to_weight = bin_value;
        }
        else if(key == "torque_to_inertia"){
            parsed.torque_to_inertia = bin_value;
        }
        else if(key == "motor_delay"){
            parsed.motor_delay = bin_value;
        }
        else if(key == "curve_shape"){
            parsed.curve_shape = bin_value;
        }
        else{
            return false;
        }
    }
    bins = parsed;
    return true;
}

bool has_next_value(int argc, char** argv, int i){
    return i + 1 < argc && argv[i + 1][0] != '-';
}

bool parse_options(int argc, char** argv, Options& options){
    for(int i = 1; i < argc; i++){
        std::string arg = argv[i];
        if(arg == "--gpu-rollout"){
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-against-cpu"){
            options.validate_against_cpu = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-observation"){
            options.validate_observation = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-actor-forward"){
            options.validate_actor_forward = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-critic-backward"){
            options.validate_critic_backward = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-closed-loop"){
            options.validate_closed_loop = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-action-gradient-injection"){
            options.validate_action_gradient_injection = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-actor-backward"){
            options.validate_actor_backward = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-adam-update"){
            options.validate_adam_update = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-correlated-size-mass-sampler"){
            options.validate_correlated_size_mass_sampler = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-trajectory-sampler"){
            options.validate_trajectory_sampler = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-validate-deployment-adapter"){
            options.validate_deployment_adapter = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--validation-step" && i + 1 < argc){
            options.validation_step = std::stoull(argv[++i]);
        }
        else if(arg == "--gpu-stage9-debug"){
            options.stage9_debug = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--stage9-debug-steps" && i + 1 < argc){
            options.stage9_debug_steps = std::stoull(argv[++i]);
            options.stage9_debug = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--stage9-debug-replay-path" && i + 1 < argc){
            options.stage9_debug_replay_path = argv[++i];
            options.stage9_debug = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--gpu-benchmark"){
            options.benchmark = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--batch-size" && i + 1 < argc){
            options.batch_size = std::stoull(argv[++i]);
        }
        else if(arg == "--gpu-batch-size" && i + 1 < argc){
            options.gpu_batch_size = std::stoull(argv[++i]);
            options.gpu_rollout = true;
        }
        else if(arg == "--horizon" && i + 1 < argc){
            options.horizon = std::stoull(argv[++i]);
        }
        else if(arg == "--seed" && i + 1 < argc){
            options.seed = static_cast<unsigned>(std::stoul(argv[++i]));
        }
        else if(arg == "--gpu-device" && i + 1 < argc){
            options.gpu_device = std::stoi(argv[++i]);
        }
        else if(arg == "--gpu-benchmark-iterations" && i + 1 < argc){
            options.benchmark_iterations = std::stoi(argv[++i]);
        }
        else if(arg == "--diff-model" && i + 1 < argc){
            options.diff_model = argv[++i];
        }
        else if(arg == "--steps" && i + 1 < argc){
            options.steps = std::stoull(argv[++i]);
        }
        else if((arg == "--learning-rate" || arg == "--lr") && i + 1 < argc){
            options.learning_rate = std::stof(argv[++i]);
        }
        else if(arg == "--diff-rollout-loss-weight" && i + 1 < argc){
            options.diff_rollout_loss_weight = std::stof(argv[++i]);
        }
        else if(arg == "--disable-physics-gradient"){
            options.disable_physics_gradient = true;
        }
        else if(arg == "--action-grad-clip"){
            options.action_grad_clip_enabled = true;
            if(has_next_value(argc, argv, i)){
                options.action_grad_clip_norm = std::stof(argv[++i]);
            }
        }
        else if(arg == "--action-grad-clip-norm" && i + 1 < argc){
            options.action_grad_clip_enabled = true;
            options.action_grad_clip_norm = std::stof(argv[++i]);
        }
        else if(arg == "--disable-action-grad-clip"){
            options.action_grad_clip_enabled = false;
        }
        else if(arg == "--actor-grad-clip"){
            options.actor_grad_clip_enabled = true;
            if(has_next_value(argc, argv, i)){
                options.actor_grad_clip_norm = std::stof(argv[++i]);
            }
        }
        else if(arg == "--actor-grad-clip-norm" && i + 1 < argc){
            options.actor_grad_clip_enabled = true;
            options.actor_grad_clip_norm = std::stof(argv[++i]);
        }
        else if(arg == "--disable-actor-grad-clip"){
            options.actor_grad_clip_enabled = false;
        }
        else if(arg == "--actor-grad-skip-norm" && i + 1 < argc){
            options.actor_grad_skip_norm = std::stof(argv[++i]);
        }
        else if(arg == "--actor-grad-eps" && i + 1 < argc){
            options.actor_grad_eps = std::stof(argv[++i]);
        }
        else if(arg == "--reset-hidden-each-step"){
            options.reset_hidden_each_step = true;
        }
        else if(arg == "--no-load-optimizer" || arg == "--reset-optimizer-on-load"){
            options.load_optimizer_state = false;
        }
        else if(arg == "--log-path" && i + 1 < argc){
            options.log_path = argv[++i];
        }
        else if(arg == "--fixed-dynamics"){
            options.sample_dynamics = false;
        }
        else if(arg == "--sample-dynamics"){
            options.sample_dynamics = true;
        }
        else if(arg == "--balanced-dynamics-sampling"){
            options.balanced_dynamics_sampling = true;
        }
        else if(arg == "--correlated-size-mass-sampling"){
            options.correlated_size_mass_sampling = true;
            options.sample_dynamics = true;
        }
        else if(arg == "--sampled-dynamics-level" && i + 1 < argc){
            options.sampled_dynamics_level = argv[++i];
        }
        else if(arg == "--trajectory-mode" && i + 1 < argc){
            options.trajectory_mode = parse_trajectory_mode(argv[++i]);
        }
        else if(arg == "--trajectory-amplitude" && i + 1 < argc){
            options.trajectory_amplitude = std::stof(argv[++i]);
        }
        else if(arg == "--trajectory-frequency-hz" && i + 1 < argc){
            options.trajectory_frequency_hz = std::stof(argv[++i]);
        }
        else if(arg == "--tracking-gate-mode" && i + 1 < argc){
            options.tracking_gate_mode = argv[++i];
            if(options.tracking_gate_mode != "local" && options.tracking_gate_mode != "recovery"){
                throw std::runtime_error("Unknown --tracking-gate-mode value: " + options.tracking_gate_mode);
            }
        }
        else if(arg == "--throughout-gate-start-step" && i + 1 < argc){
            options.throughout_gate_start_step = std::stoull(argv[++i]);
        }
        else if(arg == "--initial-position-scale" && i + 1 < argc){
            options.initial_position_scale = std::stof(argv[++i]);
        }
        else if(arg == "--initial-velocity-scale" && i + 1 < argc){
            options.initial_velocity_scale = std::stof(argv[++i]);
        }
        else if(arg == "--initial-angular-velocity-scale" && i + 1 < argc){
            options.initial_angular_velocity_scale = std::stof(argv[++i]);
        }
        else if(arg == "--initial-position-scale-local" && i + 1 < argc){
            options.initial_position_scale_local = std::stof(argv[++i]);
        }
        else if(arg == "--initial-velocity-scale-local" && i + 1 < argc){
            options.initial_velocity_scale_local = std::stof(argv[++i]);
        }
        else if(arg == "--initial-attitude-scale-local" && i + 1 < argc){
            options.initial_attitude_scale_local = std::stof(argv[++i]);
        }
        else if(arg == "--initial-angular-velocity-scale-local" && i + 1 < argc){
            options.initial_angular_velocity_scale_local = std::stof(argv[++i]);
        }
        else if(arg == "--w-v" && i + 1 < argc){
            options.loss_velocity_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-action-magnitude" && i + 1 < argc){
            options.action_magnitude_weight = std::stof(argv[++i]);
        }
        else if((arg == "--w-u" || arg == "--w-action-smoothness") && i + 1 < argc){
            options.action_smoothness_weight = std::stof(argv[++i]);
        }
        else if((arg == "--w-sat" || arg == "--w-saturation") && i + 1 < argc){
            options.action_saturation_weight = std::stof(argv[++i]);
        }
        else if((arg == "--action-saturation-start" || arg == "--u-safe") && i + 1 < argc){
            options.action_saturation_start = std::stof(argv[++i]);
        }
        else if(arg == "--terminal-loss-scale" && i + 1 < argc){
            options.terminal_loss_scale = std::stof(argv[++i]);
        }
        else if(arg == "--w-terminal-v" && i + 1 < argc){
            options.terminal_velocity_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-angular-v" && i + 1 < argc){
            options.loss_angular_velocity_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-terminal-angular-v" && i + 1 < argc){
            options.terminal_angular_velocity_weight = std::stof(argv[++i]);
        }
        else if(arg == "--save-path" && i + 1 < argc){
            options.save_path = argv[++i];
        }
        else if(arg == "--load-path" && i + 1 < argc){
            options.load_path = argv[++i];
        }
        else if(arg == "--production-objective-trace"){
            options.production_objective_trace = true;
        }
        else if(arg == "--objective-trace-path" && i + 1 < argc){
            options.objective_trace_path = argv[++i];
            options.production_objective_trace = true;
        }
        else if(arg == "--stage9-6-objective-parity"){
            options.stage9_6_objective_parity = true;
            options.production_objective_trace = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--stage9-6-sampler-parity"){
            options.stage9_6_sampler_parity = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--stage9-6-eval-parity"){
            options.stage9_6_eval_parity = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--stage9-6-checkpoint-parity"){
            options.stage9_6_checkpoint_parity = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--stage9-6-steps" && i + 1 < argc){
            options.stage9_6_steps = std::stoull(argv[++i]);
        }
        else if(arg == "--stage9-6-replay-path" && i + 1 < argc){
            options.stage9_6_replay_path = argv[++i];
        }
        else if(arg == "--stage11" || arg == "--stage-11"){
            options.stage11 = true;
            options.gpu_rollout = true;
            options.sample_dynamics = true;
            options.balanced_dynamics_sampling = true;
            options.sampled_dynamics_level = "broad";
        }
        else if((arg == "--stage11-steps" || arg == "--stage-11-steps") && i + 1 < argc){
            options.stage11_steps = std::stoull(argv[++i]);
            options.stage11 = true;
            options.gpu_rollout = true;
        }
        else if((arg == "--stage11-success-threshold" || arg == "--success-threshold") && i + 1 < argc){
            options.stage11_success_threshold = std::stof(argv[++i]);
            options.stage11 = true;
            options.gpu_rollout = true;
        }
        else if((arg == "--stage11-velocity-threshold" || arg == "--velocity-threshold") && i + 1 < argc){
            options.stage11_velocity_threshold = std::stof(argv[++i]);
            options.stage11 = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--eval-only"){
            options.eval_only = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--eval-model" && i + 1 < argc){
            options.eval_model = argv[++i];
            options.gpu_rollout = true;
        }
        else if(arg == "--failure-analysis"){
            options.failure_analysis = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--eval-episodes" && i + 1 < argc){
            options.eval_episodes = std::stoull(argv[++i]);
            options.gpu_rollout = true;
        }
        else if(arg == "--eval-horizon" && i + 1 < argc){
            options.eval_horizon = std::stoull(argv[++i]);
            options.gpu_rollout = true;
        }
        else if(arg == "--strict-stability-thresholds"){
            options.success_position_threshold = 0.05f;
            options.success_velocity_threshold = 0.1f;
            options.success_attitude_threshold = 5.0f * 3.14159265358979323846f / 180.0f;
            options.success_angular_velocity_threshold = 0.2f;
            options.success_action_saturation_threshold = 0.01f;
        }
        else if(arg == "--success-position-threshold" && i + 1 < argc){
            options.success_position_threshold = std::stof(argv[++i]);
        }
        else if(arg == "--success-velocity-threshold" && i + 1 < argc){
            options.success_velocity_threshold = std::stof(argv[++i]);
        }
        else if(arg == "--success-attitude-threshold" && i + 1 < argc){
            options.success_attitude_threshold = std::stof(argv[++i]);
        }
        else if(arg == "--success-attitude-threshold-deg" && i + 1 < argc){
            options.success_attitude_threshold = std::stof(argv[++i]) * 3.14159265358979323846f / 180.0f;
        }
        else if(arg == "--success-angular-velocity-threshold" && i + 1 < argc){
            options.success_angular_velocity_threshold = std::stof(argv[++i]);
        }
        else if(arg == "--success-action-saturation-threshold" && i + 1 < argc){
            options.success_action_saturation_threshold = std::stof(argv[++i]);
        }
        else if(arg == "--horizon-curriculum"){
            options.horizon_curriculum_enabled = true;
            options.gpu_rollout = true;
            if(has_next_value(argc, argv, i)){
                options.horizon_curriculum = parse_size_list(argv[++i]);
            }
        }
        else if(arg == "--curriculum-success-threshold" && i + 1 < argc){
            options.curriculum_success_thresholds = parse_float_list(argv[++i]);
        }
        else if(arg == "--curriculum-min-steps" && i + 1 < argc){
            options.curriculum_min_steps = parse_size_list(argv[++i]);
        }
        else if(arg == "--curriculum-learning-rate-scale" && i + 1 < argc){
            options.curriculum_learning_rate_scales = parse_float_list(argv[++i]);
        }
        else if(arg == "--curriculum-diff-loss-scale" && i + 1 < argc){
            options.curriculum_diff_loss_scales = parse_float_list(argv[++i]);
        }
        else if(arg == "--curriculum-terminal-loss-scale" && i + 1 < argc){
            options.curriculum_terminal_loss_scales = parse_float_list(argv[++i]);
        }
        else if(arg == "--curriculum-actor-grad-clip-norm" && i + 1 < argc){
            options.curriculum_actor_grad_clip_norms = parse_float_list(argv[++i]);
            options.actor_grad_clip_enabled = true;
        }
        else if(arg == "--reset-optimizer-on-curriculum-transition"){
            options.reset_optimizer_on_curriculum_transition = true;
        }
        else if(arg == "--curriculum-gate-check-interval" && i + 1 < argc){
            options.curriculum_gate_check_interval = std::stoull(argv[++i]);
        }
        else if(arg == "--force-dynamics-bins" && i + 1 < argc){
            if(!parse_force_bins_spec(argv[++i], options.forced_bins)){
                std::cerr << "Invalid --force-dynamics-bins specification.\n";
                return false;
            }
            options.failure_analysis = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--save-optimizer"){
            options.save_optimizer = true;
        }
        else if(arg == "--load-optimizer"){
            options.load_optimizer = true;
            options.load_optimizer_state = true;
        }
        else if(arg == "--checkpoint-inspect"){
            options.checkpoint_inspect = true;
            if(i + 1 < argc && argv[i + 1][0] != '-'){
                options.checkpoint_inspect_path = argv[++i];
            }
        }
        else if(arg == "--checkpoint-convert-old"){
            options.checkpoint_convert_old = true;
            if(i + 1 < argc && argv[i + 1][0] != '-'){
                options.checkpoint_convert_old_path = argv[++i];
            }
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

gpu::FullGpuTrainingOptions make_full_training_options(const Options& options){
    gpu::FullGpuTrainingOptions training_options;
    training_options.device = options.gpu_device;
    training_options.batch_size = options.batch_size;
    training_options.horizon = options.horizon;
    training_options.steps = options.steps;
    training_options.seed = options.seed;
    training_options.learning_rate = options.learning_rate;
    training_options.diff_rollout_loss_weight = options.diff_rollout_loss_weight;
    training_options.action_grad_clip_enabled = options.action_grad_clip_enabled;
    training_options.action_grad_clip_norm = options.action_grad_clip_norm;
    training_options.actor_grad_clip_enabled = options.actor_grad_clip_enabled;
    training_options.actor_grad_clip_norm = options.actor_grad_clip_norm;
    training_options.actor_grad_skip_norm = options.actor_grad_skip_norm;
    training_options.actor_grad_eps = options.actor_grad_eps;
    training_options.disable_physics_gradient = options.disable_physics_gradient;
    training_options.reset_hidden_each_step = options.reset_hidden_each_step;
    training_options.sample_dynamics = options.sample_dynamics;
    training_options.load_optimizer_state = options.load_optimizer_state;
    training_options.correlated_size_mass_sampling = options.correlated_size_mass_sampling;
    training_options.trajectory_mode = options.trajectory_mode;
    training_options.trajectory_amplitude = options.trajectory_amplitude;
    training_options.trajectory_frequency_hz = options.trajectory_frequency_hz;
    training_options.initial_position_scale = options.initial_position_scale;
    training_options.initial_velocity_scale = options.initial_velocity_scale;
    training_options.initial_angular_velocity_scale = options.initial_angular_velocity_scale;
    training_options.success_position_threshold = options.success_position_threshold;
    training_options.success_velocity_threshold = options.success_velocity_threshold;
    training_options.success_attitude_threshold = options.success_attitude_threshold;
    training_options.success_angular_velocity_threshold = options.success_angular_velocity_threshold;
    training_options.success_action_saturation_threshold = options.success_action_saturation_threshold;
    training_options.log_path = options.log_path;
    training_options.save_path = options.save_path;
    training_options.load_path = options.load_path;
    return training_options;
}

gpu::GpuPolicyEvalOptions make_gpu_eval_options(const Options& options){
    gpu::GpuPolicyEvalOptions eval_options;
    eval_options.device = options.gpu_device;
    eval_options.episodes = options.eval_episodes;
    eval_options.horizon = options.eval_horizon;
    eval_options.seed = options.seed;
    eval_options.sample_dynamics = options.sample_dynamics;
    eval_options.reset_hidden_each_step = options.reset_hidden_each_step;
    eval_options.correlated_size_mass_sampling = options.correlated_size_mass_sampling;
    eval_options.trajectory_mode = options.trajectory_mode;
    eval_options.trajectory_amplitude = options.trajectory_amplitude;
    eval_options.trajectory_frequency_hz = options.trajectory_frequency_hz;
    eval_options.initial_position_scale = options.initial_position_scale;
    eval_options.initial_velocity_scale = options.initial_velocity_scale;
    eval_options.initial_angular_velocity_scale = options.initial_angular_velocity_scale;
    eval_options.success_position_threshold = options.success_position_threshold;
    eval_options.success_velocity_threshold = options.success_velocity_threshold;
    eval_options.success_attitude_threshold = options.success_attitude_threshold;
    eval_options.success_angular_velocity_threshold = options.success_angular_velocity_threshold;
    eval_options.success_action_saturation_threshold = options.success_action_saturation_threshold;
    eval_options.throughout_gate_start_step = options.throughout_gate_start_step;
    eval_options.forced_bins = options.forced_bins;
    eval_options.load_path = options.load_path;
    eval_options.log_path = options.log_path;
    return eval_options;
}

std::string stage11_phase_suffix(std::size_t horizon, const std::string& suffix){
    return std::string("_h") + std::to_string(horizon) + suffix;
}

void print_validation(const gpu::ValidationSummary& summary){
    std::cout << "gpu_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "forward_close=" << (summary.forward_close ? "true" : "false") << "\n";
    std::cout << "loss_close=" << (summary.loss_close ? "true" : "false") << "\n";
    std::cout << "action_gradient_close=" << (summary.action_gradient_close ? "true" : "false") << "\n";
    std::cout << "max_forward_abs_error=" << summary.max_forward_abs_error << "\n";
    std::cout << "max_rpm_abs_error=" << summary.max_rpm_abs_error << "\n";
    std::cout << "max_rpm_rel_error=" << summary.max_rpm_rel_error << "\n";
    std::cout << "max_loss_abs_error=" << summary.max_loss_abs_error << "\n";
    std::cout << "max_loss_rel_error=" << summary.max_loss_rel_error << "\n";
    std::cout << "max_action_gradient_abs_error=" << summary.max_action_gradient_abs_error << "\n";
    std::cout << "max_action_gradient_rel_error=" << summary.max_action_gradient_rel_error << "\n";
    std::cout << "action_gradient_l2_rel_error=" << summary.action_gradient_l2_rel_error << "\n";
}

void print_observation_validation(const gpu::ObservationValidationSummary& summary){
    std::cout << "gpu_observation_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "observation_max_abs_error=" << summary.max_abs_error << "\n";
    std::cout << "observation_mean_abs_error=" << summary.mean_abs_error << "\n";
    std::cout << "observation_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_deployment_adapter_validation(const gpu::DeploymentAdapterValidationSummary& summary){
    std::cout << "deployment_adapter_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "deployment_adapter_deterministic=" << (summary.deterministic ? "true" : "false") << "\n";
    std::cout << "deployment_adapter_relative_offsets_close=" << (summary.relative_offsets_close ? "true" : "false") << "\n";
    std::cout << "deployment_adapter_max_abs_error=" << summary.max_abs_error << "\n";
    std::cout << "deployment_adapter_mean_abs_error=" << summary.mean_abs_error << "\n";
    std::cout << "deployment_adapter_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_actor_forward_validation(const gpu::ActorForwardValidationSummary& summary){
    std::cout << "gpu_actor_forward_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "raw_action_close=" << (summary.raw_action_close ? "true" : "false") << "\n";
    std::cout << "bounded_action_close=" << (summary.bounded_action_close ? "true" : "false") << "\n";
    std::cout << "hidden_close=" << (summary.hidden_close ? "true" : "false") << "\n";
    std::cout << "critic_output_close=" << (summary.critic_output_close ? "true" : "false") << "\n";
    std::cout << "critic_forward_checked=" << (summary.critic_checked ? "true" : "false") << "\n";
    std::cout << "actor_forward_max_raw_action_abs_error=" << summary.max_raw_action_abs_error << "\n";
    std::cout << "actor_forward_max_bounded_action_abs_error=" << summary.max_bounded_action_abs_error << "\n";
    std::cout << "actor_forward_max_hidden_abs_error=" << summary.max_hidden_abs_error << "\n";
    std::cout << "critic_forward_max_output_abs_error=" << summary.max_critic_output_abs_error << "\n";
    std::cout << "actor_forward_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_closed_loop_validation(const gpu::ClosedLoopValidationSummary& summary){
    std::cout << "gpu_closed_loop_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "closed_loop_state_close=" << (summary.state_close ? "true" : "false") << "\n";
    std::cout << "closed_loop_action_close=" << (summary.action_close ? "true" : "false") << "\n";
    std::cout << "closed_loop_hidden_close=" << (summary.hidden_close ? "true" : "false") << "\n";
    std::cout << "closed_loop_loss_close=" << (summary.loss_close ? "true" : "false") << "\n";
    std::cout << "closed_loop_max_state_abs_error=" << summary.max_state_abs_error << "\n";
    std::cout << "closed_loop_max_rpm_abs_error=" << summary.max_rpm_abs_error << "\n";
    std::cout << "closed_loop_max_rpm_rel_error=" << summary.max_rpm_rel_error << "\n";
    std::cout << "closed_loop_max_action_abs_error=" << summary.max_action_abs_error << "\n";
    std::cout << "closed_loop_max_hidden_abs_error=" << summary.max_hidden_abs_error << "\n";
    std::cout << "closed_loop_max_loss_abs_error=" << summary.max_loss_abs_error << "\n";
    std::cout << "closed_loop_max_loss_rel_error=" << summary.max_loss_rel_error << "\n";
    std::cout << "closed_loop_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_action_gradient_injection_validation(const gpu::ActionGradientInjectionValidationSummary& summary){
    std::cout << "gpu_action_gradient_injection_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "action_gradient_close=" << (summary.action_gradient_close ? "true" : "false") << "\n";
    std::cout << "raw_action_gradient_close=" << (summary.raw_action_gradient_close ? "true" : "false") << "\n";
    std::cout << "action_derivative_close=" << (summary.derivative_close ? "true" : "false") << "\n";
    std::cout << "clamp_zero_ok=" << (summary.clamp_zero_ok ? "true" : "false") << "\n";
    std::cout << "action_gradient_max_abs_error=" << summary.max_action_gradient_abs_error << "\n";
    std::cout << "action_gradient_max_rel_error=" << summary.max_action_gradient_rel_error << "\n";
    std::cout << "action_gradient_l2_rel_error=" << summary.action_gradient_l2_rel_error << "\n";
    std::cout << "raw_action_gradient_max_abs_error=" << summary.max_raw_action_gradient_abs_error << "\n";
    std::cout << "raw_action_gradient_max_rel_error=" << summary.max_raw_action_gradient_rel_error << "\n";
    std::cout << "raw_action_gradient_l2_rel_error=" << summary.raw_action_gradient_l2_rel_error << "\n";
    std::cout << "action_derivative_max_abs_error=" << summary.max_action_derivative_abs_error << "\n";
    std::cout << "clamped_action_count=" << summary.clamped_action_count << "\n";
    std::cout << "clamp_zero_violation_count=" << summary.clamp_zero_violation_count << "\n";
    std::cout << "action_gradient_injection_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_actor_backward_validation(const gpu::ActorBackwardValidationSummary& summary){
    std::cout << "gpu_actor_backward_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "actor_backward_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "actor_backward_nonzero=" << (summary.nonzero ? "true" : "false") << "\n";
    std::cout << "actor_backward_cosine_close=" << (summary.cosine_close ? "true" : "false") << "\n";
    std::cout << "encoder_grad_norm=" << summary.encoder_grad_norm << "\n";
    std::cout << "gru_input_grad_norm=" << summary.gru_input_grad_norm << "\n";
    std::cout << "gru_hidden_grad_norm=" << summary.gru_hidden_grad_norm << "\n";
    std::cout << "actor_head_grad_norm=" << summary.actor_head_grad_norm << "\n";
    std::cout << "h0_grad_norm=" << summary.h0_grad_norm << "\n";
    std::cout << "encoder_grad_cosine=" << summary.encoder_grad_cosine << "\n";
    std::cout << "gru_input_grad_cosine=" << summary.gru_input_grad_cosine << "\n";
    std::cout << "gru_hidden_grad_cosine=" << summary.gru_hidden_grad_cosine << "\n";
    std::cout << "actor_head_grad_cosine=" << summary.actor_head_grad_cosine << "\n";
    std::cout << "h0_grad_cosine=" << summary.h0_grad_cosine << "\n";
    std::cout << "actor_backward_max_abs_error=" << summary.max_abs_error << "\n";
    std::cout << "actor_backward_l2_rel_error=" << summary.l2_rel_error << "\n";
    std::cout << "actor_backward_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_critic_backward_validation(const gpu::CriticBackwardValidationSummary& summary){
    std::cout << "gpu_critic_backward_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "critic_backward_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "critic_backward_nonzero=" << (summary.nonzero ? "true" : "false") << "\n";
    std::cout << "critic_backward_cosine_close=" << (summary.cosine_close ? "true" : "false") << "\n";
    std::cout << "critic_backward_actor_head_zero=" << (summary.actor_head_zero ? "true" : "false") << "\n";
    std::cout << "critic_encoder_grad_norm=" << summary.encoder_grad_norm << "\n";
    std::cout << "critic_gru_input_grad_norm=" << summary.gru_input_grad_norm << "\n";
    std::cout << "critic_gru_hidden_grad_norm=" << summary.gru_hidden_grad_norm << "\n";
    std::cout << "critic_head_grad_norm=" << summary.critic_head_grad_norm << "\n";
    std::cout << "critic_actor_head_grad_norm=" << summary.actor_head_grad_norm << "\n";
    std::cout << "critic_h0_grad_norm=" << summary.h0_grad_norm << "\n";
    std::cout << "critic_encoder_grad_cosine=" << summary.encoder_grad_cosine << "\n";
    std::cout << "critic_gru_input_grad_cosine=" << summary.gru_input_grad_cosine << "\n";
    std::cout << "critic_gru_hidden_grad_cosine=" << summary.gru_hidden_grad_cosine << "\n";
    std::cout << "critic_head_grad_cosine=" << summary.critic_head_grad_cosine << "\n";
    std::cout << "critic_h0_grad_cosine=" << summary.h0_grad_cosine << "\n";
    std::cout << "critic_backward_max_abs_error=" << summary.max_abs_error << "\n";
    std::cout << "critic_backward_l2_rel_error=" << summary.l2_rel_error << "\n";
    std::cout << "critic_backward_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_adam_update_validation(const gpu::AdamUpdateValidationSummary& summary){
    std::cout << "gpu_adam_update_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "adam_weights_close=" << (summary.weights_close ? "true" : "false") << "\n";
    std::cout << "adam_moments_close=" << (summary.moments_close ? "true" : "false") << "\n";
    std::cout << "adam_step_close=" << (summary.step_close ? "true" : "false") << "\n";
    std::cout << "adam_max_weight_abs_error=" << summary.max_weight_abs_error << "\n";
    std::cout << "adam_max_weight_rel_error=" << summary.max_weight_rel_error << "\n";
    std::cout << "adam_max_first_moment_abs_error=" << summary.max_first_moment_abs_error << "\n";
    std::cout << "adam_max_first_moment_rel_error=" << summary.max_first_moment_rel_error << "\n";
    std::cout << "adam_max_second_moment_abs_error=" << summary.max_second_moment_abs_error << "\n";
    std::cout << "adam_max_second_moment_rel_error=" << summary.max_second_moment_rel_error << "\n";
    std::cout << "adam_update_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_full_gpu_training_summary(const gpu::FullGpuTrainingSummary& summary){
    std::cout << "gpu_full_training_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "gpu_full_training_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "gpu_full_training_final_loss=" << summary.final_loss << "\n";
    std::cout << "gpu_full_training_final_grad_norm=" << summary.final_grad_norm << "\n";
    std::cout << "gpu_full_training_final_critic_loss=" << summary.final_critic_loss << "\n";
    std::cout << "gpu_full_training_final_critic_error_norm=" << summary.final_critic_error_norm << "\n";
    std::cout << "gpu_full_training_final_critic_output_mean=" << summary.final_critic_output_mean << "\n";
    std::cout << "gpu_full_training_final_critic_target_mean=" << summary.final_critic_target_mean << "\n";
    std::cout << "gpu_full_training_final_success_rate=" << summary.final_success_rate << "\n";
    std::cout << "gpu_full_training_final_action_saturation=" << summary.final_action_saturation << "\n";
    std::cout << "gpu_full_training_final_position_norm_mean=" << summary.final_position_norm_mean << "\n";
    std::cout << "gpu_full_training_final_velocity_norm_mean=" << summary.final_velocity_norm_mean << "\n";
    std::cout << "gpu_full_training_final_attitude_error_mean=" << summary.final_attitude_error_mean << "\n";
    std::cout << "gpu_full_training_final_angular_velocity_norm_mean=" << summary.final_angular_velocity_norm_mean << "\n";
    std::cout << "gpu_full_training_nan_inf_count=" << summary.nan_inf_count << "\n";
    std::cout << "gpu_checkpoint_saved=" << (summary.checkpoint_saved ? "true" : "false") << "\n";
    std::cout << "gpu_checkpoint_loaded=" << (summary.checkpoint_loaded ? "true" : "false") << "\n";
}

void print_gpu_eval_summary(const gpu::GpuPolicyEvalSummary& summary){
    std::cout << "gpu_eval_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "gpu_eval_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "gpu_eval_checkpoint_loaded=" << (summary.checkpoint_loaded ? "true" : "false") << "\n";
    std::cout << "gpu_eval_success_rate=" << summary.success_rate << "\n";
    std::cout << "gpu_eval_near_success_rate_p=" << summary.near_success_rate_p << "\n";
    std::cout << "gpu_eval_near_success_rate_pv=" << summary.near_success_rate_pv << "\n";
    std::cout << "gpu_eval_mean_total_loss=" << summary.mean_total_loss << "\n";
    std::cout << "gpu_eval_mean_final_position_norm=" << summary.mean_final_position_norm << "\n";
    std::cout << "gpu_eval_mean_final_velocity_norm=" << summary.mean_final_velocity_norm << "\n";
    std::cout << "gpu_eval_mean_final_attitude_error=" << summary.mean_final_attitude_error << "\n";
    std::cout << "gpu_eval_mean_final_angular_velocity_norm=" << summary.mean_final_angular_velocity_norm << "\n";
    std::cout << "gpu_eval_position_rmse=" << summary.position_rmse << "\n";
    std::cout << "gpu_eval_velocity_rmse=" << summary.velocity_rmse << "\n";
    std::cout << "gpu_eval_attitude_rmse=" << summary.attitude_rmse << "\n";
    std::cout << "gpu_eval_angular_velocity_rmse=" << summary.angular_velocity_rmse << "\n";
    std::cout << "gpu_eval_p90_final_position_norm=" << summary.p90_final_position_norm << "\n";
    std::cout << "gpu_eval_p90_final_velocity_norm=" << summary.p90_final_velocity_norm << "\n";
    std::cout << "gpu_eval_p90_final_attitude_error=" << summary.p90_final_attitude_error << "\n";
    std::cout << "gpu_eval_p90_final_angular_velocity_norm=" << summary.p90_final_angular_velocity_norm << "\n";
    std::cout << "gpu_eval_throughout_success_rate=" << summary.throughout_success_rate << "\n";
    std::cout << "gpu_eval_post_burnin_throughout_success_rate=" << summary.post_burnin_throughout_success_rate << "\n";
    std::cout << "gpu_eval_mean_time_inside_fraction=" << summary.mean_time_inside_fraction << "\n";
    std::cout << "gpu_eval_mean_first_failure_time_s=" << summary.mean_first_failure_time_s << "\n";
    std::cout << "gpu_eval_mean_max_position_norm=" << summary.mean_max_position_norm << "\n";
    std::cout << "gpu_eval_mean_max_velocity_norm=" << summary.mean_max_velocity_norm << "\n";
    std::cout << "gpu_eval_mean_max_attitude_error=" << summary.mean_max_attitude_error << "\n";
    std::cout << "gpu_eval_mean_max_angular_velocity_norm=" << summary.mean_max_angular_velocity_norm << "\n";
    std::cout << "gpu_eval_p90_max_position_norm=" << summary.p90_max_position_norm << "\n";
    std::cout << "gpu_eval_p90_max_velocity_norm=" << summary.p90_max_velocity_norm << "\n";
    std::cout << "gpu_eval_p90_max_attitude_error=" << summary.p90_max_attitude_error << "\n";
    std::cout << "gpu_eval_p90_max_angular_velocity_norm=" << summary.p90_max_angular_velocity_norm << "\n";
    std::cout << "gpu_eval_max_action_abs=" << summary.max_action_abs << "\n";
    std::cout << "gpu_eval_action_saturation_rate=" << summary.action_saturation_rate << "\n";
    std::cout << "gpu_eval_stability_gate_passed=" << (summary.stability_gate_passed ? "true" : "false") << "\n";
    std::cout << "gpu_eval_invalid_or_nan_rate=" << summary.invalid_or_nan_rate << "\n";
    std::cout << "gpu_eval_nan_inf_count=" << summary.nan_inf_count << "\n";
}

float curriculum_threshold_for_stage(const Options& options, std::size_t stage){
    if(options.curriculum_success_thresholds.empty()){
        return options.stage11_success_threshold;
    }
    if(options.curriculum_success_thresholds.size() == 1){
        return options.curriculum_success_thresholds[0];
    }
    return options.curriculum_success_thresholds[std::min(stage, options.curriculum_success_thresholds.size() - 1)];
}

std::size_t curriculum_min_steps_for_stage(const Options& options, std::size_t stage){
    if(options.curriculum_min_steps.empty()){
        return 0;
    }
    return options.curriculum_min_steps[std::min(stage, options.curriculum_min_steps.size() - 1)];
}

float stage_scale_or_default(const std::vector<float>& values, std::size_t stage, float fallback){
    if(values.empty()){
        return fallback;
    }
    return values[std::min(stage, values.size() - 1)];
}

std::string curriculum_chunk_log_path(const std::string& summary_log_path, std::size_t chunk_index){
    const std::string prefix = summary_log_path.empty()
        ? std::string("/tmp/gpu_horizon_curriculum")
        : summary_log_path;
    return prefix + ".chunk_" + std::to_string(chunk_index) + ".csv";
}

std::string curriculum_chunk_checkpoint_path(const std::string& checkpoint_path, std::size_t chunk_index, std::size_t horizon){
    const std::string prefix = checkpoint_path.empty()
        ? std::string("/tmp/gpu_horizon_curriculum_final.ckpt")
        : checkpoint_path;
    return prefix + ".chunk_" + std::to_string(chunk_index) + "_h" + std::to_string(horizon) + ".ckpt";
}

bool copy_file(const std::string& src, const std::string& dst){
    std::ifstream in(src, std::ios::binary);
    if(!in){
        return false;
    }
    std::ofstream out(dst, std::ios::binary);
    if(!out){
        return false;
    }
    out << in.rdbuf();
    return static_cast<bool>(out);
}

float read_chunk_mean_success_rate(const std::string& chunk_log_path, float fallback){
    std::ifstream in(chunk_log_path);
    if(!in){
        return fallback;
    }
    std::string line;
    if(!std::getline(in, line)){
        return fallback;
    }
    double sum = 0.0;
    std::size_t count = 0;
    while(std::getline(in, line)){
        const auto columns = split_csv(line);
        if(columns.size() <= 12 || columns[12].empty()){
            continue;
        }
        sum += std::stod(columns[12]);
        count++;
    }
    return count > 0 ? static_cast<float>(sum / static_cast<double>(count)) : fallback;
}

bool run_horizon_curriculum(const Options& options, const gpu::EulerGpuLossWeights& weights){
    if(options.horizon_curriculum.empty()){
        throw std::runtime_error("Horizon curriculum was enabled but no horizons were provided");
    }
    if(options.steps == 0){
        throw std::runtime_error("Horizon curriculum requires --steps > 0");
    }
    if(options.curriculum_gate_check_interval == 0){
        throw std::runtime_error("Horizon curriculum requires --curriculum-gate-check-interval > 0");
    }
    const std::string summary_log_path = options.log_path.empty()
        ? std::string("/tmp/gpu_horizon_curriculum_summary.csv")
        : options.log_path;
    const std::string checkpoint_path = options.save_path.empty()
        ? std::string("/tmp/gpu_horizon_curriculum_final.ckpt")
        : options.save_path;
    std::ofstream summary_log(summary_log_path);
    if(!summary_log){
        throw std::runtime_error("Failed to open horizon curriculum summary log: " + summary_log_path);
    }
    summary_log
        << "global_step,chunk_index,horizon,chunk_steps,stage_steps,success_threshold,min_stage_steps,"
        << "learning_rate,diff_rollout_loss_weight,terminal_loss_scale,actor_grad_clip_norm,"
        << "optimizer_state_loaded,"
        << "loss,grad_norm,critic_loss,critic_error_norm,success_rate_batch,action_saturation_rate,"
        << "chunk_mean_success_rate,"
        << "final_position_norm_mean,final_velocity_norm_mean,final_angular_velocity_norm_mean,"
        << "finite,checkpoint_saved,checkpoint_loaded,latest_checkpoint_copied,transitioned,checkpoint_path,latest_checkpoint_path,chunk_log_path\n";

    std::cout << "horizon_curriculum_enabled=true\n";
    std::cout << "horizon_curriculum_sequence=";
    for(std::size_t i = 0; i < options.horizon_curriculum.size(); i++){
        std::cout << (i == 0 ? "" : ",") << options.horizon_curriculum[i];
    }
    std::cout << "\n";
    std::cout << "horizon_curriculum_summary_log_path=" << summary_log_path << "\n";
    std::cout << "horizon_curriculum_checkpoint_path=" << checkpoint_path << "\n";

    std::string previous_checkpoint = options.load_path;
    std::size_t stage = 0;
    std::size_t stage_steps = 0;
    std::size_t completed_steps = 0;
    std::size_t chunk_index = 0;
    bool previous_chunk_transitioned = false;
    bool passed = true;
    while(completed_steps < options.steps){
        const std::size_t remaining = options.steps - completed_steps;
        const std::size_t chunk_steps = std::min(options.curriculum_gate_check_interval, remaining);
        const std::size_t horizon = options.horizon_curriculum[stage];
        const float threshold = curriculum_threshold_for_stage(options, stage);
        const std::size_t min_stage_steps = curriculum_min_steps_for_stage(options, stage);
        const std::string chunk_log = curriculum_chunk_log_path(summary_log_path, chunk_index);
        const std::string chunk_checkpoint = curriculum_chunk_checkpoint_path(checkpoint_path, chunk_index, horizon);
        const float lr_scale = stage_scale_or_default(options.curriculum_learning_rate_scales, stage, 1.0f);
        const float diff_scale = stage_scale_or_default(options.curriculum_diff_loss_scales, stage, 1.0f);
        const float terminal_scale = stage_scale_or_default(options.curriculum_terminal_loss_scales, stage, 1.0f);
        const float stage_actor_grad_clip_norm = stage_scale_or_default(options.curriculum_actor_grad_clip_norms, stage, options.actor_grad_clip_norm);

        gpu::FullGpuTrainingOptions phase_options = make_full_training_options(options);
        phase_options.horizon = horizon;
        phase_options.steps = chunk_steps;
        phase_options.seed = options.seed + static_cast<unsigned>(completed_steps);
        phase_options.load_path = previous_checkpoint;
        phase_options.save_path = chunk_checkpoint;
        phase_options.log_path = chunk_log;
        phase_options.learning_rate = options.learning_rate * lr_scale;
        phase_options.diff_rollout_loss_weight = options.diff_rollout_loss_weight * diff_scale;
        phase_options.actor_grad_clip_norm = stage_actor_grad_clip_norm;
        phase_options.load_optimizer_state = options.load_optimizer_state &&
            !(options.reset_optimizer_on_curriculum_transition && previous_chunk_transitioned);

        std::cout << "curriculum_chunk_index=" << chunk_index << "\n";
        std::cout << "curriculum_chunk_horizon=" << horizon << "\n";
        std::cout << "curriculum_chunk_steps=" << chunk_steps << "\n";
        std::cout << "curriculum_chunk_learning_rate=" << phase_options.learning_rate << "\n";
        std::cout << "curriculum_chunk_diff_rollout_loss_weight=" << phase_options.diff_rollout_loss_weight << "\n";
        std::cout << "curriculum_chunk_terminal_loss_scale=" << terminal_scale << "\n";
        std::cout << "curriculum_chunk_actor_grad_clip_norm=" << phase_options.actor_grad_clip_norm << "\n";
        std::cout << "curriculum_chunk_load_optimizer_state=" << (phase_options.load_optimizer_state ? "true" : "false") << "\n";
        std::cout << "curriculum_chunk_log_path=" << chunk_log << "\n";
        std::cout << "curriculum_chunk_checkpoint_path=" << chunk_checkpoint << "\n";
        if(!phase_options.load_path.empty()){
            std::cout << "curriculum_chunk_load_path=" << phase_options.load_path << "\n";
        }
        gpu::EulerGpuLossWeights chunk_weights = weights;
        chunk_weights.terminal_loss_scale *= terminal_scale;
        auto phase_summary = gpu::run_full_gpu_training(phase_options, chunk_weights);
        print_full_gpu_training_summary(phase_summary);
        const float chunk_mean_success_rate = read_chunk_mean_success_rate(chunk_log, phase_summary.final_success_rate);
        const bool latest_checkpoint_copied = phase_summary.checkpoint_saved && copy_file(chunk_checkpoint, checkpoint_path);

        completed_steps += chunk_steps;
        stage_steps += chunk_steps;
        bool transitioned = false;
        if(stage + 1 < options.horizon_curriculum.size() &&
           stage_steps >= min_stage_steps &&
           chunk_mean_success_rate > threshold){
            stage++;
            stage_steps = 0;
            transitioned = true;
        }

        summary_log
            << completed_steps << "," << chunk_index << "," << horizon << "," << chunk_steps << "," << stage_steps << ","
            << threshold << "," << min_stage_steps << ","
            << phase_options.learning_rate << "," << phase_options.diff_rollout_loss_weight << ","
            << chunk_weights.terminal_loss_scale << "," << phase_options.actor_grad_clip_norm << ","
            << (phase_options.load_optimizer_state ? "true" : "false") << ","
            << phase_summary.final_loss << "," << phase_summary.final_grad_norm << ","
            << phase_summary.final_critic_loss << "," << phase_summary.final_critic_error_norm << ","
            << phase_summary.final_success_rate << "," << phase_summary.final_action_saturation << ","
            << chunk_mean_success_rate << ","
            << phase_summary.final_position_norm_mean << "," << phase_summary.final_velocity_norm_mean << ","
            << phase_summary.final_angular_velocity_norm_mean << ","
            << (phase_summary.finite ? "true" : "false") << ","
            << (phase_summary.checkpoint_saved ? "true" : "false") << ","
            << (phase_summary.checkpoint_loaded ? "true" : "false") << ","
            << (latest_checkpoint_copied ? "true" : "false") << ","
            << (transitioned ? "true" : "false") << ","
            << chunk_checkpoint << "," << checkpoint_path << "," << chunk_log << "\n";
        summary_log.flush();

        std::cout << "curriculum_global_step=" << completed_steps << "\n";
        std::cout << "curriculum_success_threshold=" << threshold << "\n";
        std::cout << "curriculum_chunk_mean_success_rate=" << chunk_mean_success_rate << "\n";
        std::cout << "curriculum_stage_transitioned=" << (transitioned ? "true" : "false") << "\n";
        std::cout << "curriculum_current_horizon=" << options.horizon_curriculum[stage] << "\n";

        passed = passed && phase_summary.passed && phase_summary.checkpoint_saved && latest_checkpoint_copied;
        if(!phase_summary.passed || !phase_summary.checkpoint_saved || !latest_checkpoint_copied){
            std::cout << "horizon_curriculum_passed=false\n";
            return false;
        }
        previous_checkpoint = chunk_checkpoint;
        previous_chunk_transitioned = transitioned;
        chunk_index++;
    }
    std::cout << "horizon_curriculum_final_horizon=" << options.horizon_curriculum[stage] << "\n";
    std::cout << "horizon_curriculum_final_checkpoint_path=" << checkpoint_path << "\n";
    std::cout << "horizon_curriculum_passed=" << (passed ? "true" : "false") << "\n";
    return passed;
}

void print_stage9_replay_debug_summary(const gpu::Stage9ReplayDebugSummary& summary){
    std::cout << "gpu_stage9_replay_debug_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "gpu_stage9_replay_debug_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "gpu_stage9_replay_debug_close=" << (summary.close ? "true" : "false") << "\n";
    std::cout << "stage9_replay_written=" << (summary.replay_written ? "true" : "false") << "\n";
    std::cout << "stage9_replay_loaded=" << (summary.replay_loaded ? "true" : "false") << "\n";
    std::cout << "stage9_final_cpu_loss=" << summary.final_cpu_loss << "\n";
    std::cout << "stage9_final_gpu_loss=" << summary.final_gpu_loss << "\n";
    std::cout << "stage9_max_loss_abs_error=" << summary.max_loss_abs_error << "\n";
    std::cout << "stage9_max_action_mean_abs_error=" << summary.max_action_mean_abs_error << "\n";
    std::cout << "stage9_max_action_std_abs_error=" << summary.max_action_std_abs_error << "\n";
    std::cout << "stage9_max_action_saturation_abs_error=" << summary.max_action_saturation_abs_error << "\n";
    std::cout << "stage9_max_action_gradient_norm_abs_error=" << summary.max_action_gradient_norm_abs_error << "\n";
    std::cout << "stage9_max_actor_gradient_norm_abs_error=" << summary.max_actor_gradient_norm_abs_error << "\n";
    std::cout << "stage9_max_weight_abs_error=" << summary.max_weight_abs_error << "\n";
    std::cout << "stage9_max_weight_l2_rel_error=" << summary.max_weight_l2_rel_error << "\n";
    std::cout << "stage9_max_adam_m_norm_abs_error=" << summary.max_adam_m_norm_abs_error << "\n";
    std::cout << "stage9_max_adam_v_norm_abs_error=" << summary.max_adam_v_norm_abs_error << "\n";
    std::cout << "stage9_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_stage9_sampler_parity_summary(const gpu::Stage9SamplerParitySummary& summary){
    std::cout << "stage9_6_sampler_parity_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "stage9_6_sampler_replay_written=" << (summary.replay_written ? "true" : "false") << "\n";
    std::cout << "stage9_6_sampler_replay_loaded=" << (summary.replay_loaded ? "true" : "false") << "\n";
    std::cout << "stage9_6_sampler_metadata_present=" << (summary.metadata_present ? "true" : "false") << "\n";
    std::cout << "stage9_6_sampler_bins_balanced=" << (summary.bins_balanced ? "true" : "false") << "\n";
    std::cout << "stage9_6_sampler_group_weights_close=" << (summary.group_weights_close ? "true" : "false") << "\n";
    std::cout << "stage9_6_sampler_reset_masks_replayed=" << (summary.reset_masks_replayed ? "true" : "false") << "\n";
    std::cout << "stage9_6_sampler_samples=" << summary.samples << "\n";
    std::cout << "stage9_6_sampler_groups=" << summary.groups << "\n";
    std::cout << "stage9_6_sampler_rejected_total=" << summary.rejected_total << "\n";
    std::cout << "stage9_6_sampler_group_weight_sum_min=" << summary.group_weight_sum_min << "\n";
    std::cout << "stage9_6_sampler_group_weight_sum_max=" << summary.group_weight_sum_max << "\n";
    std::cout << "stage9_6_sampler_metadata_mismatch_count=" << summary.metadata_mismatch_count << "\n";
    std::cout << "stage9_6_sampler_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_correlated_size_mass_sampler_validation(const gpu::CorrelatedSizeMassSamplerValidationSummary& summary){
    std::cout << "correlated_size_mass_sampler_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "correlated_size_mass_sampler_default_disabled=" << (summary.default_disabled ? "true" : "false") << "\n";
    std::cout << "correlated_size_mass_sampler_fixed_nominal_unchanged=" << (summary.fixed_nominal_unchanged ? "true" : "false") << "\n";
    std::cout << "correlated_size_mass_sampler_formula_close=" << (summary.correlated_formula_close ? "true" : "false") << "\n";
    std::cout << "correlated_size_mass_sampler_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "correlated_size_mass_sampler_samples=" << summary.samples << "\n";
    std::cout << "correlated_size_mass_sampler_mass_min=" << summary.mass_min << "\n";
    std::cout << "correlated_size_mass_sampler_mass_max=" << summary.mass_max << "\n";
    std::cout << "correlated_size_mass_sampler_size_factor_min=" << summary.size_factor_min << "\n";
    std::cout << "correlated_size_mass_sampler_size_factor_max=" << summary.size_factor_max << "\n";
    std::cout << "correlated_size_mass_sampler_max_default_batch_abs_error=" << summary.max_default_batch_abs_error << "\n";
    std::cout << "correlated_size_mass_sampler_max_nominal_abs_error=" << summary.max_nominal_abs_error << "\n";
    std::cout << "correlated_size_mass_sampler_max_size_factor_bounds_error=" << summary.max_size_factor_bounds_error << "\n";
    std::cout << "correlated_size_mass_sampler_max_thrust_factor_abs_error=" << summary.max_thrust_factor_abs_error << "\n";
    std::cout << "correlated_size_mass_sampler_max_thrust_to_weight_bounds_error=" << summary.max_thrust_to_weight_bounds_error << "\n";
    std::cout << "correlated_size_mass_sampler_max_inertia_bounds_error=" << summary.max_inertia_bounds_error << "\n";
    std::cout << "correlated_size_mass_sampler_max_j_inverse_abs_error=" << summary.max_j_inverse_abs_error << "\n";
    std::cout << "correlated_size_mass_sampler_formula_mismatch_count=" << summary.formula_mismatch_count << "\n";
    std::cout << "correlated_size_mass_sampler_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_trajectory_sampler_validation(const gpu::TrajectorySamplerValidationSummary& summary){
    std::cout << "trajectory_sampler_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "trajectory_sampler_fixed_backward_compatible=" << (summary.fixed_backward_compatible ? "true" : "false") << "\n";
    std::cout << "trajectory_sampler_deterministic=" << (summary.deterministic ? "true" : "false") << "\n";
    std::cout << "trajectory_sampler_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "trajectory_sampler_mixed_covers_all_modes=" << (summary.mixed_covers_all_modes ? "true" : "false") << "\n";
    std::cout << "trajectory_sampler_samples=" << summary.samples << "\n";
    std::cout << "trajectory_sampler_horizon=" << summary.horizon << "\n";
    std::cout << "trajectory_sampler_fixed_reference_max_abs=" << summary.fixed_reference_max_abs << "\n";
    std::cout << "trajectory_sampler_deterministic_max_abs=" << summary.deterministic_max_abs << "\n";
    std::cout << "trajectory_sampler_max_reference_abs=" << summary.max_reference_abs << "\n";
    std::cout << "trajectory_sampler_max_reference_velocity_abs=" << summary.max_reference_velocity_abs << "\n";
    std::cout << "trajectory_sampler_mixed_mode_coverage_mask=" << summary.mixed_mode_coverage_mask << "\n";
    std::cout << "trajectory_sampler_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_stage9_eval_parity_summary(const gpu::Stage9EvalParitySummary& summary){
    std::cout << "stage9_6_eval_parity_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "stage9_6_eval_replay_written=" << (summary.replay_written ? "true" : "false") << "\n";
    std::cout << "stage9_6_eval_replay_loaded=" << (summary.replay_loaded ? "true" : "false") << "\n";
    std::cout << "stage9_6_eval_final_state_close=" << (summary.final_state_close ? "true" : "false") << "\n";
    std::cout << "stage9_6_eval_loss_close=" << (summary.loss_close ? "true" : "false") << "\n";
    std::cout << "stage9_6_eval_success_close=" << (summary.success_close ? "true" : "false") << "\n";
    std::cout << "stage9_6_eval_action_saturation_close=" << (summary.action_saturation_close ? "true" : "false") << "\n";
    std::cout << "stage9_6_eval_cpu_success_rate=" << summary.cpu_success_rate << "\n";
    std::cout << "stage9_6_eval_gpu_success_rate=" << summary.gpu_success_rate << "\n";
    std::cout << "stage9_6_eval_cpu_action_saturation=" << summary.cpu_action_saturation << "\n";
    std::cout << "stage9_6_eval_gpu_action_saturation=" << summary.gpu_action_saturation << "\n";
    std::cout << "stage9_6_eval_mean_cpu_final_position_norm=" << summary.mean_cpu_final_position_norm << "\n";
    std::cout << "stage9_6_eval_mean_gpu_final_position_norm=" << summary.mean_gpu_final_position_norm << "\n";
    std::cout << "stage9_6_eval_max_final_state_abs_error=" << summary.max_final_state_abs_error << "\n";
    std::cout << "stage9_6_eval_max_loss_abs_error=" << summary.max_loss_abs_error << "\n";
    std::cout << "stage9_6_eval_success_mismatch_count=" << summary.success_mismatch_count << "\n";
    std::cout << "stage9_6_eval_saturation_mismatch_count=" << summary.saturation_mismatch_count << "\n";
    std::cout << "stage9_6_eval_nan_inf_count=" << summary.nan_inf_count << "\n";
}

void print_stage9_checkpoint_parity_summary(const gpu::Stage9CheckpointParitySummary& summary){
    std::cout << "stage9_6_checkpoint_parity_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "stage9_6_checkpoint_saved=" << (summary.saved ? "true" : "false") << "\n";
    std::cout << "stage9_6_checkpoint_loaded=" << (summary.loaded ? "true" : "false") << "\n";
    std::cout << "stage9_6_checkpoint_metadata_ok=" << (summary.metadata_ok ? "true" : "false") << "\n";
    std::cout << "stage9_6_checkpoint_weights_close=" << (summary.weights_close ? "true" : "false") << "\n";
    std::cout << "stage9_6_checkpoint_moments_close=" << (summary.moments_close ? "true" : "false") << "\n";
    std::cout << "stage9_6_checkpoint_max_weight_abs_error=" << summary.max_weight_abs_error << "\n";
    std::cout << "stage9_6_checkpoint_max_first_moment_abs_error=" << summary.max_first_moment_abs_error << "\n";
    std::cout << "stage9_6_checkpoint_max_second_moment_abs_error=" << summary.max_second_moment_abs_error << "\n";
}

void print_benchmark(const gpu::BenchmarkSummary& gpu_summary, const gpu::BenchmarkSummary& cpu_summary){
    std::cout << "batch_size=" << gpu_summary.batch_size << "\n";
    std::cout << "horizon=" << gpu_summary.horizon << "\n";
    std::cout << "iterations=" << gpu_summary.iterations << "\n";
    std::cout << "gpu_transitions_per_second=" << gpu_summary.transitions_per_second << "\n";
    std::cout << "gpu_rollouts_per_second=" << gpu_summary.rollouts_per_second << "\n";
    std::cout << "cpu_transitions_per_second=" << cpu_summary.transitions_per_second << "\n";
    std::cout << "cpu_rollouts_per_second=" << cpu_summary.rollouts_per_second << "\n";
    std::cout << "speedup_vs_cpu=" << (cpu_summary.transitions_per_second > 0 ? gpu_summary.transitions_per_second / cpu_summary.transitions_per_second : 0.0) << "\n";
    std::cout << "forward_time_ms=" << gpu_summary.mean_timings.forward_ms << "\n";
    std::cout << "backward_vjp_time_ms=" << gpu_summary.mean_timings.backward_vjp_ms << "\n";
    std::cout << "loss_time_ms=" << gpu_summary.mean_timings.loss_ms << "\n";
    std::cout << "actor_forward_time_ms=0\n";
    std::cout << "actor_backward_time_ms=0\n";
    std::cout << "cpu_gpu_transfer_time_ms=" << (gpu_summary.mean_timings.host_to_device_ms + gpu_summary.mean_timings.device_to_host_ms) << "\n";
    std::cout << "gpu_total_rollout_time_ms=" << gpu_summary.mean_timings.total_ms << "\n";
    std::cout << "cpu_total_rollout_time_ms=" << cpu_summary.mean_timings.total_ms << "\n";
}

void append_benchmark_csv(const std::string& path, const gpu::BenchmarkSummary& gpu_summary, const gpu::BenchmarkSummary& cpu_summary){
    const bool exists = std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if(!exists){
        out << "batch_size,horizon,iterations,gpu_transitions_per_second,gpu_rollouts_per_second,"
            << "cpu_transitions_per_second,cpu_rollouts_per_second,speedup_vs_cpu,"
            << "forward_time_ms,backward_vjp_time_ms,loss_time_ms,actor_forward_time_ms,"
            << "actor_backward_time_ms,cpu_gpu_transfer_time_ms,gpu_total_rollout_time_ms,cpu_total_rollout_time_ms\n";
    }
    out << gpu_summary.batch_size << ","
        << gpu_summary.horizon << ","
        << gpu_summary.iterations << ","
        << gpu_summary.transitions_per_second << ","
        << gpu_summary.rollouts_per_second << ","
        << cpu_summary.transitions_per_second << ","
        << cpu_summary.rollouts_per_second << ","
        << (cpu_summary.transitions_per_second > 0 ? gpu_summary.transitions_per_second / cpu_summary.transitions_per_second : 0.0) << ","
        << gpu_summary.mean_timings.forward_ms << ","
        << gpu_summary.mean_timings.backward_vjp_ms << ","
        << gpu_summary.mean_timings.loss_ms << ","
        << 0 << ","
        << 0 << ","
        << (gpu_summary.mean_timings.host_to_device_ms + gpu_summary.mean_timings.device_to_host_ms) << ","
        << gpu_summary.mean_timings.total_ms << ","
        << cpu_summary.mean_timings.total_ms << "\n";
}

} // namespace

int main(int argc, char** argv){
    Options options;
    if(!parse_options(argc, argv, options)){
        return 1;
    }
    if(options.diff_model != "euler"){
        std::cerr << "CUDA rollout target currently supports --diff-model euler only.\n";
        return 1;
    }
    if(options.production_objective_trace && !options.objective_trace_path.empty() && options.log_path.empty()){
        options.log_path = options.objective_trace_path;
    }
    if(options.production_objective_trace && options.steps == 0 && !options.stage9_6_objective_parity){
        options.steps = options.stage9_6_steps;
    }
    if(options.checkpoint_inspect || options.checkpoint_convert_old){
        std::cerr << "CUDA checkpoint inspect and old-checkpoint conversion are not implemented in this target yet.\n";
        return 1;
    }
    if(options.gpu_batch_size > 0){
        options.batch_size = options.gpu_batch_size;
    }
    if(options.tracking_gate_mode == "local"){
        options.initial_position_scale = options.initial_position_scale_local;
        options.initial_velocity_scale = options.initial_velocity_scale_local;
        options.initial_angular_velocity_scale = options.initial_angular_velocity_scale_local;
        options.throughout_gate_start_step = 0;
    }
    if(options.horizon_curriculum_enabled){
        if(options.horizon_curriculum.empty()){
            options.horizon_curriculum = {16, 32, 64, options.horizon};
        }
        options.horizon = options.horizon_curriculum.back();
    }
    if(options.stage11){
        options.batch_size = 128;
        options.horizon = 128;
        options.sample_dynamics = true;
        options.balanced_dynamics_sampling = true;
        options.sampled_dynamics_level = "broad";
        options.loss_velocity_weight = 3.0f;
        options.terminal_velocity_weight = 20.0f;
        options.success_velocity_threshold = options.stage11_velocity_threshold;
        if(options.stage11_steps == 0){
            std::cerr << "Stage 11 requires --stage11-steps > 0.\n";
            return 1;
        }
    }
    if(options.eval_only && options.load_path.empty()){
        std::cerr << "CUDA eval requires --load-path.\n";
        return 1;
    }
    if(options.eval_only && options.eval_model != "euler"){
        std::cerr << "CUDA eval currently supports --eval-model euler only.\n";
        return 1;
    }
    if(options.batch_size == 0 || options.horizon == 0){
        std::cerr << "Batch size and horizon must be non-zero.\n";
        return 1;
    }

    std::cout << "cuda_stage=physics_rollout_loss_vjp\n";
    std::cout << "gpu_rollout=" << (options.gpu_rollout ? "true" : "false") << "\n";
    std::cout << "actor_output_dim=4\n";
    std::cout << "critic_output_dim=1\n";
    std::cout << "actor_forward_on_gpu=true\n";
    std::cout << "critic_head_on_gpu=true\n";
    std::cout << "gru_backward_on_gpu=true\n";
    std::cout << "optimizer_on_gpu=true\n";
    std::cout << "full_gpu_training_enabled=" << (options.steps > 0 || options.stage11 ? "true" : "false") << "\n";
    std::cout << "stage11_enabled=" << (options.stage11 ? "true" : "false") << "\n";
    std::cout << "stage11_total_steps=" << options.stage11_steps << "\n";
    std::cout << "stage11_success_threshold=" << options.stage11_success_threshold << "\n";
    std::cout << "stage11_velocity_threshold=" << options.stage11_velocity_threshold << "\n";
    std::cout << "eval_only=" << (options.eval_only ? "true" : "false") << "\n";
    std::cout << "eval_model=" << options.eval_model << "\n";
    std::cout << "failure_analysis=" << (options.failure_analysis ? "true" : "false") << "\n";
    std::cout << "horizon_curriculum_enabled=" << (options.horizon_curriculum_enabled ? "true" : "false") << "\n";
    std::cout << "curriculum_gate_check_interval=" << options.curriculum_gate_check_interval << "\n";
    std::cout << "success_position_threshold=" << options.success_position_threshold << "\n";
    std::cout << "success_velocity_threshold=" << options.success_velocity_threshold << "\n";
    std::cout << "success_attitude_threshold=" << options.success_attitude_threshold << "\n";
    std::cout << "success_angular_velocity_threshold=" << options.success_angular_velocity_threshold << "\n";
    std::cout << "success_action_saturation_threshold=" << options.success_action_saturation_threshold << "\n";
    std::cout << "forced_dynamics_bins_enabled=" << (options.forced_bins.enabled ? "true" : "false") << "\n";
    if(options.forced_bins.enabled){
        std::cout << "forced_dynamics_bins=size_mass=" << options.forced_bins.size_mass
                  << ",thrust_to_weight=" << options.forced_bins.thrust_to_weight
                  << ",torque_to_inertia=" << options.forced_bins.torque_to_inertia
                  << ",motor_delay=" << options.forced_bins.motor_delay
                  << ",curve_shape=" << options.forced_bins.curve_shape << "\n";
    }
    std::cout << "balanced_dynamics_sampling=" << (options.balanced_dynamics_sampling ? "true" : "false") << "\n";
    std::cout << "sampled_dynamics_level=" << options.sampled_dynamics_level << "\n";
    std::cout << "correlated_size_mass_sampling=" << (options.correlated_size_mass_sampling ? "true" : "false") << "\n";
    std::cout << "trajectory_mode=" << trajectory_mode_name(options.trajectory_mode) << "\n";
    std::cout << "trajectory_amplitude=" << options.trajectory_amplitude << "\n";
    std::cout << "trajectory_frequency_hz=" << options.trajectory_frequency_hz << "\n";
    std::cout << "tracking_gate_mode=" << options.tracking_gate_mode << "\n";
    std::cout << "throughout_gate_start_step=" << options.throughout_gate_start_step << "\n";
    std::cout << "initial_position_scale=" << options.initial_position_scale << "\n";
    std::cout << "initial_velocity_scale=" << options.initial_velocity_scale << "\n";
    std::cout << "initial_attitude_scale_local=" << options.initial_attitude_scale_local << "\n";
    std::cout << "initial_angular_velocity_scale=" << options.initial_angular_velocity_scale << "\n";
    std::cout << "simulation_frequency_hz=100\n";
    std::cout << "dt=0.01\n";
    std::cout << "sample_dynamics=" << (options.sample_dynamics ? "true" : "false") << "\n";
    std::cout << "learning_rate=" << options.learning_rate << "\n";
    std::cout << "action_magnitude_weight=" << options.action_magnitude_weight << "\n";
    std::cout << "action_smoothness_weight=" << options.action_smoothness_weight << "\n";
    std::cout << "action_saturation_weight=" << options.action_saturation_weight << "\n";
    std::cout << "action_saturation_start=" << options.action_saturation_start << "\n";
    std::cout << "loss_velocity_weight=" << options.loss_velocity_weight << "\n";
    std::cout << "loss_angular_velocity_weight=" << options.loss_angular_velocity_weight << "\n";
    std::cout << "terminal_loss_scale=" << options.terminal_loss_scale << "\n";
    std::cout << "terminal_velocity_weight=" << options.terminal_velocity_weight << "\n";
    std::cout << "terminal_angular_velocity_weight=" << options.terminal_angular_velocity_weight << "\n";
    std::cout << "diff_rollout_loss_weight=" << options.diff_rollout_loss_weight << "\n";
    std::cout << "action_grad_clip_enabled=" << (options.action_grad_clip_enabled ? "true" : "false") << "\n";
    std::cout << "action_grad_clip_norm=" << options.action_grad_clip_norm << "\n";
    std::cout << "actor_grad_clip_enabled=" << (options.actor_grad_clip_enabled ? "true" : "false") << "\n";
    std::cout << "actor_grad_clip_norm=" << options.actor_grad_clip_norm << "\n";
    std::cout << "actor_grad_skip_norm=" << options.actor_grad_skip_norm << "\n";
    std::cout << "disable_physics_gradient=" << (options.disable_physics_gradient ? "true" : "false") << "\n";
    std::cout << "reset_hidden_each_step=" << (options.reset_hidden_each_step ? "true" : "false") << "\n";
    std::cout << "load_optimizer_state=" << (options.load_optimizer_state ? "true" : "false") << "\n";
    std::cout << "reset_optimizer_on_curriculum_transition=" << (options.reset_optimizer_on_curriculum_transition ? "true" : "false") << "\n";

    gpu::EulerGpuBatch batch;
    gpu::EulerGpuLossWeights weights;
    weights.action_magnitude = options.action_magnitude_weight;
    weights.action_smoothness = options.action_smoothness_weight;
    weights.saturation = options.action_saturation_weight;
    weights.saturation_start = options.action_saturation_start;
    weights.velocity = options.loss_velocity_weight;
    weights.angular_velocity = options.loss_angular_velocity_weight;
    weights.terminal_loss_scale = options.terminal_loss_scale;
    weights.terminal_velocity = options.terminal_velocity_weight;
    weights.terminal_angular_velocity = options.terminal_angular_velocity_weight;
    gpu::EulerGpuRunOptions run_options;
    run_options.device = options.gpu_device;
    run_options.compute_action_gradients = true;
    gpu::generate_validation_batch(
        batch,
        options.batch_size,
        options.horizon,
        options.seed,
        nullptr,
        !options.sample_dynamics,
        options.correlated_size_mass_sampling,
        options.trajectory_mode,
        options.trajectory_amplitude,
        options.trajectory_frequency_hz,
        options.initial_position_scale,
        options.initial_velocity_scale,
        options.initial_angular_velocity_scale
    );

    try{
        bool formal_gate_run = false;
        bool formal_gate_passed = true;
        auto make_stage9_options = [&](){
            gpu::Stage9ReplayDebugOptions debug_options;
            debug_options.device = options.gpu_device;
            debug_options.batch_size = options.batch_size;
            debug_options.horizon = options.horizon;
            debug_options.steps = options.stage9_6_steps;
            debug_options.seed = options.seed;
            debug_options.learning_rate = options.learning_rate;
            debug_options.diff_rollout_loss_weight = options.diff_rollout_loss_weight;
            debug_options.action_grad_clip_enabled = options.action_grad_clip_enabled;
            debug_options.action_grad_clip_norm = options.action_grad_clip_norm;
            debug_options.actor_grad_clip_enabled = options.actor_grad_clip_enabled;
            debug_options.actor_grad_clip_norm = options.actor_grad_clip_norm;
            debug_options.actor_grad_skip_norm = options.actor_grad_skip_norm;
            debug_options.actor_grad_eps = options.actor_grad_eps;
            debug_options.disable_physics_gradient = options.disable_physics_gradient;
            debug_options.reset_hidden_each_step = options.reset_hidden_each_step;
            debug_options.replay_path = options.stage9_6_replay_path.empty()
                ? std::string("/tmp/stage9_6_formal_replay.bin")
                : options.stage9_6_replay_path;
            debug_options.log_path = options.log_path;
            return debug_options;
        };
        if(options.stage11){
            const std::vector<std::size_t> horizons = {16, 32, 64, 128};
            const std::size_t phase_count = horizons.size();
            const std::size_t base_steps = options.stage11_steps / phase_count;
            const std::size_t extra_steps = options.stage11_steps % phase_count;
            const std::string log_prefix = options.log_path.empty()
                ? std::string("/tmp/stage11_gpu_curriculum")
                : options.log_path;
            const std::string checkpoint_prefix = options.save_path.empty()
                ? std::string("/tmp/stage11_gpu_curriculum")
                : options.save_path;
            const std::string final_checkpoint = options.save_path.empty()
                ? std::string("/tmp/stage11_gpu_curriculum_final.ckpt")
                : options.save_path;
            const std::string summary_log_path = log_prefix + "_summary.csv";
            std::ofstream stage11_log(summary_log_path);
            if(!stage11_log){
                throw std::runtime_error("Failed to open Stage 11 summary log: " + summary_log_path);
            }
            stage11_log
                << "phase,horizon,steps,checkpoint,loaded_checkpoint,passed,checkpoint_saved,checkpoint_loaded,"
                << "final_loss,final_grad_norm,final_critic_loss,final_critic_error_norm,nan_inf_count\n";
            std::cout << "stage11_horizon_curriculum=16,32,64,128\n";
            std::cout << "stage11_batch_size=128\n";
            std::cout << "stage11_sampled_dynamics=broad\n";
            std::cout << "stage11_balanced_dynamics_sampling=true\n";
            std::cout << "stage11_w_v=3\n";
            std::cout << "stage11_w_terminal_v=20\n";
            std::cout << "stage11_summary_log_path=" << summary_log_path << "\n";

            std::string previous_checkpoint = options.load_path;
            bool stage11_passed = true;
            for(std::size_t phase = 0; phase < phase_count; phase++){
                const std::size_t horizon = horizons[phase];
                const std::size_t phase_steps = base_steps + (phase < extra_steps ? 1 : 0);
                if(phase_steps == 0){
                    continue;
                }
                gpu::FullGpuTrainingOptions phase_options = make_full_training_options(options);
                phase_options.batch_size = 128;
                phase_options.horizon = horizon;
                phase_options.steps = phase_steps;
                phase_options.seed = options.seed + static_cast<unsigned>(phase * 100000u);
                phase_options.sample_dynamics = true;
                phase_options.load_path = previous_checkpoint;
                phase_options.save_path = phase + 1 == phase_count
                    ? final_checkpoint
                    : checkpoint_prefix + stage11_phase_suffix(horizon, ".ckpt");
                phase_options.log_path = log_prefix + stage11_phase_suffix(horizon, ".csv");
                std::cout << "stage11_phase=" << (phase + 1) << "\n";
                std::cout << "stage11_phase_horizon=" << horizon << "\n";
                std::cout << "stage11_phase_steps=" << phase_steps << "\n";
                std::cout << "stage11_phase_log_path=" << phase_options.log_path << "\n";
                std::cout << "stage11_phase_checkpoint_path=" << phase_options.save_path << "\n";
                if(!phase_options.load_path.empty()){
                    std::cout << "stage11_phase_load_path=" << phase_options.load_path << "\n";
                }
                auto phase_summary = gpu::run_full_gpu_training(phase_options, weights);
                print_full_gpu_training_summary(phase_summary);
                stage11_log
                    << (phase + 1) << "," << horizon << "," << phase_steps << ","
                    << phase_options.save_path << "," << phase_options.load_path << ","
                    << (phase_summary.passed ? "true" : "false") << ","
                    << (phase_summary.checkpoint_saved ? "true" : "false") << ","
                    << (phase_summary.checkpoint_loaded ? "true" : "false") << ","
                    << phase_summary.final_loss << ","
                    << phase_summary.final_grad_norm << ","
                    << phase_summary.final_critic_loss << ","
                    << phase_summary.final_critic_error_norm << ","
                    << phase_summary.nan_inf_count << "\n";
                stage11_passed = stage11_passed && phase_summary.passed && phase_summary.checkpoint_saved;
                if(!phase_summary.passed || !phase_summary.checkpoint_saved){
                    std::cout << "stage11_passed=false\n";
                    return 1;
                }
                previous_checkpoint = phase_options.save_path;
            }
            std::cout << "stage11_final_checkpoint_path=" << final_checkpoint << "\n";
            std::cout << "stage11_passed=" << (stage11_passed ? "true" : "false") << "\n";
            return stage11_passed ? 0 : 1;
        }
        if(options.eval_only){
            auto eval_options = make_gpu_eval_options(options);
            auto eval_summary = gpu::run_gpu_policy_eval(eval_options, weights);
            print_gpu_eval_summary(eval_summary);
            if(!eval_summary.passed){
                return 1;
            }
            return 0;
        }
        if(options.horizon_curriculum_enabled){
            const bool curriculum_passed = run_horizon_curriculum(options, weights);
            return curriculum_passed ? 0 : 1;
        }
        if(options.stage9_6_checkpoint_parity){
            formal_gate_run = true;
            const std::string checkpoint_path = options.save_path.empty()
                ? std::string("/tmp/stage9_6_checkpoint_parity_v4.ckpt")
                : options.save_path;
            auto checkpoint_summary = gpu::run_stage9_checkpoint_parity(checkpoint_path, options.seed);
            print_stage9_checkpoint_parity_summary(checkpoint_summary);
            formal_gate_passed = formal_gate_passed && checkpoint_summary.passed;
        }
        if(options.stage9_6_sampler_parity){
            formal_gate_run = true;
            auto sampler_summary = gpu::run_stage9_sampler_parity(make_stage9_options());
            print_stage9_sampler_parity_summary(sampler_summary);
            formal_gate_passed = formal_gate_passed && sampler_summary.passed;
        }
        if(options.stage9_6_eval_parity){
            formal_gate_run = true;
            auto eval_summary = gpu::run_stage9_eval_parity(make_stage9_options(), weights);
            print_stage9_eval_parity_summary(eval_summary);
            formal_gate_passed = formal_gate_passed && eval_summary.passed;
        }
        if(options.stage9_6_objective_parity){
            formal_gate_run = true;
            auto objective_summary = gpu::run_stage9_replay_debug(make_stage9_options(), weights);
            std::cout << "stage9_6_objective_parity_passed=" << (objective_summary.passed ? "true" : "false") << "\n";
            print_stage9_replay_debug_summary(objective_summary);
            formal_gate_passed = formal_gate_passed && objective_summary.passed;
        }
        if(formal_gate_run){
            std::cout << "stage9_6_formal_gates_passed=" << (formal_gate_passed ? "true" : "false") << "\n";
            return formal_gate_passed ? 0 : 1;
        }
        if(options.validate_correlated_size_mass_sampler){
            auto sampler_validation = gpu::validate_correlated_size_mass_sampler(options.batch_size, options.horizon, options.seed);
            print_correlated_size_mass_sampler_validation(sampler_validation);
            if(!sampler_validation.passed){
                return 1;
            }
        }
        if(options.validate_trajectory_sampler){
            auto trajectory_validation = gpu::validate_trajectory_sampler(
                options.batch_size,
                options.horizon,
                options.seed,
                options.trajectory_amplitude,
                options.trajectory_frequency_hz
            );
            print_trajectory_sampler_validation(trajectory_validation);
            if(!trajectory_validation.passed){
                return 1;
            }
        }
        if(options.validate_observation){
            auto observation_validation = gpu::validate_observations_against_cpu(
                batch,
                run_options,
                std::min(options.validation_step, options.horizon)
            );
            print_observation_validation(observation_validation);
            if(!observation_validation.passed){
                return 1;
            }
        }
        if(options.validate_deployment_adapter){
            auto deployment_validation = gpu::validate_deployment_adapter(
                options.batch_size,
                options.horizon,
                options.seed,
                options.trajectory_mode,
                options.trajectory_amplitude,
                options.trajectory_frequency_hz,
                options.validation_step == 0 ? options.horizon / 2 : options.validation_step
            );
            print_deployment_adapter_validation(deployment_validation);
            if(!deployment_validation.passed){
                return 1;
            }
        }
        if(options.validate_actor_forward){
            auto actor_validation = gpu::validate_actor_forward_against_cpu(options.batch_size, options.horizon, options.seed, run_options);
            print_actor_forward_validation(actor_validation);
            if(!actor_validation.passed){
                return 1;
            }
        }
        if(options.validate_critic_backward){
            auto critic_backward_validation = gpu::validate_critic_backward_against_cpu(options.batch_size, options.horizon, options.seed, run_options);
            print_critic_backward_validation(critic_backward_validation);
            if(!critic_backward_validation.passed){
                return 1;
            }
        }
        if(options.validate_closed_loop){
            auto closed_loop_validation = gpu::validate_closed_loop_against_cpu(options.batch_size, options.horizon, options.seed, weights, run_options);
            print_closed_loop_validation(closed_loop_validation);
            if(!closed_loop_validation.passed){
                return 1;
            }
        }
        if(options.validate_action_gradient_injection){
            auto gradient_validation = gpu::validate_action_gradient_injection_against_cpu(options.batch_size, options.horizon, options.seed, weights, run_options);
            print_action_gradient_injection_validation(gradient_validation);
            if(!gradient_validation.passed){
                return 1;
            }
        }
        if(options.validate_actor_backward){
            auto backward_validation = gpu::validate_actor_backward_against_cpu(options.batch_size, options.horizon, options.seed, weights, run_options);
            print_actor_backward_validation(backward_validation);
            if(!backward_validation.passed){
                return 1;
            }
        }
        if(options.validate_adam_update){
            auto adam_validation = gpu::validate_adam_update_against_cpu(options.seed, run_options);
            print_adam_update_validation(adam_validation);
            if(!adam_validation.passed){
                return 1;
            }
        }
        if(options.validate_against_cpu){
            auto validation = gpu::validate_against_cpu(batch, weights, run_options);
            print_validation(validation);
            if(!validation.passed){
                return 1;
            }
        }
        if(options.benchmark){
            auto summary = gpu::benchmark_gpu_rollout(batch, weights, run_options, options.benchmark_iterations);
            auto cpu_summary = gpu::benchmark_cpu_reference(batch, weights, 1);
            print_benchmark(summary, cpu_summary);
            if(!options.log_path.empty()){
                append_benchmark_csv(options.log_path, summary, cpu_summary);
                std::cout << "benchmark_log_path=" << options.log_path << "\n";
            }
        }
        if(options.stage9_debug){
            gpu::Stage9ReplayDebugOptions debug_options;
            debug_options.device = options.gpu_device;
            debug_options.batch_size = options.batch_size;
            debug_options.horizon = options.horizon;
            debug_options.steps = options.stage9_debug_steps;
            debug_options.seed = options.seed;
            debug_options.learning_rate = options.learning_rate;
            debug_options.diff_rollout_loss_weight = options.diff_rollout_loss_weight;
            debug_options.action_grad_clip_enabled = options.action_grad_clip_enabled;
            debug_options.action_grad_clip_norm = options.action_grad_clip_norm;
            debug_options.actor_grad_clip_enabled = options.actor_grad_clip_enabled;
            debug_options.actor_grad_clip_norm = options.actor_grad_clip_norm;
            debug_options.actor_grad_skip_norm = options.actor_grad_skip_norm;
            debug_options.actor_grad_eps = options.actor_grad_eps;
            debug_options.disable_physics_gradient = options.disable_physics_gradient;
            debug_options.reset_hidden_each_step = options.reset_hidden_each_step;
            debug_options.replay_path = options.stage9_debug_replay_path;
            debug_options.log_path = options.log_path;
            auto debug_summary = gpu::run_stage9_replay_debug(debug_options, weights);
            print_stage9_replay_debug_summary(debug_summary);
            if(!debug_summary.passed){
                return 1;
            }
        }
        if(options.steps > 0){
            gpu::FullGpuTrainingOptions training_options = make_full_training_options(options);
            auto training_summary = gpu::run_full_gpu_training(training_options, weights);
            print_full_gpu_training_summary(training_summary);
            if(!training_summary.passed){
                return 1;
            }
        }
        if(options.steps == 0 && !options.validate_observation && !options.validate_actor_forward && !options.validate_closed_loop &&
           !options.validate_action_gradient_injection && !options.validate_actor_backward &&
           !options.validate_critic_backward && !options.validate_adam_update &&
           !options.validate_correlated_size_mass_sampler && !options.validate_trajectory_sampler &&
           !options.validate_deployment_adapter &&
           !options.validate_against_cpu && !options.benchmark && !options.stage9_debug){
            gpu::EulerGpuResult result;
            gpu::EulerGpuTimings timings;
            const int status = gpu::run_euler_gpu_rollout(batch, weights, run_options, result, timings);
            if(status != 0){
                return status;
            }
            std::cout << "single_gpu_rollout_passed=true\n";
            std::cout << "forward_time_ms=" << timings.forward_ms << "\n";
            std::cout << "backward_vjp_time_ms=" << timings.backward_vjp_ms << "\n";
            std::cout << "cpu_gpu_transfer_time_ms=" << (timings.host_to_device_ms + timings.device_to_host_ms) << "\n";
            std::cout << "total_rollout_time_ms=" << timings.total_ms << "\n";
        }
    }
    catch(const std::exception& e){
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}
