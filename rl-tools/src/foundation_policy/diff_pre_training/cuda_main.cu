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

constexpr std::size_t ACTIVE_TRAINING_HORIZON = 16;

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
    bool diagnose_objective_gradient_conflicts = false;
    bool validate_adam_update = false;
    bool validate_correlated_size_mass_sampler = false;
    bool validate_trajectory_sampler = false;
    bool validate_local_initial_conditions = false;
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
    float action_magnitude_center = 0.0f;
    bool hover_relative_action_magnitude = false;
    float action_smoothness_weight = 0.03f;
    float action_saturation_weight = 0.05f;
    float action_saturation_start = 0.95f;
    float loss_velocity_weight = 0.8f;
    float velocity_reference_gain = 0.0f;
    float progress_weight = 0.0f;
    float outward_velocity_weight = 0.0f;
    float clf_weight = 0.0f;
    float window_clf_weight = 0.0f;
    float clf_alpha = 0.2f;
    float clf_position_weight = 8.0f;
    float clf_velocity_weight = 0.8f;
    float clf_attitude_weight = 4.0f;
    float clf_angular_velocity_weight = 0.8f;
    float clf_position_velocity_cross_beta = 0.0f;
    float clf_attitude_angular_velocity_cross_beta = 0.0f;
    float window_clf_epsilon = 1e-3f;
    float window_clf_huber_delta = 1.0f;
    float velocity_barrier_weight = 0.0f;
    float velocity_barrier_safe = 10.0f;
    float angular_velocity_barrier_weight = 0.0f;
    float angular_velocity_barrier_safe = 15.0f;
    float attitude_barrier_weight = 0.0f;
    float attitude_barrier_safe = 4.0f;
    float attitude_control_weight = 0.0f;
    float attitude_control_k_R = 2.0f;
    float attitude_control_k_omega = 1.0f;
    float loss_angular_velocity_weight = 0.8f;
    float loss_linear_acceleration_weight = 0.0f;
    float loss_angular_acceleration_weight = 0.0f;
    float terminal_loss_scale = 0.0f;
    float terminal_velocity_weight = 4.0f;
    float terminal_angular_velocity_weight = 4.0f;
    float diff_rollout_loss_weight = 5e-4f;
    bool action_grad_clip_enabled = false;
    float action_grad_clip_norm = 100.0f;
    bool actor_grad_clip_enabled = true;
    float actor_grad_clip_norm = 100.0f;
    float actor_grad_skip_norm = 1e12f;
    float actor_grad_eps = 1e-6f;
    float temporal_gradient_decay_alpha = 0.0f;
    float velocity_observation_noise = 0.0f;
    std::size_t velocity_observation_delay_steps = 0;
    bool persistent_episode_training = true;
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
    bool eval_only = false;
    bool failure_analysis = false;
    std::string eval_model = "euler";
    std::size_t eval_episodes = 100;
    std::size_t eval_horizon = 500;
    bool h1000_gate_enabled = false;
    std::size_t h1000_gate_interval = 0;
    std::size_t h1000_gate_horizon = 1000;
    std::size_t h1000_gate_episodes = 256;
    bool h1000_gate_sample_dynamics = false;
    std::string h1000_gate_dynamics_level = "small";
    bool h1000_gate_correlated_size_mass_sampling = false;
    float h1000_gate_max_action_saturation_rate = 0.05f;
    float h1000_gate_max_action_abs = 0.98f;
    float h1000_gate_mean_max_angular_velocity_norm = 15.0f;
    float h1000_gate_max_angular_velocity_norm = 40.0f;
    float h1000_gate_mean_max_position_norm = 10.0f;
    float h1000_gate_mean_max_velocity_norm = 15.0f;
    std::string h1000_gate_best_path;
    std::string h1000_gate_candidate_path;
    std::string h1000_gate_log_path;
    bool failure_replay_enabled = false;
    float failure_replay_ratio = 0.30f;
    std::size_t failure_replay_capacity = 512;
    std::size_t failure_replay_backtrack_min = 32;
    std::size_t failure_replay_backtrack_max = 128;
    float failure_replay_position_norm = 20.0f;
    float failure_replay_velocity_norm = 20.0f;
    float failure_replay_angular_velocity_norm = 30.0f;
    float failure_replay_attitude_error = 2.6179938779914943654f;
    float success_position_threshold = 1.0f;
    float success_velocity_threshold = 2.0f;
    float success_attitude_threshold = 3.14159265358979323846f;
    float success_angular_velocity_threshold = 5.0f;
    float success_action_saturation_threshold = 1.0f;
    gpu::ForcedDynamicsBins forced_bins;
    bool balanced_dynamics_sampling = false;
    std::string sampled_dynamics_level = "broad";
    bool correlated_size_mass_sampling = false;
    gpu::TrajectoryMode trajectory_mode = gpu::TrajectoryMode::FIXED;
    float trajectory_amplitude = 0.03f;
    float trajectory_frequency_hz = 0.5f;
    std::size_t training_episode_steps = 500;
    std::string tracking_gate_mode = "recovery";
    std::size_t throughout_gate_start_step = 0;
    float initial_position_scale = 2.5f;
    float initial_velocity_scale = 20.0f;
    float initial_attitude_scale = 18.947368f;
    float initial_angular_velocity_scale = 50.0f;
    float near_zero_guidance_probability = 0.10f;
    float initial_position_scale_local = 0.14f;
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
        << "    [--gpu-diagnose-objective-gradient-conflicts]\n"
        << "    [--gpu-validate-adam-update]\n"
        << "    [--gpu-validate-correlated-size-mass-sampler]\n"
        << "    [--gpu-validate-trajectory-sampler]\n"
        << "    [--gpu-validate-local-initial-conditions]\n"
        << "    [--gpu-validate-deployment-adapter] [--validation-step N]\n"
        << "    [--gpu-stage9-debug] [--stage9-debug-steps N] [--stage9-debug-replay-path PATH]\n"
        << "    [--gpu-benchmark] [--gpu-benchmark-iterations N]\n"
        << "    [--steps N] [--learning-rate LR]\n"
        << "    [--diff-rollout-loss-weight W] [--disable-physics-gradient]\n"
        << "    [--temporal-gradient-decay-alpha A]\n"
        << "    [--velocity-observation-noise STD] [--velocity-observation-delay-steps N]\n"
        << "    [--action-grad-clip VALUE] [--disable-action-grad-clip]\n"
        << "    [--actor-grad-clip VALUE] [--disable-actor-grad-clip] [--actor-grad-skip-norm VALUE]\n"
        << "    [--reset-hidden-each-step]\n"
        << "    [--production-objective-trace] [--objective-trace-path PATH]\n"
        << "    [--stage9-6-objective-parity] [--stage9-6-sampler-parity] [--stage9-6-eval-parity]\n"
        << "    [--stage9-6-checkpoint-parity] [--save-optimizer] [--load-optimizer]\n"
        << "    [--checkpoint-inspect [PATH]] [--checkpoint-convert-old [PATH]]\n"
        << "    [--stage9-6-steps N] [--stage9-6-replay-path PATH]\n"
        << "    active training defaults to H16 origin recovery; non-fixed trajectories and curriculum flags are rejected\n"
        << "    [--eval-only] [--eval-episodes N] [--eval-horizon N]\n"
        << "    [--h1000-gate] [--h1000-gate-interval N] [--h1000-gate-horizon N] [--h1000-gate-episodes N]\n"
        << "    [--h1000-gate-small-dynamics|--h1000-gate-fixed-dynamics] [--h1000-gate-correlated-size-mass-sampling]\n"
        << "    [--h1000-gate-best-path PATH] [--h1000-gate-log-path PATH]\n"
        << "    [--failure-analysis] [--force-dynamics-bins size_mass=3,thrust_to_weight=3,...]\n"
        << "    [--strict-stability-thresholds]\n"
        << "    [--success-position-threshold VALUE] [--success-velocity-threshold VALUE]\n"
        << "    [--success-attitude-threshold RAD] [--success-attitude-threshold-deg DEG]\n"
        << "    [--success-angular-velocity-threshold VALUE]\n"
        << "    [--success-action-saturation-threshold VALUE]\n"
        << "    [--w-action-magnitude W] [--w-u W] [--w-sat W]\n"
        << "    [--action-magnitude-center VALUE] [--hover-relative-action-magnitude]\n"
        << "    [--w-linear-acceleration W] [--w-angular-acceleration W]\n"
        << "    [--w-progress W] [--w-outward-velocity W] [--velocity-reference-gain K]\n"
        << "    [--w-window-clf W] [--w-clf W] [--clf-alpha A] [--clf-pv-beta B] [--clf-rw-beta B]\n"
        << "    [--window-clf-eps EPS] [--window-clf-huber-delta D]\n"
        << "    [--w-velocity-barrier W] [--velocity-barrier-safe V]\n"
        << "    [--w-omega-barrier W] [--omega-barrier-safe WSAFE]\n"
        << "    [--w-attitude-barrier W] [--attitude-barrier-safe PSI]\n"
        << "    [--w-clf W] [--w-window-clf W] [--clf-alpha A] [--w-clf-position W] [--w-clf-velocity W]\n"
        << "    [--w-clf-attitude W] [--w-clf-angular-velocity W]\n"
        << "    [--w-attitude-control W] [--attitude-control-k-r K] [--attitude-control-k-omega K]\n"
        << "    [--action-saturation-start VALUE] [--terminal-loss-scale VALUE]\n"
        << "    [--no-load-optimizer]\n"
        << "    [--correlated-size-mass-sampling]\n"
        << "    [--trajectory-mode fixed|step|circle|figure8|mixed]  # eval/deployment setpoint shifting only\n"
        << "    [--trajectory-amplitude M] [--trajectory-frequency-hz F] [--training-episode-steps N]\n"
        << "    [--h1000-gate] [--failure-replay] [--failure-replay-ratio P]\n"
        << "    [--tracking-gate-mode local|recovery] [--throughout-gate-start-step N]\n"
        << "    [--initial-position-scale S] [--initial-velocity-scale S]\n"
        << "    [--initial-attitude-scale S] [--initial-angular-velocity-scale S]\n"
        << "    [--near-zero-guidance-probability P]\n"
        << "    [--initial-position-scale-local S] [--initial-velocity-scale-local S]\n"
        << "    [--initial-attitude-scale-local S] [--initial-angular-velocity-scale-local S]\n"
        << "    [--log-path PATH] [--save-path PATH] [--load-path PATH]\n"
        << "\n"
        << "This CUDA target implements the active origin-recovery position-controller path:\n"
        << "fixed H16 differentiable Euler rollout, physics VJP, RDAC actor/GRU/critic-head\n"
        << "BPTT, and Adam update on GPU. Non-origin setpoints are deployment/eval-time\n"
        << "setpoint shifting through relative p/v offsets, not a training curriculum.\n";
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

gpu::TrajectoryMode parse_trajectory_mode(const std::string& value){
    if(value == "fixed") return gpu::TrajectoryMode::FIXED;
    if(value == "step") return gpu::TrajectoryMode::STEP;
    if(value == "circle") return gpu::TrajectoryMode::CIRCLE;
    if(value == "figure8" || value == "figure-eight") return gpu::TrajectoryMode::FIGURE8;
    if(value == "mixed") return gpu::TrajectoryMode::MIXED;
    throw std::runtime_error("Unknown --trajectory-mode value: " + value);
}

gpu::DynamicsRandomizationLevel parse_dynamics_randomization_level(const std::string& value){
    if(value == "broad"){
        return gpu::DynamicsRandomizationLevel::BROAD;
    }
    if(value == "small"){
        return gpu::DynamicsRandomizationLevel::SMALL;
    }
    throw std::runtime_error("Unknown --sampled-dynamics-level value: " + value);
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
        else if(arg == "--gpu-diagnose-objective-gradient-conflicts"){
            options.diagnose_objective_gradient_conflicts = true;
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
        else if(arg == "--gpu-validate-local-initial-conditions"){
            options.validate_local_initial_conditions = true;
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
        else if(arg == "--temporal-gradient-decay-alpha" && i + 1 < argc){
            options.temporal_gradient_decay_alpha = std::stof(argv[++i]);
        }
        else if(arg == "--velocity-observation-noise" && i + 1 < argc){
            options.velocity_observation_noise = std::stof(argv[++i]);
        }
        else if(arg == "--velocity-observation-delay-steps" && i + 1 < argc){
            options.velocity_observation_delay_steps = std::stoull(argv[++i]);
        }
        else if(arg == "--persistent-episode-training"){
            options.persistent_episode_training = true;
        }
        else if(arg == "--disable-persistent-episode-training" || arg == "--fresh-h16-batches"){
            options.persistent_episode_training = false;
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
            (void)parse_dynamics_randomization_level(options.sampled_dynamics_level);
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
        else if(arg == "--training-episode-steps" && i + 1 < argc){
            options.training_episode_steps = std::stoull(argv[++i]);
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
        else if(arg == "--initial-attitude-scale" && i + 1 < argc){
            options.initial_attitude_scale = std::stof(argv[++i]);
        }
        else if(arg == "--initial-angular-velocity-scale" && i + 1 < argc){
            options.initial_angular_velocity_scale = std::stof(argv[++i]);
        }
        else if(arg == "--near-zero-guidance-probability" && i + 1 < argc){
            options.near_zero_guidance_probability = std::stof(argv[++i]);
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
        else if(arg == "--velocity-reference-gain" && i + 1 < argc){
            options.velocity_reference_gain = std::stof(argv[++i]);
        }
        else if(arg == "--w-progress" && i + 1 < argc){
            options.progress_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-outward-velocity" && i + 1 < argc){
            options.outward_velocity_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-clf" && i + 1 < argc){
            options.clf_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-window-clf" && i + 1 < argc){
            options.window_clf_weight = std::stof(argv[++i]);
        }
        else if(arg == "--clf-alpha" && i + 1 < argc){
            options.clf_alpha = std::stof(argv[++i]);
        }
        else if(arg == "--w-clf-position" && i + 1 < argc){
            options.clf_position_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-clf-velocity" && i + 1 < argc){
            options.clf_velocity_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-clf-attitude" && i + 1 < argc){
            options.clf_attitude_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-clf-angular-velocity" && i + 1 < argc){
            options.clf_angular_velocity_weight = std::stof(argv[++i]);
        }
        else if(arg == "--clf-pv-beta" && i + 1 < argc){
            options.clf_position_velocity_cross_beta = std::stof(argv[++i]);
        }
        else if(arg == "--clf-rw-beta" && i + 1 < argc){
            options.clf_attitude_angular_velocity_cross_beta = std::stof(argv[++i]);
        }
        else if(arg == "--window-clf-eps" && i + 1 < argc){
            options.window_clf_epsilon = std::stof(argv[++i]);
        }
        else if(arg == "--window-clf-huber-delta" && i + 1 < argc){
            options.window_clf_huber_delta = std::stof(argv[++i]);
        }
        else if(arg == "--w-velocity-barrier" && i + 1 < argc){
            options.velocity_barrier_weight = std::stof(argv[++i]);
        }
        else if(arg == "--velocity-barrier-safe" && i + 1 < argc){
            options.velocity_barrier_safe = std::stof(argv[++i]);
        }
        else if((arg == "--w-angular-velocity-barrier" || arg == "--w-omega-barrier") && i + 1 < argc){
            options.angular_velocity_barrier_weight = std::stof(argv[++i]);
        }
        else if((arg == "--angular-velocity-barrier-safe" || arg == "--omega-barrier-safe") && i + 1 < argc){
            options.angular_velocity_barrier_safe = std::stof(argv[++i]);
        }
        else if(arg == "--w-attitude-barrier" && i + 1 < argc){
            options.attitude_barrier_weight = std::stof(argv[++i]);
        }
        else if(arg == "--attitude-barrier-safe" && i + 1 < argc){
            options.attitude_barrier_safe = std::stof(argv[++i]);
        }
        else if(arg == "--w-attitude-control" && i + 1 < argc){
            options.attitude_control_weight = std::stof(argv[++i]);
        }
        else if(arg == "--attitude-control-k-r" && i + 1 < argc){
            options.attitude_control_k_R = std::stof(argv[++i]);
        }
        else if(arg == "--attitude-control-k-omega" && i + 1 < argc){
            options.attitude_control_k_omega = std::stof(argv[++i]);
        }
        else if(arg == "--w-linear-acceleration" && i + 1 < argc){
            options.loss_linear_acceleration_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-angular-acceleration" && i + 1 < argc){
            options.loss_angular_acceleration_weight = std::stof(argv[++i]);
        }
        else if(arg == "--w-action-magnitude" && i + 1 < argc){
            options.action_magnitude_weight = std::stof(argv[++i]);
        }
        else if(arg == "--action-magnitude-center" && i + 1 < argc){
            options.action_magnitude_center = std::stof(argv[++i]);
            options.hover_relative_action_magnitude = false;
        }
        else if(arg == "--hover-relative-action-magnitude"){
            options.hover_relative_action_magnitude = true;
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
        else if(arg == "--h1000-gate"){
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--h1000-gate-interval" && i + 1 < argc){
            options.h1000_gate_interval = static_cast<std::size_t>(std::stoul(argv[++i]));
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--h1000-gate-horizon" && i + 1 < argc){
            options.h1000_gate_horizon = static_cast<std::size_t>(std::stoul(argv[++i]));
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--h1000-gate-episodes" && i + 1 < argc){
            options.h1000_gate_episodes = static_cast<std::size_t>(std::stoul(argv[++i]));
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--h1000-gate-fixed-dynamics"){
            options.h1000_gate_sample_dynamics = false;
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--h1000-gate-small-dynamics"){
            options.h1000_gate_sample_dynamics = true;
            options.h1000_gate_dynamics_level = "small";
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--h1000-gate-broad-dynamics"){
            options.h1000_gate_sample_dynamics = true;
            options.h1000_gate_dynamics_level = "broad";
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--h1000-gate-correlated-size-mass-sampling"){
            options.h1000_gate_correlated_size_mass_sampling = true;
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--h1000-gate-max-action-saturation-rate" && i + 1 < argc){
            options.h1000_gate_max_action_saturation_rate = std::stof(argv[++i]);
        }
        else if(arg == "--h1000-gate-max-action-abs" && i + 1 < argc){
            options.h1000_gate_max_action_abs = std::stof(argv[++i]);
        }
        else if(arg == "--h1000-gate-mean-max-omega" && i + 1 < argc){
            options.h1000_gate_mean_max_angular_velocity_norm = std::stof(argv[++i]);
        }
        else if(arg == "--h1000-gate-max-omega" && i + 1 < argc){
            options.h1000_gate_max_angular_velocity_norm = std::stof(argv[++i]);
        }
        else if(arg == "--h1000-gate-mean-max-position" && i + 1 < argc){
            options.h1000_gate_mean_max_position_norm = std::stof(argv[++i]);
        }
        else if(arg == "--h1000-gate-mean-max-velocity" && i + 1 < argc){
            options.h1000_gate_mean_max_velocity_norm = std::stof(argv[++i]);
        }
        else if(arg == "--h1000-gate-best-path" && i + 1 < argc){
            options.h1000_gate_best_path = argv[++i];
            options.h1000_gate_enabled = true;
        }
        else if(arg == "--h1000-gate-candidate-path" && i + 1 < argc){
            options.h1000_gate_candidate_path = argv[++i];
            options.h1000_gate_enabled = true;
        }
        else if(arg == "--h1000-gate-log-path" && i + 1 < argc){
            options.h1000_gate_log_path = argv[++i];
            options.h1000_gate_enabled = true;
        }
        else if(arg == "--failure-replay"){
            options.failure_replay_enabled = true;
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--failure-replay-ratio" && i + 1 < argc){
            options.failure_replay_ratio = std::stof(argv[++i]);
            options.failure_replay_enabled = true;
            options.h1000_gate_enabled = true;
            options.gpu_rollout = true;
        }
        else if(arg == "--failure-replay-capacity" && i + 1 < argc){
            options.failure_replay_capacity = static_cast<std::size_t>(std::stoul(argv[++i]));
            options.failure_replay_enabled = true;
        }
        else if(arg == "--failure-replay-backtrack-min" && i + 1 < argc){
            options.failure_replay_backtrack_min = static_cast<std::size_t>(std::stoul(argv[++i]));
            options.failure_replay_enabled = true;
        }
        else if(arg == "--failure-replay-backtrack-max" && i + 1 < argc){
            options.failure_replay_backtrack_max = static_cast<std::size_t>(std::stoul(argv[++i]));
            options.failure_replay_enabled = true;
        }
        else if(arg == "--failure-replay-position-norm" && i + 1 < argc){
            options.failure_replay_position_norm = std::stof(argv[++i]);
            options.failure_replay_enabled = true;
        }
        else if(arg == "--failure-replay-velocity-norm" && i + 1 < argc){
            options.failure_replay_velocity_norm = std::stof(argv[++i]);
            options.failure_replay_enabled = true;
        }
        else if((arg == "--failure-replay-angular-velocity-norm" || arg == "--failure-replay-omega-norm") && i + 1 < argc){
            options.failure_replay_angular_velocity_norm = std::stof(argv[++i]);
            options.failure_replay_enabled = true;
        }
        else if(arg == "--failure-replay-attitude-error" && i + 1 < argc){
            options.failure_replay_attitude_error = std::stof(argv[++i]);
            options.failure_replay_enabled = true;
        }
        else if(arg == "--failure-replay-attitude-error-deg" && i + 1 < argc){
            options.failure_replay_attitude_error = std::stof(argv[++i]) * 3.14159265358979323846f / 180.0f;
            options.failure_replay_enabled = true;
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
    training_options.temporal_gradient_decay_alpha = options.temporal_gradient_decay_alpha;
    training_options.clf_weight = options.clf_weight;
    training_options.window_clf_weight = options.window_clf_weight;
    training_options.clf_alpha = options.clf_alpha;
    training_options.clf_position = options.clf_position_weight;
    training_options.clf_velocity = options.clf_velocity_weight;
    training_options.clf_attitude = options.clf_attitude_weight;
    training_options.clf_angular_velocity = options.clf_angular_velocity_weight;
    training_options.clf_position_velocity_cross_beta = options.clf_position_velocity_cross_beta;
    training_options.clf_attitude_angular_velocity_cross_beta = options.clf_attitude_angular_velocity_cross_beta;
    training_options.window_clf_epsilon = options.window_clf_epsilon;
    training_options.window_clf_huber_delta = options.window_clf_huber_delta;
    training_options.velocity_barrier_weight = options.velocity_barrier_weight;
    training_options.velocity_barrier_safe = options.velocity_barrier_safe;
    training_options.angular_velocity_barrier_weight = options.angular_velocity_barrier_weight;
    training_options.angular_velocity_barrier_safe = options.angular_velocity_barrier_safe;
    training_options.attitude_barrier_weight = options.attitude_barrier_weight;
    training_options.attitude_barrier_safe = options.attitude_barrier_safe;
    training_options.outward_velocity_weight = options.outward_velocity_weight;
    training_options.attitude_control_weight = options.attitude_control_weight;
    training_options.attitude_control_k_R = options.attitude_control_k_R;
    training_options.attitude_control_k_omega = options.attitude_control_k_omega;
    training_options.velocity_observation_noise = options.velocity_observation_noise;
    training_options.velocity_observation_delay_steps = options.velocity_observation_delay_steps;
    training_options.persistent_episode_training = options.persistent_episode_training;
    training_options.disable_physics_gradient = options.disable_physics_gradient;
    training_options.reset_hidden_each_step = options.reset_hidden_each_step;
    training_options.sample_dynamics = options.sample_dynamics;
    training_options.load_optimizer_state = options.load_optimizer_state;
    training_options.dynamics_randomization_level = parse_dynamics_randomization_level(options.sampled_dynamics_level);
    training_options.correlated_size_mass_sampling = options.correlated_size_mass_sampling;
    training_options.trajectory_mode = gpu::TrajectoryMode::FIXED;
    training_options.trajectory_amplitude = 0.0f;
    training_options.trajectory_frequency_hz = 0.0f;
    training_options.training_episode_steps = options.training_episode_steps;
    training_options.initial_position_scale = options.initial_position_scale;
    training_options.initial_velocity_scale = options.initial_velocity_scale;
    training_options.initial_attitude_scale = options.initial_attitude_scale;
    training_options.initial_angular_velocity_scale = options.initial_angular_velocity_scale;
    training_options.near_zero_guidance_probability = options.near_zero_guidance_probability;
    training_options.success_position_threshold = options.success_position_threshold;
    training_options.success_velocity_threshold = options.success_velocity_threshold;
    training_options.success_attitude_threshold = options.success_attitude_threshold;
    training_options.success_angular_velocity_threshold = options.success_angular_velocity_threshold;
    training_options.success_action_saturation_threshold = options.success_action_saturation_threshold;
    training_options.log_path = options.log_path;
    training_options.save_path = options.save_path;
    training_options.load_path = options.load_path;
    training_options.h1000_gate_enabled = options.h1000_gate_enabled;
    training_options.h1000_gate_interval = options.h1000_gate_interval;
    training_options.h1000_gate_horizon = options.h1000_gate_horizon;
    training_options.h1000_gate_episodes = options.h1000_gate_episodes;
    training_options.h1000_gate_sample_dynamics = options.h1000_gate_sample_dynamics;
    training_options.h1000_gate_dynamics_randomization_level = parse_dynamics_randomization_level(options.h1000_gate_dynamics_level);
    training_options.h1000_gate_correlated_size_mass_sampling = options.h1000_gate_correlated_size_mass_sampling;
    training_options.h1000_gate_max_action_saturation_rate = options.h1000_gate_max_action_saturation_rate;
    training_options.h1000_gate_max_action_abs = options.h1000_gate_max_action_abs;
    training_options.h1000_gate_mean_max_angular_velocity_norm = options.h1000_gate_mean_max_angular_velocity_norm;
    training_options.h1000_gate_max_angular_velocity_norm = options.h1000_gate_max_angular_velocity_norm;
    training_options.h1000_gate_mean_max_position_norm = options.h1000_gate_mean_max_position_norm;
    training_options.h1000_gate_mean_max_velocity_norm = options.h1000_gate_mean_max_velocity_norm;
    training_options.h1000_gate_best_path = options.h1000_gate_best_path;
    training_options.h1000_gate_candidate_path = options.h1000_gate_candidate_path;
    training_options.h1000_gate_log_path = options.h1000_gate_log_path;
    training_options.failure_replay_enabled = options.failure_replay_enabled;
    training_options.failure_replay_ratio = options.failure_replay_ratio;
    training_options.failure_replay_capacity = options.failure_replay_capacity;
    training_options.failure_replay_backtrack_min = options.failure_replay_backtrack_min;
    training_options.failure_replay_backtrack_max = options.failure_replay_backtrack_max;
    training_options.failure_replay_position_norm = options.failure_replay_position_norm;
    training_options.failure_replay_velocity_norm = options.failure_replay_velocity_norm;
    training_options.failure_replay_angular_velocity_norm = options.failure_replay_angular_velocity_norm;
    training_options.failure_replay_attitude_error = options.failure_replay_attitude_error;
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
    eval_options.dynamics_randomization_level = parse_dynamics_randomization_level(options.sampled_dynamics_level);
    eval_options.correlated_size_mass_sampling = options.correlated_size_mass_sampling;
    eval_options.trajectory_mode = options.trajectory_mode;
    eval_options.trajectory_amplitude = options.trajectory_amplitude;
    eval_options.trajectory_frequency_hz = options.trajectory_frequency_hz;
    eval_options.initial_position_scale = options.initial_position_scale;
    eval_options.initial_velocity_scale = options.initial_velocity_scale;
    eval_options.initial_attitude_scale = options.initial_attitude_scale;
    eval_options.initial_angular_velocity_scale = options.initial_angular_velocity_scale;
    eval_options.near_zero_guidance_probability = options.near_zero_guidance_probability;
    eval_options.success_position_threshold = options.success_position_threshold;
    eval_options.success_velocity_threshold = options.success_velocity_threshold;
    eval_options.success_attitude_threshold = options.success_attitude_threshold;
    eval_options.success_angular_velocity_threshold = options.success_angular_velocity_threshold;
    eval_options.success_action_saturation_threshold = options.success_action_saturation_threshold;
    eval_options.throughout_gate_start_step = options.throughout_gate_start_step;
    eval_options.velocity_observation_noise = options.velocity_observation_noise;
    eval_options.velocity_observation_delay_steps = options.velocity_observation_delay_steps;
    eval_options.forced_bins = options.forced_bins;
    eval_options.load_path = options.load_path;
    eval_options.log_path = options.log_path;
    return eval_options;
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

void print_objective_gradient_conflict_diagnostic(const gpu::ObjectiveGradientConflictSummary& summary){
    std::cout << "objective_gradient_conflict_diagnostic_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "objective_gradient_conflict_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "objective_gradient_conflict_physics_nonzero=" << (summary.physics_nonzero ? "true" : "false") << "\n";
    std::cout << "objective_gradient_conflict_critic_nonzero=" << (summary.critic_nonzero ? "true" : "false") << "\n";
    std::cout << "objective_gradient_conflict_active_cuda_transition_consistency_present="
              << (summary.active_cuda_transition_consistency_present ? "true" : "false") << "\n";
    std::cout << "objective_gradient_conflict_batch_size=" << summary.batch_size << "\n";
    std::cout << "objective_gradient_conflict_horizon=" << summary.horizon << "\n";
    std::cout << "objective_gradient_conflict_diff_loss_scaled=" << summary.diff_loss_scaled << "\n";
    std::cout << "objective_gradient_conflict_critic_loss_scaled=" << summary.critic_loss_scaled << "\n";
    std::cout << "objective_gradient_conflict_physics_shared_norm=" << summary.physics_shared_norm << "\n";
    std::cout << "objective_gradient_conflict_critic_shared_norm=" << summary.critic_shared_norm << "\n";
    std::cout << "objective_gradient_conflict_combined_shared_norm=" << summary.combined_shared_norm << "\n";
    std::cout << "objective_gradient_conflict_physics_actor_head_norm=" << summary.physics_actor_head_norm << "\n";
    std::cout << "objective_gradient_conflict_critic_actor_head_norm=" << summary.critic_actor_head_norm << "\n";
    std::cout << "objective_gradient_conflict_critic_head_norm=" << summary.critic_head_norm << "\n";
    std::cout << "objective_gradient_conflict_shared_cosine=" << summary.shared_cosine << "\n";
    std::cout << "objective_gradient_conflict_encoder_cosine=" << summary.encoder_cosine << "\n";
    std::cout << "objective_gradient_conflict_gru_input_cosine=" << summary.gru_input_cosine << "\n";
    std::cout << "objective_gradient_conflict_gru_hidden_cosine=" << summary.gru_hidden_cosine << "\n";
    std::cout << "objective_gradient_conflict_h0_cosine=" << summary.h0_cosine << "\n";
    std::cout << "objective_gradient_conflict_nan_inf_count=" << summary.nan_inf_count << "\n";
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
    std::cout << "gpu_full_training_returns_mean=" << summary.returns_mean << "\n";
    std::cout << "gpu_full_training_returns_std=" << summary.returns_std << "\n";
    std::cout << "gpu_full_training_episode_length_mean=" << summary.episode_length_mean << "\n";
    std::cout << "gpu_full_training_episode_length_std=" << summary.episode_length_std << "\n";
    std::cout << "gpu_full_training_num_terminated=" << summary.num_terminated << "\n";
    std::cout << "gpu_full_training_share_terminated=" << summary.share_terminated << "\n";
    std::cout << "gpu_full_training_position_cost=" << summary.position_cost << "\n";
    std::cout << "gpu_full_training_orientation_cost=" << summary.orientation_cost << "\n";
    std::cout << "gpu_full_training_linear_velocity_cost=" << summary.linear_velocity_cost << "\n";
    std::cout << "gpu_full_training_angular_velocity_cost=" << summary.angular_velocity_cost << "\n";
    std::cout << "gpu_full_training_action_cost=" << summary.action_cost << "\n";
    std::cout << "gpu_full_training_weighted_cost=" << summary.weighted_cost << "\n";
    std::cout << "gpu_full_training_reward=" << summary.reward_mean << "\n";
    std::cout << "gpu_full_training_nan_inf_count=" << summary.nan_inf_count << "\n";
    std::cout << "gpu_full_training_episode_steps=" << summary.training_episode_steps << "\n";
    std::cout << "gpu_full_training_persistent_episode_training=" << (summary.persistent_episode_training ? "true" : "false") << "\n";
    std::cout << "gpu_full_training_persistent_episode_step=" << summary.persistent_episode_step << "\n";
    std::cout << "gpu_full_training_segment_start=" << summary.segment_start << "\n";
    std::cout << "gpu_full_training_segment_end=" << summary.segment_end << "\n";
    std::cout << "gpu_full_training_episode_reset_count=" << summary.episode_reset_count << "\n";
    std::cout << "gpu_full_training_final_window_start_mean=" << summary.final_window_start_mean << "\n";
    std::cout << "gpu_full_training_h1000_gate_enabled=" << (summary.h1000_gate_enabled ? "true" : "false") << "\n";
    std::cout << "gpu_full_training_h1000_gate_eval_count=" << summary.h1000_gate_eval_count << "\n";
    std::cout << "gpu_full_training_h1000_gate_pass_count=" << summary.h1000_gate_pass_count << "\n";
    std::cout << "gpu_full_training_h1000_gate_fail_count=" << summary.h1000_gate_fail_count << "\n";
    std::cout << "gpu_full_training_h1000_gate_rollback_count=" << summary.h1000_gate_rollback_count << "\n";
    std::cout << "gpu_full_training_h1000_gate_last_passed=" << (summary.h1000_gate_last_passed ? "true" : "false") << "\n";
    std::cout << "gpu_full_training_h1000_gate_last_weighted_cost=" << summary.h1000_gate_last_weighted_cost << "\n";
    std::cout << "gpu_full_training_h1000_gate_last_mean_max_omega=" << summary.h1000_gate_last_mean_max_angular_velocity_norm << "\n";
    std::cout << "gpu_full_training_h1000_gate_last_max_omega=" << summary.h1000_gate_last_max_angular_velocity_norm << "\n";
    std::cout << "gpu_full_training_h1000_gate_last_action_saturation=" << summary.h1000_gate_last_action_saturation_rate << "\n";
    std::cout << "gpu_full_training_failure_replay_enabled=" << (summary.failure_replay_enabled ? "true" : "false") << "\n";
    std::cout << "gpu_full_training_failure_replay_buffer_size=" << summary.failure_replay_buffer_size << "\n";
    std::cout << "gpu_full_training_failure_replay_added_count=" << summary.failure_replay_added_count << "\n";
    std::cout << "gpu_full_training_failure_replay_used_count=" << summary.failure_replay_used_count << "\n";
    std::cout << "gpu_checkpoint_saved=" << (summary.checkpoint_saved ? "true" : "false") << "\n";
    std::cout << "gpu_checkpoint_loaded=" << (summary.checkpoint_loaded ? "true" : "false") << "\n";
}

void print_gpu_eval_summary(const gpu::GpuPolicyEvalSummary& summary){
    std::cout << "gpu_eval_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "gpu_eval_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "gpu_eval_checkpoint_loaded=" << (summary.checkpoint_loaded ? "true" : "false") << "\n";
    std::cout << "gpu_eval_returns_mean=" << summary.returns_mean << "\n";
    std::cout << "gpu_eval_returns_std=" << summary.returns_std << "\n";
    std::cout << "gpu_eval_episode_length_mean=" << summary.episode_length_mean << "\n";
    std::cout << "gpu_eval_episode_length_std=" << summary.episode_length_std << "\n";
    std::cout << "gpu_eval_num_terminated=" << summary.num_terminated << "\n";
    std::cout << "gpu_eval_share_terminated=" << summary.share_terminated << "\n";
    std::cout << "gpu_eval_position_cost=" << summary.position_cost << "\n";
    std::cout << "gpu_eval_orientation_cost=" << summary.orientation_cost << "\n";
    std::cout << "gpu_eval_linear_velocity_cost=" << summary.linear_velocity_cost << "\n";
    std::cout << "gpu_eval_angular_velocity_cost=" << summary.angular_velocity_cost << "\n";
    std::cout << "gpu_eval_action_cost=" << summary.action_cost << "\n";
    std::cout << "gpu_eval_weighted_cost=" << summary.weighted_cost << "\n";
    std::cout << "gpu_eval_reward=" << summary.reward_mean << "\n";
    std::cout << "gpu_eval_mean_final_position_norm=" << summary.mean_final_position_norm << "\n";
    std::cout << "gpu_eval_mean_final_velocity_norm=" << summary.mean_final_velocity_norm << "\n";
    std::cout << "gpu_eval_mean_final_attitude_error=" << summary.mean_final_attitude_error << "\n";
    std::cout << "gpu_eval_mean_final_angular_velocity_norm=" << summary.mean_final_angular_velocity_norm << "\n";
    std::cout << "gpu_eval_mean_max_position_norm=" << summary.mean_max_position_norm << "\n";
    std::cout << "gpu_eval_mean_max_velocity_norm=" << summary.mean_max_velocity_norm << "\n";
    std::cout << "gpu_eval_mean_max_attitude_error=" << summary.mean_max_attitude_error << "\n";
    std::cout << "gpu_eval_mean_max_angular_velocity_norm=" << summary.mean_max_angular_velocity_norm << "\n";
    std::cout << "gpu_eval_max_position_norm=" << summary.max_position_norm << "\n";
    std::cout << "gpu_eval_max_velocity_norm=" << summary.max_velocity_norm << "\n";
    std::cout << "gpu_eval_max_attitude_error=" << summary.max_attitude_error << "\n";
    std::cout << "gpu_eval_max_angular_velocity_norm=" << summary.max_angular_velocity_norm << "\n";
    std::cout << "gpu_eval_mean_action_magnitude=" << summary.mean_action_magnitude << "\n";
    std::cout << "gpu_eval_max_action_magnitude=" << summary.max_action_magnitude << "\n";
    std::cout << "gpu_eval_mean_action_smoothness=" << summary.mean_action_smoothness << "\n";
    std::cout << "gpu_eval_max_action_smoothness=" << summary.max_action_smoothness << "\n";
    std::cout << "gpu_eval_max_action_abs=" << summary.max_action_abs << "\n";
    std::cout << "gpu_eval_action_saturation_rate=" << summary.action_saturation_rate << "\n";
    std::cout << "gpu_eval_sample_dynamics=" << (summary.sample_dynamics ? "true" : "false") << "\n";
    std::cout << "gpu_eval_correlated_size_mass_sampling=" << (summary.correlated_size_mass_sampling ? "true" : "false") << "\n";
    std::cout << "gpu_eval_mass_min=" << summary.mass_min << "\n";
    std::cout << "gpu_eval_mass_mean=" << summary.mass_mean << "\n";
    std::cout << "gpu_eval_mass_max=" << summary.mass_max << "\n";
    std::cout << "gpu_eval_thrust_to_weight_min=" << summary.thrust_to_weight_min << "\n";
    std::cout << "gpu_eval_thrust_to_weight_mean=" << summary.thrust_to_weight_mean << "\n";
    std::cout << "gpu_eval_thrust_to_weight_max=" << summary.thrust_to_weight_max << "\n";
    std::cout << "gpu_eval_motor_delay_min=" << summary.motor_delay_min << "\n";
    std::cout << "gpu_eval_motor_delay_mean=" << summary.motor_delay_mean << "\n";
    std::cout << "gpu_eval_motor_delay_max=" << summary.motor_delay_max << "\n";
    std::cout << "gpu_eval_nan_inf_count=" << summary.nan_inf_count << "\n";
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

void print_local_initial_condition_validation(const gpu::LocalInitialConditionValidationSummary& summary){
    std::cout << "local_initial_condition_validation_passed=" << (summary.passed ? "true" : "false") << "\n";
    std::cout << "local_initial_condition_position_ok=" << (summary.position_ok ? "true" : "false") << "\n";
    std::cout << "local_initial_condition_velocity_ok=" << (summary.velocity_ok ? "true" : "false") << "\n";
    std::cout << "local_initial_condition_attitude_ok=" << (summary.attitude_ok ? "true" : "false") << "\n";
    std::cout << "local_initial_condition_angular_velocity_ok=" << (summary.angular_velocity_ok ? "true" : "false") << "\n";
    std::cout << "local_initial_condition_finite=" << (summary.finite ? "true" : "false") << "\n";
    std::cout << "local_initial_condition_samples=" << summary.samples << "\n";
    std::cout << "local_initial_condition_horizon=" << summary.horizon << "\n";
    std::cout << "local_initial_condition_position_threshold=" << summary.position_threshold << "\n";
    std::cout << "local_initial_condition_velocity_threshold=" << summary.velocity_threshold << "\n";
    std::cout << "local_initial_condition_attitude_threshold=" << summary.attitude_threshold << "\n";
    std::cout << "local_initial_condition_angular_velocity_threshold=" << summary.angular_velocity_threshold << "\n";
    std::cout << "local_initial_condition_max_position_error=" << summary.max_position_error << "\n";
    std::cout << "local_initial_condition_max_velocity_error=" << summary.max_velocity_error << "\n";
    std::cout << "local_initial_condition_max_attitude_error=" << summary.max_attitude_error << "\n";
    std::cout << "local_initial_condition_max_angular_velocity_norm=" << summary.max_angular_velocity_norm << "\n";
    std::cout << "local_initial_condition_nan_inf_count=" << summary.nan_inf_count << "\n";
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
        options.initial_attitude_scale = options.initial_attitude_scale_local;
        options.initial_angular_velocity_scale = options.initial_angular_velocity_scale_local;
        options.throughout_gate_start_step = 0;
    }
    const bool active_training_requested = options.steps > 0;
    if(active_training_requested && options.trajectory_mode != gpu::TrajectoryMode::FIXED){
        std::cerr << "Active CUDA training is origin recovery only. "
                  << "Use non-fixed --trajectory-mode only for eval/deployment setpoint shifting.\n";
        return 1;
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
    std::cout << "full_gpu_training_enabled=" << (options.steps > 0 ? "true" : "false") << "\n";
    std::cout << "active_training_task=origin_recovery_position_controller\n";
    std::cout << "active_training_default_horizon=" << ACTIVE_TRAINING_HORIZON << "\n";
    std::cout << "active_training_requested_horizon=" << options.horizon << "\n";
    std::cout << "setpoint_shifting_for_eval_and_deployment=true\n";
    std::cout << "eval_only=" << (options.eval_only ? "true" : "false") << "\n";
    std::cout << "eval_model=" << options.eval_model << "\n";
    std::cout << "failure_analysis=" << (options.failure_analysis ? "true" : "false") << "\n";
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
    std::cout << "training_reference_mode=origin\n";
    std::cout << "trajectory_amplitude=" << options.trajectory_amplitude << "\n";
    std::cout << "trajectory_frequency_hz=" << options.trajectory_frequency_hz << "\n";
    std::cout << "training_episode_steps=" << options.training_episode_steps << "\n";
    std::cout << "tracking_gate_mode=" << options.tracking_gate_mode << "\n";
    std::cout << "initial_position_scale=" << options.initial_position_scale << "\n";
    std::cout << "initial_velocity_scale=" << options.initial_velocity_scale << "\n";
    std::cout << "initial_attitude_scale=" << options.initial_attitude_scale << "\n";
    std::cout << "initial_attitude_scale_local=" << options.initial_attitude_scale_local << "\n";
    std::cout << "initial_angular_velocity_scale=" << options.initial_angular_velocity_scale << "\n";
    std::cout << "near_zero_guidance_probability=" << options.near_zero_guidance_probability << "\n";
    std::cout << "simulation_frequency_hz=100\n";
    std::cout << "dt=0.01\n";
    std::cout << "sample_dynamics=" << (options.sample_dynamics ? "true" : "false") << "\n";
    std::cout << "learning_rate=" << options.learning_rate << "\n";
    std::cout << "action_magnitude_weight=" << options.action_magnitude_weight << "\n";
    std::cout << "action_magnitude_center=" << options.action_magnitude_center << "\n";
    std::cout << "hover_relative_action_magnitude=" << (options.hover_relative_action_magnitude ? "true" : "false") << "\n";
    std::cout << "action_smoothness_weight=" << options.action_smoothness_weight << "\n";
    std::cout << "action_saturation_weight=" << options.action_saturation_weight << "\n";
    std::cout << "action_saturation_start=" << options.action_saturation_start << "\n";
    std::cout << "loss_velocity_weight=" << options.loss_velocity_weight << "\n";
    std::cout << "velocity_reference_gain=" << options.velocity_reference_gain << "\n";
    std::cout << "progress_weight=" << options.progress_weight << "\n";
    std::cout << "outward_velocity_weight=" << options.outward_velocity_weight << "\n";
    std::cout << "clf_weight=" << options.clf_weight << "\n";
    std::cout << "window_clf_weight=" << options.window_clf_weight << "\n";
    std::cout << "clf_alpha=" << options.clf_alpha << "\n";
    std::cout << "clf_position_weight=" << options.clf_position_weight << "\n";
    std::cout << "clf_velocity_weight=" << options.clf_velocity_weight << "\n";
    std::cout << "clf_attitude_weight=" << options.clf_attitude_weight << "\n";
    std::cout << "clf_angular_velocity_weight=" << options.clf_angular_velocity_weight << "\n";
    std::cout << "clf_position_velocity_cross_beta=" << options.clf_position_velocity_cross_beta << "\n";
    std::cout << "clf_attitude_angular_velocity_cross_beta=" << options.clf_attitude_angular_velocity_cross_beta << "\n";
    std::cout << "window_clf_epsilon=" << options.window_clf_epsilon << "\n";
    std::cout << "window_clf_huber_delta=" << options.window_clf_huber_delta << "\n";
    std::cout << "velocity_barrier_weight=" << options.velocity_barrier_weight << "\n";
    std::cout << "velocity_barrier_safe=" << options.velocity_barrier_safe << "\n";
    std::cout << "angular_velocity_barrier_weight=" << options.angular_velocity_barrier_weight << "\n";
    std::cout << "angular_velocity_barrier_safe=" << options.angular_velocity_barrier_safe << "\n";
    std::cout << "attitude_barrier_weight=" << options.attitude_barrier_weight << "\n";
    std::cout << "attitude_barrier_safe=" << options.attitude_barrier_safe << "\n";
    std::cout << "attitude_control_weight=" << options.attitude_control_weight << "\n";
    std::cout << "attitude_control_k_R=" << options.attitude_control_k_R << "\n";
    std::cout << "attitude_control_k_omega=" << options.attitude_control_k_omega << "\n";
    std::cout << "loss_angular_velocity_weight=" << options.loss_angular_velocity_weight << "\n";
    std::cout << "loss_linear_acceleration_weight=" << options.loss_linear_acceleration_weight << "\n";
    std::cout << "loss_angular_acceleration_weight=" << options.loss_angular_acceleration_weight << "\n";
    std::cout << "terminal_loss_scale=" << options.terminal_loss_scale << "\n";
    std::cout << "terminal_velocity_weight=" << options.terminal_velocity_weight << "\n";
    std::cout << "terminal_angular_velocity_weight=" << options.terminal_angular_velocity_weight << "\n";
    std::cout << "diff_rollout_loss_weight=" << options.diff_rollout_loss_weight << "\n";
    std::cout << "temporal_gradient_decay_alpha=" << options.temporal_gradient_decay_alpha << "\n";
    std::cout << "velocity_observation_noise=" << options.velocity_observation_noise << "\n";
    std::cout << "velocity_observation_delay_steps=" << options.velocity_observation_delay_steps << "\n";
    std::cout << "persistent_episode_training=" << (options.persistent_episode_training ? "true" : "false") << "\n";
    std::cout << "action_grad_clip_enabled=" << (options.action_grad_clip_enabled ? "true" : "false") << "\n";
    std::cout << "action_grad_clip_norm=" << options.action_grad_clip_norm << "\n";
    std::cout << "actor_grad_clip_enabled=" << (options.actor_grad_clip_enabled ? "true" : "false") << "\n";
    std::cout << "actor_grad_clip_norm=" << options.actor_grad_clip_norm << "\n";
    std::cout << "actor_grad_skip_norm=" << options.actor_grad_skip_norm << "\n";
    std::cout << "disable_physics_gradient=" << (options.disable_physics_gradient ? "true" : "false") << "\n";
    std::cout << "reset_hidden_each_step=" << (options.reset_hidden_each_step ? "true" : "false") << "\n";
    std::cout << "load_optimizer_state=" << (options.load_optimizer_state ? "true" : "false") << "\n";

    gpu::EulerGpuBatch batch;
    gpu::EulerGpuLossWeights weights;
    weights.action_magnitude = options.action_magnitude_weight;
    weights.action_magnitude_center = options.action_magnitude_center;
    weights.action_smoothness = options.action_smoothness_weight;
    weights.saturation = options.action_saturation_weight;
    weights.saturation_start = options.action_saturation_start;
    weights.velocity = options.loss_velocity_weight;
    weights.velocity_reference_gain = options.velocity_reference_gain;
    weights.progress = options.progress_weight;
    weights.outward_velocity = options.outward_velocity_weight;
    weights.clf = options.clf_weight;
    weights.window_clf = options.window_clf_weight;
    weights.clf_alpha = options.clf_alpha;
    weights.clf_position = options.clf_position_weight;
    weights.clf_velocity = options.clf_velocity_weight;
    weights.clf_attitude = options.clf_attitude_weight;
    weights.clf_angular_velocity = options.clf_angular_velocity_weight;
    weights.clf_position_velocity_cross_beta = options.clf_position_velocity_cross_beta;
    weights.clf_attitude_angular_velocity_cross_beta = options.clf_attitude_angular_velocity_cross_beta;
    weights.window_clf_epsilon = options.window_clf_epsilon;
    weights.window_clf_huber_delta = options.window_clf_huber_delta;
    weights.velocity_barrier = options.velocity_barrier_weight;
    weights.velocity_barrier_safe = options.velocity_barrier_safe;
    weights.angular_velocity_barrier = options.angular_velocity_barrier_weight;
    weights.angular_velocity_barrier_safe = options.angular_velocity_barrier_safe;
    weights.attitude_barrier = options.attitude_barrier_weight;
    weights.attitude_barrier_safe = options.attitude_barrier_safe;
    weights.hover_relative_action_magnitude = options.hover_relative_action_magnitude;
    weights.attitude_control = options.attitude_control_weight;
    weights.attitude_control_k_R = options.attitude_control_k_R;
    weights.attitude_control_k_omega = options.attitude_control_k_omega;
    weights.angular_velocity = options.loss_angular_velocity_weight;
    weights.linear_acceleration = options.loss_linear_acceleration_weight;
    weights.angular_acceleration = options.loss_angular_acceleration_weight;
    weights.terminal_loss_scale = options.terminal_loss_scale;
    weights.terminal_velocity = options.terminal_velocity_weight;
    weights.terminal_angular_velocity = options.terminal_angular_velocity_weight;
    gpu::EulerGpuRunOptions run_options;
    run_options.device = options.gpu_device;
    run_options.compute_action_gradients = true;
    run_options.temporal_gradient_decay_alpha = options.temporal_gradient_decay_alpha;
    run_options.clf_weight = options.clf_weight;
    run_options.window_clf_weight = options.window_clf_weight;
    run_options.clf_alpha = options.clf_alpha;
    run_options.clf_position = options.clf_position_weight;
    run_options.clf_velocity = options.clf_velocity_weight;
    run_options.clf_attitude = options.clf_attitude_weight;
    run_options.clf_angular_velocity = options.clf_angular_velocity_weight;
    run_options.clf_position_velocity_cross_beta = options.clf_position_velocity_cross_beta;
    run_options.clf_attitude_angular_velocity_cross_beta = options.clf_attitude_angular_velocity_cross_beta;
    run_options.window_clf_epsilon = options.window_clf_epsilon;
    run_options.window_clf_huber_delta = options.window_clf_huber_delta;
    run_options.velocity_barrier_weight = options.velocity_barrier_weight;
    run_options.velocity_barrier_safe = options.velocity_barrier_safe;
    run_options.angular_velocity_barrier_weight = options.angular_velocity_barrier_weight;
    run_options.angular_velocity_barrier_safe = options.angular_velocity_barrier_safe;
    run_options.attitude_barrier_weight = options.attitude_barrier_weight;
    run_options.attitude_barrier_safe = options.attitude_barrier_safe;
    run_options.outward_velocity_weight = options.outward_velocity_weight;
    run_options.attitude_control_weight = options.attitude_control_weight;
    run_options.attitude_control_k_R = options.attitude_control_k_R;
    run_options.attitude_control_k_omega = options.attitude_control_k_omega;
    run_options.velocity_observation_noise = options.velocity_observation_noise;
    run_options.velocity_observation_delay_steps = options.velocity_observation_delay_steps;
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
        options.initial_angular_velocity_scale,
        options.initial_attitude_scale,
        options.near_zero_guidance_probability,
        options.training_episode_steps,
        parse_dynamics_randomization_level(options.sampled_dynamics_level)
    );
    if(options.hover_relative_action_magnitude){
        std::cout << "hover_relative_action_magnitude_center=per_sample_initial_previous_action\n";
    }

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
        if(options.eval_only){
            auto eval_options = make_gpu_eval_options(options);
            auto eval_summary = gpu::run_gpu_policy_eval(eval_options, weights);
            print_gpu_eval_summary(eval_summary);
            if(!eval_summary.passed){
                return 1;
            }
            return 0;
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
        if(options.validate_local_initial_conditions){
            auto local_initial_validation = gpu::validate_local_initial_conditions(
                options.batch_size,
                options.horizon,
                options.seed,
                options.trajectory_mode,
                options.trajectory_amplitude,
                options.trajectory_frequency_hz,
                options.correlated_size_mass_sampling,
                options.initial_position_scale,
                options.initial_velocity_scale,
                options.initial_angular_velocity_scale,
                options.initial_attitude_scale,
                options.near_zero_guidance_probability
            );
            print_local_initial_condition_validation(local_initial_validation);
            if(!local_initial_validation.passed){
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
        if(options.diagnose_objective_gradient_conflicts){
            auto conflict_summary = gpu::diagnose_objective_gradient_conflicts(
                options.batch_size,
                options.horizon,
                options.seed,
                weights,
                run_options,
                options.diff_rollout_loss_weight
            );
            print_objective_gradient_conflict_diagnostic(conflict_summary);
            if(!conflict_summary.passed){
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
           !options.validate_critic_backward && !options.diagnose_objective_gradient_conflicts &&
           !options.validate_adam_update &&
           !options.validate_correlated_size_mass_sampler && !options.validate_trajectory_sampler &&
           !options.validate_local_initial_conditions &&
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
