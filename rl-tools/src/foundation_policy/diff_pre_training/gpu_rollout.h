#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rl_tools::foundation_policy::diff_pre_training::gpu{

constexpr std::size_t EULER_OBSERVATION_DIM = 22;
constexpr std::size_t RDAC_POLICY_INPUT_DIM = 48;
constexpr std::size_t RDAC_HIDDEN_DIM = 16;
constexpr std::size_t RDAC_ACTION_DIM = 4;
constexpr std::size_t RDAC_ACTOR_HEAD_INPUT_DIM = EULER_OBSERVATION_DIM + RDAC_HIDDEN_DIM;
constexpr std::size_t RDAC_CRITIC_DIM = 1;
constexpr std::size_t RDAC_CRITIC_HEAD_INPUT_DIM = EULER_OBSERVATION_DIM + RDAC_HIDDEN_DIM;
constexpr float RDAC_CRITIC_LOSS_WEIGHT = 0.10f;

enum class TrajectoryMode : std::uint32_t{
    FIXED = 0,
    STEP = 1,
    CIRCLE = 2,
    FIGURE8 = 3,
    MIXED = 4
};

struct EulerGpuLossWeights{
    float position = 8.0f;
    float velocity = 0.8f;
    float attitude = 4.0f;
    float angular_velocity = 0.8f;
    float action_magnitude = 0.005f;
    float action_smoothness = 0.03f;
    float saturation = 0.05f;
    float saturation_start = 0.95f;
    float terminal_loss_weight = 4.0f;
    float terminal_loss_scale = 1.0f;
    float terminal_position = 12.0f;
    float terminal_velocity = 4.0f;
    float terminal_attitude = 8.0f;
    float terminal_angular_velocity = 4.0f;
};

struct EulerGpuTimings{
    float host_to_device_ms = 0.0f;
    float forward_ms = 0.0f;
    float loss_ms = 0.0f;
    float backward_vjp_ms = 0.0f;
    float device_to_host_ms = 0.0f;
    float total_ms = 0.0f;
};

struct EulerGpuBatch{
    std::size_t batch_size = 0;
    std::size_t horizon = 0;

    std::vector<float> initial_p;               // [B, 3]
    std::vector<float> initial_v;               // [B, 3]
    std::vector<float> initial_R;               // [B, 9]
    std::vector<float> initial_omega;           // [B, 3]
    std::vector<float> initial_rpm;             // [B, 4]
    std::vector<float> initial_previous_action; // [B, 4]
    std::vector<float> reference_p;             // [B, 3], fixed-reference compatibility metadata
    std::vector<float> reference_v;             // [B, 3], fixed-reference compatibility metadata
    std::vector<float> reference_p_traj;        // [H + 1, B, 3]
    std::vector<float> reference_v_traj;        // [H + 1, B, 3]
    std::vector<float> actions;                 // [H, B, 4]

    std::vector<float> mass;                    // [B]
    std::vector<float> gravity;                 // [B, 3]
    std::vector<float> J;                       // [B, 9]
    std::vector<float> J_inv;                   // [B, 9]
    std::vector<float> rotor_positions;         // [B, 4, 3]
    std::vector<float> rotor_thrust_directions; // [B, 4, 3]
    std::vector<float> rotor_torque_directions; // [B, 4, 3]
    std::vector<float> rotor_torque_constants;  // [B, 4]
    std::vector<float> rotor_time_rising;       // [B, 4]
    std::vector<float> rotor_time_falling;      // [B, 4]
    std::vector<float> rotor_thrust_coeffs;     // [B, 4, 3]
    std::vector<float> action_min;              // [B]
    std::vector<float> action_max;              // [B]
    std::vector<std::uint32_t> dynamics_size_mass_bin;          // [B]
    std::vector<std::uint32_t> dynamics_thrust_to_weight_bin;   // [B]
    std::vector<std::uint32_t> dynamics_torque_to_inertia_bin;  // [B]
    std::vector<std::uint32_t> dynamics_motor_delay_bin;        // [B]
    std::vector<std::uint32_t> dynamics_curve_shape_bin;        // [B]
    std::vector<std::uint32_t> dynamics_group_key;              // [B]
    std::vector<std::uint32_t> rejected_before_accept;          // [B]
    std::vector<float> group_weight;                            // [B]
    std::vector<std::uint32_t> reset_mask;                      // [H, B]
    std::vector<std::uint32_t> hidden_reset_mask;               // [H, B]
    std::uint32_t replay_schema_version = 2;
    std::uint32_t sampler_seed = 0;
    std::uint32_t sampler_balance_bins = 4;
    std::uint32_t hidden_reset_enabled = 0;
    std::uint32_t trajectory_mode = static_cast<std::uint32_t>(TrajectoryMode::FIXED);

    void resize(std::size_t new_batch_size, std::size_t new_horizon);
};

struct EulerGpuResult{
    std::vector<float> final_p;          // [B, 3]
    std::vector<float> final_v;          // [B, 3]
    std::vector<float> final_R;          // [B, 9]
    std::vector<float> final_omega;      // [B, 3]
    std::vector<float> final_rpm;        // [B, 4]
    std::vector<float> loss;             // [B]
    std::vector<float> action_gradients; // [H, B, 4]

    void resize(std::size_t batch_size, std::size_t horizon);
};

struct DeploymentAdapterInput{
    float p[3] = {};
    float v[3] = {};
    float R[9] = {};
    float omega[3] = {};
    float previous_action[4] = {};
    float reference_p[3] = {};
    float reference_v[3] = {};
};

void build_deployment_observation(
    const DeploymentAdapterInput& input,
    float observation[EULER_OBSERVATION_DIM]
);

struct EulerGpuRunOptions{
    int device = 0;
    bool compute_action_gradients = true;
};

struct ForcedDynamicsBins{
    bool enabled = false;
    std::uint32_t size_mass = 0;
    std::uint32_t thrust_to_weight = 0;
    std::uint32_t torque_to_inertia = 0;
    std::uint32_t motor_delay = 0;
    std::uint32_t curve_shape = 0;
};

struct ValidationSummary{
    float max_forward_abs_error = 0.0f;
    float max_rpm_abs_error = 0.0f;
    float max_rpm_rel_error = 0.0f;
    float max_loss_abs_error = 0.0f;
    float max_loss_rel_error = 0.0f;
    float max_action_gradient_abs_error = 0.0f;
    float max_action_gradient_rel_error = 0.0f;
    float action_gradient_l2_rel_error = 0.0f;
    bool forward_close = false;
    bool loss_close = false;
    bool action_gradient_close = false;
    bool passed = false;
};

struct ObservationValidationSummary{
    float max_abs_error = 0.0f;
    float mean_abs_error = 0.0f;
    std::size_t nan_inf_count = 0;
    bool passed = false;
};

struct ActorForwardValidationSummary{
    float max_raw_action_abs_error = 0.0f;
    float max_bounded_action_abs_error = 0.0f;
    float max_hidden_abs_error = 0.0f;
    float max_critic_output_abs_error = 0.0f;
    std::size_t nan_inf_count = 0;
    bool raw_action_close = false;
    bool bounded_action_close = false;
    bool hidden_close = false;
    bool critic_output_close = false;
    bool critic_checked = false;
    bool passed = false;
};

struct ClosedLoopValidationSummary{
    float max_state_abs_error = 0.0f;
    float max_rpm_abs_error = 0.0f;
    float max_rpm_rel_error = 0.0f;
    float max_action_abs_error = 0.0f;
    float max_hidden_abs_error = 0.0f;
    float max_loss_abs_error = 0.0f;
    float max_loss_rel_error = 0.0f;
    std::size_t nan_inf_count = 0;
    bool state_close = false;
    bool action_close = false;
    bool hidden_close = false;
    bool loss_close = false;
    bool passed = false;
};

struct ActionGradientInjectionValidationSummary{
    float max_action_gradient_abs_error = 0.0f;
    float max_action_gradient_rel_error = 0.0f;
    float action_gradient_l2_rel_error = 0.0f;
    float max_raw_action_gradient_abs_error = 0.0f;
    float max_raw_action_gradient_rel_error = 0.0f;
    float raw_action_gradient_l2_rel_error = 0.0f;
    float max_action_derivative_abs_error = 0.0f;
    std::size_t nan_inf_count = 0;
    std::size_t clamped_action_count = 0;
    std::size_t clamp_zero_violation_count = 0;
    bool action_gradient_close = false;
    bool raw_action_gradient_close = false;
    bool derivative_close = false;
    bool clamp_zero_ok = false;
    bool passed = false;
};

struct ActorBackwardValidationSummary{
    float encoder_grad_norm = 0.0f;
    float gru_input_grad_norm = 0.0f;
    float gru_hidden_grad_norm = 0.0f;
    float actor_head_grad_norm = 0.0f;
    float h0_grad_norm = 0.0f;
    float encoder_grad_cosine = 0.0f;
    float gru_input_grad_cosine = 0.0f;
    float gru_hidden_grad_cosine = 0.0f;
    float actor_head_grad_cosine = 0.0f;
    float h0_grad_cosine = 0.0f;
    float max_abs_error = 0.0f;
    float l2_rel_error = 0.0f;
    std::size_t nan_inf_count = 0;
    bool finite = false;
    bool nonzero = false;
    bool cosine_close = false;
    bool passed = false;
};

struct CriticBackwardValidationSummary{
    float encoder_grad_norm = 0.0f;
    float gru_input_grad_norm = 0.0f;
    float gru_hidden_grad_norm = 0.0f;
    float critic_head_grad_norm = 0.0f;
    float actor_head_grad_norm = 0.0f;
    float h0_grad_norm = 0.0f;
    float encoder_grad_cosine = 0.0f;
    float gru_input_grad_cosine = 0.0f;
    float gru_hidden_grad_cosine = 0.0f;
    float critic_head_grad_cosine = 0.0f;
    float h0_grad_cosine = 0.0f;
    float max_abs_error = 0.0f;
    float l2_rel_error = 0.0f;
    std::size_t nan_inf_count = 0;
    bool finite = false;
    bool nonzero = false;
    bool cosine_close = false;
    bool actor_head_zero = false;
    bool passed = false;
};

struct AdamUpdateValidationSummary{
    float max_weight_abs_error = 0.0f;
    float max_weight_rel_error = 0.0f;
    float max_first_moment_abs_error = 0.0f;
    float max_first_moment_rel_error = 0.0f;
    float max_second_moment_abs_error = 0.0f;
    float max_second_moment_rel_error = 0.0f;
    std::size_t nan_inf_count = 0;
    bool weights_close = false;
    bool moments_close = false;
    bool step_close = false;
    bool passed = false;
};

struct FullGpuTrainingOptions{
    int device = 0;
    std::size_t batch_size = 128;
    std::size_t horizon = 16;
    std::size_t steps = 0;
    unsigned seed = 0;
    float learning_rate = 1e-4f;
    float diff_rollout_loss_weight = 5e-4f;
    bool action_grad_clip_enabled = false;
    float action_grad_clip_norm = 100.0f;
    bool actor_grad_clip_enabled = true;
    float actor_grad_clip_norm = 100.0f;
    float actor_grad_skip_norm = 1e12f;
    float actor_grad_eps = 1e-6f;
    bool disable_physics_gradient = false;
    bool reset_hidden_each_step = false;
    bool sample_dynamics = true;
    bool load_optimizer_state = true;
    bool correlated_size_mass_sampling = false;
    TrajectoryMode trajectory_mode = TrajectoryMode::FIXED;
    float trajectory_amplitude = 0.03f;
    float trajectory_frequency_hz = 0.5f;
    float initial_position_scale = 1.0f;
    float initial_velocity_scale = 1.0f;
    float initial_angular_velocity_scale = 1.0f;
    float success_position_threshold = 1.0f;
    float success_velocity_threshold = 2.0f;
    float success_attitude_threshold = 3.14159265358979323846f;
    float success_angular_velocity_threshold = 5.0f;
    float success_action_saturation_threshold = 1.0f;
    std::string log_path;
    std::string save_path;
    std::string load_path;
};

struct FullGpuTrainingSummary{
    float final_loss = 0.0f;
    float final_grad_norm = 0.0f;
    float final_critic_loss = 0.0f;
    float final_critic_error_norm = 0.0f;
    float final_critic_output_mean = 0.0f;
    float final_critic_target_mean = 0.0f;
    float final_success_rate = 0.0f;
    float final_action_saturation = 0.0f;
    float final_position_norm_mean = 0.0f;
    float final_velocity_norm_mean = 0.0f;
    float final_attitude_error_mean = 0.0f;
    float final_angular_velocity_norm_mean = 0.0f;
    std::size_t nan_inf_count = 0;
    bool finite = false;
    bool checkpoint_saved = false;
    bool checkpoint_loaded = false;
    bool passed = false;
};

struct GpuPolicyEvalOptions{
    int device = 0;
    std::size_t episodes = 100;
    std::size_t horizon = 128;
    unsigned seed = 0;
    bool sample_dynamics = true;
    bool reset_hidden_each_step = false;
    bool correlated_size_mass_sampling = false;
    TrajectoryMode trajectory_mode = TrajectoryMode::FIXED;
    float trajectory_amplitude = 0.03f;
    float trajectory_frequency_hz = 0.5f;
    float initial_position_scale = 1.0f;
    float initial_velocity_scale = 1.0f;
    float initial_angular_velocity_scale = 1.0f;
    float success_position_threshold = 1.0f;
    float success_velocity_threshold = 2.0f;
    float success_attitude_threshold = 3.14159265358979323846f;
    float success_angular_velocity_threshold = 5.0f;
    float success_action_saturation_threshold = 1.0f;
    ForcedDynamicsBins forced_bins;
    std::string load_path;
    std::string log_path;
};

struct GpuPolicyEvalSummary{
    float success_rate = 0.0f;
    float near_success_rate_p = 0.0f;
    float near_success_rate_pv = 0.0f;
    float mean_total_loss = 0.0f;
    float mean_final_position_norm = 0.0f;
    float mean_final_velocity_norm = 0.0f;
    float mean_final_attitude_error = 0.0f;
    float mean_final_angular_velocity_norm = 0.0f;
    float position_rmse = 0.0f;
    float velocity_rmse = 0.0f;
    float attitude_rmse = 0.0f;
    float angular_velocity_rmse = 0.0f;
    float median_final_position_norm = 0.0f;
    float median_final_velocity_norm = 0.0f;
    float median_final_attitude_error = 0.0f;
    float median_final_angular_velocity_norm = 0.0f;
    float p90_final_position_norm = 0.0f;
    float p90_final_velocity_norm = 0.0f;
    float p90_final_attitude_error = 0.0f;
    float p90_final_angular_velocity_norm = 0.0f;
    float throughout_success_rate = 0.0f;
    float mean_time_inside_fraction = 0.0f;
    float mean_first_failure_time_s = 0.0f;
    float mean_max_position_norm = 0.0f;
    float mean_max_velocity_norm = 0.0f;
    float mean_max_attitude_error = 0.0f;
    float mean_max_angular_velocity_norm = 0.0f;
    float p90_max_position_norm = 0.0f;
    float p90_max_velocity_norm = 0.0f;
    float p90_max_attitude_error = 0.0f;
    float p90_max_angular_velocity_norm = 0.0f;
    float max_action_abs = 0.0f;
    float action_saturation_rate = 0.0f;
    float invalid_or_nan_rate = 0.0f;
    std::size_t nan_inf_count = 0;
    bool checkpoint_loaded = false;
    bool finite = false;
    bool stability_gate_passed = false;
    bool passed = false;
};

struct Stage9ReplayDebugOptions{
    int device = 0;
    std::size_t batch_size = 64;
    std::size_t horizon = 16;
    std::size_t steps = 10;
    unsigned seed = 0;
    float learning_rate = 1e-4f;
    float diff_rollout_loss_weight = 5e-4f;
    bool action_grad_clip_enabled = false;
    float action_grad_clip_norm = 100.0f;
    bool actor_grad_clip_enabled = true;
    float actor_grad_clip_norm = 100.0f;
    float actor_grad_skip_norm = 1e12f;
    float actor_grad_eps = 1e-6f;
    bool disable_physics_gradient = false;
    bool reset_hidden_each_step = false;
    std::string replay_path;
    std::string log_path;
};

struct Stage9ReplayDebugSummary{
    float final_cpu_loss = 0.0f;
    float final_gpu_loss = 0.0f;
    float max_loss_abs_error = 0.0f;
    float max_action_mean_abs_error = 0.0f;
    float max_action_std_abs_error = 0.0f;
    float max_action_saturation_abs_error = 0.0f;
    float max_action_gradient_norm_abs_error = 0.0f;
    float max_actor_gradient_norm_abs_error = 0.0f;
    float max_weight_abs_error = 0.0f;
    float max_weight_l2_rel_error = 0.0f;
    float max_adam_m_norm_abs_error = 0.0f;
    float max_adam_v_norm_abs_error = 0.0f;
    std::size_t nan_inf_count = 0;
    bool replay_written = false;
    bool replay_loaded = false;
    bool finite = false;
    bool close = false;
    bool passed = false;
};

struct Stage9SamplerParitySummary{
    std::size_t samples = 0;
    std::size_t groups = 0;
    std::size_t rejected_total = 0;
    std::size_t reset_mask_count = 0;
    std::size_t hidden_reset_mask_count = 0;
    std::size_t metadata_mismatch_count = 0;
    std::size_t nan_inf_count = 0;
    float group_weight_sum_min = 0.0f;
    float group_weight_sum_max = 0.0f;
    bool replay_written = false;
    bool replay_loaded = false;
    bool metadata_present = false;
    bool bins_balanced = false;
    bool group_weights_close = false;
    bool reset_masks_replayed = false;
    bool finite = false;
    bool passed = false;
};

struct CorrelatedSizeMassSamplerValidationSummary{
    std::size_t samples = 0;
    std::size_t formula_mismatch_count = 0;
    std::size_t nan_inf_count = 0;
    float mass_min = 0.0f;
    float mass_max = 0.0f;
    float size_factor_min = 0.0f;
    float size_factor_max = 0.0f;
    float max_default_batch_abs_error = 0.0f;
    float max_nominal_abs_error = 0.0f;
    float max_size_factor_bounds_error = 0.0f;
    float max_thrust_factor_abs_error = 0.0f;
    float max_thrust_to_weight_bounds_error = 0.0f;
    float max_inertia_bounds_error = 0.0f;
    float max_j_inverse_abs_error = 0.0f;
    bool default_disabled = false;
    bool fixed_nominal_unchanged = false;
    bool correlated_formula_close = false;
    bool finite = false;
    bool passed = false;
};

struct TrajectorySamplerValidationSummary{
    std::size_t samples = 0;
    std::size_t horizon = 0;
    std::size_t nan_inf_count = 0;
    std::uint32_t mixed_mode_coverage_mask = 0u;
    float fixed_reference_max_abs = 0.0f;
    float deterministic_max_abs = 0.0f;
    float max_reference_abs = 0.0f;
    float max_reference_velocity_abs = 0.0f;
    bool fixed_backward_compatible = false;
    bool deterministic = false;
    bool finite = false;
    bool mixed_covers_all_modes = false;
    bool passed = false;
};

struct Stage9EvalParitySummary{
    float max_final_state_abs_error = 0.0f;
    float max_loss_abs_error = 0.0f;
    float cpu_success_rate = 0.0f;
    float gpu_success_rate = 0.0f;
    float cpu_action_saturation = 0.0f;
    float gpu_action_saturation = 0.0f;
    float mean_cpu_final_position_norm = 0.0f;
    float mean_gpu_final_position_norm = 0.0f;
    std::size_t success_mismatch_count = 0;
    std::size_t saturation_mismatch_count = 0;
    std::size_t nan_inf_count = 0;
    bool replay_written = false;
    bool replay_loaded = false;
    bool final_state_close = false;
    bool loss_close = false;
    bool success_close = false;
    bool action_saturation_close = false;
    bool finite = false;
    bool passed = false;
};

struct Stage9CheckpointParitySummary{
    float max_weight_abs_error = 0.0f;
    float max_first_moment_abs_error = 0.0f;
    float max_second_moment_abs_error = 0.0f;
    bool saved = false;
    bool loaded = false;
    bool metadata_ok = false;
    bool weights_close = false;
    bool moments_close = false;
    bool passed = false;
};

struct BenchmarkSummary{
    std::size_t batch_size = 0;
    std::size_t horizon = 0;
    int iterations = 0;
    double transitions_per_second = 0.0;
    double rollouts_per_second = 0.0;
    EulerGpuTimings mean_timings;
};

int run_euler_gpu_rollout(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options,
    EulerGpuResult& result,
    EulerGpuTimings& timings
);

void generate_validation_batch(
    EulerGpuBatch& batch,
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const ForcedDynamicsBins* forced_bins = nullptr,
    bool nominal_dynamics = false,
    bool correlated_size_mass_sampling = false,
    TrajectoryMode trajectory_mode = TrajectoryMode::FIXED,
    float trajectory_amplitude = 0.03f,
    float trajectory_frequency_hz = 0.5f,
    float initial_position_scale = 1.0f,
    float initial_velocity_scale = 1.0f,
    float initial_angular_velocity_scale = 1.0f
);

int assemble_observations_gpu(
    const EulerGpuBatch& batch,
    const EulerGpuRunOptions& options,
    std::size_t step_i,
    std::vector<float>& observations
);

ObservationValidationSummary validate_observations_against_cpu(
    const EulerGpuBatch& batch,
    const EulerGpuRunOptions& options,
    std::size_t step_i
);

ActorForwardValidationSummary validate_actor_forward_against_cpu(
    std::size_t batch_size,
    std::size_t sequence_length,
    unsigned seed,
    const EulerGpuRunOptions& options
);

ClosedLoopValidationSummary validate_closed_loop_against_cpu(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options
);

ActionGradientInjectionValidationSummary validate_action_gradient_injection_against_cpu(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options
);

ActorBackwardValidationSummary validate_actor_backward_against_cpu(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options
);

CriticBackwardValidationSummary validate_critic_backward_against_cpu(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const EulerGpuRunOptions& options
);

AdamUpdateValidationSummary validate_adam_update_against_cpu(
    unsigned seed,
    const EulerGpuRunOptions& options
);

FullGpuTrainingSummary run_full_gpu_training(
    const FullGpuTrainingOptions& training_options,
    const EulerGpuLossWeights& weights
);

GpuPolicyEvalSummary run_gpu_policy_eval(
    const GpuPolicyEvalOptions& eval_options,
    const EulerGpuLossWeights& weights
);

Stage9ReplayDebugSummary run_stage9_replay_debug(
    const Stage9ReplayDebugOptions& debug_options,
    const EulerGpuLossWeights& weights
);

Stage9SamplerParitySummary run_stage9_sampler_parity(
    const Stage9ReplayDebugOptions& debug_options
);

CorrelatedSizeMassSamplerValidationSummary validate_correlated_size_mass_sampler(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed
);

TrajectorySamplerValidationSummary validate_trajectory_sampler(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    float amplitude,
    float frequency_hz
);

Stage9EvalParitySummary run_stage9_eval_parity(
    const Stage9ReplayDebugOptions& debug_options,
    const EulerGpuLossWeights& weights
);

Stage9CheckpointParitySummary run_stage9_checkpoint_parity(
    const std::string& checkpoint_path,
    unsigned seed
);

ValidationSummary validate_against_cpu(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options
);

BenchmarkSummary benchmark_gpu_rollout(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options,
    int iterations
);

BenchmarkSummary benchmark_cpu_reference(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    int iterations
);

} // namespace rl_tools::foundation_policy::diff_pre_training::gpu
