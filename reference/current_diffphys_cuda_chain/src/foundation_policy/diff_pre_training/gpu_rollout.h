#pragma once

#include <array>
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
constexpr std::size_t LOSS_COMPONENT_COUNT = 24;

enum class TrajectoryMode : std::uint32_t{
    FIXED = 0,
    STEP = 1,
    CIRCLE = 2,
    FIGURE8 = 3,
    MIXED = 4
};

enum class DynamicsRandomizationLevel : std::uint32_t{
    BROAD = 0,
    SMALL = 1
};

enum class FailureReplayTriggerMode : std::uint32_t{
    ANY = 0,
    STATE = 1,
    ACTION = 2
};

struct EulerGpuLossWeights{
    float position = 8.0f;
    float velocity = 0.8f;
    float velocity_reference_gain = 0.0f;
    float progress = 0.0f;
    float outward_velocity = 0.0f;
    float attitude = 4.0f;
    float angular_velocity = 0.8f;
    float linear_acceleration = 0.0f;
    float angular_acceleration = 0.0f;
    float clf = 0.0f;
    float window_clf = 0.0f;
    float clf_alpha = 0.2f;
    float clf_position = 8.0f;
    float clf_velocity = 0.8f;
    float clf_attitude = 4.0f;
    float clf_angular_velocity = 0.8f;
    float clf_position_velocity_cross_beta = 0.0f;
    float clf_attitude_angular_velocity_cross_beta = 0.0f;
    float window_clf_epsilon = 1e-3f;
    float window_clf_huber_delta = 1.0f;
    float velocity_barrier = 0.0f;
    float velocity_barrier_safe = 10.0f;
    float angular_velocity_barrier = 0.0f;
    float angular_velocity_barrier_safe = 15.0f;
    float attitude_barrier = 0.0f;
    float attitude_barrier_safe = 4.0f;
    float barrier_huber_delta = 1.0f;
    float barrier_gradient_cap = 0.0f;
    float attitude_control = 0.0f;
    float attitude_control_k_R = 2.0f;
    float attitude_control_k_omega = 1.0f;
    float action_magnitude = 0.005f;
    float action_magnitude_center = 0.0f;
    bool hover_relative_action_magnitude = false;
    float action_smoothness = 0.03f;
    float saturation = 0.05f;
    float saturation_start = 0.95f;
    float replay_recovery_action_magnitude = 0.0f;
    float replay_recovery_saturation = 0.0f;
    float replay_recovery_linear_velocity = 0.0f;
    float replay_recovery_velocity_barrier = 0.0f;
    float replay_recovery_velocity_barrier_safe = 2.0f;
    float replay_recovery_angular_velocity = 0.0f;
    float replay_recovery_position_progress = 0.0f;
    float replay_recovery_progress_margin = 0.0f;
    float replay_recovery_progress_epsilon = 1e-3f;
    float replay_recovery_progress_huber_delta = 1.0f;
    std::uint32_t replay_recovery_start_segment = 1u; // zero-based replay segment index
    float terminal_loss_weight = 4.0f;
    float terminal_loss_scale = 0.0f;
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

struct LossComponentMeans{
    float position = 0.0f;
    float velocity = 0.0f;
    float attitude = 0.0f;
    float angular_velocity = 0.0f;
    float linear_acceleration = 0.0f;
    float angular_acceleration = 0.0f;
    float clf = 0.0f;
    float window_clf = 0.0f;
    float outward_velocity = 0.0f;
    float attitude_control = 0.0f;
    float velocity_barrier = 0.0f;
    float angular_velocity_barrier = 0.0f;
    float attitude_barrier = 0.0f;
    float action_magnitude = 0.0f;
    float action_smoothness = 0.0f;
    float saturation = 0.0f;
    float replay_recovery = 0.0f;
    float replay_recovery_progress = 0.0f;
    float replay_recovery_velocity_barrier = 0.0f;
    float terminal = 0.0f;
    float terminal_position = 0.0f;
    float terminal_velocity = 0.0f;
    float terminal_attitude = 0.0f;
    float terminal_angular_velocity = 0.0f;
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
    std::vector<float> action_hover_center;      // [B, 4], per-sample hover action center for action magnitude loss
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
    std::vector<std::uint32_t> trajectory_start_step;           // [B]
    std::vector<float> group_weight;                            // [B]
    std::vector<std::uint32_t> reset_mask;                      // [H, B]
    std::vector<std::uint32_t> hidden_reset_mask;               // [H, B]
    std::vector<std::uint32_t> failure_replay_segment_index;    // [B], 0 inactive, otherwise 1 + zero-based replay segment index
    std::uint32_t replay_schema_version = 4;
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

struct FailureReplaySample{
    std::array<float, 3> p = {};
    std::array<float, 3> v = {};
    std::array<float, 9> R = {};
    std::array<float, 3> omega = {};
    std::array<float, 4> rpm = {};
    std::array<float, 4> previous_action = {};
    std::array<float, 4> action_hover_center = {};
    std::array<float, RDAC_HIDDEN_DIM> hidden = {};
    std::array<float, 3> reference_p = {};
    std::array<float, 3> reference_v = {};
    std::array<float, 3> gravity = {};
    std::array<float, 9> J = {};
    std::array<float, 9> J_inv = {};
    std::array<float, 4 * 3> rotor_positions = {};
    std::array<float, 4 * 3> rotor_thrust_directions = {};
    std::array<float, 4 * 3> rotor_torque_directions = {};
    std::array<float, 4> rotor_torque_constants = {};
    std::array<float, 4> rotor_time_rising = {};
    std::array<float, 4> rotor_time_falling = {};
    std::array<float, 4 * 3> rotor_thrust_coeffs = {};
    float mass = 0.0f;
    float action_min = 0.0f;
    float action_max = 0.0f;
    std::uint32_t dynamics_size_mass_bin = 0;
    std::uint32_t dynamics_thrust_to_weight_bin = 0;
    std::uint32_t dynamics_torque_to_inertia_bin = 0;
    std::uint32_t dynamics_motor_delay_bin = 0;
    std::uint32_t dynamics_curve_shape_bin = 0;
    std::uint32_t dynamics_group_key = 0;
    std::uint32_t source_episode = 0;
    std::uint32_t source_failure_step = 0;
    std::uint32_t source_replay_step = 0;
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
    float temporal_gradient_decay_alpha = 0.0f;
    float clf_weight = 0.0f;
    float window_clf_weight = 0.0f;
    float clf_alpha = 0.2f;
    float clf_position = 8.0f;
    float clf_velocity = 0.8f;
    float clf_attitude = 4.0f;
    float clf_angular_velocity = 0.8f;
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
    float barrier_huber_delta = 1.0f;
    float barrier_gradient_cap = 0.0f;
    float outward_velocity_weight = 0.0f;
    float attitude_control_weight = 0.0f;
    float attitude_control_k_R = 2.0f;
    float attitude_control_k_omega = 1.0f;
    float velocity_observation_noise = 0.0f;
    std::size_t velocity_observation_delay_steps = 0;
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

struct DeploymentAdapterValidationSummary{
    float max_abs_error = 0.0f;
    float mean_abs_error = 0.0f;
    std::size_t nan_inf_count = 0;
    bool deterministic = false;
    bool relative_offsets_close = false;
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

struct ObjectiveGradientConflictSummary{
    std::size_t batch_size = 0;
    std::size_t horizon = 0;
    float diff_loss_scaled = 0.0f;
    float critic_loss_scaled = 0.0f;
    float physics_shared_norm = 0.0f;
    float critic_shared_norm = 0.0f;
    float combined_shared_norm = 0.0f;
    float physics_actor_head_norm = 0.0f;
    float critic_actor_head_norm = 0.0f;
    float critic_head_norm = 0.0f;
    float shared_cosine = 0.0f;
    float encoder_cosine = 0.0f;
    float gru_input_cosine = 0.0f;
    float gru_hidden_cosine = 0.0f;
    float h0_cosine = 0.0f;
    std::size_t nan_inf_count = 0;
    bool finite = false;
    bool physics_nonzero = false;
    bool critic_nonzero = false;
    bool active_cuda_transition_consistency_present = false;
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
    bool direct_h500_training = false;
    bool critic_training_enabled = true;
    float learning_rate = 1e-4f;
    float diff_rollout_loss_weight = 5e-4f;
    bool action_grad_clip_enabled = false;
    float action_grad_clip_norm = 100.0f;
    bool actor_grad_clip_enabled = true;
    float actor_grad_clip_norm = 100.0f;
    float actor_grad_skip_norm = 1e12f;
    float actor_grad_eps = 1e-6f;
    float temporal_gradient_decay_alpha = 0.0f;
    float clf_weight = 0.0f;
    float window_clf_weight = 0.0f;
    float clf_alpha = 0.2f;
    float clf_position = 8.0f;
    float clf_velocity = 0.8f;
    float clf_attitude = 4.0f;
    float clf_angular_velocity = 0.8f;
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
    float barrier_huber_delta = 1.0f;
    float barrier_gradient_cap = 0.0f;
    float outward_velocity_weight = 0.0f;
    float attitude_control_weight = 0.0f;
    float attitude_control_k_R = 2.0f;
    float attitude_control_k_omega = 1.0f;
    float velocity_observation_noise = 0.0f;
    std::size_t velocity_observation_delay_steps = 0;
    bool persistent_episode_training = true;
    bool disable_physics_gradient = false;
    bool reset_hidden_each_step = false;
    bool sample_dynamics = true;
    bool load_optimizer_state = true;
    DynamicsRandomizationLevel dynamics_randomization_level = DynamicsRandomizationLevel::BROAD;
    bool correlated_size_mass_sampling = false;
    TrajectoryMode trajectory_mode = TrajectoryMode::FIXED; // Training invariant: origin recovery only.
    float trajectory_amplitude = 0.0f;
    float trajectory_frequency_hz = 0.0f;
    std::size_t training_episode_steps = 500;
    float initial_position_scale = 2.5f;
    float initial_velocity_scale = 20.0f;
    float initial_attitude_scale = 18.947368f;
    float initial_angular_velocity_scale = 50.0f;
    float near_zero_guidance_probability = 0.10f;
    float success_position_threshold = 1.0f;
    float success_velocity_threshold = 2.0f;
    float success_attitude_threshold = 3.14159265358979323846f;
    float success_angular_velocity_threshold = 5.0f;
    float success_action_saturation_threshold = 1.0f;
    std::string log_path;
    std::string save_path;
    std::string load_path;
    bool h1000_gate_enabled = false;
    bool h1000_gate_rollback_enabled = true;
    std::size_t h1000_gate_interval = 0;
    std::size_t h1000_gate_horizon = 1000;
    std::size_t h1000_gate_episodes = 256;
    bool h1000_gate_sample_dynamics = false;
    DynamicsRandomizationLevel h1000_gate_dynamics_randomization_level = DynamicsRandomizationLevel::SMALL;
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
    std::size_t failure_replay_segments = 1;
    FailureReplayTriggerMode failure_replay_trigger_mode = FailureReplayTriggerMode::ANY;
    std::size_t failure_replay_capacity = 512;
    std::size_t failure_replay_backtrack_min = 32;
    std::size_t failure_replay_backtrack_max = 128;
    bool failure_replay_response_sampling = false;
    float failure_replay_response_probability = 0.0f;
    std::size_t failure_replay_response_min = 0;
    std::size_t failure_replay_response_max = 0;
    float failure_replay_position_norm = 20.0f;
    float failure_replay_velocity_norm = 20.0f;
    float failure_replay_angular_velocity_norm = 30.0f;
    float failure_replay_attitude_error = 2.6179938779914943654f; // 150 deg
    float failure_replay_action_abs = 1.01f;
    float replay_recovery_action_magnitude_weight = 0.0f;
    float replay_recovery_saturation_weight = 0.0f;
    float replay_recovery_linear_velocity_weight = 0.0f;
    float replay_recovery_velocity_barrier_weight = 0.0f;
    float replay_recovery_velocity_barrier_safe = 2.0f;
    float replay_recovery_angular_velocity_weight = 0.0f;
    float replay_recovery_position_progress_weight = 0.0f;
    float replay_recovery_progress_margin = 0.0f;
    float replay_recovery_progress_epsilon = 1e-3f;
    float replay_recovery_progress_huber_delta = 1.0f;
    std::uint32_t replay_recovery_start_segment = 1u;
};

struct FullGpuTrainingSummary{
    float final_loss = 0.0f;
    float final_grad_norm = 0.0f;
    float final_critic_loss = 0.0f;
    float final_critic_error_norm = 0.0f;
    float final_critic_output_mean = 0.0f;
    float final_critic_target_mean = 0.0f;
    float returns_mean = 0.0f;
    float returns_std = 0.0f;
    float episode_length_mean = 0.0f;
    float episode_length_std = 0.0f;
    std::size_t num_terminated = 0;
    float share_terminated = 0.0f;
    float position_cost = 0.0f;
    float orientation_cost = 0.0f;
    float linear_velocity_cost = 0.0f;
    float angular_velocity_cost = 0.0f;
    float action_cost = 0.0f;
    float weighted_cost = 0.0f;
    float reward_mean = 0.0f;
    float reward_std = 0.0f;
    float action_saturation_rate = 0.0f;
    float action_abs_mean = 0.0f;
    float action_abs_max = 0.0f;
    float action_delta_mean = 0.0f;
    float action_delta_max = 0.0f;
    float hover_relative_action_mean = 0.0f;
    float hover_relative_action_max = 0.0f;
    float action_gradient_norm = 0.0f;
    float raw_action_gradient_norm_pre_clip = 0.0f;
    float raw_action_gradient_norm_post_clip = 0.0f;
    float actor_gradient_norm_pre_clip = 0.0f;
    float actor_update_norm = 0.0f;
    LossComponentMeans final_loss_components;
    std::size_t nan_inf_count = 0;
    float final_window_start_mean = 0.0f;
    std::size_t training_episode_steps = 0;
    std::size_t persistent_episode_step = 0;
    std::size_t segment_start = 0;
    std::size_t segment_end = 0;
    std::size_t episode_reset_count = 0;
    bool persistent_episode_training = false;
    bool direct_h500_training = false;
    bool critic_training_enabled = true;
    bool finite = false;
    bool checkpoint_saved = false;
    bool checkpoint_loaded = false;
    bool h1000_gate_enabled = false;
    bool h1000_gate_last_passed = false;
    bool h1000_gate_best_available = false;
    std::size_t h1000_gate_eval_count = 0;
    std::size_t h1000_gate_pass_count = 0;
    std::size_t h1000_gate_fail_count = 0;
    std::size_t h1000_gate_rollback_count = 0;
    std::size_t h1000_gate_last_step = 0;
    float h1000_gate_last_weighted_cost = 0.0f;
    float h1000_gate_last_action_saturation_rate = 0.0f;
    float h1000_gate_last_max_action_abs = 0.0f;
    float h1000_gate_last_mean_max_angular_velocity_norm = 0.0f;
    float h1000_gate_last_max_angular_velocity_norm = 0.0f;
    float h1000_gate_last_mean_max_position_norm = 0.0f;
    float h1000_gate_last_mean_max_velocity_norm = 0.0f;
    float h1000_gate_last_mean_first_failure_time_s = 0.0f;
    std::size_t h1000_gate_last_nan_inf_count = 0;
    std::size_t h1000_gate_last_first_failure_position_count = 0;
    std::size_t h1000_gate_last_first_failure_velocity_count = 0;
    std::size_t h1000_gate_last_first_failure_attitude_count = 0;
    std::size_t h1000_gate_last_first_failure_angular_velocity_count = 0;
    std::size_t h1000_gate_last_first_failure_nonfinite_count = 0;
    float h1000_gate_last_shadow_score = 0.0f;
    float h1000_gate_best_weighted_cost = 0.0f;
    float h1000_gate_best_shadow_score = 0.0f;
    bool failure_replay_enabled = false;
    std::size_t failure_replay_buffer_size = 0;
    std::size_t failure_replay_added_count = 0;
    std::size_t failure_replay_used_count = 0;
    std::size_t failure_replay_last_added = 0;
    std::size_t failure_replay_active_slots = 0;
    std::size_t failure_replay_new_slots = 0;
    std::size_t failure_replay_chain_length = 1;
    std::size_t failure_replay_episode_count = 0;
    std::size_t failure_replay_segment_count = 0;
    std::size_t failure_replay_completed_chain_count = 0;
    std::size_t failure_replay_last_state_trigger_count = 0;
    std::size_t failure_replay_last_action_trigger_count = 0;
    std::size_t failure_replay_last_nonfinite_trigger_count = 0;
    float failure_replay_mean_segment_index = 0.0f;
    std::size_t failure_replay_max_segment_index = 0;
    std::size_t failure_replay_observed_max_segment_index = 0;
    bool failure_replay_state_carry_enabled = false;
    bool failure_replay_hidden_carry_enabled = false;
    float failure_replay_action_saturation_rate = 0.0f;
    std::size_t failure_replay_carry_check_slots = 0;
    float failure_replay_carry_position_error = 0.0f;
    float failure_replay_carry_velocity_error = 0.0f;
    float failure_replay_carry_rotation_error = 0.0f;
    float failure_replay_carry_angular_velocity_error = 0.0f;
    float failure_replay_carry_rpm_error = 0.0f;
    float failure_replay_carry_previous_action_error = 0.0f;
    float failure_replay_carry_hidden_error = 0.0f;
    bool failure_replay_last_episode = false;
    bool passed = false;
};

struct GpuPolicyEvalOptions{
    int device = 0;
    std::size_t episodes = 100;
    std::size_t horizon = 500;
    unsigned seed = 0;
    bool sample_dynamics = true;
    bool reset_hidden_each_step = false;
    DynamicsRandomizationLevel dynamics_randomization_level = DynamicsRandomizationLevel::BROAD;
    bool correlated_size_mass_sampling = false;
    TrajectoryMode trajectory_mode = TrajectoryMode::FIXED;
    float trajectory_amplitude = 0.03f;
    float trajectory_frequency_hz = 0.5f;
    float initial_position_scale = 2.5f;
    float initial_velocity_scale = 20.0f;
    float initial_attitude_scale = 18.947368f;
    float initial_angular_velocity_scale = 50.0f;
    float near_zero_guidance_probability = 0.10f;
    float success_position_threshold = 1.0f;
    float success_velocity_threshold = 2.0f;
    float success_attitude_threshold = 3.14159265358979323846f;
    float success_angular_velocity_threshold = 5.0f;
    float success_action_saturation_threshold = 1.0f;
    std::size_t throughout_gate_start_step = 0;
    float velocity_observation_noise = 0.0f;
    std::size_t velocity_observation_delay_steps = 0;
    ForcedDynamicsBins forced_bins;
    bool collect_failure_replay = false;
    FailureReplayTriggerMode failure_replay_trigger_mode = FailureReplayTriggerMode::ANY;
    std::size_t failure_replay_capacity = 0;
    std::size_t failure_replay_backtrack_min = 32;
    std::size_t failure_replay_backtrack_max = 128;
    bool failure_replay_response_sampling = false;
    float failure_replay_response_probability = 0.0f;
    std::size_t failure_replay_response_min = 0;
    std::size_t failure_replay_response_max = 0;
    float failure_replay_position_norm = 20.0f;
    float failure_replay_velocity_norm = 20.0f;
    float failure_replay_angular_velocity_norm = 30.0f;
    float failure_replay_attitude_error = 2.6179938779914943654f; // 150 deg
    float failure_replay_action_abs = 1.01f;
    std::string load_path;
    std::string log_path;
};

struct GpuPolicyEvalSummary{
    float returns_mean = 0.0f;
    float returns_std = 0.0f;
    float episode_length_mean = 0.0f;
    float episode_length_std = 0.0f;
    std::size_t num_terminated = 0;
    float share_terminated = 0.0f;
    float position_cost = 0.0f;
    float orientation_cost = 0.0f;
    float linear_velocity_cost = 0.0f;
    float angular_velocity_cost = 0.0f;
    float action_cost = 0.0f;
    float weighted_cost = 0.0f;
    float reward_mean = 0.0f;
    float reward_std = 0.0f;
    bool sample_dynamics = true;
    DynamicsRandomizationLevel dynamics_randomization_level = DynamicsRandomizationLevel::BROAD;
    bool correlated_size_mass_sampling = false;
    float mass_min = 0.0f;
    float mass_mean = 0.0f;
    float mass_max = 0.0f;
    float thrust_to_weight_min = 0.0f;
    float thrust_to_weight_mean = 0.0f;
    float thrust_to_weight_max = 0.0f;
    float motor_delay_min = 0.0f;
    float motor_delay_mean = 0.0f;
    float motor_delay_max = 0.0f;
    float mean_total_loss = 0.0f;
    float mean_final_position_norm = 0.0f;
    float mean_final_velocity_norm = 0.0f;
    float mean_final_attitude_error = 0.0f;
    float mean_final_angular_velocity_norm = 0.0f;
    float position_rmse = 0.0f;
    float velocity_rmse = 0.0f;
    float attitude_rmse = 0.0f;
    float angular_velocity_rmse = 0.0f;
    float linear_acceleration_error_rmse = 0.0f;
    float angular_acceleration_error_rmse = 0.0f;
    float position_mean_error_mean = 0.0f;
    float angle_mean_error_mean = 0.0f;
    float linear_velocity_mean_error_mean = 0.0f;
    float angular_velocity_mean_error_mean = 0.0f;
    float angular_acceleration_mean_error_mean = 0.0f;
    float action_mean_error_mean = 0.0f;
    float action_relative_mean_error_mean = 0.0f;
    float position_max_error_mean = 0.0f;
    float angle_max_error_mean = 0.0f;
    float linear_velocity_max_error_mean = 0.0f;
    float angular_velocity_max_error_mean = 0.0f;
    float angular_acceleration_max_error_mean = 0.0f;
    float action_max_error_mean = 0.0f;
    float action_relative_max_error_mean = 0.0f;
    float position_max_error_std = 0.0f;
    float angle_max_error_std = 0.0f;
    float linear_velocity_max_error_std = 0.0f;
    float angular_velocity_max_error_std = 0.0f;
    float angular_acceleration_max_error_std = 0.0f;
    float action_max_error_std = 0.0f;
    float action_relative_max_error_std = 0.0f;
    float median_final_position_norm = 0.0f;
    float median_final_velocity_norm = 0.0f;
    float median_final_attitude_error = 0.0f;
    float median_final_angular_velocity_norm = 0.0f;
    float p90_final_position_norm = 0.0f;
    float p90_final_velocity_norm = 0.0f;
    float p90_final_attitude_error = 0.0f;
    float p90_final_angular_velocity_norm = 0.0f;
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
    float max_position_norm = 0.0f;
    float max_velocity_norm = 0.0f;
    float max_attitude_error = 0.0f;
    float max_angular_velocity_norm = 0.0f;
    float mean_action_magnitude = 0.0f;
    float max_action_magnitude = 0.0f;
    float mean_action_smoothness = 0.0f;
    float max_action_smoothness = 0.0f;
    float max_action_abs = 0.0f;
    float action_saturation_rate = 0.0f;
    float invalid_or_nan_rate = 0.0f;
    std::size_t nan_inf_count = 0;
    std::size_t first_failure_position_count = 0;
    std::size_t first_failure_velocity_count = 0;
    std::size_t first_failure_attitude_count = 0;
    std::size_t first_failure_angular_velocity_count = 0;
    std::size_t first_failure_nonfinite_count = 0;
    std::vector<FailureReplaySample> failure_replay_samples;
    std::size_t failure_replay_sample_count = 0;
    std::size_t failure_replay_state_trigger_count = 0;
    std::size_t failure_replay_action_trigger_count = 0;
    std::size_t failure_replay_nonfinite_trigger_count = 0;
    bool checkpoint_loaded = false;
    bool finite = false;
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

struct LocalInitialConditionValidationSummary{
    std::size_t samples = 0;
    std::size_t horizon = 0;
    std::size_t nan_inf_count = 0;
    float position_threshold = 0.05f;
    float velocity_threshold = 0.1f;
    float attitude_threshold = 0.0872664626f;
    float angular_velocity_threshold = 0.2f;
    float max_position_error = 0.0f;
    float max_velocity_error = 0.0f;
    float max_attitude_error = 0.0f;
    float max_angular_velocity_norm = 0.0f;
    bool position_ok = false;
    bool velocity_ok = false;
    bool attitude_ok = false;
    bool angular_velocity_ok = false;
    bool finite = false;
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
    float initial_angular_velocity_scale = 1.0f,
    float initial_attitude_scale = 0.0f,
    float near_zero_guidance_probability = 0.10f,
    std::size_t trajectory_episode_steps = 0,
    DynamicsRandomizationLevel dynamics_randomization_level = DynamicsRandomizationLevel::BROAD
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

DeploymentAdapterValidationSummary validate_deployment_adapter(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    TrajectoryMode trajectory_mode,
    float trajectory_amplitude,
    float trajectory_frequency_hz,
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

ObjectiveGradientConflictSummary diagnose_objective_gradient_conflicts(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options,
    float diff_rollout_loss_weight
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

LocalInitialConditionValidationSummary validate_local_initial_conditions(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    TrajectoryMode trajectory_mode,
    float trajectory_amplitude,
    float trajectory_frequency_hz,
    bool correlated_size_mass_sampling,
    float initial_position_scale,
    float initial_velocity_scale,
    float initial_angular_velocity_scale,
    float initial_attitude_scale,
    float near_zero_guidance_probability = 0.10f
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
