#ifndef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
// #define RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
#endif

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f/diff_euler_model.h>
#include <rl_tools/rl/environments/l2f/diff_euler_rollout.h>

#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cli_options.h"
#include "eval_utils.h"
#include "checkpoint_io.h"
#include "logging_utils.h"
#include "gradient_utils.h"
#include "loop.h"
#include "rdac_operations.h"
#include "../equivalent_dynamics.h"

namespace fp = rl_tools::foundation_policy::diff_pre_training;
namespace rlt = rl_tools;
namespace l2f_diff = rl_tools::rl::environments::l2f::diff;
namespace eq_dyn = rl_tools::foundation_policy::equivalent_dynamics;

using DiffModel = fp::DiffModel;
using EvalModel = fp::EvalModel;
using RuntimeOptions = fp::RuntimeOptions;

std::string sibling_output_path(const std::string& anchor_path, const std::string& filename){
    const std::size_t separator = anchor_path.find_last_of("/\\");
    if(separator == std::string::npos){
        return filename;
    }
    return anchor_path.substr(0, separator + 1) + filename;
}

void write_production_objective_trace_header(std::ostream& out){
    out << "step,seed,batch_size,horizon,sample_mode,fixed_or_sampled_dynamics,"
        << "sampled_dynamics_level,balanced_dynamics_sampling_enabled,"
        << "total_loss_for_backprop,diff_rollout_loss_raw,diff_rollout_loss_weight,diff_rollout_loss_scaled,"
        << "position_loss,velocity_loss,attitude_loss,angular_velocity_loss,"
        << "terminal_loss,terminal_position_loss,terminal_velocity_loss,terminal_attitude_loss,terminal_angular_velocity_loss,"
        << "clf_loss,window_clf_loss,outward_velocity_loss,attitude_control_loss,"
        << "velocity_barrier_loss,angular_velocity_barrier_loss,attitude_barrier_loss,"
        << "action_magnitude_loss,action_smoothness_loss,saturation_loss,"
        << "critic_loss,actor_critic_loss,transition_consistency_loss,value_loss,entropy_loss,teacher_loss,other_loss_terms,"
        << "action_grad_norm_before_scale,action_grad_norm_after_diff_weight,action_grad_norm_after_action_clip,"
        << "actor_grad_norm_before_clip,actor_grad_norm_after_clip,actor_grad_skip_triggered,"
        << "actor_update_norm,adam_step,adam_m_norm,adam_v_norm,"
        << "success_rate_batch,final_position_norm_mean,final_velocity_norm_mean,final_angular_velocity_norm_mean,action_saturation_rate\n";
}

template <typename TI>
void decode_dynamics_group_key(TI key, TI num_bins, TI& size_mass, TI& thrust_to_weight, TI& torque_to_inertia, TI& motor_delay, TI& curve_shape){
    curve_shape = key % num_bins;
    key /= num_bins;
    motor_delay = key % num_bins;
    key /= num_bins;
    torque_to_inertia = key % num_bins;
    key /= num_bins;
    thrust_to_weight = key % num_bins;
    key /= num_bins;
    size_mass = key % num_bins;
}

template <typename T>
T ramp_difficulty(long long step, long long stage_steps, long long stages){
    if(stage_steps <= 0 || stages <= 0){
        return (T)1;
    }
    return fp::clamp01((T)step / (T)(stage_steps * stages));
}

template <typename TI>
TI scheduled_horizon(const RuntimeOptions& options, TI training_step){
    if(!options.horizon_curriculum){
        return options.horizon;
    }
    TI horizon = std::min(options.horizon, (TI)fp::HORIZON_START);
    TI stage = options.horizon_stage_steps > 0 ? training_step / options.horizon_stage_steps : options.horizon;
    for(TI stage_i = 0; stage_i < stage; stage_i++){
        TI next_horizon = std::min(options.horizon, horizon * (TI)2);
        // Hold at H=64 for extra steps before transitioning to H=128
        if(options.hold_h64_extra_steps > 0 && horizon == (TI)64 && next_horizon > (TI)64 && options.horizon > (TI)64){
            TI steps_into_h64 = training_step - (TI)(stage_i * options.horizon_stage_steps);
            if(steps_into_h64 < options.hold_h64_extra_steps + options.horizon_stage_steps){
                return horizon;
            }
        }
        horizon = next_horizon;
    }
    return std::max((TI)1, horizon);
}

template <typename TI>
TI curriculum_stage_bucket(bool enabled, TI training_step, TI stage_steps, TI max_bucket){
    if(!enabled || stage_steps <= 0){
        return max_bucket;
    }
    return std::min(max_bucket, training_step / stage_steps);
}

template <typename TI>
eq_dyn::DynamicsBins<TI> scheduled_balanced_dynamics_bins(TI training_step, TI batch_i, TI batch_size, TI num_bins){
    eq_dyn::DynamicsBins<TI> bins;
    if(num_bins <= 1){
        return bins;
    }
    TI sample_index = training_step * std::max((TI)1, batch_size) + batch_i;
    bins.size_mass = sample_index % num_bins;
    sample_index /= num_bins;
    bins.thrust_to_weight = sample_index % num_bins;
    sample_index /= num_bins;
    bins.torque_to_inertia = sample_index % num_bins;
    sample_index /= num_bins;
    bins.motor_delay = sample_index % num_bins;
    sample_index /= num_bins;
    bins.curve_shape = sample_index % num_bins;
    return bins;
}

// Success-gated curriculum: explicit alternating stage plan.
struct CurriculumStage{
    long long horizon;
    float state_difficulty;
    const char* name;
    CurriculumStage(long long h, float d, const char* n) : horizon(h), state_difficulty(d), name(n) {}
};

template <typename TI, typename T>
void get_success_gated_curriculum_stages(TI max_horizon, std::vector<CurriculumStage>& stages){
    stages.clear();
    stages.push_back(CurriculumStage(16, (T)0.25, "H16_easy"));
    stages.push_back(CurriculumStage(32, (T)0.25, "H32_easy"));
    stages.push_back(CurriculumStage(64, (T)0.25, "H64_easy"));
    stages.push_back(CurriculumStage(64, (T)0.60, "H64_medium"));
    stages.push_back(CurriculumStage(std::min(max_horizon, (TI)128), (T)0.60, "H128_medium"));
    stages.push_back(CurriculumStage(std::min(max_horizon, (TI)128), (T)1.00, "H128_full"));
}

template <typename T, typename TI>
const char* check_curriculum_gate(
    const RuntimeOptions& options,
    TI stage_age,
    const std::vector<bool>& success_window,
    TI training_step,
    TI final_stage_index
){
    if(!options.success_gated_curriculum) return "none";
    if(stage_age < options.curriculum_min_stage_steps) return "min_stage_hold";
    if(options.curriculum_gate_check_interval > 0 && training_step % options.curriculum_gate_check_interval != 0) return "gate_not_checked";
    TI count = 0;
    TI total = 0;
    for(auto s : success_window){ total++; if(s) count++; }
    if(total < options.curriculum_success_window) return "window_not_full";
    T mean = total > 0 ? (T)count / (T)total : (T)0;
    if(mean >= options.curriculum_success_threshold) return "success_gate_pass";
    // Hybrid: force advance if max stage steps reached and not disabled
    if(!options.curriculum_no_forced_advance && stage_age >= options.curriculum_max_stage_steps){
        return "max_stage_forced";
    }
    return "gate_fail_hold";
}

template <typename T, typename TI>
bool compute_training_success_flag(
    const l2f_diff::EulerState<T, TI>& final_state,
    const l2f_diff::TrackingReference<T>& ref
){
    const T p_norm = fp::euler_position_error_norm<T, TI>(final_state, ref);
    const T v_norm = fp::euler_velocity_error_norm<T, TI>(final_state, ref);
    const T w_norm = fp::norm3(final_state.omega);
    return p_norm < fp::SUCCESS_POSITION_THRESHOLD
        && v_norm < fp::SUCCESS_VELOCITY_THRESHOLD
        && w_norm < fp::SUCCESS_ANGULAR_VELOCITY_THRESHOLD;
}

template <typename T, typename TI>
bool compute_training_success_flag(
    const l2f_diff::EulerState<T, TI>& final_state
){
    const auto ref = l2f_diff::zero_tracking_reference<T>();
    return compute_training_success_flag<T, TI>(final_state, ref);
}

// H128-prioritized curriculum schedule
struct H128ScheduleStep{ long long horizon; float difficulty; const char* phase; };

template <typename TI, typename T>
void get_h128_prioritized_schedule(const std::string& name, std::vector<H128ScheduleStep>& out){
    out.clear();
    if(name == "short_warmup_12000"){
        for(TI s=0; s<500; s++)  out.push_back({16, (T)0.25, "H16_easy"});
        for(TI s=500; s<1000; s++) out.push_back({32, (T)0.25, "H32_easy"});
        for(TI s=1000; s<1500; s++) out.push_back({64, (T)0.25, "H64_easy"});
        for(TI s=1500; s<4000; s++) out.push_back({128, (T)0.25, "H128_easy"});
        for(TI s=4000; s<6000; s++) out.push_back({128, (T)0.60, "H128_medium"});
        for(TI s=6000; s<12000; s++) out.push_back({128, (T)1.00, "H128_full"});
    }
    else if(name == "balanced_16000"){
        for(TI s=0; s<1000; s++)  out.push_back({16, (T)0.25, "H16_easy"});
        for(TI s=1000; s<2000; s++) out.push_back({32, (T)0.25, "H32_easy"});
        for(TI s=2000; s<3000; s++) out.push_back({64, (T)0.25, "H64_easy"});
        for(TI s=3000; s<6000; s++) out.push_back({128, (T)0.25, "H128_easy"});
        for(TI s=6000; s<9000; s++) out.push_back({128, (T)0.60, "H128_medium"});
        for(TI s=9000; s<16000; s++) out.push_back({128, (T)1.00, "H128_full"});
    }
    else if(name == "direct_H128_full"){
        for(TI s=0; s<16000; s++) out.push_back({128, (T)1.00, "H128_full_direct"});
    }
}

template <typename T, typename TI>
T terminal_ramp_multiplier(const RuntimeOptions& options, TI steps_since_horizon_change){
    if(!options.terminal_ramp_after_horizon_change || options.terminal_ramp_steps <= 0){
        return (T)1;
    }
    const T progress = fp::clamp01((T)steps_since_horizon_change / (T)options.terminal_ramp_steps);
    return options.terminal_ramp_min + ((T)1 - options.terminal_ramp_min) * progress;
}

template <typename T>
T curriculum_scale(T start, T difficulty){
    return start + ((T)1 - start) * fp::clamp01(difficulty);
}

template <typename STATE, typename T>
void apply_state_curriculum(STATE& state, T difficulty){
    const T position_scale = curriculum_scale<T>(fp::STATE_CURRICULUM_POSITION_START, difficulty);
    const T velocity_scale = curriculum_scale<T>(fp::STATE_CURRICULUM_VELOCITY_START, difficulty);
    const T attitude_scale = curriculum_scale<T>(fp::STATE_CURRICULUM_ATTITUDE_START, difficulty);
    const T angular_velocity_scale = curriculum_scale<T>(fp::STATE_CURRICULUM_ANGULAR_VELOCITY_START, difficulty);
    for(int i = 0; i < 3; i++){
        state.position[i] *= position_scale;
        state.linear_velocity[i] *= velocity_scale;
        state.angular_velocity[i] *= angular_velocity_scale;
        state.orientation[i + 1] *= attitude_scale;
    }
    const T sign = state.orientation[0] >= (T)0 ? (T)1 : (T)-1;
    T vector_norm_sq = 0;
    for(int i = 1; i < 4; i++){
        vector_norm_sq += state.orientation[i] * state.orientation[i];
    }
    if(vector_norm_sq >= (T)0.999){
        const T factor = std::sqrt((T)0.999 / vector_norm_sq);
        for(int i = 1; i < 4; i++){
            state.orientation[i] *= factor;
        }
        vector_norm_sq = (T)0.999;
    }
    state.orientation[0] = sign * std::sqrt(std::max((T)0, (T)1 - vector_norm_sq));
}

template <typename PARAMETERS, typename T, typename TI>
T estimated_thrust_to_weight(const PARAMETERS& parameters){
    T thrust_z = 0;
    const T max_rpm = parameters.dynamics.action_limit.max;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        const T c0 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][0];
        const T c1 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][1];
        const T c2 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][2];
        const T force = c0 + c1 * max_rpm + c2 * max_rpm * max_rpm;
        thrust_z += parameters.dynamics.rotor_thrust_directions[rotor_i][2] * force;
    }
    const T gravity_norm = std::max((T)1e-6, std::abs(parameters.dynamics.gravity[2]));
    return std::abs(thrust_z) / std::max((T)1e-6, parameters.dynamics.mass * gravity_norm);
}

template <typename PARAMETERS, typename T, typename TI>
T estimated_max_thrust(const PARAMETERS& parameters){
    T max_thrust = 0;
    const T max_rpm = parameters.dynamics.action_limit.max;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        const T c0 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][0];
        const T c1 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][1];
        const T c2 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][2];
        max_thrust += c0 + c1 * max_rpm + c2 * max_rpm * max_rpm;
    }
    return std::abs(max_thrust);
}

template <typename PARAMETERS, typename T, typename TI>
T estimated_torque_to_inertia_ratio(const PARAMETERS& parameters){
    const T gravity_norm = std::max((T)1e-6, std::abs(parameters.dynamics.gravity[2]));
    const T thrust_to_weight = estimated_thrust_to_weight<PARAMETERS, T, TI>(parameters);
    const T max_thrust_per_rotor = thrust_to_weight * parameters.dynamics.mass * gravity_norm / (T)4;
    const T rotor_distance = std::abs(parameters.dynamics.rotor_positions[0][0]);
    const T max_torque = rotor_distance * (T)1.414213562373095 * max_thrust_per_rotor;
    return max_torque / std::max((T)1e-12, parameters.dynamics.J[0][0]);
}

template <typename PARAMETERS, typename T, typename TI>
T average_rotor_torque_constant_abs(const PARAMETERS& parameters){
    T sum = 0;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        sum += std::abs(parameters.dynamics.rotor_torque_constants[rotor_i]);
    }
    return sum / (T)4;
}

template <typename PARAMETERS, typename T, typename TI>
void motor_tau_stats(const PARAMETERS& parameters, T& mean, T& min_value, T& max_value){
    mean = 0;
    min_value = std::numeric_limits<T>::infinity();
    max_value = -std::numeric_limits<T>::infinity();
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        const T rising = parameters.dynamics.rotor_time_constants_rising[rotor_i];
        const T falling = parameters.dynamics.rotor_time_constants_falling[rotor_i];
        mean += rising + falling;
        min_value = std::min(min_value, std::min(rising, falling));
        max_value = std::max(max_value, std::max(rising, falling));
    }
    mean /= (T)8;
}

template <typename PARAMETERS, typename T, typename TI>
void motor_tau_response_means(const PARAMETERS& parameters, T& rise_mean, T& fall_mean){
    rise_mean = 0;
    fall_mean = 0;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        rise_mean += parameters.dynamics.rotor_time_constants_rising[rotor_i];
        fall_mean += parameters.dynamics.rotor_time_constants_falling[rotor_i];
    }
    rise_mean /= (T)4;
    fall_mean /= (T)4;
}

template <typename T, typename DEVICE, typename RNG>
T sample_uniform_range(DEVICE& device, RNG& rng, T min_value, T max_value){
    if(!(max_value > min_value)){
        return min_value;
    }
    return rlt::random::uniform_real_distribution(device.random, min_value, max_value, rng);
}

template <typename T>
T clamp_value(T value, T min_value, T max_value){
    return std::max(min_value, std::min(max_value, value));
}

template <typename PARAMETERS, typename T, typename TI>
T solve_hover_throttle_relative(const PARAMETERS& parameters){
    const T min_action = parameters.dynamics.action_limit.min;
    const T max_action = parameters.dynamics.action_limit.max;
    const T action_range = std::max((T)1e-12, max_action - min_action);
    const T gravity_norm = std::max((T)1e-6, std::abs(parameters.dynamics.gravity[2]));
    const T hover_force = parameters.dynamics.mass * gravity_norm / (T)4;
    const T c0 = parameters.dynamics.rotor_thrust_coefficients[0][0];
    const T c1 = parameters.dynamics.rotor_thrust_coefficients[0][1];
    const T c2 = parameters.dynamics.rotor_thrust_coefficients[0][2];
    T hover_action = min_action;
    if(std::abs(c2) > (T)1e-12){
        const T discriminant = c1 * c1 - (T)4 * c2 * (c0 - hover_force);
        if(discriminant >= (T)0){
            hover_action = (-c1 + std::sqrt(discriminant)) / ((T)2 * c2);
        }
        else{
            hover_action = max_action;
        }
    }
    else if(std::abs(c1) > (T)1e-12){
        hover_action = (hover_force - c0) / c1;
    }
    hover_action = clamp_value<T>(hover_action, min_action, max_action);
    return clamp_value<T>((hover_action - min_action) / action_range, (T)0, (T)1);
}

template <typename PARAMETERS, typename T, typename TI>
bool sampled_parameters_inside_allowed_ranges(const PARAMETERS& parameters, T dynamics_difficulty){
    const auto& domain = parameters.domain_randomization;
    const T safe_mass_max = (T)0.06 + dynamics_difficulty * (fp::SAMPLED_DYNAMICS_MAX_MASS - (T)0.06);
    const T mass_upper = std::min(domain.mass_max, safe_mass_max);
    const T thrust_to_weight = estimated_thrust_to_weight<PARAMETERS, T, TI>(parameters);
    const T torque_to_inertia = estimated_torque_to_inertia_ratio<PARAMETERS, T, TI>(parameters);
    const T range_epsilon = (T)1e-6;
    T tau_rise;
    T tau_fall;
    motor_tau_response_means<PARAMETERS, T, TI>(parameters, tau_rise, tau_fall);
    if(!fp::finite_value(parameters.dynamics.mass) ||
       parameters.dynamics.mass < domain.mass_min - range_epsilon ||
       parameters.dynamics.mass > mass_upper + range_epsilon){
        return false;
    }
    if(!fp::finite_value(thrust_to_weight) ||
       thrust_to_weight < domain.thrust_to_weight_min - range_epsilon ||
       thrust_to_weight > domain.thrust_to_weight_max + range_epsilon ||
       thrust_to_weight < fp::SAMPLED_DYNAMICS_MIN_THRUST_TO_WEIGHT){
        return false;
    }
    if(!fp::finite_value(torque_to_inertia) ||
       torque_to_inertia < domain.torque_to_inertia_min - range_epsilon ||
       torque_to_inertia > domain.torque_to_inertia_max + range_epsilon){
        return false;
    }
    if(!fp::finite_value(tau_rise) ||
       tau_rise < domain.rotor_time_constant_rising_min - range_epsilon ||
       tau_rise > domain.rotor_time_constant_rising_max + range_epsilon ||
       !fp::finite_value(tau_fall) ||
       tau_fall < domain.rotor_time_constant_falling_min - range_epsilon ||
       tau_fall > domain.rotor_time_constant_falling_max + range_epsilon){
        return false;
    }
    for(TI i = 0; i < 3; i++){
        if(!fp::finite_value(parameters.dynamics.J[i][i]) || parameters.dynamics.J[i][i] <= (T)0) return false;
        if(!fp::finite_value(parameters.dynamics.J_inv[i][i]) || parameters.dynamics.J_inv[i][i] <= (T)0) return false;
    }
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        const T torque_constant = parameters.dynamics.rotor_torque_constants[rotor_i];
        if(!fp::finite_value(torque_constant) ||
           torque_constant < domain.rotor_torque_constant_min - range_epsilon ||
           torque_constant > domain.rotor_torque_constant_max + range_epsilon){
            return false;
        }
        const T c0 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][0];
        const T c1 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][1];
        const T c2 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][2];
        const T max_action = parameters.dynamics.action_limit.max;
        const T max_force = c0 + c1 * max_action + c2 * max_action * max_action;
        if(!fp::finite_value(c0) || !fp::finite_value(c1) || !fp::finite_value(c2) ||
           !fp::finite_value(max_force) || c0 < (T)0 || c1 < (T)0 || c2 <= (T)0 || max_force <= (T)0){
            return false;
        }
        if(!fp::finite_value(parameters.dynamics.rotor_time_constants_rising[rotor_i]) ||
           !fp::finite_value(parameters.dynamics.rotor_time_constants_falling[rotor_i]) ||
           parameters.dynamics.rotor_time_constants_rising[rotor_i] < domain.rotor_time_constant_rising_min - range_epsilon ||
           parameters.dynamics.rotor_time_constants_rising[rotor_i] > domain.rotor_time_constant_rising_max + range_epsilon ||
           parameters.dynamics.rotor_time_constants_falling[rotor_i] < domain.rotor_time_constant_falling_min - range_epsilon ||
           parameters.dynamics.rotor_time_constants_falling[rotor_i] > domain.rotor_time_constant_falling_max + range_epsilon){
            return false;
        }
    }
    const T max_disturbance_multiple = std::max((T)0, thrust_to_weight - (T)1) * domain.disturbance_force_max;
    const T max_disturbance_std = max_disturbance_multiple * thrust_to_weight * parameters.dynamics.mass / (T)3;
    if(!fp::finite_value(parameters.disturbances.random_force.mean) ||
       !fp::finite_value(parameters.disturbances.random_force.std) ||
       std::abs(parameters.disturbances.random_force.mean) > (T)1e-12 ||
       parameters.disturbances.random_force.std < (T)0 ||
       parameters.disturbances.random_force.std > max_disturbance_std + (T)1e-9){
        return false;
    }
    if(!fp::finite_value(parameters.dynamics.hovering_throttle_relative) ||
       parameters.dynamics.hovering_throttle_relative < (T)0 ||
       parameters.dynamics.hovering_throttle_relative > (T)1){
        return false;
    }
    return true;
}

template <typename PARAMETERS, typename T, typename TI>
struct SampledDynamicsDiagnostics{
    T mass = 0;
    T thrust_to_weight_ratio = 0;
    T torque_to_inertia_ratio = 0;
    T motor_tau_mean = 0;
    T motor_tau_min = 0;
    T motor_tau_max = 0;
    T sampled_tau_rise = 0;
    T sampled_tau_fall = 0;
    T sampled_curve_shape = 0;
    T sampled_parameters_inside_allowed_ranges = 0;
    T inertia_trace = 0;
    T thrust_scale = 0;
    T torque_scale = 0;
    T dynamics_size_mass_bin = 0;
    T dynamics_thrust_to_weight_bin = 0;
    T dynamics_torque_to_inertia_bin = 0;
    T dynamics_motor_delay_bin = 0;
    T dynamics_curve_shape_bin = 0;
    T equivalent_dynamics_diag_thrust_to_acceleration_gain = 0;
    T equivalent_dynamics_diag_roll_pitch_torque_to_angular_acceleration_gain = 0;
    T equivalent_dynamics_diag_yaw_torque_to_angular_acceleration_gain = 0;
    T equivalent_dynamics_diag_motor_rise_time_constant = 0;
    T equivalent_dynamics_diag_motor_fall_time_constant = 0;
    T equivalent_dynamics_diag_thrust_curve_shape = 0;
    T equivalent_dynamics_diag_torque_curve_shape = 0;
    T equivalent_dynamics_diag_residual_force_bias = 0;
    T equivalent_dynamics_diag_residual_torque_bias = 0;
};

template <typename PARAMETERS, typename T, typename TI>
SampledDynamicsDiagnostics<PARAMETERS, T, TI> sampled_dynamics_diagnostics(
    const PARAMETERS& parameters,
    const PARAMETERS& nominal_parameters,
    const eq_dyn::DynamicsBins<TI>& bins,
    T dynamics_difficulty
){
    SampledDynamicsDiagnostics<PARAMETERS, T, TI> stats;
    const auto equivalent_dynamics_diag = eq_dyn::normalized_target_from_parameters<PARAMETERS>(parameters, nominal_parameters);
    stats.mass = parameters.dynamics.mass;
    stats.thrust_to_weight_ratio = estimated_thrust_to_weight<PARAMETERS, T, TI>(parameters);
    stats.torque_to_inertia_ratio = estimated_torque_to_inertia_ratio<PARAMETERS, T, TI>(parameters);
    motor_tau_stats<PARAMETERS, T, TI>(parameters, stats.motor_tau_mean, stats.motor_tau_min, stats.motor_tau_max);
    motor_tau_response_means<PARAMETERS, T, TI>(parameters, stats.sampled_tau_rise, stats.sampled_tau_fall);
    stats.sampled_curve_shape = eq_dyn::thrust_curve_shape<PARAMETERS>(parameters);
    stats.sampled_parameters_inside_allowed_ranges =
        sampled_parameters_inside_allowed_ranges<PARAMETERS, T, TI>(parameters, dynamics_difficulty) ? (T)1 : (T)0;
    stats.inertia_trace = parameters.dynamics.J[0][0] + parameters.dynamics.J[1][1] + parameters.dynamics.J[2][2];
    stats.thrust_scale = estimated_max_thrust<PARAMETERS, T, TI>(parameters)
        / std::max((T)1e-12, estimated_max_thrust<PARAMETERS, T, TI>(nominal_parameters));
    stats.torque_scale = average_rotor_torque_constant_abs<PARAMETERS, T, TI>(parameters)
        / std::max((T)1e-12, average_rotor_torque_constant_abs<PARAMETERS, T, TI>(nominal_parameters));
    stats.dynamics_size_mass_bin = (T)bins.size_mass;
    stats.dynamics_thrust_to_weight_bin = (T)bins.thrust_to_weight;
    stats.dynamics_torque_to_inertia_bin = (T)bins.torque_to_inertia;
    stats.dynamics_motor_delay_bin = (T)bins.motor_delay;
    stats.dynamics_curve_shape_bin = (T)bins.curve_shape;
    stats.equivalent_dynamics_diag_thrust_to_acceleration_gain = equivalent_dynamics_diag.thrust_to_acceleration_gain;
    stats.equivalent_dynamics_diag_roll_pitch_torque_to_angular_acceleration_gain = equivalent_dynamics_diag.roll_pitch_torque_to_angular_acceleration_gain;
    stats.equivalent_dynamics_diag_yaw_torque_to_angular_acceleration_gain = equivalent_dynamics_diag.yaw_torque_to_angular_acceleration_gain;
    stats.equivalent_dynamics_diag_motor_rise_time_constant = equivalent_dynamics_diag.motor_rise_time_constant;
    stats.equivalent_dynamics_diag_motor_fall_time_constant = equivalent_dynamics_diag.motor_fall_time_constant;
    stats.equivalent_dynamics_diag_thrust_curve_shape = equivalent_dynamics_diag.thrust_curve_shape;
    stats.equivalent_dynamics_diag_torque_curve_shape = equivalent_dynamics_diag.torque_curve_shape;
    stats.equivalent_dynamics_diag_residual_force_bias = equivalent_dynamics_diag.residual_force_bias;
    stats.equivalent_dynamics_diag_residual_torque_bias = equivalent_dynamics_diag.residual_torque_bias;
    return stats;
}

template <typename PARAMETERS, typename T, typename TI>
void add_sampled_dynamics_diagnostics(
    SampledDynamicsDiagnostics<PARAMETERS, T, TI>& accumulator,
    const SampledDynamicsDiagnostics<PARAMETERS, T, TI>& value
){
    accumulator.mass += value.mass;
    accumulator.thrust_to_weight_ratio += value.thrust_to_weight_ratio;
    accumulator.torque_to_inertia_ratio += value.torque_to_inertia_ratio;
    accumulator.motor_tau_mean += value.motor_tau_mean;
    accumulator.motor_tau_min += value.motor_tau_min;
    accumulator.motor_tau_max += value.motor_tau_max;
    accumulator.sampled_tau_rise += value.sampled_tau_rise;
    accumulator.sampled_tau_fall += value.sampled_tau_fall;
    accumulator.sampled_curve_shape += value.sampled_curve_shape;
    accumulator.sampled_parameters_inside_allowed_ranges += value.sampled_parameters_inside_allowed_ranges;
    accumulator.inertia_trace += value.inertia_trace;
    accumulator.thrust_scale += value.thrust_scale;
    accumulator.torque_scale += value.torque_scale;
    accumulator.dynamics_size_mass_bin += value.dynamics_size_mass_bin;
    accumulator.dynamics_thrust_to_weight_bin += value.dynamics_thrust_to_weight_bin;
    accumulator.dynamics_torque_to_inertia_bin += value.dynamics_torque_to_inertia_bin;
    accumulator.dynamics_motor_delay_bin += value.dynamics_motor_delay_bin;
    accumulator.dynamics_curve_shape_bin += value.dynamics_curve_shape_bin;
    accumulator.equivalent_dynamics_diag_thrust_to_acceleration_gain += value.equivalent_dynamics_diag_thrust_to_acceleration_gain;
    accumulator.equivalent_dynamics_diag_roll_pitch_torque_to_angular_acceleration_gain += value.equivalent_dynamics_diag_roll_pitch_torque_to_angular_acceleration_gain;
    accumulator.equivalent_dynamics_diag_yaw_torque_to_angular_acceleration_gain += value.equivalent_dynamics_diag_yaw_torque_to_angular_acceleration_gain;
    accumulator.equivalent_dynamics_diag_motor_rise_time_constant += value.equivalent_dynamics_diag_motor_rise_time_constant;
    accumulator.equivalent_dynamics_diag_motor_fall_time_constant += value.equivalent_dynamics_diag_motor_fall_time_constant;
    accumulator.equivalent_dynamics_diag_thrust_curve_shape += value.equivalent_dynamics_diag_thrust_curve_shape;
    accumulator.equivalent_dynamics_diag_torque_curve_shape += value.equivalent_dynamics_diag_torque_curve_shape;
    accumulator.equivalent_dynamics_diag_residual_force_bias += value.equivalent_dynamics_diag_residual_force_bias;
    accumulator.equivalent_dynamics_diag_residual_torque_bias += value.equivalent_dynamics_diag_residual_torque_bias;
}

template <typename PARAMETERS, typename T, typename TI>
void scale_sampled_dynamics_diagnostics(SampledDynamicsDiagnostics<PARAMETERS, T, TI>& value, T scale){
    value.mass *= scale;
    value.thrust_to_weight_ratio *= scale;
    value.torque_to_inertia_ratio *= scale;
    value.motor_tau_mean *= scale;
    value.motor_tau_min *= scale;
    value.motor_tau_max *= scale;
    value.sampled_tau_rise *= scale;
    value.sampled_tau_fall *= scale;
    value.sampled_curve_shape *= scale;
    value.sampled_parameters_inside_allowed_ranges *= scale;
    value.inertia_trace *= scale;
    value.thrust_scale *= scale;
    value.torque_scale *= scale;
    value.dynamics_size_mass_bin *= scale;
    value.dynamics_thrust_to_weight_bin *= scale;
    value.dynamics_torque_to_inertia_bin *= scale;
    value.dynamics_motor_delay_bin *= scale;
    value.dynamics_curve_shape_bin *= scale;
    value.equivalent_dynamics_diag_thrust_to_acceleration_gain *= scale;
    value.equivalent_dynamics_diag_roll_pitch_torque_to_angular_acceleration_gain *= scale;
    value.equivalent_dynamics_diag_yaw_torque_to_angular_acceleration_gain *= scale;
    value.equivalent_dynamics_diag_motor_rise_time_constant *= scale;
    value.equivalent_dynamics_diag_motor_fall_time_constant *= scale;
    value.equivalent_dynamics_diag_thrust_curve_shape *= scale;
    value.equivalent_dynamics_diag_torque_curve_shape *= scale;
    value.equivalent_dynamics_diag_residual_force_bias *= scale;
    value.equivalent_dynamics_diag_residual_torque_bias *= scale;
}

template <typename ENVIRONMENT>
void configure_sampled_dynamics_level(ENVIRONMENT& sampling_env, fp::SampledDynamicsLevel level){
    using T = typename ENVIRONMENT::T;
    using TI = typename ENVIRONMENT::TI;
    using PARAMETERS = typename ENVIRONMENT::Parameters;
    if(level == fp::SampledDynamicsLevel::BROAD || level == fp::SampledDynamicsLevel::FIXED){
        return;
    }
    const T variation = level == fp::SampledDynamicsLevel::NARROW ? (T)0.10 : (T)0.25;
    auto& domain = sampling_env.parameters.domain_randomization;
    const PARAMETERS& nominal = sampling_env.parameters;
    const T nominal_mass = nominal.dynamics.mass;
    const T nominal_ttw = estimated_thrust_to_weight<PARAMETERS, T, TI>(nominal);
    const T nominal_torque_to_inertia = estimated_torque_to_inertia_ratio<PARAMETERS, T, TI>(nominal);
    T nominal_rising_mean;
    T nominal_tau_min;
    T nominal_tau_max;
    motor_tau_stats<PARAMETERS, T, TI>(nominal, nominal_rising_mean, nominal_tau_min, nominal_tau_max);
    const T nominal_rising = nominal.dynamics.rotor_time_constants_rising[0];
    const T nominal_falling = nominal.dynamics.rotor_time_constants_falling[0];
    const T nominal_torque_constant = average_rotor_torque_constant_abs<PARAMETERS, T, TI>(nominal);

    domain.mass_min = std::max((T)1e-4, nominal_mass * ((T)1 - variation));
    domain.mass_max = std::max(domain.mass_min + (T)1e-5, nominal_mass * ((T)1 + variation));
    domain.thrust_to_weight_min = std::max((T)1.5, nominal_ttw * ((T)1 - variation));
    domain.thrust_to_weight_max = std::max(domain.thrust_to_weight_min + (T)1e-3, nominal_ttw * ((T)1 + variation));
    domain.torque_to_inertia_min = std::max((T)1e-6, nominal_torque_to_inertia * ((T)1 - variation));
    domain.torque_to_inertia_max = std::max(domain.torque_to_inertia_min + (T)1e-3, nominal_torque_to_inertia * ((T)1 + variation));
    domain.mass_size_deviation = variation;
    domain.rotor_time_constant_rising_min = std::max((T)0.005, nominal_rising * ((T)1 - variation));
    domain.rotor_time_constant_rising_max = std::max(domain.rotor_time_constant_rising_min + (T)1e-4, nominal_rising * ((T)1 + variation));
    domain.rotor_time_constant_falling_min = std::max((T)0.005, nominal_falling * ((T)1 - variation));
    domain.rotor_time_constant_falling_max = std::max(domain.rotor_time_constant_falling_min + (T)1e-4, nominal_falling * ((T)1 + variation));
    domain.rotor_torque_constant_min = std::max((T)1e-6, nominal_torque_constant * ((T)1 - variation));
    domain.rotor_torque_constant_max = std::max(domain.rotor_torque_constant_min + (T)1e-6, nominal_torque_constant * ((T)1 + variation));
    domain.orientation_offset_angle_max = (T)0;
    domain.disturbance_force_max = level == fp::SampledDynamicsLevel::NARROW ? (T)0.05 : (T)0.15;
}

template <typename T>
fp::SampledDynamicsLevel effective_sampled_dynamics_level(const RuntimeOptions& options, T dynamics_difficulty){
    if(!options.sample_dynamics){
        return fp::SampledDynamicsLevel::FIXED;
    }
    if(!options.sampled_dynamics_curriculum_levels){
        return options.sampled_dynamics_level;
    }
    if(dynamics_difficulty < (T)0.25){
        return fp::SampledDynamicsLevel::FIXED;
    }
    if(dynamics_difficulty < (T)0.60){
        return fp::SampledDynamicsLevel::NARROW;
    }
    if(dynamics_difficulty < (T)0.90){
        return fp::SampledDynamicsLevel::MEDIUM;
    }
    return options.sampled_dynamics_level == fp::SampledDynamicsLevel::BROAD
        ? fp::SampledDynamicsLevel::BROAD
        : fp::SampledDynamicsLevel::MEDIUM;
}

template <typename PARAMETERS, typename T, typename TI>
bool sampled_parameters_safe(const PARAMETERS& parameters, T dynamics_difficulty){
    return sampled_parameters_inside_allowed_ranges<PARAMETERS, T, TI>(parameters, dynamics_difficulty);
}

template <typename ENVIRONMENT, typename T>
void clamp_sampling_domain_to_existing_safety(ENVIRONMENT& sampling_env, T dynamics_difficulty){
    auto& domain = sampling_env.parameters.domain_randomization;
    const T safe_mass_max = (T)0.06 + dynamics_difficulty * (fp::SAMPLED_DYNAMICS_MAX_MASS - (T)0.06);
    domain.mass_max = std::min(domain.mass_max, safe_mass_max);
    domain.thrust_to_weight_min = std::max(domain.thrust_to_weight_min, fp::SAMPLED_DYNAMICS_MIN_THRUST_TO_WEIGHT);
    if(domain.mass_max < domain.mass_min){
        domain.mass_min = domain.mass_max;
    }
    if(domain.thrust_to_weight_max < domain.thrust_to_weight_min){
        domain.thrust_to_weight_min = domain.thrust_to_weight_max;
    }
    if(domain.torque_to_inertia_max < domain.torque_to_inertia_min){
        domain.torque_to_inertia_min = domain.torque_to_inertia_max;
    }
    if(domain.rotor_time_constant_rising_max < domain.rotor_time_constant_rising_min){
        domain.rotor_time_constant_rising_min = domain.rotor_time_constant_rising_max;
    }
    if(domain.rotor_time_constant_falling_max < domain.rotor_time_constant_falling_min){
        domain.rotor_time_constant_falling_min = domain.rotor_time_constant_falling_max;
    }
    if(domain.rotor_torque_constant_max < domain.rotor_torque_constant_min){
        domain.rotor_torque_constant_min = domain.rotor_torque_constant_max;
    }
}

template <typename DEVICE, typename ENVIRONMENT, typename PARAMETERS, typename RNG, typename T, typename TI>
void sample_equivalent_input_response_parameters(
    DEVICE& device,
    ENVIRONMENT& env,
    const ENVIRONMENT& sampling_env,
    PARAMETERS& parameters,
    RNG& rng,
    const eq_dyn::DynamicsBins<TI>* balanced_bins
){
    rlt::initial_parameters(device, env, parameters);
    parameters.domain_randomization = sampling_env.parameters.domain_randomization;
    const auto& domain = parameters.domain_randomization;
    T curve_shape_min = 0;
    T curve_shape_max = 1;
    if(balanced_bins != nullptr){
        eq_dyn::restrict_range_to_bin<T, TI>(curve_shape_min, curve_shape_max, balanced_bins->curve_shape, fp::DYNAMICS_BALANCE_BINS);
    }

    const T mass_size_min = std::cbrt(std::max((T)1e-12, domain.mass_min));
    const T mass_size_max = std::cbrt(std::max((T)1e-12, domain.mass_max));
    const T sampled_size = sample_uniform_range<T, DEVICE, RNG>(device, rng, mass_size_min, mass_size_max);
    const T sampled_mass = clamp_value<T>(sampled_size * sampled_size * sampled_size, domain.mass_min, domain.mass_max);
    const T sampled_thrust_to_weight = sample_uniform_range<T, DEVICE, RNG>(device, rng, domain.thrust_to_weight_min, domain.thrust_to_weight_max);
    const T sampled_torque_to_inertia = sample_uniform_range<T, DEVICE, RNG>(device, rng, domain.torque_to_inertia_min, domain.torque_to_inertia_max);
    const T sampled_tau_rise = sample_uniform_range<T, DEVICE, RNG>(device, rng, domain.rotor_time_constant_rising_min, domain.rotor_time_constant_rising_max);
    const T sampled_tau_fall = sample_uniform_range<T, DEVICE, RNG>(device, rng, domain.rotor_time_constant_falling_min, domain.rotor_time_constant_falling_max);
    const T sampled_curve_shape = sample_uniform_range<T, DEVICE, RNG>(device, rng, curve_shape_min, curve_shape_max);

    parameters.dynamics.mass = sampled_mass;
    const T nominal_mass = std::max((T)1e-12, env.parameters.dynamics.mass);
    const T size_scale = std::cbrt(sampled_mass / nominal_mass);
    const T geometry_factor = std::max((T)0.1, (T)1 + ((T)2 * sampled_curve_shape - (T)1) * domain.mass_size_deviation);
    const T arm_scale = size_scale * geometry_factor;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        for(TI axis_i = 0; axis_i < 3; axis_i++){
            parameters.dynamics.rotor_positions[rotor_i][axis_i] = env.parameters.dynamics.rotor_positions[rotor_i][axis_i] * arm_scale;
        }
    }

    const T gravity_norm = std::max((T)1e-6, std::abs(parameters.dynamics.gravity[2]));
    const T total_max_thrust = sampled_thrust_to_weight * sampled_mass * gravity_norm;
    const T per_rotor_max_thrust = total_max_thrust / (T)4;
    const T max_action = std::max((T)1e-6, parameters.dynamics.action_limit.max);
    const T quadratic_fraction = clamp_value<T>(sampled_curve_shape, (T)0.05, (T)0.95);
    const T remaining_fraction = std::max((T)0, (T)1 - quadratic_fraction);
    const T offset_fraction = remaining_fraction * (T)0.20 * ((T)1 - sampled_curve_shape);
    const T linear_fraction = std::max((T)0, remaining_fraction - offset_fraction);
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        parameters.dynamics.rotor_thrust_coefficients[rotor_i][0] = per_rotor_max_thrust * offset_fraction;
        parameters.dynamics.rotor_thrust_coefficients[rotor_i][1] = per_rotor_max_thrust * linear_fraction / max_action;
        parameters.dynamics.rotor_thrust_coefficients[rotor_i][2] = per_rotor_max_thrust * quadratic_fraction / (max_action * max_action);
    }

    const T rotor_distance_x = std::max((T)1e-9, std::abs(parameters.dynamics.rotor_positions[0][0]));
    const T max_roll_pitch_torque = rotor_distance_x * (T)1.414213562373095 * per_rotor_max_thrust;
    const T j_x = std::max((T)1e-12, max_roll_pitch_torque / std::max((T)1e-12, sampled_torque_to_inertia));
    const T nominal_j_x = std::max((T)1e-12, env.parameters.dynamics.J[0][0]);
    const T j_y = j_x * std::max((T)1e-6, env.parameters.dynamics.J[1][1] / nominal_j_x);
    const T j_z = j_x * std::max((T)1e-6, env.parameters.dynamics.J[2][2] / nominal_j_x);
    for(TI row_i = 0; row_i < 3; row_i++){
        for(TI col_i = 0; col_i < 3; col_i++){
            parameters.dynamics.J[row_i][col_i] = (T)0;
            parameters.dynamics.J_inv[row_i][col_i] = (T)0;
        }
    }
    parameters.dynamics.J[0][0] = j_x;
    parameters.dynamics.J[1][1] = j_y;
    parameters.dynamics.J[2][2] = j_z;
    parameters.dynamics.J_inv[0][0] = (T)1 / j_x;
    parameters.dynamics.J_inv[1][1] = (T)1 / j_y;
    parameters.dynamics.J_inv[2][2] = (T)1 / j_z;

    const T torque_to_inertia_norm = (domain.torque_to_inertia_max > domain.torque_to_inertia_min)
        ? (sampled_torque_to_inertia - domain.torque_to_inertia_min) / (domain.torque_to_inertia_max - domain.torque_to_inertia_min)
        : (T)0.5;
    const T torque_curve_shape = clamp_value<T>(((T)0.5 * sampled_curve_shape) + ((T)0.5 * torque_to_inertia_norm), (T)0, (T)1);
    const T rotor_torque_constant = domain.rotor_torque_constant_min
        + (domain.rotor_torque_constant_max - domain.rotor_torque_constant_min) * torque_curve_shape;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        parameters.dynamics.rotor_torque_constants[rotor_i] = rotor_torque_constant;
        parameters.dynamics.rotor_time_constants_rising[rotor_i] = sampled_tau_rise;
        parameters.dynamics.rotor_time_constants_falling[rotor_i] = sampled_tau_fall;
    }

    const T surplus_thrust_to_weight = std::max((T)0, sampled_thrust_to_weight - (T)1);
    const T disturbance_multiple = sample_uniform_range<T, DEVICE, RNG>(
        device,
        rng,
        (T)0,
        surplus_thrust_to_weight * domain.disturbance_force_max
    );
    parameters.disturbances.random_force.mean = 0;
    parameters.disturbances.random_force.std = disturbance_multiple * sampled_thrust_to_weight * sampled_mass / (T)3;
    parameters.dynamics.hovering_throttle_relative = solve_hover_throttle_relative<PARAMETERS, T, TI>(parameters);

    T max_rotor_distance = 0;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        const T x = parameters.dynamics.rotor_positions[rotor_i][0];
        const T y = parameters.dynamics.rotor_positions[rotor_i][1];
        const T z = parameters.dynamics.rotor_positions[rotor_i][2];
        max_rotor_distance = std::max(max_rotor_distance, std::sqrt(x * x + y * y + z * z));
    }
    if(max_rotor_distance > (T)0){
        parameters.mdp.termination.position_threshold = max_rotor_distance * (T)20;
        parameters.mdp.init.max_position = max_rotor_distance * (T)10;
    }
}

template <typename DEVICE, typename ENVIRONMENT, typename PARAMETERS, typename RNG, typename T, typename TI>
TI sample_training_parameters(
    DEVICE& device,
    ENVIRONMENT& env,
    PARAMETERS& parameters,
    RNG& rng,
    const RuntimeOptions& options,
    T dynamics_difficulty,
    const eq_dyn::DynamicsBins<TI>* balanced_bins
){
    const auto level = effective_sampled_dynamics_level<T>(options, dynamics_difficulty);
    if(!options.sample_dynamics || level == fp::SampledDynamicsLevel::FIXED){
        rlt::initial_parameters(device, env, parameters);
        return 0;
    }
    const T safety_difficulty = options.dynamics_curriculum ? dynamics_difficulty : (T)1;
    ENVIRONMENT sampling_env = env;
    configure_sampled_dynamics_level(sampling_env, level);
    clamp_sampling_domain_to_existing_safety<ENVIRONMENT, T>(sampling_env, safety_difficulty);
    if(options.balanced_dynamics_sampling && balanced_bins != nullptr){
        eq_dyn::restrict_domain_to_bins<typename ENVIRONMENT::Parameters::DomainRandomization, T, TI>(
            sampling_env.parameters.domain_randomization,
            *balanced_bins,
            fp::DYNAMICS_BALANCE_BINS
        );
    }
    if(!options.dynamics_curriculum || dynamics_difficulty <= (T)0){
        if(options.dynamics_curriculum){
            rlt::initial_parameters(device, env, parameters);
        }
        else{
            TI rejected = 0;
            for(TI attempt_i = 0; attempt_i < 64; attempt_i++){
                sample_equivalent_input_response_parameters<DEVICE, ENVIRONMENT, PARAMETERS, RNG, T, TI>(
                    device, env, sampling_env, parameters, rng, balanced_bins
                );
                if(sampled_parameters_safe<PARAMETERS, T, TI>(parameters, safety_difficulty)){
                    return rejected;
                }
                rejected++;
            }
            rlt::initial_parameters(device, env, parameters);
            return rejected;
        }
        return 0;
    }
    TI rejected = 0;
    for(TI attempt_i = 0; attempt_i < 64; attempt_i++){
        sample_equivalent_input_response_parameters<DEVICE, ENVIRONMENT, PARAMETERS, RNG, T, TI>(
            device, env, sampling_env, parameters, rng, balanced_bins
        );
        if(sampled_parameters_safe<PARAMETERS, T, TI>(parameters, safety_difficulty)){
            return rejected;
        }
        rejected++;
    }
    rlt::initial_parameters(device, env, parameters);
    return rejected;
}

template <typename T>
T bounded_action(T raw_action, bool enabled, T bound, T& derivative, bool& clamped){
    derivative = (T)1;
    clamped = false;
    if(!enabled) return raw_action;
    if(raw_action > bound){
        derivative = (T)0;
        clamped = true;
        return bound;
    }
    if(raw_action < -bound){
        derivative = (T)0;
        clamped = true;
        return -bound;
    }
    return raw_action;
}

int main(int argc, char** argv){
    using DEVICE = fp::DEVICE;
    using RNG = fp::RNG;
    using TI = fp::TI;
    using T = fp::T;
    using ENVIRONMENT = fp::ENVIRONMENT;
    using ACTOR = fp::ACTOR;
    using ROLLOUT_ACTOR = fp::ROLLOUT_ACTOR;
    using STATE = typename ENVIRONMENT::State;
    using PARAMETERS = typename ENVIRONMENT::Parameters;
    static constexpr TI MAX_HORIZON = fp::DIFF_TRAINING_HORIZON;
    static constexpr TI MAX_BATCH_SIZE = fp::DIFF_TRAINING_BATCH_SIZE;

    RuntimeOptions options;
    if(!fp::parse_options(argc, argv, options)){
        return 1;
    }
    if(options.gpu_rollout || options.gpu_validate_against_cpu || options.gpu_benchmark){
        std::cerr << "WARNING: foundation_policy_diff_pre_training is the CPU baseline target. "
                  << "Use foundation_policy_diff_pre_training_cuda for --gpu-rollout validation and benchmarks. "
                  << "Continuing with the unchanged CPU rollout path.\n";
    }
    if(options.horizon == 0 || options.horizon > MAX_HORIZON){
        std::cerr << "--horizon must be in [1, " << MAX_HORIZON << "] for this build.\n";
        return 1;
    }
    if(options.batch_size == 0 || options.batch_size > MAX_BATCH_SIZE){
        std::cerr << "--batch-size must be in [1, " << MAX_BATCH_SIZE << "] for this build.\n";
        return 1;
    }
    if(options.terminal_ramp_min < (T)0 || options.terminal_ramp_min > (T)1){
        std::cerr << "--terminal-ramp-min must be in [0, 1].\n";
        return 1;
    }
    if(options.terminal_ramp_steps < 0){
        std::cerr << "--terminal-ramp-steps must be non-negative.\n";
        return 1;
    }
    if(options.diff_model == DiffModel::L2F_APPROX && options.horizon != MAX_HORIZON && !options.eval_only){
        std::cerr << "l2f_approx training path currently uses the compile-time horizon " << MAX_HORIZON << ".\n";
        return 1;
    }
    if(options.diff_model == DiffModel::L2F_APPROX && options.horizon_curriculum && !options.eval_only){
        std::cerr << "l2f_approx training path does not support --horizon-curriculum.\n";
        return 1;
    }

    DEVICE device;
    RNG rng;
    ENVIRONMENT env;
    ACTOR actor;
    typename ACTOR::template Buffer<> actor_buffer;
    ROLLOUT_ACTOR rollout_actor;
    typename ROLLOUT_ACTOR::template Buffer<fp::DYNAMIC_ALLOCATION> rollout_actor_buffer;
    typename ROLLOUT_ACTOR::template State<fp::DYNAMIC_ALLOCATION> rollout_actor_state;
    fp::OPTIMIZER actor_optimizer;

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, fp::DIFF_TRAINING_SEQUENCE_LENGTH, fp::DIFF_TRAINING_BATCH_SIZE, fp::POLICY_INPUT_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, fp::DIFF_TRAINING_BATCH_SIZE, fp::POLICY_INPUT_DIM>>> step_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, fp::DIFF_TRAINING_BATCH_SIZE, fp::ACTOR_OUTPUT_DIM>>> step_actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, ACTOR::OUTPUT_SHAPE>> d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, fp::CRITIC_OUTPUT_SHAPE>> q_targets;
    rlt::Tensor<rlt::tensor::Specification<T, TI, fp::CRITIC_OUTPUT_SHAPE>> q_weights;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, fp::DIFF_TRAINING_SEQUENCE_LENGTH, fp::DIFF_TRAINING_BATCH_SIZE, 1>>> reset;

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM>> observation_row;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM>> predicted_observation_row;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action_row;

    rlt::init(device);
    rlt::malloc(device, rng);
    fp::rdac_malloc(device, actor);
    fp::rdac_malloc_buffer<DEVICE, typename ACTOR::CAPABILITY_TYPE, fp::DYNAMIC_ALLOCATION>(device, actor_buffer);
    fp::rdac_malloc(device, rollout_actor);
    fp::rdac_malloc_buffer<DEVICE, typename ROLLOUT_ACTOR::CAPABILITY_TYPE, fp::DYNAMIC_ALLOCATION>(device, rollout_actor_buffer);
    rlt::malloc(device, rollout_actor_state);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, observations);
    rlt::malloc(device, step_observations);
    rlt::malloc(device, step_actions);
    rlt::malloc(device, d_output);
    rlt::malloc(device, q_targets);
    rlt::malloc(device, q_weights);
    rlt::malloc(device, reset);
    rlt::malloc(device, observation_row);
    rlt::malloc(device, predicted_observation_row);
    rlt::malloc(device, action_row);

    rlt::init(device, rng, options.seed);
    rlt::init(device, env);
    fp::rdac_init_weights(device, actor, rng);
    fp::rdac_reset_optimizer_state(device, actor_optimizer, actor);

    if(options.sampler_dump_samples > 0){
        if(options.sampler_dump_path.empty()){
            std::cerr << "--sampler-dump-path is required when --sampler-dump-samples is positive.\n";
            return 1;
        }
        std::ofstream sampler_dump(options.sampler_dump_path);
        if(!sampler_dump.is_open()){
            std::cerr << "Failed to open sampler dump path: " << options.sampler_dump_path << "\n";
            return 1;
        }
        sampler_dump << "sample_i,training_step,batch_i,"
                     << "dynamics_size_mass_bin,dynamics_thrust_to_weight_bin,dynamics_torque_to_inertia_bin,dynamics_motor_delay_bin,dynamics_curve_shape_bin,"
                     << "dynamics_group_key,rejected_before_accept,"
                     << "mass,thrust_to_weight_ratio,torque_to_inertia_ratio,sampled_tau_rise,sampled_tau_fall,sampled_curve_shape,"
                     << "rotor_torque_constant,inertia_trace,thrust_scale,torque_scale,sampled_parameters_inside_allowed_ranges\n";
        TI rejected_total = 0;
        for(TI sample_i = 0; sample_i < options.sampler_dump_samples; sample_i++){
            const TI training_step = sample_i / std::max((TI)1, options.batch_size);
            const TI batch_i = sample_i % std::max((TI)1, options.batch_size);
            const auto scheduled_bins = scheduled_balanced_dynamics_bins<TI>(training_step, batch_i, options.batch_size, fp::DYNAMICS_BALANCE_BINS);
            PARAMETERS sample_parameters;
            const TI rejected = sample_training_parameters<DEVICE, ENVIRONMENT, PARAMETERS, RNG, T, TI>(
                device,
                env,
                sample_parameters,
                rng,
                options,
                (T)1,
                options.balanced_dynamics_sampling && options.sample_dynamics ? &scheduled_bins : nullptr
            );
            rejected_total += rejected;
            const auto dump_bins = options.balanced_dynamics_sampling && options.sample_dynamics
                ? scheduled_bins
                : eq_dyn::bins_from_parameters<PARAMETERS, TI>(sample_parameters, fp::DYNAMICS_BALANCE_BINS);
            const TI group_key = ((((dump_bins.size_mass * fp::DYNAMICS_BALANCE_BINS + dump_bins.thrust_to_weight)
                * fp::DYNAMICS_BALANCE_BINS + dump_bins.torque_to_inertia)
                * fp::DYNAMICS_BALANCE_BINS + dump_bins.motor_delay)
                * fp::DYNAMICS_BALANCE_BINS + dump_bins.curve_shape);
            const auto diag = sampled_dynamics_diagnostics<PARAMETERS, T, TI>(
                sample_parameters,
                env.parameters,
                dump_bins,
                (T)1
            );
            sampler_dump << sample_i << ","
                         << training_step << ","
                         << batch_i << ","
                         << dump_bins.size_mass << ","
                         << dump_bins.thrust_to_weight << ","
                         << dump_bins.torque_to_inertia << ","
                         << dump_bins.motor_delay << ","
                         << dump_bins.curve_shape << ","
                         << group_key << ","
                         << rejected << ","
                         << diag.mass << ","
                         << diag.thrust_to_weight_ratio << ","
                         << diag.torque_to_inertia_ratio << ","
                         << diag.sampled_tau_rise << ","
                         << diag.sampled_tau_fall << ","
                         << diag.sampled_curve_shape << ","
                         << average_rotor_torque_constant_abs<PARAMETERS, T, TI>(sample_parameters) << ","
                         << diag.inertia_trace << ","
                         << diag.thrust_scale << ","
                         << diag.torque_scale << ","
                         << (diag.sampled_parameters_inside_allowed_ranges > (T)0.999 ? "true" : "false")
                         << "\n";
        }
        std::cout << "sampler_dump_path=" << options.sampler_dump_path
                  << " sampler_dump_samples=" << options.sampler_dump_samples
                  << " rejected_dynamics_count=" << rejected_total
                  << " sampled_dynamics_level=" << fp::sampled_dynamics_level_name(options.sample_dynamics ? options.sampled_dynamics_level : fp::SampledDynamicsLevel::FIXED)
                  << " balanced_dynamics_sampling=" << (options.balanced_dynamics_sampling ? "true" : "false")
                  << "\n";
        return 0;
    }

    if(options.checkpoint_inspect){
        const std::string inspect_path = !options.checkpoint_inspect_path.empty()
            ? options.checkpoint_inspect_path
            : options.load_path;
        if(inspect_path.empty()){
            std::cerr << "--checkpoint-inspect requires a path or --load-path\n";
            return 1;
        }
        return fp::inspect_checkpoint_file(inspect_path) ? 0 : 1;
    }

    if(options.checkpoint_convert_old){
        const std::string source_path = !options.checkpoint_convert_old_path.empty()
            ? options.checkpoint_convert_old_path
            : options.load_path;
        const std::string target_path = !options.save_path.empty()
            ? options.save_path
            : source_path + ".v4";
        if(source_path.empty()){
            std::cerr << "--checkpoint-convert-old requires a source path or --load-path\n";
            return 1;
        }
        if(!fp::load_actor_checkpoint(device, actor, source_path)){
            return 1;
        }
        if(!fp::save_actor_optimizer_checkpoint(device, actor, actor_optimizer, target_path)){
            return 1;
        }
        std::cout << "checkpoint_converted_from=" << source_path
                  << " checkpoint_converted_to=" << target_path << "\n";
        return 0;
    }

    bool init_actor_loaded_flag = false;
    std::string actor_checkpoint_to_load;
    if(options.eval_only){
        actor_checkpoint_to_load = options.load_path;
    }
    else if(!options.init_actor_path.empty()){
        actor_checkpoint_to_load = options.init_actor_path;
        init_actor_loaded_flag = true;
    }
    else{
        actor_checkpoint_to_load = options.load_path;
    }
    if(!actor_checkpoint_to_load.empty()){
        const bool loaded_ok = options.load_optimizer
            ? fp::load_actor_optimizer_checkpoint(device, actor, actor_optimizer, actor_checkpoint_to_load, true)
            : fp::load_actor_checkpoint(device, actor, actor_checkpoint_to_load);
        if(!loaded_ok){
            return 1;
        }
        std::cout << "loaded_actor_checkpoint=" << actor_checkpoint_to_load << "\n";
        if(options.load_optimizer){
            std::cout << "loaded_optimizer_state=true optimizer_age="
                      << rlt::get(device, actor_optimizer.age, 0) << "\n";
        }
        if(init_actor_loaded_flag){
            std::cout << "init_actor_loaded=true init_actor_path=" << options.init_actor_path << "\n";
        }
    }

    const fp::LossWeights<T> l2f_approx_weights{
        fp::LOSS_POSITION_WEIGHT,
        options.loss_velocity_weight,
        fp::LOSS_ATTITUDE_WEIGHT,
        options.loss_angular_velocity_weight,
        fp::LOSS_ACTION_MAGNITUDE_WEIGHT,
        fp::LOSS_ACTION_SMOOTHNESS_WEIGHT,
        fp::LOSS_SATURATION_WEIGHT
    };
    const T action_magnitude_center = options.hover_relative_action_magnitude
        ? (T)2 * solve_hover_throttle_relative<PARAMETERS, T, TI>(env.parameters) - (T)1
        : options.action_magnitude_center;
    const l2f_diff::EulerLossWeights<T> euler_weights{
        fp::LOSS_POSITION_WEIGHT,
        options.loss_velocity_weight,
        fp::LOSS_ATTITUDE_WEIGHT,
        options.loss_angular_velocity_weight,
        fp::LOSS_ACTION_MAGNITUDE_WEIGHT,
        fp::LOSS_ACTION_SMOOTHNESS_WEIGHT,
        fp::LOSS_SATURATION_WEIGHT,
        fp::TERMINAL_LOSS_WEIGHT,
        fp::TERMINAL_POSITION_WEIGHT,
        options.terminal_velocity_weight,
        fp::TERMINAL_ATTITUDE_WEIGHT,
        options.terminal_angular_velocity_weight,
        options.loss_clf_weight,
        options.loss_window_clf_weight,
        options.loss_clf_alpha,
        options.loss_clf_position_weight,
        options.loss_clf_velocity_weight,
        options.loss_clf_attitude_weight,
        options.loss_clf_angular_velocity_weight,
        options.loss_outward_velocity_weight,
        options.loss_attitude_control_weight,
        options.attitude_control_k_R,
        options.attitude_control_k_omega,
        action_magnitude_center,
        options.action_saturation_start,
        options.loss_clf_position_velocity_cross_beta,
        options.loss_clf_attitude_angular_velocity_cross_beta,
        options.loss_window_clf_epsilon,
        options.loss_window_clf_huber_delta,
        options.loss_velocity_barrier_weight,
        options.velocity_barrier_safe,
        options.loss_angular_velocity_barrier_weight,
        options.angular_velocity_barrier_safe,
        options.loss_attitude_barrier_weight,
        options.attitude_barrier_safe,
        (T)1,
        (T)0,
        options.hover_relative_action_magnitude
    };
    auto set_euler_action_hover_center = [&](l2f_diff::EulerState<T, TI>& state){
        for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
            state.action_hover_center[action_i] = action_magnitude_center;
        }
    };
    const auto tracking_reference = l2f_diff::zero_tracking_reference<T>();

    auto run_evaluation = [&](){
        fp::rdac_copy(device, device, actor, rollout_actor);
        rlt::Mode<rlt::mode::Evaluation<>> evaluation_mode;
        fp::EvalMetrics<T> eval_metrics;
        TI successes = 0;
        TI near_success_p = 0;
        TI near_success_pv = 0;
        TI invalid_count = 0;
        T total_action_norm = 0;
        std::vector<T> final_positions;
        std::vector<T> final_velocities;
        std::vector<T> final_angular_velocities;
        final_positions.reserve(options.eval_episodes);
        final_velocities.reserve(options.eval_episodes);
        final_angular_velocities.reserve(options.eval_episodes);

        for(TI episode_i = 0; episode_i < options.eval_episodes; episode_i++){
            PARAMETERS parameters;
            if(options.sample_dynamics){
                sample_training_parameters<DEVICE, ENVIRONMENT, PARAMETERS, RNG, T, TI>(device, env, parameters, rng, options, (T)1, nullptr);
            }
            else{
                rlt::initial_parameters(device, env, parameters);
            }
            STATE state;
            rlt::sample_initial_state(device, env, parameters, state, rng);
            l2f_diff::EulerState<T, TI> euler_state;
            l2f_diff::from_l2f_state<STATE, T, TI>(state, euler_state);
            set_euler_action_hover_center(euler_state);
            rlt::reset(device, rollout_actor.trunk, rollout_actor_state, rng);
            T previous_action[ENVIRONMENT::ACTION_DIM] = {};
            T previous_response_error[fp::RESPONSE_ERROR_DIM] = {};

            fp::ScalarTerms<T> episode_terms;
            bool invalid = false;
            T episode_action_norm = 0;
            for(TI step_i = 0; step_i < options.eval_horizon; step_i++){
                if(options.reset_hidden_each_step){
                    rlt::reset(device, rollout_actor.trunk, rollout_actor_state, rng);
                }
                if(options.eval_model == EvalModel::EULER){
                    l2f_diff::observe_with_reference<T, TI>(euler_state, tracking_reference, observation_row);
                }
                else{
                    rlt::observe(device, env, parameters, state, typename ENVIRONMENT::Observation{}, observation_row, rng);
                    l2f_diff::apply_reference_error_to_observation<T, TI>(observation_row, tracking_reference);
                }
                rlt::set_all(device, step_observations, (T)0);
                for(TI observation_i = 0; observation_i < ENVIRONMENT::Observation::DIM; observation_i++){
                    rlt::set(device, step_observations, rlt::get(observation_row, 0, observation_i), 0, observation_i);
                }
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    rlt::set(device, step_observations, previous_action[action_i], 0, ENVIRONMENT::Observation::DIM + action_i);
                }
                for(TI error_i = 0; error_i < fp::RESPONSE_ERROR_DIM; error_i++){
                    rlt::set(device, step_observations, previous_response_error[error_i], 0, ENVIRONMENT::Observation::DIM + ENVIRONMENT::ACTION_DIM + error_i);
                }
                fp::rdac_evaluate_step(device, rollout_actor, step_observations, rollout_actor_state, step_actions, rollout_actor_buffer, rng, evaluation_mode);
                T action[4];
                T action_norm_sq = 0;
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    T derivative;
                    bool clamped;
                    action[action_i] = bounded_action<T>(
                        rlt::get(device, step_actions, 0, action_i),
                        options.action_bound_enabled,
                        options.action_bound_value,
                        derivative,
                        clamped
                    );
                    action_norm_sq += action[action_i] * action[action_i];
                    rlt::set(action_row, 0, action_i, action[action_i]);
                    previous_action[action_i] = action[action_i];
                }
                episode_action_norm += std::sqrt(action_norm_sq);
                if(options.eval_model == EvalModel::EULER){
                    l2f_diff::EulerState<T, TI> next_state;
                    l2f_diff::EulerStepCache<T> cache;
                    l2f_diff::step<PARAMETERS, T, TI>(parameters, euler_state, action, next_state, cache);
                    euler_state = next_state;
                    auto terms = fp::euler_state_loss_terms<T, TI>(euler_state, tracking_reference, euler_weights);
                    episode_terms.total += terms.total / (T)options.eval_horizon;
                }
                else{
                    STATE next_state;
                    rlt::step(device, env, parameters, state, action_row, next_state, rng);
                    state = next_state;
                    auto terms = fp::l2f_state_loss_terms<STATE, T, TI>(state, tracking_reference, l2f_approx_weights);
                    episode_terms.total += terms.total / (T)options.eval_horizon;
                }
            }

            T final_p = 0;
            T final_v = 0;
            T final_w = 0;
            if(options.eval_model == EvalModel::EULER){
                final_p = fp::euler_position_error_norm<T, TI>(euler_state, tracking_reference);
                final_v = fp::euler_velocity_error_norm<T, TI>(euler_state, tracking_reference);
                final_w = fp::norm3(euler_state.omega);
            }
            else{
                final_p = fp::l2f_position_error_norm<STATE, T, TI>(state, tracking_reference);
                final_v = fp::l2f_velocity_error_norm<STATE, T, TI>(state, tracking_reference);
                final_w = fp::norm3(state.angular_velocity);
            }
            invalid = invalid || !fp::finite_value(episode_terms.total) || !fp::finite_value(final_p) || !fp::finite_value(final_v) || !fp::finite_value(final_w);
            if(invalid){
                invalid_count++;
            }
            else{
                final_positions.push_back(final_p);
                final_velocities.push_back(final_v);
                final_angular_velocities.push_back(final_w);
                if(final_p < fp::SUCCESS_POSITION_THRESHOLD){
                    near_success_p++;
                }
                if(final_p < fp::SUCCESS_POSITION_THRESHOLD && final_v < fp::SUCCESS_VELOCITY_THRESHOLD){
                    near_success_pv++;
                }
                if(final_p < fp::SUCCESS_POSITION_THRESHOLD && final_v < fp::SUCCESS_VELOCITY_THRESHOLD && final_w < fp::SUCCESS_ANGULAR_VELOCITY_THRESHOLD){
                    successes++;
                }
            }
            eval_metrics.mean_total_loss += episode_terms.total;
            eval_metrics.mean_final_position_norm += final_p;
            eval_metrics.mean_final_velocity_norm += final_v;
            eval_metrics.mean_final_angular_velocity_norm += final_w;
            total_action_norm += episode_action_norm / (T)options.eval_horizon;
        }

        const T normalizer = options.eval_episodes > 0 ? (T)1 / (T)options.eval_episodes : (T)1;
        eval_metrics.success_rate = successes * normalizer;
        eval_metrics.near_success_rate_p = near_success_p * normalizer;
        eval_metrics.near_success_rate_pv = near_success_pv * normalizer;
        eval_metrics.mean_total_loss *= normalizer;
        eval_metrics.mean_final_position_norm *= normalizer;
        eval_metrics.mean_final_velocity_norm *= normalizer;
        eval_metrics.mean_final_angular_velocity_norm *= normalizer;
        eval_metrics.mean_action_norm = total_action_norm * normalizer;
        eval_metrics.invalid_or_nan_rate = invalid_count * normalizer;
        auto percentile = [](std::vector<T>& values, T q){
            if(values.empty()) return (T)0;
            std::sort(values.begin(), values.end());
            const T scaled = q * (T)(values.size() - 1);
            const std::size_t index = (std::size_t)std::round(scaled);
            return values[std::min(index, values.size() - 1)];
        };
        eval_metrics.median_final_position_norm = percentile(final_positions, (T)0.5);
        eval_metrics.median_final_velocity_norm = percentile(final_velocities, (T)0.5);
        eval_metrics.median_final_angular_velocity_norm = percentile(final_angular_velocities, (T)0.5);
        eval_metrics.p90_final_position_norm = percentile(final_positions, (T)0.9);
        eval_metrics.p90_final_velocity_norm = percentile(final_velocities, (T)0.9);

        std::cout << "evaluation"
                  << " eval_model=" << fp::eval_model_name(options.eval_model)
                  << " mode=" << (options.sample_dynamics ? "sampled" : "fixed")
                  << " sampled_dynamics_level=" << fp::sampled_dynamics_level_name(options.sample_dynamics ? options.sampled_dynamics_level : fp::SampledDynamicsLevel::FIXED)
                  << " eval_episodes=" << options.eval_episodes
                  << " eval_horizon=" << options.eval_horizon
                  << " success_rate=" << eval_metrics.success_rate
                  << " near_success_rate_p=" << eval_metrics.near_success_rate_p
                  << " near_success_rate_pv=" << eval_metrics.near_success_rate_pv
                  << " mean_total_loss=" << eval_metrics.mean_total_loss
                  << " mean_final_position_norm=" << eval_metrics.mean_final_position_norm
                  << " mean_final_velocity_norm=" << eval_metrics.mean_final_velocity_norm
                  << " mean_final_angular_velocity_norm=" << eval_metrics.mean_final_angular_velocity_norm
                  << " median_final_position_norm=" << eval_metrics.median_final_position_norm
                  << " median_final_velocity_norm=" << eval_metrics.median_final_velocity_norm
                  << " median_final_angular_velocity_norm=" << eval_metrics.median_final_angular_velocity_norm
                  << " p90_final_position_norm=" << eval_metrics.p90_final_position_norm
                  << " p90_final_velocity_norm=" << eval_metrics.p90_final_velocity_norm
                  << " mean_action_norm=" << eval_metrics.mean_action_norm
                  << " invalid_or_nan_rate=" << eval_metrics.invalid_or_nan_rate
                  << " explicit_physical_parameters_in_observation=false\n";
        if(!options.log_path.empty()){
            std::ofstream eval_log(options.log_path);
            fp::write_eval_csv_header(eval_log);
            eval_log << fp::eval_model_name(options.eval_model) << ","
                     << (options.sample_dynamics ? "sampled" : "fixed") << ","
                     << (options.sample_dynamics ? "sampled" : "fixed") << ","
                     << fp::sampled_dynamics_level_name(options.sample_dynamics ? options.sampled_dynamics_level : fp::SampledDynamicsLevel::FIXED) << ","
                     << options.eval_episodes << ","
                     << options.eval_horizon << ","
                     << eval_metrics.success_rate << ","
                     << eval_metrics.near_success_rate_p << ","
                     << eval_metrics.near_success_rate_pv << ","
                     << eval_metrics.mean_total_loss << ","
                     << eval_metrics.mean_final_position_norm << ","
                     << eval_metrics.mean_final_velocity_norm << ","
                     << eval_metrics.mean_final_angular_velocity_norm << ","
                     << eval_metrics.median_final_position_norm << ","
                     << eval_metrics.median_final_velocity_norm << ","
                     << eval_metrics.median_final_angular_velocity_norm << ","
                     << eval_metrics.p90_final_position_norm << ","
                     << eval_metrics.p90_final_velocity_norm << ","
                     << eval_metrics.mean_action_norm << ","
                     << eval_metrics.invalid_or_nan_rate << "\n";
        }
        return eval_metrics.invalid_or_nan_rate == (T)1 ? 1 : 0;
    };

    auto run_failure_analysis = [&](){
        fp::rdac_copy(device, device, actor, rollout_actor);
        rlt::Mode<rlt::mode::Evaluation<>> evaluation_mode;
        RuntimeOptions analysis_options = options;
        if(options.force_dynamics_bins){
            analysis_options.sample_dynamics = true;
            analysis_options.balanced_dynamics_sampling = true;
        }
        eq_dyn::DynamicsBins<TI> forced_bins;
        forced_bins.size_mass = options.force_size_mass_bin;
        forced_bins.thrust_to_weight = options.force_thrust_to_weight_bin;
        forced_bins.torque_to_inertia = options.force_torque_to_inertia_bin;
        forced_bins.motor_delay = options.force_motor_delay_bin;
        forced_bins.curve_shape = options.force_curve_shape_bin;
        const eq_dyn::DynamicsBins<TI>* bins_ptr = options.force_dynamics_bins ? &forced_bins : nullptr;
        const std::string output_path = !options.failure_analysis_path.empty()
            ? options.failure_analysis_path
            : (!options.log_path.empty() ? options.log_path : std::string("failure_analysis.csv"));
        std::ofstream episode_log(output_path);
        if(!episode_log.is_open()){
            std::cerr << "Failed to open failure analysis path: " << output_path << "\n";
            return 1;
        }
        episode_log << "episode,success,primary_failure,final_p_norm,final_v_norm,final_w_norm,final_attitude_error,"
                    << "initial_p_norm,initial_v_norm,mid_p_norm,mid_v_norm,max_p_norm,max_v_norm,max_w_norm,max_attitude_error,"
                    << "action_saturation_ratio,max_action_abs,attitude_turn_failure,speed_not_braked,position_overshoot,action_saturation_failure,horizon_still_improving,"
                    << "size_mass_bin,thrust_to_weight_bin,torque_to_inertia_bin,motor_delay_bin,curve_shape_bin\n";

        auto attitude_error_from_state = [](const STATE& state){
            const T orientation_sign = fp::sign_for_shortest_quaternion(state.orientation);
            T sum = 0;
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                const T e = (T)2 * orientation_sign * state.orientation[dim_i + 1];
                sum += e * e;
            }
            return std::sqrt(sum);
        };
        auto position_error_norm = [&](const STATE& state){
            return fp::l2f_position_error_norm<STATE, T, TI>(state, tracking_reference);
        };
        auto velocity_error_norm = [&](const STATE& state){
            return fp::l2f_velocity_error_norm<STATE, T, TI>(state, tracking_reference);
        };

        TI success_count = 0;
        TI invalid_count = 0;
        TI attitude_failure_count = 0;
        TI speed_failure_count = 0;
        TI overshoot_failure_count = 0;
        TI saturation_failure_count = 0;
        TI horizon_improving_count = 0;
        TI primary_attitude = 0;
        TI primary_speed = 0;
        TI primary_overshoot = 0;
        TI primary_saturation = 0;
        TI primary_horizon = 0;
        TI primary_other = 0;
        T sum_final_p = 0;
        T sum_final_v = 0;
        T sum_final_w = 0;
        T sum_final_attitude = 0;
        T sum_saturation_ratio = 0;
        T sum_max_action_abs = 0;

        for(TI episode_i = 0; episode_i < options.eval_episodes; episode_i++){
            PARAMETERS parameters;
            if(analysis_options.sample_dynamics){
                sample_training_parameters<DEVICE, ENVIRONMENT, PARAMETERS, RNG, T, TI>(device, env, parameters, rng, analysis_options, (T)1, bins_ptr);
            }
            else{
                rlt::initial_parameters(device, env, parameters);
            }
            STATE state;
            rlt::sample_initial_state(device, env, parameters, state, rng);
            l2f_diff::EulerState<T, TI> euler_state;
            l2f_diff::from_l2f_state<STATE, T, TI>(state, euler_state);
            set_euler_action_hover_center(euler_state);
            rlt::reset(device, rollout_actor.trunk, rollout_actor_state, rng);
            T previous_action[ENVIRONMENT::ACTION_DIM] = {};
            T previous_response_error[fp::RESPONSE_ERROR_DIM] = {};

            const T initial_p_norm = position_error_norm(state);
            const T initial_v_norm = velocity_error_norm(state);
            T mid_p_norm = initial_p_norm;
            T mid_v_norm = initial_v_norm;
            T max_p_norm = initial_p_norm;
            T max_v_norm = initial_v_norm;
            T max_w_norm = fp::norm3(state.angular_velocity);
            T max_attitude_error = attitude_error_from_state(state);
            T max_action_abs = 0;
            TI saturation_count = 0;
            TI action_count = 0;
            T initial_p[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                initial_p[dim_i] = state.position[dim_i] - tracking_reference.p[dim_i];
            }

            for(TI step_i = 0; step_i < options.eval_horizon; step_i++){
                if(options.reset_hidden_each_step){
                    rlt::reset(device, rollout_actor.trunk, rollout_actor_state, rng);
                }
                rlt::observe(device, env, parameters, state, typename ENVIRONMENT::Observation{}, observation_row, rng);
                l2f_diff::apply_reference_error_to_observation<T, TI>(observation_row, tracking_reference);
                rlt::set_all(device, step_observations, (T)0);
                for(TI observation_i = 0; observation_i < ENVIRONMENT::Observation::DIM; observation_i++){
                    rlt::set(device, step_observations, rlt::get(observation_row, 0, observation_i), 0, observation_i);
                }
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    rlt::set(device, step_observations, previous_action[action_i], 0, ENVIRONMENT::Observation::DIM + action_i);
                }
                for(TI error_i = 0; error_i < fp::RESPONSE_ERROR_DIM; error_i++){
                    rlt::set(device, step_observations, previous_response_error[error_i], 0, ENVIRONMENT::Observation::DIM + ENVIRONMENT::ACTION_DIM + error_i);
                }
                fp::rdac_evaluate_step(device, rollout_actor, step_observations, rollout_actor_state, step_actions, rollout_actor_buffer, rng, evaluation_mode);

                T action[4];
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    T derivative;
                    bool clamped;
                    action[action_i] = bounded_action<T>(
                        rlt::get(device, step_actions, 0, action_i),
                        options.action_bound_enabled,
                        options.action_bound_value,
                        derivative,
                        clamped
                    );
                    max_action_abs = std::max(max_action_abs, std::abs(action[action_i]));
                    if(std::abs(action[action_i]) > (T)0.98){
                        saturation_count++;
                    }
                    action_count++;
                    previous_action[action_i] = action[action_i];
                    rlt::set(action_row, 0, action_i, action[action_i]);
                }

                STATE next_state;
                rlt::step(device, env, parameters, state, action_row, next_state, rng);
                l2f_diff::EulerState<T, TI> next_euler_state;
                l2f_diff::EulerStepCache<T> euler_cache;
                l2f_diff::step<PARAMETERS, T, TI>(parameters, euler_state, action, next_euler_state, euler_cache);
                rlt::observe(device, env, parameters, next_state, typename ENVIRONMENT::Observation{}, observation_row, rng);
                l2f_diff::apply_reference_error_to_observation<T, TI>(observation_row, tracking_reference);
                l2f_diff::observe_with_reference<T, TI>(next_euler_state, tracking_reference, predicted_observation_row);
                for(TI error_i = 0; error_i < fp::RESPONSE_ERROR_DIM; error_i++){
                    previous_response_error[error_i] = rlt::get(observation_row, 0, error_i) - rlt::get(predicted_observation_row, 0, error_i);
                }
                state = next_state;
                euler_state = next_euler_state;

                const T p_norm = position_error_norm(state);
                const T v_norm = velocity_error_norm(state);
                const T w_norm = fp::norm3(state.angular_velocity);
                const T attitude_error = attitude_error_from_state(state);
                max_p_norm = std::max(max_p_norm, p_norm);
                max_v_norm = std::max(max_v_norm, v_norm);
                max_w_norm = std::max(max_w_norm, w_norm);
                max_attitude_error = std::max(max_attitude_error, attitude_error);
                if(step_i + 1 == options.eval_horizon / 2){
                    mid_p_norm = p_norm;
                    mid_v_norm = v_norm;
                }
            }

            const T final_p_norm = position_error_norm(state);
            const T final_v_norm = velocity_error_norm(state);
            const T final_w_norm = fp::norm3(state.angular_velocity);
            const T final_attitude_error = attitude_error_from_state(state);
            const T saturation_ratio = action_count > 0 ? (T)saturation_count / (T)action_count : (T)0;
            T final_p[3];
            T initial_final_dot = 0;
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                final_p[dim_i] = state.position[dim_i] - tracking_reference.p[dim_i];
                initial_final_dot += initial_p[dim_i] * final_p[dim_i];
            }
            const bool finite = fp::finite_value(final_p_norm) && fp::finite_value(final_v_norm) && fp::finite_value(final_w_norm) && fp::finite_value(final_attitude_error);
            const bool success = finite &&
                final_p_norm < fp::SUCCESS_POSITION_THRESHOLD &&
                final_v_norm < fp::SUCCESS_VELOCITY_THRESHOLD &&
                final_w_norm < fp::SUCCESS_ANGULAR_VELOCITY_THRESHOLD;
            const bool attitude_turn_failure = !success && (final_attitude_error > (T)0.75 || final_w_norm >= fp::SUCCESS_ANGULAR_VELOCITY_THRESHOLD);
            const bool speed_not_braked = !success && final_v_norm >= fp::SUCCESS_VELOCITY_THRESHOLD;
            const bool position_overshoot = !success && ((initial_final_dot < (T)0 && final_p_norm >= fp::SUCCESS_POSITION_THRESHOLD) || max_p_norm > std::max((T)1, initial_p_norm) * (T)1.25);
            const bool action_saturation_failure = !success && saturation_ratio > (T)0.05;
            const bool horizon_still_improving = !success && final_p_norm < mid_p_norm && final_v_norm < mid_v_norm;
            const char* primary_failure = "success";
            if(!success){
                if(action_saturation_failure){ primary_failure = "action_saturation"; primary_saturation++; }
                else if(attitude_turn_failure){ primary_failure = "attitude_turn"; primary_attitude++; }
                else if(speed_not_braked){ primary_failure = "speed_not_braked"; primary_speed++; }
                else if(position_overshoot){ primary_failure = "position_overshoot"; primary_overshoot++; }
                else if(horizon_still_improving){ primary_failure = "horizon_still_improving"; primary_horizon++; }
                else{ primary_failure = "other"; primary_other++; }
            }
            success_count += success ? 1 : 0;
            invalid_count += finite ? 0 : 1;
            attitude_failure_count += attitude_turn_failure ? 1 : 0;
            speed_failure_count += speed_not_braked ? 1 : 0;
            overshoot_failure_count += position_overshoot ? 1 : 0;
            saturation_failure_count += action_saturation_failure ? 1 : 0;
            horizon_improving_count += horizon_still_improving ? 1 : 0;
            sum_final_p += final_p_norm;
            sum_final_v += final_v_norm;
            sum_final_w += final_w_norm;
            sum_final_attitude += final_attitude_error;
            sum_saturation_ratio += saturation_ratio;
            sum_max_action_abs += max_action_abs;
            episode_log << episode_i << "," << (success ? "true" : "false") << "," << primary_failure << ","
                        << final_p_norm << "," << final_v_norm << "," << final_w_norm << "," << final_attitude_error << ","
                        << initial_p_norm << "," << initial_v_norm << "," << mid_p_norm << "," << mid_v_norm << ","
                        << max_p_norm << "," << max_v_norm << "," << max_w_norm << "," << max_attitude_error << ","
                        << saturation_ratio << "," << max_action_abs << ","
                        << (attitude_turn_failure ? "true" : "false") << ","
                        << (speed_not_braked ? "true" : "false") << ","
                        << (position_overshoot ? "true" : "false") << ","
                        << (action_saturation_failure ? "true" : "false") << ","
                        << (horizon_still_improving ? "true" : "false") << ","
                        << (options.force_dynamics_bins ? options.force_size_mass_bin : (TI)-1) << ","
                        << (options.force_dynamics_bins ? options.force_thrust_to_weight_bin : (TI)-1) << ","
                        << (options.force_dynamics_bins ? options.force_torque_to_inertia_bin : (TI)-1) << ","
                        << (options.force_dynamics_bins ? options.force_motor_delay_bin : (TI)-1) << ","
                        << (options.force_dynamics_bins ? options.force_curve_shape_bin : (TI)-1) << "\n";
        }
        const T normalizer = options.eval_episodes > 0 ? (T)1 / (T)options.eval_episodes : (T)1;
        std::cout << "failure_analysis"
                  << " episodes=" << options.eval_episodes
                  << " eval_horizon=" << options.eval_horizon
                  << " output_path=" << output_path
                  << " force_dynamics_bins=" << (options.force_dynamics_bins ? "true" : "false")
                  << " size_mass_bin=" << (options.force_dynamics_bins ? options.force_size_mass_bin : (TI)-1)
                  << " thrust_to_weight_bin=" << (options.force_dynamics_bins ? options.force_thrust_to_weight_bin : (TI)-1)
                  << " torque_to_inertia_bin=" << (options.force_dynamics_bins ? options.force_torque_to_inertia_bin : (TI)-1)
                  << " motor_delay_bin=" << (options.force_dynamics_bins ? options.force_motor_delay_bin : (TI)-1)
                  << " curve_shape_bin=" << (options.force_dynamics_bins ? options.force_curve_shape_bin : (TI)-1)
                  << " success_rate=" << (T)success_count * normalizer
                  << " invalid_rate=" << (T)invalid_count * normalizer
                  << " mean_final_p=" << sum_final_p * normalizer
                  << " mean_final_v=" << sum_final_v * normalizer
                  << " mean_final_w=" << sum_final_w * normalizer
                  << " mean_final_attitude=" << sum_final_attitude * normalizer
                  << " mean_action_saturation_ratio=" << sum_saturation_ratio * normalizer
                  << " mean_max_action_abs=" << sum_max_action_abs * normalizer
                  << " attitude_turn_failure_rate=" << (T)attitude_failure_count * normalizer
                  << " speed_not_braked_rate=" << (T)speed_failure_count * normalizer
                  << " position_overshoot_rate=" << (T)overshoot_failure_count * normalizer
                  << " action_saturation_failure_rate=" << (T)saturation_failure_count * normalizer
                  << " horizon_still_improving_rate=" << (T)horizon_improving_count * normalizer
                  << " primary_action_saturation=" << primary_saturation
                  << " primary_attitude=" << primary_attitude
                  << " primary_speed=" << primary_speed
                  << " primary_overshoot=" << primary_overshoot
                  << " primary_horizon=" << primary_horizon
                  << " primary_other=" << primary_other
                  << "\n";
        return invalid_count == options.eval_episodes ? 1 : 0;
    };

    if(options.stage9_6_objective_parity || options.stage9_6_sampler_parity || options.stage9_6_eval_parity || options.stage9_6_checkpoint_parity){
        std::cerr << "Stage 9.6 parity orchestration is blocked: production sampler replay, "
                  << "unified CPU/CUDA checkpoint cross-load, objective replay parity, and eval parity "
                  << "are not fully implemented yet.\n";
        return 1;
    }

    if(options.failure_analysis){
        return run_failure_analysis();
    }

    if(options.eval_only){
        return run_evaluation();
    }

    std::ofstream log_file;
    std::ofstream batch_bin_counts_file;
    std::ofstream objective_trace_file;
    if(!options.log_path.empty()){
        log_file.open(options.log_path);
        fp::write_training_csv_header(log_file);
        batch_bin_counts_file.open(sibling_output_path(options.log_path, "batch_bin_counts.csv"));
        if(batch_bin_counts_file.is_open()){
            batch_bin_counts_file << "step,batch_size,dynamics_num_groups,dynamics_group_count_min,dynamics_group_count_max,dynamics_batch_balanced";
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << ",size_mass_bin_" << bin_i;
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << ",thrust_to_weight_bin_" << bin_i;
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << ",torque_to_inertia_bin_" << bin_i;
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << ",motor_delay_bin_" << bin_i;
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << ",curve_shape_bin_" << bin_i;
            batch_bin_counts_file << "\n";
        }
    }
    if(options.production_objective_trace){
        const std::string trace_path = !options.objective_trace_path.empty()
            ? options.objective_trace_path
            : (!options.log_path.empty() ? sibling_output_path(options.log_path, "objective_trace_cpu.csv") : std::string("objective_trace_cpu.csv"));
        objective_trace_file.open(trace_path);
        if(!objective_trace_file.is_open()){
            std::cerr << "Failed to open production objective trace: " << trace_path << "\n";
            return 1;
        }
        write_production_objective_trace_header(objective_trace_file);
        std::cout << "production_objective_trace_path=" << trace_path << "\n";
    }

    std::cout << "foundation_policy_diff_pre_training\n";
    std::cout << "diff_model=" << fp::diff_model_name(options.diff_model)
              << " sampled_dynamics=" << (options.sample_dynamics ? "true" : "false")
              << " sampled_dynamics_level=" << fp::sampled_dynamics_level_name(options.sample_dynamics ? options.sampled_dynamics_level : fp::SampledDynamicsLevel::FIXED)
              << " balanced_dynamics_sampling=" << (options.balanced_dynamics_sampling ? "true" : "false")
              << " dynamics_balance_bins=" << fp::DYNAMICS_BALANCE_BINS
              << " batch_size=" << options.batch_size
              << " effective_batch_size=" << options.batch_size
              << " actor_output_dim=" << fp::ACTOR_OUTPUT_DIM
              << " horizon=" << options.horizon
              << " steps=" << options.num_steps
              << " horizon_curriculum=" << (options.horizon_curriculum ? "true" : "false")
              << " state_curriculum=" << (options.state_curriculum ? "true" : "false")
              << " dynamics_curriculum=" << (options.dynamics_curriculum ? "true" : "false")
              << " action_grad_clip_enabled=" << (options.action_grad_clip_enabled ? "true" : "false")
              << " action_grad_clip_norm=" << options.action_grad_clip_norm
              << " actor_grad_clip_enabled=" << (options.actor_grad_clip_enabled ? "true" : "false")
              << " actor_grad_clip_norm=" << options.actor_grad_clip_norm
              << " actor_grad_skip_norm=" << options.actor_grad_skip_norm
              << " action_bound_enabled=" << (options.action_bound_enabled ? "true" : "false")
              << " action_bound_value=" << options.action_bound_value
              << " terminal_ramp_after_horizon_change=" << (options.terminal_ramp_after_horizon_change ? "true" : "false")
              << " terminal_ramp_min=" << options.terminal_ramp_min
              << " terminal_ramp_steps=" << options.terminal_ramp_steps
              << " terminal_ramp_terminal_only=" << (options.terminal_ramp_terminal_only ? "true" : "false")
              << " loss_clf_weight=" << options.loss_clf_weight
              << " loss_window_clf_weight=" << options.loss_window_clf_weight
              << " loss_clf_alpha=" << options.loss_clf_alpha
              << " loss_outward_velocity_weight=" << options.loss_outward_velocity_weight
              << " loss_attitude_control_weight=" << options.loss_attitude_control_weight
              << " action_magnitude_center=" << action_magnitude_center
              << " hover_relative_action_magnitude=" << (options.hover_relative_action_magnitude ? "true" : "false")
              << " reset_optimizer_on_curriculum_transition=" << (options.reset_optimizer_on_curriculum_transition ? "true" : "false")
              << " physics_gradient_enabled=" << (!options.disable_physics_gradient ? "true" : "false")
              << " reset_hidden_each_step=" << (options.reset_hidden_each_step ? "true" : "false")
              << " init_actor_path=" << (options.init_actor_path.empty() ? "none" : options.init_actor_path)
              << " init_actor_loaded=" << (init_actor_loaded_flag ? "true" : "false")
              << "\n";
    std::cout << "actor=RDAC_GRU_ADAPTIVE_ACTOR observation_dim=" << ENVIRONMENT::Observation::DIM
              << " policy_input_dim=" << fp::POLICY_INPUT_DIM
              << " response_error_dim=" << fp::RESPONSE_ERROR_DIM
              << " action_dim=" << ENVIRONMENT::ACTION_DIM
              << " actor_output_dim=" << fp::ACTOR_OUTPUT_DIM
              << " adaptive_memory=gru_hidden_only"
              << " explicit_physical_parameters_in_observation=false\n";

    STATE states[MAX_BATCH_SIZE][MAX_HORIZON + 1];
    l2f_diff::EulerState<T, TI> euler_states[MAX_BATCH_SIZE][MAX_HORIZON + 1];
    l2f_diff::EulerState<T, TI> euler_gradient_states[MAX_HORIZON + 1];
    l2f_diff::EulerStepCache<T> euler_caches[MAX_HORIZON];
    PARAMETERS parameters[MAX_BATCH_SIZE];
    T actions[MAX_BATCH_SIZE][MAX_HORIZON][ENVIRONMENT::ACTION_DIM];
    T action_derivatives[MAX_BATCH_SIZE][MAX_HORIZON][ENVIRONMENT::ACTION_DIM];
    T output_gradients[MAX_BATCH_SIZE][MAX_HORIZON][ENVIRONMENT::ACTION_DIM];
    T action_gradients[MAX_HORIZON][ENVIRONMENT::ACTION_DIM];
    T response_errors[MAX_BATCH_SIZE][MAX_HORIZON][fp::RESPONSE_ERROR_DIM];
    T sample_group_weights[MAX_BATCH_SIZE];
    TI sample_group_keys[MAX_BATCH_SIZE];
    T sample_action_saturation_ratios[MAX_BATCH_SIZE];
    TI sample_action_value_counts[MAX_BATCH_SIZE];
    TI sample_action_saturation_counts[MAX_BATCH_SIZE];
    T step_costs[MAX_BATCH_SIZE][MAX_HORIZON];
    T q_returns[MAX_BATCH_SIZE][MAX_HORIZON];

    constexpr TI NUM_DYNAMICS_BIN_GROUPS = fp::DYNAMICS_BALANCE_BINS * fp::DYNAMICS_BALANCE_BINS * fp::DYNAMICS_BALANCE_BINS * fp::DYNAMICS_BALANCE_BINS * fp::DYNAMICS_BALANCE_BINS;
    T dynamics_bin_loss_sum[NUM_DYNAMICS_BIN_GROUPS] = {};
    T dynamics_bin_success_sum[NUM_DYNAMICS_BIN_GROUPS] = {};
    T dynamics_bin_action_saturation_sum[NUM_DYNAMICS_BIN_GROUPS] = {};
    TI dynamics_bin_sample_count[NUM_DYNAMICS_BIN_GROUPS] = {};

    bool training_invalid = false;
    TI num_skipped_steps = 0;
    TI num_applied_steps = 0;

    // Success-gated curriculum state
    std::vector<CurriculumStage> sg_stages;
    TI sg_current_stage = 0;
    TI sg_stage_age = 0;
    TI sg_num_gate_checks = 0;
    TI sg_num_gate_passes = 0;
    TI sg_num_forced_advances = 0;
    TI sg_num_stage_advances = 0;
    std::vector<bool> sg_success_window;
    bool sg_current_success_flag = false;
    T sg_success_window_mean = 0;
    bool sg_gate_checked = false;
    bool sg_gate_passed = false;
    bool sg_stage_advanced = false;
    bool sg_forced_advance = false;
    const char* sg_transition_reason = "none";
    if(options.success_gated_curriculum){
        get_success_gated_curriculum_stages<TI, T>(MAX_HORIZON, sg_stages);
    }

    // H128-prioritized curriculum schedule
    std::vector<H128ScheduleStep> h128_schedule_steps;
    std::vector<TI> h128_phase_indices;
    std::vector<TI> h128_phase_starts;
    const char* h128_phase_name = "none";
    if(options.h128_prioritized_curriculum){
        get_h128_prioritized_schedule<TI, T>(options.h128_schedule, h128_schedule_steps);
        h128_phase_indices.resize(h128_schedule_steps.size(), 0);
        h128_phase_starts.resize(h128_schedule_steps.size(), 0);
        TI phase_index_builder = 0;
        TI phase_start_builder = 0;
        for(TI i = 0; i < (TI)h128_schedule_steps.size(); i++){
            if(i > 0 && std::strcmp(h128_schedule_steps[i].phase, h128_schedule_steps[i - 1].phase) != 0){
                phase_index_builder++;
                phase_start_builder = i;
            }
            h128_phase_indices[i] = phase_index_builder;
            h128_phase_starts[i] = phase_start_builder;
        }
    }

    TI previous_horizon = scheduled_horizon<TI>(options, 0);
    TI previous_state_bucket = curriculum_stage_bucket<TI>(options.state_curriculum, 0, options.state_curriculum_stage_steps, 3);
    TI steps_since_horizon_change = options.terminal_ramp_steps;
    TI steps_since_state_change = options.state_curriculum_stage_steps;
    for(TI training_step = 0; training_step < options.num_steps; training_step++){
        TI current_horizon = scheduled_horizon<TI>(options, training_step);
        T current_state_difficulty = (T)1;
        TI h128_phase_index = (TI)-1;
        TI h128_steps_in_phase = training_step;
        // Success-gated curriculum: override horizon and state difficulty from stage plan
        sg_stage_advanced = false;
        sg_gate_checked = false;
        sg_gate_passed = false;
        if(options.success_gated_curriculum && !sg_stages.empty()){
            sg_stage_age++;
            TI si = std::min(sg_current_stage, (TI)(sg_stages.size() - 1));
            current_horizon = sg_stages[si].horizon;
            current_state_difficulty = sg_stages[si].state_difficulty;
        }
        // H128-prioritized curriculum: override from fixed schedule
        if(options.h128_prioritized_curriculum && !h128_schedule_steps.empty()){
            TI si = std::min(training_step, (TI)(h128_schedule_steps.size() - 1));
            auto& sched = h128_schedule_steps[si];
            current_horizon = sched.horizon;
            current_state_difficulty = sched.difficulty;
            h128_phase_name = sched.phase;
            h128_phase_index = h128_phase_indices[si];
            h128_steps_in_phase = training_step - h128_phase_starts[si];
        }
        const TI current_state_bucket = curriculum_stage_bucket<TI>(options.state_curriculum, training_step, options.state_curriculum_stage_steps, 3);
        const bool horizon_changed = training_step > 0 && current_horizon != previous_horizon;
        const bool state_curriculum_changed = training_step > 0 && current_state_bucket != previous_state_bucket;
        if(horizon_changed){
            steps_since_horizon_change = 0;
        }
        else if(training_step > 0){
            steps_since_horizon_change++;
        }
        if(state_curriculum_changed){
            steps_since_state_change = 0;
        }
        else if(training_step > 0){
            steps_since_state_change++;
        }
        bool optimizer_reset_flag = false;
        if(options.reset_optimizer_on_curriculum_transition && (horizon_changed || state_curriculum_changed)){
            fp::rdac_reset_optimizer_state(device, actor_optimizer, actor);
            optimizer_reset_flag = true;
        }
        const T terminal_ramp_multiplier_value = terminal_ramp_multiplier<T, TI>(options, steps_since_horizon_change);
        l2f_diff::EulerLossWeights<T> step_euler_weights = euler_weights;
        step_euler_weights.terminal_velocity = options.terminal_velocity_weight * terminal_ramp_multiplier_value;
        step_euler_weights.terminal_angular_velocity = options.terminal_angular_velocity_weight * terminal_ramp_multiplier_value;
        if(options.terminal_ramp_terminal_only){
            step_euler_weights.terminal_position = fp::TERMINAL_POSITION_WEIGHT * terminal_ramp_multiplier_value;
            step_euler_weights.terminal_attitude = fp::TERMINAL_ATTITUDE_WEIGHT * terminal_ramp_multiplier_value;
        }
        // Decoupled curriculum: state difficulty uses a lagged effective step
        TI effective_state_step = training_step;
        if(options.decoupled_curriculum && options.state_curriculum_lag_steps > 0){
            effective_state_step = std::max((TI)0, training_step - options.state_curriculum_lag_steps);
        }
        // One-axis-at-a-time: if both horizon and state would change, delay state
        TI delayed_state_bucket = current_state_bucket;
        if(options.one_curriculum_axis_at_a_time && horizon_changed && state_curriculum_changed){
            delayed_state_bucket = previous_state_bucket;
        }
        // State difficulty: fixed override > success-gated > step curriculum > default 1.0
        T state_difficulty = (T)1;
        if(options.fixed_state_difficulty >= (T)0){
            state_difficulty = options.fixed_state_difficulty;
        }
        else if(options.success_gated_curriculum){
            state_difficulty = current_state_difficulty;
        }
        else if(options.state_curriculum){
            state_difficulty = ramp_difficulty<T>(effective_state_step, options.state_curriculum_stage_steps, 3);
        }
        const T dynamics_difficulty = options.dynamics_curriculum
            ? ramp_difficulty<T>(training_step, options.dynamics_curriculum_stage_steps, 3)
            : (T)1;
        TI rejected_dynamics_count = 0;
        TI valid_rollout_count = 0;
        TI invalid_rollout_count = 0;
        T action_gradient_norm_sq_before_scale = 0;
        T action_gradient_norm_sq_before_clip = 0;
        T action_gradient_norm_sq_after_clip = 0;
        bool action_gradient_clipped = false;
        fp::rdac_copy(device, device, actor, rollout_actor);
        rlt::reset(device, rollout_actor.trunk, rollout_actor_state, rng);
        rlt::set_all(device, observations, (T)0);
        rlt::set_all(device, step_observations, (T)0);
        rlt::set_all(device, d_output, (T)0);
        rlt::set_all(device, q_targets, (T)0);
        rlt::set_all(device, q_weights, (T)0);
        rlt::set_all(device, reset, false);
        for(TI batch_i = 0; batch_i < MAX_BATCH_SIZE; batch_i++){
            sample_action_saturation_ratios[batch_i] = (T)0;
            sample_action_value_counts[batch_i] = 0;
            sample_action_saturation_counts[batch_i] = 0;
            for(TI horizon_i = 0; horizon_i < MAX_HORIZON; horizon_i++){
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    output_gradients[batch_i][horizon_i][action_i] = (T)0;
                    action_derivatives[batch_i][horizon_i][action_i] = (T)1;
                }
                for(TI error_i = 0; error_i < fp::RESPONSE_ERROR_DIM; error_i++){
                    response_errors[batch_i][horizon_i][error_i] = (T)0;
                }
            }
        }

        fp::RolloutMetrics<T> mean_rollout_metrics;
        const T diagnostic_nan = std::numeric_limits<T>::quiet_NaN();
        const T diff_rollout_weight = fp::LOSS_DIFF_ROLLOUT_WEIGHT;
        T action_value_sum = 0;
        T action_value_sq_sum = 0;
        T action_abs_sum = 0;
        T action_delta_sum = 0;
        T action_delta_max = 0;
        T action_value_min = std::numeric_limits<T>::infinity();
        T action_value_max = -std::numeric_limits<T>::infinity();
        TI action_value_count = 0;
        TI action_saturation_count = 0;
        TI action_delta_count = 0;
        T rpm_sum = 0;
        T rpm_min = std::numeric_limits<T>::infinity();
        T rpm_max = -std::numeric_limits<T>::infinity();
        TI rpm_count = 0;
        T thrust_norm_sum = 0;
        T torque_norm_sum = 0;
        TI force_cache_count = 0;
        T transition_consistency_loss = 0;
        bool single_step_state_finite = current_horizon > 0;
        bool single_step_action_finite = current_horizon > 0;
        bool single_step_reward_finite = current_horizon > 0;
        bool single_step_done_finite = current_horizon > 0;
        SampledDynamicsDiagnostics<PARAMETERS, T, TI> sampled_dynamics_stats;
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            const auto dynamics_bins = scheduled_balanced_dynamics_bins<TI>(training_step, batch_i, options.batch_size, fp::DYNAMICS_BALANCE_BINS);
            const TI sample_key_raw = (((dynamics_bins.size_mass * fp::DYNAMICS_BALANCE_BINS + dynamics_bins.thrust_to_weight)
                * fp::DYNAMICS_BALANCE_BINS + dynamics_bins.torque_to_inertia)
                * fp::DYNAMICS_BALANCE_BINS + dynamics_bins.motor_delay)
                * fp::DYNAMICS_BALANCE_BINS + dynamics_bins.curve_shape;
            const TI sample_key = options.sample_dynamics && options.balanced_dynamics_sampling ? sample_key_raw : (TI)0;
            sample_group_keys[batch_i] = sample_key;
            sample_group_weights[batch_i] = (T)0;
            rejected_dynamics_count += sample_training_parameters<DEVICE, ENVIRONMENT, PARAMETERS, RNG, T, TI>(
                device, env, parameters[batch_i], rng, options, dynamics_difficulty,
                options.balanced_dynamics_sampling && options.sample_dynamics ? &dynamics_bins : nullptr
            );
            add_sampled_dynamics_diagnostics<PARAMETERS, T, TI>(
                sampled_dynamics_stats,
                sampled_dynamics_diagnostics<PARAMETERS, T, TI>(
                    parameters[batch_i],
                    env.parameters,
                    options.balanced_dynamics_sampling && options.sample_dynamics
                        ? dynamics_bins
                        : eq_dyn::bins_from_parameters<PARAMETERS, TI>(parameters[batch_i], fp::DYNAMICS_BALANCE_BINS),
                    options.dynamics_curriculum ? dynamics_difficulty : (T)1
                )
            );
            rlt::sample_initial_state(device, env, parameters[batch_i], states[batch_i][0], rng);
            if(options.state_curriculum || options.fixed_state_difficulty >= (T)0){
                apply_state_curriculum<STATE, T>(states[batch_i][0], state_difficulty);
            }
            l2f_diff::from_l2f_state<STATE, T, TI>(states[batch_i][0], euler_states[batch_i][0]);
            set_euler_action_hover_center(euler_states[batch_i][0]);
            mean_rollout_metrics.initial_position_norm += fp::l2f_position_error_norm<STATE, T, TI>(states[batch_i][0], tracking_reference);
            mean_rollout_metrics.initial_velocity_norm += fp::l2f_velocity_error_norm<STATE, T, TI>(states[batch_i][0], tracking_reference);
            mean_rollout_metrics.initial_angular_velocity_norm += fp::norm3(states[batch_i][0].angular_velocity);
        }
        TI num_dynamics_groups = 0;
        TI dynamics_group_count_min = options.batch_size > 0 ? options.batch_size : 0;
        TI dynamics_group_count_max = 0;
        T dynamics_group_weight_sum_min = std::numeric_limits<T>::infinity();
        T dynamics_group_weight_sum_max = -std::numeric_limits<T>::infinity();
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            bool first_in_group = true;
            for(TI previous_i = 0; previous_i < batch_i; previous_i++){
                if(sample_group_keys[previous_i] == sample_group_keys[batch_i]){
                    first_in_group = false;
                    break;
                }
            }
            if(first_in_group){
                num_dynamics_groups++;
            }
        }
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            TI group_count = 0;
            for(TI other_i = 0; other_i < options.batch_size; other_i++){
                if(sample_group_keys[other_i] == sample_group_keys[batch_i]){
                    group_count++;
                }
            }
            sample_group_weights[batch_i] = num_dynamics_groups > 0 && group_count > 0
                ? (T)1 / ((T)num_dynamics_groups * (T)group_count)
                : (T)1 / (T)options.batch_size;
        }
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            bool first_in_group = true;
            for(TI previous_i = 0; previous_i < batch_i; previous_i++){
                if(sample_group_keys[previous_i] == sample_group_keys[batch_i]){
                    first_in_group = false;
                    break;
                }
            }
            if(!first_in_group){
                continue;
            }
            TI group_count = 0;
            T group_weight_sum = 0;
            for(TI other_i = 0; other_i < options.batch_size; other_i++){
                if(sample_group_keys[other_i] == sample_group_keys[batch_i]){
                    group_count++;
                    group_weight_sum += sample_group_weights[other_i];
                }
            }
            dynamics_group_count_min = std::min(dynamics_group_count_min, group_count);
            dynamics_group_count_max = std::max(dynamics_group_count_max, group_count);
            dynamics_group_weight_sum_min = std::min(dynamics_group_weight_sum_min, group_weight_sum);
            dynamics_group_weight_sum_max = std::max(dynamics_group_weight_sum_max, group_weight_sum);
        }
        if(num_dynamics_groups == 0){
            dynamics_group_weight_sum_min = (T)0;
            dynamics_group_weight_sum_max = (T)0;
        }
        const bool dynamics_batch_balanced =
            num_dynamics_groups > 0 &&
            dynamics_group_count_min == dynamics_group_count_max &&
            std::abs(dynamics_group_weight_sum_max - dynamics_group_weight_sum_min) < (T)1e-6;
        TI batch_size_mass_bin_counts[fp::DYNAMICS_BALANCE_BINS] = {};
        TI batch_thrust_to_weight_bin_counts[fp::DYNAMICS_BALANCE_BINS] = {};
        TI batch_torque_to_inertia_bin_counts[fp::DYNAMICS_BALANCE_BINS] = {};
        TI batch_motor_delay_bin_counts[fp::DYNAMICS_BALANCE_BINS] = {};
        TI batch_curve_shape_bin_counts[fp::DYNAMICS_BALANCE_BINS] = {};
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            TI size_mass_bin;
            TI thrust_to_weight_bin;
            TI torque_to_inertia_bin;
            TI motor_delay_bin;
            TI curve_shape_bin;
            decode_dynamics_group_key<TI>(sample_group_keys[batch_i], fp::DYNAMICS_BALANCE_BINS, size_mass_bin, thrust_to_weight_bin, torque_to_inertia_bin, motor_delay_bin, curve_shape_bin);
            batch_size_mass_bin_counts[size_mass_bin]++;
            batch_thrust_to_weight_bin_counts[thrust_to_weight_bin]++;
            batch_torque_to_inertia_bin_counts[torque_to_inertia_bin]++;
            batch_motor_delay_bin_counts[motor_delay_bin]++;
            batch_curve_shape_bin_counts[curve_shape_bin]++;
        }
        if(batch_bin_counts_file.is_open()){
            batch_bin_counts_file << training_step << "," << options.batch_size << "," << num_dynamics_groups << ","
                                  << dynamics_group_count_min << "," << dynamics_group_count_max << ","
                                  << (dynamics_batch_balanced ? "true" : "false");
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << "," << batch_size_mass_bin_counts[bin_i];
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << "," << batch_thrust_to_weight_bin_counts[bin_i];
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << "," << batch_torque_to_inertia_bin_counts[bin_i];
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << "," << batch_motor_delay_bin_counts[bin_i];
            for(TI bin_i = 0; bin_i < fp::DYNAMICS_BALANCE_BINS; bin_i++) batch_bin_counts_file << "," << batch_curve_shape_bin_counts[bin_i];
            batch_bin_counts_file << "\n";
        }
        scale_sampled_dynamics_diagnostics<PARAMETERS, T, TI>(sampled_dynamics_stats, (T)1 / (T)options.batch_size);
        const T dynamics_rejection_rate = rejected_dynamics_count > 0
            ? (T)rejected_dynamics_count / ((T)rejected_dynamics_count + (T)options.batch_size)
            : (T)0;

        rlt::Mode<rlt::mode::Evaluation<>> evaluation_mode;
        for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
            if(options.reset_hidden_each_step){
                rlt::reset(device, rollout_actor.trunk, rollout_actor_state, rng);
            }
            rlt::set_all(device, step_observations, (T)0);
            for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
                rlt::observe(device, env, parameters[batch_i], states[batch_i][horizon_i], typename ENVIRONMENT::Observation{}, observation_row, rng);
                l2f_diff::apply_reference_error_to_observation<T, TI>(observation_row, tracking_reference);
                for(TI observation_i = 0; observation_i < ENVIRONMENT::Observation::DIM; observation_i++){
                    const T value = rlt::get(observation_row, 0, observation_i);
                    rlt::set(device, step_observations, value, batch_i, observation_i);
                    rlt::set(device, observations, value, horizon_i, batch_i, observation_i);
                }
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    const T previous_action = horizon_i > 0 ? actions[batch_i][horizon_i - 1][action_i] : (T)0;
                    rlt::set(device, step_observations, previous_action, batch_i, ENVIRONMENT::Observation::DIM + action_i);
                    rlt::set(device, observations, previous_action, horizon_i, batch_i, ENVIRONMENT::Observation::DIM + action_i);
                }
                for(TI error_i = 0; error_i < fp::RESPONSE_ERROR_DIM; error_i++){
                    const T response_error = horizon_i > 0 ? response_errors[batch_i][horizon_i - 1][error_i] : (T)0;
                    rlt::set(device, step_observations, response_error, batch_i, ENVIRONMENT::Observation::DIM + ENVIRONMENT::ACTION_DIM + error_i);
                    rlt::set(device, observations, response_error, horizon_i, batch_i, ENVIRONMENT::Observation::DIM + ENVIRONMENT::ACTION_DIM + error_i);
                }
                rlt::set(device, reset, options.reset_hidden_each_step || horizon_i == 0, horizon_i, batch_i, 0);
            }

            fp::rdac_evaluate_step(device, rollout_actor, step_observations, rollout_actor_state, step_actions, rollout_actor_buffer, rng, evaluation_mode);

            for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
                T action_norm_sq = 0;
                T raw_action_norm_sq = 0;
                TI clamped_this_step = 0;
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    const T raw_action = rlt::get(device, step_actions, batch_i, action_i);
                    T action_derivative;
                    bool clamped;
                    const T action = bounded_action<T>(raw_action, options.action_bound_enabled, options.action_bound_value, action_derivative, clamped);
                    if(horizon_i == 0){
                        single_step_action_finite = single_step_action_finite &&
                            fp::finite_value(raw_action) &&
                            fp::finite_value(action) &&
                            fp::finite_value(action_derivative);
                    }
                    actions[batch_i][horizon_i][action_i] = action;
                    action_derivatives[batch_i][horizon_i][action_i] = action_derivative;
                    raw_action_norm_sq += raw_action * raw_action;
                    action_norm_sq += action * action;
                    action_value_sum += action;
                    action_value_sq_sum += action * action;
                    action_abs_sum += std::abs(action);
                    action_value_min = std::min(action_value_min, action);
                    action_value_max = std::max(action_value_max, action);
                    action_value_count++;
                    if(horizon_i > 0){
                        const T delta = std::abs(action - actions[batch_i][horizon_i - 1][action_i]);
                        action_delta_sum += delta;
                        action_delta_max = std::max(action_delta_max, delta);
                        action_delta_count++;
                    }
                    sample_action_value_counts[batch_i]++;
                    if(std::abs(action) > (T)0.98){
                        action_saturation_count++;
                        sample_action_saturation_counts[batch_i]++;
                    }
                    clamped_this_step += clamped ? 1 : 0;
                    rlt::set(action_row, 0, action_i, action);
                }
                const T post_action_norm = std::sqrt(action_norm_sq);
                mean_rollout_metrics.preclamp_action_norm += std::sqrt(raw_action_norm_sq) / (T)current_horizon;
                mean_rollout_metrics.postclamp_action_norm += post_action_norm / (T)current_horizon;
                mean_rollout_metrics.action_norm += post_action_norm / (T)current_horizon;
                mean_rollout_metrics.max_action_norm = std::max(mean_rollout_metrics.max_action_norm, post_action_norm);
                mean_rollout_metrics.action_clamp_rate += (T)clamped_this_step / ((T)current_horizon * (T)ENVIRONMENT::ACTION_DIM);
                if(options.diff_model == DiffModel::EULER){
                    l2f_diff::EulerStepCache<T> cache;
                    l2f_diff::step<PARAMETERS, T, TI>(parameters[batch_i], euler_states[batch_i][horizon_i], actions[batch_i][horizon_i], euler_states[batch_i][horizon_i + 1], cache);
                    rlt::step(device, env, parameters[batch_i], states[batch_i][horizon_i], action_row, states[batch_i][horizon_i + 1], rng);
                    rlt::observe(device, env, parameters[batch_i], states[batch_i][horizon_i + 1], typename ENVIRONMENT::Observation{}, observation_row, rng);
                    l2f_diff::apply_reference_error_to_observation<T, TI>(observation_row, tracking_reference);
                    l2f_diff::observe_with_reference<T, TI>(euler_states[batch_i][horizon_i + 1], tracking_reference, predicted_observation_row);
                    const T transition_weight = fp::LOSS_TRANSITION_CONSISTENCY_WEIGHT
                        * sample_group_weights[batch_i]
                        / std::max((T)1, (T)current_horizon);
                    for(TI error_i = 0; error_i < fp::RESPONSE_ERROR_DIM; error_i++){
                        const T response_error = rlt::get(observation_row, 0, error_i) - rlt::get(predicted_observation_row, 0, error_i);
                        response_errors[batch_i][horizon_i][error_i] = response_error;
                        transition_consistency_loss += transition_weight * response_error * response_error;
                    }
                    if(horizon_i == 0){
                        const bool done = compute_training_success_flag<T, TI>(euler_states[batch_i][horizon_i + 1], tracking_reference);
                        single_step_state_finite = single_step_state_finite &&
                            fp::finite_value(l2f_diff::state_norm<T, TI>(euler_states[batch_i][horizon_i + 1])) &&
                            fp::finite_value(fp::norm3(states[batch_i][horizon_i + 1].position)) &&
                            fp::finite_value(fp::norm3(states[batch_i][horizon_i + 1].linear_velocity)) &&
                            fp::finite_value(fp::norm3(states[batch_i][horizon_i + 1].angular_velocity));
                        single_step_done_finite = single_step_done_finite && (done == true || done == false);
                    }
                    for(TI rotor_i = 0; rotor_i < ENVIRONMENT::ACTION_DIM; rotor_i++){
                        rpm_sum += cache.rpm_next[rotor_i];
                        rpm_min = std::min(rpm_min, cache.rpm_next[rotor_i]);
                        rpm_max = std::max(rpm_max, cache.rpm_next[rotor_i]);
                        rpm_count++;
                    }
                    thrust_norm_sum += std::sqrt(
                        cache.thrust_body[0] * cache.thrust_body[0] +
                        cache.thrust_body[1] * cache.thrust_body[1] +
                        cache.thrust_body[2] * cache.thrust_body[2]
                    );
                    torque_norm_sum += std::sqrt(
                        cache.torque_body[0] * cache.torque_body[0] +
                        cache.torque_body[1] * cache.torque_body[1] +
                        cache.torque_body[2] * cache.torque_body[2]
                    );
                    force_cache_count++;
                }
                else{
                    rlt::step(device, env, parameters[batch_i], states[batch_i][horizon_i], action_row, states[batch_i][horizon_i + 1], rng);
                    if(horizon_i == 0){
                        const T p_norm = fp::l2f_position_error_norm<STATE, T, TI>(states[batch_i][horizon_i + 1], tracking_reference);
                        const T v_norm = fp::l2f_velocity_error_norm<STATE, T, TI>(states[batch_i][horizon_i + 1], tracking_reference);
                        const T w_norm = fp::norm3(states[batch_i][horizon_i + 1].angular_velocity);
                        const bool done = p_norm < fp::SUCCESS_POSITION_THRESHOLD &&
                            v_norm < fp::SUCCESS_VELOCITY_THRESHOLD &&
                            w_norm < fp::SUCCESS_ANGULAR_VELOCITY_THRESHOLD;
                        single_step_state_finite = single_step_state_finite &&
                            fp::finite_value(p_norm) &&
                            fp::finite_value(v_norm) &&
                            fp::finite_value(w_norm);
                        single_step_done_finite = single_step_done_finite && (done == true || done == false);
                    }
                }
            }
        }

        fp::ScalarTerms<T> mean_terms;
        T diff_rollout_loss_raw = 0;
        T diff_rollout_loss_scaled = 0;
        TI training_success_count = 0;
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            const T sample_weight = sample_group_weights[batch_i];
            const T diff_sample_weight = sample_weight * diff_rollout_weight;
            T sample_weighted_loss_for_bin = (T)0;
            bool sample_success_for_bin = false;
            sample_action_saturation_ratios[batch_i] = sample_action_value_counts[batch_i] > 0
                ? (T)sample_action_saturation_counts[batch_i] / (T)sample_action_value_counts[batch_i]
                : (T)0;
            if(options.diff_model == DiffModel::EULER){
                auto terms = l2f_diff::rollout_loss_and_gradients<PARAMETERS, T, TI, MAX_HORIZON>(
                    parameters[batch_i],
                    euler_states[batch_i][0],
                    actions[batch_i],
                    current_horizon,
                    step_euler_weights,
                    tracking_reference,
                    euler_gradient_states,
                    euler_caches,
                    action_gradients
                );
                diff_rollout_loss_raw += sample_weight * terms.total();
                diff_rollout_loss_scaled += diff_sample_weight * terms.total();
                mean_terms.total += diff_sample_weight * terms.total();
                mean_terms.position += diff_sample_weight * terms.position;
                mean_terms.velocity += diff_sample_weight * terms.velocity;
                mean_terms.attitude += diff_sample_weight * terms.attitude;
                mean_terms.angular_velocity += diff_sample_weight * terms.angular_velocity;
                mean_terms.action_magnitude += diff_sample_weight * terms.action_magnitude;
                mean_terms.action_smoothness += diff_sample_weight * terms.action_smoothness;
                mean_terms.saturation += diff_sample_weight * terms.saturation;
                mean_terms.terminal += diff_sample_weight * terms.terminal;
                mean_terms.terminal_position += diff_sample_weight * terms.terminal_position;
                mean_terms.terminal_velocity += diff_sample_weight * terms.terminal_velocity;
                mean_terms.terminal_attitude += diff_sample_weight * terms.terminal_attitude;
                mean_terms.terminal_angular_velocity += diff_sample_weight * terms.terminal_angular_velocity;
                mean_terms.clf += diff_sample_weight * terms.clf;
                mean_terms.window_clf += diff_sample_weight * terms.window_clf;
                mean_terms.outward_velocity += diff_sample_weight * terms.outward_velocity;
                mean_terms.attitude_control += diff_sample_weight * terms.attitude_control;
                mean_terms.velocity_barrier += diff_sample_weight * terms.velocity_barrier;
                mean_terms.angular_velocity_barrier += diff_sample_weight * terms.angular_velocity_barrier;
                mean_terms.attitude_barrier += diff_sample_weight * terms.attitude_barrier;
                sample_weighted_loss_for_bin = diff_rollout_weight * terms.total();
                mean_rollout_metrics.final_state_norm += l2f_diff::state_norm<T, TI>(euler_gradient_states[current_horizon]);
                mean_rollout_metrics.final_position_norm += fp::euler_position_error_norm<T, TI>(euler_gradient_states[current_horizon], tracking_reference);
                mean_rollout_metrics.final_velocity_norm += fp::euler_velocity_error_norm<T, TI>(euler_gradient_states[current_horizon], tracking_reference);
                mean_rollout_metrics.final_angular_velocity_norm += fp::norm3(euler_gradient_states[current_horizon].omega);
                sample_success_for_bin = compute_training_success_flag<T, TI>(euler_gradient_states[current_horizon], tracking_reference);
                if(sample_success_for_bin){
                    training_success_count++;
                }
            }
            else{
                auto terms = fp::stabilization_loss_and_action_gradients<DEVICE, PARAMETERS, STATE, T, TI, MAX_HORIZON>(
                    device,
                    parameters[batch_i],
                    states[batch_i],
                    actions[batch_i],
                    action_gradients,
                    l2f_approx_weights,
                    tracking_reference
                );
                diff_rollout_loss_raw += sample_weight * terms.total();
                diff_rollout_loss_scaled += diff_sample_weight * terms.total();
                mean_terms.total += diff_sample_weight * terms.total();
                mean_terms.position += diff_sample_weight * terms.position;
                mean_terms.velocity += diff_sample_weight * terms.velocity;
                mean_terms.attitude += diff_sample_weight * terms.attitude;
                mean_terms.angular_velocity += diff_sample_weight * terms.angular_velocity;
                mean_terms.action_magnitude += diff_sample_weight * terms.action_magnitude;
                mean_terms.action_smoothness += diff_sample_weight * terms.action_smoothness;
                mean_terms.saturation += diff_sample_weight * terms.saturation;
                mean_rollout_metrics.final_position_norm += fp::l2f_position_error_norm<STATE, T, TI>(states[batch_i][current_horizon], tracking_reference);
                mean_rollout_metrics.final_velocity_norm += fp::l2f_velocity_error_norm<STATE, T, TI>(states[batch_i][current_horizon], tracking_reference);
                mean_rollout_metrics.final_angular_velocity_norm += fp::norm3(states[batch_i][current_horizon].angular_velocity);
                const T p_norm = fp::l2f_position_error_norm<STATE, T, TI>(states[batch_i][current_horizon], tracking_reference);
                const T v_norm = fp::l2f_velocity_error_norm<STATE, T, TI>(states[batch_i][current_horizon], tracking_reference);
                const T w_norm = fp::norm3(states[batch_i][current_horizon].angular_velocity);
                sample_weighted_loss_for_bin = diff_rollout_weight * terms.total();
                sample_success_for_bin = p_norm < fp::SUCCESS_POSITION_THRESHOLD &&
                    v_norm < fp::SUCCESS_VELOCITY_THRESHOLD &&
                    w_norm < fp::SUCCESS_ANGULAR_VELOCITY_THRESHOLD;
                if(sample_success_for_bin){
                    training_success_count++;
                }
            }
            if(sample_group_keys[batch_i] >= 0 && sample_group_keys[batch_i] < NUM_DYNAMICS_BIN_GROUPS){
                const TI group_key = sample_group_keys[batch_i];
                dynamics_bin_loss_sum[group_key] += sample_weighted_loss_for_bin;
                dynamics_bin_success_sum[group_key] += sample_success_for_bin ? (T)1 : (T)0;
                dynamics_bin_action_saturation_sum[group_key] += sample_action_saturation_ratios[batch_i];
                dynamics_bin_sample_count[group_key]++;
            }
            if(options.disable_physics_gradient){
                for(TI horizon_i = 0; horizon_i < MAX_HORIZON; horizon_i++){
                    for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                        action_gradients[horizon_i][action_i] = 0;
                    }
                }
            }
            for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    const T raw_gradient = action_gradients[horizon_i][action_i] * action_derivatives[batch_i][horizon_i][action_i];
                    const T gradient = raw_gradient * diff_sample_weight;
                    output_gradients[batch_i][horizon_i][action_i] = gradient;
                    action_gradient_norm_sq_before_scale += raw_gradient * raw_gradient;
                    action_gradient_norm_sq_before_clip += gradient * gradient;
                }
            }
        }
        constexpr T AC_GAMMA = (T)0.99;
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
                T cost = 0;
                if(options.diff_model == DiffModel::EULER){
                    const auto& state = euler_states[batch_i][horizon_i + 1];
                    for(TI dim_i = 0; dim_i < 3; dim_i++){
                        const T e_p = state.p[dim_i] - tracking_reference.p[dim_i];
                        const T e_v = state.v[dim_i] - tracking_reference.v[dim_i];
                        cost += fp::LOSS_POSITION_WEIGHT * e_p * e_p;
                        cost += options.loss_velocity_weight * e_v * e_v;
                        cost += options.loss_angular_velocity_weight * state.omega[dim_i] * state.omega[dim_i];
                    }
                    for(TI row_i = 0; row_i < 3; row_i++){
                        for(TI col_i = 0; col_i < 3; col_i++){
                            const T target = row_i == col_i ? (T)1 : (T)0;
                            const T e = state.R[row_i][col_i] - target;
                            cost += fp::LOSS_ATTITUDE_WEIGHT * e * e;
                        }
                    }
                }
                else{
                    const auto& state = states[batch_i][horizon_i + 1];
                    const T orientation_sign = fp::sign_for_shortest_quaternion(state.orientation);
                    for(TI dim_i = 0; dim_i < 3; dim_i++){
                        const T e_p = state.position[dim_i] - tracking_reference.p[dim_i];
                        const T e_v = state.linear_velocity[dim_i] - tracking_reference.v[dim_i];
                        const T e_R = (T)2 * orientation_sign * state.orientation[dim_i + 1];
                        cost += fp::LOSS_POSITION_WEIGHT * e_p * e_p;
                        cost += options.loss_velocity_weight * e_v * e_v;
                        cost += fp::LOSS_ATTITUDE_WEIGHT * e_R * e_R;
                        cost += options.loss_angular_velocity_weight * state.angular_velocity[dim_i] * state.angular_velocity[dim_i];
                    }
                }
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    const T centered_action = actions[batch_i][horizon_i][action_i] - step_euler_weights.action_magnitude_center;
                    cost += fp::LOSS_ACTION_MAGNITUDE_WEIGHT * centered_action * centered_action;
                }
                step_costs[batch_i][horizon_i] = cost;
                if(horizon_i == 0){
                    const T reward = -cost;
                    single_step_reward_finite = single_step_reward_finite && fp::finite_value(reward);
                }
            }
            T running_return = 0;
            for(TI reverse_i = 0; reverse_i < current_horizon; reverse_i++){
                const TI horizon_i = current_horizon - reverse_i - 1;
                running_return = -step_costs[batch_i][horizon_i] + AC_GAMMA * running_return;
                q_returns[batch_i][horizon_i] = running_return;
            }
        }
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            bool first_in_group = true;
            for(TI previous_i = 0; previous_i < batch_i; previous_i++){
                if(sample_group_keys[previous_i] == sample_group_keys[batch_i]){
                    first_in_group = false;
                    break;
                }
            }
            if(!first_in_group){
                continue;
            }
            T mean_return = 0;
            T variance_return = 0;
            TI count = 0;
            for(TI other_i = 0; other_i < options.batch_size; other_i++){
                if(sample_group_keys[other_i] != sample_group_keys[batch_i]) continue;
                for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
                    mean_return += q_returns[other_i][horizon_i];
                    count++;
                }
            }
            mean_return = count > 0 ? mean_return / (T)count : (T)0;
            for(TI other_i = 0; other_i < options.batch_size; other_i++){
                if(sample_group_keys[other_i] != sample_group_keys[batch_i]) continue;
                for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
                    const T centered = q_returns[other_i][horizon_i] - mean_return;
                    variance_return += centered * centered;
                }
            }
            const T std_return = count > 1 ? std::sqrt(std::max((T)1e-6, variance_return / (T)(count - 1))) : (T)1;
            for(TI other_i = 0; other_i < options.batch_size; other_i++){
                if(sample_group_keys[other_i] != sample_group_keys[batch_i]) continue;
                const T sample_weight = sample_group_weights[other_i] / std::max((T)1, (T)current_horizon);
                for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
                    const T normalized_return = (q_returns[other_i][horizon_i] - mean_return) / std_return;
                    rlt::set(device, q_targets, normalized_return, horizon_i, other_i, (TI)0);
                    rlt::set(device, q_weights, sample_weight, horizon_i, other_i, (TI)0);
                }
            }
        }
        const T batch_normalizer = (T)1 / (T)options.batch_size;
        mean_rollout_metrics.initial_position_norm *= batch_normalizer;
        mean_rollout_metrics.final_position_norm *= batch_normalizer;
        mean_rollout_metrics.initial_velocity_norm *= batch_normalizer;
        mean_rollout_metrics.final_velocity_norm *= batch_normalizer;
        mean_rollout_metrics.initial_angular_velocity_norm *= batch_normalizer;
        mean_rollout_metrics.final_angular_velocity_norm *= batch_normalizer;
        mean_rollout_metrics.action_norm *= batch_normalizer;
        mean_rollout_metrics.preclamp_action_norm *= batch_normalizer;
        mean_rollout_metrics.postclamp_action_norm *= batch_normalizer;
        mean_rollout_metrics.action_clamp_rate *= batch_normalizer;
        mean_rollout_metrics.final_state_norm *= batch_normalizer;
        const T training_success_rate = (T)training_success_count * batch_normalizer;
        const T action_mean = action_value_count > 0 ? action_value_sum / (T)action_value_count : diagnostic_nan;
        const T action_variance = action_value_count > 0
            ? std::max((T)0, action_value_sq_sum / (T)action_value_count - action_mean * action_mean)
            : diagnostic_nan;
        const T action_std = action_value_count > 0 ? std::sqrt(action_variance) : diagnostic_nan;
        const T action_min = action_value_count > 0 ? action_value_min : diagnostic_nan;
        const T action_max = action_value_count > 0 ? action_value_max : diagnostic_nan;
        const T action_abs_mean = action_value_count > 0 ? action_abs_sum / (T)action_value_count : diagnostic_nan;
        const T action_saturation_ratio = action_value_count > 0 ? (T)action_saturation_count / (T)action_value_count : diagnostic_nan;
        const T action_delta_mean = action_delta_count > 0 ? action_delta_sum / (T)action_delta_count : diagnostic_nan;
        const T action_delta_max_value = action_delta_count > 0 ? action_delta_max : diagnostic_nan;
        const T rpm_mean = rpm_count > 0 ? rpm_sum / (T)rpm_count : diagnostic_nan;
        const T rpm_min_value = rpm_count > 0 ? rpm_min : diagnostic_nan;
        const T rpm_max_value = rpm_count > 0 ? rpm_max : diagnostic_nan;
        const T thrust_mean = force_cache_count > 0 ? thrust_norm_sum / (T)force_cache_count : diagnostic_nan;
        const T torque_norm_mean = force_cache_count > 0 ? torque_norm_sum / (T)force_cache_count : diagnostic_nan;
        const T loss_stabilization = mean_terms.position + mean_terms.velocity + mean_terms.attitude + mean_terms.angular_velocity +
            mean_terms.terminal + mean_terms.clf + mean_terms.window_clf + mean_terms.outward_velocity +
            mean_terms.attitude_control + mean_terms.velocity_barrier + mean_terms.angular_velocity_barrier +
            mean_terms.attitude_barrier;
        const T loss_action_regularization = mean_terms.action_magnitude + mean_terms.action_smoothness + mean_terms.saturation;

        // Action-gradient clipping: clips dL/du (rollout-output gradients) before BPTT.
        // This is optional; the recommended stabilizer is actor-gradient scaling (below).
        const T action_grad_norm_before_clip = std::sqrt(action_gradient_norm_sq_before_clip);
        T action_gradient_scale = (T)1;
        if(options.action_grad_clip_enabled && !options.disable_physics_gradient && action_grad_norm_before_clip > options.action_grad_clip_norm){
            action_gradient_scale = options.action_grad_clip_norm / std::max((T)1e-12, action_grad_norm_before_clip);
            action_gradient_clipped = true;
        }
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    const T scaled_gradient = output_gradients[batch_i][horizon_i][action_i] * action_gradient_scale;
                    action_gradient_norm_sq_after_clip += scaled_gradient * scaled_gradient;
                    rlt::set(device, d_output, scaled_gradient, horizon_i, batch_i, action_i);
                }
            }
        }
        const T action_grad_norm_after_clip = std::sqrt(action_gradient_norm_sq_after_clip);

        // Backprop through the RDAC trunk/heads, then compute and scale parameter gradients.
        rlt::Mode<rlt::nn::layers::gru::ResetMode<rlt::mode::Default<>, rlt::nn::layers::gru::ResetModeSpecification<TI, decltype(reset)>>> training_mode;
        training_mode.reset_container = reset;
        fp::rdac_forward(device, actor, observations, actor_buffer, rng, training_mode);
        T hidden_group_means[MAX_BATCH_SIZE][fp::HIDDEN_DIM] = {};
        T hidden_global_mean[fp::HIDDEN_DIM] = {};
        TI hidden_group_counts[MAX_BATCH_SIZE] = {};
        TI hidden_group_keys[MAX_BATCH_SIZE] = {};
        TI hidden_num_groups = 0;
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            TI group_index = hidden_num_groups;
            for(TI group_i = 0; group_i < hidden_num_groups; group_i++){
                if(hidden_group_keys[group_i] == sample_group_keys[batch_i]){
                    group_index = group_i;
                    break;
                }
            }
            if(group_index == hidden_num_groups){
                hidden_group_keys[group_index] = sample_group_keys[batch_i];
                hidden_num_groups++;
            }
            for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
                hidden_group_counts[group_index]++;
                for(TI hidden_i = 0; hidden_i < fp::HIDDEN_DIM; hidden_i++){
                    const T value = rlt::get(device, actor_buffer.hidden, horizon_i, batch_i, hidden_i);
                    hidden_group_means[group_index][hidden_i] += value;
                    hidden_global_mean[hidden_i] += value;
                }
            }
        }
        TI hidden_total_count = 0;
        for(TI group_i = 0; group_i < hidden_num_groups; group_i++){
            hidden_total_count += hidden_group_counts[group_i];
            const T inv_count = hidden_group_counts[group_i] > 0 ? (T)1 / (T)hidden_group_counts[group_i] : (T)0;
            for(TI hidden_i = 0; hidden_i < fp::HIDDEN_DIM; hidden_i++){
                hidden_group_means[group_i][hidden_i] *= inv_count;
            }
        }
        const T hidden_inv_total = hidden_total_count > 0 ? (T)1 / (T)hidden_total_count : (T)0;
        for(TI hidden_i = 0; hidden_i < fp::HIDDEN_DIM; hidden_i++){
            hidden_global_mean[hidden_i] *= hidden_inv_total;
        }
        T hidden_dynamics_between_var = 0;
        T hidden_dynamics_within_var = 0;
        for(TI group_i = 0; group_i < hidden_num_groups; group_i++){
            T group_distance_sq = 0;
            for(TI hidden_i = 0; hidden_i < fp::HIDDEN_DIM; hidden_i++){
                const T centered = hidden_group_means[group_i][hidden_i] - hidden_global_mean[hidden_i];
                group_distance_sq += centered * centered;
            }
            hidden_dynamics_between_var += (T)hidden_group_counts[group_i] * group_distance_sq;
        }
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            TI group_index = 0;
            for(TI group_i = 0; group_i < hidden_num_groups; group_i++){
                if(hidden_group_keys[group_i] == sample_group_keys[batch_i]){
                    group_index = group_i;
                    break;
                }
            }
            for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
                T sample_distance_sq = 0;
                for(TI hidden_i = 0; hidden_i < fp::HIDDEN_DIM; hidden_i++){
                    const T centered = rlt::get(device, actor_buffer.hidden, horizon_i, batch_i, hidden_i) - hidden_group_means[group_index][hidden_i];
                    sample_distance_sq += centered * centered;
                }
                hidden_dynamics_within_var += sample_distance_sq;
            }
        }
        hidden_dynamics_between_var *= hidden_inv_total;
        hidden_dynamics_within_var *= hidden_inv_total;
        const T hidden_dynamics_separation_ratio = hidden_dynamics_between_var / std::max((T)1e-12, hidden_dynamics_within_var);
        const bool hidden_dynamics_separable =
            hidden_num_groups > 1 &&
            fp::finite_value(hidden_dynamics_separation_ratio) &&
            hidden_dynamics_separation_ratio > (T)1e-8;
        mean_terms.total += transition_consistency_loss;
        fp::rdac_zero_gradient(device, actor);
        const auto rdac_ac_terms = fp::rdac_backward(
            device,
            actor,
            observations,
            d_output,
            q_targets,
            q_weights,
            actor_buffer,
            training_mode
        );
        const auto rdac_gradient_diag_before_clip = fp::rdac_gradient_diagnostics(device, actor);
        mean_terms.total += rdac_ac_terms.actor_critic_actor + rdac_ac_terms.actor_critic_critic;

        // Actor parameter-gradient scaling: scales dL/dtheta after BPTT.
        // This is the recommended default stabilization mechanism.
        const auto actor_grad_clip = fp::clip_actor_gradients_by_global_norm<DEVICE, ACTOR, T>(
            device,
            actor,
            options.actor_grad_clip_enabled,
            options.actor_grad_clip_norm,
            options.actor_grad_eps
        );
        const auto rdac_gradient_diag_after_clip = fp::rdac_gradient_diagnostics(device, actor);

        const bool finite_loss = fp::finite_scalar_terms(mean_terms);
        const bool finite_actor_gradient_before_clip = fp::finite_value(actor_grad_clip.raw_norm);
        const bool finite_actor_gradient_after_clip = fp::finite_value(actor_grad_clip.scaled_norm);
        const bool finite_metrics =
            fp::finite_value(mean_rollout_metrics.initial_position_norm) &&
            fp::finite_value(mean_rollout_metrics.final_position_norm) &&
            fp::finite_value(mean_rollout_metrics.initial_velocity_norm) &&
            fp::finite_value(mean_rollout_metrics.final_velocity_norm) &&
            fp::finite_value(mean_rollout_metrics.initial_angular_velocity_norm) &&
            fp::finite_value(mean_rollout_metrics.final_angular_velocity_norm) &&
            fp::finite_value(mean_rollout_metrics.action_norm) &&
            single_step_state_finite &&
            single_step_action_finite &&
            single_step_reward_finite &&
            single_step_done_finite;
        const bool nan_or_inf = !finite_loss || !finite_actor_gradient_before_clip || !finite_actor_gradient_after_clip || !finite_metrics;
        valid_rollout_count = nan_or_inf ? 0 : options.batch_size;
        invalid_rollout_count = nan_or_inf ? options.batch_size : 0;
        bool actor_step_skipped = false;
        bool actor_step_applied = false;
        const char* skip_reason = "none";
        // Optimizer step is skipped only for nonfinite losses/gradients/metrics,
        // or when the raw actor gradient norm exceeds the extreme safety threshold.
        // Finite large gradients are scaled, not skipped.
        if(!finite_loss){
            actor_step_skipped = true;
            skip_reason = "nonfinite_loss";
        }
        else if(!finite_metrics){
            actor_step_skipped = true;
            skip_reason = "nonfinite_metrics";
        }
        else if(!finite_actor_gradient_before_clip){
            actor_step_skipped = true;
            skip_reason = "nonfinite_actor_grad_before_clip";
        }
        else if(actor_grad_clip.raw_norm > options.actor_grad_skip_norm){
            actor_step_skipped = true;
            skip_reason = "actor_grad_above_skip_norm";
        }
        else if(!finite_actor_gradient_after_clip){
            actor_step_skipped = true;
            skip_reason = "nonfinite_actor_grad_after_clip";
        }
        if(!actor_step_skipped){
            fp::rdac_step(device, actor_optimizer, actor);
            actor_step_applied = true;
            num_applied_steps++;
        }
        else{
            num_skipped_steps++;
        }

        // Success-gated curriculum: compute success flag and push to sliding window
        sg_current_success_flag = false;
        if(options.success_gated_curriculum && !nan_or_inf){
            const TI fi = current_horizon;
            sg_current_success_flag = compute_training_success_flag<T, TI>(euler_gradient_states[fi]);
            sg_success_window.push_back(sg_current_success_flag);
            if((TI)sg_success_window.size() > options.curriculum_success_window){
                sg_success_window.erase(sg_success_window.begin());
            }
            TI sc = 0;
            for(auto s : sg_success_window) if(s) sc++;
            sg_success_window_mean = sg_success_window.empty() ? (T)0 : (T)sc / (T)sg_success_window.size();

            // Gate check
            sg_gate_checked = true;
            sg_num_gate_checks++;
            sg_transition_reason = check_curriculum_gate<T, TI>(
                options, sg_stage_age, sg_success_window, training_step,
                (TI)(sg_stages.empty() ? 0 : sg_stages.size() - 1)
            );
            sg_forced_advance = false;
            if(strcmp(sg_transition_reason, "success_gate_pass") == 0){
                sg_gate_passed = true;
                sg_num_gate_passes++;
            }
            if(strcmp(sg_transition_reason, "success_gate_pass") == 0 ||
               strcmp(sg_transition_reason, "max_stage_forced") == 0){
                if(strcmp(sg_transition_reason, "max_stage_forced") == 0){
                    sg_forced_advance = true;
                    sg_num_forced_advances++;
                }
                if(sg_current_stage + 1 < (TI)sg_stages.size()){
                    sg_current_stage++;
                    sg_stage_age = 0;
                    sg_stage_advanced = true;
                    sg_num_stage_advances++;
                }
            }
        }

        if(objective_trace_file.is_open()){
            const T action_grad_norm_before_scale = std::sqrt(action_gradient_norm_sq_before_scale);
            const T critic_loss = rdac_ac_terms.actor_critic_critic;
            const T actor_critic_loss = rdac_ac_terms.actor_critic_actor + rdac_ac_terms.actor_critic_critic;
            const T value_loss = critic_loss;
            const T entropy_loss = (T)0;
            const T teacher_loss = (T)0;
            const T other_loss_terms = (T)0;
            const T actor_update_norm = diagnostic_nan;
            const T adam_m_norm = diagnostic_nan;
            const T adam_v_norm = diagnostic_nan;
            objective_trace_file << training_step << ","
                                 << options.seed << ","
                                 << options.batch_size << ","
                                 << current_horizon << ","
                                 << (options.sample_dynamics ? "sampled" : "fixed") << ","
                                 << (options.sample_dynamics ? "sampled" : "fixed") << ","
                                 << fp::sampled_dynamics_level_name(options.sample_dynamics ? effective_sampled_dynamics_level<T>(options, dynamics_difficulty) : fp::SampledDynamicsLevel::FIXED) << ","
                                 << (options.balanced_dynamics_sampling ? "true" : "false") << ","
                                 << mean_terms.total << ","
                                 << diff_rollout_loss_raw << ","
                                 << diff_rollout_weight << ","
                                 << diff_rollout_loss_scaled << ","
                                 << mean_terms.position << ","
                                 << mean_terms.velocity << ","
                                 << mean_terms.attitude << ","
                                 << mean_terms.angular_velocity << ","
                                 << mean_terms.terminal << ","
                                 << mean_terms.terminal_position << ","
                                 << mean_terms.terminal_velocity << ","
                                 << mean_terms.terminal_attitude << ","
                                 << mean_terms.terminal_angular_velocity << ","
                                 << mean_terms.clf << ","
                                 << mean_terms.window_clf << ","
                                 << mean_terms.outward_velocity << ","
                                 << mean_terms.attitude_control << ","
                                 << mean_terms.velocity_barrier << ","
                                 << mean_terms.angular_velocity_barrier << ","
                                 << mean_terms.attitude_barrier << ","
                                 << mean_terms.action_magnitude << ","
                                 << mean_terms.action_smoothness << ","
                                 << mean_terms.saturation << ","
                                 << critic_loss << ","
                                 << actor_critic_loss << ","
                                 << transition_consistency_loss << ","
                                 << value_loss << ","
                                 << entropy_loss << ","
                                 << teacher_loss << ","
                                 << other_loss_terms << ","
                                 << action_grad_norm_before_scale << ","
                                 << action_grad_norm_before_clip << ","
                                 << action_grad_norm_after_clip << ","
                                 << actor_grad_clip.raw_norm << ","
                                 << actor_grad_clip.scaled_norm << ","
                                 << (actor_step_skipped ? "true" : "false") << ","
                                 << actor_update_norm << ","
                                 << num_applied_steps << ","
                                 << adam_m_norm << ","
                                 << adam_v_norm << ","
                                 << training_success_rate << ","
                                 << mean_rollout_metrics.final_position_norm << ","
                                 << mean_rollout_metrics.final_velocity_norm << ","
                                 << mean_rollout_metrics.final_angular_velocity_norm << ","
                                 << action_saturation_ratio << "\n";
        }

        std::cout << "training_step=" << training_step
                  << " mean_total_loss=" << mean_terms.total
                  << " mean_position_loss=" << mean_terms.position
                  << " mean_velocity_loss=" << mean_terms.velocity
                  << " mean_attitude_loss=" << mean_terms.attitude
                  << " mean_angular_velocity_loss=" << mean_terms.angular_velocity
                  << " mean_action_magnitude_loss=" << mean_terms.action_magnitude
                  << " mean_action_smoothness_loss=" << mean_terms.action_smoothness
                  << " mean_saturation_loss=" << mean_terms.saturation
                  << " mean_terminal_loss=" << mean_terms.terminal
                  << " mean_terminal_position_loss=" << mean_terms.terminal_position
                  << " mean_terminal_velocity_loss=" << mean_terms.terminal_velocity
                  << " mean_terminal_attitude_loss=" << mean_terms.terminal_attitude
                  << " mean_terminal_angular_velocity_loss=" << mean_terms.terminal_angular_velocity
                  << " mean_clf_loss=" << mean_terms.clf
                  << " mean_window_clf_loss=" << mean_terms.window_clf
                  << " mean_outward_velocity_loss=" << mean_terms.outward_velocity
                  << " mean_attitude_control_loss=" << mean_terms.attitude_control
                  << " mean_velocity_barrier_loss=" << mean_terms.velocity_barrier
                  << " mean_angular_velocity_barrier_loss=" << mean_terms.angular_velocity_barrier
                  << " mean_attitude_barrier_loss=" << mean_terms.attitude_barrier
                  << " transition_consistency_loss=" << transition_consistency_loss
                  << " actor_critic_actor_loss=" << rdac_ac_terms.actor_critic_actor
                  << " actor_critic_critic_loss=" << rdac_ac_terms.actor_critic_critic
                  << " single_step_state_finite=" << (single_step_state_finite ? "true" : "false")
                  << " single_step_action_finite=" << (single_step_action_finite ? "true" : "false")
                  << " single_step_reward_finite=" << (single_step_reward_finite ? "true" : "false")
                  << " single_step_done_finite=" << (single_step_done_finite ? "true" : "false")
                  << " diff_rollout_loss_weight=" << diff_rollout_weight
                  << " rdac_encoder_grad_norm_before_clip=" << rdac_gradient_diag_before_clip.encoder
                  << " rdac_gru_grad_norm_before_clip=" << rdac_gradient_diag_before_clip.gru
                  << " rdac_actor_head_grad_norm_before_clip=" << rdac_gradient_diag_before_clip.actor_head
                  << " rdac_critic_head_grad_norm_before_clip=" << rdac_gradient_diag_before_clip.critic_head
                  << " rdac_encoder_grad_norm_after_clip=" << rdac_gradient_diag_after_clip.encoder
                  << " rdac_gru_grad_norm_after_clip=" << rdac_gradient_diag_after_clip.gru
                  << " rdac_actor_head_grad_norm_after_clip=" << rdac_gradient_diag_after_clip.actor_head
                  << " rdac_critic_head_grad_norm_after_clip=" << rdac_gradient_diag_after_clip.critic_head
                  << " hidden_dynamics_separation_ratio=" << hidden_dynamics_separation_ratio
                  << " hidden_dynamics_between_var=" << hidden_dynamics_between_var
                  << " hidden_dynamics_within_var=" << hidden_dynamics_within_var
                  << " hidden_dynamics_separable=" << (hidden_dynamics_separable ? "true" : "false")
                  << " mean_initial_position_norm=" << mean_rollout_metrics.initial_position_norm
                  << " mean_final_position_norm=" << mean_rollout_metrics.final_position_norm
                  << " mean_initial_velocity_norm=" << mean_rollout_metrics.initial_velocity_norm
                  << " mean_final_velocity_norm=" << mean_rollout_metrics.final_velocity_norm
                  << " mean_initial_angular_velocity_norm=" << mean_rollout_metrics.initial_angular_velocity_norm
                  << " mean_final_angular_velocity_norm=" << mean_rollout_metrics.final_angular_velocity_norm
                  << " mean_action_norm=" << mean_rollout_metrics.action_norm
                  << " mean_preclamp_action_norm=" << mean_rollout_metrics.preclamp_action_norm
                  << " mean_postclamp_action_norm=" << mean_rollout_metrics.postclamp_action_norm
                  << " max_action_norm=" << mean_rollout_metrics.max_action_norm
                  << " action_clamp_rate=" << mean_rollout_metrics.action_clamp_rate
                  << " action_mean=" << action_mean
                  << " action_std=" << action_std
                  << " action_min=" << action_min
                  << " action_max=" << action_max
                  << " action_abs_mean=" << action_abs_mean
                  << " action_saturation_ratio=" << action_saturation_ratio
                  << " action_delta_mean=" << action_delta_mean
                  << " action_delta_max=" << action_delta_max_value
                  << " training_success_rate=" << training_success_rate
                  << " rpm_mean=" << rpm_mean
                  << " rpm_min=" << rpm_min_value
                  << " rpm_max=" << rpm_max_value
                  << " thrust_mean=" << thrust_mean
                  << " torque_norm_mean=" << torque_norm_mean
                  << " mean_final_state_norm=" << mean_rollout_metrics.final_state_norm
                  << " sampled_dynamics=" << (options.sample_dynamics ? "true" : "false")
                  << " sampled_dynamics_level=" << fp::sampled_dynamics_level_name(options.sample_dynamics ? effective_sampled_dynamics_level<T>(options, dynamics_difficulty) : fp::SampledDynamicsLevel::FIXED)
                  << " batch_size=" << options.batch_size
                  << " horizon=" << current_horizon
                  << " requested_horizon=" << options.horizon
                  << " state_difficulty=" << state_difficulty
                  << " dynamics_difficulty=" << dynamics_difficulty
                  << " h128_prio=" << (options.h128_prioritized_curriculum ? "true" : "false")
                  << " h128_schedule=" << options.h128_schedule
                  << " h128_phase=" << h128_phase_name
                  << " phase_index=" << h128_phase_index
                  << " steps_in_current_phase=" << h128_steps_in_phase
                  << " horizon_changed=" << (horizon_changed ? "true" : "false")
                  << " state_curriculum_changed=" << (state_curriculum_changed ? "true" : "false")
                  << " steps_since_horizon_change=" << steps_since_horizon_change
                  << " steps_since_state_change=" << steps_since_state_change
                  << " effective_terminal_velocity_weight=" << step_euler_weights.terminal_velocity
                  << " effective_terminal_angular_velocity_weight=" << step_euler_weights.terminal_angular_velocity
                  << " terminal_ramp_multiplier=" << terminal_ramp_multiplier_value
                  << " optimizer_reset=" << (optimizer_reset_flag ? "true" : "false")
                  << " rejected_dynamics_count=" << rejected_dynamics_count
                  << " dynamics_rejection_rate=" << dynamics_rejection_rate
                  << " mass=" << sampled_dynamics_stats.mass
                  << " thrust_to_weight_ratio=" << sampled_dynamics_stats.thrust_to_weight_ratio
                  << " torque_to_inertia_ratio=" << sampled_dynamics_stats.torque_to_inertia_ratio
                  << " motor_tau_mean=" << sampled_dynamics_stats.motor_tau_mean
                  << " motor_tau_min=" << sampled_dynamics_stats.motor_tau_min
                  << " motor_tau_max=" << sampled_dynamics_stats.motor_tau_max
                  << " sampled_tau_rise=" << sampled_dynamics_stats.sampled_tau_rise
                  << " sampled_tau_fall=" << sampled_dynamics_stats.sampled_tau_fall
                  << " sampled_curve_shape=" << sampled_dynamics_stats.sampled_curve_shape
                  << " sampled_parameters_inside_allowed_ranges=" << (sampled_dynamics_stats.sampled_parameters_inside_allowed_ranges > (T)0.999 ? "true" : "false")
                  << " inertia_trace=" << sampled_dynamics_stats.inertia_trace
                  << " thrust_scale=" << sampled_dynamics_stats.thrust_scale
                  << " torque_scale=" << sampled_dynamics_stats.torque_scale
                  << " balanced_dynamics_sampling=" << (options.balanced_dynamics_sampling ? "true" : "false")
                  << " dynamics_num_groups=" << num_dynamics_groups
                  << " dynamics_group_count_min=" << dynamics_group_count_min
                  << " dynamics_group_count_max=" << dynamics_group_count_max
                  << " dynamics_group_weight_sum_min=" << dynamics_group_weight_sum_min
                  << " dynamics_group_weight_sum_max=" << dynamics_group_weight_sum_max
                  << " dynamics_batch_balanced=" << (dynamics_batch_balanced ? "true" : "false")
                  << " dynamics_size_mass_bin=" << sampled_dynamics_stats.dynamics_size_mass_bin
                  << " dynamics_thrust_to_weight_bin=" << sampled_dynamics_stats.dynamics_thrust_to_weight_bin
                  << " dynamics_torque_to_inertia_bin=" << sampled_dynamics_stats.dynamics_torque_to_inertia_bin
                  << " dynamics_motor_delay_bin=" << sampled_dynamics_stats.dynamics_motor_delay_bin
                  << " dynamics_curve_shape_bin=" << sampled_dynamics_stats.dynamics_curve_shape_bin
                  << " equivalent_dynamics_diag_thrust_to_acceleration_gain=" << sampled_dynamics_stats.equivalent_dynamics_diag_thrust_to_acceleration_gain
                  << " equivalent_dynamics_diag_roll_pitch_torque_to_angular_acceleration_gain=" << sampled_dynamics_stats.equivalent_dynamics_diag_roll_pitch_torque_to_angular_acceleration_gain
                  << " equivalent_dynamics_diag_yaw_torque_to_angular_acceleration_gain=" << sampled_dynamics_stats.equivalent_dynamics_diag_yaw_torque_to_angular_acceleration_gain
                  << " equivalent_dynamics_diag_motor_rise_time_constant=" << sampled_dynamics_stats.equivalent_dynamics_diag_motor_rise_time_constant
                  << " equivalent_dynamics_diag_motor_fall_time_constant=" << sampled_dynamics_stats.equivalent_dynamics_diag_motor_fall_time_constant
                  << " equivalent_dynamics_diag_thrust_curve_shape=" << sampled_dynamics_stats.equivalent_dynamics_diag_thrust_curve_shape
                  << " equivalent_dynamics_diag_torque_curve_shape=" << sampled_dynamics_stats.equivalent_dynamics_diag_torque_curve_shape
                  << " equivalent_dynamics_diag_residual_force_bias=" << sampled_dynamics_stats.equivalent_dynamics_diag_residual_force_bias
                  << " equivalent_dynamics_diag_residual_torque_bias=" << sampled_dynamics_stats.equivalent_dynamics_diag_residual_torque_bias
                  << " valid_rollout_count=" << valid_rollout_count
                  << " invalid_rollout_count=" << invalid_rollout_count
                  << " diff_model=" << fp::diff_model_name(options.diff_model)
                  << " physics_gradient_enabled=" << (!options.disable_physics_gradient ? "true" : "false")
                  << " reset_hidden_each_step=" << (options.reset_hidden_each_step ? "true" : "false")
                  << " init_actor_loaded=" << (init_actor_loaded_flag ? "true" : "false")
                  << " action_grad_norm_before_clip=" << action_grad_norm_before_clip
                  << " action_grad_norm_after_clip=" << action_grad_norm_after_clip
                  << " action_grad_scale=" << action_gradient_scale
                  << " action_grad_clipped=" << (action_gradient_clipped ? "true" : "false")
                  << " actor_grad_norm_before_clip=" << actor_grad_clip.raw_norm
                  << " actor_grad_norm_after_clip=" << actor_grad_clip.scaled_norm
                  << " actor_grad_scale=" << actor_grad_clip.scale
                  << " actor_grad_clipped=" << (actor_grad_clip.clipped ? "true" : "false")
                  << " actor_step_skipped=" << (actor_step_skipped ? "true" : "false")
                  << " actor_step_applied=" << (actor_step_applied ? "true" : "false")
                  << " num_skipped_steps=" << num_skipped_steps
                  << " num_applied_steps=" << num_applied_steps
                  << " skip_reason=" << skip_reason
                  << " optimizer_step=" << (actor_step_applied ? "true" : "false")
                  << " finite=" << (!nan_or_inf ? "true" : "false")
                  << " success_gated=" << (options.success_gated_curriculum ? "true" : "false")
                  << " sg_stage_index=" << sg_current_stage
                  << " sg_stage_age=" << sg_stage_age
                  << " sg_success_mean=" << sg_success_window_mean
                  << " sg_gate_checked=" << (sg_gate_checked ? "true" : "false")
                  << " sg_gate_passed=" << (sg_gate_passed ? "true" : "false")
                  << " sg_stage_advanced=" << (sg_stage_advanced ? "true" : "false")
                  << " sg_forced_advance=" << (sg_forced_advance ? "true" : "false")
                  << " sg_reason=" << sg_transition_reason
                  << " sg_max_stage=" << options.curriculum_max_stage_steps
                  << " train_success=" << (training_success_rate > (T)0 ? "true" : "false")
                  << "\n";
        if(log_file.is_open()){
            log_file << options.seed << ","
                     << training_step << ","
                     << (options.sample_dynamics ? "sampled" : "fixed") << ","
                     << fp::diff_model_name(options.diff_model) << ","
                     << (!options.disable_physics_gradient ? "true" : "false") << ","
                     << (options.reset_hidden_each_step ? "true" : "false") << ","
                     << current_horizon << ","
                     << state_difficulty << ","
                     << dynamics_difficulty << ","
                     << fp::sampled_dynamics_level_name(options.sample_dynamics ? effective_sampled_dynamics_level<T>(options, dynamics_difficulty) : fp::SampledDynamicsLevel::FIXED) << ","
                     << (options.sample_dynamics ? "true" : "false") << ","
                     << rejected_dynamics_count << ","
                     << dynamics_rejection_rate << ","
                     << (options.init_actor_path.empty() ? "none" : options.init_actor_path) << ","
                     << (init_actor_loaded_flag ? "true" : "false") << ","
                     << (options.h128_prioritized_curriculum ? "true" : "false") << ","
                     << options.h128_schedule << ","
                     << h128_phase_name << ","
                     << h128_phase_index << ","
                     << h128_steps_in_phase << ","
                     << (horizon_changed ? "true" : "false") << ","
                     << (state_curriculum_changed ? "true" : "false") << ","
                     << steps_since_horizon_change << ","
                     << steps_since_state_change << ","
                     << step_euler_weights.terminal_velocity << ","
                     << step_euler_weights.terminal_angular_velocity << ","
                     << terminal_ramp_multiplier_value << ","
                     << (optimizer_reset_flag ? "true" : "false") << ","
                     << rejected_dynamics_count << ","
                     << valid_rollout_count << ","
                     << invalid_rollout_count << ","
                     << sampled_dynamics_stats.mass << ","
                     << sampled_dynamics_stats.thrust_to_weight_ratio << ","
                     << sampled_dynamics_stats.torque_to_inertia_ratio << ","
                     << sampled_dynamics_stats.motor_tau_mean << ","
                     << sampled_dynamics_stats.motor_tau_min << ","
                     << sampled_dynamics_stats.motor_tau_max << ","
                     << sampled_dynamics_stats.sampled_tau_rise << ","
                     << sampled_dynamics_stats.sampled_tau_fall << ","
                     << sampled_dynamics_stats.sampled_curve_shape << ","
                     << (sampled_dynamics_stats.sampled_parameters_inside_allowed_ranges > (T)0.999 ? "true" : "false") << ","
                     << sampled_dynamics_stats.inertia_trace << ","
                     << sampled_dynamics_stats.thrust_scale << ","
                     << sampled_dynamics_stats.torque_scale << ","
                     << (options.balanced_dynamics_sampling ? "true" : "false") << ","
                     << num_dynamics_groups << ","
                     << dynamics_group_count_min << ","
                     << dynamics_group_count_max << ","
                     << dynamics_group_weight_sum_min << ","
                     << dynamics_group_weight_sum_max << ","
                     << (dynamics_batch_balanced ? "true" : "false") << ","
                     << sampled_dynamics_stats.dynamics_size_mass_bin << ","
                     << sampled_dynamics_stats.dynamics_thrust_to_weight_bin << ","
                     << sampled_dynamics_stats.dynamics_torque_to_inertia_bin << ","
                     << sampled_dynamics_stats.dynamics_motor_delay_bin << ","
                     << sampled_dynamics_stats.dynamics_curve_shape_bin << ","
                     << sampled_dynamics_stats.equivalent_dynamics_diag_thrust_to_acceleration_gain << ","
                     << sampled_dynamics_stats.equivalent_dynamics_diag_roll_pitch_torque_to_angular_acceleration_gain << ","
                     << sampled_dynamics_stats.equivalent_dynamics_diag_yaw_torque_to_angular_acceleration_gain << ","
                     << sampled_dynamics_stats.equivalent_dynamics_diag_motor_rise_time_constant << ","
                     << sampled_dynamics_stats.equivalent_dynamics_diag_motor_fall_time_constant << ","
                     << sampled_dynamics_stats.equivalent_dynamics_diag_thrust_curve_shape << ","
                     << sampled_dynamics_stats.equivalent_dynamics_diag_torque_curve_shape << ","
                     << sampled_dynamics_stats.equivalent_dynamics_diag_residual_force_bias << ","
                     << sampled_dynamics_stats.equivalent_dynamics_diag_residual_torque_bias << ","
                     << mean_terms.total << ","
                     << mean_terms.position << ","
                     << mean_terms.velocity << ","
                     << mean_terms.attitude << ","
                     << mean_terms.angular_velocity << ","
                     << mean_terms.action_magnitude << ","
                     << mean_terms.action_smoothness << ","
                     << mean_terms.saturation << ","
                     << mean_terms.terminal << ","
                     << mean_terms.terminal_position << ","
                     << mean_terms.terminal_velocity << ","
                     << mean_terms.terminal_attitude << ","
                     << mean_terms.terminal_angular_velocity << ","
                     << mean_terms.clf << ","
                     << mean_terms.window_clf << ","
                     << mean_terms.outward_velocity << ","
                     << mean_terms.attitude_control << ","
                     << mean_terms.velocity_barrier << ","
                     << mean_terms.angular_velocity_barrier << ","
                     << mean_terms.attitude_barrier << ","
                     << transition_consistency_loss << ","
                     << rdac_ac_terms.actor_critic_actor << ","
                     << rdac_ac_terms.actor_critic_critic << ","
                     << (single_step_state_finite ? "true" : "false") << ","
                     << (single_step_action_finite ? "true" : "false") << ","
                     << (single_step_reward_finite ? "true" : "false") << ","
                     << (single_step_done_finite ? "true" : "false") << ","
                     << diff_rollout_weight << ","
                     << rdac_gradient_diag_before_clip.encoder << ","
                     << rdac_gradient_diag_before_clip.gru << ","
                     << rdac_gradient_diag_before_clip.actor_head << ","
                     << rdac_gradient_diag_before_clip.critic_head << ","
                     << rdac_gradient_diag_after_clip.encoder << ","
                     << rdac_gradient_diag_after_clip.gru << ","
                     << rdac_gradient_diag_after_clip.actor_head << ","
                     << rdac_gradient_diag_after_clip.critic_head << ","
                     << hidden_dynamics_separation_ratio << ","
                     << hidden_dynamics_between_var << ","
                     << hidden_dynamics_within_var << ","
                     << (hidden_dynamics_separable ? "true" : "false") << ","
                     << action_grad_norm_before_clip << ","
                     << action_grad_norm_after_clip << ","
                     << action_gradient_scale << ","
                     << (action_gradient_clipped ? "true" : "false") << ","
                     << actor_grad_clip.raw_norm << ","
                     << actor_grad_clip.scaled_norm << ","
                     << actor_grad_clip.scale << ","
                     << (actor_grad_clip.clipped ? "true" : "false") << ","
                     << ((!finite_actor_gradient_before_clip || !finite_actor_gradient_after_clip) ? "true" : "false") << ","
                     << (actor_step_skipped ? "true" : "false") << ","
                     << (actor_step_applied ? "true" : "false") << ","
                     << num_skipped_steps << ","
                     << num_applied_steps << ","
                     << skip_reason << ","
                     << mean_rollout_metrics.initial_position_norm << ","
                     << mean_rollout_metrics.final_position_norm << ","
                     << mean_rollout_metrics.initial_velocity_norm << ","
                     << mean_rollout_metrics.final_velocity_norm << ","
                     << mean_rollout_metrics.initial_angular_velocity_norm << ","
                     << mean_rollout_metrics.final_angular_velocity_norm << ","
                     << mean_rollout_metrics.action_norm << ","
                     << mean_rollout_metrics.preclamp_action_norm << ","
                     << mean_rollout_metrics.postclamp_action_norm << ","
                     << mean_rollout_metrics.max_action_norm << ","
                     << mean_rollout_metrics.action_clamp_rate << ","
                     << (nan_or_inf ? "true" : "false") << ","
                     << mean_terms.total << ","
                     << mean_terms.position << ","
                     << mean_terms.velocity << ","
                     << mean_terms.angular_velocity << ","
                     << mean_terms.terminal_position << ","
                     << mean_terms.terminal_velocity << ","
                     << mean_terms.terminal_angular_velocity << ","
                     << mean_terms.clf << ","
                     << mean_terms.window_clf << ","
                     << mean_terms.outward_velocity << ","
                     << mean_terms.attitude_control << ","
                     << mean_terms.action_magnitude << ","
                     << mean_terms.action_smoothness << ","
                     << mean_terms.saturation << ","
                     << loss_stabilization << ","
                     << loss_action_regularization << ","
                     << actor_grad_clip.raw_norm << ","
                     << actor_grad_clip.scaled_norm << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << action_grad_norm_before_clip << ","
                     << action_grad_norm_after_clip << ","
                     << action_gradient_scale << ","
                     << (action_gradient_clipped ? "true" : "false") << ","
                     << action_mean << ","
                     << action_std << ","
                     << action_min << ","
                     << action_max << ","
                     << action_abs_mean << ","
                     << action_saturation_ratio << ","
                     << action_delta_mean << ","
                     << action_delta_max_value << ","
                     << training_success_rate << ","
                     << rpm_mean << ","
                     << rpm_min_value << ","
                     << rpm_max_value << ","
                     << thrust_mean << ","
                     << torque_norm_mean << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << diagnostic_nan << ","
                     << "false" << ","
                     << "actor_grad_max_abs;actor_update_norm;actor_parameter_norm;actor_param_norm;actor_param_max_abs;actor_update_to_param_norm_ratio;adam_m_norm;adam_v_norm;gru_hidden_norm_mean;gru_hidden_norm_max;gru_hidden_abs_mean;gru_hidden_abs_max" << "\n";
        }
        previous_horizon = current_horizon;
        previous_state_bucket = current_state_bucket;
        if(nan_or_inf){
            if(options.sample_dynamics){
                std::cerr << "Sampled-dynamics mode: skipping this invalid rollout and continuing.\n";
                continue;
            }
            else{
                std::cerr << "Stopping because loss or actor gradient norm is not finite.\n";
                training_invalid = true;
                break;
            }
        }
    }

    if(!options.log_path.empty()){
        const std::string per_bin_path = sibling_output_path(options.log_path, "per_dynamics_bin_summary.csv");
        std::ofstream per_bin_file(per_bin_path);
        if(per_bin_file.is_open()){
            per_bin_file << "group_key,size_mass_bin,thrust_to_weight_bin,torque_to_inertia_bin,motor_delay_bin,curve_shape_bin,sample_count,loss_mean,success_mean,action_saturation_mean\n";
            TI min_bin_sample_count = dynamics_bin_sample_count[0];
            TI max_bin_sample_count = dynamics_bin_sample_count[0];
            for(TI group_key = 0; group_key < NUM_DYNAMICS_BIN_GROUPS; group_key++){
                min_bin_sample_count = std::min(min_bin_sample_count, dynamics_bin_sample_count[group_key]);
                max_bin_sample_count = std::max(max_bin_sample_count, dynamics_bin_sample_count[group_key]);
                TI size_mass_bin;
                TI thrust_to_weight_bin;
                TI torque_to_inertia_bin;
                TI motor_delay_bin;
                TI curve_shape_bin;
                decode_dynamics_group_key<TI>(group_key, fp::DYNAMICS_BALANCE_BINS, size_mass_bin, thrust_to_weight_bin, torque_to_inertia_bin, motor_delay_bin, curve_shape_bin);
                const T inv_count = dynamics_bin_sample_count[group_key] > 0 ? (T)1 / (T)dynamics_bin_sample_count[group_key] : (T)0;
                per_bin_file << group_key << ","
                             << size_mass_bin << ","
                             << thrust_to_weight_bin << ","
                             << torque_to_inertia_bin << ","
                             << motor_delay_bin << ","
                             << curve_shape_bin << ","
                             << dynamics_bin_sample_count[group_key] << ","
                             << dynamics_bin_loss_sum[group_key] * inv_count << ","
                             << dynamics_bin_success_sum[group_key] * inv_count << ","
                             << dynamics_bin_action_saturation_sum[group_key] * inv_count << "\n";
            }
            std::cout << "per_dynamics_bin_summary=" << per_bin_path
                      << " dynamics_bin_sample_count_min=" << min_bin_sample_count
                      << " dynamics_bin_sample_count_max=" << max_bin_sample_count
                      << " dynamics_bins_sampled_evenly=" << (min_bin_sample_count == max_bin_sample_count ? "true" : "false")
                      << "\n";
        }
    }

    if(!options.save_path.empty()){
        if(training_invalid){
            std::cerr << "Not saving actor checkpoint because training encountered NaN/Inf.\n";
            return 1;
        }
        const bool saved_ok = options.save_optimizer
            ? fp::save_actor_optimizer_checkpoint(device, actor, actor_optimizer, options.save_path)
            : fp::save_actor_checkpoint(device, actor, options.save_path);
        if(!saved_ok){
            return 1;
        }
        std::cout << "saved_actor_checkpoint=" << options.save_path << "\n";
        if(options.save_optimizer){
            std::cout << "saved_optimizer_state=true optimizer_age="
                      << rlt::get(device, actor_optimizer.age, 0) << "\n";
        }
    }

    rlt::free(device, observation_row);
    rlt::free(device, predicted_observation_row);
    rlt::free(device, action_row);
    rlt::free(device, reset);
    rlt::free(device, q_weights);
    rlt::free(device, q_targets);
    rlt::free(device, d_output);
    rlt::free(device, step_actions);
    rlt::free(device, step_observations);
    rlt::free(device, observations);
    rlt::free(device, actor_optimizer);
    rlt::free(device, rollout_actor_state);
    fp::rdac_free_buffer<DEVICE, typename ROLLOUT_ACTOR::CAPABILITY_TYPE, fp::DYNAMIC_ALLOCATION>(device, rollout_actor_buffer);
    fp::rdac_free(device, rollout_actor);
    fp::rdac_free_buffer<DEVICE, typename ACTOR::CAPABILITY_TYPE, fp::DYNAMIC_ALLOCATION>(device, actor_buffer);
    fp::rdac_free(device, actor);
    rlt::free(device, rng);
    return 0;
}
