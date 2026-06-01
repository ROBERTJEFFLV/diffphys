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

namespace fp = rl_tools::foundation_policy::diff_pre_training;
namespace rlt = rl_tools;
namespace l2f_diff = rl_tools::rl::environments::l2f::diff;

using DiffModel = fp::DiffModel;
using EvalModel = fp::EvalModel;
using RuntimeOptions = fp::RuntimeOptions;

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
    const l2f_diff::EulerState<T, TI>& final_state
){
    const T p_norm = std::sqrt(final_state.p[0]*final_state.p[0] + final_state.p[1]*final_state.p[1] + final_state.p[2]*final_state.p[2]);
    const T v_norm = std::sqrt(final_state.v[0]*final_state.v[0] + final_state.v[1]*final_state.v[1] + final_state.v[2]*final_state.v[2]);
    const T w_norm = std::sqrt(final_state.omega[0]*final_state.omega[0] + final_state.omega[1]*final_state.omega[1] + final_state.omega[2]*final_state.omega[2]);
    return p_norm < fp::SUCCESS_POSITION_THRESHOLD
        && v_norm < fp::SUCCESS_VELOCITY_THRESHOLD
        && w_norm < fp::SUCCESS_ANGULAR_VELOCITY_THRESHOLD;
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
struct SampledDynamicsDiagnostics{
    T mass = 0;
    T thrust_to_weight_ratio = 0;
    T torque_to_inertia_ratio = 0;
    T motor_tau_mean = 0;
    T motor_tau_min = 0;
    T motor_tau_max = 0;
    T inertia_trace = 0;
    T thrust_scale = 0;
    T torque_scale = 0;
};

template <typename PARAMETERS, typename T, typename TI>
SampledDynamicsDiagnostics<PARAMETERS, T, TI> sampled_dynamics_diagnostics(
    const PARAMETERS& parameters,
    const PARAMETERS& nominal_parameters
){
    SampledDynamicsDiagnostics<PARAMETERS, T, TI> stats;
    stats.mass = parameters.dynamics.mass;
    stats.thrust_to_weight_ratio = estimated_thrust_to_weight<PARAMETERS, T, TI>(parameters);
    stats.torque_to_inertia_ratio = estimated_torque_to_inertia_ratio<PARAMETERS, T, TI>(parameters);
    motor_tau_stats<PARAMETERS, T, TI>(parameters, stats.motor_tau_mean, stats.motor_tau_min, stats.motor_tau_max);
    stats.inertia_trace = parameters.dynamics.J[0][0] + parameters.dynamics.J[1][1] + parameters.dynamics.J[2][2];
    stats.thrust_scale = estimated_max_thrust<PARAMETERS, T, TI>(parameters)
        / std::max((T)1e-12, estimated_max_thrust<PARAMETERS, T, TI>(nominal_parameters));
    stats.torque_scale = average_rotor_torque_constant_abs<PARAMETERS, T, TI>(parameters)
        / std::max((T)1e-12, average_rotor_torque_constant_abs<PARAMETERS, T, TI>(nominal_parameters));
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
    accumulator.inertia_trace += value.inertia_trace;
    accumulator.thrust_scale += value.thrust_scale;
    accumulator.torque_scale += value.torque_scale;
}

template <typename PARAMETERS, typename T, typename TI>
void scale_sampled_dynamics_diagnostics(SampledDynamicsDiagnostics<PARAMETERS, T, TI>& value, T scale){
    value.mass *= scale;
    value.thrust_to_weight_ratio *= scale;
    value.torque_to_inertia_ratio *= scale;
    value.motor_tau_mean *= scale;
    value.motor_tau_min *= scale;
    value.motor_tau_max *= scale;
    value.inertia_trace *= scale;
    value.thrust_scale *= scale;
    value.torque_scale *= scale;
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
    const T max_mass = (T)0.06 + dynamics_difficulty * (fp::SAMPLED_DYNAMICS_MAX_MASS - (T)0.06);
    if(!fp::finite_value(parameters.dynamics.mass) || parameters.dynamics.mass <= (T)0 || parameters.dynamics.mass > max_mass){
        return false;
    }
    if(estimated_thrust_to_weight<PARAMETERS, T, TI>(parameters) < fp::SAMPLED_DYNAMICS_MIN_THRUST_TO_WEIGHT){
        return false;
    }
    for(TI i = 0; i < 3; i++){
        if(!fp::finite_value(parameters.dynamics.J[i][i]) || parameters.dynamics.J[i][i] <= (T)0) return false;
        if(!fp::finite_value(parameters.dynamics.J_inv[i][i]) || parameters.dynamics.J_inv[i][i] <= (T)0) return false;
    }
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        if(!fp::finite_value(parameters.dynamics.rotor_time_constants_rising[rotor_i]) ||
           !fp::finite_value(parameters.dynamics.rotor_time_constants_falling[rotor_i]) ||
           parameters.dynamics.rotor_time_constants_rising[rotor_i] < (T)0.005 ||
           parameters.dynamics.rotor_time_constants_falling[rotor_i] < (T)0.005 ||
           parameters.dynamics.rotor_time_constants_rising[rotor_i] > (T)0.5 ||
           parameters.dynamics.rotor_time_constants_falling[rotor_i] > (T)0.5){
            return false;
        }
        if(!fp::finite_value(parameters.dynamics.rotor_thrust_coefficients[rotor_i][2]) ||
           parameters.dynamics.rotor_thrust_coefficients[rotor_i][2] <= (T)0){
            return false;
        }
    }
    return true;
}

template <typename DEVICE, typename ENVIRONMENT, typename PARAMETERS, typename RNG, typename T, typename TI>
TI sample_training_parameters(DEVICE& device, ENVIRONMENT& env, PARAMETERS& parameters, RNG& rng, const RuntimeOptions& options, T dynamics_difficulty){
    const auto level = effective_sampled_dynamics_level<T>(options, dynamics_difficulty);
    if(!options.sample_dynamics || level == fp::SampledDynamicsLevel::FIXED){
        rlt::initial_parameters(device, env, parameters);
        return 0;
    }
    ENVIRONMENT sampling_env = env;
    configure_sampled_dynamics_level(sampling_env, level);
    const bool level_uses_filter = level == fp::SampledDynamicsLevel::NARROW || level == fp::SampledDynamicsLevel::MEDIUM;
    if(!options.dynamics_curriculum || dynamics_difficulty <= (T)0){
        if(options.dynamics_curriculum){
            rlt::initial_parameters(device, env, parameters);
        }
        else if(level_uses_filter){
            TI rejected = 0;
            for(TI attempt_i = 0; attempt_i < 64; attempt_i++){
                rlt::sample_initial_parameters(device, sampling_env, parameters, rng);
                if(sampled_parameters_safe<PARAMETERS, T, TI>(parameters, (T)1)){
                    return rejected;
                }
                rejected++;
            }
            rlt::initial_parameters(device, env, parameters);
            return rejected;
        }
        else{
            rlt::sample_initial_parameters(device, sampling_env, parameters, rng);
        }
        return 0;
    }
    TI rejected = 0;
    for(TI attempt_i = 0; attempt_i < 64; attempt_i++){
        rlt::sample_initial_parameters(device, sampling_env, parameters, rng);
        if(sampled_parameters_safe<PARAMETERS, T, TI>(parameters, dynamics_difficulty)){
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

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, fp::DIFF_TRAINING_SEQUENCE_LENGTH, fp::DIFF_TRAINING_BATCH_SIZE, ENVIRONMENT::Observation::DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, fp::DIFF_TRAINING_BATCH_SIZE, ENVIRONMENT::Observation::DIM>>> step_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, fp::DIFF_TRAINING_BATCH_SIZE, ENVIRONMENT::ACTION_DIM>>> step_actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, ACTOR::OUTPUT_SHAPE>> d_output;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, fp::DIFF_TRAINING_SEQUENCE_LENGTH, fp::DIFF_TRAINING_BATCH_SIZE, 1>>> reset;

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM>> observation_row;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action_row;

    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, rollout_actor);
    rlt::malloc(device, rollout_actor_buffer);
    rlt::malloc(device, rollout_actor_state);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, observations);
    rlt::malloc(device, step_observations);
    rlt::malloc(device, step_actions);
    rlt::malloc(device, d_output);
    rlt::malloc(device, reset);
    rlt::malloc(device, observation_row);
    rlt::malloc(device, action_row);

    rlt::init(device, rng, options.seed);
    rlt::init(device, env);
    rlt::init_weights(device, actor, rng);
    rlt::reset_optimizer_state(device, actor_optimizer, actor);

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
        if(!fp::load_actor_checkpoint(device, actor, actor_checkpoint_to_load)){
            return 1;
        }
        std::cout << "loaded_actor_checkpoint=" << actor_checkpoint_to_load << "\n";
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
        options.terminal_angular_velocity_weight
    };

    auto run_evaluation = [&](){
        rlt::copy(device, device, actor, rollout_actor);
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
                sample_training_parameters<DEVICE, ENVIRONMENT, PARAMETERS, RNG, T, TI>(device, env, parameters, rng, options, (T)1);
            }
            else{
                rlt::initial_parameters(device, env, parameters);
            }
            STATE state;
            rlt::sample_initial_state(device, env, parameters, state, rng);
            l2f_diff::EulerState<T, TI> euler_state;
            l2f_diff::from_l2f_state<STATE, T, TI>(state, euler_state);
            rlt::reset(device, rollout_actor, rollout_actor_state, rng);

            fp::ScalarTerms<T> episode_terms;
            bool invalid = false;
            T episode_action_norm = 0;
            for(TI step_i = 0; step_i < options.eval_horizon; step_i++){
                if(options.reset_hidden_each_step){
                    rlt::reset(device, rollout_actor, rollout_actor_state, rng);
                }
                if(options.eval_model == EvalModel::EULER){
                    l2f_diff::observe<T, TI>(euler_state, observation_row);
                }
                else{
                    rlt::observe(device, env, parameters, state, typename ENVIRONMENT::Observation{}, observation_row, rng);
                }
                for(TI observation_i = 0; observation_i < ENVIRONMENT::Observation::DIM; observation_i++){
                    rlt::set(device, step_observations, rlt::get(observation_row, 0, observation_i), 0, observation_i);
                }
                rlt::evaluate_step(device, rollout_actor, step_observations, rollout_actor_state, step_actions, rollout_actor_buffer, rng, evaluation_mode);
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
                }
                episode_action_norm += std::sqrt(action_norm_sq);
                if(options.eval_model == EvalModel::EULER){
                    l2f_diff::EulerState<T, TI> next_state;
                    l2f_diff::EulerStepCache<T> cache;
                    l2f_diff::step<PARAMETERS, T, TI>(parameters, euler_state, action, next_state, cache);
                    euler_state = next_state;
                    auto terms = fp::euler_state_loss_terms<T, TI>(euler_state, euler_weights);
                    episode_terms.total += terms.total / (T)options.eval_horizon;
                }
                else{
                    STATE next_state;
                    rlt::step(device, env, parameters, state, action_row, next_state, rng);
                    state = next_state;
                    auto terms = fp::l2f_state_loss_terms<STATE, T, TI>(state, l2f_approx_weights);
                    episode_terms.total += terms.total / (T)options.eval_horizon;
                }
            }

            T final_p = 0;
            T final_v = 0;
            T final_w = 0;
            if(options.eval_model == EvalModel::EULER){
                final_p = fp::norm3(euler_state.p);
                final_v = fp::norm3(euler_state.v);
                final_w = fp::norm3(euler_state.omega);
            }
            else{
                final_p = fp::norm3(state.position);
                final_v = fp::norm3(state.linear_velocity);
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

    if(options.eval_only){
        return run_evaluation();
    }

    std::ofstream log_file;
    if(!options.log_path.empty()){
        log_file.open(options.log_path);
        fp::write_training_csv_header(log_file);
    }

    std::cout << "foundation_policy_diff_pre_training\n";
    std::cout << "diff_model=" << fp::diff_model_name(options.diff_model)
              << " sampled_dynamics=" << (options.sample_dynamics ? "true" : "false")
              << " sampled_dynamics_level=" << fp::sampled_dynamics_level_name(options.sample_dynamics ? options.sampled_dynamics_level : fp::SampledDynamicsLevel::FIXED)
              << " batch_size=" << options.batch_size
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
              << " reset_optimizer_on_curriculum_transition=" << (options.reset_optimizer_on_curriculum_transition ? "true" : "false")
              << " physics_gradient_enabled=" << (!options.disable_physics_gradient ? "true" : "false")
              << " reset_hidden_each_step=" << (options.reset_hidden_each_step ? "true" : "false")
              << " init_actor_path=" << (options.init_actor_path.empty() ? "none" : options.init_actor_path)
              << " init_actor_loaded=" << (init_actor_loaded_flag ? "true" : "false")
              << "\n";
    std::cout << "actor=RAPTOR_GRU_FOUNDATION_POLICY observation_dim=" << ENVIRONMENT::Observation::DIM
              << " action_dim=" << ENVIRONMENT::ACTION_DIM
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
            rlt::reset_optimizer_state(device, actor_optimizer, actor);
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
        T action_gradient_norm_sq_before_clip = 0;
        T action_gradient_norm_sq_after_clip = 0;
        bool action_gradient_clipped = false;
        rlt::copy(device, device, actor, rollout_actor);
        rlt::reset(device, rollout_actor, rollout_actor_state, rng);
        rlt::set_all(device, observations, (T)0);
        rlt::set_all(device, d_output, (T)0);
        rlt::set_all(device, reset, false);
        for(TI batch_i = 0; batch_i < MAX_BATCH_SIZE; batch_i++){
            for(TI horizon_i = 0; horizon_i < MAX_HORIZON; horizon_i++){
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    output_gradients[batch_i][horizon_i][action_i] = (T)0;
                    action_derivatives[batch_i][horizon_i][action_i] = (T)1;
                }
            }
        }

        fp::RolloutMetrics<T> mean_rollout_metrics;
        const T diagnostic_nan = std::numeric_limits<T>::quiet_NaN();
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
        SampledDynamicsDiagnostics<PARAMETERS, T, TI> sampled_dynamics_stats;
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            rejected_dynamics_count += sample_training_parameters<DEVICE, ENVIRONMENT, PARAMETERS, RNG, T, TI>(
                device, env, parameters[batch_i], rng, options, dynamics_difficulty
            );
            add_sampled_dynamics_diagnostics<PARAMETERS, T, TI>(
                sampled_dynamics_stats,
                sampled_dynamics_diagnostics<PARAMETERS, T, TI>(parameters[batch_i], env.parameters)
            );
            rlt::sample_initial_state(device, env, parameters[batch_i], states[batch_i][0], rng);
            if(options.state_curriculum || options.fixed_state_difficulty >= (T)0){
                apply_state_curriculum<STATE, T>(states[batch_i][0], state_difficulty);
            }
            l2f_diff::from_l2f_state<STATE, T, TI>(states[batch_i][0], euler_states[batch_i][0]);
            mean_rollout_metrics.initial_position_norm += fp::norm3(states[batch_i][0].position);
            mean_rollout_metrics.initial_velocity_norm += fp::norm3(states[batch_i][0].linear_velocity);
            mean_rollout_metrics.initial_angular_velocity_norm += fp::norm3(states[batch_i][0].angular_velocity);
        }
        scale_sampled_dynamics_diagnostics<PARAMETERS, T, TI>(sampled_dynamics_stats, (T)1 / (T)options.batch_size);
        const T dynamics_rejection_rate = rejected_dynamics_count > 0
            ? (T)rejected_dynamics_count / ((T)rejected_dynamics_count + (T)options.batch_size)
            : (T)0;

        rlt::Mode<rlt::mode::Evaluation<>> evaluation_mode;
        for(TI horizon_i = 0; horizon_i < current_horizon; horizon_i++){
            if(options.reset_hidden_each_step){
                rlt::reset(device, rollout_actor, rollout_actor_state, rng);
            }
            for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
                if(options.diff_model == DiffModel::EULER){
                    l2f_diff::observe<T, TI>(euler_states[batch_i][horizon_i], observation_row);
                }
                else{
                    rlt::observe(device, env, parameters[batch_i], states[batch_i][horizon_i], typename ENVIRONMENT::Observation{}, observation_row, rng);
                }
                for(TI observation_i = 0; observation_i < ENVIRONMENT::Observation::DIM; observation_i++){
                    const T value = rlt::get(observation_row, 0, observation_i);
                    rlt::set(device, step_observations, value, batch_i, observation_i);
                    rlt::set(device, observations, value, horizon_i, batch_i, observation_i);
                }
                rlt::set(device, reset, options.reset_hidden_each_step || horizon_i == 0, horizon_i, batch_i, 0);
            }

            rlt::evaluate_step(device, rollout_actor, step_observations, rollout_actor_state, step_actions, rollout_actor_buffer, rng, evaluation_mode);

            for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
                T action_norm_sq = 0;
                T raw_action_norm_sq = 0;
                TI clamped_this_step = 0;
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    const T raw_action = rlt::get(device, step_actions, batch_i, action_i);
                    T action_derivative;
                    bool clamped;
                    const T action = bounded_action<T>(raw_action, options.action_bound_enabled, options.action_bound_value, action_derivative, clamped);
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
                    if(std::abs(action) > (T)0.98){
                        action_saturation_count++;
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
                }
            }
        }

        fp::ScalarTerms<T> mean_terms;
        TI training_success_count = 0;
        for(TI batch_i = 0; batch_i < options.batch_size; batch_i++){
            if(options.diff_model == DiffModel::EULER){
                auto terms = l2f_diff::rollout_loss_and_gradients<PARAMETERS, T, TI, MAX_HORIZON>(
                    parameters[batch_i],
                    euler_states[batch_i][0],
                    actions[batch_i],
                    current_horizon,
                    step_euler_weights,
                    euler_gradient_states,
                    euler_caches,
                    action_gradients
                );
                mean_terms.total += terms.total();
                mean_terms.position += terms.position;
                mean_terms.velocity += terms.velocity;
                mean_terms.attitude += terms.attitude;
                mean_terms.angular_velocity += terms.angular_velocity;
                mean_terms.action_magnitude += terms.action_magnitude;
                mean_terms.action_smoothness += terms.action_smoothness;
                mean_terms.saturation += terms.saturation;
                mean_terms.terminal += terms.terminal;
                mean_terms.terminal_position += terms.terminal_position;
                mean_terms.terminal_velocity += terms.terminal_velocity;
                mean_terms.terminal_attitude += terms.terminal_attitude;
                mean_terms.terminal_angular_velocity += terms.terminal_angular_velocity;
                mean_rollout_metrics.final_state_norm += l2f_diff::state_norm<T, TI>(euler_gradient_states[current_horizon]);
                mean_rollout_metrics.final_position_norm += fp::norm3(euler_gradient_states[current_horizon].p);
                mean_rollout_metrics.final_velocity_norm += fp::norm3(euler_gradient_states[current_horizon].v);
                mean_rollout_metrics.final_angular_velocity_norm += fp::norm3(euler_gradient_states[current_horizon].omega);
                if(compute_training_success_flag<T, TI>(euler_gradient_states[current_horizon])){
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
                    l2f_approx_weights
                );
                mean_terms.total += terms.total();
                mean_terms.position += terms.position;
                mean_terms.velocity += terms.velocity;
                mean_terms.attitude += terms.attitude;
                mean_terms.angular_velocity += terms.angular_velocity;
                mean_terms.action_magnitude += terms.action_magnitude;
                mean_terms.action_smoothness += terms.action_smoothness;
                mean_terms.saturation += terms.saturation;
                mean_rollout_metrics.final_position_norm += fp::norm3(states[batch_i][current_horizon].position);
                mean_rollout_metrics.final_velocity_norm += fp::norm3(states[batch_i][current_horizon].linear_velocity);
                mean_rollout_metrics.final_angular_velocity_norm += fp::norm3(states[batch_i][current_horizon].angular_velocity);
                const T p_norm = fp::norm3(states[batch_i][current_horizon].position);
                const T v_norm = fp::norm3(states[batch_i][current_horizon].linear_velocity);
                const T w_norm = fp::norm3(states[batch_i][current_horizon].angular_velocity);
                if(p_norm < fp::SUCCESS_POSITION_THRESHOLD &&
                   v_norm < fp::SUCCESS_VELOCITY_THRESHOLD &&
                   w_norm < fp::SUCCESS_ANGULAR_VELOCITY_THRESHOLD){
                    training_success_count++;
                }
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
                    const T gradient = action_gradients[horizon_i][action_i] * action_derivatives[batch_i][horizon_i][action_i] / (T)options.batch_size;
                    output_gradients[batch_i][horizon_i][action_i] = gradient;
                    action_gradient_norm_sq_before_clip += gradient * gradient;
                }
            }
        }
        const T batch_normalizer = (T)1 / (T)options.batch_size;
        mean_terms.total *= batch_normalizer;
        mean_terms.position *= batch_normalizer;
        mean_terms.velocity *= batch_normalizer;
        mean_terms.attitude *= batch_normalizer;
        mean_terms.angular_velocity *= batch_normalizer;
        mean_terms.action_magnitude *= batch_normalizer;
        mean_terms.action_smoothness *= batch_normalizer;
        mean_terms.saturation *= batch_normalizer;
        mean_terms.terminal *= batch_normalizer;
        mean_terms.terminal_position *= batch_normalizer;
        mean_terms.terminal_velocity *= batch_normalizer;
        mean_terms.terminal_attitude *= batch_normalizer;
        mean_terms.terminal_angular_velocity *= batch_normalizer;
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
        const T loss_stabilization = mean_terms.position + mean_terms.velocity + mean_terms.attitude + mean_terms.angular_velocity + mean_terms.terminal;
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

        // Backprop through the GRU actor, then compute and scale actor parameter gradients.
        rlt::Mode<rlt::nn::layers::gru::ResetMode<rlt::mode::Default<>, rlt::nn::layers::gru::ResetModeSpecification<TI, decltype(reset)>>> training_mode;
        training_mode.reset_container = reset;
        rlt::forward(device, actor, observations, actor_buffer, rng, training_mode);
        rlt::zero_gradient(device, actor);
        rlt::backward(device, actor, observations, d_output, actor_buffer, training_mode);

        // Actor parameter-gradient scaling: scales dL/dtheta after BPTT.
        // This is the recommended default stabilization mechanism.
        const auto actor_grad_clip = fp::clip_actor_gradients_by_global_norm<DEVICE, ACTOR, T>(
            device,
            actor,
            options.actor_grad_clip_enabled,
            options.actor_grad_clip_norm,
            options.actor_grad_eps
        );

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
            fp::finite_value(mean_rollout_metrics.action_norm);
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
            rlt::step(device, actor_optimizer, actor);
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
                  << " inertia_trace=" << sampled_dynamics_stats.inertia_trace
                  << " thrust_scale=" << sampled_dynamics_stats.thrust_scale
                  << " torque_scale=" << sampled_dynamics_stats.torque_scale
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
                     << sampled_dynamics_stats.inertia_trace << ","
                     << sampled_dynamics_stats.thrust_scale << ","
                     << sampled_dynamics_stats.torque_scale << ","
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

    if(!options.save_path.empty()){
        if(training_invalid){
            std::cerr << "Not saving actor checkpoint because training encountered NaN/Inf.\n";
            return 1;
        }
        if(!fp::save_actor_checkpoint(device, actor, options.save_path)){
            return 1;
        }
        std::cout << "saved_actor_checkpoint=" << options.save_path << "\n";
    }

    rlt::free(device, observation_row);
    rlt::free(device, action_row);
    rlt::free(device, reset);
    rlt::free(device, d_output);
    rlt::free(device, step_actions);
    rlt::free(device, step_observations);
    rlt::free(device, observations);
    rlt::free(device, actor_optimizer);
    rlt::free(device, rollout_actor_state);
    rlt::free(device, rollout_actor_buffer);
    rlt::free(device, rollout_actor);
    rlt::free(device, actor_buffer);
    rlt::free(device, actor);
    rlt::free(device, rng);
    return 0;
}
