#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/nn_analytics/config.h>
#include <rl_tools/rl/utils/evaluation/operations_generic.h>

#include <fstream>

#include "environment.h"
#include "../pre_training/config.h"
#include "../pre_training/options.h"

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = DEVICE::index_t;
using T = float;

#define RL_TOOLS_POST_TRAINING
#include "config.h"

using EVAL_ACTOR_BATCH = ACTOR::CHANGE_BATCH_SIZE<TI, 1>;
using EVAL_ACTOR_SEQ = EVAL_ACTOR_BATCH::CHANGE_SEQUENCE_LENGTH<TI, 1>;
using EVAL_ACTOR = EVAL_ACTOR_SEQ::CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;

using SAMPLER_PARAMETER_FACTORY = rlt::rl::environments::l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI, rlt::rl::environments::l2f::parameters::DEFAULT_DOMAIN_RANDOMIZATION_OPTIONS<true>>;
using SAMPLER_ENVIRONMENT = rlt::rl::environments::Multirotor<rlt::rl::environments::l2f::Specification<T, TI, SAMPLER_PARAMETER_FACTORY::STATIC_PARAMETERS>>;

struct RuntimeOptions{
    std::string checkpoint_path;
    std::string domain = "fixed";
    std::string csv_path;
    TI seed = 10000;
    TI episodes = 100;
    TI horizon = 128;
};

void print_usage(){
    std::cout
        << "Usage: foundation_policy_post_training_evaluate --checkpoint PATH [options]\n"
        << "Options:\n"
        << "  --domain fixed|narrow|medium\n"
        << "  --seed N\n"
        << "  --episodes N\n"
        << "  --horizon N\n"
        << "  --csv PATH\n";
}

bool parse_non_negative_ti(const std::string& option_name, const char* raw, TI& target){
    try{
        const long long parsed = std::stoll(raw);
        if(parsed < 0){
            std::cerr << option_name << " must be non-negative.\n";
            return false;
        }
        target = static_cast<TI>(parsed);
        return true;
    }
    catch(const std::exception&){
        std::cerr << "Invalid integer for " << option_name << ": " << raw << "\n";
        return false;
    }
}

bool parse_options(int argc, char** argv, RuntimeOptions& options){
    for(int arg_i = 1; arg_i < argc; arg_i++){
        std::string arg = argv[arg_i];
        auto need_value = [&](const std::string& name){
            if(arg_i + 1 >= argc){
                std::cerr << "Missing value for " << name << "\n";
                return false;
            }
            return true;
        };
        if(arg == "--help" || arg == "-h"){
            print_usage();
            return false;
        }
        else if(arg == "--checkpoint"){
            if(!need_value(arg)) return false;
            options.checkpoint_path = argv[++arg_i];
        }
        else if(arg == "--domain"){
            if(!need_value(arg)) return false;
            options.domain = argv[++arg_i];
            if(options.domain != "fixed" && options.domain != "narrow" && options.domain != "medium"){
                std::cerr << "Unsupported --domain: " << options.domain << "\n";
                return false;
            }
        }
        else if(arg == "--seed"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.seed)) return false;
        }
        else if(arg == "--episodes"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.episodes)) return false;
        }
        else if(arg == "--horizon"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.horizon)) return false;
        }
        else if(arg == "--csv"){
            if(!need_value(arg)) return false;
            options.csv_path = argv[++arg_i];
        }
        else{
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage();
            return false;
        }
    }
    if(options.checkpoint_path.empty()){
        std::cerr << "Missing --checkpoint.\n";
        return false;
    }
    return true;
}

template <typename PARAMETERS>
T norm3(const T v[3]){
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

bool finite_value(T value){
    return value == value && value > (T)-1e20 && value < (T)1e20;
}

template <typename PARAMETERS>
T gravity_norm(const PARAMETERS& parameters){
    return std::sqrt(parameters.dynamics.gravity[0] * parameters.dynamics.gravity[0] +
                     parameters.dynamics.gravity[1] * parameters.dynamics.gravity[1] +
                     parameters.dynamics.gravity[2] * parameters.dynamics.gravity[2]);
}

template <typename PARAMETERS>
T estimated_thrust_to_weight(const PARAMETERS& parameters){
    const T max_action = parameters.dynamics.action_limit.max;
    T max_thrust = 0;
    for(TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
        max_thrust += parameters.dynamics.rotor_thrust_coefficients[rotor_i][0] +
                      parameters.dynamics.rotor_thrust_coefficients[rotor_i][1] * max_action +
                      parameters.dynamics.rotor_thrust_coefficients[rotor_i][2] * max_action * max_action;
    }
    return max_thrust / std::max((T)1e-12, parameters.dynamics.mass * gravity_norm(parameters));
}

template <typename PARAMETERS>
T estimated_torque_to_inertia(const PARAMETERS& parameters){
    const T ttw = estimated_thrust_to_weight(parameters);
    const T max_thrust_per_rotor = ttw * parameters.dynamics.mass * gravity_norm(parameters) / (T)PARAMETERS::N;
    const T rotor_distance = std::abs(parameters.dynamics.rotor_positions[0][0]);
    const T max_torque = rotor_distance * (T)1.4142135623730951 * max_thrust_per_rotor;
    return max_torque / std::max((T)1e-12, parameters.dynamics.J[0][0]);
}

template <typename PARAMETERS>
T average_abs_rotor_torque_constant(const PARAMETERS& parameters){
    T value = 0;
    for(TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
        value += std::abs(parameters.dynamics.rotor_torque_constants[rotor_i]);
    }
    return value / (T)PARAMETERS::N;
}

template <typename PARAMETERS>
void configure_sampling_domain(PARAMETERS& parameters, const std::string& domain_name){
    if(domain_name == "fixed"){
        return;
    }
    const T variation = domain_name == "narrow" ? (T)0.10 : (T)0.25;
    const T nominal_mass = parameters.dynamics.mass;
    const T nominal_ttw = estimated_thrust_to_weight(parameters);
    const T nominal_tti = estimated_torque_to_inertia(parameters);
    const T nominal_rising = parameters.dynamics.rotor_time_constants_rising[0];
    const T nominal_falling = parameters.dynamics.rotor_time_constants_falling[0];
    const T nominal_torque_constant = average_abs_rotor_torque_constant(parameters);

    auto& domain = parameters.domain_randomization;
    domain.mass_min = std::max((T)1e-4, nominal_mass * ((T)1 - variation));
    domain.mass_max = std::max(domain.mass_min + (T)1e-5, nominal_mass * ((T)1 + variation));
    domain.thrust_to_weight_min = std::max((T)1.5, nominal_ttw * ((T)1 - variation));
    domain.thrust_to_weight_max = std::max(domain.thrust_to_weight_min + (T)1e-3, nominal_ttw * ((T)1 + variation));
    domain.torque_to_inertia_min = std::max((T)1e-6, nominal_tti * ((T)1 - variation));
    domain.torque_to_inertia_max = std::max(domain.torque_to_inertia_min + (T)1e-3, nominal_tti * ((T)1 + variation));
    domain.mass_size_deviation = variation;
    domain.rotor_time_constant_rising_min = std::max((T)0.005, nominal_rising * ((T)1 - variation));
    domain.rotor_time_constant_rising_max = std::max(domain.rotor_time_constant_rising_min + (T)1e-4, nominal_rising * ((T)1 + variation));
    domain.rotor_time_constant_falling_min = std::max((T)0.005, nominal_falling * ((T)1 - variation));
    domain.rotor_time_constant_falling_max = std::max(domain.rotor_time_constant_falling_min + (T)1e-4, nominal_falling * ((T)1 + variation));
    domain.rotor_torque_constant_min = std::max((T)1e-6, nominal_torque_constant * ((T)1 - variation));
    domain.rotor_torque_constant_max = std::max(domain.rotor_torque_constant_min + (T)1e-6, nominal_torque_constant * ((T)1 + variation));
    domain.orientation_offset_angle_max = 0;
    domain.disturbance_force_max = domain_name == "narrow" ? (T)0.05 : (T)0.15;
}

template <typename PARAMETERS>
bool sampled_parameters_safe(const PARAMETERS& parameters){
    if(!finite_value(parameters.dynamics.mass) || parameters.dynamics.mass <= 0){
        return false;
    }
    if(estimated_thrust_to_weight(parameters) < (T)1.5){
        return false;
    }
    for(TI i = 0; i < 3; i++){
        if(!finite_value(parameters.dynamics.J[i][i]) || parameters.dynamics.J[i][i] <= 0){
            return false;
        }
        if(!finite_value(parameters.dynamics.J_inv[i][i]) || parameters.dynamics.J_inv[i][i] <= 0){
            return false;
        }
    }
    for(TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
        if(!finite_value(parameters.dynamics.rotor_time_constants_rising[rotor_i]) ||
           !finite_value(parameters.dynamics.rotor_time_constants_falling[rotor_i]) ||
           parameters.dynamics.rotor_time_constants_rising[rotor_i] < (T)0.005 ||
           parameters.dynamics.rotor_time_constants_falling[rotor_i] < (T)0.005 ||
           parameters.dynamics.rotor_time_constants_rising[rotor_i] > (T)0.5 ||
           parameters.dynamics.rotor_time_constants_falling[rotor_i] > (T)0.5){
            return false;
        }
        if(!finite_value(parameters.dynamics.rotor_thrust_coefficients[rotor_i][2]) ||
           parameters.dynamics.rotor_thrust_coefficients[rotor_i][2] <= 0){
            return false;
        }
    }
    return true;
}

template <typename DEVICE_T, typename RNG_T>
void sample_eval_parameters(DEVICE_T& device, ENVIRONMENT& env, SAMPLER_ENVIRONMENT& sampler_env, const RuntimeOptions& options, ENVIRONMENT::Parameters& parameters, RNG_T& rng){
    if(options.domain == "fixed"){
        rlt::initial_parameters(device, env, parameters);
        return;
    }
    SAMPLER_ENVIRONMENT::Parameters sampled;
    for(TI attempt_i = 0; attempt_i < 64; attempt_i++){
        rlt::sample_initial_parameters(device, sampler_env, sampled, rng);
        if(sampled_parameters_safe(sampled)){
            break;
        }
    }
    sampled.domain_randomization = rlt::rl::environments::l2f::parameters::domain_randomization_disabled<T>;
    const std::string json = rlt::json(device, sampler_env, sampled);
    rlt::from_json(device, env, json, parameters);
}

int main(int argc, char** argv){
    RuntimeOptions options;
    if(!parse_options(argc, argv, options)){
        return 1;
    }

    DEVICE device;
    RNG rng;
    EVAL_ACTOR actor;
    typename EVAL_ACTOR::template Buffer<DYNAMIC_ALLOCATION> actor_buffer;
    typename EVAL_ACTOR::template State<DYNAMIC_ALLOCATION> actor_state;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, ENVIRONMENT::Observation::DIM>>> step_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, ENVIRONMENT::ACTION_DIM>>> step_actions;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM>> observation_row;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action_row;

    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, actor_state);
    rlt::malloc(device, step_observations);
    rlt::malloc(device, step_actions);
    rlt::malloc(device, observation_row);
    rlt::malloc(device, action_row);
    rlt::init(device, rng, options.seed);

    try{
        auto file = HighFive::File(options.checkpoint_path, HighFive::File::ReadOnly);
        rlt::load(device, actor, file.getGroup("actor"));
    }
    catch(const HighFive::Exception& e){
        std::cerr << "Failed to load actor checkpoint " << options.checkpoint_path << ": " << e.what() << "\n";
        return 1;
    }

    ENVIRONMENT env;
    SAMPLER_ENVIRONMENT sampler_env;
    rlt::init(device, env);
    rlt::init(device, sampler_env);
    rlt::initial_parameters(device, sampler_env, sampler_env.parameters);
    configure_sampling_domain(sampler_env.parameters, options.domain);

    TI successes = 0;
    TI invalid_count = 0;
    T mean_final_p = 0;
    T mean_final_v = 0;
    T mean_final_w = 0;
    T mean_action_norm = 0;
    rlt::Mode<rlt::mode::Evaluation<>> mode;

    for(TI episode_i = 0; episode_i < options.episodes; episode_i++){
        ENVIRONMENT::Parameters parameters;
        sample_eval_parameters(device, env, sampler_env, options, parameters, rng);
        ENVIRONMENT::State state;
        rlt::sample_initial_state(device, env, parameters, state, rng);
        rlt::reset(device, actor, actor_state, rng);
        T episode_action_norm = 0;
        for(TI step_i = 0; step_i < options.horizon; step_i++){
            rlt::observe(device, env, parameters, state, typename ENVIRONMENT::Observation{}, observation_row, rng);
            for(TI observation_i = 0; observation_i < ENVIRONMENT::Observation::DIM; observation_i++){
                rlt::set(device, step_observations, rlt::get(observation_row, 0, observation_i), 0, observation_i);
            }
            rlt::evaluate_step(device, actor, step_observations, actor_state, step_actions, actor_buffer, rng, mode);
            T action_norm_sq = 0;
            for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                const T action_value = rlt::get(device, step_actions, 0, action_i);
                rlt::set(action_row, 0, action_i, action_value);
                action_norm_sq += action_value * action_value;
            }
            episode_action_norm += std::sqrt(action_norm_sq);
            ENVIRONMENT::State next_state;
            rlt::step(device, env, parameters, state, action_row, next_state, rng);
            state = next_state;
        }
        const T final_p = norm3<ENVIRONMENT::Parameters>(state.position);
        const T final_v = norm3<ENVIRONMENT::Parameters>(state.linear_velocity);
        const T final_w = norm3<ENVIRONMENT::Parameters>(state.angular_velocity);
        const bool invalid = !finite_value(final_p) || !finite_value(final_v) || !finite_value(final_w);
        if(invalid){
            invalid_count++;
        }
        else if(final_p < (T)1.0 && final_v < (T)2.0 && final_w < (T)5.0){
            successes++;
        }
        mean_final_p += final_p;
        mean_final_v += final_v;
        mean_final_w += final_w;
        mean_action_norm += episode_action_norm / std::max((T)1, (T)options.horizon);
    }

    const T normalizer = options.episodes > 0 ? (T)1 / (T)options.episodes : (T)1;
    const T success_rate = successes * normalizer;
    const T invalid_rate = invalid_count * normalizer;
    mean_final_p *= normalizer;
    mean_final_v *= normalizer;
    mean_final_w *= normalizer;
    mean_action_norm *= normalizer;

    std::cout << "student_evaluation"
              << " checkpoint=" << options.checkpoint_path
              << " domain=" << options.domain
              << " seed=" << options.seed
              << " episodes=" << options.episodes
              << " horizon=" << options.horizon
              << " success_rate=" << success_rate
              << " mean_final_position_norm=" << mean_final_p
              << " mean_final_velocity_norm=" << mean_final_v
              << " mean_final_angular_velocity_norm=" << mean_final_w
              << " invalid_or_nan_rate=" << invalid_rate
              << " mean_action_norm=" << mean_action_norm
              << " explicit_physical_parameters_in_observation=false\n";

    if(!options.csv_path.empty()){
        std::filesystem::create_directories(std::filesystem::path(options.csv_path).parent_path());
        std::ofstream csv(options.csv_path);
        csv << "checkpoint,domain,seed,episodes,horizon,success_rate,mean_final_position_norm,mean_final_velocity_norm,mean_final_angular_velocity_norm,invalid_or_nan_rate,mean_action_norm\n";
        csv << options.checkpoint_path << ","
            << options.domain << ","
            << options.seed << ","
            << options.episodes << ","
            << options.horizon << ","
            << success_rate << ","
            << mean_final_p << ","
            << mean_final_v << ","
            << mean_final_w << ","
            << invalid_rate << ","
            << mean_action_norm << "\n";
    }

    rlt::free(device, rng);
    rlt::free(device, actor);
    rlt::free(device, actor_buffer);
    rlt::free(device, actor_state);
    rlt::free(device, step_observations);
    rlt::free(device, step_actions);
    rlt::free(device, observation_row);
    rlt::free(device, action_row);
    return invalid_count == options.episodes ? 1 : 0;
}
