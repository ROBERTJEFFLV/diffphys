#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>

#include "../equivalent_dynamics.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace eq_dyn = rl_tools::foundation_policy::equivalent_dynamics;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = DEVICE::index_t;
using T = double;
constexpr bool DYNAMIC_ALLOCATION = true;

using PARAMETER_FACTORY = rlt::rl::environments::l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI, rlt::rl::environments::l2f::parameters::DEFAULT_DOMAIN_RANDOMIZATION_OPTIONS<true>>;
using ENVIRONMENT = rlt::rl::environments::Multirotor<rlt::rl::environments::l2f::Specification<T, TI, PARAMETER_FACTORY::STATIC_PARAMETERS>>;

struct RuntimeOptions{
    TI seed = 0;
    TI count = 1000;
    std::string output_path = "./src/foundation_policy/dynamics_parameters/";
    std::string registry_output_path = "./src/foundation_policy/registry/";
    std::string manifest_path;
    std::string domain = "broad";
    bool balanced = true;
    TI balance_bins = 4;
};

void print_usage(){
    std::cout
        << "Usage: foundation_policy_pre_training_sample_dynamics_parameters [options]\n"
        << "Options:\n"
        << "  --seed N                    Sampling seed.\n"
        << "  --count N                   Number of dynamics JSON files to write.\n"
        << "  --output-path PATH          Output directory for dynamics JSON files.\n"
        << "  --registry-output-path PATH Output directory for registry JSON files.\n"
        << "  --manifest-path PATH        CSV manifest output path.\n"
        << "  --domain broad|narrow|medium Sampling domain.\n"
        << "  --balanced                  Enable balanced root-dynamics bins (default).\n"
        << "  --disable-balanced          Disable balanced root-dynamics bins.\n"
        << "  --balance-bins N            Number of bins per root dynamics variable.\n";
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
        else if(arg == "--seed"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.seed)) return false;
        }
        else if(arg == "--count"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.count)) return false;
        }
        else if(arg == "--output-path"){
            if(!need_value(arg)) return false;
            options.output_path = argv[++arg_i];
        }
        else if(arg == "--registry-output-path"){
            if(!need_value(arg)) return false;
            options.registry_output_path = argv[++arg_i];
        }
        else if(arg == "--manifest-path"){
            if(!need_value(arg)) return false;
            options.manifest_path = argv[++arg_i];
        }
        else if(arg == "--domain"){
            if(!need_value(arg)) return false;
            options.domain = argv[++arg_i];
            if(options.domain != "broad" && options.domain != "narrow" && options.domain != "medium"){
                std::cerr << "Unsupported --domain: " << options.domain << "\n";
                return false;
            }
        }
        else if(arg == "--balanced"){
            options.balanced = true;
        }
        else if(arg == "--disable-balanced"){
            options.balanced = false;
        }
        else if(arg == "--balance-bins"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.balance_bins)) return false;
            if(options.balance_bins == 0){
                std::cerr << "--balance-bins must be positive.\n";
                return false;
            }
        }
        else{
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage();
            return false;
        }
    }
    return true;
}

template <typename TI>
eq_dyn::DynamicsBins<TI> scheduled_balanced_dynamics_bins(TI sample_i, TI num_bins){
    eq_dyn::DynamicsBins<TI> bins;
    if(num_bins <= 1){
        return bins;
    }
    bins.size_mass = sample_i % num_bins;
    bins.thrust_to_weight = ((TI)3 * sample_i) % num_bins;
    bins.torque_to_inertia = ((TI)5 * sample_i + sample_i / std::max((TI)1, num_bins)) % num_bins;
    bins.motor_delay = ((TI)7 * sample_i + sample_i / std::max((TI)1, num_bins * num_bins)) % num_bins;
    return bins;
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
void motor_tau_stats(const PARAMETERS& parameters, T& mean, T& min_v, T& max_v){
    mean = 0;
    min_v = std::numeric_limits<T>::infinity();
    max_v = 0;
    for(TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
        const T rising = parameters.dynamics.rotor_time_constants_rising[rotor_i];
        const T falling = parameters.dynamics.rotor_time_constants_falling[rotor_i];
        mean += rising + falling;
        min_v = std::min(min_v, std::min(rising, falling));
        max_v = std::max(max_v, std::max(rising, falling));
    }
    mean /= (T)(2 * PARAMETERS::N);
}

template <typename PARAMETERS>
void configure_common(PARAMETERS& parameters){
    parameters.mdp.reward = {
        false, 1.00, 1.50, -100.00,
        1.00, 0.00, 0.10, 0.00,
        0.00, 0.00, 0.00, 0.00,
        1.00, 0.00,
    };
}

template <typename PARAMETERS>
void configure_sampling_domain(PARAMETERS& parameters, const std::string& domain_name){
    configure_common(parameters);
    if(domain_name == "broad"){
        parameters.domain_randomization = {
            1.5, 5.0,
            40, 1200,
            0.02, 5.00,
            0.1,
            0.03, 0.10,
            0.03, 0.30,
            0.005, 0.05,
            0.0, 0.3
        };
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
bool finite_value(T value){
    return value == value && value > (T)-1e20 && value < (T)1e20;
}

template <typename PARAMETERS>
bool sampled_parameters_safe(const PARAMETERS& parameters){
    if(!finite_value<PARAMETERS>(parameters.dynamics.mass) || parameters.dynamics.mass <= 0){
        return false;
    }
    if(estimated_thrust_to_weight(parameters) < (T)1.5){
        return false;
    }
    for(TI i = 0; i < 3; i++){
        if(!finite_value<PARAMETERS>(parameters.dynamics.J[i][i]) || parameters.dynamics.J[i][i] <= 0){
            return false;
        }
        if(!finite_value<PARAMETERS>(parameters.dynamics.J_inv[i][i]) || parameters.dynamics.J_inv[i][i] <= 0){
            return false;
        }
    }
    for(TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
        if(!finite_value<PARAMETERS>(parameters.dynamics.rotor_time_constants_rising[rotor_i]) ||
           !finite_value<PARAMETERS>(parameters.dynamics.rotor_time_constants_falling[rotor_i]) ||
           parameters.dynamics.rotor_time_constants_rising[rotor_i] < (T)0.005 ||
           parameters.dynamics.rotor_time_constants_falling[rotor_i] < (T)0.005 ||
           parameters.dynamics.rotor_time_constants_rising[rotor_i] > (T)0.5 ||
           parameters.dynamics.rotor_time_constants_falling[rotor_i] > (T)0.5){
            return false;
        }
        if(!finite_value<PARAMETERS>(parameters.dynamics.rotor_thrust_coefficients[rotor_i][2]) ||
           parameters.dynamics.rotor_thrust_coefficients[rotor_i][2] <= 0){
            return false;
        }
    }
    return true;
}

template <typename PARAMETERS>
void write_manifest_row(
    std::ofstream& manifest,
    TI dynamics_id,
    const PARAMETERS& parameters,
    const PARAMETERS& nominal_parameters,
    const eq_dyn::DynamicsBins<TI>& bins,
    TI rejected,
    bool safe
){
    T tau_mean = 0;
    T tau_min = 0;
    T tau_max = 0;
    motor_tau_stats(parameters, tau_mean, tau_min, tau_max);
    const auto latent_target = eq_dyn::normalized_target_from_parameters<PARAMETERS>(parameters, nominal_parameters);
    manifest << dynamics_id << ","
             << std::setprecision(12)
             << bins.size_mass << ","
             << bins.thrust_to_weight << ","
             << bins.torque_to_inertia << ","
             << bins.motor_delay << ","
             << latent_target.thrust_to_acceleration_gain << ","
             << latent_target.roll_pitch_torque_to_angular_acceleration_gain << ","
             << latent_target.yaw_torque_to_angular_acceleration_gain << ","
             << latent_target.motor_rise_time_constant << ","
             << latent_target.motor_fall_time_constant << ","
             << latent_target.thrust_curve_shape << ","
             << latent_target.torque_curve_shape << ","
             << latent_target.residual_force_bias << ","
             << latent_target.residual_torque_bias << ","
             << parameters.dynamics.mass << ","
             << parameters.dynamics.J[0][0] << ","
             << parameters.dynamics.J[1][1] << ","
             << parameters.dynamics.J[2][2] << ","
             << estimated_thrust_to_weight(parameters) << ","
             << estimated_torque_to_inertia(parameters) << ","
             << tau_mean << ","
             << tau_min << ","
             << tau_max << ","
             << average_abs_rotor_torque_constant(parameters) << ","
             << rejected << ","
             << (safe ? 1 : 0) << "\n";
}

int main(int argc, char** argv){
    RuntimeOptions options;
    if(!parse_options(argc, argv, options)){
        return 1;
    }

    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, options.seed);

    ENVIRONMENT env;
    ENVIRONMENT::Parameters params;
    rlt::init(device, env);
    rlt::initial_parameters(device, env, env.parameters);
    configure_sampling_domain(env.parameters, options.domain);

    const std::filesystem::path output_path = options.output_path;
    std::filesystem::create_directories(output_path);
    if(!options.manifest_path.empty()){
        std::filesystem::create_directories(std::filesystem::path(options.manifest_path).parent_path());
    }
    std::ofstream manifest;
    if(!options.manifest_path.empty()){
        manifest.open(options.manifest_path);
        manifest << "dynamics_id,size_mass_bin,thrust_to_weight_bin,torque_to_inertia_bin,motor_delay_bin,"
                 << "latent_target_thrust_to_acceleration_gain,"
                 << "latent_target_roll_pitch_torque_to_angular_acceleration_gain,"
                 << "latent_target_yaw_torque_to_angular_acceleration_gain,"
                 << "latent_target_motor_rise_time_constant,latent_target_motor_fall_time_constant,"
                 << "latent_target_thrust_curve_shape,latent_target_torque_curve_shape,"
                 << "latent_target_residual_force_bias,latent_target_residual_torque_bias,"
                 << "mass_diagnostic,Jxx_diagnostic,Jyy_diagnostic,Jzz_diagnostic,"
                 << "thrust_to_weight_diagnostic,torque_to_inertia_diagnostic,"
                 << "motor_tau_mean_diagnostic,motor_tau_min_diagnostic,motor_tau_max_diagnostic,"
                 << "rotor_torque_constant_abs_mean_diagnostic,rejected_samples,safety_filter_pass\n";
    }

    for(TI set_i = 0; set_i < options.count; ++set_i){
        auto sampling_env = env;
        auto bins = scheduled_balanced_dynamics_bins<TI>(set_i, options.balance_bins);
        if(options.balanced){
            eq_dyn::restrict_domain_to_bins<ENVIRONMENT::Parameters::DomainRandomization, T, TI>(
                sampling_env.parameters.domain_randomization,
                bins,
                options.balance_bins
            );
        }
        else{
            bins = eq_dyn::bins_from_parameters<ENVIRONMENT::Parameters, TI>(sampling_env.parameters, options.balance_bins);
        }
        TI rejected = 0;
        bool safe = false;
        for(TI attempt_i = 0; attempt_i < 64; attempt_i++){
            rlt::sample_initial_parameters(device, sampling_env, params, rng);
            configure_common(params);
            safe = sampled_parameters_safe(params);
            if(safe){
                break;
            }
            rejected++;
        }
        auto params_copy = params;
        params_copy.domain_randomization = rlt::rl::environments::l2f::parameters::domain_randomization_disabled<T>;
        std::ofstream output(output_path / (std::to_string(set_i) + ".json"));
        output << rlt::json(device, env, params_copy);
        output.close();
        if(manifest){
            write_manifest_row(manifest, set_i, params, env.parameters, bins, rejected, safe);
        }
    }

    const std::filesystem::path output_path_registry = options.registry_output_path;
    std::filesystem::create_directories(output_path_registry);
    std::vector<std::tuple<std::string, ENVIRONMENT::Parameters::Dynamics>> registry;
    auto permute_rotors_px4_to_cf = [&device, &env](const auto& dynamics){
        auto copy = dynamics;
        rlt::permute_rotors(device, env, copy, 0, 3, 1, 2);
        return copy;
    };
    registry.emplace_back("crazyflie", rlt::rl::environments::l2f::parameters::dynamics::crazyflie<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>);
    registry.emplace_back("x500", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::x500::real<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));
    registry.emplace_back("mrs", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::mrs<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));
    registry.emplace_back("fs", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::fs::base<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));
    registry.emplace_back("race", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::race<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));
    registry.emplace_back("flightmare", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::flightmare<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));

    rlt::initial_parameters(device, env, params);
    configure_common(params);
    for(const auto& [name, dynamics] : registry){
        params.dynamics = dynamics;
        std::ofstream output(output_path_registry / (name + ".json"));
        output << rlt::json(device, env, params);
        output.close();
    }

    std::cout << "sampled_dynamics_parameters count=" << options.count
              << " domain=" << options.domain
              << " balanced=" << (options.balanced ? "true" : "false")
              << " balance_bins=" << options.balance_bins
              << " output_path=" << output_path.string()
              << " manifest_path=" << options.manifest_path << std::endl;
    rlt::free(device, rng);
    return 0;
}
