#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/td3_sampling/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#include <rl_tools/rl/components/replay_buffer/persist.h>

#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/gru/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn/layers/td3_sampling/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist_code.h>

#include "environment.h"

#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>

#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/nn_analytics/operations_cpu.h>

#include <rl_tools/rl/utils/evaluation/operations_cpu.h>

#include "config.h"
#include "options.h"

#include <limits>
#include <sstream>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;
constexpr bool DYNAMIC_ALLOCATION = true;

using OPTIONS = OPTIONS_PRE_TRAINING;

using FACTORY = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS, DYNAMIC_ALLOCATION>;
using LOOP_CORE_CONFIG = FACTORY::LOOP_CORE_CONFIG;
using LOOP_CONFIG = builder::LOOP_ASSEMBLY<LOOP_CORE_CONFIG>::LOOP_CONFIG;

struct TeacherPreTrainingRuntimeOptions{
    TI seed = 0;
    TI max_steps = 0;
    std::string output_root;
    std::string experiment;
    std::vector<std::string> file_paths;
};

void print_teacher_pre_training_usage(){
    std::cout
        << "Usage: foundation_policy_pre_training [options] [dynamics_json ...]\n"
        << "Options:\n"
        << "  --seed N              Training seed.\n"
        << "  --max-steps N         Stop after N environment steps and save a final HDF5 actor checkpoint.\n"
        << "  --output-root PATH    ExTrack base path for teacher checkpoints.\n"
        << "  --experiment NAME     ExTrack experiment name.\n"
        << "  --help                Show this message.\n";
}

bool parse_non_negative_ti_teacher(const std::string& option_name, const char* raw, TI& target){
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

bool parse_teacher_pre_training_options(int argc, char** argv, TeacherPreTrainingRuntimeOptions& options){
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
            print_teacher_pre_training_usage();
            return false;
        }
        else if(arg == "--seed"){
            if(!need_value(arg) || !parse_non_negative_ti_teacher(arg, argv[++arg_i], options.seed)) return false;
        }
        else if(arg == "--max-steps"){
            if(!need_value(arg) || !parse_non_negative_ti_teacher(arg, argv[++arg_i], options.max_steps)) return false;
        }
        else if(arg == "--output-root"){
            if(!need_value(arg)) return false;
            options.output_root = argv[++arg_i];
        }
        else if(arg == "--experiment"){
            if(!need_value(arg)) return false;
            options.experiment = argv[++arg_i];
        }
        else if(!arg.empty() && arg[0] == '-'){
            std::cerr << "Unknown option: " << arg << "\n";
            print_teacher_pre_training_usage();
            return false;
        }
        else{
            options.file_paths.push_back(arg);
        }
    }
    return true;
}

// note: make sure that the rng_params is invoked in the exact same way in pre- as in post-training, to make sure the params used to sample parameters to generate data from the trained policy are matching the ones seen by the particular policy for the seed during pretraining

int main(int argc, char** argv){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    TeacherPreTrainingRuntimeOptions runtime_options;
    if(!parse_teacher_pre_training_options(argc, argv, runtime_options)){
        return 1;
    }
    const TI seed = runtime_options.seed;
    rlt::init(device, rng, seed);

    std::vector<std::string> file_paths = runtime_options.file_paths;
    if (file_paths.empty()){
        // iterate dynamics_parameters directory
        std::filesystem::path dynamics_parameters_path = "./src/foundation_policy/dynamics_parameters/";
        if (!std::filesystem::exists(dynamics_parameters_path)){
            std::cerr << "Dynamics parameters path does not exist: " << dynamics_parameters_path << std::endl;
            return 1;
        }
        for (const auto& entry : std::filesystem::directory_iterator(dynamics_parameters_path)){
            file_paths.push_back(entry.path().string());
        }
        std::sort(file_paths.begin(), file_paths.end());
    }
    for (const auto& file_path_string : file_paths){
        typename LOOP_CONFIG::template State <LOOP_CONFIG> ts;
        rlt::malloc(device, ts);
        ts.extrack_config.name = "foundation-policy-pre-training";
        if(!runtime_options.output_root.empty()){
            ts.extrack_config.base_path = runtime_options.output_root;
        }
        if(!runtime_options.experiment.empty()){
            ts.extrack_config.experiment = runtime_options.experiment;
        }

        auto& base_env = rlt::get(ts.off_policy_runner.envs, 0, 0);
        std::filesystem::path file_path = file_path_string;
        if (file_path.filename().string()[0] == '.'){
            continue;
        }
        if (!std::filesystem::exists(file_path)){
            std::cerr << "Dynamics parameters path does not exist: " << file_path << std::endl;
            break;
        }
        std::ifstream file(file_path, std::ios::in | std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + file_path.string());
        }
        std::cout << "Loading dynamics parameters from: " << file_path.string() << std::endl;
        std::ostringstream buffer;
        buffer << file.rdbuf();
        decltype(base_env.parameters) new_params;
        rlt::from_json(device, base_env, buffer.str(), new_params);
        ts.extrack_config.population_variates = "dynamics-id";
        ts.extrack_config.population_values = file_path.filename().stem().string();
        rlt::init(device, ts, seed);
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
        rlt::init(device, device.logger, ts.extrack_paths.seed);
#endif
        // auto old_init = base_env.parameters.mdp.init;
        base_env.parameters = new_params;
        // base_env.parameters.mdp.init = old_init;
        for (TI env_i = 1; env_i < LOOP_CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS; env_i++){
            auto& env = rlt::get(ts.off_policy_runner.envs, 0, env_i);
            env.parameters = base_env.parameters;
        }
        ts.env_eval.parameters = base_env.parameters;


        bool finished = false;
        while(!finished){
            // if (ts.step % LOOP_CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL == 0){
            //     auto step_folder = rlt::get_step_folder(device, ts.extrack_config, ts.extrack_paths, ts.step);
            //     std::filesystem::path checkpoint_path = std::filesystem::path(step_folder) / "critic_checkpoint.h5";
            //     std::cerr << "Checkpointing critic to: " << checkpoint_path.string() << std::endl;
            //     auto file = HighFive::File(checkpoint_path.string(), HighFive::File::Overwrite);
            //     auto group_0 = file.createGroup("critic_0");
            //     auto group_1 = file.createGroup("critic_1");
            //     rl_tools::save(device, ts.actor_critic.critics[0], group_0);
            //     rl_tools::save(device, ts.actor_critic.critics[1], group_1);
            // }
            finished = rlt::step(device, ts);
            if(runtime_options.max_steps > 0 && ts.step >= runtime_options.max_steps){
                auto step_folder = rlt::get_step_folder(device, ts.extrack_config, ts.extrack_paths, ts.step);
                auto& actor_checkpoint = rlt::get_actor(ts);
                rlt::rl::loop::steps::checkpoint::save<LOOP_CONFIG::DYNAMIC_ALLOCATION, typename LOOP_CONFIG::ENVIRONMENT, typename LOOP_CONFIG::CHECKPOINT_PARAMETERS>(device, step_folder.string(), actor_checkpoint, ts.rng_checkpoint);
                std::cout << "manual_final_checkpoint=" << (step_folder / "checkpoint.h5").string() << " step=" << ts.step << std::endl;
                finished = true;
            }
        }
        std::filesystem::create_directories(ts.extrack_paths.seed);
        std::ofstream return_file(ts.extrack_paths.seed / "return.json");
        return_file << "[";
        for(TI evaluation_i = 0; evaluation_i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS; evaluation_i++){
            auto& result = get(ts.evaluation_results, 0, evaluation_i);
            return_file << rlt::json(device, result, LOOP_CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL * LOOP_CONFIG::ENVIRONMENT_STEPS_PER_LOOP_STEP * evaluation_i);
            if(evaluation_i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS - 1){
                return_file << ", ";
            }
        }
        return_file << "]";
        std::ofstream return_file_confirmation(ts.extrack_paths.seed / "return.json.set");
        return_file_confirmation.close();
        rlt::free(device, ts);
    }
    rlt::free(device, rng);
    return 0;
}