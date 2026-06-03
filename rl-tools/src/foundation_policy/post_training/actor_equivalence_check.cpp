#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include "environment.h"
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/nn_analytics/operations_cpu.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include "../pre_training/config.h"
#include "../pre_training/options.h"
namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = DEVICE::index_t;
using T = float;
#define RL_TOOLS_POST_TRAINING
#include "config.h"
#include "../diff_pre_training/config.h"
#include "../diff_pre_training/checkpoint_io.h"

namespace diff_pt = rl_tools::foundation_policy::diff_pre_training;

struct Options{
    std::string checkpoint_path;
    std::string write_random_checkpoint_path;
    TI seed = 0;
    T max_abs_threshold = (T)1e-5;
    T mean_abs_threshold = (T)1e-6;
};

void print_usage(){
    std::cout
        << "Usage: foundation_policy_actor_equivalence_check --checkpoint PATH [options]\n"
        << "       foundation_policy_actor_equivalence_check --write-random-checkpoint PATH [--seed N]\n"
        << "Options:\n"
        << "  --checkpoint PATH                Diff-pretraining actor text checkpoint.\n"
        << "  --write-random-checkpoint PATH   Write a random compatible text checkpoint for smoke testing.\n"
        << "  --seed N                         Input/checkpoint seed.\n"
        << "  --max-abs-threshold VALUE        Default 1e-5.\n"
        << "  --mean-abs-threshold VALUE       Default 1e-6.\n";
}

bool parse_options(int argc, char** argv, Options& options){
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
        else if(arg == "--write-random-checkpoint"){
            if(!need_value(arg)) return false;
            options.write_random_checkpoint_path = argv[++arg_i];
        }
        else if(arg == "--seed"){
            if(!need_value(arg)) return false;
            options.seed = std::stoll(argv[++arg_i]);
        }
        else if(arg == "--max-abs-threshold"){
            if(!need_value(arg)) return false;
            options.max_abs_threshold = std::stof(argv[++arg_i]);
        }
        else if(arg == "--mean-abs-threshold"){
            if(!need_value(arg)) return false;
            options.mean_abs_threshold = std::stof(argv[++arg_i]);
        }
        else{
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage();
            return false;
        }
    }
    if(options.checkpoint_path.empty() && options.write_random_checkpoint_path.empty()){
        std::cerr << "Either --checkpoint or --write-random-checkpoint is required.\n";
        print_usage();
        return false;
    }
    return true;
}

int main(int argc, char** argv){
    Options options;
    if(!parse_options(argc, argv, options)){
        return 1;
    }

    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, options.seed);

    constexpr TI CHECK_SEQUENCE_LENGTH = 17;
    constexpr TI CHECK_BATCH_SIZE = 9;
    static_assert(diff_pt::ENVIRONMENT::Observation::DIM == ENVIRONMENT::Observation::DIM,
                  "Diff-pretraining and post-training observation dimensions must match");
    static_assert(diff_pt::ENVIRONMENT::ACTION_DIM == ENVIRONMENT::ACTION_DIM,
                  "Diff-pretraining and post-training action dimensions must match");

    using CHECK_INPUT_SHAPE = rlt::tensor::Shape<TI, CHECK_SEQUENCE_LENGTH, CHECK_BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
    using DIFF_ACTOR = rlt::nn_models::sequential::Build<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>, diff_pt::MODULE_CHAIN, CHECK_INPUT_SHAPE>;
    using POST_ACTOR = rlt::nn_models::sequential::Build<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>, MODULE_CHAIN, CHECK_INPUT_SHAPE>;
    using INPUT = rlt::Tensor<rlt::tensor::Specification<T, TI, CHECK_INPUT_SHAPE>>;
    using OUTPUT = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, CHECK_SEQUENCE_LENGTH, CHECK_BATCH_SIZE, ENVIRONMENT::ACTION_DIM>>>;
    using RESET = rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, CHECK_SEQUENCE_LENGTH, CHECK_BATCH_SIZE, 1>>>;

    if(!options.write_random_checkpoint_path.empty()){
        POST_ACTOR actor;
        rlt::malloc(device, actor);
        rlt::init_weights(device, actor, rng);
        const bool ok = diff_pt::save_actor_checkpoint(device, actor, options.write_random_checkpoint_path);
        rlt::free(device, actor);
        rlt::free(device, rng);
        if(!ok){
            return 1;
        }
        std::cout << "Wrote random actor text checkpoint: " << options.write_random_checkpoint_path << "\n";
        if(options.checkpoint_path.empty()){
            return 0;
        }
    }

    DIFF_ACTOR diff_actor;
    POST_ACTOR post_actor;
    typename DIFF_ACTOR::Buffer<DYNAMIC_ALLOCATION> diff_buffer;
    typename POST_ACTOR::Buffer<DYNAMIC_ALLOCATION> post_buffer;
    INPUT input;
    OUTPUT diff_output;
    OUTPUT post_output;
    RESET reset;

    rlt::malloc(device, diff_actor);
    rlt::malloc(device, post_actor);
    rlt::malloc(device, diff_buffer);
    rlt::malloc(device, post_buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, diff_output);
    rlt::malloc(device, post_output);
    rlt::malloc(device, reset);

    if(!diff_pt::load_actor_checkpoint(device, diff_actor, options.checkpoint_path)){
        return 1;
    }
    if(!diff_pt::load_actor_checkpoint(device, post_actor, options.checkpoint_path)){
        return 1;
    }

    rlt::randn(device, input, rng);
    for(TI step_i = 0; step_i < CHECK_SEQUENCE_LENGTH; step_i++){
        for(TI batch_i = 0; batch_i < CHECK_BATCH_SIZE; batch_i++){
            rlt::set(device, reset, step_i == 0, step_i, batch_i, 0);
        }
    }

    using RESET_MODE_SPEC = rlt::nn::layers::gru::ResetModeSpecification<TI, RESET>;
    using RESET_MODE = rlt::nn::layers::gru::ResetMode<rlt::mode::Default<>, RESET_MODE_SPEC>;
    rlt::Mode<RESET_MODE> mode;
    mode.reset_container = reset;

    rlt::evaluate(device, diff_actor, input, diff_output, diff_buffer, rng, mode);
    rlt::evaluate(device, post_actor, input, post_output, post_buffer, rng, mode);
    T max_abs_error = 0;
    T mean_abs_error = 0;
    TI count = 0;
    for(TI step_i = 0; step_i < CHECK_SEQUENCE_LENGTH; step_i++){
        for(TI batch_i = 0; batch_i < CHECK_BATCH_SIZE; batch_i++){
            for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                const T error = std::abs(rlt::get(device, diff_output, step_i, batch_i, action_i) - rlt::get(device, post_output, step_i, batch_i, action_i));
                max_abs_error = std::max(max_abs_error, error);
                mean_abs_error += error;
                count++;
            }
        }
    }
    mean_abs_error /= std::max<TI>(1, count);

    std::cout << "actor_equivalence max_abs_error=" << max_abs_error
              << " mean_abs_error=" << mean_abs_error
              << " count=" << count << "\n";

    const bool pass = max_abs_error <= options.max_abs_threshold && mean_abs_error <= options.mean_abs_threshold;
    std::cout << (pass ? "PASS" : "FAIL") << "\n";

    rlt::free(device, diff_actor);
    rlt::free(device, post_actor);
    rlt::free(device, diff_buffer);
    rlt::free(device, post_buffer);
    rlt::free(device, input);
    rlt::free(device, diff_output);
    rlt::free(device, post_output);
    rlt::free(device, reset);
    rlt::free(device, rng);
    return pass ? 0 : 1;
}
