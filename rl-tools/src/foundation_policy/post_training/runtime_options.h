#pragma once

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <cstdlib>

struct PostTrainingRuntimeOptions{
    TI seed = 0;
    TI num_teachers = NUM_TEACHERS;
    TI num_active_teachers = NUM_ACTIVE_TEACHERS;
    TI num_epochs = N_EPOCH;
    TI teacher_forcing_epochs = EPOCH_TEACHER_FORCING;
    TI max_batches_per_epoch = 0;
    T min_teacher_return = SOLVED_RETURN;
    std::string init_actor_path;
    std::string run_path;
    std::string teacher_experiment = "2025-04-16_20-10-58";
    std::string teacher_search_root = "1k-experiments";
    std::string dynamics_parameter_index;
    std::string dynamics_parameters_path;
};

inline void print_post_training_usage(){
    std::cout
        << "Usage: foundation_policy_post_training [options]\n"
        << "Options:\n"
        << "  --seed N                         Training seed.\n"
        << "  --init-actor-path PATH           Diff-pretraining actor text checkpoint for student actor init.\n"
        << "  --num-teachers N                 Number of teacher checkpoints to load.\n"
        << "  --teacher-count N                Alias for --num-teachers.\n"
        << "  --num-active-teachers N          Number of lowest-return teachers to add during student epochs.\n"
        << "  --epochs N                       Number of post-training epochs.\n"
        << "  --teacher-forcing-epochs N       Behavioral-cloning-only epochs.\n"
        << "  --teacher-experiment NAME        Teacher checkpoint experiment id.\n"
        << "  --teacher-search-root PATH       Root passed to find_latest_run for teacher HDF5 checkpoints.\n"
        << "  --teacher-index-path PATH        Teacher dynamics/checkpoint index file.\n"
        << "  --dynamics-parameters-path PATH  Directory containing teacher dynamics JSON files.\n"
        << "  --run-path PATH                  Output run directory.\n"
        << "  --help                           Show this message.\n";
}

inline bool parse_non_negative_ti(const std::string& option_name, const char* raw, TI& target){
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

inline bool parse_float_option(const std::string& option_name, const char* raw, T& target){
    try{
        target = static_cast<T>(std::stod(raw));
        return true;
    }
    catch(const std::exception&){
        std::cerr << "Invalid float for " << option_name << ": " << raw << "\n";
        return false;
    }
}

inline bool parse_post_training_options(int argc, char** argv, PostTrainingRuntimeOptions& options){
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
            print_post_training_usage();
            return false;
        }
        else if(arg == "--seed"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.seed)) return false;
        }
        else if(arg == "--init-actor-path"){
            if(!need_value(arg)) return false;
            options.init_actor_path = argv[++arg_i];
        }
        else if(arg == "--num-teachers" || arg == "--teacher-count"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.num_teachers)) return false;
        }
        else if(arg == "--num-active-teachers"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.num_active_teachers)) return false;
        }
        else if(arg == "--epochs"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.num_epochs)) return false;
        }
        else if(arg == "--teacher-forcing-epochs"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.teacher_forcing_epochs)) return false;
        }
        else if(arg == "--max-batches-per-epoch"){
            if(!need_value(arg) || !parse_non_negative_ti(arg, argv[++arg_i], options.max_batches_per_epoch)) return false;
        }
        else if(arg == "--min-teacher-return"){
            if(!need_value(arg) || !parse_float_option(arg, argv[++arg_i], options.min_teacher_return)) return false;
        }
        else if(arg == "--teacher-experiment"){
            if(!need_value(arg)) return false;
            options.teacher_experiment = argv[++arg_i];
        }
        else if(arg == "--teacher-search-root"){
            if(!need_value(arg)) return false;
            options.teacher_search_root = argv[++arg_i];
        }
        else if(arg == "--teacher-index-path"){
            if(!need_value(arg)) return false;
            options.dynamics_parameter_index = argv[++arg_i];
        }
        else if(arg == "--dynamics-parameters-path"){
            if(!need_value(arg)) return false;
            options.dynamics_parameters_path = argv[++arg_i];
        }
        else if(arg == "--run-path"){
            if(!need_value(arg)) return false;
            options.run_path = argv[++arg_i];
        }
        else{
            std::cerr << "Unknown option: " << arg << "\n";
            print_post_training_usage();
            return false;
        }
    }
    if(options.num_teachers > NUM_TEACHERS){
        std::cerr << "--num-teachers exceeds compiled capacity " << NUM_TEACHERS << "\n";
        return false;
    }
    options.num_active_teachers = std::min(options.num_active_teachers, options.num_teachers);
    return true;
}
