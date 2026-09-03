#pragma once

#include "config.h"

#include <rl_tools/containers/matrix/matrix.h>
#include <rl_tools/containers/tensor/tensor.h>
#include <rl_tools/operations/cpu_mux.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace rl_tools::foundation_policy::diff_pre_training{
    namespace rlt = rl_tools;

    struct CheckpointMetadata{
        long long format_version = 3;
        long long optimizer_age = 1;
        bool has_optimizer = false;
    };

    inline void save_checkpoint_metadata_text(std::ostream& out, const CheckpointMetadata& metadata){
        out << "metadata 9\n";
        out << "format_version " << metadata.format_version << "\n";
        out << "has_optimizer " << (metadata.has_optimizer ? 1 : 0) << "\n";
        out << "optimizer_age " << metadata.optimizer_age << "\n";
        out << "scalar_type float32\n";
        out << "actor_arch rdac_hidden_gru\n";
        out << "input_dim " << POLICY_INPUT_DIM << "\n";
        out << "hidden_dim " << HIDDEN_DIM << "\n";
        out << "action_dim " << ENVIRONMENT::ACTION_DIM << "\n";
        out << "critic_dim " << CRITIC_DIM << "\n";
        out << "end_metadata\n";
    }

    inline bool load_checkpoint_metadata_text(std::istream& in, CheckpointMetadata& metadata){
        std::string label;
        long long count = 0;
        if(!(in >> label >> count) || label != "metadata" || count < 0){
            std::cerr << "Checkpoint metadata header mismatch\n";
            return false;
        }
        for(long long i = 0; i < count; i++){
            std::string key;
            std::string value;
            if(!(in >> key >> value)){
                std::cerr << "Checkpoint metadata entry mismatch\n";
                return false;
            }
            if(key == "format_version"){
                metadata.format_version = std::stoll(value);
            }
            else if(key == "has_optimizer"){
                metadata.has_optimizer = std::stoll(value) != 0;
            }
            else if(key == "optimizer_age"){
                metadata.optimizer_age = std::stoll(value);
            }
        }
        std::string end_label;
        if(!(in >> end_label) || end_label != "end_metadata"){
            std::cerr << "Checkpoint metadata terminator mismatch\n";
            return false;
        }
        return true;
    }

    template <typename MAT>
    void save_matrix_text(std::ostream& out, const MAT& matrix, const std::string& name){
        out << name << " " << MAT::ROWS << " " << MAT::COLS << "\n";
        out << std::setprecision(10);
        for(typename MAT::TI row_i = 0; row_i < MAT::ROWS; row_i++){
            for(typename MAT::TI col_i = 0; col_i < MAT::COLS; col_i++){
                out << rlt::get(matrix, row_i, col_i) << "\n";
            }
        }
    }

    template <typename MAT>
    bool load_matrix_text(std::istream& in, MAT& matrix, const std::string& expected_name){
        std::string name;
        typename MAT::TI rows, cols;
        if(!(in >> name >> rows >> cols)){
            return false;
        }
        if(name != expected_name || rows != MAT::ROWS || cols != MAT::COLS){
            std::cerr << "Checkpoint matrix mismatch for " << expected_name << ": got " << name << " " << rows << "x" << cols << "\n";
            return false;
        }
        for(typename MAT::TI row_i = 0; row_i < MAT::ROWS; row_i++){
            for(typename MAT::TI col_i = 0; col_i < MAT::COLS; col_i++){
                typename MAT::T value;
                in >> value;
                rlt::set(matrix, row_i, col_i, value);
            }
        }
        return true;
    }

    template <typename DEVICE, typename TENSOR>
    void save_tensor_text(DEVICE& device, std::ostream& out, const TENSOR& tensor, const std::string& name){
        out << name << " " << TENSOR::SPEC::SIZE << "\n";
        out << std::setprecision(10);
        for(typename DEVICE::index_t i = 0; i < TENSOR::SPEC::SIZE; i++){
            out << rlt::get_flat(device, tensor, i) << "\n";
        }
    }

    template <typename DEVICE, typename TENSOR>
    bool load_tensor_text(DEVICE& device, std::istream& in, TENSOR& tensor, const std::string& expected_name){
        std::string name;
        typename DEVICE::index_t size;
        if(!(in >> name >> size)){
            return false;
        }
        if(name != expected_name || size != TENSOR::SPEC::SIZE){
            std::cerr << "Checkpoint tensor mismatch for " << expected_name << ": got " << name << " size=" << size << "\n";
            return false;
        }
        for(typename DEVICE::index_t i = 0; i < TENSOR::SPEC::SIZE; i++){
            typename TENSOR::T value;
            in >> value;
            rlt::data(tensor)[i] = value;
        }
        return true;
    }

    template <typename DEVICE, typename ACTOR>
    void save_actor_weights_text(DEVICE& device, std::ostream& out, ACTOR& actor){
        save_matrix_text(out, actor.trunk.content.weights.parameters, "input_dense_weights");
        save_matrix_text(out, actor.trunk.content.biases.parameters, "input_dense_biases");
        save_tensor_text(device, out, actor.trunk.next_module.content.weights_input.parameters, "gru_weights_input");
        save_tensor_text(device, out, actor.trunk.next_module.content.biases_input.parameters, "gru_biases_input");
        save_tensor_text(device, out, actor.trunk.next_module.content.weights_hidden.parameters, "gru_weights_hidden");
        save_tensor_text(device, out, actor.trunk.next_module.content.biases_hidden.parameters, "gru_biases_hidden");
        save_tensor_text(device, out, actor.trunk.next_module.content.initial_hidden_state.parameters, "gru_initial_hidden_state");
        save_matrix_text(out, actor.actor_head.weights.parameters, "actor_head_weights");
        save_matrix_text(out, actor.actor_head.biases.parameters, "actor_head_biases");
        save_matrix_text(out, actor.critic_head.weights.parameters, "critic_head_weights");
        save_matrix_text(out, actor.critic_head.biases.parameters, "critic_head_biases");
    }

    template <typename DEVICE, typename ACTOR>
    bool load_actor_weights_text_v3(DEVICE& device, std::istream& in, ACTOR& actor){
        bool ok = true;
        ok = load_matrix_text(in, actor.trunk.content.weights.parameters, "input_dense_weights") && ok;
        ok = load_matrix_text(in, actor.trunk.content.biases.parameters, "input_dense_biases") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.weights_input.parameters, "gru_weights_input") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.biases_input.parameters, "gru_biases_input") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.weights_hidden.parameters, "gru_weights_hidden") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.biases_hidden.parameters, "gru_biases_hidden") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.initial_hidden_state.parameters, "gru_initial_hidden_state") && ok;
        ok = load_matrix_text(in, actor.actor_head.weights.parameters, "actor_head_weights") && ok;
        ok = load_matrix_text(in, actor.actor_head.biases.parameters, "actor_head_biases") && ok;
        ok = load_matrix_text(in, actor.critic_head.weights.parameters, "critic_head_weights") && ok;
        ok = load_matrix_text(in, actor.critic_head.biases.parameters, "critic_head_biases") && ok;
        return ok;
    }

    template <typename DEVICE, typename ACTOR>
    void save_actor_adam_moments_text(DEVICE& device, std::ostream& out, ACTOR& actor){
        save_matrix_text(out, actor.trunk.content.weights.gradient_first_order_moment, "input_dense_weights_adam_m");
        save_matrix_text(out, actor.trunk.content.weights.gradient_second_order_moment, "input_dense_weights_adam_v");
        save_matrix_text(out, actor.trunk.content.biases.gradient_first_order_moment, "input_dense_biases_adam_m");
        save_matrix_text(out, actor.trunk.content.biases.gradient_second_order_moment, "input_dense_biases_adam_v");
        save_tensor_text(device, out, actor.trunk.next_module.content.weights_input.gradient_first_order_moment, "gru_weights_input_adam_m");
        save_tensor_text(device, out, actor.trunk.next_module.content.weights_input.gradient_second_order_moment, "gru_weights_input_adam_v");
        save_tensor_text(device, out, actor.trunk.next_module.content.biases_input.gradient_first_order_moment, "gru_biases_input_adam_m");
        save_tensor_text(device, out, actor.trunk.next_module.content.biases_input.gradient_second_order_moment, "gru_biases_input_adam_v");
        save_tensor_text(device, out, actor.trunk.next_module.content.weights_hidden.gradient_first_order_moment, "gru_weights_hidden_adam_m");
        save_tensor_text(device, out, actor.trunk.next_module.content.weights_hidden.gradient_second_order_moment, "gru_weights_hidden_adam_v");
        save_tensor_text(device, out, actor.trunk.next_module.content.biases_hidden.gradient_first_order_moment, "gru_biases_hidden_adam_m");
        save_tensor_text(device, out, actor.trunk.next_module.content.biases_hidden.gradient_second_order_moment, "gru_biases_hidden_adam_v");
        save_tensor_text(device, out, actor.trunk.next_module.content.initial_hidden_state.gradient_first_order_moment, "gru_initial_hidden_state_adam_m");
        save_tensor_text(device, out, actor.trunk.next_module.content.initial_hidden_state.gradient_second_order_moment, "gru_initial_hidden_state_adam_v");
        save_matrix_text(out, actor.actor_head.weights.gradient_first_order_moment, "actor_head_weights_adam_m");
        save_matrix_text(out, actor.actor_head.weights.gradient_second_order_moment, "actor_head_weights_adam_v");
        save_matrix_text(out, actor.actor_head.biases.gradient_first_order_moment, "actor_head_biases_adam_m");
        save_matrix_text(out, actor.actor_head.biases.gradient_second_order_moment, "actor_head_biases_adam_v");
        save_matrix_text(out, actor.critic_head.weights.gradient_first_order_moment, "critic_head_weights_adam_m");
        save_matrix_text(out, actor.critic_head.weights.gradient_second_order_moment, "critic_head_weights_adam_v");
        save_matrix_text(out, actor.critic_head.biases.gradient_first_order_moment, "critic_head_biases_adam_m");
        save_matrix_text(out, actor.critic_head.biases.gradient_second_order_moment, "critic_head_biases_adam_v");
    }

    template <typename DEVICE, typename ACTOR>
    bool load_actor_adam_moments_text(DEVICE& device, std::istream& in, ACTOR& actor){
        bool ok = true;
        ok = load_matrix_text(in, actor.trunk.content.weights.gradient_first_order_moment, "input_dense_weights_adam_m") && ok;
        ok = load_matrix_text(in, actor.trunk.content.weights.gradient_second_order_moment, "input_dense_weights_adam_v") && ok;
        ok = load_matrix_text(in, actor.trunk.content.biases.gradient_first_order_moment, "input_dense_biases_adam_m") && ok;
        ok = load_matrix_text(in, actor.trunk.content.biases.gradient_second_order_moment, "input_dense_biases_adam_v") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.weights_input.gradient_first_order_moment, "gru_weights_input_adam_m") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.weights_input.gradient_second_order_moment, "gru_weights_input_adam_v") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.biases_input.gradient_first_order_moment, "gru_biases_input_adam_m") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.biases_input.gradient_second_order_moment, "gru_biases_input_adam_v") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.weights_hidden.gradient_first_order_moment, "gru_weights_hidden_adam_m") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.weights_hidden.gradient_second_order_moment, "gru_weights_hidden_adam_v") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.biases_hidden.gradient_first_order_moment, "gru_biases_hidden_adam_m") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.biases_hidden.gradient_second_order_moment, "gru_biases_hidden_adam_v") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.initial_hidden_state.gradient_first_order_moment, "gru_initial_hidden_state_adam_m") && ok;
        ok = load_tensor_text(device, in, actor.trunk.next_module.content.initial_hidden_state.gradient_second_order_moment, "gru_initial_hidden_state_adam_v") && ok;
        ok = load_matrix_text(in, actor.actor_head.weights.gradient_first_order_moment, "actor_head_weights_adam_m") && ok;
        ok = load_matrix_text(in, actor.actor_head.weights.gradient_second_order_moment, "actor_head_weights_adam_v") && ok;
        ok = load_matrix_text(in, actor.actor_head.biases.gradient_first_order_moment, "actor_head_biases_adam_m") && ok;
        ok = load_matrix_text(in, actor.actor_head.biases.gradient_second_order_moment, "actor_head_biases_adam_v") && ok;
        ok = load_matrix_text(in, actor.critic_head.weights.gradient_first_order_moment, "critic_head_weights_adam_m") && ok;
        ok = load_matrix_text(in, actor.critic_head.weights.gradient_second_order_moment, "critic_head_weights_adam_v") && ok;
        ok = load_matrix_text(in, actor.critic_head.biases.gradient_first_order_moment, "critic_head_biases_adam_m") && ok;
        ok = load_matrix_text(in, actor.critic_head.biases.gradient_second_order_moment, "critic_head_biases_adam_v") && ok;
        return ok;
    }

    template <typename DEVICE, typename ACTOR>
    bool save_actor_checkpoint(DEVICE& device, ACTOR& actor, const std::string& path){
        try{
            std::ofstream out(path);
            out << "foundation_policy_diff_pre_training_rdac_hidden_actor_v3\n";
            save_actor_weights_text(device, out, actor);
            return true;
        }
        catch(const std::exception& e){
            std::cerr << "Failed to save actor checkpoint to " << path << ": " << e.what() << "\n";
            return false;
        }
    }

    template <typename DEVICE, typename ACTOR, typename OPTIMIZER>
    bool save_actor_optimizer_checkpoint(DEVICE& device, ACTOR& actor, OPTIMIZER& optimizer, const std::string& path){
        try{
            std::ofstream out(path);
            out << "foundation_policy_diff_pre_training_rdac_hidden_actor_v4\n";
            CheckpointMetadata metadata;
            metadata.format_version = 4;
            metadata.has_optimizer = true;
            metadata.optimizer_age = rlt::get(device, optimizer.age, 0);
            save_checkpoint_metadata_text(out, metadata);
            save_actor_weights_text(device, out, actor);
            save_actor_adam_moments_text(device, out, actor);
            return true;
        }
        catch(const std::exception& e){
            std::cerr << "Failed to save unified actor checkpoint to " << path << ": " << e.what() << "\n";
            return false;
        }
    }

    template <typename DEVICE, typename ACTOR>
    bool load_actor_checkpoint(DEVICE& device, ACTOR& actor, const std::string& path){
        try{
            std::ifstream in(path);
            std::string magic;
            in >> magic;
            if(magic == "foundation_policy_diff_pre_training_rdac_hidden_actor_v4"){
                CheckpointMetadata metadata;
                if(!load_checkpoint_metadata_text(in, metadata)){
                    return false;
                }
                return load_actor_weights_text_v3(device, in, actor);
            }
            if(magic != "foundation_policy_diff_pre_training_rdac_hidden_actor_v3"){
                std::cerr << "Unsupported checkpoint type: " << magic << "\n";
                return false;
            }
            return load_actor_weights_text_v3(device, in, actor);
        }
        catch(const std::exception& e){
            std::cerr << "Failed to load actor checkpoint from " << path << ": " << e.what() << "\n";
            return false;
        }
    }

    template <typename DEVICE, typename ACTOR, typename OPTIMIZER>
    bool load_actor_optimizer_checkpoint(DEVICE& device, ACTOR& actor, OPTIMIZER& optimizer, const std::string& path, bool require_optimizer){
        try{
            std::ifstream in(path);
            std::string magic;
            in >> magic;
            if(magic == "foundation_policy_diff_pre_training_rdac_hidden_actor_v3"){
                if(require_optimizer){
                    std::cerr << "Checkpoint does not contain optimizer state: " << path << "\n";
                    return false;
                }
                return load_actor_weights_text_v3(device, in, actor);
            }
            if(magic != "foundation_policy_diff_pre_training_rdac_hidden_actor_v4"){
                std::cerr << "Unsupported checkpoint type: " << magic << "\n";
                return false;
            }
            CheckpointMetadata metadata;
            if(!load_checkpoint_metadata_text(in, metadata)){
                return false;
            }
            bool ok = load_actor_weights_text_v3(device, in, actor);
            if(metadata.has_optimizer){
                ok = load_actor_adam_moments_text(device, in, actor) && ok;
                rlt::set(device, optimizer.age, static_cast<typename DEVICE::index_t>(metadata.optimizer_age), 0);
            }
            else if(require_optimizer){
                std::cerr << "Checkpoint metadata says optimizer state is absent: " << path << "\n";
                return false;
            }
            return ok;
        }
        catch(const std::exception& e){
            std::cerr << "Failed to load unified actor checkpoint from " << path << ": " << e.what() << "\n";
            return false;
        }
    }

    inline bool inspect_checkpoint_file(const std::string& path){
        std::ifstream in(path);
        if(!in){
            std::cerr << "Failed to open checkpoint for inspect: " << path << "\n";
            return false;
        }
        std::string magic;
        in >> magic;
        std::cout << "checkpoint_magic=" << magic << "\n";
        if(magic == "foundation_policy_diff_pre_training_rdac_hidden_actor_v4"){
            CheckpointMetadata metadata;
            if(!load_checkpoint_metadata_text(in, metadata)){
                return false;
            }
            std::cout << "checkpoint_format_version=" << metadata.format_version
                      << " checkpoint_has_optimizer=" << (metadata.has_optimizer ? "true" : "false")
                      << " checkpoint_optimizer_age=" << metadata.optimizer_age << "\n";
        }
        return true;
    }
}
