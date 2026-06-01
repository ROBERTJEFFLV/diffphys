#pragma once

#include "config.h"

#include <rl_tools/containers/matrix/matrix.h>
#include <rl_tools/containers/tensor/tensor.h>
#include <rl_tools/operations/cpu_mux.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace rl_tools::foundation_policy::diff_pre_training{
    namespace rlt = rl_tools;

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
    bool save_actor_checkpoint(DEVICE& device, ACTOR& actor, const std::string& path){
        try{
            std::ofstream out(path);
            out << "foundation_policy_diff_pre_training_actor_v1\n";
            save_matrix_text(out, actor.content.weights.parameters, "input_dense_weights");
            save_matrix_text(out, actor.content.biases.parameters, "input_dense_biases");
            save_tensor_text(device, out, actor.next_module.content.weights_input.parameters, "gru_weights_input");
            save_tensor_text(device, out, actor.next_module.content.biases_input.parameters, "gru_biases_input");
            save_tensor_text(device, out, actor.next_module.content.weights_hidden.parameters, "gru_weights_hidden");
            save_tensor_text(device, out, actor.next_module.content.biases_hidden.parameters, "gru_biases_hidden");
            save_tensor_text(device, out, actor.next_module.content.initial_hidden_state.parameters, "gru_initial_hidden_state");
            save_matrix_text(out, actor.next_module.next_module.content.weights.parameters, "output_dense_weights");
            save_matrix_text(out, actor.next_module.next_module.content.biases.parameters, "output_dense_biases");
            return true;
        }
        catch(const std::exception& e){
            std::cerr << "Failed to save actor checkpoint to " << path << ": " << e.what() << "\n";
            return false;
        }
    }

    template <typename DEVICE, typename ACTOR>
    bool load_actor_checkpoint(DEVICE& device, ACTOR& actor, const std::string& path){
        try{
            std::ifstream in(path);
            std::string magic;
            in >> magic;
            if(magic != "foundation_policy_diff_pre_training_actor_v1"){
                std::cerr << "Unsupported checkpoint type: " << magic << "\n";
                return false;
            }
            bool ok = true;
            ok = load_matrix_text(in, actor.content.weights.parameters, "input_dense_weights") && ok;
            ok = load_matrix_text(in, actor.content.biases.parameters, "input_dense_biases") && ok;
            ok = load_tensor_text(device, in, actor.next_module.content.weights_input.parameters, "gru_weights_input") && ok;
            ok = load_tensor_text(device, in, actor.next_module.content.biases_input.parameters, "gru_biases_input") && ok;
            ok = load_tensor_text(device, in, actor.next_module.content.weights_hidden.parameters, "gru_weights_hidden") && ok;
            ok = load_tensor_text(device, in, actor.next_module.content.biases_hidden.parameters, "gru_biases_hidden") && ok;
            ok = load_tensor_text(device, in, actor.next_module.content.initial_hidden_state.parameters, "gru_initial_hidden_state") && ok;
            ok = load_matrix_text(in, actor.next_module.next_module.content.weights.parameters, "output_dense_weights") && ok;
            ok = load_matrix_text(in, actor.next_module.next_module.content.biases.parameters, "output_dense_biases") && ok;
            return ok;
        }
        catch(const std::exception& e){
            std::cerr << "Failed to load actor checkpoint from " << path << ": " << e.what() << "\n";
            return false;
        }
    }
}
