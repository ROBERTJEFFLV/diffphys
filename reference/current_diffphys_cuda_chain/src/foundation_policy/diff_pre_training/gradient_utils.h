#pragma once

#include "config.h"
#include "rdac_operations.h"

#include <rl_tools/containers/matrix/matrix.h>
#include <rl_tools/containers/tensor/tensor.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn/layers/gru/layer.h>
#include <rl_tools/nn_models/sequential/model.h>

#include <algorithm>
#include <type_traits>

namespace rl_tools::foundation_policy::diff_pre_training{
    namespace rlt = rl_tools;

    template <typename T>
    struct ActorGradientClipResult{
        T raw_norm = 0;
        T scaled_norm = 0;
        T scale = 1;
        bool clipped = false;
    };

    template <typename DEVICE, typename SPEC>
    void scale_gradient_container(DEVICE& device, rlt::Matrix<SPEC>& matrix, typename SPEC::T factor){
        rlt::multiply_all(device, matrix, factor);
    }

    template <typename DEVICE, typename SPEC>
    void scale_gradient_container(DEVICE& device, rlt::Tensor<SPEC>& tensor, typename SPEC::T factor){
        rlt::scale(device, tensor, factor);
    }

    template <typename DEVICE, typename SPEC, typename T>
    void scale_parameter_gradient(DEVICE& device, rlt::nn::parameters::Gradient::instance<SPEC>& parameter, T factor){
        scale_gradient_container(device, parameter.gradient, (typename SPEC::CONTAINER::T)factor);
    }

    template <typename DEVICE, typename SPEC, typename T>
    void scale_layer_gradients(DEVICE& device, rlt::nn::layers::dense::LayerGradient<SPEC>& layer, T factor){
        scale_parameter_gradient(device, layer.weights, factor);
        scale_parameter_gradient(device, layer.biases, factor);
    }

    template <typename DEVICE, typename SPEC, typename T>
    void scale_layer_gradients(DEVICE& device, rlt::nn::layers::gru::LayerGradient<SPEC>& layer, T factor){
        scale_parameter_gradient(device, layer.weights_input, factor);
        scale_parameter_gradient(device, layer.biases_input, factor);
        scale_parameter_gradient(device, layer.weights_hidden, factor);
        scale_parameter_gradient(device, layer.biases_hidden, factor);
        scale_parameter_gradient(device, layer.initial_hidden_state, factor);
    }

    template <typename DEVICE, typename MODULE, typename T>
    void scale_actor_gradients(DEVICE& device, MODULE& module, T factor);

    template <typename DEVICE, typename SPEC, typename T>
    void scale_actor_gradients(DEVICE& device, rlt::nn_models::sequential::ModuleGradient<SPEC>& module, T factor){
        scale_layer_gradients(device, module.content, factor);
        if constexpr(!std::is_same_v<typename SPEC::NEXT_MODULE, rlt::nn_models::sequential::OutputModule>){
            scale_actor_gradients(device, module.next_module, factor);
        }
    }

    template <typename DEVICE, typename CAPABILITY, typename MODULE, typename INPUT_SHAPE, typename T>
    void scale_actor_gradients(DEVICE& device, rlt::nn_models::sequential::Build<CAPABILITY, MODULE, INPUT_SHAPE>& actor, T factor){
        using BASE_MODULE = typename rlt::nn_models::sequential::_Chain<CAPABILITY, MODULE, INPUT_SHAPE>::MODULE;
        scale_actor_gradients(device, static_cast<BASE_MODULE&>(actor), factor);
    }

    template <typename DEVICE, typename CAPABILITY, typename T>
    void scale_actor_gradients(DEVICE& device, RDACActor<CAPABILITY>& actor, T factor){
        scale_actor_gradients(device, actor.trunk, factor);
        scale_layer_gradients(device, actor.actor_head, factor);
        scale_layer_gradients(device, actor.critic_head, factor);
    }

    template <typename DEVICE, typename ACTOR>
    auto compute_actor_gradient_norm(DEVICE& device, ACTOR& actor){
        return rlt::gradient_norm(device, actor);
    }

    template <typename DEVICE, typename CAPABILITY>
    auto compute_actor_gradient_norm(DEVICE& device, RDACActor<CAPABILITY>& actor){
        return rdac_gradient_norm(device, actor);
    }

    template <typename DEVICE, typename ACTOR, typename T>
    ActorGradientClipResult<T> clip_actor_gradients_by_global_norm(
        DEVICE& device,
        ACTOR& actor,
        bool enabled,
        T max_norm,
        T eps
    ){
        ActorGradientClipResult<T> result;
        result.raw_norm = compute_actor_gradient_norm(device, actor);
        result.scaled_norm = result.raw_norm;
        result.scale = (T)1;
        result.clipped = false;
        if(enabled && result.raw_norm == result.raw_norm && result.raw_norm > max_norm){
            result.scale = max_norm / (result.raw_norm + std::max(eps, (T)0));
            scale_actor_gradients(device, actor, result.scale);
            result.scaled_norm = compute_actor_gradient_norm(device, actor);
            result.clipped = true;
        }
        return result;
    }
}
