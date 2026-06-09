#pragma once

#include "config.h"

namespace rl_tools::foundation_policy::diff_pre_training{
    namespace rlt = rl_tools;

    template <typename T>
    struct RDACBackwardTerms{
        T actor_critic_actor = 0;
        T actor_critic_critic = 0;
    };

    template <typename T>
    struct RDACGradientDiagnostics{
        T encoder = 0;
        T gru = 0;
        T actor_head = 0;
        T critic_head = 0;
    };

    template <typename DEVICE, typename CAPABILITY>
    void rdac_malloc(DEVICE& device, RDACActor<CAPABILITY>& model){
        rlt::malloc(device, model.trunk);
        rlt::malloc(device, model.actor_head);
        rlt::malloc(device, model.critic_head);
    }

    template <typename DEVICE, typename CAPABILITY>
    void rdac_free(DEVICE& device, RDACActor<CAPABILITY>& model){
        rlt::free(device, model.critic_head);
        rlt::free(device, model.actor_head);
        rlt::free(device, model.trunk);
    }

    template <typename DEVICE, typename CAPABILITY, bool DYNAMIC>
    void rdac_malloc_buffer(DEVICE& device, typename RDACActor<CAPABILITY>::template Buffer<DYNAMIC>& buffer){
        rlt::malloc(device, buffer.trunk);
        rlt::malloc(device, buffer.actor_head);
        rlt::malloc(device, buffer.critic_head);
        rlt::malloc(device, buffer.hidden);
        rlt::malloc(device, buffer.action);
        rlt::malloc(device, buffer.actor_input);
        rlt::malloc(device, buffer.critic_input);
        rlt::malloc(device, buffer.q);
        rlt::malloc(device, buffer.d_hidden);
        rlt::malloc(device, buffer.d_action_total);
        rlt::malloc(device, buffer.d_actor_input);
        rlt::malloc(device, buffer.d_critic_input);
        rlt::malloc(device, buffer.d_q_critic);
        rlt::malloc(device, buffer.step_hidden);
        rlt::malloc(device, buffer.step_actor_input);
        rlt::malloc(device, buffer.step_critic_input);
        rlt::malloc(device, buffer.step_q);
    }

    template <typename DEVICE, typename CAPABILITY, bool DYNAMIC>
    void rdac_free_buffer(DEVICE& device, typename RDACActor<CAPABILITY>::template Buffer<DYNAMIC>& buffer){
        rlt::free(device, buffer.step_q);
        rlt::free(device, buffer.step_critic_input);
        rlt::free(device, buffer.step_actor_input);
        rlt::free(device, buffer.step_hidden);
        rlt::free(device, buffer.d_q_critic);
        rlt::free(device, buffer.d_critic_input);
        rlt::free(device, buffer.d_actor_input);
        rlt::free(device, buffer.d_action_total);
        rlt::free(device, buffer.d_hidden);
        rlt::free(device, buffer.q);
        rlt::free(device, buffer.critic_input);
        rlt::free(device, buffer.actor_input);
        rlt::free(device, buffer.action);
        rlt::free(device, buffer.hidden);
        rlt::free(device, buffer.critic_head);
        rlt::free(device, buffer.actor_head);
        rlt::free(device, buffer.trunk);
    }

    template <typename DEVICE, typename CAPABILITY, typename RNG>
    void rdac_init_weights(DEVICE& device, RDACActor<CAPABILITY>& model, RNG& rng){
        rlt::init_weights(device, model.trunk, rng);
        rlt::init_weights(device, model.actor_head, rng);
        rlt::init_weights(device, model.critic_head, rng);
    }

    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_CAPABILITY, typename TARGET_CAPABILITY>
    void rdac_copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const RDACActor<SOURCE_CAPABILITY>& source, RDACActor<TARGET_CAPABILITY>& target){
        rlt::copy(source_device, target_device, source.trunk, target.trunk);
        rlt::copy(source_device, target_device, source.actor_head, target.actor_head);
        rlt::copy(source_device, target_device, source.critic_head, target.critic_head);
    }

    template <typename DEVICE, typename CAPABILITY>
    void rdac_zero_gradient(DEVICE& device, RDACActor<CAPABILITY>& model){
        rlt::zero_gradient(device, model.trunk);
        rlt::zero_gradient(device, model.actor_head);
        rlt::zero_gradient(device, model.critic_head);
    }

    template <typename DEVICE, typename CAPABILITY, typename OPTIMIZER>
    void rdac_reset_optimizer_state(DEVICE& device, OPTIMIZER& optimizer, RDACActor<CAPABILITY>& model){
        rlt::set(device, optimizer.age, 1, 0);
        rlt::_reset_optimizer_state(device, model.trunk, optimizer);
        rlt::_reset_optimizer_state(device, model.actor_head, optimizer);
        rlt::_reset_optimizer_state(device, model.critic_head, optimizer);
    }

    template <typename DEVICE, typename CAPABILITY, typename OPTIMIZER>
    void rdac_step(DEVICE& device, OPTIMIZER& optimizer, RDACActor<CAPABILITY>& model){
        rlt::_step(device, optimizer);
        rlt::update(device, model.trunk, optimizer);
        rlt::update(device, model.actor_head, optimizer);
        rlt::update(device, model.critic_head, optimizer);
    }

    template <typename DEVICE, typename CAPABILITY>
    auto rdac_gradient_norm(DEVICE& device, const RDACActor<CAPABILITY>& model){
        using TT = T;
        TT value = 0;
        value += rlt::gradient_norm(device, model.trunk, false);
        value += rlt::gradient_norm(device, model.actor_head);
        value += rlt::gradient_norm(device, model.critic_head);
        return rlt::math::sqrt(device.math, value);
    }

    template <typename DEVICE, typename CAPABILITY>
    RDACGradientDiagnostics<T> rdac_gradient_diagnostics(DEVICE& device, const RDACActor<CAPABILITY>& model){
        RDACGradientDiagnostics<T> result;
        result.encoder = rlt::math::sqrt(device.math, rlt::gradient_norm(device, model.trunk.content));
        result.gru = rlt::math::sqrt(device.math, rlt::gradient_norm(device, model.trunk.next_module.content));
        result.actor_head = rlt::math::sqrt(device.math, rlt::gradient_norm(device, model.actor_head));
        result.critic_head = rlt::math::sqrt(device.math, rlt::gradient_norm(device, model.critic_head));
        return result;
    }

    template <typename DEVICE, typename TENSOR, typename VALUE>
    void add_to_tensor3(DEVICE& device, TENSOR& tensor, VALUE value, typename DEVICE::index_t i, typename DEVICE::index_t j, typename DEVICE::index_t k){
        rlt::set(device, tensor, rlt::get(device, tensor, i, j, k) + value, i, j, k);
    }

    template <typename DEVICE, typename POLICY_INPUT, typename HIDDEN, typename HEAD_INPUT>
    void assemble_head_input_sequence(DEVICE& device, const POLICY_INPUT& policy_input, const HIDDEN& hidden, HEAD_INPUT& head_input){
        using TI = typename DEVICE::index_t;
        for(TI t = 0; t < DIFF_TRAINING_SEQUENCE_LENGTH; t++){
            for(TI b = 0; b < DIFF_TRAINING_BATCH_SIZE; b++){
                for(TI o = 0; o < ENVIRONMENT::Observation::DIM; o++){
                    rlt::set(device, head_input, rlt::get(device, policy_input, t, b, o), t, b, o);
                }
                for(TI h = 0; h < HIDDEN_DIM; h++){
                    rlt::set(device, head_input, rlt::get(device, hidden, t, b, h), t, b, ENVIRONMENT::Observation::DIM + h);
                }
            }
        }
    }

    template <typename DEVICE, typename POLICY_INPUT, typename HIDDEN, typename HEAD_INPUT>
    void assemble_head_input_step(DEVICE& device, const POLICY_INPUT& policy_input, const HIDDEN& hidden, HEAD_INPUT& head_input){
        using TI = typename DEVICE::index_t;
        for(TI b = 0; b < DIFF_TRAINING_BATCH_SIZE; b++){
            for(TI o = 0; o < ENVIRONMENT::Observation::DIM; o++){
                rlt::set(device, head_input, rlt::get(device, policy_input, b, o), b, o);
            }
            for(TI h = 0; h < HIDDEN_DIM; h++){
                rlt::set(device, head_input, rlt::get(device, hidden, b, h), b, ENVIRONMENT::Observation::DIM + h);
            }
        }
    }

    template <typename DEVICE, typename CAPABILITY, typename INPUT, typename STATE, typename OUTPUT, typename BUFFER, typename RNG, typename MODE>
    void rdac_evaluate_step(DEVICE& device, const RDACActor<CAPABILITY>& model, const INPUT& input, STATE& state, OUTPUT& output, BUFFER& buffer, RNG& rng, const rlt::Mode<MODE>& mode){
        rlt::evaluate_step(device, model.trunk, input, state, buffer.step_hidden, buffer.trunk, rng, mode);
        assemble_head_input_step(device, input, buffer.step_hidden, buffer.step_actor_input);
        rlt::evaluate(device, model.actor_head, buffer.step_actor_input, output, buffer.actor_head, rng, mode);
        assemble_head_input_step(device, input, buffer.step_hidden, buffer.step_critic_input);
        rlt::evaluate(device, model.critic_head, buffer.step_critic_input, buffer.step_q, buffer.critic_head, rng, mode);
    }

    template <typename DEVICE, typename CAPABILITY, typename INPUT, typename BUFFER, typename RNG, typename MODE>
    void rdac_forward(DEVICE& device, RDACActor<CAPABILITY>& model, INPUT& input, BUFFER& buffer, RNG& rng, const rlt::Mode<MODE>& mode){
        rlt::forward(device, model.trunk, input, buffer.trunk, rng, mode);
        auto hidden = rlt::output(device, model.trunk);
        rlt::copy(device, device, hidden, buffer.hidden);
        assemble_head_input_sequence(device, input, buffer.hidden, buffer.actor_input);
        rlt::forward(device, model.actor_head, buffer.actor_input, buffer.action, buffer.actor_head, rng, mode);
        assemble_head_input_sequence(device, input, buffer.hidden, buffer.critic_input);
        rlt::forward(device, model.critic_head, buffer.critic_input, buffer.q, buffer.critic_head, rng, mode);
    }

    template <typename DEVICE, typename CAPABILITY, typename INPUT, typename D_ACTION, typename Q_TARGET, typename Q_WEIGHT, typename BUFFER, typename MODE>
    RDACBackwardTerms<T> rdac_backward(
        DEVICE& device,
        RDACActor<CAPABILITY>& model,
        INPUT& input,
        D_ACTION& d_action_diff,
        Q_TARGET& q_target,
        Q_WEIGHT& q_weight,
        BUFFER& buffer,
        const rlt::Mode<MODE>& mode
    ){
        using TT = T;
        using TI = typename DEVICE::index_t;
        RDACBackwardTerms<TT> terms;
        rlt::set_all(device, buffer.d_hidden, (TT)0);
        rlt::set_all(device, buffer.d_action_total, (TT)0);
        rlt::set_all(device, buffer.d_actor_input, (TT)0);
        rlt::set_all(device, buffer.d_critic_input, (TT)0);
        rlt::set_all(device, buffer.d_q_critic, (TT)0);

        for(TI t = 0; t < DIFF_TRAINING_SEQUENCE_LENGTH; t++){
            for(TI b = 0; b < DIFF_TRAINING_BATCH_SIZE; b++){
                const TT weight = rlt::get(device, q_weight, t, b, (TI)0);
                const TT q = rlt::get(device, buffer.q, t, b, (TI)0);
                const TT target = rlt::get(device, q_target, t, b, (TI)0);
                const TT error = q - target;
                terms.actor_critic_critic += (TT)0.5 * LOSS_AC_CRITIC_WEIGHT * weight * error * error;
                rlt::set(device, buffer.d_q_critic, LOSS_AC_CRITIC_WEIGHT * weight * error, t, b, (TI)0);
            }
        }

        rlt::backward_full(device, model.critic_head, buffer.critic_input, buffer.d_q_critic, buffer.d_critic_input, buffer.critic_head, mode);
        for(TI t = 0; t < DIFF_TRAINING_SEQUENCE_LENGTH; t++){
            for(TI b = 0; b < DIFF_TRAINING_BATCH_SIZE; b++){
                for(TI h = 0; h < HIDDEN_DIM; h++){
                    add_to_tensor3(device, buffer.d_hidden, rlt::get(device, buffer.d_critic_input, t, b, ENVIRONMENT::Observation::DIM + h), t, b, h);
                }
                for(TI a = 0; a < ENVIRONMENT::ACTION_DIM; a++){
                    rlt::set(device, buffer.d_action_total, rlt::get(device, d_action_diff, t, b, a), t, b, a);
                }
            }
        }

        rlt::backward_full(device, model.actor_head, buffer.actor_input, buffer.d_action_total, buffer.d_actor_input, buffer.actor_head, mode);
        for(TI t = 0; t < DIFF_TRAINING_SEQUENCE_LENGTH; t++){
            for(TI b = 0; b < DIFF_TRAINING_BATCH_SIZE; b++){
                for(TI h = 0; h < HIDDEN_DIM; h++){
                    add_to_tensor3(device, buffer.d_hidden, rlt::get(device, buffer.d_actor_input, t, b, ENVIRONMENT::Observation::DIM + h), t, b, h);
                }
            }
        }

        rlt::backward(device, model.trunk, input, buffer.d_hidden, buffer.trunk, mode);
        return terms;
    }
}
