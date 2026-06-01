#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_DIFF_ACTOR_UPDATE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_DIFF_ACTOR_UPDATE_H

#include <type_traits>

#include <rl_tools/rl/environments/l2f/diff_rollout_loss.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::sac{
    namespace detail{
        template<typename PARAMETERS, typename = void>
        struct has_differentiable_physics_actor_loss: std::false_type{};

        template<typename PARAMETERS>
        struct has_differentiable_physics_actor_loss<PARAMETERS, std::void_t<decltype(PARAMETERS::DIFFERENTIABLE_PHYSICS_ACTOR_LOSS)>>: std::true_type{};
    }

    template<typename DEVICE, typename T_CONFIG>
    void train_actor_differentiable_physics(DEVICE& device, rl::algorithms::sac::loop::core::State<T_CONFIG>& ts){
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using SAC_PARAMETERS = typename CONFIG::CORE_PARAMETERS::SAC_PARAMETERS;
        if constexpr(detail::has_differentiable_physics_actor_loss<SAC_PARAMETERS>::value){
            if constexpr(SAC_PARAMETERS::DIFFERENTIABLE_PHYSICS_ACTOR_LOSS){
                static_assert(CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::SEQUENCE_LENGTH == 1, "Differentiable L2F physics update currently expects SAC sequence length 1");
                constexpr TI BATCH_SIZE = CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::ACTOR_BATCH_SIZE;
                constexpr TI N_ENVIRONMENTS = CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS;
                constexpr TI OBSERVATION_DIM = CONFIG::ENVIRONMENT::Observation::DIM;
                constexpr TI ACTION_DIM = CONFIG::ENVIRONMENT::ACTION_DIM;
                static_assert(ACTION_DIM == 4, "L2F differentiable physics update expects four rotor commands");

                zero_gradient(device, ts.actor_critic.actor);
                set_all(device, ts.action_noise_actor, (T)0);
                set_all(device, ts.actor_batch.reset, false);

                for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                    const TI env_i = batch_i % N_ENVIRONMENTS;
                    for(TI observation_i = 0; observation_i < OBSERVATION_DIM; observation_i++){
                        const T observation_value = get(ts.off_policy_runner.buffers.next_observations, env_i, observation_i);
                        set(device, ts.actor_batch.observations_current, observation_value, 0, batch_i, observation_i);
                    }
                }

                auto& actor_buffers = ts.actor_buffers[1];
                auto& sample_and_squashing_buffer = get_last_buffer(actor_buffers);
                copy(device, device, ts.action_noise_actor, sample_and_squashing_buffer.noise);

                using SAMPLE_AND_SQUASH_MODE = nn::layers::sample_and_squash::mode::ExternalNoise<mode::Default<>>;
                using RESET_MODE_SAS_SPEC = nn::layers::gru::ResetModeSpecification<TI, decltype(ts.actor_batch.reset)>;
                using RESET_MODE_SAS = nn::layers::gru::ResetMode<SAMPLE_AND_SQUASH_MODE, RESET_MODE_SAS_SPEC>;
                Mode<RESET_MODE_SAS> reset_mode_sas;
                reset_mode_sas.reset_container = ts.actor_batch.reset;

                forward(device, ts.actor_critic.actor, ts.actor_batch.observations_current, ts.actor_training_buffers.actions, actor_buffers, ts.rng, reset_mode_sas);
                set_all(device, ts.actor_training_buffers.d_actor_output_squashing, (T)0);

                for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                    const TI env_i = batch_i % N_ENVIRONMENTS;
                    const auto& state = get(ts.off_policy_runner.states, 0, env_i);
                    const auto& parameters = get(ts.off_policy_runner.env_parameters, 0, env_i);
                    T action_normalized[4];
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        action_normalized[action_i] = get(device, ts.actor_training_buffers.actions, 0, batch_i, action_i);
                    }
                    T action_gradient[4];
                    rl::environments::l2f::differentiable_rollout_action_gradient<
                        DEVICE,
                        typename CONFIG::ENVIRONMENT::Parameters,
                        typename CONFIG::ENVIRONMENT::State,
                        T,
                        TI,
                        SAC_PARAMETERS::DIFFERENTIABLE_PHYSICS_HORIZON
                    >(
                        device,
                        parameters,
                        state,
                        action_normalized,
                        action_gradient,
                        SAC_PARAMETERS::DIFFERENTIABLE_PHYSICS_POSITION_WEIGHT,
                        SAC_PARAMETERS::DIFFERENTIABLE_PHYSICS_VELOCITY_WEIGHT,
                        SAC_PARAMETERS::DIFFERENTIABLE_PHYSICS_ORIENTATION_WEIGHT,
                        SAC_PARAMETERS::DIFFERENTIABLE_PHYSICS_ANGULAR_VELOCITY_WEIGHT,
                        SAC_PARAMETERS::DIFFERENTIABLE_PHYSICS_ACTION_WEIGHT,
                        SAC_PARAMETERS::DIFFERENTIABLE_PHYSICS_SATURATION_WEIGHT
                    );
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        const T scaled_gradient = SAC_PARAMETERS::DIFFERENTIABLE_PHYSICS_WEIGHT * action_gradient[action_i] / (T)BATCH_SIZE;
                        set(device, ts.actor_training_buffers.d_actor_output_squashing, scaled_gradient, 0, batch_i, action_i);
                    }
                }

                backward(device, ts.actor_critic.actor, ts.actor_batch.observations_current, ts.actor_training_buffers.d_actor_output_squashing, actor_buffers, reset_mode_sas);
                step(device, ts.actor_critic.actor_optimizer, ts.actor_critic.actor);
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
