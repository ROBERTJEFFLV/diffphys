#pragma once

#include "environment.h"

#include <rl_tools/nn_models/sequential/model.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn/layers/gru/layer.h>

namespace rl_tools::foundation_policy::diff_pre_training{
    namespace rlt = rl_tools;

    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    using TI = DEVICE::index_t;
    using T = float;

    static_assert(sizeof(TI) == 8);

    constexpr bool DYNAMIC_ALLOCATION = true;
    constexpr bool DIFF_TRAINING_SAMPLE_DYNAMICS = true;
    constexpr TI DIFF_TRAINING_BATCH_SIZE = 4;
    constexpr TI DIFF_TRAINING_SEQUENCE_LENGTH = 128;
    constexpr TI DIFF_TRAINING_HORIZON = DIFF_TRAINING_SEQUENCE_LENGTH;
    constexpr TI DIFF_TRAINING_NUM_STEPS = 3;
    constexpr T DIFF_TRAINING_LEARNING_RATE = 1e-4;
    constexpr TI HIDDEN_DIM = 16;
    constexpr TI CRITIC_DIM = 1;
    constexpr T LOSS_AC_ACTOR_WEIGHT = 0.05;
    constexpr T LOSS_AC_CRITIC_WEIGHT = 0.10;
    constexpr T LOSS_DIFF_ROLLOUT_WEIGHT = 5e-4;
    constexpr T LOSS_TRANSITION_CONSISTENCY_WEIGHT = 0.05;

    constexpr T LOSS_POSITION_WEIGHT = 8.0;
    constexpr T LOSS_VELOCITY_WEIGHT = 0.8;
    constexpr T LOSS_ATTITUDE_WEIGHT = 4.0;
    constexpr T LOSS_ANGULAR_VELOCITY_WEIGHT = 0.8;
    constexpr T LOSS_ACTION_MAGNITUDE_WEIGHT = 0.005;
    constexpr T LOSS_ACTION_SMOOTHNESS_WEIGHT = 0.03;
    constexpr T LOSS_SATURATION_WEIGHT = 0.05;

    constexpr bool HORIZON_CURRICULUM_ENABLED = false;
    constexpr TI HORIZON_START = 16;
    constexpr TI HORIZON_STAGE_STEPS = 1000;

    constexpr bool STATE_CURRICULUM_ENABLED = false;
    constexpr TI STATE_CURRICULUM_STAGE_STEPS = 1000;
    constexpr T STATE_CURRICULUM_POSITION_START = 0.25;
    constexpr T STATE_CURRICULUM_VELOCITY_START = 0.10;
    constexpr T STATE_CURRICULUM_ATTITUDE_START = 0.15;
    constexpr T STATE_CURRICULUM_ANGULAR_VELOCITY_START = 0.10;

    constexpr T TERMINAL_LOSS_WEIGHT = 4.0;
    constexpr T TERMINAL_POSITION_WEIGHT = 12.0;
    constexpr T TERMINAL_VELOCITY_WEIGHT = 4.0;
    constexpr T TERMINAL_ATTITUDE_WEIGHT = 8.0;
    constexpr T TERMINAL_ANGULAR_VELOCITY_WEIGHT = 4.0;

    constexpr bool ACTION_BOUND_ENABLED = true;
    constexpr T ACTION_BOUND_VALUE = 1.0;
    constexpr bool ACTION_GRAD_CLIP_ENABLED = false;
    constexpr T ACTION_GRAD_CLIP_NORM = 100.0;
    constexpr bool ACTOR_GRAD_CLIP_ENABLED = true;
    constexpr T ACTOR_GRAD_CLIP_NORM = 100.0;
    constexpr T ACTOR_GRAD_SKIP_NORM = 1e12;
    constexpr T ACTOR_GRAD_EPS = 1e-6;

    // Deprecated compatibility aliases. Do not use in new code.
    // --grad-clip is a legacy alias for action-gradient clipping.
    // Use ACTION_GRAD_CLIP_* and ACTOR_GRAD_CLIP_* explicitly instead.
    constexpr bool GRAD_CLIP_ENABLED = ACTION_GRAD_CLIP_ENABLED;
    constexpr T GRAD_CLIP_DEFAULT = ACTION_GRAD_CLIP_NORM;
    constexpr T ACTION_GRADIENT_CLIP = ACTION_GRAD_CLIP_NORM;
    constexpr T ACTOR_GRADIENT_SKIP_THRESHOLD = ACTOR_GRAD_SKIP_NORM;

    constexpr bool DYNAMICS_CURRICULUM_ENABLED = false;
    constexpr TI DYNAMICS_CURRICULUM_STAGE_STEPS = 500;
    constexpr bool BALANCED_DYNAMICS_SAMPLING_ENABLED = true;
    constexpr TI DYNAMICS_BALANCE_BINS = 4;
    constexpr T SAMPLED_DYNAMICS_MAX_MASS = 0.25;
    constexpr T SAMPLED_DYNAMICS_MIN_THRUST_TO_WEIGHT = 1.25;

    constexpr T SUCCESS_POSITION_THRESHOLD = 1.0;
    constexpr T SUCCESS_VELOCITY_THRESHOLD = 2.0;
    constexpr T SUCCESS_ANGULAR_VELOCITY_THRESHOLD = 5.0;
    constexpr T EVAL_SUCCESS_POSITION_THRESHOLD = SUCCESS_POSITION_THRESHOLD;
    constexpr T EVAL_SUCCESS_VELOCITY_THRESHOLD = SUCCESS_VELOCITY_THRESHOLD;
    constexpr T EVAL_SUCCESS_ANGULAR_VELOCITY_THRESHOLD = SUCCESS_ANGULAR_VELOCITY_THRESHOLD;

    using ENVIRONMENT = typename EnvironmentFactory<DEVICE, T, TI>::ENVIRONMENT;

    template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
    using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

    using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Input>;
    using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
    using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, HIDDEN_DIM, rlt::nn::parameters::groups::Normal>;
    using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
    constexpr TI ACTOR_OUTPUT_DIM = ENVIRONMENT::ACTION_DIM;
    constexpr TI RESPONSE_ERROR_DIM = ENVIRONMENT::Observation::DIM;
    constexpr TI POLICY_INPUT_DIM = ENVIRONMENT::Observation::DIM + ENVIRONMENT::ACTION_DIM + RESPONSE_ERROR_DIM;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, DYNAMIC_ALLOCATION>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, DIFF_TRAINING_SEQUENCE_LENGTH, DIFF_TRAINING_BATCH_SIZE, POLICY_INPUT_DIM>;
    using HIDDEN_SHAPE = rlt::tensor::Shape<TI, DIFF_TRAINING_SEQUENCE_LENGTH, DIFF_TRAINING_BATCH_SIZE, HIDDEN_DIM>;
    using ACTION_OUTPUT_SHAPE = rlt::tensor::Shape<TI, DIFF_TRAINING_SEQUENCE_LENGTH, DIFF_TRAINING_BATCH_SIZE, ENVIRONMENT::ACTION_DIM>;
    using ACTOR_HEAD_INPUT_SHAPE = rlt::tensor::Shape<TI, DIFF_TRAINING_SEQUENCE_LENGTH, DIFF_TRAINING_BATCH_SIZE, ENVIRONMENT::Observation::DIM + HIDDEN_DIM>;
    using CRITIC_HEAD_INPUT_SHAPE = rlt::tensor::Shape<TI, DIFF_TRAINING_SEQUENCE_LENGTH, DIFF_TRAINING_BATCH_SIZE, ENVIRONMENT::Observation::DIM + HIDDEN_DIM>;
    using CRITIC_OUTPUT_SHAPE = rlt::tensor::Shape<TI, DIFF_TRAINING_SEQUENCE_LENGTH, DIFF_TRAINING_BATCH_SIZE, CRITIC_DIM>;

    using ACTOR_HEAD_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Output>;
    using CRITIC_HEAD_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, CRITIC_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Output>;

    using TRUNK_MODULE_CHAIN = Module<INPUT_LAYER, Module<GRU>>;

    template <typename T_CAPABILITY>
    struct RDACActor{
        using CAPABILITY_TYPE = T_CAPABILITY;
        using TRUNK = rlt::nn_models::sequential::Build<T_CAPABILITY, TRUNK_MODULE_CHAIN, INPUT_SHAPE>;
        using ACTOR_HEAD = rlt::nn::layers::dense::Layer<ACTOR_HEAD_CONFIG, T_CAPABILITY, ACTOR_HEAD_INPUT_SHAPE>;
        using CRITIC_HEAD = rlt::nn::layers::dense::Layer<CRITIC_HEAD_CONFIG, T_CAPABILITY, CRITIC_HEAD_INPUT_SHAPE>;
        using OUTPUT_SHAPE = ACTION_OUTPUT_SHAPE;

        TRUNK trunk;
        ACTOR_HEAD actor_head;
        CRITIC_HEAD critic_head;

        template <typename T_NEW_CAPABILITY>
        using CHANGE_CAPABILITY = RDACActor<T_NEW_CAPABILITY>;

        template <bool T_DYNAMIC_ALLOCATION=true>
        struct Buffer{
            typename TRUNK::template Buffer<T_DYNAMIC_ALLOCATION> trunk;
            typename ACTOR_HEAD::template Buffer<T_DYNAMIC_ALLOCATION> actor_head;
            typename CRITIC_HEAD::template Buffer<T_DYNAMIC_ALLOCATION> critic_head;
            rlt::Tensor<rlt::tensor::Specification<T, TI, HIDDEN_SHAPE, T_DYNAMIC_ALLOCATION>> hidden;
            rlt::Tensor<rlt::tensor::Specification<T, TI, ACTION_OUTPUT_SHAPE, T_DYNAMIC_ALLOCATION>> action;
            rlt::Tensor<rlt::tensor::Specification<T, TI, ACTOR_HEAD_INPUT_SHAPE, T_DYNAMIC_ALLOCATION>> actor_input;
            rlt::Tensor<rlt::tensor::Specification<T, TI, CRITIC_HEAD_INPUT_SHAPE, T_DYNAMIC_ALLOCATION>> critic_input;
            rlt::Tensor<rlt::tensor::Specification<T, TI, CRITIC_OUTPUT_SHAPE, T_DYNAMIC_ALLOCATION>> q;
            rlt::Tensor<rlt::tensor::Specification<T, TI, HIDDEN_SHAPE, T_DYNAMIC_ALLOCATION>> d_hidden;
            rlt::Tensor<rlt::tensor::Specification<T, TI, ACTION_OUTPUT_SHAPE, T_DYNAMIC_ALLOCATION>> d_action_total;
            rlt::Tensor<rlt::tensor::Specification<T, TI, ACTOR_HEAD_INPUT_SHAPE, T_DYNAMIC_ALLOCATION>> d_actor_input;
            rlt::Tensor<rlt::tensor::Specification<T, TI, CRITIC_HEAD_INPUT_SHAPE, T_DYNAMIC_ALLOCATION>> d_critic_input;
            rlt::Tensor<rlt::tensor::Specification<T, TI, CRITIC_OUTPUT_SHAPE, T_DYNAMIC_ALLOCATION>> d_q_critic;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DIFF_TRAINING_BATCH_SIZE, HIDDEN_DIM>, T_DYNAMIC_ALLOCATION>> step_hidden;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DIFF_TRAINING_BATCH_SIZE, ENVIRONMENT::Observation::DIM + HIDDEN_DIM>, T_DYNAMIC_ALLOCATION>> step_actor_input;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DIFF_TRAINING_BATCH_SIZE, ENVIRONMENT::Observation::DIM + HIDDEN_DIM>, T_DYNAMIC_ALLOCATION>> step_critic_input;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DIFF_TRAINING_BATCH_SIZE, CRITIC_DIM>, T_DYNAMIC_ALLOCATION>> step_q;
        };

        template <bool T_DYNAMIC_ALLOCATION=true>
        using State = typename TRUNK::template State<T_DYNAMIC_ALLOCATION>;
    };

    using ACTOR = RDACActor<CAPABILITY>;
    using ROLLOUT_ACTOR = ACTOR::CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;

    struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
        static constexpr T ALPHA = DIFF_TRAINING_LEARNING_RATE;
        static constexpr T WEIGHT_DECAY = 0;
        static constexpr T WEIGHT_DECAY_INPUT = 0;
        static constexpr T WEIGHT_DECAY_OUTPUT = 0;
    };
    using OPTIMIZER = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<T, TI, ADAM_PARAMETERS>>;
}
