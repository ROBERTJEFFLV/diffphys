namespace rl_tools::rl::zoo::l2f::sac_diff{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG, bool DYNAMIC_ALLOCATION=true>
    struct FACTORY{
        using BASE = sac::FACTORY<DEVICE, T, TI, RNG, DYNAMIC_ALLOCATION>;
        using ENVIRONMENT = typename BASE::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: BASE::LOOP_CORE_PARAMETERS{
            struct SAC_PARAMETERS: BASE::LOOP_CORE_PARAMETERS::SAC_PARAMETERS{
                static constexpr bool DIFFERENTIABLE_PHYSICS_ACTOR_LOSS = true;
                static constexpr TI DIFFERENTIABLE_PHYSICS_HORIZON = 8;
                static constexpr T DIFFERENTIABLE_PHYSICS_WEIGHT = 0.05;
                static constexpr T DIFFERENTIABLE_PHYSICS_POSITION_WEIGHT = 10.0;
                static constexpr T DIFFERENTIABLE_PHYSICS_VELOCITY_WEIGHT = 0.10;
                static constexpr T DIFFERENTIABLE_PHYSICS_ORIENTATION_WEIGHT = 4.0;
                static constexpr T DIFFERENTIABLE_PHYSICS_ANGULAR_VELOCITY_WEIGHT = 0.10;
                static constexpr T DIFFERENTIABLE_PHYSICS_ACTION_WEIGHT = 0.001;
                static constexpr T DIFFERENTIABLE_PHYSICS_SATURATION_WEIGHT = 0.01;
            };
        };

        using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsMLP, DYNAMIC_ALLOCATION>;
    };
}
