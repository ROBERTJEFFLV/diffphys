#pragma once

#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>

#include <rl_tools/utils/generic/typing.h>

namespace rl_tools::foundation_policy::diff_pre_training{
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;

    template <typename DEVICE, typename T, typename TI>
    struct EnvironmentFactory{
        using BASE_ENV = rl_tools::rl::environments::Multirotor<Specification<T, TI>>;
        static constexpr auto MODEL = parameters::dynamics::REGISTRY::crazyflie;

        using REWARD_FUNCTION = parameters::reward_functions::Squared<T>;
        struct DOMAIN_RANDOMIZATION_OPTIONS{
            static constexpr bool ENABLED = true;
            static constexpr bool THRUST_TO_WEIGHT = ENABLED;
            static constexpr bool MASS = ENABLED;
            static constexpr bool TORQUE_TO_INERTIA = ENABLED;
            static constexpr bool MASS_SIZE_DEVIATION = ENABLED;
            static constexpr bool ROTOR_TORQUE_CONSTANT = ENABLED;
            static constexpr bool DISTURBANCE_FORCE = ENABLED;
            static constexpr bool ROTOR_TIME_CONSTANT = ENABLED;
        };
        struct TRAJECTORY_OPTIONS{
            static constexpr bool LANGEVIN = false;
        };

        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = ParametersTrajectory<ParametersTrajectorySpecification<T, TI, TRAJECTORY_OPTIONS, ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, ParametersDisturbances<ParametersSpecification<T, TI, ParametersBase<PARAMETERS_SPEC>>>>>>>;

        static constexpr TI SIMULATION_FREQUENCY = 100;
        static constexpr auto BASE_PARAMS = BASE_ENV::SPEC::PARAMETER_VALUES;

        static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
            1.5,  // thrust_to_weight_min
            5.0,  // thrust_to_weight_max
            40.0, // torque_to_inertia_min
            1200.0, // torque_to_inertia_max
            0.02, // mass_min
            5.00, // mass_max
            0.10, // mass_size_deviation
            0.03, // rotor_time_constant_rising_min
            0.10, // rotor_time_constant_rising_max
            0.03, // rotor_time_constant_falling_min
            0.30, // rotor_time_constant_falling_max
            0.005, // rotor_torque_constant_min
            0.05, // rotor_torque_constant_max
            0.0, // orientation_offset_angle_max
            0.3  // disturbance_force_max
        };

        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0},
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0}
        };

        static constexpr typename PARAMETERS_TYPE::Trajectory trajectory = {
            {1.0, 0.0},
            typename PARAMETERS_TYPE::Trajectory::Langevin{
                1.0,
                2.0,
                0.0,
                0.01
            }
        };

        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                {
                    {
                        BASE_PARAMS.dynamics,
                        BASE_PARAMS.integration,
                        BASE_PARAMS.mdp
                    },
                    disturbances
                },
                domain_randomization
            },
            trajectory
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 1;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * SIMULATION_FREQUENCY;
            static constexpr bool CLOSED_FORM = false;
            static constexpr TI ANGULAR_VELOCITY_DELAY = 0;
            using STATE_BASE = StateAngularVelocityDelay<StateAngularVelocityDelaySpecification<T, TI, ANGULAR_VELOCITY_DELAY, StateLastAction<StateSpecification<T, TI, StateBase<StateSpecification<T, TI>>>>>>;
            using STATE_TYPE = StateTrajectory<StateSpecification<T, TI, StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>>>;
            using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                    observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                    observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH
            >>>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr T STATE_LIMIT_POSITION = 100000;
            static constexpr T STATE_LIMIT_VELOCITY = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };

        using ENVIRONMENT_SPEC = Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
