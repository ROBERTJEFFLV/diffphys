#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_TRANSITION_LINEARIZATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_TRANSITION_LINEARIZATION_H

#include "60_dynamics_jacobian.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    template<typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T normalized_action_scale(const PARAMETERS& parameters){
        return (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) / (T)2;
    }

    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void motor_transition_and_gradient(DEVICE& device, const PARAMETERS& parameters, T rpm, T normalized_action, typename DEVICE::index_t rotor_i, T& next_rpm, T& d_next_rpm_d_normalized_action){
        const T half_range = normalized_action_scale<PARAMETERS, T>(parameters);
        const T unclamped = normalized_action * half_range + parameters.dynamics.action_limit.min + half_range;
        const T setpoint = math::clamp(device.math, unclamped, parameters.dynamics.action_limit.min, parameters.dynamics.action_limit.max);
        const T d_setpoint_d_normalized_action = unclamped == setpoint ? half_range : (T)0;
        const T tau = setpoint >= rpm ? parameters.dynamics.rotor_time_constants_rising[rotor_i] : parameters.dynamics.rotor_time_constants_falling[rotor_i];
        const T alpha = math::exp(device.math, -parameters.integration.dt / tau);
        next_rpm = alpha * rpm + ((T)1 - alpha) * setpoint;
        d_next_rpm_d_normalized_action = ((T)1 - alpha) * d_setpoint_d_normalized_action;
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void transition_linearization(DEVICE& device, const Multirotor<SPEC>& env, PARAMETERS& parameters, const StateRotors<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& normalized_action, DynamicsJacobian<typename SPEC::T, typename DEVICE::index_t, StateRotors<STATE_SPEC>::DIM>& linearization, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using STATE = StateRotors<STATE_SPEC>;
        zero_dynamics_jacobian(linearization);

        T next_rpm[4];
        T d_next_rpm_d_action[4];
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            motor_transition_and_gradient(device, parameters, state.rpm[rotor_i], get(normalized_action, 0, rotor_i), rotor_i, next_rpm[rotor_i], d_next_rpm_d_action[rotor_i]);
        }

        RotorAccelerationJacobian<T, TI> acceleration_jacobian;
        rotor_acceleration_jacobian_from_quaternion(device, parameters, state.orientation, next_rpm, acceleration_jacobian);
        const T dt = parameters.integration.dt;
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                const T d_a = acceleration_jacobian.linear[rotor_i][dim_i] * d_next_rpm_d_action[rotor_i];
                const T d_alpha = acceleration_jacobian.angular[rotor_i][dim_i] * d_next_rpm_d_action[rotor_i];
                linearization.B[0 + dim_i][rotor_i] = (T)0.5 * dt * dt * d_a;
                linearization.B[7 + dim_i][rotor_i] = dt * d_a;
                linearization.B[10 + dim_i][rotor_i] = dt * d_alpha;
            }
            linearization.B[STATE::NEXT_COMPONENT::DIM + rotor_i][rotor_i] = d_next_rpm_d_action[rotor_i];
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
