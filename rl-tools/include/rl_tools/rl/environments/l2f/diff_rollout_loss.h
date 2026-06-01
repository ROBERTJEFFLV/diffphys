#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_DIFF_ROLLOUT_LOSS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_DIFF_ROLLOUT_LOSS_H

#include "operations_generic/65_transition_linearization.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    template<typename DEVICE, typename PARAMETERS, typename STATE, typename T, typename TI, TI HORIZON>
    RL_TOOLS_FUNCTION_PLACEMENT void differentiable_rollout_action_gradient(DEVICE& device, const PARAMETERS& parameters, const STATE& state, const T action_normalized[4], T gradient[4], T position_weight, T velocity_weight, T orientation_weight, T angular_velocity_weight, T action_weight, T saturation_weight){
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            gradient[rotor_i] = (T)0;
        }

        T position[3];
        T linear_velocity[3];
        T orientation_error[3];
        T angular_velocity[3];
        T rpm[4];
        T d_position[4][3];
        T d_linear_velocity[4][3];
        T d_orientation_error[4][3];
        T d_angular_velocity[4][3];
        T d_rpm[4];

        const T orientation_sign = state.orientation[0] >= (T)0 ? (T)1 : (T)-1;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            position[dim_i] = state.position[dim_i];
            linear_velocity[dim_i] = state.linear_velocity[dim_i];
            orientation_error[dim_i] = orientation_sign * state.orientation[1 + dim_i];
            angular_velocity[dim_i] = state.angular_velocity[dim_i];
        }
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            rpm[rotor_i] = state.rpm[rotor_i];
            d_rpm[rotor_i] = (T)0;
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                d_position[rotor_i][dim_i] = (T)0;
                d_linear_velocity[rotor_i][dim_i] = (T)0;
                d_orientation_error[rotor_i][dim_i] = (T)0;
                d_angular_velocity[rotor_i][dim_i] = (T)0;
            }
        }

        const T dt = parameters.integration.dt;
        const T horizon_normalizer = HORIZON > 0 ? (T)1 / (T)HORIZON : (T)1;
        for(TI step_i = 0; step_i < HORIZON; step_i++){
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                T next_rpm;
                T d_next_rpm_d_current_action;
                motor_transition_and_gradient(device, parameters, rpm[rotor_i], action_normalized[rotor_i], rotor_i, next_rpm, d_next_rpm_d_current_action);
                const T tau = next_rpm >= rpm[rotor_i] ? parameters.dynamics.rotor_time_constants_rising[rotor_i] : parameters.dynamics.rotor_time_constants_falling[rotor_i];
                const T alpha = math::exp(device.math, -dt / tau);
                d_rpm[rotor_i] = alpha * d_rpm[rotor_i] + d_next_rpm_d_current_action;
                rpm[rotor_i] = next_rpm;
            }

            T linear_acceleration[3];
            T angular_acceleration[3];
            RotorAccelerationJacobian<T, TI> acceleration_jacobian;
            rotor_accelerations_from_quaternion(device, parameters, state.orientation, angular_velocity, rpm, linear_acceleration, angular_acceleration);
            rotor_acceleration_jacobian_from_quaternion(device, parameters, state.orientation, rpm, acceleration_jacobian);

            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                for(TI dim_i = 0; dim_i < 3; dim_i++){
                    const T d_a = acceleration_jacobian.linear[rotor_i][dim_i] * d_rpm[rotor_i];
                    const T d_alpha = acceleration_jacobian.angular[rotor_i][dim_i] * d_rpm[rotor_i];
                    d_position[rotor_i][dim_i] += dt * d_linear_velocity[rotor_i][dim_i] + (T)0.5 * dt * dt * d_a;
                    d_linear_velocity[rotor_i][dim_i] += dt * d_a;
                    d_orientation_error[rotor_i][dim_i] += (T)0.5 * dt * d_angular_velocity[rotor_i][dim_i] + (T)0.25 * dt * dt * d_alpha;
                    d_angular_velocity[rotor_i][dim_i] += dt * d_alpha;
                }
            }

            for(TI dim_i = 0; dim_i < 3; dim_i++){
                position[dim_i] += dt * linear_velocity[dim_i] + (T)0.5 * dt * dt * linear_acceleration[dim_i];
                linear_velocity[dim_i] += dt * linear_acceleration[dim_i];
                orientation_error[dim_i] += (T)0.5 * dt * angular_velocity[dim_i] + (T)0.25 * dt * dt * angular_acceleration[dim_i];
                angular_velocity[dim_i] += dt * angular_acceleration[dim_i];
            }

            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                for(TI dim_i = 0; dim_i < 3; dim_i++){
                    gradient[rotor_i] += horizon_normalizer * (
                        (T)2 * position_weight * position[dim_i] * d_position[rotor_i][dim_i] +
                        (T)2 * velocity_weight * linear_velocity[dim_i] * d_linear_velocity[rotor_i][dim_i] +
                        (T)2 * orientation_weight * orientation_error[dim_i] * d_orientation_error[rotor_i][dim_i] +
                        (T)2 * angular_velocity_weight * angular_velocity[dim_i] * d_angular_velocity[rotor_i][dim_i]
                    );
                }
            }
        }

        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            gradient[rotor_i] += (T)2 * action_weight * action_normalized[rotor_i];
            const T abs_action = math::abs(device.math, action_normalized[rotor_i]);
            constexpr T SATURATION_START = (T)0.95;
            if(abs_action > SATURATION_START){
                const T sign = action_normalized[rotor_i] >= (T)0 ? (T)1 : (T)-1;
                gradient[rotor_i] += (T)2 * saturation_weight * (abs_action - SATURATION_START) * sign;
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
