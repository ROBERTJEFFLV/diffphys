#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_DYNAMICS_JACOBIAN_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_DYNAMICS_JACOBIAN_H

#include "../multirotor.h"
#include "../quaternion_helper.h"
#include "60_dynamics.h"

#include <rl_tools/utils/generic/vector_operations.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    template<typename T, typename TI, TI T_STATE_DIM, TI T_ACTION_DIM=4, TI T_PARAMETER_DIM=0>
    struct DynamicsJacobian{
        static constexpr TI STATE_DIM = T_STATE_DIM;
        static constexpr TI ACTION_DIM = T_ACTION_DIM;
        static constexpr TI PARAMETER_DIM = T_PARAMETER_DIM;
        static constexpr TI PARAMETER_DIM_STORAGE = PARAMETER_DIM == 0 ? 1 : PARAMETER_DIM;
        T A[STATE_DIM][STATE_DIM];
        T B[STATE_DIM][ACTION_DIM];
        T P[STATE_DIM][PARAMETER_DIM_STORAGE];
    };

    template<typename T, typename TI>
    struct RotorAccelerationJacobian{
        static constexpr TI ACTION_DIM = 4;
        T linear[ACTION_DIM][3];
        T angular[ACTION_DIM][3];
    };

    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void rotor_force_torque_derivative(DEVICE& device, const PARAMETERS& params, typename DEVICE::index_t rotor_i, T rpm, T d_thrust_body[3], T d_torque_body[3]){
        using TI = typename DEVICE::index_t;
        const T d_thrust_magnitude_d_rpm = params.dynamics.rotor_thrust_coefficients[rotor_i][1] + (T)2 * params.dynamics.rotor_thrust_coefficients[rotor_i][2] * rpm;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            d_thrust_body[dim_i] = params.dynamics.rotor_thrust_directions[rotor_i][dim_i] * d_thrust_magnitude_d_rpm;
            d_torque_body[dim_i] = params.dynamics.rotor_torque_directions[rotor_i][dim_i] * params.dynamics.rotor_torque_constants[rotor_i] * d_thrust_magnitude_d_rpm;
        }
        T arm_torque[3];
        rl_tools::utils::vector_operations::cross_product<DEVICE, T>(params.dynamics.rotor_positions[rotor_i], d_thrust_body, arm_torque);
        rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(arm_torque, d_torque_body);
    }

    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void rotor_acceleration_jacobian_from_rotation_matrix(DEVICE& device, const PARAMETERS& params, const T rotation_body_to_world[9], const T rpm[4], RotorAccelerationJacobian<T, typename DEVICE::index_t>& jacobian){
        using TI = typename DEVICE::index_t;
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            T d_thrust_body[3];
            T d_torque_body[3];
            rotor_force_torque_derivative(device, params, rotor_i, rpm[rotor_i], d_thrust_body, d_torque_body);
            for(TI row_i = 0; row_i < 3; row_i++){
                jacobian.linear[rotor_i][row_i] = (
                    rotation_body_to_world[row_i * 3 + 0] * d_thrust_body[0] +
                    rotation_body_to_world[row_i * 3 + 1] * d_thrust_body[1] +
                    rotation_body_to_world[row_i * 3 + 2] * d_thrust_body[2]
                ) / params.dynamics.mass;
            }
            rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, d_torque_body, jacobian.angular[rotor_i]);
        }
    }

    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void rotor_acceleration_jacobian_from_quaternion(DEVICE& device, const PARAMETERS& params, const T orientation[4], const T rpm[4], RotorAccelerationJacobian<T, typename DEVICE::index_t>& jacobian){
        using TI = typename DEVICE::index_t;
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            T d_thrust_body[3];
            T d_torque_body[3];
            rotor_force_torque_derivative(device, params, rotor_i, rpm[rotor_i], d_thrust_body, d_torque_body);
            rotate_vector_by_quaternion<DEVICE, T>(orientation, d_thrust_body, jacobian.linear[rotor_i]);
            rl_tools::utils::vector_operations::scalar_multiply<DEVICE, T, 3>(jacobian.linear[rotor_i], (T)1 / params.dynamics.mass);
            rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, d_torque_body, jacobian.angular[rotor_i]);
        }
    }

    template<typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void rotor_accelerations_from_quaternion(DEVICE& device, const PARAMETERS& params, const T orientation[4], const T angular_velocity[3], const T rpm[4], T linear_acceleration[3], T angular_acceleration[3]){
        using TI = typename DEVICE::index_t;
        T thrust[3] = {0, 0, 0};
        T torque[3] = {0, 0, 0};
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            const T thrust_magnitude = params.dynamics.rotor_thrust_coefficients[rotor_i][0] + params.dynamics.rotor_thrust_coefficients[rotor_i][1] * rpm[rotor_i] + params.dynamics.rotor_thrust_coefficients[rotor_i][2] * rpm[rotor_i] * rpm[rotor_i];
            T rotor_thrust[3];
            rl_tools::utils::vector_operations::scalar_multiply<DEVICE, T, 3>(params.dynamics.rotor_thrust_directions[rotor_i], thrust_magnitude, rotor_thrust);
            rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(rotor_thrust, thrust);
            rl_tools::utils::vector_operations::scalar_multiply_accumulate<DEVICE, T, 3>(params.dynamics.rotor_torque_directions[rotor_i], thrust_magnitude * params.dynamics.rotor_torque_constants[rotor_i], torque);
            rl_tools::utils::vector_operations::cross_product_accumulate<DEVICE, T>(params.dynamics.rotor_positions[rotor_i], rotor_thrust, torque);
        }

        rotate_vector_by_quaternion<DEVICE, T>(orientation, thrust, linear_acceleration);
        rl_tools::utils::vector_operations::scalar_multiply<DEVICE, T, 3>(linear_acceleration, (T)1 / params.dynamics.mass);
        rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(params.dynamics.gravity, linear_acceleration);

        T angular_momentum[3];
        T coriolis[3];
        T angular_rhs[3];
        rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J, angular_velocity, angular_momentum);
        rl_tools::utils::vector_operations::cross_product<DEVICE, T>(angular_velocity, angular_momentum, coriolis);
        rl_tools::utils::vector_operations::sub<DEVICE, T, 3>(torque, coriolis, angular_rhs);
        rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, angular_rhs, angular_acceleration);
    }

    template<typename T, typename TI, TI STATE_DIM, TI ACTION_DIM, TI PARAMETER_DIM>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_dynamics_jacobian(DynamicsJacobian<T, TI, STATE_DIM, ACTION_DIM, PARAMETER_DIM>& jacobian){
        for(TI row_i = 0; row_i < STATE_DIM; row_i++){
            for(TI col_i = 0; col_i < STATE_DIM; col_i++){
                jacobian.A[row_i][col_i] = row_i == col_i ? (T)1 : (T)0;
            }
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                jacobian.B[row_i][action_i] = (T)0;
            }
            for(TI parameter_i = 0; parameter_i < decltype(jacobian)::PARAMETER_DIM_STORAGE; parameter_i++){
                jacobian.P[row_i][parameter_i] = (T)0;
            }
        }
    }

    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void dynamics_jacobian(DEVICE& device, const PARAMETERS& params, const StateBase<STATE_SPEC>& state, const T* action_rpm, DynamicsJacobian<T, typename DEVICE::index_t, StateBase<STATE_SPEC>::DIM>& jacobian){
        using TI = typename DEVICE::index_t;
        zero_dynamics_jacobian(jacobian);
        RotorAccelerationJacobian<T, TI> rotor_jacobian;
        rotor_acceleration_jacobian_from_quaternion(device, params, state.orientation, action_rpm, rotor_jacobian);
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                jacobian.B[7 + dim_i][rotor_i] = rotor_jacobian.linear[rotor_i][dim_i];
                jacobian.B[10 + dim_i][rotor_i] = rotor_jacobian.angular[rotor_i][dim_i];
            }
        }
    }

    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void dynamics_jacobian(DEVICE& device, const PARAMETERS& params, const StateRotors<STATE_SPEC>& state, const T* action_setpoint, DynamicsJacobian<T, typename DEVICE::index_t, StateRotors<STATE_SPEC>::DIM>& jacobian){
        using STATE = StateRotors<STATE_SPEC>;
        using TI = typename DEVICE::index_t;
        zero_dynamics_jacobian(jacobian);
        RotorAccelerationJacobian<T, TI> rotor_jacobian;
        rotor_acceleration_jacobian_from_quaternion(device, params, state.orientation, state.rpm, rotor_jacobian);
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                jacobian.B[7 + dim_i][rotor_i] = rotor_jacobian.linear[rotor_i][dim_i];
                jacobian.B[10 + dim_i][rotor_i] = rotor_jacobian.angular[rotor_i][dim_i];
            }
            if constexpr(!STATE_SPEC::CLOSED_FORM){
                const T tau = action_setpoint[rotor_i] >= state.rpm[rotor_i] ? params.dynamics.rotor_time_constants_rising[rotor_i] : params.dynamics.rotor_time_constants_falling[rotor_i];
                jacobian.B[STATE::NEXT_COMPONENT::DIM + rotor_i][rotor_i] = (T)1 / tau;
            }
        }
    }

    template<typename DEVICE, typename PARAMETERS, typename STATE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void forward_dynamics(DEVICE& device, const PARAMETERS& params, const STATE& state, const T* action, STATE& state_change){
        multirotor_dynamics(device, params, state, action, state_change);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
