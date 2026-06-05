#pragma once

#include "config.h"

#include <rl_tools/rl/environments/l2f/diff_euler_model.h>
#include <rl_tools/rl/environments/l2f/operations_generic/65_transition_linearization.h>

namespace rl_tools::foundation_policy::diff_pre_training{
    template <typename T>
    struct LossWeights{
        T position;
        T velocity;
        T attitude;
        T angular_velocity;
        T action_magnitude;
        T action_smoothness;
        T saturation;
    };

    template <typename T>
    struct LossTerms{
        T position = 0;
        T velocity = 0;
        T attitude = 0;
        T angular_velocity = 0;
        T action_magnitude = 0;
        T action_smoothness = 0;
        T saturation = 0;

        T total() const{
            return position + velocity + attitude + angular_velocity + action_magnitude + action_smoothness + saturation;
        }
        void add(const LossTerms& other){
            position += other.position;
            velocity += other.velocity;
            attitude += other.attitude;
            angular_velocity += other.angular_velocity;
            action_magnitude += other.action_magnitude;
            action_smoothness += other.action_smoothness;
            saturation += other.saturation;
        }
        void scale(T factor){
            position *= factor;
            velocity *= factor;
            attitude *= factor;
            angular_velocity *= factor;
            action_magnitude *= factor;
            action_smoothness *= factor;
            saturation *= factor;
        }
    };

    template <typename STATE, typename T, typename TI>
    void previous_action_from_state(const STATE& state, T previous_action[4]){
        static_assert(STATE::HISTORY_LENGTH >= 1, "diff pre-training expects action history in the state");
        const TI current_step = state.current_step == 0 ? STATE::HISTORY_LENGTH - 1 : state.current_step - 1;
        for(TI action_i = 0; action_i < 4; action_i++){
            previous_action[action_i] = state.action_history[current_step][action_i];
        }
    }

    template <typename T>
    T sign_for_shortest_quaternion(const T q[4]){
        return q[0] >= (T)0 ? (T)1 : (T)-1;
    }

    template <typename DEVICE, typename PARAMETERS, typename STATE, typename T, typename TI, TI HORIZON>
    LossTerms<T> stabilization_loss_and_action_gradients(
        DEVICE& device,
        const PARAMETERS& parameters,
        const STATE (&states)[HORIZON + 1],
        const T (&actions)[HORIZON][4],
        T (&action_gradients)[HORIZON][4],
        const LossWeights<T>& weights,
        const rl::environments::l2f::diff::TrackingReference<T>& ref
    ){
        namespace l2f = rl_tools::rl::environments::l2f;
        LossTerms<T> terms;
        for(TI step_i = 0; step_i < HORIZON; step_i++){
            for(TI action_i = 0; action_i < 4; action_i++){
                action_gradients[step_i][action_i] = (T)0;
            }
        }

        T d_position[HORIZON][4][3] = {};
        T d_velocity[HORIZON][4][3] = {};
        T d_attitude[HORIZON][4][3] = {};
        T d_angular_velocity[HORIZON][4][3] = {};
        T d_rpm[HORIZON][4] = {};

        T previous_action[4];
        previous_action_from_state<STATE, T, TI>(states[0], previous_action);

        const T dt = parameters.integration.dt;
        const T normalizer = HORIZON > 0 ? (T)1 / (T)HORIZON : (T)1;

        for(TI step_i = 0; step_i < HORIZON; step_i++){
            T next_rpm[4];
            T d_next_rpm_d_action[4];
            T next_d_rpm[HORIZON][4] = {};
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                l2f::motor_transition_and_gradient(device, parameters, states[step_i].rpm[rotor_i], actions[step_i][rotor_i], rotor_i, next_rpm[rotor_i], d_next_rpm_d_action[rotor_i]);
                const T half_range = l2f::normalized_action_scale<PARAMETERS, T>(parameters);
                const T unclamped_setpoint = actions[step_i][rotor_i] * half_range + parameters.dynamics.action_limit.min + half_range;
                const T setpoint = math::clamp(device.math, unclamped_setpoint, parameters.dynamics.action_limit.min, parameters.dynamics.action_limit.max);
                const T tau = setpoint >= states[step_i].rpm[rotor_i] ? parameters.dynamics.rotor_time_constants_rising[rotor_i] : parameters.dynamics.rotor_time_constants_falling[rotor_i];
                const T alpha = math::exp(device.math, -dt / tau);
                for(TI past_step_i = 0; past_step_i < step_i; past_step_i++){
                    next_d_rpm[past_step_i][rotor_i] = alpha * d_rpm[past_step_i][rotor_i];
                }
                next_d_rpm[step_i][rotor_i] = d_next_rpm_d_action[rotor_i];
            }

            l2f::RotorAccelerationJacobian<T, TI> acceleration_jacobian;
            l2f::rotor_acceleration_jacobian_from_quaternion(device, parameters, states[step_i].orientation, next_rpm, acceleration_jacobian);

            // MVP gradient model: propagate action -> rpm -> thrust/torque ->
            // acceleration -> stabilization loss. This intentionally omits the
            // full A = dx_next/dx Jacobian. The missing terms to add next are
            // orientation effects on thrust direction across multiple steps,
            // Coriolis/gyroscopic coupling derivatives, and quaternion/rotation
            // integration gradients. The structure below keeps those extensions
            // localized to this transition sensitivity update.
            for(TI past_step_i = 0; past_step_i <= step_i; past_step_i++){
                for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                    for(TI dim_i = 0; dim_i < 3; dim_i++){
                        const T d_a = acceleration_jacobian.linear[rotor_i][dim_i] * next_d_rpm[past_step_i][rotor_i];
                        const T d_alpha = acceleration_jacobian.angular[rotor_i][dim_i] * next_d_rpm[past_step_i][rotor_i];
                        d_position[past_step_i][rotor_i][dim_i] += dt * d_velocity[past_step_i][rotor_i][dim_i] + (T)0.5 * dt * dt * d_a;
                        d_velocity[past_step_i][rotor_i][dim_i] += dt * d_a;
                        d_attitude[past_step_i][rotor_i][dim_i] += (T)0.5 * dt * d_angular_velocity[past_step_i][rotor_i][dim_i] + (T)0.25 * dt * dt * d_alpha;
                        d_angular_velocity[past_step_i][rotor_i][dim_i] += dt * d_alpha;
                    }
                    d_rpm[past_step_i][rotor_i] = next_d_rpm[past_step_i][rotor_i];
                }
            }

            const STATE& state = states[step_i + 1];
            T attitude_error[3];
            const T orientation_sign = sign_for_shortest_quaternion(state.orientation);
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                const T p = state.position[dim_i] - ref.p[dim_i];
                const T v = state.linear_velocity[dim_i] - ref.v[dim_i];
                const T e_R = (T)2 * orientation_sign * state.orientation[dim_i + 1];
                const T w = state.angular_velocity[dim_i];
                attitude_error[dim_i] = e_R;
                terms.position += normalizer * weights.position * p * p;
                terms.velocity += normalizer * weights.velocity * v * v;
                terms.attitude += normalizer * weights.attitude * e_R * e_R;
                terms.angular_velocity += normalizer * weights.angular_velocity * w * w;
                for(TI past_step_i = 0; past_step_i <= step_i; past_step_i++){
                    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                        action_gradients[past_step_i][rotor_i] += normalizer * (
                            (T)2 * weights.position * p * d_position[past_step_i][rotor_i][dim_i] +
                            (T)2 * weights.velocity * v * d_velocity[past_step_i][rotor_i][dim_i] +
                            (T)2 * weights.attitude * e_R * d_attitude[past_step_i][rotor_i][dim_i] +
                            (T)2 * weights.angular_velocity * w * d_angular_velocity[past_step_i][rotor_i][dim_i]
                        );
                    }
                }
            }

            for(TI action_i = 0; action_i < 4; action_i++){
                terms.action_magnitude += normalizer * weights.action_magnitude * actions[step_i][action_i] * actions[step_i][action_i];
                action_gradients[step_i][action_i] += normalizer * (T)2 * weights.action_magnitude * actions[step_i][action_i];
                const T action_diff = actions[step_i][action_i] - previous_action[action_i];
                terms.action_smoothness += normalizer * weights.action_smoothness * action_diff * action_diff;
                action_gradients[step_i][action_i] += normalizer * (T)2 * weights.action_smoothness * action_diff;
                if(step_i > 0){
                    action_gradients[step_i - 1][action_i] -= normalizer * (T)2 * weights.action_smoothness * action_diff;
                }
                previous_action[action_i] = actions[step_i][action_i];

                const T abs_action = math::abs(device.math, actions[step_i][action_i]);
                constexpr T SATURATION_START = (T)0.95;
                if(abs_action > SATURATION_START){
                    const T sign = actions[step_i][action_i] >= (T)0 ? (T)1 : (T)-1;
                    const T excess = abs_action - SATURATION_START;
                    terms.saturation += normalizer * weights.saturation * excess * excess;
                    action_gradients[step_i][action_i] += normalizer * (T)2 * weights.saturation * excess * sign;
                }
            }
        }
        return terms;
    }

    template <typename DEVICE, typename PARAMETERS, typename STATE, typename T, typename TI, TI HORIZON>
    LossTerms<T> stabilization_loss_and_action_gradients(
        DEVICE& device,
        const PARAMETERS& parameters,
        const STATE (&states)[HORIZON + 1],
        const T (&actions)[HORIZON][4],
        T (&action_gradients)[HORIZON][4],
        const LossWeights<T>& weights
    ){
        const auto ref = rl::environments::l2f::diff::zero_tracking_reference<T>();
        return stabilization_loss_and_action_gradients<DEVICE, PARAMETERS, STATE, T, TI, HORIZON>(device, parameters, states, actions, action_gradients, weights, ref);
    }
}
