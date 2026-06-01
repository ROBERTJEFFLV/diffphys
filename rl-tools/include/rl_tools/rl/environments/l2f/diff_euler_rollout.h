#pragma once

#include "diff_euler_model.h"

namespace rl_tools::rl::environments::l2f::diff{
    template <typename T>
    struct EulerLossWeights{
        T position;
        T velocity;
        T attitude;
        T angular_velocity;
        T action_magnitude;
        T action_smoothness;
        T saturation;
        T terminal_loss_weight;
        T terminal_position;
        T terminal_velocity;
        T terminal_attitude;
        T terminal_angular_velocity;
    };

    template <typename T>
    struct EulerLossTerms{
        T position = 0;
        T velocity = 0;
        T attitude = 0;
        T angular_velocity = 0;
        T action_magnitude = 0;
        T action_smoothness = 0;
        T saturation = 0;
        T terminal = 0;
        T terminal_position = 0;
        T terminal_velocity = 0;
        T terminal_attitude = 0;
        T terminal_angular_velocity = 0;

        T total() const{
            return position + velocity + attitude + angular_velocity + action_magnitude + action_smoothness + saturation + terminal;
        }
        void add(const EulerLossTerms& other){
            position += other.position;
            velocity += other.velocity;
            attitude += other.attitude;
            angular_velocity += other.angular_velocity;
            action_magnitude += other.action_magnitude;
            action_smoothness += other.action_smoothness;
            saturation += other.saturation;
            terminal += other.terminal;
            terminal_position += other.terminal_position;
            terminal_velocity += other.terminal_velocity;
            terminal_attitude += other.terminal_attitude;
            terminal_angular_velocity += other.terminal_angular_velocity;
        }
        void scale(T factor){
            position *= factor;
            velocity *= factor;
            attitude *= factor;
            angular_velocity *= factor;
            action_magnitude *= factor;
            action_smoothness *= factor;
            saturation *= factor;
            terminal *= factor;
            terminal_position *= factor;
            terminal_velocity *= factor;
            terminal_attitude *= factor;
            terminal_angular_velocity *= factor;
        }
    };

    template <typename T, typename TI>
    void add_state_loss(
        const EulerState<T, TI>& state,
        const EulerLossWeights<T>& weights,
        T normalizer,
        EulerLossTerms<T>& terms,
        EulerStateAdjoint<T>& lambda
    ){
        for(TI i = 0; i < 3; i++){
            terms.position += normalizer * weights.position * state.p[i] * state.p[i];
            lambda.p[i] += normalizer * (T)2 * weights.position * state.p[i];
            terms.velocity += normalizer * weights.velocity * state.v[i] * state.v[i];
            lambda.v[i] += normalizer * (T)2 * weights.velocity * state.v[i];
            terms.angular_velocity += normalizer * weights.angular_velocity * state.omega[i] * state.omega[i];
            lambda.omega[i] += normalizer * (T)2 * weights.angular_velocity * state.omega[i];
        }
        for(TI i = 0; i < 3; i++){
            for(TI j = 0; j < 3; j++){
                const T target = i == j ? (T)1 : (T)0;
                const T error = state.R[i][j] - target;
                terms.attitude += normalizer * weights.attitude * error * error;
                lambda.R[i][j] += normalizer * (T)2 * weights.attitude * error;
            }
        }
    }

    template <typename T, typename TI>
    void add_terminal_loss(
        const EulerState<T, TI>& state,
        const EulerLossWeights<T>& weights,
        EulerLossTerms<T>& terms,
        EulerStateAdjoint<T>& lambda
    ){
        const T wp = weights.terminal_loss_weight * weights.terminal_position;
        const T wv = weights.terminal_loss_weight * weights.terminal_velocity;
        const T wR = weights.terminal_loss_weight * weights.terminal_attitude;
        const T ww = weights.terminal_loss_weight * weights.terminal_angular_velocity;
        for(TI i = 0; i < 3; i++){
            const T p_loss = wp * state.p[i] * state.p[i];
            const T v_loss = wv * state.v[i] * state.v[i];
            const T w_loss = ww * state.omega[i] * state.omega[i];
            terms.terminal_position += p_loss;
            terms.terminal_velocity += v_loss;
            terms.terminal_angular_velocity += w_loss;
            terms.terminal += p_loss + v_loss + w_loss;
            lambda.p[i] += (T)2 * wp * state.p[i];
            lambda.v[i] += (T)2 * wv * state.v[i];
            lambda.omega[i] += (T)2 * ww * state.omega[i];
        }
        for(TI i = 0; i < 3; i++){
            for(TI j = 0; j < 3; j++){
                const T target = i == j ? (T)1 : (T)0;
                const T error = state.R[i][j] - target;
                const T R_loss = wR * error * error;
                terms.terminal_attitude += R_loss;
                terms.terminal += R_loss;
                lambda.R[i][j] += (T)2 * wR * error;
            }
        }
    }

    template <typename T, typename TI, TI MAX_HORIZON>
    void add_action_loss(
        const EulerState<T, TI>& initial_state,
        const T (&actions)[MAX_HORIZON][4],
        TI horizon,
        const EulerLossWeights<T>& weights,
        T normalizer,
        EulerLossTerms<T>& terms,
        T (&action_gradients)[MAX_HORIZON][4]
    ){
        T previous_action[4];
        for(TI action_i = 0; action_i < 4; action_i++){
            previous_action[action_i] = initial_state.previous_action[action_i];
        }
        for(TI step_i = 0; step_i < horizon; step_i++){
            for(TI action_i = 0; action_i < 4; action_i++){
                terms.action_magnitude += normalizer * weights.action_magnitude * actions[step_i][action_i] * actions[step_i][action_i];
                action_gradients[step_i][action_i] += normalizer * (T)2 * weights.action_magnitude * actions[step_i][action_i];
                const T diff = actions[step_i][action_i] - previous_action[action_i];
                terms.action_smoothness += normalizer * weights.action_smoothness * diff * diff;
                action_gradients[step_i][action_i] += normalizer * (T)2 * weights.action_smoothness * diff;
                if(step_i > 0){
                    action_gradients[step_i - 1][action_i] -= normalizer * (T)2 * weights.action_smoothness * diff;
                }
                const T abs_action = std::abs(actions[step_i][action_i]);
                constexpr T SATURATION_START = (T)0.95;
                if(abs_action > SATURATION_START){
                    const T sign = actions[step_i][action_i] >= (T)0 ? (T)1 : (T)-1;
                    const T excess = abs_action - SATURATION_START;
                    terms.saturation += normalizer * weights.saturation * excess * excess;
                    action_gradients[step_i][action_i] += normalizer * (T)2 * weights.saturation * excess * sign;
                }
                previous_action[action_i] = actions[step_i][action_i];
            }
        }
    }

    template <typename PARAMETERS, typename T, typename TI, TI MAX_HORIZON>
    EulerLossTerms<T> rollout_loss(
        const PARAMETERS& parameters,
        const EulerState<T, TI>& initial_state,
        const T (&actions)[MAX_HORIZON][4],
        TI horizon,
        const EulerLossWeights<T>& weights,
        EulerState<T, TI> (&states)[MAX_HORIZON + 1],
        EulerStepCache<T> (&caches)[MAX_HORIZON]
    ){
        states[0] = initial_state;
        for(TI step_i = 0; step_i < horizon; step_i++){
            step<PARAMETERS, T, TI>(parameters, states[step_i], actions[step_i], states[step_i + 1], caches[step_i]);
        }
        EulerLossTerms<T> terms;
        const T normalizer = horizon > 0 ? (T)1 / (T)horizon : (T)1;
        EulerStateAdjoint<T> unused;
        zero(unused);
        for(TI step_i = 0; step_i < horizon; step_i++){
            EulerStateAdjoint<T> state_loss;
            zero(state_loss);
            add_state_loss<T, TI>(states[step_i + 1], weights, normalizer, terms, state_loss);
        }
        T dummy_gradients[MAX_HORIZON][4] = {};
        add_action_loss<T, TI, MAX_HORIZON>(initial_state, actions, horizon, weights, normalizer, terms, dummy_gradients);
        EulerStateAdjoint<T> terminal_loss;
        zero(terminal_loss);
        add_terminal_loss<T, TI>(states[horizon], weights, terms, terminal_loss);
        return terms;
    }

    template <typename PARAMETERS, typename T, typename TI, TI MAX_HORIZON>
    EulerLossTerms<T> rollout_loss_and_gradients(
        const PARAMETERS& parameters,
        const EulerState<T, TI>& initial_state,
        const T (&actions)[MAX_HORIZON][4],
        TI horizon,
        const EulerLossWeights<T>& weights,
        EulerState<T, TI> (&states)[MAX_HORIZON + 1],
        EulerStepCache<T> (&caches)[MAX_HORIZON],
        T (&action_gradients)[MAX_HORIZON][4]
    ){
        for(TI step_i = 0; step_i < MAX_HORIZON; step_i++){
            for(TI action_i = 0; action_i < 4; action_i++){
                action_gradients[step_i][action_i] = 0;
            }
        }

        states[0] = initial_state;
        for(TI step_i = 0; step_i < horizon; step_i++){
            step<PARAMETERS, T, TI>(parameters, states[step_i], actions[step_i], states[step_i + 1], caches[step_i]);
        }

        EulerStateAdjoint<T> lambdas[MAX_HORIZON + 1];
        for(TI step_i = 0; step_i <= MAX_HORIZON; step_i++){
            zero(lambdas[step_i]);
        }

        EulerLossTerms<T> terms;
        const T normalizer = horizon > 0 ? (T)1 / (T)horizon : (T)1;
        for(TI step_i = 0; step_i < horizon; step_i++){
            add_state_loss<T, TI>(states[step_i + 1], weights, normalizer, terms, lambdas[step_i + 1]);
        }
        add_terminal_loss<T, TI>(states[horizon], weights, terms, lambdas[horizon]);
        add_action_loss<T, TI, MAX_HORIZON>(initial_state, actions, horizon, weights, normalizer, terms, action_gradients);

        for(TI reverse_i = 0; reverse_i < horizon; reverse_i++){
            const TI step_i = horizon - 1 - reverse_i;
            EulerStateAdjoint<T> lambda_state;
            T transition_grad[4];
            step_vjp<PARAMETERS, T, TI>(parameters, states[step_i], actions[step_i], caches[step_i], lambdas[step_i + 1], lambda_state, transition_grad);
            for(TI action_i = 0; action_i < 4; action_i++){
                action_gradients[step_i][action_i] += transition_grad[action_i];
            }
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                lambdas[step_i].p[dim_i] += lambda_state.p[dim_i];
                lambdas[step_i].v[dim_i] += lambda_state.v[dim_i];
                lambdas[step_i].omega[dim_i] += lambda_state.omega[dim_i];
                for(TI dim_j = 0; dim_j < 3; dim_j++){
                    lambdas[step_i].R[dim_i][dim_j] += lambda_state.R[dim_i][dim_j];
                }
            }
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                lambdas[step_i].rpm[rotor_i] += lambda_state.rpm[rotor_i];
            }
        }
        return terms;
    }

    template <typename T, typename TI, TI MAX_HORIZON>
    T action_gradient_norm(const T (&action_gradients)[MAX_HORIZON][4], TI horizon){
        T sum = 0;
        for(TI step_i = 0; step_i < horizon; step_i++){
            for(TI action_i = 0; action_i < 4; action_i++){
                sum += action_gradients[step_i][action_i] * action_gradients[step_i][action_i];
            }
        }
        return std::sqrt(sum);
    }
}
