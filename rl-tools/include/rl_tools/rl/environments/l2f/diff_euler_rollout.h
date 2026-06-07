#pragma once

#include "diff_euler_model.h"

#include <algorithm>
#include <cmath>

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
        T clf = 0;
        T window_clf = 0;
        T clf_alpha = (T)0.2;
        T clf_position = 8;
        T clf_velocity = 0.8;
        T clf_attitude = 4;
        T clf_angular_velocity = 0.8;
        T outward_velocity = 0;
        T attitude_control = 0;
        T attitude_control_k_R = 2;
        T attitude_control_k_omega = 1;
        T action_magnitude_center = 0;
        T saturation_start = (T)0.95;
        T clf_position_velocity_cross_beta = 0;
        T clf_attitude_angular_velocity_cross_beta = 0;
        T window_clf_epsilon = (T)1e-3;
        T window_clf_huber_delta = (T)1;
        T velocity_barrier = 0;
        T velocity_barrier_safe = (T)10;
        T angular_velocity_barrier = 0;
        T angular_velocity_barrier_safe = (T)15;
        T attitude_barrier = 0;
        T attitude_barrier_safe = (T)4;
        T barrier_huber_delta = (T)1;
        T barrier_gradient_cap = 0;
        bool hover_relative_action_magnitude = false;
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
        T clf = 0;
        T window_clf = 0;
        T outward_velocity = 0;
        T attitude_control = 0;
        T velocity_barrier = 0;
        T angular_velocity_barrier = 0;
        T attitude_barrier = 0;

        T total() const{
            return position + velocity + attitude + angular_velocity + action_magnitude + action_smoothness + saturation + terminal +
                clf + window_clf + outward_velocity + attitude_control + velocity_barrier + angular_velocity_barrier + attitude_barrier;
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
            clf += other.clf;
            window_clf += other.window_clf;
            outward_velocity += other.outward_velocity;
            attitude_control += other.attitude_control;
            velocity_barrier += other.velocity_barrier;
            angular_velocity_barrier += other.angular_velocity_barrier;
            attitude_barrier += other.attitude_barrier;
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
            clf *= factor;
            window_clf *= factor;
            outward_velocity *= factor;
            attitude_control *= factor;
            velocity_barrier *= factor;
            angular_velocity_barrier *= factor;
            attitude_barrier *= factor;
        }
    };

    template <typename T, typename TI>
    void attitude_identity_error_vee(const EulerState<T, TI>& state, T (&e_R)[3]){
        e_R[0] = (T)0.5 * (state.R[2][1] - state.R[1][2]);
        e_R[1] = (T)0.5 * (state.R[0][2] - state.R[2][0]);
        e_R[2] = (T)0.5 * (state.R[1][0] - state.R[0][1]);
    }

    template <typename T, typename TI>
    void add_attitude_identity_error_vee_adjoint(T scale, const T (&grad_e_R)[3], EulerStateAdjoint<T>& lambda){
        lambda.R[2][1] += scale * (T)0.5 * grad_e_R[0];
        lambda.R[1][2] -= scale * (T)0.5 * grad_e_R[0];
        lambda.R[0][2] += scale * (T)0.5 * grad_e_R[1];
        lambda.R[2][0] -= scale * (T)0.5 * grad_e_R[1];
        lambda.R[1][0] += scale * (T)0.5 * grad_e_R[2];
        lambda.R[0][1] -= scale * (T)0.5 * grad_e_R[2];
    }

    template <typename T>
    T safe_cross_weight(T beta, T a, T b){
        const T beta_clamped = std::max((T)-0.999, std::min((T)0.999, beta));
        return beta_clamped * std::sqrt(std::max((T)0, a * b));
    }

    template <typename T>
    T positive_huber(T x, T delta){
        const T d = delta > (T)0 ? delta : (T)1;
        return x <= d ? (T)0.5 * x * x : d * (x - (T)0.5 * d);
    }

    template <typename T>
    T positive_huber_grad(T x, T delta){
        const T d = delta > (T)0 ? delta : (T)1;
        return x <= d ? x : d;
    }

    template <typename T>
    T cap_barrier_gradient(T value, const EulerLossWeights<T>& weights){
        if(weights.barrier_gradient_cap <= (T)0){
            return value;
        }
        return std::max(-weights.barrier_gradient_cap, std::min(weights.barrier_gradient_cap, value));
    }

    template <typename T, typename TI>
    T lyapunov_energy_identity(
        const EulerState<T, TI>& state,
        const TrackingReference<T>& ref,
        const EulerLossWeights<T>& weights
    ){
        T energy = 0;
        const T c_pv = safe_cross_weight<T>(
            weights.clf_position_velocity_cross_beta,
            weights.clf_position,
            weights.clf_velocity
        );
        for(TI i = 0; i < 3; i++){
            const T e_p = state.p[i] - ref.p[i];
            const T e_v = state.v[i] - ref.v[i];
            energy += (T)0.5 * weights.clf_position * e_p * e_p;
            energy += c_pv * e_p * e_v;
            energy += (T)0.5 * weights.clf_velocity * e_v * e_v;
        }
        T e_R[3];
        attitude_identity_error_vee<T, TI>(state, e_R);
        const T c_Rw = safe_cross_weight<T>(
            weights.clf_attitude_angular_velocity_cross_beta,
            weights.clf_attitude,
            weights.clf_angular_velocity
        );
        for(TI i = 0; i < 3; i++){
            energy += (T)0.5 * weights.clf_attitude * e_R[i] * e_R[i];
            energy += c_Rw * e_R[i] * state.omega[i];
            energy += (T)0.5 * weights.clf_angular_velocity * state.omega[i] * state.omega[i];
        }
        return energy;
    }

    template <typename T, typename TI>
    void add_lyapunov_energy_adjoint_identity(
        const EulerState<T, TI>& state,
        const TrackingReference<T>& ref,
        const EulerLossWeights<T>& weights,
        T scale,
        EulerStateAdjoint<T>& lambda
    ){
        const T c_pv = safe_cross_weight<T>(
            weights.clf_position_velocity_cross_beta,
            weights.clf_position,
            weights.clf_velocity
        );
        for(TI i = 0; i < 3; i++){
            const T e_p = state.p[i] - ref.p[i];
            const T e_v = state.v[i] - ref.v[i];
            lambda.p[i] += scale * (weights.clf_position * e_p + c_pv * e_v);
            lambda.v[i] += scale * (weights.clf_velocity * e_v + c_pv * e_p);
        }
        T e_R[3];
        attitude_identity_error_vee<T, TI>(state, e_R);
        const T c_Rw = safe_cross_weight<T>(
            weights.clf_attitude_angular_velocity_cross_beta,
            weights.clf_attitude,
            weights.clf_angular_velocity
        );
        T grad_e_R[3];
        for(TI i = 0; i < 3; i++){
            grad_e_R[i] = weights.clf_attitude * e_R[i] + c_Rw * state.omega[i];
            lambda.omega[i] += scale * (weights.clf_angular_velocity * state.omega[i] + c_Rw * e_R[i]);
        }
        add_attitude_identity_error_vee_adjoint<T, TI>(scale, grad_e_R, lambda);
    }

    template <typename T, typename TI>
    void add_state_loss(
        const EulerState<T, TI>& state,
        const TrackingReference<T>& ref,
        const EulerLossWeights<T>& weights,
        T normalizer,
        EulerLossTerms<T>& terms,
        EulerStateAdjoint<T>& lambda
    ){
        for(TI i = 0; i < 3; i++){
            const T e_p = state.p[i] - ref.p[i];
            const T e_v = state.v[i] - ref.v[i];
            terms.position += normalizer * weights.position * e_p * e_p;
            lambda.p[i] += normalizer * (T)2 * weights.position * e_p;
            terms.velocity += normalizer * weights.velocity * e_v * e_v;
            lambda.v[i] += normalizer * (T)2 * weights.velocity * e_v;
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
    void add_state_loss(
        const EulerState<T, TI>& state,
        const EulerLossWeights<T>& weights,
        T normalizer,
        EulerLossTerms<T>& terms,
        EulerStateAdjoint<T>& lambda
    ){
        const auto ref = zero_tracking_reference<T>();
        add_state_loss<T, TI>(state, ref, weights, normalizer, terms, lambda);
    }

    template <typename T, typename TI>
    void add_terminal_loss(
        const EulerState<T, TI>& state,
        const TrackingReference<T>& ref,
        const EulerLossWeights<T>& weights,
        EulerLossTerms<T>& terms,
        EulerStateAdjoint<T>& lambda
    ){
        const T wp = weights.terminal_loss_weight * weights.terminal_position;
        const T wv = weights.terminal_loss_weight * weights.terminal_velocity;
        const T wR = weights.terminal_loss_weight * weights.terminal_attitude;
        const T ww = weights.terminal_loss_weight * weights.terminal_angular_velocity;
        for(TI i = 0; i < 3; i++){
            const T e_p = state.p[i] - ref.p[i];
            const T e_v = state.v[i] - ref.v[i];
            const T p_loss = wp * e_p * e_p;
            const T v_loss = wv * e_v * e_v;
            const T w_loss = ww * state.omega[i] * state.omega[i];
            terms.terminal_position += p_loss;
            terms.terminal_velocity += v_loss;
            terms.terminal_angular_velocity += w_loss;
            terms.terminal += p_loss + v_loss + w_loss;
            lambda.p[i] += (T)2 * wp * e_p;
            lambda.v[i] += (T)2 * wv * e_v;
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

    template <typename T, typename TI>
    void add_terminal_loss(
        const EulerState<T, TI>& state,
        const EulerLossWeights<T>& weights,
        EulerLossTerms<T>& terms,
        EulerStateAdjoint<T>& lambda
    ){
        const auto ref = zero_tracking_reference<T>();
        add_terminal_loss<T, TI>(state, ref, weights, terms, lambda);
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
                const T action_center = weights.hover_relative_action_magnitude
                    ? initial_state.action_hover_center[action_i]
                    : weights.action_magnitude_center;
                const T centered_action = actions[step_i][action_i] - action_center;
                terms.action_magnitude += normalizer * weights.action_magnitude * centered_action * centered_action;
                action_gradients[step_i][action_i] += normalizer * (T)2 * weights.action_magnitude * centered_action;
                const T diff = actions[step_i][action_i] - previous_action[action_i];
                terms.action_smoothness += normalizer * weights.action_smoothness * diff * diff;
                action_gradients[step_i][action_i] += normalizer * (T)2 * weights.action_smoothness * diff;
                if(step_i > 0){
                    action_gradients[step_i - 1][action_i] -= normalizer * (T)2 * weights.action_smoothness * diff;
                }
                const T abs_action = std::abs(actions[step_i][action_i]);
                if(abs_action > weights.saturation_start){
                    const T sign = actions[step_i][action_i] >= (T)0 ? (T)1 : (T)-1;
                    const T excess = abs_action - weights.saturation_start;
                    terms.saturation += normalizer * weights.saturation * excess * excess;
                    action_gradients[step_i][action_i] += normalizer * (T)2 * weights.saturation * excess * sign;
                }
                previous_action[action_i] = actions[step_i][action_i];
            }
        }
    }

    template <typename T, typename TI>
    void add_barrier_loss(
        const EulerState<T, TI>& state,
        const EulerLossWeights<T>& weights,
        T normalizer,
        EulerLossTerms<T>& terms,
        EulerStateAdjoint<T>& lambda
    ){
        if(weights.velocity_barrier != (T)0){
            T norm_sq = 0;
            for(TI i = 0; i < 3; i++){
                norm_sq += state.v[i] * state.v[i];
            }
            const T safe_sq = std::max((T)1.0e-6, weights.velocity_barrier_safe * weights.velocity_barrier_safe);
            const T excess = norm_sq / safe_sq - (T)1;
            if(excess > (T)0){
                terms.velocity_barrier += normalizer * weights.velocity_barrier *
                    positive_huber<T>(excess, weights.barrier_huber_delta);
                const T grad = normalizer * weights.velocity_barrier *
                    positive_huber_grad<T>(excess, weights.barrier_huber_delta) * (T)2 / safe_sq;
                for(TI i = 0; i < 3; i++){
                    lambda.v[i] += cap_barrier_gradient<T>(grad * state.v[i], weights);
                }
            }
        }
        if(weights.angular_velocity_barrier != (T)0){
            T norm_sq = 0;
            for(TI i = 0; i < 3; i++){
                norm_sq += state.omega[i] * state.omega[i];
            }
            const T safe_sq = std::max((T)1.0e-6, weights.angular_velocity_barrier_safe * weights.angular_velocity_barrier_safe);
            const T excess = norm_sq / safe_sq - (T)1;
            if(excess > (T)0){
                terms.angular_velocity_barrier += normalizer * weights.angular_velocity_barrier *
                    positive_huber<T>(excess, weights.barrier_huber_delta);
                const T grad = normalizer * weights.angular_velocity_barrier *
                    positive_huber_grad<T>(excess, weights.barrier_huber_delta) * (T)2 / safe_sq;
                for(TI i = 0; i < 3; i++){
                    lambda.omega[i] += cap_barrier_gradient<T>(grad * state.omega[i], weights);
                }
            }
        }
        if(weights.attitude_barrier != (T)0){
            T psi = 0;
            for(TI i = 0; i < 3; i++){
                for(TI j = 0; j < 3; j++){
                    const T target = i == j ? (T)1 : (T)0;
                    const T error = state.R[i][j] - target;
                    psi += error * error;
                }
            }
            const T safe = std::max((T)1.0e-6, weights.attitude_barrier_safe);
            const T excess = psi / safe - (T)1;
            if(excess > (T)0){
                terms.attitude_barrier += normalizer * weights.attitude_barrier *
                    positive_huber<T>(excess, weights.barrier_huber_delta);
                const T grad = normalizer * weights.attitude_barrier *
                    positive_huber_grad<T>(excess, weights.barrier_huber_delta) * (T)2 / safe;
                for(TI i = 0; i < 3; i++){
                    for(TI j = 0; j < 3; j++){
                        const T target = i == j ? (T)1 : (T)0;
                        lambda.R[i][j] += cap_barrier_gradient<T>(grad * (state.R[i][j] - target), weights);
                    }
                }
            }
        }
    }

    template <typename T, typename TI>
    void add_clf_transition_loss(
        const EulerState<T, TI>& previous,
        const EulerState<T, TI>& next,
        const TrackingReference<T>& ref,
        const EulerLossWeights<T>& weights,
        T dt,
        T normalizer,
        EulerLossTerms<T>& terms,
        EulerStateAdjoint<T>& previous_lambda,
        EulerStateAdjoint<T>& next_lambda
    ){
        if(weights.clf != (T)0){
            const T V_prev = lyapunov_energy_identity<T, TI>(previous, ref, weights);
            const T V_next = lyapunov_energy_identity<T, TI>(next, ref, weights);
            const T decay = std::max((T)0, (T)1 - weights.clf_alpha * dt);
            const T target_next = decay * V_prev;
            const T delta = V_next - target_next;
            if(delta > (T)0){
                terms.clf += normalizer * weights.clf * delta * delta;
                const T grad_delta = normalizer * (T)2 * weights.clf * delta;
                add_lyapunov_energy_adjoint_identity<T, TI>(next, ref, weights, grad_delta, next_lambda);
                add_lyapunov_energy_adjoint_identity<T, TI>(
                    previous,
                    ref,
                    weights,
                    -grad_delta * decay,
                    previous_lambda
                );
            }
        }
        if(weights.outward_velocity != (T)0){
            T radial = 0;
            T e_p[3];
            T e_v[3];
            for(TI i = 0; i < 3; i++){
                e_p[i] = next.p[i] - ref.p[i];
                e_v[i] = next.v[i] - ref.v[i];
                radial += e_p[i] * e_v[i];
            }
            if(radial > (T)0){
                terms.outward_velocity += normalizer * weights.outward_velocity * radial * radial;
                const T grad = normalizer * (T)2 * weights.outward_velocity * radial;
                for(TI i = 0; i < 3; i++){
                    next_lambda.p[i] += grad * e_v[i];
                    next_lambda.v[i] += grad * e_p[i];
                }
            }
        }
        if(weights.attitude_control != (T)0){
            T e_R[3];
            attitude_identity_error_vee<T, TI>(previous, e_R);
            for(TI i = 0; i < 3; i++){
                const T omega_target_next = previous.omega[i] + dt * (
                    -weights.attitude_control_k_R * e_R[i] -
                    weights.attitude_control_k_omega * previous.omega[i]
                );
                const T error = next.omega[i] - omega_target_next;
                terms.attitude_control += normalizer * weights.attitude_control * error * error;
                next_lambda.omega[i] += normalizer * (T)2 * weights.attitude_control * error;
            }
        }
    }

    template <typename T, typename TI>
    void add_window_clf_loss(
        const EulerState<T, TI>& initial,
        const EulerState<T, TI>& terminal,
        const TrackingReference<T>& ref,
        const EulerLossWeights<T>& weights,
        T dt,
        TI horizon,
        EulerLossTerms<T>& terms,
        EulerStateAdjoint<T>& initial_lambda,
        EulerStateAdjoint<T>& terminal_lambda
    ){
        if(weights.window_clf == (T)0 || horizon <= 0){
            return;
        }
        const T V_initial = lyapunov_energy_identity<T, TI>(initial, ref, weights);
        const T V_terminal = lyapunov_energy_identity<T, TI>(terminal, ref, weights);
        const T per_step_decay = std::max((T)0, (T)1 - weights.clf_alpha * dt);
        T window_decay = (T)1;
        for(TI step_i = 0; step_i < horizon; step_i++){
            window_decay *= per_step_decay;
        }
        const T delta_raw = V_terminal - window_decay * V_initial;
        const T denom = std::max(weights.window_clf_epsilon + V_initial, (T)1e-12);
        const T delta_norm = delta_raw / denom;
        if(delta_norm > (T)0){
            const T huber = positive_huber<T>(delta_norm, weights.window_clf_huber_delta);
            const T huber_grad = weights.window_clf * positive_huber_grad<T>(delta_norm, weights.window_clf_huber_delta);
            terms.window_clf += weights.window_clf * huber;
            const T grad_terminal = huber_grad / denom;
            const T grad_initial = -huber_grad * (window_decay * denom + delta_raw) / (denom * denom);
            add_lyapunov_energy_adjoint_identity<T, TI>(terminal, ref, weights, grad_terminal, terminal_lambda);
            add_lyapunov_energy_adjoint_identity<T, TI>(initial, ref, weights, grad_initial, initial_lambda);
        }
    }

    template <typename PARAMETERS, typename T, typename TI, TI MAX_HORIZON>
    EulerLossTerms<T> rollout_loss(
        const PARAMETERS& parameters,
        const EulerState<T, TI>& initial_state,
        const T (&actions)[MAX_HORIZON][4],
        TI horizon,
        const EulerLossWeights<T>& weights,
        const TrackingReference<T>& ref,
        EulerState<T, TI> (&states)[MAX_HORIZON + 1],
        EulerStepCache<T> (&caches)[MAX_HORIZON]
    ){
        states[0] = initial_state;
        for(TI step_i = 0; step_i < horizon; step_i++){
            step<PARAMETERS, T, TI>(parameters, states[step_i], actions[step_i], states[step_i + 1], caches[step_i]);
        }
        EulerLossTerms<T> terms;
        const T dt = parameters.integration.dt;
        const T normalizer = horizon > 0 ? (T)1 / (T)horizon : (T)1;
        EulerStateAdjoint<T> unused;
        zero(unused);
        for(TI step_i = 0; step_i < horizon; step_i++){
            EulerStateAdjoint<T> state_loss;
            zero(state_loss);
            add_state_loss<T, TI>(states[step_i + 1], ref, weights, normalizer, terms, state_loss);
            add_barrier_loss<T, TI>(states[step_i + 1], weights, normalizer, terms, state_loss);
            EulerStateAdjoint<T> previous_loss;
            zero(previous_loss);
            add_clf_transition_loss<T, TI>(
                states[step_i],
                states[step_i + 1],
                ref,
                weights,
                dt,
                normalizer,
                terms,
                previous_loss,
                state_loss
            );
        }
        add_window_clf_loss<T, TI>(
            states[0],
            states[horizon],
            ref,
            weights,
            dt,
            horizon,
            terms,
            unused,
            unused
        );
        T dummy_gradients[MAX_HORIZON][4] = {};
        add_action_loss<T, TI, MAX_HORIZON>(initial_state, actions, horizon, weights, normalizer, terms, dummy_gradients);
        EulerStateAdjoint<T> terminal_loss;
        zero(terminal_loss);
        add_terminal_loss<T, TI>(states[horizon], ref, weights, terms, terminal_loss);
        return terms;
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
        const auto ref = zero_tracking_reference<T>();
        return rollout_loss<PARAMETERS, T, TI, MAX_HORIZON>(parameters, initial_state, actions, horizon, weights, ref, states, caches);
    }

    template <typename PARAMETERS, typename T, typename TI, TI MAX_HORIZON>
    EulerLossTerms<T> rollout_loss_and_gradients(
        const PARAMETERS& parameters,
        const EulerState<T, TI>& initial_state,
        const T (&actions)[MAX_HORIZON][4],
        TI horizon,
        const EulerLossWeights<T>& weights,
        const TrackingReference<T>& ref,
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
        const T dt = parameters.integration.dt;
        const T normalizer = horizon > 0 ? (T)1 / (T)horizon : (T)1;
        for(TI step_i = 0; step_i < horizon; step_i++){
            add_state_loss<T, TI>(states[step_i + 1], ref, weights, normalizer, terms, lambdas[step_i + 1]);
            add_barrier_loss<T, TI>(states[step_i + 1], weights, normalizer, terms, lambdas[step_i + 1]);
            add_clf_transition_loss<T, TI>(
                states[step_i],
                states[step_i + 1],
                ref,
                weights,
                dt,
                normalizer,
                terms,
                lambdas[step_i],
                lambdas[step_i + 1]
            );
        }
        add_window_clf_loss<T, TI>(
            states[0],
            states[horizon],
            ref,
            weights,
            dt,
            horizon,
            terms,
            lambdas[0],
            lambdas[horizon]
        );
        add_terminal_loss<T, TI>(states[horizon], ref, weights, terms, lambdas[horizon]);
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
        const auto ref = zero_tracking_reference<T>();
        return rollout_loss_and_gradients<PARAMETERS, T, TI, MAX_HORIZON>(parameters, initial_state, actions, horizon, weights, ref, states, caches, action_gradients);
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
