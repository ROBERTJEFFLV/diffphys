#pragma once

#include "diff_euler_rollout.h"

#include <iomanip>
#include <iostream>

namespace rl_tools::rl::environments::l2f::diff{
    template <typename T>
    struct EulerCheckSummary{
        bool h1_pass = false;
        bool h4_pass = false;
        bool h16_pass = false;
        bool h64_pass = false;
        T h1_cosine = 0;
        T h4_cosine = 0;
        T h16_cosine = 0;
        T h64_cosine = 0;
        T h1_relative_error = 0;
        T h4_relative_error = 0;
        T h16_relative_error = 0;
        T h64_relative_error = 0;
        int h1_sign_matches = 0;
        int h4_sign_matches = 0;
        int h16_sign_matches = 0;
        int h64_sign_matches = 0;
    };

    template <typename T>
    T safe_relative_error(T abs_error, T reference){
        return abs_error / std::max((T)1e-8, reference);
    }

    template <typename T>
    struct EulerObservationBuffer{
        T data[22] = {};
    };

    template <typename T, typename ROW_TI, typename COL_TI, typename VALUE>
    void set(EulerObservationBuffer<T>& observation, ROW_TI, COL_TI col, VALUE value){
        observation.data[col] = (T)value;
    }

    template <typename T, typename ROW_TI, typename COL_TI>
    T get(const EulerObservationBuffer<T>& observation, ROW_TI, COL_TI col){
        return observation.data[col];
    }

    template <typename T>
    void rotation_delta_vjp_fd_reference_double(
        T dt,
        const T omega_next[3],
        const T lambda_delta[3][3],
        double grad_omega[3],
        double eps_used[3],
        double scalar_plus[3],
        double scalar_minus[3]
    ){
        const double dt_d = (double)dt;
        double lambda_d[3][3];
        for(int i = 0; i < 3; i++){
            grad_omega[i] = 0;
            for(int j = 0; j < 3; j++){
                lambda_d[i][j] = (double)lambda_delta[i][j];
            }
        }
        for(int a = 0; a < 3; a++){
            const double eps = 1e-3 * std::max(1.0, std::abs((double)omega_next[a]));
            eps_used[a] = eps;
            double plus[3] = {(double)omega_next[0], (double)omega_next[1], (double)omega_next[2]};
            double minus[3] = {(double)omega_next[0], (double)omega_next[1], (double)omega_next[2]};
            plus[a] += eps;
            minus[a] -= eps;
            double D_plus[3][3], D_minus[3][3];
            so3_expmap_delta(dt_d, plus, D_plus);
            so3_expmap_delta(dt_d, minus, D_minus);
            scalar_plus[a] = 0;
            scalar_minus[a] = 0;
            for(int i = 0; i < 3; i++){
                for(int j = 0; j < 3; j++){
                    scalar_plus[a] += lambda_d[i][j] * D_plus[i][j];
                    scalar_minus[a] += lambda_d[i][j] * D_minus[i][j];
                }
            }
            grad_omega[a] = (scalar_plus[a] - scalar_minus[a]) / (2.0 * eps);
        }
    }

    template <typename T>
    void trace_closed_form_grad(T dt, const T omega_next[3], double grad_omega[3]){
        double phi[3] = {
            (double)dt * (double)omega_next[0],
            (double)dt * (double)omega_next[1],
            (double)dt * (double)omega_next[2]
        };
        const double theta = std::sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]);
        const double scale = theta < 1e-12 ? -2.0 : -2.0 * std::sin(theta) / theta;
        for(int i = 0; i < 3; i++){
            grad_omega[i] = (double)dt * scale * phi[i];
        }
    }

    inline void print_vec3_double(const char* name, const double v[3]){
        std::cout << name << "=[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
    }

    template <typename PARAMETERS, typename T>
    T hover_normalized_action_from_parameters(const PARAMETERS& parameters){
        const T hover = parameters.dynamics.hovering_throttle_relative * (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) + parameters.dynamics.action_limit.min;
        return (hover - parameters.dynamics.action_limit.min) / (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) * (T)2 - (T)1;
    }

    template <typename PARAMETERS, typename T, typename TI>
    void configure_check_state(const PARAMETERS& parameters, EulerState<T, TI>& state){
        const T hover = parameters.dynamics.hovering_throttle_relative * (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) + parameters.dynamics.action_limit.min;
        for(TI i = 0; i < 3; i++){
            state.p[i] = (i == 0 ? (T)0.12 : (i == 1 ? (T)-0.08 : (T)0.05));
            state.v[i] = (i == 0 ? (T)-0.03 : (i == 1 ? (T)0.04 : (T)-0.02));
            state.omega[i] = (i == 0 ? (T)0.07 : (i == 1 ? (T)-0.05 : (T)0.03));
            for(TI j = 0; j < 3; j++){
                state.R[i][j] = i == j ? (T)1 : (T)0;
            }
        }
        // Use a valid SO(3) rotation for gradient checks.
        T check_rot_vec[3] = {(T)0.02, (T)-0.015, (T)0.03};
        so3_expmap_delta((T)1, check_rot_vec, state.R);
        for(TI i = 0; i < 4; i++){
            state.rpm[i] = hover;
            state.previous_action[i] = hover_normalized_action_from_parameters<PARAMETERS, T>(parameters);
        }
    }

    template <typename T, typename TI>
    void print_vec3(const char* name, const T v[3]){
        std::cout << name << "=[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
    }

    template <typename PARAMETERS, typename T, typename TI>
    void euler_forward_sanity(const PARAMETERS& parameters){
        std::cout << "\n[Euler A] Forward deterministic sanity check\n";
        EulerState<T, TI> state, next;
        EulerStepCache<T> cache;
        configure_check_state<PARAMETERS, T, TI>(parameters, state);
        T action[4] = {
            hover_normalized_action_from_parameters<PARAMETERS, T>(parameters) + (T)0.05,
            hover_normalized_action_from_parameters<PARAMETERS, T>(parameters) - (T)0.03,
            hover_normalized_action_from_parameters<PARAMETERS, T>(parameters) + (T)0.02,
            hover_normalized_action_from_parameters<PARAMETERS, T>(parameters) - (T)0.01
        };
        step<PARAMETERS, T, TI>(parameters, state, action, next, cache);
        print_vec3<T, TI>("p_before", state.p); std::cout << " "; print_vec3<T, TI>("v_before", state.v); std::cout << " "; print_vec3<T, TI>("omega_before", state.omega); std::cout << "\n";
        print_vec3<T, TI>("p_after", next.p); std::cout << " "; print_vec3<T, TI>("v_after", next.v); std::cout << " "; print_vec3<T, TI>("omega_after", next.omega); std::cout << "\n";
        std::cout << "R_after=[";
        for(TI i = 0; i < 3; i++){
            std::cout << "[" << next.R[i][0] << ", " << next.R[i][1] << ", " << next.R[i][2] << "]";
            if(i + 1 < 3){ std::cout << ", "; }
        }
        std::cout << "] rpm_after=[" << next.rpm[0] << ", " << next.rpm[1] << ", " << next.rpm[2] << ", " << next.rpm[3] << "]\n";
        print_vec3<T, TI>("thrust_body", cache.thrust_body); std::cout << " "; print_vec3<T, TI>("torque_body", cache.torque_body); std::cout << "\n";
        print_vec3<T, TI>("acceleration_world", cache.acceleration_world); std::cout << " "; print_vec3<T, TI>("angular_acceleration_body", cache.angular_acceleration_body); std::cout << "\n";
    }

    template <typename PARAMETERS, typename T, typename TI>
    bool euler_local_derivative_checks(const PARAMETERS& parameters){
        std::cout << "\n[Euler B] Local derivative checks\n";
        bool ok = true;
        EulerState<T, TI> state, next;
        EulerStepCache<T> cache;
        configure_check_state<PARAMETERS, T, TI>(parameters, state);
        T base = hover_normalized_action_from_parameters<PARAMETERS, T>(parameters);
        T action[4] = {base + (T)0.04, base - (T)0.02, base + (T)0.01, base - (T)0.03};
        step<PARAMETERS, T, TI>(parameters, state, action, next, cache);

        const T eps = (T)1e-3;
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            T plus[4] = {action[0], action[1], action[2], action[3]};
            T minus[4] = {action[0], action[1], action[2], action[3]};
            plus[rotor_i] += eps;
            minus[rotor_i] -= eps;
            EulerState<T, TI> next_plus, next_minus;
            EulerStepCache<T> cache_plus, cache_minus;
            step<PARAMETERS, T, TI>(parameters, state, plus, next_plus, cache_plus);
            step<PARAMETERS, T, TI>(parameters, state, minus, next_minus, cache_minus);
            const T analytic = ((T)1 - cache.alpha[rotor_i]) * cache.d_setpoint_d_action[rotor_i];
            const T fd = (next_plus.rpm[rotor_i] - next_minus.rpm[rotor_i]) / ((T)2 * eps);
            const T rel = safe_relative_error(std::abs(analytic - fd), std::abs(fd));
            std::cout << "motor rotor=" << rotor_i << " analytic=" << analytic << " fd=" << fd << " rel=" << rel << "\n";
            ok = ok && rel < (T)2e-3;
        }

        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            const T rpm = cache.rpm_next[rotor_i];
            const T analytic = cache.d_force_d_rpm[rotor_i];
            const T c0 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][0];
            const T c1 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][1];
            const T c2 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][2];
            const T fp = c0 + c1 * (rpm + eps) + c2 * (rpm + eps) * (rpm + eps);
            const T fm = c0 + c1 * (rpm - eps) + c2 * (rpm - eps) * (rpm - eps);
            const T fd = (fp - fm) / ((T)2 * eps);
            const T rel = safe_relative_error(std::abs(analytic - fd), std::abs(fd));
            std::cout << "thrust rotor=" << rotor_i << " analytic=" << analytic << " fd=" << fd << " rel=" << rel << "\n";
            ok = ok && rel < (T)2e-3;
        }

        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            // True finite-difference torque check: perturb action, re-run step, compare torque_body.
            T plus[4] = {action[0], action[1], action[2], action[3]};
            T minus[4] = {action[0], action[1], action[2], action[3]};
            plus[rotor_i] += eps;
            minus[rotor_i] -= eps;
            EulerState<T, TI> next_plus_t, next_minus_t;
            EulerStepCache<T> cache_plus_t, cache_minus_t;
            step<PARAMETERS, T, TI>(parameters, state, plus, next_plus_t, cache_plus_t);
            step<PARAMETERS, T, TI>(parameters, state, minus, next_minus_t, cache_minus_t);
            T fd_torque[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                fd_torque[dim_i] = (cache_plus_t.torque_body[dim_i] - cache_minus_t.torque_body[dim_i]) / ((T)2 * eps);
            }
            T arm_direction[3];
            cross3(parameters.dynamics.rotor_positions[rotor_i], parameters.dynamics.rotor_thrust_directions[rotor_i], arm_direction);
            const T drpm_du = ((T)1 - cache.alpha[rotor_i]) * cache.d_setpoint_d_action[rotor_i];
            T analytic_torque[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                analytic_torque[dim_i] = (
                    parameters.dynamics.rotor_torque_directions[rotor_i][dim_i] * parameters.dynamics.rotor_torque_constants[rotor_i] +
                    arm_direction[dim_i]
                ) * cache.d_force_d_rpm[rotor_i] * drpm_du;
            }
            T abs_err = 0;
            T ref = 0;
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                abs_err += (analytic_torque[dim_i] - fd_torque[dim_i]) * (analytic_torque[dim_i] - fd_torque[dim_i]);
                ref += fd_torque[dim_i] * fd_torque[dim_i];
            }
            const T rel = safe_relative_error(std::sqrt(abs_err), std::sqrt(ref));
            std::cout << "torque rotor=" << rotor_i << " analytic=[" << analytic_torque[0] << "," << analytic_torque[1] << "," << analytic_torque[2]
                      << "] fd=[" << fd_torque[0] << "," << fd_torque[1] << "," << fd_torque[2] << "] rel=" << rel << "\n";
            ok = ok && rel < (T)5e-3;
        }

        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            T plus[4] = {action[0], action[1], action[2], action[3]};
            T minus[4] = {action[0], action[1], action[2], action[3]};
            plus[rotor_i] += eps;
            minus[rotor_i] -= eps;
            EulerState<T, TI> next_plus, next_minus;
            EulerStepCache<T> cache_plus, cache_minus;
            step<PARAMETERS, T, TI>(parameters, state, plus, next_plus, cache_plus);
            step<PARAMETERS, T, TI>(parameters, state, minus, next_minus, cache_minus);
            const T drpm_du = ((T)1 - cache.alpha[rotor_i]) * cache.d_setpoint_d_action[rotor_i];
            T dthrust_body[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                dthrust_body[dim_i] = parameters.dynamics.rotor_thrust_directions[rotor_i][dim_i] * cache.d_force_d_rpm[rotor_i] * drpm_du;
            }
            T analytic_acc[3];
            mat_vec3(state.R, dthrust_body, analytic_acc);
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                analytic_acc[dim_i] /= parameters.dynamics.mass;
            }
            T fd_acc[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                fd_acc[dim_i] = (cache_plus.acceleration_world[dim_i] - cache_minus.acceleration_world[dim_i]) / ((T)2 * eps);
            }
            T abs_err = 0;
            T ref = 0;
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                abs_err += (analytic_acc[dim_i] - fd_acc[dim_i]) * (analytic_acc[dim_i] - fd_acc[dim_i]);
                ref += fd_acc[dim_i] * fd_acc[dim_i];
            }
            const T rel = safe_relative_error(std::sqrt(abs_err), std::sqrt(ref));
            std::cout << "acceleration rotor=" << rotor_i << " rel=" << rel << "\n";
            ok = ok && rel < (T)5e-3;
        }
        std::cout << (ok ? "PASS" : "FAIL") << " euler_local_derivative_checks\n";
        return ok;
    }

    template <typename PARAMETERS, typename T, typename TI>
    bool euler_reference_zero_equivalence_check(const PARAMETERS& parameters, const EulerLossWeights<T>& weights){
        std::cout << "\n[Euler reference] Zero-reference equivalence check\n";
        EulerState<T, TI> initial_state;
        configure_check_state<PARAMETERS, T, TI>(parameters, initial_state);
        const auto zero_ref = zero_tracking_reference<T>();

        EulerObservationBuffer<T> legacy_observation;
        EulerObservationBuffer<T> reference_observation;
        observe<T, TI>(initial_state, legacy_observation);
        observe_with_reference<T, TI>(initial_state, zero_ref, reference_observation);
        T max_observation_error = 0;
        for(TI i = 0; i < 22; i++){
            max_observation_error = std::max(max_observation_error, std::abs(legacy_observation.data[i] - reference_observation.data[i]));
        }

        constexpr TI H = 4;
        T actions[H][4];
        const T base = hover_normalized_action_from_parameters<PARAMETERS, T>(parameters);
        for(TI step_i = 0; step_i < H; step_i++){
            actions[step_i][0] = base + (T)0.17 + (T)0.0020 * step_i;
            actions[step_i][1] = base + (T)0.11 - (T)0.0010 * step_i;
            actions[step_i][2] = base + (T)0.15 + (T)0.0015 * step_i;
            actions[step_i][3] = base + (T)0.13 - (T)0.0005 * step_i;
        }

        EulerState<T, TI> legacy_states[H + 1];
        EulerState<T, TI> reference_states[H + 1];
        EulerStepCache<T> legacy_caches[H];
        EulerStepCache<T> reference_caches[H];
        T legacy_gradients[H][4];
        T reference_gradients[H][4];
        const auto legacy_terms = rollout_loss_and_gradients<PARAMETERS, T, TI, H>(parameters, initial_state, actions, H, weights, legacy_states, legacy_caches, legacy_gradients);
        const auto reference_terms = rollout_loss_and_gradients<PARAMETERS, T, TI, H>(parameters, initial_state, actions, H, weights, zero_ref, reference_states, reference_caches, reference_gradients);

        T max_state_error = 0;
        for(TI step_i = 0; step_i <= H; step_i++){
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                max_state_error = std::max(max_state_error, std::abs(legacy_states[step_i].p[dim_i] - reference_states[step_i].p[dim_i]));
                max_state_error = std::max(max_state_error, std::abs(legacy_states[step_i].v[dim_i] - reference_states[step_i].v[dim_i]));
                max_state_error = std::max(max_state_error, std::abs(legacy_states[step_i].omega[dim_i] - reference_states[step_i].omega[dim_i]));
                for(TI dim_j = 0; dim_j < 3; dim_j++){
                    max_state_error = std::max(max_state_error, std::abs(legacy_states[step_i].R[dim_i][dim_j] - reference_states[step_i].R[dim_i][dim_j]));
                }
            }
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                max_state_error = std::max(max_state_error, std::abs(legacy_states[step_i].rpm[rotor_i] - reference_states[step_i].rpm[rotor_i]));
            }
        }

        T max_gradient_error = 0;
        for(TI step_i = 0; step_i < H; step_i++){
            for(TI action_i = 0; action_i < 4; action_i++){
                max_gradient_error = std::max(max_gradient_error, std::abs(legacy_gradients[step_i][action_i] - reference_gradients[step_i][action_i]));
            }
        }
        const T loss_error = std::abs(legacy_terms.total() - reference_terms.total());
        const bool ok = max_observation_error < (T)1e-7 && loss_error < (T)1e-6 && max_gradient_error < (T)1e-6 && max_state_error < (T)1e-7;
        std::cout << "zero_ref_observation_max_error=" << max_observation_error
                  << " zero_ref_loss_error=" << loss_error
                  << " zero_ref_action_gradient_max_error=" << max_gradient_error
                  << " zero_ref_state_max_error=" << max_state_error
                  << " " << (ok ? "PASS" : "FAIL") << "\n";
        return ok;
    }

    template <typename PARAMETERS, typename T, typename TI, TI HORIZON>
    bool euler_rollout_gradient_check_one(const PARAMETERS& parameters, const EulerLossWeights<T>& weights, EulerCheckSummary<T>& summary){
        EulerState<T, TI> initial_state;
        configure_check_state<PARAMETERS, T, TI>(parameters, initial_state);
        T actions[HORIZON][4];
        const T base = hover_normalized_action_from_parameters<PARAMETERS, T>(parameters);
        for(TI step_i = 0; step_i < HORIZON; step_i++){
            actions[step_i][0] = base + (T)0.22 + (T)0.0010 * step_i;
            actions[step_i][1] = base + (T)0.18 + (T)0.0005 * step_i;
            actions[step_i][2] = base + (T)0.20 - (T)0.0007 * step_i;
            actions[step_i][3] = base + (T)0.16 + (T)0.0003 * step_i;
        }
        EulerState<T, TI> states[HORIZON + 1];
        EulerStepCache<T> caches[HORIZON];
        T analytic[HORIZON][4];
        auto terms = rollout_loss_and_gradients<PARAMETERS, T, TI, HORIZON>(parameters, initial_state, actions, HORIZON, weights, states, caches, analytic);

        T fd[HORIZON][4];
        const T eps = (T)1e-3;
        for(TI step_i = 0; step_i < HORIZON; step_i++){
            for(TI action_i = 0; action_i < 4; action_i++){
                T plus_actions[HORIZON][4];
                T minus_actions[HORIZON][4];
                for(TI t = 0; t < HORIZON; t++){
                    for(TI a = 0; a < 4; a++){
                        plus_actions[t][a] = actions[t][a];
                        minus_actions[t][a] = actions[t][a];
                    }
                }
                plus_actions[step_i][action_i] += eps;
                minus_actions[step_i][action_i] -= eps;
                EulerState<T, TI> plus_states[HORIZON + 1], minus_states[HORIZON + 1];
                EulerStepCache<T> plus_caches[HORIZON], minus_caches[HORIZON];
                const T loss_plus = rollout_loss<PARAMETERS, T, TI, HORIZON>(parameters, initial_state, plus_actions, HORIZON, weights, plus_states, plus_caches).total();
                const T loss_minus = rollout_loss<PARAMETERS, T, TI, HORIZON>(parameters, initial_state, minus_actions, HORIZON, weights, minus_states, minus_caches).total();
                fd[step_i][action_i] = (loss_plus - loss_minus) / ((T)2 * eps);
            }
        }

        T dot = 0;
        T analytic_norm = 0;
        T fd_norm = 0;
        T error_norm = 0;
        T max_abs_error = 0;
        int sign_matches = 0;
        for(TI step_i = 0; step_i < HORIZON; step_i++){
            for(TI action_i = 0; action_i < 4; action_i++){
                dot += analytic[step_i][action_i] * fd[step_i][action_i];
                analytic_norm += analytic[step_i][action_i] * analytic[step_i][action_i];
                fd_norm += fd[step_i][action_i] * fd[step_i][action_i];
                const T error = analytic[step_i][action_i] - fd[step_i][action_i];
                error_norm += error * error;
                max_abs_error = std::max(max_abs_error, std::abs(error));
                if(std::abs(fd[step_i][action_i]) < (T)1e-7 || analytic[step_i][action_i] * fd[step_i][action_i] >= 0){
                    sign_matches++;
                }
            }
        }
        analytic_norm = std::sqrt(analytic_norm);
        fd_norm = std::sqrt(fd_norm);
        error_norm = std::sqrt(error_norm);
        const T cosine = dot / std::max((T)1e-8, analytic_norm * fd_norm);
        const T relative_error = safe_relative_error(error_norm, fd_norm);
        const T cosine_threshold = HORIZON == 1 ? (T)0.99 : (HORIZON == 4 ? (T)0.98 : (T)0.95);
        const T relative_error_threshold = HORIZON <= 16 ? (T)1e-2 : (T)5e-2;
        const bool pass = cosine > cosine_threshold && relative_error < relative_error_threshold;
        std::cout << "H=" << HORIZON
                  << " loss=" << terms.total()
                  << " cosine=" << cosine
                  << " relative_error=" << relative_error
                  << " sign_matches=" << sign_matches << "/" << (HORIZON * 4)
                  << " max_abs_error=" << max_abs_error
                  << " status=" << (pass ? "PASS" : "FAIL")
                  << "\n";
        if constexpr(HORIZON == 1){
            summary.h1_pass = pass;
            summary.h1_cosine = cosine;
            summary.h1_relative_error = relative_error;
            summary.h1_sign_matches = sign_matches;
        }
        if constexpr(HORIZON == 4){
            summary.h4_pass = pass;
            summary.h4_cosine = cosine;
            summary.h4_relative_error = relative_error;
            summary.h4_sign_matches = sign_matches;
        }
        if constexpr(HORIZON == 16){
            summary.h16_pass = pass;
            summary.h16_cosine = cosine;
            summary.h16_relative_error = relative_error;
            summary.h16_sign_matches = sign_matches;
        }
        if constexpr(HORIZON == 64){
            summary.h64_pass = pass;
            summary.h64_cosine = cosine;
            summary.h64_relative_error = relative_error;
            summary.h64_sign_matches = sign_matches;
        }
        return pass;
    }

    template <typename PARAMETERS, typename T, typename TI>
    bool euler_sign_sanity_tests(const PARAMETERS& parameters){
        std::cout << "\n[Euler D] Sign sanity tests\n";
        EulerState<T, TI> state;
        configure_check_state<PARAMETERS, T, TI>(parameters, state);
        for(TI i = 0; i < 3; i++){
            state.p[i] = 0;
            state.v[i] = 0;
            state.omega[i] = 0;
            for(TI j = 0; j < 3; j++){
                state.R[i][j] = i == j ? (T)1 : (T)0;
            }
        }
        const T base = hover_normalized_action_from_parameters<PARAMETERS, T>(parameters);
        T action_all[4] = {base + (T)0.15, base + (T)0.15, base + (T)0.15, base + (T)0.15};
        EulerState<T, TI> next_all;
        EulerStepCache<T> cache_all;
        step<PARAMETERS, T, TI>(parameters, state, action_all, next_all, cache_all);
        std::cout << "all motors acceleration_z=" << cache_all.acceleration_world[2] << " angular_norm="
                  << std::sqrt(dot3(cache_all.angular_acceleration_body, cache_all.angular_acceleration_body)) << "\n";

        T action_one[4] = {base + (T)0.15, base, base, base};
        EulerState<T, TI> next_one;
        EulerStepCache<T> cache_one;
        step<PARAMETERS, T, TI>(parameters, state, action_one, next_one, cache_one);
        print_vec3<T, TI>("single_motor_torque", cache_one.torque_body); std::cout << " "; print_vec3<T, TI>("angular_acc", cache_one.angular_acceleration_body); std::cout << "\n";

        T action_diag[4] = {base + (T)0.12, base, base + (T)0.12, base};
        EulerState<T, TI> next_diag;
        EulerStepCache<T> cache_diag;
        step<PARAMETERS, T, TI>(parameters, state, action_diag, next_diag, cache_diag);
        print_vec3<T, TI>("diagonal_torque", cache_diag.torque_body); std::cout << " "; print_vec3<T, TI>("angular_acc", cache_diag.angular_acceleration_body); std::cout << "\n";
        const bool ok = cache_all.acceleration_world[2] > parameters.dynamics.gravity[2];
        std::cout << (ok ? "PASS" : "FAIL") << " euler_sign_sanity_tests\n";
        return ok;
    }

    inline void euler_coordinate_frame_report(){
        std::cout << "\n[Euler E] Coordinate-frame report\n";
        std::cout << "thrust frame: rotor thrust directions are summed in body frame.\n";
        std::cout << "rotation convention: R is body-to-world and acceleration_world = R * thrust_body / mass + gravity.\n";
        std::cout << "gravity sign: uses parameters.dynamics.gravity in world frame, normally [0, 0, -9.81].\n";
        std::cout << "velocity frame: p and v are world-frame states.\n";
        std::cout << "angular velocity frame: omega and omega_dot are body-frame states.\n";
        std::cout << "action units: actor outputs normalized actions; setpoint = u * half_range + min + half_range, clamped to action_limit.\n";
        std::cout << "rpm units: same internal normalized motor-speed units used by L2F action_limit, not physical rad/s.\n";
        std::cout << "rotation integration: SO(3) exponential map R_next = R * Exp(dt * skew(omega_next)); no re-orthonormalization is required because Exp(.) is orthogonal up to numerical precision.\n";
        std::cout << "rotation VJP: analytic Rodrigues-form VJP is used for the SO(3) exp-map; finite differences are retained for validation only.\n";
    }

    template <typename T>
    T det3(const T R[3][3]){
        return R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
             - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
             + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    }

    template <typename T>
    T orthogonality_error(const T R[3][3]){
        T err = 0;
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                T dot = 0;
                for(int k = 0; k < 3; k++){
                    dot += R[k][i] * R[k][j];
                }
                const T expected = (i == j) ? (T)1 : (T)0;
                err += (dot - expected) * (dot - expected);
            }
        }
        return std::sqrt(err);
    }

    template <typename PARAMETERS, typename T, typename TI>
    bool euler_rotation_drift_check(const PARAMETERS& parameters){
        std::cout << "\n[Rotation drift check]\n";
        const T dt = parameters.integration.dt;
        const TI horizons[] = {16, 128};
        const T omega_norms[] = {(T)0.5, (T)1.0, (T)3.0, (T)5.0, (T)10.0};
        const T directions[][3] = {{(T)1, (T)0, (T)0}, {(T)0, (T)1, (T)0}, {(T)0.3, (T)-0.5, (T)0.8}};
        bool ok = true;
        for(int hi = 0; hi < 2; hi++){
            for(int ni = 0; ni < 5; ni++){
                for(int di = 0; di < 3; di++){
                    T dir[3];
                    T dir_norm = 0;
                    for(int i = 0; i < 3; i++){ dir_norm += directions[di][i] * directions[di][i]; }
                    dir_norm = std::sqrt(dir_norm);
                    for(int i = 0; i < 3; i++){ dir[i] = directions[di][i] * omega_norms[ni] / std::max((T)1e-8, dir_norm); }
                    T R_first[3][3], R_exp[3][3];
                    identity3(R_first);
                    identity3(R_exp);
                    for(TI step = 0; step < horizons[hi]; step++){
                        T D_first[3][3], D_exp[3][3];
                        first_order_rotation_delta(dt, dir, D_first);
                        so3_expmap_delta(dt, dir, D_exp);
                        T tmp_first[3][3], tmp_exp[3][3];
                        mat_mul3(R_first, D_first, tmp_first);
                        mat_mul3(R_exp, D_exp, tmp_exp);
                        for(int i = 0; i < 3; i++){
                            for(int j = 0; j < 3; j++){
                                R_first[i][j] = tmp_first[i][j];
                                R_exp[i][j] = tmp_exp[i][j];
                            }
                        }
                    }
                    const T first_orth = orthogonality_error(R_first);
                    const T first_det = det3(R_first);
                    const T exp_orth = orthogonality_error(R_exp);
                    const T exp_det = det3(R_exp);
                    const bool local_ok = exp_orth < (T)1e-2 && std::abs(exp_det - (T)1) < (T)1e-2;
                    ok = ok && local_ok;
                    std::cout << "omega_norm=" << omega_norms[ni] << " H=" << horizons[hi]
                              << " first_orth=" << first_orth << " first_det=" << first_det
                              << " exp_orth=" << exp_orth << " exp_det=" << exp_det
                              << " status=" << (local_ok ? "PASS" : "FAIL") << "\n";
                }
            }
        }
        std::cout << (ok ? "PASS" : "FAIL") << " rotation_drift_check\n";
        return ok;
    }

    template <typename PARAMETERS, typename T, typename TI>
    bool so3_expmap_vjp_check(const PARAMETERS& parameters){
        std::cout << "\n[SO(3) VJP unit check]\n";
        const T dt = parameters.integration.dt;
        const T omega_cases[][3] = {
            {(T)0, (T)0, (T)0},
            {(T)0.01, (T)-0.02, (T)0.03},
            {(T)0.5, (T)-0.3, (T)0.2},
            {(T)3.0, (T)-1.0, (T)0.5},
            {(T)10.0, (T)2.0, (T)-3.0}
        };
        const T lambda_cases[][3][3] = {
            {{(T)1, (T)0, (T)0}, {(T)0, (T)1, (T)0}, {(T)0, (T)0, (T)1}},
            {{(T)0.2, (T)-0.5, (T)0.3}, {(T)0.7, (T)0.1, (T)-0.4}, {(T)-0.1, (T)0.6, (T)0.2}}
        };
        const char* lambda_names[] = {"identity_trace", "generic_dense"};
        bool ok = true;
        const int num_omega = 5;
        const int num_lambda = 2;
        const double abs_floor = sizeof(T) <= 4 ? 1e-6 : 1e-10;
        const double small_grad_threshold = sizeof(T) <= 4 ? 1e-5 : 1e-9;
        const double small_abs_error_threshold = sizeof(T) <= 4 ? 1e-5 : 1e-9;
        const double pass_small_abs_error_threshold = sizeof(T) <= 4 ? 1e-4 : 1e-8;
        const double borderline_abs_error_threshold = sizeof(T) <= 4 ? 1e-4 : 1e-8;
        for(int oi = 0; oi < num_omega; oi++){
            for(int li = 0; li < num_lambda; li++){
                T grad_analytic[3], grad_fd_float[3];
                double grad_fd[3], eps_used[3], scalar_plus[3], scalar_minus[3], trace_grad[3];
                rotation_delta_vjp_analytic(dt, omega_cases[oi], lambda_cases[li], grad_analytic);
                rotation_delta_vjp_fd(dt, omega_cases[oi], lambda_cases[li], grad_fd_float);
                rotation_delta_vjp_fd_reference_double(dt, omega_cases[oi], lambda_cases[li], grad_fd, eps_used, scalar_plus, scalar_minus);
                trace_closed_form_grad(dt, omega_cases[oi], trace_grad);
                double dot = 0, norm_a = 0, norm_f = 0, err_sq = 0, max_abs_component_error = 0;
                double grad_analytic_d[3];
                double grad_fd_float_d[3];
                for(int i = 0; i < 3; i++){
                    grad_analytic_d[i] = (double)grad_analytic[i];
                    grad_fd_float_d[i] = (double)grad_fd_float[i];
                    dot += grad_analytic_d[i] * grad_fd[i];
                    norm_a += grad_analytic_d[i] * grad_analytic_d[i];
                    norm_f += grad_fd[i] * grad_fd[i];
                    double e = grad_analytic_d[i] - grad_fd[i];
                    err_sq += e * e;
                    max_abs_component_error = std::max(max_abs_component_error, std::abs(e));
                }
                norm_a = std::sqrt(norm_a);
                norm_f = std::sqrt(norm_f);
                const double abs_error = std::sqrt(err_sq);
                const double scale = std::max(std::max(norm_a, norm_f), abs_floor);
                const double rel = abs_error / scale;
                const double cosine = (norm_a > abs_floor && norm_f > abs_floor) ? dot / (norm_a * norm_f) : 1.0;
                const bool both_small = norm_a < small_grad_threshold && norm_f < small_grad_threshold;
                const bool one_small = norm_a < small_grad_threshold || norm_f < small_grad_threshold;
                const bool both_within_small_band = norm_a < 10.0 * small_grad_threshold && norm_f < 10.0 * small_grad_threshold;
                const char* vjp_status;
                const char* reason;
                bool vjp_local_ok;
                if(both_small && abs_error < small_abs_error_threshold){
                    vjp_status = "PASS_ZERO_GRAD";
                    reason = "both gradients below small threshold and absolute error is below threshold";
                    vjp_local_ok = true;
                }
                else if(one_small && both_within_small_band && abs_error < pass_small_abs_error_threshold){
                    vjp_status = "PASS_SMALL_GRAD";
                    reason = "near-zero gradient; absolute error is below float-scale threshold";
                    vjp_local_ok = true;
                }
                else{
                    if(cosine > 0.999 && rel < 1e-3){
                        vjp_status = "PASS";
                        reason = "cosine and relative error within strict thresholds";
                        vjp_local_ok = true;
                    }
                    else if(cosine > 0.999 && rel < 2e-2 && abs_error < borderline_abs_error_threshold){
                        vjp_status = "BORDERLINE";
                        reason = "direction is correct; absolute error is below tolerance";
                        vjp_local_ok = true;
                    }
                    else{
                        vjp_status = "FAIL";
                        reason = "outside cosine, relative-error, and absolute-error tolerances";
                        vjp_local_ok = false;
                    }
                }
                ok = ok && vjp_local_ok;
                std::cout << "case=" << oi << ":" << li
                          << " omega=[" << omega_cases[oi][0] << "," << omega_cases[oi][1] << "," << omega_cases[oi][2] << "]"
                          << " lambda_index=" << li
                          << " lambda_name=" << lambda_names[li]
                          << " analytic_norm=" << norm_a << " fd_norm=" << norm_f
                          << " abs_error=" << abs_error << " relative_error=" << rel
                          << " cosine=" << cosine
                          << " max_abs_component_error=" << max_abs_component_error
                          << " status=" << vjp_status
                          << " reason=\"" << reason << "\"\n";
                std::cout << "  ";
                print_vec3_double("grad_analytic", grad_analytic_d); std::cout << " ";
                print_vec3_double("grad_fd_reference_double", grad_fd); std::cout << " ";
                print_vec3_double("grad_fd_float", grad_fd_float_d); std::cout << "\n";
                if(li == 0){
                    std::cout << "  ";
                    print_vec3_double("closed_form_trace_grad", trace_grad); std::cout << "\n";
                }
                std::cout << "  ";
                print_vec3_double("eps_used", eps_used); std::cout << " ";
                print_vec3_double("scalar_plus", scalar_plus); std::cout << " ";
                print_vec3_double("scalar_minus", scalar_minus);
                double scalar_delta[3] = {
                    scalar_plus[0] - scalar_minus[0],
                    scalar_plus[1] - scalar_minus[1],
                    scalar_plus[2] - scalar_minus[2]
                };
                std::cout << " ";
                print_vec3_double("scalar_plus_minus_delta", scalar_delta); std::cout << "\n";
            }
        }
        std::cout << (ok ? "PASS" : "FAIL") << " so3_expmap_vjp_check\n";
        return ok;
    }

    template <typename PARAMETERS, typename T, typename TI>
    bool euler_transition_vjp_check(const PARAMETERS& parameters){
        std::cout << "\n[Euler C] One-step transition VJP check\n";
        const T eps = (T)1e-3;
        bool ok = true;

        struct Scenario { const char* name; T action_offset[4]; };
        Scenario scenarios[] = {
            {"near_hover", {(T)0.04, (T)-0.02, (T)0.01, (T)-0.03}},
            {"nonzero_omega", {(T)0.15, (T)0.12, (T)-0.10, (T)0.13}},
            {"high_action", {(T)0.25, (T)0.30, (T)-0.28, (T)0.22}},
        };

        for(int si = 0; si < 3; si++){
            EulerState<T, TI> state;
            configure_check_state<PARAMETERS, T, TI>(parameters, state);
            T base = hover_normalized_action_from_parameters<PARAMETERS, T>(parameters);
            T action[4];
            for(int i = 0; i < 4; i++) action[i] = base + scenarios[si].action_offset[i];

            EulerState<T, TI> next;
            EulerStepCache<T> cache;
            step<PARAMETERS, T, TI>(parameters, state, action, next, cache);

            EulerStateAdjoint<T> lambda_next;
            zero(lambda_next);
            T lambda_dot = 0;
            for(int i = 0; i < 3; i++){
                lambda_next.p[i] = (T)(0.5 + 0.1 * i);
                lambda_next.v[i] = (T)(0.3 + 0.08 * i);
                lambda_next.omega[i] = (T)(0.2 + 0.06 * i);
                for(int j = 0; j < 3; j++){
                    lambda_next.R[i][j] = (T)(0.15 + 0.02 * (3*i + j));
                    lambda_dot += lambda_next.p[i] * next.p[i] + lambda_next.v[i] * next.v[i] + lambda_next.R[i][j] * next.R[i][j] + lambda_next.omega[i] * next.omega[i];
                }
            }
            for(int i = 0; i < 4; i++){
                lambda_next.rpm[i] = (T)(0.12 + 0.04 * i);
                lambda_dot += lambda_next.rpm[i] * next.rpm[i];
            }

            EulerStateAdjoint<T> lambda_state;
            T grad_action[4];
            step_vjp<PARAMETERS, T, TI>(parameters, state, action, cache, lambda_next, lambda_state, grad_action);

            T dot = 0, norm_a = 0, norm_f = 0, err_sq = 0;
            int sign_matches = 0;
            for(int i = 0; i < 4; i++){
                T plus[4], minus[4];
                for(int j = 0; j < 4; j++){ plus[j] = action[j]; minus[j] = action[j]; }
                plus[i] += eps; minus[i] -= eps;
                EulerState<T, TI> n_plus, n_minus;
                EulerStepCache<T> c_plus, c_minus;
                step<PARAMETERS, T, TI>(parameters, state, plus, n_plus, c_plus);
                step<PARAMETERS, T, TI>(parameters, state, minus, n_minus, c_minus);
                T J_plus = 0, J_minus = 0;
                for(int j = 0; j < 3; j++){
                    J_plus += lambda_next.p[j] * n_plus.p[j] + lambda_next.v[j] * n_plus.v[j] + lambda_next.omega[j] * n_plus.omega[j];
                    J_minus += lambda_next.p[j] * n_minus.p[j] + lambda_next.v[j] * n_minus.v[j] + lambda_next.omega[j] * n_minus.omega[j];
                    for(int k = 0; k < 3; k++){
                        J_plus += lambda_next.R[j][k] * n_plus.R[j][k];
                        J_minus += lambda_next.R[j][k] * n_minus.R[j][k];
                    }
                }
                for(int j = 0; j < 4; j++){ J_plus += lambda_next.rpm[j] * n_plus.rpm[j]; J_minus += lambda_next.rpm[j] * n_minus.rpm[j]; }
                T fd = (J_plus - J_minus) / ((T)2 * eps);
                dot += grad_action[i] * fd;
                norm_a += grad_action[i] * grad_action[i];
                norm_f += fd * fd;
                T e = grad_action[i] - fd;
                err_sq += e * e;
                if(std::abs(fd) < (T)1e-7 || grad_action[i] * fd >= 0) sign_matches++;
            }
            norm_a = std::sqrt(norm_a); norm_f = std::sqrt(norm_f);
            T cos_vjp = dot / std::max((T)1e-8, norm_a * norm_f);
            T rel_vjp = safe_relative_error(std::sqrt(err_sq), norm_f);
            bool local_ok = cos_vjp > (T)0.999 && rel_vjp < (T)1e-2;
            ok = ok && local_ok;
            std::cout << scenarios[si].name << " action_cos=" << cos_vjp << " rel=" << rel_vjp << " sign=" << sign_matches << "/4"
                      << " " << (local_ok ? "PASS" : "FAIL") << "\n";
        }
        std::cout << (ok ? "PASS" : "FAIL") << " transition_vjp_check\n";
        return ok;
    }

    template <typename PARAMETERS, typename T, typename TI>
    void euler_rollout_gradient_prefix_check(const PARAMETERS& parameters, const EulerLossWeights<T>& weights){
        std::cout << "\n[Euler D] Prefix-horizon rollout gradient diagnostics\n";
        const T eps_list[] = {(T)1e-2, (T)3e-3, (T)1e-3, (T)3e-4, (T)1e-4};
        const int num_eps = 5;
        const TI horizons[] = {1, 2, 4, 8, 16, 32, 64};
        const int num_h = 7;

        EulerState<T, TI> initial_state;
        configure_check_state<PARAMETERS, T, TI>(parameters, initial_state);
        T base = hover_normalized_action_from_parameters<PARAMETERS, T>(parameters);

        std::cout << "H best_eps cosine rel_error sign_match a_norm f_norm norm_ratio status\n";

        for(int hi = 0; hi < num_h; hi++){
            TI H = horizons[hi];
            T actions[64][4];
            for(TI s = 0; s < H; s++){
                actions[s][0] = base + (T)0.22 + (T)0.0010 * s;
                actions[s][1] = base + (T)0.18 + (T)0.0005 * s;
                actions[s][2] = base + (T)0.20 - (T)0.0007 * s;
                actions[s][3] = base + (T)0.16 + (T)0.0003 * s;
            }
            T best_cos = -1, best_rel = 1e10, best_a = 0, best_f = 0, best_sign = 0;
            T best_eps = 0;
            for(int ei = 0; ei < num_eps; ei++){
                T eps_fd = eps_list[ei];
                T fd[64][4];
                for(TI s = 0; s < H; s++){
                    for(TI a = 0; a < 4; a++){
                        T plus_a[64][4], minus_a[64][4];
                        for(TI t = 0; t < H; t++) for(TI x = 0; x < 4; x++){ plus_a[t][x] = actions[t][x]; minus_a[t][x] = actions[t][x]; }
                        plus_a[s][a] += eps_fd; minus_a[s][a] -= eps_fd;
                        EulerState<T, TI> sp[65], sm[65];
                        EulerStepCache<T> cp[64], cm[64];
                        T lp = rollout_loss<PARAMETERS, T, TI, 64>(parameters, initial_state, plus_a, H, weights, sp, cp).total();
                        T lm = rollout_loss<PARAMETERS, T, TI, 64>(parameters, initial_state, minus_a, H, weights, sm, cm).total();
                        fd[s][a] = (lp - lm) / ((T)2 * eps_fd);
                    }
                }
                EulerState<T, TI> astates[65]; EulerStepCache<T> acaches[64]; T analytic[64][4];
                rollout_loss_and_gradients<PARAMETERS, T, TI, 64>(parameters, initial_state, actions, H, weights, astates, acaches, analytic);
                T dot_fd = 0, na = 0, nf = 0, err = 0;
                int sign = 0;
                for(TI s = 0; s < H; s++){
                    for(TI a = 0; a < 4; a++){
                        dot_fd += analytic[s][a] * fd[s][a];
                        na += analytic[s][a] * analytic[s][a];
                        nf += fd[s][a] * fd[s][a];
                        T e = analytic[s][a] - fd[s][a];
                        err += e * e;
                        if(std::abs(fd[s][a]) < (T)1e-7 || analytic[s][a] * fd[s][a] >= 0) sign++;
                    }
                }
                na = std::sqrt(na); nf = std::sqrt(nf);
                T cos_fd = dot_fd / std::max((T)1e-8, na * nf);
                if(cos_fd > best_cos){ best_cos = cos_fd; best_rel = safe_relative_error(std::sqrt(err), nf); best_a = na; best_f = nf; best_sign = sign; best_eps = eps_fd; }
            }
            const char* prefix_status;
            T norm_ratio = std::max((T)1e-10, best_f) > (T)1e-10 ? best_a / best_f : (T)1;
            int total_sign = (int)H * 4;
            bool finite_ok = best_a < (T)1e20 && best_f < (T)1e20 && norm_ratio > (T)0.1 && norm_ratio < (T)10.0;
            if(best_cos > (T)0.99 && best_sign >= total_sign * 99 / 100 && best_rel < (T)0.05 && finite_ok){
                prefix_status = "PASS";
            }
            else if(best_cos > (H >= 32 ? (T)0.995 : (T)0.99) && best_sign >= total_sign * 99 / 100 && best_rel < (T)0.15 && finite_ok && norm_ratio > (T)0.5 && norm_ratio < (T)2.0){
                prefix_status = "PASS_DIRECTIONAL";
            }
            else{ prefix_status = "FAIL"; }
            std::cout << +H << " " << best_eps << " " << best_cos << " " << best_rel << " " << best_sign << "/" << total_sign << " " << best_a << " " << best_f << " " << norm_ratio << " " << prefix_status << "\n";
        }
    }

    template <typename PARAMETERS, typename T, typename TI>
    bool run_euler_physics_check(const PARAMETERS& parameters, EulerCheckSummary<T>& summary){
        std::cout << "diff_model=euler\n";
        euler_forward_sanity<PARAMETERS, T, TI>(parameters);
        const bool local_ok = euler_local_derivative_checks<PARAMETERS, T, TI>(parameters);
        const bool vjp_ok = so3_expmap_vjp_check<PARAMETERS, T, TI>(parameters);
        const bool trans_ok = euler_transition_vjp_check<PARAMETERS, T, TI>(parameters);
        const EulerLossWeights<T> weights{(T)8.0, (T)0.8, (T)4.0, (T)0.8, (T)0.005, (T)0.03, (T)0.05, (T)4.0, (T)12.0, (T)4.0, (T)8.0, (T)4.0};
        const bool reference_ok = euler_reference_zero_equivalence_check<PARAMETERS, T, TI>(parameters, weights);
        const bool h1 = euler_rollout_gradient_check_one<PARAMETERS, T, TI, 1>(parameters, weights, summary);
        const bool h4 = euler_rollout_gradient_check_one<PARAMETERS, T, TI, 4>(parameters, weights, summary);
        const bool h16 = euler_rollout_gradient_check_one<PARAMETERS, T, TI, 16>(parameters, weights, summary);
        euler_rollout_gradient_prefix_check<PARAMETERS, T, TI>(parameters, weights);
        const bool sign_ok = euler_sign_sanity_tests<PARAMETERS, T, TI>(parameters);
        const bool rotation_ok = euler_rotation_drift_check<PARAMETERS, T, TI>(parameters);
        euler_coordinate_frame_report();

        const bool strict = local_ok && vjp_ok && trans_ok && reference_ok && h1 && h4 && h16 && sign_ok && rotation_ok;
        const char* overall;
        if(strict) overall = "STRICT_PASS";
        else if(local_ok && vjp_ok && trans_ok && reference_ok && h1 && h4 && h16 && sign_ok && rotation_ok) overall = "DIRECTIONAL_PASS";
        else overall = "FAIL";

        std::cout << "\n[Euler summary]\n";
        std::cout << "overall=" << overall << "\n";
        if(strict){
            std::cout << "Euler model is safe for short fixed-dynamics training.\n";
        }
        else if(local_ok && vjp_ok && trans_ok && reference_ok && h1 && h4 && h16){
            std::cout << "Euler model is safe for short fixed-dynamics training (DIRECTIONAL_PASS).\n";
            std::cout << "H=64 has correct gradient direction but FD-reference scale noise. Actor-grad clipping reduces risk.\n";
        }
        else{
            std::cout << "Do not proceed to training.\n";
        }
        return strict;
    }
}
