#pragma once

#include <cmath>
#include <iostream>

namespace rl_tools::rl::environments::l2f::diff{
    template <typename T, typename TI>
    struct EulerState{
        T p[3];
        T v[3];
        T R[3][3];
        T omega[3];
        T rpm[4];
        T previous_action[4];
        T action_hover_center[4];
    };

    template <typename T>
    struct TrackingReference{
        T p[3];
        T v[3];
    };

    template <typename T>
    TrackingReference<T> zero_tracking_reference(){
        TrackingReference<T> ref{};
        for(int i = 0; i < 3; i++){
            ref.p[i] = (T)0;
            ref.v[i] = (T)0;
        }
        return ref;
    }

    template <typename T>
    void set_zero(TrackingReference<T>& ref){
        for(int i = 0; i < 3; i++){
            ref.p[i] = (T)0;
            ref.v[i] = (T)0;
        }
    }

    template <typename T>
    struct EulerStepCache{
        T normalized_action[4];
        T setpoint[4];
        T d_setpoint_d_action[4];
        T alpha[4];
        T rpm_next[4];
        T force[4];
        T d_force_d_rpm[4];
        T thrust_body[3];
        T torque_body[3];
        T acceleration_world[3];
        T angular_acceleration_body[3];
        T J_omega[3];
        T gyro[3];
        T skew_omega_next[3][3];
        T rotation_update[3][3];
        T omega_next[3];
    };

    template <typename T>
    struct EulerStateAdjoint{
        T p[3];
        T v[3];
        T R[3][3];
        T omega[3];
        T rpm[4];
    };

    template <typename T>
    void zero(EulerStateAdjoint<T>& a){
        for(int i = 0; i < 3; i++){
            a.p[i] = 0;
            a.v[i] = 0;
            a.omega[i] = 0;
            for(int j = 0; j < 3; j++){
                a.R[i][j] = 0;
            }
        }
        for(int i = 0; i < 4; i++){
            a.rpm[i] = 0;
        }
    }

    template <typename T>
    T dot3(const T a[3], const T b[3]){
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    template <typename T>
    void cross3(const T a[3], const T b[3], T out[3]){
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
    }

    template <typename T>
    void mat_vec3(const T A[3][3], const T x[3], T y[3]){
        for(int i = 0; i < 3; i++){
            y[i] = 0;
            for(int j = 0; j < 3; j++){
                y[i] += A[i][j] * x[j];
            }
        }
    }

    template <typename T>
    void skew3(const T w[3], T S[3][3]){
        S[0][0] = 0;     S[0][1] = -w[2]; S[0][2] =  w[1];
        S[1][0] = w[2];  S[1][1] = 0;     S[1][2] = -w[0];
        S[2][0] = -w[1]; S[2][1] = w[0];  S[2][2] = 0;
    }

    template <typename T>
    void mat_mul3(const T A[3][3], const T B[3][3], T C[3][3]){
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                C[i][j] = 0;
                for(int k = 0; k < 3; k++){
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    template <typename T>
    T mat_inner3(const T A[3][3], const T B[3][3]){
        T value = 0;
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                value += A[i][j] * B[i][j];
            }
        }
        return value;
    }

    template <typename T>
    void identity3(T I[3][3]){
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                I[i][j] = i == j ? (T)1 : (T)0;
            }
        }
    }

    template <typename T>
    void so3_expmap_delta(T dt, const T omega[3], T Delta[3][3]){
        T phi[3];
        for(int i = 0; i < 3; i++){ phi[i] = dt * omega[i]; }
        const T theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
        const T theta = std::sqrt(theta_sq);
        T K[3][3];
        skew3(phi, K);
        T K2[3][3];
        mat_mul3(K, K, K2);
        T A, B;
        if(theta < (T)1e-4){
            const T t2 = theta_sq;
            A = (T)1 - t2 / (T)6 + t2 * t2 / (T)120;
            B = (T)0.5 - t2 / (T)24 + t2 * t2 / (T)720;
        }
        else{
            A = std::sin(theta) / theta;
            B = ((T)1 - std::cos(theta)) / theta_sq;
        }
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                Delta[i][j] = (i == j ? (T)1 : (T)0) + A * K[i][j] + B * K2[i][j];
            }
        }
    }

    // Legacy first-order rotation delta for drift comparison only.
    template <typename T>
    void first_order_rotation_delta(T dt, const T omega[3], T Delta[3][3]){
        T S[3][3];
        skew3(omega, S);
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                Delta[i][j] = (i == j ? (T)1 : (T)0) + dt * S[i][j];
            }
        }
    }

    template <typename T>
    void basis_skew3(int idx, T E[3][3]){
        for(int r = 0; r < 3; r++){
            for(int c = 0; c < 3; c++){
                E[r][c] = 0;
            }
        }
        if(idx == 0){
            E[1][2] = -1; E[2][1] = 1;
        }
        else if(idx == 1){
            E[0][2] = 1; E[2][0] = -1;
        }
        else{
            E[0][1] = -1; E[1][0] = 1;
        }
    }

    // Analytic Rodrigues-form VJP for Delta = Exp(dt * skew(omega_next)).
    // Uses dA/dphi and dB/dphi to differentiate I + A*K + B*K2 with respect to phi.
    template <typename T>
    void rotation_delta_vjp_analytic(
        T dt,
        const T omega_next[3],
        const T lambda_delta[3][3],
        T grad_omega[3]
    ){
        T phi[3] = {dt * omega_next[0], dt * omega_next[1], dt * omega_next[2]};
        const T theta2 = dot3(phi, phi);
        const T theta = std::sqrt(theta2);
        T K[3][3];
        skew3(phi, K);
        T K2[3][3];
        mat_mul3(K, K, K2);
        T A, B;
        T dA_dphi[3] = {0, 0, 0};
        T dB_dphi[3] = {0, 0, 0};
        if(theta < (T)1e-4){
            const T t2 = theta2;
            const T t4 = t2 * t2;
            A = (T)1 - t2 / (T)6 + t4 / (T)120;
            B = (T)0.5 - t2 / (T)24 + t4 / (T)720;
            const T cA = (T)(-1.0/3.0) + t2 / (T)30 - t4 / (T)840;
            const T cB = (T)(-1.0/12.0) + t2 / (T)180 - t4 / (T)6720;
            for(int i = 0; i < 3; i++){
                dA_dphi[i] = cA * phi[i];
                dB_dphi[i] = cB * phi[i];
            }
        }
        else{
            A = std::sin(theta) / theta;
            B = ((T)1 - std::cos(theta)) / theta2;
            const T dA_dtheta = (theta * std::cos(theta) - std::sin(theta)) / theta2;
            const T dB_dtheta = (theta * std::sin(theta) - (T)2 * ((T)1 - std::cos(theta))) / (theta2 * theta);
            for(int i = 0; i < 3; i++){
                dA_dphi[i] = dA_dtheta * phi[i] / theta;
                dB_dphi[i] = dB_dtheta * phi[i] / theta;
            }
        }

        const T inner_GK = mat_inner3(lambda_delta, K);
        const T inner_GK2 = mat_inner3(lambda_delta, K2);

        for(int i = 0; i < 3; i++){
            T E[3][3];
            basis_skew3(i, E);
            T EK[3][3];
            T KE[3][3];
            mat_mul3(E, K, EK);
            mat_mul3(K, E, KE);
            T dK2[3][3];
            for(int r = 0; r < 3; r++){
                for(int c = 0; c < 3; c++){
                    dK2[r][c] = EK[r][c] + KE[r][c];
                }
            }
            T term = 0;
            term += dA_dphi[i] * inner_GK;
            term += A * mat_inner3(lambda_delta, E);
            term += dB_dphi[i] * inner_GK2;
            term += B * mat_inner3(lambda_delta, dK2);
            grad_omega[i] = dt * term;
        }
    }

    // Finite-difference VJP retained only for validation.
    // The training VJP uses rotation_delta_vjp_analytic above.
    template <typename T>
    void rotation_delta_vjp_fd(
        T dt,
        const T omega_next[3],
        const T lambda_delta[3][3],
        T grad_omega[3]
    ){
        grad_omega[0] = 0;
        grad_omega[1] = 0;
        grad_omega[2] = 0;
        for(int a = 0; a < 3; a++){
            const T eps = (T)1e-3 * std::max((T)1, std::abs(omega_next[a]));
            T plus[3] = {omega_next[0], omega_next[1], omega_next[2]};
            T minus[3] = {omega_next[0], omega_next[1], omega_next[2]};
            plus[a] += eps;
            minus[a] -= eps;
            T D_plus[3][3], D_minus[3][3];
            so3_expmap_delta(dt, plus, D_plus);
            so3_expmap_delta(dt, minus, D_minus);
            T deriv = 0;
            for(int i = 0; i < 3; i++){
                for(int j = 0; j < 3; j++){
                    deriv += lambda_delta[i][j] * (D_plus[i][j] - D_minus[i][j]);
                }
            }
            deriv /= (T)2 * eps;
            grad_omega[a] += deriv;
        }
    }

    template <typename PARAMETERS, typename T>
    T action_scale(const PARAMETERS& parameters){
        return (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) / (T)2;
    }

    template <typename PARAMETERS, typename T>
    T normalized_to_setpoint(const PARAMETERS& parameters, T normalized_action, T& derivative){
        const T half_range = action_scale<PARAMETERS, T>(parameters);
        const T unclamped = normalized_action * half_range + parameters.dynamics.action_limit.min + half_range;
        if(unclamped < parameters.dynamics.action_limit.min){
            derivative = 0;
            return parameters.dynamics.action_limit.min;
        }
        if(unclamped > parameters.dynamics.action_limit.max){
            derivative = 0;
            return parameters.dynamics.action_limit.max;
        }
        derivative = half_range;
        return unclamped;
    }

    template <typename T, typename STATE>
    void quaternion_to_rotation_matrix(const STATE& state, T R[3][3]){
        const T w = state.orientation[0];
        const T x = state.orientation[1];
        const T y = state.orientation[2];
        const T z = state.orientation[3];
        R[0][0] = 1 - 2 * (y * y + z * z);
        R[0][1] = 2 * (x * y - z * w);
        R[0][2] = 2 * (x * z + y * w);
        R[1][0] = 2 * (x * y + z * w);
        R[1][1] = 1 - 2 * (x * x + z * z);
        R[1][2] = 2 * (y * z - x * w);
        R[2][0] = 2 * (x * z - y * w);
        R[2][1] = 2 * (y * z + x * w);
        R[2][2] = 1 - 2 * (x * x + y * y);
    }

    template <typename STATE, typename T, typename TI>
    void from_l2f_state(const STATE& state, EulerState<T, TI>& out){
        for(TI i = 0; i < 3; i++){
            out.p[i] = state.position[i];
            out.v[i] = state.linear_velocity[i];
            out.omega[i] = state.angular_velocity[i];
        }
        quaternion_to_rotation_matrix<T>(state, out.R);
        for(TI i = 0; i < 4; i++){
            out.rpm[i] = state.rpm[i];
        }
        static_assert(STATE::HISTORY_LENGTH >= 1, "Euler diff pre-training expects action history");
        const TI current_step = state.current_step == 0 ? STATE::HISTORY_LENGTH - 1 : state.current_step - 1;
        for(TI i = 0; i < 4; i++){
            out.previous_action[i] = state.action_history[current_step][i];
        }
    }

    template <typename T, typename TI, typename OBS_MATRIX>
    void observe_with_reference(const EulerState<T, TI>& state, const TrackingReference<T>& ref, OBS_MATRIX& observation){
        TI idx = 0;
        for(TI i = 0; i < 3; i++){
            set(observation, 0, idx++, state.p[i] - ref.p[i]);
        }
        for(TI i = 0; i < 3; i++){
            for(TI j = 0; j < 3; j++){
                set(observation, 0, idx++, state.R[i][j]);
            }
        }
        for(TI i = 0; i < 3; i++){
            set(observation, 0, idx++, state.v[i] - ref.v[i]);
        }
        for(TI i = 0; i < 3; i++){
            set(observation, 0, idx++, state.omega[i]);
        }
        for(TI i = 0; i < 4; i++){
            set(observation, 0, idx++, state.previous_action[i]);
        }
    }

    template <typename T, typename TI, typename OBS_MATRIX>
    void observe(const EulerState<T, TI>& state, OBS_MATRIX& observation){
        const auto ref = zero_tracking_reference<T>();
        observe_with_reference<T, TI>(state, ref, observation);
    }

    template <typename T, typename TI, typename OBS_MATRIX>
    void apply_reference_error_to_observation(OBS_MATRIX& observation, const TrackingReference<T>& ref){
        constexpr TI POSITION_OFFSET = 0;
        constexpr TI VELOCITY_OFFSET = 3 + 9;
        for(TI i = 0; i < 3; i++){
            set(observation, 0, POSITION_OFFSET + i, get(observation, 0, POSITION_OFFSET + i) - ref.p[i]);
            set(observation, 0, VELOCITY_OFFSET + i, get(observation, 0, VELOCITY_OFFSET + i) - ref.v[i]);
        }
    }

    template <typename PARAMETERS, typename T, typename TI>
    void motor_transition(const PARAMETERS& parameters, const EulerState<T, TI>& state, const T action[4], EulerStepCache<T>& cache){
        const T dt = parameters.integration.dt;
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            cache.normalized_action[rotor_i] = action[rotor_i];
            cache.setpoint[rotor_i] = normalized_to_setpoint<PARAMETERS, T>(parameters, action[rotor_i], cache.d_setpoint_d_action[rotor_i]);
            const T tau = cache.setpoint[rotor_i] >= state.rpm[rotor_i]
                ? parameters.dynamics.rotor_time_constants_rising[rotor_i]
                : parameters.dynamics.rotor_time_constants_falling[rotor_i];
            cache.alpha[rotor_i] = std::exp(-dt / tau);
            cache.rpm_next[rotor_i] = cache.alpha[rotor_i] * state.rpm[rotor_i] + ((T)1 - cache.alpha[rotor_i]) * cache.setpoint[rotor_i];
        }
    }

    template <typename PARAMETERS, typename T, typename TI>
    void forces_and_accelerations(const PARAMETERS& parameters, const EulerState<T, TI>& state, EulerStepCache<T>& cache){
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            cache.thrust_body[dim_i] = 0;
            cache.torque_body[dim_i] = 0;
        }
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            const T rpm = cache.rpm_next[rotor_i];
            const T c0 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][0];
            const T c1 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][1];
            const T c2 = parameters.dynamics.rotor_thrust_coefficients[rotor_i][2];
            cache.force[rotor_i] = c0 + c1 * rpm + c2 * rpm * rpm;
            cache.d_force_d_rpm[rotor_i] = c1 + (T)2 * c2 * rpm;
            T rotor_thrust[3];
            T arm_torque[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                rotor_thrust[dim_i] = parameters.dynamics.rotor_thrust_directions[rotor_i][dim_i] * cache.force[rotor_i];
                cache.thrust_body[dim_i] += rotor_thrust[dim_i];
                cache.torque_body[dim_i] += parameters.dynamics.rotor_torque_directions[rotor_i][dim_i] * parameters.dynamics.rotor_torque_constants[rotor_i] * cache.force[rotor_i];
            }
            cross3(parameters.dynamics.rotor_positions[rotor_i], rotor_thrust, arm_torque);
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                cache.torque_body[dim_i] += arm_torque[dim_i];
            }
        }

        T rotated_thrust[3];
        mat_vec3(state.R, cache.thrust_body, rotated_thrust);
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            cache.acceleration_world[dim_i] = rotated_thrust[dim_i] / parameters.dynamics.mass + parameters.dynamics.gravity[dim_i];
        }

        mat_vec3(parameters.dynamics.J, state.omega, cache.J_omega);
        cross3(state.omega, cache.J_omega, cache.gyro);
        T angular_rhs[3];
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            angular_rhs[dim_i] = cache.torque_body[dim_i] - cache.gyro[dim_i];
        }
        mat_vec3(parameters.dynamics.J_inv, angular_rhs, cache.angular_acceleration_body);
    }

    template <typename PARAMETERS, typename T, typename TI>
    void step(const PARAMETERS& parameters, const EulerState<T, TI>& state, const T action[4], EulerState<T, TI>& next, EulerStepCache<T>& cache){
        const T dt = parameters.integration.dt;
        motor_transition<PARAMETERS, T, TI>(parameters, state, action, cache);
        forces_and_accelerations<PARAMETERS, T, TI>(parameters, state, cache);

        for(TI dim_i = 0; dim_i < 3; dim_i++){
            next.omega[dim_i] = state.omega[dim_i] + dt * cache.angular_acceleration_body[dim_i];
            next.v[dim_i] = state.v[dim_i] + dt * cache.acceleration_world[dim_i];
            next.p[dim_i] = state.p[dim_i] + dt * next.v[dim_i];
            next.rpm[dim_i] = cache.rpm_next[dim_i];
            next.previous_action[dim_i] = action[dim_i];
            next.action_hover_center[dim_i] = state.action_hover_center[dim_i];
        }
        next.rpm[3] = cache.rpm_next[3];
        next.previous_action[3] = action[3];
        next.action_hover_center[3] = state.action_hover_center[3];

        skew3(next.omega, cache.skew_omega_next);
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            cache.omega_next[dim_i] = next.omega[dim_i];
        }
        so3_expmap_delta(dt, cache.omega_next, cache.rotation_update);
        T tmp[3][3];
        mat_mul3(state.R, cache.rotation_update, tmp);
        for(TI i = 0; i < 3; i++){
            for(TI j = 0; j < 3; j++){
                next.R[i][j] = tmp[i][j];
            }
        }
    }

    template <typename PARAMETERS, typename T, typename TI>
    void step_vjp(
        const PARAMETERS& parameters,
        const EulerState<T, TI>& state,
        const T action[4],
        const EulerStepCache<T>& cache,
        const EulerStateAdjoint<T>& lambda_next,
        EulerStateAdjoint<T>& lambda_state,
        T grad_action[4]
    ){
        const T dt = parameters.integration.dt;
        zero(lambda_state);
        for(TI i = 0; i < 4; i++){
            grad_action[i] = 0;
        }

        T lambda_v_next[3];
        T lambda_omega_next[3];
        for(TI i = 0; i < 3; i++){
            lambda_state.p[i] += lambda_next.p[i];
            lambda_v_next[i] = lambda_next.v[i] + dt * lambda_next.p[i];
            lambda_omega_next[i] = lambda_next.omega[i];
        }

        T lambda_delta[3][3] = {};
        for(TI i = 0; i < 3; i++){
            for(TI k = 0; k < 3; k++){
                T value = 0;
                for(TI j = 0; j < 3; j++){
                    value += lambda_next.R[i][j] * cache.rotation_update[k][j];
                    lambda_delta[k][j] += state.R[i][k] * lambda_next.R[i][j];
                }
                lambda_state.R[i][k] += value;
            }
        }
        T lambda_omega_from_rotation[3] = {0, 0, 0};
        // Analytic VJP for Delta = Exp(dt * skew(omega_next)).
        // The finite-difference VJP (rotation_delta_vjp_fd) remains only for validation.
        rotation_delta_vjp_analytic(dt, cache.omega_next, lambda_delta, lambda_omega_from_rotation);
        for(TI i = 0; i < 3; i++){
            lambda_omega_next[i] += lambda_omega_from_rotation[i];
        }

        T lambda_acceleration[3];
        T lambda_omega_dot[3];
        for(TI i = 0; i < 3; i++){
            lambda_state.v[i] += lambda_v_next[i];
            lambda_acceleration[i] = dt * lambda_v_next[i];
            lambda_state.omega[i] += lambda_omega_next[i];
            lambda_omega_dot[i] = dt * lambda_omega_next[i];
        }

        T lambda_thrust_body[3] = {};
        for(TI i = 0; i < 3; i++){
            for(TI j = 0; j < 3; j++){
                lambda_state.R[i][j] += lambda_acceleration[i] * cache.thrust_body[j] / parameters.dynamics.mass;
                lambda_thrust_body[j] += lambda_acceleration[i] * state.R[i][j] / parameters.dynamics.mass;
            }
        }

        T lambda_angular_rhs[3] = {};
        for(TI i = 0; i < 3; i++){
            for(TI j = 0; j < 3; j++){
                lambda_angular_rhs[j] += parameters.dynamics.J_inv[i][j] * lambda_omega_dot[i];
            }
        }
        T lambda_torque_body[3] = {lambda_angular_rhs[0], lambda_angular_rhs[1], lambda_angular_rhs[2]};
        T lambda_gyro[3] = {-lambda_angular_rhs[0], -lambda_angular_rhs[1], -lambda_angular_rhs[2]};

        T lambda_omega_from_cross[3];
        T lambda_J_omega[3];
        cross3(cache.J_omega, lambda_gyro, lambda_omega_from_cross);
        cross3(lambda_gyro, state.omega, lambda_J_omega);
        for(TI i = 0; i < 3; i++){
            lambda_state.omega[i] += lambda_omega_from_cross[i];
            for(TI j = 0; j < 3; j++){
                lambda_state.omega[j] += parameters.dynamics.J[i][j] * lambda_J_omega[i];
            }
        }

        T lambda_rpm_next[4] = {lambda_next.rpm[0], lambda_next.rpm[1], lambda_next.rpm[2], lambda_next.rpm[3]};
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            T lambda_force = 0;
            T arm_direction[3];
            cross3(parameters.dynamics.rotor_positions[rotor_i], parameters.dynamics.rotor_thrust_directions[rotor_i], arm_direction);
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                lambda_force += parameters.dynamics.rotor_thrust_directions[rotor_i][dim_i] * lambda_thrust_body[dim_i];
                lambda_force += parameters.dynamics.rotor_torque_directions[rotor_i][dim_i] * parameters.dynamics.rotor_torque_constants[rotor_i] * lambda_torque_body[dim_i];
                lambda_force += arm_direction[dim_i] * lambda_torque_body[dim_i];
            }
            lambda_rpm_next[rotor_i] += lambda_force * cache.d_force_d_rpm[rotor_i];
        }

        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            lambda_state.rpm[rotor_i] += cache.alpha[rotor_i] * lambda_rpm_next[rotor_i];
            const T d_next_d_setpoint = (T)1 - cache.alpha[rotor_i];
            grad_action[rotor_i] += d_next_d_setpoint * cache.d_setpoint_d_action[rotor_i] * lambda_rpm_next[rotor_i];
        }
    }

    template <typename T, typename TI>
    T state_norm(const EulerState<T, TI>& state){
        T value = 0;
        for(TI i = 0; i < 3; i++){
            value += state.p[i] * state.p[i] + state.v[i] * state.v[i] + state.omega[i] * state.omega[i];
        }
        for(TI i = 0; i < 3; i++){
            for(TI j = 0; j < 3; j++){
                value += state.R[i][j] * state.R[i][j];
            }
        }
        for(TI i = 0; i < 4; i++){
            value += state.rpm[i] * state.rpm[i];
        }
        return std::sqrt(value);
    }
}
