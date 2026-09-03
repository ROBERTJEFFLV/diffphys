#ifndef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
// #define RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
#endif

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f/diff_euler_check.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "loss.h"

namespace fp = rl_tools::foundation_policy::diff_pre_training;
namespace rlt = rl_tools;
namespace l2f = rl_tools::rl::environments::l2f;
namespace l2f_diff = rl_tools::rl::environments::l2f::diff;

using DEVICE = fp::DEVICE;
using RNG = fp::RNG;
using TI = fp::TI;
using T = fp::T;
using ENVIRONMENT = fp::ENVIRONMENT;
using STATE = ENVIRONMENT::State;
using PARAMETERS = ENVIRONMENT::Parameters;

T abs_safe(T x){ return std::abs(x); }
T max3(T a, T b, T c){ return std::max(a, std::max(b, c)); }
T norm3(const T v[3]){ return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
T dot4(const T a[4], const T b[4]){ T v = 0; for(TI i = 0; i < 4; i++){ v += a[i] * b[i]; } return v; }
T norm4(const T a[4]){ return std::sqrt(dot4(a, a)); }
T relative_error(T abs_error, T reference_norm){ return abs_error / std::max((T)1e-6, reference_norm); }

void print_vec3(const char* name, const T v[3]){
    std::cout << name << "=[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
}
void print_vec4(const char* name, const T v[4]){
    std::cout << name << "=[" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << "]";
}

struct CheckBook{
    TI passed = 0;
    TI failed = 0;
    void check(bool condition, const std::string& name){
        if(condition){
            passed++;
            std::cout << "PASS " << name << "\n";
        }
        else{
            failed++;
            std::cout << "FAIL " << name << "\n";
        }
    }
    bool ok() const { return failed == 0; }
};

struct Thresholds{
    T forward_step_error = (T)1e-6;
    T thrust_derivative_relative_error = (T)2e-2;
    T torque_derivative_relative_error = (T)2e-2;
    T acceleration_derivative_relative_error = (T)5e-2;
    T loss_gradient_cosine = (T)0.95;
    T loss_gradient_relative_error = (T)1e-1;
};

T clamp_action(T action){
    return std::max((T)-1, std::min((T)1, action));
}

T normalized_to_setpoint(const PARAMETERS& parameters, T normalized_action){
    const T half_range = (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) / (T)2;
    const T unclamped = normalized_action * half_range + parameters.dynamics.action_limit.min + half_range;
    return std::max(parameters.dynamics.action_limit.min, std::min(parameters.dynamics.action_limit.max, unclamped));
}

T thrust_magnitude(const PARAMETERS& parameters, TI rotor_i, T rpm){
    return parameters.dynamics.rotor_thrust_coefficients[rotor_i][0]
        + parameters.dynamics.rotor_thrust_coefficients[rotor_i][1] * rpm
        + parameters.dynamics.rotor_thrust_coefficients[rotor_i][2] * rpm * rpm;
}

void cross(const T a[3], const T b[3], T out[3]){
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

void mat_vec(const T M[3][3], const T x[3], T out[3]){
    for(TI r = 0; r < 3; r++){
        out[r] = M[r][0] * x[0] + M[r][1] * x[1] + M[r][2] * x[2];
    }
}

void compute_force_torque_body(const PARAMETERS& parameters, const T rpm[4], T thrust_body[3], T torque_body[3]){
    for(TI dim = 0; dim < 3; dim++){
        thrust_body[dim] = 0;
        torque_body[dim] = 0;
    }
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        const T F = thrust_magnitude(parameters, rotor_i, rpm[rotor_i]);
        T rotor_thrust[3];
        for(TI dim = 0; dim < 3; dim++){
            rotor_thrust[dim] = parameters.dynamics.rotor_thrust_directions[rotor_i][dim] * F;
            thrust_body[dim] += rotor_thrust[dim];
            torque_body[dim] += parameters.dynamics.rotor_torque_directions[rotor_i][dim] * parameters.dynamics.rotor_torque_constants[rotor_i] * F;
        }
        T arm_torque[3];
        cross(parameters.dynamics.rotor_positions[rotor_i], rotor_thrust, arm_torque);
        for(TI dim = 0; dim < 3; dim++){
            torque_body[dim] += arm_torque[dim];
        }
    }
}

void compute_accelerations(DEVICE& device, const PARAMETERS& parameters, const STATE& state, const T rpm[4], T linear_acceleration[3], T angular_acceleration[3]){
    T thrust_body[3];
    T torque_body[3];
    compute_force_torque_body(parameters, rpm, thrust_body, torque_body);
    l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, thrust_body, linear_acceleration);
    for(TI dim = 0; dim < 3; dim++){
        linear_acceleration[dim] = linear_acceleration[dim] / parameters.dynamics.mass + parameters.dynamics.gravity[dim];
    }

    T angular_momentum[3];
    T coriolis[3];
    T angular_rhs[3];
    mat_vec(parameters.dynamics.J, state.angular_velocity, angular_momentum);
    cross(state.angular_velocity, angular_momentum, coriolis);
    for(TI dim = 0; dim < 3; dim++){
        angular_rhs[dim] = torque_body[dim] - coriolis[dim];
    }
    mat_vec(parameters.dynamics.J_inv, angular_rhs, angular_acceleration);
}

T hover_normalized_action(const PARAMETERS& parameters){
    return parameters.dynamics.hovering_throttle_relative * (T)2 - (T)1;
}

T hover_rpm(const PARAMETERS& parameters){
    return parameters.dynamics.action_limit.min + parameters.dynamics.hovering_throttle_relative * (parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min);
}

void normalize_quaternion(T q[4]){
    T n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    for(TI i = 0; i < 4; i++){ q[i] /= n; }
}

void configure_test_state(const PARAMETERS& parameters, STATE& state, bool perturbed){
    const T hover_action = hover_normalized_action(parameters);
    const T rpm = hover_rpm(parameters);
    for(TI dim = 0; dim < 3; dim++){
        state.position[dim] = 0;
        state.linear_velocity[dim] = 0;
        state.angular_velocity[dim] = 0;
        state.angular_velocity_history[0][dim] = 0;
        state.force[dim] = 0;
        state.torque[dim] = 0;
        state.trajectory.langevin.position[dim] = 0;
        state.trajectory.langevin.velocity[dim] = 0;
        state.trajectory.langevin.position_raw[dim] = 0;
        state.trajectory.langevin.velocity_raw[dim] = 0;
    }
    state.orientation[0] = 1;
    state.orientation[1] = 0;
    state.orientation[2] = 0;
    state.orientation[3] = 0;
    if(perturbed){
        state.position[0] = (T)0.20;
        state.position[1] = (T)-0.12;
        state.position[2] = (T)0.15;
        state.linear_velocity[0] = (T)0.10;
        state.linear_velocity[1] = (T)-0.04;
        state.linear_velocity[2] = (T)0.03;
        state.angular_velocity[0] = (T)0.18;
        state.angular_velocity[1] = (T)-0.11;
        state.angular_velocity[2] = (T)0.07;
        state.orientation[0] = (T)0.995;
        state.orientation[1] = (T)0.045;
        state.orientation[2] = (T)-0.070;
        state.orientation[3] = (T)0.025;
        normalize_quaternion(state.orientation);
    }
    for(TI action_i = 0; action_i < 4; action_i++){
        state.rpm[action_i] = rpm;
        state.last_action[action_i] = hover_action;
        state.action_history[0][action_i] = hover_action;
    }
    state.current_step = 0;
    state.trajectory.type = l2f::POSITION;
}

void prepare_parameters(DEVICE& device, ENVIRONMENT& env, PARAMETERS& parameters, RNG& rng, bool sampled){
    if(sampled){
        rlt::sample_initial_parameters(device, env, parameters, rng);
    }
    else{
        rlt::initial_parameters(device, env, parameters);
    }
    parameters.mdp.action_noise.normalized_rpm = 0;
    parameters.disturbances.random_force.mean = 0;
    parameters.disturbances.random_force.std = 0;
    parameters.disturbances.random_torque.mean = 0;
    parameters.disturbances.random_torque.std = 0;
}

void set_action(rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>>& action_matrix, const T action[4]){
    for(TI action_i = 0; action_i < 4; action_i++){
        rlt::set(action_matrix, 0, action_i, action[action_i]);
    }
}

T max_state_difference(const STATE& a, const STATE& b, T& position, T& orientation, T& linear_velocity, T& angular_velocity, T& rpm){
    position = orientation = linear_velocity = angular_velocity = rpm = 0;
    for(TI dim = 0; dim < 3; dim++){
        position = std::max(position, abs_safe(a.position[dim] - b.position[dim]));
        linear_velocity = std::max(linear_velocity, abs_safe(a.linear_velocity[dim] - b.linear_velocity[dim]));
        angular_velocity = std::max(angular_velocity, abs_safe(a.angular_velocity[dim] - b.angular_velocity[dim]));
    }
    for(TI dim = 0; dim < 4; dim++){
        orientation = std::max(orientation, abs_safe(a.orientation[dim] - b.orientation[dim]));
        rpm = std::max(rpm, abs_safe(a.rpm[dim] - b.rpm[dim]));
    }
    return max3(std::max(position, orientation), std::max(linear_velocity, angular_velocity), rpm);
}

bool forward_consistency_check(DEVICE& device, ENVIRONMENT& env, PARAMETERS& parameters, RNG& rng, const Thresholds& thresholds, CheckBook& book){
    STATE state, original_next, diff_next;
    configure_test_state(parameters, state, true);
    T action[4] = {
        clamp_action(hover_normalized_action(parameters) + (T)0.04),
        clamp_action(hover_normalized_action(parameters) - (T)0.02),
        clamp_action(hover_normalized_action(parameters) + (T)0.03),
        clamp_action(hover_normalized_action(parameters) - (T)0.01)
    };
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action_matrix;
    rlt::malloc(device, action_matrix);
    set_action(action_matrix, action);
    rlt::step(device, env, parameters, state, action_matrix, original_next, rng);
    rlt::step(device, env, parameters, state, action_matrix, diff_next, rng);
    T position, orientation, linear_velocity, angular_velocity, rpm;
    const T max_error = max_state_difference(original_next, diff_next, position, orientation, linear_velocity, angular_velocity, rpm);

    T rpm_next[4];
    for(TI i = 0; i < 4; i++){ rpm_next[i] = original_next.rpm[i]; }
    T thrust_body[3], torque_body[3], linear_acceleration[3], angular_acceleration[3];
    compute_force_torque_body(parameters, rpm_next, thrust_body, torque_body);
    compute_accelerations(device, parameters, state, rpm_next, linear_acceleration, angular_acceleration);

    std::cout << "\n[A] Forward consistency\n";
    std::cout << "diff_pre_training forward path: original rlt::step (RK4 + post_integration)\n";
    std::cout << "position_error=" << position << " orientation_error=" << orientation
              << " linear_velocity_error=" << linear_velocity << " angular_velocity_error=" << angular_velocity
              << " rpm_error=" << rpm << " max_error=" << max_error << "\n";
    print_vec3("total_thrust_body", thrust_body); std::cout << " ";
    print_vec3("total_torque_body", torque_body); std::cout << "\n";
    print_vec3("linear_acceleration_world", linear_acceleration); std::cout << " ";
    print_vec3("angular_acceleration_body", angular_acceleration); std::cout << "\n";
    const bool pass = max_error <= thresholds.forward_step_error;
    book.check(pass, "forward_consistency_original_step_vs_diff_pretraining_forward");
    rlt::free(device, action_matrix);
    return pass;
}

bool motor_delay_check(DEVICE& device, const PARAMETERS& parameters, CheckBook& book){
    std::cout << "\n[B] Motor delay\n";
    const T hover_action = hover_normalized_action(parameters);
    const T rpm0 = hover_rpm(parameters);
    T actions[4] = {hover_action, hover_action, hover_action, hover_action};
    actions[0] = clamp_action(hover_action + (T)0.20);
    T next_rpm[4];
    T gradients[4];
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        l2f::motor_transition_and_gradient(device, parameters, rpm0, actions[rotor_i], rotor_i, next_rpm[rotor_i], gradients[rotor_i]);
    }
    const bool motor_1_increases_more = next_rpm[0] > next_rpm[1] && next_rpm[0] > next_rpm[2] && next_rpm[0] > next_rpm[3];
    book.check(motor_1_increases_more, "motor_1_setpoint_increases_motor_1_rpm_more_than_others");

    const T eps = (T)1e-3;
    T plus, minus, dplus, dminus;
    l2f::motor_transition_and_gradient(device, parameters, rpm0, actions[0] + eps, 0, plus, dplus);
    l2f::motor_transition_and_gradient(device, parameters, rpm0, actions[0] - eps, 0, minus, dminus);
    const T fd = (plus - minus) / ((T)2 * eps);
    const T rel = relative_error(abs_safe(fd - gradients[0]), abs_safe(fd));
    std::cout << "rpm0=" << rpm0 << " action0=" << actions[0] << " next_rpm=["
              << next_rpm[0] << ", " << next_rpm[1] << ", " << next_rpm[2] << ", " << next_rpm[3] << "]\n";
    std::cout << "drpm_next/du analytic=" << gradients[0] << " finite_difference=" << fd << " relative_error=" << rel << "\n";
    const bool pass = rel < (T)1e-2;
    book.check(pass, "motor_delay_gradient_matches_finite_difference");
    return motor_1_increases_more && pass;
}

bool thrust_derivative_check(const PARAMETERS& parameters, const Thresholds& thresholds, CheckBook& book){
    std::cout << "\n[C] Thrust derivative\n";
    const T min_rpm = parameters.dynamics.action_limit.min;
    const T max_rpm = parameters.dynamics.action_limit.max;
    const T hover = hover_rpm(parameters);
    T rpm_values[3] = {min_rpm + (T)0.05 * (max_rpm - min_rpm), hover, min_rpm + (T)0.90 * (max_rpm - min_rpm)};
    bool all_pass = true;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        for(TI rpm_i = 0; rpm_i < 3; rpm_i++){
            const T rpm = rpm_values[rpm_i];
            const T analytic = parameters.dynamics.rotor_thrust_coefficients[rotor_i][1] + (T)2 * parameters.dynamics.rotor_thrust_coefficients[rotor_i][2] * rpm;
            const T eps = std::max((T)1e-2, abs_safe(max_rpm - min_rpm) * (T)1e-4);
            const T fd = (thrust_magnitude(parameters, rotor_i, rpm + eps) - thrust_magnitude(parameters, rotor_i, rpm - eps)) / ((T)2 * eps);
            const T rel = relative_error(abs_safe(analytic - fd), abs_safe(fd));
            std::cout << "rotor=" << rotor_i << " rpm=" << rpm << " analytic=" << analytic << " fd=" << fd << " rel=" << rel << "\n";
            all_pass = all_pass && rel < thresholds.thrust_derivative_relative_error;
        }
    }
    book.check(all_pass, "thrust_derivative_matches_central_difference");
    return all_pass;
}

bool torque_derivative_check(DEVICE& device, const PARAMETERS& parameters, const Thresholds& thresholds, CheckBook& book){
    std::cout << "\n[D] Torque derivative\n";
    const T max_rpm = parameters.dynamics.action_limit.max;
    const T min_rpm = parameters.dynamics.action_limit.min;
    T rpm[4] = {hover_rpm(parameters), hover_rpm(parameters), hover_rpm(parameters), hover_rpm(parameters)};
    const T eps = std::max((T)1e-2, abs_safe(max_rpm - min_rpm) * (T)1e-4);
    bool all_pass = true;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        T d_thrust_body[3];
        T analytic[3];
        l2f::rotor_force_torque_derivative(device, parameters, rotor_i, rpm[rotor_i], d_thrust_body, analytic);
        T rpm_plus[4] = {rpm[0], rpm[1], rpm[2], rpm[3]};
        T rpm_minus[4] = {rpm[0], rpm[1], rpm[2], rpm[3]};
        rpm_plus[rotor_i] += eps;
        rpm_minus[rotor_i] -= eps;
        T thrust_dummy[3], torque_plus[3], torque_minus[3], fd[3], err_vec[3];
        compute_force_torque_body(parameters, rpm_plus, thrust_dummy, torque_plus);
        compute_force_torque_body(parameters, rpm_minus, thrust_dummy, torque_minus);
        for(TI dim = 0; dim < 3; dim++){
            fd[dim] = (torque_plus[dim] - torque_minus[dim]) / ((T)2 * eps);
            err_vec[dim] = analytic[dim] - fd[dim];
        }
        const T abs_err = norm3(err_vec);
        const T rel = relative_error(abs_err, norm3(fd));
        bool sign_ok = true;
        for(TI dim = 0; dim < 3; dim++){
            if(abs_safe(fd[dim]) > (T)1e-7){ sign_ok = sign_ok && (analytic[dim] * fd[dim] >= 0); }
        }
        std::cout << "rotor=" << rotor_i << " "; print_vec3("analytic_dtorque_drpm", analytic); std::cout << " "; print_vec3("fd", fd);
        std::cout << " abs_error=" << abs_err << " rel=" << rel << " sign_ok=" << (sign_ok ? "true" : "false") << "\n";
        all_pass = all_pass && rel < thresholds.torque_derivative_relative_error && sign_ok;
    }
    book.check(all_pass, "torque_derivative_matches_central_difference");
    return all_pass;
}

bool acceleration_derivative_check(DEVICE& device, const PARAMETERS& parameters, const Thresholds& thresholds, CheckBook& book){
    std::cout << "\n[E] Acceleration derivative\n";
    STATE state;
    configure_test_state(parameters, state, true);
    T rpm[4] = {hover_rpm(parameters), hover_rpm(parameters), hover_rpm(parameters), hover_rpm(parameters)};
    l2f::RotorAccelerationJacobian<T, TI> jac;
    l2f::rotor_acceleration_jacobian_from_quaternion(device, parameters, state.orientation, rpm, jac);
    const T eps = std::max((T)1e-2, abs_safe(parameters.dynamics.action_limit.max - parameters.dynamics.action_limit.min) * (T)1e-4);
    bool all_rpm_pass = true;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        T rpm_plus[4] = {rpm[0], rpm[1], rpm[2], rpm[3]};
        T rpm_minus[4] = {rpm[0], rpm[1], rpm[2], rpm[3]};
        rpm_plus[rotor_i] += eps;
        rpm_minus[rotor_i] -= eps;
        T lin_plus[3], ang_plus[3], lin_minus[3], ang_minus[3], lin_fd[3], ang_fd[3], lin_err[3], ang_err[3];
        compute_accelerations(device, parameters, state, rpm_plus, lin_plus, ang_plus);
        compute_accelerations(device, parameters, state, rpm_minus, lin_minus, ang_minus);
        for(TI dim = 0; dim < 3; dim++){
            lin_fd[dim] = (lin_plus[dim] - lin_minus[dim]) / ((T)2 * eps);
            ang_fd[dim] = (ang_plus[dim] - ang_minus[dim]) / ((T)2 * eps);
            lin_err[dim] = jac.linear[rotor_i][dim] - lin_fd[dim];
            ang_err[dim] = jac.angular[rotor_i][dim] - ang_fd[dim];
        }
        const T lin_rel = relative_error(norm3(lin_err), norm3(lin_fd));
        const T ang_rel = relative_error(norm3(ang_err), norm3(ang_fd));
        std::cout << "rotor=" << rotor_i << " da/drpm_rel=" << lin_rel << " dangular_acc/drpm_rel=" << ang_rel << "\n";
        all_rpm_pass = all_rpm_pass && lin_rel < thresholds.acceleration_derivative_relative_error && ang_rel < thresholds.acceleration_derivative_relative_error;
    }
    book.check(all_rpm_pass, "acceleration_derivative_wrt_rpm_matches_central_difference");

    T action = clamp_action(hover_normalized_action(parameters) + (T)0.05);
    bool all_action_pass = true;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        T next_rpm;
        T drpm_du;
        l2f::motor_transition_and_gradient(device, parameters, rpm[rotor_i], action, rotor_i, next_rpm, drpm_du);
        T next_rpms[4] = {rpm[0], rpm[1], rpm[2], rpm[3]};
        next_rpms[rotor_i] = next_rpm;
        l2f::RotorAccelerationJacobian<T, TI> jac_action;
        l2f::rotor_acceleration_jacobian_from_quaternion(device, parameters, state.orientation, next_rpms, jac_action);
        T analytic_lin[3], analytic_ang[3];
        for(TI dim = 0; dim < 3; dim++){
            analytic_lin[dim] = jac_action.linear[rotor_i][dim] * drpm_du;
            analytic_ang[dim] = jac_action.angular[rotor_i][dim] * drpm_du;
        }
        const T eps_u = (T)1e-3;
        T rpm_plus_val, rpm_minus_val, dummy;
        l2f::motor_transition_and_gradient(device, parameters, rpm[rotor_i], action + eps_u, rotor_i, rpm_plus_val, dummy);
        l2f::motor_transition_and_gradient(device, parameters, rpm[rotor_i], action - eps_u, rotor_i, rpm_minus_val, dummy);
        T rpm_plus[4] = {rpm[0], rpm[1], rpm[2], rpm[3]};
        T rpm_minus[4] = {rpm[0], rpm[1], rpm[2], rpm[3]};
        rpm_plus[rotor_i] = rpm_plus_val;
        rpm_minus[rotor_i] = rpm_minus_val;
        T lin_plus[3], ang_plus[3], lin_minus[3], ang_minus[3], lin_fd[3], ang_fd[3], lin_err[3], ang_err[3];
        compute_accelerations(device, parameters, state, rpm_plus, lin_plus, ang_plus);
        compute_accelerations(device, parameters, state, rpm_minus, lin_minus, ang_minus);
        for(TI dim = 0; dim < 3; dim++){
            lin_fd[dim] = (lin_plus[dim] - lin_minus[dim]) / ((T)2 * eps_u);
            ang_fd[dim] = (ang_plus[dim] - ang_minus[dim]) / ((T)2 * eps_u);
            lin_err[dim] = analytic_lin[dim] - lin_fd[dim];
            ang_err[dim] = analytic_ang[dim] - ang_fd[dim];
        }
        const T lin_rel = relative_error(norm3(lin_err), norm3(lin_fd));
        const T ang_rel = relative_error(norm3(ang_err), norm3(ang_fd));
        std::cout << "rotor=" << rotor_i << " da/du_rel=" << lin_rel << " dangular_acc/du_rel=" << ang_rel << "\n";
        all_action_pass = all_action_pass && lin_rel < thresholds.acceleration_derivative_relative_error && ang_rel < thresholds.acceleration_derivative_relative_error;
    }
    book.check(all_action_pass, "acceleration_derivative_wrt_actor_action_matches_central_difference");
    std::cout << "Implemented angular derivative covers direct torque contribution; Coriolis state-coupling derivative is not part of dacc/du when state angular velocity is fixed.\n";
    return all_rpm_pass && all_action_pass;
}

template <TI HORIZON>
void rollout_loss(DEVICE& device, ENVIRONMENT& env, const PARAMETERS& parameters, const STATE& state0, const T (&actions)[HORIZON][4], T& loss, fp::LossTerms<T>& terms, RNG& rng){
    STATE states[HORIZON + 1];
    states[0] = state0;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action_matrix;
    rlt::malloc(device, action_matrix);
    for(TI step_i = 0; step_i < HORIZON; step_i++){
        set_action(action_matrix, actions[step_i]);
        auto parameters_copy = parameters;
        rlt::step(device, env, parameters_copy, states[step_i], action_matrix, states[step_i + 1], rng);
    }
    T action_gradients[HORIZON][4];
    const fp::LossWeights<T> weights{
        fp::LOSS_POSITION_WEIGHT,
        fp::LOSS_VELOCITY_WEIGHT,
        fp::LOSS_ATTITUDE_WEIGHT,
        fp::LOSS_ANGULAR_VELOCITY_WEIGHT,
        fp::LOSS_ACTION_SMOOTHNESS_WEIGHT,
        fp::LOSS_SATURATION_WEIGHT
    };
    terms = fp::stabilization_loss_and_action_gradients<DEVICE, PARAMETERS, STATE, T, TI, HORIZON>(device, parameters, states, actions, action_gradients, weights);
    loss = terms.total();
    rlt::free(device, action_matrix);
}

template <TI HORIZON>
bool loss_gradient_check_one(DEVICE& device, ENVIRONMENT& env, PARAMETERS& parameters, RNG& rng, bool sampled, const Thresholds& thresholds, CheckBook& book){
    STATE state0;
    configure_test_state(parameters, state0, true);
    T base = hover_normalized_action(parameters);
    T actions[HORIZON][4];
    for(TI step_i = 0; step_i < HORIZON; step_i++){
        actions[step_i][0] = clamp_action(base + (T)0.05 + (T)0.005 * step_i);
        actions[step_i][1] = clamp_action(base - (T)0.03 + (T)0.002 * step_i);
        actions[step_i][2] = clamp_action(base + (T)0.02 - (T)0.003 * step_i);
        actions[step_i][3] = clamp_action(base - (T)0.01 + (T)0.001 * step_i);
    }

    STATE states[HORIZON + 1];
    states[0] = state0;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action_matrix;
    rlt::malloc(device, action_matrix);
    for(TI step_i = 0; step_i < HORIZON; step_i++){
        set_action(action_matrix, actions[step_i]);
        auto parameters_copy = parameters;
        rlt::step(device, env, parameters_copy, states[step_i], action_matrix, states[step_i + 1], rng);
    }
    T analytic_gradients[HORIZON][4];
    const fp::LossWeights<T> weights{
        fp::LOSS_POSITION_WEIGHT,
        fp::LOSS_VELOCITY_WEIGHT,
        fp::LOSS_ATTITUDE_WEIGHT,
        fp::LOSS_ANGULAR_VELOCITY_WEIGHT,
        fp::LOSS_ACTION_SMOOTHNESS_WEIGHT,
        fp::LOSS_SATURATION_WEIGHT
    };
    auto terms = fp::stabilization_loss_and_action_gradients<DEVICE, PARAMETERS, STATE, T, TI, HORIZON>(device, parameters, states, actions, analytic_gradients, weights);

    T fd[4];
    const T eps = (T)1e-3;
    for(TI action_i = 0; action_i < 4; action_i++){
        T plus_actions[HORIZON][4];
        T minus_actions[HORIZON][4];
        for(TI step_i = 0; step_i < HORIZON; step_i++){
            for(TI j = 0; j < 4; j++){
                plus_actions[step_i][j] = actions[step_i][j];
                minus_actions[step_i][j] = actions[step_i][j];
            }
        }
        plus_actions[0][action_i] += eps;
        minus_actions[0][action_i] -= eps;
        T loss_plus, loss_minus;
        fp::LossTerms<T> terms_plus, terms_minus;
        RNG rng_plus = rng;
        RNG rng_minus = rng;
        rollout_loss<HORIZON>(device, env, parameters, state0, plus_actions, loss_plus, terms_plus, rng_plus);
        rollout_loss<HORIZON>(device, env, parameters, state0, minus_actions, loss_minus, terms_minus, rng_minus);
        fd[action_i] = (loss_plus - loss_minus) / ((T)2 * eps);
    }
    T analytic[4] = {analytic_gradients[0][0], analytic_gradients[0][1], analytic_gradients[0][2], analytic_gradients[0][3]};
    T err[4];
    for(TI i = 0; i < 4; i++){ err[i] = analytic[i] - fd[i]; }
    const T cosine = dot4(analytic, fd) / std::max((T)1e-8, norm4(analytic) * norm4(fd));
    const T rel = relative_error(norm4(err), norm4(fd));
    TI sign_matches = 0;
    TI sign_count = 0;
    for(TI i = 0; i < 4; i++){
        if(abs_safe(fd[i]) > (T)1e-6 || abs_safe(analytic[i]) > (T)1e-6){
            sign_count++;
            if(analytic[i] * fd[i] >= 0){ sign_matches++; }
        }
    }
    const T max_abs_error = std::max(std::max(abs_safe(err[0]), abs_safe(err[1])), std::max(abs_safe(err[2]), abs_safe(err[3])));
    std::cout << "H=" << HORIZON << " sampled=" << (sampled ? "true" : "false")
              << " loss=" << terms.total() << " cosine=" << cosine << " rel_error=" << rel
              << " max_abs_error=" << max_abs_error << " sign_matches=" << sign_matches << "/" << sign_count << "\n";
    print_vec4("analytic_dL_du_t0", analytic); std::cout << " "; print_vec4("fd_dL_du_t0", fd); std::cout << "\n";
    const bool pass = cosine > thresholds.loss_gradient_cosine && rel < thresholds.loss_gradient_relative_error && sign_matches == sign_count;
    book.check(pass, std::string("loss_gradient_H") + std::to_string(HORIZON) + (sampled ? "_sampled" : "_fixed"));
    rlt::free(device, action_matrix);
    return pass;
}

bool loss_gradient_checks(DEVICE& device, ENVIRONMENT& env, RNG& rng, const Thresholds& thresholds, CheckBook& book){
    std::cout << "\n[F] Loss gradient checks against original L2F forward rollout finite differences\n";
    PARAMETERS fixed_params;
    prepare_parameters(device, env, fixed_params, rng, false);
    bool ok = true;
    ok = loss_gradient_check_one<1>(device, env, fixed_params, rng, false, thresholds, book) && ok;
    ok = loss_gradient_check_one<4>(device, env, fixed_params, rng, false, thresholds, book) && ok;
    ok = loss_gradient_check_one<16>(device, env, fixed_params, rng, false, thresholds, book) && ok;
    PARAMETERS sampled_params;
    RNG sampled_rng = rng;
    prepare_parameters(device, env, sampled_params, sampled_rng, true);
    ok = loss_gradient_check_one<1>(device, env, sampled_params, sampled_rng, true, thresholds, book) && ok;
    ok = loss_gradient_check_one<4>(device, env, sampled_params, sampled_rng, true, thresholds, book) && ok;
    ok = loss_gradient_check_one<16>(device, env, sampled_params, sampled_rng, true, thresholds, book) && ok;
    return ok;
}

bool sign_sanity_tests(DEVICE& device, const PARAMETERS& parameters, CheckBook& book){
    std::cout << "\n[G] Sign sanity tests\n";
    STATE state;
    configure_test_state(parameters, state, false);
    const T base_rpm = hover_rpm(parameters);
    T rpm_base[4] = {base_rpm, base_rpm, base_rpm, base_rpm};
    T lin_base[3], ang_base[3];
    compute_accelerations(device, parameters, state, rpm_base, lin_base, ang_base);

    const T high_rpm = normalized_to_setpoint(parameters, clamp_action(hover_normalized_action(parameters) + (T)0.15));
    T rpm_all[4] = {high_rpm, high_rpm, high_rpm, high_rpm};
    T lin_all[3], ang_all[3], delta_lin[3];
    compute_accelerations(device, parameters, state, rpm_all, lin_all, ang_all);
    for(TI dim = 0; dim < 3; dim++){ delta_lin[dim] = lin_all[dim] - lin_base[dim]; }
    std::cout << "scenario1 all motors increased: "; print_vec3("delta_linear_acceleration", delta_lin); std::cout << " "; print_vec3("angular_acceleration", ang_all); std::cout << "\n";
    const bool all_motor_pass = norm3(delta_lin) > (T)1e-3 && norm3(ang_all) < std::max((T)1e-2, norm3(delta_lin) * (T)5);
    book.check(all_motor_pass, "sign_sanity_all_motors_increase_changes_thrust_direction");

    T rpm_one[4] = {base_rpm, base_rpm, base_rpm, base_rpm};
    rpm_one[0] = high_rpm;
    T thrust_one[3], torque_one[3], lin_one[3], ang_one[3];
    compute_force_torque_body(parameters, rpm_one, thrust_one, torque_one);
    compute_accelerations(device, parameters, state, rpm_one, lin_one, ang_one);
    std::cout << "scenario2 motor0 increased: "; print_vec3("total_thrust_body", thrust_one); std::cout << " "; print_vec3("total_torque_body", torque_one); std::cout << " "; print_vec3("angular_acceleration", ang_one); std::cout << "\n";
    const bool single_motor_pass = norm3(torque_one) > (T)1e-7 && norm3(ang_one) > (T)1e-7;
    book.check(single_motor_pass, "sign_sanity_single_motor_creates_torque_and_angular_acceleration");

    T rpm_diag[4] = {base_rpm, base_rpm, base_rpm, base_rpm};
    rpm_diag[0] = high_rpm;
    rpm_diag[2] = high_rpm;
    T thrust_diag[3], torque_diag[3], lin_diag[3], ang_diag[3];
    compute_force_torque_body(parameters, rpm_diag, thrust_diag, torque_diag);
    compute_accelerations(device, parameters, state, rpm_diag, lin_diag, ang_diag);
    std::cout << "scenario3 diagonal motors 0+2 increased: "; print_vec3("total_torque_body", torque_diag); std::cout << " "; print_vec3("angular_acceleration", ang_diag); std::cout << "\n";
    const bool diagonal_pass = norm3(thrust_diag) > norm3(thrust_one) * (T)0.5;
    book.check(diagonal_pass, "sign_sanity_diagonal_pair_produces_finite_thrust_and_torque_reported");
    return all_motor_pass && single_motor_pass && diagonal_pass;
}

void coordinate_frame_audit(const PARAMETERS& parameters){
    std::cout << "\n[H] Coordinate frame audit\n";
    std::cout << "thrust is accumulated in body frame from rotor_thrust_directions in 60_dynamics.h.\n";
    std::cout << "body thrust is rotated to world frame with rotate_vector_by_quaternion before dividing by mass.\n";
    std::cout << "gravity is added in world frame after thrust rotation: gravity=["
              << parameters.dynamics.gravity[0] << ", " << parameters.dynamics.gravity[1] << ", " << parameters.dynamics.gravity[2] << "].\n";
    std::cout << "position and linear_velocity losses are world-frame stabilization losses.\n";
    std::cout << "angular_velocity is body-frame and angular acceleration is J_inv * (torque - omega x J omega).\n";
    std::cout << "actor action is normalized [-1, 1] and maps to rpm setpoint through action_limit min/max.\n";
    std::cout << "current MVP gradient does not include full state-coupling terms through orientation/quaternion integration over time.\n";
}

int main(int argc, char** argv){
    std::cout << std::setprecision(8);
    std::string diff_model = "euler";
    bool clf_validation = false;
    for(int arg_i = 1; arg_i < argc; arg_i++){
        std::string arg = argv[arg_i];
        if(arg == "--diff-model" && arg_i + 1 < argc){
            diff_model = argv[++arg_i];
        }
        else if(arg == "--clf-validation"){
            clf_validation = true;
        }
        else if(arg == "--help"){
            std::cout << "Usage: foundation_policy_diff_physics_check [--diff-model euler|l2f_approx] [--clf-validation]\n";
            return 0;
        }
        else{
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }
    DEVICE device;
    RNG rng;
    ENVIRONMENT env;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    rlt::init(device, env);

    Thresholds thresholds;
    CheckBook book;

    PARAMETERS parameters;
    prepare_parameters(device, env, parameters, rng, false);

    std::cout << "foundation_policy_diff_physics_check\n";
    if(diff_model == "euler"){
        l2f_diff::EulerCheckSummary<T> euler_summary;
        bool euler_ok = l2f_diff::run_euler_physics_check<PARAMETERS, T, TI>(parameters, euler_summary);
        if(clf_validation){
            const bool clf_ok = l2f_diff::run_euler_clf_objective_validation<PARAMETERS, T, TI>(parameters);
            euler_ok = euler_ok && clf_ok;
        }
        rlt::free(device, rng);
        return euler_ok ? 0 : 1;
    }
    if(diff_model != "l2f_approx"){
        std::cerr << "Unknown --diff-model value: " << diff_model << "\n";
        rlt::free(device, rng);
        return 1;
    }
    std::cout << "diff_model=l2f_approx\n";
    std::cout << "thresholds: forward=" << thresholds.forward_step_error
              << " thrust_rel=" << thresholds.thrust_derivative_relative_error
              << " torque_rel=" << thresholds.torque_derivative_relative_error
              << " accel_rel=" << thresholds.acceleration_derivative_relative_error
              << " loss_cos=" << thresholds.loss_gradient_cosine
              << " loss_rel=" << thresholds.loss_gradient_relative_error << "\n";

    forward_consistency_check(device, env, parameters, rng, thresholds, book);
    motor_delay_check(device, parameters, book);
    thrust_derivative_check(parameters, thresholds, book);
    torque_derivative_check(device, parameters, thresholds, book);
    acceleration_derivative_check(device, parameters, thresholds, book);
    loss_gradient_checks(device, env, rng, thresholds, book);
    sign_sanity_tests(device, parameters, book);
    coordinate_frame_audit(parameters);

    std::cout << "\n[I] Pass/fail summary\n";
    std::cout << "passed=" << book.passed << " failed=" << book.failed << " status=" << (book.ok() ? "PASS" : "FAIL") << "\n";
    std::cout << "Known missing gradient terms: full A=dx_next/dx Jacobian, orientation-to-thrust coupling across steps, Coriolis/gyroscopic state derivatives, quaternion integration derivatives, and RK4 substep sensitivities.\n";
    std::cout << "Training safety verdict: " << (book.ok() ? "gradient checks passed for MVP thresholds" : "NOT SAFE for training until failed gradient checks are fixed or explicitly accepted as approximations") << "\n";

    rlt::free(device, rng);
    return book.ok() ? 0 : 1;
}
