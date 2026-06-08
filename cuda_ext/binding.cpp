#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> l2f_step_forward_cuda(
    torch::Tensor position,
    torch::Tensor velocity,
    torch::Tensor rotation,
    torch::Tensor omega,
    torch::Tensor motor,
    torch::Tensor action,
    torch::Tensor external_force,
    double dt,
    double mass,
    double gravity,
    double arm_length,
    double yaw_drag,
    double motor_tau,
    double motor_authority,
    double inertia_x,
    double inertia_y,
    double inertia_z);

std::vector<torch::Tensor> l2f_step_backward_cuda(
    torch::Tensor position,
    torch::Tensor velocity,
    torch::Tensor rotation,
    torch::Tensor omega,
    torch::Tensor motor,
    torch::Tensor action,
    torch::Tensor grad_position,
    torch::Tensor grad_velocity,
    torch::Tensor grad_rotation,
    torch::Tensor grad_omega,
    torch::Tensor grad_motor,
    torch::Tensor grad_previous_action,
    double dt,
    double mass,
    double gravity,
    double arm_length,
    double yaw_drag,
    double motor_tau,
    double motor_authority,
    double inertia_x,
    double inertia_y,
    double inertia_z,
    double grad_decay);

std::vector<torch::Tensor> l2f_full_rollout_cuda(
    torch::Tensor position0,
    torch::Tensor velocity0,
    torch::Tensor rotation0,
    torch::Tensor omega0,
    torch::Tensor motor0,
    torch::Tensor previous_action0,
    torch::Tensor external_force0,
    torch::Tensor encoder0_w,
    torch::Tensor encoder0_b,
    torch::Tensor encoder1_w,
    torch::Tensor encoder1_b,
    torch::Tensor gru_w_ih,
    torch::Tensor gru_w_hh,
    torch::Tensor gru_b_ih,
    torch::Tensor gru_b_hh,
    torch::Tensor motor_head_w,
    torch::Tensor motor_head_b,
    int horizon,
    int tail_steps,
    double dt,
    double mass,
    double gravity,
    double arm_length,
    double yaw_drag,
    double motor_tau,
    double motor_authority,
    double inertia_x,
    double inertia_y,
    double inertia_z,
    double state_grad_decay,
    double hidden_grad_decay,
    double p_scale,
    double v_scale,
    double omega_scale,
    double huber_beta,
    double w_p,
    double w_v,
    double w_r,
    double w_omega,
    double clf_kappa,
    double u_soft,
    double lambda_clf,
    double lambda_out,
    double lambda_tail,
    double lambda_du,
    double lambda_ddu,
    double lambda_sat,
    double negative_slope,
    int noise_seed,
    double external_torque_max,
    double action_noise_max,
    double observation_noise_max);

namespace {

void check_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat, name, " must be float32");
}

void check_shapes(
    const torch::Tensor& position,
    const torch::Tensor& velocity,
    const torch::Tensor& rotation,
    const torch::Tensor& omega,
    const torch::Tensor& motor,
    const torch::Tensor& action) {
    TORCH_CHECK(position.dim() == 2 && position.size(1) == 3, "position must have shape [B, 3]");
    const auto batch = position.size(0);
    TORCH_CHECK(velocity.sizes() == position.sizes(), "velocity must have shape [B, 3]");
    TORCH_CHECK(omega.sizes() == position.sizes(), "omega must have shape [B, 3]");
    TORCH_CHECK(rotation.dim() == 3 && rotation.size(0) == batch && rotation.size(1) == 3 && rotation.size(2) == 3,
        "rotation must have shape [B, 3, 3]");
    TORCH_CHECK(motor.dim() == 2 && motor.size(0) == batch && motor.size(1) == 4, "motor must have shape [B, 4]");
    TORCH_CHECK(action.sizes() == motor.sizes(), "action must have shape [B, 4]");
}

void check_2d(const torch::Tensor& tensor, const char* name, int64_t rows, int64_t cols) {
    check_tensor(tensor, name);
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == cols,
        name, " must have shape [", rows, ", ", cols, "]");
}

void check_1d(const torch::Tensor& tensor, const char* name, int64_t size) {
    check_tensor(tensor, name);
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == size,
        name, " must have shape [", size, "]");
}

void check_state_inputs(
    const torch::Tensor& position,
    const torch::Tensor& velocity,
    const torch::Tensor& rotation,
    const torch::Tensor& omega,
    const torch::Tensor& motor,
    const torch::Tensor& action) {
    check_tensor(position, "position");
    check_tensor(velocity, "velocity");
    check_tensor(rotation, "rotation");
    check_tensor(omega, "omega");
    check_tensor(motor, "motor");
    check_tensor(action, "action");
    check_shapes(position, velocity, rotation, omega, motor, action);
}

} // namespace

std::vector<torch::Tensor> step_forward(
    torch::Tensor position,
    torch::Tensor velocity,
    torch::Tensor rotation,
    torch::Tensor omega,
    torch::Tensor motor,
    torch::Tensor action,
    torch::Tensor external_force,
    double dt,
    double mass,
    double gravity,
    double arm_length,
    double yaw_drag,
    double motor_tau,
    double motor_authority,
    double inertia_x,
    double inertia_y,
    double inertia_z) {
    check_state_inputs(position, velocity, rotation, omega, motor, action);
    check_2d(external_force, "external_force", position.size(0), 3);
    return l2f_step_forward_cuda(
        position, velocity, rotation, omega, motor, action, external_force,
        dt, mass, gravity, arm_length, yaw_drag, motor_tau, motor_authority,
        inertia_x, inertia_y, inertia_z);
}

std::vector<torch::Tensor> step_backward(
    torch::Tensor position,
    torch::Tensor velocity,
    torch::Tensor rotation,
    torch::Tensor omega,
    torch::Tensor motor,
    torch::Tensor action,
    torch::Tensor grad_position,
    torch::Tensor grad_velocity,
    torch::Tensor grad_rotation,
    torch::Tensor grad_omega,
    torch::Tensor grad_motor,
    torch::Tensor grad_previous_action,
    double dt,
    double mass,
    double gravity,
    double arm_length,
    double yaw_drag,
    double motor_tau,
    double motor_authority,
    double inertia_x,
    double inertia_y,
    double inertia_z,
    double grad_decay) {
    check_state_inputs(position, velocity, rotation, omega, motor, action);
    check_state_inputs(grad_position, grad_velocity, grad_rotation, grad_omega, grad_motor, grad_previous_action);
    return l2f_step_backward_cuda(
        position, velocity, rotation, omega, motor, action,
        grad_position, grad_velocity, grad_rotation, grad_omega, grad_motor, grad_previous_action,
        dt, mass, gravity, arm_length, yaw_drag, motor_tau, motor_authority,
        inertia_x, inertia_y, inertia_z, grad_decay);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step_forward", &step_forward, "L2F Euler step forward (CUDA)");
    m.def("step_backward", &step_backward, "L2F Euler step analytic VJP (CUDA)");
    m.def("full_rollout", [](torch::Tensor position0,
                              torch::Tensor velocity0,
                              torch::Tensor rotation0,
                              torch::Tensor omega0,
                              torch::Tensor motor0,
                              torch::Tensor previous_action0,
                              torch::Tensor external_force0,
                              torch::Tensor encoder0_w,
                              torch::Tensor encoder0_b,
                              torch::Tensor encoder1_w,
                              torch::Tensor encoder1_b,
                              torch::Tensor gru_w_ih,
                              torch::Tensor gru_w_hh,
                              torch::Tensor gru_b_ih,
                              torch::Tensor gru_b_hh,
                              torch::Tensor motor_head_w,
                              torch::Tensor motor_head_b,
                              int horizon,
                              int tail_steps,
                              double dt,
                              double mass,
                              double gravity,
                              double arm_length,
                              double yaw_drag,
                              double motor_tau,
                              double motor_authority,
                              double inertia_x,
                              double inertia_y,
                              double inertia_z,
                              double state_grad_decay,
                              double hidden_grad_decay,
                              double p_scale,
                              double v_scale,
                              double omega_scale,
                              double huber_beta,
                              double w_p,
                              double w_v,
                              double w_r,
                              double w_omega,
                              double clf_kappa,
                              double u_soft,
                              double lambda_clf,
                              double lambda_out,
                              double lambda_tail,
                              double lambda_du,
                              double lambda_ddu,
                              double lambda_sat,
                              double negative_slope,
                              int noise_seed,
                              double external_torque_max,
                              double action_noise_max,
                              double observation_noise_max) {
        check_state_inputs(position0, velocity0, rotation0, omega0, motor0, previous_action0);
        check_2d(external_force0, "external_force0", position0.size(0), 3);
        check_2d(encoder0_w, "encoder0_w", 192, 40);
        check_1d(encoder0_b, "encoder0_b", 192);
        check_2d(encoder1_w, "encoder1_w", 192, 192);
        check_1d(encoder1_b, "encoder1_b", 192);
        check_2d(gru_w_ih, "gru_w_ih", 576, 192);
        check_2d(gru_w_hh, "gru_w_hh", 576, 192);
        check_1d(gru_b_ih, "gru_b_ih", 576);
        check_1d(gru_b_hh, "gru_b_hh", 576);
        check_2d(motor_head_w, "motor_head_w", 4, 192);
        check_1d(motor_head_b, "motor_head_b", 4);
        TORCH_CHECK(horizon > 0, "horizon must be positive");
        return l2f_full_rollout_cuda(
            position0, velocity0, rotation0, omega0, motor0, previous_action0, external_force0,
            encoder0_w, encoder0_b, encoder1_w, encoder1_b,
            gru_w_ih, gru_w_hh, gru_b_ih, gru_b_hh,
            motor_head_w, motor_head_b,
            horizon, tail_steps,
            dt, mass, gravity, arm_length, yaw_drag, motor_tau, motor_authority,
            inertia_x, inertia_y, inertia_z,
            state_grad_decay, hidden_grad_decay,
            p_scale, v_scale, omega_scale, huber_beta,
            w_p, w_v, w_r, w_omega, clf_kappa, u_soft,
            lambda_clf, lambda_out, lambda_tail, lambda_du, lambda_ddu, lambda_sat,
            negative_slope, noise_seed, external_torque_max,
            action_noise_max, observation_noise_max);
    }, "Full H-step L2F rollout, loss, physics VJP, and actor backward (CUDA)");
}
