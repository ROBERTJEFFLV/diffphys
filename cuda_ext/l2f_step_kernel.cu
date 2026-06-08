#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <vector>

namespace {

__device__ inline float clampf(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

__device__ inline void cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void skew3(const float v[3], float k[9]) {
    k[0] = 0.0f;
    k[1] = -v[2];
    k[2] = v[1];
    k[3] = v[2];
    k[4] = 0.0f;
    k[5] = -v[0];
    k[6] = -v[1];
    k[7] = v[0];
    k[8] = 0.0f;
}

__device__ inline void matmul3(const float a[9], const float b[9], float out[9]) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float value = 0.0f;
            for (int k = 0; k < 3; k++) {
                value += a[r * 3 + k] * b[k * 3 + c];
            }
            out[r * 3 + c] = value;
        }
    }
}

__device__ inline void so3_exp(const float phi[3], float e[9]) {
    float k[9];
    float k2[9];
    skew3(phi, k);
    matmul3(k, k, k2);

    const float theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    float a;
    float b;
    if (theta2 < 1.0e-8f) {
        const float theta4 = theta2 * theta2;
        a = 1.0f - theta2 / 6.0f + theta4 / 120.0f;
        b = 0.5f - theta2 / 24.0f + theta4 / 720.0f;
    } else {
        const float theta = sqrtf(theta2);
        a = sinf(theta) / theta;
        b = (1.0f - cosf(theta)) / theta2;
    }

    for (int i = 0; i < 9; i++) {
        e[i] = a * k[i] + b * k2[i];
    }
    e[0] += 1.0f;
    e[4] += 1.0f;
    e[8] += 1.0f;
}

__device__ inline void so3_exp_with_derivatives(const float phi[3], float e[9], float de[3][9]) {
    float k[9];
    float k2[9];
    skew3(phi, k);
    matmul3(k, k, k2);

    const float theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    float a;
    float b;
    float da_coeff;
    float db_coeff;
    if (theta2 < 1.0e-8f) {
        const float theta4 = theta2 * theta2;
        a = 1.0f - theta2 / 6.0f + theta4 / 120.0f;
        b = 0.5f - theta2 / 24.0f + theta4 / 720.0f;
        da_coeff = -1.0f / 3.0f + theta2 / 30.0f - theta4 / 840.0f;
        db_coeff = -1.0f / 12.0f + theta2 / 180.0f - theta4 / 6720.0f;
    } else {
        const float theta = sqrtf(theta2);
        const float sin_t = sinf(theta);
        const float cos_t = cosf(theta);
        a = sin_t / theta;
        b = (1.0f - cos_t) / theta2;
        const float da_dtheta = (theta * cos_t - sin_t) / theta2;
        const float db_dtheta = (theta * sin_t - 2.0f * (1.0f - cos_t)) / (theta2 * theta);
        da_coeff = da_dtheta / theta;
        db_coeff = db_dtheta / theta;
    }

    for (int i = 0; i < 9; i++) {
        e[i] = a * k[i] + b * k2[i];
    }
    e[0] += 1.0f;
    e[4] += 1.0f;
    e[8] += 1.0f;

    const float dk[3][9] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f},
        {0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    };
    for (int axis = 0; axis < 3; axis++) {
        float dk_k[9];
        float k_dk[9];
        matmul3(dk[axis], k, dk_k);
        matmul3(k, dk[axis], k_dk);
        const float da = da_coeff * phi[axis];
        const float db = db_coeff * phi[axis];
        for (int i = 0; i < 9; i++) {
            de[axis][i] =
                da * k[i] +
                a * dk[axis][i] +
                db * k2[i] +
                b * (dk_k[i] + k_dk[i]);
        }
    }
}

__global__ void l2f_step_forward_kernel(
    const float* __restrict__ position,
    const float* __restrict__ velocity,
    const float* __restrict__ rotation,
    const float* __restrict__ omega,
    const float* __restrict__ motor,
    const float* __restrict__ action,
    float* __restrict__ out_position,
    float* __restrict__ out_velocity,
    float* __restrict__ out_rotation,
    float* __restrict__ out_omega,
    float* __restrict__ out_motor,
    float* __restrict__ out_previous_action,
    int batch,
    float dt,
    float mass,
    float gravity,
    float arm_length,
    float yaw_drag,
    float motor_tau,
    float motor_authority,
    float inertia_x,
    float inertia_y,
    float inertia_z) {
    const int bidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= batch) {
        return;
    }

    const float alpha = clampf(dt / motor_tau, 0.0f, 1.0f);
    const float hover_thrust = mass * gravity * 0.25f;
    float command[4];
    float next_motor[4];
    float thrust[4];
    for (int i = 0; i < 4; i++) {
        const int idx4 = bidx * 4 + i;
        command[i] = clampf(action[idx4], -1.0f, 1.0f);
        next_motor[i] = motor[idx4] + alpha * (command[i] - motor[idx4]);
        const float thrust_pre = hover_thrust * (1.0f + motor_authority * next_motor[i]);
        thrust[i] = fmaxf(thrust_pre, 0.0f);
        out_motor[idx4] = next_motor[i];
        out_previous_action[idx4] = command[i];
    }

    float r[9];
    for (int i = 0; i < 9; i++) {
        r[i] = rotation[bidx * 9 + i];
    }

    const float total_thrust = thrust[0] + thrust[1] + thrust[2] + thrust[3];
    float acceleration[3];
    acceleration[0] = r[2] * (total_thrust / mass);
    acceleration[1] = r[5] * (total_thrust / mass);
    acceleration[2] = r[8] * (total_thrust / mass) - gravity;
    for (int i = 0; i < 3; i++) {
        const int idx3 = bidx * 3 + i;
        const float next_velocity = velocity[idx3] + dt * acceleration[i];
        out_velocity[idx3] = next_velocity;
        out_position[idx3] = position[idx3] + dt * next_velocity;
    }

    const float torque[3] = {
        arm_length * (thrust[1] - thrust[3]),
        arm_length * (thrust[2] - thrust[0]),
        yaw_drag * (thrust[0] - thrust[1] + thrust[2] - thrust[3]),
    };
    const float inertia[3] = {inertia_x, inertia_y, inertia_z};
    float w[3];
    float iw[3];
    for (int i = 0; i < 3; i++) {
        w[i] = omega[bidx * 3 + i];
        iw[i] = inertia[i] * w[i];
    }
    float omega_cross[3];
    cross3(w, iw, omega_cross);
    float next_omega[3];
    for (int i = 0; i < 3; i++) {
        next_omega[i] = w[i] + dt * ((torque[i] - omega_cross[i]) / inertia[i]);
        out_omega[bidx * 3 + i] = next_omega[i];
    }

    const float phi[3] = {dt * next_omega[0], dt * next_omega[1], dt * next_omega[2]};
    float e[9];
    float next_rotation[9];
    so3_exp(phi, e);
    matmul3(r, e, next_rotation);
    for (int i = 0; i < 9; i++) {
        out_rotation[bidx * 9 + i] = next_rotation[i];
    }
}

__global__ void l2f_step_backward_kernel(
    const float* __restrict__ position,
    const float* __restrict__ velocity,
    const float* __restrict__ rotation,
    const float* __restrict__ omega,
    const float* __restrict__ motor,
    const float* __restrict__ action,
    const float* __restrict__ grad_position,
    const float* __restrict__ grad_velocity,
    const float* __restrict__ grad_rotation,
    const float* __restrict__ grad_omega,
    const float* __restrict__ grad_motor,
    const float* __restrict__ grad_previous_action,
    float* __restrict__ out_grad_position,
    float* __restrict__ out_grad_velocity,
    float* __restrict__ out_grad_rotation,
    float* __restrict__ out_grad_omega,
    float* __restrict__ out_grad_motor,
    float* __restrict__ out_grad_action,
    int batch,
    float dt,
    float mass,
    float gravity,
    float arm_length,
    float yaw_drag,
    float motor_tau,
    float motor_authority,
    float inertia_x,
    float inertia_y,
    float inertia_z,
    float grad_decay) {
    const int bidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= batch) {
        return;
    }

    const float alpha = clampf(dt / motor_tau, 0.0f, 1.0f);
    const float hover_thrust = mass * gravity * 0.25f;
    const float inertia[3] = {inertia_x, inertia_y, inertia_z};

    float r[9];
    for (int i = 0; i < 9; i++) {
        r[i] = rotation[bidx * 9 + i];
    }
    float command[4];
    float next_motor[4];
    float thrust[4];
    for (int i = 0; i < 4; i++) {
        const int idx4 = bidx * 4 + i;
        command[i] = clampf(action[idx4], -1.0f, 1.0f);
        next_motor[i] = motor[idx4] + alpha * (command[i] - motor[idx4]);
        const float thrust_pre = hover_thrust * (1.0f + motor_authority * next_motor[i]);
        thrust[i] = fmaxf(thrust_pre, 0.0f);
    }

    const float torque[3] = {
        arm_length * (thrust[1] - thrust[3]),
        arm_length * (thrust[2] - thrust[0]),
        yaw_drag * (thrust[0] - thrust[1] + thrust[2] - thrust[3]),
    };
    float w[3];
    float iw[3];
    for (int i = 0; i < 3; i++) {
        w[i] = omega[bidx * 3 + i];
        iw[i] = inertia[i] * w[i];
    }
    float omega_cross[3];
    cross3(w, iw, omega_cross);
    float next_omega[3];
    for (int i = 0; i < 3; i++) {
        next_omega[i] = w[i] + dt * ((torque[i] - omega_cross[i]) / inertia[i]);
    }

    const float phi[3] = {dt * next_omega[0], dt * next_omega[1], dt * next_omega[2]};
    float e[9];
    float de[3][9];
    so3_exp_with_derivatives(phi, e, de);

    float gp[3];
    float gv_next[3];
    float gr_next[9];
    float gw_next[3];
    float gm_next[4];
    float gc[4];
    for (int i = 0; i < 3; i++) {
        gp[i] = grad_decay * grad_position[bidx * 3 + i];
        gv_next[i] = grad_decay * grad_velocity[bidx * 3 + i];
        gw_next[i] = grad_decay * grad_omega[bidx * 3 + i];
    }
    for (int i = 0; i < 9; i++) {
        gr_next[i] = grad_decay * grad_rotation[bidx * 9 + i];
    }
    for (int i = 0; i < 4; i++) {
        gm_next[i] = grad_decay * grad_motor[bidx * 4 + i];
        gc[i] = grad_decay * grad_previous_action[bidx * 4 + i];
    }

    float gp_in[3] = {0.0f, 0.0f, 0.0f};
    float gv_in[3] = {0.0f, 0.0f, 0.0f};
    float gr_in[9] = {0.0f};
    float gw_in[3] = {0.0f, 0.0f, 0.0f};
    float gm_in[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float ga_in[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float gthrust[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float ge[9] = {0.0f};
    for (int rr = 0; rr < 3; rr++) {
        for (int cc = 0; cc < 3; cc++) {
            float v_r = 0.0f;
            for (int kk = 0; kk < 3; kk++) {
                v_r += gr_next[rr * 3 + kk] * e[cc * 3 + kk];
                ge[cc * 3 + kk] += r[rr * 3 + cc] * gr_next[rr * 3 + kk];
            }
            gr_in[rr * 3 + cc] += v_r;
        }
    }
    for (int axis = 0; axis < 3; axis++) {
        float gphi = 0.0f;
        for (int i = 0; i < 9; i++) {
            gphi += ge[i] * de[axis][i];
        }
        gw_next[axis] += dt * gphi;
    }

    float gv_from_position[3];
    float gacc[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; i++) {
        gp_in[i] += gp[i];
        gv_from_position[i] = gv_next[i] + dt * gp[i];
        gv_in[i] += gv_from_position[i];
        gacc[i] += dt * gv_from_position[i];
    }

    const float total_thrust = thrust[0] + thrust[1] + thrust[2] + thrust[3];
    float gtotal_thrust = 0.0f;
    for (int i = 0; i < 3; i++) {
        const float body_z = r[i * 3 + 2];
        gr_in[i * 3 + 2] += gacc[i] * (total_thrust / mass);
        gtotal_thrust += gacc[i] * body_z / mass;
    }
    for (int i = 0; i < 4; i++) {
        gthrust[i] += gtotal_thrust;
    }

    float gy[3];
    float gtorque[3];
    for (int i = 0; i < 3; i++) {
        gw_in[i] += gw_next[i];
        gtorque[i] = gw_next[i] * dt / inertia[i];
        gy[i] = -gw_next[i] * dt / inertia[i];
    }
    float tmp_a[3];
    float tmp_b[3];
    cross3(iw, gy, tmp_a);
    cross3(gy, w, tmp_b);
    for (int i = 0; i < 3; i++) {
        gw_in[i] += tmp_a[i] + inertia[i] * tmp_b[i];
    }

    gthrust[1] += arm_length * gtorque[0];
    gthrust[3] -= arm_length * gtorque[0];
    gthrust[2] += arm_length * gtorque[1];
    gthrust[0] -= arm_length * gtorque[1];
    gthrust[0] += yaw_drag * gtorque[2];
    gthrust[1] -= yaw_drag * gtorque[2];
    gthrust[2] += yaw_drag * gtorque[2];
    gthrust[3] -= yaw_drag * gtorque[2];

    for (int i = 0; i < 4; i++) {
        const float thrust_pre = hover_thrust * (1.0f + motor_authority * next_motor[i]);
        if (thrust_pre > 0.0f) {
            gm_next[i] += gthrust[i] * hover_thrust * motor_authority;
        }
        gm_in[i] += (1.0f - alpha) * gm_next[i];
        gc[i] += alpha * gm_next[i];
        if (action[bidx * 4 + i] >= -1.0f && action[bidx * 4 + i] <= 1.0f) {
            ga_in[i] += gc[i];
        }
    }

    for (int i = 0; i < 3; i++) {
        out_grad_position[bidx * 3 + i] = gp_in[i];
        out_grad_velocity[bidx * 3 + i] = gv_in[i];
        out_grad_omega[bidx * 3 + i] = gw_in[i];
    }
    for (int i = 0; i < 9; i++) {
        out_grad_rotation[bidx * 9 + i] = gr_in[i];
    }
    for (int i = 0; i < 4; i++) {
        out_grad_motor[bidx * 4 + i] = gm_in[i];
        out_grad_action[bidx * 4 + i] = ga_in[i];
    }
}

} // namespace

std::vector<torch::Tensor> l2f_step_forward_cuda(
    torch::Tensor position,
    torch::Tensor velocity,
    torch::Tensor rotation,
    torch::Tensor omega,
    torch::Tensor motor,
    torch::Tensor action,
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
    const c10::cuda::CUDAGuard device_guard(position.device());
    auto out_position = torch::empty_like(position);
    auto out_velocity = torch::empty_like(velocity);
    auto out_rotation = torch::empty_like(rotation);
    auto out_omega = torch::empty_like(omega);
    auto out_motor = torch::empty_like(motor);
    auto out_previous_action = torch::empty_like(action);

    const int batch = static_cast<int>(position.size(0));
    const int threads = 128;
    const int blocks = (batch + threads - 1) / threads;
    l2f_step_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        position.data_ptr<float>(),
        velocity.data_ptr<float>(),
        rotation.data_ptr<float>(),
        omega.data_ptr<float>(),
        motor.data_ptr<float>(),
        action.data_ptr<float>(),
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_rotation.data_ptr<float>(),
        out_omega.data_ptr<float>(),
        out_motor.data_ptr<float>(),
        out_previous_action.data_ptr<float>(),
        batch,
        static_cast<float>(dt),
        static_cast<float>(mass),
        static_cast<float>(gravity),
        static_cast<float>(arm_length),
        static_cast<float>(yaw_drag),
        static_cast<float>(motor_tau),
        static_cast<float>(motor_authority),
        static_cast<float>(inertia_x),
        static_cast<float>(inertia_y),
        static_cast<float>(inertia_z));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_position, out_velocity, out_rotation, out_omega, out_motor, out_previous_action};
}

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
    double grad_decay) {
    const c10::cuda::CUDAGuard device_guard(position.device());
    auto out_grad_position = torch::empty_like(position);
    auto out_grad_velocity = torch::empty_like(velocity);
    auto out_grad_rotation = torch::empty_like(rotation);
    auto out_grad_omega = torch::empty_like(omega);
    auto out_grad_motor = torch::empty_like(motor);
    auto out_grad_action = torch::empty_like(action);

    const int batch = static_cast<int>(position.size(0));
    const int threads = 128;
    const int blocks = (batch + threads - 1) / threads;
    l2f_step_backward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        position.data_ptr<float>(),
        velocity.data_ptr<float>(),
        rotation.data_ptr<float>(),
        omega.data_ptr<float>(),
        motor.data_ptr<float>(),
        action.data_ptr<float>(),
        grad_position.data_ptr<float>(),
        grad_velocity.data_ptr<float>(),
        grad_rotation.data_ptr<float>(),
        grad_omega.data_ptr<float>(),
        grad_motor.data_ptr<float>(),
        grad_previous_action.data_ptr<float>(),
        out_grad_position.data_ptr<float>(),
        out_grad_velocity.data_ptr<float>(),
        out_grad_rotation.data_ptr<float>(),
        out_grad_omega.data_ptr<float>(),
        out_grad_motor.data_ptr<float>(),
        out_grad_action.data_ptr<float>(),
        batch,
        static_cast<float>(dt),
        static_cast<float>(mass),
        static_cast<float>(gravity),
        static_cast<float>(arm_length),
        static_cast<float>(yaw_drag),
        static_cast<float>(motor_tau),
        static_cast<float>(motor_authority),
        static_cast<float>(inertia_x),
        static_cast<float>(inertia_y),
        static_cast<float>(inertia_z),
        static_cast<float>(grad_decay));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        out_grad_position,
        out_grad_velocity,
        out_grad_rotation,
        out_grad_omega,
        out_grad_motor,
        out_grad_action,
    };
}
