#include <torch/types.h>

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

__device__ inline void matTvec3(const float a[9], const float x[3], float out[3]) {
    for (int c = 0; c < 3; c++) {
        out[c] = a[c] * x[0] + a[3 + c] * x[1] + a[6 + c] * x[2];
    }
}

__device__ inline void transpose3(const float a[9], float out[9]) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            out[r * 3 + c] = a[c * 3 + r];
        }
    }
}

__device__ inline void solve3x3(const float a[9], const float b[3], float x[3]) {
    const float det =
        a[0] * (a[4] * a[8] - a[5] * a[7]) -
        a[1] * (a[3] * a[8] - a[5] * a[6]) +
        a[2] * (a[3] * a[7] - a[4] * a[6]);
    const float det_safe = fabsf(det) < 1.0e-12f
        ? copysignf(1.0e-12f, det == 0.0f ? 1.0f : det)
        : det;
    x[0] = (
        b[0] * (a[4] * a[8] - a[5] * a[7]) -
        a[1] * (b[1] * a[8] - a[5] * b[2]) +
        a[2] * (b[1] * a[7] - a[4] * b[2])
    ) / det_safe;
    x[1] = (
        a[0] * (b[1] * a[8] - a[5] * b[2]) -
        b[0] * (a[3] * a[8] - a[5] * a[6]) +
        a[2] * (a[3] * b[2] - b[1] * a[6])
    ) / det_safe;
    x[2] = (
        a[0] * (a[4] * b[2] - b[1] * a[7]) -
        a[1] * (a[3] * b[2] - b[1] * a[6]) +
        b[0] * (a[3] * a[7] - a[4] * a[6])
    ) / det_safe;
}

__device__ inline void gyro_jacobian3(const float omega[3], const float inertia[3], float out[9]) {
    const float jx = inertia[0];
    const float jy = inertia[1];
    const float jz = inertia[2];
    out[0] = 0.0f;
    out[1] = (jz - jy) * omega[2];
    out[2] = (jz - jy) * omega[1];
    out[3] = (jx - jz) * omega[2];
    out[4] = 0.0f;
    out[5] = (jx - jz) * omega[0];
    out[6] = (jy - jx) * omega[1];
    out[7] = (jy - jx) * omega[0];
    out[8] = 0.0f;
}

__device__ inline void implicit_midpoint_jacobian3(
    const float omega_mid[3],
    const float inertia[3],
    float dt,
    float out[9]) {
    float cjac[9];
    gyro_jacobian3(omega_mid, inertia, cjac);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            out[r * 3 + c] = (r == c ? 1.0f : 0.0f)
                + 0.5f * dt * cjac[r * 3 + c] / inertia[r];
        }
    }
}

__device__ inline void implicit_midpoint_residual3(
    const float omega_next[3],
    const float omega[3],
    const float torque[3],
    const float inertia[3],
    float dt,
    float residual[3]) {
    float omega_mid[3], j_mid[3], gyro_mid[3];
    for (int i = 0; i < 3; i++) {
        omega_mid[i] = 0.5f * (omega[i] + omega_next[i]);
        j_mid[i] = inertia[i] * omega_mid[i];
    }
    cross3(omega_mid, j_mid, gyro_mid);
    for (int i = 0; i < 3; i++) {
        residual[i] = omega_next[i] - omega[i]
            - dt * ((torque[i] - gyro_mid[i]) / inertia[i]);
    }
}

__device__ inline void implicit_midpoint_omega3(
    const float omega[3],
    const float torque[3],
    const float inertia[3],
    float dt,
    float omega_next[3]) {
    float j_omega[3], gyro[3];
    for (int i = 0; i < 3; i++) {
        j_omega[i] = inertia[i] * omega[i];
    }
    cross3(omega, j_omega, gyro);
    for (int i = 0; i < 3; i++) {
        omega_next[i] = omega[i] + dt * ((torque[i] - gyro[i]) / inertia[i]);
    }
    for (int iter = 0; iter < 4; iter++) {
        float omega_mid[3], residual[3], jacobian[9], delta[3];
        for (int i = 0; i < 3; i++) {
            omega_mid[i] = 0.5f * (omega[i] + omega_next[i]);
        }
        implicit_midpoint_residual3(omega_next, omega, torque, inertia, dt, residual);
        implicit_midpoint_jacobian3(omega_mid, inertia, dt, jacobian);
        solve3x3(jacobian, residual, delta);
        for (int i = 0; i < 3; i++) {
            omega_next[i] -= delta[i];
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
    const float* __restrict__ external_force,
    const float* __restrict__ mass,
    const float* __restrict__ thrust_coeff_c0,
    const float* __restrict__ thrust_coeff_c1,
    const float* __restrict__ thrust_coeff_c2,
    const float* __restrict__ motor_time_rising,
    const float* __restrict__ motor_time_falling,
    const float* __restrict__ arm_length,
    const float* __restrict__ inertia_x,
    const float* __restrict__ inertia_y,
    const float* __restrict__ inertia_z,
    const float* __restrict__ rotor_torque_constant,
    float* __restrict__ out_position,
    float* __restrict__ out_velocity,
    float* __restrict__ out_rotation,
    float* __restrict__ out_omega,
    float* __restrict__ out_motor,
    float* __restrict__ out_previous_action,
    int batch,
    float dt,
    float gravity) {
    const int bidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= batch) {
        return;
    }

    const float mass_b = mass[bidx];
    const float arm_b = arm_length[bidx];
    const float inertia[3] = {inertia_x[bidx], inertia_y[bidx], inertia_z[bidx]};
    const float yaw_b = rotor_torque_constant[bidx];
    const int coeff_base = bidx * 4;
    float command[4];
    float next_motor[4];
    float thrust[4];
    for (int i = 0; i < 4; i++) {
        const int idx4 = bidx * 4 + i;
        command[i] = clampf(action[idx4], -1.0f, 1.0f);
        const float tau = command[i] >= motor[idx4]
            ? motor_time_rising[bidx]
            : motor_time_falling[bidx];
        const float alpha = clampf(dt / tau, 0.0f, 1.0f);
        next_motor[i] = motor[idx4] + alpha * (command[i] - motor[idx4]);
        const float thrust_pre =
            thrust_coeff_c0[coeff_base + i]
            + thrust_coeff_c1[coeff_base + i] * next_motor[i]
            + thrust_coeff_c2[coeff_base + i] * next_motor[i] * next_motor[i];
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
    acceleration[0] = r[2] * (total_thrust / mass_b) + external_force[bidx * 3 + 0] / mass_b;
    acceleration[1] = r[5] * (total_thrust / mass_b) + external_force[bidx * 3 + 1] / mass_b;
    acceleration[2] = r[8] * (total_thrust / mass_b) - gravity + external_force[bidx * 3 + 2] / mass_b;
    for (int i = 0; i < 3; i++) {
        const int idx3 = bidx * 3 + i;
        const float next_velocity = velocity[idx3] + dt * acceleration[i];
        out_velocity[idx3] = next_velocity;
        out_position[idx3] = position[idx3] + dt * next_velocity;
    }

    const float torque[3] = {
        arm_b * (thrust[1] - thrust[3]),
        arm_b * (thrust[2] - thrust[0]),
        yaw_b * (thrust[0] - thrust[1] + thrust[2] - thrust[3]),
    };
    float w[3];
    for (int i = 0; i < 3; i++) {
        w[i] = omega[bidx * 3 + i];
    }
    float next_omega[3];
    implicit_midpoint_omega3(w, torque, inertia, dt, next_omega);
    for (int i = 0; i < 3; i++) {
        out_omega[bidx * 3 + i] = next_omega[i];
    }

    const float phi[3] = {
        0.5f * dt * (w[0] + next_omega[0]),
        0.5f * dt * (w[1] + next_omega[1]),
        0.5f * dt * (w[2] + next_omega[2]),
    };
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
    const float* __restrict__ mass,
    const float* __restrict__ thrust_coeff_c0,
    const float* __restrict__ thrust_coeff_c1,
    const float* __restrict__ thrust_coeff_c2,
    const float* __restrict__ motor_time_rising,
    const float* __restrict__ motor_time_falling,
    const float* __restrict__ arm_length,
    const float* __restrict__ inertia_x,
    const float* __restrict__ inertia_y,
    const float* __restrict__ inertia_z,
    const float* __restrict__ rotor_torque_constant,
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
    float grad_decay) {
    const int bidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= batch) {
        return;
    }

    const float mass_b = mass[bidx];
    const float arm_b = arm_length[bidx];
    const float inertia[3] = {inertia_x[bidx], inertia_y[bidx], inertia_z[bidx]};
    const float yaw_b = rotor_torque_constant[bidx];
    const int coeff_base = bidx * 4;

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
        const float tau = command[i] >= motor[idx4]
            ? motor_time_rising[bidx]
            : motor_time_falling[bidx];
        const float alpha = clampf(dt / tau, 0.0f, 1.0f);
        next_motor[i] = motor[idx4] + alpha * (command[i] - motor[idx4]);
        const float thrust_pre =
            thrust_coeff_c0[coeff_base + i]
            + thrust_coeff_c1[coeff_base + i] * next_motor[i]
            + thrust_coeff_c2[coeff_base + i] * next_motor[i] * next_motor[i];
        thrust[i] = fmaxf(thrust_pre, 0.0f);
    }

    const float torque[3] = {
        arm_b * (thrust[1] - thrust[3]),
        arm_b * (thrust[2] - thrust[0]),
        yaw_b * (thrust[0] - thrust[1] + thrust[2] - thrust[3]),
    };
    float w[3];
    for (int i = 0; i < 3; i++) {
        w[i] = omega[bidx * 3 + i];
    }
    float next_omega[3];
    implicit_midpoint_omega3(w, torque, inertia, dt, next_omega);

    const float omega_mid[3] = {
        0.5f * (w[0] + next_omega[0]),
        0.5f * (w[1] + next_omega[1]),
        0.5f * (w[2] + next_omega[2]),
    };
    const float phi[3] = {dt * omega_mid[0], dt * omega_mid[1], dt * omega_mid[2]};
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
        const float gmid = dt * gphi;
        gw_in[axis] += 0.5f * gmid;
        gw_next[axis] += 0.5f * gmid;
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
        gr_in[i * 3 + 2] += gacc[i] * (total_thrust / mass_b);
        gtotal_thrust += gacc[i] * body_z / mass_b;
    }
    for (int i = 0; i < 4; i++) {
        gthrust[i] += gtotal_thrust;
    }

    float implicit_jacobian[9], implicit_jacobian_t[9];
    float lambda[3], inv_inertia_lambda[3], gyro_jacobian[9], gyro_term[3], gtorque[3];
    implicit_midpoint_jacobian3(omega_mid, inertia, dt, implicit_jacobian);
    transpose3(implicit_jacobian, implicit_jacobian_t);
    solve3x3(implicit_jacobian_t, gw_next, lambda);
    gyro_jacobian3(omega_mid, inertia, gyro_jacobian);
    for (int i = 0; i < 3; i++) {
        inv_inertia_lambda[i] = lambda[i] / inertia[i];
        gtorque[i] = dt * inv_inertia_lambda[i];
    }
    matTvec3(gyro_jacobian, inv_inertia_lambda, gyro_term);
    for (int i = 0; i < 3; i++) {
        gw_in[i] += lambda[i] - 0.5f * dt * gyro_term[i];
    }

    gthrust[1] += arm_b * gtorque[0];
    gthrust[3] -= arm_b * gtorque[0];
    gthrust[2] += arm_b * gtorque[1];
    gthrust[0] -= arm_b * gtorque[1];
    gthrust[0] += yaw_b * gtorque[2];
    gthrust[1] -= yaw_b * gtorque[2];
    gthrust[2] += yaw_b * gtorque[2];
    gthrust[3] -= yaw_b * gtorque[2];

    for (int i = 0; i < 4; i++) {
        const float thrust_pre =
            thrust_coeff_c0[coeff_base + i]
            + thrust_coeff_c1[coeff_base + i] * next_motor[i]
            + thrust_coeff_c2[coeff_base + i] * next_motor[i] * next_motor[i];
        const float dthrust = thrust_coeff_c1[coeff_base + i]
            + 2.0f * thrust_coeff_c2[coeff_base + i] * next_motor[i];
        if (thrust_pre > 0.0f) {
            gm_next[i] += gthrust[i] * dthrust;
        }
        const float tau = command[i] >= motor[bidx * 4 + i]
            ? motor_time_rising[bidx]
            : motor_time_falling[bidx];
        const float alpha = clampf(dt / tau, 0.0f, 1.0f);
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
    torch::Tensor external_force,
    torch::Tensor mass,
    torch::Tensor thrust_coeff_c0,
    torch::Tensor thrust_coeff_c1,
    torch::Tensor thrust_coeff_c2,
    torch::Tensor motor_time_rising,
    torch::Tensor motor_time_falling,
    torch::Tensor arm_length,
    torch::Tensor inertia_x,
    torch::Tensor inertia_y,
    torch::Tensor inertia_z,
    torch::Tensor rotor_torque_constant,
    double dt,
    double gravity) {
    const c10::cuda::CUDAGuard device_guard(position.device());
    auto out_position = at::empty_like(position);
    auto out_velocity = at::empty_like(velocity);
    auto out_rotation = at::empty_like(rotation);
    auto out_omega = at::empty_like(omega);
    auto out_motor = at::empty_like(motor);
    auto out_previous_action = at::empty_like(action);

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
        external_force.data_ptr<float>(),
        mass.data_ptr<float>(),
        thrust_coeff_c0.data_ptr<float>(),
        thrust_coeff_c1.data_ptr<float>(),
        thrust_coeff_c2.data_ptr<float>(),
        motor_time_rising.data_ptr<float>(),
        motor_time_falling.data_ptr<float>(),
        arm_length.data_ptr<float>(),
        inertia_x.data_ptr<float>(),
        inertia_y.data_ptr<float>(),
        inertia_z.data_ptr<float>(),
        rotor_torque_constant.data_ptr<float>(),
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_rotation.data_ptr<float>(),
        out_omega.data_ptr<float>(),
        out_motor.data_ptr<float>(),
        out_previous_action.data_ptr<float>(),
        batch,
        static_cast<float>(dt),
        static_cast<float>(gravity));
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
    torch::Tensor mass,
    torch::Tensor thrust_coeff_c0,
    torch::Tensor thrust_coeff_c1,
    torch::Tensor thrust_coeff_c2,
    torch::Tensor motor_time_rising,
    torch::Tensor motor_time_falling,
    torch::Tensor arm_length,
    torch::Tensor inertia_x,
    torch::Tensor inertia_y,
    torch::Tensor inertia_z,
    torch::Tensor rotor_torque_constant,
    torch::Tensor grad_position,
    torch::Tensor grad_velocity,
    torch::Tensor grad_rotation,
    torch::Tensor grad_omega,
    torch::Tensor grad_motor,
    torch::Tensor grad_previous_action,
    double dt,
    double grad_decay) {
    const c10::cuda::CUDAGuard device_guard(position.device());
    auto out_grad_position = at::empty_like(position);
    auto out_grad_velocity = at::empty_like(velocity);
    auto out_grad_rotation = at::empty_like(rotation);
    auto out_grad_omega = at::empty_like(omega);
    auto out_grad_motor = at::empty_like(motor);
    auto out_grad_action = at::empty_like(action);

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
        mass.data_ptr<float>(),
        thrust_coeff_c0.data_ptr<float>(),
        thrust_coeff_c1.data_ptr<float>(),
        thrust_coeff_c2.data_ptr<float>(),
        motor_time_rising.data_ptr<float>(),
        motor_time_falling.data_ptr<float>(),
        arm_length.data_ptr<float>(),
        inertia_x.data_ptr<float>(),
        inertia_y.data_ptr<float>(),
        inertia_z.data_ptr<float>(),
        rotor_torque_constant.data_ptr<float>(),
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
