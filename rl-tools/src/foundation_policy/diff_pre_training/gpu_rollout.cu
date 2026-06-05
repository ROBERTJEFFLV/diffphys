#include "gpu_rollout.h"

#include <cuda_runtime.h>

#include <rl_tools/rl/environments/l2f/diff_euler_rollout.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace rl_tools::foundation_policy::diff_pre_training::gpu{

namespace{

constexpr float DT = 0.01f;
constexpr float G = 9.81f;
constexpr std::uint64_t CPU_MAX_HORIZON = 256;
constexpr std::uint64_t GPU_EVAL_MAX_HORIZON = 4096;

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if(err__ != cudaSuccess){ \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(false)

__host__ __device__ inline std::size_t idx3(std::size_t t, std::size_t b, std::size_t d, std::size_t batch_size){
    return (t * batch_size + b) * 3 + d;
}

__host__ __device__ inline std::size_t idx4(std::size_t t, std::size_t b, std::size_t d, std::size_t batch_size){
    return (t * batch_size + b) * 4 + d;
}

__host__ __device__ inline std::size_t idx9(std::size_t t, std::size_t b, std::size_t d, std::size_t batch_size){
    return (t * batch_size + b) * 9 + d;
}

__host__ __device__ inline std::size_t idx_obs(std::size_t t, std::size_t b, std::size_t d, std::size_t batch_size){
    return (t * batch_size + b) * EULER_OBSERVATION_DIM + d;
}

__host__ __device__ inline std::size_t pidx3(std::size_t b, std::size_t d){
    return b * 3 + d;
}

__host__ __device__ inline std::size_t pidx4(std::size_t b, std::size_t d){
    return b * 4 + d;
}

__host__ __device__ inline std::size_t pidx9(std::size_t b, std::size_t d){
    return b * 9 + d;
}

__host__ __device__ inline std::size_t rotor3(std::size_t b, std::size_t r, std::size_t d){
    return (b * 4 + r) * 3 + d;
}

__host__ __device__ inline float dot3(const float a[3], const float b[3]){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__host__ __device__ inline void cross3(const float a[3], const float b[3], float out[3]){
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__host__ __device__ inline void mat_vec3(const float A[9], const float x[3], float y[3]){
    for(int i = 0; i < 3; i++){
        y[i] = 0.0f;
        for(int j = 0; j < 3; j++){
            y[i] += A[i * 3 + j] * x[j];
        }
    }
}

__host__ __device__ inline void mat_mul3(const float A[9], const float B[9], float C[9]){
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            float value = 0.0f;
            for(int k = 0; k < 3; k++){
                value += A[i * 3 + k] * B[k * 3 + j];
            }
            C[i * 3 + j] = value;
        }
    }
}

__host__ __device__ inline float mat_inner3(const float A[9], const float B[9]){
    float value = 0.0f;
    for(int i = 0; i < 9; i++){
        value += A[i] * B[i];
    }
    return value;
}

__host__ __device__ inline void skew3(const float w[3], float S[9]){
    S[0] = 0.0f;   S[1] = -w[2];  S[2] = w[1];
    S[3] = w[2];   S[4] = 0.0f;   S[5] = -w[0];
    S[6] = -w[1];  S[7] = w[0];   S[8] = 0.0f;
}

__host__ __device__ inline void basis_skew3(int idx, float E[9]){
    for(int i = 0; i < 9; i++){
        E[i] = 0.0f;
    }
    if(idx == 0){
        E[5] = -1.0f; E[7] = 1.0f;
    }
    else if(idx == 1){
        E[2] = 1.0f; E[6] = -1.0f;
    }
    else{
        E[1] = -1.0f; E[3] = 1.0f;
    }
}

__host__ __device__ inline void so3_expmap_delta(float dt, const float omega[3], float Delta[9]){
    float phi[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
    const float theta_sq = dot3(phi, phi);
    const float theta = sqrtf(theta_sq);
    float K[9];
    skew3(phi, K);
    float K2[9];
    mat_mul3(K, K, K2);
    float A, B;
    if(theta < 1e-4f){
        const float t2 = theta_sq;
        A = 1.0f - t2 / 6.0f + t2 * t2 / 120.0f;
        B = 0.5f - t2 / 24.0f + t2 * t2 / 720.0f;
    }
    else{
        A = sinf(theta) / theta;
        B = (1.0f - cosf(theta)) / theta_sq;
    }
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            Delta[i * 3 + j] = (i == j ? 1.0f : 0.0f) + A * K[i * 3 + j] + B * K2[i * 3 + j];
        }
    }
}

__host__ __device__ inline void rotation_delta_vjp_analytic(float dt, const float omega_next[3], const float lambda_delta[9], float grad_omega[3]){
    float phi[3] = {dt * omega_next[0], dt * omega_next[1], dt * omega_next[2]};
    const float theta2 = dot3(phi, phi);
    const float theta = sqrtf(theta2);
    float K[9];
    skew3(phi, K);
    float K2[9];
    mat_mul3(K, K, K2);
    float A, B;
    float dA_dphi[3] = {0.0f, 0.0f, 0.0f};
    float dB_dphi[3] = {0.0f, 0.0f, 0.0f};
    if(theta < 1e-4f){
        const float t2 = theta2;
        const float t4 = t2 * t2;
        A = 1.0f - t2 / 6.0f + t4 / 120.0f;
        B = 0.5f - t2 / 24.0f + t4 / 720.0f;
        const float cA = -1.0f / 3.0f + t2 / 30.0f - t4 / 840.0f;
        const float cB = -1.0f / 12.0f + t2 / 180.0f - t4 / 6720.0f;
        for(int i = 0; i < 3; i++){
            dA_dphi[i] = cA * phi[i];
            dB_dphi[i] = cB * phi[i];
        }
    }
    else{
        A = sinf(theta) / theta;
        B = (1.0f - cosf(theta)) / theta2;
        const float dA_dtheta = (theta * cosf(theta) - sinf(theta)) / theta2;
        const float dB_dtheta = (theta * sinf(theta) - 2.0f * (1.0f - cosf(theta))) / (theta2 * theta);
        for(int i = 0; i < 3; i++){
            dA_dphi[i] = dA_dtheta * phi[i] / theta;
            dB_dphi[i] = dB_dtheta * phi[i] / theta;
        }
    }

    const float inner_GK = mat_inner3(lambda_delta, K);
    const float inner_GK2 = mat_inner3(lambda_delta, K2);

    for(int i = 0; i < 3; i++){
        float E[9];
        basis_skew3(i, E);
        float EK[9], KE[9];
        mat_mul3(E, K, EK);
        mat_mul3(K, E, KE);
        float dK2[9];
        for(int j = 0; j < 9; j++){
            dK2[j] = EK[j] + KE[j];
        }
        float term = 0.0f;
        term += dA_dphi[i] * inner_GK;
        term += A * mat_inner3(lambda_delta, E);
        term += dB_dphi[i] * inner_GK2;
        term += B * mat_inner3(lambda_delta, dK2);
        grad_omega[i] = dt * term;
    }
}

struct DeviceArrays{
    float* p = nullptr;
    float* v = nullptr;
    float* R = nullptr;
    float* omega = nullptr;
    float* rpm = nullptr;
    float* actions = nullptr;
    float* initial_previous_action = nullptr;
    float* reference_p = nullptr;
    float* reference_v = nullptr;
    float* mass = nullptr;
    float* gravity = nullptr;
    float* J = nullptr;
    float* J_inv = nullptr;
    float* rotor_positions = nullptr;
    float* rotor_thrust_directions = nullptr;
    float* rotor_torque_directions = nullptr;
    float* rotor_torque_constants = nullptr;
    float* rotor_time_rising = nullptr;
    float* rotor_time_falling = nullptr;
    float* rotor_thrust_coeffs = nullptr;
    float* action_min = nullptr;
    float* action_max = nullptr;
    float* cache_alpha = nullptr;
    float* cache_d_setpoint_d_action = nullptr;
    float* cache_d_force_d_rpm = nullptr;
    float* cache_thrust_body = nullptr;
    float* cache_rotation_update = nullptr;
    float* cache_omega_next = nullptr;
    float* cache_J_omega = nullptr;
    float* lambda_p = nullptr;
    float* lambda_v = nullptr;
    float* lambda_R = nullptr;
    float* lambda_omega = nullptr;
    float* lambda_rpm = nullptr;
    float* loss = nullptr;
    float* action_gradients = nullptr;
    float* observations = nullptr;
    float* diagnostics = nullptr;
};

void cuda_free(DeviceArrays& d){
    float* ptrs[] = {
        d.p, d.v, d.R, d.omega, d.rpm, d.actions, d.initial_previous_action, d.reference_p, d.reference_v,
        d.mass, d.gravity, d.J, d.J_inv, d.rotor_positions, d.rotor_thrust_directions, d.rotor_torque_directions,
        d.rotor_torque_constants, d.rotor_time_rising, d.rotor_time_falling, d.rotor_thrust_coeffs, d.action_min,
        d.action_max, d.cache_alpha, d.cache_d_setpoint_d_action, d.cache_d_force_d_rpm, d.cache_thrust_body,
        d.cache_rotation_update, d.cache_omega_next, d.cache_J_omega, d.lambda_p, d.lambda_v, d.lambda_R,
        d.lambda_omega, d.lambda_rpm, d.loss, d.action_gradients, d.observations, d.diagnostics
    };
    for(float* ptr: ptrs){
        if(ptr != nullptr){
            cudaFree(ptr);
        }
    }
    d = DeviceArrays{};
}

void cuda_alloc(float*& ptr, std::size_t count){
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(float) * count));
}

void allocate(DeviceArrays& d, std::size_t batch_size, std::size_t horizon){
    const std::size_t state_count = (horizon + 1) * batch_size;
    const std::size_t step_count = horizon * batch_size;
    cuda_alloc(d.p, state_count * 3);
    cuda_alloc(d.v, state_count * 3);
    cuda_alloc(d.R, state_count * 9);
    cuda_alloc(d.omega, state_count * 3);
    cuda_alloc(d.rpm, state_count * 4);
    cuda_alloc(d.actions, step_count * 4);
    cuda_alloc(d.initial_previous_action, batch_size * 4);
    cuda_alloc(d.reference_p, batch_size * 3);
    cuda_alloc(d.reference_v, batch_size * 3);
    cuda_alloc(d.mass, batch_size);
    cuda_alloc(d.gravity, batch_size * 3);
    cuda_alloc(d.J, batch_size * 9);
    cuda_alloc(d.J_inv, batch_size * 9);
    cuda_alloc(d.rotor_positions, batch_size * 4 * 3);
    cuda_alloc(d.rotor_thrust_directions, batch_size * 4 * 3);
    cuda_alloc(d.rotor_torque_directions, batch_size * 4 * 3);
    cuda_alloc(d.rotor_torque_constants, batch_size * 4);
    cuda_alloc(d.rotor_time_rising, batch_size * 4);
    cuda_alloc(d.rotor_time_falling, batch_size * 4);
    cuda_alloc(d.rotor_thrust_coeffs, batch_size * 4 * 3);
    cuda_alloc(d.action_min, batch_size);
    cuda_alloc(d.action_max, batch_size);
    cuda_alloc(d.cache_alpha, step_count * 4);
    cuda_alloc(d.cache_d_setpoint_d_action, step_count * 4);
    cuda_alloc(d.cache_d_force_d_rpm, step_count * 4);
    cuda_alloc(d.cache_thrust_body, step_count * 3);
    cuda_alloc(d.cache_rotation_update, step_count * 9);
    cuda_alloc(d.cache_omega_next, step_count * 3);
    cuda_alloc(d.cache_J_omega, step_count * 3);
    cuda_alloc(d.lambda_p, state_count * 3);
    cuda_alloc(d.lambda_v, state_count * 3);
    cuda_alloc(d.lambda_R, state_count * 9);
    cuda_alloc(d.lambda_omega, state_count * 3);
    cuda_alloc(d.lambda_rpm, state_count * 4);
    cuda_alloc(d.loss, batch_size);
    cuda_alloc(d.action_gradients, step_count * 4);
    cuda_alloc(d.observations, state_count * EULER_OBSERVATION_DIM);
    cuda_alloc(d.diagnostics, 8);
}

__global__ void observation_kernel(DeviceArrays d, std::size_t batch_size, std::size_t step_i){
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= batch_size){
        return;
    }
    std::size_t out = 0;
    for(int dim = 0; dim < 3; dim++){
        d.observations[idx_obs(step_i, b, out++, batch_size)] =
            d.p[idx3(step_i, b, dim, batch_size)] - d.reference_p[pidx3(b, dim)];
    }
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            d.observations[idx_obs(step_i, b, out++, batch_size)] =
                d.R[idx9(step_i, b, r * 3 + c, batch_size)];
        }
    }
    for(int dim = 0; dim < 3; dim++){
        d.observations[idx_obs(step_i, b, out++, batch_size)] =
            d.v[idx3(step_i, b, dim, batch_size)] - d.reference_v[pidx3(b, dim)];
    }
    for(int dim = 0; dim < 3; dim++){
        d.observations[idx_obs(step_i, b, out++, batch_size)] =
            d.omega[idx3(step_i, b, dim, batch_size)];
    }
    for(int a = 0; a < 4; a++){
        const float previous_action = step_i == 0
            ? d.initial_previous_action[pidx4(b, a)]
            : d.actions[idx4(step_i - 1, b, a, batch_size)];
        d.observations[idx_obs(step_i, b, out++, batch_size)] = previous_action;
    }
}

__global__ void forward_step_kernel(DeviceArrays d, std::size_t batch_size, std::size_t step_i){
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= batch_size){
        return;
    }

    float thrust_body[3] = {0.0f, 0.0f, 0.0f};
    float torque_body[3] = {0.0f, 0.0f, 0.0f};
    float rpm_next[4];
    float d_force_d_rpm[4];

    const float min_action = d.action_min[b];
    const float max_action = d.action_max[b];
    const float half_range = (max_action - min_action) * 0.5f;

    for(int r = 0; r < 4; r++){
        const float action = d.actions[idx4(step_i, b, r, batch_size)];
        const float unclamped = action * half_range + min_action + half_range;
        float setpoint = unclamped;
        float d_setpoint = half_range;
        if(unclamped < min_action){
            setpoint = min_action;
            d_setpoint = 0.0f;
        }
        if(unclamped > max_action){
            setpoint = max_action;
            d_setpoint = 0.0f;
        }
        const float rpm_prev = d.rpm[idx4(step_i, b, r, batch_size)];
        const float tau = setpoint >= rpm_prev ? d.rotor_time_rising[pidx4(b, r)] : d.rotor_time_falling[pidx4(b, r)];
        const float alpha = expf(-DT / tau);
        rpm_next[r] = alpha * rpm_prev + (1.0f - alpha) * setpoint;
        d.cache_alpha[idx4(step_i, b, r, batch_size)] = alpha;
        d.cache_d_setpoint_d_action[idx4(step_i, b, r, batch_size)] = d_setpoint;

        const float c0 = d.rotor_thrust_coeffs[rotor3(b, r, 0)];
        const float c1 = d.rotor_thrust_coeffs[rotor3(b, r, 1)];
        const float c2 = d.rotor_thrust_coeffs[rotor3(b, r, 2)];
        const float force = c0 + c1 * rpm_next[r] + c2 * rpm_next[r] * rpm_next[r];
        d_force_d_rpm[r] = c1 + 2.0f * c2 * rpm_next[r];
        d.cache_d_force_d_rpm[idx4(step_i, b, r, batch_size)] = d_force_d_rpm[r];

        float rotor_thrust[3];
        float arm_torque[3];
        float rotor_pos[3];
        for(int dim = 0; dim < 3; dim++){
            rotor_thrust[dim] = d.rotor_thrust_directions[rotor3(b, r, dim)] * force;
            rotor_pos[dim] = d.rotor_positions[rotor3(b, r, dim)];
            thrust_body[dim] += rotor_thrust[dim];
            torque_body[dim] += d.rotor_torque_directions[rotor3(b, r, dim)] * d.rotor_torque_constants[pidx4(b, r)] * force;
        }
        cross3(rotor_pos, rotor_thrust, arm_torque);
        for(int dim = 0; dim < 3; dim++){
            torque_body[dim] += arm_torque[dim];
        }
    }

    for(int dim = 0; dim < 3; dim++){
        d.cache_thrust_body[idx3(step_i, b, dim, batch_size)] = thrust_body[dim];
    }

    float R[9];
    for(int i = 0; i < 9; i++){
        R[i] = d.R[idx9(step_i, b, i, batch_size)];
    }
    float rotated_thrust[3];
    mat_vec3(R, thrust_body, rotated_thrust);
    float acceleration_world[3];
    for(int dim = 0; dim < 3; dim++){
        acceleration_world[dim] = rotated_thrust[dim] / d.mass[b] + d.gravity[pidx3(b, dim)];
    }

    float J[9], J_inv[9], omega[3];
    for(int i = 0; i < 9; i++){
        J[i] = d.J[pidx9(b, i)];
        J_inv[i] = d.J_inv[pidx9(b, i)];
    }
    for(int dim = 0; dim < 3; dim++){
        omega[dim] = d.omega[idx3(step_i, b, dim, batch_size)];
    }
    float J_omega[3];
    mat_vec3(J, omega, J_omega);
    for(int dim = 0; dim < 3; dim++){
        d.cache_J_omega[idx3(step_i, b, dim, batch_size)] = J_omega[dim];
    }
    float gyro[3];
    cross3(omega, J_omega, gyro);
    float angular_rhs[3];
    for(int dim = 0; dim < 3; dim++){
        angular_rhs[dim] = torque_body[dim] - gyro[dim];
    }
    float angular_acceleration[3];
    mat_vec3(J_inv, angular_rhs, angular_acceleration);

    float omega_next[3];
    for(int dim = 0; dim < 3; dim++){
        omega_next[dim] = omega[dim] + DT * angular_acceleration[dim];
        const float v_next = d.v[idx3(step_i, b, dim, batch_size)] + DT * acceleration_world[dim];
        d.omega[idx3(step_i + 1, b, dim, batch_size)] = omega_next[dim];
        d.v[idx3(step_i + 1, b, dim, batch_size)] = v_next;
        d.p[idx3(step_i + 1, b, dim, batch_size)] = d.p[idx3(step_i, b, dim, batch_size)] + DT * v_next;
        d.cache_omega_next[idx3(step_i, b, dim, batch_size)] = omega_next[dim];
    }
    for(int r = 0; r < 4; r++){
        d.rpm[idx4(step_i + 1, b, r, batch_size)] = rpm_next[r];
    }

    float rotation_update[9];
    so3_expmap_delta(DT, omega_next, rotation_update);
    for(int i = 0; i < 9; i++){
        d.cache_rotation_update[idx9(step_i, b, i, batch_size)] = rotation_update[i];
    }
    float next_R[9];
    mat_mul3(R, rotation_update, next_R);
    for(int i = 0; i < 9; i++){
        d.R[idx9(step_i + 1, b, i, batch_size)] = next_R[i];
    }
}

__global__ void loss_and_action_kernel(DeviceArrays d, EulerGpuLossWeights weights, std::size_t batch_size, std::size_t horizon){
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= batch_size){
        return;
    }

    float total = 0.0f;
    const float normalizer = horizon > 0 ? 1.0f / static_cast<float>(horizon) : 1.0f;
    for(std::size_t step_i = 0; step_i < horizon; step_i++){
        const std::size_t state_step = step_i + 1;
        for(int dim = 0; dim < 3; dim++){
            const float e_p = d.p[idx3(state_step, b, dim, batch_size)] - d.reference_p[pidx3(b, dim)];
            const float e_v = d.v[idx3(state_step, b, dim, batch_size)] - d.reference_v[pidx3(b, dim)];
            const float omega = d.omega[idx3(state_step, b, dim, batch_size)];
            total += normalizer * weights.position * e_p * e_p;
            total += normalizer * weights.velocity * e_v * e_v;
            total += normalizer * weights.angular_velocity * omega * omega;
            d.lambda_p[idx3(state_step, b, dim, batch_size)] += normalizer * 2.0f * weights.position * e_p;
            d.lambda_v[idx3(state_step, b, dim, batch_size)] += normalizer * 2.0f * weights.velocity * e_v;
            d.lambda_omega[idx3(state_step, b, dim, batch_size)] += normalizer * 2.0f * weights.angular_velocity * omega;
        }
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                const int flat = i * 3 + j;
                const float target = i == j ? 1.0f : 0.0f;
                const float error = d.R[idx9(state_step, b, flat, batch_size)] - target;
                total += normalizer * weights.attitude * error * error;
                d.lambda_R[idx9(state_step, b, flat, batch_size)] += normalizer * 2.0f * weights.attitude * error;
            }
        }
    }

    const std::size_t terminal = horizon;
    const float terminal_scale = weights.terminal_loss_scale;
    const float wp = terminal_scale * weights.terminal_loss_weight * weights.terminal_position;
    const float wv = terminal_scale * weights.terminal_loss_weight * weights.terminal_velocity;
    const float wR = terminal_scale * weights.terminal_loss_weight * weights.terminal_attitude;
    const float ww = terminal_scale * weights.terminal_loss_weight * weights.terminal_angular_velocity;
    for(int dim = 0; dim < 3; dim++){
        const float e_p = d.p[idx3(terminal, b, dim, batch_size)] - d.reference_p[pidx3(b, dim)];
        const float e_v = d.v[idx3(terminal, b, dim, batch_size)] - d.reference_v[pidx3(b, dim)];
        const float omega = d.omega[idx3(terminal, b, dim, batch_size)];
        total += wp * e_p * e_p + wv * e_v * e_v + ww * omega * omega;
        d.lambda_p[idx3(terminal, b, dim, batch_size)] += 2.0f * wp * e_p;
        d.lambda_v[idx3(terminal, b, dim, batch_size)] += 2.0f * wv * e_v;
        d.lambda_omega[idx3(terminal, b, dim, batch_size)] += 2.0f * ww * omega;
    }
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            const int flat = i * 3 + j;
            const float target = i == j ? 1.0f : 0.0f;
            const float error = d.R[idx9(terminal, b, flat, batch_size)] - target;
            total += wR * error * error;
            d.lambda_R[idx9(terminal, b, flat, batch_size)] += 2.0f * wR * error;
        }
    }

    float previous_action[4];
    for(int a = 0; a < 4; a++){
        previous_action[a] = d.initial_previous_action[pidx4(b, a)];
    }
    for(std::size_t step_i = 0; step_i < horizon; step_i++){
        for(int a = 0; a < 4; a++){
            const float action = d.actions[idx4(step_i, b, a, batch_size)];
            total += normalizer * weights.action_magnitude * action * action;
            d.action_gradients[idx4(step_i, b, a, batch_size)] += normalizer * 2.0f * weights.action_magnitude * action;
            const float diff = action - previous_action[a];
            total += normalizer * weights.action_smoothness * diff * diff;
            d.action_gradients[idx4(step_i, b, a, batch_size)] += normalizer * 2.0f * weights.action_smoothness * diff;
            if(step_i > 0){
                d.action_gradients[idx4(step_i - 1, b, a, batch_size)] -= normalizer * 2.0f * weights.action_smoothness * diff;
            }
            const float abs_action = fabsf(action);
            const float SATURATION_START = weights.saturation_start;
            if(abs_action > SATURATION_START){
                const float sign = action >= 0.0f ? 1.0f : -1.0f;
                const float excess = abs_action - SATURATION_START;
                total += normalizer * weights.saturation * excess * excess;
                d.action_gradients[idx4(step_i, b, a, batch_size)] += normalizer * 2.0f * weights.saturation * excess * sign;
            }
            previous_action[a] = action;
        }
    }

    d.loss[b] = total;
}

__global__ void rollout_diagnostics_kernel(
    DeviceArrays d,
    std::size_t batch_size,
    std::size_t horizon,
    float position_threshold,
    float velocity_threshold,
    float angular_velocity_threshold
){
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= batch_size){
        return;
    }
    float p_norm_sq = 0.0f;
    float v_norm_sq = 0.0f;
    float w_norm_sq = 0.0f;
    bool finite = true;
    for(int dim = 0; dim < 3; dim++){
        const float p = d.p[idx3(horizon, b, dim, batch_size)];
        const float v = d.v[idx3(horizon, b, dim, batch_size)];
        const float w = d.omega[idx3(horizon, b, dim, batch_size)];
        p_norm_sq += p * p;
        v_norm_sq += v * v;
        w_norm_sq += w * w;
        finite = finite && isfinite(p) && isfinite(v) && isfinite(w);
    }
    const float p_norm = sqrtf(p_norm_sq);
    const float v_norm = sqrtf(v_norm_sq);
    const float w_norm = sqrtf(w_norm_sq);
    finite = finite && isfinite(d.loss[b]) && isfinite(p_norm) && isfinite(v_norm) && isfinite(w_norm);

    std::uint32_t saturation_count = 0;
    for(std::size_t t = 0; t < horizon; t++){
        for(int a = 0; a < 4; a++){
            const float action = d.actions[idx4(t, b, a, batch_size)];
            finite = finite && isfinite(action);
            saturation_count += fabsf(action) >= 0.95f ? 1u : 0u;
        }
    }
    if(finite){
        const bool success = p_norm < position_threshold &&
            v_norm < velocity_threshold &&
            w_norm < angular_velocity_threshold;
        atomicAdd(&d.diagnostics[0], success ? 1.0f : 0.0f);
        atomicAdd(&d.diagnostics[1], 1.0f);
        atomicAdd(&d.diagnostics[3], static_cast<float>(saturation_count));
        atomicAdd(&d.diagnostics[4], static_cast<float>(horizon * 4));
        atomicAdd(&d.diagnostics[5], p_norm);
        atomicAdd(&d.diagnostics[6], v_norm);
        atomicAdd(&d.diagnostics[7], w_norm);
    }
    else{
        atomicAdd(&d.diagnostics[2], 1.0f);
    }
}

__global__ void backward_step_kernel(DeviceArrays d, std::size_t batch_size, std::size_t step_i){
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= batch_size){
        return;
    }

    float lambda_state_p[3] = {0.0f, 0.0f, 0.0f};
    float lambda_state_v[3] = {0.0f, 0.0f, 0.0f};
    float lambda_state_R[9] = {0.0f};
    float lambda_state_omega[3] = {0.0f, 0.0f, 0.0f};
    float lambda_state_rpm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float grad_action[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float lambda_v_next[3];
    float lambda_omega_next[3];
    for(int i = 0; i < 3; i++){
        const float lp_next = d.lambda_p[idx3(step_i + 1, b, i, batch_size)];
        lambda_state_p[i] += lp_next;
        lambda_v_next[i] = d.lambda_v[idx3(step_i + 1, b, i, batch_size)] + DT * lp_next;
        lambda_omega_next[i] = d.lambda_omega[idx3(step_i + 1, b, i, batch_size)];
    }

    float state_R[9];
    float rotation_update[9];
    for(int i = 0; i < 9; i++){
        state_R[i] = d.R[idx9(step_i, b, i, batch_size)];
        rotation_update[i] = d.cache_rotation_update[idx9(step_i, b, i, batch_size)];
    }

    float lambda_delta[9] = {0.0f};
    for(int i = 0; i < 3; i++){
        for(int k = 0; k < 3; k++){
            float value = 0.0f;
            for(int j = 0; j < 3; j++){
                const float lambda_next_R = d.lambda_R[idx9(step_i + 1, b, i * 3 + j, batch_size)];
                value += lambda_next_R * rotation_update[k * 3 + j];
                lambda_delta[k * 3 + j] += state_R[i * 3 + k] * lambda_next_R;
            }
            lambda_state_R[i * 3 + k] += value;
        }
    }
    float omega_next[3];
    for(int i = 0; i < 3; i++){
        omega_next[i] = d.cache_omega_next[idx3(step_i, b, i, batch_size)];
    }
    float lambda_omega_from_rotation[3] = {0.0f, 0.0f, 0.0f};
    rotation_delta_vjp_analytic(DT, omega_next, lambda_delta, lambda_omega_from_rotation);
    for(int i = 0; i < 3; i++){
        lambda_omega_next[i] += lambda_omega_from_rotation[i];
    }

    float lambda_acceleration[3];
    float lambda_omega_dot[3];
    for(int i = 0; i < 3; i++){
        lambda_state_v[i] += lambda_v_next[i];
        lambda_acceleration[i] = DT * lambda_v_next[i];
        lambda_state_omega[i] += lambda_omega_next[i];
        lambda_omega_dot[i] = DT * lambda_omega_next[i];
    }

    float thrust_body[3];
    float lambda_thrust_body[3] = {0.0f, 0.0f, 0.0f};
    for(int i = 0; i < 3; i++){
        thrust_body[i] = d.cache_thrust_body[idx3(step_i, b, i, batch_size)];
    }
    const float inv_mass = 1.0f / d.mass[b];
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            lambda_state_R[i * 3 + j] += lambda_acceleration[i] * thrust_body[j] * inv_mass;
            lambda_thrust_body[j] += lambda_acceleration[i] * state_R[i * 3 + j] * inv_mass;
        }
    }

    float J[9], J_inv[9];
    for(int i = 0; i < 9; i++){
        J[i] = d.J[pidx9(b, i)];
        J_inv[i] = d.J_inv[pidx9(b, i)];
    }
    float lambda_angular_rhs[3] = {0.0f, 0.0f, 0.0f};
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            lambda_angular_rhs[j] += J_inv[i * 3 + j] * lambda_omega_dot[i];
        }
    }
    float lambda_torque_body[3] = {lambda_angular_rhs[0], lambda_angular_rhs[1], lambda_angular_rhs[2]};
    float lambda_gyro[3] = {-lambda_angular_rhs[0], -lambda_angular_rhs[1], -lambda_angular_rhs[2]};

    float state_omega[3], J_omega[3];
    for(int i = 0; i < 3; i++){
        state_omega[i] = d.omega[idx3(step_i, b, i, batch_size)];
        J_omega[i] = d.cache_J_omega[idx3(step_i, b, i, batch_size)];
    }
    float lambda_omega_from_cross[3];
    float lambda_J_omega[3];
    cross3(J_omega, lambda_gyro, lambda_omega_from_cross);
    cross3(lambda_gyro, state_omega, lambda_J_omega);
    for(int i = 0; i < 3; i++){
        lambda_state_omega[i] += lambda_omega_from_cross[i];
        for(int j = 0; j < 3; j++){
            lambda_state_omega[j] += J[i * 3 + j] * lambda_J_omega[i];
        }
    }

    float lambda_rpm_next[4];
    for(int r = 0; r < 4; r++){
        lambda_rpm_next[r] = d.lambda_rpm[idx4(step_i + 1, b, r, batch_size)];
    }
    for(int r = 0; r < 4; r++){
        float lambda_force = 0.0f;
        float rotor_pos[3], thrust_dir[3], torque_dir[3];
        for(int dim = 0; dim < 3; dim++){
            rotor_pos[dim] = d.rotor_positions[rotor3(b, r, dim)];
            thrust_dir[dim] = d.rotor_thrust_directions[rotor3(b, r, dim)];
            torque_dir[dim] = d.rotor_torque_directions[rotor3(b, r, dim)];
        }
        float arm_direction[3];
        cross3(rotor_pos, thrust_dir, arm_direction);
        for(int dim = 0; dim < 3; dim++){
            lambda_force += thrust_dir[dim] * lambda_thrust_body[dim];
            lambda_force += torque_dir[dim] * d.rotor_torque_constants[pidx4(b, r)] * lambda_torque_body[dim];
            lambda_force += arm_direction[dim] * lambda_torque_body[dim];
        }
        lambda_rpm_next[r] += lambda_force * d.cache_d_force_d_rpm[idx4(step_i, b, r, batch_size)];
    }

    for(int r = 0; r < 4; r++){
        const float alpha = d.cache_alpha[idx4(step_i, b, r, batch_size)];
        lambda_state_rpm[r] += alpha * lambda_rpm_next[r];
        const float d_next_d_setpoint = 1.0f - alpha;
        grad_action[r] += d_next_d_setpoint * d.cache_d_setpoint_d_action[idx4(step_i, b, r, batch_size)] * lambda_rpm_next[r];
        d.action_gradients[idx4(step_i, b, r, batch_size)] += grad_action[r];
    }

    for(int i = 0; i < 3; i++){
        d.lambda_p[idx3(step_i, b, i, batch_size)] += lambda_state_p[i];
        d.lambda_v[idx3(step_i, b, i, batch_size)] += lambda_state_v[i];
        d.lambda_omega[idx3(step_i, b, i, batch_size)] += lambda_state_omega[i];
    }
    for(int i = 0; i < 9; i++){
        d.lambda_R[idx9(step_i, b, i, batch_size)] += lambda_state_R[i];
    }
    for(int r = 0; r < 4; r++){
        d.lambda_rpm[idx4(step_i, b, r, batch_size)] += lambda_state_rpm[r];
    }
}

void copy_input_to_device(const EulerGpuBatch& batch, DeviceArrays& d){
    const std::size_t B = batch.batch_size;
    const std::size_t H = batch.horizon;
    CUDA_CHECK(cudaMemcpy(d.p, batch.initial_p.data(), sizeof(float) * B * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.v, batch.initial_v.data(), sizeof(float) * B * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.R, batch.initial_R.data(), sizeof(float) * B * 9, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.omega, batch.initial_omega.data(), sizeof(float) * B * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rpm, batch.initial_rpm.data(), sizeof(float) * B * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.actions, batch.actions.data(), sizeof(float) * H * B * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.initial_previous_action, batch.initial_previous_action.data(), sizeof(float) * B * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.reference_p, batch.reference_p.data(), sizeof(float) * B * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.reference_v, batch.reference_v.data(), sizeof(float) * B * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.mass, batch.mass.data(), sizeof(float) * B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.gravity, batch.gravity.data(), sizeof(float) * B * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.J, batch.J.data(), sizeof(float) * B * 9, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.J_inv, batch.J_inv.data(), sizeof(float) * B * 9, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rotor_positions, batch.rotor_positions.data(), sizeof(float) * B * 4 * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rotor_thrust_directions, batch.rotor_thrust_directions.data(), sizeof(float) * B * 4 * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rotor_torque_directions, batch.rotor_torque_directions.data(), sizeof(float) * B * 4 * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rotor_torque_constants, batch.rotor_torque_constants.data(), sizeof(float) * B * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rotor_time_rising, batch.rotor_time_rising.data(), sizeof(float) * B * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rotor_time_falling, batch.rotor_time_falling.data(), sizeof(float) * B * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rotor_thrust_coeffs, batch.rotor_thrust_coeffs.data(), sizeof(float) * B * 4 * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.action_min, batch.action_min.data(), sizeof(float) * B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.action_max, batch.action_max.data(), sizeof(float) * B, cudaMemcpyHostToDevice));
}

void copy_output_to_host(const EulerGpuBatch& batch, const DeviceArrays& d, EulerGpuResult& result){
    const std::size_t B = batch.batch_size;
    const std::size_t H = batch.horizon;
    result.resize(B, H);
    CUDA_CHECK(cudaMemcpy(result.final_p.data(), d.p + H * B * 3, sizeof(float) * B * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.final_v.data(), d.v + H * B * 3, sizeof(float) * B * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.final_R.data(), d.R + H * B * 9, sizeof(float) * B * 9, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.final_omega.data(), d.omega + H * B * 3, sizeof(float) * B * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.final_rpm.data(), d.rpm + H * B * 4, sizeof(float) * B * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.loss.data(), d.loss, sizeof(float) * B, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.action_gradients.data(), d.action_gradients, sizeof(float) * H * B * 4, cudaMemcpyDeviceToHost));
}

float elapsed(cudaEvent_t start, cudaEvent_t stop){
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

struct CpuSimpleDynamics{
    float action_limit_min;
    float action_limit_max;
    float mass;
    float gravity[3];
    float J[3][3];
    float J_inv[3][3];
    float rotor_positions[4][3];
    float rotor_thrust_directions[4][3];
    float rotor_torque_directions[4][3];
    float rotor_torque_constants[4];
    float rotor_time_constants_rising[4];
    float rotor_time_constants_falling[4];
    float rotor_thrust_coefficients[4][3];
    struct ActionLimit{
        float min;
        float max;
    } action_limit;
};

struct CpuSimpleIntegration{
    float dt;
};

struct CpuSimpleParameters{
    CpuSimpleDynamics dynamics;
    CpuSimpleIntegration integration;
};

CpuSimpleParameters cpu_parameters_from_batch(const EulerGpuBatch& batch, std::size_t b){
    CpuSimpleParameters parameters{};
    parameters.integration.dt = DT;
    parameters.dynamics.mass = batch.mass[b];
    parameters.dynamics.action_limit.min = batch.action_min[b];
    parameters.dynamics.action_limit.max = batch.action_max[b];
    for(int i = 0; i < 3; i++){
        parameters.dynamics.gravity[i] = batch.gravity[pidx3(b, i)];
        for(int j = 0; j < 3; j++){
            parameters.dynamics.J[i][j] = batch.J[pidx9(b, i * 3 + j)];
            parameters.dynamics.J_inv[i][j] = batch.J_inv[pidx9(b, i * 3 + j)];
        }
    }
    for(int r = 0; r < 4; r++){
        parameters.dynamics.rotor_torque_constants[r] = batch.rotor_torque_constants[pidx4(b, r)];
        parameters.dynamics.rotor_time_constants_rising[r] = batch.rotor_time_rising[pidx4(b, r)];
        parameters.dynamics.rotor_time_constants_falling[r] = batch.rotor_time_falling[pidx4(b, r)];
        for(int i = 0; i < 3; i++){
            parameters.dynamics.rotor_positions[r][i] = batch.rotor_positions[rotor3(b, r, i)];
            parameters.dynamics.rotor_thrust_directions[r][i] = batch.rotor_thrust_directions[rotor3(b, r, i)];
            parameters.dynamics.rotor_torque_directions[r][i] = batch.rotor_torque_directions[rotor3(b, r, i)];
            parameters.dynamics.rotor_thrust_coefficients[r][i] = batch.rotor_thrust_coeffs[rotor3(b, r, i)];
        }
    }
    return parameters;
}

void run_cpu_reference(const EulerGpuBatch& batch, const EulerGpuLossWeights& gpu_weights, EulerGpuResult& result){
    namespace diff = rl_tools::rl::environments::l2f::diff;
    using TI = std::uint64_t;
    if(batch.horizon > CPU_MAX_HORIZON){
        throw std::runtime_error("CPU validation horizon exceeds CPU_MAX_HORIZON");
    }
    result.resize(batch.batch_size, batch.horizon);
    diff::EulerLossWeights<float> weights{
        gpu_weights.position,
        gpu_weights.velocity,
        gpu_weights.attitude,
        gpu_weights.angular_velocity,
        gpu_weights.action_magnitude,
        gpu_weights.action_smoothness,
        gpu_weights.saturation,
        gpu_weights.terminal_loss_weight,
        gpu_weights.terminal_position,
        gpu_weights.terminal_velocity,
        gpu_weights.terminal_attitude,
        gpu_weights.terminal_angular_velocity
    };

    for(std::size_t b = 0; b < batch.batch_size; b++){
        CpuSimpleParameters parameters = cpu_parameters_from_batch(batch, b);
        diff::EulerState<float, TI> initial{};
        for(int i = 0; i < 3; i++){
            initial.p[i] = batch.initial_p[pidx3(b, i)];
            initial.v[i] = batch.initial_v[pidx3(b, i)];
            initial.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                initial.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            initial.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            initial.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        diff::TrackingReference<float> ref{};
        for(int i = 0; i < 3; i++){
            ref.p[i] = batch.reference_p[pidx3(b, i)];
            ref.v[i] = batch.reference_v[pidx3(b, i)];
        }
        float actions[CPU_MAX_HORIZON][4] = {};
        float action_gradients[CPU_MAX_HORIZON][4] = {};
        for(std::size_t t = 0; t < batch.horizon; t++){
            for(int a = 0; a < 4; a++){
                actions[t][a] = batch.actions[idx4(t, b, a, batch.batch_size)];
            }
        }
        diff::EulerState<float, TI> states[CPU_MAX_HORIZON + 1];
        diff::EulerStepCache<float> caches[CPU_MAX_HORIZON];
        auto terms = diff::rollout_loss_and_gradients<CpuSimpleParameters, float, TI, CPU_MAX_HORIZON>(
            parameters, initial, actions, static_cast<TI>(batch.horizon), weights, ref, states, caches, action_gradients
        );
        result.loss[b] = terms.total();
        for(int i = 0; i < 3; i++){
            result.final_p[pidx3(b, i)] = states[batch.horizon].p[i];
            result.final_v[pidx3(b, i)] = states[batch.horizon].v[i];
            result.final_omega[pidx3(b, i)] = states[batch.horizon].omega[i];
            for(int j = 0; j < 3; j++){
                result.final_R[pidx9(b, i * 3 + j)] = states[batch.horizon].R[i][j];
            }
        }
        for(int i = 0; i < 4; i++){
            result.final_rpm[pidx4(b, i)] = states[batch.horizon].rpm[i];
        }
        for(std::size_t t = 0; t < batch.horizon; t++){
            for(int a = 0; a < 4; a++){
                result.action_gradients[idx4(t, b, a, batch.batch_size)] = action_gradients[t][a];
            }
        }
    }
}

} // namespace

void EulerGpuBatch::resize(std::size_t new_batch_size, std::size_t new_horizon){
    batch_size = new_batch_size;
    horizon = new_horizon;
    initial_p.assign(batch_size * 3, 0.0f);
    initial_v.assign(batch_size * 3, 0.0f);
    initial_R.assign(batch_size * 9, 0.0f);
    initial_omega.assign(batch_size * 3, 0.0f);
    initial_rpm.assign(batch_size * 4, 0.0f);
    initial_previous_action.assign(batch_size * 4, 0.0f);
    reference_p.assign(batch_size * 3, 0.0f);
    reference_v.assign(batch_size * 3, 0.0f);
    actions.assign(horizon * batch_size * 4, 0.0f);
    mass.assign(batch_size, 0.0f);
    gravity.assign(batch_size * 3, 0.0f);
    J.assign(batch_size * 9, 0.0f);
    J_inv.assign(batch_size * 9, 0.0f);
    rotor_positions.assign(batch_size * 4 * 3, 0.0f);
    rotor_thrust_directions.assign(batch_size * 4 * 3, 0.0f);
    rotor_torque_directions.assign(batch_size * 4 * 3, 0.0f);
    rotor_torque_constants.assign(batch_size * 4, 0.0f);
    rotor_time_rising.assign(batch_size * 4, 0.0f);
    rotor_time_falling.assign(batch_size * 4, 0.0f);
    rotor_thrust_coeffs.assign(batch_size * 4 * 3, 0.0f);
    action_min.assign(batch_size, 0.0f);
    action_max.assign(batch_size, 0.0f);
    dynamics_size_mass_bin.assign(batch_size, 0u);
    dynamics_thrust_to_weight_bin.assign(batch_size, 0u);
    dynamics_torque_to_inertia_bin.assign(batch_size, 0u);
    dynamics_motor_delay_bin.assign(batch_size, 0u);
    dynamics_curve_shape_bin.assign(batch_size, 0u);
    dynamics_group_key.assign(batch_size, 0u);
    rejected_before_accept.assign(batch_size, 0u);
    group_weight.assign(batch_size, batch_size > 0 ? 1.0f / static_cast<float>(batch_size) : 0.0f);
    reset_mask.assign(horizon * batch_size, 0u);
    hidden_reset_mask.assign(horizon * batch_size, 0u);
}

void EulerGpuResult::resize(std::size_t batch_size, std::size_t horizon){
    final_p.assign(batch_size * 3, 0.0f);
    final_v.assign(batch_size * 3, 0.0f);
    final_R.assign(batch_size * 9, 0.0f);
    final_omega.assign(batch_size * 3, 0.0f);
    final_rpm.assign(batch_size * 4, 0.0f);
    loss.assign(batch_size, 0.0f);
    action_gradients.assign(horizon * batch_size * 4, 0.0f);
}

int run_euler_gpu_rollout(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options,
    EulerGpuResult& result,
    EulerGpuTimings& timings
){
    if(batch.batch_size == 0 || batch.horizon == 0){
        std::cerr << "GPU rollout requires non-zero batch size and horizon.\n";
        return 1;
    }
    DeviceArrays d;
    cudaEvent_t start, after_h2d, after_forward, after_loss, after_backward, after_d2h;
    try{
        CUDA_CHECK(cudaSetDevice(options.device));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&after_h2d));
        CUDA_CHECK(cudaEventCreate(&after_forward));
        CUDA_CHECK(cudaEventCreate(&after_loss));
        CUDA_CHECK(cudaEventCreate(&after_backward));
        CUDA_CHECK(cudaEventCreate(&after_d2h));

        allocate(d, batch.batch_size, batch.horizon);
        CUDA_CHECK(cudaEventRecord(start));
        copy_input_to_device(batch, d);
        CUDA_CHECK(cudaEventRecord(after_h2d));

        const int block = 256;
        const int grid = static_cast<int>((batch.batch_size + block - 1) / block);
        for(std::size_t step_i = 0; step_i < batch.horizon; step_i++){
            forward_step_kernel<<<grid, block>>>(d, batch.batch_size, step_i);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(after_forward));

        const std::size_t state_count = (batch.horizon + 1) * batch.batch_size;
        const std::size_t step_count = batch.horizon * batch.batch_size;
        CUDA_CHECK(cudaMemset(d.lambda_p, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_v, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_R, 0, sizeof(float) * state_count * 9));
        CUDA_CHECK(cudaMemset(d.lambda_omega, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_rpm, 0, sizeof(float) * state_count * 4));
        CUDA_CHECK(cudaMemset(d.action_gradients, 0, sizeof(float) * step_count * 4));
        CUDA_CHECK(cudaMemset(d.loss, 0, sizeof(float) * batch.batch_size));
        loss_and_action_kernel<<<grid, block>>>(d, weights, batch.batch_size, batch.horizon);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(after_loss));

        if(options.compute_action_gradients){
            for(std::size_t reverse_i = 0; reverse_i < batch.horizon; reverse_i++){
                const std::size_t step_i = batch.horizon - 1 - reverse_i;
                backward_step_kernel<<<grid, block>>>(d, batch.batch_size, step_i);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        CUDA_CHECK(cudaEventRecord(after_backward));

        copy_output_to_host(batch, d, result);
        CUDA_CHECK(cudaEventRecord(after_d2h));
        CUDA_CHECK(cudaEventSynchronize(after_d2h));

        timings.host_to_device_ms = elapsed(start, after_h2d);
        timings.forward_ms = elapsed(after_h2d, after_forward);
        timings.loss_ms = elapsed(after_forward, after_loss);
        timings.backward_vjp_ms = elapsed(after_loss, after_backward);
        timings.device_to_host_ms = elapsed(after_backward, after_d2h);
        timings.total_ms = elapsed(start, after_d2h);

        cuda_free(d);
        cudaEventDestroy(start);
        cudaEventDestroy(after_h2d);
        cudaEventDestroy(after_forward);
        cudaEventDestroy(after_loss);
        cudaEventDestroy(after_backward);
        cudaEventDestroy(after_d2h);
        return 0;
    }
    catch(const std::exception& e){
        std::cerr << e.what() << "\n";
        cuda_free(d);
        return 1;
    }
}

float sample_binned_value(
    std::mt19937& rng,
    std::uniform_real_distribution<float>& unit,
    float min_value,
    float max_value,
    std::uint32_t bin,
    std::uint32_t bins
){
    const std::uint32_t clamped_bin = std::min(bin, bins - 1u);
    const float width = (max_value - min_value) / static_cast<float>(bins);
    const float lo = min_value + width * static_cast<float>(clamped_bin);
    const float hi = clamped_bin + 1u == bins ? max_value : lo + width;
    return lo + unit(rng) * (hi - lo);
}

float sample_reciprocal_randomization_factor(std::mt19937& rng, std::uniform_real_distribution<float>& unit, float deviation){
    const float upper = std::max(1.0f, 1.0f + deviation);
    const float lower = 1.0f / upper;
    return lower + unit(rng) * (upper - lower);
}

void generate_validation_batch(
    EulerGpuBatch& batch,
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const ForcedDynamicsBins* forced_bins,
    bool nominal_dynamics,
    bool correlated_size_mass_sampling
){
    batch.resize(batch_size, horizon);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_real_distribution<float> action_delta_dist(-0.03f, 0.03f);
    std::uniform_real_distribution<float> pos_dist(-0.2f, 0.2f);
    std::uniform_real_distribution<float> vel_dist(-0.05f, 0.05f);
    std::uniform_real_distribution<float> omega_dist(-0.02f, 0.02f);
    constexpr float max_rpm = 22000.0f;
    constexpr std::uint32_t balance_bins = 4u;
    batch.replay_schema_version = 2u;
    batch.sampler_seed = seed;
    batch.sampler_balance_bins = balance_bins;

    for(std::size_t b = 0; b < batch_size; b++){
        const std::uint32_t sample_i = static_cast<std::uint32_t>(b);
        const bool force_bins = forced_bins != nullptr && forced_bins->enabled && !nominal_dynamics;
        const std::uint32_t size_bin = nominal_dynamics ? 0u : (force_bins ? forced_bins->size_mass % balance_bins : sample_i % balance_bins);
        const std::uint32_t thrust_bin = nominal_dynamics ? 0u : (force_bins ? forced_bins->thrust_to_weight % balance_bins : (sample_i + 1u) % balance_bins);
        const std::uint32_t torque_bin = nominal_dynamics ? 0u : (force_bins ? forced_bins->torque_to_inertia % balance_bins : (sample_i + 2u) % balance_bins);
        const std::uint32_t delay_bin = nominal_dynamics ? 0u : (force_bins ? forced_bins->motor_delay % balance_bins : (sample_i + 3u) % balance_bins);
        const std::uint32_t curve_bin = nominal_dynamics ? 0u : (force_bins ? forced_bins->curve_shape % balance_bins : (sample_i + seed) % balance_bins);
        const std::uint32_t group_key = (((size_bin * balance_bins + thrust_bin) * balance_bins + torque_bin) * balance_bins + delay_bin) * balance_bins + curve_bin;
        batch.dynamics_size_mass_bin[b] = size_bin;
        batch.dynamics_thrust_to_weight_bin[b] = thrust_bin;
        batch.dynamics_torque_to_inertia_bin[b] = torque_bin;
        batch.dynamics_motor_delay_bin[b] = delay_bin;
        batch.dynamics_curve_shape_bin[b] = curve_bin;
        batch.dynamics_group_key[b] = group_key;
        batch.rejected_before_accept[b] = 0u;

        constexpr float nominal_mass = 0.05f;
        constexpr float nominal_arm = 0.09f;
        constexpr float nominal_thrust_to_weight = 3.0f;
        constexpr float nominal_torque_to_inertia = 250.0f;
        constexpr float nominal_torque_constant = 0.02f;
        constexpr float nominal_jx = 0.55f * nominal_mass * nominal_arm * nominal_arm;
        constexpr float nominal_jy = nominal_jx;
        constexpr float nominal_jz = 1.10f * nominal_mass * nominal_arm * nominal_arm;
        const bool correlated_size_mass = correlated_size_mass_sampling && !nominal_dynamics;

        float mass = nominal_dynamics ? nominal_mass : sample_binned_value(rng, unit, 0.02f, 0.25f, size_bin, balance_bins);
        float arm = nominal_dynamics ? nominal_arm : sample_binned_value(rng, unit, 0.035f, 0.215f, balance_bins - 1u - torque_bin, balance_bins);
        const float thrust_to_weight = nominal_dynamics ? nominal_thrust_to_weight : sample_binned_value(rng, unit, 1.5f, 5.0f, thrust_bin, balance_bins);
        float per_rotor_max_thrust = thrust_to_weight * mass * G / 4.0f;
        const float sampled_torque_to_inertia = nominal_dynamics ? nominal_torque_to_inertia : sample_binned_value(rng, unit, 125.0f, 500.0f, torque_bin, balance_bins);
        const float torque_constant = nominal_dynamics ? nominal_torque_constant : sample_binned_value(rng, unit, 0.005f, 0.05f, torque_bin, balance_bins);
        const float tau_rise = nominal_dynamics ? 0.05f : sample_binned_value(rng, unit, 0.03f, 0.10f, delay_bin, balance_bins);
        const float tau_fall = nominal_dynamics ? 0.08f : sample_binned_value(rng, unit, 0.03f, 0.30f, delay_bin, balance_bins);
        const float linear_curve_share = nominal_dynamics ? 0.15f : static_cast<float>(curve_bin) * 0.15f;

        float jx = std::max(1e-6f, 0.55f * mass * arm * arm);
        float jy = std::max(1e-6f, 0.55f * mass * arm * arm);
        float jz = std::max(1e-6f, 1.10f * mass * arm * arm);
        if(correlated_size_mass){
            constexpr float mass_min = 0.02f;
            constexpr float mass_max = 0.25f;
            constexpr float mass_size_deviation = 0.20f;
            const float relative_size_min = std::cbrt(mass_min);
            const float relative_size_max = std::cbrt(mass_max);
            const float size_new = relative_size_min + unit(rng) * (relative_size_max - relative_size_min);
            mass = std::max(mass_min, std::min(mass_max, size_new * size_new * size_new));
            const float scale_relative = std::cbrt(mass / nominal_mass);
            const float factor_mass = mass / nominal_mass;
            const float size_factor = sample_reciprocal_randomization_factor(rng, unit, mass_size_deviation);
            const float rotor_distance_factor = scale_relative * size_factor;
            arm = nominal_arm * rotor_distance_factor;
            const float factor_thrust_to_weight = thrust_to_weight / nominal_thrust_to_weight;
            const float factor_thrust_coefficients = factor_thrust_to_weight * factor_mass;
            const float nominal_per_rotor_max_thrust = nominal_thrust_to_weight * nominal_mass * G / 4.0f;
            per_rotor_max_thrust = nominal_per_rotor_max_thrust * factor_thrust_coefficients;
            const float torque_to_inertia_factor = sampled_torque_to_inertia / nominal_torque_to_inertia;
            const float inertia_factor = torque_to_inertia_factor / std::max(1e-6f, rotor_distance_factor);
            jx = std::max(1e-6f, nominal_jx / std::max(1e-6f, inertia_factor));
            jy = std::max(1e-6f, nominal_jy / std::max(1e-6f, inertia_factor));
            jz = std::max(1e-6f, nominal_jz / std::max(1e-6f, inertia_factor));
        }

        batch.mass[b] = mass;
        batch.gravity[pidx3(b, 0)] = 0.0f;
        batch.gravity[pidx3(b, 1)] = 0.0f;
        batch.gravity[pidx3(b, 2)] = -G;
        batch.action_min[b] = 0.0f;
        batch.action_max[b] = max_rpm;

        for(int i = 0; i < 9; i++){
            batch.J[pidx9(b, i)] = 0.0f;
            batch.J_inv[pidx9(b, i)] = 0.0f;
        }
        batch.J[pidx9(b, 0)] = jx;
        batch.J[pidx9(b, 4)] = jy;
        batch.J[pidx9(b, 8)] = jz;
        batch.J_inv[pidx9(b, 0)] = 1.0f / jx;
        batch.J_inv[pidx9(b, 4)] = 1.0f / jy;
        batch.J_inv[pidx9(b, 8)] = 1.0f / jz;

        const float rotor_xy[4][2] = {
            { arm,  arm},
            {-arm,  arm},
            {-arm, -arm},
            { arm, -arm}
        };
        const float yaw_sign[4] = {1.0f, -1.0f, 1.0f, -1.0f};
        for(int r = 0; r < 4; r++){
            batch.rotor_positions[rotor3(b, r, 0)] = rotor_xy[r][0];
            batch.rotor_positions[rotor3(b, r, 1)] = rotor_xy[r][1];
            batch.rotor_positions[rotor3(b, r, 2)] = 0.0f;
            batch.rotor_thrust_directions[rotor3(b, r, 0)] = 0.0f;
            batch.rotor_thrust_directions[rotor3(b, r, 1)] = 0.0f;
            batch.rotor_thrust_directions[rotor3(b, r, 2)] = 1.0f;
            batch.rotor_torque_directions[rotor3(b, r, 0)] = 0.0f;
            batch.rotor_torque_directions[rotor3(b, r, 1)] = 0.0f;
            batch.rotor_torque_directions[rotor3(b, r, 2)] = yaw_sign[r];
            batch.rotor_torque_constants[pidx4(b, r)] = torque_constant;
            batch.rotor_time_rising[pidx4(b, r)] = tau_rise;
            batch.rotor_time_falling[pidx4(b, r)] = tau_fall;
            batch.rotor_thrust_coeffs[rotor3(b, r, 0)] = 0.0f;
            batch.rotor_thrust_coeffs[rotor3(b, r, 1)] = linear_curve_share * per_rotor_max_thrust / max_rpm;
            batch.rotor_thrust_coeffs[rotor3(b, r, 2)] = (1.0f - linear_curve_share) * per_rotor_max_thrust / (max_rpm * max_rpm);
        }

        for(int i = 0; i < 3; i++){
            batch.initial_p[pidx3(b, i)] = pos_dist(rng);
            batch.initial_v[pidx3(b, i)] = vel_dist(rng);
            batch.initial_omega[pidx3(b, i)] = omega_dist(rng);
            batch.reference_p[pidx3(b, i)] = 0.0f;
            batch.reference_v[pidx3(b, i)] = 0.0f;
        }
        for(int i = 0; i < 9; i++){
            batch.initial_R[pidx9(b, i)] = 0.0f;
        }
        batch.initial_R[pidx9(b, 0)] = 1.0f;
        batch.initial_R[pidx9(b, 4)] = 1.0f;
        batch.initial_R[pidx9(b, 8)] = 1.0f;
        const float hover_force = mass * G / 4.0f;
        const float c1 = linear_curve_share * per_rotor_max_thrust / max_rpm;
        const float c2 = (1.0f - linear_curve_share) * per_rotor_max_thrust / (max_rpm * max_rpm);
        float hover_rpm = max_rpm * std::sqrt(std::max(0.0f, hover_force / std::max(1e-12f, per_rotor_max_thrust)));
        if(c2 > 1e-12f){
            const float discriminant = c1 * c1 + 4.0f * c2 * hover_force;
            hover_rpm = (-c1 + std::sqrt(std::max(0.0f, discriminant))) / (2.0f * c2);
        }
        else if(c1 > 1e-12f){
            hover_rpm = hover_force / c1;
        }
        hover_rpm = std::max(0.0f, std::min(max_rpm, hover_rpm));
        const float hover_action = 2.0f * hover_rpm / max_rpm - 1.0f;
        for(int r = 0; r < 4; r++){
            batch.initial_rpm[pidx4(b, r)] = (hover_action * 0.5f + 0.5f) * max_rpm;
            batch.initial_previous_action[pidx4(b, r)] = hover_action;
        }
    }
    for(std::size_t t = 0; t < horizon; t++){
        for(std::size_t b = 0; b < batch_size; b++){
            for(int a = 0; a < 4; a++){
                const float base = batch.initial_previous_action[pidx4(b, a)];
            batch.actions[idx4(t, b, a, batch_size)] = std::max(-0.95f, std::min(0.95f, base + action_delta_dist(rng)));
            }
            batch.reset_mask[t * batch_size + b] = t == 0 ? 1u : 0u;
            batch.hidden_reset_mask[t * batch_size + b] = batch.hidden_reset_enabled != 0u && t == 0 ? 1u : 0u;
        }
    }
    for(std::size_t b = 0; b < batch_size; b++){
        std::size_t group_count = 0;
        std::size_t group_unique_count = 0;
        for(std::size_t i = 0; i < batch_size; i++){
            bool first = true;
            for(std::size_t j = 0; j < i; j++){
                if(batch.dynamics_group_key[j] == batch.dynamics_group_key[i]){
                    first = false;
                    break;
                }
            }
            if(first){
                group_unique_count++;
            }
            if(batch.dynamics_group_key[i] == batch.dynamics_group_key[b]){
                group_count++;
            }
        }
        batch.group_weight[b] = group_unique_count > 0 && group_count > 0
            ? 1.0f / (static_cast<float>(group_unique_count) * static_cast<float>(group_count))
            : (batch_size > 0 ? 1.0f / static_cast<float>(batch_size) : 0.0f);
    }
}

CorrelatedSizeMassSamplerValidationSummary validate_correlated_size_mass_sampler(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed
){
    CorrelatedSizeMassSamplerValidationSummary summary;
    summary.samples = std::max<std::size_t>(batch_size, 64);
    const std::size_t validation_horizon = std::max<std::size_t>(horizon, 1);

    EulerGpuBatch default_batch;
    EulerGpuBatch explicit_default_batch;
    EulerGpuBatch nominal_batch;
    EulerGpuBatch correlated_batch;
    generate_validation_batch(default_batch, summary.samples, validation_horizon, seed, nullptr, false);
    generate_validation_batch(explicit_default_batch, summary.samples, validation_horizon, seed, nullptr, false, false);
    generate_validation_batch(nominal_batch, summary.samples, validation_horizon, seed, nullptr, true, true);
    generate_validation_batch(correlated_batch, summary.samples, validation_horizon, seed, nullptr, false, true);

    summary.default_disabled =
        !FullGpuTrainingOptions{}.correlated_size_mass_sampling &&
        !GpuPolicyEvalOptions{}.correlated_size_mass_sampling;

    auto update_max_abs = [](float& current, float lhs, float rhs){
        current = std::max(current, std::fabs(lhs - rhs));
    };

    for(std::size_t b = 0; b < summary.samples; b++){
        update_max_abs(summary.max_default_batch_abs_error, default_batch.mass[b], explicit_default_batch.mass[b]);
        update_max_abs(summary.max_default_batch_abs_error, default_batch.group_weight[b], explicit_default_batch.group_weight[b]);
        update_max_abs(summary.max_default_batch_abs_error, default_batch.J[pidx9(b, 0)], explicit_default_batch.J[pidx9(b, 0)]);
        update_max_abs(summary.max_default_batch_abs_error, default_batch.J[pidx9(b, 4)], explicit_default_batch.J[pidx9(b, 4)]);
        update_max_abs(summary.max_default_batch_abs_error, default_batch.J[pidx9(b, 8)], explicit_default_batch.J[pidx9(b, 8)]);
        update_max_abs(summary.max_default_batch_abs_error, default_batch.rotor_positions[rotor3(b, 0, 0)], explicit_default_batch.rotor_positions[rotor3(b, 0, 0)]);
        update_max_abs(summary.max_default_batch_abs_error, default_batch.rotor_thrust_coeffs[rotor3(b, 0, 1)], explicit_default_batch.rotor_thrust_coeffs[rotor3(b, 0, 1)]);
        update_max_abs(summary.max_default_batch_abs_error, default_batch.rotor_thrust_coeffs[rotor3(b, 0, 2)], explicit_default_batch.rotor_thrust_coeffs[rotor3(b, 0, 2)]);
    }
    summary.default_disabled = summary.default_disabled && summary.max_default_batch_abs_error <= 1e-7f;

    constexpr float nominal_mass = 0.05f;
    constexpr float nominal_arm = 0.09f;
    constexpr float nominal_thrust_to_weight = 3.0f;
    constexpr float nominal_torque_to_inertia = 250.0f;
    constexpr float nominal_jx = 0.55f * nominal_mass * nominal_arm * nominal_arm;
    constexpr float nominal_jz = 1.10f * nominal_mass * nominal_arm * nominal_arm;
    constexpr float nominal_per_rotor_max_thrust = nominal_thrust_to_weight * nominal_mass * G / 4.0f;
    constexpr float mass_min = 0.02f;
    constexpr float mass_max = 0.25f;
    constexpr float size_factor_low = 1.0f / 1.20f;
    constexpr float size_factor_high = 1.20f;
    constexpr float max_rpm = 22000.0f;

    summary.fixed_nominal_unchanged = true;
    for(std::size_t b = 0; b < summary.samples; b++){
        update_max_abs(summary.max_nominal_abs_error, nominal_batch.mass[b], nominal_mass);
        update_max_abs(summary.max_nominal_abs_error, nominal_batch.J[pidx9(b, 0)], nominal_jx);
        update_max_abs(summary.max_nominal_abs_error, nominal_batch.J[pidx9(b, 4)], nominal_jx);
        update_max_abs(summary.max_nominal_abs_error, nominal_batch.J[pidx9(b, 8)], nominal_jz);
        update_max_abs(summary.max_nominal_abs_error, nominal_batch.rotor_positions[rotor3(b, 0, 0)], nominal_arm);
        update_max_abs(summary.max_nominal_abs_error, nominal_batch.rotor_positions[rotor3(b, 0, 1)], nominal_arm);
        if(nominal_batch.dynamics_size_mass_bin[b] != 0u ||
           nominal_batch.dynamics_thrust_to_weight_bin[b] != 0u ||
           nominal_batch.dynamics_torque_to_inertia_bin[b] != 0u ||
           nominal_batch.dynamics_motor_delay_bin[b] != 0u ||
           nominal_batch.dynamics_curve_shape_bin[b] != 0u ||
           nominal_batch.dynamics_group_key[b] != 0u){
            summary.fixed_nominal_unchanged = false;
        }
    }
    summary.fixed_nominal_unchanged = summary.fixed_nominal_unchanged && summary.max_nominal_abs_error <= 1e-6f;

    summary.mass_min = std::numeric_limits<float>::infinity();
    summary.mass_max = -std::numeric_limits<float>::infinity();
    summary.size_factor_min = std::numeric_limits<float>::infinity();
    summary.size_factor_max = -std::numeric_limits<float>::infinity();
    for(std::size_t b = 0; b < summary.samples; b++){
        const float mass = correlated_batch.mass[b];
        const float arm_x = std::fabs(correlated_batch.rotor_positions[rotor3(b, 0, 0)]);
        const float arm_y = std::fabs(correlated_batch.rotor_positions[rotor3(b, 0, 1)]);
        const float arm = 0.5f * (arm_x + arm_y);
        const float rotor_distance_factor = arm / nominal_arm;
        const float scale_relative = std::cbrt(mass / nominal_mass);
        const float size_factor = rotor_distance_factor / std::max(1e-6f, scale_relative);
        const float c1 = correlated_batch.rotor_thrust_coeffs[rotor3(b, 0, 1)];
        const float c2 = correlated_batch.rotor_thrust_coeffs[rotor3(b, 0, 2)];
        const float per_rotor_max_thrust = c1 * max_rpm + c2 * max_rpm * max_rpm;
        const float thrust_to_weight = per_rotor_max_thrust * 4.0f / std::max(1e-6f, mass * G);
        const float factor_thrust_coefficients = per_rotor_max_thrust / nominal_per_rotor_max_thrust;
        const float expected_factor_thrust_coefficients = (thrust_to_weight / nominal_thrust_to_weight) * (mass / nominal_mass);
        const float jx = correlated_batch.J[pidx9(b, 0)];
        const float jy = correlated_batch.J[pidx9(b, 4)];
        const float jz = correlated_batch.J[pidx9(b, 8)];
        const float inferred_torque_to_inertia =
            (nominal_jx / std::max(1e-12f, jx)) * rotor_distance_factor * nominal_torque_to_inertia;

        summary.mass_min = std::min(summary.mass_min, mass);
        summary.mass_max = std::max(summary.mass_max, mass);
        summary.size_factor_min = std::min(summary.size_factor_min, size_factor);
        summary.size_factor_max = std::max(summary.size_factor_max, size_factor);

        if(!std::isfinite(mass) || !std::isfinite(arm) || !std::isfinite(size_factor) ||
           !std::isfinite(per_rotor_max_thrust) || !std::isfinite(thrust_to_weight) ||
           !std::isfinite(jx) || !std::isfinite(jy) || !std::isfinite(jz)){
            summary.nan_inf_count++;
        }
        if(mass < mass_min){
            summary.max_size_factor_bounds_error = std::max(summary.max_size_factor_bounds_error, mass_min - mass);
        }
        if(mass > mass_max){
            summary.max_size_factor_bounds_error = std::max(summary.max_size_factor_bounds_error, mass - mass_max);
        }
        if(size_factor < size_factor_low){
            summary.max_size_factor_bounds_error = std::max(summary.max_size_factor_bounds_error, size_factor_low - size_factor);
        }
        if(size_factor > size_factor_high){
            summary.max_size_factor_bounds_error = std::max(summary.max_size_factor_bounds_error, size_factor - size_factor_high);
        }
        summary.max_thrust_factor_abs_error = std::max(
            summary.max_thrust_factor_abs_error,
            std::fabs(factor_thrust_coefficients - expected_factor_thrust_coefficients)
        );
        if(thrust_to_weight < 1.5f){
            summary.max_thrust_to_weight_bounds_error = std::max(summary.max_thrust_to_weight_bounds_error, 1.5f - thrust_to_weight);
        }
        if(thrust_to_weight > 5.0f){
            summary.max_thrust_to_weight_bounds_error = std::max(summary.max_thrust_to_weight_bounds_error, thrust_to_weight - 5.0f);
        }
        if(inferred_torque_to_inertia < 125.0f){
            summary.max_inertia_bounds_error = std::max(summary.max_inertia_bounds_error, 125.0f - inferred_torque_to_inertia);
        }
        if(inferred_torque_to_inertia > 500.0f){
            summary.max_inertia_bounds_error = std::max(summary.max_inertia_bounds_error, inferred_torque_to_inertia - 500.0f);
        }
        update_max_abs(summary.max_j_inverse_abs_error, correlated_batch.J_inv[pidx9(b, 0)] * jx, 1.0f);
        update_max_abs(summary.max_j_inverse_abs_error, correlated_batch.J_inv[pidx9(b, 4)] * jy, 1.0f);
        update_max_abs(summary.max_j_inverse_abs_error, correlated_batch.J_inv[pidx9(b, 8)] * jz, 1.0f);
    }

    if(summary.max_size_factor_bounds_error > 1e-5f) summary.formula_mismatch_count++;
    if(summary.max_thrust_factor_abs_error > 1e-5f) summary.formula_mismatch_count++;
    if(summary.max_thrust_to_weight_bounds_error > 1e-5f) summary.formula_mismatch_count++;
    if(summary.max_inertia_bounds_error > 1e-3f) summary.formula_mismatch_count++;
    if(summary.max_j_inverse_abs_error > 1e-5f) summary.formula_mismatch_count++;
    summary.finite = summary.nan_inf_count == 0;
    summary.correlated_formula_close = summary.formula_mismatch_count == 0;
    summary.passed =
        summary.default_disabled &&
        summary.fixed_nominal_unchanged &&
        summary.correlated_formula_close &&
        summary.finite;
    return summary;
}

int assemble_observations_gpu(
    const EulerGpuBatch& batch,
    const EulerGpuRunOptions& options,
    std::size_t step_i,
    std::vector<float>& observations
){
    if(batch.batch_size == 0 || batch.horizon == 0 || step_i > batch.horizon){
        std::cerr << "GPU observation assembly requires non-zero batch/horizon and a valid step index.\n";
        return 1;
    }
    DeviceArrays d;
    try{
        CUDA_CHECK(cudaSetDevice(options.device));
        allocate(d, batch.batch_size, batch.horizon);
        copy_input_to_device(batch, d);
        const int block = 256;
        const int grid = static_cast<int>((batch.batch_size + block - 1) / block);
        for(std::size_t t = 0; t < step_i; t++){
            forward_step_kernel<<<grid, block>>>(d, batch.batch_size, t);
            CUDA_CHECK(cudaGetLastError());
        }
        observation_kernel<<<grid, block>>>(d, batch.batch_size, step_i);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        observations.assign(batch.batch_size * EULER_OBSERVATION_DIM, 0.0f);
        CUDA_CHECK(cudaMemcpy(
            observations.data(),
            d.observations + step_i * batch.batch_size * EULER_OBSERVATION_DIM,
            sizeof(float) * observations.size(),
            cudaMemcpyDeviceToHost
        ));
        cuda_free(d);
        return 0;
    }
    catch(const std::exception& e){
        std::cerr << e.what() << "\n";
        cuda_free(d);
        return 1;
    }
}

ObservationValidationSummary validate_observations_against_cpu(
    const EulerGpuBatch& batch,
    const EulerGpuRunOptions& options,
    std::size_t step_i
){
    namespace diff = rl_tools::rl::environments::l2f::diff;
    using TI = std::uint64_t;
    if(step_i > batch.horizon || batch.horizon > CPU_MAX_HORIZON){
        throw std::runtime_error("Observation validation step or horizon is out of range");
    }

    std::vector<float> gpu_observations;
    if(assemble_observations_gpu(batch, options, step_i, gpu_observations) != 0){
        throw std::runtime_error("GPU observation assembly failed");
    }

    std::vector<float> cpu_observations(batch.batch_size * EULER_OBSERVATION_DIM, 0.0f);
    for(std::size_t b = 0; b < batch.batch_size; b++){
        CpuSimpleParameters parameters = cpu_parameters_from_batch(batch, b);
        diff::EulerState<float, TI> state{};
        for(int i = 0; i < 3; i++){
            state.p[i] = batch.initial_p[pidx3(b, i)];
            state.v[i] = batch.initial_v[pidx3(b, i)];
            state.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                state.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            state.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            state.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        diff::EulerStepCache<float> cache{};
        for(std::size_t t = 0; t < step_i; t++){
            float action[4];
            for(int a = 0; a < 4; a++){
                action[a] = batch.actions[idx4(t, b, a, batch.batch_size)];
            }
            diff::EulerState<float, TI> next{};
            diff::step<CpuSimpleParameters, float, TI>(parameters, state, action, next, cache);
            state = next;
        }
        diff::TrackingReference<float> ref{};
        for(int i = 0; i < 3; i++){
            ref.p[i] = batch.reference_p[pidx3(b, i)];
            ref.v[i] = batch.reference_v[pidx3(b, i)];
        }

        std::size_t out = 0;
        for(int i = 0; i < 3; i++){
            cpu_observations[b * EULER_OBSERVATION_DIM + out++] = state.p[i] - ref.p[i];
        }
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                cpu_observations[b * EULER_OBSERVATION_DIM + out++] = state.R[i][j];
            }
        }
        for(int i = 0; i < 3; i++){
            cpu_observations[b * EULER_OBSERVATION_DIM + out++] = state.v[i] - ref.v[i];
        }
        for(int i = 0; i < 3; i++){
            cpu_observations[b * EULER_OBSERVATION_DIM + out++] = state.omega[i];
        }
        for(int i = 0; i < 4; i++){
            cpu_observations[b * EULER_OBSERVATION_DIM + out++] = state.previous_action[i];
        }
    }

    ObservationValidationSummary summary;
    double error_sum = 0.0;
    for(std::size_t i = 0; i < cpu_observations.size(); i++){
        const float gpu_value = gpu_observations[i];
        const float cpu_value = cpu_observations[i];
        if(!std::isfinite(gpu_value) || !std::isfinite(cpu_value)){
            summary.nan_inf_count++;
            continue;
        }
        const float abs_error = std::abs(gpu_value - cpu_value);
        summary.max_abs_error = std::max(summary.max_abs_error, abs_error);
        error_sum += abs_error;
    }
    summary.mean_abs_error = static_cast<float>(error_sum / std::max<std::size_t>(1, cpu_observations.size()));
    summary.passed = summary.nan_inf_count == 0 && summary.max_abs_error < 1e-5f;
    return summary;
}

namespace{

__host__ __device__ inline std::size_t actor_idx3(std::size_t t, std::size_t b, std::size_t d, std::size_t batch_size, std::size_t dim){
    return (t * batch_size + b) * dim + d;
}

__host__ __device__ inline float rdac_sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ inline float rdac_relu(float x){
    return x > 0.0f ? x : 0.0f;
}

struct ActorWeightsDevice{
    float* encoder_w = nullptr;      // [16, 48]
    float* encoder_b = nullptr;      // [16]
    float* gru_w_input = nullptr;    // [48, 16]
    float* gru_b_input = nullptr;    // [48]
    float* gru_w_hidden = nullptr;   // [48, 16]
    float* gru_b_hidden = nullptr;   // [48]
    float* gru_h0 = nullptr;         // [16]
    float* actor_w = nullptr;        // [4, 38]
    float* actor_b = nullptr;        // [4]
    float* critic_w = nullptr;       // [1, 38]
    float* critic_b = nullptr;       // [1]
};

struct ActorWeightsHost{
    std::vector<float> encoder_w;
    std::vector<float> encoder_b;
    std::vector<float> gru_w_input;
    std::vector<float> gru_b_input;
    std::vector<float> gru_w_hidden;
    std::vector<float> gru_b_hidden;
    std::vector<float> gru_h0;
    std::vector<float> actor_w;
    std::vector<float> actor_b;
    std::vector<float> critic_w;
    std::vector<float> critic_b;

    void resize(){
        encoder_w.resize(RDAC_HIDDEN_DIM * RDAC_POLICY_INPUT_DIM);
        encoder_b.resize(RDAC_HIDDEN_DIM);
        gru_w_input.resize(3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM);
        gru_b_input.resize(3 * RDAC_HIDDEN_DIM);
        gru_w_hidden.resize(3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM);
        gru_b_hidden.resize(3 * RDAC_HIDDEN_DIM);
        gru_h0.resize(RDAC_HIDDEN_DIM);
        actor_w.resize(RDAC_ACTION_DIM * RDAC_ACTOR_HEAD_INPUT_DIM);
        actor_b.resize(RDAC_ACTION_DIM);
        critic_w.resize(RDAC_CRITIC_DIM * RDAC_CRITIC_HEAD_INPUT_DIM);
        critic_b.resize(RDAC_CRITIC_DIM);
    }
};

void generate_actor_weights(ActorWeightsHost& weights, std::mt19937& rng, float scale = 0.12f){
    weights.resize();
    std::uniform_real_distribution<float> weight_dist(-scale, scale);
    for(auto* vec: {
        &weights.encoder_w, &weights.encoder_b, &weights.gru_w_input, &weights.gru_b_input,
        &weights.gru_w_hidden, &weights.gru_b_hidden, &weights.gru_h0, &weights.actor_w, &weights.actor_b,
        &weights.critic_w, &weights.critic_b
    }){
        for(float& value: *vec){
            value = weight_dist(rng);
        }
    }
}

struct ActorBuffersDevice{
    float* policy_input = nullptr;      // [T, B, 48]
    float* observation = nullptr;       // [T, B, 22]
    float* encoder = nullptr;           // [T, B, 16]
    float* hidden = nullptr;            // [T, B, 16]
    float* raw_action = nullptr;        // [T, B, 4]
    float* bounded_action = nullptr;    // [T, B, 4]
    float* action_derivative = nullptr; // [T, B, 4]
    float* raw_action_gradient = nullptr; // [T, B, 4]
    float* critic_output = nullptr;     // [T, B, 1]
    float* critic_target = nullptr;     // [T, B, 1]
    float* critic_weight = nullptr;     // [T, B, 1]
    float* critic_output_gradient = nullptr; // [T, B, 1]
};

struct ActorGradientsDevice{
    float* encoder_w = nullptr;
    float* encoder_b = nullptr;
    float* gru_w_input = nullptr;
    float* gru_b_input = nullptr;
    float* gru_w_hidden = nullptr;
    float* gru_b_hidden = nullptr;
    float* gru_h0 = nullptr;
    float* actor_w = nullptr;
    float* actor_b = nullptr;
    float* critic_w = nullptr;
    float* critic_b = nullptr;
    float* hidden_adjoint = nullptr; // [B, 16]
};

void actor_cuda_alloc(float*& ptr, std::size_t count){
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(float) * count));
}

void actor_allocate_weights(ActorWeightsDevice& weights){
    actor_cuda_alloc(weights.encoder_w, RDAC_HIDDEN_DIM * RDAC_POLICY_INPUT_DIM);
    actor_cuda_alloc(weights.encoder_b, RDAC_HIDDEN_DIM);
    actor_cuda_alloc(weights.gru_w_input, 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(weights.gru_b_input, 3 * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(weights.gru_w_hidden, 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(weights.gru_b_hidden, 3 * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(weights.gru_h0, RDAC_HIDDEN_DIM);
    actor_cuda_alloc(weights.actor_w, RDAC_ACTION_DIM * RDAC_ACTOR_HEAD_INPUT_DIM);
    actor_cuda_alloc(weights.actor_b, RDAC_ACTION_DIM);
    actor_cuda_alloc(weights.critic_w, RDAC_CRITIC_DIM * RDAC_CRITIC_HEAD_INPUT_DIM);
    actor_cuda_alloc(weights.critic_b, RDAC_CRITIC_DIM);
}

void actor_free(ActorWeightsDevice& weights){
    float* ptrs[] = {
        weights.encoder_w, weights.encoder_b, weights.gru_w_input, weights.gru_b_input,
        weights.gru_w_hidden, weights.gru_b_hidden, weights.gru_h0, weights.actor_w, weights.actor_b,
        weights.critic_w, weights.critic_b
    };
    for(float* ptr: ptrs){
        if(ptr != nullptr){
            cudaFree(ptr);
        }
    }
    weights = ActorWeightsDevice{};
}

void actor_free(ActorBuffersDevice& buffers){
    float* ptrs[] = {
        buffers.policy_input, buffers.observation, buffers.encoder, buffers.hidden,
        buffers.raw_action, buffers.bounded_action, buffers.action_derivative,
        buffers.raw_action_gradient, buffers.critic_output, buffers.critic_target,
        buffers.critic_weight, buffers.critic_output_gradient
    };
    for(float* ptr: ptrs){
        if(ptr != nullptr){
            cudaFree(ptr);
        }
    }
    buffers = ActorBuffersDevice{};
}

void actor_free(ActorGradientsDevice& gradients){
    float* ptrs[] = {
        gradients.encoder_w, gradients.encoder_b, gradients.gru_w_input, gradients.gru_b_input,
        gradients.gru_w_hidden, gradients.gru_b_hidden, gradients.gru_h0, gradients.actor_w,
        gradients.actor_b, gradients.critic_w, gradients.critic_b, gradients.hidden_adjoint
    };
    for(float* ptr: ptrs){
        if(ptr != nullptr){
            cudaFree(ptr);
        }
    }
    gradients = ActorGradientsDevice{};
}

void actor_allocate(ActorWeightsDevice& weights, ActorBuffersDevice& buffers, std::size_t batch_size, std::size_t sequence_length){
    actor_allocate_weights(weights);

    actor_cuda_alloc(buffers.policy_input, sequence_length * batch_size * RDAC_POLICY_INPUT_DIM);
    actor_cuda_alloc(buffers.observation, sequence_length * batch_size * EULER_OBSERVATION_DIM);
    actor_cuda_alloc(buffers.encoder, sequence_length * batch_size * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(buffers.hidden, sequence_length * batch_size * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(buffers.raw_action, sequence_length * batch_size * RDAC_ACTION_DIM);
    actor_cuda_alloc(buffers.bounded_action, sequence_length * batch_size * RDAC_ACTION_DIM);
    actor_cuda_alloc(buffers.action_derivative, sequence_length * batch_size * RDAC_ACTION_DIM);
    actor_cuda_alloc(buffers.raw_action_gradient, sequence_length * batch_size * RDAC_ACTION_DIM);
    actor_cuda_alloc(buffers.critic_output, sequence_length * batch_size * RDAC_CRITIC_DIM);
    actor_cuda_alloc(buffers.critic_target, sequence_length * batch_size * RDAC_CRITIC_DIM);
    actor_cuda_alloc(buffers.critic_weight, sequence_length * batch_size * RDAC_CRITIC_DIM);
    actor_cuda_alloc(buffers.critic_output_gradient, sequence_length * batch_size * RDAC_CRITIC_DIM);
}

void actor_allocate(ActorGradientsDevice& gradients, std::size_t batch_size){
    actor_cuda_alloc(gradients.encoder_w, RDAC_HIDDEN_DIM * RDAC_POLICY_INPUT_DIM);
    actor_cuda_alloc(gradients.encoder_b, RDAC_HIDDEN_DIM);
    actor_cuda_alloc(gradients.gru_w_input, 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(gradients.gru_b_input, 3 * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(gradients.gru_w_hidden, 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(gradients.gru_b_hidden, 3 * RDAC_HIDDEN_DIM);
    actor_cuda_alloc(gradients.gru_h0, RDAC_HIDDEN_DIM);
    actor_cuda_alloc(gradients.actor_w, RDAC_ACTION_DIM * RDAC_ACTOR_HEAD_INPUT_DIM);
    actor_cuda_alloc(gradients.actor_b, RDAC_ACTION_DIM);
    actor_cuda_alloc(gradients.critic_w, RDAC_CRITIC_DIM * RDAC_CRITIC_HEAD_INPUT_DIM);
    actor_cuda_alloc(gradients.critic_b, RDAC_CRITIC_DIM);
    actor_cuda_alloc(gradients.hidden_adjoint, batch_size * RDAC_HIDDEN_DIM);
}

void zero_actor_gradients(ActorGradientsDevice& gradients, std::size_t batch_size){
    CUDA_CHECK(cudaMemset(gradients.encoder_w, 0, sizeof(float) * RDAC_HIDDEN_DIM * RDAC_POLICY_INPUT_DIM));
    CUDA_CHECK(cudaMemset(gradients.encoder_b, 0, sizeof(float) * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(gradients.gru_w_input, 0, sizeof(float) * 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(gradients.gru_b_input, 0, sizeof(float) * 3 * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(gradients.gru_w_hidden, 0, sizeof(float) * 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(gradients.gru_b_hidden, 0, sizeof(float) * 3 * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(gradients.gru_h0, 0, sizeof(float) * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(gradients.actor_w, 0, sizeof(float) * RDAC_ACTION_DIM * RDAC_ACTOR_HEAD_INPUT_DIM));
    CUDA_CHECK(cudaMemset(gradients.actor_b, 0, sizeof(float) * RDAC_ACTION_DIM));
    CUDA_CHECK(cudaMemset(gradients.critic_w, 0, sizeof(float) * RDAC_CRITIC_DIM * RDAC_CRITIC_HEAD_INPUT_DIM));
    CUDA_CHECK(cudaMemset(gradients.critic_b, 0, sizeof(float) * RDAC_CRITIC_DIM));
    CUDA_CHECK(cudaMemset(gradients.hidden_adjoint, 0, sizeof(float) * batch_size * RDAC_HIDDEN_DIM));
}

void copy_actor_weights_to_device(const ActorWeightsHost& source, ActorWeightsDevice& target){
    CUDA_CHECK(cudaMemcpy(target.encoder_w, source.encoder_w.data(), sizeof(float) * source.encoder_w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.encoder_b, source.encoder_b.data(), sizeof(float) * source.encoder_b.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_w_input, source.gru_w_input.data(), sizeof(float) * source.gru_w_input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_b_input, source.gru_b_input.data(), sizeof(float) * source.gru_b_input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_w_hidden, source.gru_w_hidden.data(), sizeof(float) * source.gru_w_hidden.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_b_hidden, source.gru_b_hidden.data(), sizeof(float) * source.gru_b_hidden.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_h0, source.gru_h0.data(), sizeof(float) * source.gru_h0.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.actor_w, source.actor_w.data(), sizeof(float) * source.actor_w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.actor_b, source.actor_b.data(), sizeof(float) * source.actor_b.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.critic_w, source.critic_w.data(), sizeof(float) * source.critic_w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.critic_b, source.critic_b.data(), sizeof(float) * source.critic_b.size(), cudaMemcpyHostToDevice));
}

void zero_actor_weights_device(ActorWeightsDevice& target){
    CUDA_CHECK(cudaMemset(target.encoder_w, 0, sizeof(float) * RDAC_HIDDEN_DIM * RDAC_POLICY_INPUT_DIM));
    CUDA_CHECK(cudaMemset(target.encoder_b, 0, sizeof(float) * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(target.gru_w_input, 0, sizeof(float) * 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(target.gru_b_input, 0, sizeof(float) * 3 * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(target.gru_w_hidden, 0, sizeof(float) * 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(target.gru_b_hidden, 0, sizeof(float) * 3 * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(target.gru_h0, 0, sizeof(float) * RDAC_HIDDEN_DIM));
    CUDA_CHECK(cudaMemset(target.actor_w, 0, sizeof(float) * RDAC_ACTION_DIM * RDAC_ACTOR_HEAD_INPUT_DIM));
    CUDA_CHECK(cudaMemset(target.actor_b, 0, sizeof(float) * RDAC_ACTION_DIM));
    CUDA_CHECK(cudaMemset(target.critic_w, 0, sizeof(float) * RDAC_CRITIC_DIM * RDAC_CRITIC_HEAD_INPUT_DIM));
    CUDA_CHECK(cudaMemset(target.critic_b, 0, sizeof(float) * RDAC_CRITIC_DIM));
}

void copy_actor_gradients_to_device(const ActorWeightsHost& source, ActorGradientsDevice& target){
    CUDA_CHECK(cudaMemcpy(target.encoder_w, source.encoder_w.data(), sizeof(float) * source.encoder_w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.encoder_b, source.encoder_b.data(), sizeof(float) * source.encoder_b.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_w_input, source.gru_w_input.data(), sizeof(float) * source.gru_w_input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_b_input, source.gru_b_input.data(), sizeof(float) * source.gru_b_input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_w_hidden, source.gru_w_hidden.data(), sizeof(float) * source.gru_w_hidden.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_b_hidden, source.gru_b_hidden.data(), sizeof(float) * source.gru_b_hidden.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.gru_h0, source.gru_h0.data(), sizeof(float) * source.gru_h0.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.actor_w, source.actor_w.data(), sizeof(float) * source.actor_w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.actor_b, source.actor_b.data(), sizeof(float) * source.actor_b.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.critic_w, source.critic_w.data(), sizeof(float) * source.critic_w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(target.critic_b, source.critic_b.data(), sizeof(float) * source.critic_b.size(), cudaMemcpyHostToDevice));
}

void copy_actor_weights_to_host(const ActorWeightsDevice& source, ActorWeightsHost& target){
    target.resize();
    CUDA_CHECK(cudaMemcpy(target.encoder_w.data(), source.encoder_w, sizeof(float) * target.encoder_w.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.encoder_b.data(), source.encoder_b, sizeof(float) * target.encoder_b.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_w_input.data(), source.gru_w_input, sizeof(float) * target.gru_w_input.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_b_input.data(), source.gru_b_input, sizeof(float) * target.gru_b_input.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_w_hidden.data(), source.gru_w_hidden, sizeof(float) * target.gru_w_hidden.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_b_hidden.data(), source.gru_b_hidden, sizeof(float) * target.gru_b_hidden.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_h0.data(), source.gru_h0, sizeof(float) * target.gru_h0.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.actor_w.data(), source.actor_w, sizeof(float) * target.actor_w.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.actor_b.data(), source.actor_b, sizeof(float) * target.actor_b.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.critic_w.data(), source.critic_w, sizeof(float) * target.critic_w.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.critic_b.data(), source.critic_b, sizeof(float) * target.critic_b.size(), cudaMemcpyDeviceToHost));
}

void copy_actor_gradients_to_host(const ActorGradientsDevice& source, ActorWeightsHost& target){
    target.resize();
    CUDA_CHECK(cudaMemcpy(target.encoder_w.data(), source.encoder_w, sizeof(float) * target.encoder_w.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.encoder_b.data(), source.encoder_b, sizeof(float) * target.encoder_b.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_w_input.data(), source.gru_w_input, sizeof(float) * target.gru_w_input.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_b_input.data(), source.gru_b_input, sizeof(float) * target.gru_b_input.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_w_hidden.data(), source.gru_w_hidden, sizeof(float) * target.gru_w_hidden.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_b_hidden.data(), source.gru_b_hidden, sizeof(float) * target.gru_b_hidden.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.gru_h0.data(), source.gru_h0, sizeof(float) * target.gru_h0.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.actor_w.data(), source.actor_w, sizeof(float) * target.actor_w.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.actor_b.data(), source.actor_b, sizeof(float) * target.actor_b.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.critic_w.data(), source.critic_w, sizeof(float) * target.critic_w.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(target.critic_b.data(), source.critic_b, sizeof(float) * target.critic_b.size(), cudaMemcpyDeviceToHost));
}

__global__ void build_policy_input_from_observation_kernel(
    DeviceArrays rollout,
    ActorBuffersDevice actor,
    std::size_t batch_size,
    std::size_t step_i
){
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= batch_size){
        return;
    }
    for(int o = 0; o < static_cast<int>(EULER_OBSERVATION_DIM); o++){
        const float value = rollout.observations[idx_obs(step_i, b, o, batch_size)];
        actor.observation[actor_idx3(step_i, b, o, batch_size, EULER_OBSERVATION_DIM)] = value;
        actor.policy_input[actor_idx3(step_i, b, o, batch_size, RDAC_POLICY_INPUT_DIM)] = value;
    }
    for(int a = 0; a < static_cast<int>(RDAC_ACTION_DIM); a++){
        actor.policy_input[actor_idx3(step_i, b, EULER_OBSERVATION_DIM + a, batch_size, RDAC_POLICY_INPUT_DIM)] =
            rollout.observations[idx_obs(step_i, b, 18 + a, batch_size)];
    }
    for(int e = 0; e < static_cast<int>(EULER_OBSERVATION_DIM); e++){
        actor.policy_input[actor_idx3(step_i, b, EULER_OBSERVATION_DIM + RDAC_ACTION_DIM + e, batch_size, RDAC_POLICY_INPUT_DIM)] = 0.0f;
    }
}

__global__ void copy_bounded_action_to_rollout_kernel(
    ActorBuffersDevice actor,
    DeviceArrays rollout,
    std::size_t batch_size,
    std::size_t step_i
){
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= batch_size){
        return;
    }
    for(int a = 0; a < static_cast<int>(RDAC_ACTION_DIM); a++){
        rollout.actions[idx4(step_i, b, a, batch_size)] =
            actor.bounded_action[actor_idx3(step_i, b, a, batch_size, RDAC_ACTION_DIM)];
    }
}

__global__ void inject_action_gradient_kernel(
    DeviceArrays rollout,
    ActorBuffersDevice actor,
    std::size_t batch_size,
    std::size_t horizon
){
    const std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t count = horizon * batch_size * RDAC_ACTION_DIM;
    if(linear >= count){
        return;
    }
    const std::size_t action_dim = RDAC_ACTION_DIM;
    const std::size_t action = linear % action_dim;
    const std::size_t batch = (linear / action_dim) % batch_size;
    const std::size_t step = linear / (batch_size * action_dim);
    const float dloss_daction = rollout.action_gradients[idx4(step, batch, action, batch_size)];
    const float derivative = actor.action_derivative[actor_idx3(step, batch, action, batch_size, RDAC_ACTION_DIM)];
    actor.raw_action_gradient[actor_idx3(step, batch, action, batch_size, RDAC_ACTION_DIM)] =
        dloss_daction * derivative;
}

__global__ void rdac_actor_backward_step_kernel(
    ActorWeightsDevice weights,
    ActorBuffersDevice buffers,
    ActorGradientsDevice gradients,
    std::size_t batch_size,
    std::size_t step_i
){
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= batch_size){
        return;
    }

    float actor_input[RDAC_ACTOR_HEAD_INPUT_DIM];
    for(int o = 0; o < static_cast<int>(EULER_OBSERVATION_DIM); o++){
        actor_input[o] = buffers.observation[actor_idx3(step_i, b, o, batch_size, EULER_OBSERVATION_DIM)];
    }
    float h_next[RDAC_HIDDEN_DIM];
    float h_prev[RDAC_HIDDEN_DIM];
    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        h_next[h] = buffers.hidden[actor_idx3(step_i, b, h, batch_size, RDAC_HIDDEN_DIM)];
        h_prev[h] = step_i == 0
            ? weights.gru_h0[h]
            : buffers.hidden[actor_idx3(step_i - 1, b, h, batch_size, RDAC_HIDDEN_DIM)];
        actor_input[EULER_OBSERVATION_DIM + h] = h_next[h];
    }

    float d_actor_input[RDAC_ACTOR_HEAD_INPUT_DIM];
    for(int k = 0; k < static_cast<int>(RDAC_ACTOR_HEAD_INPUT_DIM); k++){
        d_actor_input[k] = 0.0f;
    }
    float d_critic_input[RDAC_CRITIC_HEAD_INPUT_DIM];
    for(int k = 0; k < static_cast<int>(RDAC_CRITIC_HEAD_INPUT_DIM); k++){
        d_critic_input[k] = 0.0f;
    }
    for(int c = 0; c < static_cast<int>(RDAC_CRITIC_DIM); c++){
        const float d_q = buffers.critic_output_gradient[actor_idx3(step_i, b, c, batch_size, RDAC_CRITIC_DIM)];
        atomicAdd(&gradients.critic_b[c], d_q);
        for(int k = 0; k < static_cast<int>(RDAC_CRITIC_HEAD_INPUT_DIM); k++){
            atomicAdd(&gradients.critic_w[c * RDAC_CRITIC_HEAD_INPUT_DIM + k], d_q * actor_input[k]);
            d_critic_input[k] += weights.critic_w[c * RDAC_CRITIC_HEAD_INPUT_DIM + k] * d_q;
        }
    }
    for(int a = 0; a < static_cast<int>(RDAC_ACTION_DIM); a++){
        const float d_raw = buffers.raw_action_gradient[actor_idx3(step_i, b, a, batch_size, RDAC_ACTION_DIM)];
        atomicAdd(&gradients.actor_b[a], d_raw);
        for(int k = 0; k < static_cast<int>(RDAC_ACTOR_HEAD_INPUT_DIM); k++){
            atomicAdd(&gradients.actor_w[a * RDAC_ACTOR_HEAD_INPUT_DIM + k], d_raw * actor_input[k]);
            d_actor_input[k] += weights.actor_w[a * RDAC_ACTOR_HEAD_INPUT_DIM + k] * d_raw;
        }
    }

    float d_h_next[RDAC_HIDDEN_DIM];
    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        d_h_next[h] = gradients.hidden_adjoint[b * RDAC_HIDDEN_DIM + h] +
            d_actor_input[EULER_OBSERVATION_DIM + h] +
            d_critic_input[EULER_OBSERVATION_DIM + h];
    }

    float encoder[RDAC_HIDDEN_DIM];
    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        encoder[h] = buffers.encoder[actor_idx3(step_i, b, h, batch_size, RDAC_HIDDEN_DIM)];
    }

    float hidden_affine[3 * RDAC_HIDDEN_DIM];
    float gate[3 * RDAC_HIDDEN_DIM];
    for(int g = 0; g < static_cast<int>(3 * RDAC_HIDDEN_DIM); g++){
        float h_value = weights.gru_b_hidden[g];
        for(int k = 0; k < static_cast<int>(RDAC_HIDDEN_DIM); k++){
            h_value += weights.gru_w_hidden[g * RDAC_HIDDEN_DIM + k] * h_prev[k];
        }
        hidden_affine[g] = h_value;
        float value = h_value + weights.gru_b_input[g];
        for(int k = 0; k < static_cast<int>(RDAC_HIDDEN_DIM); k++){
            value += weights.gru_w_input[g * RDAC_HIDDEN_DIM + k] * encoder[k];
        }
        gate[g] = value;
    }

    float d_gate[3 * RDAC_HIDDEN_DIM];
    float d_hidden_affine[3 * RDAC_HIDDEN_DIM];
    for(int g = 0; g < static_cast<int>(3 * RDAC_HIDDEN_DIM); g++){
        d_gate[g] = 0.0f;
        d_hidden_affine[g] = 0.0f;
    }
    float d_h_prev[RDAC_HIDDEN_DIM];
    float d_encoder[RDAC_HIDDEN_DIM];
    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        d_h_prev[h] = 0.0f;
        d_encoder[h] = 0.0f;
    }

    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        const float r = rdac_sigmoid(gate[h]);
        const float z = rdac_sigmoid(gate[RDAC_HIDDEN_DIM + h]);
        const float n_pre = gate[2 * RDAC_HIDDEN_DIM + h] + r * hidden_affine[2 * RDAC_HIDDEN_DIM + h];
        const float n = tanhf(n_pre);
        const float dh = d_h_next[h];
        const float d_z = dh * (h_prev[h] - n);
        const float d_n = dh * (1.0f - z);
        d_h_prev[h] += dh * z;
        const float d_n_pre = d_n * (1.0f - n * n);
        d_gate[2 * RDAC_HIDDEN_DIM + h] += d_n_pre;
        d_hidden_affine[2 * RDAC_HIDDEN_DIM + h] += d_n_pre * r;
        const float d_r = d_n_pre * hidden_affine[2 * RDAC_HIDDEN_DIM + h];
        d_gate[h] += d_r * r * (1.0f - r);
        d_gate[RDAC_HIDDEN_DIM + h] += d_z * z * (1.0f - z);
    }

    for(int g = 0; g < static_cast<int>(3 * RDAC_HIDDEN_DIM); g++){
        d_hidden_affine[g] += d_gate[g];
        atomicAdd(&gradients.gru_b_input[g], d_gate[g]);
        atomicAdd(&gradients.gru_b_hidden[g], d_hidden_affine[g]);
        for(int k = 0; k < static_cast<int>(RDAC_HIDDEN_DIM); k++){
            atomicAdd(&gradients.gru_w_input[g * RDAC_HIDDEN_DIM + k], d_gate[g] * encoder[k]);
            atomicAdd(&gradients.gru_w_hidden[g * RDAC_HIDDEN_DIM + k], d_hidden_affine[g] * h_prev[k]);
            d_encoder[k] += weights.gru_w_input[g * RDAC_HIDDEN_DIM + k] * d_gate[g];
            d_h_prev[k] += weights.gru_w_hidden[g * RDAC_HIDDEN_DIM + k] * d_hidden_affine[g];
        }
    }

    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        const float d_encoder_pre = encoder[h] > 0.0f ? d_encoder[h] : 0.0f;
        atomicAdd(&gradients.encoder_b[h], d_encoder_pre);
        for(int k = 0; k < static_cast<int>(RDAC_POLICY_INPUT_DIM); k++){
            const float input = buffers.policy_input[actor_idx3(step_i, b, k, batch_size, RDAC_POLICY_INPUT_DIM)];
            atomicAdd(&gradients.encoder_w[h * RDAC_POLICY_INPUT_DIM + k], d_encoder_pre * input);
        }
    }

    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        if(step_i == 0){
            atomicAdd(&gradients.gru_h0[h], d_h_prev[h]);
        }
        else{
            gradients.hidden_adjoint[b * RDAC_HIDDEN_DIM + h] = d_h_prev[h];
        }
    }
}

__global__ void adam_update_kernel(
    float* parameters,
    const float* gradients,
    float* first_moment,
    float* second_moment,
    std::size_t count,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int step
){
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count){
        return;
    }
    const float gradient = gradients[i];
    const float m = beta1 * first_moment[i] + (1.0f - beta1) * gradient;
    const float v = beta2 * second_moment[i] + (1.0f - beta2) * gradient * gradient;
    first_moment[i] = m;
    second_moment[i] = v;
    const float bias_correction1 = 1.0f - powf(beta1, static_cast<float>(step));
    const float bias_correction2 = 1.0f - powf(beta2, static_cast<float>(step));
    const float m_hat = m / bias_correction1;
    const float v_hat = v / bias_correction2;
    parameters[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
}

__global__ void scale_buffer_kernel(float* values, std::size_t count, float scale){
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count){
        return;
    }
    values[i] *= scale;
}

void scale_device_buffer(float* values, std::size_t count, float scale){
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    scale_buffer_kernel<<<grid, block>>>(values, count, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_adam_update(
    float* parameters,
    const float* gradients,
    float* first_moment,
    float* second_moment,
    std::size_t count,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int step
){
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    adam_update_kernel<<<grid, block>>>(
        parameters, gradients, first_moment, second_moment, count,
        learning_rate, beta1, beta2, epsilon, step
    );
    CUDA_CHECK(cudaGetLastError());
}

void adam_update_actor(
    ActorWeightsDevice& parameters,
    const ActorGradientsDevice& gradients,
    ActorWeightsDevice& first_moment,
    ActorWeightsDevice& second_moment,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int step
){
    launch_adam_update(parameters.encoder_w, gradients.encoder_w, first_moment.encoder_w, second_moment.encoder_w, RDAC_HIDDEN_DIM * RDAC_POLICY_INPUT_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.encoder_b, gradients.encoder_b, first_moment.encoder_b, second_moment.encoder_b, RDAC_HIDDEN_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.gru_w_input, gradients.gru_w_input, first_moment.gru_w_input, second_moment.gru_w_input, 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.gru_b_input, gradients.gru_b_input, first_moment.gru_b_input, second_moment.gru_b_input, 3 * RDAC_HIDDEN_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.gru_w_hidden, gradients.gru_w_hidden, first_moment.gru_w_hidden, second_moment.gru_w_hidden, 3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.gru_b_hidden, gradients.gru_b_hidden, first_moment.gru_b_hidden, second_moment.gru_b_hidden, 3 * RDAC_HIDDEN_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.gru_h0, gradients.gru_h0, first_moment.gru_h0, second_moment.gru_h0, RDAC_HIDDEN_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.actor_w, gradients.actor_w, first_moment.actor_w, second_moment.actor_w, RDAC_ACTION_DIM * RDAC_ACTOR_HEAD_INPUT_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.actor_b, gradients.actor_b, first_moment.actor_b, second_moment.actor_b, RDAC_ACTION_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.critic_w, gradients.critic_w, first_moment.critic_w, second_moment.critic_w, RDAC_CRITIC_DIM * RDAC_CRITIC_HEAD_INPUT_DIM, learning_rate, beta1, beta2, epsilon, step);
    launch_adam_update(parameters.critic_b, gradients.critic_b, first_moment.critic_b, second_moment.critic_b, RDAC_CRITIC_DIM, learning_rate, beta1, beta2, epsilon, step);
}

__global__ void rdac_actor_forward_step_kernel(
    ActorWeightsDevice weights,
    ActorBuffersDevice buffers,
    std::size_t batch_size,
    std::size_t step_i,
    float action_bound,
    bool reset_hidden_each_step
){
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= batch_size){
        return;
    }

    float encoder[RDAC_HIDDEN_DIM];
    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        float value = weights.encoder_b[h];
        for(int k = 0; k < static_cast<int>(RDAC_POLICY_INPUT_DIM); k++){
            value += weights.encoder_w[h * RDAC_POLICY_INPUT_DIM + k] *
                buffers.policy_input[actor_idx3(step_i, b, k, batch_size, RDAC_POLICY_INPUT_DIM)];
        }
        encoder[h] = rdac_relu(value);
        buffers.encoder[actor_idx3(step_i, b, h, batch_size, RDAC_HIDDEN_DIM)] = encoder[h];
    }

    float previous_hidden[RDAC_HIDDEN_DIM];
    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        previous_hidden[h] = (step_i == 0 || reset_hidden_each_step)
            ? weights.gru_h0[h]
            : buffers.hidden[actor_idx3(step_i - 1, b, h, batch_size, RDAC_HIDDEN_DIM)];
    }

    float hidden_affine[3 * RDAC_HIDDEN_DIM];
    float gate[3 * RDAC_HIDDEN_DIM];
    for(int g = 0; g < static_cast<int>(3 * RDAC_HIDDEN_DIM); g++){
        float h_value = weights.gru_b_hidden[g];
        for(int k = 0; k < static_cast<int>(RDAC_HIDDEN_DIM); k++){
            h_value += weights.gru_w_hidden[g * RDAC_HIDDEN_DIM + k] * previous_hidden[k];
        }
        hidden_affine[g] = h_value;

        float value = h_value + weights.gru_b_input[g];
        for(int k = 0; k < static_cast<int>(RDAC_HIDDEN_DIM); k++){
            value += weights.gru_w_input[g * RDAC_HIDDEN_DIM + k] * encoder[k];
        }
        gate[g] = value;
    }

    float next_hidden[RDAC_HIDDEN_DIM];
    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        const float r = rdac_sigmoid(gate[h]);
        const float z = rdac_sigmoid(gate[RDAC_HIDDEN_DIM + h]);
        const float n_pre = gate[2 * RDAC_HIDDEN_DIM + h] + r * hidden_affine[2 * RDAC_HIDDEN_DIM + h];
        const float n = tanhf(n_pre);
        next_hidden[h] = (1.0f - z) * n + z * previous_hidden[h];
        buffers.hidden[actor_idx3(step_i, b, h, batch_size, RDAC_HIDDEN_DIM)] = next_hidden[h];
    }

    float actor_input[RDAC_ACTOR_HEAD_INPUT_DIM];
    for(int o = 0; o < static_cast<int>(EULER_OBSERVATION_DIM); o++){
        actor_input[o] = buffers.observation[actor_idx3(step_i, b, o, batch_size, EULER_OBSERVATION_DIM)];
    }
    for(int h = 0; h < static_cast<int>(RDAC_HIDDEN_DIM); h++){
        actor_input[EULER_OBSERVATION_DIM + h] = next_hidden[h];
    }
    for(int a = 0; a < static_cast<int>(RDAC_ACTION_DIM); a++){
        float raw = weights.actor_b[a];
        for(int k = 0; k < static_cast<int>(RDAC_ACTOR_HEAD_INPUT_DIM); k++){
            raw += weights.actor_w[a * RDAC_ACTOR_HEAD_INPUT_DIM + k] * actor_input[k];
        }
        float derivative = 1.0f;
        float bounded = raw;
        if(raw > action_bound){
            bounded = action_bound;
            derivative = 0.0f;
        }
        if(raw < -action_bound){
            bounded = -action_bound;
            derivative = 0.0f;
        }
        buffers.raw_action[actor_idx3(step_i, b, a, batch_size, RDAC_ACTION_DIM)] = raw;
        buffers.bounded_action[actor_idx3(step_i, b, a, batch_size, RDAC_ACTION_DIM)] = bounded;
        buffers.action_derivative[actor_idx3(step_i, b, a, batch_size, RDAC_ACTION_DIM)] = derivative;
    }
    for(int c = 0; c < static_cast<int>(RDAC_CRITIC_DIM); c++){
        float q = weights.critic_b[c];
        for(int k = 0; k < static_cast<int>(RDAC_CRITIC_HEAD_INPUT_DIM); k++){
            q += weights.critic_w[c * RDAC_CRITIC_HEAD_INPUT_DIM + k] * actor_input[k];
        }
        buffers.critic_output[actor_idx3(step_i, b, c, batch_size, RDAC_CRITIC_DIM)] = q;
    }
}

float cpu_rdac_sigmoid(float x){
    return 1.0f / (1.0f + std::exp(-x));
}

void cpu_actor_forward(
    std::size_t batch_size,
    std::size_t sequence_length,
    const std::vector<float>& encoder_w,
    const std::vector<float>& encoder_b,
    const std::vector<float>& gru_w_input,
    const std::vector<float>& gru_b_input,
    const std::vector<float>& gru_w_hidden,
    const std::vector<float>& gru_b_hidden,
    const std::vector<float>& gru_h0,
    const std::vector<float>& actor_w,
    const std::vector<float>& actor_b,
    const std::vector<float>& policy_input,
    const std::vector<float>& observation,
    std::vector<float>& raw_action,
    std::vector<float>& bounded_action,
    std::vector<float>& hidden,
    float action_bound
){
    raw_action.assign(sequence_length * batch_size * RDAC_ACTION_DIM, 0.0f);
    bounded_action.assign(sequence_length * batch_size * RDAC_ACTION_DIM, 0.0f);
    hidden.assign(sequence_length * batch_size * RDAC_HIDDEN_DIM, 0.0f);
    for(std::size_t t = 0; t < sequence_length; t++){
        for(std::size_t b = 0; b < batch_size; b++){
            float encoder[RDAC_HIDDEN_DIM];
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                float value = encoder_b[h];
                for(std::size_t k = 0; k < RDAC_POLICY_INPUT_DIM; k++){
                    value += encoder_w[h * RDAC_POLICY_INPUT_DIM + k] *
                        policy_input[actor_idx3(t, b, k, batch_size, RDAC_POLICY_INPUT_DIM)];
                }
                encoder[h] = value > 0.0f ? value : 0.0f;
            }
            float previous_hidden[RDAC_HIDDEN_DIM];
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                previous_hidden[h] = t == 0
                    ? gru_h0[h]
                    : hidden[actor_idx3(t - 1, b, h, batch_size, RDAC_HIDDEN_DIM)];
            }
            float hidden_affine[3 * RDAC_HIDDEN_DIM];
            float gate[3 * RDAC_HIDDEN_DIM];
            for(std::size_t g = 0; g < 3 * RDAC_HIDDEN_DIM; g++){
                float h_value = gru_b_hidden[g];
                for(std::size_t k = 0; k < RDAC_HIDDEN_DIM; k++){
                    h_value += gru_w_hidden[g * RDAC_HIDDEN_DIM + k] * previous_hidden[k];
                }
                hidden_affine[g] = h_value;
                float value = h_value + gru_b_input[g];
                for(std::size_t k = 0; k < RDAC_HIDDEN_DIM; k++){
                    value += gru_w_input[g * RDAC_HIDDEN_DIM + k] * encoder[k];
                }
                gate[g] = value;
            }
            float next_hidden[RDAC_HIDDEN_DIM];
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                const float r = cpu_rdac_sigmoid(gate[h]);
                const float z = cpu_rdac_sigmoid(gate[RDAC_HIDDEN_DIM + h]);
                const float n_pre = gate[2 * RDAC_HIDDEN_DIM + h] + r * hidden_affine[2 * RDAC_HIDDEN_DIM + h];
                const float n = std::tanh(n_pre);
                next_hidden[h] = (1.0f - z) * n + z * previous_hidden[h];
                hidden[actor_idx3(t, b, h, batch_size, RDAC_HIDDEN_DIM)] = next_hidden[h];
            }
            float actor_input[RDAC_ACTOR_HEAD_INPUT_DIM];
            for(std::size_t o = 0; o < EULER_OBSERVATION_DIM; o++){
                actor_input[o] = observation[actor_idx3(t, b, o, batch_size, EULER_OBSERVATION_DIM)];
            }
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                actor_input[EULER_OBSERVATION_DIM + h] = next_hidden[h];
            }
            for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                float raw = actor_b[a];
                for(std::size_t k = 0; k < RDAC_ACTOR_HEAD_INPUT_DIM; k++){
                    raw += actor_w[a * RDAC_ACTOR_HEAD_INPUT_DIM + k] * actor_input[k];
                }
                raw_action[actor_idx3(t, b, a, batch_size, RDAC_ACTION_DIM)] = raw;
                bounded_action[actor_idx3(t, b, a, batch_size, RDAC_ACTION_DIM)] =
                    std::max(-action_bound, std::min(action_bound, raw));
            }
        }
    }
}

void cpu_actor_forward_step(
    const ActorWeightsHost& weights,
    const float policy_input[RDAC_POLICY_INPUT_DIM],
    const float observation[EULER_OBSERVATION_DIM],
    const float previous_hidden[RDAC_HIDDEN_DIM],
    float raw_action[RDAC_ACTION_DIM],
    float bounded_action[RDAC_ACTION_DIM],
    float next_hidden[RDAC_HIDDEN_DIM],
    float action_bound
){
    float encoder[RDAC_HIDDEN_DIM];
    for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
        float value = weights.encoder_b[h];
        for(std::size_t k = 0; k < RDAC_POLICY_INPUT_DIM; k++){
            value += weights.encoder_w[h * RDAC_POLICY_INPUT_DIM + k] * policy_input[k];
        }
        encoder[h] = value > 0.0f ? value : 0.0f;
    }
    float hidden_affine[3 * RDAC_HIDDEN_DIM];
    float gate[3 * RDAC_HIDDEN_DIM];
    for(std::size_t g = 0; g < 3 * RDAC_HIDDEN_DIM; g++){
        float h_value = weights.gru_b_hidden[g];
        for(std::size_t k = 0; k < RDAC_HIDDEN_DIM; k++){
            h_value += weights.gru_w_hidden[g * RDAC_HIDDEN_DIM + k] * previous_hidden[k];
        }
        hidden_affine[g] = h_value;
        float value = h_value + weights.gru_b_input[g];
        for(std::size_t k = 0; k < RDAC_HIDDEN_DIM; k++){
            value += weights.gru_w_input[g * RDAC_HIDDEN_DIM + k] * encoder[k];
        }
        gate[g] = value;
    }
    for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
        const float r = cpu_rdac_sigmoid(gate[h]);
        const float z = cpu_rdac_sigmoid(gate[RDAC_HIDDEN_DIM + h]);
        const float n_pre = gate[2 * RDAC_HIDDEN_DIM + h] + r * hidden_affine[2 * RDAC_HIDDEN_DIM + h];
        const float n = std::tanh(n_pre);
        next_hidden[h] = (1.0f - z) * n + z * previous_hidden[h];
    }
    float actor_input[RDAC_ACTOR_HEAD_INPUT_DIM];
    for(std::size_t o = 0; o < EULER_OBSERVATION_DIM; o++){
        actor_input[o] = observation[o];
    }
    for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
        actor_input[EULER_OBSERVATION_DIM + h] = next_hidden[h];
    }
    for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
        float raw = weights.actor_b[a];
        for(std::size_t k = 0; k < RDAC_ACTOR_HEAD_INPUT_DIM; k++){
            raw += weights.actor_w[a * RDAC_ACTOR_HEAD_INPUT_DIM + k] * actor_input[k];
        }
        raw_action[a] = raw;
        bounded_action[a] = std::max(-action_bound, std::min(action_bound, raw));
    }
}

void zero_actor_weights_host(ActorWeightsHost& values){
    values.resize();
    for(auto* vec: {
        &values.encoder_w, &values.encoder_b, &values.gru_w_input, &values.gru_b_input,
        &values.gru_w_hidden, &values.gru_b_hidden, &values.gru_h0, &values.actor_w, &values.actor_b,
        &values.critic_w, &values.critic_b
    }){
        std::fill(vec->begin(), vec->end(), 0.0f);
    }
}

void scale_actor_weights_host(ActorWeightsHost& values, float scale){
    for(auto* vec: {
        &values.encoder_w, &values.encoder_b, &values.gru_w_input, &values.gru_b_input,
        &values.gru_w_hidden, &values.gru_b_hidden, &values.gru_h0, &values.actor_w, &values.actor_b,
        &values.critic_w, &values.critic_b
    }){
        for(float& value: *vec){
            value *= scale;
        }
    }
}

void cpu_actor_backward_sequence(
    const ActorWeightsHost& weights,
    std::size_t batch_size,
    std::size_t horizon,
    const std::vector<float>& policy_input,
    const std::vector<float>& observation,
    const std::vector<float>& hidden,
    const std::vector<float>& raw_action_gradient,
    ActorWeightsHost& gradients,
    const std::vector<float>* critic_output_gradient = nullptr
){
    zero_actor_weights_host(gradients);
    std::vector<float> hidden_adjoint(batch_size * RDAC_HIDDEN_DIM, 0.0f);
    for(std::size_t reverse_i = 0; reverse_i < horizon; reverse_i++){
        const std::size_t step_i = horizon - 1 - reverse_i;
        for(std::size_t b = 0; b < batch_size; b++){
            float actor_input[RDAC_ACTOR_HEAD_INPUT_DIM];
            for(std::size_t o = 0; o < EULER_OBSERVATION_DIM; o++){
                actor_input[o] = observation[actor_idx3(step_i, b, o, batch_size, EULER_OBSERVATION_DIM)];
            }
            float h_next[RDAC_HIDDEN_DIM];
            float h_prev[RDAC_HIDDEN_DIM];
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                h_next[h] = hidden[actor_idx3(step_i, b, h, batch_size, RDAC_HIDDEN_DIM)];
                h_prev[h] = step_i == 0
                    ? weights.gru_h0[h]
                    : hidden[actor_idx3(step_i - 1, b, h, batch_size, RDAC_HIDDEN_DIM)];
                actor_input[EULER_OBSERVATION_DIM + h] = h_next[h];
            }

            float d_actor_input[RDAC_ACTOR_HEAD_INPUT_DIM] = {};
            float d_critic_input[RDAC_CRITIC_HEAD_INPUT_DIM] = {};
            if(critic_output_gradient != nullptr){
                for(std::size_t c = 0; c < RDAC_CRITIC_DIM; c++){
                    const float d_q = (*critic_output_gradient)[actor_idx3(step_i, b, c, batch_size, RDAC_CRITIC_DIM)];
                    gradients.critic_b[c] += d_q;
                    for(std::size_t k = 0; k < RDAC_CRITIC_HEAD_INPUT_DIM; k++){
                        gradients.critic_w[c * RDAC_CRITIC_HEAD_INPUT_DIM + k] += d_q * actor_input[k];
                        d_critic_input[k] += weights.critic_w[c * RDAC_CRITIC_HEAD_INPUT_DIM + k] * d_q;
                    }
                }
            }
            for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                const float d_raw = raw_action_gradient[actor_idx3(step_i, b, a, batch_size, RDAC_ACTION_DIM)];
                gradients.actor_b[a] += d_raw;
                for(std::size_t k = 0; k < RDAC_ACTOR_HEAD_INPUT_DIM; k++){
                    gradients.actor_w[a * RDAC_ACTOR_HEAD_INPUT_DIM + k] += d_raw * actor_input[k];
                    d_actor_input[k] += weights.actor_w[a * RDAC_ACTOR_HEAD_INPUT_DIM + k] * d_raw;
                }
            }

            float d_h_next[RDAC_HIDDEN_DIM];
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                d_h_next[h] = hidden_adjoint[b * RDAC_HIDDEN_DIM + h] +
                    d_actor_input[EULER_OBSERVATION_DIM + h] +
                    d_critic_input[EULER_OBSERVATION_DIM + h];
            }

            float encoder[RDAC_HIDDEN_DIM];
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                float value = weights.encoder_b[h];
                for(std::size_t k = 0; k < RDAC_POLICY_INPUT_DIM; k++){
                    value += weights.encoder_w[h * RDAC_POLICY_INPUT_DIM + k] *
                        policy_input[actor_idx3(step_i, b, k, batch_size, RDAC_POLICY_INPUT_DIM)];
                }
                encoder[h] = value > 0.0f ? value : 0.0f;
            }

            float hidden_affine[3 * RDAC_HIDDEN_DIM];
            float gate[3 * RDAC_HIDDEN_DIM];
            for(std::size_t g = 0; g < 3 * RDAC_HIDDEN_DIM; g++){
                float h_value = weights.gru_b_hidden[g];
                for(std::size_t k = 0; k < RDAC_HIDDEN_DIM; k++){
                    h_value += weights.gru_w_hidden[g * RDAC_HIDDEN_DIM + k] * h_prev[k];
                }
                hidden_affine[g] = h_value;
                float value = h_value + weights.gru_b_input[g];
                for(std::size_t k = 0; k < RDAC_HIDDEN_DIM; k++){
                    value += weights.gru_w_input[g * RDAC_HIDDEN_DIM + k] * encoder[k];
                }
                gate[g] = value;
            }

            float d_gate[3 * RDAC_HIDDEN_DIM] = {};
            float d_hidden_affine[3 * RDAC_HIDDEN_DIM] = {};
            float d_h_prev[RDAC_HIDDEN_DIM] = {};
            float d_encoder[RDAC_HIDDEN_DIM] = {};

            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                const float r = cpu_rdac_sigmoid(gate[h]);
                const float z = cpu_rdac_sigmoid(gate[RDAC_HIDDEN_DIM + h]);
                const float n_pre = gate[2 * RDAC_HIDDEN_DIM + h] + r * hidden_affine[2 * RDAC_HIDDEN_DIM + h];
                const float n = std::tanh(n_pre);
                const float dh = d_h_next[h];
                const float d_z = dh * (h_prev[h] - n);
                const float d_n = dh * (1.0f - z);
                d_h_prev[h] += dh * z;
                const float d_n_pre = d_n * (1.0f - n * n);
                d_gate[2 * RDAC_HIDDEN_DIM + h] += d_n_pre;
                d_hidden_affine[2 * RDAC_HIDDEN_DIM + h] += d_n_pre * r;
                const float d_r = d_n_pre * hidden_affine[2 * RDAC_HIDDEN_DIM + h];
                d_gate[h] += d_r * r * (1.0f - r);
                d_gate[RDAC_HIDDEN_DIM + h] += d_z * z * (1.0f - z);
            }

            for(std::size_t g = 0; g < 3 * RDAC_HIDDEN_DIM; g++){
                d_hidden_affine[g] += d_gate[g];
                gradients.gru_b_input[g] += d_gate[g];
                gradients.gru_b_hidden[g] += d_hidden_affine[g];
                for(std::size_t k = 0; k < RDAC_HIDDEN_DIM; k++){
                    gradients.gru_w_input[g * RDAC_HIDDEN_DIM + k] += d_gate[g] * encoder[k];
                    gradients.gru_w_hidden[g * RDAC_HIDDEN_DIM + k] += d_hidden_affine[g] * h_prev[k];
                    d_encoder[k] += weights.gru_w_input[g * RDAC_HIDDEN_DIM + k] * d_gate[g];
                    d_h_prev[k] += weights.gru_w_hidden[g * RDAC_HIDDEN_DIM + k] * d_hidden_affine[g];
                }
            }

            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                const float d_encoder_pre = encoder[h] > 0.0f ? d_encoder[h] : 0.0f;
                gradients.encoder_b[h] += d_encoder_pre;
                for(std::size_t k = 0; k < RDAC_POLICY_INPUT_DIM; k++){
                    gradients.encoder_w[h * RDAC_POLICY_INPUT_DIM + k] += d_encoder_pre *
                        policy_input[actor_idx3(step_i, b, k, batch_size, RDAC_POLICY_INPUT_DIM)];
                }
            }

            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                if(step_i == 0){
                    gradients.gru_h0[h] += d_h_prev[h];
                }
                else{
                    hidden_adjoint[b * RDAC_HIDDEN_DIM + h] = d_h_prev[h];
                }
            }
        }
    }
}

float actor_weights_l2_norm(const ActorWeightsHost& values){
    double sum = 0.0;
    for(const auto* vec: {
        &values.encoder_w, &values.encoder_b, &values.gru_w_input, &values.gru_b_input,
        &values.gru_w_hidden, &values.gru_b_hidden, &values.gru_h0, &values.actor_w, &values.actor_b,
        &values.critic_w, &values.critic_b
    }){
        for(float value: *vec){
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
    }
    return static_cast<float>(std::sqrt(sum));
}

bool actor_weights_finite(const ActorWeightsHost& values){
    for(const auto* vec: {
        &values.encoder_w, &values.encoder_b, &values.gru_w_input, &values.gru_b_input,
        &values.gru_w_hidden, &values.gru_b_hidden, &values.gru_h0, &values.actor_w, &values.actor_b,
        &values.critic_w, &values.critic_b
    }){
        for(float value: *vec){
            if(!std::isfinite(value)){
                return false;
            }
        }
    }
    return true;
}

void write_actor_checkpoint_weights_text(std::ostream& out, const ActorWeightsHost& weights){
    auto write_matrix = [&](const char* name, std::size_t rows, std::size_t cols, const std::vector<float>& values){
        out << name << " " << rows << " " << cols << "\n";
        for(float value: values){
            out << value << "\n";
        }
    };
    auto write_tensor = [&](const char* name, const std::vector<float>& values){
        out << name << " " << values.size() << "\n";
        for(float value: values){
            out << value << "\n";
        }
    };
    write_matrix("input_dense_weights", RDAC_HIDDEN_DIM, RDAC_POLICY_INPUT_DIM, weights.encoder_w);
    write_matrix("input_dense_biases", 1, RDAC_HIDDEN_DIM, weights.encoder_b);
    write_tensor("gru_weights_input", weights.gru_w_input);
    write_tensor("gru_biases_input", weights.gru_b_input);
    write_tensor("gru_weights_hidden", weights.gru_w_hidden);
    write_tensor("gru_biases_hidden", weights.gru_b_hidden);
    write_tensor("gru_initial_hidden_state", weights.gru_h0);
    write_matrix("actor_head_weights", RDAC_ACTION_DIM, RDAC_ACTOR_HEAD_INPUT_DIM, weights.actor_w);
    write_matrix("actor_head_biases", 1, RDAC_ACTION_DIM, weights.actor_b);
    write_matrix("critic_head_weights", RDAC_CRITIC_DIM, RDAC_CRITIC_HEAD_INPUT_DIM, weights.critic_w);
    write_matrix("critic_head_biases", 1, RDAC_CRITIC_DIM, weights.critic_b);
}

void write_actor_checkpoint_moments_text(std::ostream& out, const ActorWeightsHost& first_moment, const ActorWeightsHost& second_moment){
    auto write_matrix = [&](const char* name, std::size_t rows, std::size_t cols, const std::vector<float>& values){
        out << name << " " << rows << " " << cols << "\n";
        for(float value: values){
            out << value << "\n";
        }
    };
    auto write_tensor = [&](const char* name, const std::vector<float>& values){
        out << name << " " << values.size() << "\n";
        for(float value: values){
            out << value << "\n";
        }
    };
    write_matrix("input_dense_weights_adam_m", RDAC_HIDDEN_DIM, RDAC_POLICY_INPUT_DIM, first_moment.encoder_w);
    write_matrix("input_dense_weights_adam_v", RDAC_HIDDEN_DIM, RDAC_POLICY_INPUT_DIM, second_moment.encoder_w);
    write_matrix("input_dense_biases_adam_m", 1, RDAC_HIDDEN_DIM, first_moment.encoder_b);
    write_matrix("input_dense_biases_adam_v", 1, RDAC_HIDDEN_DIM, second_moment.encoder_b);
    write_tensor("gru_weights_input_adam_m", first_moment.gru_w_input);
    write_tensor("gru_weights_input_adam_v", second_moment.gru_w_input);
    write_tensor("gru_biases_input_adam_m", first_moment.gru_b_input);
    write_tensor("gru_biases_input_adam_v", second_moment.gru_b_input);
    write_tensor("gru_weights_hidden_adam_m", first_moment.gru_w_hidden);
    write_tensor("gru_weights_hidden_adam_v", second_moment.gru_w_hidden);
    write_tensor("gru_biases_hidden_adam_m", first_moment.gru_b_hidden);
    write_tensor("gru_biases_hidden_adam_v", second_moment.gru_b_hidden);
    write_tensor("gru_initial_hidden_state_adam_m", first_moment.gru_h0);
    write_tensor("gru_initial_hidden_state_adam_v", second_moment.gru_h0);
    write_matrix("actor_head_weights_adam_m", RDAC_ACTION_DIM, RDAC_ACTOR_HEAD_INPUT_DIM, first_moment.actor_w);
    write_matrix("actor_head_weights_adam_v", RDAC_ACTION_DIM, RDAC_ACTOR_HEAD_INPUT_DIM, second_moment.actor_w);
    write_matrix("actor_head_biases_adam_m", 1, RDAC_ACTION_DIM, first_moment.actor_b);
    write_matrix("actor_head_biases_adam_v", 1, RDAC_ACTION_DIM, second_moment.actor_b);
    write_matrix("critic_head_weights_adam_m", RDAC_CRITIC_DIM, RDAC_CRITIC_HEAD_INPUT_DIM, first_moment.critic_w);
    write_matrix("critic_head_weights_adam_v", RDAC_CRITIC_DIM, RDAC_CRITIC_HEAD_INPUT_DIM, second_moment.critic_w);
    write_matrix("critic_head_biases_adam_m", 1, RDAC_CRITIC_DIM, first_moment.critic_b);
    write_matrix("critic_head_biases_adam_v", 1, RDAC_CRITIC_DIM, second_moment.critic_b);
}

bool save_actor_checkpoint(
    const std::string& path,
    const ActorWeightsHost& weights,
    const ActorWeightsHost& first_moment,
    const ActorWeightsHost& second_moment,
    std::size_t optimizer_age
){
    if(path.empty()){
        return false;
    }
    std::ofstream out(path);
    if(!out){
        return false;
    }
    out << "foundation_policy_diff_pre_training_rdac_hidden_actor_v4\n";
    out << std::setprecision(10);
    out << "metadata 9\n";
    out << "format_version 4\n";
    out << "has_optimizer 1\n";
    out << "optimizer_age " << optimizer_age << "\n";
    out << "scalar_type float32\n";
    out << "actor_arch rdac_hidden_gru\n";
    out << "input_dim " << RDAC_POLICY_INPUT_DIM << "\n";
    out << "hidden_dim " << RDAC_HIDDEN_DIM << "\n";
    out << "action_dim " << RDAC_ACTION_DIM << "\n";
    out << "critic_dim " << RDAC_CRITIC_DIM << "\n";
    out << "end_metadata\n";
    write_actor_checkpoint_weights_text(out, weights);
    write_actor_checkpoint_moments_text(out, first_moment, second_moment);
    return static_cast<bool>(out);
}

bool save_actor_checkpoint(const std::string& path, const ActorWeightsHost& weights){
    ActorWeightsHost first_moment;
    ActorWeightsHost second_moment;
    zero_actor_weights_host(first_moment);
    zero_actor_weights_host(second_moment);
    return save_actor_checkpoint(path, weights, first_moment, second_moment, 1);
}

bool load_actor_checkpoint(const std::string& path, ActorWeightsHost& weights){
    if(path.empty()){
        return false;
    }
    {
        std::ifstream in(path);
        if(in){
            std::string magic_text;
            in >> magic_text;
            if(magic_text == "foundation_policy_diff_pre_training_rdac_hidden_actor_v3" ||
               magic_text == "foundation_policy_diff_pre_training_rdac_hidden_actor_v4"){
                if(magic_text == "foundation_policy_diff_pre_training_rdac_hidden_actor_v4"){
                    std::string metadata_label;
                    std::size_t metadata_count = 0;
                    in >> metadata_label >> metadata_count;
                    if(metadata_label != "metadata"){
                        in.setstate(std::ios::failbit);
                        return false;
                    }
                    for(std::size_t i = 0; i < metadata_count; i++){
                        std::string key;
                        std::string value;
                        in >> key >> value;
                    }
                    std::string end_label;
                    in >> end_label;
                    if(end_label != "end_metadata"){
                        in.setstate(std::ios::failbit);
                        return false;
                    }
                }
                weights.resize();
                auto read_matrix = [&](const char* expected_name, std::size_t expected_rows, std::size_t expected_cols, std::vector<float>& values){
                    std::string name;
                    std::size_t rows = 0;
                    std::size_t cols = 0;
                    in >> name >> rows >> cols;
                    if(name != expected_name || rows != expected_rows || cols != expected_cols || values.size() != rows * cols){
                        in.setstate(std::ios::failbit);
                        return;
                    }
                    for(float& value: values){
                        in >> value;
                    }
                };
                auto read_tensor = [&](const char* expected_name, std::vector<float>& values){
                    std::string name;
                    std::size_t size = 0;
                    in >> name >> size;
                    if(name != expected_name || size != values.size()){
                        in.setstate(std::ios::failbit);
                        return;
                    }
                    for(float& value: values){
                        in >> value;
                    }
                };
                read_matrix("input_dense_weights", RDAC_HIDDEN_DIM, RDAC_POLICY_INPUT_DIM, weights.encoder_w);
                read_matrix("input_dense_biases", 1, RDAC_HIDDEN_DIM, weights.encoder_b);
                read_tensor("gru_weights_input", weights.gru_w_input);
                read_tensor("gru_biases_input", weights.gru_b_input);
                read_tensor("gru_weights_hidden", weights.gru_w_hidden);
                read_tensor("gru_biases_hidden", weights.gru_b_hidden);
                read_tensor("gru_initial_hidden_state", weights.gru_h0);
                read_matrix("actor_head_weights", RDAC_ACTION_DIM, RDAC_ACTOR_HEAD_INPUT_DIM, weights.actor_w);
                read_matrix("actor_head_biases", 1, RDAC_ACTION_DIM, weights.actor_b);
                read_matrix("critic_head_weights", RDAC_CRITIC_DIM, RDAC_CRITIC_HEAD_INPUT_DIM, weights.critic_w);
                read_matrix("critic_head_biases", 1, RDAC_CRITIC_DIM, weights.critic_b);
                return static_cast<bool>(in) && actor_weights_finite(weights);
            }
        }
    }

    std::ifstream in(path, std::ios::binary);
    if(!in){
        return false;
    }
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if(magic != 0x52444143u || (version != 1u && version != 2u)){
        return false;
    }
    weights.resize();
    auto read_vector = [&](std::vector<float>& values){
        std::uint64_t size = 0;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        if(size != values.size()){
            in.setstate(std::ios::failbit);
            return;
        }
        in.read(reinterpret_cast<char*>(values.data()), sizeof(float) * values.size());
    };
    read_vector(weights.encoder_w);
    read_vector(weights.encoder_b);
    read_vector(weights.gru_w_input);
    read_vector(weights.gru_b_input);
    read_vector(weights.gru_w_hidden);
    read_vector(weights.gru_b_hidden);
    read_vector(weights.gru_h0);
    read_vector(weights.actor_w);
    read_vector(weights.actor_b);
    if(version >= 2u){
        read_vector(weights.critic_w);
        read_vector(weights.critic_b);
    }
    else{
        std::fill(weights.critic_w.begin(), weights.critic_w.end(), 0.0f);
        std::fill(weights.critic_b.begin(), weights.critic_b.end(), 0.0f);
    }
    return static_cast<bool>(in) && actor_weights_finite(weights);
}

bool load_actor_checkpoint_with_moments(
    const std::string& path,
    ActorWeightsHost& weights,
    ActorWeightsHost& first_moment,
    ActorWeightsHost& second_moment,
    std::uint32_t& optimizer_age
){
    std::ifstream in(path);
    if(!in){
        return false;
    }
    std::string magic_text;
    in >> magic_text;
    if(magic_text != "foundation_policy_diff_pre_training_rdac_hidden_actor_v4"){
        return false;
    }
    std::string metadata_label;
    std::size_t metadata_count = 0;
    in >> metadata_label >> metadata_count;
    if(metadata_label != "metadata"){
        return false;
    }
    bool has_optimizer = false;
    for(std::size_t i = 0; i < metadata_count; i++){
        std::string key;
        std::string value;
        in >> key >> value;
        if(key == "optimizer_age"){
            optimizer_age = static_cast<std::uint32_t>(std::stoul(value));
        }
        else if(key == "has_optimizer"){
            has_optimizer = std::stoul(value) != 0;
        }
    }
    std::string end_label;
    in >> end_label;
    if(end_label != "end_metadata" || !has_optimizer){
        return false;
    }
    weights.resize();
    first_moment.resize();
    second_moment.resize();
    auto read_matrix = [&](const char* expected_name, std::size_t expected_rows, std::size_t expected_cols, std::vector<float>& values){
        std::string name;
        std::size_t rows = 0;
        std::size_t cols = 0;
        in >> name >> rows >> cols;
        if(name != expected_name || rows != expected_rows || cols != expected_cols || values.size() != rows * cols){
            in.setstate(std::ios::failbit);
            return;
        }
        for(float& value: values){
            in >> value;
        }
    };
    auto read_tensor = [&](const char* expected_name, std::vector<float>& values){
        std::string name;
        std::size_t size = 0;
        in >> name >> size;
        if(name != expected_name || size != values.size()){
            in.setstate(std::ios::failbit);
            return;
        }
        for(float& value: values){
            in >> value;
        }
    };
    read_matrix("input_dense_weights", RDAC_HIDDEN_DIM, RDAC_POLICY_INPUT_DIM, weights.encoder_w);
    read_matrix("input_dense_biases", 1, RDAC_HIDDEN_DIM, weights.encoder_b);
    read_tensor("gru_weights_input", weights.gru_w_input);
    read_tensor("gru_biases_input", weights.gru_b_input);
    read_tensor("gru_weights_hidden", weights.gru_w_hidden);
    read_tensor("gru_biases_hidden", weights.gru_b_hidden);
    read_tensor("gru_initial_hidden_state", weights.gru_h0);
    read_matrix("actor_head_weights", RDAC_ACTION_DIM, RDAC_ACTOR_HEAD_INPUT_DIM, weights.actor_w);
    read_matrix("actor_head_biases", 1, RDAC_ACTION_DIM, weights.actor_b);
    read_matrix("critic_head_weights", RDAC_CRITIC_DIM, RDAC_CRITIC_HEAD_INPUT_DIM, weights.critic_w);
    read_matrix("critic_head_biases", 1, RDAC_CRITIC_DIM, weights.critic_b);

    read_matrix("input_dense_weights_adam_m", RDAC_HIDDEN_DIM, RDAC_POLICY_INPUT_DIM, first_moment.encoder_w);
    read_matrix("input_dense_weights_adam_v", RDAC_HIDDEN_DIM, RDAC_POLICY_INPUT_DIM, second_moment.encoder_w);
    read_matrix("input_dense_biases_adam_m", 1, RDAC_HIDDEN_DIM, first_moment.encoder_b);
    read_matrix("input_dense_biases_adam_v", 1, RDAC_HIDDEN_DIM, second_moment.encoder_b);
    read_tensor("gru_weights_input_adam_m", first_moment.gru_w_input);
    read_tensor("gru_weights_input_adam_v", second_moment.gru_w_input);
    read_tensor("gru_biases_input_adam_m", first_moment.gru_b_input);
    read_tensor("gru_biases_input_adam_v", second_moment.gru_b_input);
    read_tensor("gru_weights_hidden_adam_m", first_moment.gru_w_hidden);
    read_tensor("gru_weights_hidden_adam_v", second_moment.gru_w_hidden);
    read_tensor("gru_biases_hidden_adam_m", first_moment.gru_b_hidden);
    read_tensor("gru_biases_hidden_adam_v", second_moment.gru_b_hidden);
    read_tensor("gru_initial_hidden_state_adam_m", first_moment.gru_h0);
    read_tensor("gru_initial_hidden_state_adam_v", second_moment.gru_h0);
    read_matrix("actor_head_weights_adam_m", RDAC_ACTION_DIM, RDAC_ACTOR_HEAD_INPUT_DIM, first_moment.actor_w);
    read_matrix("actor_head_weights_adam_v", RDAC_ACTION_DIM, RDAC_ACTOR_HEAD_INPUT_DIM, second_moment.actor_w);
    read_matrix("actor_head_biases_adam_m", 1, RDAC_ACTION_DIM, first_moment.actor_b);
    read_matrix("actor_head_biases_adam_v", 1, RDAC_ACTION_DIM, second_moment.actor_b);
    read_matrix("critic_head_weights_adam_m", RDAC_CRITIC_DIM, RDAC_CRITIC_HEAD_INPUT_DIM, first_moment.critic_w);
    read_matrix("critic_head_weights_adam_v", RDAC_CRITIC_DIM, RDAC_CRITIC_HEAD_INPUT_DIM, second_moment.critic_w);
    read_matrix("critic_head_biases_adam_m", 1, RDAC_CRITIC_DIM, first_moment.critic_b);
    read_matrix("critic_head_biases_adam_v", 1, RDAC_CRITIC_DIM, second_moment.critic_b);
    return static_cast<bool>(in) &&
        actor_weights_finite(weights) &&
        actor_weights_finite(first_moment) &&
        actor_weights_finite(second_moment);
}

struct CriticLossMetrics{
    float raw_loss = 0.0f;
    float weight = RDAC_CRITIC_LOSS_WEIGHT;
    float scaled_loss = 0.0f;
    float output_mean = 0.0f;
    float target_mean = 0.0f;
    float error_mean = 0.0f;
    float error_norm = 0.0f;
    bool finite = true;
};

CriticLossMetrics compute_critic_targets_and_gradients(
    const EulerGpuBatch& batch,
    const DeviceArrays& d,
    ActorBuffersDevice& buffers,
    const EulerGpuLossWeights& weights
){
    const std::size_t B = batch.batch_size;
    const std::size_t H = batch.horizon;
    const std::size_t state_count = (H + 1) * B;
    const std::size_t step_count = H * B;
    std::vector<float> host_p(state_count * 3, 0.0f);
    std::vector<float> host_v(state_count * 3, 0.0f);
    std::vector<float> host_R(state_count * 9, 0.0f);
    std::vector<float> host_omega(state_count * 3, 0.0f);
    std::vector<float> host_actions(step_count * RDAC_ACTION_DIM, 0.0f);
    std::vector<float> host_q(step_count * RDAC_CRITIC_DIM, 0.0f);
    CUDA_CHECK(cudaMemcpy(host_p.data(), d.p, sizeof(float) * host_p.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_v.data(), d.v, sizeof(float) * host_v.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_R.data(), d.R, sizeof(float) * host_R.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_omega.data(), d.omega, sizeof(float) * host_omega.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_actions.data(), d.actions, sizeof(float) * host_actions.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_q.data(), buffers.critic_output, sizeof(float) * host_q.size(), cudaMemcpyDeviceToHost));

    std::vector<float> returns(step_count, 0.0f);
    std::vector<float> targets(step_count * RDAC_CRITIC_DIM, 0.0f);
    std::vector<float> q_weights(step_count * RDAC_CRITIC_DIM, 0.0f);
    std::vector<float> q_gradients(step_count * RDAC_CRITIC_DIM, 0.0f);
    constexpr float gamma = 0.99f;
    for(std::size_t b = 0; b < B; b++){
        float running_return = 0.0f;
        for(std::size_t reverse_i = 0; reverse_i < H; reverse_i++){
            const std::size_t t = H - reverse_i - 1;
            float cost = 0.0f;
            for(int dim = 0; dim < 3; dim++){
                const float e_p = host_p[idx3(t + 1, b, dim, B)] - batch.reference_p[pidx3(b, dim)];
                const float e_v = host_v[idx3(t + 1, b, dim, B)] - batch.reference_v[pidx3(b, dim)];
                cost += weights.position * e_p * e_p;
                cost += weights.velocity * e_v * e_v;
                cost += weights.angular_velocity * host_omega[idx3(t + 1, b, dim, B)] * host_omega[idx3(t + 1, b, dim, B)];
            }
            for(int row = 0; row < 3; row++){
                for(int col = 0; col < 3; col++){
                    const float target = row == col ? 1.0f : 0.0f;
                    const float e_R = host_R[idx9(t + 1, b, row * 3 + col, B)] - target;
                    cost += weights.attitude * e_R * e_R;
                }
            }
            for(int action = 0; action < static_cast<int>(RDAC_ACTION_DIM); action++){
                const float u = host_actions[idx4(t, b, action, B)];
                cost += weights.action_magnitude * u * u;
            }
            running_return = -cost + gamma * running_return;
            returns[t * B + b] = running_return;
        }
    }

    double mean_return = 0.0;
    for(float value: returns){
        mean_return += static_cast<double>(value);
    }
    const double count = static_cast<double>(std::max<std::size_t>(1, returns.size()));
    mean_return /= count;
    double variance_return = 0.0;
    for(float value: returns){
        const double centered = static_cast<double>(value) - mean_return;
        variance_return += centered * centered;
    }
    const double std_return = returns.size() > 1
        ? std::sqrt(std::max(1e-6, variance_return / static_cast<double>(returns.size() - 1)))
        : 1.0;
    CriticLossMetrics metrics;
    metrics.weight = RDAC_CRITIC_LOSS_WEIGHT;
    double output_sum = 0.0;
    double target_sum = 0.0;
    double error_sum = 0.0;
    double error_sq = 0.0;
    double raw_loss = 0.0;
    for(std::size_t t = 0; t < H; t++){
        for(std::size_t b = 0; b < B; b++){
            const std::size_t i = actor_idx3(t, b, 0, B, RDAC_CRITIC_DIM);
            const float target = static_cast<float>((static_cast<double>(returns[t * B + b]) - mean_return) / std_return);
            const float q = host_q[i];
            const float error = q - target;
            const float q_weight = batch.group_weight.size() == B
                ? batch.group_weight[b] / std::max(1.0f, static_cast<float>(H))
                : 1.0f / std::max(1.0f, static_cast<float>(step_count));
            targets[i] = target;
            q_weights[i] = q_weight;
            q_gradients[i] = RDAC_CRITIC_LOSS_WEIGHT * q_weight * error;
            raw_loss += 0.5 * static_cast<double>(q_weight) * static_cast<double>(error) * static_cast<double>(error);
            output_sum += q;
            target_sum += target;
            error_sum += error;
            error_sq += static_cast<double>(error) * static_cast<double>(error);
            metrics.finite = metrics.finite &&
                std::isfinite(q) &&
                std::isfinite(target) &&
                std::isfinite(error) &&
                std::isfinite(q_gradients[i]);
        }
    }
    const double inv_count = 1.0 / std::max(1.0, static_cast<double>(step_count));
    metrics.raw_loss = static_cast<float>(raw_loss);
    metrics.scaled_loss = static_cast<float>(RDAC_CRITIC_LOSS_WEIGHT * raw_loss);
    metrics.output_mean = static_cast<float>(output_sum * inv_count);
    metrics.target_mean = static_cast<float>(target_sum * inv_count);
    metrics.error_mean = static_cast<float>(error_sum * inv_count);
    metrics.error_norm = static_cast<float>(std::sqrt(error_sq));

    CUDA_CHECK(cudaMemcpy(buffers.critic_target, targets.data(), sizeof(float) * targets.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.critic_weight, q_weights.data(), sizeof(float) * q_weights.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.critic_output_gradient, q_gradients.data(), sizeof(float) * q_gradients.size(), cudaMemcpyHostToDevice));
    return metrics;
}

void write_float_vector(std::ostream& out, const std::vector<float>& values){
    out.write(reinterpret_cast<const char*>(values.data()), sizeof(float) * values.size());
}

void read_float_vector(std::istream& in, std::vector<float>& values){
    in.read(reinterpret_cast<char*>(values.data()), sizeof(float) * values.size());
}

void write_uint32_vector(std::ostream& out, const std::vector<std::uint32_t>& values){
    out.write(reinterpret_cast<const char*>(values.data()), sizeof(std::uint32_t) * values.size());
}

void read_uint32_vector(std::istream& in, std::vector<std::uint32_t>& values){
    in.read(reinterpret_cast<char*>(values.data()), sizeof(std::uint32_t) * values.size());
}

void write_replay_batch(std::ostream& out, const EulerGpuBatch& batch){
    write_float_vector(out, batch.initial_p);
    write_float_vector(out, batch.initial_v);
    write_float_vector(out, batch.initial_R);
    write_float_vector(out, batch.initial_omega);
    write_float_vector(out, batch.initial_rpm);
    write_float_vector(out, batch.initial_previous_action);
    write_float_vector(out, batch.reference_p);
    write_float_vector(out, batch.reference_v);
    write_float_vector(out, batch.actions);
    write_float_vector(out, batch.mass);
    write_float_vector(out, batch.gravity);
    write_float_vector(out, batch.J);
    write_float_vector(out, batch.J_inv);
    write_float_vector(out, batch.rotor_positions);
    write_float_vector(out, batch.rotor_thrust_directions);
    write_float_vector(out, batch.rotor_torque_directions);
    write_float_vector(out, batch.rotor_torque_constants);
    write_float_vector(out, batch.rotor_time_rising);
    write_float_vector(out, batch.rotor_time_falling);
    write_float_vector(out, batch.rotor_thrust_coeffs);
    write_float_vector(out, batch.action_min);
    write_float_vector(out, batch.action_max);
    write_uint32_vector(out, batch.dynamics_size_mass_bin);
    write_uint32_vector(out, batch.dynamics_thrust_to_weight_bin);
    write_uint32_vector(out, batch.dynamics_torque_to_inertia_bin);
    write_uint32_vector(out, batch.dynamics_motor_delay_bin);
    write_uint32_vector(out, batch.dynamics_curve_shape_bin);
    write_uint32_vector(out, batch.dynamics_group_key);
    write_uint32_vector(out, batch.rejected_before_accept);
    write_float_vector(out, batch.group_weight);
    write_uint32_vector(out, batch.reset_mask);
    write_uint32_vector(out, batch.hidden_reset_mask);
    out.write(reinterpret_cast<const char*>(&batch.replay_schema_version), sizeof(batch.replay_schema_version));
    out.write(reinterpret_cast<const char*>(&batch.sampler_seed), sizeof(batch.sampler_seed));
    out.write(reinterpret_cast<const char*>(&batch.sampler_balance_bins), sizeof(batch.sampler_balance_bins));
    out.write(reinterpret_cast<const char*>(&batch.hidden_reset_enabled), sizeof(batch.hidden_reset_enabled));
}

void read_replay_batch(std::istream& in, EulerGpuBatch& batch){
    read_float_vector(in, batch.initial_p);
    read_float_vector(in, batch.initial_v);
    read_float_vector(in, batch.initial_R);
    read_float_vector(in, batch.initial_omega);
    read_float_vector(in, batch.initial_rpm);
    read_float_vector(in, batch.initial_previous_action);
    read_float_vector(in, batch.reference_p);
    read_float_vector(in, batch.reference_v);
    read_float_vector(in, batch.actions);
    read_float_vector(in, batch.mass);
    read_float_vector(in, batch.gravity);
    read_float_vector(in, batch.J);
    read_float_vector(in, batch.J_inv);
    read_float_vector(in, batch.rotor_positions);
    read_float_vector(in, batch.rotor_thrust_directions);
    read_float_vector(in, batch.rotor_torque_directions);
    read_float_vector(in, batch.rotor_torque_constants);
    read_float_vector(in, batch.rotor_time_rising);
    read_float_vector(in, batch.rotor_time_falling);
    read_float_vector(in, batch.rotor_thrust_coeffs);
    read_float_vector(in, batch.action_min);
    read_float_vector(in, batch.action_max);
    read_uint32_vector(in, batch.dynamics_size_mass_bin);
    read_uint32_vector(in, batch.dynamics_thrust_to_weight_bin);
    read_uint32_vector(in, batch.dynamics_torque_to_inertia_bin);
    read_uint32_vector(in, batch.dynamics_motor_delay_bin);
    read_uint32_vector(in, batch.dynamics_curve_shape_bin);
    read_uint32_vector(in, batch.dynamics_group_key);
    read_uint32_vector(in, batch.rejected_before_accept);
    read_float_vector(in, batch.group_weight);
    read_uint32_vector(in, batch.reset_mask);
    read_uint32_vector(in, batch.hidden_reset_mask);
    in.read(reinterpret_cast<char*>(&batch.replay_schema_version), sizeof(batch.replay_schema_version));
    in.read(reinterpret_cast<char*>(&batch.sampler_seed), sizeof(batch.sampler_seed));
    in.read(reinterpret_cast<char*>(&batch.sampler_balance_bins), sizeof(batch.sampler_balance_bins));
    in.read(reinterpret_cast<char*>(&batch.hidden_reset_enabled), sizeof(batch.hidden_reset_enabled));
}

bool write_stage9_replay_file(
    const std::string& path,
    std::size_t steps,
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed
){
    if(path.empty()){
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if(!out){
        return false;
    }
    const std::uint32_t magic = 0x52394744u; // R9GD
    const std::uint32_t version = 2u;
    const std::uint64_t saved_steps = static_cast<std::uint64_t>(steps);
    const std::uint64_t saved_batch = static_cast<std::uint64_t>(batch_size);
    const std::uint64_t saved_horizon = static_cast<std::uint64_t>(horizon);
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&saved_steps), sizeof(saved_steps));
    out.write(reinterpret_cast<const char*>(&saved_batch), sizeof(saved_batch));
    out.write(reinterpret_cast<const char*>(&saved_horizon), sizeof(saved_horizon));
    EulerGpuBatch batch;
    for(std::size_t step = 0; step < steps; step++){
        generate_validation_batch(batch, batch_size, horizon, seed + static_cast<unsigned>(step));
        write_replay_batch(out, batch);
    }
    return static_cast<bool>(out);
}

bool open_stage9_replay_file(
    const std::string& path,
    std::ifstream& in,
    std::size_t expected_steps,
    std::size_t expected_batch_size,
    std::size_t expected_horizon
){
    if(path.empty()){
        return false;
    }
    in.open(path, std::ios::binary);
    if(!in){
        return false;
    }
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint64_t steps = 0;
    std::uint64_t batch_size = 0;
    std::uint64_t horizon = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&steps), sizeof(steps));
    in.read(reinterpret_cast<char*>(&batch_size), sizeof(batch_size));
    in.read(reinterpret_cast<char*>(&horizon), sizeof(horizon));
    return magic == 0x52394744u &&
        version == 2u &&
        steps == expected_steps &&
        batch_size == expected_batch_size &&
        horizon == expected_horizon &&
        static_cast<bool>(in);
}

void adam_update_host(
    ActorWeightsHost& parameters,
    const ActorWeightsHost& gradients,
    ActorWeightsHost& first_moment,
    ActorWeightsHost& second_moment,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int step
){
    auto update = [&](std::vector<float>& p, const std::vector<float>& g, std::vector<float>& m_vec, std::vector<float>& v_vec){
        for(std::size_t i = 0; i < p.size(); i++){
            const float gradient = g[i];
            const float m = beta1 * m_vec[i] + (1.0f - beta1) * gradient;
            const float v = beta2 * v_vec[i] + (1.0f - beta2) * gradient * gradient;
            m_vec[i] = m;
            v_vec[i] = v;
            const float m_hat = m / (1.0f - std::pow(beta1, static_cast<float>(step)));
            const float v_hat = v / (1.0f - std::pow(beta2, static_cast<float>(step)));
            p[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    };
    update(parameters.encoder_w, gradients.encoder_w, first_moment.encoder_w, second_moment.encoder_w);
    update(parameters.encoder_b, gradients.encoder_b, first_moment.encoder_b, second_moment.encoder_b);
    update(parameters.gru_w_input, gradients.gru_w_input, first_moment.gru_w_input, second_moment.gru_w_input);
    update(parameters.gru_b_input, gradients.gru_b_input, first_moment.gru_b_input, second_moment.gru_b_input);
    update(parameters.gru_w_hidden, gradients.gru_w_hidden, first_moment.gru_w_hidden, second_moment.gru_w_hidden);
    update(parameters.gru_b_hidden, gradients.gru_b_hidden, first_moment.gru_b_hidden, second_moment.gru_b_hidden);
    update(parameters.gru_h0, gradients.gru_h0, first_moment.gru_h0, second_moment.gru_h0);
    update(parameters.actor_w, gradients.actor_w, first_moment.actor_w, second_moment.actor_w);
    update(parameters.actor_b, gradients.actor_b, first_moment.actor_b, second_moment.actor_b);
    update(parameters.critic_w, gradients.critic_w, first_moment.critic_w, second_moment.critic_w);
    update(parameters.critic_b, gradients.critic_b, first_moment.critic_b, second_moment.critic_b);
}

float actor_weights_max_abs_error(const ActorWeightsHost& a, const ActorWeightsHost& b){
    float max_error = 0.0f;
    auto update = [&](const std::vector<float>& av, const std::vector<float>& bv){
        for(std::size_t i = 0; i < av.size(); i++){
            max_error = std::max(max_error, std::abs(av[i] - bv[i]));
        }
    };
    update(a.encoder_w, b.encoder_w);
    update(a.encoder_b, b.encoder_b);
    update(a.gru_w_input, b.gru_w_input);
    update(a.gru_b_input, b.gru_b_input);
    update(a.gru_w_hidden, b.gru_w_hidden);
    update(a.gru_b_hidden, b.gru_b_hidden);
    update(a.gru_h0, b.gru_h0);
    update(a.actor_w, b.actor_w);
    update(a.actor_b, b.actor_b);
    update(a.critic_w, b.critic_w);
    update(a.critic_b, b.critic_b);
    return max_error;
}

float actor_weights_l2_rel_error(const ActorWeightsHost& a, const ActorWeightsHost& b){
    double error_sq = 0.0;
    double reference_sq = 0.0;
    auto update = [&](const std::vector<float>& av, const std::vector<float>& bv){
        for(std::size_t i = 0; i < av.size(); i++){
            const double error = static_cast<double>(av[i]) - static_cast<double>(bv[i]);
            error_sq += error * error;
            reference_sq += static_cast<double>(bv[i]) * static_cast<double>(bv[i]);
        }
    };
    update(a.encoder_w, b.encoder_w);
    update(a.encoder_b, b.encoder_b);
    update(a.gru_w_input, b.gru_w_input);
    update(a.gru_b_input, b.gru_b_input);
    update(a.gru_w_hidden, b.gru_w_hidden);
    update(a.gru_b_hidden, b.gru_b_hidden);
    update(a.gru_h0, b.gru_h0);
    update(a.actor_w, b.actor_w);
    update(a.actor_b, b.actor_b);
    update(a.critic_w, b.critic_w);
    update(a.critic_b, b.critic_b);
    return static_cast<float>(std::sqrt(error_sq) / std::max(1e-12, std::sqrt(reference_sq)));
}

float actor_weights_diff_l2_norm(const ActorWeightsHost& a, const ActorWeightsHost& b){
    double error_sq = 0.0;
    auto update = [&](const std::vector<float>& av, const std::vector<float>& bv){
        for(std::size_t i = 0; i < av.size(); i++){
            const double error = static_cast<double>(av[i]) - static_cast<double>(bv[i]);
            error_sq += error * error;
        }
    };
    update(a.encoder_w, b.encoder_w);
    update(a.encoder_b, b.encoder_b);
    update(a.gru_w_input, b.gru_w_input);
    update(a.gru_b_input, b.gru_b_input);
    update(a.gru_w_hidden, b.gru_w_hidden);
    update(a.gru_b_hidden, b.gru_b_hidden);
    update(a.gru_h0, b.gru_h0);
    update(a.actor_w, b.actor_w);
    update(a.actor_b, b.actor_b);
    update(a.critic_w, b.critic_w);
    update(a.critic_b, b.critic_b);
    return static_cast<float>(std::sqrt(error_sq));
}

struct Stage9StepMetrics{
    float loss = 0.0f;
    float action_mean = 0.0f;
    float action_std = 0.0f;
    float action_saturation = 0.0f;
    float action_gradient_norm = 0.0f;
    float raw_action_gradient_norm = 0.0f;
    float actor_gradient_norm = 0.0f;
    float weight_delta_norm = 0.0f;
    float adam_m_norm = 0.0f;
    float adam_v_norm = 0.0f;
    bool finite = true;
};

Stage9StepMetrics cpu_stage9_replay_step(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    const Stage9ReplayDebugOptions& debug_options,
    ActorWeightsHost& actor_weights,
    ActorWeightsHost& adam_first_moment,
    ActorWeightsHost& adam_second_moment,
    std::size_t update_step
){
    namespace diff = rl_tools::rl::environments::l2f::diff;
    using TI = std::uint64_t;
    Stage9StepMetrics metrics;
    ActorWeightsHost weights_before = actor_weights;

    const std::size_t sequence_count = batch.horizon * batch.batch_size;
    std::vector<float> policy_input(sequence_count * RDAC_POLICY_INPUT_DIM, 0.0f);
    std::vector<float> observation(sequence_count * EULER_OBSERVATION_DIM, 0.0f);
    std::vector<float> hidden(sequence_count * RDAC_HIDDEN_DIM, 0.0f);
    std::vector<float> raw_action_gradient(sequence_count * RDAC_ACTION_DIM, 0.0f);
    std::vector<float> critic_output(sequence_count * RDAC_CRITIC_DIM, 0.0f);
    std::vector<float> critic_output_gradient(sequence_count * RDAC_CRITIC_DIM, 0.0f);
    std::vector<float> q_returns(sequence_count, 0.0f);

    double loss_sum = 0.0;
    double action_sum = 0.0;
    double action_sq_sum = 0.0;
    std::size_t action_count = 0;
    std::size_t saturated_count = 0;
    double action_gradient_sq = 0.0;
    double raw_action_gradient_sq = 0.0;

    diff::EulerLossWeights<float> cpu_weights{
        weights.position, weights.velocity, weights.attitude, weights.angular_velocity,
        weights.action_magnitude, weights.action_smoothness, weights.saturation,
        weights.terminal_loss_weight, weights.terminal_position, weights.terminal_velocity,
        weights.terminal_attitude, weights.terminal_angular_velocity
    };

    for(std::size_t b = 0; b < batch.batch_size; b++){
        CpuSimpleParameters parameters = cpu_parameters_from_batch(batch, b);
        diff::EulerState<float, TI> state{};
        for(int i = 0; i < 3; i++){
            state.p[i] = batch.initial_p[pidx3(b, i)];
            state.v[i] = batch.initial_v[pidx3(b, i)];
            state.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                state.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            state.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            state.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        diff::TrackingReference<float> ref{};
        for(int i = 0; i < 3; i++){
            ref.p[i] = batch.reference_p[pidx3(b, i)];
            ref.v[i] = batch.reference_v[pidx3(b, i)];
        }

        float previous_hidden[RDAC_HIDDEN_DIM];
        for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
            previous_hidden[h] = actor_weights.gru_h0[h];
        }
        float actions_for_loss[CPU_MAX_HORIZON][4] = {};
        float action_derivatives[CPU_MAX_HORIZON][4] = {};
        for(std::size_t t = 0; t < batch.horizon; t++){
            if(debug_options.reset_hidden_each_step){
                for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                    previous_hidden[h] = actor_weights.gru_h0[h];
                }
            }
            float obs[EULER_OBSERVATION_DIM];
            std::size_t out = 0;
            for(int i = 0; i < 3; i++){ obs[out++] = state.p[i] - ref.p[i]; }
            for(int i = 0; i < 3; i++){ for(int j = 0; j < 3; j++){ obs[out++] = state.R[i][j]; } }
            for(int i = 0; i < 3; i++){ obs[out++] = state.v[i] - ref.v[i]; }
            for(int i = 0; i < 3; i++){ obs[out++] = state.omega[i]; }
            for(int i = 0; i < 4; i++){ obs[out++] = state.previous_action[i]; }
            float input[RDAC_POLICY_INPUT_DIM] = {};
            for(std::size_t i = 0; i < EULER_OBSERVATION_DIM; i++){
                input[i] = obs[i];
                observation[actor_idx3(t, b, i, batch.batch_size, EULER_OBSERVATION_DIM)] = obs[i];
            }
            for(std::size_t i = 0; i < RDAC_ACTION_DIM; i++){
                input[EULER_OBSERVATION_DIM + i] = obs[18 + i];
            }
            for(std::size_t i = 0; i < RDAC_POLICY_INPUT_DIM; i++){
                policy_input[actor_idx3(t, b, i, batch.batch_size, RDAC_POLICY_INPUT_DIM)] = input[i];
            }
            float raw_action[RDAC_ACTION_DIM];
            float bounded_action[RDAC_ACTION_DIM];
            float next_hidden[RDAC_HIDDEN_DIM];
            cpu_actor_forward_step(actor_weights, input, obs, previous_hidden, raw_action, bounded_action, next_hidden, 1.0f);
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                hidden[actor_idx3(t, b, h, batch.batch_size, RDAC_HIDDEN_DIM)] = next_hidden[h];
                previous_hidden[h] = next_hidden[h];
            }
            float critic_input[RDAC_CRITIC_HEAD_INPUT_DIM];
            for(std::size_t o = 0; o < EULER_OBSERVATION_DIM; o++){
                critic_input[o] = obs[o];
            }
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                critic_input[EULER_OBSERVATION_DIM + h] = next_hidden[h];
            }
            for(std::size_t c = 0; c < RDAC_CRITIC_DIM; c++){
                float q = actor_weights.critic_b[c];
                for(std::size_t k = 0; k < RDAC_CRITIC_HEAD_INPUT_DIM; k++){
                    q += actor_weights.critic_w[c * RDAC_CRITIC_HEAD_INPUT_DIM + k] * critic_input[k];
                }
                critic_output[actor_idx3(t, b, c, batch.batch_size, RDAC_CRITIC_DIM)] = q;
            }
            for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                actions_for_loss[t][a] = bounded_action[a];
                action_derivatives[t][a] = (raw_action[a] > 1.0f || raw_action[a] < -1.0f) ? 0.0f : 1.0f;
                action_sum += bounded_action[a];
                action_sq_sum += static_cast<double>(bounded_action[a]) * static_cast<double>(bounded_action[a]);
                saturated_count += std::abs(bounded_action[a]) >= 0.95f ? 1 : 0;
                action_count++;
            }
            diff::EulerState<float, TI> next{};
            diff::EulerStepCache<float> cache{};
            diff::step<CpuSimpleParameters, float, TI>(parameters, state, bounded_action, next, cache);
            state = next;
        }

        diff::EulerState<float, TI> initial{};
        for(int i = 0; i < 3; i++){
            initial.p[i] = batch.initial_p[pidx3(b, i)];
            initial.v[i] = batch.initial_v[pidx3(b, i)];
            initial.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                initial.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            initial.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            initial.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        diff::EulerState<float, TI> states[CPU_MAX_HORIZON + 1];
        diff::EulerStepCache<float> caches[CPU_MAX_HORIZON];
        float action_gradients[CPU_MAX_HORIZON][4] = {};
        auto terms = diff::rollout_loss_and_gradients<CpuSimpleParameters, float, TI, CPU_MAX_HORIZON>(
            parameters, initial, actions_for_loss, static_cast<TI>(batch.horizon), cpu_weights, ref, states, caches, action_gradients
        );
        loss_sum += terms.total();
        float running_return = 0.0f;
        for(std::size_t reverse_i = 0; reverse_i < batch.horizon; reverse_i++){
            const std::size_t t = batch.horizon - reverse_i - 1;
            float cost = 0.0f;
            for(int dim = 0; dim < 3; dim++){
                const float e_p = states[t + 1].p[dim] - ref.p[dim];
                const float e_v = states[t + 1].v[dim] - ref.v[dim];
                cost += weights.position * e_p * e_p;
                cost += weights.velocity * e_v * e_v;
                cost += weights.angular_velocity * states[t + 1].omega[dim] * states[t + 1].omega[dim];
            }
            for(int row = 0; row < 3; row++){
                for(int col = 0; col < 3; col++){
                    const float target = row == col ? 1.0f : 0.0f;
                    const float e_R = states[t + 1].R[row][col] - target;
                    cost += weights.attitude * e_R * e_R;
                }
            }
            for(std::size_t action = 0; action < RDAC_ACTION_DIM; action++){
                cost += weights.action_magnitude * actions_for_loss[t][action] * actions_for_loss[t][action];
            }
            running_return = -cost + 0.99f * running_return;
            q_returns[t * batch.batch_size + b] = running_return;
        }
        for(std::size_t t = 0; t < batch.horizon; t++){
            for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                const float action_gradient = action_gradients[t][a];
                const float base_scale = debug_options.disable_physics_gradient
                    ? 0.0f
                    : debug_options.diff_rollout_loss_weight / std::max<float>(1.0f, static_cast<float>(batch.batch_size));
                const float raw_gradient = action_gradient * action_derivatives[t][a] * base_scale;
                action_gradient_sq += static_cast<double>(action_gradient) * static_cast<double>(action_gradient);
                raw_action_gradient_sq += static_cast<double>(raw_gradient) * static_cast<double>(raw_gradient);
                raw_action_gradient[actor_idx3(t, b, a, batch.batch_size, RDAC_ACTION_DIM)] = raw_gradient;
            }
        }
    }

    double mean_return = 0.0;
    for(float value: q_returns){
        mean_return += static_cast<double>(value);
    }
    mean_return /= std::max(1.0, static_cast<double>(q_returns.size()));
    double variance_return = 0.0;
    for(float value: q_returns){
        const double centered = static_cast<double>(value) - mean_return;
        variance_return += centered * centered;
    }
    const double std_return = q_returns.size() > 1
        ? std::sqrt(std::max(1e-6, variance_return / static_cast<double>(q_returns.size() - 1)))
        : 1.0;
    double critic_raw_loss = 0.0;
    for(std::size_t t = 0; t < batch.horizon; t++){
        for(std::size_t b = 0; b < batch.batch_size; b++){
            const std::size_t i = actor_idx3(t, b, 0, batch.batch_size, RDAC_CRITIC_DIM);
            const float target = static_cast<float>((static_cast<double>(q_returns[t * batch.batch_size + b]) - mean_return) / std_return);
            const float error = critic_output[i] - target;
            const float critic_weight = batch.group_weight.size() == batch.batch_size
                ? batch.group_weight[b] / std::max(1.0f, static_cast<float>(batch.horizon))
                : 1.0f / std::max(1.0f, static_cast<float>(sequence_count));
            critic_output_gradient[i] = RDAC_CRITIC_LOSS_WEIGHT * critic_weight * error;
            critic_raw_loss += 0.5 * static_cast<double>(critic_weight) * static_cast<double>(error) * static_cast<double>(error);
        }
    }

    const float raw_action_gradient_norm = static_cast<float>(std::sqrt(raw_action_gradient_sq));
    if(debug_options.action_grad_clip_enabled && raw_action_gradient_norm > debug_options.action_grad_clip_norm){
        const float scale = debug_options.action_grad_clip_norm / std::max(1e-12f, raw_action_gradient_norm);
        raw_action_gradient_sq = 0.0;
        for(float& value: raw_action_gradient){
            value *= scale;
            raw_action_gradient_sq += static_cast<double>(value) * static_cast<double>(value);
        }
    }

    ActorWeightsHost actor_gradients;
    cpu_actor_backward_sequence(actor_weights, batch.batch_size, batch.horizon, policy_input, observation, hidden, raw_action_gradient, actor_gradients, &critic_output_gradient);
    const float diff_loss_scaled = static_cast<float>(
        (debug_options.disable_physics_gradient ? 0.0 : debug_options.diff_rollout_loss_weight) *
        loss_sum / std::max<std::size_t>(1, batch.batch_size)
    );
    metrics.loss = diff_loss_scaled + static_cast<float>(RDAC_CRITIC_LOSS_WEIGHT * critic_raw_loss);
    metrics.action_mean = static_cast<float>(action_sum / std::max<std::size_t>(1, action_count));
    const double action_mean = metrics.action_mean;
    metrics.action_std = static_cast<float>(std::sqrt(std::max(0.0, action_sq_sum / std::max<std::size_t>(1, action_count) - action_mean * action_mean)));
    metrics.action_saturation = static_cast<float>(static_cast<double>(saturated_count) / std::max<std::size_t>(1, action_count));
    metrics.action_gradient_norm = static_cast<float>(std::sqrt(action_gradient_sq));
    metrics.raw_action_gradient_norm = static_cast<float>(std::sqrt(raw_action_gradient_sq));
    const float raw_actor_gradient_norm = actor_weights_l2_norm(actor_gradients);
    bool skip_actor_step = raw_actor_gradient_norm > debug_options.actor_grad_skip_norm;
    if(debug_options.actor_grad_clip_enabled && raw_actor_gradient_norm > debug_options.actor_grad_clip_norm){
        const float scale = debug_options.actor_grad_clip_norm / std::max(debug_options.actor_grad_eps, raw_actor_gradient_norm);
        scale_actor_weights_host(actor_gradients, scale);
    }
    metrics.actor_gradient_norm = actor_weights_l2_norm(actor_gradients);
    if(!skip_actor_step){
        adam_update_host(actor_weights, actor_gradients, adam_first_moment, adam_second_moment, debug_options.learning_rate, 0.9f, 0.999f, 1e-8f, static_cast<int>(update_step));
    }
    metrics.weight_delta_norm = actor_weights_diff_l2_norm(actor_weights, weights_before);
    metrics.adam_m_norm = actor_weights_l2_norm(adam_first_moment);
    metrics.adam_v_norm = actor_weights_l2_norm(adam_second_moment);
    metrics.finite = std::isfinite(metrics.loss) &&
        std::isfinite(metrics.action_mean) &&
        std::isfinite(metrics.action_std) &&
        std::isfinite(metrics.action_gradient_norm) &&
        std::isfinite(metrics.actor_gradient_norm) &&
        actor_weights_finite(actor_weights) &&
        actor_weights_finite(adam_first_moment) &&
        actor_weights_finite(adam_second_moment);
    return metrics;
}

Stage9StepMetrics gpu_stage9_replay_step(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    const Stage9ReplayDebugOptions& debug_options,
    DeviceArrays& d,
    ActorWeightsDevice& actor_weights,
    ActorBuffersDevice& actor_buffers,
    ActorGradientsDevice& actor_gradients,
    ActorWeightsDevice& adam_first_moment,
    ActorWeightsDevice& adam_second_moment,
    std::size_t update_step
){
    Stage9StepMetrics metrics;
    ActorWeightsHost weights_before;
    copy_actor_weights_to_host(actor_weights, weights_before);

    copy_input_to_device(batch, d);
    zero_actor_gradients(actor_gradients, batch.batch_size);
    const int block = 256;
    const int grid = static_cast<int>((batch.batch_size + block - 1) / block);
    const std::size_t state_count = (batch.horizon + 1) * batch.batch_size;
    const std::size_t step_count = batch.horizon * batch.batch_size;
    const std::size_t gradient_count = step_count * RDAC_ACTION_DIM;
    const std::size_t critic_count = step_count * RDAC_CRITIC_DIM;
    CUDA_CHECK(cudaMemset(d.lambda_p, 0, sizeof(float) * state_count * 3));
    CUDA_CHECK(cudaMemset(d.lambda_v, 0, sizeof(float) * state_count * 3));
    CUDA_CHECK(cudaMemset(d.lambda_R, 0, sizeof(float) * state_count * 9));
    CUDA_CHECK(cudaMemset(d.lambda_omega, 0, sizeof(float) * state_count * 3));
    CUDA_CHECK(cudaMemset(d.lambda_rpm, 0, sizeof(float) * state_count * 4));
    CUDA_CHECK(cudaMemset(d.action_gradients, 0, sizeof(float) * step_count * 4));
    CUDA_CHECK(cudaMemset(d.loss, 0, sizeof(float) * batch.batch_size));
    CUDA_CHECK(cudaMemset(actor_buffers.raw_action_gradient, 0, sizeof(float) * gradient_count));
    CUDA_CHECK(cudaMemset(actor_buffers.critic_output, 0, sizeof(float) * critic_count));
    CUDA_CHECK(cudaMemset(actor_buffers.critic_target, 0, sizeof(float) * critic_count));
    CUDA_CHECK(cudaMemset(actor_buffers.critic_weight, 0, sizeof(float) * critic_count));
    CUDA_CHECK(cudaMemset(actor_buffers.critic_output_gradient, 0, sizeof(float) * critic_count));

    for(std::size_t t = 0; t < batch.horizon; t++){
        observation_kernel<<<grid, block>>>(d, batch.batch_size, t);
        CUDA_CHECK(cudaGetLastError());
        build_policy_input_from_observation_kernel<<<grid, block>>>(d, actor_buffers, batch.batch_size, t);
        CUDA_CHECK(cudaGetLastError());
        rdac_actor_forward_step_kernel<<<grid, block>>>(actor_weights, actor_buffers, batch.batch_size, t, 1.0f, debug_options.reset_hidden_each_step);
        CUDA_CHECK(cudaGetLastError());
        copy_bounded_action_to_rollout_kernel<<<grid, block>>>(actor_buffers, d, batch.batch_size, t);
        CUDA_CHECK(cudaGetLastError());
        forward_step_kernel<<<grid, block>>>(d, batch.batch_size, t);
        CUDA_CHECK(cudaGetLastError());
    }
    loss_and_action_kernel<<<grid, block>>>(d, weights, batch.batch_size, batch.horizon);
    CUDA_CHECK(cudaGetLastError());
    for(std::size_t reverse_i = 0; reverse_i < batch.horizon; reverse_i++){
        const std::size_t step_i = batch.horizon - 1 - reverse_i;
        backward_step_kernel<<<grid, block>>>(d, batch.batch_size, step_i);
        CUDA_CHECK(cudaGetLastError());
    }
    const int grad_grid = static_cast<int>((gradient_count + block - 1) / block);
    inject_action_gradient_kernel<<<grad_grid, block>>>(d, actor_buffers, batch.batch_size, batch.horizon);
    CUDA_CHECK(cudaGetLastError());
    const float base_scale = debug_options.disable_physics_gradient
        ? 0.0f
        : debug_options.diff_rollout_loss_weight / std::max<float>(1.0f, static_cast<float>(batch.batch_size));
    scale_device_buffer(actor_buffers.raw_action_gradient, gradient_count, base_scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    if(debug_options.action_grad_clip_enabled){
        std::vector<float> clip_probe(gradient_count, 0.0f);
        CUDA_CHECK(cudaMemcpy(clip_probe.data(), actor_buffers.raw_action_gradient, sizeof(float) * clip_probe.size(), cudaMemcpyDeviceToHost));
        double norm_sq = 0.0;
        for(float value: clip_probe){
            norm_sq += static_cast<double>(value) * static_cast<double>(value);
        }
        const float norm = static_cast<float>(std::sqrt(norm_sq));
        if(norm > debug_options.action_grad_clip_norm){
            const float scale = debug_options.action_grad_clip_norm / std::max(1e-12f, norm);
            scale_device_buffer(actor_buffers.raw_action_gradient, gradient_count, scale);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    const CriticLossMetrics critic_metrics = compute_critic_targets_and_gradients(batch, d, actor_buffers, weights);
    for(std::size_t reverse_i = 0; reverse_i < batch.horizon; reverse_i++){
        const std::size_t step_i = batch.horizon - 1 - reverse_i;
        rdac_actor_backward_step_kernel<<<grid, block>>>(actor_weights, actor_buffers, actor_gradients, batch.batch_size, step_i);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host_loss(batch.batch_size, 0.0f);
    std::vector<float> host_actions(gradient_count, 0.0f);
    std::vector<float> host_action_gradients(gradient_count, 0.0f);
    std::vector<float> host_raw_action_gradients(gradient_count, 0.0f);
    CUDA_CHECK(cudaMemcpy(host_loss.data(), d.loss, sizeof(float) * host_loss.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_actions.data(), d.actions, sizeof(float) * host_actions.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_action_gradients.data(), d.action_gradients, sizeof(float) * host_action_gradients.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_raw_action_gradients.data(), actor_buffers.raw_action_gradient, sizeof(float) * host_raw_action_gradients.size(), cudaMemcpyDeviceToHost));
    ActorWeightsHost host_gradients;
    copy_actor_gradients_to_host(actor_gradients, host_gradients);

    double loss_sum = 0.0;
    for(float value: host_loss){
        loss_sum += value;
        metrics.finite = metrics.finite && std::isfinite(value);
    }
    double action_sum = 0.0;
    double action_sq_sum = 0.0;
    std::size_t saturated_count = 0;
    for(float value: host_actions){
        action_sum += value;
        action_sq_sum += static_cast<double>(value) * static_cast<double>(value);
        saturated_count += std::abs(value) >= 0.95f ? 1 : 0;
        metrics.finite = metrics.finite && std::isfinite(value);
    }
    double action_gradient_sq = 0.0;
    for(float value: host_action_gradients){
        action_gradient_sq += static_cast<double>(value) * static_cast<double>(value);
        metrics.finite = metrics.finite && std::isfinite(value);
    }
    double raw_action_gradient_sq = 0.0;
    for(float value: host_raw_action_gradients){
        raw_action_gradient_sq += static_cast<double>(value) * static_cast<double>(value);
        metrics.finite = metrics.finite && std::isfinite(value);
    }
    const float diff_loss_scaled = static_cast<float>(
        (debug_options.disable_physics_gradient ? 0.0 : debug_options.diff_rollout_loss_weight) *
        loss_sum / std::max<std::size_t>(1, batch.batch_size)
    );
    metrics.loss = diff_loss_scaled + critic_metrics.scaled_loss;
    metrics.action_mean = static_cast<float>(action_sum / std::max<std::size_t>(1, host_actions.size()));
    const double action_mean = metrics.action_mean;
    metrics.action_std = static_cast<float>(std::sqrt(std::max(0.0, action_sq_sum / std::max<std::size_t>(1, host_actions.size()) - action_mean * action_mean)));
    metrics.action_saturation = static_cast<float>(static_cast<double>(saturated_count) / std::max<std::size_t>(1, host_actions.size()));
    metrics.action_gradient_norm = static_cast<float>(std::sqrt(action_gradient_sq));
    metrics.raw_action_gradient_norm = static_cast<float>(std::sqrt(raw_action_gradient_sq));
    const float raw_actor_gradient_norm = actor_weights_l2_norm(host_gradients);
    bool skip_actor_step = raw_actor_gradient_norm > debug_options.actor_grad_skip_norm;
    if(debug_options.actor_grad_clip_enabled && raw_actor_gradient_norm > debug_options.actor_grad_clip_norm){
        const float scale = debug_options.actor_grad_clip_norm / std::max(debug_options.actor_grad_eps, raw_actor_gradient_norm);
        scale_actor_weights_host(host_gradients, scale);
        copy_actor_gradients_to_device(host_gradients, actor_gradients);
    }
    metrics.actor_gradient_norm = actor_weights_l2_norm(host_gradients);

    if(!skip_actor_step){
        adam_update_actor(
            actor_weights,
            actor_gradients,
            adam_first_moment,
            adam_second_moment,
            debug_options.learning_rate,
            0.9f,
            0.999f,
            1e-8f,
            static_cast<int>(update_step)
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ActorWeightsHost weights_after;
    ActorWeightsHost adam_m_host;
    ActorWeightsHost adam_v_host;
    copy_actor_weights_to_host(actor_weights, weights_after);
    copy_actor_weights_to_host(adam_first_moment, adam_m_host);
    copy_actor_weights_to_host(adam_second_moment, adam_v_host);
    metrics.weight_delta_norm = actor_weights_diff_l2_norm(weights_after, weights_before);
    metrics.adam_m_norm = actor_weights_l2_norm(adam_m_host);
    metrics.adam_v_norm = actor_weights_l2_norm(adam_v_host);
    metrics.finite = metrics.finite &&
        actor_weights_finite(host_gradients) &&
        actor_weights_finite(weights_after) &&
        actor_weights_finite(adam_m_host) &&
        actor_weights_finite(adam_v_host) &&
        critic_metrics.finite &&
        std::isfinite(metrics.loss) &&
        std::isfinite(metrics.actor_gradient_norm) &&
        std::isfinite(metrics.weight_delta_norm);
    return metrics;
}

} // namespace

ActorForwardValidationSummary validate_actor_forward_against_cpu(
    std::size_t batch_size,
    std::size_t sequence_length,
    unsigned seed,
    const EulerGpuRunOptions& options
){
    if(batch_size == 0 || sequence_length == 0){
        throw std::runtime_error("Actor forward validation requires non-zero batch and sequence length");
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> weight_dist(-0.12f, 0.12f);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::vector<float> encoder_w(RDAC_HIDDEN_DIM * RDAC_POLICY_INPUT_DIM);
    std::vector<float> encoder_b(RDAC_HIDDEN_DIM);
    std::vector<float> gru_w_input(3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM);
    std::vector<float> gru_b_input(3 * RDAC_HIDDEN_DIM);
    std::vector<float> gru_w_hidden(3 * RDAC_HIDDEN_DIM * RDAC_HIDDEN_DIM);
    std::vector<float> gru_b_hidden(3 * RDAC_HIDDEN_DIM);
    std::vector<float> gru_h0(RDAC_HIDDEN_DIM);
    std::vector<float> actor_w(RDAC_ACTION_DIM * RDAC_ACTOR_HEAD_INPUT_DIM);
    std::vector<float> actor_b(RDAC_ACTION_DIM);
    std::vector<float> critic_w(RDAC_CRITIC_DIM * RDAC_CRITIC_HEAD_INPUT_DIM);
    std::vector<float> critic_b(RDAC_CRITIC_DIM);
    for(auto* vec: {&encoder_w, &encoder_b, &gru_w_input, &gru_b_input, &gru_w_hidden, &gru_b_hidden, &gru_h0, &actor_w, &actor_b, &critic_w, &critic_b}){
        for(float& value: *vec){
            value = weight_dist(rng);
        }
    }

    std::vector<float> policy_input(sequence_length * batch_size * RDAC_POLICY_INPUT_DIM);
    std::vector<float> observation(sequence_length * batch_size * EULER_OBSERVATION_DIM);
    for(float& value: policy_input){
        value = input_dist(rng);
    }
    for(std::size_t t = 0; t < sequence_length; t++){
        for(std::size_t b = 0; b < batch_size; b++){
            for(std::size_t o = 0; o < EULER_OBSERVATION_DIM; o++){
                observation[actor_idx3(t, b, o, batch_size, EULER_OBSERVATION_DIM)] =
                    policy_input[actor_idx3(t, b, o, batch_size, RDAC_POLICY_INPUT_DIM)];
            }
        }
    }

    constexpr float ACTION_BOUND = 1.0f;
    std::vector<float> cpu_raw_action;
    std::vector<float> cpu_bounded_action;
    std::vector<float> cpu_hidden;
    std::vector<float> cpu_critic_output(sequence_length * batch_size * RDAC_CRITIC_DIM, 0.0f);
    cpu_actor_forward(
        batch_size, sequence_length,
        encoder_w, encoder_b,
        gru_w_input, gru_b_input, gru_w_hidden, gru_b_hidden, gru_h0,
        actor_w, actor_b, policy_input, observation,
        cpu_raw_action, cpu_bounded_action, cpu_hidden, ACTION_BOUND
    );
    for(std::size_t t = 0; t < sequence_length; t++){
        for(std::size_t b = 0; b < batch_size; b++){
            float critic_input[RDAC_CRITIC_HEAD_INPUT_DIM];
            for(std::size_t o = 0; o < EULER_OBSERVATION_DIM; o++){
                critic_input[o] = observation[actor_idx3(t, b, o, batch_size, EULER_OBSERVATION_DIM)];
            }
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                critic_input[EULER_OBSERVATION_DIM + h] = cpu_hidden[actor_idx3(t, b, h, batch_size, RDAC_HIDDEN_DIM)];
            }
            for(std::size_t c = 0; c < RDAC_CRITIC_DIM; c++){
                float q = critic_b[c];
                for(std::size_t k = 0; k < RDAC_CRITIC_HEAD_INPUT_DIM; k++){
                    q += critic_w[c * RDAC_CRITIC_HEAD_INPUT_DIM + k] * critic_input[k];
                }
                cpu_critic_output[actor_idx3(t, b, c, batch_size, RDAC_CRITIC_DIM)] = q;
            }
        }
    }

    ActorWeightsDevice d_weights;
    ActorBuffersDevice d_buffers;
    std::vector<float> gpu_raw_action(cpu_raw_action.size(), 0.0f);
    std::vector<float> gpu_bounded_action(cpu_bounded_action.size(), 0.0f);
    std::vector<float> gpu_hidden(cpu_hidden.size(), 0.0f);
    std::vector<float> gpu_critic_output(cpu_critic_output.size(), 0.0f);
    try{
        CUDA_CHECK(cudaSetDevice(options.device));
        actor_allocate(d_weights, d_buffers, batch_size, sequence_length);
        CUDA_CHECK(cudaMemcpy(d_weights.encoder_w, encoder_w.data(), sizeof(float) * encoder_w.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.encoder_b, encoder_b.data(), sizeof(float) * encoder_b.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.gru_w_input, gru_w_input.data(), sizeof(float) * gru_w_input.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.gru_b_input, gru_b_input.data(), sizeof(float) * gru_b_input.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.gru_w_hidden, gru_w_hidden.data(), sizeof(float) * gru_w_hidden.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.gru_b_hidden, gru_b_hidden.data(), sizeof(float) * gru_b_hidden.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.gru_h0, gru_h0.data(), sizeof(float) * gru_h0.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.actor_w, actor_w.data(), sizeof(float) * actor_w.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.actor_b, actor_b.data(), sizeof(float) * actor_b.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.critic_w, critic_w.data(), sizeof(float) * critic_w.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.critic_b, critic_b.data(), sizeof(float) * critic_b.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_buffers.policy_input, policy_input.data(), sizeof(float) * policy_input.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_buffers.observation, observation.data(), sizeof(float) * observation.size(), cudaMemcpyHostToDevice));
        const int block = 256;
        const int grid = static_cast<int>((batch_size + block - 1) / block);
        for(std::size_t t = 0; t < sequence_length; t++){
            rdac_actor_forward_step_kernel<<<grid, block>>>(d_weights, d_buffers, batch_size, t, ACTION_BOUND, false);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(gpu_raw_action.data(), d_buffers.raw_action, sizeof(float) * gpu_raw_action.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_bounded_action.data(), d_buffers.bounded_action, sizeof(float) * gpu_bounded_action.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_hidden.data(), d_buffers.hidden, sizeof(float) * gpu_hidden.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_critic_output.data(), d_buffers.critic_output, sizeof(float) * gpu_critic_output.size(), cudaMemcpyDeviceToHost));
        actor_free(d_buffers);
        actor_free(d_weights);
    }
    catch(const std::exception& e){
        actor_free(d_buffers);
        actor_free(d_weights);
        throw;
    }

    ActorForwardValidationSummary summary;
    auto compare = [&](const std::vector<float>& gpu_values, const std::vector<float>& cpu_values, float& max_error){
        for(std::size_t i = 0; i < gpu_values.size(); i++){
            if(!std::isfinite(gpu_values[i]) || !std::isfinite(cpu_values[i])){
                summary.nan_inf_count++;
                continue;
            }
            max_error = std::max(max_error, std::abs(gpu_values[i] - cpu_values[i]));
        }
    };
    compare(gpu_raw_action, cpu_raw_action, summary.max_raw_action_abs_error);
    compare(gpu_bounded_action, cpu_bounded_action, summary.max_bounded_action_abs_error);
    compare(gpu_hidden, cpu_hidden, summary.max_hidden_abs_error);
    compare(gpu_critic_output, cpu_critic_output, summary.max_critic_output_abs_error);
    summary.raw_action_close = summary.max_raw_action_abs_error < 1e-4f;
    summary.bounded_action_close = summary.max_bounded_action_abs_error < 1e-4f;
    summary.hidden_close = summary.max_hidden_abs_error < 1e-4f;
    summary.critic_output_close = summary.max_critic_output_abs_error < 1e-4f;
    summary.critic_checked = true;
    summary.passed = summary.nan_inf_count == 0 &&
        summary.raw_action_close &&
        summary.bounded_action_close &&
        summary.hidden_close &&
        summary.critic_output_close;
    return summary;
}

ClosedLoopValidationSummary validate_closed_loop_against_cpu(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options
){
    namespace diff = rl_tools::rl::environments::l2f::diff;
    using TI = std::uint64_t;
    if(batch_size == 0 || horizon == 0 || horizon > CPU_MAX_HORIZON){
        throw std::runtime_error("Closed-loop validation requires non-zero batch/horizon within CPU_MAX_HORIZON");
    }

    EulerGpuBatch batch;
    generate_validation_batch(batch, batch_size, horizon, seed);
    std::mt19937 actor_rng(seed + 991u);
    ActorWeightsHost actor_weights;
    generate_actor_weights(actor_weights, actor_rng, 0.015f);

    std::vector<float> gpu_p((horizon + 1) * batch_size * 3, 0.0f);
    std::vector<float> gpu_v((horizon + 1) * batch_size * 3, 0.0f);
    std::vector<float> gpu_R((horizon + 1) * batch_size * 9, 0.0f);
    std::vector<float> gpu_omega((horizon + 1) * batch_size * 3, 0.0f);
    std::vector<float> gpu_rpm((horizon + 1) * batch_size * 4, 0.0f);
    std::vector<float> gpu_actions(horizon * batch_size * RDAC_ACTION_DIM, 0.0f);
    std::vector<float> gpu_hidden(horizon * batch_size * RDAC_HIDDEN_DIM, 0.0f);
    std::vector<float> gpu_loss(batch_size, 0.0f);

    DeviceArrays d;
    ActorWeightsDevice d_actor_weights;
    ActorBuffersDevice d_actor_buffers;
    try{
        CUDA_CHECK(cudaSetDevice(options.device));
        allocate(d, batch_size, horizon);
        actor_allocate(d_actor_weights, d_actor_buffers, batch_size, horizon);
        copy_input_to_device(batch, d);
        copy_actor_weights_to_device(actor_weights, d_actor_weights);
        const int block = 256;
        const int grid = static_cast<int>((batch_size + block - 1) / block);
        for(std::size_t t = 0; t < horizon; t++){
            observation_kernel<<<grid, block>>>(d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            build_policy_input_from_observation_kernel<<<grid, block>>>(d, d_actor_buffers, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            rdac_actor_forward_step_kernel<<<grid, block>>>(d_actor_weights, d_actor_buffers, batch_size, t, 1.0f, false);
            CUDA_CHECK(cudaGetLastError());
            copy_bounded_action_to_rollout_kernel<<<grid, block>>>(d_actor_buffers, d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            forward_step_kernel<<<grid, block>>>(d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
        }
        const std::size_t state_count = (horizon + 1) * batch_size;
        const std::size_t step_count = horizon * batch_size;
        CUDA_CHECK(cudaMemset(d.lambda_p, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_v, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_R, 0, sizeof(float) * state_count * 9));
        CUDA_CHECK(cudaMemset(d.lambda_omega, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_rpm, 0, sizeof(float) * state_count * 4));
        CUDA_CHECK(cudaMemset(d.action_gradients, 0, sizeof(float) * step_count * 4));
        CUDA_CHECK(cudaMemset(d.loss, 0, sizeof(float) * batch_size));
        loss_and_action_kernel<<<grid, block>>>(d, weights, batch_size, horizon);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(gpu_p.data(), d.p, sizeof(float) * gpu_p.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_v.data(), d.v, sizeof(float) * gpu_v.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_R.data(), d.R, sizeof(float) * gpu_R.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_omega.data(), d.omega, sizeof(float) * gpu_omega.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_rpm.data(), d.rpm, sizeof(float) * gpu_rpm.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_actions.data(), d.actions, sizeof(float) * gpu_actions.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_hidden.data(), d_actor_buffers.hidden, sizeof(float) * gpu_hidden.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_loss.data(), d.loss, sizeof(float) * gpu_loss.size(), cudaMemcpyDeviceToHost));
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
    }
    catch(const std::exception&){
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
        throw;
    }

    std::vector<float> cpu_p((horizon + 1) * batch_size * 3, 0.0f);
    std::vector<float> cpu_v((horizon + 1) * batch_size * 3, 0.0f);
    std::vector<float> cpu_R((horizon + 1) * batch_size * 9, 0.0f);
    std::vector<float> cpu_omega((horizon + 1) * batch_size * 3, 0.0f);
    std::vector<float> cpu_rpm((horizon + 1) * batch_size * 4, 0.0f);
    std::vector<float> cpu_actions(horizon * batch_size * RDAC_ACTION_DIM, 0.0f);
    std::vector<float> cpu_hidden(horizon * batch_size * RDAC_HIDDEN_DIM, 0.0f);
    std::vector<float> cpu_loss(batch_size, 0.0f);

    diff::EulerLossWeights<float> cpu_weights{
        weights.position, weights.velocity, weights.attitude, weights.angular_velocity,
        weights.action_magnitude, weights.action_smoothness, weights.saturation,
        weights.terminal_loss_weight, weights.terminal_position, weights.terminal_velocity,
        weights.terminal_attitude, weights.terminal_angular_velocity
    };

    for(std::size_t b = 0; b < batch_size; b++){
        CpuSimpleParameters parameters = cpu_parameters_from_batch(batch, b);
        diff::EulerState<float, TI> state{};
        for(int i = 0; i < 3; i++){
            state.p[i] = batch.initial_p[pidx3(b, i)];
            state.v[i] = batch.initial_v[pidx3(b, i)];
            state.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                state.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            state.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            state.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        auto store_state = [&](std::size_t t, const diff::EulerState<float, TI>& s){
            for(int i = 0; i < 3; i++){
                cpu_p[idx3(t, b, i, batch_size)] = s.p[i];
                cpu_v[idx3(t, b, i, batch_size)] = s.v[i];
                cpu_omega[idx3(t, b, i, batch_size)] = s.omega[i];
                for(int j = 0; j < 3; j++){
                    cpu_R[idx9(t, b, i * 3 + j, batch_size)] = s.R[i][j];
                }
            }
            for(int i = 0; i < 4; i++){
                cpu_rpm[idx4(t, b, i, batch_size)] = s.rpm[i];
            }
        };
        store_state(0, state);
        float previous_hidden[RDAC_HIDDEN_DIM];
        for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
            previous_hidden[h] = actor_weights.gru_h0[h];
        }
        float actions_for_loss[CPU_MAX_HORIZON][4] = {};
        diff::TrackingReference<float> ref{};
        for(int i = 0; i < 3; i++){
            ref.p[i] = batch.reference_p[pidx3(b, i)];
            ref.v[i] = batch.reference_v[pidx3(b, i)];
        }
        for(std::size_t t = 0; t < horizon; t++){
            float observation[EULER_OBSERVATION_DIM];
            std::size_t out = 0;
            for(int i = 0; i < 3; i++){ observation[out++] = state.p[i] - ref.p[i]; }
            for(int i = 0; i < 3; i++){ for(int j = 0; j < 3; j++){ observation[out++] = state.R[i][j]; } }
            for(int i = 0; i < 3; i++){ observation[out++] = state.v[i] - ref.v[i]; }
            for(int i = 0; i < 3; i++){ observation[out++] = state.omega[i]; }
            for(int i = 0; i < 4; i++){ observation[out++] = state.previous_action[i]; }
            float policy_input[RDAC_POLICY_INPUT_DIM] = {};
            for(std::size_t i = 0; i < EULER_OBSERVATION_DIM; i++){
                policy_input[i] = observation[i];
            }
            for(std::size_t i = 0; i < RDAC_ACTION_DIM; i++){
                policy_input[EULER_OBSERVATION_DIM + i] = observation[18 + i];
            }
            float raw_action[RDAC_ACTION_DIM];
            float bounded_action[RDAC_ACTION_DIM];
            float next_hidden[RDAC_HIDDEN_DIM];
            cpu_actor_forward_step(actor_weights, policy_input, observation, previous_hidden, raw_action, bounded_action, next_hidden, 1.0f);
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                cpu_hidden[actor_idx3(t, b, h, batch_size, RDAC_HIDDEN_DIM)] = next_hidden[h];
                previous_hidden[h] = next_hidden[h];
            }
            for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                cpu_actions[idx4(t, b, a, batch_size)] = bounded_action[a];
                actions_for_loss[t][a] = bounded_action[a];
            }
            diff::EulerState<float, TI> next{};
            diff::EulerStepCache<float> cache{};
            diff::step<CpuSimpleParameters, float, TI>(parameters, state, bounded_action, next, cache);
            state = next;
            store_state(t + 1, state);
        }
        diff::EulerState<float, TI> loss_states[CPU_MAX_HORIZON + 1];
        diff::EulerStepCache<float> loss_caches[CPU_MAX_HORIZON];
        diff::EulerState<float, TI> initial{};
        for(int i = 0; i < 3; i++){
            initial.p[i] = batch.initial_p[pidx3(b, i)];
            initial.v[i] = batch.initial_v[pidx3(b, i)];
            initial.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                initial.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            initial.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            initial.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        auto terms = diff::rollout_loss<CpuSimpleParameters, float, TI, CPU_MAX_HORIZON>(
            parameters, initial, actions_for_loss, static_cast<TI>(horizon), cpu_weights, ref, loss_states, loss_caches
        );
        cpu_loss[b] = terms.total();
    }

    ClosedLoopValidationSummary summary;
    auto compare_abs = [&](const std::vector<float>& gpu_values, const std::vector<float>& cpu_values, float& max_error){
        for(std::size_t i = 0; i < gpu_values.size(); i++){
            if(!std::isfinite(gpu_values[i]) || !std::isfinite(cpu_values[i])){
                summary.nan_inf_count++;
                continue;
            }
            max_error = std::max(max_error, std::abs(gpu_values[i] - cpu_values[i]));
        }
    };
    compare_abs(gpu_p, cpu_p, summary.max_state_abs_error);
    compare_abs(gpu_v, cpu_v, summary.max_state_abs_error);
    compare_abs(gpu_R, cpu_R, summary.max_state_abs_error);
    compare_abs(gpu_omega, cpu_omega, summary.max_state_abs_error);
    compare_abs(gpu_actions, cpu_actions, summary.max_action_abs_error);
    compare_abs(gpu_hidden, cpu_hidden, summary.max_hidden_abs_error);
    for(std::size_t i = 0; i < gpu_rpm.size(); i++){
        if(!std::isfinite(gpu_rpm[i]) || !std::isfinite(cpu_rpm[i])){
            summary.nan_inf_count++;
            continue;
        }
        const float abs_error = std::abs(gpu_rpm[i] - cpu_rpm[i]);
        summary.max_rpm_abs_error = std::max(summary.max_rpm_abs_error, abs_error);
        summary.max_rpm_rel_error = std::max(summary.max_rpm_rel_error, abs_error / std::max(1.0f, std::abs(cpu_rpm[i])));
    }
    for(std::size_t i = 0; i < gpu_loss.size(); i++){
        if(!std::isfinite(gpu_loss[i]) || !std::isfinite(cpu_loss[i])){
            summary.nan_inf_count++;
            continue;
        }
        const float abs_error = std::abs(gpu_loss[i] - cpu_loss[i]);
        summary.max_loss_abs_error = std::max(summary.max_loss_abs_error, abs_error);
        summary.max_loss_rel_error = std::max(summary.max_loss_rel_error, abs_error / std::max(1e-6f, std::abs(cpu_loss[i])));
    }
    summary.state_close = summary.max_state_abs_error < 1e-3f && summary.max_rpm_rel_error < 1e-5f;
    summary.action_close = summary.max_action_abs_error < 1e-4f;
    summary.hidden_close = summary.max_hidden_abs_error < 1e-4f;
    summary.loss_close = summary.max_loss_rel_error < 1e-3f || summary.max_loss_abs_error < 1e-4f;
    summary.passed = summary.nan_inf_count == 0 && summary.state_close && summary.action_close && summary.hidden_close && summary.loss_close;
    return summary;
}

ActionGradientInjectionValidationSummary validate_action_gradient_injection_against_cpu(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options
){
    namespace diff = rl_tools::rl::environments::l2f::diff;
    using TI = std::uint64_t;
    if(batch_size == 0 || horizon == 0 || horizon > CPU_MAX_HORIZON){
        throw std::runtime_error("Action-gradient injection validation requires non-zero batch/horizon within CPU_MAX_HORIZON");
    }

    EulerGpuBatch batch;
    generate_validation_batch(batch, batch_size, horizon, seed);
    std::mt19937 actor_rng(seed + 991u);
    ActorWeightsHost actor_weights;
    generate_actor_weights(actor_weights, actor_rng, 0.015f);
    for(std::size_t k = 0; k < RDAC_ACTOR_HEAD_INPUT_DIM; k++){
        actor_weights.actor_w[k] = 0.0f;
    }
    actor_weights.actor_b[0] = 1.10f;

    const std::size_t gradient_count = horizon * batch_size * RDAC_ACTION_DIM;
    std::vector<float> gpu_action_gradients(gradient_count, 0.0f);
    std::vector<float> gpu_raw_action_gradients(gradient_count, 0.0f);
    std::vector<float> gpu_action_derivatives(gradient_count, 0.0f);

    DeviceArrays d;
    ActorWeightsDevice d_actor_weights;
    ActorBuffersDevice d_actor_buffers;
    try{
        CUDA_CHECK(cudaSetDevice(options.device));
        allocate(d, batch_size, horizon);
        actor_allocate(d_actor_weights, d_actor_buffers, batch_size, horizon);
        copy_input_to_device(batch, d);
        copy_actor_weights_to_device(actor_weights, d_actor_weights);
        const int block = 256;
        const int grid = static_cast<int>((batch_size + block - 1) / block);
        for(std::size_t t = 0; t < horizon; t++){
            observation_kernel<<<grid, block>>>(d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            build_policy_input_from_observation_kernel<<<grid, block>>>(d, d_actor_buffers, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            rdac_actor_forward_step_kernel<<<grid, block>>>(d_actor_weights, d_actor_buffers, batch_size, t, 1.0f, false);
            CUDA_CHECK(cudaGetLastError());
            copy_bounded_action_to_rollout_kernel<<<grid, block>>>(d_actor_buffers, d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            forward_step_kernel<<<grid, block>>>(d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
        }

        const std::size_t state_count = (horizon + 1) * batch_size;
        const std::size_t step_count = horizon * batch_size;
        CUDA_CHECK(cudaMemset(d.lambda_p, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_v, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_R, 0, sizeof(float) * state_count * 9));
        CUDA_CHECK(cudaMemset(d.lambda_omega, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_rpm, 0, sizeof(float) * state_count * 4));
        CUDA_CHECK(cudaMemset(d.action_gradients, 0, sizeof(float) * step_count * 4));
        CUDA_CHECK(cudaMemset(d.loss, 0, sizeof(float) * batch_size));
        CUDA_CHECK(cudaMemset(d_actor_buffers.raw_action_gradient, 0, sizeof(float) * gradient_count));
        CUDA_CHECK(cudaMemset(d_actor_buffers.critic_output_gradient, 0, sizeof(float) * step_count * RDAC_CRITIC_DIM));
        loss_and_action_kernel<<<grid, block>>>(d, weights, batch_size, horizon);
        CUDA_CHECK(cudaGetLastError());
        for(std::size_t reverse_i = 0; reverse_i < horizon; reverse_i++){
            const std::size_t step_i = horizon - 1 - reverse_i;
            backward_step_kernel<<<grid, block>>>(d, batch_size, step_i);
            CUDA_CHECK(cudaGetLastError());
        }
        const int grad_grid = static_cast<int>((gradient_count + block - 1) / block);
        inject_action_gradient_kernel<<<grad_grid, block>>>(d, d_actor_buffers, batch_size, horizon);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(gpu_action_gradients.data(), d.action_gradients, sizeof(float) * gradient_count, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_raw_action_gradients.data(), d_actor_buffers.raw_action_gradient, sizeof(float) * gradient_count, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_action_derivatives.data(), d_actor_buffers.action_derivative, sizeof(float) * gradient_count, cudaMemcpyDeviceToHost));
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
    }
    catch(const std::exception&){
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
        throw;
    }

    std::vector<float> cpu_action_gradients(gradient_count, 0.0f);
    std::vector<float> cpu_raw_action_gradients(gradient_count, 0.0f);
    std::vector<float> cpu_action_derivatives(gradient_count, 0.0f);

    diff::EulerLossWeights<float> cpu_weights{
        weights.position, weights.velocity, weights.attitude, weights.angular_velocity,
        weights.action_magnitude, weights.action_smoothness, weights.saturation,
        weights.terminal_loss_weight, weights.terminal_position, weights.terminal_velocity,
        weights.terminal_attitude, weights.terminal_angular_velocity
    };

    for(std::size_t b = 0; b < batch_size; b++){
        CpuSimpleParameters parameters = cpu_parameters_from_batch(batch, b);
        diff::EulerState<float, TI> state{};
        for(int i = 0; i < 3; i++){
            state.p[i] = batch.initial_p[pidx3(b, i)];
            state.v[i] = batch.initial_v[pidx3(b, i)];
            state.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                state.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            state.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            state.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        diff::TrackingReference<float> ref{};
        for(int i = 0; i < 3; i++){
            ref.p[i] = batch.reference_p[pidx3(b, i)];
            ref.v[i] = batch.reference_v[pidx3(b, i)];
        }
        float previous_hidden[RDAC_HIDDEN_DIM];
        for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
            previous_hidden[h] = actor_weights.gru_h0[h];
        }
        float actions_for_loss[CPU_MAX_HORIZON][4] = {};
        float action_derivatives[CPU_MAX_HORIZON][4] = {};
        for(std::size_t t = 0; t < horizon; t++){
            float observation[EULER_OBSERVATION_DIM];
            std::size_t out = 0;
            for(int i = 0; i < 3; i++){ observation[out++] = state.p[i] - ref.p[i]; }
            for(int i = 0; i < 3; i++){ for(int j = 0; j < 3; j++){ observation[out++] = state.R[i][j]; } }
            for(int i = 0; i < 3; i++){ observation[out++] = state.v[i] - ref.v[i]; }
            for(int i = 0; i < 3; i++){ observation[out++] = state.omega[i]; }
            for(int i = 0; i < 4; i++){ observation[out++] = state.previous_action[i]; }
            float policy_input[RDAC_POLICY_INPUT_DIM] = {};
            for(std::size_t i = 0; i < EULER_OBSERVATION_DIM; i++){
                policy_input[i] = observation[i];
            }
            for(std::size_t i = 0; i < RDAC_ACTION_DIM; i++){
                policy_input[EULER_OBSERVATION_DIM + i] = observation[18 + i];
            }
            float raw_action[RDAC_ACTION_DIM];
            float bounded_action[RDAC_ACTION_DIM];
            float next_hidden[RDAC_HIDDEN_DIM];
            cpu_actor_forward_step(actor_weights, policy_input, observation, previous_hidden, raw_action, bounded_action, next_hidden, 1.0f);
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                previous_hidden[h] = next_hidden[h];
            }
            for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                actions_for_loss[t][a] = bounded_action[a];
                action_derivatives[t][a] = (raw_action[a] > 1.0f || raw_action[a] < -1.0f) ? 0.0f : 1.0f;
                cpu_action_derivatives[actor_idx3(t, b, a, batch_size, RDAC_ACTION_DIM)] = action_derivatives[t][a];
            }
            diff::EulerState<float, TI> next{};
            diff::EulerStepCache<float> cache{};
            diff::step<CpuSimpleParameters, float, TI>(parameters, state, bounded_action, next, cache);
            state = next;
        }

        diff::EulerState<float, TI> initial{};
        for(int i = 0; i < 3; i++){
            initial.p[i] = batch.initial_p[pidx3(b, i)];
            initial.v[i] = batch.initial_v[pidx3(b, i)];
            initial.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                initial.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            initial.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            initial.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        diff::EulerState<float, TI> states[CPU_MAX_HORIZON + 1];
        diff::EulerStepCache<float> caches[CPU_MAX_HORIZON];
        float action_gradients[CPU_MAX_HORIZON][4] = {};
        (void)diff::rollout_loss_and_gradients<CpuSimpleParameters, float, TI, CPU_MAX_HORIZON>(
            parameters, initial, actions_for_loss, static_cast<TI>(horizon), cpu_weights, ref, states, caches, action_gradients
        );
        for(std::size_t t = 0; t < horizon; t++){
            for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                const std::size_t index = actor_idx3(t, b, a, batch_size, RDAC_ACTION_DIM);
                cpu_action_gradients[index] = action_gradients[t][a];
                cpu_raw_action_gradients[index] = action_gradients[t][a] * action_derivatives[t][a];
            }
        }
    }

    ActionGradientInjectionValidationSummary summary;
    auto compare_gradient = [&](const std::vector<float>& gpu_values, const std::vector<float>& cpu_values, float& max_abs, float& max_rel, float& l2_rel){
        double error_sq = 0.0;
        double reference_sq = 0.0;
        for(std::size_t i = 0; i < gpu_values.size(); i++){
            if(!std::isfinite(gpu_values[i]) || !std::isfinite(cpu_values[i])){
                summary.nan_inf_count++;
                continue;
            }
            const float abs_error = std::abs(gpu_values[i] - cpu_values[i]);
            max_abs = std::max(max_abs, abs_error);
            max_rel = std::max(max_rel, abs_error / std::max(1e-6f, std::abs(cpu_values[i])));
            error_sq += static_cast<double>(abs_error) * static_cast<double>(abs_error);
            reference_sq += static_cast<double>(cpu_values[i]) * static_cast<double>(cpu_values[i]);
        }
        l2_rel = static_cast<float>(std::sqrt(error_sq) / std::max(1e-12, std::sqrt(reference_sq)));
    };

    compare_gradient(
        gpu_action_gradients,
        cpu_action_gradients,
        summary.max_action_gradient_abs_error,
        summary.max_action_gradient_rel_error,
        summary.action_gradient_l2_rel_error
    );
    compare_gradient(
        gpu_raw_action_gradients,
        cpu_raw_action_gradients,
        summary.max_raw_action_gradient_abs_error,
        summary.max_raw_action_gradient_rel_error,
        summary.raw_action_gradient_l2_rel_error
    );

    for(std::size_t i = 0; i < gradient_count; i++){
        if(!std::isfinite(gpu_action_derivatives[i]) || !std::isfinite(cpu_action_derivatives[i])){
            summary.nan_inf_count++;
            continue;
        }
        const float derivative_error = std::abs(gpu_action_derivatives[i] - cpu_action_derivatives[i]);
        summary.max_action_derivative_abs_error = std::max(summary.max_action_derivative_abs_error, derivative_error);
        if(cpu_action_derivatives[i] == 0.0f){
            summary.clamped_action_count++;
            if(std::abs(gpu_raw_action_gradients[i]) > 1e-6f || std::abs(cpu_raw_action_gradients[i]) > 1e-6f){
                summary.clamp_zero_violation_count++;
            }
        }
    }

    summary.action_gradient_close = summary.action_gradient_l2_rel_error < 1e-2f || summary.max_action_gradient_abs_error < 1e-3f;
    summary.raw_action_gradient_close = summary.raw_action_gradient_l2_rel_error < 1e-2f || summary.max_raw_action_gradient_abs_error < 1e-3f;
    summary.derivative_close = summary.max_action_derivative_abs_error < 1e-6f;
    summary.clamp_zero_ok = summary.clamped_action_count > 0 && summary.clamp_zero_violation_count == 0;
    summary.passed = summary.nan_inf_count == 0 &&
        summary.action_gradient_close &&
        summary.raw_action_gradient_close &&
        summary.derivative_close &&
        summary.clamp_zero_ok;
    return summary;
}

ActorBackwardValidationSummary validate_actor_backward_against_cpu(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options
){
    namespace diff = rl_tools::rl::environments::l2f::diff;
    using TI = std::uint64_t;
    if(batch_size == 0 || horizon == 0 || horizon > CPU_MAX_HORIZON){
        throw std::runtime_error("Actor backward validation requires non-zero batch/horizon within CPU_MAX_HORIZON");
    }

    EulerGpuBatch batch;
    generate_validation_batch(batch, batch_size, horizon, seed);
    std::mt19937 actor_rng(seed + 991u);
    ActorWeightsHost actor_weights;
    generate_actor_weights(actor_weights, actor_rng, 0.015f);

    ActorWeightsHost gpu_gradients;
    DeviceArrays d;
    ActorWeightsDevice d_actor_weights;
    ActorBuffersDevice d_actor_buffers;
    ActorGradientsDevice d_actor_gradients;
    try{
        CUDA_CHECK(cudaSetDevice(options.device));
        allocate(d, batch_size, horizon);
        actor_allocate(d_actor_weights, d_actor_buffers, batch_size, horizon);
        actor_allocate(d_actor_gradients, batch_size);
        copy_input_to_device(batch, d);
        copy_actor_weights_to_device(actor_weights, d_actor_weights);
        zero_actor_gradients(d_actor_gradients, batch_size);
        const int block = 256;
        const int grid = static_cast<int>((batch_size + block - 1) / block);
        for(std::size_t t = 0; t < horizon; t++){
            observation_kernel<<<grid, block>>>(d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            build_policy_input_from_observation_kernel<<<grid, block>>>(d, d_actor_buffers, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            rdac_actor_forward_step_kernel<<<grid, block>>>(d_actor_weights, d_actor_buffers, batch_size, t, 1.0f, false);
            CUDA_CHECK(cudaGetLastError());
            copy_bounded_action_to_rollout_kernel<<<grid, block>>>(d_actor_buffers, d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            forward_step_kernel<<<grid, block>>>(d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
        }

        const std::size_t state_count = (horizon + 1) * batch_size;
        const std::size_t step_count = horizon * batch_size;
        const std::size_t gradient_count = step_count * RDAC_ACTION_DIM;
        CUDA_CHECK(cudaMemset(d.lambda_p, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_v, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_R, 0, sizeof(float) * state_count * 9));
        CUDA_CHECK(cudaMemset(d.lambda_omega, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_rpm, 0, sizeof(float) * state_count * 4));
        CUDA_CHECK(cudaMemset(d.action_gradients, 0, sizeof(float) * step_count * 4));
        CUDA_CHECK(cudaMemset(d.loss, 0, sizeof(float) * batch_size));
        CUDA_CHECK(cudaMemset(d_actor_buffers.raw_action_gradient, 0, sizeof(float) * gradient_count));
        CUDA_CHECK(cudaMemset(d_actor_buffers.critic_output_gradient, 0, sizeof(float) * step_count * RDAC_CRITIC_DIM));
        loss_and_action_kernel<<<grid, block>>>(d, weights, batch_size, horizon);
        CUDA_CHECK(cudaGetLastError());
        for(std::size_t reverse_i = 0; reverse_i < horizon; reverse_i++){
            const std::size_t step_i = horizon - 1 - reverse_i;
            backward_step_kernel<<<grid, block>>>(d, batch_size, step_i);
            CUDA_CHECK(cudaGetLastError());
        }
        const int grad_grid = static_cast<int>((gradient_count + block - 1) / block);
        inject_action_gradient_kernel<<<grad_grid, block>>>(d, d_actor_buffers, batch_size, horizon);
        CUDA_CHECK(cudaGetLastError());
        for(std::size_t reverse_i = 0; reverse_i < horizon; reverse_i++){
            const std::size_t step_i = horizon - 1 - reverse_i;
            rdac_actor_backward_step_kernel<<<grid, block>>>(d_actor_weights, d_actor_buffers, d_actor_gradients, batch_size, step_i);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        copy_actor_gradients_to_host(d_actor_gradients, gpu_gradients);
        actor_free(d_actor_gradients);
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
    }
    catch(const std::exception&){
        actor_free(d_actor_gradients);
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
        throw;
    }

    const std::size_t sequence_count = horizon * batch_size;
    std::vector<float> cpu_policy_input(sequence_count * RDAC_POLICY_INPUT_DIM, 0.0f);
    std::vector<float> cpu_observation(sequence_count * EULER_OBSERVATION_DIM, 0.0f);
    std::vector<float> cpu_hidden(sequence_count * RDAC_HIDDEN_DIM, 0.0f);
    std::vector<float> cpu_raw_action_gradient(sequence_count * RDAC_ACTION_DIM, 0.0f);

    diff::EulerLossWeights<float> cpu_weights{
        weights.position, weights.velocity, weights.attitude, weights.angular_velocity,
        weights.action_magnitude, weights.action_smoothness, weights.saturation,
        weights.terminal_loss_weight, weights.terminal_position, weights.terminal_velocity,
        weights.terminal_attitude, weights.terminal_angular_velocity
    };

    for(std::size_t b = 0; b < batch_size; b++){
        CpuSimpleParameters parameters = cpu_parameters_from_batch(batch, b);
        diff::EulerState<float, TI> state{};
        for(int i = 0; i < 3; i++){
            state.p[i] = batch.initial_p[pidx3(b, i)];
            state.v[i] = batch.initial_v[pidx3(b, i)];
            state.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                state.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            state.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            state.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        diff::TrackingReference<float> ref{};
        for(int i = 0; i < 3; i++){
            ref.p[i] = batch.reference_p[pidx3(b, i)];
            ref.v[i] = batch.reference_v[pidx3(b, i)];
        }

        float previous_hidden[RDAC_HIDDEN_DIM];
        for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
            previous_hidden[h] = actor_weights.gru_h0[h];
        }
        float actions_for_loss[CPU_MAX_HORIZON][4] = {};
        float action_derivatives[CPU_MAX_HORIZON][4] = {};
        for(std::size_t t = 0; t < horizon; t++){
            float observation[EULER_OBSERVATION_DIM];
            std::size_t out = 0;
            for(int i = 0; i < 3; i++){ observation[out++] = state.p[i] - ref.p[i]; }
            for(int i = 0; i < 3; i++){ for(int j = 0; j < 3; j++){ observation[out++] = state.R[i][j]; } }
            for(int i = 0; i < 3; i++){ observation[out++] = state.v[i] - ref.v[i]; }
            for(int i = 0; i < 3; i++){ observation[out++] = state.omega[i]; }
            for(int i = 0; i < 4; i++){ observation[out++] = state.previous_action[i]; }
            float policy_input[RDAC_POLICY_INPUT_DIM] = {};
            for(std::size_t i = 0; i < EULER_OBSERVATION_DIM; i++){
                policy_input[i] = observation[i];
                cpu_observation[actor_idx3(t, b, i, batch_size, EULER_OBSERVATION_DIM)] = observation[i];
            }
            for(std::size_t i = 0; i < RDAC_ACTION_DIM; i++){
                policy_input[EULER_OBSERVATION_DIM + i] = observation[18 + i];
            }
            for(std::size_t i = 0; i < RDAC_POLICY_INPUT_DIM; i++){
                cpu_policy_input[actor_idx3(t, b, i, batch_size, RDAC_POLICY_INPUT_DIM)] = policy_input[i];
            }
            float raw_action[RDAC_ACTION_DIM];
            float bounded_action[RDAC_ACTION_DIM];
            float next_hidden[RDAC_HIDDEN_DIM];
            cpu_actor_forward_step(actor_weights, policy_input, observation, previous_hidden, raw_action, bounded_action, next_hidden, 1.0f);
            for(std::size_t h = 0; h < RDAC_HIDDEN_DIM; h++){
                cpu_hidden[actor_idx3(t, b, h, batch_size, RDAC_HIDDEN_DIM)] = next_hidden[h];
                previous_hidden[h] = next_hidden[h];
            }
            for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                actions_for_loss[t][a] = bounded_action[a];
                action_derivatives[t][a] = (raw_action[a] > 1.0f || raw_action[a] < -1.0f) ? 0.0f : 1.0f;
            }
            diff::EulerState<float, TI> next{};
            diff::EulerStepCache<float> cache{};
            diff::step<CpuSimpleParameters, float, TI>(parameters, state, bounded_action, next, cache);
            state = next;
        }

        diff::EulerState<float, TI> initial{};
        for(int i = 0; i < 3; i++){
            initial.p[i] = batch.initial_p[pidx3(b, i)];
            initial.v[i] = batch.initial_v[pidx3(b, i)];
            initial.omega[i] = batch.initial_omega[pidx3(b, i)];
            for(int j = 0; j < 3; j++){
                initial.R[i][j] = batch.initial_R[pidx9(b, i * 3 + j)];
            }
        }
        for(int i = 0; i < 4; i++){
            initial.rpm[i] = batch.initial_rpm[pidx4(b, i)];
            initial.previous_action[i] = batch.initial_previous_action[pidx4(b, i)];
        }
        diff::EulerState<float, TI> states[CPU_MAX_HORIZON + 1];
        diff::EulerStepCache<float> caches[CPU_MAX_HORIZON];
        float action_gradients[CPU_MAX_HORIZON][4] = {};
        (void)diff::rollout_loss_and_gradients<CpuSimpleParameters, float, TI, CPU_MAX_HORIZON>(
            parameters, initial, actions_for_loss, static_cast<TI>(horizon), cpu_weights, ref, states, caches, action_gradients
        );
        for(std::size_t t = 0; t < horizon; t++){
            for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                cpu_raw_action_gradient[actor_idx3(t, b, a, batch_size, RDAC_ACTION_DIM)] =
                    action_gradients[t][a] * action_derivatives[t][a];
            }
        }
    }

    ActorWeightsHost cpu_gradients;
    cpu_actor_backward_sequence(
        actor_weights,
        batch_size,
        horizon,
        cpu_policy_input,
        cpu_observation,
        cpu_hidden,
        cpu_raw_action_gradient,
        cpu_gradients
    );

    ActorBackwardValidationSummary summary;
    double total_error_sq = 0.0;
    double total_reference_sq = 0.0;
    auto compare_group = [&](const std::vector<float>& gpu_values, const std::vector<float>& cpu_values, float& norm, float& cosine){
        double dot = 0.0;
        double gpu_sq = 0.0;
        double cpu_sq = 0.0;
        for(std::size_t i = 0; i < gpu_values.size(); i++){
            const float gpu_value = gpu_values[i];
            const float cpu_value = cpu_values[i];
            if(!std::isfinite(gpu_value) || !std::isfinite(cpu_value)){
                summary.nan_inf_count++;
                continue;
            }
            const float abs_error = std::abs(gpu_value - cpu_value);
            summary.max_abs_error = std::max(summary.max_abs_error, abs_error);
            dot += static_cast<double>(gpu_value) * static_cast<double>(cpu_value);
            gpu_sq += static_cast<double>(gpu_value) * static_cast<double>(gpu_value);
            cpu_sq += static_cast<double>(cpu_value) * static_cast<double>(cpu_value);
            total_error_sq += static_cast<double>(abs_error) * static_cast<double>(abs_error);
            total_reference_sq += static_cast<double>(cpu_value) * static_cast<double>(cpu_value);
        }
        norm = static_cast<float>(std::sqrt(gpu_sq));
        cosine = static_cast<float>(dot / std::max(1e-12, std::sqrt(gpu_sq) * std::sqrt(cpu_sq)));
    };

    auto concat = [](const std::vector<float>& a, const std::vector<float>& b){
        std::vector<float> out;
        out.reserve(a.size() + b.size());
        out.insert(out.end(), a.begin(), a.end());
        out.insert(out.end(), b.begin(), b.end());
        return out;
    };
    const auto gpu_encoder = concat(gpu_gradients.encoder_w, gpu_gradients.encoder_b);
    const auto cpu_encoder = concat(cpu_gradients.encoder_w, cpu_gradients.encoder_b);
    const auto gpu_gru_input = concat(gpu_gradients.gru_w_input, gpu_gradients.gru_b_input);
    const auto cpu_gru_input = concat(cpu_gradients.gru_w_input, cpu_gradients.gru_b_input);
    const auto gpu_gru_hidden = concat(gpu_gradients.gru_w_hidden, gpu_gradients.gru_b_hidden);
    const auto cpu_gru_hidden = concat(cpu_gradients.gru_w_hidden, cpu_gradients.gru_b_hidden);
    const auto gpu_actor_head = concat(gpu_gradients.actor_w, gpu_gradients.actor_b);
    const auto cpu_actor_head = concat(cpu_gradients.actor_w, cpu_gradients.actor_b);
    compare_group(gpu_encoder, cpu_encoder, summary.encoder_grad_norm, summary.encoder_grad_cosine);
    compare_group(gpu_gru_input, cpu_gru_input, summary.gru_input_grad_norm, summary.gru_input_grad_cosine);
    compare_group(gpu_gru_hidden, cpu_gru_hidden, summary.gru_hidden_grad_norm, summary.gru_hidden_grad_cosine);
    compare_group(gpu_actor_head, cpu_actor_head, summary.actor_head_grad_norm, summary.actor_head_grad_cosine);
    compare_group(gpu_gradients.gru_h0, cpu_gradients.gru_h0, summary.h0_grad_norm, summary.h0_grad_cosine);

    summary.l2_rel_error = static_cast<float>(std::sqrt(total_error_sq) / std::max(1e-12, std::sqrt(total_reference_sq)));
    summary.finite = summary.nan_inf_count == 0;
    summary.nonzero = summary.encoder_grad_norm > 1e-9f &&
        summary.gru_input_grad_norm > 1e-9f &&
        summary.gru_hidden_grad_norm > 1e-9f &&
        summary.actor_head_grad_norm > 1e-9f;
    summary.cosine_close = summary.encoder_grad_cosine > 0.99f &&
        summary.gru_input_grad_cosine > 0.99f &&
        summary.gru_hidden_grad_cosine > 0.99f &&
        summary.actor_head_grad_cosine > 0.99f &&
        summary.h0_grad_cosine > 0.99f;
    summary.passed = summary.finite && summary.nonzero && summary.cosine_close && summary.l2_rel_error < 1e-2f;
    return summary;
}

CriticBackwardValidationSummary validate_critic_backward_against_cpu(
    std::size_t batch_size,
    std::size_t horizon,
    unsigned seed,
    const EulerGpuRunOptions& options
){
    if(batch_size == 0 || horizon == 0 || horizon > CPU_MAX_HORIZON){
        throw std::runtime_error("Critic backward validation requires non-zero batch/horizon within CPU_MAX_HORIZON");
    }

    EulerGpuBatch batch;
    generate_validation_batch(batch, batch_size, horizon, seed);
    std::mt19937 actor_rng(seed + 991u);
    ActorWeightsHost actor_weights;
    generate_actor_weights(actor_weights, actor_rng, 0.015f);

    const std::size_t sequence_count = horizon * batch_size;
    std::vector<float> critic_output_gradient(sequence_count * RDAC_CRITIC_DIM, 0.0f);
    std::mt19937 grad_rng(seed + 17001u);
    std::uniform_real_distribution<float> grad_dist(-0.05f, 0.05f);
    for(float& value: critic_output_gradient){
        value = grad_dist(grad_rng);
    }

    ActorWeightsHost gpu_gradients;
    std::vector<float> host_policy_input(sequence_count * RDAC_POLICY_INPUT_DIM, 0.0f);
    std::vector<float> host_observation(sequence_count * EULER_OBSERVATION_DIM, 0.0f);
    std::vector<float> host_hidden(sequence_count * RDAC_HIDDEN_DIM, 0.0f);

    DeviceArrays d;
    ActorWeightsDevice d_actor_weights;
    ActorBuffersDevice d_actor_buffers;
    ActorGradientsDevice d_actor_gradients;
    try{
        CUDA_CHECK(cudaSetDevice(options.device));
        allocate(d, batch_size, horizon);
        actor_allocate(d_actor_weights, d_actor_buffers, batch_size, horizon);
        actor_allocate(d_actor_gradients, batch_size);
        copy_input_to_device(batch, d);
        copy_actor_weights_to_device(actor_weights, d_actor_weights);
        zero_actor_gradients(d_actor_gradients, batch_size);
        const int block = 256;
        const int grid = static_cast<int>((batch_size + block - 1) / block);
        for(std::size_t t = 0; t < horizon; t++){
            observation_kernel<<<grid, block>>>(d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            build_policy_input_from_observation_kernel<<<grid, block>>>(d, d_actor_buffers, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            rdac_actor_forward_step_kernel<<<grid, block>>>(d_actor_weights, d_actor_buffers, batch_size, t, 1.0f, false);
            CUDA_CHECK(cudaGetLastError());
            copy_bounded_action_to_rollout_kernel<<<grid, block>>>(d_actor_buffers, d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
            forward_step_kernel<<<grid, block>>>(d, batch_size, t);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaMemset(d_actor_buffers.raw_action_gradient, 0, sizeof(float) * sequence_count * RDAC_ACTION_DIM));
        CUDA_CHECK(cudaMemcpy(d_actor_buffers.critic_output_gradient, critic_output_gradient.data(), sizeof(float) * critic_output_gradient.size(), cudaMemcpyHostToDevice));
        for(std::size_t reverse_i = 0; reverse_i < horizon; reverse_i++){
            const std::size_t step_i = horizon - 1 - reverse_i;
            rdac_actor_backward_step_kernel<<<grid, block>>>(d_actor_weights, d_actor_buffers, d_actor_gradients, batch_size, step_i);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        copy_actor_gradients_to_host(d_actor_gradients, gpu_gradients);
        CUDA_CHECK(cudaMemcpy(host_policy_input.data(), d_actor_buffers.policy_input, sizeof(float) * host_policy_input.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_observation.data(), d_actor_buffers.observation, sizeof(float) * host_observation.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_hidden.data(), d_actor_buffers.hidden, sizeof(float) * host_hidden.size(), cudaMemcpyDeviceToHost));
        actor_free(d_actor_gradients);
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
    }
    catch(const std::exception&){
        actor_free(d_actor_gradients);
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
        throw;
    }

    std::vector<float> raw_action_gradient(sequence_count * RDAC_ACTION_DIM, 0.0f);
    ActorWeightsHost cpu_gradients;
    cpu_actor_backward_sequence(
        actor_weights,
        batch_size,
        horizon,
        host_policy_input,
        host_observation,
        host_hidden,
        raw_action_gradient,
        cpu_gradients,
        &critic_output_gradient
    );

    CriticBackwardValidationSummary summary;
    double total_error_sq = 0.0;
    double total_reference_sq = 0.0;
    auto compare_group = [&](const std::vector<float>& gpu_values, const std::vector<float>& cpu_values, float& norm, float& cosine){
        double dot = 0.0;
        double gpu_sq = 0.0;
        double cpu_sq = 0.0;
        for(std::size_t i = 0; i < gpu_values.size(); i++){
            const float gpu_value = gpu_values[i];
            const float cpu_value = cpu_values[i];
            if(!std::isfinite(gpu_value) || !std::isfinite(cpu_value)){
                summary.nan_inf_count++;
                continue;
            }
            const float abs_error = std::abs(gpu_value - cpu_value);
            summary.max_abs_error = std::max(summary.max_abs_error, abs_error);
            dot += static_cast<double>(gpu_value) * static_cast<double>(cpu_value);
            gpu_sq += static_cast<double>(gpu_value) * static_cast<double>(gpu_value);
            cpu_sq += static_cast<double>(cpu_value) * static_cast<double>(cpu_value);
            total_error_sq += static_cast<double>(abs_error) * static_cast<double>(abs_error);
            total_reference_sq += static_cast<double>(cpu_value) * static_cast<double>(cpu_value);
        }
        norm = static_cast<float>(std::sqrt(gpu_sq));
        cosine = static_cast<float>(dot / std::max(1e-12, std::sqrt(gpu_sq) * std::sqrt(cpu_sq)));
    };
    auto concat = [](const std::vector<float>& a, const std::vector<float>& b){
        std::vector<float> out;
        out.reserve(a.size() + b.size());
        out.insert(out.end(), a.begin(), a.end());
        out.insert(out.end(), b.begin(), b.end());
        return out;
    };
    const auto gpu_encoder = concat(gpu_gradients.encoder_w, gpu_gradients.encoder_b);
    const auto cpu_encoder = concat(cpu_gradients.encoder_w, cpu_gradients.encoder_b);
    const auto gpu_gru_input = concat(gpu_gradients.gru_w_input, gpu_gradients.gru_b_input);
    const auto cpu_gru_input = concat(cpu_gradients.gru_w_input, cpu_gradients.gru_b_input);
    const auto gpu_gru_hidden = concat(gpu_gradients.gru_w_hidden, gpu_gradients.gru_b_hidden);
    const auto cpu_gru_hidden = concat(cpu_gradients.gru_w_hidden, cpu_gradients.gru_b_hidden);
    const auto gpu_critic_head = concat(gpu_gradients.critic_w, gpu_gradients.critic_b);
    const auto cpu_critic_head = concat(cpu_gradients.critic_w, cpu_gradients.critic_b);
    const auto gpu_actor_head = concat(gpu_gradients.actor_w, gpu_gradients.actor_b);
    const auto cpu_actor_head = concat(cpu_gradients.actor_w, cpu_gradients.actor_b);
    compare_group(gpu_encoder, cpu_encoder, summary.encoder_grad_norm, summary.encoder_grad_cosine);
    compare_group(gpu_gru_input, cpu_gru_input, summary.gru_input_grad_norm, summary.gru_input_grad_cosine);
    compare_group(gpu_gru_hidden, cpu_gru_hidden, summary.gru_hidden_grad_norm, summary.gru_hidden_grad_cosine);
    compare_group(gpu_critic_head, cpu_critic_head, summary.critic_head_grad_norm, summary.critic_head_grad_cosine);
    compare_group(gpu_gradients.gru_h0, cpu_gradients.gru_h0, summary.h0_grad_norm, summary.h0_grad_cosine);

    double actor_head_sq = 0.0;
    for(float value: gpu_actor_head){
        actor_head_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    for(float value: cpu_actor_head){
        actor_head_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    summary.actor_head_grad_norm = static_cast<float>(std::sqrt(actor_head_sq));
    summary.l2_rel_error = static_cast<float>(std::sqrt(total_error_sq) / std::max(1e-12, std::sqrt(total_reference_sq)));
    summary.finite = summary.nan_inf_count == 0;
    summary.nonzero = summary.encoder_grad_norm > 1e-9f &&
        summary.gru_input_grad_norm > 1e-9f &&
        summary.gru_hidden_grad_norm > 1e-9f &&
        summary.critic_head_grad_norm > 1e-9f;
    summary.cosine_close = summary.encoder_grad_cosine > 0.99f &&
        summary.gru_input_grad_cosine > 0.99f &&
        summary.gru_hidden_grad_cosine > 0.99f &&
        summary.critic_head_grad_cosine > 0.99f &&
        summary.h0_grad_cosine > 0.99f;
    summary.actor_head_zero = summary.actor_head_grad_norm < 1e-9f;
    summary.passed = summary.finite &&
        summary.nonzero &&
        summary.cosine_close &&
        summary.actor_head_zero &&
        summary.l2_rel_error < 1e-2f;
    return summary;
}

AdamUpdateValidationSummary validate_adam_update_against_cpu(
    unsigned seed,
    const EulerGpuRunOptions& options
){
    std::mt19937 rng(seed + 7001u);
    ActorWeightsHost initial_weights;
    ActorWeightsHost gradients;
    generate_actor_weights(initial_weights, rng, 0.05f);
    generate_actor_weights(gradients, rng, 0.20f);

    constexpr float learning_rate = 1e-3f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float epsilon = 1e-8f;
    constexpr int step = 1;

    ActorWeightsHost gpu_weights;
    ActorWeightsHost gpu_first_moment;
    ActorWeightsHost gpu_second_moment;
    ActorWeightsDevice d_weights;
    ActorGradientsDevice d_gradients;
    ActorWeightsDevice d_first_moment;
    ActorWeightsDevice d_second_moment;
    try{
        CUDA_CHECK(cudaSetDevice(options.device));
        actor_allocate_weights(d_weights);
        actor_allocate(d_gradients, 1);
        actor_allocate_weights(d_first_moment);
        actor_allocate_weights(d_second_moment);
        copy_actor_weights_to_device(initial_weights, d_weights);
        copy_actor_gradients_to_device(gradients, d_gradients);
        zero_actor_weights_device(d_first_moment);
        zero_actor_weights_device(d_second_moment);
        adam_update_actor(d_weights, d_gradients, d_first_moment, d_second_moment, learning_rate, beta1, beta2, epsilon, step);
        CUDA_CHECK(cudaDeviceSynchronize());
        copy_actor_weights_to_host(d_weights, gpu_weights);
        copy_actor_weights_to_host(d_first_moment, gpu_first_moment);
        copy_actor_weights_to_host(d_second_moment, gpu_second_moment);
        actor_free(d_second_moment);
        actor_free(d_first_moment);
        actor_free(d_gradients);
        actor_free(d_weights);
    }
    catch(const std::exception&){
        actor_free(d_second_moment);
        actor_free(d_first_moment);
        actor_free(d_gradients);
        actor_free(d_weights);
        throw;
    }

    ActorWeightsHost cpu_weights = initial_weights;
    ActorWeightsHost cpu_first_moment;
    ActorWeightsHost cpu_second_moment;
    zero_actor_weights_host(cpu_first_moment);
    zero_actor_weights_host(cpu_second_moment);
    auto update_host = [&](std::vector<float>& parameters, const std::vector<float>& grad, std::vector<float>& first, std::vector<float>& second){
        for(std::size_t i = 0; i < parameters.size(); i++){
            const float gradient = grad[i];
            const float m = beta1 * first[i] + (1.0f - beta1) * gradient;
            const float v = beta2 * second[i] + (1.0f - beta2) * gradient * gradient;
            first[i] = m;
            second[i] = v;
            const float m_hat = m / (1.0f - std::pow(beta1, static_cast<float>(step)));
            const float v_hat = v / (1.0f - std::pow(beta2, static_cast<float>(step)));
            parameters[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    };
    update_host(cpu_weights.encoder_w, gradients.encoder_w, cpu_first_moment.encoder_w, cpu_second_moment.encoder_w);
    update_host(cpu_weights.encoder_b, gradients.encoder_b, cpu_first_moment.encoder_b, cpu_second_moment.encoder_b);
    update_host(cpu_weights.gru_w_input, gradients.gru_w_input, cpu_first_moment.gru_w_input, cpu_second_moment.gru_w_input);
    update_host(cpu_weights.gru_b_input, gradients.gru_b_input, cpu_first_moment.gru_b_input, cpu_second_moment.gru_b_input);
    update_host(cpu_weights.gru_w_hidden, gradients.gru_w_hidden, cpu_first_moment.gru_w_hidden, cpu_second_moment.gru_w_hidden);
    update_host(cpu_weights.gru_b_hidden, gradients.gru_b_hidden, cpu_first_moment.gru_b_hidden, cpu_second_moment.gru_b_hidden);
    update_host(cpu_weights.gru_h0, gradients.gru_h0, cpu_first_moment.gru_h0, cpu_second_moment.gru_h0);
    update_host(cpu_weights.actor_w, gradients.actor_w, cpu_first_moment.actor_w, cpu_second_moment.actor_w);
    update_host(cpu_weights.actor_b, gradients.actor_b, cpu_first_moment.actor_b, cpu_second_moment.actor_b);
    update_host(cpu_weights.critic_w, gradients.critic_w, cpu_first_moment.critic_w, cpu_second_moment.critic_w);
    update_host(cpu_weights.critic_b, gradients.critic_b, cpu_first_moment.critic_b, cpu_second_moment.critic_b);

    AdamUpdateValidationSummary summary;
    auto compare_vector = [&](const std::vector<float>& gpu_values, const std::vector<float>& cpu_values, float& max_abs, float& max_rel){
        for(std::size_t i = 0; i < gpu_values.size(); i++){
            if(!std::isfinite(gpu_values[i]) || !std::isfinite(cpu_values[i])){
                summary.nan_inf_count++;
                continue;
            }
            const float abs_error = std::abs(gpu_values[i] - cpu_values[i]);
            max_abs = std::max(max_abs, abs_error);
            max_rel = std::max(max_rel, abs_error / std::max(1e-6f, std::abs(cpu_values[i])));
        }
    };
    auto compare_weights = [&](const ActorWeightsHost& gpu_values, const ActorWeightsHost& cpu_values, float& max_abs, float& max_rel){
        compare_vector(gpu_values.encoder_w, cpu_values.encoder_w, max_abs, max_rel);
        compare_vector(gpu_values.encoder_b, cpu_values.encoder_b, max_abs, max_rel);
        compare_vector(gpu_values.gru_w_input, cpu_values.gru_w_input, max_abs, max_rel);
        compare_vector(gpu_values.gru_b_input, cpu_values.gru_b_input, max_abs, max_rel);
        compare_vector(gpu_values.gru_w_hidden, cpu_values.gru_w_hidden, max_abs, max_rel);
        compare_vector(gpu_values.gru_b_hidden, cpu_values.gru_b_hidden, max_abs, max_rel);
        compare_vector(gpu_values.gru_h0, cpu_values.gru_h0, max_abs, max_rel);
        compare_vector(gpu_values.actor_w, cpu_values.actor_w, max_abs, max_rel);
        compare_vector(gpu_values.actor_b, cpu_values.actor_b, max_abs, max_rel);
        compare_vector(gpu_values.critic_w, cpu_values.critic_w, max_abs, max_rel);
        compare_vector(gpu_values.critic_b, cpu_values.critic_b, max_abs, max_rel);
    };
    compare_weights(gpu_weights, cpu_weights, summary.max_weight_abs_error, summary.max_weight_rel_error);
    compare_weights(gpu_first_moment, cpu_first_moment, summary.max_first_moment_abs_error, summary.max_first_moment_rel_error);
    compare_weights(gpu_second_moment, cpu_second_moment, summary.max_second_moment_abs_error, summary.max_second_moment_rel_error);
    summary.weights_close = summary.max_weight_rel_error < 1e-4f || summary.max_weight_abs_error < 1e-6f;
    summary.moments_close =
        (summary.max_first_moment_rel_error < 1e-5f || summary.max_first_moment_abs_error < 1e-7f) &&
        (summary.max_second_moment_rel_error < 1e-5f || summary.max_second_moment_abs_error < 1e-9f);
    summary.step_close = true;
    summary.passed = summary.nan_inf_count == 0 && summary.weights_close && summary.moments_close && summary.step_close;
    return summary;
}

FullGpuTrainingSummary run_full_gpu_training(
    const FullGpuTrainingOptions& training_options,
    const EulerGpuLossWeights& weights
){
    if(training_options.steps == 0 || training_options.batch_size == 0 || training_options.horizon == 0){
        throw std::runtime_error("Full GPU training requires non-zero steps, batch size, and horizon");
    }
    if(training_options.horizon > CPU_MAX_HORIZON){
        throw std::runtime_error("Full GPU training horizon exceeds current validation horizon limit");
    }

    std::mt19937 actor_rng(training_options.seed + 991u);
    ActorWeightsHost initial_weights;
    ActorWeightsHost initial_first_moment;
    ActorWeightsHost initial_second_moment;
    generate_actor_weights(initial_weights, actor_rng, 0.015f);
    zero_actor_weights_host(initial_first_moment);
    zero_actor_weights_host(initial_second_moment);
    std::uint32_t initial_optimizer_age = 1u;
    bool loaded_optimizer_state = false;
    FullGpuTrainingSummary summary;
    if(!training_options.load_path.empty()){
        if(training_options.load_optimizer_state){
            loaded_optimizer_state = load_actor_checkpoint_with_moments(
                training_options.load_path,
                initial_weights,
                initial_first_moment,
                initial_second_moment,
                initial_optimizer_age
            );
        }
        if(!loaded_optimizer_state && !load_actor_checkpoint(training_options.load_path, initial_weights)){
            throw std::runtime_error("Failed to load CUDA RDAC actor checkpoint: " + training_options.load_path);
        }
        summary.checkpoint_loaded = true;
    }

    std::ofstream log;
    if(!training_options.log_path.empty()){
        log.open(training_options.log_path);
        if(!log){
            throw std::runtime_error("Failed to open GPU training log path: " + training_options.log_path);
        }
        log << "step,horizon,loss,diff_rollout_loss_scaled,critic_loss_raw,critic_loss_weight,critic_loss_scaled,"
            << "critic_output_mean,critic_target_mean,critic_error_mean,critic_error_norm,grad_norm,"
            << "success_rate_batch,action_saturation_rate,final_position_norm_mean,final_velocity_norm_mean,"
            << "final_angular_velocity_norm_mean,finite\n";
    }

    EulerGpuBatch batch;
    DeviceArrays d;
    ActorWeightsDevice d_actor_weights;
    ActorBuffersDevice d_actor_buffers;
    ActorGradientsDevice d_actor_gradients;
    ActorWeightsDevice d_adam_first_moment;
    ActorWeightsDevice d_adam_second_moment;
    try{
        CUDA_CHECK(cudaSetDevice(training_options.device));
        batch.resize(training_options.batch_size, training_options.horizon);
        if(!training_options.sample_dynamics){
            generate_validation_batch(batch, training_options.batch_size, training_options.horizon, training_options.seed, nullptr, true);
        }
        allocate(d, training_options.batch_size, training_options.horizon);
        actor_allocate(d_actor_weights, d_actor_buffers, training_options.batch_size, training_options.horizon);
        actor_allocate(d_actor_gradients, training_options.batch_size);
        actor_allocate_weights(d_adam_first_moment);
        actor_allocate_weights(d_adam_second_moment);
        copy_actor_weights_to_device(initial_weights, d_actor_weights);
        if(loaded_optimizer_state){
            copy_actor_weights_to_device(initial_first_moment, d_adam_first_moment);
            copy_actor_weights_to_device(initial_second_moment, d_adam_second_moment);
        }
        else{
            zero_actor_weights_device(d_adam_first_moment);
            zero_actor_weights_device(d_adam_second_moment);
        }

        const int block = 256;
        const int grid = static_cast<int>((training_options.batch_size + block - 1) / block);
        const std::size_t state_count = (training_options.horizon + 1) * training_options.batch_size;
        const std::size_t step_count = training_options.horizon * training_options.batch_size;
        const std::size_t gradient_count = step_count * RDAC_ACTION_DIM;
        const std::size_t critic_count = step_count * RDAC_CRITIC_DIM;
        std::vector<float> host_loss(training_options.batch_size, 0.0f);
        std::vector<float> host_diagnostics(8, 0.0f);
        ActorWeightsHost host_gradients;
        for(std::size_t step = 0; step < training_options.steps; step++){
            if(training_options.sample_dynamics){
                generate_validation_batch(
                    batch,
                    training_options.batch_size,
                    training_options.horizon,
                    training_options.seed + static_cast<unsigned>(step),
                    nullptr,
                    false,
                    training_options.correlated_size_mass_sampling
                );
            }
            copy_input_to_device(batch, d);
            zero_actor_gradients(d_actor_gradients, training_options.batch_size);
            CUDA_CHECK(cudaMemset(d.lambda_p, 0, sizeof(float) * state_count * 3));
            CUDA_CHECK(cudaMemset(d.lambda_v, 0, sizeof(float) * state_count * 3));
            CUDA_CHECK(cudaMemset(d.lambda_R, 0, sizeof(float) * state_count * 9));
            CUDA_CHECK(cudaMemset(d.lambda_omega, 0, sizeof(float) * state_count * 3));
            CUDA_CHECK(cudaMemset(d.lambda_rpm, 0, sizeof(float) * state_count * 4));
            CUDA_CHECK(cudaMemset(d.action_gradients, 0, sizeof(float) * step_count * 4));
            CUDA_CHECK(cudaMemset(d.loss, 0, sizeof(float) * training_options.batch_size));
            CUDA_CHECK(cudaMemset(d_actor_buffers.raw_action_gradient, 0, sizeof(float) * gradient_count));
            CUDA_CHECK(cudaMemset(d_actor_buffers.critic_output, 0, sizeof(float) * critic_count));
            CUDA_CHECK(cudaMemset(d_actor_buffers.critic_target, 0, sizeof(float) * critic_count));
            CUDA_CHECK(cudaMemset(d_actor_buffers.critic_weight, 0, sizeof(float) * critic_count));
            CUDA_CHECK(cudaMemset(d_actor_buffers.critic_output_gradient, 0, sizeof(float) * critic_count));

            for(std::size_t t = 0; t < training_options.horizon; t++){
                observation_kernel<<<grid, block>>>(d, training_options.batch_size, t);
                CUDA_CHECK(cudaGetLastError());
                build_policy_input_from_observation_kernel<<<grid, block>>>(d, d_actor_buffers, training_options.batch_size, t);
                CUDA_CHECK(cudaGetLastError());
                rdac_actor_forward_step_kernel<<<grid, block>>>(d_actor_weights, d_actor_buffers, training_options.batch_size, t, 1.0f, training_options.reset_hidden_each_step);
                CUDA_CHECK(cudaGetLastError());
                copy_bounded_action_to_rollout_kernel<<<grid, block>>>(d_actor_buffers, d, training_options.batch_size, t);
                CUDA_CHECK(cudaGetLastError());
                forward_step_kernel<<<grid, block>>>(d, training_options.batch_size, t);
                CUDA_CHECK(cudaGetLastError());
            }
            loss_and_action_kernel<<<grid, block>>>(d, weights, training_options.batch_size, training_options.horizon);
            CUDA_CHECK(cudaGetLastError());
            for(std::size_t reverse_i = 0; reverse_i < training_options.horizon; reverse_i++){
                const std::size_t step_i = training_options.horizon - 1 - reverse_i;
                backward_step_kernel<<<grid, block>>>(d, training_options.batch_size, step_i);
                CUDA_CHECK(cudaGetLastError());
            }
            const int grad_grid = static_cast<int>((gradient_count + block - 1) / block);
            inject_action_gradient_kernel<<<grad_grid, block>>>(d, d_actor_buffers, training_options.batch_size, training_options.horizon);
            CUDA_CHECK(cudaGetLastError());
            const float base_scale = training_options.disable_physics_gradient
                ? 0.0f
                : training_options.diff_rollout_loss_weight / std::max<float>(1.0f, static_cast<float>(training_options.batch_size));
            scale_device_buffer(d_actor_buffers.raw_action_gradient, gradient_count, base_scale);
            CUDA_CHECK(cudaDeviceSynchronize());
            if(training_options.action_grad_clip_enabled){
                std::vector<float> clip_probe(gradient_count, 0.0f);
                CUDA_CHECK(cudaMemcpy(clip_probe.data(), d_actor_buffers.raw_action_gradient, sizeof(float) * clip_probe.size(), cudaMemcpyDeviceToHost));
                double norm_sq = 0.0;
                for(float value: clip_probe){
                    norm_sq += static_cast<double>(value) * static_cast<double>(value);
                }
                const float norm = static_cast<float>(std::sqrt(norm_sq));
                if(norm > training_options.action_grad_clip_norm){
                    const float scale = training_options.action_grad_clip_norm / std::max(1e-12f, norm);
                    scale_device_buffer(d_actor_buffers.raw_action_gradient, gradient_count, scale);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }
            const CriticLossMetrics critic_metrics = compute_critic_targets_and_gradients(batch, d, d_actor_buffers, weights);
            for(std::size_t reverse_i = 0; reverse_i < training_options.horizon; reverse_i++){
                const std::size_t step_i = training_options.horizon - 1 - reverse_i;
                rdac_actor_backward_step_kernel<<<grid, block>>>(d_actor_weights, d_actor_buffers, d_actor_gradients, training_options.batch_size, step_i);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(d.diagnostics, 0, sizeof(float) * host_diagnostics.size()));
            rollout_diagnostics_kernel<<<grid, block>>>(
                d,
                training_options.batch_size,
                training_options.horizon,
                training_options.success_position_threshold,
                training_options.success_velocity_threshold,
                training_options.success_angular_velocity_threshold
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(host_loss.data(), d.loss, sizeof(float) * host_loss.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_diagnostics.data(), d.diagnostics, sizeof(float) * host_diagnostics.size(), cudaMemcpyDeviceToHost));
            copy_actor_gradients_to_host(d_actor_gradients, host_gradients);
            double loss_sum = 0.0;
            bool finite_step = true;
            for(float value: host_loss){
                if(!std::isfinite(value)){
                    finite_step = false;
                    summary.nan_inf_count++;
                }
                loss_sum += static_cast<double>(value);
            }
            const float diff_loss_scaled = static_cast<float>(
                (training_options.disable_physics_gradient ? 0.0 : training_options.diff_rollout_loss_weight) *
                loss_sum / std::max<std::size_t>(1, host_loss.size())
            );
            summary.final_critic_loss = critic_metrics.scaled_loss;
            summary.final_critic_error_norm = critic_metrics.error_norm;
            summary.final_critic_output_mean = critic_metrics.output_mean;
            summary.final_critic_target_mean = critic_metrics.target_mean;
            summary.final_loss = diff_loss_scaled + critic_metrics.scaled_loss;
            const float valid_count = std::max(1.0f, host_diagnostics[1]);
            summary.final_success_rate = host_diagnostics[0] / valid_count;
            summary.final_action_saturation = host_diagnostics[4] > 0.0f ? host_diagnostics[3] / host_diagnostics[4] : 0.0f;
            summary.final_position_norm_mean = host_diagnostics[5] / valid_count;
            summary.final_velocity_norm_mean = host_diagnostics[6] / valid_count;
            summary.final_angular_velocity_norm_mean = host_diagnostics[7] / valid_count;
            if(host_diagnostics[2] > 0.0f){
                summary.nan_inf_count += static_cast<std::size_t>(host_diagnostics[2]);
            }
            const float raw_actor_grad_norm = actor_weights_l2_norm(host_gradients);
            bool skip_actor_step = raw_actor_grad_norm > training_options.actor_grad_skip_norm;
            if(training_options.actor_grad_clip_enabled && raw_actor_grad_norm > training_options.actor_grad_clip_norm){
                const float scale = training_options.actor_grad_clip_norm / std::max(training_options.actor_grad_eps, raw_actor_grad_norm);
                scale_actor_weights_host(host_gradients, scale);
                copy_actor_gradients_to_device(host_gradients, d_actor_gradients);
            }
            summary.final_grad_norm = actor_weights_l2_norm(host_gradients);
            finite_step = finite_step && actor_weights_finite(host_gradients) && critic_metrics.finite;
            if(!finite_step){
                summary.finite = false;
                if(log){
                    log << step << "," << training_options.horizon << "," << summary.final_loss << "," << diff_loss_scaled << ","
                        << critic_metrics.raw_loss << "," << critic_metrics.weight << "," << critic_metrics.scaled_loss << ","
                        << critic_metrics.output_mean << "," << critic_metrics.target_mean << ","
                        << critic_metrics.error_mean << "," << critic_metrics.error_norm << ","
                        << summary.final_grad_norm << ","
                        << summary.final_success_rate << "," << summary.final_action_saturation << ","
                        << summary.final_position_norm_mean << "," << summary.final_velocity_norm_mean << ","
                        << summary.final_angular_velocity_norm_mean << ",false\n";
                }
                break;
            }
            if(!skip_actor_step){
                adam_update_actor(
                    d_actor_weights,
                    d_actor_gradients,
                    d_adam_first_moment,
                    d_adam_second_moment,
                    training_options.learning_rate,
                    0.9f,
                    0.999f,
                    1e-8f,
                    static_cast<int>(initial_optimizer_age + step)
                );
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            if(log){
                log << step << "," << training_options.horizon << "," << summary.final_loss << "," << diff_loss_scaled << ","
                    << critic_metrics.raw_loss << "," << critic_metrics.weight << "," << critic_metrics.scaled_loss << ","
                    << critic_metrics.output_mean << "," << critic_metrics.target_mean << ","
                    << critic_metrics.error_mean << "," << critic_metrics.error_norm << ","
                    << summary.final_grad_norm << ","
                    << summary.final_success_rate << "," << summary.final_action_saturation << ","
                    << summary.final_position_norm_mean << "," << summary.final_velocity_norm_mean << ","
                    << summary.final_angular_velocity_norm_mean << ",true\n";
            }
            summary.finite = true;
        }

        ActorWeightsHost final_weights;
        ActorWeightsHost final_first_moment;
        ActorWeightsHost final_second_moment;
        copy_actor_weights_to_host(d_actor_weights, final_weights);
        copy_actor_weights_to_host(d_adam_first_moment, final_first_moment);
        copy_actor_weights_to_host(d_adam_second_moment, final_second_moment);
        if(!actor_weights_finite(final_weights)){
            summary.nan_inf_count++;
            summary.finite = false;
        }
        summary.checkpoint_saved = save_actor_checkpoint(
            training_options.save_path,
            final_weights,
            final_first_moment,
            final_second_moment,
            initial_optimizer_age + training_options.steps
        );
        actor_free(d_adam_second_moment);
        actor_free(d_adam_first_moment);
        actor_free(d_actor_gradients);
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
    }
    catch(const std::exception&){
        actor_free(d_adam_second_moment);
        actor_free(d_adam_first_moment);
        actor_free(d_actor_gradients);
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
        throw;
    }
    summary.passed = summary.finite && summary.nan_inf_count == 0;
    return summary;
}

GpuPolicyEvalSummary run_gpu_policy_eval(
    const GpuPolicyEvalOptions& eval_options,
    const EulerGpuLossWeights& weights
){
    if(eval_options.episodes == 0 || eval_options.horizon == 0){
        throw std::runtime_error("GPU eval requires non-zero episodes and horizon");
    }
    if(eval_options.horizon > GPU_EVAL_MAX_HORIZON){
        throw std::runtime_error("GPU eval horizon exceeds GPU_EVAL_MAX_HORIZON");
    }
    ActorWeightsHost actor_weights;
    GpuPolicyEvalSummary summary;
    summary.checkpoint_loaded = load_actor_checkpoint(eval_options.load_path, actor_weights);
    if(!summary.checkpoint_loaded){
        throw std::runtime_error("Failed to load CUDA eval checkpoint: " + eval_options.load_path);
    }

    EulerGpuBatch batch;
    generate_validation_batch(
        batch,
        eval_options.episodes,
        eval_options.horizon,
        eval_options.seed,
        eval_options.forced_bins.enabled ? &eval_options.forced_bins : nullptr,
        !eval_options.sample_dynamics && !eval_options.forced_bins.enabled,
        eval_options.correlated_size_mass_sampling
    );

    DeviceArrays d;
    ActorWeightsDevice d_actor_weights;
    ActorBuffersDevice d_actor_buffers;
    try{
        CUDA_CHECK(cudaSetDevice(eval_options.device));
        allocate(d, eval_options.episodes, eval_options.horizon);
        actor_allocate(d_actor_weights, d_actor_buffers, eval_options.episodes, eval_options.horizon);
        copy_actor_weights_to_device(actor_weights, d_actor_weights);
        copy_input_to_device(batch, d);
        const int block = 256;
        const int grid = static_cast<int>((eval_options.episodes + block - 1) / block);
        const std::size_t state_count = (eval_options.horizon + 1) * eval_options.episodes;
        const std::size_t step_count = eval_options.horizon * eval_options.episodes;
        const std::size_t gradient_count = step_count * RDAC_ACTION_DIM;
        CUDA_CHECK(cudaMemset(d.lambda_p, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_v, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_R, 0, sizeof(float) * state_count * 9));
        CUDA_CHECK(cudaMemset(d.lambda_omega, 0, sizeof(float) * state_count * 3));
        CUDA_CHECK(cudaMemset(d.lambda_rpm, 0, sizeof(float) * state_count * 4));
        CUDA_CHECK(cudaMemset(d.action_gradients, 0, sizeof(float) * gradient_count));
        CUDA_CHECK(cudaMemset(d.loss, 0, sizeof(float) * eval_options.episodes));
        CUDA_CHECK(cudaMemset(d_actor_buffers.raw_action_gradient, 0, sizeof(float) * gradient_count));
        for(std::size_t t = 0; t < eval_options.horizon; t++){
            observation_kernel<<<grid, block>>>(d, eval_options.episodes, t);
            CUDA_CHECK(cudaGetLastError());
            build_policy_input_from_observation_kernel<<<grid, block>>>(d, d_actor_buffers, eval_options.episodes, t);
            CUDA_CHECK(cudaGetLastError());
            rdac_actor_forward_step_kernel<<<grid, block>>>(d_actor_weights, d_actor_buffers, eval_options.episodes, t, 1.0f, eval_options.reset_hidden_each_step);
            CUDA_CHECK(cudaGetLastError());
            copy_bounded_action_to_rollout_kernel<<<grid, block>>>(d_actor_buffers, d, eval_options.episodes, t);
            CUDA_CHECK(cudaGetLastError());
            forward_step_kernel<<<grid, block>>>(d, eval_options.episodes, t);
            CUDA_CHECK(cudaGetLastError());
        }
        loss_and_action_kernel<<<grid, block>>>(d, weights, eval_options.episodes, eval_options.horizon);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemset(d.diagnostics, 0, sizeof(float) * 8));
        rollout_diagnostics_kernel<<<grid, block>>>(
            d,
            eval_options.episodes,
            eval_options.horizon,
            eval_options.success_position_threshold,
            eval_options.success_velocity_threshold,
            eval_options.success_angular_velocity_threshold
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        EulerGpuResult result;
        copy_output_to_host(batch, d, result);
        std::vector<float> host_diagnostics(8, 0.0f);
        CUDA_CHECK(cudaMemcpy(host_diagnostics.data(), d.diagnostics, sizeof(float) * host_diagnostics.size(), cudaMemcpyDeviceToHost));
        std::vector<float> host_p(state_count * 3, 0.0f);
        std::vector<float> host_v(state_count * 3, 0.0f);
        std::vector<float> host_omega(state_count * 3, 0.0f);
        std::vector<float> host_actions(step_count * RDAC_ACTION_DIM, 0.0f);
        CUDA_CHECK(cudaMemcpy(host_p.data(), d.p, sizeof(float) * host_p.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_v.data(), d.v, sizeof(float) * host_v.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_omega.data(), d.omega, sizeof(float) * host_omega.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_actions.data(), d.actions, sizeof(float) * host_actions.size(), cudaMemcpyDeviceToHost));
        const float valid_count = host_diagnostics[1];
        const float invalid_count = host_diagnostics[2];
        summary.success_rate = valid_count > 0.0f ? host_diagnostics[0] / valid_count : 0.0f;
        summary.action_saturation_rate = host_diagnostics[4] > 0.0f ? host_diagnostics[3] / host_diagnostics[4] : 0.0f;
        summary.invalid_or_nan_rate = (invalid_count + (static_cast<float>(eval_options.episodes) - valid_count - invalid_count)) /
            std::max(1.0f, static_cast<float>(eval_options.episodes));
        summary.nan_inf_count = static_cast<std::size_t>(invalid_count);

        std::vector<float> p_norms;
        std::vector<float> v_norms;
        std::vector<float> w_norms;
        std::vector<float> max_p_norms;
        std::vector<float> max_v_norms;
        std::vector<float> max_w_norms;
        p_norms.reserve(eval_options.episodes);
        v_norms.reserve(eval_options.episodes);
        w_norms.reserve(eval_options.episodes);
        max_p_norms.reserve(eval_options.episodes);
        max_v_norms.reserve(eval_options.episodes);
        max_w_norms.reserve(eval_options.episodes);
        double loss_sum = 0.0;
        std::size_t loss_count = 0;
        std::size_t near_p_count = 0;
        std::size_t near_pv_count = 0;
        std::size_t valid_trajectory_count = 0;
        std::size_t throughout_success_count = 0;
        std::size_t first_failure_count = 0;
        double inside_fraction_sum = 0.0;
        double first_failure_time_sum = 0.0;
        float max_action_abs = 0.0f;
        for(std::size_t t = 0; t < eval_options.horizon; t++){
            for(std::size_t b = 0; b < eval_options.episodes; b++){
                for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                    const float action = host_actions[idx4(t, b, a, eval_options.episodes)];
                    if(std::isfinite(action)){
                        max_action_abs = std::max(max_action_abs, std::abs(action));
                    }
                }
            }
        }
        for(std::size_t b = 0; b < eval_options.episodes; b++){
            double p_sq = 0.0;
            double v_sq = 0.0;
            double w_sq = 0.0;
            bool finite = true;
            for(int dim = 0; dim < 3; dim++){
                const float p = result.final_p[pidx3(b, dim)];
                const float v = result.final_v[pidx3(b, dim)];
                const float w = result.final_omega[pidx3(b, dim)];
                finite = finite && std::isfinite(p) && std::isfinite(v) && std::isfinite(w);
                p_sq += static_cast<double>(p) * p;
                v_sq += static_cast<double>(v) * v;
                w_sq += static_cast<double>(w) * w;
            }
            finite = finite && std::isfinite(result.loss[b]);
            if(!finite){
                summary.nan_inf_count++;
                continue;
            }
            const float p_norm = static_cast<float>(std::sqrt(p_sq));
            const float v_norm = static_cast<float>(std::sqrt(v_sq));
            const float w_norm = static_cast<float>(std::sqrt(w_sq));
            p_norms.push_back(p_norm);
            v_norms.push_back(v_norm);
            w_norms.push_back(w_norm);
            near_p_count += p_norm < eval_options.success_position_threshold ? 1u : 0u;
            near_pv_count += (p_norm < eval_options.success_position_threshold && v_norm < eval_options.success_velocity_threshold) ? 1u : 0u;
            loss_sum += result.loss[b];
            loss_count++;

            float max_p_norm = 0.0f;
            float max_v_norm = 0.0f;
            float max_w_norm = 0.0f;
            std::size_t inside_steps = 0;
            bool failed = false;
            std::size_t first_failure_step = eval_options.horizon + 1;
            for(std::size_t t = 0; t <= eval_options.horizon; t++){
                double step_p_sq = 0.0;
                double step_v_sq = 0.0;
                double step_w_sq = 0.0;
                bool step_finite = true;
                for(int dim = 0; dim < 3; dim++){
                    const float p = host_p[idx3(t, b, dim, eval_options.episodes)];
                    const float v = host_v[idx3(t, b, dim, eval_options.episodes)];
                    const float w = host_omega[idx3(t, b, dim, eval_options.episodes)];
                    step_finite = step_finite && std::isfinite(p) && std::isfinite(v) && std::isfinite(w);
                    step_p_sq += static_cast<double>(p) * p;
                    step_v_sq += static_cast<double>(v) * v;
                    step_w_sq += static_cast<double>(w) * w;
                }
                const float step_p_norm = step_finite ? static_cast<float>(std::sqrt(step_p_sq)) : std::numeric_limits<float>::infinity();
                const float step_v_norm = step_finite ? static_cast<float>(std::sqrt(step_v_sq)) : std::numeric_limits<float>::infinity();
                const float step_w_norm = step_finite ? static_cast<float>(std::sqrt(step_w_sq)) : std::numeric_limits<float>::infinity();
                max_p_norm = std::max(max_p_norm, step_p_norm);
                max_v_norm = std::max(max_v_norm, step_v_norm);
                max_w_norm = std::max(max_w_norm, step_w_norm);
                const bool inside = step_finite &&
                    step_p_norm < eval_options.success_position_threshold &&
                    step_v_norm < eval_options.success_velocity_threshold &&
                    step_w_norm < eval_options.success_angular_velocity_threshold;
                inside_steps += inside ? 1u : 0u;
                if(!inside && !failed){
                    failed = true;
                    first_failure_step = t;
                }
            }
            valid_trajectory_count++;
            max_p_norms.push_back(max_p_norm);
            max_v_norms.push_back(max_v_norm);
            max_w_norms.push_back(max_w_norm);
            inside_fraction_sum += static_cast<double>(inside_steps) / static_cast<double>(eval_options.horizon + 1);
            if(!failed){
                throughout_success_count++;
            }
            else{
                first_failure_count++;
                first_failure_time_sum += static_cast<double>(first_failure_step) * static_cast<double>(DT);
            }
        }
        auto mean = [](const std::vector<float>& values){
            double sum = 0.0;
            for(float value: values){ sum += value; }
            return values.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(values.size()));
        };
        auto percentile = [](std::vector<float> values, double q){
            if(values.empty()) return 0.0f;
            std::sort(values.begin(), values.end());
            const std::size_t index = static_cast<std::size_t>(std::min<double>(values.size() - 1, std::floor(q * static_cast<double>(values.size() - 1))));
            return values[index];
        };
        summary.mean_total_loss = loss_count > 0 ? static_cast<float>(loss_sum / static_cast<double>(loss_count)) : 0.0f;
        summary.mean_final_position_norm = mean(p_norms);
        summary.mean_final_velocity_norm = mean(v_norms);
        summary.mean_final_angular_velocity_norm = mean(w_norms);
        summary.median_final_position_norm = percentile(p_norms, 0.5);
        summary.median_final_velocity_norm = percentile(v_norms, 0.5);
        summary.median_final_angular_velocity_norm = percentile(w_norms, 0.5);
        summary.p90_final_position_norm = percentile(p_norms, 0.9);
        summary.p90_final_velocity_norm = percentile(v_norms, 0.9);
        summary.p90_final_angular_velocity_norm = percentile(w_norms, 0.9);
        const float valid_trajectory_float = std::max(1.0f, static_cast<float>(valid_trajectory_count));
        summary.throughout_success_rate = static_cast<float>(throughout_success_count) / valid_trajectory_float;
        summary.mean_time_inside_fraction = valid_trajectory_count > 0
            ? static_cast<float>(inside_fraction_sum / static_cast<double>(valid_trajectory_count))
            : 0.0f;
        summary.mean_first_failure_time_s = first_failure_count > 0
            ? static_cast<float>(first_failure_time_sum / static_cast<double>(first_failure_count))
            : static_cast<float>(eval_options.horizon) * DT;
        summary.mean_max_position_norm = mean(max_p_norms);
        summary.mean_max_velocity_norm = mean(max_v_norms);
        summary.mean_max_angular_velocity_norm = mean(max_w_norms);
        summary.p90_max_position_norm = percentile(max_p_norms, 0.9);
        summary.p90_max_velocity_norm = percentile(max_v_norms, 0.9);
        summary.p90_max_angular_velocity_norm = percentile(max_w_norms, 0.9);
        summary.max_action_abs = max_action_abs;
        const float valid_eval_count = std::max(1.0f, static_cast<float>(p_norms.size()));
        summary.near_success_rate_p = static_cast<float>(near_p_count) / valid_eval_count;
        summary.near_success_rate_pv = static_cast<float>(near_pv_count) / valid_eval_count;
        summary.finite = summary.nan_inf_count == 0;
        summary.passed = summary.checkpoint_loaded && summary.finite;

        if(!eval_options.log_path.empty()){
            std::ofstream log(eval_options.log_path);
            if(!log){
                throw std::runtime_error("Failed to open GPU eval log path: " + eval_options.log_path);
            }
            log << "eval_model,mode_fixed_or_sampled,eval_dynamics_mode,eval_sampled_dynamics_level,eval_episodes,eval_horizon,"
                << "success_rate,near_success_rate_p,near_success_rate_pv,mean_total_loss,mean_final_position_norm,"
                << "mean_final_velocity_norm,mean_final_angular_velocity_norm,median_final_position_norm,"
                << "median_final_velocity_norm,median_final_angular_velocity_norm,p90_final_position_norm,"
                << "p90_final_velocity_norm,p90_final_angular_velocity_norm,throughout_success_rate,mean_time_inside_fraction,"
                << "mean_first_failure_time_s,mean_max_position_norm,mean_max_velocity_norm,mean_max_angular_velocity_norm,"
                << "p90_max_position_norm,p90_max_velocity_norm,p90_max_angular_velocity_norm,max_action_abs,"
                << "action_saturation_rate,invalid_or_nan_rate,"
                << "forced_bins_enabled,size_mass_bin,thrust_to_weight_bin,torque_to_inertia_bin,motor_delay_bin,curve_shape_bin\n";
            log << "euler,"
                << (eval_options.sample_dynamics ? "sampled" : "fixed") << ","
                << (eval_options.sample_dynamics ? "sampled" : "fixed") << ",broad,"
                << eval_options.episodes << "," << eval_options.horizon << ","
                << summary.success_rate << "," << summary.near_success_rate_p << "," << summary.near_success_rate_pv << ","
                << summary.mean_total_loss << "," << summary.mean_final_position_norm << ","
                << summary.mean_final_velocity_norm << "," << summary.mean_final_angular_velocity_norm << ","
                << summary.median_final_position_norm << "," << summary.median_final_velocity_norm << ","
                << summary.median_final_angular_velocity_norm << "," << summary.p90_final_position_norm << ","
                << summary.p90_final_velocity_norm << "," << summary.p90_final_angular_velocity_norm << ","
                << summary.throughout_success_rate << "," << summary.mean_time_inside_fraction << ","
                << summary.mean_first_failure_time_s << "," << summary.mean_max_position_norm << ","
                << summary.mean_max_velocity_norm << "," << summary.mean_max_angular_velocity_norm << ","
                << summary.p90_max_position_norm << "," << summary.p90_max_velocity_norm << ","
                << summary.p90_max_angular_velocity_norm << "," << summary.max_action_abs << ","
                << summary.action_saturation_rate << "," << summary.invalid_or_nan_rate << ","
                << (eval_options.forced_bins.enabled ? "true" : "false") << ","
                << eval_options.forced_bins.size_mass << "," << eval_options.forced_bins.thrust_to_weight << ","
                << eval_options.forced_bins.torque_to_inertia << "," << eval_options.forced_bins.motor_delay << ","
                << eval_options.forced_bins.curve_shape << "\n";
        }

        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
    }
    catch(const std::exception&){
        actor_free(d_actor_buffers);
        actor_free(d_actor_weights);
        cuda_free(d);
        throw;
    }
    return summary;
}

Stage9ReplayDebugSummary run_stage9_replay_debug(
    const Stage9ReplayDebugOptions& debug_options,
    const EulerGpuLossWeights& weights
){
    if(debug_options.steps == 0 || debug_options.batch_size == 0 || debug_options.horizon == 0){
        throw std::runtime_error("Stage9 replay debug requires non-zero steps, batch size, and horizon");
    }
    if(debug_options.horizon > CPU_MAX_HORIZON){
        throw std::runtime_error("Stage9 replay debug horizon exceeds CPU_MAX_HORIZON");
    }

    Stage9ReplayDebugSummary summary;
    summary.replay_written = write_stage9_replay_file(
        debug_options.replay_path,
        debug_options.steps,
        debug_options.batch_size,
        debug_options.horizon,
        debug_options.seed
    );
    std::ifstream replay_in;
    summary.replay_loaded = open_stage9_replay_file(
        debug_options.replay_path,
        replay_in,
        debug_options.steps,
        debug_options.batch_size,
        debug_options.horizon
    );
    if(!debug_options.replay_path.empty() && (!summary.replay_written || !summary.replay_loaded)){
        throw std::runtime_error("Stage9 replay debug failed to write/read replay file: " + debug_options.replay_path);
    }

    std::ofstream log;
    if(!debug_options.log_path.empty()){
        log.open(debug_options.log_path);
        if(!log){
            throw std::runtime_error("Failed to open Stage9 replay debug log: " + debug_options.log_path);
        }
        log << "step,cpu_loss,gpu_loss,loss_abs_error,"
            << "cpu_action_mean,gpu_action_mean,action_mean_abs_error,"
            << "cpu_action_std,gpu_action_std,action_std_abs_error,"
            << "cpu_action_saturation,gpu_action_saturation,action_saturation_abs_error,"
            << "cpu_action_gradient_norm,gpu_action_gradient_norm,action_gradient_norm_abs_error,"
            << "cpu_raw_action_gradient_norm,gpu_raw_action_gradient_norm,raw_action_gradient_norm_abs_error,"
            << "cpu_actor_gradient_norm,gpu_actor_gradient_norm,actor_gradient_norm_abs_error,"
            << "cpu_weight_delta_norm,gpu_weight_delta_norm,weight_delta_norm_abs_error,"
            << "cpu_adam_m_norm,gpu_adam_m_norm,adam_m_norm_abs_error,"
            << "cpu_adam_v_norm,gpu_adam_v_norm,adam_v_norm_abs_error,"
            << "weight_max_abs_error,weight_l2_rel_error,cpu_finite,gpu_finite\n";
    }

    std::mt19937 actor_rng(debug_options.seed + 991u);
    ActorWeightsHost cpu_weights;
    ActorWeightsHost gpu_initial_weights;
    generate_actor_weights(cpu_weights, actor_rng, 0.015f);
    gpu_initial_weights = cpu_weights;
    ActorWeightsHost cpu_adam_m;
    ActorWeightsHost cpu_adam_v;
    zero_actor_weights_host(cpu_adam_m);
    zero_actor_weights_host(cpu_adam_v);

    EulerGpuBatch batch;
    batch.resize(debug_options.batch_size, debug_options.horizon);
    DeviceArrays d;
    ActorWeightsDevice d_weights;
    ActorBuffersDevice d_buffers;
    ActorGradientsDevice d_gradients;
    ActorWeightsDevice d_adam_m;
    ActorWeightsDevice d_adam_v;
    try{
        CUDA_CHECK(cudaSetDevice(debug_options.device));
        allocate(d, debug_options.batch_size, debug_options.horizon);
        actor_allocate(d_weights, d_buffers, debug_options.batch_size, debug_options.horizon);
        actor_allocate(d_gradients, debug_options.batch_size);
        actor_allocate_weights(d_adam_m);
        actor_allocate_weights(d_adam_v);
        copy_actor_weights_to_device(gpu_initial_weights, d_weights);
        zero_actor_weights_device(d_adam_m);
        zero_actor_weights_device(d_adam_v);

        for(std::size_t step = 0; step < debug_options.steps; step++){
            if(summary.replay_loaded){
                read_replay_batch(replay_in, batch);
                if(!replay_in){
                    throw std::runtime_error("Failed to read Stage9 replay batch at step " + std::to_string(step));
                }
            }
            else{
                generate_validation_batch(batch, debug_options.batch_size, debug_options.horizon, debug_options.seed + static_cast<unsigned>(step));
            }

            const auto cpu_metrics = cpu_stage9_replay_step(
                batch,
                weights,
                debug_options,
                cpu_weights,
                cpu_adam_m,
                cpu_adam_v,
                step + 1
            );
            const auto gpu_metrics = gpu_stage9_replay_step(
                batch,
                weights,
                debug_options,
                d,
                d_weights,
                d_buffers,
                d_gradients,
                d_adam_m,
                d_adam_v,
                step + 1
            );

            ActorWeightsHost gpu_weights_after;
            copy_actor_weights_to_host(d_weights, gpu_weights_after);
            const float weight_max_abs_error = actor_weights_max_abs_error(gpu_weights_after, cpu_weights);
            const float weight_l2_rel_error = actor_weights_l2_rel_error(gpu_weights_after, cpu_weights);

            const float loss_error = std::abs(gpu_metrics.loss - cpu_metrics.loss);
            const float action_mean_error = std::abs(gpu_metrics.action_mean - cpu_metrics.action_mean);
            const float action_std_error = std::abs(gpu_metrics.action_std - cpu_metrics.action_std);
            const float action_saturation_error = std::abs(gpu_metrics.action_saturation - cpu_metrics.action_saturation);
            const float action_gradient_norm_error = std::abs(gpu_metrics.action_gradient_norm - cpu_metrics.action_gradient_norm);
            const float actor_gradient_norm_error = std::abs(gpu_metrics.actor_gradient_norm - cpu_metrics.actor_gradient_norm);
            const float adam_m_norm_error = std::abs(gpu_metrics.adam_m_norm - cpu_metrics.adam_m_norm);
            const float adam_v_norm_error = std::abs(gpu_metrics.adam_v_norm - cpu_metrics.adam_v_norm);

            summary.final_cpu_loss = cpu_metrics.loss;
            summary.final_gpu_loss = gpu_metrics.loss;
            summary.max_loss_abs_error = std::max(summary.max_loss_abs_error, loss_error);
            summary.max_action_mean_abs_error = std::max(summary.max_action_mean_abs_error, action_mean_error);
            summary.max_action_std_abs_error = std::max(summary.max_action_std_abs_error, action_std_error);
            summary.max_action_saturation_abs_error = std::max(summary.max_action_saturation_abs_error, action_saturation_error);
            summary.max_action_gradient_norm_abs_error = std::max(summary.max_action_gradient_norm_abs_error, action_gradient_norm_error);
            summary.max_actor_gradient_norm_abs_error = std::max(summary.max_actor_gradient_norm_abs_error, actor_gradient_norm_error);
            summary.max_weight_abs_error = std::max(summary.max_weight_abs_error, weight_max_abs_error);
            summary.max_weight_l2_rel_error = std::max(summary.max_weight_l2_rel_error, weight_l2_rel_error);
            summary.max_adam_m_norm_abs_error = std::max(summary.max_adam_m_norm_abs_error, adam_m_norm_error);
            summary.max_adam_v_norm_abs_error = std::max(summary.max_adam_v_norm_abs_error, adam_v_norm_error);

            const bool finite_step = cpu_metrics.finite &&
                gpu_metrics.finite &&
                actor_weights_finite(cpu_weights) &&
                actor_weights_finite(gpu_weights_after);
            if(!finite_step){
                summary.nan_inf_count++;
            }
            if(log){
                log << step << ","
                    << cpu_metrics.loss << "," << gpu_metrics.loss << "," << loss_error << ","
                    << cpu_metrics.action_mean << "," << gpu_metrics.action_mean << "," << action_mean_error << ","
                    << cpu_metrics.action_std << "," << gpu_metrics.action_std << "," << action_std_error << ","
                    << cpu_metrics.action_saturation << "," << gpu_metrics.action_saturation << "," << action_saturation_error << ","
                    << cpu_metrics.action_gradient_norm << "," << gpu_metrics.action_gradient_norm << "," << action_gradient_norm_error << ","
                    << cpu_metrics.raw_action_gradient_norm << "," << gpu_metrics.raw_action_gradient_norm << "," << std::abs(gpu_metrics.raw_action_gradient_norm - cpu_metrics.raw_action_gradient_norm) << ","
                    << cpu_metrics.actor_gradient_norm << "," << gpu_metrics.actor_gradient_norm << "," << actor_gradient_norm_error << ","
                    << cpu_metrics.weight_delta_norm << "," << gpu_metrics.weight_delta_norm << "," << std::abs(gpu_metrics.weight_delta_norm - cpu_metrics.weight_delta_norm) << ","
                    << cpu_metrics.adam_m_norm << "," << gpu_metrics.adam_m_norm << "," << adam_m_norm_error << ","
                    << cpu_metrics.adam_v_norm << "," << gpu_metrics.adam_v_norm << "," << adam_v_norm_error << ","
                    << weight_max_abs_error << "," << weight_l2_rel_error << ","
                    << (cpu_metrics.finite ? "true" : "false") << ","
                    << (gpu_metrics.finite ? "true" : "false") << "\n";
            }
        }

        actor_free(d_adam_v);
        actor_free(d_adam_m);
        actor_free(d_gradients);
        actor_free(d_buffers);
        actor_free(d_weights);
        cuda_free(d);
    }
    catch(const std::exception&){
        actor_free(d_adam_v);
        actor_free(d_adam_m);
        actor_free(d_gradients);
        actor_free(d_buffers);
        actor_free(d_weights);
        cuda_free(d);
        throw;
    }

    summary.finite = summary.nan_inf_count == 0;
    summary.close =
        summary.max_loss_abs_error < 1e-2f &&
        summary.max_action_mean_abs_error < 1e-4f &&
        summary.max_action_std_abs_error < 1e-4f &&
        summary.max_action_saturation_abs_error < 1e-6f &&
        summary.max_weight_l2_rel_error < 1e-3f;
    summary.passed = summary.finite && summary.close;
    return summary;
}

Stage9SamplerParitySummary run_stage9_sampler_parity(
    const Stage9ReplayDebugOptions& debug_options
){
    Stage9SamplerParitySummary summary;
    const std::string replay_path = debug_options.replay_path.empty()
        ? std::string("/tmp/stage9_6_sampler_parity_replay.bin")
        : debug_options.replay_path;
    summary.replay_written = write_stage9_replay_file(
        replay_path,
        debug_options.steps,
        debug_options.batch_size,
        debug_options.horizon,
        debug_options.seed
    );
    std::ifstream replay_in;
    summary.replay_loaded = open_stage9_replay_file(
        replay_path,
        replay_in,
        debug_options.steps,
        debug_options.batch_size,
        debug_options.horizon
    );
    if(!summary.replay_written || !summary.replay_loaded){
        return summary;
    }

    constexpr std::size_t bins = 4;
    std::uint64_t bin_counts[5][bins] = {};
    EulerGpuBatch batch;
    batch.resize(debug_options.batch_size, debug_options.horizon);
    summary.group_weight_sum_min = std::numeric_limits<float>::infinity();
    summary.group_weight_sum_max = -std::numeric_limits<float>::infinity();
    for(std::size_t step = 0; step < debug_options.steps; step++){
        read_replay_batch(replay_in, batch);
        if(!replay_in){
            summary.metadata_mismatch_count++;
            break;
        }
        summary.metadata_present = summary.metadata_present ||
            (batch.replay_schema_version == 2u && batch.sampler_balance_bins == bins);
        if(batch.replay_schema_version != 2u || batch.sampler_balance_bins != bins){
            summary.metadata_mismatch_count++;
        }
        std::vector<std::uint32_t> unique_groups;
        for(std::size_t b = 0; b < batch.batch_size; b++){
            const std::uint32_t values[5] = {
                batch.dynamics_size_mass_bin[b],
                batch.dynamics_thrust_to_weight_bin[b],
                batch.dynamics_torque_to_inertia_bin[b],
                batch.dynamics_motor_delay_bin[b],
                batch.dynamics_curve_shape_bin[b]
            };
            for(std::size_t dim = 0; dim < 5; dim++){
                if(values[dim] >= bins){
                    summary.metadata_mismatch_count++;
                }
                else{
                    bin_counts[dim][values[dim]]++;
                }
            }
            const std::uint32_t expected_group =
                (((values[0] * bins + values[1]) * bins + values[2]) * bins + values[3]) * bins + values[4];
            if(expected_group != batch.dynamics_group_key[b]){
                summary.metadata_mismatch_count++;
            }
            if(!std::isfinite(batch.group_weight[b])){
                summary.nan_inf_count++;
            }
            summary.rejected_total += batch.rejected_before_accept[b];
            bool seen = false;
            for(std::uint32_t key: unique_groups){
                if(key == batch.dynamics_group_key[b]){
                    seen = true;
                    break;
                }
            }
            if(!seen){
                unique_groups.push_back(batch.dynamics_group_key[b]);
            }
        }
        summary.groups += unique_groups.size();
        for(std::uint32_t key: unique_groups){
            float group_sum = 0.0f;
            for(std::size_t b = 0; b < batch.batch_size; b++){
                if(batch.dynamics_group_key[b] == key){
                    group_sum += batch.group_weight[b];
                }
            }
            summary.group_weight_sum_min = std::min(summary.group_weight_sum_min, group_sum);
            summary.group_weight_sum_max = std::max(summary.group_weight_sum_max, group_sum);
        }
        for(std::size_t t = 0; t < batch.horizon; t++){
            for(std::size_t b = 0; b < batch.batch_size; b++){
                const std::size_t i = t * batch.batch_size + b;
                summary.reset_mask_count += batch.reset_mask[i] != 0u ? 1 : 0;
                summary.hidden_reset_mask_count += batch.hidden_reset_mask[i] != 0u ? 1 : 0;
                const std::uint32_t expected_reset = t == 0 ? 1u : 0u;
                if(batch.reset_mask[i] != expected_reset){
                    summary.metadata_mismatch_count++;
                }
            }
        }
        summary.samples += batch.batch_size;
    }
    bool balanced = true;
    for(std::size_t dim = 0; dim < 5; dim++){
        std::uint64_t min_count = bin_counts[dim][0];
        std::uint64_t max_count = bin_counts[dim][0];
        for(std::size_t bin = 1; bin < bins; bin++){
            min_count = std::min(min_count, bin_counts[dim][bin]);
            max_count = std::max(max_count, bin_counts[dim][bin]);
        }
        balanced = balanced && min_count == max_count && min_count > 0;
    }
    if(!std::isfinite(summary.group_weight_sum_min)){
        summary.group_weight_sum_min = 0.0f;
        summary.group_weight_sum_max = 0.0f;
    }
    summary.bins_balanced = balanced;
    summary.group_weights_close = std::abs(summary.group_weight_sum_max - summary.group_weight_sum_min) < 1e-6f;
    summary.reset_masks_replayed = summary.reset_mask_count == debug_options.steps * debug_options.batch_size;
    summary.finite = summary.nan_inf_count == 0;
    summary.passed = summary.replay_written &&
        summary.replay_loaded &&
        summary.metadata_present &&
        summary.bins_balanced &&
        summary.group_weights_close &&
        summary.reset_masks_replayed &&
        summary.metadata_mismatch_count == 0 &&
        summary.finite;
    return summary;
}

Stage9EvalParitySummary run_stage9_eval_parity(
    const Stage9ReplayDebugOptions& debug_options,
    const EulerGpuLossWeights& weights
){
    Stage9EvalParitySummary summary;
    const std::string replay_path = debug_options.replay_path.empty()
        ? std::string("/tmp/stage9_6_eval_parity_replay.bin")
        : debug_options.replay_path;
    summary.replay_written = write_stage9_replay_file(
        replay_path,
        debug_options.steps,
        debug_options.batch_size,
        debug_options.horizon,
        debug_options.seed
    );
    std::ifstream replay_in;
    summary.replay_loaded = open_stage9_replay_file(
        replay_path,
        replay_in,
        debug_options.steps,
        debug_options.batch_size,
        debug_options.horizon
    );
    if(!summary.replay_written || !summary.replay_loaded){
        return summary;
    }
    EulerGpuRunOptions run_options;
    run_options.device = debug_options.device;
    run_options.compute_action_gradients = false;
    EulerGpuBatch batch;
    batch.resize(debug_options.batch_size, debug_options.horizon);
    std::size_t total = 0;
    std::size_t cpu_success_count = 0;
    std::size_t gpu_success_count = 0;
    std::size_t cpu_saturation_count = 0;
    std::size_t gpu_saturation_count = 0;
    double cpu_final_p_sum = 0.0;
    double gpu_final_p_sum = 0.0;
    constexpr float success_p = 1.0f;
    constexpr float success_v = 3.0f;
    constexpr float success_w = 3.0f;
    for(std::size_t step = 0; step < debug_options.steps; step++){
        read_replay_batch(replay_in, batch);
        if(!replay_in){
            summary.nan_inf_count++;
            break;
        }
        EulerGpuResult cpu_result;
        EulerGpuResult gpu_result;
        EulerGpuTimings timings;
        run_cpu_reference(batch, weights, cpu_result);
        if(run_euler_gpu_rollout(batch, weights, run_options, gpu_result, timings) != 0){
            summary.nan_inf_count++;
            break;
        }
        for(std::size_t b = 0; b < batch.batch_size; b++){
            const auto norm3 = [&](const std::vector<float>& values, std::size_t base){
                double sum = 0.0;
                for(std::size_t i = 0; i < 3; i++){
                    sum += static_cast<double>(values[base + i]) * static_cast<double>(values[base + i]);
                }
                return static_cast<float>(std::sqrt(sum));
            };
            const float cpu_p = norm3(cpu_result.final_p, b * 3);
            const float cpu_v = norm3(cpu_result.final_v, b * 3);
            const float cpu_w = norm3(cpu_result.final_omega, b * 3);
            const float gpu_p = norm3(gpu_result.final_p, b * 3);
            const float gpu_v = norm3(gpu_result.final_v, b * 3);
            const float gpu_w = norm3(gpu_result.final_omega, b * 3);
            const bool cpu_success = cpu_p < success_p && cpu_v < success_v && cpu_w < success_w;
            const bool gpu_success = gpu_p < success_p && gpu_v < success_v && gpu_w < success_w;
            cpu_success_count += cpu_success ? 1 : 0;
            gpu_success_count += gpu_success ? 1 : 0;
            summary.success_mismatch_count += cpu_success != gpu_success ? 1 : 0;
            cpu_final_p_sum += cpu_p;
            gpu_final_p_sum += gpu_p;
            for(std::size_t d = 0; d < 3; d++){
                summary.max_final_state_abs_error = std::max(summary.max_final_state_abs_error, std::abs(cpu_result.final_p[b * 3 + d] - gpu_result.final_p[b * 3 + d]));
                summary.max_final_state_abs_error = std::max(summary.max_final_state_abs_error, std::abs(cpu_result.final_v[b * 3 + d] - gpu_result.final_v[b * 3 + d]));
                summary.max_final_state_abs_error = std::max(summary.max_final_state_abs_error, std::abs(cpu_result.final_omega[b * 3 + d] - gpu_result.final_omega[b * 3 + d]));
            }
            for(std::size_t d = 0; d < 9; d++){
                summary.max_final_state_abs_error = std::max(summary.max_final_state_abs_error, std::abs(cpu_result.final_R[b * 9 + d] - gpu_result.final_R[b * 9 + d]));
            }
            for(std::size_t t = 0; t < batch.horizon; t++){
                for(std::size_t a = 0; a < RDAC_ACTION_DIM; a++){
                    const bool cpu_sat = std::abs(batch.actions[idx4(t, b, a, batch.batch_size)]) >= 0.95f;
                    const bool gpu_sat = cpu_sat;
                    cpu_saturation_count += cpu_sat ? 1 : 0;
                    gpu_saturation_count += gpu_sat ? 1 : 0;
                    summary.saturation_mismatch_count += cpu_sat != gpu_sat ? 1 : 0;
                }
            }
            total++;
        }
        for(std::size_t i = 0; i < cpu_result.loss.size(); i++){
            summary.max_loss_abs_error = std::max(summary.max_loss_abs_error, std::abs(cpu_result.loss[i] - gpu_result.loss[i]));
        }
    }
    const float inv_total = total > 0 ? 1.0f / static_cast<float>(total) : 0.0f;
    summary.cpu_success_rate = static_cast<float>(cpu_success_count) * inv_total;
    summary.gpu_success_rate = static_cast<float>(gpu_success_count) * inv_total;
    summary.mean_cpu_final_position_norm = static_cast<float>(cpu_final_p_sum) * inv_total;
    summary.mean_gpu_final_position_norm = static_cast<float>(gpu_final_p_sum) * inv_total;
    const float action_count = static_cast<float>(std::max<std::size_t>(1, total * debug_options.horizon * RDAC_ACTION_DIM));
    summary.cpu_action_saturation = static_cast<float>(cpu_saturation_count) / action_count;
    summary.gpu_action_saturation = static_cast<float>(gpu_saturation_count) / action_count;
    summary.final_state_close = summary.max_final_state_abs_error < 1e-4f;
    summary.loss_close = summary.max_loss_abs_error < 1e-4f;
    summary.success_close = summary.success_mismatch_count == 0 && std::abs(summary.cpu_success_rate - summary.gpu_success_rate) < 1e-6f;
    summary.action_saturation_close = summary.saturation_mismatch_count == 0 && std::abs(summary.cpu_action_saturation - summary.gpu_action_saturation) < 1e-6f;
    summary.finite = summary.nan_inf_count == 0;
    summary.passed = summary.replay_written &&
        summary.replay_loaded &&
        summary.final_state_close &&
        summary.loss_close &&
        summary.success_close &&
        summary.action_saturation_close &&
        summary.finite;
    return summary;
}

Stage9CheckpointParitySummary run_stage9_checkpoint_parity(
    const std::string& checkpoint_path,
    unsigned seed
){
    Stage9CheckpointParitySummary summary;
    const std::string path = checkpoint_path.empty()
        ? std::string("/tmp/stage9_6_cuda_checkpoint_parity_v4.ckpt")
        : checkpoint_path;
    std::mt19937 rng(seed + 173u);
    ActorWeightsHost weights;
    ActorWeightsHost first_moment;
    ActorWeightsHost second_moment;
    generate_actor_weights(weights, rng, 0.02f);
    generate_actor_weights(first_moment, rng, 0.001f);
    generate_actor_weights(second_moment, rng, 0.0001f);
    summary.saved = save_actor_checkpoint(path, weights, first_moment, second_moment, 37u);
    ActorWeightsHost loaded_weights;
    ActorWeightsHost loaded_first_moment;
    ActorWeightsHost loaded_second_moment;
    std::uint32_t optimizer_age = 0;
    summary.loaded = load_actor_checkpoint_with_moments(
        path,
        loaded_weights,
        loaded_first_moment,
        loaded_second_moment,
        optimizer_age
    );
    summary.metadata_ok = optimizer_age == 37u;
    if(summary.loaded){
        summary.max_weight_abs_error = actor_weights_max_abs_error(loaded_weights, weights);
        summary.max_first_moment_abs_error = actor_weights_max_abs_error(loaded_first_moment, first_moment);
        summary.max_second_moment_abs_error = actor_weights_max_abs_error(loaded_second_moment, second_moment);
    }
    summary.weights_close = summary.max_weight_abs_error < 1e-8f;
    summary.moments_close = summary.max_first_moment_abs_error < 1e-8f && summary.max_second_moment_abs_error < 1e-8f;
    summary.passed = summary.saved && summary.loaded && summary.metadata_ok && summary.weights_close && summary.moments_close;
    return summary;
}

ValidationSummary validate_against_cpu(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options
){
    EulerGpuResult gpu_result;
    EulerGpuTimings timings;
    if(run_euler_gpu_rollout(batch, weights, options, gpu_result, timings) != 0){
        throw std::runtime_error("GPU rollout failed during validation");
    }
    EulerGpuResult cpu_result;
    run_cpu_reference(batch, weights, cpu_result);

    ValidationSummary summary;
    auto update_abs = [](float& value, float candidate){
        value = std::max(value, std::abs(candidate));
    };
    for(std::size_t i = 0; i < gpu_result.final_p.size(); i++){
        update_abs(summary.max_forward_abs_error, gpu_result.final_p[i] - cpu_result.final_p[i]);
        update_abs(summary.max_forward_abs_error, gpu_result.final_v[i] - cpu_result.final_v[i]);
        update_abs(summary.max_forward_abs_error, gpu_result.final_omega[i] - cpu_result.final_omega[i]);
    }
    for(std::size_t i = 0; i < gpu_result.final_R.size(); i++){
        update_abs(summary.max_forward_abs_error, gpu_result.final_R[i] - cpu_result.final_R[i]);
    }
    for(std::size_t i = 0; i < gpu_result.final_rpm.size(); i++){
        const float abs_error = std::abs(gpu_result.final_rpm[i] - cpu_result.final_rpm[i]);
        const float denom = std::max(1.0f, std::abs(cpu_result.final_rpm[i]));
        summary.max_rpm_abs_error = std::max(summary.max_rpm_abs_error, abs_error);
        summary.max_rpm_rel_error = std::max(summary.max_rpm_rel_error, abs_error / denom);
    }
    for(std::size_t i = 0; i < gpu_result.loss.size(); i++){
        const float abs_error = std::abs(gpu_result.loss[i] - cpu_result.loss[i]);
        const float denom = std::max(1e-6f, std::abs(cpu_result.loss[i]));
        summary.max_loss_abs_error = std::max(summary.max_loss_abs_error, abs_error);
        summary.max_loss_rel_error = std::max(summary.max_loss_rel_error, abs_error / denom);
    }
    double action_gradient_error_sq = 0.0;
    double action_gradient_reference_sq = 0.0;
    for(std::size_t i = 0; i < gpu_result.action_gradients.size(); i++){
        const float abs_error = std::abs(gpu_result.action_gradients[i] - cpu_result.action_gradients[i]);
        const float denom = std::max(1e-6f, std::abs(cpu_result.action_gradients[i]));
        summary.max_action_gradient_abs_error = std::max(summary.max_action_gradient_abs_error, abs_error);
        summary.max_action_gradient_rel_error = std::max(summary.max_action_gradient_rel_error, abs_error / denom);
        action_gradient_error_sq += static_cast<double>(abs_error) * static_cast<double>(abs_error);
        action_gradient_reference_sq += static_cast<double>(cpu_result.action_gradients[i]) * static_cast<double>(cpu_result.action_gradients[i]);
    }
    summary.action_gradient_l2_rel_error = static_cast<float>(
        std::sqrt(action_gradient_error_sq) / std::max(1e-12, std::sqrt(action_gradient_reference_sq))
    );
    summary.forward_close = summary.max_forward_abs_error < 1e-3f && summary.max_rpm_rel_error < 1e-6f;
    summary.loss_close = summary.max_loss_rel_error < 1e-3f || summary.max_loss_abs_error < 1e-4f;
    summary.action_gradient_close = summary.action_gradient_l2_rel_error < 1e-2f || summary.max_action_gradient_abs_error < 1e-3f;
    summary.passed = summary.forward_close && summary.loss_close && summary.action_gradient_close;
    return summary;
}

BenchmarkSummary benchmark_gpu_rollout(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    const EulerGpuRunOptions& options,
    int iterations
){
    BenchmarkSummary summary;
    summary.batch_size = batch.batch_size;
    summary.horizon = batch.horizon;
    summary.iterations = iterations;
    EulerGpuResult result;
    EulerGpuTimings timings;
    // Warm up once outside the measured loop.
    if(run_euler_gpu_rollout(batch, weights, options, result, timings) != 0){
        throw std::runtime_error("GPU rollout warmup failed");
    }
    EulerGpuTimings accum;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++){
        if(run_euler_gpu_rollout(batch, weights, options, result, timings) != 0){
            throw std::runtime_error("GPU rollout benchmark failed");
        }
        accum.host_to_device_ms += timings.host_to_device_ms;
        accum.forward_ms += timings.forward_ms;
        accum.loss_ms += timings.loss_ms;
        accum.backward_vjp_ms += timings.backward_vjp_ms;
        accum.device_to_host_ms += timings.device_to_host_ms;
        accum.total_ms += timings.total_ms;
    }
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    const double transitions = static_cast<double>(batch.batch_size) * static_cast<double>(batch.horizon) * static_cast<double>(iterations);
    summary.transitions_per_second = transitions / elapsed_seconds;
    summary.rollouts_per_second = static_cast<double>(batch.batch_size) * static_cast<double>(iterations) / elapsed_seconds;
    if(iterations > 0){
        summary.mean_timings.host_to_device_ms = accum.host_to_device_ms / iterations;
        summary.mean_timings.forward_ms = accum.forward_ms / iterations;
        summary.mean_timings.loss_ms = accum.loss_ms / iterations;
        summary.mean_timings.backward_vjp_ms = accum.backward_vjp_ms / iterations;
        summary.mean_timings.device_to_host_ms = accum.device_to_host_ms / iterations;
        summary.mean_timings.total_ms = accum.total_ms / iterations;
    }
    return summary;
}

BenchmarkSummary benchmark_cpu_reference(
    const EulerGpuBatch& batch,
    const EulerGpuLossWeights& weights,
    int iterations
){
    BenchmarkSummary summary;
    summary.batch_size = batch.batch_size;
    summary.horizon = batch.horizon;
    summary.iterations = iterations;
    EulerGpuResult result;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++){
        run_cpu_reference(batch, weights, result);
    }
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    const double transitions = static_cast<double>(batch.batch_size) * static_cast<double>(batch.horizon) * static_cast<double>(iterations);
    summary.transitions_per_second = transitions / elapsed_seconds;
    summary.rollouts_per_second = static_cast<double>(batch.batch_size) * static_cast<double>(iterations) / elapsed_seconds;
    summary.mean_timings.total_ms = static_cast<float>((elapsed_seconds * 1000.0) / std::max(1, iterations));
    summary.mean_timings.forward_ms = summary.mean_timings.total_ms;
    return summary;
}

} // namespace rl_tools::foundation_policy::diff_pre_training::gpu
