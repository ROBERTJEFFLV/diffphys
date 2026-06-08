#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int STATE_DIM = 40;
constexpr int ENC_DIM = 192;
constexpr int HID_DIM = 192;
constexpr int GATE_DIM = 3 * HID_DIM;
constexpr int ACT_DIM = 4;
constexpr int METRIC_DIM = 12;

enum MetricIndex {
    M_LOSS = 0,
    M_TRACKING = 1,
    M_POSITION = 2,
    M_VELOCITY = 3,
    M_ATTITUDE = 4,
    M_OMEGA = 5,
    M_CLF = 6,
    M_OUTWARD = 7,
    M_TAIL = 8,
    M_DU = 9,
    M_DDU = 10,
    M_SAT = 11,
};

struct DynamicsParams {
    float dt;
    float mass;
    float gravity;
    float arm_length;
    float yaw_drag;
    float motor_tau;
    float motor_authority;
    float inertia_x;
    float inertia_y;
    float inertia_z;
    float state_grad_decay;
    int noise_seed;
    float external_force_max;
    float external_torque_max;
    float action_noise_max;
};

struct LossParams {
    float hidden_grad_decay;
    float p_scale;
    float v_scale;
    float omega_scale;
    float huber_beta;
    float w_p;
    float w_v;
    float w_r;
    float w_omega;
    float clf_kappa;
    float u_soft;
    float lambda_clf;
    float lambda_out;
    float lambda_tail;
    float lambda_du;
    float lambda_ddu;
    float lambda_sat;
    float negative_slope;
    int noise_seed;
    float observation_noise_max;
};

__device__ inline float clampf(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

__device__ inline float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float leaky(float x, float slope) {
    return x >= 0.0f ? x : slope * x;
}

__device__ inline float leaky_grad(float y, float slope) {
    return y >= 0.0f ? 1.0f : slope;
}

__device__ inline unsigned int mix_u32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ inline float signed_noise(int seed, int t, int b, int channel, int stream) {
    unsigned int x = static_cast<unsigned int>(seed);
    x ^= static_cast<unsigned int>(t + 1) * 0x9e3779b9u;
    x ^= static_cast<unsigned int>(b + 1) * 0x85ebca6bu;
    x ^= static_cast<unsigned int>(channel + 1) * 0xc2b2ae35u;
    x ^= static_cast<unsigned int>(stream + 1) * 0x27d4eb2fu;
    const unsigned int h = mix_u32(x);
    const float unit = static_cast<float>(h & 0x00ffffffu) * (1.0f / 16777215.0f);
    return 2.0f * unit - 1.0f;
}

__device__ inline float add_observation_noise(float value, const LossParams lp, int t, int b, int k) {
    if (lp.observation_noise_max <= 0.0f) return value;
    return value + lp.observation_noise_max * signed_noise(lp.noise_seed, t, b, k, 0);
}

__device__ inline void cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void skew3(const float v[3], float k[9]) {
    k[0] = 0.0f;  k[1] = -v[2]; k[2] = v[1];
    k[3] = v[2];  k[4] = 0.0f;  k[5] = -v[0];
    k[6] = -v[1]; k[7] = v[0];  k[8] = 0.0f;
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
    float k[9], k2[9];
    skew3(phi, k);
    matmul3(k, k, k2);
    const float theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    float a, b;
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
    float k[9], k2[9];
    skew3(phi, k);
    matmul3(k, k, k2);
    const float theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    float a, b, da_coeff, db_coeff;
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
        da_coeff = ((theta * cos_t - sin_t) / theta2) / theta;
        db_coeff = ((theta * sin_t - 2.0f * (1.0f - cos_t)) / (theta2 * theta)) / theta;
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
        float dk_k[9], k_dk[9];
        matmul3(dk[axis], k, dk_k);
        matmul3(k, dk[axis], k_dk);
        const float da = da_coeff * phi[axis];
        const float db = db_coeff * phi[axis];
        for (int i = 0; i < 9; i++) {
            de[axis][i] = da * k[i] + a * dk[axis][i] + db * k2[i] + b * (dk_k[i] + k_dk[i]);
        }
    }
}

__device__ inline float smooth_l1_value(float x, float beta) {
    const float ax = fabsf(x);
    return ax < beta ? 0.5f * x * x / beta : ax - 0.5f * beta;
}

__device__ inline float smooth_l1_grad(float x, float beta) {
    const float ax = fabsf(x);
    return ax < beta ? x / beta : copysignf(1.0f, x);
}

__device__ float potential_and_components(
    const float* p,
    const float* v,
    const float* R,
    const float* omega,
    const LossParams lp,
    float* comps) {
    float pos = 0.0f, vel = 0.0f, att = 0.0f, omg = 0.0f;
    for (int i = 0; i < 3; i++) {
        pos += lp.w_p * smooth_l1_value(p[i] / lp.p_scale, lp.huber_beta);
        vel += lp.w_v * smooth_l1_value(v[i] / lp.v_scale, lp.huber_beta);
        omg += lp.w_omega * smooth_l1_value(omega[i] / lp.omega_scale, lp.huber_beta);
    }
    for (int i = 0; i < 9; i++) {
        const float target = (i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f;
        att += lp.w_r * smooth_l1_value(R[i] - target, lp.huber_beta);
    }
    comps[0] = pos;
    comps[1] = vel;
    comps[2] = att;
    comps[3] = omg;
    return pos + vel + att + omg;
}

__device__ void add_potential_grad(
    const float* p,
    const float* v,
    const float* R,
    const float* omega,
    const LossParams lp,
    float coeff,
    float* lp_out,
    float* lv_out,
    float* lR_out,
    float* lw_out) {
    for (int i = 0; i < 3; i++) {
        atomicAdd(&lp_out[i], coeff * lp.w_p * smooth_l1_grad(p[i] / lp.p_scale, lp.huber_beta) / lp.p_scale);
        atomicAdd(&lv_out[i], coeff * lp.w_v * smooth_l1_grad(v[i] / lp.v_scale, lp.huber_beta) / lp.v_scale);
        atomicAdd(&lw_out[i], coeff * lp.w_omega * smooth_l1_grad(omega[i] / lp.omega_scale, lp.huber_beta) / lp.omega_scale);
    }
    for (int i = 0; i < 9; i++) {
        const float target = (i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f;
        atomicAdd(&lR_out[i], coeff * lp.w_r * smooth_l1_grad(R[i] - target, lp.huber_beta));
    }
}

__global__ void initialize_rollout_kernel(
    const float* p0, const float* v0, const float* R0, const float* w0,
    const float* m0, const float* pa0,
    float* p, float* v, float* R, float* w, float* m, float* pa,
    float* hidden,
    int batch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int state3 = batch * 3;
    const int state4 = batch * 4;
    const int state9 = batch * 9;
    if (idx < state3) {
        p[idx] = p0[idx];
        v[idx] = v0[idx];
        w[idx] = w0[idx];
    }
    if (idx < state9) {
        R[idx] = R0[idx];
    }
    if (idx < state4) {
        m[idx] = m0[idx];
        pa[idx] = pa0[idx];
    }
    const int hidden_total = batch * HID_DIM;
    if (idx < hidden_total) {
        hidden[idx] = 0.0f;
    }
}

__global__ void actor_forward_kernel(
    int t,
    int batch,
    const float* p, const float* v, const float* R, const float* w, const float* m, const float* pa,
    const float* encoder0_w, const float* encoder0_b,
    const float* encoder1_w, const float* encoder1_b,
    const float* gru_w_ih, const float* gru_w_hh,
    const float* gru_b_ih, const float* gru_b_hh,
    const float* head_w, const float* head_b,
    float* hidden,
    float* enc0_store,
    float* enc1_store,
    float* gate_i_store,
    float* gate_h_store,
    float* actions,
    LossParams lp) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float input[STATE_DIM];
    __shared__ float e0[ENC_DIM];
    __shared__ float e1[ENC_DIM];
    __shared__ float hprev[HID_DIM];
    __shared__ float gate_i[GATE_DIM];
    __shared__ float gate_h[GATE_DIM];

    const int state_base3 = (t * batch + b) * 3;
    const int state_base9 = (t * batch + b) * 9;
    const int state_base4 = (t * batch + b) * 4;
    if (tid < 3) {
        input[tid] = add_observation_noise(p[state_base3 + tid], lp, t, b, tid);
        input[3 + tid] = add_observation_noise(v[state_base3 + tid], lp, t, b, 3 + tid);
        input[15 + tid] = add_observation_noise(w[state_base3 + tid], lp, t, b, 15 + tid);
        input[18 + tid] = add_observation_noise(p[state_base3 + tid], lp, t, b, 18 + tid);
        input[21 + tid] = add_observation_noise(v[state_base3 + tid], lp, t, b, 21 + tid);
        input[33 + tid] = add_observation_noise(w[state_base3 + tid], lp, t, b, 33 + tid);
    }
    if (tid < 9) {
        const float r = R[state_base9 + tid];
        const float target = (tid == 0 || tid == 4 || tid == 8) ? 1.0f : 0.0f;
        input[6 + tid] = add_observation_noise(r, lp, t, b, 6 + tid);
        input[24 + tid] = add_observation_noise(r - target, lp, t, b, 24 + tid);
    }
    if (tid < 4) {
        input[36 + tid] = add_observation_noise(pa[state_base4 + tid], lp, t, b, 36 + tid);
    }
    if (tid < HID_DIM) {
        hprev[tid] = hidden[(t * batch + b) * HID_DIM + tid];
    }
    __syncthreads();

    for (int h = tid; h < ENC_DIM; h += blockDim.x) {
        float sum = encoder0_b[h];
        for (int k = 0; k < STATE_DIM; k++) {
            sum += encoder0_w[h * STATE_DIM + k] * input[k];
        }
        const float value = leaky(sum, lp.negative_slope);
        e0[h] = value;
        enc0_store[(t * batch + b) * ENC_DIM + h] = value;
    }
    __syncthreads();

    for (int h = tid; h < ENC_DIM; h += blockDim.x) {
        float sum = encoder1_b[h];
        for (int k = 0; k < ENC_DIM; k++) {
            sum += encoder1_w[h * ENC_DIM + k] * e0[k];
        }
        const float value = leaky(sum, lp.negative_slope);
        e1[h] = value;
        enc1_store[(t * batch + b) * ENC_DIM + h] = value;
    }
    __syncthreads();

    for (int g = tid; g < GATE_DIM; g += blockDim.x) {
        float gi = gru_b_ih[g];
        float gh = gru_b_hh[g];
        for (int k = 0; k < ENC_DIM; k++) {
            gi += gru_w_ih[g * ENC_DIM + k] * e1[k];
            gh += gru_w_hh[g * HID_DIM + k] * hprev[k];
        }
        gate_i[g] = gi;
        gate_h[g] = gh;
        gate_i_store[(t * batch + b) * GATE_DIM + g] = gi;
        gate_h_store[(t * batch + b) * GATE_DIM + g] = gh;
    }
    __syncthreads();

    for (int h = tid; h < HID_DIM; h += blockDim.x) {
        const float r = sigmoidf_fast(gate_i[h] + gate_h[h]);
        const float z = sigmoidf_fast(gate_i[HID_DIM + h] + gate_h[HID_DIM + h]);
        const float n = tanhf(gate_i[2 * HID_DIM + h] + r * gate_h[2 * HID_DIM + h]);
        hidden[((t + 1) * batch + b) * HID_DIM + h] = n + z * (hprev[h] - n);
    }
    __syncthreads();

    if (tid < ACT_DIM) {
        float raw = head_b[tid];
        for (int h = 0; h < HID_DIM; h++) {
            const float hv = hidden[((t + 1) * batch + b) * HID_DIM + h];
            raw += head_w[tid * HID_DIM + h] * leaky(hv, lp.negative_slope);
        }
        actions[(t * batch + b) * ACT_DIM + tid] = tanhf(raw);
    }
}

__global__ void physics_forward_kernel(
    int t,
    int batch,
    const float* actions,
    float* p, float* v, float* R, float* w, float* m, float* pa,
    DynamicsParams dp) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const int base3 = (t * batch + b) * 3;
    const int next3 = ((t + 1) * batch + b) * 3;
    const int base9 = (t * batch + b) * 9;
    const int next9 = ((t + 1) * batch + b) * 9;
    const int base4 = (t * batch + b) * 4;
    const int next4 = ((t + 1) * batch + b) * 4;
    const int act4 = (t * batch + b) * 4;
    const float alpha = clampf(dp.dt / dp.motor_tau, 0.0f, 1.0f);
    const float hover = dp.mass * dp.gravity * 0.25f;
    float command[4], nm[4], thrust[4];
    for (int i = 0; i < 4; i++) {
        const float action_noise = dp.action_noise_max > 0.0f ? dp.action_noise_max * signed_noise(dp.noise_seed, t, b, i, 1) : 0.0f;
        command[i] = clampf(actions[act4 + i] + action_noise, -1.0f, 1.0f);
        nm[i] = m[base4 + i] + alpha * (command[i] - m[base4 + i]);
        thrust[i] = fmaxf(hover * (1.0f + dp.motor_authority * nm[i]), 0.0f);
        m[next4 + i] = nm[i];
        pa[next4 + i] = command[i];
    }
    float rmat[9];
    for (int i = 0; i < 9; i++) rmat[i] = R[base9 + i];
    const float total = thrust[0] + thrust[1] + thrust[2] + thrust[3];
    const float acc[3] = {
        rmat[2] * total / dp.mass + (dp.external_force_max > 0.0f ? dp.external_force_max * signed_noise(dp.noise_seed, t, b, 0, 2) / dp.mass : 0.0f),
        rmat[5] * total / dp.mass + (dp.external_force_max > 0.0f ? dp.external_force_max * signed_noise(dp.noise_seed, t, b, 1, 2) / dp.mass : 0.0f),
        rmat[8] * total / dp.mass - dp.gravity + (dp.external_force_max > 0.0f ? dp.external_force_max * signed_noise(dp.noise_seed, t, b, 2, 2) / dp.mass : 0.0f),
    };
    for (int i = 0; i < 3; i++) {
        const float nv = v[base3 + i] + dp.dt * acc[i];
        v[next3 + i] = nv;
        p[next3 + i] = p[base3 + i] + dp.dt * nv;
    }
    const float torque[3] = {
        dp.arm_length * (thrust[1] - thrust[3]) + (dp.external_torque_max > 0.0f ? dp.external_torque_max * signed_noise(dp.noise_seed, t, b, 0, 3) : 0.0f),
        dp.arm_length * (thrust[2] - thrust[0]) + (dp.external_torque_max > 0.0f ? dp.external_torque_max * signed_noise(dp.noise_seed, t, b, 1, 3) : 0.0f),
        dp.yaw_drag * (thrust[0] - thrust[1] + thrust[2] - thrust[3]) + (dp.external_torque_max > 0.0f ? dp.external_torque_max * signed_noise(dp.noise_seed, t, b, 2, 3) : 0.0f),
    };
    const float inertia[3] = {dp.inertia_x, dp.inertia_y, dp.inertia_z};
    float omega[3], iw[3], oc[3], nw[3];
    for (int i = 0; i < 3; i++) {
        omega[i] = w[base3 + i];
        iw[i] = inertia[i] * omega[i];
    }
    cross3(omega, iw, oc);
    for (int i = 0; i < 3; i++) {
        nw[i] = omega[i] + dp.dt * ((torque[i] - oc[i]) / inertia[i]);
        w[next3 + i] = nw[i];
    }
    const float phi[3] = {dp.dt * nw[0], dp.dt * nw[1], dp.dt * nw[2]};
    float expR[9], nextR[9];
    so3_exp(phi, expR);
    matmul3(rmat, expR, nextR);
    for (int i = 0; i < 9; i++) R[next9 + i] = nextR[i];
}

__global__ void potential_kernel(
    int total_states,
    int batch,
    const float* p, const float* v, const float* R, const float* w,
    float* potentials,
    LossParams lp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_states * batch) return;
    float comps[4];
    potentials[idx] = potential_and_components(
        &p[idx * 3], &v[idx * 3], &R[idx * 9], &w[idx * 3], lp, comps);
}

__global__ void loss_adjoint_kernel(
    int horizon,
    int tail_count,
    int batch,
    const float* p, const float* v, const float* R, const float* w,
    const float* m, const float* pa,
    const float* actions,
    const float* potentials,
    float* lp_adj, float* lv_adj, float* lR_adj, float* lw_adj, float* lm_adj, float* lpa_adj,
    float* action_adj,
    float* metrics,
    LossParams lp,
    DynamicsParams dp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = horizon * batch;
    if (idx >= total) return;
    const int t = idx / batch;
    const int b = idx - t * batch;
    const int state_idx = t + 1;
    const int s3 = (state_idx * batch + b) * 3;
    const int s9 = (state_idx * batch + b) * 9;
    const int s4 = (state_idx * batch + b) * 4;
    const int prev4 = (t * batch + b) * 4;
    const int act4 = (t * batch + b) * 4;
    const float inv_bh = 1.0f / (static_cast<float>(batch) * static_cast<float>(horizon));
    float comps[4];
    const float V = potential_and_components(&p[s3], &v[s3], &R[s9], &w[s3], lp, comps);
    float v_coeff = inv_bh;
    atomicAdd(&metrics[M_TRACKING], V * inv_bh);
    atomicAdd(&metrics[M_POSITION], comps[0] * inv_bh);
    atomicAdd(&metrics[M_VELOCITY], comps[1] * inv_bh);
    atomicAdd(&metrics[M_ATTITUDE], comps[2] * inv_bh);
    atomicAdd(&metrics[M_OMEGA], comps[3] * inv_bh);
    atomicAdd(&metrics[M_LOSS], V * inv_bh);
    if (state_idx > horizon - tail_count) {
        const float tail_unit = V / (static_cast<float>(batch) * static_cast<float>(tail_count));
        atomicAdd(&metrics[M_TAIL], tail_unit);
        atomicAdd(&metrics[M_LOSS], lp.lambda_tail * tail_unit);
        v_coeff += lp.lambda_tail / (static_cast<float>(batch) * static_cast<float>(tail_count));
    }
    const float clf_decay = fmaxf(0.0f, 1.0f - lp.clf_kappa * dp.dt);
    const float delta = V - clf_decay * potentials[t * batch + b];
    if (delta > 0.0f) {
        const float clf_unit = delta * delta * inv_bh;
        atomicAdd(&metrics[M_CLF], clf_unit);
        atomicAdd(&metrics[M_LOSS], lp.lambda_clf * clf_unit);
        v_coeff += lp.lambda_clf * 2.0f * delta * inv_bh;
    }
    add_potential_grad(
        &p[s3], &v[s3], &R[s9], &w[s3], lp, v_coeff,
        &lp_adj[s3], &lv_adj[s3], &lR_adj[s9], &lw_adj[s3]);

    float outward = 0.0f;
    for (int d = 0; d < 3; d++) {
        outward += (p[s3 + d] / lp.p_scale) * (v[s3 + d] / lp.v_scale);
    }
    if (outward > 0.0f) {
        const float out_unit = outward * outward * inv_bh;
        const float coeff = lp.lambda_out * 2.0f * outward * inv_bh / (lp.p_scale * lp.v_scale);
        atomicAdd(&metrics[M_OUTWARD], out_unit);
        atomicAdd(&metrics[M_LOSS], lp.lambda_out * out_unit);
        for (int d = 0; d < 3; d++) {
            atomicAdd(&lp_adj[s3 + d], coeff * v[s3 + d]);
            atomicAdd(&lv_adj[s3 + d], coeff * p[s3 + d]);
        }
    }

    const float inv_bh4 = inv_bh * 0.25f;
    const float inv_ddu = horizon > 1 ? 1.0f / (static_cast<float>(batch) * 4.0f * static_cast<float>(horizon - 1)) : 0.0f;
    for (int a = 0; a < 4; a++) {
        const float du = actions[act4 + a] - pa[prev4 + a];
        const float du_unit = du * du * inv_bh4;
        atomicAdd(&metrics[M_DU], du_unit);
        atomicAdd(&metrics[M_LOSS], lp.lambda_du * du_unit);
        const float du_grad = lp.lambda_du * 2.0f * du * inv_bh4;
        atomicAdd(&action_adj[act4 + a], du_grad);
        atomicAdd(&lpa_adj[prev4 + a], -du_grad);

        if (t > 0) {
            const int prev_act4 = ((t - 1) * batch + b) * 4;
            const int prev_state4 = ((t - 1) * batch + b) * 4;
            const float du_prev = actions[prev_act4 + a] - pa[prev_state4 + a];
            const float ddu = du - du_prev;
            const float ddu_unit = ddu * ddu * inv_ddu;
            atomicAdd(&metrics[M_DDU], ddu_unit);
            atomicAdd(&metrics[M_LOSS], lp.lambda_ddu * ddu_unit);
            const float ddu_grad = lp.lambda_ddu * 2.0f * ddu * inv_ddu;
            atomicAdd(&action_adj[act4 + a], ddu_grad);
            atomicAdd(&lpa_adj[prev4 + a], -ddu_grad);
            atomicAdd(&action_adj[prev_act4 + a], -ddu_grad);
            atomicAdd(&lpa_adj[prev_state4 + a], ddu_grad);
        }

        const float abs_u = fabsf(actions[act4 + a]);
        if (abs_u > lp.u_soft) {
            const float excess = abs_u - lp.u_soft;
            const float sat_unit = excess * excess * inv_bh4;
            atomicAdd(&metrics[M_SAT], sat_unit);
            atomicAdd(&metrics[M_LOSS], lp.lambda_sat * sat_unit);
            const float sat_grad = lp.lambda_sat * 2.0f * excess * copysignf(1.0f, actions[act4 + a]) * inv_bh4;
            atomicAdd(&action_adj[act4 + a], sat_grad);
        }
    }
    (void)m;
}

__global__ void physics_backward_kernel(
    int t,
    int batch,
    const float* p, const float* v, const float* R, const float* w, const float* m, const float* actions,
    float* lp_adj, float* lv_adj, float* lR_adj, float* lw_adj, float* lm_adj, float* lpa_adj,
    float* action_adj,
    DynamicsParams dp) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const int base3 = (t * batch + b) * 3;
    const int next3 = ((t + 1) * batch + b) * 3;
    const int base9 = (t * batch + b) * 9;
    const int next9 = ((t + 1) * batch + b) * 9;
    const int base4 = (t * batch + b) * 4;
    const int next4 = ((t + 1) * batch + b) * 4;
    const int act4 = (t * batch + b) * 4;

    const float alpha = clampf(dp.dt / dp.motor_tau, 0.0f, 1.0f);
    const float hover = dp.mass * dp.gravity * 0.25f;
    const float inertia[3] = {dp.inertia_x, dp.inertia_y, dp.inertia_z};
    float rmat[9];
    for (int i = 0; i < 9; i++) rmat[i] = R[base9 + i];
    float command[4], nm[4], thrust[4];
    for (int i = 0; i < 4; i++) {
        const float action_noise = dp.action_noise_max > 0.0f ? dp.action_noise_max * signed_noise(dp.noise_seed, t, b, i, 1) : 0.0f;
        command[i] = clampf(actions[act4 + i] + action_noise, -1.0f, 1.0f);
        nm[i] = m[base4 + i] + alpha * (command[i] - m[base4 + i]);
        thrust[i] = fmaxf(hover * (1.0f + dp.motor_authority * nm[i]), 0.0f);
    }
    const float torque[3] = {
        dp.arm_length * (thrust[1] - thrust[3]) + (dp.external_torque_max > 0.0f ? dp.external_torque_max * signed_noise(dp.noise_seed, t, b, 0, 3) : 0.0f),
        dp.arm_length * (thrust[2] - thrust[0]) + (dp.external_torque_max > 0.0f ? dp.external_torque_max * signed_noise(dp.noise_seed, t, b, 1, 3) : 0.0f),
        dp.yaw_drag * (thrust[0] - thrust[1] + thrust[2] - thrust[3]) + (dp.external_torque_max > 0.0f ? dp.external_torque_max * signed_noise(dp.noise_seed, t, b, 2, 3) : 0.0f),
    };
    float omega[3], iw[3], oc[3], nw[3];
    for (int i = 0; i < 3; i++) {
        omega[i] = w[base3 + i];
        iw[i] = inertia[i] * omega[i];
    }
    cross3(omega, iw, oc);
    for (int i = 0; i < 3; i++) {
        nw[i] = omega[i] + dp.dt * ((torque[i] - oc[i]) / inertia[i]);
    }
    const float phi[3] = {dp.dt * nw[0], dp.dt * nw[1], dp.dt * nw[2]};
    float expR[9], de[3][9];
    so3_exp_with_derivatives(phi, expR, de);

    float gp[3], gv_next[3], gr_next[9], gw_next[3], gm_next[4], gc[4];
    for (int i = 0; i < 3; i++) {
        gp[i] = dp.state_grad_decay * lp_adj[next3 + i];
        gv_next[i] = dp.state_grad_decay * lv_adj[next3 + i];
        gw_next[i] = dp.state_grad_decay * lw_adj[next3 + i];
    }
    for (int i = 0; i < 9; i++) gr_next[i] = dp.state_grad_decay * lR_adj[next9 + i];
    for (int i = 0; i < 4; i++) {
        gm_next[i] = dp.state_grad_decay * lm_adj[next4 + i];
        gc[i] = dp.state_grad_decay * lpa_adj[next4 + i];
    }

    float gp_in[3] = {0, 0, 0};
    float gv_in[3] = {0, 0, 0};
    float gr_in[9] = {0};
    float gw_in[3] = {0, 0, 0};
    float gm_in[4] = {0, 0, 0, 0};
    float ga_in[4] = {0, 0, 0, 0};
    float gthrust[4] = {0, 0, 0, 0};

    float ge[9] = {0};
    for (int rr = 0; rr < 3; rr++) {
        for (int cc = 0; cc < 3; cc++) {
            float g_r = 0.0f;
            for (int kk = 0; kk < 3; kk++) {
                g_r += gr_next[rr * 3 + kk] * expR[cc * 3 + kk];
                ge[cc * 3 + kk] += rmat[rr * 3 + cc] * gr_next[rr * 3 + kk];
            }
            gr_in[rr * 3 + cc] += g_r;
        }
    }
    for (int axis = 0; axis < 3; axis++) {
        float gphi = 0.0f;
        for (int i = 0; i < 9; i++) gphi += ge[i] * de[axis][i];
        gw_next[axis] += dp.dt * gphi;
    }

    float gacc[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        gp_in[i] += gp[i];
        const float gv_total = gv_next[i] + dp.dt * gp[i];
        gv_in[i] += gv_total;
        gacc[i] += dp.dt * gv_total;
    }
    const float total_thrust = thrust[0] + thrust[1] + thrust[2] + thrust[3];
    float gtotal = 0.0f;
    for (int i = 0; i < 3; i++) {
        gr_in[i * 3 + 2] += gacc[i] * total_thrust / dp.mass;
        gtotal += gacc[i] * rmat[i * 3 + 2] / dp.mass;
    }
    for (int i = 0; i < 4; i++) gthrust[i] += gtotal;

    float gy[3], gtorque[3];
    for (int i = 0; i < 3; i++) {
        gw_in[i] += gw_next[i];
        gtorque[i] = gw_next[i] * dp.dt / inertia[i];
        gy[i] = -gw_next[i] * dp.dt / inertia[i];
    }
    float tmp_a[3], tmp_b[3];
    cross3(iw, gy, tmp_a);
    cross3(gy, omega, tmp_b);
    for (int i = 0; i < 3; i++) gw_in[i] += tmp_a[i] + inertia[i] * tmp_b[i];

    gthrust[1] += dp.arm_length * gtorque[0];
    gthrust[3] -= dp.arm_length * gtorque[0];
    gthrust[2] += dp.arm_length * gtorque[1];
    gthrust[0] -= dp.arm_length * gtorque[1];
    gthrust[0] += dp.yaw_drag * gtorque[2];
    gthrust[1] -= dp.yaw_drag * gtorque[2];
    gthrust[2] += dp.yaw_drag * gtorque[2];
    gthrust[3] -= dp.yaw_drag * gtorque[2];

    for (int i = 0; i < 4; i++) {
        const float thrust_pre = hover * (1.0f + dp.motor_authority * nm[i]);
        if (thrust_pre > 0.0f) gm_next[i] += gthrust[i] * hover * dp.motor_authority;
        gm_in[i] += (1.0f - alpha) * gm_next[i];
        gc[i] += alpha * gm_next[i];
        const float action_noise = dp.action_noise_max > 0.0f ? dp.action_noise_max * signed_noise(dp.noise_seed, t, b, i, 1) : 0.0f;
        const float noisy_action = actions[act4 + i] + action_noise;
        if (noisy_action >= -1.0f && noisy_action <= 1.0f) ga_in[i] += gc[i];
    }
    for (int i = 0; i < 3; i++) {
        atomicAdd(&lp_adj[base3 + i], gp_in[i]);
        atomicAdd(&lv_adj[base3 + i], gv_in[i]);
        atomicAdd(&lw_adj[base3 + i], gw_in[i]);
    }
    for (int i = 0; i < 9; i++) atomicAdd(&lR_adj[base9 + i], gr_in[i]);
    for (int i = 0; i < 4; i++) {
        atomicAdd(&lm_adj[base4 + i], gm_in[i]);
        atomicAdd(&action_adj[act4 + i], ga_in[i]);
    }
}

__global__ void actor_backward_kernel(
    int t,
    int batch,
    const float* p, const float* v, const float* R, const float* w, const float* m, const float* pa,
    const float* actions,
    const float* hidden,
    const float* enc0_store,
    const float* enc1_store,
    const float* gate_i_store,
    const float* gate_h_store,
    const float* encoder0_w, const float* encoder1_w,
    const float* gru_w_ih, const float* gru_w_hh,
    const float* head_w,
    float* lp_adj, float* lv_adj, float* lR_adj, float* lw_adj, float* lm_adj, float* lpa_adj,
    float* action_adj,
    float* hidden_adj,
    float* g_encoder0_w, float* g_encoder0_b,
    float* g_encoder1_w, float* g_encoder1_b,
    float* g_gru_w_ih, float* g_gru_w_hh,
    float* g_gru_b_ih, float* g_gru_b_hh,
    float* g_head_w, float* g_head_b,
    LossParams lp) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float input[STATE_DIM];
    __shared__ float e0[ENC_DIM], e1[ENC_DIM], hprev[HID_DIM], hcur[HID_DIM];
    __shared__ float d_h[HID_DIM], d_hprev[HID_DIM], d_e1[ENC_DIM], d_e0[ENC_DIM], d_input[STATE_DIM];
    __shared__ float d_gate_i[GATE_DIM], d_gate_h[GATE_DIM];
    __shared__ float gate_i[GATE_DIM], gate_h[GATE_DIM];

    const int s3 = (t * batch + b) * 3;
    const int s9 = (t * batch + b) * 9;
    const int s4 = (t * batch + b) * 4;
    const int act4 = (t * batch + b) * 4;
    if (tid < STATE_DIM) d_input[tid] = 0.0f;
    for (int h = tid; h < ENC_DIM; h += blockDim.x) {
        d_e0[h] = 0.0f;
        d_e1[h] = 0.0f;
        d_hprev[h] = 0.0f;
        e0[h] = enc0_store[(t * batch + b) * ENC_DIM + h];
        e1[h] = enc1_store[(t * batch + b) * ENC_DIM + h];
        hprev[h] = hidden[(t * batch + b) * HID_DIM + h];
        hcur[h] = hidden[((t + 1) * batch + b) * HID_DIM + h];
        d_h[h] = hidden_adj[((t + 1) * batch + b) * HID_DIM + h];
    }
    for (int g = tid; g < GATE_DIM; g += blockDim.x) {
        d_gate_i[g] = 0.0f;
        d_gate_h[g] = 0.0f;
        gate_i[g] = gate_i_store[(t * batch + b) * GATE_DIM + g];
        gate_h[g] = gate_h_store[(t * batch + b) * GATE_DIM + g];
    }
    if (tid < 3) {
        input[tid] = add_observation_noise(p[s3 + tid], lp, t, b, tid);
        input[3 + tid] = add_observation_noise(v[s3 + tid], lp, t, b, 3 + tid);
        input[15 + tid] = add_observation_noise(w[s3 + tid], lp, t, b, 15 + tid);
        input[18 + tid] = add_observation_noise(p[s3 + tid], lp, t, b, 18 + tid);
        input[21 + tid] = add_observation_noise(v[s3 + tid], lp, t, b, 21 + tid);
        input[33 + tid] = add_observation_noise(w[s3 + tid], lp, t, b, 33 + tid);
    }
    if (tid < 9) {
        const float rv = R[s9 + tid];
        const float target = (tid == 0 || tid == 4 || tid == 8) ? 1.0f : 0.0f;
        input[6 + tid] = add_observation_noise(rv, lp, t, b, 6 + tid);
        input[24 + tid] = add_observation_noise(rv - target, lp, t, b, 24 + tid);
    }
    if (tid < 4) {
        input[36 + tid] = add_observation_noise(pa[s4 + tid], lp, t, b, 36 + tid);
    }
    __syncthreads();

    for (int h = tid; h < HID_DIM; h += blockDim.x) {
        const float hleaky = leaky(hcur[h], lp.negative_slope);
        float dh_add = 0.0f;
        for (int a = 0; a < ACT_DIM; a++) {
            const float u = actions[act4 + a];
            const float d_raw = action_adj[act4 + a] * (1.0f - u * u);
            dh_add += head_w[a * HID_DIM + h] * d_raw;
            atomicAdd(&g_head_w[a * HID_DIM + h], d_raw * hleaky);
        }
        d_h[h] += dh_add * leaky_grad(hcur[h], lp.negative_slope);
    }
    if (tid < ACT_DIM) {
        const float u = actions[act4 + tid];
        atomicAdd(&g_head_b[tid], action_adj[act4 + tid] * (1.0f - u * u));
    }
    __syncthreads();

    for (int h = tid; h < HID_DIM; h += blockDim.x) {
        const float r = sigmoidf_fast(gate_i[h] + gate_h[h]);
        const float z = sigmoidf_fast(gate_i[HID_DIM + h] + gate_h[HID_DIM + h]);
        const float h_aff_n = gate_h[2 * HID_DIM + h];
        const float n = tanhf(gate_i[2 * HID_DIM + h] + r * h_aff_n);
        const float dh = d_h[h];
        const float dz = dh * (hprev[h] - n);
        const float dn = dh * (1.0f - z);
        d_hprev[h] += dh * z;
        const float dn_pre = dn * (1.0f - n * n);
        const float dr = dn_pre * h_aff_n;
        const float dh_aff_n = dn_pre * r;
        d_gate_i[h] = dr * r * (1.0f - r);
        d_gate_h[h] = d_gate_i[h];
        d_gate_i[HID_DIM + h] = dz * z * (1.0f - z);
        d_gate_h[HID_DIM + h] = d_gate_i[HID_DIM + h];
        d_gate_i[2 * HID_DIM + h] = dn_pre;
        d_gate_h[2 * HID_DIM + h] = dh_aff_n;
    }
    __syncthreads();

    for (int g = tid; g < GATE_DIM; g += blockDim.x) {
        const float dgi = d_gate_i[g];
        const float dgh = d_gate_h[g];
        atomicAdd(&g_gru_b_ih[g], dgi);
        atomicAdd(&g_gru_b_hh[g], dgh);
        for (int k = 0; k < HID_DIM; k++) {
            atomicAdd(&g_gru_w_ih[g * HID_DIM + k], dgi * e1[k]);
            atomicAdd(&g_gru_w_hh[g * HID_DIM + k], dgh * hprev[k]);
        }
    }
    __syncthreads();

    for (int k = tid; k < HID_DIM; k += blockDim.x) {
        float de1 = 0.0f;
        float dhp = d_hprev[k];
        for (int g = 0; g < GATE_DIM; g++) {
            de1 += gru_w_ih[g * HID_DIM + k] * d_gate_i[g];
            dhp += gru_w_hh[g * HID_DIM + k] * d_gate_h[g];
        }
        d_e1[k] = de1;
        atomicAdd(&hidden_adj[(t * batch + b) * HID_DIM + k], lp.hidden_grad_decay * dhp);
    }
    __syncthreads();

    for (int h = tid; h < ENC_DIM; h += blockDim.x) {
        const float dpre = d_e1[h] * leaky_grad(e1[h], lp.negative_slope);
        atomicAdd(&g_encoder1_b[h], dpre);
        for (int k = 0; k < ENC_DIM; k++) {
            atomicAdd(&g_encoder1_w[h * ENC_DIM + k], dpre * e0[k]);
        }
    }
    __syncthreads();

    for (int k = tid; k < ENC_DIM; k += blockDim.x) {
        float de0 = 0.0f;
        for (int h = 0; h < ENC_DIM; h++) {
            const float dpre = d_e1[h] * leaky_grad(e1[h], lp.negative_slope);
            de0 += encoder1_w[h * ENC_DIM + k] * dpre;
        }
        d_e0[k] = de0;
    }
    __syncthreads();

    for (int h = tid; h < ENC_DIM; h += blockDim.x) {
        const float dpre = d_e0[h] * leaky_grad(e0[h], lp.negative_slope);
        atomicAdd(&g_encoder0_b[h], dpre);
        for (int k = 0; k < STATE_DIM; k++) {
            atomicAdd(&g_encoder0_w[h * STATE_DIM + k], dpre * input[k]);
        }
    }
    __syncthreads();

    for (int k = tid; k < STATE_DIM; k += blockDim.x) {
        float din = 0.0f;
        for (int h = 0; h < ENC_DIM; h++) {
            const float dpre = d_e0[h] * leaky_grad(e0[h], lp.negative_slope);
            din += encoder0_w[h * STATE_DIM + k] * dpre;
        }
        d_input[k] = din;
    }
    __syncthreads();

    if (tid < 3) {
        atomicAdd(&lp_adj[s3 + tid], d_input[tid] + d_input[18 + tid]);
        atomicAdd(&lv_adj[s3 + tid], d_input[3 + tid] + d_input[21 + tid]);
        atomicAdd(&lw_adj[s3 + tid], d_input[15 + tid] + d_input[33 + tid]);
    }
    if (tid < 9) {
        atomicAdd(&lR_adj[s9 + tid], d_input[6 + tid] + d_input[24 + tid]);
    }
    if (tid < 4) {
        atomicAdd(&lpa_adj[s4 + tid], d_input[36 + tid]);
    }
}

template <typename T>
void zero_tensor(T& tensor) {
    tensor.zero_();
}

} // namespace

std::vector<torch::Tensor> l2f_full_rollout_cuda(
    torch::Tensor position0,
    torch::Tensor velocity0,
    torch::Tensor rotation0,
    torch::Tensor omega0,
    torch::Tensor motor0,
    torch::Tensor previous_action0,
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
    double external_force_max,
    double external_torque_max,
    double action_noise_max,
    double observation_noise_max) {
    const c10::cuda::CUDAGuard device_guard(position0.device());
    const int batch = static_cast<int>(position0.size(0));
    const int states = horizon + 1;
    auto opts = position0.options();
    auto p = torch::empty({states, batch, 3}, opts);
    auto v = torch::empty({states, batch, 3}, opts);
    auto R = torch::empty({states, batch, 3, 3}, opts);
    auto w = torch::empty({states, batch, 3}, opts);
    auto m = torch::empty({states, batch, 4}, opts);
    auto pa = torch::empty({states, batch, 4}, opts);
    auto hidden = torch::empty({states, batch, HID_DIM}, opts);
    auto enc0 = torch::empty({horizon, batch, ENC_DIM}, opts);
    auto enc1 = torch::empty({horizon, batch, ENC_DIM}, opts);
    auto gate_i = torch::empty({horizon, batch, GATE_DIM}, opts);
    auto gate_h = torch::empty({horizon, batch, GATE_DIM}, opts);
    auto actions = torch::empty({horizon, batch, ACT_DIM}, opts);
    auto potentials = torch::empty({states, batch}, opts);

    auto lp_adj = torch::zeros({states, batch, 3}, opts);
    auto lv_adj = torch::zeros({states, batch, 3}, opts);
    auto lR_adj = torch::zeros({states, batch, 3, 3}, opts);
    auto lw_adj = torch::zeros({states, batch, 3}, opts);
    auto lm_adj = torch::zeros({states, batch, 4}, opts);
    auto lpa_adj = torch::zeros({states, batch, 4}, opts);
    auto action_adj = torch::zeros({horizon, batch, ACT_DIM}, opts);
    auto hidden_adj = torch::zeros({states, batch, HID_DIM}, opts);
    auto metrics = torch::zeros({METRIC_DIM}, opts);

    auto g_encoder0_w = torch::zeros_like(encoder0_w);
    auto g_encoder0_b = torch::zeros_like(encoder0_b);
    auto g_encoder1_w = torch::zeros_like(encoder1_w);
    auto g_encoder1_b = torch::zeros_like(encoder1_b);
    auto g_gru_w_ih = torch::zeros_like(gru_w_ih);
    auto g_gru_w_hh = torch::zeros_like(gru_w_hh);
    auto g_gru_b_ih = torch::zeros_like(gru_b_ih);
    auto g_gru_b_hh = torch::zeros_like(gru_b_hh);
    auto g_head_w = torch::zeros_like(motor_head_w);
    auto g_head_b = torch::zeros_like(motor_head_b);

    DynamicsParams dp{
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
        static_cast<float>(state_grad_decay),
        noise_seed,
        static_cast<float>(external_force_max),
        static_cast<float>(external_torque_max),
        static_cast<float>(action_noise_max),
    };
    LossParams lp{
        static_cast<float>(hidden_grad_decay),
        static_cast<float>(p_scale),
        static_cast<float>(v_scale),
        static_cast<float>(omega_scale),
        static_cast<float>(huber_beta),
        static_cast<float>(w_p),
        static_cast<float>(w_v),
        static_cast<float>(w_r),
        static_cast<float>(w_omega),
        static_cast<float>(clf_kappa),
        static_cast<float>(u_soft),
        static_cast<float>(lambda_clf),
        static_cast<float>(lambda_out),
        static_cast<float>(lambda_tail),
        static_cast<float>(lambda_du),
        static_cast<float>(lambda_ddu),
        static_cast<float>(lambda_sat),
        static_cast<float>(negative_slope),
        noise_seed,
        static_cast<float>(observation_noise_max),
    };

    const int init_total = std::max({batch * 9, batch * 4, batch * HID_DIM});
    const int threads = 256;
    initialize_rollout_kernel<<<(init_total + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        position0.data_ptr<float>(), velocity0.data_ptr<float>(), rotation0.data_ptr<float>(),
        omega0.data_ptr<float>(), motor0.data_ptr<float>(), previous_action0.data_ptr<float>(),
        p.data_ptr<float>(), v.data_ptr<float>(), R.data_ptr<float>(), w.data_ptr<float>(),
        m.data_ptr<float>(), pa.data_ptr<float>(), hidden.data_ptr<float>(), batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int t = 0; t < horizon; t++) {
        actor_forward_kernel<<<batch, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            t, batch,
            p.data_ptr<float>(), v.data_ptr<float>(), R.data_ptr<float>(), w.data_ptr<float>(),
            m.data_ptr<float>(), pa.data_ptr<float>(),
            encoder0_w.data_ptr<float>(), encoder0_b.data_ptr<float>(),
            encoder1_w.data_ptr<float>(), encoder1_b.data_ptr<float>(),
            gru_w_ih.data_ptr<float>(), gru_w_hh.data_ptr<float>(),
            gru_b_ih.data_ptr<float>(), gru_b_hh.data_ptr<float>(),
            motor_head_w.data_ptr<float>(), motor_head_b.data_ptr<float>(),
            hidden.data_ptr<float>(), enc0.data_ptr<float>(), enc1.data_ptr<float>(),
            gate_i.data_ptr<float>(), gate_h.data_ptr<float>(), actions.data_ptr<float>(), lp);
        physics_forward_kernel<<<(batch + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            t, batch, actions.data_ptr<float>(),
            p.data_ptr<float>(), v.data_ptr<float>(), R.data_ptr<float>(), w.data_ptr<float>(),
            m.data_ptr<float>(), pa.data_ptr<float>(), dp);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int total_states = states * batch;
    potential_kernel<<<(total_states + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        states, batch, p.data_ptr<float>(), v.data_ptr<float>(), R.data_ptr<float>(), w.data_ptr<float>(),
        potentials.data_ptr<float>(), lp);
    const int tail_count = std::min(std::max(tail_steps, 1), horizon);
    const int total_steps = horizon * batch;
    loss_adjoint_kernel<<<(total_steps + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        horizon, tail_count, batch,
        p.data_ptr<float>(), v.data_ptr<float>(), R.data_ptr<float>(), w.data_ptr<float>(),
        m.data_ptr<float>(), pa.data_ptr<float>(), actions.data_ptr<float>(), potentials.data_ptr<float>(),
        lp_adj.data_ptr<float>(), lv_adj.data_ptr<float>(), lR_adj.data_ptr<float>(), lw_adj.data_ptr<float>(),
        lm_adj.data_ptr<float>(), lpa_adj.data_ptr<float>(), action_adj.data_ptr<float>(),
        metrics.data_ptr<float>(), lp, dp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int rev = 0; rev < horizon; rev++) {
        const int t = horizon - 1 - rev;
        physics_backward_kernel<<<(batch + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            t, batch,
            p.data_ptr<float>(), v.data_ptr<float>(), R.data_ptr<float>(), w.data_ptr<float>(),
            m.data_ptr<float>(), actions.data_ptr<float>(),
            lp_adj.data_ptr<float>(), lv_adj.data_ptr<float>(), lR_adj.data_ptr<float>(), lw_adj.data_ptr<float>(),
            lm_adj.data_ptr<float>(), lpa_adj.data_ptr<float>(), action_adj.data_ptr<float>(), dp);
        actor_backward_kernel<<<batch, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            t, batch,
            p.data_ptr<float>(), v.data_ptr<float>(), R.data_ptr<float>(), w.data_ptr<float>(),
            m.data_ptr<float>(), pa.data_ptr<float>(), actions.data_ptr<float>(), hidden.data_ptr<float>(),
            enc0.data_ptr<float>(), enc1.data_ptr<float>(), gate_i.data_ptr<float>(), gate_h.data_ptr<float>(),
            encoder0_w.data_ptr<float>(), encoder1_w.data_ptr<float>(), gru_w_ih.data_ptr<float>(),
            gru_w_hh.data_ptr<float>(), motor_head_w.data_ptr<float>(),
            lp_adj.data_ptr<float>(), lv_adj.data_ptr<float>(), lR_adj.data_ptr<float>(), lw_adj.data_ptr<float>(),
            lm_adj.data_ptr<float>(), lpa_adj.data_ptr<float>(), action_adj.data_ptr<float>(), hidden_adj.data_ptr<float>(),
            g_encoder0_w.data_ptr<float>(), g_encoder0_b.data_ptr<float>(),
            g_encoder1_w.data_ptr<float>(), g_encoder1_b.data_ptr<float>(),
            g_gru_w_ih.data_ptr<float>(), g_gru_w_hh.data_ptr<float>(),
            g_gru_b_ih.data_ptr<float>(), g_gru_b_hh.data_ptr<float>(),
            g_head_w.data_ptr<float>(), g_head_b.data_ptr<float>(), lp);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        metrics,
        g_encoder0_w, g_encoder0_b,
        g_encoder1_w, g_encoder1_b,
        g_gru_w_ih, g_gru_w_hh,
        g_gru_b_ih, g_gru_b_hh,
        g_head_w, g_head_b,
    };
}
