from __future__ import annotations

import torch

from env_l2f import L2FLossConfig, L2FParams, L2FState
from l2f_cuda_backend import load_extension
from model import MotorGRUPolicy


METRIC_NAMES = (
    "loss",
    "tracking",
    "position",
    "velocity",
    "attitude",
    "omega",
    "clf",
    "outward",
    "tail",
    "du",
    "ddu",
    "sat",
)


def _policy_tensors(policy: MotorGRUPolicy) -> tuple[torch.Tensor, ...]:
    if len(policy.encoder) != 4:
        raise ValueError("cuda-full backend currently requires encoder_depth=2")
    if policy.encoder[0].weight.shape != (192, 40):
        raise ValueError("cuda-full backend currently requires encoder[0] weight shape [192,40]")
    if policy.encoder[2].weight.shape != (192, 192):
        raise ValueError("cuda-full backend currently requires encoder[1] weight shape [192,192]")
    if policy.gru.weight_ih.shape != (576, 192) or policy.gru.weight_hh.shape != (576, 192):
        raise ValueError("cuda-full backend currently requires GRU weight shape [576,192]")
    if policy.motor_head.weight.shape != (4, 192):
        raise ValueError("cuda-full backend currently requires motor head weight shape [4,192]")
    return (
        policy.encoder[0].weight,
        policy.encoder[0].bias,
        policy.encoder[2].weight,
        policy.encoder[2].bias,
        policy.gru.weight_ih,
        policy.gru.weight_hh,
        policy.gru.bias_ih,
        policy.gru.bias_hh,
        policy.motor_head.weight,
        policy.motor_head.bias,
    )


class _FullCudaRollout(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        position0: torch.Tensor,
        velocity0: torch.Tensor,
        rotation0: torch.Tensor,
        omega0: torch.Tensor,
        motor0: torch.Tensor,
        previous_action0: torch.Tensor,
        encoder0_w: torch.Tensor,
        encoder0_b: torch.Tensor,
        encoder1_w: torch.Tensor,
        encoder1_b: torch.Tensor,
        gru_w_ih: torch.Tensor,
        gru_w_hh: torch.Tensor,
        gru_b_ih: torch.Tensor,
        gru_b_hh: torch.Tensor,
        motor_head_w: torch.Tensor,
        motor_head_b: torch.Tensor,
        horizon: int,
        tail_steps: int,
        dt: float,
        mass: float,
        gravity: float,
        arm_length: float,
        yaw_drag: float,
        motor_tau: float,
        motor_authority: float,
        inertia_x: float,
        inertia_y: float,
        inertia_z: float,
        state_grad_decay: float,
        hidden_grad_decay: float,
        p_scale: float,
        v_scale: float,
        omega_scale: float,
        huber_beta: float,
        w_p: float,
        w_v: float,
        w_r: float,
        w_omega: float,
        clf_kappa: float,
        u_soft: float,
        lambda_clf: float,
        lambda_out: float,
        lambda_tail: float,
        lambda_du: float,
        lambda_ddu: float,
        lambda_sat: float,
        negative_slope: float,
        noise_seed: int,
        external_force_max: float,
        external_torque_max: float,
        action_noise_max: float,
        observation_noise_max: float,
    ) -> torch.Tensor:
        ext = load_extension()
        outputs = ext.full_rollout(
            position0.contiguous(),
            velocity0.contiguous(),
            rotation0.contiguous(),
            omega0.contiguous(),
            motor0.contiguous(),
            previous_action0.contiguous(),
            encoder0_w.contiguous(),
            encoder0_b.contiguous(),
            encoder1_w.contiguous(),
            encoder1_b.contiguous(),
            gru_w_ih.contiguous(),
            gru_w_hh.contiguous(),
            gru_b_ih.contiguous(),
            gru_b_hh.contiguous(),
            motor_head_w.contiguous(),
            motor_head_b.contiguous(),
            int(horizon),
            int(tail_steps),
            float(dt),
            float(mass),
            float(gravity),
            float(arm_length),
            float(yaw_drag),
            float(motor_tau),
            float(motor_authority),
            float(inertia_x),
            float(inertia_y),
            float(inertia_z),
            float(state_grad_decay),
            float(hidden_grad_decay),
            float(p_scale),
            float(v_scale),
            float(omega_scale),
            float(huber_beta),
            float(w_p),
            float(w_v),
            float(w_r),
            float(w_omega),
            float(clf_kappa),
            float(u_soft),
            float(lambda_clf),
            float(lambda_out),
            float(lambda_tail),
            float(lambda_du),
            float(lambda_ddu),
            float(lambda_sat),
            float(negative_slope),
            int(noise_seed),
            float(external_force_max),
            float(external_torque_max),
            float(action_noise_max),
            float(observation_noise_max),
        )
        metrics = outputs[0]
        ctx.save_for_backward(*outputs[1:])
        return metrics

    @staticmethod
    def backward(ctx, grad_metrics: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        grads = ctx.saved_tensors
        scale = grad_metrics[0].contiguous()
        scaled_grads = tuple(grad * scale for grad in grads)
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            *scaled_grads,
            *([None] * 36),
        )


def full_cuda_rollout_metrics(
    policy: MotorGRUPolicy,
    initial_state: L2FState,
    params: L2FParams,
    loss_config: L2FLossConfig,
    *,
    horizon: int,
    tail_steps: int,
    state_step_decay: float,
    hidden_step_decay: float,
    clf_kappa: float,
    u_soft: float,
    lambda_clf: float,
    lambda_out: float,
    lambda_tail: float,
    lambda_du: float,
    lambda_ddu: float,
    lambda_sat: float,
    noise_seed: int = 0,
    external_force_max: float = 0.0,
    external_torque_max: float = 0.0,
    action_noise_max: float = 0.0,
    observation_noise_max: float = 0.0,
) -> torch.Tensor:
    if not initial_state.position.is_cuda:
        raise RuntimeError("cuda-full backend requires CUDA initial state tensors")
    tensors = _policy_tensors(policy)
    return _FullCudaRollout.apply(
        initial_state.position,
        initial_state.velocity,
        initial_state.rotation,
        initial_state.omega,
        initial_state.motor,
        initial_state.previous_action,
        *tensors,
        horizon,
        tail_steps,
        params.dt,
        params.mass,
        params.gravity,
        params.arm_length,
        params.yaw_drag,
        params.motor_tau,
        params.motor_authority,
        params.inertia_x,
        params.inertia_y,
        params.inertia_z,
        state_step_decay,
        hidden_step_decay,
        loss_config.p_scale,
        loss_config.v_scale,
        loss_config.omega_scale,
        loss_config.huber_beta,
        loss_config.w_p,
        loss_config.w_v,
        loss_config.w_r,
        loss_config.w_omega,
        clf_kappa,
        u_soft,
        lambda_clf,
        lambda_out,
        lambda_tail,
        lambda_du,
        lambda_ddu,
        lambda_sat,
        policy.negative_slope,
        int(noise_seed),
        external_force_max,
        external_torque_max,
        action_noise_max,
        observation_noise_max,
    )
