from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from env_l2f import L2FParams, L2FState


_EXTENSION: Any | None = None
_LOAD_ERROR: Exception | None = None


def _source_paths() -> list[str]:
    root = Path(__file__).resolve().parent
    return [
        str(root / "cuda_ext" / "binding.cpp"),
        str(root / "cuda_ext" / "l2f_step_kernel.cu"),
        str(root / "cuda_ext" / "full_rollout_kernel.cu"),
    ]


def _discover_cuda_home() -> str | None:
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured and (Path(configured) / "bin" / "nvcc").exists():
        return configured

    torch_cuda = torch.version.cuda or ""
    preferred_major = torch_cuda.split(".", 1)[0]
    candidates = sorted(Path("/usr/local").glob("cuda-*"), reverse=True)
    for candidate in candidates:
        nvcc = candidate / "bin" / "nvcc"
        if not nvcc.exists():
            continue
        if preferred_major and candidate.name.startswith(f"cuda-{preferred_major}."):
            return str(candidate)
    for candidate in [Path("/usr/local/cuda"), *candidates]:
        if (candidate / "bin" / "nvcc").exists():
            return str(candidate)
    return None


def load_extension() -> Any:
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise RuntimeError("L2F CUDA extension is unavailable") from _LOAD_ERROR

    cuda_home = _discover_cuda_home()
    if cuda_home is not None:
        os.environ.setdefault("CUDA_HOME", cuda_home)
    if not torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0"

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        _LOAD_ERROR = RuntimeError("CUDA_HOME is not set; nvcc is required")
        raise RuntimeError("L2F CUDA extension is unavailable") from _LOAD_ERROR

    build_dir = Path(__file__).resolve().parent / ".torch_extensions" / "l2f_cuda_ext"
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        _EXTENSION = load(
            name="l2f_cuda_ext",
            sources=_source_paths(),
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain
        _LOAD_ERROR = exc
        raise RuntimeError("L2F CUDA extension build/load failed") from exc
    return _EXTENSION


def cuda_backend_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        load_extension()
    except RuntimeError:
        return False
    return True


class _CudaL2FStep(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        position: torch.Tensor,
        velocity: torch.Tensor,
        rotation: torch.Tensor,
        omega: torch.Tensor,
        motor: torch.Tensor,
        action: torch.Tensor,
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
        grad_decay: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ext = load_extension()
        tensors = (
            position.contiguous(),
            velocity.contiguous(),
            rotation.contiguous(),
            omega.contiguous(),
            motor.contiguous(),
            action.contiguous(),
        )
        ctx.save_for_backward(*tensors)
        ctx.params = (
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
            float(grad_decay),
        )
        return tuple(ext.step_forward(*tensors, *ctx.params[:-1]))

    @staticmethod
    def backward(
        ctx,
        grad_position: torch.Tensor,
        grad_velocity: torch.Tensor,
        grad_rotation: torch.Tensor,
        grad_omega: torch.Tensor,
        grad_motor: torch.Tensor,
        grad_previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        ext = load_extension()
        saved = ctx.saved_tensors
        grads = (
            grad_position.contiguous(),
            grad_velocity.contiguous(),
            grad_rotation.contiguous(),
            grad_omega.contiguous(),
            grad_motor.contiguous(),
            grad_previous_action.contiguous(),
        )
        outputs = ext.step_backward(*saved, *grads, *ctx.params)
        return (*outputs, *([None] * 11))


def cuda_step(
    state: L2FState,
    action: torch.Tensor,
    params: L2FParams,
    *,
    grad_decay: float,
) -> L2FState:
    if not state.position.is_cuda:
        raise RuntimeError("cuda_step requires CUDA tensors")
    outputs = _CudaL2FStep.apply(
        state.position,
        state.velocity,
        state.rotation,
        state.omega,
        state.motor,
        action,
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
        grad_decay,
    )
    return L2FState(
        position=outputs[0],
        velocity=outputs[1],
        rotation=outputs[2],
        omega=outputs[3],
        motor=outputs[4],
        previous_action=outputs[5],
    )
