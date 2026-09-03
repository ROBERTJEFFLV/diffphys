from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch


ACTION_STATE_KEYS = (
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
    "gru.weight_ih",
    "gru.weight_hh",
    "gru.bias_ih",
    "gru.bias_hh",
    "motor_head.weight",
    "motor_head.bias",
)
DEFAULT_NEGATIVE_SLOPE = 0.05
DEPLOYMENT_INPUT_DIMS = (22, 25, 40)
DEPLOYMENT_ACTION_DIM = 4
INTEGRAL_RESIDUAL_KEYS = (
    "integral_residual_head.0.weight",
    "integral_residual_head.0.bias",
    "integral_residual_head.2.weight",
    "integral_residual_head.2.bias",
)
DAMPING_RESIDUAL_KEYS = (
    "motor_state_head.weight",
    "motor_state_head.bias",
    "damping_residual_head.0.weight",
    "damping_residual_head.0.bias",
    "damping_residual_head.2.weight",
    "damping_residual_head.2.bias",
)


def mode_from_observation_dim(width: int) -> str:
    modes = {22: "compact22", 25: "integral25", 40: "legacy40"}
    try:
        return modes[int(width)]
    except KeyError as exc:
        raise ValueError(f"unsupported observation width: {width}") from exc


def _as_double_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def _state_dict_from_checkpoint(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a state_dict or a mapping containing 'model'")
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint['model'] must be a state_dict mapping")
    return state_dict


def _checkpoint_arg(checkpoint: object, name: str) -> Any | None:
    if not isinstance(checkpoint, Mapping):
        return None
    checkpoint_args = checkpoint.get("args")
    if isinstance(checkpoint_args, Mapping):
        return checkpoint_args.get(name)
    return getattr(checkpoint_args, name, None)


def _validate_action_state_dict(state_dict: Mapping[str, torch.Tensor]) -> None:
    missing = [key for key in ACTION_STATE_KEYS if key not in state_dict]
    if missing:
        raise KeyError(f"checkpoint is missing MotorGRUPolicy action keys: {missing}")

    for key in ACTION_STATE_KEYS:
        if not isinstance(state_dict[key], torch.Tensor):
            raise TypeError(f"checkpoint value {key!r} must be a torch.Tensor")

    encoder0_w = state_dict["encoder.0.weight"]
    encoder0_b = state_dict["encoder.0.bias"]
    encoder1_w = state_dict["encoder.2.weight"]
    encoder1_b = state_dict["encoder.2.bias"]
    gru_w_ih = state_dict["gru.weight_ih"]
    gru_w_hh = state_dict["gru.weight_hh"]
    gru_b_ih = state_dict["gru.bias_ih"]
    gru_b_hh = state_dict["gru.bias_hh"]
    motor_head_w = state_dict["motor_head.weight"]
    motor_head_b = state_dict["motor_head.bias"]

    if encoder0_w.ndim != 2 or encoder0_w.shape[1] not in DEPLOYMENT_INPUT_DIMS:
        raise ValueError(
            "encoder.0.weight must use a supported deployed observation width (22, 25, or 40)"
        )
    encoder_dim = int(encoder0_w.shape[0])
    expected_shapes = {
        "encoder.0.bias": (encoder_dim,),
        "encoder.2.weight": (encoder_dim, encoder_dim),
        "encoder.2.bias": (encoder_dim,),
    }

    if gru_w_hh.ndim != 2 or gru_w_hh.shape[0] != 3 * gru_w_hh.shape[1]:
        raise ValueError("gru.weight_hh must have shape [3*hidden_dim,hidden_dim]")
    hidden_dim = int(gru_w_hh.shape[1])
    expected_shapes.update(
        {
            "gru.weight_ih": (3 * hidden_dim, encoder_dim),
            "gru.bias_ih": (3 * hidden_dim,),
            "gru.bias_hh": (3 * hidden_dim,),
            "motor_head.weight": (DEPLOYMENT_ACTION_DIM, hidden_dim),
            "motor_head.bias": (DEPLOYMENT_ACTION_DIM,),
        }
    )
    tensors = {
        "encoder.0.bias": encoder0_b,
        "encoder.2.weight": encoder1_w,
        "encoder.2.bias": encoder1_b,
        "gru.weight_ih": gru_w_ih,
        "gru.bias_ih": gru_b_ih,
        "gru.bias_hh": gru_b_hh,
        "motor_head.weight": motor_head_w,
        "motor_head.bias": motor_head_b,
    }
    for key, expected in expected_shapes.items():
        actual = tuple(tensors[key].shape)
        if actual != expected:
            raise ValueError(f"{key} must have shape {list(expected)}, got {list(actual)}")


def _required_optional_keys(
    state_dict: Mapping[str, torch.Tensor],
    keys: tuple[str, ...],
    *,
    enabled: bool,
    branch_name: str,
    deployment_indicator: str | None = None,
) -> None:
    present = [key for key in keys if key in state_dict]
    if enabled and len(present) != len(keys):
        missing = [key for key in keys if key not in state_dict]
        raise KeyError(f"enabled {branch_name} is missing checkpoint keys: {missing}")
    active_present = (
        [key for key in present if key.startswith(deployment_indicator)]
        if deployment_indicator is not None
        else present
    )
    if not enabled and active_present:
        raise ValueError(
            f"checkpoint contains {branch_name} weights but its deployment flag is disabled"
        )


def _export_linear(
    payload: dict[str, np.ndarray],
    state_dict: Mapping[str, torch.Tensor],
    *,
    source_prefix: str,
    target_prefix: str,
) -> None:
    payload[f"{target_prefix}_weight"] = _as_double_array(
        state_dict[f"{source_prefix}.weight"]
    )
    payload[f"{target_prefix}_bias"] = _as_double_array(
        state_dict[f"{source_prefix}.bias"]
    ).reshape(1, -1)


def build_matlab_export(
    checkpoint: object,
    *,
    source_checkpoint: str = "",
    negative_slope: float | None = None,
    integral_input_multiplier: float | None = None,
) -> dict[str, np.ndarray]:
    """Build an action-only MATLAB payload from old or auxiliary checkpoints."""
    state_dict = _state_dict_from_checkpoint(checkpoint)
    _validate_action_state_dict(state_dict)
    if negative_slope is None:
        stored_slope = _checkpoint_arg(checkpoint, "negative_slope")
        negative_slope = DEFAULT_NEGATIVE_SLOPE if stored_slope is None else float(stored_slope)
    hidden_dim = int(state_dict["gru.weight_hh"].shape[1])
    input_dim = int(state_dict["encoder.0.weight"].shape[1])
    stored_mode = _checkpoint_arg(checkpoint, "observation_mode")
    observation_mode = mode_from_observation_dim(input_dim) if stored_mode is None else str(stored_mode)
    if observation_mode != mode_from_observation_dim(input_dim):
        raise ValueError(
            f"checkpoint observation_mode={observation_mode!r} disagrees with encoder width {input_dim}"
        )
    integral_limit = _checkpoint_arg(checkpoint, "integral_limit")
    integral_leak = _checkpoint_arg(checkpoint, "integral_leak")
    integral_input_frame = _checkpoint_arg(checkpoint, "integral_input_frame")
    integral_input_frame = "world" if integral_input_frame is None else str(integral_input_frame)
    if integral_input_frame not in ("world", "body"):
        raise ValueError("integral_input_frame must be 'world' or 'body'")
    stored_integral_input_multiplier = _checkpoint_arg(
        checkpoint,
        "integral_input_multiplier",
    )
    if integral_input_multiplier is None:
        integral_input_multiplier = (
            1.0
            if stored_integral_input_multiplier is None
            else float(stored_integral_input_multiplier)
        )
    if not np.isfinite(integral_input_multiplier) or integral_input_multiplier < 0.0:
        raise ValueError("integral_input_multiplier must be finite and non-negative")
    stored_integral_enabled = _checkpoint_arg(checkpoint, "enable_integral_residual")
    stored_damping_enabled = _checkpoint_arg(checkpoint, "enable_rate_damping_residual")
    enable_integral_residual = (
        any(key in state_dict for key in INTEGRAL_RESIDUAL_KEYS)
        if stored_integral_enabled is None
        else bool(stored_integral_enabled)
    )
    enable_damping_residual = (
        any(key in state_dict for key in DAMPING_RESIDUAL_KEYS if key.startswith("damping_"))
        if stored_damping_enabled is None
        else bool(stored_damping_enabled)
    )
    _required_optional_keys(
        state_dict,
        INTEGRAL_RESIDUAL_KEYS,
        enabled=enable_integral_residual,
        branch_name="integral residual",
    )
    _required_optional_keys(
        state_dict,
        DAMPING_RESIDUAL_KEYS,
        enabled=enable_damping_residual,
        branch_name="damping residual",
        deployment_indicator="damping_residual_head.",
    )
    if (enable_integral_residual or enable_damping_residual) and input_dim != 25:
        raise ValueError("residual deployment branches require a 25D policy")
    integral_residual_scale = _checkpoint_arg(checkpoint, "integral_residual_scale")
    damping_residual_scale = _checkpoint_arg(checkpoint, "damping_residual_scale")
    payload = {
        "encoder_0_weight": _as_double_array(state_dict["encoder.0.weight"]),
        "encoder_0_bias": _as_double_array(state_dict["encoder.0.bias"]).reshape(1, -1),
        "encoder_2_weight": _as_double_array(state_dict["encoder.2.weight"]),
        "encoder_2_bias": _as_double_array(state_dict["encoder.2.bias"]).reshape(1, -1),
        "gru_weight_ih": _as_double_array(state_dict["gru.weight_ih"]),
        "gru_weight_hh": _as_double_array(state_dict["gru.weight_hh"]),
        "gru_bias_ih": _as_double_array(state_dict["gru.bias_ih"]).reshape(1, -1),
        "gru_bias_hh": _as_double_array(state_dict["gru.bias_hh"]).reshape(1, -1),
        "motor_head_weight": _as_double_array(state_dict["motor_head.weight"]),
        "motor_head_bias": _as_double_array(state_dict["motor_head.bias"]).reshape(1, -1),
        "negative_slope": np.array([[float(negative_slope)]], dtype=np.float64),
        "hidden_dim": np.array([[hidden_dim]], dtype=np.float64),
        "input_dim": np.array([[input_dim]], dtype=np.float64),
        "observation_mode": np.array(observation_mode),
        "integral_input_frame": np.array(integral_input_frame),
        "integral_input_multiplier": np.array(
            [[float(integral_input_multiplier)]], dtype=np.float64
        ),
        "integral_limit": np.array([[0.5 if integral_limit is None else float(integral_limit)]], dtype=np.float64),
        "integral_leak": np.array([[0.0 if integral_leak is None else float(integral_leak)]], dtype=np.float64),
        "enable_integral_residual": np.array([[float(enable_integral_residual)]], dtype=np.float64),
        "enable_rate_damping_residual": np.array([[float(enable_damping_residual)]], dtype=np.float64),
        "integral_residual_scale": np.array([[1.0 if integral_residual_scale is None else float(integral_residual_scale)]], dtype=np.float64),
        "damping_residual_scale": np.array([[1.0 if damping_residual_scale is None else float(damping_residual_scale)]], dtype=np.float64),
        "source_checkpoint": np.array([[source_checkpoint]], dtype=object),
    }
    if enable_integral_residual:
        _export_linear(
            payload,
            state_dict,
            source_prefix="integral_residual_head.0",
            target_prefix="integral_residual_0",
        )
        _export_linear(
            payload,
            state_dict,
            source_prefix="integral_residual_head.2",
            target_prefix="integral_residual_2",
        )
    if enable_damping_residual:
        _export_linear(
            payload,
            state_dict,
            source_prefix="motor_state_head",
            target_prefix="motor_state_head",
        )
        _export_linear(
            payload,
            state_dict,
            source_prefix="damping_residual_head.0",
            target_prefix="damping_residual_0",
        )
        _export_linear(
            payload,
            state_dict,
            source_prefix="damping_residual_head.2",
            target_prefix="damping_residual_2",
        )
    return payload


def export_checkpoint_to_mat(
    checkpoint_path: Path,
    output_path: Path,
    *,
    negative_slope: float | None = None,
    integral_input_multiplier: float | None = None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    export = build_matlab_export(
        checkpoint,
        source_checkpoint=str(checkpoint_path),
        negative_slope=negative_slope,
        integral_input_multiplier=integral_input_multiplier,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(output_path, export, do_compression=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MotorGRUPolicy .pt weights to a MATLAB .mat file.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--negative-slope",
        type=float,
        default=None,
        help="Override LeakyReLU slope; defaults to checkpoint args or 0.05.",
    )
    parser.add_argument(
        "--integral-input-multiplier",
        type=float,
        default=None,
        help="Override the checkpoint multiplier for sensitivity evaluation.",
    )
    args = parser.parse_args()

    export_checkpoint_to_mat(
        args.checkpoint,
        args.output,
        negative_slope=args.negative_slope,
        integral_input_multiplier=args.integral_input_multiplier,
    )
    print(f"exported {args.checkpoint} -> {args.output}")


if __name__ == "__main__":
    main()
