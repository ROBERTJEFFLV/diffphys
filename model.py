from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


LEGACY_INPUT_DIM = 40
COMPACT_INPUT_DIM = 22
INTEGRAL_INPUT_DIM = 25
DEFAULT_INPUT_DIM = INTEGRAL_INPUT_DIM
ACTION_DIM = 4
CHECKPOINT_FORMAT_VERSION = 2
POLICY_ARCHITECTURE = "motor-gru-policy"
POLICY_ARCHITECTURE_VERSION = 2
CAPABILITY_DIM = 6
RESPONSE_DIM = 6
INTEGRAL_POSITION_DIM = 3
INTEGRAL_INPUT_SLICE = slice(18, 21)
AUXILIARY_STATE_PREFIXES = (
    "motor_state_head.",
    "capability_head.",
    "response_head.",
    "integral_residual_head.",
    "damping_residual_head.",
)
DAMPING_DEPLOYMENT_STATE_KEYS = (
    "motor_state_head.weight",
    "motor_state_head.bias",
    "damping_residual_head.0.weight",
    "damping_residual_head.0.bias",
    "damping_residual_head.2.weight",
    "damping_residual_head.2.bias",
)
INTEGRAL_DEPLOYMENT_STATE_KEYS = (
    "integral_residual_head.0.weight",
    "integral_residual_head.0.bias",
    "integral_residual_head.2.weight",
    "integral_residual_head.2.bias",
)


class MotorGRUPolicy(nn.Module):
    """Deployable recurrent motor policy with training-only belief heads."""

    def __init__(
        self,
        observation_dim: int = DEFAULT_INPUT_DIM,
        encoder_dim: int = 192,
        hidden_dim: int = 192,
        encoder_depth: int = 2,
        negative_slope: float = 0.05,
        enable_integral_residual: bool = False,
        enable_damping_residual: bool = False,
        integral_residual_hidden_dim: int = 16,
        damping_residual_hidden_dim: int = 32,
        integral_residual_scale: float = 1.0,
        damping_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be >= 1")

        layers: list[nn.Module] = []
        if observation_dim not in (LEGACY_INPUT_DIM, COMPACT_INPUT_DIM, INTEGRAL_INPUT_DIM):
            raise ValueError("observation_dim must be 22, 25, or legacy-compatible 40")
        if (enable_integral_residual or enable_damping_residual) and observation_dim != INTEGRAL_INPUT_DIM:
            raise ValueError("residual control branches require the 25D integral observation")
        if integral_residual_hidden_dim < 1 or damping_residual_hidden_dim < 1:
            raise ValueError("residual hidden dimensions must be positive")
        if integral_residual_scale < 0.0 or damping_residual_scale < 0.0:
            raise ValueError("residual scales must be non-negative")
        input_dim = observation_dim
        for layer_idx in range(encoder_depth):
            in_features = input_dim if layer_idx == 0 else encoder_dim
            layers.append(nn.Linear(in_features, encoder_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))

        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRUCell(encoder_dim, hidden_dim)
        # Keep this name and action path stable for checkpoint/MATLAB compatibility.
        self.motor_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.motor_state_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.capability_head = nn.Linear(hidden_dim, CAPABILITY_DIM)
        self.response_head = nn.Linear(hidden_dim + ACTION_DIM, RESPONSE_DIM)
        self.integral_residual_head = (
            nn.Sequential(
                nn.Linear(INTEGRAL_POSITION_DIM, integral_residual_hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Linear(integral_residual_hidden_dim, ACTION_DIM),
            )
            if enable_integral_residual
            else None
        )
        damping_input_dim = hidden_dim + 3 + ACTION_DIM + ACTION_DIM
        self.damping_residual_head = (
            nn.Sequential(
                nn.Linear(damping_input_dim, damping_residual_hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Linear(damping_residual_hidden_dim, ACTION_DIM),
            )
            if enable_damping_residual
            else None
        )
        self.negative_slope = negative_slope
        self.hidden_dim = hidden_dim
        self.observation_dim = observation_dim
        self.enable_integral_residual = bool(enable_integral_residual)
        self.enable_damping_residual = bool(enable_damping_residual)
        self.integral_residual_scale = float(integral_residual_scale)
        self.damping_residual_scale = float(damping_residual_scale)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.uniform_(self.motor_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.motor_head.bias)
        nn.init.orthogonal_(self.motor_state_head.weight)
        nn.init.zeros_(self.motor_state_head.bias)
        nn.init.orthogonal_(self.capability_head.weight)
        nn.init.zeros_(self.capability_head.bias)
        nn.init.orthogonal_(self.response_head.weight)
        nn.init.zeros_(self.response_head.bias)
        if self.integral_residual_head is not None:
            self._reset_residual_head(self.integral_residual_head)
        if self.damping_residual_head is not None:
            self._reset_residual_head(self.damping_residual_head)

    @staticmethod
    def _reset_residual_head(head: nn.Sequential) -> None:
        first = head[0]
        last = head[2]
        if not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear):
            raise TypeError("residual head must use Linear, activation, Linear")
        nn.init.orthogonal_(first.weight)
        nn.init.zeros_(first.bias)
        # Exact zero output preserves the loaded baseline action at update zero.
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def architecture_metadata(self) -> dict[str, object]:
        """Describe this policy without changing or evaluating its forward path."""

        return {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": POLICY_ARCHITECTURE,
            "architecture_version": POLICY_ARCHITECTURE_VERSION,
            "observation_dim": self.observation_dim,
            "encoder_dim": int(self.encoder[0].out_features),
            "encoder_depth": len(self.encoder) // 2,
            "hidden_dim": self.hidden_dim,
            "action_dim": ACTION_DIM,
            "enable_integral_residual": self.enable_integral_residual,
            "enable_rate_damping_residual": self.enable_damping_residual,
            "deployment_privileged_inputs": False,
        }

    def _encode_recurrent(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.initial_hidden(
                observation.shape[0],
                device=observation.device,
                dtype=observation.dtype,
            )

        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observation must have shape [batch,{self.observation_dim}], "
                f"got {tuple(observation.shape)}"
            )
        encoded = self.encoder(observation)
        hidden = self.gru(encoded, hidden)
        latent = F.leaky_relu(hidden, negative_slope=self.negative_slope)
        return latent, hidden

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deployment interface; privileged quantities never enter this path."""
        latent, hidden = self._encode_recurrent(observation, hidden)
        motor_command, _ = self._action_from_latent(observation, latent)
        return motor_command, hidden

    def _action_from_latent(
        self,
        observation: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        main_logits = self.motor_head(latent)
        zeros = torch.zeros_like(main_logits)
        integral_logits = zeros
        if self.integral_residual_head is not None:
            integral_body = observation[:, 18:21]
            integral_logits = (
                self.integral_residual_scale * self.integral_residual_head(integral_body)
            )

        damping_logits = zeros
        if self.damping_residual_head is not None:
            motor_state_hat = torch.tanh(self.motor_state_head(latent))
            omega = observation[:, 15:18]
            previous_action = observation[:, 21:25]
            damping_input = torch.cat(
                (latent, omega, previous_action, motor_state_hat),
                dim=-1,
            )
            damping_logits = (
                self.damping_residual_scale * self.damping_residual_head(damping_input)
            )

        main_action = torch.tanh(main_logits)
        integral_action = torch.tanh(main_logits + integral_logits)
        action = torch.tanh(main_logits + integral_logits + damping_logits)
        details: dict[str, torch.Tensor] = {
            "main_logits": main_logits,
            "integral_residual_logits": integral_logits,
            "damping_residual_logits": damping_logits,
            "integral_action_contribution": integral_action - main_action,
            "damping_action_contribution": action - integral_action,
        }
        if self.damping_residual_head is not None:
            details["motor_state"] = motor_state_hat
        return action, details

    def forward_with_aux(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Training interface returning predictions from the recurrent belief."""
        latent, hidden = self._encode_recurrent(observation, hidden)
        motor_command, action_details = self._action_from_latent(observation, latent)
        motor_state_hat = action_details.get("motor_state")
        if motor_state_hat is None:
            motor_state_hat = torch.tanh(self.motor_state_head(latent))
        auxiliary = {
            "motor_state": motor_state_hat,
            "capability": self.capability_head(latent),
            # The action is an explanatory input, not an auxiliary gradient path
            # into the deployable motor head.
            "response": self.response_head(
                torch.cat((latent, motor_command.detach()), dim=-1)
            ),
            **action_details,
        }
        return motor_command, hidden, auxiliary

    def load_compatible_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[list[str], list[str]]:
        """Load old action-only checkpoints while rejecting unrelated mismatch."""
        compatible = convert_observation_state_dict(
            state_dict,
            target_observation_dim=self.observation_dim,
        )
        if any(key.startswith("damping_residual_head.") for key in compatible):
            missing_damping_dependencies = [
                key for key in DAMPING_DEPLOYMENT_STATE_KEYS if key not in compatible
            ]
            if missing_damping_dependencies:
                raise RuntimeError(
                    "incomplete damping deployment checkpoint: "
                    f"missing={missing_damping_dependencies}"
                )
        if any(key.startswith("integral_residual_head.") for key in compatible):
            missing_integral_dependencies = [
                key for key in INTEGRAL_DEPLOYMENT_STATE_KEYS if key not in compatible
            ]
            if missing_integral_dependencies:
                raise RuntimeError(
                    "incomplete integral deployment checkpoint: "
                    f"missing={missing_integral_dependencies}"
                )
        incompatible = self.load_state_dict(compatible, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [
            key
            for key in missing
            if not key.startswith(AUXILIARY_STATE_PREFIXES)
        ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                "incompatible MotorGRUPolicy checkpoint: "
                f"missing={invalid_missing}, unexpected={unexpected}"
            )
        return missing, unexpected


def convert_observation_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    target_observation_dim: int,
) -> dict[str, torch.Tensor]:
    """Convert the first encoder layer while preserving zero-integral behavior."""
    if "encoder.0.weight" not in state_dict:
        return dict(state_dict)
    source = state_dict["encoder.0.weight"]
    source_dim = int(source.shape[1])
    if source_dim == target_observation_dim:
        return dict(state_dict)
    if target_observation_dim not in (COMPACT_INPUT_DIM, INTEGRAL_INPUT_DIM):
        raise RuntimeError(
            f"cannot convert observation width {source_dim} to {target_observation_dim}"
        )

    converted = dict(state_dict)
    if source_dim == LEGACY_INPUT_DIM:
        state_weight = source[:, :18]
        error_weight = source[:, 18:36]
        action_weight = source[:, 36:40]
        new_weight = source.new_zeros(source.shape[0], target_observation_dim)
        new_weight[:, :18] = state_weight + error_weight
        action_start = 18 if target_observation_dim == COMPACT_INPUT_DIM else 21
        new_weight[:, action_start:action_start + 4] = action_weight
        identity_offset = source.new_zeros(18)
        identity_offset[(6, 10, 14),] = 1.0
        converted["encoder.0.weight"] = new_weight
        converted["encoder.0.bias"] = (
            state_dict["encoder.0.bias"] - error_weight @ identity_offset
        )
        return converted

    if source_dim == COMPACT_INPUT_DIM and target_observation_dim == INTEGRAL_INPUT_DIM:
        new_weight = source.new_zeros(source.shape[0], INTEGRAL_INPUT_DIM)
        new_weight[:, :18] = source[:, :18]
        new_weight[:, 21:25] = source[:, 18:22]
        converted["encoder.0.weight"] = new_weight
        return converted

    raise RuntimeError(
        f"cannot losslessly convert observation width {source_dim} to {target_observation_dim}"
    )


def compensate_integral_input_scale_(
    policy: MotorGRUPolicy,
    multiplier: float,
) -> None:
    """Preserve the policy function when its three integral inputs are scaled.

    This is intentionally an initialization-time parameter transform.  Both
    deployment paths that directly consume the integral feature are adjusted;
    recurrent, motor, damping and auxiliary weights are left untouched.
    """

    if multiplier <= 0.0:
        raise ValueError("integral input compensation requires multiplier > 0")
    if policy.observation_dim != INTEGRAL_INPUT_DIM:
        raise ValueError("integral input compensation requires a 25D policy")
    first_encoder = policy.encoder[0]
    if not isinstance(first_encoder, nn.Linear):
        raise TypeError("policy encoder must start with Linear")
    with torch.no_grad():
        first_encoder.weight[:, INTEGRAL_INPUT_SLICE].div_(float(multiplier))
        if policy.integral_residual_head is not None:
            first_integral = policy.integral_residual_head[0]
            if not isinstance(first_integral, nn.Linear):
                raise TypeError("integral residual head must start with Linear")
            first_integral.weight.div_(float(multiplier))
