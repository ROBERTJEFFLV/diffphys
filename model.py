from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


STATE_DIM = 22
ERROR_DIM = 18
PREVIOUS_ACTION_DIM = 4
ACTION_DIM = 4


class MotorGRUPolicy(nn.Module):
    """state/error/previous_action -> MLP encoder -> GRU -> 4 motors."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        error_dim: int = ERROR_DIM,
        previous_action_dim: int = PREVIOUS_ACTION_DIM,
        encoder_dim: int = 192,
        hidden_dim: int = 192,
        encoder_depth: int = 2,
        negative_slope: float = 0.05,
    ) -> None:
        super().__init__()
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be >= 1")

        layers: list[nn.Module] = []
        input_dim = state_dim + error_dim + previous_action_dim
        for layer_idx in range(encoder_depth):
            in_features = input_dim if layer_idx == 0 else encoder_dim
            layers.append(nn.Linear(in_features, encoder_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))

        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRUCell(encoder_dim, hidden_dim)
        self.motor_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.negative_slope = negative_slope
        self.hidden_dim = hidden_dim

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

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def forward(
        self,
        state: torch.Tensor,
        error: torch.Tensor,
        previous_action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.initial_hidden(
                state.shape[0],
                device=state.device,
                dtype=state.dtype,
            )

        features = torch.cat((state, error, previous_action), dim=-1)
        encoded = self.encoder(features)
        hidden = self.gru(encoded, hidden)
        motor_command = torch.tanh(
            self.motor_head(F.leaky_relu(hidden, negative_slope=self.negative_slope))
        )
        return motor_command, hidden
