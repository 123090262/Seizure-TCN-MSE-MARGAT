from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNBaseline(nn.Module):
    """TCN baseline.

    Input: `[B, C, T]`
    Output: logits `[B, num_classes]`
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(hidden_channels, kernel_size, 2**i, dropout) for i in range(levels)]
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_channels, num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.blocks(self.proj(x))
        return self.head(h), {}
