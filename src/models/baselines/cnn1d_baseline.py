from __future__ import annotations

import torch
from torch import nn


class CNN1DBaseline(nn.Module):
    """1D CNN baseline.

    Input: `[B, C, T]`
    Output: logits `[B, num_classes]`
    """

    def __init__(self, in_channels: int, num_classes: int = 2, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.net(x)
        return self.head(h), {}
