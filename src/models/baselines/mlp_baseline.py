from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """Simple MLP baseline for flattened EEG windows."""

    def __init__(self, in_channels: int, window_size: int, hidden: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * window_size, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.net(x), {}
