from __future__ import annotations

from typing import Any

import torch

from src.models.baselines.cnn1d_baseline import CNN1DBaseline
from src.models.baselines.tcn_baseline import TCNBaseline
from src.models.tcn_mse_margat import TCNMSEMARGAT


def build_model(
    cfg: Any,
    num_channels: int,
    window_size: int,
    channel_names: list[str] | None = None,
) -> torch.nn.Module:
    name = str(cfg.model.name).lower()
    if name == "tcn_mse_margat":
        if channel_names is None:
            raise ValueError("tcn_mse_margat requires ordered bipolar channel names")
        return TCNMSEMARGAT(cfg, num_channels=num_channels, channel_names=channel_names)
    if name == "cnn1d":
        return CNN1DBaseline(in_channels=num_channels, num_classes=int(cfg.model.num_classes))
    if name == "tcn":
        return TCNBaseline(
            in_channels=num_channels,
            hidden_channels=int(cfg.model.hidden_channels),
            levels=int(cfg.model.levels),
            kernel_size=int(cfg.model.kernel_size),
            dropout=float(cfg.model.dropout),
            num_classes=int(cfg.model.num_classes),
        )
    raise ValueError(f"Unsupported model name: {name}")
