from __future__ import annotations

from typing import Any

import torch


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    name = str(cfg.training.optimizer).lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    raise ValueError(f"Unsupported optimizer: {name}")
