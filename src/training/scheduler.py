from __future__ import annotations

from typing import Any

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Any) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(cfg.training.scheduler).lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.training.epochs))
    raise ValueError(f"Unsupported scheduler: {name}")
