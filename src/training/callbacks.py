from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, epoch: int, metrics: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    map_location: torch.device | str | None = None,
) -> dict[str, Any]:
    """Restore model weights and return the full checkpoint payload."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return checkpoint
