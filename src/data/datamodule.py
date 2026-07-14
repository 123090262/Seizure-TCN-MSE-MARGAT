from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from src.data.datasets import EEGWindowDataset, make_balanced_sampler
from src.data.normalization import Normalizer, build_normalizer


def resolve_balance_strategy(cfg: Any) -> str:
    """Resolve the new exclusive strategy, with an explicit legacy migration."""
    training = cfg.training
    configured = training.get("balance_strategy") if hasattr(training, "get") else None
    loss_cfg = training.get("loss", {}) if hasattr(training, "get") else {}
    legacy_weight = loss_cfg.get("class_weight") if hasattr(loss_cfg, "get") else None
    if configured is None:
        strategy = "class_weight" if legacy_weight == "auto" else "none"
    else:
        strategy = str(configured).lower()
        if legacy_weight not in (None, "none") and strategy != "class_weight":
            raise ValueError(
                "Legacy training.loss.class_weight conflicts with training.balance_strategy. "
                "Remove class_weight and select exactly one balance strategy."
            )
    if strategy not in {"sampler", "class_weight", "none"}:
        raise ValueError("training.balance_strategy must be sampler, class_weight, or none")
    return strategy


def build_dataloaders(
    split: dict[str, Any],
    cfg: Any,
    channels: list[str] | None = None,
    normalizer: Normalizer | None = None,
) -> dict[str, DataLoader]:
    train_windows = list(split["train_windows"])
    val_windows = list(split["val_windows"])
    test_windows = list(split["test_windows"])
    if normalizer is None:
        normalizer = build_normalizer(split, cfg, channel_names=channels)
    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)
    train_dataset = EEGWindowDataset(train_windows, normalizer=normalizer, channels=channels, cache=True)
    val_dataset = EEGWindowDataset(val_windows, normalizer=normalizer, channels=channels, cache=True)
    test_dataset = EEGWindowDataset(test_windows, normalizer=normalizer, channels=channels, cache=True)
    strategy = resolve_balance_strategy(cfg)
    sampler = make_balanced_sampler(train_windows, int(cfg.model.num_classes)) if strategy == "sampler" and train_windows else None
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None and bool(train_windows),
            num_workers=num_workers,
        ),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
