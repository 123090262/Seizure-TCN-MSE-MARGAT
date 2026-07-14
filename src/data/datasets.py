from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.normalization import ZScoreNormalizer
from src.utils.io import load_array


class EEGWindowDataset(Dataset[dict[str, Any]]):
    """Lazy EEG window dataset returning tensors shaped `[C, T]`."""

    def __init__(
        self,
        windows: list[dict[str, Any]],
        normalizer: ZScoreNormalizer | None = None,
        channels: list[str] | None = None,
        cache: bool = False,
    ) -> None:
        self.windows = windows
        self.normalizer = normalizer
        self.channels = channels or []
        self.cache = cache
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def _load_record(self, path: str) -> np.ndarray:
        if self.cache and path in self._cache:
            return self._cache[path]
        data = load_array(path, mmap_mode="r")
        if self.cache:
            self._cache[path] = data
        return data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.windows[idx]
        data = self._load_record(str(item["data_path"]))
        x = data[:, int(item["start_sample"]): int(item["end_sample"])].astype(np.float32)
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(int(item["label"]), dtype=torch.long),
            "subject_id": str(item["subject_id"]),
            "record_id": str(item["record_id"]),
            "window_start": float(item["window_start"]),
            "window_end": float(item["window_end"]),
            "channels": self.channels,
        }


def make_balanced_sampler(windows: list[dict[str, Any]], num_classes: int | None = None) -> WeightedRandomSampler:
    if not windows:
        raise ValueError("Cannot build a balanced sampler for an empty training set")
    labels = torch.tensor([int(w["label"]) for w in windows], dtype=torch.long)
    if (labels < 0).any():
        raise ValueError("Class labels must be non-negative integers")
    classes = int(num_classes) if num_classes is not None else int(labels.max()) + 1
    if labels.max() >= classes:
        raise ValueError(f"Found label {int(labels.max())}, but num_classes={classes}")
    counts = torch.bincount(labels, minlength=classes).float()
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
