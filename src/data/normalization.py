from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.io import load_array, write_json


@dataclass
class ZScoreNormalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6
    channel_names: list[str] | None = None

    @classmethod
    def fit(cls, windows: list[dict[str, object]], eps: float = 1e-6, channel_names: list[str] | None = None) -> "ZScoreNormalizer":
        """Fit per-channel statistics from training windows only."""
        if not windows:
            raise ValueError("Cannot fit normalization on an empty training window list.")
        sums: np.ndarray | None = None
        sq_sums: np.ndarray | None = None
        count = 0
        for window in windows:
            data = load_array(str(window["data_path"]), mmap_mode="r")
            segment = data[:, int(window["start_sample"]): int(window["end_sample"])].astype(np.float64)
            if sums is None:
                sums = np.zeros(segment.shape[0], dtype=np.float64)
                sq_sums = np.zeros(segment.shape[0], dtype=np.float64)
            sums += segment.sum(axis=1)
            sq_sums += np.square(segment).sum(axis=1)
            count += segment.shape[1]
        assert sums is not None and sq_sums is not None
        mean = sums / count
        var = np.maximum(sq_sums / count - np.square(mean), 0.0)
        std = np.sqrt(var)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32), eps=eps, channel_names=channel_names)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean[:, None]) / (self.std[:, None] + self.eps)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        write_json(
            {
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "channel_names": self.channel_names or [],
                "fit_on": "train_only",
                "eps": self.eps,
            },
            path,
        )
