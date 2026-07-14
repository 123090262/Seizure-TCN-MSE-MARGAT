from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import load_array, read_json, write_json


@dataclass
class ZScoreNormalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6
    channel_names: list[str] | None = None

    @classmethod
    def fit(
        cls,
        windows: list[dict[str, object]],
        eps: float = 1e-6,
        channel_names: list[str] | None = None,
    ) -> "ZScoreNormalizer":
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

    def transform(self, x: np.ndarray, subject_id: str | None = None) -> np.ndarray:
        return ((x - self.mean[:, None]) / (self.std[:, None] + self.eps)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        write_json(
            {
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "channel_names": self.channel_names or [],
                "method": "zscore",
                "scope": "train_only",
                "fit_on": "train_only",
                "eps": self.eps,
            },
            path,
        )


@dataclass
class SubjectGlobalZScoreNormalizer:
    """Per-subject, per-channel Z-score statistics fitted on full records."""

    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]
    eps: float = 1e-6
    channel_names: list[str] | None = None
    source_metadata_path: str | None = None
    source_metadata_sha256: str | None = None

    @classmethod
    def fit_records(
        cls,
        records: list[dict[str, object]],
        *,
        eps: float = 1e-6,
        channel_names: list[str] | None = None,
        chunk_samples: int = 1_000_000,
        source_metadata_path: str | None = None,
        source_metadata_sha256: str | None = None,
    ) -> "SubjectGlobalZScoreNormalizer":
        if not records:
            raise ValueError("Cannot fit subject-global normalization on empty metadata.")
        if chunk_samples < 1:
            raise ValueError("chunk_samples must be positive")

        paths_by_subject: dict[str, set[str]] = {}
        for record in records:
            subject_id = str(record["subject_id"])
            raw_path = record.get("processed_path", record.get("data_path"))
            if raw_path is None:
                raise ValueError("Subject-global normalization records require processed_path or data_path")
            paths_by_subject.setdefault(subject_id, set()).add(str(raw_path))

        means: dict[str, np.ndarray] = {}
        stds: dict[str, np.ndarray] = {}
        expected_channels = len(channel_names) if channel_names else None
        for subject_id, paths in sorted(paths_by_subject.items()):
            sums: np.ndarray | None = None
            sq_sums: np.ndarray | None = None
            count = 0
            for path in sorted(paths):
                data = load_array(path, mmap_mode="r")
                if data.ndim != 2:
                    raise ValueError(f"Expected [channels,time] EEG array for {path}, got {data.shape}")
                if expected_channels is not None and data.shape[0] != expected_channels:
                    raise ValueError(
                        f"Channel count mismatch for {path}: expected {expected_channels}, got {data.shape[0]}"
                    )
                if sums is None:
                    sums = np.zeros(data.shape[0], dtype=np.float64)
                    sq_sums = np.zeros(data.shape[0], dtype=np.float64)
                elif data.shape[0] != len(sums):
                    raise ValueError(f"Inconsistent channel count for subject {subject_id}: {path}")
                for start in range(0, data.shape[1], chunk_samples):
                    chunk = np.asarray(data[:, start : start + chunk_samples], dtype=np.float64)
                    sums += chunk.sum(axis=1)
                    sq_sums += np.square(chunk).sum(axis=1)
                    count += chunk.shape[1]
            if sums is None or sq_sums is None or count == 0:
                raise ValueError(f"No EEG samples found for subject {subject_id}")
            mean = sums / count
            variance = np.maximum(sq_sums / count - np.square(mean), 0.0)
            means[subject_id] = mean.astype(np.float32)
            stds[subject_id] = np.sqrt(variance).astype(np.float32)
        return cls(
            means=means,
            stds=stds,
            eps=eps,
            channel_names=channel_names,
            source_metadata_path=source_metadata_path,
            source_metadata_sha256=source_metadata_sha256,
        )

    def transform(self, x: np.ndarray, subject_id: str | None = None) -> np.ndarray:
        if subject_id is None:
            raise ValueError("subject_id is required for subject-global normalization")
        key = str(subject_id)
        if key not in self.means:
            raise KeyError(f"No subject-global normalization statistics for {key}")
        return ((x - self.means[key][:, None]) / (self.stds[key][:, None] + self.eps)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        write_json(
            {
                "method": "zscore",
                "scope": "subject_global",
                "fit_on": "all_preprocessed_records_per_subject",
                "means": {subject: value.tolist() for subject, value in sorted(self.means.items())},
                "stds": {subject: value.tolist() for subject, value in sorted(self.stds.items())},
                "channel_names": self.channel_names or [],
                "eps": self.eps,
                "source_metadata_path": self.source_metadata_path,
                "source_metadata_sha256": self.source_metadata_sha256,
            },
            path,
        )


Normalizer = ZScoreNormalizer | SubjectGlobalZScoreNormalizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_normalizer(path: str | Path, *, default_eps: float = 1e-6) -> Normalizer:
    stats = read_json(path)
    scope = str(stats.get("scope", "train_only"))
    channel_names = list(stats.get("channel_names", [])) or None
    eps = float(stats.get("eps", default_eps))
    if scope == "train_only":
        return ZScoreNormalizer(
            mean=np.asarray(stats["mean"], dtype=np.float32),
            std=np.asarray(stats["std"], dtype=np.float32),
            eps=eps,
            channel_names=channel_names,
        )
    if scope == "subject_global":
        return SubjectGlobalZScoreNormalizer(
            means={key: np.asarray(value, dtype=np.float32) for key, value in stats["means"].items()},
            stds={key: np.asarray(value, dtype=np.float32) for key, value in stats["stds"].items()},
            eps=eps,
            channel_names=channel_names,
            source_metadata_path=stats.get("source_metadata_path"),
            source_metadata_sha256=stats.get("source_metadata_sha256"),
        )
    raise ValueError(f"Unsupported normalization scope in {path}: {scope}")


def build_normalizer(
    split: dict[str, Any],
    cfg: Any,
    *,
    channel_names: list[str] | None = None,
) -> Normalizer:
    scope = str(cfg.normalization.scope)
    eps = float(cfg.normalization.eps)
    if scope == "train_only":
        return ZScoreNormalizer.fit(
            list(split["train_windows"]),
            eps=eps,
            channel_names=channel_names,
        )
    if scope != "subject_global":
        raise ValueError(f"normalization.scope must be train_only or subject_global, got {scope!r}")

    configured_metadata = (
        cfg.normalization.get("metadata_path")
        if hasattr(cfg.normalization, "get")
        else None
    )
    metadata_path = Path(str(configured_metadata or cfg.data.metadata_path))
    stats_path = Path(str(cfg.normalization.stats_path))
    if not metadata_path.exists():
        raise FileNotFoundError(f"Subject-global source metadata not found: {metadata_path}")
    metadata_sha256 = _sha256(metadata_path)
    if stats_path.exists():
        cached = json.loads(stats_path.read_text(encoding="utf-8"))
        if cached.get("scope") != "subject_global":
            raise ValueError(f"Normalization cache has the wrong scope: {stats_path}")
        if cached.get("source_metadata_sha256") != metadata_sha256:
            raise ValueError(
                f"Normalization cache does not match current metadata: {stats_path}. "
                "Regenerate the isolated cache before training."
            )
        if float(cached.get("eps", eps)) != eps:
            raise ValueError(f"Normalization cache epsilon does not match the config: {stats_path}")
        cached_channels = list(cached.get("channel_names", []))
        if channel_names and cached_channels != channel_names:
            raise ValueError(f"Normalization cache channels do not match the config: {stats_path}")
        return load_normalizer(stats_path, default_eps=eps)

    print(f"Computing subject-global normalization statistics from {metadata_path}", flush=True)
    metadata = pd.read_csv(metadata_path)
    normalizer = SubjectGlobalZScoreNormalizer.fit_records(
        metadata.to_dict("records"),
        eps=eps,
        channel_names=channel_names,
        source_metadata_path=str(metadata_path),
        source_metadata_sha256=metadata_sha256,
    )
    normalizer.save(stats_path)
    print(f"Saved subject-global normalization cache: {stats_path}", flush=True)
    return normalizer
