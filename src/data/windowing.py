from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowIndex:
    subject_id: str
    record_id: str
    file_name: str
    data_path: str
    start_sample: int
    end_sample: int
    window_start: float
    window_end: float
    label: int
    block_id: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _parse_intervals(value: object) -> list[dict[str, float]]:
    if isinstance(value, str):
        if not value:
            return []
        return json.loads(value)
    if isinstance(value, list):
        return value
    return []


def _window_label(
    start_sec: float,
    end_sec: float,
    seizure_intervals: list[dict[str, float]],
    interictal_exclusion_sec: float,
) -> int | None:
    for interval in seizure_intervals:
        if start_sec >= float(interval["start_sec"]) and end_sec <= float(interval["end_sec"]):
            return 1
    for interval in seizure_intervals:
        if start_sec < float(interval["end_sec"]) + interictal_exclusion_sec and end_sec > float(interval["start_sec"]) - interictal_exclusion_sec:
            return None
    return 0


def generate_windows_for_record(row: dict[str, object], cfg: Any) -> list[WindowIndex]:
    """Generate non-leaking labels for one continuous recording."""
    sfreq = float(row["sampling_rate"])
    window_size = int(cfg.windowing.window_size_samples)
    ictal_step = max(1, int(window_size * (1.0 - float(cfg.windowing.ictal_overlap))))
    interictal_step = max(1, int(window_size * (1.0 - float(cfg.windowing.interictal_overlap))))
    duration_sec = float(row["duration_sec"])
    n_samples = int(round(duration_sec * sfreq))
    intervals = _parse_intervals(row.get("seizure_intervals", "[]"))
    data_path = str(row.get("processed_path") or row.get("edf_path"))
    subject_id = str(row["subject_id"])
    file_name = str(row["file_name"])
    record_id = file_name.rsplit(".", 1)[0]
    windows: list[WindowIndex] = []

    for interval_idx, interval in enumerate(intervals):
        start = int(round(float(interval["start_sec"]) * sfreq))
        end = int(round(float(interval["end_sec"]) * sfreq))
        for sample_start in range(start, max(start, end - window_size + 1), ictal_step):
            sample_end = sample_start + window_size
            if sample_end > end or sample_end > n_samples:
                continue
            windows.append(
                WindowIndex(
                    subject_id=subject_id,
                    record_id=record_id,
                    file_name=file_name,
                    data_path=data_path,
                    start_sample=sample_start,
                    end_sample=sample_end,
                    window_start=sample_start / sfreq,
                    window_end=sample_end / sfreq,
                    label=1,
                    block_id=f"{record_id}:seizure_{interval_idx}",
                )
            )

    for sample_start in range(0, max(0, n_samples - window_size + 1), interictal_step):
        sample_end = sample_start + window_size
        start_sec = sample_start / sfreq
        end_sec = sample_end / sfreq
        label = _window_label(start_sec, end_sec, intervals, float(cfg.windowing.interictal_exclusion_sec))
        if label != 0:
            continue
        block_idx = int(start_sec // max(float(cfg.windowing.interictal_exclusion_sec), 1.0))
        windows.append(
            WindowIndex(
                subject_id=subject_id,
                record_id=record_id,
                file_name=file_name,
                data_path=data_path,
                start_sample=sample_start,
                end_sample=sample_end,
                window_start=start_sec,
                window_end=end_sec,
                label=0,
                block_id=f"{record_id}:interictal_{block_idx}",
            )
        )
    return windows


def generate_window_index(metadata: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    windows: list[dict[str, object]] = []
    for row in metadata.to_dict("records"):
        windows.extend(window.to_dict() for window in generate_windows_for_record(row, cfg))
    return pd.DataFrame(windows)


def balance_windows(
    windows: pd.DataFrame,
    ratio: float = 1.0,
    seed: int = 42,
    context: str | None = None,
) -> pd.DataFrame:
    """Downsample interictal windows relative to ictal windows."""
    if ratio < 0:
        raise ValueError(f"balance ratio must be non-negative, got {ratio}")
    pos = windows[windows["label"] == 1]
    neg = windows[windows["label"] == 0]
    if pos.empty or neg.empty:
        missing = "seizure" if pos.empty else "non-seizure"
        location = f" in {context}" if context else ""
        warnings.warn(
            f"Skipping class balancing{location}: no {missing} windows; keeping the original set.",
            RuntimeWarning,
            stacklevel=2,
        )
        return windows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_neg = min(len(neg), int(round(len(pos) * ratio)))
    neg = neg.sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def windows_to_records(windows: pd.DataFrame) -> list[dict[str, object]]:
    return windows.to_dict("records")
