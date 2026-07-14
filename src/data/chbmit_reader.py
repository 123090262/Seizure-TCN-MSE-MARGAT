from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import mne
import pandas as pd

from src.data.annotations import intervals_to_json, parse_summary_file


COMMON_18_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]

COMMON_21_CHANNELS = COMMON_18_CHANNELS + ["T7-FT9", "FT9-FT10", "FT10-T8"]
COMMON_22_CHANNELS = COMMON_21_CHANNELS + ["P7-T7"]


def normalize_channel_name(name: str) -> str:
    """Normalize CHB-MIT bipolar channel names."""
    name = name.strip().upper()
    name = re.sub(r"\s+", "", name)
    name = "-".join({"01": "O1", "02": "O2"}.get(part, part) for part in name.split("-"))
    name = name.replace("T3", "T7").replace("T4", "T8").replace("T5", "P7").replace("T6", "P8")
    name = re.sub(r"-+", "-", name)
    name = re.sub(r"-\d+$", "", name)
    return name


def get_channel_set(name: str, custom_channels: Iterable[str] | None = None) -> list[str]:
    if name == "common_18":
        return COMMON_18_CHANNELS.copy()
    if name == "common_21":
        return COMMON_21_CHANNELS.copy()
    if name == "common_22":
        return COMMON_22_CHANNELS.copy()
    if name == "custom":
        if custom_channels is None:
            raise ValueError("custom channel_set requires custom_channels.")
        return [normalize_channel_name(ch) for ch in custom_channels]
    raise ValueError(f"Unknown channel_set: {name}")


def find_subject_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.glob("chb*") if path.is_dir())


def read_edf_metadata(edf_path: str | Path, preload: bool = False) -> dict[str, object]:
    """Read lightweight EDF metadata with MNE."""
    edf_path = Path(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    channels = [normalize_channel_name(ch) for ch in raw.ch_names]
    duration_sec = float(raw.n_times / sfreq)
    raw.close()
    return {
        "edf_path": str(edf_path),
        "duration_sec": duration_sec,
        "sampling_rate": sfreq,
        "channels": channels,
    }


def build_metadata(
    root: str | Path,
    channel_set: str = "common_18",
    custom_channels: Iterable[str] | None = None,
    expected_sampling_rate: int = 256,
) -> pd.DataFrame:
    """Build one metadata table from CHB-MIT EDF files and summary annotations."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"CHB-MIT root not found: {root}")

    target_channels = get_channel_set(channel_set, custom_channels)
    rows: list[dict[str, object]] = []
    for subject_dir in find_subject_dirs(root):
        subject_id = subject_dir.name
        summaries = sorted(subject_dir.glob("*summary.txt"))
        if not summaries:
            raise FileNotFoundError(f"No summary file found for {subject_id} in {subject_dir}")
        annotations = parse_summary_file(summaries[0])
        for edf_path in sorted(subject_dir.glob("*.edf")):
            meta = read_edf_metadata(edf_path)
            sfreq = float(meta["sampling_rate"])
            if abs(sfreq - expected_sampling_rate) > 1e-3:
                raise ValueError(f"{edf_path} sampling rate is {sfreq}, expected {expected_sampling_rate}")
            available_channels = set(meta["channels"])  # type: ignore[arg-type]
            missing = [ch for ch in target_channels if ch not in available_channels]
            intervals = annotations.get(edf_path.name, [])
            rows.append(
                {
                    "subject_id": subject_id,
                    "file_name": edf_path.name,
                    "edf_path": str(edf_path),
                    "duration_sec": meta["duration_sec"],
                    "sampling_rate": sfreq,
                    "channels": json.dumps(meta["channels"]),
                    "selected_channels": json.dumps(target_channels),
                    "missing_selected_channels": json.dumps(missing),
                    "seizure_intervals": json.dumps(intervals_to_json(intervals)),
                    "has_seizure": bool(intervals),
                }
            )
    return pd.DataFrame(rows)
