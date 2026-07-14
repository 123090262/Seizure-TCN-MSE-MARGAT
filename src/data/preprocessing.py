from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from src.data.chbmit_reader import get_channel_set, normalize_channel_name
from src.utils.io import save_array


REFERENCE_NAMES = {"CS2", "REF", "A1", "A2", "AVG"}


def _first_matching_channels(ch_names: list[str], channels: list[str]) -> dict[str, str]:
    normalized_to_raw: dict[str, str] = {}
    for ch_name in ch_names:
        normalized_to_raw.setdefault(normalize_channel_name(ch_name), ch_name)
    return {channel: normalized_to_raw[channel] for channel in channels if channel in normalized_to_raw}


def _electrode_name(ch_name: str) -> str | None:
    normalized = normalize_channel_name(ch_name)
    if normalized == "-":
        return None
    parts = normalized.split("-")
    if len(parts) == 1:
        return normalized
    if len(parts) == 2 and parts[1] in REFERENCE_NAMES:
        return parts[0]
    return None


def _first_matching_electrodes(ch_names: list[str]) -> dict[str, str]:
    electrode_to_raw: dict[str, str] = {}
    for ch_name in ch_names:
        electrode = _electrode_name(ch_name)
        if electrode:
            electrode_to_raw.setdefault(electrode, ch_name)
    return electrode_to_raw


def _derive_bipolar_channels(raw: mne.io.BaseRaw, channels: list[str]) -> mne.io.BaseRaw | None:
    electrode_to_raw = _first_matching_electrodes(raw.ch_names)
    anodes: list[str] = []
    cathodes: list[str] = []
    for channel in channels:
        if "-" not in channel:
            return None
        anode, cathode = channel.split("-", 1)
        if anode not in electrode_to_raw or cathode not in electrode_to_raw:
            return None
        anodes.append(electrode_to_raw[anode])
        cathodes.append(electrode_to_raw[cathode])
    derived = mne.set_bipolar_reference(
        raw,
        anode=anodes,
        cathode=cathodes,
        ch_name=channels,
        drop_refs=True,
        copy=False,
        verbose="ERROR",
    )
    derived.pick(channels)
    return derived


def load_selected_edf(
    edf_path: str | Path,
    channel_set: str,
    custom_channels: list[str] | None = None,
    preload: bool = True,
) -> mne.io.BaseRaw:
    """Load EDF and select normalized bipolar channels."""
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose="ERROR")
    channels = get_channel_set(channel_set, custom_channels)
    selected = _first_matching_channels(raw.ch_names, channels)
    missing = [ch for ch in channels if ch not in selected]
    if missing:
        derived = _derive_bipolar_channels(raw, channels)
        if derived is None:
            raise ValueError(f"{edf_path} is missing selected channels: {missing}")
        return derived
    raw_names = [selected[ch] for ch in channels]
    raw.pick(raw_names)
    rename_map = {raw_name: ch for raw_name, ch in zip(raw_names, channels) if raw_name != ch}
    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def artifact_mask(data: np.ndarray, threshold_uv: float) -> np.ndarray:
    """Return a boolean mask over time points with amplitudes inside threshold."""
    threshold_v = threshold_uv * 1e-6
    return np.max(np.abs(data), axis=0) <= threshold_v


def preprocess_edf(
    edf_path: str | Path,
    output_path: str | Path,
    cfg: Any,
) -> dict[str, object]:
    """Preprocess one EDF file and save a continuous array `[C, T]`."""
    channel_set = str(cfg.preprocessing.channel_set)
    custom_channels = list(getattr(cfg.data, "custom_channels", []) or [])
    raw = load_selected_edf(edf_path, channel_set=channel_set, custom_channels=custom_channels)

    target_sfreq = float(cfg.preprocessing.sampling_rate)
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-3:
        raw.resample(target_sfreq)

    bandpass = cfg.preprocessing.bandpass
    if bool(bandpass.enabled):
        raw.filter(
            l_freq=float(bandpass.low),
            h_freq=float(bandpass.high),
            method="iir",
            iir_params={"order": int(bandpass.order), "ftype": "butter"},
            verbose="ERROR",
        )

    notch = cfg.preprocessing.notch
    if bool(notch.enabled):
        raw.notch_filter(freqs=[float(notch.freq)], verbose="ERROR")

    data = raw.get_data().astype(np.float32)
    valid_fraction = 1.0
    screening = cfg.preprocessing.artifact_screening
    if bool(screening.enabled):
        mask = artifact_mask(data, float(screening.amplitude_threshold_uv))
        valid_fraction = float(mask.mean())

    save_array(data, output_path)
    info = {
        "processed_path": str(output_path),
        "channels": raw.ch_names,
        "sampling_rate": float(raw.info["sfreq"]),
        "n_times": int(raw.n_times),
        "duration_sec": float(raw.n_times / raw.info["sfreq"]),
        "artifact_valid_fraction": valid_fraction,
    }
    raw.close()
    return info


def _existing_preprocessed_info(output_path: Path, cfg: Any) -> dict[str, object] | None:
    if not output_path.exists() or output_path.suffix != ".npy":
        return None
    try:
        data = np.load(output_path, mmap_mode="r")
    except Exception:
        return None
    if data.ndim != 2:
        return None

    channel_set = str(cfg.preprocessing.channel_set)
    custom_channels = list(getattr(cfg.data, "custom_channels", []) or [])
    channels = get_channel_set(channel_set, custom_channels)
    if data.shape[0] != len(channels):
        return None

    sampling_rate = float(cfg.preprocessing.sampling_rate)
    n_times = int(data.shape[1])
    return {
        "processed_path": str(output_path),
        "channels": channels,
        "sampling_rate": sampling_rate,
        "n_times": n_times,
        "duration_sec": float(n_times / sampling_rate),
        "artifact_valid_fraction": float("nan"),
    }


def preprocess_from_metadata(metadata: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Preprocess all rows in a metadata table."""
    processed_dir = Path(cfg.data.processed_dir)
    overwrite = bool(getattr(cfg.preprocessing, "overwrite_processed", False))
    rows: list[dict[str, object]] = []
    records = metadata.to_dict("records")
    total = len(records)
    for index, row in enumerate(records, start=1):
        subject_id = str(row["subject_id"])
        file_stem = Path(str(row["file_name"])).stem
        suffix = ".pt" if str(cfg.preprocessing.save_format) == "pt" else ".npy"
        output_path = processed_dir / subject_id / f"{file_stem}{suffix}"
        info = None if overwrite else _existing_preprocessed_info(output_path, cfg)
        if info is None:
            print(f"[{index}/{total}] preprocessing {subject_id}/{row['file_name']}", flush=True)
            info = preprocess_edf(row["edf_path"], output_path, cfg)
        else:
            print(f"[{index}/{total}] skipping existing {subject_id}/{row['file_name']}", flush=True)
        merged = dict(row)
        merged.update(info)
        merged["channels"] = json.dumps(info["channels"])
        rows.append(merged)
    return pd.DataFrame(rows)
