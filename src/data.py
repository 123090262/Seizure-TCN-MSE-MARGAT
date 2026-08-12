"""CHB-MIT preprocessing, compact caches, and reproducible data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)
CACHE_VERSION = 1


def find_edf_files(raw_dir: Path) -> list[Path]:
    """Find EDF files recursively without relying on extension case."""
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".edf"
    )


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Load YAML files left-to-right; later values override earlier ones."""
    config: dict[str, Any] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            content = yaml.safe_load(stream) or {}
        if not isinstance(content, dict):
            raise ValueError(f"Config must contain a mapping: {path}")
        config = _merge(config, content)
    return config


def _balance(
    indices: np.ndarray, labels: np.ndarray, ratio: float, rng: np.random.Generator
) -> np.ndarray:
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError(
            "Every split must contain both seizure and non-seizure windows"
        )
    wanted = min(len(negative), max(1, int(round(len(positive) * ratio))))
    chosen = rng.choice(negative, size=wanted, replace=False)
    return rng.permutation(np.concatenate([positive, chosen])).astype(np.int64)


def make_split(
    labels: np.ndarray,
    patients: np.ndarray,
    protocol: str,
    split_id: int | str,
    seed: int,
    folds: int,
    val_fraction: float,
    balance_ratio: float,
) -> dict[str, np.ndarray]:
    """Create balanced mixed-window or leave-one-patient-out indices."""
    labels = np.asarray(labels, dtype=np.int64)
    patients = np.asarray(patients)
    if len(labels) != len(patients):
        raise ValueError("labels and patients must have equal length")

    all_indices = np.arange(len(labels))
    rng = np.random.default_rng(seed)
    if protocol == "mixed_10fold":
        fold = int(split_id)
        if not 0 <= fold < folds:
            raise ValueError(f"fold must be in [0, {folds - 1}], got {fold}")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        train_val, test = list(splitter.split(all_indices, labels))[fold]
        train, val = train_test_split(
            train_val,
            test_size=val_fraction,
            stratify=labels[train_val],
            random_state=seed + fold,
        )
    elif protocol == "lopo":
        test_patient = str(split_id)
        test = all_indices[patients == test_patient]
        remaining_patients = np.unique(patients[patients != test_patient])
        if len(test) == 0:
            raise ValueError(f"Unknown test patient: {test_patient}")
        if len(remaining_patients) < 2:
            raise ValueError("LOPO requires at least three patients")
        rng.shuffle(remaining_patients)
        count = min(
            len(remaining_patients) - 1,
            max(1, round(len(remaining_patients) * val_fraction)),
        )
        val_patients = remaining_patients[:count]
        val = all_indices[np.isin(patients, val_patients)]
        train = all_indices[
            (patients != test_patient) & ~np.isin(patients, val_patients)
        ]
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    return {
        name: _balance(indices, labels, balance_ratio, rng)
        for name, indices in (("train", train), ("val", val), ("test", test))
    }


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _channel_key(name: str) -> str:
    key = name.upper().replace("EEG", "").replace(" ", "").replace("–", "-")
    return re.sub(r"-(REF|LE|AVG|0|1)$", "", key)


def _read_seizure_times(raw_dir: Path) -> dict[str, list[tuple[float, float]]]:
    seizures: dict[str, list[tuple[float, float]]] = {}
    summaries = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and "summary" in path.name.lower()
        and path.suffix.lower() == ".txt"
    )
    for summary in summaries:
        text = summary.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"(?=File Name\s*:)", text, flags=re.IGNORECASE)
        for block in blocks:
            name = re.search(r"File Name\s*:\s*(\S+\.edf)", block, re.IGNORECASE)
            starts = re.findall(
                r"Seizure \d+ Start Time\s*:\s*(\d+)\s*seconds", block, re.IGNORECASE
            )
            ends = re.findall(
                r"Seizure \d+ End Time\s*:\s*(\d+)\s*seconds", block, re.IGNORECASE
            )
            if name:
                seizures[name.group(1).lower()] = [
                    (float(a), float(b)) for a, b in zip(starts, ends)
                ]
    return seizures


def _filter_signal(
    signal_uv: np.ndarray, sample_rate: int, low: float, high: float, notch: float
) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, signal_uv, axis=-1)
    b_notch, a_notch = iirnotch(notch, 30.0, fs=sample_rate)
    return filtfilt(b_notch, a_notch, filtered, axis=-1).astype(np.float32)


def prepare_records(config: dict[str, Any], force: bool = False) -> Path:
    """Convert EDF files once into filtered, continuous float32 record caches."""
    import mne  # EDF support is only needed during preparation.

    data = config["data"]
    raw_dir = Path(data["raw_dir"])
    prepared = Path(data["prepared_dir"])
    records_dir = prepared / "records"
    manifest_path = prepared / "manifest.json"
    prepared.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(exist_ok=True)

    preprocessing = {
        "version": CACHE_VERSION,
        "channels": data["channels"],
        "sample_rate": data["sample_rate"],
        "bandpass_hz": data["bandpass_hz"],
        "notch_hz": data["notch_hz"],
    }
    fingerprint = _fingerprint(preprocessing)
    if manifest_path.exists() and not force:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("fingerprint") == fingerprint:
            LOGGER.info("Using existing record cache: %s", prepared)
            return manifest_path
        raise RuntimeError("Preprocessing settings changed; rerun with --force")

    edf_paths = find_edf_files(raw_dir)
    if not edf_paths:
        raise FileNotFoundError(f"No EDF files found under {raw_dir}")
    seizure_times = _read_seizure_times(raw_dir)
    records: list[dict[str, Any]] = []
    moments: dict[str, dict[str, np.ndarray | int]] = {}

    for edf_path in edf_paths:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        available = {_channel_key(name): name for name in raw.ch_names}
        missing = [
            name for name in data["channels"] if _channel_key(name) not in available
        ]
        if missing:
            LOGGER.warning("Skipping %s; missing channels: %s", edf_path.name, missing)
            continue
        raw.pick([available[_channel_key(name)] for name in data["channels"]])
        if round(raw.info["sfreq"]) != data["sample_rate"]:
            raw.resample(data["sample_rate"])
        signal_uv = raw.get_data() * 1e6
        signal_uv = _filter_signal(
            signal_uv,
            data["sample_rate"],
            data["bandpass_hz"][0],
            data["bandpass_hz"][1],
            data["notch_hz"],
        )
        patient_match = re.match(r"(chb\d+[a-z]?)", edf_path.stem, re.IGNORECASE)
        if not patient_match:
            LOGGER.warning("Skipping unrecognized filename: %s", edf_path.name)
            continue
        patient = patient_match.group(1).lower()
        cache_path = records_dir / f"{edf_path.stem.lower()}.npy"
        np.save(cache_path, signal_uv)
        records.append(
            {
                "name": edf_path.name,
                "patient": patient,
                "path": str(cache_path),
                "samples": signal_uv.shape[1],
                "seizures_seconds": seizure_times.get(edf_path.name.lower(), []),
            }
        )
        state = moments.setdefault(
            patient,
            {
                "count": 0,
                "sum": np.zeros(signal_uv.shape[0], dtype=np.float64),
                "sumsq": np.zeros(signal_uv.shape[0], dtype=np.float64),
            },
        )
        state["count"] = int(state["count"]) + signal_uv.shape[1]
        state["sum"] += signal_uv.sum(axis=1, dtype=np.float64)  # type: ignore[operator]
        state["sumsq"] += np.square(signal_uv, dtype=np.float64).sum(axis=1)  # type: ignore[operator]

    if not records:
        raise RuntimeError("No EDF record contained all configured channels")
    statistics: dict[str, np.ndarray] = {}
    for patient, state in moments.items():
        count = int(state["count"])
        mean = state["sum"] / count
        variance = np.maximum(state["sumsq"] / count - np.square(mean), 1e-12)
        statistics[f"{patient}_mean"] = mean.astype(np.float32)
        statistics[f"{patient}_std"] = np.sqrt(variance).astype(np.float32)
    np.savez(prepared / "normalization.npz", **statistics)
    manifest = {
        "fingerprint": fingerprint,
        "preprocessing": preprocessing,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Prepared %d records in %s", len(records), prepared)
    return manifest_path


def _valid_window(
    signal: np.ndarray, start: int, size: int, amplitude: float, min_std: float
) -> bool:
    window = signal[:, start : start + size]
    return bool(
        window.shape[1] == size
        and np.isfinite(window).all()
        and np.max(np.abs(window)) <= amplitude
        and np.all(window.std(axis=1) >= min_std)
    )


def prepare_windows(
    config: dict[str, Any], window_seconds: float, force: bool = False
) -> Path:
    """Build a small index catalog for one window length without copying EEG."""
    data = config["data"]
    prepared = Path(data["prepared_dir"])
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    label_settings = {
        "record_fingerprint": manifest["fingerprint"],
        "window_seconds": window_seconds,
        "seizure_overlap": data["seizure_overlap"],
        "min_seizure_overlap": data["min_seizure_overlap"],
        "safety_margin_seconds": data["safety_margin_seconds"],
        "artifact_uv": data["artifact_uv"],
        "constant_std_uv": data["constant_std_uv"],
    }
    windows_dir = prepared / "windows"
    windows_dir.mkdir(exist_ok=True)
    label = f"{window_seconds:g}s"
    catalog_path = windows_dir / f"{label}.npz"
    metadata_path = windows_dir / f"{label}.json"
    fingerprint = _fingerprint(label_settings)
    if catalog_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            LOGGER.info("Using existing %s window catalog", label)
            return catalog_path
        raise RuntimeError(f"Window settings changed for {label}; rerun with --force")

    sample_rate = int(data["sample_rate"])
    size = int(round(window_seconds * sample_rate))
    seizure_step = max(1, int(round(size * (1.0 - data["seizure_overlap"]))))
    minimum_overlap = int(np.ceil(size * data["min_seizure_overlap"]))
    safety = int(round(data["safety_margin_seconds"] * sample_rate))
    record_ids: list[int] = []
    starts: list[int] = []
    labels: list[int] = []

    for record_id, record in enumerate(manifest["records"]):
        signal = np.load(record["path"], mmap_mode="r")
        intervals = [
            (int(a * sample_rate), int(b * sample_rate))
            for a, b in record["seizures_seconds"]
        ]
        seizure_starts: set[int] = set()
        for seizure_start, seizure_end in intervals:
            lower = max(0, seizure_start - size)
            upper = min(signal.shape[1] - size, seizure_end)
            for start in range(lower, upper + 1, seizure_step):
                overlap = max(
                    0, min(start + size, seizure_end) - max(start, seizure_start)
                )
                if overlap >= minimum_overlap:
                    seizure_starts.add(start)
        nonseizure_starts = []
        for start in range(0, signal.shape[1] - size + 1, size):
            end = start + size
            safe = all(
                end <= seizure_start - safety or start >= seizure_end + safety
                for seizure_start, seizure_end in intervals
            )
            if safe:
                nonseizure_starts.append(start)
        for target, target_label in (
            (sorted(seizure_starts), 1),
            (nonseizure_starts, 0),
        ):
            for start in target:
                if _valid_window(
                    signal, start, size, data["artifact_uv"], data["constant_std_uv"]
                ):
                    record_ids.append(record_id)
                    starts.append(start)
                    labels.append(target_label)

    if not labels or len(set(labels)) != 2:
        raise RuntimeError(f"The {label} catalog does not contain both classes")
    np.savez_compressed(
        catalog_path,
        record=np.asarray(record_ids, dtype=np.int32),
        start=np.asarray(starts, dtype=np.int64),
        label=np.asarray(labels, dtype=np.uint8),
    )
    metadata_path.write_text(
        json.dumps({"fingerprint": fingerprint, **label_settings}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Indexed %d windows for %s", len(labels), label)
    return catalog_path


class EEGWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read indexed windows lazily from memory-mapped continuous records."""

    def __init__(self, config: dict[str, Any], indices: np.ndarray):
        data = config["data"]
        self.size = int(round(data["window_seconds"] * data["sample_rate"]))
        prepared = Path(data["prepared_dir"])
        self.records = json.loads(
            (prepared / "manifest.json").read_text(encoding="utf-8")
        )["records"]
        catalog = np.load(prepared / "windows" / f"{data['window_seconds']:g}s.npz")
        self.record_ids = catalog["record"][indices]
        self.starts = catalog["start"][indices]
        self.labels = catalog["label"][indices]
        stats = np.load(prepared / "normalization.npz")
        self.means = {
            record["patient"]: stats[f"{record['patient']}_mean"]
            for record in self.records
        }
        self.stds = {
            record["patient"]: stats[f"{record['patient']}_std"]
            for record in self.records
        }
        self._arrays: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_id = int(self.record_ids[index])
        if record_id not in self._arrays:
            self._arrays[record_id] = np.load(
                self.records[record_id]["path"], mmap_mode="r"
            )
        start = int(self.starts[index])
        patient = self.records[record_id]["patient"]
        window = self._arrays[record_id][:, start : start + self.size]
        window = (
            (window - self.means[patient][:, None]) / self.stds[patient][:, None]
        ).astype(np.float32)
        return torch.from_numpy(window), torch.tensor(
            int(self.labels[index]), dtype=torch.long
        )


class InMemoryEEGWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Materialize one split while preserving its logical sample order."""

    def __init__(self, source: EEGWindowDataset, split_name: str):
        channels = len(next(iter(source.means.values())))
        shape = (len(source), channels, source.size)
        gib = np.prod(shape, dtype=np.int64) * np.dtype(np.float32).itemsize / 1024**3
        LOGGER.info(
            "Materializing %s: %d windows, %.2f GiB", split_name, len(source), gib
        )
        started = time.perf_counter()
        try:
            inputs = np.empty(shape, dtype=np.float32)
        except MemoryError as error:
            raise MemoryError(
                f"Cannot materialize {split_name} ({gib:.2f} GiB); "
                "set train.cache_in_memory to false"
            ) from error
        order = np.lexsort((source.starts, source.record_ids))
        for position in order:
            inputs[int(position)] = source[int(position)][0].numpy()
        self.inputs = torch.from_numpy(inputs)
        self.labels = torch.from_numpy(source.labels.astype(np.int64, copy=True))
        LOGGER.info(
            "Materialized %s in %.1f seconds",
            split_name,
            time.perf_counter() - started,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.labels[index]


def create_dataloaders(
    config: dict[str, Any], split_id: int | str
) -> dict[str, DataLoader]:
    data = config["data"]
    prepared = Path(data["prepared_dir"])
    catalog_path = prepared / "windows" / f"{data['window_seconds']:g}s.npz"
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Missing {catalog_path}; run python -m src.data ... --prepare"
        )
    catalog = np.load(catalog_path)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    patient_by_record = np.asarray(
        [record["patient"] for record in manifest["records"]]
    )
    patients = patient_by_record[catalog["record"]]
    split = make_split(
        catalog["label"],
        patients,
        config["split"]["protocol"],
        split_id,
        config["seed"],
        config["split"]["folds"],
        config["split"]["val_fraction"],
        config["split"]["balance_ratio"],
    )
    split_dir = (
        prepared
        / "splits"
        / f"{data['window_seconds']:g}s"
        / f"{config['split']['protocol']}_seed{config['seed']}"
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(split_dir / f"{split_id}.npz", **split)
    cache_in_memory = bool(config["train"].get("cache_in_memory", False))
    workers = 0 if cache_in_memory else int(config["train"]["num_workers"])
    datasets: dict[str, Dataset] = {}
    for name, indices in split.items():
        source = EEGWindowDataset(config, indices)
        datasets[name] = (
            InMemoryEEGWindowDataset(source, name) if cache_in_memory else source
        )
    return {
        name: DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=name == "train",
            num_workers=workers,
            pin_memory=config["train"]["device"] == "cuda",
            persistent_workers=workers > 0,
        )
        for name, dataset in datasets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare read-only CHB-MIT EDF data")
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Build filtered records and window catalogs",
    )
    parser.add_argument("--windows", nargs="+", type=float, default=[1, 2, 4])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    if not args.prepare:
        parser.error("Specify --prepare")
    config = load_config(args.configs)
    prepare_records(config, args.force)
    for seconds in args.windows:
        prepare_windows(config, seconds, args.force)


if __name__ == "__main__":
    main()
