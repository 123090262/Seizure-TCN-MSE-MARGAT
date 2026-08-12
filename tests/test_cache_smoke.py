import json
from pathlib import Path

import numpy as np
import torch

from src.data import (
    EEGWindowDataset,
    InMemoryEEGWindowDataset,
    create_dataloaders,
    prepare_windows,
)


def _indexed_cache_config(tmp_path: Path) -> dict:
    prepared = tmp_path / "indexed"
    records = prepared / "records"
    windows = prepared / "windows"
    records.mkdir(parents=True)
    windows.mkdir()
    signal = np.arange(18 * 800, dtype=np.float32).reshape(18, 800)
    record_path = records / "chb01_01.npy"
    np.save(record_path, signal)
    manifest = {
        "records": [
            {
                "name": "chb01_01.edf",
                "patient": "chb01",
                "path": str(record_path),
                "samples": 800,
                "seizures_seconds": [],
            }
        ]
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(
        prepared / "normalization.npz",
        chb01_mean=np.zeros(18, dtype=np.float32),
        chb01_std=np.ones(18, dtype=np.float32),
    )
    np.savez(
        windows / "2s.npz",
        record=np.zeros(40, dtype=np.int32),
        start=np.arange(40, dtype=np.int64) * 20,
        label=np.tile(np.array([0, 1], dtype=np.uint8), 20),
    )
    return {
        "seed": 42,
        "data": {
            "prepared_dir": str(prepared),
            "sample_rate": 10,
            "window_seconds": 2,
        },
        "split": {
            "protocol": "mixed_10fold",
            "folds": 2,
            "val_fraction": 0.25,
            "balance_ratio": 1.0,
        },
        "train": {"batch_size": 4, "num_workers": 2, "device": "cpu"},
    }


def test_in_memory_dataset_matches_mmap_values_and_split_order(
    tmp_path: Path,
) -> None:
    config = _indexed_cache_config(tmp_path)
    indices = np.array([3, 0, 2, 1], dtype=np.int64)
    lazy = EEGWindowDataset(config, indices)
    cached = InMemoryEEGWindowDataset(lazy, "train")

    assert len(cached) == len(lazy)
    for index in range(len(lazy)):
        expected_input, expected_label = lazy[index]
        actual_input, actual_label = cached[index]
        torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
        assert actual_input.dtype == torch.float32
        assert actual_label.item() == expected_label.item()


def test_window_cache_and_loader_use_shared_continuous_record(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    records = prepared / "records"
    records.mkdir(parents=True)
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 20, size=(18, 1000)).astype(np.float32)
    record_path = records / "chb01_01.npy"
    np.save(record_path, signal)
    manifest = {
        "fingerprint": "records-v1",
        "records": [
            {
                "name": "chb01_01.edf",
                "patient": "chb01",
                "path": str(record_path),
                "samples": 1000,
                "seizures_seconds": [[20, 30]],
            }
        ],
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(
        prepared / "normalization.npz",
        chb01_mean=signal.mean(1),
        chb01_std=signal.std(1),
    )
    config = {
        "seed": 42,
        "data": {
            "prepared_dir": str(prepared),
            "sample_rate": 10,
            "window_seconds": 2,
            "seizure_overlap": 0.75,
            "min_seizure_overlap": 0.5,
            "safety_margin_seconds": 2,
            "artifact_uv": 1000,
            "constant_std_uv": 0.1,
        },
        "split": {
            "protocol": "mixed_10fold",
            "folds": 2,
            "val_fraction": 0.25,
            "balance_ratio": 1.0,
        },
        "train": {"batch_size": 4, "num_workers": 0, "device": "cpu"},
    }

    catalog = prepare_windows(config, 2)
    loaders = create_dataloaders(config, 0)
    inputs, labels = next(iter(loaders["train"]))

    assert catalog.name == "2s.npz"
    assert inputs.shape == (4, 18, 20)
    assert set(labels.tolist()) <= {0, 1}
    assert record_path.exists()
