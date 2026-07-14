from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from src.utils.config import load_config
from src.utils.paths import window_index_path


VARIANTS = [
    ("win1s_ov50", 1.0, 256),
    ("win2s_ov50", 2.0, 512),
]


@pytest.mark.parametrize("tag,seconds,samples", VARIANTS)
def test_window_sweep_variant_is_fully_partitioned(tag: str, seconds: float, samples: int) -> None:
    cfg = load_config(f"configs/window_sweeps/{tag}.yaml")

    assert str(cfg.experiment.output_tag) == tag
    assert float(cfg.windowing.window_size_sec) == seconds
    assert int(cfg.windowing.window_size_samples) == samples
    assert float(cfg.windowing.ictal_overlap) == 0.5
    assert float(cfg.windowing.interictal_overlap) == 0.5
    assert window_index_path(cfg) == Path(f"data/window_sweeps/{tag}/window_index.csv")
    assert Path(str(cfg.data.metadata_path)) == Path(f"data/window_sweeps/{tag}/metadata.csv")
    assert Path(str(cfg.data.splits_dir)) == Path(f"data/window_sweeps/{tag}/splits")
    assert Path(str(cfg.project.output_dir)) == Path(f"outputs/window_sweeps/{tag}")


def test_window_sweep_variants_share_only_the_preprocessed_signal_source() -> None:
    base = load_config("configs/exp_mixed_5fold.yaml")
    configs = [load_config(f"configs/window_sweeps/{tag}.yaml") for tag, _, _ in VARIANTS]

    assert len({str(window_index_path(cfg)) for cfg in configs}) == len(configs)
    assert len({str(cfg.data.splits_dir) for cfg in configs}) == len(configs)
    assert len({str(cfg.project.output_dir) for cfg in configs}) == len(configs)
    for cfg in configs:
        assert str(cfg.data.processed_dir) == str(base.data.processed_dir)
        assert str(cfg.data.source_metadata_path) == str(base.data.metadata_path)
        assert cfg.preprocessing == base.preprocessing
        assert cfg.normalization == base.normalization
        assert cfg.training == base.training
        assert cfg.evaluation == base.evaluation
        assert cfg.windowing.task == base.windowing.task
        assert cfg.windowing.interictal_exclusion_sec == base.windowing.interictal_exclusion_sec
        assert cfg.windowing.balance_after_split == base.windowing.balance_after_split
        assert cfg.windowing.balance_ratio == base.windowing.balance_ratio
        assert cfg.windowing.balance_sets == base.windowing.balance_sets
        assert cfg.experiment.strategy == base.experiment.strategy
        assert cfg.experiment.n_splits == base.experiment.n_splits
        assert cfg.experiment.val_fraction == base.experiment.val_fraction


def test_variant_preparation_writes_manifest_and_reuses_an_exact_match(tmp_path: Path) -> None:
    source_metadata = tmp_path / "source_metadata.csv"
    variant_dir = tmp_path / "variant"
    pd.DataFrame(
        [
            {
                "subject_id": "test01",
                "file_name": "test01.edf",
                "processed_path": "shared/test01.npy",
                "sampling_rate": 256,
                "duration_sec": 20,
                "seizure_intervals": "[]",
            }
        ]
    ).to_csv(source_metadata, index=False)
    config_path = tmp_path / "variant.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "sampling_rate": 256,
                    "channel_set": "common_18",
                    "source_metadata_path": str(source_metadata),
                    "metadata_path": str(variant_dir / "metadata.csv"),
                    "window_index_path": str(variant_dir / "window_index.csv"),
                    "splits_dir": str(variant_dir / "splits"),
                },
                "windowing": {
                    "window_size_samples": 256,
                    "window_size_sec": 1.0,
                    "ictal_overlap": 0.5,
                    "interictal_overlap": 0.5,
                    "interictal_exclusion_sec": 0,
                },
                "experiment": {"output_tag": "test_win1s_ov50"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    command = [sys.executable, "scripts/prepare_window_variant.py", "--config", str(config_path)]

    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)

    index = pd.read_csv(variant_dir / "window_index.csv")
    manifest = json.loads((variant_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(index) == 39
    assert set(index["label"]) == {0}
    assert manifest["window_count"] == 39
    assert manifest["window_size_sec"] == 1.0
    assert manifest["ictal_overlap"] == 0.5
    assert manifest["interictal_overlap"] == 0.5
    assert "Saved window index" in first.stdout
    assert "Reusing matching window variant" in second.stdout
