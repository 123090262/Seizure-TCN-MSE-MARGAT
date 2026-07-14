from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_config


@pytest.mark.parametrize(
    "config_name,base_config,seconds,samples,output_tag",
    [
        ("subject_global", "configs/exp_mixed_5fold.yaml", 4.0, 1024, "subject_global_win4s_baseline"),
        (
            "subject_global_win1s_ov50",
            "configs/window_sweeps/win1s_ov50.yaml",
            1.0,
            256,
            "subject_global_win1s_ov50",
        ),
        (
            "subject_global_win2s_ov50",
            "configs/window_sweeps/win2s_ov50.yaml",
            2.0,
            512,
            "subject_global_win2s_ov50",
        ),
    ],
)
def test_subject_global_branch_only_changes_normalization_and_partition_paths(
    config_name: str,
    base_config: str,
    seconds: float,
    samples: int,
    output_tag: str,
) -> None:
    base = load_config(base_config)
    branch = load_config(f"configs/normalization/{config_name}.yaml")

    assert branch.experiment.branch_name == "全局归一化"
    assert branch.experiment.output_tag == output_tag
    assert branch.normalization.method == base.normalization.method == "zscore"
    assert branch.normalization.scope == "subject_global"
    assert base.normalization.scope == "train_only"
    assert branch.normalization.per_channel == base.normalization.per_channel
    assert branch.normalization.eps == base.normalization.eps
    assert Path(str(branch.normalization.stats_path)) == Path(
        "data/normalization/subject_global/chbmit_common18_stats.json"
    )
    assert Path(str(branch.normalization.metadata_path)) == Path("data/processed/chbmit/metadata.csv")
    assert float(branch.windowing.window_size_sec) == seconds
    assert int(branch.windowing.window_size_samples) == samples
    assert str(branch.project.output_dir).startswith("outputs/normalization/subject_global/")
    assert branch.data == base.data
    assert branch.preprocessing == base.preprocessing
    assert branch.windowing == base.windowing
    assert branch.training == base.training
    assert branch.evaluation == base.evaluation
    assert branch.experiment.strategy == base.experiment.strategy
    assert branch.experiment.n_splits == base.experiment.n_splits
    assert branch.experiment.val_fraction == base.experiment.val_fraction
