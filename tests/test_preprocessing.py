from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data.chbmit_reader import get_channel_set, normalize_channel_name
from src.data.normalization import (
    SubjectGlobalZScoreNormalizer,
    ZScoreNormalizer,
    build_normalizer,
    load_normalizer,
)
from src.data.preprocessing import _first_matching_channels, _first_matching_electrodes


def test_normalization_fit_train_only(tmp_path) -> None:
    train = np.ones((2, 8), dtype=np.float32)
    test = np.ones((2, 8), dtype=np.float32) * 100
    train_path = tmp_path / "train.npy"
    test_path = tmp_path / "test.npy"
    np.save(train_path, train)
    np.save(test_path, test)
    normalizer = ZScoreNormalizer.fit(
        [{"data_path": str(train_path), "start_sample": 0, "end_sample": 8}],
        eps=1e-6,
        channel_names=["A", "B"],
    )
    assert np.allclose(normalizer.mean, [1, 1])
    assert not np.allclose(normalizer.mean, [100, 100])
    transformed = normalizer.transform(test)
    assert transformed.mean() > 1


def test_subject_global_normalization_uses_each_subjects_full_records(tmp_path) -> None:
    subject_a = tmp_path / "a.npy"
    subject_b = tmp_path / "b.npy"
    np.save(subject_a, np.asarray([[1, 3], [10, 14]], dtype=np.float32))
    np.save(subject_b, np.asarray([[100, 104], [20, 22]], dtype=np.float32))

    normalizer = SubjectGlobalZScoreNormalizer.fit_records(
        [
            {"subject_id": "a", "processed_path": str(subject_a)},
            {"subject_id": "b", "processed_path": str(subject_b)},
        ],
        channel_names=["C1", "C2"],
    )

    assert normalizer.means["a"].tolist() == [2.0, 12.0]
    assert normalizer.means["b"].tolist() == [102.0, 21.0]
    assert np.allclose(normalizer.transform(np.load(subject_a), subject_id="a").mean(axis=1), 0.0)
    assert np.allclose(normalizer.transform(np.load(subject_b), subject_id="b").mean(axis=1), 0.0)
    assert not np.allclose(
        normalizer.transform(np.load(subject_b), subject_id="b"),
        normalizer.transform(np.load(subject_b), subject_id="a"),
    )


def test_subject_global_normalization_cache_round_trip(tmp_path) -> None:
    record = tmp_path / "record.npy"
    metadata_path = tmp_path / "metadata.csv"
    stats_path = tmp_path / "stats.json"
    np.save(record, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    pd.DataFrame(
        [{"subject_id": "chb01", "processed_path": str(record)}]
    ).to_csv(metadata_path, index=False)
    cfg = SimpleNamespace(
        data=SimpleNamespace(metadata_path=str(metadata_path)),
        normalization=SimpleNamespace(
            scope="subject_global",
            eps=1e-6,
            stats_path=str(stats_path),
        ),
    )

    created = build_normalizer({}, cfg, channel_names=["C1", "C2"])
    loaded = build_normalizer({}, cfg, channel_names=["C1", "C2"])
    direct = load_normalizer(stats_path)

    assert isinstance(created, SubjectGlobalZScoreNormalizer)
    assert isinstance(loaded, SubjectGlobalZScoreNormalizer)
    assert isinstance(direct, SubjectGlobalZScoreNormalizer)
    assert np.allclose(created.means["chb01"], loaded.means["chb01"])
    assert np.allclose(created.stds["chb01"], direct.stds["chb01"])


def test_normalizer_loader_remains_compatible_with_legacy_train_only_stats(tmp_path) -> None:
    stats_path = tmp_path / "legacy_stats.json"
    stats_path.write_text(
        json.dumps({"mean": [1.0, 2.0], "std": [2.0, 4.0], "eps": 1e-6}),
        encoding="utf-8",
    )

    normalizer = load_normalizer(stats_path)

    assert isinstance(normalizer, ZScoreNormalizer)
    assert normalizer.mean.tolist() == [1.0, 2.0]
    assert normalizer.std.tolist() == [2.0, 4.0]


def test_chb12_referential_labels_can_form_common_18() -> None:
    channels = get_channel_set("common_18")
    chb12_27_labels = [
        "F7-CS2",
        "T7-CS2",
        "P7-CS2",
        "-",
        "FP1-CS2",
        "F3-CS2",
        "C3-CS2",
        "P3-CS2",
        "O1-CS2",
        "-",
        "FZ-CS2",
        "CZ-CS2",
        "PZ-CS2",
        "-",
        "FP2-CS2",
        "F4-CS2",
        "C4-CS2",
        "P4-CS2",
        "O2-CS2",
        "-",
        "F8-CS2",
        "T8-CS2",
        "P8-CS2",
    ]
    electrodes = _first_matching_electrodes(chb12_27_labels)

    assert not _first_matching_channels(chb12_27_labels, channels)
    assert all(
        channel.split("-", 1)[0] in electrodes and channel.split("-", 1)[1] in electrodes
        for channel in channels
    )


def test_normalize_channel_name_handles_chb12_o_digit_alias() -> None:
    assert normalize_channel_name("01") == "O1"
    assert normalize_channel_name("P7-01") == "P7-O1"
