from __future__ import annotations

import numpy as np

from src.data.chbmit_reader import get_channel_set, normalize_channel_name
from src.data.normalization import ZScoreNormalizer
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
