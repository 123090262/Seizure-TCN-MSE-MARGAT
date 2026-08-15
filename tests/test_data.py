from pathlib import Path

import numpy as np

from src.data import canonical_case_id, load_config, make_split


def test_later_yaml_overrides_only_selected_values(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"
    base.write_text(
        "data:\n  window_seconds: 2\n  sample_rate: 256\nseed: 42\n", encoding="utf-8"
    )
    overlay.write_text("data:\n  window_seconds: 4\n", encoding="utf-8")

    config = load_config([base, overlay])

    assert config == {
        "data": {"window_seconds": 4, "sample_rate": 256},
        "seed": 42,
    }


def test_mixed_split_is_reproducible_disjoint_and_balanced() -> None:
    labels = np.array([0] * 100 + [1] * 20, dtype=np.int64)
    patients = np.array([f"chb{i % 10:02d}" for i in range(120)])

    first = make_split(labels, patients, "mixed_10fold", 0, 42, 10, 0.1, 1.0)
    second = make_split(labels, patients, "mixed_10fold", 0, 42, 10, 0.1, 1.0)

    for name in ("train", "val", "test"):
        assert np.array_equal(first[name], second[name])
        subset_labels = labels[first[name]]
        assert np.count_nonzero(subset_labels == 0) == np.count_nonzero(
            subset_labels == 1
        )
    assert not set(first["train"]) & set(first["val"])
    assert not set(first["train"]) & set(first["test"])
    assert not set(first["val"]) & set(first["test"])


def test_mixed_fold_zero_indices_are_unchanged() -> None:
    labels = np.array([0] * 100 + [1] * 20, dtype=np.int64)
    patients = np.array([f"chb{i % 10:02d}" for i in range(120)])

    split = make_split(labels, patients, "mixed_10fold", 0, 42, 10, 0.1, 1.0)

    assert split["train"].tolist() == [
        101,
        115,
        107,
        104,
        113,
        21,
        45,
        110,
        81,
        29,
        12,
        102,
        34,
        27,
        116,
        109,
        66,
        74,
        67,
        111,
        119,
        14,
        24,
        44,
        108,
        23,
        117,
        47,
        118,
        106,
        105,
        94,
    ]
    assert split["val"].tolist() == [112, 100, 68, 25]
    assert split["test"].tolist() == [42, 103, 54, 114]


def test_case_id_merge_is_enabled_only_for_lopo_artifacts() -> None:
    assert canonical_case_id("chb01_03", merge_chb17=True) == "chb01"
    assert canonical_case_id("chb17a_03", merge_chb17=True) == "chb17"
    assert canonical_case_id("chb17b_57", merge_chb17=True) == "chb17"
    assert canonical_case_id("chb17c_02", merge_chb17=True) == "chb17"
    assert canonical_case_id("chb17a_03", merge_chb17=False) == "chb17a"
    assert canonical_case_id("unrecognized", merge_chb17=True) is None


def test_lopo_keeps_test_patient_out_of_train_and_validation() -> None:
    labels = np.tile(np.array([0, 0, 1]), 20)
    patients = np.repeat(["chb01", "chb02", "chb03", "chb04"], 15)

    split = make_split(labels, patients, "lopo", "chb04", 7, 10, 0.25, 1.0)

    assert set(patients[split["test"]]) == {"chb04"}
    assert "chb04" not in set(patients[split["train"]])
    assert "chb04" not in set(patients[split["val"]])
    assert not set(patients[split["train"]]) & set(patients[split["val"]])
