from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data import (
    EXPECTED_LOPO_CASE_IDS,
    _read_seizure_times,
    _record_preprocessing_settings,
    _split_options,
    _split_variant_name,
    canonical_case_id,
    load_config,
    main as data_main,
    make_split,
    validate_lopo_artifacts,
)


def _write_lopo_artifacts(
    tmp_path: Path,
    case_ids: tuple[str, ...] = tuple(f"chb{i:02d}" for i in range(1, 25)),
    missing_positive_case: str | None = None,
) -> dict:
    prepared = tmp_path / "prepared_lopo_24case"
    windows = prepared / "windows"
    windows.mkdir(parents=True)
    records = [{"patient": case_id} for case_id in case_ids]
    (prepared / "manifest.json").write_text(
        json.dumps({"records": records}), encoding="utf-8"
    )
    statistics = {
        f"{case_id}_{suffix}": np.ones(18, dtype=np.float32)
        for case_id in case_ids
        for suffix in ("mean", "std")
    }
    np.savez(prepared / "normalization.npz", **statistics)
    record_ids = []
    labels = []
    for record_id, case_id in enumerate(case_ids):
        record_ids.append(record_id)
        labels.append(0)
        if case_id != missing_positive_case:
            record_ids.append(record_id)
            labels.append(1)
    np.savez(
        windows / "2s.npz",
        record=np.asarray(record_ids, dtype=np.int32),
        start=np.zeros(len(labels), dtype=np.int64),
        label=np.asarray(labels, dtype=np.uint8),
    )
    return {
        "data": {
            "prepared_dir": str(prepared),
            "window_seconds": 2,
            "merge_chb17": True,
        },
        "split": {"protocol": "lopo"},
    }


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


def test_annotation_parser_change_invalidates_prepared_caches() -> None:
    data = {
        "channels": ["FP1-F7"],
        "sample_rate": 256,
        "bandpass_hz": [0.5, 70.0],
        "notch_hz": 60.0,
    }

    assert _record_preprocessing_settings(data) == {
        "version": 2,
        "channels": ["FP1-F7"],
        "sample_rate": 256,
        "bandpass_hz": [0.5, 70.0],
        "notch_hz": 60.0,
    }
    assert _record_preprocessing_settings({**data, "merge_chb17": True}) == {
        "version": 2,
        "channels": ["FP1-F7"],
        "sample_rate": 256,
        "bandpass_hz": [0.5, 70.0],
        "notch_hz": 60.0,
        "case_id_mapping": "merge-chb17-v1",
    }


def test_seizure_times_accept_numbered_and_unnumbered_entries(tmp_path: Path) -> None:
    (tmp_path / "chb-summary.txt").write_text(
        "File Name: chb01_03.edf\n"
        "Number of Seizures in File: 1\n"
        "Seizure Start Time: 2996 seconds\n"
        "Seizure End Time: 3036 seconds\n"
        "\n"
        "File Name: chb06_04.edf\n"
        "Number of Seizures in File: 2\n"
        "Seizure 1 Start Time: 100 seconds\n"
        "Seizure 1 End Time: 120 seconds\n"
        "Seizure 2 Start Time: 200 seconds\n"
        "Seizure 2 End Time: 240 seconds\n",
        encoding="utf-8",
    )

    assert _read_seizure_times(tmp_path) == {
        "chb01_03.edf": [(2996.0, 3036.0)],
        "chb06_04.edf": [(100.0, 120.0), (200.0, 240.0)],
    }


def test_seizure_times_reject_incomplete_summary_entries(tmp_path: Path) -> None:
    (tmp_path / "chb-summary.txt").write_text(
        "File Name: chb24_01.edf\n"
        "Number of Seizures in File: 2\n"
        "Seizure Start Time: 480 seconds\n"
        "Seizure End Time: 505 seconds\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        _read_seizure_times(tmp_path)

    assert "chb24_01.edf" in str(error.value)
    assert "declares 2 seizures but parsed 1" in str(error.value)


def test_lopo_artifacts_require_exactly_24_cases(tmp_path: Path) -> None:
    config = _write_lopo_artifacts(tmp_path)

    summary = validate_lopo_artifacts(config, 2)

    assert summary == {
        "case_count": 24,
        "case_ids": list(EXPECTED_LOPO_CASE_IDS),
        "normalization_arrays": 48,
        "window_seconds": 2.0,
        "fold_ids": list(EXPECTED_LOPO_CASE_IDS),
    }


def test_lopo_artifact_validation_requires_chb17_merge(tmp_path: Path) -> None:
    config = _write_lopo_artifacts(tmp_path)
    config["data"]["merge_chb17"] = False

    with pytest.raises(ValueError) as error:
        validate_lopo_artifacts(config, 2)

    assert "data.merge_chb17=true" in str(error.value)


@pytest.mark.parametrize(
    ("case_ids", "message"),
    [
        (tuple(f"chb{i:02d}" for i in range(1, 24)), "missing=['chb24']"),
        (
            tuple(f"chb{i:02d}" for i in range(1, 25)) + ("chb17a",),
            "unexpected=['chb17a']",
        ),
    ],
)
def test_lopo_artifacts_report_case_set_errors(
    tmp_path: Path, case_ids: tuple[str, ...], message: str
) -> None:
    config = _write_lopo_artifacts(tmp_path, case_ids)

    with pytest.raises(ValueError) as error:
        validate_lopo_artifacts(config, 2)

    assert message in str(error.value)


def test_lopo_artifacts_require_two_normalization_arrays_per_case(
    tmp_path: Path,
) -> None:
    config = _write_lopo_artifacts(tmp_path)
    normalization_path = Path(config["data"]["prepared_dir"]) / "normalization.npz"
    with np.load(normalization_path) as archive:
        statistics = {key: archive[key] for key in archive.files if key != "chb17_std"}
    np.savez(normalization_path, **statistics)

    with pytest.raises(ValueError) as error:
        validate_lopo_artifacts(config, 2)

    assert "missing=['chb17_std']" in str(error.value)


def test_lopo_artifacts_require_both_classes_for_every_case(tmp_path: Path) -> None:
    config = _write_lopo_artifacts(tmp_path, missing_positive_case="chb08")

    with pytest.raises(ValueError) as error:
        validate_lopo_artifacts(config, 2)

    assert "chb08" in str(error.value)
    assert "negative=1" in str(error.value)
    assert "positive=0" in str(error.value)


def test_check_lopo_cli_prints_validated_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    config = _write_lopo_artifacts(tmp_path)
    config_path = tmp_path / "lopo.yaml"
    config_path.write_text(
        "data:\n"
        f"  prepared_dir: {config['data']['prepared_dir']}\n"
        "  merge_chb17: true\n"
        "split:\n"
        "  protocol: lopo\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["src.data", str(config_path), "--check-lopo", "--windows", "2"],
    )

    data_main()

    summary = json.loads(capsys.readouterr().out)
    assert summary["case_count"] == 24
    assert summary["normalization_arrays"] == 48
    assert summary["fold_ids"] == list(EXPECTED_LOPO_CASE_IDS)


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


def test_all_24_lopo_folds_are_case_disjoint_and_balanced() -> None:
    labels = np.tile(np.array([0, 0, 1], dtype=np.int64), 24 * 5)
    patients = np.repeat(EXPECTED_LOPO_CASE_IDS, 15)
    tested = []

    for fold_id in EXPECTED_LOPO_CASE_IDS:
        split = make_split(labels, patients, "lopo", fold_id, 42, 10, 0.15, 1.0)
        assert set(patients[split["test"]]) == {fold_id}
        assert fold_id not in set(patients[split["train"]])
        assert fold_id not in set(patients[split["val"]])
        assert not set(patients[split["train"]]) & set(patients[split["val"]])
        for indices in split.values():
            assert set(labels[indices]) == {0, 1}
        tested.append(fold_id)

    assert tested == list(EXPECTED_LOPO_CASE_IDS)


def test_lopo_mixed_window_validation_is_stratified_reproducible_and_test_isolated() -> None:
    patients = np.repeat(["chb01", "chb02", "chb03", "chb04"], 20)
    labels = np.tile(np.array([0, 1], dtype=np.int64), 40)

    first = make_split(
        labels,
        patients,
        "lopo",
        "chb04",
        42,
        10,
        0.10,
        1.0,
        validation_strategy="mixed_windows",
    )
    second = make_split(
        labels,
        patients,
        "lopo",
        "chb04",
        42,
        10,
        0.10,
        1.0,
        validation_strategy="mixed_windows",
    )

    assert np.array_equal(first["val"], second["val"])
    assert len(first["val"]) == 6
    assert np.count_nonzero(labels[first["val"]] == 0) == 3
    assert np.count_nonzero(labels[first["val"]] == 1) == 3
    assert set(patients[first["test"]]) == {"chb04"}
    assert "chb04" not in set(patients[first["train"]])
    assert "chb04" not in set(patients[first["val"]])
    assert set(patients[first["train"]]) & set(patients[first["val"]])
    assert not set(first["train"]) & set(first["val"])


def test_lopo_supports_independent_negative_ratios() -> None:
    patients = np.repeat(["chb01", "chb02", "chb03", "chb04"], 50)
    labels = np.tile(np.array([0, 0, 0, 0, 1], dtype=np.int64), 40)

    split = make_split(
        labels,
        patients,
        "lopo",
        "chb04",
        42,
        10,
        0.10,
        1.0,
        validation_strategy="mixed_windows",
        negative_ratios={"train": 2.0, "val": 1.0, "test": 1.0},
    )

    for name, expected_ratio in {"train": 2, "val": 1, "test": 1}.items():
        split_labels = labels[split[name]]
        positives = np.count_nonzero(split_labels == 1)
        negatives = np.count_nonzero(split_labels == 0)
        assert negatives == positives * expected_ratio


@pytest.mark.parametrize(
    ("validation_strategy", "negative_ratios", "message"),
    [
        ("unknown", None, "Unsupported LOPO validation strategy"),
        (
            "mixed_windows",
            {"train": 0.0, "val": 1.0, "test": 1.0},
            "train negative ratio",
        ),
    ],
)
def test_lopo_rejects_invalid_split_options(
    validation_strategy: str,
    negative_ratios: dict[str, float] | None,
    message: str,
) -> None:
    labels = np.tile(np.array([0, 0, 1], dtype=np.int64), 20)
    patients = np.repeat(["chb01", "chb02", "chb03", "chb04"], 15)

    with pytest.raises(ValueError, match=message):
        make_split(
            labels,
            patients,
            "lopo",
            "chb04",
            42,
            10,
            0.10,
            1.0,
            validation_strategy=validation_strategy,
            negative_ratios=negative_ratios,
        )


def test_split_options_preserve_legacy_defaults() -> None:
    strategy, ratios = _split_options({"balance_ratio": 1.0})

    assert strategy == "case_holdout"
    assert ratios == {"train": 1.0, "val": 1.0, "test": 1.0}


def test_mixed_validation_uses_distinct_split_variant_name() -> None:
    baseline = _split_variant_name(
        "lopo",
        42,
        {
            "balance_ratio": 1.0,
            "val_fraction": 0.15,
        },
    )
    mixed_validation = _split_variant_name(
        "lopo",
        42,
        {
            "balance_ratio": 1.0,
            "validation_strategy": "mixed_windows",
            "val_fraction": 0.10,
            "train_negative_ratio": 2.0,
            "val_negative_ratio": 1.0,
            "test_negative_ratio": 1.0,
        },
    )

    assert baseline == "lopo_seed42"
    assert mixed_validation.startswith("lopo_mixed_windows_")
    assert mixed_validation.endswith("_seed42")
    assert mixed_validation != baseline
