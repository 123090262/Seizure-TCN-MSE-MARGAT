from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.splitters import (
    build_intra_patient_10fold_splits,
    build_lopo_splits,
    build_mixed_5fold_splits,
    save_intra_patient_10fold_splits,
    save_lopo_splits,
    save_mixed_5fold_splits,
)
from src.utils.config import load_config


def _cfg(
    splits_dir: str | Path = "data/splits",
    *,
    balance: bool = False,
    n_splits: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        experiment=SimpleNamespace(n_splits=n_splits, val_fraction=0.2, val_subjects=1),
        training=SimpleNamespace(seed=42),
        data=SimpleNamespace(splits_dir=str(splits_dir)),
        windowing=SimpleNamespace(
            balance_after_split=balance,
            balance_ratio=1.0,
            balance_sets=["train", "val", "test"],
        ),
    )


def _assert_balanced(split: dict[str, object]) -> None:
    for name in ("train", "val", "test"):
        counts = Counter(int(window["label"]) for window in split[f"{name}_windows"])
        assert counts[0] == counts[1]


def _balanced_intra_rows() -> list[dict[str, object]]:
    rows = []
    for block in range(6):
        for label, count in ((1, 2), (0, 6)):
            for index in range(count):
                window_start = block * 1000 + label * 100 + index * 0.25
                rows.append(
                    {
                        "window_id": f"b{block}-y{label}-{index}",
                        "subject_id": "chb01",
                        "record_id": f"r{block}",
                        "block_id": "shared_block",
                        "label": label,
                        "window_start": window_start,
                        "window_end": window_start + 1,
                    }
                )
    return rows


def _balanced_lopo_rows(subjects: list[str]) -> list[dict[str, object]]:
    rows = []
    for subject in subjects:
        for label, count in ((1, 2), (0, 6)):
            for index in range(count):
                rows.append(
                    {
                        "window_id": f"{subject}-y{label}-{index}",
                        "subject_id": subject,
                        "record_id": f"{subject}_r",
                        "block_id": f"{subject}_b",
                        "label": label,
                    }
                )
    return rows


def _mixed_rows() -> list[dict[str, object]]:
    rows = []
    for subject_index in range(5):
        subject = f"chb{subject_index + 1:02d}"
        for label, count in ((1, 10), (0, 30)):
            for index in range(count):
                start_sample = label * 100_000 + index * (768 if label else 1024)
                rows.append(
                    {
                        "window_id": f"{subject}-y{label}-{index}",
                        "subject_id": subject,
                        "record_id": f"{subject}_r",
                        "block_id": f"{subject}_b",
                        "data_path": f"data/{subject}.npy",
                        "start_sample": start_sample,
                        "end_sample": start_sample + 1024,
                        "window_start": start_sample / 256,
                        "window_end": (start_sample + 1024) / 256,
                        "label": label,
                    }
                )
    return rows


def _sample_ids(windows: list[dict[str, object]]) -> set[tuple[object, ...]]:
    return {
        (window["subject_id"], window["record_id"], window["window_start"], window["window_end"])
        for window in windows
    }


def _has_partial_overlap(left: list[dict[str, object]], right: list[dict[str, object]]) -> bool:
    return any(
        first["record_id"] == second["record_id"]
        and (first["window_start"], first["window_end"]) != (second["window_start"], second["window_end"])
        and max(first["window_start"], second["window_start"])
        < min(first["window_end"], second["window_end"])
        for first in left
        for second in right
    )


def test_intra_patient_splits_are_stratified_by_window_not_block() -> None:
    rows = []
    for label in (0, 1):
        for index in range(12):
            window_start = label * 100 + index * 0.25
            rows.append(
                {
                    "subject_id": "chb01",
                    "record_id": "r0",
                    "block_id": "shared_block",
                    "label": label,
                    "window_start": window_start,
                    "window_end": window_start + 1,
                }
            )
    splits = build_intra_patient_10fold_splits("chb01", pd.DataFrame(rows), _cfg())
    for split in splits:
        sets = {name: split[f"{name}_windows"] for name in ("train", "val", "test")}
        for windows in sets.values():
            assert {int(window["label"]) for window in windows} == {0, 1}
            assert {window["block_id"] for window in windows} == {"shared_block"}
        ids = {name: _sample_ids(items) for name, items in sets.items()}
        assert ids["train"].isdisjoint(ids["val"])
        assert ids["train"].isdisjoint(ids["test"])
        assert ids["val"].isdisjoint(ids["test"])
        assert _has_partial_overlap(sets["train"], sets["test"])


def test_lopo_test_subject_not_in_train_or_val() -> None:
    rows = []
    subjects = ["chb01", "chb02", "chb03", "chb04"]
    for subject in subjects:
        rows.append({"subject_id": subject, "record_id": f"{subject}_r", "block_id": f"{subject}_b", "label": 0})
    splits = build_lopo_splits(subjects, pd.DataFrame(rows), _cfg())
    for split in splits:
        test_subject = split["test_subject"]
        assert test_subject not in split["train_subjects"]
        assert test_subject not in split["val_subjects"]


def test_intra_patient_json_balances_each_set_after_split_and_keeps_index(tmp_path: Path) -> None:
    index_path = tmp_path / "window_index.csv"
    pd.DataFrame(_balanced_intra_rows()).to_csv(index_path, index=False)
    original_index = index_path.read_bytes()
    windows = pd.read_csv(index_path)

    paths = save_intra_patient_10fold_splits("chb01", windows, _cfg(tmp_path / "splits", balance=True))

    assert index_path.read_bytes() == original_index
    assert len(paths) == 3
    for path in paths:
        split = json.loads(path.read_text(encoding="utf-8"))
        _assert_balanced(split)
        ids = {
            name: _sample_ids(split[f"{name}_windows"])
            for name in ("train", "val", "test")
        }
        assert ids["train"].isdisjoint(ids["val"])
        assert ids["train"].isdisjoint(ids["test"])
        assert ids["val"].isdisjoint(ids["test"])
        assert {window["block_id"] for window in split["train_windows"]} == {"shared_block"}
        assert {window["block_id"] for window in split["test_windows"]} == {"shared_block"}


def test_lopo_json_balances_each_set_after_subject_split(tmp_path: Path) -> None:
    subjects = ["chb01", "chb02", "chb03", "chb04"]
    windows = pd.DataFrame(_balanced_lopo_rows(subjects))

    paths = save_lopo_splits(subjects, windows, _cfg(tmp_path / "splits", balance=True))

    assert len(paths) == len(subjects)
    for path in paths:
        split = json.loads(path.read_text(encoding="utf-8"))
        _assert_balanced(split)
        test_subject = split["test_subject"]
        assert {window["subject_id"] for window in split["test_windows"]} == {test_subject}
        assert test_subject not in {window["subject_id"] for window in split["train_windows"]}
        assert test_subject not in {window["subject_id"] for window in split["val_windows"]}


def test_balancing_is_reproducible_for_each_fold() -> None:
    windows = pd.DataFrame(_balanced_intra_rows())
    cfg = _cfg(balance=True)

    first = build_intra_patient_10fold_splits("chb01", windows, cfg)
    second = build_intra_patient_10fold_splits("chb01", windows, cfg)

    assert first == second


def test_intra_patient_keeps_exactly_ten_stratified_folds() -> None:
    rows = []
    for label, count in ((1, 30), (0, 90)):
        for index in range(count):
            window_start = label * 1000 + index * 0.25
            rows.append(
                {
                    "subject_id": "chb01",
                    "record_id": "r0",
                    "block_id": "shared_block",
                    "label": label,
                    "window_start": window_start,
                    "window_end": window_start + 1,
                }
            )

    splits = build_intra_patient_10fold_splits("chb01", pd.DataFrame(rows), _cfg(n_splits=10))

    assert len(splits) == 10
    all_test_ids: set[tuple[object, ...]] = set()
    for split in splits:
        for name in ("train", "val", "test"):
            assert {int(window["label"]) for window in split[f"{name}_windows"]} == {0, 1}
        test_ids = _sample_ids(split["test_windows"])
        assert all_test_ids.isdisjoint(test_ids)
        all_test_ids.update(test_ids)
    assert len(all_test_ids) == len(rows)


def test_intra_patient_rejects_ten_folds_when_a_class_has_too_few_exact_windows() -> None:
    rows = []
    for label, count in ((1, 9), (0, 30)):
        for index in range(count):
            rows.append(
                {
                    "subject_id": "chb01",
                    "record_id": "r0",
                    "block_id": "shared_block",
                    "label": label,
                    "window_start": label * 1000 + index,
                    "window_end": label * 1000 + index + 1,
                }
            )

    with pytest.raises(ValueError, match="each class needs at least 10 samples"):
        build_intra_patient_10fold_splits("chb01", pd.DataFrame(rows), _cfg(n_splits=10))


def test_mixed_5fold_json_pools_subjects_and_balances_each_set_after_split(tmp_path: Path) -> None:
    index_path = tmp_path / "window_index.csv"
    pd.DataFrame(_mixed_rows()).to_csv(index_path, index=False)
    original_index = index_path.read_bytes()
    windows = pd.read_csv(index_path)

    paths = save_mixed_5fold_splits(windows, _cfg(tmp_path / "splits", balance=True, n_splits=5))

    assert index_path.read_bytes() == original_index
    assert len(paths) == 5
    for path in paths:
        split = json.loads(path.read_text(encoding="utf-8"))
        assert split["strategy"] == "mixed_5fold"
        _assert_balanced(split)
        ids = {
            name: {
                (window["data_path"], int(window["start_sample"]), int(window["end_sample"]))
                for window in split[f"{name}_windows"]
            }
            for name in ("train", "val", "test")
        }
        assert ids["train"].isdisjoint(ids["val"])
        assert ids["train"].isdisjoint(ids["test"])
        assert ids["val"].isdisjoint(ids["test"])
        for name in ("train", "val", "test"):
            assert len({window["subject_id"] for window in split[f"{name}_windows"]}) > 1


def test_mixed_5fold_is_reproducible_and_allows_partial_overlap_across_sets() -> None:
    windows = pd.DataFrame(_mixed_rows())
    cfg = _cfg(balance=True, n_splits=5)

    first = build_mixed_5fold_splits(windows, cfg)
    second = build_mixed_5fold_splits(windows, cfg)

    assert first == second
    assert any(_has_partial_overlap(split["train_windows"], split["test_windows"]) for split in first)


def test_mixed_5fold_config_uses_25_percent_ictal_overlap() -> None:
    cfg = load_config("configs/exp_mixed_5fold.yaml")

    assert int(cfg.experiment.n_splits) == 5
    assert float(cfg.windowing.ictal_overlap) == 0.25
    assert list(cfg.windowing.balance_sets) == ["train", "val", "test"]
