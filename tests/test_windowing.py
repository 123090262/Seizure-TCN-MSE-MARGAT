from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.windowing import balance_windows, generate_windows_for_record


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        windowing=SimpleNamespace(
            window_size_samples=256,
            ictal_overlap=0.75,
            interictal_overlap=0.0,
            interictal_exclusion_sec=10,
        )
    )


def test_window_labels_and_interictal_exclusion() -> None:
    row = {
        "subject_id": "chb01",
        "file_name": "chb01_01.edf",
        "processed_path": "dummy.npy",
        "sampling_rate": 256,
        "duration_sec": 120,
        "seizure_intervals": json.dumps([{"start_sec": 50, "end_sec": 60}]),
    }
    windows = generate_windows_for_record(row, _cfg())
    assert any(w.label == 1 for w in windows)
    interictal = [w for w in windows if w.label == 0]
    assert interictal
    assert all(w.window_end <= 40 or w.window_start >= 70 for w in interictal)


@pytest.mark.parametrize("label, missing_class", [(0, "seizure"), (1, "non-seizure")])
def test_balance_windows_keeps_single_class_set_with_warning(label: int, missing_class: str) -> None:
    windows = pd.DataFrame([{"window_id": index, "label": label} for index in range(4)])

    with pytest.warns(RuntimeWarning, match=f"no {missing_class} windows"):
        balanced = balance_windows(windows, ratio=1.0, seed=42, context="test set")

    assert sorted(balanced["window_id"].tolist()) == [0, 1, 2, 3]
    assert set(balanced["label"]) == {label}
