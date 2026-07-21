from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.fold_report import TEST_METRIC_KEYS, summarize_fold_reports


def _report(value: float, confusion_matrix: list[list[int]]) -> dict[str, object]:
    report: dict[str, object] = {key: value for key in TEST_METRIC_KEYS}
    report["confusion_matrix"] = confusion_matrix
    return report


def test_summarize_fold_reports_saves_means_std_and_confusion_matrix(tmp_path: Path) -> None:
    paths = []
    for fold, report in enumerate((
        _report(0.8, [[8, 2], [1, 9]]),
        _report(1.0, [[10, 0], [0, 10]]),
    )):
        path = tmp_path / f"fold_{fold}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)

    output_dir = tmp_path / "summary"
    summary = summarize_fold_reports(paths, output_dir, expected_folds=2)

    assert summary["fold_count"] == 2
    assert summary["metrics"]["accuracy"]["mean"] == pytest.approx(0.9)
    assert summary["metrics"]["accuracy"]["std"] == pytest.approx(2**0.5 / 10)
    assert summary["confusion_matrix_sum"] == [[18, 2], [1, 19]]
    saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    table = pd.read_csv(output_dir / "summary.csv")
    assert set(table["metric"]) == set(TEST_METRIC_KEYS)


def test_summarize_fold_reports_rejects_incomplete_run(tmp_path: Path) -> None:
    path = tmp_path / "fold_0.json"
    path.write_text(json.dumps(_report(0.8, [[1, 0], [0, 1]])), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 5 fold reports, got 1"):
        summarize_fold_reports([path], tmp_path / "summary", expected_folds=5)


def test_summarize_fold_reports_includes_v2_threshold_metrics_when_complete(tmp_path: Path) -> None:
    paths = []
    for fold, value in enumerate((0.4, 0.6)):
        report = _report(0.9, [[9, 1], [1, 9]])
        report.update({"fixed_threshold_accuracy": 0.8 + fold * 0.1, "decision_threshold": value})
        path = tmp_path / f"fold_{fold}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)

    summary = summarize_fold_reports(paths, tmp_path / "summary", expected_folds=2)
    assert summary["metrics"]["fixed_threshold_accuracy"]["mean"] == pytest.approx(0.85)
    assert summary["metrics"]["decision_threshold"]["mean"] == pytest.approx(0.5)
