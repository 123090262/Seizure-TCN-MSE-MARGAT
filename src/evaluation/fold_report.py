from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import write_json


TEST_METRIC_KEYS = (
    "test_loss",
    "accuracy",
    "sensitivity",
    "recall",
    "specificity",
    "precision",
    "f1",
    "auroc",
    "auprc",
)


def save_fold_report(metrics: dict[str, Any], path: str | Path) -> None:
    write_json(metrics, path)


def summarize_fold_reports(
    report_paths: list[str | Path],
    output_dir: str | Path,
    *,
    expected_folds: int | None = None,
) -> dict[str, Any]:
    """Aggregate completed fold reports and save machine- and human-readable summaries."""
    paths = [Path(path) for path in report_paths]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot summarize missing fold reports: {missing}")
    if expected_folds is not None and len(paths) != expected_folds:
        raise ValueError(f"Expected {expected_folds} fold reports, got {len(paths)}")

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = []
    metrics: dict[str, dict[str, float]] = {}
    for key in TEST_METRIC_KEYS:
        values = np.asarray([float(report[key]) for report in reports], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        metrics[key] = {"mean": mean, "std": std}
        rows.append({"metric": key, "mean": mean, "std": std})

    confusion_matrix_sum = np.sum(
        [np.asarray(report["confusion_matrix"], dtype=int) for report in reports],
        axis=0,
    ).tolist()
    summary: dict[str, Any] = {
        "fold_count": len(reports),
        "metrics": metrics,
        "confusion_matrix_sum": confusion_matrix_sum,
    }
    normalization_scopes = {
        str(report["normalization_scope"])
        for report in reports
        if "normalization_scope" in report
    }
    if len(normalization_scopes) == 1:
        summary["normalization_scope"] = normalization_scopes.pop()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_json(summary, output_path / "summary.json")
    pd.DataFrame(rows).to_csv(output_path / "summary.csv", index=False)
    return summary
