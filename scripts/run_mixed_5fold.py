from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from src.data.splitters import save_mixed_5fold_splits
from src.evaluation.fold_report import TEST_METRIC_KEYS, summarize_fold_reports
from src.utils.config import load_config
from src.utils.io import write_json
from src.utils.paths import window_index_path
from src.utils.run_naming import timestamped_run_name, validate_run_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--generate_only", action="store_true", help="Only generate split files; do not train.")
    parser.add_argument("--run_name", help="Optional run directory name; defaults to a unique timestamp.")
    args = parser.parse_args()
    cfg = load_config(args.config, args.model_config)
    window_path = window_index_path(cfg)
    if not window_path.exists():
        raise FileNotFoundError(
            f"Window index not found: {window_path}. Prepare it before generating splits or training."
        )
    windows = pd.read_csv(window_path)
    paths = save_mixed_5fold_splits(windows, cfg)
    model_name = str(cfg.model.name)
    output_tag = getattr(cfg.experiment, "output_tag", None)
    run_name = (
        validate_run_name(args.run_name)
        if args.run_name
        else timestamped_run_name(str(output_tag)) if output_tag else timestamped_run_name()
    )
    model_output_dir = Path(cfg.project.output_dir) / "mixed_5fold" / model_name / run_name
    report_paths = []
    if not args.generate_only:
        write_json(
            {
                "output_tag": str(output_tag) if output_tag else None,
                "run_name": run_name,
                "window_index_path": str(window_path),
                "split_root": str(cfg.data.splits_dir),
                "window_size_sec": float(cfg.windowing.window_size_sec),
                "window_size_samples": int(cfg.windowing.window_size_samples),
                "ictal_overlap": float(cfg.windowing.ictal_overlap),
                "interictal_overlap": float(cfg.windowing.interictal_overlap),
                "config": args.config,
                "model_config": args.model_config,
            },
            model_output_dir / "run_manifest.json",
        )
        print(f"Run directory: {model_output_dir}", flush=True)
    for path in paths:
        if not args.generate_only:
            out_dir = model_output_dir / path.stem
            subprocess.run(
                [
                    sys.executable,
                    "scripts/train_one_fold.py",
                    "--config",
                    args.config,
                    "--model_config",
                    args.model_config,
                    "--split",
                    str(path),
                    "--output_dir",
                    str(out_dir),
                ],
                check=True,
            )
            report_paths.append(out_dir / "report.json")
    if args.generate_only:
        print("Mixed-window 5-fold split generation complete.")
        return

    summary = summarize_fold_reports(report_paths, model_output_dir, expected_folds=len(paths))
    print(f"\n{len(paths)}-fold test metric summary for {model_name}:")
    for key in TEST_METRIC_KEYS:
        stats = summary["metrics"][key]
        print(f"{key}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    print(f"confusion_matrix_sum: {summary['confusion_matrix_sum']}")
    print(f"Saved summary: {model_output_dir / 'summary.json'}")
    print(f"Saved summary table: {model_output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
