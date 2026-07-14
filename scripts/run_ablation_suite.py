from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from src.data.splitters import save_mixed_5fold_splits
from src.evaluation.fold_report import TEST_METRIC_KEYS, summarize_fold_reports
from src.utils.config import load_config
from src.utils.io import write_json
from src.utils.paths import window_index_path
from src.utils.run_naming import timestamped_run_name, validate_run_name


PAIR_METRICS = ("accuracy", "sensitivity", "specificity", "precision", "f1", "auroc", "auprc", "test_loss")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("experiments"), list):
        raise ValueError(f"Invalid ablation manifest: {path}")
    names = [str(item.get("name", "")) for item in manifest["experiments"]]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Ablation experiment names must be non-empty and unique")
    reference = str(manifest.get("reference", "full"))
    if reference not in names:
        raise ValueError(f"Reference experiment {reference!r} is missing from the manifest")
    return manifest


def read_fold_reports(experiment_dir: Path) -> list[dict[str, Any]]:
    fold_dirs = sorted(experiment_dir.glob("fold_*"), key=lambda path: int(path.name.split("_")[-1]))
    reports = []
    for fold_dir in fold_dirs:
        report_path = fold_dir / "report.json"
        if report_path.exists():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    return reports


def paired_effect(reference: list[dict[str, Any]], ablation: list[dict[str, Any]], metric: str) -> tuple[float, float, float]:
    # Positive effect means the removed/replaced module helped the full model.
    if metric == "test_loss":
        values = np.asarray([float(ablated[metric]) - float(full[metric]) for full, ablated in zip(reference, ablation)])
    else:
        values = np.asarray([float(full[metric]) - float(ablated[metric]) for full, ablated in zip(reference, ablation)])
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    # 95% paired CI for the fixed five-fold screening design (t_0.975,4=2.776).
    half_width = 2.776 * std / math.sqrt(len(values)) if len(values) == 5 else float("nan")
    return mean, std, half_width


def summarize_suite(suite_dir: Path, manifest: dict[str, Any], expected_folds: int) -> pd.DataFrame:
    experiments = {str(item["name"]): item for item in manifest["experiments"]}
    reference_name = str(manifest.get("reference", "full"))
    reference = read_fold_reports(suite_dir / reference_name)
    if len(reference) != expected_folds:
        raise FileNotFoundError(f"Reference {reference_name!r} has {len(reference)}/{expected_folds} completed folds")

    rows: list[dict[str, Any]] = []
    for name, experiment in experiments.items():
        reports = read_fold_reports(suite_dir / name)
        if len(reports) != expected_folds:
            continue
        row: dict[str, Any] = {
            "experiment": name,
            "group": experiment.get("group", ""),
            "removed_or_replaced_module": experiment.get("removed_module", "none"),
            "fold_count": len(reports),
            "parameter_count": int(reports[0]["parameter_count"]),
            "mean_fold_training_time_minutes": float(np.mean([float(report["total_training_time"]) for report in reports]) / 60.0),
        }
        for metric in TEST_METRIC_KEYS:
            values = np.asarray([float(report[metric]) for report in reports])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        for metric in PAIR_METRICS:
            effect, effect_std, ci = paired_effect(reference, reports, metric)
            row[f"module_effect_{metric}"] = effect
            row[f"module_effect_{metric}_paired_std"] = effect_std
            row[f"module_effect_{metric}_ci95_half_width"] = ci
        accuracy_effect = float(row["module_effect_accuracy"])
        if name == reference_name:
            verdict = "reference"
        elif accuracy_effect >= 0.005:
            verdict = "helpful_module"
        elif accuracy_effect <= -0.005:
            verdict = "potentially_harmful_module"
        else:
            verdict = "small_or_inconclusive"
        row["accuracy_verdict"] = verdict
        row["question"] = experiment.get("tests", "")
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)
    frame.to_csv(suite_dir / "ablation_summary.csv", index=False)
    write_json(frame.to_dict("records"), suite_dir / "ablation_summary.json")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_mixed_5fold.yaml")
    parser.add_argument("--model_config", default="configs/model_tcn_mse_margat.yaml")
    parser.add_argument("--manifest", default="configs/ablations/suite.yaml")
    parser.add_argument("--run_name", help="Suite directory name; defaults to a unique timestamp.")
    parser.add_argument("--run_dir", help="Existing suite directory, primarily for --summarize_only.")
    parser.add_argument("--experiments", nargs="*", help="Optional subset of manifest experiment names.")
    parser.add_argument("--resume", action="store_true", help="Skip folds whose report.json already exists.")
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    base_cfg = load_config(args.config, args.model_config)
    expected_folds = int(base_cfg.experiment.n_splits)
    model_name = str(base_cfg.model.name)
    output_tag = getattr(base_cfg.experiment, "output_tag", None)
    run_name = validate_run_name(args.run_name) if args.run_name else timestamped_run_name(str(output_tag)) if output_tag else timestamped_run_name()
    suite_dir = Path(args.run_dir) if args.run_dir else Path(base_cfg.project.output_dir) / "ablations" / "mixed_5fold" / model_name / run_name

    if args.summarize_only:
        frame = summarize_suite(suite_dir, manifest, expected_folds)
        print(frame[["experiment", "accuracy_mean", "module_effect_accuracy", "accuracy_verdict"]].to_string(index=False))
        print(f"Saved summary: {suite_dir / 'ablation_summary.csv'}")
        return

    if suite_dir.exists() and any(suite_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Suite directory is not empty: {suite_dir}. Pass --resume or choose another run name.")
    suite_dir.mkdir(parents=True, exist_ok=True)
    saved_manifest = suite_dir / "suite_manifest.yaml"
    if saved_manifest.exists() and saved_manifest.read_bytes() != manifest_path.read_bytes():
        raise ValueError("The current manifest differs from the saved suite manifest; use a new run name")
    if not saved_manifest.exists():
        shutil.copy2(manifest_path, saved_manifest)
    print(f"Ablation suite directory: {suite_dir}", flush=True)

    windows = pd.read_csv(window_index_path(base_cfg))
    split_paths = save_mixed_5fold_splits(windows, base_cfg)
    selected = set(args.experiments or [])
    known = {str(item["name"]) for item in manifest["experiments"]}
    unknown = selected - known
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")

    for experiment in manifest["experiments"]:
        name = str(experiment["name"])
        if selected and name not in selected:
            continue
        overlay = experiment.get("config")
        output_dir = suite_dir / name
        report_paths = []
        print(f"\n=== {name}: {experiment.get('tests', '')} ===", flush=True)
        for split_path in split_paths:
            fold_dir = output_dir / split_path.stem
            report_path = fold_dir / "report.json"
            if args.resume and report_path.exists():
                print(f"Skipping completed {fold_dir}", flush=True)
            else:
                if fold_dir.exists() and any(fold_dir.iterdir()):
                    archived = fold_dir.with_name(f"{fold_dir.name}.incomplete_{timestamped_run_name()}")
                    fold_dir.rename(archived)
                    print(f"Archived incomplete fold to {archived}", flush=True)
                command = [
                    sys.executable,
                    "scripts/train_one_fold.py",
                    "--config",
                    args.config,
                    "--model_config",
                    args.model_config,
                    "--split",
                    str(split_path),
                    "--output_dir",
                    str(fold_dir),
                ]
                if overlay:
                    command.extend(["--override_config", str(overlay)])
                subprocess.run(command, check=True)
            report_paths.append(report_path)
        summarize_fold_reports(report_paths, output_dir, expected_folds=expected_folds)

    reference_name = str(manifest.get("reference", "full"))
    if (suite_dir / reference_name / "summary.json").exists():
        frame = summarize_suite(suite_dir, manifest, expected_folds)
        print("\nAccuracy-ranked completed experiments:")
        print(frame[["experiment", "accuracy_mean", "module_effect_accuracy", "accuracy_verdict"]].to_string(index=False))
        print(f"Saved summary: {suite_dir / 'ablation_summary.csv'}")
    else:
        print("Reference experiment is incomplete; suite-level paired summary was not generated.")


if __name__ == "__main__":
    main()
