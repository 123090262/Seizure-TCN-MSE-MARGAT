from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.data.chbmit_reader import get_channel_set
from src.data.datamodule import build_dataloaders
from src.data.normalization import ZScoreNormalizer
from src.evaluation.fold_report import summarize_fold_reports
from src.evaluation.metrics import binary_metrics, parameter_count
from src.models.factory import build_model
from src.training.callbacks import load_checkpoint
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        required=True,
        help="A fold directory or a model directory containing fold_* directories.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    fold_dirs = [run_dir] if (run_dir / "best.pt").exists() else sorted(run_dir.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No best.pt or fold_* directories found under {run_dir}")

    report_paths = []
    device = get_device()
    for fold_dir in fold_dirs:
        cfg = load_config(fold_dir / "config.yaml")
        split = read_json(fold_dir / "split.json")
        stats = read_json(fold_dir / "normalization_stats.json")
        normalizer = ZScoreNormalizer(
            mean=np.asarray(stats["mean"], dtype=np.float32),
            std=np.asarray(stats["std"], dtype=np.float32),
            eps=float(stats.get("eps", cfg.normalization.eps)),
            channel_names=list(stats.get("channel_names", [])) or None,
        )
        channels = get_channel_set(str(cfg.data.channel_set), list(cfg.data.custom_channels or []))
        loaders = build_dataloaders(split, cfg, channels=channels, normalizer=normalizer)
        model = build_model(
            cfg,
            num_channels=len(channels),
            window_size=int(cfg.windowing.window_size_samples),
            channel_names=channels,
        ).to(device)
        checkpoint = load_checkpoint(fold_dir / "best.pt", model, map_location=device)

        # Reuse the exact evaluation loop and metric implementation used during training.
        trainer = Trainer(model, cfg, device, fold_dir)
        test_loss, test_y, test_p = trainer._run_epoch(loaders["test"], train=False)
        test_metrics = binary_metrics(test_y, test_p, threshold=float(cfg.evaluation.threshold))

        report_path = fold_dir / "report.json"
        old_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        backup_path = fold_dir / "report.pre_best_restore.json"
        if report_path.exists() and not backup_path.exists():
            shutil.copy2(report_path, backup_path)
        report = {
            **old_report,
            "best_epoch": int(checkpoint["epoch"]),
            "best_metrics": checkpoint.get("metrics", old_report.get("best_metrics", {})),
            "test_loss": test_loss,
            **test_metrics,
            "parameter_count": parameter_count(model),
            "evaluation_checkpoint": "best.pt",
        }
        write_json(report, report_path)
        report_paths.append(report_path)
        print(f"Re-evaluated {fold_dir} from best.pt")

    if len(fold_dirs) > 1:
        summarize_fold_reports(report_paths, run_dir, expected_folds=len(fold_dirs))
        print(f"Updated summary: {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
