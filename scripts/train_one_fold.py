from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from src.data.chbmit_reader import get_channel_set
from src.data.datamodule import build_dataloaders
from src.data.normalization import ZScoreNormalizer
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.utils.device import get_device
from src.utils.io import read_json, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument(
        "--override_config",
        action="append",
        default=[],
        help="Additional YAML overlay; may be supplied more than once.",
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in a non-empty output directory.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.model_config, *args.override_config)
    set_seed(int(cfg.training.seed))
    split = read_json(args.split)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty run directory: {output_dir}. "
            "Use a new --output_dir or pass --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")
    write_json(split, output_dir / "split.json")

    channels = get_channel_set(str(cfg.data.channel_set), list(cfg.data.custom_channels or []))
    normalizer = ZScoreNormalizer.fit(split["train_windows"], eps=float(cfg.normalization.eps), channel_names=channels)
    normalizer.save(output_dir / "normalization_stats.json")

    loaders = build_dataloaders(split, cfg, channels=channels, normalizer=normalizer)
    model = build_model(
        cfg,
        num_channels=len(channels),
        window_size=int(cfg.windowing.window_size_samples),
        channel_names=channels,
    )
    trainer = Trainer(model, cfg, get_device(), output_dir)
    report = trainer.fit(loaders)
    write_json(report, output_dir / "report.json")
    pd.DataFrame(report.get("history", [])).to_csv(output_dir / "history.csv", index=False)
    print(f"Saved fold report: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
