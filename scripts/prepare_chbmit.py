from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.data.chbmit_reader import build_metadata
from src.data.preprocessing import preprocess_from_metadata
from src.data.windowing import generate_window_index
from src.utils.config import load_config
from src.utils.io import write_dataframe
from src.utils.paths import window_index_path
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.training.seed))

    metadata = build_metadata(
        root=cfg.data.root,
        channel_set=str(cfg.data.channel_set),
        custom_channels=list(cfg.data.custom_channels or []),
        expected_sampling_rate=int(cfg.data.sampling_rate),
    )
    metadata = preprocess_from_metadata(metadata, cfg)
    write_dataframe(metadata, cfg.data.metadata_path)

    windows = generate_window_index(metadata, cfg)
    window_path = window_index_path(cfg)
    write_dataframe(windows, window_path)
    print(f"Saved metadata: {cfg.data.metadata_path}")
    print(f"Saved window index: {window_path}")


if __name__ == "__main__":
    main()
