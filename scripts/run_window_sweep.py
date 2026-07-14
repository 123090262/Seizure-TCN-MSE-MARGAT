from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.config import load_config
from src.utils.paths import window_index_path
from src.utils.run_naming import validate_run_name


DEFAULT_CONFIGS = [
    "configs/window_sweeps/win1s_ov50.yaml",
    "configs/window_sweeps/win2s_ov50.yaml",
]


def _isolation_key(cfg: Any) -> tuple[str, str, str, str]:
    return (
        validate_run_name(str(cfg.experiment.output_tag)),
        str(window_index_path(cfg)),
        str(cfg.data.splits_dir),
        str(cfg.project.output_dir),
    )


def _validate_isolation(config_paths: list[str]) -> None:
    keys = [_isolation_key(load_config(path)) for path in config_paths]
    labels = ["output_tag", "window_index_path", "splits_dir", "output_dir"]
    for index, label in enumerate(labels):
        values = [key[index] for key in keys]
        if len(values) != len(set(values)):
            raise ValueError(f"Window sweep configs must use distinct {label} values: {values}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and run the isolated 1s/50% and 2s/50% mixed-5-fold experiments."
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--model_config", default="configs/model_tcn_mse_margat.yaml")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--overwrite_data", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--generate_only", action="store_true", help="Prepare data and split JSON files without training.")
    args = parser.parse_args()

    config_paths = [str(Path(path)) for path in args.configs]
    _validate_isolation(config_paths)
    for config_path in config_paths:
        cfg = load_config(config_path)
        tag = str(cfg.experiment.output_tag)
        print(f"\n=== Window variant: {tag} ===", flush=True)
        if not args.skip_prepare:
            prepare_command = [
                sys.executable,
                "scripts/prepare_window_variant.py",
                "--config",
                config_path,
            ]
            if args.overwrite_data:
                prepare_command.append("--overwrite")
            subprocess.run(prepare_command, check=True)
        if args.prepare_only:
            continue

        run_command = [
            sys.executable,
            "scripts/run_mixed_5fold.py",
            "--config",
            config_path,
            "--model_config",
            args.model_config,
        ]
        if args.generate_only:
            run_command.append("--generate_only")
        subprocess.run(run_command, check=True)


if __name__ == "__main__":
    main()
