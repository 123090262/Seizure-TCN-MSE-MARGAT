from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from src.data.splitters import save_intra_patient_10fold_splits
from src.utils.config import load_config
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
    windows = pd.read_csv(window_path)
    subjects = sorted(windows["subject_id"].astype(str).unique().tolist())
    output_tag = getattr(cfg.experiment, "output_tag", None)
    run_name = (
        validate_run_name(args.run_name)
        if args.run_name
        else timestamped_run_name(str(output_tag)) if output_tag else timestamped_run_name()
    )
    run_dir = Path(cfg.project.output_dir) / "intra_10fold" / str(cfg.model.name) / run_name
    if not args.generate_only:
        print(f"Run directory: {run_dir}", flush=True)
    for subject_id in subjects:
        paths = save_intra_patient_10fold_splits(subject_id, windows, cfg)
        for path in paths:
            if not args.generate_only:
                out_dir = run_dir / subject_id / path.stem
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
    print("Intra-patient split generation complete.")


if __name__ == "__main__":
    main()
