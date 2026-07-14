from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import pandas as pd

from src.data.windowing import generate_windows_for_record
from src.utils.config import load_config
from src.utils.io import write_dataframe, write_json
from src.utils.paths import window_index_path
from src.utils.run_naming import timestamped_run_name, validate_run_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant_signature(cfg: Any, source_metadata: Path) -> dict[str, Any]:
    sampling_rate = int(cfg.data.sampling_rate)
    window_samples = int(cfg.windowing.window_size_samples)
    window_seconds = float(cfg.windowing.window_size_sec)
    expected_samples = round(window_seconds * sampling_rate)
    if window_samples != expected_samples:
        raise ValueError(
            f"window_size_samples={window_samples} does not match "
            f"window_size_sec*sampling_rate={expected_samples}"
        )
    overlaps = {
        "ictal_overlap": float(cfg.windowing.ictal_overlap),
        "interictal_overlap": float(cfg.windowing.interictal_overlap),
    }
    if any(not 0.0 <= value < 1.0 for value in overlaps.values()):
        raise ValueError(f"Window overlaps must lie in [0,1): {overlaps}")
    output_tag = validate_run_name(str(cfg.experiment.output_tag))
    return {
        "output_tag": output_tag,
        "source_metadata_path": str(source_metadata),
        "source_metadata_sha256": _sha256(source_metadata),
        "sampling_rate": sampling_rate,
        "window_size_samples": window_samples,
        "window_size_sec": window_seconds,
        **overlaps,
        "interictal_exclusion_sec": float(cfg.windowing.interictal_exclusion_sec),
        "channel_set": str(cfg.data.channel_set),
    }


def _matches_existing(manifest_path: Path, signature: dict[str, Any]) -> bool:
    if not manifest_path.exists():
        return False
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    return all(existing.get(key) == value for key, value in signature.items())


def prepare_variant(config_path: str | Path, *, overwrite: bool = False) -> Path:
    cfg = load_config(config_path)
    source_metadata = Path(str(cfg.data.source_metadata_path))
    if not source_metadata.exists():
        raise FileNotFoundError(f"Source metadata not found: {source_metadata}")

    metadata_path = Path(str(cfg.data.metadata_path))
    index_path = window_index_path(cfg)
    manifest_path = index_path.parent / "dataset_manifest.json"
    signature = _variant_signature(cfg, source_metadata)

    if index_path.exists() and not overwrite:
        if _matches_existing(manifest_path, signature):
            print(f"Reusing matching window variant: {index_path}")
            return index_path
        raise FileExistsError(
            f"Refusing to replace existing window data with a different or missing manifest: {index_path}. "
            "Use --overwrite only after checking the target variant directory."
        )

    metadata = pd.read_csv(source_metadata)
    required = {"subject_id", "file_name", "processed_path", "sampling_rate", "duration_sec", "seizure_intervals"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Source metadata is missing required columns: {sorted(missing)}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    incomplete_path = index_path.with_name(
        f"{index_path.stem}.incomplete_{timestamped_run_name()}{index_path.suffix}"
    )
    counts: Counter[int] = Counter()
    total = 0
    wrote_header = False
    try:
        for row in metadata.to_dict("records"):
            records = [window.to_dict() for window in generate_windows_for_record(row, cfg)]
            if not records:
                continue
            frame = pd.DataFrame.from_records(records)
            frame.to_csv(
                incomplete_path,
                mode="a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            label_counts = frame["label"].astype(int).value_counts().to_dict()
            counts.update({int(label): int(count) for label, count in label_counts.items()})
            total += len(frame)
        if not wrote_header:
            raise ValueError("Window generation produced no rows")
        incomplete_path.replace(index_path)
    except BaseException:
        print(f"Incomplete window index retained for inspection: {incomplete_path}")
        raise

    write_dataframe(metadata, metadata_path)
    write_json(
        {
            **signature,
            "metadata_snapshot_path": str(metadata_path),
            "window_index_path": str(index_path),
            "split_root": str(cfg.data.splits_dir),
            "window_count": total,
            "class_counts": {str(label): count for label, count in sorted(counts.items())},
            "created_at": datetime.now().astimezone().isoformat(),
        },
        manifest_path,
    )
    print(f"Saved metadata snapshot: {metadata_path}")
    print(f"Saved window index: {index_path} ({total:,} rows)")
    print(f"Saved dataset manifest: {manifest_path}")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing variant only when its isolated target directory is intentional.",
    )
    args = parser.parse_args()
    prepare_variant(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
