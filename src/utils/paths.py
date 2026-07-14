from __future__ import annotations

from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(base) if base is not None else project_root()) / path


def window_index_path(cfg: Any) -> Path:
    """Resolve a variant-specific window index with legacy fallback."""
    configured = getattr(cfg.data, "window_index_path", None)
    if configured:
        return Path(str(configured))
    return Path(str(cfg.data.processed_dir)) / "window_index.csv"
