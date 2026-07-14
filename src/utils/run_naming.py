from __future__ import annotations

from datetime import datetime
from pathlib import Path


def timestamped_run_name(prefix: str | None = None) -> str:
    """Return a sortable, collision-resistant local timestamp for one experiment run."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return f"{validate_run_name(prefix)}_{timestamp}" if prefix else timestamp


def validate_run_name(value: str) -> str:
    """Accept a single safe path component as a user-provided run name."""
    name = value.strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Run name must be a single non-empty directory name, got {value!r}")
    return name
