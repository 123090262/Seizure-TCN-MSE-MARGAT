from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_array(array: np.ndarray | torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pt":
        torch.save(array, path)
    elif path.suffix == ".npy":
        np.save(path, array)
    else:
        raise ValueError(f"Unsupported array suffix: {path.suffix}")


def load_array(path: str | Path, mmap_mode: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".pt":
        tensor = torch.load(path, map_location="cpu")
        return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode=mmap_mode)
    raise ValueError(f"Unsupported array suffix: {path.suffix}")


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
