from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

try:
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:  # pragma: no cover - exercised when optional dependency is absent
    DictConfig = Any  # type: ignore[misc, assignment]
    OmegaConf = None  # type: ignore[assignment]


class AttrDict(dict):
    """Small attr-access dict used when OmegaConf is unavailable."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(path) if OmegaConf is not None else yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = cfg or {}
    defaults = cfg.pop("defaults", None)
    if not defaults:
        return cfg if OmegaConf is not None else _wrap(cfg)

    merged = OmegaConf.create() if OmegaConf is not None else {}
    for item in defaults:
        if isinstance(item, str):
            default_path = path.parent / f"{item}.yaml"
        elif isinstance(item, dict):
            name = next(iter(item.values()))
            default_path = path.parent / f"{name}.yaml"
        else:
            raise TypeError(f"Unsupported defaults item in {path}: {item!r}")
        loaded = _load_yaml(default_path)
        merged = OmegaConf.merge(merged, loaded) if OmegaConf is not None else _deep_merge(_unwrap(merged), _unwrap(loaded))
    return OmegaConf.merge(merged, cfg) if OmegaConf is not None else _wrap(_deep_merge(_unwrap(merged), cfg))


def load_config(*paths: str | Path) -> Any:
    """Load and deep-merge YAML configs."""
    if not paths:
        raise ValueError("At least one config path is required.")
    merged = OmegaConf.create() if OmegaConf is not None else {}
    for path in paths:
        loaded = _load_yaml(Path(path))
        merged = OmegaConf.merge(merged, loaded) if OmegaConf is not None else _deep_merge(_unwrap(merged), _unwrap(loaded))
    return merged if OmegaConf is not None else _wrap(merged)


def save_config(cfg: DictConfig | dict[str, Any], path: str | Path) -> None:
    """Save an OmegaConf-compatible config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if OmegaConf is not None:
        OmegaConf.save(config=cfg, f=path)
    else:
        path.write_text(yaml.safe_dump(_unwrap(cfg), sort_keys=False), encoding="utf-8")


def to_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Convert config to a plain resolved dictionary."""
    if OmegaConf is not None:
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return _unwrap(cfg)
