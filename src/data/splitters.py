from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.windowing import balance_windows
from src.utils.io import write_json


_SPLIT_SET_NAMES = ("train", "val", "test")


def _balance_split_sets(
    split_sets: dict[str, pd.DataFrame],
    cfg: Any,
    *,
    strategy: str,
    fold: int,
) -> dict[str, pd.DataFrame]:
    """Balance configured sets after their leakage-safe split boundaries are fixed."""
    windowing_cfg = getattr(cfg, "windowing", None)
    if windowing_cfg is None or not bool(getattr(windowing_cfg, "balance_after_split", False)):
        return split_sets

    configured_sets = getattr(windowing_cfg, "balance_sets", _SPLIT_SET_NAMES)
    if isinstance(configured_sets, str):
        configured_sets = [configured_sets]
    balance_sets = {str(name) for name in configured_sets}
    unknown_sets = balance_sets.difference(_SPLIT_SET_NAMES)
    if unknown_sets:
        raise ValueError(f"Unknown windowing.balance_sets entries: {sorted(unknown_sets)}")

    ratio = float(getattr(windowing_cfg, "balance_ratio", 1.0))
    base_seed = int(cfg.training.seed) + fold * len(_SPLIT_SET_NAMES)
    balanced: dict[str, pd.DataFrame] = {}
    for offset, name in enumerate(_SPLIT_SET_NAMES):
        frame = split_sets[name]
        balanced[name] = (
            balance_windows(
                frame,
                ratio=ratio,
                seed=base_seed + offset,
                context=f"{strategy} fold {fold} {name} set",
            )
            if name in balance_sets
            else frame
        )
    return balanced


def _records_from_windows(windows: pd.DataFrame) -> list[str]:
    if windows.empty:
        return []
    return sorted(windows["record_id"].astype(str).unique().tolist())


def _split_dict(
    strategy: str,
    fold: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    subject_id: str | None = None,
    test_subject: str | None = None,
) -> dict[str, object]:
    out: dict[str, object] = {
        "strategy": strategy,
        "fold": fold,
        "train_records": _records_from_windows(train),
        "val_records": _records_from_windows(val),
        "test_records": _records_from_windows(test),
        "train_windows": train.to_dict("records"),
        "val_windows": val.to_dict("records"),
        "test_windows": test.to_dict("records"),
        "notes": "normalization must be fitted on train only",
    }
    if subject_id is not None:
        out["subject_id"] = subject_id
    if test_subject is not None:
        out["test_subject"] = test_subject
    return out


def _with_sample_groups(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach IDs that bind only windows referencing the exact same source samples."""
    preferred = ["data_path", "start_sample", "end_sample"]
    fallback = ["subject_id", "record_id", "window_start", "window_end"]
    identity_columns = preferred if all(column in windows for column in preferred) else fallback
    if not all(column in windows for column in identity_columns):
        missing = [column for column in identity_columns if column not in windows]
        raise ValueError(f"Cannot identify exact EEG windows; missing columns: {missing}")

    grouped = windows.copy()
    identity = pd.MultiIndex.from_frame(grouped[identity_columns])
    grouped["_sample_group"] = pd.factorize(identity, sort=False)[0]
    label_counts = grouped.groupby("_sample_group")["label"].nunique()
    if bool((label_counts > 1).any()):
        raise ValueError("Identical EEG windows have conflicting labels.")
    unique_samples = grouped.drop_duplicates("_sample_group")[["_sample_group", "label"]].reset_index(drop=True)
    return grouped, unique_samples


def _require_both_classes(frame: pd.DataFrame, *, context: str, fold: int, set_name: str) -> None:
    labels = set(frame["label"].astype(int).unique().tolist())
    if labels != {0, 1}:
        raise ValueError(
            f"Stratified split for {context} fold {fold} {set_name} set does not contain both classes: "
            f"found labels {sorted(labels)}"
        )


def _build_stratified_window_splits(
    windows: pd.DataFrame,
    cfg: Any,
    *,
    strategy: str,
    subject_id: str | None = None,
) -> list[dict[str, object]]:
    """Build sample-level stratified folds while keeping exact duplicate windows together."""
    context = subject_id or strategy
    if windows.empty:
        raise ValueError(f"No windows found for {context}")
    grouped_windows, unique_samples = _with_sample_groups(windows.reset_index(drop=True))
    n_splits = int(cfg.experiment.n_splits)
    if n_splits < 2:
        raise ValueError(f"Need at least 2 folds for {strategy}, got {n_splits}")

    class_counts = unique_samples["label"].astype(int).value_counts()
    if set(class_counts.index.tolist()) != {0, 1} or int(class_counts.min()) < n_splits:
        raise ValueError(
            f"Cannot build {n_splits} stratified folds for {context}; exact-window class counts are "
            f"{class_counts.sort_index().to_dict()} and each class needs at least {n_splits} samples."
        )

    labels = unique_samples["label"].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.training.seed))
    out: list[dict[str, object]] = []
    for fold, (train_val_idx, test_idx) in enumerate(splitter.split(unique_samples, labels)):
        train_val_samples = unique_samples.iloc[train_val_idx].reset_index(drop=True)
        test_groups = unique_samples.iloc[test_idx]["_sample_group"]
        test = grouped_windows[grouped_windows["_sample_group"].isin(test_groups)]

        val_fraction = float(cfg.experiment.val_fraction)
        train_val_counts = train_val_samples["label"].astype(int).value_counts()
        if val_fraction <= 0 or int(train_val_counts.min()) < 2:
            raise ValueError(
                f"Cannot create stratified train/val sets for {context} fold {fold}; "
                f"val_fraction={val_fraction}, class counts={train_val_counts.sort_index().to_dict()}"
            )
        val_parts = []
        for label, class_samples in train_val_samples.groupby("label", sort=True):
            val_count = min(len(class_samples) - 1, max(1, int(round(len(class_samples) * val_fraction))))
            val_parts.append(
                class_samples.sample(
                    n=val_count,
                    random_state=int(cfg.training.seed) + fold * 2 + int(label),
                )
            )
        val_samples = pd.concat(val_parts, ignore_index=True)
        val_groups = val_samples["_sample_group"]
        train_groups = train_val_samples.loc[
            ~train_val_samples["_sample_group"].isin(val_groups), "_sample_group"
        ]
        train = grouped_windows[grouped_windows["_sample_group"].isin(train_groups)]
        val = grouped_windows[grouped_windows["_sample_group"].isin(val_groups)]

        split_sets = {
            name: frame.drop(columns="_sample_group").reset_index(drop=True)
            for name, frame in {"train": train, "val": val, "test": test}.items()
        }
        for name, frame in split_sets.items():
            _require_both_classes(frame, context=context, fold=fold, set_name=name)
        balanced_sets = _balance_split_sets(
            split_sets,
            cfg,
            strategy=strategy,
            fold=fold,
        )
        train, val, test = (balanced_sets[name] for name in _SPLIT_SET_NAMES)
        split = _split_dict(strategy, fold, train, val, test, subject_id=subject_id)
        out.append(split)
    return out


def build_intra_patient_10fold_splits(subject_id: str, windows: pd.DataFrame, cfg: Any) -> list[dict[str, object]]:
    """Build fixed-count sample-level stratified folds for one patient."""
    subject_windows = windows[windows["subject_id"] == subject_id]
    return _build_stratified_window_splits(
        subject_windows,
        cfg,
        strategy="intra_10fold",
        subject_id=subject_id,
    )


def build_mixed_5fold_splits(windows: pd.DataFrame, cfg: Any) -> list[dict[str, object]]:
    """Build stratified folds from windows pooled across all patients."""
    return _build_stratified_window_splits(windows, cfg, strategy="mixed_5fold")


def save_intra_patient_10fold_splits(subject_id: str, windows: pd.DataFrame, cfg: Any) -> list[Path]:
    splits = build_intra_patient_10fold_splits(subject_id, windows, cfg)
    base = Path(cfg.data.splits_dir) / "intra_10fold" / subject_id
    paths = []
    for split in splits:
        path = base / f"fold_{split['fold']}.json"
        write_json(split, path)
        paths.append(path)
    return paths


def save_mixed_5fold_splits(windows: pd.DataFrame, cfg: Any) -> list[Path]:
    splits = build_mixed_5fold_splits(windows, cfg)
    base = Path(cfg.data.splits_dir) / "mixed_5fold"
    paths = []
    for split in splits:
        path = base / f"fold_{split['fold']}.json"
        write_json(split, path)
        paths.append(path)
    return paths


def build_lopo_splits(subjects: list[str], windows: pd.DataFrame, cfg: Any) -> list[dict[str, object]]:
    """Build leave-one-patient-out splits."""
    subjects = sorted(subjects)
    if len(subjects) < 3:
        raise ValueError("LOPO requires at least 3 subjects to create train/val/test splits.")
    splits: list[dict[str, object]] = []
    val_count = int(cfg.experiment.val_subjects)
    for fold, test_subject in enumerate(subjects):
        remaining = [s for s in subjects if s != test_subject]
        val_subjects = remaining[:val_count]
        train_subjects = [s for s in remaining if s not in val_subjects]
        train = windows[windows["subject_id"].isin(train_subjects)]
        val = windows[windows["subject_id"].isin(val_subjects)]
        test = windows[windows["subject_id"] == test_subject]
        balanced_sets = _balance_split_sets(
            {"train": train, "val": val, "test": test},
            cfg,
            strategy="lopo",
            fold=fold,
        )
        train, val, test = (balanced_sets[name] for name in _SPLIT_SET_NAMES)
        split = _split_dict("lopo", fold, train, val, test, test_subject=test_subject)
        split["train_subjects"] = train_subjects
        split["val_subjects"] = val_subjects
        splits.append(split)
    return splits


def save_lopo_splits(subjects: list[str], windows: pd.DataFrame, cfg: Any) -> list[Path]:
    splits = build_lopo_splits(subjects, windows, cfg)
    base = Path(cfg.data.splits_dir) / "lopo"
    paths = []
    for split in splits:
        path = base / f"test_{split['test_subject']}.json"
        write_json(split, path)
        paths.append(path)
    return paths
