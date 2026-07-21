from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(tn + fp, 1)
    out: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    out["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    out["auprc"] = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return out


def select_binary_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    objective: str = "accuracy",
    default: float = 0.5,
) -> float:
    """Select a decision threshold using validation predictions only.

    Ties are resolved toward ``default`` so calibration does not introduce an
    unnecessary threshold shift. Degenerate one-class validation sets also use
    the configured default.
    """
    if objective != "accuracy":
        raise ValueError("Threshold selection currently supports only accuracy")
    if not 0.0 <= default <= 1.0:
        raise ValueError("Default threshold must lie in [0,1]")

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float(default)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have identical shapes")
    if not np.isfinite(y_prob).all():
        raise ValueError("Validation probabilities must be finite")

    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    positives = float(np.sum(y_true == 1))
    negatives = float(np.sum(y_true == 0))
    scores = (tpr * positives + (1.0 - fpr) * negatives) / (positives + negatives)

    candidates: list[tuple[float, float]] = [
        (float(default), float(accuracy_score(y_true, y_prob >= default))),
        (0.0, float(accuracy_score(y_true, y_prob >= 0.0))),
        (1.0, float(accuracy_score(y_true, y_prob >= 1.0))),
    ]
    candidates.extend(
        (float(threshold), float(score))
        for threshold, score in zip(thresholds, scores)
        if np.isfinite(threshold) and 0.0 <= threshold <= 1.0
    )
    best_score = max(score for _, score in candidates)
    tied = [threshold for threshold, score in candidates if np.isclose(score, best_score, rtol=0.0, atol=1e-12)]
    return float(min(tied, key=lambda threshold: (abs(threshold - default), threshold)))


def parameter_count(model: object) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))  # type: ignore[attr-defined]
