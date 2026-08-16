"""Evaluate a saved checkpoint with seizure-relevant binary metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.data import create_dataloaders
from src.model import TCNMSEMARGAT


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1",
) -> float:
    """Select a deterministic decision threshold using validation data only."""
    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        if len(thresholds) == 0:
            return 0.5
        scores = (
            2
            * precision[:-1]
            * recall[:-1]
            / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        )
        return float(thresholds[int(np.argmax(scores))])
    if metric == "balanced_accuracy":
        thresholds = np.unique(probabilities)
        scores = []
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(np.int64)
            tn, fp, fn, tp = confusion_matrix(
                labels, predictions, labels=[0, 1]
            ).ravel()
            sensitivity = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            scores.append((sensitivity + specificity) / 2)
        best = np.flatnonzero(np.isclose(scores, np.max(scores)))
        return float(thresholds[int(best[-1])])
    raise ValueError(f"Unsupported threshold metric: {metric}")


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Preserve the original public F1 threshold API."""
    return select_threshold(labels, probabilities, "f1")


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    two_classes = len(np.unique(labels)) == 2
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": (
            float(roc_auc_score(labels, probabilities)) if two_classes else float("nan")
        ),
        "auprc": (
            float(average_precision_score(labels, probabilities))
            if two_classes
            else float("nan")
        ),
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    for inputs, targets in loader:
        logits = model(inputs.to(device, non_blocking=True))
        probabilities.append(logits.softmax(dim=-1)[:, 1].cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def build_model(config: dict[str, Any]) -> TCNMSEMARGAT:
    model = config["model"]
    return TCNMSEMARGAT(
        config["data"]["channels"],
        hidden=model["hidden"],
        node_dim=model["node_dim"],
        tcn_levels=model["tcn_levels"],
        heads=model["heads"],
        dropout=model["dropout"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TCN-MSE-MARGAT")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    requested = args.device or config["train"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --device cpu only for checks"
        )
    device = torch.device(requested)
    loaders = create_dataloaders(config, checkpoint["split_id"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    labels, probabilities = predict(model, loaders["test"], device)
    metrics = binary_metrics(labels, probabilities, checkpoint["threshold"])
    output = args.checkpoint.parent / "test_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
