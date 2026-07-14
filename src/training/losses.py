from __future__ import annotations

import torch
from torch import nn


def inverse_frequency_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Training-only inverse-frequency weights, normalized over present classes."""
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if not labels:
        raise ValueError("Cannot compute class weights from an empty training set")
    target = torch.tensor(labels, dtype=torch.long)
    if (target < 0).any() or (target >= num_classes).any():
        raise ValueError(f"Training labels must be in [0, {num_classes - 1}]")
    counts = torch.bincount(target, minlength=num_classes).float()
    present = counts > 0
    weights = torch.zeros(num_classes, dtype=torch.float32)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights


def build_loss(class_weight: torch.Tensor | None = None, name: str = "cross_entropy") -> nn.Module:
    if str(name).lower() != "cross_entropy":
        raise ValueError("training.loss.name currently supports only 'cross_entropy'")
    return nn.CrossEntropyLoss(weight=class_weight)
