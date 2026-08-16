import numpy as np
import pytest

from src.evaluate import best_f1_threshold, binary_metrics, select_threshold


def test_metrics_include_imbalanced_class_measures() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = binary_metrics(labels, probabilities, threshold=0.5)

    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == 2 / 3
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_validation_threshold_maximizes_f1() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])

    threshold = best_f1_threshold(labels, probabilities)

    assert threshold == 0.45


def test_metrics_include_balanced_accuracy() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = binary_metrics(labels, probabilities, threshold=0.5)

    assert metrics["balanced_accuracy"] == 0.75


def test_balanced_accuracy_threshold_uses_highest_threshold_on_tie() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.4, 0.9])

    threshold = select_threshold(labels, probabilities, "balanced_accuracy")

    assert threshold == 0.9


def test_threshold_selection_rejects_unknown_metric() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])

    with pytest.raises(ValueError, match="Unsupported threshold metric"):
        select_threshold(labels, probabilities, "accuracy")
