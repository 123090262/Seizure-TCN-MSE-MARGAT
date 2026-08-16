import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import probability_summary, seed_everything, selection_score, train_epoch


def test_seed_everything_repeats_numpy_and_torch_draws() -> None:
    seed_everything(17)
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)

    seed_everything(17)

    assert np.array_equal(np.random.rand(3), first_numpy)
    assert torch.equal(torch.rand(3), first_torch)


def test_train_epoch_runs_one_cpu_batch() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 2))
    loader = DataLoader(
        TensorDataset(torch.randn(4, 2, 4), torch.tensor([0, 1, 0, 1])),
        batch_size=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    loss = train_epoch(model, loader, optimizer, scaler, torch.device("cpu"), False)

    assert np.isfinite(loss)
    assert loss > 0


def test_probability_summary_reports_each_class_without_raw_predictions() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

    summary = probability_summary(labels, probabilities)

    assert summary["negative"]["min"] == 0.1
    assert summary["negative"]["median"] == 0.2
    assert summary["negative"]["max"] == 0.3
    assert summary["positive"]["min"] == 0.6
    assert summary["positive"]["median"] == 0.8
    assert summary["positive"]["max"] == 0.9
    assert "probabilities" not in summary


def test_selection_score_uses_configured_metric() -> None:
    metrics = {"f1": 0.81, "balanced_accuracy": 0.87}

    assert selection_score(metrics, "f1") == 0.81
    assert selection_score(metrics, "balanced_accuracy") == 0.87


def test_selection_score_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported selection metric"):
        selection_score({"f1": 0.81}, "accuracy")
