from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from src.data.datamodule import build_dataloaders, resolve_balance_strategy
from src.models.tcn_mse_margat import TCNMSEMARGAT
from src.training.losses import build_loss, inverse_frequency_class_weights
from src.training.trainer import Trainer
from src.utils.config import load_config


def test_tcn_mse_margat_uses_fp32_training_for_graph_attention_stability() -> None:
    cfg = load_config("configs/exp_mixed_5fold.yaml", "configs/model_tcn_mse_margat.yaml")

    assert cfg.model.name == "tcn_mse_margat"
    assert cfg.training.amp is False


def _empty_split_with_train_labels(labels: list[int]) -> dict:
    windows = [{"label": label} for label in labels]
    return {"train_windows": windows, "val_windows": [], "test_windows": []}


@pytest.mark.parametrize("strategy", ["sampler", "class_weight", "none"])
def test_balance_strategy_selects_exactly_one_mechanism(strategy) -> None:
    cfg = load_config("configs/default.yaml", "configs/model_tcn_mse_margat.yaml")
    cfg.training.balance_strategy = strategy
    loaders = build_dataloaders(_empty_split_with_train_labels([0, 0, 0, 1]), cfg, normalizer=object())
    if strategy == "sampler":
        assert isinstance(loaders["train"].sampler, WeightedRandomSampler)
    else:
        assert isinstance(loaders["train"].sampler, RandomSampler)
    assert resolve_balance_strategy(cfg) == strategy


def test_class_weights_are_training_only_multiclass_and_missing_safe() -> None:
    weights = inverse_frequency_class_weights([0, 0, 1], num_classes=3)
    assert weights.shape == (3,)
    assert weights.tolist() == pytest.approx([0.75, 1.5, 0.0])
    assert torch.isfinite(weights).all()
    assert inverse_frequency_class_weights([2, 2], 3).tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="empty training"):
        inverse_frequency_class_weights([], 2)
    with pytest.raises(ValueError, match="cross_entropy"):
        build_loss(name="silently_ignored_loss")


def test_legacy_class_weight_migration_and_conflict() -> None:
    cfg = load_config("configs/default.yaml", "configs/model_tcn_mse_margat.yaml")
    del cfg.training["balance_strategy"]
    cfg.training.loss.class_weight = "auto"
    assert resolve_balance_strategy(cfg) == "class_weight"
    cfg.training.balance_strategy = "sampler"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_balance_strategy(cfg)


def test_model_yaml_has_no_ignored_placeholder_options() -> None:
    cfg = load_config("configs/model_tcn_mse_margat.yaml")
    assert set(cfg.model) == {
        "name", "num_classes", "hidden_dim", "tcn", "mse", "margat",
        "fusion", "classifier", "objectives",
    }
    assert cfg.model.margat.use_montage is True
    assert cfg.model.margat.use_orientation is True


def test_margat_requires_mse_nodes() -> None:
    cfg = load_config("configs/default.yaml", "configs/model_tcn_mse_margat.yaml")
    cfg.model.mse.enabled = False
    with pytest.raises(ValueError, match="requires the multi-scale"):
        TCNMSEMARGAT(cfg, 18, [f"A{i}-B{i}" for i in range(18)])


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        self.windows = [{"label": 0}, {"label": 0}, {"label": 1}, {"label": 1}]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": torch.randn(2, 4), "y": torch.tensor(self.windows[index]["label"])}


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor):
        return self.linear(x.flatten(1)), {}


def test_training_report_records_protocol_and_resolved_class_weights(tmp_path) -> None:
    cfg = load_config("configs/default.yaml", "configs/model_tcn_mse_margat.yaml")
    cfg.training.epochs = 1
    cfg.training.batch_size = 2
    cfg.training.amp = False
    cfg.training.scheduler = "none"
    cfg.training.balance_strategy = "class_weight"
    cfg.training.early_stopping.monitor = "val_loss"
    cfg.training.early_stopping.mode = "min"
    loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)
    trainer = Trainer(_TinyModel(), cfg, torch.device("cpu"), tmp_path)
    report = trainer.fit({"train": loader, "val": loader, "test": loader})
    assert report["optimizer_steps"] == 2
    assert report["effective_batch_size"] == 2
    assert report["early_stopping_monitor"] == "val_loss"
    assert report["balance_strategy"] == "class_weight"
    assert report["class_weights"] == pytest.approx([1.0, 1.0])
    assert "graph_fusion_weights" in report and "learnable_adjacency" in report
