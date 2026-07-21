from __future__ import annotations

from pathlib import Path

import torch
import yaml
import pytest

from src.data.chbmit_reader import get_channel_set
from src.models.tcn_mse_margat import TCNMSEMARGAT
from src.utils.config import load_config


CHANNELS = get_channel_set("common_18")


def test_full_model_contract_and_gradient() -> None:
    cfg = load_config("configs/default.yaml", "configs/model_tcn_mse_margat.yaml")
    model = TCNMSEMARGAT(cfg, num_channels=18, channel_names=CHANNELS)
    x = torch.randn(2, 18, 256, requires_grad=True)
    logits, aux = model(x)
    assert logits.shape == (2, 2)
    assert aux["graph_attention"].shape[:3] == (2, 2, 4)
    assert aux["scale_weights"].shape == (2, 18, 3)
    assert aux["residual_gate"].mean().item() == pytest.approx(0.02, abs=1e-6)
    assert len(aux["expert_logits"]) == 3
    loss = logits.square().mean() + sum(item.square().mean() for item in aux["expert_logits"])
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(x.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_every_ablation_configuration_has_a_real_forward_path() -> None:
    manifest = yaml.safe_load(Path("configs/ablations/suite.yaml").read_text(encoding="utf-8"))
    x = torch.randn(1, 18, 64)
    for experiment in manifest["experiments"]:
        paths = ["configs/default.yaml", "configs/model_tcn_mse_margat.yaml"]
        if experiment["config"]:
            paths.append(experiment["config"])
        model = TCNMSEMARGAT(load_config(*paths), num_channels=18, channel_names=CHANNELS)
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, 2), experiment["name"]


def test_progressive_ablation_is_structurally_clean() -> None:
    tcn_only = TCNMSEMARGAT(
        load_config("configs/default.yaml", "configs/model_tcn_mse_margat.yaml", "configs/ablations/tcn_only.yaml"),
        18,
        CHANNELS,
    )
    assert tcn_only.tcn is not None
    assert tcn_only.mse is None and tcn_only.margat is None and tcn_only.fusion is None

    no_graph = TCNMSEMARGAT(
        load_config("configs/default.yaml", "configs/model_tcn_mse_margat.yaml", "configs/ablations/tcn_mse_no_graph.yaml"),
        18,
        CHANNELS,
    )
    assert no_graph.tcn is not None and no_graph.mse is not None
    assert no_graph.margat is None and no_graph.fusion is not None


@pytest.mark.parametrize(
    ("config_path", "samples", "levels"),
    [
        ("configs/model_tcn_mse_margat_v2_win1s.yaml", 256, 4),
        ("configs/model_tcn_mse_margat_v2_win2s.yaml", 512, 5),
        ("configs/model_tcn_mse_margat_v2_win4s.yaml", 1024, 6),
    ],
)
def test_v2_window_specific_models_route_all_expert_logits(config_path, samples, levels) -> None:
    cfg = load_config("configs/default.yaml", config_path)
    model = TCNMSEMARGAT(cfg, num_channels=18, channel_names=CHANNELS)
    assert len(model.tcn.blocks) == levels
    assert model.logit_fusion is not None
    with torch.no_grad():
        logits, aux = model(torch.randn(2, 18, samples))
    assert logits.shape == (2, 2)
    assert aux["evidence_weights"].shape == (2, 4)
    assert aux["evidence_weights"].sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
    assert aux["evidence_weights"].mean(dim=0).tolist() == pytest.approx([0.55, 0.25, 0.15, 0.05])
