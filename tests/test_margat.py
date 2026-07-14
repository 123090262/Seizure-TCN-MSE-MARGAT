from __future__ import annotations

import pytest
import torch

from src.models.graph.margat import MontageAwareResidualGAT, absolute_pearson
from src.models.graph.montage_graph import bipolar_endpoints, build_signed_line_graph


def test_bipolar_line_graph_preserves_shared_endpoint_orientation() -> None:
    channels = ["FP1-F7", "F7-T7", "FP1-F3", "P3-O1"]
    mask, relation = build_signed_line_graph(channels)
    assert bipolar_endpoints("T3-T5") == ("T7", "P7")
    assert mask[0, 1] and relation[0, 1].item() == -1
    assert mask[0, 2] and relation[0, 2].item() == 1
    assert not mask[0, 3] and relation[0, 3].item() == 0
    assert torch.all(relation.diagonal() == 2)


def test_absolute_pearson_is_finite_for_constant_channels() -> None:
    signal = torch.linspace(-1, 1, 64)
    raw = torch.stack([signal, -signal, torch.ones_like(signal)]).unsqueeze(0)
    correlation = absolute_pearson(raw)
    assert correlation[0, 0, 1].item() == pytest.approx(1.0, abs=1e-5)
    assert correlation[0, 0, 2].item() == 0.0
    assert torch.isfinite(correlation).all()


def test_margat_attention_and_structural_only_mask() -> None:
    channels = ["FP1-F7", "F7-T7", "FP1-F3", "P3-O1"]
    nodes = torch.randn(2, 4, 16, requires_grad=True)
    raw = torch.randn(2, 4, 64)
    model = MontageAwareResidualGAT(16, channels, num_heads=4, num_layers=1, use_dynamic_edges=False)
    embedding, aux = model(nodes, raw)
    attention = aux["graph_attention"][:, 0]
    assert embedding.shape == (2, 16)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 4, 4), atol=1e-6)
    non_edges = ~model.structural_mask
    assert torch.count_nonzero(attention.masked_select(non_edges[None, None])) == 0
    embedding.square().mean().backward()
    assert torch.isfinite(nodes.grad).all()


def test_structural_bias_initialization_does_not_favor_non_edges() -> None:
    model = MontageAwareResidualGAT(16, ["FP1-F7", "F7-T7"], num_heads=4, num_layers=1)
    bias = model.layers[0].type_bias.detach()
    # relation indices: opposite, non-structural, same, self
    assert torch.all(bias[:, 1] == 0)
    assert torch.all(bias[:, 0] > bias[:, 1])
    assert torch.all(bias[:, 2] > bias[:, 1])
