from __future__ import annotations

import torch

from src.models.feature_extractors.multi_scale_encoder import AdaptiveMultiScaleEncoder


def test_multi_scale_encoder_shapes_and_normalized_scale_weights() -> None:
    encoder = AdaptiveMultiScaleEncoder(stem_channels=8, node_dim=16, kernel_sizes=[3, 7, 15], dilations=[1, 1, 2])
    nodes, sequence, weights = encoder(torch.randn(2, 4, 128))
    assert nodes.shape == (2, 4, 16)
    assert sequence.shape[:3] == (2, 4, 16)
    assert weights.shape == (2, 4, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_uniform_scale_and_mean_time_controls() -> None:
    encoder = AdaptiveMultiScaleEncoder(
        stem_channels=8,
        node_dim=16,
        kernel_sizes=[3, 7],
        dilations=[1, 1],
        use_scale_attention=False,
        use_temporal_attention=False,
    )
    _, _, weights = encoder(torch.randn(1, 3, 64))
    assert torch.allclose(weights, torch.full((1, 3, 2), 0.5))
