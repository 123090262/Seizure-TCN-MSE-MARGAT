from __future__ import annotations

import torch
from torch import nn


class MultiScaleBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveMultiScaleEncoder(nn.Module):
    """Channel-independent multi-scale encoder with sample-wise scale routing.

    Input: ``[B,C,T]``. Output node features ``[B,C,D]`` and per-channel
    scale weights ``[B,C,S]``. A strided stem keeps the parallel branches
    affordable on four-second windows.
    """

    def __init__(
        self,
        stem_channels: int = 32,
        node_dim: int = 128,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
        stem_stride: int = 4,
        dropout: float = 0.15,
        use_scale_attention: bool = True,
        use_temporal_attention: bool = True,
    ) -> None:
        super().__init__()
        kernels = kernel_sizes or [7, 15, 31]
        dilation_values = dilations or [1, 2, 2]
        if len(kernels) != len(dilation_values) or not kernels:
            raise ValueError("kernel_sizes and dilations must have equal non-zero length")
        if stem_stride < 1:
            raise ValueError("stem_stride must be positive")
        self.use_scale_attention = bool(use_scale_attention)
        self.use_temporal_attention = bool(use_temporal_attention)
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=7, stride=stem_stride, padding=3),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [MultiScaleBranch(stem_channels, kernel, dilation, dropout) for kernel, dilation in zip(kernels, dilation_values)]
        )
        self.scale_score = nn.Sequential(
            nn.Linear(stem_channels, max(stem_channels // 2, 1)),
            nn.GELU(),
            nn.Linear(max(stem_channels // 2, 1), 1),
        )
        self.projection = nn.Sequential(
            nn.Conv1d(stem_channels, node_dim, kernel_size=1),
            nn.BatchNorm1d(node_dim),
            nn.GELU(),
        )
        self.temporal_score = nn.Conv1d(node_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, time = x.shape
        stem = self.stem(x.reshape(bsz * channels, 1, time))
        branch_features = torch.stack([branch(stem) for branch in self.branches], dim=1)
        descriptors = branch_features.mean(dim=-1)
        if self.use_scale_attention:
            scale_weights = torch.softmax(self.scale_score(descriptors).squeeze(-1), dim=1)
        else:
            scale_weights = torch.full(
                descriptors.shape[:2],
                1.0 / len(self.branches),
                device=x.device,
                dtype=x.dtype,
            )
        fused = torch.sum(scale_weights[..., None, None] * branch_features, dim=1)
        sequence = self.projection(fused)
        if self.use_temporal_attention:
            temporal_weights = torch.softmax(self.temporal_score(sequence), dim=-1)
            pooled = torch.sum(temporal_weights * sequence, dim=-1)
        else:
            pooled = sequence.mean(dim=-1)
        nodes = pooled.reshape(bsz, channels, -1)
        return nodes, sequence.reshape(bsz, channels, sequence.shape[-2], sequence.shape[-1]), scale_weights.reshape(bsz, channels, -1)
