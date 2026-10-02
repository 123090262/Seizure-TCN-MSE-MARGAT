"""Compact PyTorch implementation of TCN-MSE-MARGAT."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn


def lead_graph(channels: Sequence[str]) -> tuple[Tensor, Tensor]:
    """Return shared-electrode adjacency and bipolar orientation matrices."""
    parsed = [channel.upper().replace("–", "-").split("-") for channel in channels]
    if any(len(pair) != 2 for pair in parsed):
        raise ValueError("Every channel must be a bipolar pair such as FP1-F7")
    size = len(channels)
    adjacency = torch.eye(size)
    orientation = torch.eye(size)
    for i, first in enumerate(parsed):
        signs_first = {first[0]: 1.0, first[1]: -1.0}
        for j, second in enumerate(parsed):
            shared = set(first) & set(second)
            if shared:
                electrode = next(iter(shared))
                signs_second = {second[0]: 1.0, second[1]: -1.0}
                adjacency[i, j] = 1.0
                orientation[i, j] = signs_first[electrode] * signs_second[electrode]
    return adjacency, orientation


class MultiScaleTemporalBlock(nn.Module):
    """Residual multi-scale full-convolution temporal block."""

    def __init__(
        self,
        hidden: int,
        dilation: int,
        dropout: float,
        kernel_sizes: Sequence[int] = (3, 7, 15),
    ):
        super().__init__()
        branch_dim = max(16, hidden * 3 // 8)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden,
                        branch_dim,
                        kernel_size,
                        padding=dilation * (kernel_size - 1) // 2,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(branch_dim),
                    nn.GELU(),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(branch_dim * len(kernel_sizes), hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        multi_scale = torch.cat([branch(inputs) for branch in self.branches], dim=1)
        return inputs + self.fusion(multi_scale)


class SingleHeadTemporalAttentionPooling(nn.Module):
    """Pool a temporal feature map with one interpretable attention map."""

    def __init__(self, hidden: int):
        super().__init__()
        self.attention = nn.Conv1d(hidden, 1, 1)
        self.last_temporal_attention: Tensor | None = None

    def forward(self, features: Tensor) -> Tensor:
        attention_weights = self.attention(features).softmax(dim=-1)  # [B, 1, T]
        self.last_temporal_attention = attention_weights.detach()
        return (features * attention_weights).sum(dim=-1)


class TCNBranch(nn.Module):
    """Multi-scale full-convolution TCN with single-head attention pooling."""

    def __init__(
        self,
        channels: Sequence[str],
        hidden: int,
        levels: int,
        dropout: float,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(len(channels), hidden, 7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                MultiScaleTemporalBlock(hidden, 2**level, dropout)
                for level in range(levels)
            ]
        )
        self.pooling = SingleHeadTemporalAttentionPooling(hidden)

    @property
    def last_temporal_attention(self) -> Tensor | None:
        """Return detached attention weights from the most recent forward pass."""
        return self.pooling.last_temporal_attention

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.blocks(self.input_projection(inputs))  # [B, H, T]
        return self.pooling(features)


class MultiScaleEncoder(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, node_dim, kernel, stride=4, padding=kernel // 2),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for kernel in (7, 15, 31)
            ]
        )
        self.scale_score = nn.Linear(node_dim, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, samples = inputs.shape
        per_channel = inputs.reshape(batch * channels, 1, samples)
        scales = torch.stack(
            [branch(per_channel).squeeze(-1) for branch in self.branches], dim=1
        )
        weights = self.scale_score(scales).softmax(dim=1)
        nodes = (scales * weights).sum(dim=1)
        return nodes.reshape(batch, channels, -1)  # [B, C, node_dim]


class LeadGraphAttention(nn.Module):
    def __init__(
        self, channels: Sequence[str], node_dim: int, heads: int, dropout: float
    ):
        super().__init__()
        if node_dim % heads:
            raise ValueError("node_dim must be divisible by heads")
        adjacency, orientation = lead_graph(channels)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("orientation", orientation)
        self.heads = heads
        self.head_dim = node_dim // heads
        self.qkv = nn.Linear(node_dim, node_dim * 3)
        self.output = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.pool_score = nn.Linear(node_dim, 1)
        self.structure_strength = nn.Parameter(torch.tensor(1.0))
        self.orientation_strength = nn.Parameter(torch.tensor(0.5))
        self.dynamic_strength = nn.Parameter(torch.tensor(1.0))
        self.nonstructure_gate = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, signal: Tensor) -> Tensor:
        batch, channels, node_dim = nodes.shape
        qkv = self.qkv(nodes).reshape(batch, channels, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        logits = torch.einsum("bihd,bjhd->bhij", query, key) / math.sqrt(self.head_dim)

        centered = signal - signal.mean(dim=-1, keepdim=True)
        normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        correlation = torch.einsum("bit,bjt->bij", normalized, normalized).abs()
        prior = (
            self.structure_strength * self.adjacency
            + self.orientation_strength * self.orientation
        )
        gate = torch.log(torch.sigmoid(self.nonstructure_gate).clamp_min(1e-6))
        prior = prior + (1.0 - self.adjacency) * gate
        logits = (
            logits + prior[None, None] + self.dynamic_strength * correlation[:, None]
        )
        attention = self.dropout(logits.softmax(dim=-1))
        updated = torch.einsum("bhij,bjhd->bihd", attention, value).reshape(
            batch, channels, node_dim
        )
        nodes = self.norm(nodes + self.output(updated))
        pool = self.pool_score(nodes).softmax(dim=1)
        return (nodes * pool).sum(dim=1)


class TCNMSEMARGAT(nn.Module):
    """Fuse temporal, multi-scale, and lead-graph EEG representations."""

    def __init__(
        self,
        channels: Sequence[str],
        hidden: int = 128,
        node_dim: int = 128,
        tcn_levels: int = 5,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.channel_count = len(channels)
        self.tcn = TCNBranch(channels, hidden, tcn_levels, dropout)
        self.mse = MultiScaleEncoder(node_dim)
        self.graph = LeadGraphAttention(channels, node_dim, heads, dropout)
        self.correction = nn.Linear(node_dim * 2, hidden)
        self.gate = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != self.channel_count:
            raise ValueError(
                f"Expected [batch, {self.channel_count} channels, time], got {tuple(inputs.shape)}"
            )
        temporal = self.tcn(inputs)
        nodes = self.mse(inputs)
        graph = self.graph(nodes, inputs)
        correction = self.correction(torch.cat([nodes.mean(dim=1), graph], dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat([temporal, correction], dim=-1)))
        fused = self.norm(temporal + gate * correction)
        return self.classifier(fused)
