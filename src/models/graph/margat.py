from __future__ import annotations

import math

import torch
from torch import nn

from src.models.graph.montage_graph import build_signed_line_graph


def absolute_pearson(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
    unit = centered / norm.clamp_min(eps)
    correlation = torch.matmul(unit, unit.transpose(-1, -2)).clamp(-1.0, 1.0).abs()
    valid = (norm > eps) & (norm.transpose(-1, -2) > eps)
    return torch.where(valid, correlation, torch.zeros_like(correlation))


class MontageAwareResidualGATLayer(nn.Module):
    """Dense dynamic attention regularized by a signed bipolar line graph."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        structural_bias_init: float,
        non_structural_gate_init: float,
        use_dynamic_edges: bool,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if not 0.0 < non_structural_gate_init < 1.0:
            raise ValueError("non_structural_gate_init must lie in (0,1)")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_dynamic_edges = bool(use_dynamic_edges)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.gq = nn.Linear(dim, num_heads)
        self.gk = nn.Linear(dim, num_heads)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        # relation-index order: opposite orientation, non-structural, same orientation, self
        self.type_bias = nn.Parameter(
            torch.tensor(
                [structural_bias_init, 0.0, structural_bias_init, structural_bias_init + 0.5]
            ).repeat(num_heads, 1)
        )
        self.correlation_scale_raw = nn.Parameter(torch.zeros(num_heads))
        gate_logit = math.log(non_structural_gate_init / (1.0 - non_structural_gate_init))
        self.non_structural_gate_bias = nn.Parameter(torch.full((num_heads,), gate_logit))

    def forward(
        self,
        nodes: torch.Tensor,
        correlation: torch.Tensor,
        structural_mask: torch.Tensor,
        relation_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, dim = nodes.shape
        q = self.q(nodes).view(bsz, channels, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(nodes).view(bsz, channels, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(nodes).view(bsz, channels, self.num_heads, self.head_dim).transpose(1, 2)
        content = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        type_bias = self.type_bias[:, relation_index].unsqueeze(0)
        corr_scale = torch.nn.functional.softplus(self.correlation_scale_raw).view(1, self.num_heads, 1, 1)
        scores = content + type_bias + corr_scale * torch.log(correlation[:, None].clamp_min(1e-6))

        gate_query = self.gq(nodes).transpose(1, 2).unsqueeze(-1)
        gate_key = self.gk(nodes).transpose(1, 2).unsqueeze(-2)
        dynamic_gate = torch.sigmoid(gate_query + gate_key + self.non_structural_gate_bias.view(1, -1, 1, 1))
        if not self.use_dynamic_edges:
            dynamic_gate = torch.zeros_like(dynamic_gate)
        gate = torch.where(structural_mask[None, None], torch.ones_like(dynamic_gate), dynamic_gate)
        gated_scores = scores + torch.log(gate.clamp_min(1e-6))
        if not self.use_dynamic_edges:
            gated_scores = gated_scores.masked_fill(~structural_mask[None, None], torch.finfo(scores.dtype).min)
        attention = torch.softmax(gated_scores, dim=-1)
        if not self.use_dynamic_edges:
            attention = attention.masked_fill(~structural_mask[None, None], 0.0)
        message = torch.matmul(self.dropout(attention), v).transpose(1, 2).reshape(bsz, channels, dim)
        output = self.norm(nodes + self.out(message))
        non_structural = (~structural_mask)[None, None].expand_as(gate)
        sparsity = gate.masked_select(non_structural).mean() if non_structural.any() else gate.new_zeros(())
        return output, attention, sparsity


class MontageAwareResidualGAT(nn.Module):
    def __init__(
        self,
        dim: int,
        channel_names: list[str],
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
        structural_bias_init: float = 1.0,
        non_structural_gate_init: float = 0.02,
        use_montage: bool = True,
        use_orientation: bool = True,
        use_dynamic_edges: bool = True,
    ) -> None:
        super().__init__()
        if len(channel_names) < 1:
            raise ValueError("channel_names cannot be empty")
        structural_mask, relation = build_signed_line_graph(channel_names)
        if not use_montage:
            structural_mask = torch.eye(len(channel_names), dtype=torch.bool)
            relation = torch.where(structural_mask, torch.full_like(relation, 2), torch.zeros_like(relation))
        elif not use_orientation:
            relation = torch.where(relation == -1, torch.ones_like(relation), relation)
        self.use_dynamic_edges = bool(use_dynamic_edges)
        self.register_buffer("structural_mask", structural_mask)
        self.register_buffer("relation_index", relation + 1)  # -1,0,1,2 -> 0,1,2,3
        self.layers = nn.ModuleList(
            [
                MontageAwareResidualGATLayer(
                    dim,
                    num_heads,
                    dropout,
                    structural_bias_init,
                    non_structural_gate_init if use_dynamic_edges else 1e-6,
                    use_dynamic_edges,
                )
                for _ in range(num_layers)
            ]
        )
        self.pool_score = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(self, nodes: torch.Tensor, raw_x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        correlation = absolute_pearson(raw_x)
        hidden = nodes
        attentions = []
        sparsity_terms = []
        for layer in self.layers:
            hidden, attention, sparsity = layer(hidden, correlation, self.structural_mask, self.relation_index)
            attentions.append(attention)
            sparsity_terms.append(sparsity)
        pooling_weights = torch.softmax(self.pool_score(hidden), dim=1)
        graph_embedding = torch.sum(pooling_weights * hidden, dim=1)
        return graph_embedding, {
            "graph_nodes": hidden,
            "graph_attention": torch.stack(attentions, dim=1),
            "graph_pooling_weights": pooling_weights.squeeze(-1),
            "dynamic_gate_mean": torch.stack(sparsity_terms).mean(),
            "absolute_pearson": correlation,
        }
