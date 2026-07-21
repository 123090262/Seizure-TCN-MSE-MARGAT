from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.baselines.tcn_baseline import TemporalBlock
from src.models.feature_extractors.multi_scale_encoder import AdaptiveMultiScaleEncoder
from src.models.graph.margat import MontageAwareResidualGAT


class TCNFeatureBackbone(nn.Module):
    """The same temporal block family as the TCN baseline, exposed as features."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        levels: int,
        kernel_size: int,
        dropout: float,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        self.pooling = pooling.lower()
        if self.pooling not in {"attention", "attention_stats"}:
            raise ValueError("model.tcn.pooling must be attention or attention_stats")
        self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(hidden_dim, kernel_size, 2**level, dropout) for level in range(levels)]
        )
        self.pool_score = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.pool_fusion = (
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if self.pooling == "attention_stats"
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.blocks(self.proj(x))
        weights = torch.softmax(self.pool_score(sequence), dim=-1)
        attended = torch.sum(weights * sequence, dim=-1)
        if self.pool_fusion is None:
            return attended, sequence
        mean = sequence.mean(dim=-1)
        std = torch.sqrt(sequence.var(dim=-1, unbiased=False) + 1.0e-6)
        return self.pool_fusion(torch.cat([attended, mean, std], dim=-1)), sequence


class SafeResidualExpertFusion(nn.Module):
    """TCN-preserving residual adapter with a low initial graph/scale gate."""

    def __init__(self, dim: int, dropout: float, gate_init: float = 0.02) -> None:
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must lie in (0,1)")
        self.delta = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Linear(dim * 2, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, torch.logit(torch.tensor(gate_init)).item())
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tcn: torch.Tensor, scale: torch.Tensor, graph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        correction = self.delta(torch.cat([scale, graph, scale * graph], dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat([tcn, correction], dim=-1)))
        return self.norm(tcn + gate * correction), gate, correction


class DirectConcatFusion(nn.Module):
    """Unconstrained fusion control for the safe residual adapter ablation."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )

    def forward(self, tcn: torch.Tensor, scale: torch.Tensor, graph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self.net(torch.cat([tcn, scale, graph], dim=-1))
        return fused, torch.ones_like(fused), fused - tcn


class EvidenceLogitFusion(nn.Module):
    """Validation-friendly mixture of the fused classifier and expert heads.

    The old architecture trained expert heads only through an auxiliary loss.
    This layer lets their evidence directly affect the final prediction while a
    zero-initialized router starts from an explicit, conservative prior.
    """

    def __init__(self, dim: int, prior_weights: list[float], dropout: float) -> None:
        super().__init__()
        if len(prior_weights) < 2 or any(weight <= 0 for weight in prior_weights):
            raise ValueError("fusion.logit_fusion.prior_weights must contain at least two positive values")
        prior = torch.tensor(prior_weights, dtype=torch.float32)
        prior = prior / prior.sum()
        self.register_buffer("log_prior", torch.log(prior))
        self.router = nn.Sequential(
            nn.LayerNorm(dim * len(prior_weights)),
            nn.Dropout(dropout),
            nn.Linear(dim * len(prior_weights), len(prior_weights)),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.log_temperatures = nn.Parameter(torch.zeros(len(prior_weights)))

    def forward(
        self,
        embeddings: list[torch.Tensor],
        logits: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(embeddings) != len(logits) or len(logits) != self.log_prior.numel():
            raise ValueError("Evidence fusion inputs do not match configured expert prior")
        route_delta = self.router(torch.cat(embeddings, dim=-1))
        weights = torch.softmax(self.log_prior + route_delta, dim=-1)
        temperatures = self.log_temperatures.exp().clamp(0.5, 2.0)
        stacked = torch.stack(logits, dim=1) / temperatures[None, :, None]
        return torch.sum(weights[..., None] * stacked, dim=1), weights

    def prior_weights(self) -> list[float]:
        return torch.softmax(self.log_prior, dim=0).detach().cpu().tolist()

    def temperatures(self) -> list[float]:
        return self.log_temperatures.exp().clamp(0.5, 2.0).detach().cpu().tolist()


class TCNMSEMARGAT(nn.Module):
    """TCN + adaptive multi-scale encoder + montage-aware residual GAT."""

    def __init__(self, cfg: Any, num_channels: int, channel_names: list[str]) -> None:
        super().__init__()
        mcfg = cfg.model
        dim = int(mcfg.hidden_dim)
        self.tcn_enabled = bool(mcfg.tcn.enabled)
        self.mse_enabled = bool(mcfg.mse.enabled)
        self.margat_enabled = bool(mcfg.margat.enabled)
        if self.margat_enabled and not self.mse_enabled:
            raise ValueError("MARGAT requires the multi-scale node encoder")
        if not self.tcn_enabled and not self.mse_enabled:
            raise ValueError("At least one of TCN or MSE must be enabled")

        self.tcn = TCNFeatureBackbone(
            num_channels,
            dim,
            int(mcfg.tcn.levels),
            int(mcfg.tcn.kernel_size),
            float(mcfg.tcn.dropout),
            str(mcfg.tcn.get("pooling", "attention")),
        ) if self.tcn_enabled else None
        self.mse = AdaptiveMultiScaleEncoder(
            stem_channels=int(mcfg.mse.stem_channels),
            node_dim=dim,
            kernel_sizes=list(mcfg.mse.kernel_sizes),
            dilations=list(mcfg.mse.dilations),
            stem_stride=int(mcfg.mse.stem_stride),
            dropout=float(mcfg.mse.dropout),
            use_scale_attention=bool(mcfg.mse.use_scale_attention),
            use_temporal_attention=bool(mcfg.mse.use_temporal_attention),
        ) if self.mse_enabled else None
        self.scale_pool = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)) if self.mse_enabled else None
        self.margat = MontageAwareResidualGAT(
            dim=dim,
            channel_names=channel_names,
            num_heads=int(mcfg.margat.num_heads),
            num_layers=int(mcfg.margat.num_layers),
            dropout=float(mcfg.margat.dropout),
            structural_bias_init=float(mcfg.margat.structural_bias_init),
            non_structural_gate_init=float(mcfg.margat.non_structural_gate_init),
            use_montage=bool(mcfg.margat.use_montage),
            use_orientation=bool(mcfg.margat.use_orientation),
            use_dynamic_edges=bool(mcfg.margat.use_dynamic_edges),
        ) if self.margat_enabled else None
        fusion_mode = str(mcfg.fusion.get("mode", "safe_residual")).lower()
        if self.tcn_enabled and self.mse_enabled:
            if fusion_mode == "safe_residual":
                self.fusion = SafeResidualExpertFusion(dim, float(mcfg.fusion.dropout), float(mcfg.fusion.gate_init))
            elif fusion_mode == "concat":
                self.fusion = DirectConcatFusion(dim, float(mcfg.fusion.dropout))
            else:
                raise ValueError("model.fusion.mode must be 'safe_residual' or 'concat'")
        else:
            self.fusion = None
        self.classifier = nn.Sequential(nn.Dropout(float(mcfg.classifier.dropout)), nn.Linear(dim, int(mcfg.num_classes)))
        self.tcn_expert = nn.Linear(dim, int(mcfg.num_classes)) if self.tcn_enabled else None
        self.scale_expert = nn.Linear(dim, int(mcfg.num_classes)) if self.mse_enabled else None
        self.graph_expert = nn.Linear(dim, int(mcfg.num_classes)) if self.margat_enabled else None
        logit_fusion_cfg = mcfg.fusion.get("logit_fusion", {})
        logit_fusion_enabled = bool(logit_fusion_cfg.get("enabled", False)) if hasattr(logit_fusion_cfg, "get") else False
        expert_count = 1 + int(self.tcn_enabled) + int(self.mse_enabled) + int(self.margat_enabled)
        self.logit_fusion = (
            EvidenceLogitFusion(
                dim,
                list(logit_fusion_cfg.get("prior_weights", [])),
                float(logit_fusion_cfg.get("dropout", mcfg.fusion.dropout)),
            )
            if logit_fusion_enabled
            else None
        )
        if self.logit_fusion is not None and self.logit_fusion.log_prior.numel() != expert_count:
            raise ValueError(
                f"fusion.logit_fusion.prior_weights has {self.logit_fusion.log_prior.numel()} entries; "
                f"the enabled fused/expert paths require {expert_count}"
            )
        self.graph_sparsity_weight = float(mcfg.objectives.graph_sparsity_weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        aux: dict[str, Any] = {}
        tcn_embedding = None
        if self.tcn is not None:
            tcn_embedding, tcn_sequence = self.tcn(x)
            aux["tcn_sequence"] = tcn_sequence

        scale_embedding = None
        graph_embedding = None
        if self.mse is not None:
            nodes, mse_sequence, scale_weights = self.mse(x)
            node_weights = torch.softmax(self.scale_pool(nodes), dim=1) if self.scale_pool is not None else None
            assert node_weights is not None
            scale_embedding = torch.sum(node_weights * nodes, dim=1)
            aux.update({"mse_nodes": nodes, "mse_sequence": mse_sequence, "scale_weights": scale_weights, "scale_node_weights": node_weights.squeeze(-1)})
            if self.margat is not None:
                graph_embedding, graph_aux = self.margat(nodes, x)
                aux.update(graph_aux)
            else:
                graph_embedding = scale_embedding

        if tcn_embedding is not None and scale_embedding is not None:
            assert graph_embedding is not None and self.fusion is not None
            fused, residual_gate, correction = self.fusion(tcn_embedding, scale_embedding, graph_embedding)
            aux.update({"residual_gate": residual_gate, "residual_correction": correction})
        elif tcn_embedding is not None:
            fused = tcn_embedding
        else:
            assert graph_embedding is not None
            fused = graph_embedding

        expert_logits = []
        expert_embeddings = []
        if self.tcn_expert is not None and tcn_embedding is not None:
            expert_logits.append(self.tcn_expert(tcn_embedding))
            expert_embeddings.append(tcn_embedding)
        if self.scale_expert is not None and scale_embedding is not None:
            expert_logits.append(self.scale_expert(scale_embedding))
            expert_embeddings.append(scale_embedding)
        if self.graph_expert is not None and graph_embedding is not None:
            expert_logits.append(self.graph_expert(graph_embedding))
            expert_embeddings.append(graph_embedding)
        aux["expert_logits"] = expert_logits
        if "dynamic_gate_mean" in aux:
            aux["regularization_loss"] = self.graph_sparsity_weight * aux["dynamic_gate_mean"]
        fused_logits = self.classifier(fused)
        if self.logit_fusion is not None:
            fused_logits, evidence_weights = self.logit_fusion(
                [fused, *expert_embeddings],
                [fused_logits, *expert_logits],
            )
            aux["evidence_weights"] = evidence_weights
        return fused_logits, aux

    def diagnostics(self) -> dict[str, Any]:
        report = {
            "tcn_enabled": self.tcn_enabled,
            "mse_enabled": self.mse_enabled,
            "margat_enabled": self.margat_enabled,
            "logit_fusion_enabled": self.logit_fusion is not None,
        }
        if self.logit_fusion is not None:
            report["logit_fusion_prior_weights"] = self.logit_fusion.prior_weights()
            report["logit_fusion_temperatures"] = self.logit_fusion.temperatures()
        return report
