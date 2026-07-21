from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import binary_metrics, parameter_count, select_binary_threshold
from src.data.datamodule import resolve_balance_strategy
from src.training.callbacks import load_checkpoint, save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.losses import build_loss, inverse_frequency_class_weights
from src.training.optim import build_optimizer
from src.training.scheduler import build_scheduler


class Trainer:
    def __init__(self, model: torch.nn.Module, cfg: Any, device: torch.device, output_dir: str | Path) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.loss_name = str(cfg.training.loss.name)
        self.label_smoothing = float(cfg.training.loss.get("label_smoothing", 0.0))
        self.criterion = build_loss(name=self.loss_name, label_smoothing=self.label_smoothing)
        self.balance_strategy = resolve_balance_strategy(cfg)
        objectives = cfg.model.get("objectives", {}) if hasattr(cfg.model, "get") else {}
        self.expert_loss_weight = float(objectives.get("expert_loss_weight", 0.0)) if hasattr(objectives, "get") else 0.0
        self.resolved_class_weights: torch.Tensor | None = None
        self.optimizer_steps = 0
        configured_max_steps = cfg.training.get("max_optimizer_steps") if hasattr(cfg.training, "get") else None
        self.max_optimizer_steps = int(configured_max_steps) if configured_max_steps is not None else None
        if self.max_optimizer_steps is not None and self.max_optimizer_steps <= 0:
            raise ValueError("training.max_optimizer_steps must be positive or null")
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.training.amp) and device.type == "cuda")
        self.stopper = EarlyStopping(
            patience=int(cfg.training.early_stopping.patience),
            mode=str(cfg.training.early_stopping.mode),
        )

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        losses: list[float] = []
        probs: list[float] = []
        labels: list[int] = []
        phase = "train" if train else "evaluation"
        for batch_index, batch in enumerate(loader):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    logits, aux = self.model(x)
                    loss = self.criterion(logits, y)
                    if train and self.expert_loss_weight > 0:
                        expert_logits = list(aux.get("expert_logits", []))
                        if expert_logits:
                            expert_loss = torch.stack([self.criterion(expert, y) for expert in expert_logits]).mean()
                            loss = loss + self.expert_loss_weight * expert_loss
                    if train and "regularization_loss" in aux:
                        loss = loss + aux["regularization_loss"]
                loss_value = float(loss.detach().cpu())
                if not np.isfinite(loss_value):
                    raise FloatingPointError(
                        f"Non-finite loss detected during {phase} batch {batch_index} "
                        f"(AMP enabled: {self.scaler.is_enabled()}). "
                        "Check the learning rate, input normalization, and AMP compatibility."
                    )
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if float(self.cfg.training.grad_clip) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.training.grad_clip))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer_steps += 1
            losses.append(loss_value)
            probs.extend(torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())
            if train and self.max_optimizer_steps is not None and self.optimizer_steps >= self.max_optimizer_steps:
                break
        return float(np.mean(losses)) if losses else 0.0, np.asarray(labels), np.asarray(probs)

    def _configure_training_loss(self, loader: DataLoader) -> None:
        if self.balance_strategy != "class_weight":
            self.resolved_class_weights = None
            self.criterion = build_loss(name=self.loss_name, label_smoothing=self.label_smoothing)
            return
        windows = getattr(loader.dataset, "windows", None)
        if windows is None:
            raise ValueError("class_weight strategy requires a training dataset exposing training windows")
        labels = [int(window["label"]) for window in windows]
        self.resolved_class_weights = inverse_frequency_class_weights(labels, int(self.cfg.model.num_classes)).to(self.device)
        self.criterion = build_loss(
            self.resolved_class_weights,
            name=self.loss_name,
            label_smoothing=self.label_smoothing,
        )

    def select_decision_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, str]:
        configured = float(self.cfg.evaluation.threshold)
        method = str(self.cfg.evaluation.get("threshold_selection", "fixed")).lower()
        if method == "fixed":
            return configured, method
        if method == "validation_accuracy":
            return select_binary_threshold(y_true, y_prob, objective="accuracy", default=configured), method
        raise ValueError("evaluation.threshold_selection must be fixed or validation_accuracy")

    def _model_graph_report(self) -> dict[str, Any]:
        diagnostics = getattr(self.model, "diagnostics", None)
        if callable(diagnostics):
            return dict(diagnostics())
        prior = getattr(self.model, "prior", None)
        if prior is None:
            return {"graph_fusion_weights": {}, "learnable_adjacency": None}
        probability = prior.learnable_probability_matrix()
        adjacency: dict[str, float] | None = None
        if probability is not None:
            off_diagonal = ~torch.eye(probability.shape[0], device=probability.device, dtype=torch.bool)
            probs = probability[off_diagonal].detach().float()
            logits_matrix = (prior.learnable_adj_logits + prior.learnable_adj_logits.transpose(0, 1)) / 2.0
            logits = logits_matrix[off_diagonal].detach().float()
            adjacency = {
                "probability_mean": float(probs.mean().cpu()),
                "probability_std": float(probs.std(unbiased=False).cpu()),
                "logit_mean": float(logits.mean().cpu()),
                "logit_std": float(logits.std(unbiased=False).cpu()),
            }
        return {
            "graph_fusion_weights": prior.fusion_weights_named(),
            "learnable_adjacency": adjacency,
        }

    def fit(self, loaders: dict[str, DataLoader]) -> dict[str, Any]:
        self._configure_training_loss(loaders["train"])
        best_metrics: dict[str, Any] = {}
        best_epoch = -1
        history: list[dict[str, Any]] = []
        start = time.time()
        for epoch in range(int(self.cfg.training.epochs)):
            epoch_start = time.time()
            train_loss, train_y, train_p = self._run_epoch(loaders["train"], train=True)
            val_loss, val_y, val_p = self._run_epoch(loaders["val"], train=False)
            if self.scheduler is not None:
                self.scheduler.step()
            train_metrics = binary_metrics(train_y, train_p, threshold=float(self.cfg.evaluation.threshold)) if len(train_y) else {}
            val_metrics = binary_metrics(val_y, val_p, threshold=float(self.cfg.evaluation.threshold)) if len(val_y) else {}
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "training_time_per_epoch": time.time() - epoch_start,
                **{f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, float)},
                **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)},
            }
            history.append(row)
            monitor = str(self.cfg.training.early_stopping.monitor)
            score = float(row.get(monitor, -val_loss))
            if self.stopper.best is None or (score > self.stopper.best if self.stopper.mode == "max" else score < self.stopper.best):
                best_epoch = epoch
                best_metrics = row
                save_checkpoint(self.output_dir / "best.pt", self.model, epoch, row)
            if self.stopper.step(score):
                break
            if self.max_optimizer_steps is not None and self.optimizer_steps >= self.max_optimizer_steps:
                break
        if best_epoch < 0:
            raise RuntimeError("Training completed without producing a best checkpoint.")
        load_checkpoint(self.output_dir / "best.pt", self.model, map_location=self.device)
        _, threshold_y, threshold_p = self._run_epoch(loaders["val"], train=False)
        decision_threshold, threshold_selection = self.select_decision_threshold(threshold_y, threshold_p)
        threshold_validation_metrics = (
            binary_metrics(threshold_y, threshold_p, threshold=decision_threshold) if len(threshold_y) else {}
        )
        test_loss, test_y, test_p = self._run_epoch(loaders["test"], train=False)
        test_metrics = binary_metrics(test_y, test_p, threshold=decision_threshold) if len(test_y) else {}
        fixed_test_metrics = (
            binary_metrics(test_y, test_p, threshold=float(self.cfg.evaluation.threshold)) if len(test_y) else {}
        )
        initial_lr = float(self.cfg.training.lr)
        return {
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "test_loss": test_loss,
            **test_metrics,
            "parameter_count": parameter_count(self.model),
            "total_training_time": time.time() - start,
            "optimizer_steps": self.optimizer_steps,
            "effective_batch_size": int(self.cfg.training.batch_size),
            "learning_rate": initial_lr,
            "early_stopping_monitor": str(self.cfg.training.early_stopping.monitor),
            "decision_threshold": decision_threshold,
            "threshold_selection": threshold_selection,
            "threshold_validation_accuracy": threshold_validation_metrics.get("accuracy"),
            **{
                f"fixed_threshold_{key}": value
                for key, value in fixed_test_metrics.items()
                if isinstance(value, float)
            },
            "balance_strategy": self.balance_strategy,
            "normalization_scope": str(self.cfg.normalization.scope),
            "class_weights": self.resolved_class_weights.detach().cpu().tolist() if self.resolved_class_weights is not None else None,
            **self._model_graph_report(),
            "history": history,
        }
