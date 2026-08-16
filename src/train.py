"""Train TCN-MSE-MARGAT and persist every run under /workspace/output."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data import create_dataloaders_with_summary, load_config
from src.evaluate import binary_metrics, build_model, predict, select_threshold

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --device cpu only for checks"
        )
    return torch.device(name)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            loss = nn.functional.cross_entropy(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * len(targets)
    return total_loss / len(loader.dataset)


def probability_summary(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, dict[str, float]]:
    """Summarize validation probabilities by class without retaining samples."""
    result: dict[str, dict[str, float]] = {}
    for label, name in ((0, "negative"), (1, "positive")):
        values = probabilities[labels == label]
        if len(values) == 0:
            raise ValueError(f"Validation data has no {name} samples")
        result[name] = {
            "min": float(np.min(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    return result


def selection_score(metrics: dict[str, Any], metric: str) -> float:
    """Return the configured validation score used for checkpoint selection."""
    if metric not in {"f1", "balanced_accuracy"}:
        raise ValueError(f"Unsupported selection metric: {metric}")
    return float(metrics[metric])


def _setup_logging(run_dir: Path) -> None:
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def _save_environment(run_dir: Path, device: torch.device) -> None:
    information = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(information, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCN-MSE-MARGAT")
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument(
        "--fold", required=True, help="0-9 for mixed CV or patient ID for LOPO"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()

    config = load_config(args.configs)
    split_id: int | str = int(args.fold) if args.fold.isdigit() else args.fold.lower()
    requested_device = args.device or config["train"]["device"]
    config["train"]["device"] = requested_device
    device = _device(requested_device)
    seed_everything(int(config["seed"]))

    resume = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    if resume and (resume["config"] != config or resume["split_id"] != split_id):
        raise ValueError(
            "Resume checkpoint config or split does not match this command"
        )
    if resume:
        run_dir = args.resume.parent
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{config['split']['protocol']}_{config['data']['window_seconds']:g}s_{split_id}_{stamp}"
        run_dir = Path(config["train"]["output_dir"]) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
    _setup_logging(run_dir)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _save_environment(run_dir, device)

    selection_metric = str(config["train"].get("threshold_metric", "f1"))
    selection_score({"f1": 0.0, "balanced_accuracy": 0.0}, selection_metric)
    loaders, split_summary = create_dataloaders_with_summary(config, split_id)
    LOGGER.info("Split summary: %s", json.dumps(split_summary, sort_keys=True))
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    amp = bool(config["train"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_score, best_f1, stale_epochs = 0, -1.0, -1.0, 0
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = resume["epoch"] + 1
        best_score = float(resume.get("best_score", resume.get("best_f1", -1.0)))
        best_f1 = float(resume.get("best_f1", -1.0))
        stale_epochs = resume["stale_epochs"]
        LOGGER.info("Resuming at epoch %d", start_epoch)

    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        train_loss = train_epoch(
            model, loaders["train"], optimizer, scaler, device, amp
        )
        val_labels, val_probabilities = predict(model, loaders["val"], device)
        threshold = select_threshold(
            val_labels, val_probabilities, selection_metric
        )
        metrics = binary_metrics(val_labels, val_probabilities, threshold)
        score = selection_score(metrics, selection_metric)
        validation_probability_summary = probability_summary(
            val_labels, val_probabilities
        )
        improved = score > best_score
        best_score = max(best_score, score)
        best_f1 = max(best_f1, metrics["f1"])
        stale_epochs = 0 if improved else stale_epochs + 1
        state: dict[str, Any] = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "split_id": split_id,
            "threshold": threshold,
            "validation_metrics": metrics,
            "selection_metric": selection_metric,
            "selection_score": score,
            "best_score": best_score,
            "best_f1": best_f1,
            "stale_epochs": stale_epochs,
            "split_summary": split_summary,
            "validation_probability_summary": validation_probability_summary,
        }
        torch.save(state, run_dir / "last.pt")
        if improved:
            torch.save(state, run_dir / "best.pt")
            diagnostics = {
                "split_id": split_id,
                "split": split_summary,
                "best_epoch": epoch,
                "selection_metric": selection_metric,
                "selection_score": score,
                "threshold": threshold,
                "validation_metrics": metrics,
                "validation_probability_summary": validation_probability_summary,
            }
            (run_dir / "diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2), encoding="utf-8"
            )
            LOGGER.info(
                "Best validation probability summary: %s",
                json.dumps(validation_probability_summary, sort_keys=True),
            )
        LOGGER.info(
            "epoch=%d loss=%.4f val_%s=%.4f threshold=%.6f",
            epoch,
            train_loss,
            selection_metric,
            score,
            threshold,
        )
        if stale_epochs >= int(config["train"]["patience"]):
            LOGGER.info("Early stopping after %d stale epochs", stale_epochs)
            break


if __name__ == "__main__":
    main()
