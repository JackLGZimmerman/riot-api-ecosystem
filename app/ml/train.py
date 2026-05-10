"""Training entry point.

Run with:
    python -m app.ml.train
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from app.core.logging.logger import setup_logging_config
from app.ml.config import DatasetConfig, ModelConfig, TrainConfig
from app.ml.dataset import build_loaders
from app.ml.model import HybridTokenModel

setup_logging_config()
logger = logging.getLogger(__name__)

MODEL_INPUT_KEYS = (
    "champion_idx",
    "role_idx",
    "build_idx",
    "interaction_score",
    "interaction_reliability",
)


def _metric_value(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metric_value(v) for v in value]
    return value


class LiveMetrics:
    """Append-only metric stream for tailing training progress live."""

    def __init__(self, checkpoint_dir: Path, metrics_file: str, latest_file: str):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.path = checkpoint_dir / metrics_file
        self.latest_path = checkpoint_dir / latest_file
        self._t0 = time.perf_counter()
        self._fh = self.path.open("w", encoding="utf-8")

    def record(self, event: str, **fields: object) -> None:
        row = {
            "event": event,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.perf_counter() - self._t0, 3),
            **fields,
        }
        row = {k: _metric_value(v) for k, v in row.items()}
        line = json.dumps(row, sort_keys=True)
        self._fh.write(f"{line}\n")
        self._fh.flush()
        self.latest_path.write_text(
            json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
        )

    def close(self) -> None:
        self._fh.close()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check WSL GPU passthrough.")
    return torch.device(requested)


def _set_seed(seed: int, seed_cuda: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def _cuda_memory_gib(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def _lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _forward_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    return model(*(batch[key] for key in MODEL_INPUT_KEYS))


def _binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic. Returns NaN if degenerate."""
    n_pos = int(targets.sum().item())
    n_neg = targets.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(
        1, scores.numel() + 1, dtype=torch.float64, device=scores.device
    )
    sum_pos_ranks = ranks[targets > 0.5].sum().item()
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0
    score_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        logits = _forward_model(model, batch)
        target = batch["blue_win"]
        total_loss += loss_fn(logits, target).item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).to(target.dtype)
        total_correct += (preds == target).sum().item()
        total += target.numel()
        score_chunks.append(probs.detach().cpu())
        target_chunks.append(target.detach().cpu())

    if total == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "auc": float("nan"),
            "n": 0,
        }

    scores = torch.cat(score_chunks)
    targets = torch.cat(target_chunks)
    return {
        "loss": total_loss / max(1, total),
        "accuracy": total_correct / max(1, total),
        "auc": _binary_auc(scores, targets),
        "n": total,
    }


def train(
    dataset_cfg: DatasetConfig | None = None,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Path:
    dataset_cfg = dataset_cfg or DatasetConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    device = _resolve_device(train_cfg.device)
    _set_seed(train_cfg.seed, seed_cuda=device.type == "cuda")
    use_amp = train_cfg.use_amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logger.info("Using device: %s | amp=%s", device, use_amp)
    metrics = LiveMetrics(
        train_cfg.checkpoint_dir,
        train_cfg.metrics_file,
        train_cfg.latest_metrics_file,
    )
    logger.info("Live metrics: %s", metrics.path)
    metrics.record("run_start", device=str(device), amp=use_amp)

    (
        train_loader,
        val_loader,
        test_loader,
        vocab,
        layout,
    ) = build_loaders(
        dataset_cfg,
        train_cfg.batch_size,
        train_cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    n_interaction_tokens = layout.types.numel()
    logger.info(
        "Splits: train=%d val=%d test=%d | interaction_tokens=%d | vocab=%s",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        n_interaction_tokens,
        vocab,
    )
    metrics.record(
        "data_ready",
        train_games=len(train_loader.dataset),
        val_games=len(val_loader.dataset),
        test_games=len(test_loader.dataset),
        interaction_tokens=n_interaction_tokens,
        batches_per_epoch=len(train_loader),
    )

    model = HybridTokenModel(vocab, layout, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM", n_params / 1e6)
    metrics.record("model_ready", parameters=n_params)

    optimizer = AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _lr_lambda(train_cfg.warmup_steps, len(train_loader) * train_cfg.epochs),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_loss = float("inf")
    best_path = train_cfg.checkpoint_dir / "best.pt"
    checkpoint_written = False
    epochs_since_improvement = 0
    step = 0

    try:
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_n = 0
            interval_loss = 0.0
            interval_n = 0
            interval_t0 = time.perf_counter()
            t0 = time.perf_counter()
            metrics.record("epoch_start", epoch=epoch, step=step)

            for batch in train_loader:
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = _forward_model(model, batch)
                    loss = loss_fn(logits, batch["blue_win"])

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                batch_n = batch["blue_win"].numel()
                batch_loss = loss.item()
                epoch_loss += batch_loss * batch_n
                epoch_n += batch_n
                interval_loss += batch_loss * batch_n
                interval_n += batch_n
                step += 1

                if step % train_cfg.log_interval == 0:
                    interval_elapsed = time.perf_counter() - interval_t0
                    interval_avg_loss = interval_loss / max(1, interval_n)
                    samples_per_s = interval_n / max(1e-9, interval_elapsed)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "epoch %d step %d train_loss %.4f batch_loss %.4f lr %.2e samples/s %.1f",
                        epoch,
                        step,
                        interval_avg_loss,
                        batch_loss,
                        lr,
                        samples_per_s,
                    )
                    metrics.record(
                        "train_step",
                        epoch=epoch,
                        step=step,
                        train_loss=interval_avg_loss,
                        batch_loss=batch_loss,
                        lr=lr,
                        samples=epoch_n,
                        samples_per_s=samples_per_s,
                        **_cuda_memory_gib(device),
                    )
                    interval_loss = 0.0
                    interval_n = 0
                    interval_t0 = time.perf_counter()

            val_metrics = evaluate(model, val_loader, device)
            elapsed = time.perf_counter() - t0
            train_loss = epoch_loss / max(1, epoch_n)
            logger.info(
                "epoch %d done in %.1fs | train_loss %.4f | val_loss %.4f val_acc %.4f val_auc %.4f",
                epoch,
                elapsed,
                train_loss,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["auc"],
            )
            metrics.record(
                "epoch_end",
                epoch=epoch,
                step=step,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                val_auc=val_metrics["auc"],
                val_n=val_metrics["n"],
                epoch_s=elapsed,
                **_cuda_memory_gib(device),
            )

            if val_metrics["loss"] < best_val_loss or not checkpoint_written:
                best_val_loss = val_metrics["loss"]
                checkpoint_written = True
                epochs_since_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_cfg": model_cfg,
                        "vocab": vocab,
                        "n_interaction_tokens": n_interaction_tokens,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                    },
                    best_path,
                )
                logger.info(
                    "Saved checkpoint: %s (val_loss=%.4f)", best_path, best_val_loss
                )
                metrics.record(
                    "checkpoint",
                    epoch=epoch,
                    step=step,
                    path=best_path,
                    val_loss=best_val_loss,
                )
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= train_cfg.early_stopping_patience:
                    logger.info(
                        "Early stopping after %d epochs without improvement",
                        epochs_since_improvement,
                    )
                    metrics.record(
                        "early_stopping",
                        epoch=epoch,
                        step=step,
                        epochs_since_improvement=epochs_since_improvement,
                    )
                    break

        logger.info("Loading best checkpoint for final test evaluation")
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        test_metrics = evaluate(model, test_loader, device)
        logger.info(
            "test_loss %.4f test_acc %.4f test_auc %.4f n=%d",
            test_metrics["loss"],
            test_metrics["accuracy"],
            test_metrics["auc"],
            test_metrics["n"],
        )
        metrics.record(
            "test",
            step=step,
            test_loss=test_metrics["loss"],
            test_accuracy=test_metrics["accuracy"],
            test_auc=test_metrics["auc"],
            test_n=test_metrics["n"],
            checkpoint=best_path,
            **_cuda_memory_gib(device),
        )
    finally:
        metrics.close()

    return best_path


if __name__ == "__main__":
    train()
