"""Training entry point.

Run with:
    python -m app.ml.train
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from app.core.config.settings import PROJECT_ROOT
from app.core.logging.logger import setup_logging_config
from app.ml.config import DatasetConfig, ModelConfig, TrainConfig
from app.ml.dataset import InMemoryBatchLoader, build_loaders
from app.ml.model import HybridTokenModel
from app.ml.utils.live_metrics import LiveMetrics
from app.ml.utils.metrics import metric_scalar
from app.ml.utils.matched_diagnostics import (
    log_matched_moe_diagnostics,
    matched_moe_diagnostics,
)
from app.ml.utils.prediction_diagnostics import (
    log_prediction_bands,
    prediction_band_diagnostics,
    prediction_metrics,
)
from app.ml.utils.training_runtime import (
    configure_torch_runtime,
    cuda_runtime_info,
    resolve_amp_dtype,
    set_seed,
    smooth_binary_targets,
)

setup_logging_config()
logger = logging.getLogger(__name__)

MODEL_INPUT_KEYS = (
    "champion_idx",
    "role_idx",
    "build_idx",
    "player_profile",
)


def _project_relative(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _forward_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    dense_routing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        *(batch[key] for key in MODEL_INPUT_KEYS), dense_routing=dense_routing
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: InMemoryBatchLoader,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    """Evaluate `model` over `loader` with hard 0/1 targets.

    Returns (metrics, scores, targets) so callers can derive band tables
    without a second pass.
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0
    score_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []

    for batch in loader:
        with autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits, _ = _forward_model(model, batch)
            target = batch["blue_win"]
            total_loss += loss_fn(logits, target).item()
        probs = torch.sigmoid(logits.float())
        preds = probs > 0.5
        total_correct += int((preds == target).sum().item())
        total += target.numel()
        score_chunks.append(probs.detach().cpu())
        target_chunks.append(target.detach().cpu())

    scores = torch.cat(score_chunks)
    targets = torch.cat(target_chunks)
    metrics = prediction_metrics(scores, targets, total_loss, total_correct, total)
    return metrics, scores, targets


def _record_run_start_metadata(
    metrics: LiveMetrics,
    train_cfg: TrainConfig,
    device: torch.device,
    use_amp: bool,
) -> None:
    """Capture the durable run configuration in a `run_start` event."""
    metrics.record(
        "run_start",
        device=str(device),
        amp=use_amp,
        amp_dtype=train_cfg.amp_dtype,
        batch_size=train_cfg.batch_size,
        **cuda_runtime_info(device),
        target_min=train_cfg.target_min,
        target_max=train_cfg.target_max,
        optimizer="adamw",
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        adamw_betas=train_cfg.adamw_betas,
        compile_mode=train_cfg.compile_mode,
        lr_scheduler="linear_warmup_smooth_heavy_tail",
        warmup_steps=train_cfg.warmup_steps,
        lr_center_epoch=train_cfg.lr_center_epoch,
        lr_sharpness=train_cfg.lr_sharpness,
        lr_tail_strength=train_cfg.lr_tail_strength,
        lr_eta_min_ratio=train_cfg.lr_eta_min_ratio,
        grad_clip=train_cfg.grad_clip,
        early_stop_patience=train_cfg.early_stop_patience,
        run_final_test=train_cfg.run_final_test,
        checkpoint_dir=train_cfg.checkpoint_dir,
        metrics_dir=train_cfg.metrics_dir,
        tensorboard_dir=metrics.tensorboard_path,
        tensorboard_raw_mirror=train_cfg.tensorboard_raw_mirror,
    )


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    center_step: int,
    sharpness: float,
    tail_strength: float,
    eta_min_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LRScheduler, dict[str, object]]:
    """Linear warmup followed by a smooth heavy-tail decay.

    Post-warmup progress maps to a single continuous curve whose main
    fall-off sits at `center_step`. The tail decays slowly to
    `eta_min_ratio * base_lr` and reaches that floor exactly at the final
    step.
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")
    if tail_strength <= 0.0:
        raise ValueError("tail_strength must be positive")
    if not 0.0 <= eta_min_ratio <= 1.0:
        raise ValueError("eta_min_ratio must be in [0, 1]")

    total_steps = max(1, total_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    center_progress = max(1e-6, min(1.0, center_step / decay_steps))

    raw_end = (1.0 + (1.0 / center_progress) ** sharpness) ** (-tail_strength)
    denom = 1.0 - raw_end

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            start = 1.0 / warmup_steps
            return start + (1.0 - start) * (step / warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        raw = (1.0 + (progress / center_progress) ** sharpness) ** (-tail_strength)
        remaining = (raw - raw_end) / denom
        return eta_min_ratio + (1.0 - eta_min_ratio) * remaining

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    schedule_name = (
        "linear_warmup_smooth_heavy_tail" if warmup_steps > 0 else "smooth_heavy_tail"
    )
    scheduler_fields: dict[str, object] = {
        "scheduler": type(scheduler).__name__,
        "schedule": schedule_name,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "decay_steps": decay_steps,
        "center_step": center_step,
        "center_progress": center_progress,
        "sharpness": sharpness,
        "tail_strength": tail_strength,
        "eta_min_ratio": eta_min_ratio,
        "initial_lr": float(scheduler.get_last_lr()[0]),
    }
    return scheduler, scheduler_fields


def _emit_prediction_bands(
    metrics: LiveMetrics,
    split: str,
    rows: list[dict[str, object]],
    *,
    step: int,
) -> None:
    """Record the graduated band table as its own `prediction_bands` event."""
    metrics.record("prediction_bands", split=split, step=step, rows=rows)
    log_prediction_bands(logger, split, rows)


@torch.inference_mode()
def _collect_matched_moe_tensors(
    model: HybridTokenModel,
    loader: InMemoryBatchLoader,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if model.moe_head is None:
        return None
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {
        "baseline_logit": [],
        "final_logit": [],
        "target": [],
    }
    for batch in loader:
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model.matched_diagnostic_tensors(
                *(batch[key] for key in MODEL_INPUT_KEYS)
            )
        for key, value in output.items():
            chunks.setdefault(key, []).append(value.detach().float().cpu())
        chunks["target"].append(batch["blue_win"].detach().float().cpu())
    return {key: torch.cat(value) for key, value in chunks.items()}


def _maybe_compute_grad_norm(
    *,
    model: nn.Module,
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    will_log: bool,
    device: torch.device,
) -> float | None:
    """Compute (and optionally clip) the gradient norm.

    Returns the norm when clipping is on or `will_log` is set; otherwise None
    leaves the hot path untouched.
    """
    if grad_clip <= 0.0 and not will_log:
        return None
    max_norm = grad_clip if grad_clip > 0.0 else float("inf")
    scaler.unscale_(optimizer)
    return float(
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm,
            foreach=device.type == "cuda",
        ).item()
    )


def _log_epoch_summary(
    *,
    epoch: int,
    elapsed: float,
    val_metrics: dict[str, object],
    train_loss: float,
) -> None:
    logger.info(
        (
            "epoch %d done in %.1fs | train_loss[smoothed] %.4e val_loss[hard] %.4e "
            "| val_acc %.4e val_auc %.4e val_brier %.4e val_ece %.4e"
        ),
        epoch,
        elapsed,
        train_loss,
        val_metrics["loss"],
        val_metrics["accuracy"],
        val_metrics["auc"],
        val_metrics["brier"],
        val_metrics["ece"],
    )


def _run_final_test(
    *,
    model: HybridTokenModel,
    forward_model: nn.Module,
    test_loader: InMemoryBatchLoader,
    best_path: Path,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
    metrics: LiveMetrics,
    step: int,
) -> None:
    """Reload the best checkpoint, evaluate on test, record artefacts."""
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_metrics, test_scores, test_targets = evaluate(
        forward_model, test_loader, use_amp, amp_dtype, device
    )
    band_rows = prediction_band_diagnostics(test_scores, test_targets)
    _emit_prediction_bands(metrics, "test", band_rows, step=step)
    matched_tensors = _collect_matched_moe_tensors(
        model, test_loader, use_amp, amp_dtype, device
    )
    if matched_tensors is not None:
        matched_rows = matched_moe_diagnostics(
            matched_tensors["baseline_logit"],
            matched_tensors["final_logit"],
            matched_tensors["target"],
            {
                key: value
                for key, value in matched_tensors.items()
                if key not in {"baseline_logit", "final_logit", "target"}
            },
        )
        metrics.record(
            "matched_moe_diagnostics",
            split="test",
            step=step,
            **matched_rows,
        )
        log_matched_moe_diagnostics(logger, "test", matched_rows)
    logger.info(
        "test_loss %.4e test_auc %.4e test_accuracy %.4e "
        "test_brier %.4e test_ece %.4e n=%d",
        test_metrics["loss"],
        test_metrics["auc"],
        test_metrics["accuracy"],
        test_metrics["brier"],
        test_metrics["ece"],
        test_metrics["n"],
    )
    metrics.record(
        "test",
        step=step,
        test_loss=test_metrics["loss"],
        test_accuracy=test_metrics["accuracy"],
        test_auc=test_metrics["auc"],
        test_brier=test_metrics["brier"],
        test_ece=test_metrics["ece"],
        test_n=test_metrics["n"],
        checkpoint=best_path,
    )


def train(
    dataset_cfg: DatasetConfig | None = None,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Path:
    dataset_cfg = dataset_cfg or DatasetConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    device = torch.device(train_cfg.device)
    set_seed(train_cfg.seed, seed_cuda=device.type == "cuda")
    use_amp = train_cfg.use_amp and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(train_cfg.amp_dtype)
    configure_torch_runtime(device)
    logger.info(
        "Using device: %s | amp=%s | amp_dtype=%s",
        device,
        use_amp,
        train_cfg.amp_dtype,
    )
    metrics = LiveMetrics(
        train_cfg.metrics_dir,
        train_cfg.metrics_file,
        train_cfg.latest_metrics_file,
        train_cfg.tensorboard_dir,
        train_cfg.tensorboard_raw_mirror,
        tensorboard_run_name=train_cfg.tensorboard_run_name,
    )
    logger.info("Live metrics: %s", _project_relative(metrics.path))
    if metrics.tensorboard_path is not None:
        logger.info(
            "TensorBoard metrics: %s", _project_relative(metrics.tensorboard_path)
        )
    _record_run_start_metadata(metrics, train_cfg, device, use_amp)

    train_loader, val_loader, test_loader, vocab = build_loaders(
        dataset_cfg,
        train_cfg.batch_size,
        device,
    )
    logger.info(
        "Splits: train=%d val=%d test=%d | vocab=%s",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        vocab,
    )
    optimizer_steps_per_epoch = len(train_loader)
    metrics.record(
        "data_ready",
        train_games=len(train_loader.dataset),
        val_games=len(val_loader.dataset),
        test_games=len(test_loader.dataset),
        batches_per_epoch=len(train_loader),
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )

    model = HybridTokenModel(vocab, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM", n_params / 1e6)
    metrics.record(
        "model_ready",
        parameters=n_params,
        model_config=asdict(model_cfg),
    )
    forward_model: nn.Module = model
    compile_wrap_s = 0.0
    if train_cfg.compile_mode != "none":
        compile_mode = (
            None if train_cfg.compile_mode == "default" else train_cfg.compile_mode
        )
        t_compile = time.perf_counter()
        forward_model = cast(nn.Module, torch.compile(model, mode=compile_mode))
        compile_wrap_s = time.perf_counter() - t_compile
    metrics.record(
        "compile_ready",
        compile_mode=train_cfg.compile_mode,
        compile_wrap_s=compile_wrap_s,
    )

    optimizer_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.adamw_betas,
        fused=optimizer_fused,
    )
    metrics.record(
        "optimizer_ready",
        optimizer="adamw",
        optimizer_fused=optimizer_fused,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        adamw_betas=train_cfg.adamw_betas,
    )
    scheduler, scheduler_fields = _build_lr_scheduler(
        optimizer,
        total_steps=optimizer_steps_per_epoch * train_cfg.epochs,
        warmup_steps=train_cfg.warmup_steps,
        center_step=optimizer_steps_per_epoch * train_cfg.lr_center_epoch,
        sharpness=train_cfg.lr_sharpness,
        tail_strength=train_cfg.lr_tail_strength,
        eta_min_ratio=train_cfg.lr_eta_min_ratio,
    )
    metrics.record(
        "lr_scheduler_ready",
        step_unit="optimizer_step",
        base_lr=train_cfg.lr,
        **scheduler_fields,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler(
        device.type,
        enabled=use_amp and amp_dtype is torch.float16,
    )

    best_val_loss = float("inf")
    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = train_cfg.checkpoint_dir / "best.pt"
    checkpoint_written = False
    step = 0
    epochs_since_best = 0

    try:
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            if forward_model is not model:
                forward_model.train()
            epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            epoch_n = 0
            interval_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            interval_n = 0
            interval_t0 = time.perf_counter()
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            for batch in train_loader.iter_batches():
                step += 1

                with autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    logits, aux_loss = _forward_model(
                        forward_model,
                        batch,
                        dense_routing=step < model_cfg.moe_warmup_steps,
                    )
                    target = smooth_binary_targets(
                        batch["blue_win"],
                        train_cfg.target_min,
                        train_cfg.target_max,
                    )
                    loss = loss_fn(logits, target) + aux_loss

                scaler.scale(loss).backward()

                batch_n = batch["blue_win"].numel()
                batch_loss = loss.detach().float()
                epoch_loss_sum += batch_loss.double() * batch_n
                epoch_n += batch_n
                interval_loss_sum += batch_loss.double() * batch_n
                interval_n += batch_n

                will_log = step % train_cfg.log_interval == 0
                grad_norm = _maybe_compute_grad_norm(
                    model=model,
                    scaler=scaler,
                    optimizer=optimizer,
                    grad_clip=train_cfg.grad_clip,
                    will_log=will_log,
                    device=device,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if will_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    interval_elapsed = time.perf_counter() - interval_t0
                    interval_avg_loss = float((interval_loss_sum / interval_n).item())
                    batch_loss_value = float(batch_loss.item())
                    samples_per_s = int(round(interval_n / max(1e-9, interval_elapsed)))
                    lr = float(scheduler.get_last_lr()[0])
                    grad_norm_value = (
                        grad_norm if grad_norm is not None else float("nan")
                    )
                    logger.info(
                        "epoch %d step %d train_loss %.4e batch_loss %.4e "
                        "lr %.2e grad_norm %.3e samples/s %d",
                        epoch,
                        step,
                        interval_avg_loss,
                        batch_loss_value,
                        lr,
                        grad_norm_value,
                        samples_per_s,
                    )
                    metrics.record(
                        "train_step",
                        epoch=epoch,
                        step=step,
                        train_loss=interval_avg_loss,
                        batch_loss=batch_loss_value,
                        lr=lr,
                        grad_norm=grad_norm_value,
                        samples=epoch_n,
                        samples_per_s=samples_per_s,
                    )
                    interval_loss_sum.zero_()
                    interval_n = 0
                    interval_t0 = time.perf_counter()

            val_metrics, _, _ = evaluate(
                forward_model, val_loader, use_amp, amp_dtype, device
            )
            elapsed = time.perf_counter() - t0
            train_loss = float((epoch_loss_sum / epoch_n).item())
            lr_value = float(scheduler.get_last_lr()[0])

            _log_epoch_summary(
                epoch=epoch,
                elapsed=elapsed,
                val_metrics=val_metrics,
                train_loss=train_loss,
            )
            metrics.record(
                "epoch_end",
                epoch=epoch,
                step=step,
                lr=lr_value,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                val_auc=val_metrics["auc"],
                val_brier=val_metrics["brier"],
                val_ece=val_metrics["ece"],
                epoch_s=elapsed,
            )

            val_loss_for_checkpoint = metric_scalar(val_metrics["loss"])
            current_val_loss = (
                float(val_loss_for_checkpoint)
                if val_loss_for_checkpoint is not None
                else float("nan")
            )
            if current_val_loss < best_val_loss or not checkpoint_written:
                best_val_loss = current_val_loss
                checkpoint_written = True
                epochs_since_best = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_cfg": model_cfg,
                        "train_cfg": train_cfg,
                        "vocab": vocab,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                    },
                    best_path,
                )
                logger.info(
                    "Saved checkpoint: %s (val_loss=%.4e)",
                    _project_relative(best_path),
                    best_val_loss,
                )
                metrics.record(
                    "checkpoint",
                    epoch=epoch,
                    step=step,
                    path=best_path,
                    val_loss=best_val_loss,
                )
            else:
                epochs_since_best += 1

            if (
                train_cfg.early_stop_patience > 0
                and epochs_since_best >= train_cfg.early_stop_patience
            ):
                logger.info(
                    "Early stopping at epoch %d: no val_loss improvement "
                    "for %d epochs (best=%.4e)",
                    epoch,
                    train_cfg.early_stop_patience,
                    best_val_loss,
                )
                metrics.record(
                    "early_stop",
                    epoch=epoch,
                    step=step,
                    patience=train_cfg.early_stop_patience,
                    best_val_loss=best_val_loss,
                )
                break

        if train_cfg.run_final_test and checkpoint_written:
            _run_final_test(
                model=model,
                forward_model=forward_model,
                test_loader=test_loader,
                best_path=best_path,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                device=device,
                metrics=metrics,
                step=step,
            )
    finally:
        metrics.close()

    return best_path


if __name__ == "__main__":
    train()
