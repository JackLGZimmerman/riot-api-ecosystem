"""Training entry point.

Run with:
    python -m app.ml.train
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast, overload

import torch
from torch import nn

from app.core.logging.logger import setup_logging_config
from app.ml.config import DatasetConfig, ModelConfig, TrainConfig
from app.ml.dataset import InMemoryBatchLoader, build_loaders
from app.ml.model import HybridTokenModel
from app.ml.utils.attention_diagnostics import (
    AttentionMetricTracker,
    attention_summary_from_metrics,
)
from app.ml.utils.live_metrics import LiveMetrics
from app.ml.utils.metrics import metric_scalar, prefixed_fields
from app.ml.utils.prediction_diagnostics import (
    HEADLINE_CENTRAL_BAND,
    central_band_summary,
    generalization_gaps,
    log_prediction_diagnostics,
    prediction_metrics,
    prediction_summary_from_metrics,
)
from app.ml.utils.training_runtime import (
    configure_torch_runtime,
    cuda_runtime_info,
    lr_lambda,
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
)


def _slice_batch(
    batch: dict[str, torch.Tensor],
    max_examples: int,
) -> dict[str, torch.Tensor]:
    if max_examples <= 0:
        return batch
    n = min(max_examples, batch["blue_win"].shape[0])
    return {k: v[:n] for k, v in batch.items()}


@dataclass(frozen=True)
class AttentionEvalConfig:
    """Attention-diagnostic sampling settings for `evaluate`.

    `samples > 0` collects attention from leading batches until `samples`
    examples are seen (the last batch is capped so it does not overshoot).
    `model` is the uncompiled model - the manual attention path needed for
    attention maps.
    """

    batch_size: int = 0
    samples: int = 0
    model: nn.Module | None = None

    @property
    def enabled(self) -> bool:
        return self.samples > 0 and self.batch_size > 0


def _attention_sample_size(
    attention: AttentionEvalConfig,
    batch_idx: int,
    diagnostic_examples: int,
) -> int:
    """Examples to pull from this batch for attention diagnostics (0 = skip)."""
    if attention.samples > 0:
        if diagnostic_examples >= attention.samples:
            return 0
        return min(attention.batch_size, attention.samples - diagnostic_examples)
    return 0


@overload
def _forward_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    return_attention_diagnostics: Literal[False] = False,
    attention_diagnostics_sample_size: int | None = None,
) -> torch.Tensor: ...


@overload
def _forward_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    return_attention_diagnostics: Literal[True],
    attention_diagnostics_sample_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]: ...


def _forward_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    return_attention_diagnostics: bool = False,
    attention_diagnostics_sample_size: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    return model(
        *(batch[key] for key in MODEL_INPUT_KEYS),
        return_attention_diagnostics=return_attention_diagnostics,
        attention_diagnostics_sample_size=attention_diagnostics_sample_size,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: InMemoryBatchLoader,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
    attention: AttentionEvalConfig | None = None,
    full_diagnostics: bool = True,
) -> dict[str, object]:
    """Evaluate `model` over `loader` with hard 0/1 targets.

    `full_diagnostics=False` keeps only the core + headline-central-band
    prediction metrics; `attention` enables sampled attention diagnostics.
    """
    model.eval()
    attention = attention or AttentionEvalConfig()
    if attention.model is not None:
        attention.model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0
    score_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    attention_tracker = AttentionMetricTracker() if attention.enabled else None
    diagnostic_examples = 0

    for batch_idx, batch in enumerate(loader):
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = _forward_model(model, batch)
            target = batch["blue_win"]
            total_loss += loss_fn(logits, target).item()
        probs = torch.sigmoid(logits.float())
        preds = probs > 0.5
        total_correct += int((preds == target).sum().item())
        total += target.numel()
        score_chunks.append(probs.detach().cpu())
        target_chunks.append(target.detach().cpu())

        if attention_tracker is None:
            continue
        sample_size = _attention_sample_size(attention, batch_idx, diagnostic_examples)
        if sample_size <= 0:
            continue
        diag_batch = _slice_batch(batch, sample_size)
        diag_n = int(diag_batch["blue_win"].shape[0])
        if diag_n <= 0:
            continue
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            _, diagnostics = _forward_model(
                attention.model or model,
                diag_batch,
                return_attention_diagnostics=True,
            )
        attention_tracker.update(diagnostics, examples=diag_n)
        diagnostic_examples += diag_n

    scores = torch.cat(score_chunks)
    targets = torch.cat(target_chunks)
    metrics = prediction_metrics(
        scores, targets, total_loss, total_correct, total, full=full_diagnostics
    )
    if attention_tracker is not None:
        metrics.update(attention_tracker.summary())
    return metrics


def _run_diagnostic_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
    sample_size: int,
) -> dict[str, object]:
    """Run a diagnostic-only forward pass on a small slice.

    The diagnostic forward uses the manual attention path (slower than SDPA)
    and retains attention summaries only for the requested slice.
    """
    diag_batch = _slice_batch(batch, sample_size)
    with (
        torch.no_grad(),
        torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp),
    ):
        _, diagnostics = _forward_model(
            model,
            diag_batch,
            return_attention_diagnostics=True,
            attention_diagnostics_sample_size=sample_size,
        )
    return diagnostics


def _monitor_gap_fields(
    train_metrics: dict[str, object],
    held_out_metrics: dict[str, object],
) -> dict[str, object]:
    """Held-in train metrics plus train-vs-held-out generalization gaps.

    Both inputs come from the eval path (hard 0/1 targets), so they are
    directly comparable.
    """
    fields: dict[str, object] = dict(
        train_monitor_loss=train_metrics["loss"],
        train_monitor_accuracy=train_metrics["accuracy"],
        train_monitor_auc=train_metrics["auc"],
        train_monitor_brier=train_metrics["brier"],
        train_monitor_ece=train_metrics["ece"],
        train_monitor_n=train_metrics["n"],
    )
    fields.update(prefixed_fields("train", central_band_summary(train_metrics)))
    fields.update(generalization_gaps(train_metrics, held_out_metrics))
    return fields


def _assemble_epoch_fields(
    *,
    epoch: int,
    step: int,
    lr: float,
    elapsed: float,
    train_loss: float,
    train_mean_pred: float,
    train_positive_rate: float,
    val_metrics: dict[str, object],
    train_monitor_metrics: dict[str, object] | None,
    train_attention_summary: dict[str, float],
    collect_heavy_diagnostics: bool,
) -> dict[str, object]:
    """Build the epoch_end metrics row.

    `train_loss` is the epoch mean of the optimization objective - measured
    against smoothed targets (TrainConfig.target_min/target_max).
    `val_*` and `train_monitor_*` come from the eval path with
    hard 0/1 targets, so only those two are directly comparable to each other.
    """
    epoch_fields: dict[str, object] = dict(
        epoch=epoch,
        step=step,
        lr=lr,
        train_loss=train_loss,
        val_loss=val_metrics["loss"],
        val_accuracy=val_metrics["accuracy"],
        val_auc=val_metrics["auc"],
        val_brier=val_metrics["brier"],
        val_ece=val_metrics["ece"],
        train_mean_pred=train_mean_pred,
        val_mean_pred=val_metrics["mean_pred"],
        train_positive_rate=train_positive_rate,
        val_positive_rate=val_metrics["blue_win_rate"],
        baseline_logloss=val_metrics["baseline_logloss"],
        val_n=val_metrics["n"],
        epoch_s=elapsed,
    )
    epoch_fields.update(prefixed_fields("val", central_band_summary(val_metrics)))
    if train_monitor_metrics is not None:
        epoch_fields.update(_monitor_gap_fields(train_monitor_metrics, val_metrics))

    # Heavy diagnostics - sampled attention + full prediction tables - only on
    # the attention_diagnostics_interval cadence.
    if collect_heavy_diagnostics:
        epoch_fields.update(
            prefixed_fields("val", prediction_summary_from_metrics(val_metrics))
        )
        epoch_fields.update(prefixed_fields("train", train_attention_summary))
        epoch_fields.update(
            prefixed_fields("val", attention_summary_from_metrics(val_metrics))
        )
        if train_monitor_metrics is not None:
            epoch_fields.update(
                prefixed_fields(
                    "train",
                    prediction_summary_from_metrics(train_monitor_metrics),
                )
            )
    return epoch_fields


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
    effective_batch_size = train_cfg.batch_size * train_cfg.gradient_accumulation_steps
    logger.info(
        "Using device: %s | amp=%s | amp_dtype=%s",
        device,
        use_amp,
        train_cfg.amp_dtype,
    )
    logger.info(
        "Batching: micro_batch=%d accumulation=%d effective_batch=%d",
        train_cfg.batch_size,
        train_cfg.gradient_accumulation_steps,
        effective_batch_size,
    )
    metrics = LiveMetrics(
        train_cfg.metrics_dir,
        train_cfg.metrics_file,
        train_cfg.latest_metrics_file,
        train_cfg.tensorboard_dir,
        train_cfg.tensorboard_raw_mirror,
    )
    logger.info("Live metrics: %s", metrics.path)
    if metrics.tensorboard_path is not None:
        logger.info("TensorBoard metrics: %s", metrics.tensorboard_path)
    metrics.record(
        "run_start",
        device=str(device),
        amp=use_amp,
        amp_dtype=train_cfg.amp_dtype,
        batch_size=train_cfg.batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        effective_batch_size=effective_batch_size,
        **cuda_runtime_info(device),
        attention_diagnostics_interval=train_cfg.attention_diagnostics_interval,
        attention_diagnostics_interval_unit="epochs",
        attention_diagnostics_batch_size=train_cfg.attention_diagnostics_batch_size,
        attention_diagnostics_eval_samples=train_cfg.attention_diagnostics_eval_samples,
        train_monitor_samples=train_cfg.train_monitor_samples,
        target_min=train_cfg.target_min,
        target_max=train_cfg.target_max,
        optimizer="adamw",
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        adamw_betas=train_cfg.adamw_betas,
        compile_mode=train_cfg.compile_mode,
        grad_clip=train_cfg.grad_clip,
        checkpoint_dir=train_cfg.checkpoint_dir,
        metrics_dir=train_cfg.metrics_dir,
        tensorboard_dir=metrics.tensorboard_path,
        tensorboard_raw_mirror=train_cfg.tensorboard_raw_mirror,
    )

    (
        train_loader,
        val_loader,
        test_loader,
        train_monitor_loader,
        vocab,
    ) = build_loaders(
        dataset_cfg,
        train_cfg.batch_size,
        device,
        train_monitor_samples=train_cfg.train_monitor_samples,
    )
    train_monitor_games = (
        len(train_monitor_loader.dataset) if train_monitor_loader is not None else 0
    )
    logger.info(
        "Splits: train=%d val=%d test=%d | train_monitor=%d | vocab=%s",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        train_monitor_games,
        vocab,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / train_cfg.gradient_accumulation_steps
    )
    metrics.record(
        "data_ready",
        train_games=len(train_loader.dataset),
        val_games=len(val_loader.dataset),
        test_games=len(test_loader.dataset),
        train_monitor_games=train_monitor_games,
        batches_per_epoch=len(train_loader),
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )

    model = HybridTokenModel(vocab, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM", n_params / 1e6)
    token_identity_encoding = (
        "compositional: players=champion+role+build+side; "
        "no absolute token_idx embedding"
    )
    metrics.record(
        "model_ready",
        parameters=n_params,
        model_config=asdict(model_cfg),
        token_identity_encoding=token_identity_encoding,
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda(train_cfg.warmup_steps, optimizer_steps_per_epoch * train_cfg.epochs),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp and amp_dtype is torch.float16,
    )

    best_val_loss = float("inf")
    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = train_cfg.checkpoint_dir / "best.pt"
    checkpoint_written = False
    step = 0
    # Persisted across epochs so attention_drift_* is measured epoch-over-epoch.
    train_attention_tracker = AttentionMetricTracker()

    try:
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            if forward_model is not model:
                forward_model.train()
            epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            epoch_n = 0
            epoch_pred_sum = torch.zeros((), device=device, dtype=torch.float64)
            epoch_target_sum = torch.zeros((), device=device, dtype=torch.float64)
            interval_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            interval_n = 0
            interval_t0 = time.perf_counter()
            t0 = time.perf_counter()
            train_attention_tracker.reset()
            optimizer.zero_grad(set_to_none=True)
            # Core metrics record every epoch. collect_heavy_diagnostics gates
            # the heavy sampled-attention + full prediction-table fields, both
            # in the epoch_end row and inside evaluate().
            collect_heavy_diagnostics = (
                train_cfg.attention_diagnostics_interval > 0
                and epoch % train_cfg.attention_diagnostics_interval == 0
                and train_cfg.attention_diagnostics_batch_size > 0
            )
            train_attention_recorded = False

            train_batches = len(train_loader)
            train_iter = train_loader.iter_batches()
            for micro_step, batch in enumerate(train_iter, start=1):
                accumulation_boundary = (
                    micro_step % train_cfg.gradient_accumulation_steps == 0
                    or micro_step == train_batches
                )
                accumulation_group_start = (
                    (micro_step - 1) // train_cfg.gradient_accumulation_steps
                ) * train_cfg.gradient_accumulation_steps + 1
                accumulation_group_size = min(
                    train_cfg.gradient_accumulation_steps,
                    train_batches - accumulation_group_start + 1,
                )
                next_step = step + 1
                collect_attention = (
                    collect_heavy_diagnostics
                    and not train_attention_recorded
                    and accumulation_boundary
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    logits = _forward_model(forward_model, batch)
                    target = smooth_binary_targets(
                        batch["blue_win"],
                        train_cfg.target_min,
                        train_cfg.target_max,
                    )
                    loss = loss_fn(logits, target)

                scaled_loss = loss / accumulation_group_size
                scaler.scale(scaled_loss).backward()

                batch_n = batch["blue_win"].numel()
                batch_loss = loss.detach().float()
                epoch_loss_sum += batch_loss.double() * batch_n
                epoch_n += batch_n
                interval_loss_sum += batch_loss.double() * batch_n
                interval_n += batch_n
                with torch.no_grad():
                    epoch_pred_sum += torch.sigmoid(logits.detach().float()).sum()
                    epoch_target_sum += batch["blue_win"].detach().sum()

                if not accumulation_boundary:
                    continue

                will_log = next_step % train_cfg.log_interval == 0
                grad_norm: float | None = None
                if train_cfg.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            train_cfg.grad_clip,
                            foreach=device.type == "cuda",
                        ).item()
                    )
                elif will_log:
                    # Total gradient norm without clipping: max_norm=inf measures
                    # the norm and scales grads by 1.0. Only on logged steps, so
                    # the hot path is untouched.
                    scaler.unscale_(optimizer)
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float("inf"),
                            foreach=device.type == "cuda",
                        ).item()
                    )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step = next_step

                # Diagnostics run after the optimizer step on a bounded slice,
                # keeping the normal SDPA training path clear on most steps. The
                # tracker accumulates into the epoch's single epoch_end row.
                if collect_attention:
                    attention_diagnostics = _run_diagnostic_step(
                        model=model,
                        batch=batch,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        device=device,
                        sample_size=train_cfg.attention_diagnostics_batch_size,
                    )
                    train_attention_tracker.update(
                        attention_diagnostics,
                        examples=min(
                            train_cfg.attention_diagnostics_batch_size,
                            batch["blue_win"].shape[0],
                        ),
                    )
                    train_attention_recorded = True

                if step % train_cfg.log_interval == 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    interval_elapsed = time.perf_counter() - interval_t0
                    interval_avg_loss = float((interval_loss_sum / interval_n).item())
                    batch_loss_value = float(batch_loss.item())
                    samples_per_s = interval_n / max(1e-9, interval_elapsed)
                    lr = float(scheduler.get_last_lr()[0])
                    grad_norm_value = (
                        grad_norm if grad_norm is not None else float("nan")
                    )
                    logger.info(
                        "epoch %d step %d train_loss %.4e batch_loss %.4e "
                        "lr %.2e grad_norm %.3e samples/s %.1f",
                        epoch,
                        step,
                        interval_avg_loss,
                        batch_loss_value,
                        lr,
                        grad_norm_value,
                        samples_per_s,
                    )
                    # train_loss / batch_loss here are the optimization
                    # objective with smoothed targets. The eval-path val_* /
                    # train_monitor_* losses use hard targets.
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

            val_metrics = evaluate(
                forward_model,
                val_loader,
                use_amp,
                amp_dtype,
                device,
                attention=AttentionEvalConfig(
                    batch_size=train_cfg.attention_diagnostics_batch_size,
                    samples=(
                        train_cfg.attention_diagnostics_eval_samples
                        if collect_heavy_diagnostics
                        else 0
                    ),
                    model=model,
                ),
                full_diagnostics=collect_heavy_diagnostics,
            )
            # Held-in train subset through the eval path: directly comparable
            # to validation for the overfitting read. Kept light - no attention.
            train_monitor_metrics: dict[str, object] | None = None
            if train_monitor_loader is not None:
                train_monitor_metrics = evaluate(
                    forward_model,
                    train_monitor_loader,
                    use_amp,
                    amp_dtype,
                    device,
                    full_diagnostics=collect_heavy_diagnostics,
                )
            if collect_heavy_diagnostics:
                log_prediction_diagnostics(logger, "validation", val_metrics)
            elapsed = time.perf_counter() - t0
            # Epoch mean of the smoothed-target optimization objective; not
            # comparable to the hard-target eval losses.
            train_loss = float((epoch_loss_sum / epoch_n).item())
            train_mean_pred = float((epoch_pred_sum / max(epoch_n, 1)).cpu().item())
            train_positive_rate = float(
                (epoch_target_sum / max(epoch_n, 1)).cpu().item()
            )
            lr_value = float(scheduler.get_last_lr()[0])

            train_attention_summary = train_attention_tracker.summary()
            epoch_fields = _assemble_epoch_fields(
                epoch=epoch,
                step=step,
                lr=lr_value,
                elapsed=elapsed,
                train_loss=train_loss,
                train_mean_pred=train_mean_pred,
                train_positive_rate=train_positive_rate,
                val_metrics=val_metrics,
                train_monitor_metrics=train_monitor_metrics,
                train_attention_summary=train_attention_summary,
                collect_heavy_diagnostics=collect_heavy_diagnostics,
            )

            val_central = central_band_summary(val_metrics)
            val_attention_summary = attention_summary_from_metrics(val_metrics)
            # train_loss is the smoothed-target optimization loss; tr_mon_loss
            # and val_loss are eval-path losses on hard 0/1 targets (only those
            # two are directly comparable - see _assemble_epoch_fields).
            logger.info(
                (
                    "epoch %d done in %.1fs | train_loss[smoothed] %.4e "
                    "tr_mon_loss[hard] %.4e val_loss[hard] %.4e (gap %.4e) | "
                    "val_acc %.4e val_auc %.4e val_brier %.4e val_ece %.4e "
                    "acc_gap %.4e | val_central%s auc %.4e logloss %.4e | "
                    "val_attn_entropy %.4e"
                ),
                epoch,
                elapsed,
                train_loss,
                epoch_fields.get("train_monitor_loss", float("nan")),
                val_metrics["loss"],
                epoch_fields.get("gen_loss_gap", float("nan")),
                val_metrics["accuracy"],
                val_metrics["auc"],
                val_metrics["brier"],
                val_metrics["ece"],
                epoch_fields.get("gen_accuracy_gap", float("nan")),
                HEADLINE_CENTRAL_BAND.replace("_", "-"),
                val_central.get(f"central_{HEADLINE_CENTRAL_BAND}_auc", float("nan")),
                val_central.get(
                    f"central_{HEADLINE_CENTRAL_BAND}_logloss",
                    float("nan"),
                ),
                val_attention_summary.get("attention_entropy_mean", float("nan")),
            )
            metrics.record("epoch_end", **epoch_fields)

            val_loss_for_checkpoint = metric_scalar(val_metrics["loss"])
            current_val_loss = (
                float(val_loss_for_checkpoint)
                if val_loss_for_checkpoint is not None
                else float("nan")
            )
            if current_val_loss < best_val_loss or not checkpoint_written:
                best_val_loss = current_val_loss
                checkpoint_written = True
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
                    "Saved checkpoint: %s (val_loss=%.4e)", best_path, best_val_loss
                )
                metrics.record(
                    "checkpoint",
                    epoch=epoch,
                    step=step,
                    path=best_path,
                    val_loss=best_val_loss,
                )

        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        test_metrics = evaluate(
            forward_model,
            test_loader,
            use_amp,
            amp_dtype,
            device,
            attention=AttentionEvalConfig(
                batch_size=train_cfg.attention_diagnostics_batch_size,
                samples=train_cfg.attention_diagnostics_eval_samples,
                model=model,
            ),
        )
        log_prediction_diagnostics(logger, "test", test_metrics)
        test_attention_summary = attention_summary_from_metrics(test_metrics)
        test_prediction_summary = prediction_summary_from_metrics(test_metrics)
        # Final train/test gap from the best checkpoint: the end-of-run
        # overfitting summary.
        final_gap_fields: dict[str, object] = {}
        if train_monitor_loader is not None:
            final_train_metrics = evaluate(
                forward_model,
                train_monitor_loader,
                use_amp,
                amp_dtype,
                device,
            )
            final_gap_fields = _monitor_gap_fields(final_train_metrics, test_metrics)
        logger.info(
            (
                "test_loss %.4e test_auc %.4e test_accuracy %.4e "
                "test_brier %.4e test_ece %.4e test_mean_pred %.4e "
                "test_positive_rate %.4e test_baseline_logloss %.4e "
                "n=%d test_attn_entropy %.4e"
            ),
            test_metrics["loss"],
            test_metrics["auc"],
            test_metrics["accuracy"],
            test_metrics["brier"],
            test_metrics["ece"],
            test_metrics["mean_pred"],
            test_metrics["blue_win_rate"],
            test_metrics["baseline_logloss"],
            test_metrics["n"],
            test_attention_summary.get("attention_entropy_mean", float("nan")),
        )
        metrics.record(
            "test",
            step=step,
            test_loss=test_metrics["loss"],
            test_accuracy=test_metrics["accuracy"],
            test_auc=test_metrics["auc"],
            test_brier=test_metrics["brier"],
            test_ece=test_metrics["ece"],
            test_mean_pred=test_metrics["mean_pred"],
            test_positive_rate=test_metrics["blue_win_rate"],
            test_baseline_logloss=test_metrics["baseline_logloss"],
            test_n=test_metrics["n"],
            checkpoint=best_path,
            **prefixed_fields("test", central_band_summary(test_metrics)),
            **prefixed_fields("test", test_prediction_summary),
            **prefixed_fields("test", test_attention_summary),
            **final_gap_fields,
        )
    finally:
        metrics.close()

    return best_path


if __name__ == "__main__":
    train()
