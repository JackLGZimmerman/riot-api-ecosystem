"""Training entry point.

Run with:
    python -m app.ml.train
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast, overload

import torch
from torch import nn

from app.core.config.settings import PROJECT_ROOT
from app.core.logging.logger import setup_logging_config
from app.ml.config import DatasetConfig, ModelConfig, TrainConfig
from app.ml.dataset import InMemoryBatchLoader, build_loaders
from app.ml.model import N_TEAM_TOKENS, HybridTokenModel
from app.ml.utils.attention_diagnostics import (
    AttentionMetricTracker,
    attention_summary_from_metrics,
)
from app.ml.utils.live_metrics import LiveMetrics
from app.ml.utils.metrics import metric_scalar, prefixed_fields
from app.ml.utils.prediction_diagnostics import (
    HEADLINE_CENTRAL_BAND,
    central_band_summary,
    format_prediction_band_table,
    generalization_gaps,
    log_prediction_diagnostics,
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
)


def _project_relative(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _slice_batch(
    batch: dict[str, torch.Tensor],
    max_examples: int,
) -> dict[str, torch.Tensor]:
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
    bucket_table: bool = False,
    compute_band_rows: bool = False,
) -> dict[str, object]:
    """Evaluate `model` over `loader` with hard 0/1 targets.

    `bucket_table=True` attaches the console-only prediction bucket table;
    `attention` enables sampled attention diagnostics; `compute_band_rows=True`
    attaches the graduated band table under `prediction_band_rows`.
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

    for batch in loader:
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
        if diagnostic_examples >= attention.samples:
            continue
        sample_size = min(attention.batch_size, attention.samples - diagnostic_examples)
        diag_batch = _slice_batch(batch, sample_size)
        diag_n = int(diag_batch["blue_win"].shape[0])
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
        attention_tracker.update(diagnostics)
        diagnostic_examples += diag_n

    scores = torch.cat(score_chunks)
    targets = torch.cat(target_chunks)
    metrics = prediction_metrics(
        scores, targets, total_loss, total_correct, total, bucket_table=bucket_table
    )
    if attention_tracker is not None:
        metrics.update(attention_tracker.summary())
    if compute_band_rows:
        metrics["prediction_band_rows"] = prediction_band_diagnostics(scores, targets)
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
    """Held-in train loss plus train-vs-held-out generalization gaps.

    Both inputs come from the eval path (hard 0/1 targets), so they are
    directly comparable. Only train_monitor_loss is logged; the rest of the
    train-monitor metrics live inside the gap fields.
    """
    fields: dict[str, object] = {"train_monitor_loss": train_metrics["loss"]}
    fields.update(generalization_gaps(train_metrics, held_out_metrics))
    return fields


def _assemble_epoch_fields(
    *,
    epoch: int,
    step: int,
    lr: float,
    elapsed: float,
    train_loss: float,
    val_metrics: dict[str, object],
    train_monitor_metrics: dict[str, object] | None,
    train_attention_summary: dict[str, float],
    collect_heavy_diagnostics: bool,
) -> dict[str, object]:
    """Build the epoch_end metrics row.

    `train_loss` is the epoch mean of the optimization objective - measured
    against smoothed targets (TrainConfig.target_min/target_max). `val_*` and
    `train_monitor_loss` come from the eval path with hard 0/1 targets, so
    only those are directly comparable to each other.
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
        epoch_s=elapsed,
    )
    epoch_fields.update(prefixed_fields("val", central_band_summary(val_metrics)))
    if train_monitor_metrics is not None:
        epoch_fields.update(_monitor_gap_fields(train_monitor_metrics, val_metrics))

    # Heavy diagnostics: sampled attention only; bucket scalars go through
    # the dedicated prediction_bands event.
    if collect_heavy_diagnostics:
        epoch_fields.update(prefixed_fields("train", train_attention_summary))
        epoch_fields.update(
            prefixed_fields("val", attention_summary_from_metrics(val_metrics))
        )
    return epoch_fields


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
        attention_diagnostics_interval=train_cfg.attention_diagnostics_interval,
        attention_diagnostics_interval_unit="epochs",
        attention_diagnostics_batch_size=train_cfg.attention_diagnostics_batch_size,
        attention_diagnostics_eval_samples=train_cfg.attention_diagnostics_eval_samples,
        prediction_bands_enabled=train_cfg.prediction_bands_enabled,
        train_monitor_samples=train_cfg.train_monitor_samples,
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
    logger.info(
        "%s graduated prediction bands:\n%s",
        split,
        format_prediction_band_table(rows),
    )


def _maybe_compute_grad_norm(
    *,
    model: nn.Module,
    scaler: torch.amp.GradScaler,
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
    epoch_fields: dict[str, object],
    val_metrics: dict[str, object],
    train_loss: float,
) -> None:
    """One-line console summary at the end of each epoch."""
    val_central = central_band_summary(val_metrics)
    val_attention_summary = attention_summary_from_metrics(val_metrics)
    tr_mon_loss = epoch_fields.get("train_monitor_loss")
    gen_loss_gap = epoch_fields.get("gen_loss_gap")
    gen_acc_gap = epoch_fields.get("gen_accuracy_gap")
    attn_entropy = val_attention_summary.get("attention_entropy_mean")
    logger.info(
        (
            "epoch %d done in %.1fs | train_loss[smoothed] %.4e%s "
            "val_loss[hard] %.4e%s | "
            "val_acc %.4e val_auc %.4e val_brier %.4e val_ece %.4e%s"
            "val_central%s auc %.4e logloss %.4e%s"
        ),
        epoch,
        elapsed,
        train_loss,
        f" tr_mon_loss[hard] {tr_mon_loss:.4e}" if tr_mon_loss is not None else "",
        val_metrics["loss"],
        f" (gap {gen_loss_gap:.4e})" if gen_loss_gap is not None else "",
        val_metrics["accuracy"],
        val_metrics["auc"],
        val_metrics["brier"],
        val_metrics["ece"],
        f" acc_gap {gen_acc_gap:.4e} | " if gen_acc_gap is not None else " | ",
        HEADLINE_CENTRAL_BAND.replace("_", "-"),
        val_central.get(f"central_{HEADLINE_CENTRAL_BAND}_auc", float("nan")),
        val_central.get(f"central_{HEADLINE_CENTRAL_BAND}_logloss", float("nan")),
        f" | attn_entropy {attn_entropy:.4e}" if attn_entropy is not None else "",
    )


def _swap_sides(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Swap blue (positions [0:5]) and red (positions [5:10]) along the player axis.

    Side embedding is applied positionally via the model's _player_side buffer,
    so reordering the input indices alone produces the correct semantic swap:
    each player's identity travels with them but their assigned side flips.
    """
    out = dict(batch)
    for key in MODEL_INPUT_KEYS:
        x = batch[key]
        out[key] = torch.cat(
            [x[:, N_TEAM_TOKENS : N_TEAM_TOKENS * 2], x[:, :N_TEAM_TOKENS]],
            dim=1,
        )
    return out


@torch.inference_mode()
def evaluate_symmetry(
    model: nn.Module,
    loader: InMemoryBatchLoader,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    """Blue/red swap symmetry on a held-out loader.

    A perfectly side-symmetric model would have p_orig + p_swap = 1 for every
    example. The reported |p_swap - (1 - p_orig)| stats quantify how much the
    model's prediction depends on side-positional cues vs. player identity.
    Real games carry a small side win-rate gap, so some non-zero delta is
    expected; the magnitude is what matters across architecture ablations.
    """
    model.eval()
    deltas: list[torch.Tensor] = []
    for batch in loader:
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits_orig = _forward_model(model, batch)
            logits_swap = _forward_model(model, _swap_sides(batch))
        p_orig = torch.sigmoid(logits_orig.float())
        p_swap = torch.sigmoid(logits_swap.float())
        deltas.append((p_swap - (1.0 - p_orig)).detach().abs().cpu())
    delta = torch.cat(deltas)
    return {
        "symmetry_abs_delta_mean": float(delta.mean().item()),
        "symmetry_abs_delta_p50": float(delta.median().item()),
        "symmetry_abs_delta_p95": float(torch.quantile(delta, 0.95).item()),
        "symmetry_abs_delta_max": float(delta.max().item()),
        "n": int(delta.numel()),
    }


def _run_final_test(
    *,
    model: nn.Module,
    forward_model: nn.Module,
    test_loader: InMemoryBatchLoader,
    train_monitor_loader: InMemoryBatchLoader | None,
    best_path: Path,
    train_cfg: TrainConfig,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
    metrics: LiveMetrics,
    step: int,
) -> None:
    """Reload the best checkpoint, evaluate on test, record artefacts."""
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
        compute_band_rows=train_cfg.prediction_bands_enabled,
    )
    log_prediction_diagnostics(logger, "test", test_metrics)
    band_rows = test_metrics.pop("prediction_band_rows", None)
    if isinstance(band_rows, list):
        _emit_prediction_bands(metrics, "test", band_rows, step=step)
    test_attention_summary = attention_summary_from_metrics(test_metrics)
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
        **prefixed_fields("test", test_attention_summary),
        **final_gap_fields,
    )

    symmetry = evaluate_symmetry(forward_model, test_loader, use_amp, amp_dtype, device)
    logger.info(
        "test symmetry: mean=%.4e p50=%.4e p95=%.4e max=%.4e n=%d",
        symmetry["symmetry_abs_delta_mean"],
        symmetry["symmetry_abs_delta_p50"],
        symmetry["symmetry_abs_delta_p95"],
        symmetry["symmetry_abs_delta_max"],
        symmetry["n"],
    )
    metrics.record("test_symmetry", step=step, **symmetry)


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
    optimizer_steps_per_epoch = len(train_loader)
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
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp and amp_dtype is torch.float16,
    )

    best_val_loss = float("inf")
    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = train_cfg.checkpoint_dir / "best.pt"
    checkpoint_written = False
    step = 0
    epochs_since_best = 0
    # Persisted across epochs so attention_drift_* is measured epoch-over-epoch.
    train_attention_tracker = AttentionMetricTracker()

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

            for batch in train_loader.iter_batches():
                step += 1
                collect_attention = (
                    collect_heavy_diagnostics and not train_attention_recorded
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
                    train_attention_tracker.update(attention_diagnostics)
                    train_attention_recorded = True

                if step % train_cfg.log_interval == 0:
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
                bucket_table=collect_heavy_diagnostics,
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
                )
            if collect_heavy_diagnostics:
                log_prediction_diagnostics(logger, "validation", val_metrics)
            elapsed = time.perf_counter() - t0
            train_loss = float((epoch_loss_sum / epoch_n).item())
            lr_value = float(scheduler.get_last_lr()[0])

            train_attention_summary = train_attention_tracker.summary()
            epoch_fields = _assemble_epoch_fields(
                epoch=epoch,
                step=step,
                lr=lr_value,
                elapsed=elapsed,
                train_loss=train_loss,
                val_metrics=val_metrics,
                train_monitor_metrics=train_monitor_metrics,
                train_attention_summary=train_attention_summary,
                collect_heavy_diagnostics=collect_heavy_diagnostics,
            )

            _log_epoch_summary(
                epoch=epoch,
                elapsed=elapsed,
                epoch_fields=epoch_fields,
                val_metrics=val_metrics,
                train_loss=train_loss,
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
                train_monitor_loader=train_monitor_loader,
                best_path=best_path,
                train_cfg=train_cfg,
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
