"""Training entry point.

Run with:
    python -m app.ml.train
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Literal, overload

import numpy as np
import torch
from torch import nn

from app.core.logging.logger import setup_logging_config
from app.ml.config import DatasetConfig, ModelConfig, TrainConfig
from app.ml.dataset import InMemoryBatchLoader, build_loaders
from app.ml.model import HybridTokenModel
from lion_pytorch import Lion

setup_logging_config()
logger = logging.getLogger(__name__)

MODEL_INPUT_KEYS = (
    "champion_idx",
    "role_idx",
    "build_idx",
    "interaction_score",
)
METRIC_FLOAT_SIGNIFICANT_DIGITS = 6
ECE_BINS = 15
PREDICTION_BUCKET_EDGES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
CENTRAL_PREDICTION_RANGES = ((0.35, 0.65), (0.40, 0.60), (0.45, 0.55))
MetricScalar = float | int


def _smooth_binary_targets(
    target: torch.Tensor,
    target_min: float,
    target_max: float,
) -> torch.Tensor:
    return target * (target_max - target_min) + target_min


def _metric_float(value: float) -> float:
    return float(f"{value:.{METRIC_FLOAT_SIGNIFICANT_DIGITS}g}")


def _metric_scalar(value: object) -> MetricScalar | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _metric_value(value: object) -> object:
    scalar = _metric_scalar(value)
    if scalar is not None:
        return _metric_float(float(scalar)) if isinstance(scalar, float) else scalar
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metric_value(v) for v in value]
    return value


class LiveMetrics:
    """Append-only metric stream for tailing training progress live."""

    def __init__(
        self,
        checkpoint_dir: Path,
        metrics_file: str,
        latest_file: str,
        tensorboard_dir: str | None,
    ):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.path = checkpoint_dir / metrics_file
        self.latest_path = checkpoint_dir / latest_file
        self.tensorboard_path: Path | None = None
        self._writer = None
        self._t0 = time.perf_counter()
        self._fh = self.path.open("w", encoding="utf-8")
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ModuleNotFoundError:
                logger.warning(
                    "TensorBoard is unavailable; continuing with JSONL live metrics only"
                )
            else:
                self.tensorboard_path = (
                    checkpoint_dir / tensorboard_dir / Path(metrics_file).stem
                )
                self._writer = SummaryWriter(log_dir=str(self.tensorboard_path))

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
        if self._writer is not None:
            step_field = _metric_scalar(fields.get("step"))
            epoch_field = _metric_scalar(fields.get("epoch"))
            global_step = int(step_field or epoch_field or 0)
            for key, value in row.items():
                scalar = _metric_scalar(value)
                if scalar is not None:
                    self._writer.add_scalar(
                        f"{event}/{key}", float(scalar), global_step
                    )
            self._writer.flush()

    def close(self) -> None:
        self._fh.close()
        if self._writer is not None:
            self._writer.close()


def _resolve_device(requested: str) -> torch.device:
    return torch.device(requested)


def _resolve_amp_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtypes[name.lower()]
    except KeyError as exc:
        raise ValueError(
            "TrainConfig.amp_dtype must be one of: float16, fp16, bfloat16, bf16"
        ) from exc


def _validate_train_config(cfg: TrainConfig) -> None:
    if cfg.optimizer != "lion":
        raise ValueError("TrainConfig.optimizer must be 'lion'")
    if cfg.lr < 0.0:
        raise ValueError("TrainConfig.lr must be >= 0")
    if cfg.weight_decay < 0.0:
        raise ValueError("TrainConfig.weight_decay must be >= 0")
    if len(cfg.lion_betas) != 2:
        raise ValueError("TrainConfig.lion_betas must contain exactly two values")
    beta1, beta2 = cfg.lion_betas
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("TrainConfig.lion_betas values must satisfy 0 <= beta < 1")
    if not 0.0 <= cfg.target_min < cfg.target_max <= 1.0:
        raise ValueError(
            "TrainConfig target_min/target_max must satisfy "
            "0 <= target_min < target_max <= 1"
        )
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("TrainConfig.gradient_accumulation_steps must be >= 1")
    if cfg.attention_diagnostics_interval < 0:
        raise ValueError("TrainConfig.attention_diagnostics_interval must be >= 0")
    if cfg.attention_diagnostics_batch_size < 0:
        raise ValueError("TrainConfig.attention_diagnostics_batch_size must be >= 0")
    if cfg.attention_diagnostics_eval_samples < 0:
        raise ValueError("TrainConfig.attention_diagnostics_eval_samples must be >= 0")
    if cfg.attention_diagnostics_eval_batches < 0:
        raise ValueError("TrainConfig.attention_diagnostics_eval_batches must be >= 0")


def _configure_torch_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _cuda_runtime_info(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "cuda_device_name": props.name,
        "cuda_capability": f"{props.major}.{props.minor}",
        "cuda_total_memory_gib": round(props.total_memory / (1024**3), 3),
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
    }


def _set_seed(seed: int, seed_cuda: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def _lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def _slice_batch(
    batch: dict[str, torch.Tensor],
    max_examples: int,
) -> dict[str, torch.Tensor]:
    if max_examples <= 0:
        return batch
    n = min(max_examples, batch["blue_win"].shape[0])
    return {k: v[:n] for k, v in batch.items()}


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


def _is_finite_number(value: object) -> bool:
    return _metric_scalar(value) is not None


def _public_attention_fields(
    diagnostics: dict[str, object] | None,
) -> dict[str, float]:
    if not diagnostics:
        return {}
    fields: dict[str, float] = {}
    for key, value in diagnostics.items():
        scalar = _metric_scalar(value)
        if key.startswith("attention_") and scalar is not None:
            fields[key] = float(scalar)
    return fields


class AttentionMetricTracker:
    """Aggregate sampled attention diagnostics and estimate drift over time."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = {}
        self._previous_profile: torch.Tensor | None = None
        self._updates = 0
        self._examples = 0

    def update(
        self,
        diagnostics: dict[str, object] | None,
        examples: int | None = None,
    ) -> dict[str, float]:
        fields = _public_attention_fields(diagnostics)
        if fields:
            self._updates += 1
            self._examples += max(0, int(examples or 0))

        profile = diagnostics.get("_profile") if diagnostics else None
        if isinstance(profile, torch.Tensor):
            fields.update(self._drift_fields(profile))

        for key, value in fields.items():
            if math.isfinite(value):
                self._values.setdefault(key, []).append(value)
        return fields

    def _drift_fields(self, profile: torch.Tensor) -> dict[str, float]:
        current = profile.detach().float().cpu().flatten()
        fields: dict[str, float] = {}
        if (
            self._previous_profile is not None
            and self._previous_profile.shape == current.shape
        ):
            delta = current - self._previous_profile
            prev_norm = torch.linalg.vector_norm(self._previous_profile)
            current_norm = torch.linalg.vector_norm(current)
            denom = prev_norm * current_norm
            fields["attention_drift_l2"] = float(torch.linalg.vector_norm(delta).item())
            if denom > 0:
                cosine = (
                    torch.dot(current, self._previous_profile)
                    .div(denom)
                    .clamp(-1.0, 1.0)
                    .item()
                )
                fields["attention_drift_cosine"] = float(1.0 - cosine)
        self._previous_profile = current
        return fields

    def summary(self) -> dict[str, float]:
        summary: dict[str, float] = {
            "attention_diagnostic_samples": float(self._examples),
        }
        for key, values in self._values.items():
            if not values:
                continue
            if key.endswith("_max"):
                summary[key] = max(values)
            elif key.endswith("_min"):
                summary[key] = min(values)
            elif key.endswith("_observed"):
                summary[key] = max(values)
            else:
                summary[key] = sum(values) / len(values)
            if (
                key
                in {
                    "attention_entropy_mean",
                    "attention_effective_tokens_mean",
                    "attention_head_diversity_mean",
                    "attention_head_similarity_mean",
                    "attention_max_prob_mean",
                }
                and len(values) > 1
            ):
                values_tensor = torch.tensor(values, dtype=torch.float64)
                summary[f"{key}_temporal_std"] = float(
                    values_tensor.std(unbiased=False).item()
                )
        return summary


def _prefixed_fields(prefix: str, fields: Mapping[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in fields.items()}


def _prediction_summary_from_metrics(
    metrics: dict[str, object],
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {}
    for key, value in metrics.items():
        scalar = _metric_scalar(value)
        if key.startswith("pred_") and scalar is not None:
            summary[key] = scalar
    return summary


def _attention_summary_from_metrics(
    metrics: dict[str, object],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, value in metrics.items():
        scalar = _metric_scalar(value)
        if key.startswith("attention_") and scalar is not None:
            summary[key] = float(scalar)
    return summary


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


def _binary_ece(
    scores: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = ECE_BINS,
) -> float:
    scores = scores.double()
    targets = targets.double()
    n = scores.numel()
    if n == 0:
        return float("nan")

    ece = torch.zeros((), dtype=torch.float64)
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        count = int(mask.sum().item())
        if count == 0:
            continue
        confidence = scores[mask].mean()
        accuracy = targets[mask].mean()
        ece += (count / n) * torch.abs(confidence - accuracy)
    return float(ece.item())


def _baseline_logloss(targets: torch.Tensor) -> float:
    targets = targets.double()
    if targets.numel() == 0:
        return float("nan")
    rate = targets.mean()
    eps = torch.finfo(torch.float64).eps
    p = torch.clamp(rate, eps, 1.0 - eps)
    loss = -(rate * torch.log(p) + (1.0 - rate) * torch.log1p(-p))
    return float(loss.item())


def _binary_logloss(scores: torch.Tensor, targets: torch.Tensor) -> float:
    if scores.numel() == 0:
        return float("nan")
    scores = scores.double()
    targets = targets.double()
    eps = torch.finfo(torch.float64).eps
    probs = torch.clamp(scores, eps, 1.0 - eps)
    loss = -(targets * torch.log(probs) + (1.0 - targets) * torch.log1p(-probs))
    return float(loss.mean().item())


def _prediction_distribution_fields(scores: torch.Tensor) -> dict[str, float]:
    if scores.numel() == 0:
        return {}
    quantiles = torch.quantile(
        scores.double(),
        torch.tensor(
            [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99],
            dtype=torch.float64,
            device=scores.device,
        ),
    )
    return {
        "pred_std": float(scores.std(unbiased=False).item()),
        "pred_p01": float(quantiles[0].item()),
        "pred_p05": float(quantiles[1].item()),
        "pred_p10": float(quantiles[2].item()),
        "pred_p50": float(quantiles[3].item()),
        "pred_p90": float(quantiles[4].item()),
        "pred_p95": float(quantiles[5].item()),
        "pred_p99": float(quantiles[6].item()),
    }


def _confidence_bucket_fields(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    fields: dict[str, float | int] = {}
    for threshold in (0.55, 0.60, 0.65):
        mask = scores > threshold
        count = int(mask.sum().item())
        label = int(round(threshold * 100))
        fields[f"pred_gt_{label}_count"] = count
        fields[f"pred_gt_{label}_accuracy"] = (
            float(targets[mask].mean().item()) if count else float("nan")
        )
    for threshold in (0.45, 0.40, 0.35):
        mask = scores < threshold
        count = int(mask.sum().item())
        label = int(round(threshold * 100))
        fields[f"pred_lt_{label}_count"] = count
        fields[f"pred_lt_{label}_accuracy"] = (
            float((1.0 - targets[mask]).mean().item()) if count else float("nan")
        )
    return fields


def _prediction_bucket_diagnostics(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    n = scores.numel()
    rows: list[dict[str, object]] = []
    fields: dict[str, float | int] = {}
    edges = PREDICTION_BUCKET_EDGES
    bucket_specs: list[tuple[str, str, torch.Tensor]] = [
        (
            f"<{edges[0]:.2f}",
            f"lt_{int(round(edges[0] * 100))}",
            scores < edges[0],
        )
    ]
    for lower, upper in zip(edges, edges[1:]):
        lower_label = int(round(lower * 100))
        upper_label = int(round(upper * 100))
        if upper == edges[-1]:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        bucket_specs.append(
            (
                f"{lower:.2f}-{upper:.2f}",
                f"{lower_label}_{upper_label}",
                mask,
            )
        )
    bucket_specs.append(
        (
            f">{edges[-1]:.2f}",
            f"gt_{int(round(edges[-1] * 100))}",
            scores > edges[-1],
        )
    )

    for label, key, mask in bucket_specs:
        count = int(mask.sum().item())
        pct_data = 100.0 * count / n if n else float("nan")
        if count:
            bucket_scores = scores[mask]
            bucket_targets = targets[mask]
            actual_rate = float(bucket_targets.mean().item())
            mean_pred = float(bucket_scores.mean().item())
            accuracy = float(
                ((bucket_scores > 0.5) == bucket_targets).float().mean().item()
            )
            logloss = _binary_logloss(bucket_scores, bucket_targets)
            gap = actual_rate - mean_pred
        else:
            mean_pred = float("nan")
            actual_rate = float("nan")
            gap = float("nan")
            accuracy = float("nan")
            logloss = float("nan")

        row = {
            "bucket": label,
            "count": count,
            "pct_data": pct_data,
            "mean_pred": mean_pred,
            "actual_rate": actual_rate,
            "gap": gap,
            "accuracy": accuracy,
            "logloss": logloss,
        }
        rows.append(row)
        for metric, value in row.items():
            if metric == "bucket":
                continue
            fields[f"pred_bucket_{key}_{metric}"] = value
    return rows, fields


def _central_prediction_fields(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    fields: dict[str, float | int] = {}
    n = scores.numel()
    for lower, upper in CENTRAL_PREDICTION_RANGES:
        label = f"{int(round(lower * 100))}_{int(round(upper * 100))}"
        mask = (scores >= lower) & (scores <= upper)
        count = int(mask.sum().item())
        fields[f"pred_central_{label}_count"] = count
        fields[f"pred_central_{label}_pct_data"] = (
            100.0 * count / n if n else float("nan")
        )
        if count:
            central_scores = scores[mask]
            central_targets = targets[mask]
            fields[f"pred_central_{label}_auc"] = _binary_auc(
                central_scores, central_targets
            )
            fields[f"pred_central_{label}_logloss"] = _binary_logloss(
                central_scores, central_targets
            )
            fields[f"pred_central_{label}_brier"] = float(
                torch.mean((central_scores - central_targets) ** 2).item()
            )
        else:
            fields[f"pred_central_{label}_auc"] = float("nan")
            fields[f"pred_central_{label}_logloss"] = float("nan")
            fields[f"pred_central_{label}_brier"] = float("nan")
    return fields


def _format_metric_cell(value: object, decimals: int = 3) -> str:
    scalar = _metric_scalar(value)
    if scalar is None:
        return "-"
    return f"{float(scalar):.{decimals}f}"


def _format_prediction_bucket_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "bucket       count    pct_data   mean_pred   actual_rate   gap      accuracy   logloss"
    ]
    for row in rows:
        pct_data = row.get("pct_data")
        pct_scalar = _metric_scalar(pct_data)
        pct_text = f"{float(pct_scalar):.1f}%" if pct_scalar is not None else "-"
        count_scalar = _metric_scalar(row.get("count")) or 0
        lines.append(
            f"{str(row['bucket']):<11} "
            f"{int(count_scalar):>8,}  "
            f"{pct_text:>8}   "
            f"{_format_metric_cell(row.get('mean_pred')):>9}   "
            f"{_format_metric_cell(row.get('actual_rate')):>11}   "
            f"{_format_metric_cell(row.get('gap')):>6}   "
            f"{_format_metric_cell(row.get('accuracy')):>8}   "
            f"{_format_metric_cell(row.get('logloss')):>7}"
        )
    return "\n".join(lines)


def _format_central_prediction_table(metrics: dict[str, object]) -> str:
    lines = ["range    count    pct_data   auc      logloss   brier"]
    for lower, upper in CENTRAL_PREDICTION_RANGES:
        label = f"{int(round(lower * 100))}_{int(round(upper * 100))}"
        pct_data = metrics.get(f"pred_central_{label}_pct_data")
        pct_scalar = _metric_scalar(pct_data)
        pct_text = f"{float(pct_scalar):.1f}%" if pct_scalar is not None else "-"
        count_scalar = _metric_scalar(metrics.get(f"pred_central_{label}_count")) or 0
        lines.append(
            f"{lower:.2f}-{upper:.2f} "
            f"{int(count_scalar):>8,}  "
            f"{pct_text:>8}   "
            f"{_format_metric_cell(metrics.get(f'pred_central_{label}_auc')):>6}   "
            f"{_format_metric_cell(metrics.get(f'pred_central_{label}_logloss')):>7}   "
            f"{_format_metric_cell(metrics.get(f'pred_central_{label}_brier')):>6}"
        )
    return "\n".join(lines)


def _log_prediction_diagnostics(split_name: str, metrics: dict[str, object]) -> None:
    rows = metrics.get("prediction_bucket_table")
    if isinstance(rows, list):
        logger.info(
            "%s prediction buckets:\n%s",
            split_name,
            _format_prediction_bucket_table(rows),
        )
    logger.info(
        "%s central prediction metrics:\n%s",
        split_name,
        _format_central_prediction_table(metrics),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: InMemoryBatchLoader,
    attention_diagnostics_batches: int = 0,
    attention_diagnostics_batch_size: int = 0,
    attention_diagnostics_samples: int = 0,
) -> dict[str, object]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0
    score_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    collect_attention = (
        attention_diagnostics_batches > 0 or attention_diagnostics_samples > 0
    ) and attention_diagnostics_batch_size > 0
    attention_tracker = AttentionMetricTracker() if collect_attention else None
    diagnostic_examples = 0

    for batch_idx, batch in enumerate(loader):
        logits = _forward_model(model, batch)
        target = batch["blue_win"]
        total_loss += loss_fn(logits, target).item()
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        total_correct += (preds == target).sum().item()
        total += target.numel()
        score_chunks.append(probs.detach().cpu())
        target_chunks.append(target.detach().cpu())
        if attention_tracker is not None:
            reached_sample_target = (
                attention_diagnostics_samples > 0
                and diagnostic_examples >= attention_diagnostics_samples
            )
            reached_batch_target = (
                attention_diagnostics_samples <= 0
                and batch_idx >= attention_diagnostics_batches
            )
            if reached_sample_target or reached_batch_target:
                continue
            sample_size = attention_diagnostics_batch_size
            if attention_diagnostics_samples > 0:
                sample_size = min(
                    sample_size,
                    attention_diagnostics_samples - diagnostic_examples,
                )
            diag_batch = _slice_batch(batch, sample_size)
            diag_n = int(diag_batch["blue_win"].shape[0])
            if diag_n <= 0:
                continue
            _, diagnostics = _forward_model(
                model,
                diag_batch,
                return_attention_diagnostics=True,
            )
            attention_tracker.update(diagnostics, examples=diag_n)
            diagnostic_examples += diag_n

    scores = torch.cat(score_chunks)
    targets = torch.cat(target_chunks)
    mean_pred = scores.mean()
    blue_win_rate = targets.mean()
    metrics: dict[str, object] = {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "auc": _binary_auc(scores, targets),
        "brier": torch.mean((scores - targets) ** 2).item(),
        "ece": _binary_ece(scores, targets),
        "mean_pred": mean_pred.item(),
        "blue_win_rate": blue_win_rate.item(),
        "baseline_logloss": _baseline_logloss(targets),
        "n": total,
    }
    metrics.update(_prediction_distribution_fields(scores))
    metrics.update(_confidence_bucket_fields(scores, targets))
    bucket_rows, bucket_fields = _prediction_bucket_diagnostics(scores, targets)
    metrics["prediction_bucket_table"] = bucket_rows
    metrics.update(bucket_fields)
    metrics.update(_central_prediction_fields(scores, targets))
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


def train(
    dataset_cfg: DatasetConfig | None = None,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Path:
    dataset_cfg = dataset_cfg or DatasetConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    _validate_train_config(train_cfg)

    device = _resolve_device(train_cfg.device)
    _set_seed(train_cfg.seed, seed_cuda=device.type == "cuda")
    use_amp = train_cfg.use_amp and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(train_cfg.amp_dtype)
    _configure_torch_runtime(device)
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
    logger.info(
        "Training target smoothing: 0 -> %.3f, 1 -> %.3f",
        train_cfg.target_min,
        train_cfg.target_max,
    )
    metrics = LiveMetrics(
        train_cfg.checkpoint_dir,
        train_cfg.metrics_file,
        train_cfg.latest_metrics_file,
        train_cfg.tensorboard_dir,
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
        **_cuda_runtime_info(device),
        attention_diagnostics_interval=train_cfg.attention_diagnostics_interval,
        attention_diagnostics_interval_unit="epochs",
        attention_diagnostics_batch_size=train_cfg.attention_diagnostics_batch_size,
        attention_diagnostics_eval_samples=train_cfg.attention_diagnostics_eval_samples,
        attention_diagnostics_eval_batches=train_cfg.attention_diagnostics_eval_batches,
        target_min=train_cfg.target_min,
        target_max=train_cfg.target_max,
        optimizer=train_cfg.optimizer,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        lion_betas=train_cfg.lion_betas,
        tensorboard_dir=metrics.tensorboard_path,
    )

    (
        train_loader,
        val_loader,
        test_loader,
        vocab,
        layout,
    ) = build_loaders(
        dataset_cfg,
        train_cfg.batch_size,
        device,
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
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / train_cfg.gradient_accumulation_steps
    )
    metrics.record(
        "data_ready",
        train_games=len(train_loader.dataset),
        val_games=len(val_loader.dataset),
        test_games=len(test_loader.dataset),
        interaction_tokens=n_interaction_tokens,
        batches_per_epoch=len(train_loader),
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )

    model = HybridTokenModel(vocab, layout, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM", n_params / 1e6)
    token_identity_encoding = (
        "compositional: players=champion+role+build+side+type; "
        "interactions=type+side+role_slots; no absolute token_idx embedding"
    )
    logger.info("Token identity encoding: %s", token_identity_encoding)
    metrics.record(
        "model_ready",
        parameters=n_params,
        model_config=asdict(model_cfg),
        token_identity_encoding=token_identity_encoding,
    )

    optimizer = Lion(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.lion_betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _lr_lambda(
            train_cfg.warmup_steps, optimizer_steps_per_epoch * train_cfg.epochs
        ),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp and amp_dtype is torch.float16,
    )

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
            epoch_pred_sum = torch.zeros((), device=device, dtype=torch.float64)
            epoch_target_sum = torch.zeros((), device=device, dtype=torch.float64)
            interval_loss = 0.0
            interval_n = 0
            interval_t0 = time.perf_counter()
            t0 = time.perf_counter()
            train_attention_tracker = AttentionMetricTracker()
            metrics.record("epoch_start", epoch=epoch, step=step)
            optimizer.zero_grad(set_to_none=True)
            collect_epoch_attention = (
                train_cfg.attention_diagnostics_interval > 0
                and epoch % train_cfg.attention_diagnostics_interval == 0
                and train_cfg.attention_diagnostics_batch_size > 0
            )
            train_attention_recorded = False

            train_batches = len(train_loader)
            for micro_step, batch in enumerate(train_loader, start=1):
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
                    collect_epoch_attention
                    and not train_attention_recorded
                    and accumulation_boundary
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    logits = _forward_model(model, batch)
                    target = _smooth_binary_targets(
                        batch["blue_win"],
                        train_cfg.target_min,
                        train_cfg.target_max,
                    )
                    loss = loss_fn(logits, target)

                scaled_loss = loss / accumulation_group_size
                scaler.scale(scaled_loss).backward()

                batch_n = batch["blue_win"].numel()
                batch_loss = loss.item()
                epoch_loss += batch_loss * batch_n
                epoch_n += batch_n
                interval_loss += batch_loss * batch_n
                interval_n += batch_n
                with torch.no_grad():
                    epoch_pred_sum += torch.sigmoid(logits.detach().float()).sum()
                    epoch_target_sum += batch["blue_win"].detach().sum()

                if not accumulation_boundary:
                    continue

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step = next_step

                # Diagnostics run after the optimizer step on a bounded slice,
                # keeping the normal SDPA training path clear on most steps.
                attention_fields: dict[str, float] = {}
                if collect_attention:
                    attention_diagnostics = _run_diagnostic_step(
                        model=model,
                        batch=batch,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        device=device,
                        sample_size=train_cfg.attention_diagnostics_batch_size,
                    )
                    attention_fields = train_attention_tracker.update(
                        attention_diagnostics,
                        examples=min(
                            train_cfg.attention_diagnostics_batch_size,
                            batch["blue_win"].shape[0],
                        ),
                    )
                    train_attention_recorded = True
                    if step % train_cfg.log_interval != 0:
                        metrics.record(
                            "attention_step",
                            epoch=epoch,
                            step=step,
                            samples=epoch_n,
                            **attention_fields,
                        )

                if step % train_cfg.log_interval == 0:
                    interval_elapsed = time.perf_counter() - interval_t0
                    interval_avg_loss = interval_loss / interval_n
                    samples_per_s = interval_n / max(1e-9, interval_elapsed)
                    lr = scheduler.get_last_lr()[0]
                    if attention_fields:
                        logger.info(
                            (
                                "epoch %d step %d train_loss %.4e batch_loss %.4e "
                                "lr %.2e samples/s %.1f attn_entropy %.4e "
                                "head_div %.4e max_prob %.4e"
                            ),
                            epoch,
                            step,
                            interval_avg_loss,
                            batch_loss,
                            lr,
                            samples_per_s,
                            attention_fields.get(
                                "attention_entropy_mean", float("nan")
                            ),
                            attention_fields.get(
                                "attention_head_diversity_mean", float("nan")
                            ),
                            attention_fields.get(
                                "attention_max_prob_mean", float("nan")
                            ),
                        )
                    else:
                        logger.info(
                            "epoch %d step %d train_loss %.4e batch_loss %.4e lr %.2e samples/s %.1f",
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
                        **attention_fields,
                    )
                    interval_loss = 0.0
                    interval_n = 0
                    interval_t0 = time.perf_counter()

            val_metrics = evaluate(
                model,
                val_loader,
                attention_diagnostics_batches=train_cfg.attention_diagnostics_eval_batches,
                attention_diagnostics_batch_size=train_cfg.attention_diagnostics_batch_size,
                attention_diagnostics_samples=(
                    train_cfg.attention_diagnostics_eval_samples
                    if collect_epoch_attention
                    else 0
                ),
            )
            _log_prediction_diagnostics("validation", val_metrics)
            elapsed = time.perf_counter() - t0
            train_loss = epoch_loss / epoch_n
            train_mean_pred = float((epoch_pred_sum / max(epoch_n, 1)).cpu().item())
            train_positive_rate = float(
                (epoch_target_sum / max(epoch_n, 1)).cpu().item()
            )
            train_attention_summary = train_attention_tracker.summary()
            val_attention_summary = _attention_summary_from_metrics(val_metrics)
            val_prediction_summary = _prediction_summary_from_metrics(val_metrics)
            diagnostic_fields: dict[str, object] = {}
            if collect_epoch_attention:
                diagnostic_fields.update(
                    _prefixed_fields("val", val_prediction_summary)
                )
                diagnostic_fields.update(
                    _prefixed_fields("train", train_attention_summary)
                )
                diagnostic_fields.update(_prefixed_fields("val", val_attention_summary))
            logger.info(
                (
                    "epoch %d done in %.1fs | train_loss %.4e | val_loss %.4e "
                    "val_auc %.4e val_accuracy %.4e val_brier %.4e val_ece %.4e "
                    "train_mean_pred %.4e val_mean_pred %.4e "
                    "train_positive_rate %.4e val_positive_rate %.4e "
                    "baseline_logloss %.4e val_attn_entropy %.4e"
                ),
                epoch,
                elapsed,
                train_loss,
                val_metrics["loss"],
                val_metrics["auc"],
                val_metrics["accuracy"],
                val_metrics["brier"],
                val_metrics["ece"],
                train_mean_pred,
                val_metrics["mean_pred"],
                train_positive_rate,
                val_metrics["blue_win_rate"],
                val_metrics["baseline_logloss"],
                val_attention_summary.get("attention_entropy_mean", float("nan")),
            )
            metrics.record(
                "epoch_end",
                epoch=epoch,
                step=step,
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
                **diagnostic_fields,
            )

            val_loss_for_checkpoint = _metric_scalar(val_metrics["loss"])
            current_val_loss = (
                float(val_loss_for_checkpoint)
                if val_loss_for_checkpoint is not None
                else float("nan")
            )
            if current_val_loss < best_val_loss or not checkpoint_written:
                best_val_loss = current_val_loss
                checkpoint_written = True
                epochs_since_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_cfg": model_cfg,
                        "train_cfg": train_cfg,
                        "vocab": vocab,
                        "n_interaction_tokens": n_interaction_tokens,
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
        test_metrics = evaluate(
            model,
            test_loader,
            attention_diagnostics_batches=train_cfg.attention_diagnostics_eval_batches,
            attention_diagnostics_batch_size=train_cfg.attention_diagnostics_batch_size,
            attention_diagnostics_samples=train_cfg.attention_diagnostics_eval_samples,
        )
        _log_prediction_diagnostics("test", test_metrics)
        test_attention_summary = _attention_summary_from_metrics(test_metrics)
        test_prediction_summary = _prediction_summary_from_metrics(test_metrics)
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
            **_prefixed_fields("test", test_prediction_summary),
            **_prefixed_fields("test", test_attention_summary),
        )
    finally:
        metrics.close()

    return best_path


if __name__ == "__main__":
    train()
