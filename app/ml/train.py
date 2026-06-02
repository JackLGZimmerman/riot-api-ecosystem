# pyright: reportPrivateImportUsage=false

"""Train the production HGNN win-rate model.

Run with:
    python -m app.ml.train
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import nn

from app.core.config.settings import PROJECT_ROOT
from app.core.logging.logger import setup_logging_config
from app.ml.config import DatasetConfig, TrainConfig
from app.ml.dataset import SplitData, identity_meta, load_splits
from app.ml.hgnn_model import (
    HGNNConfig,
    HGNNWinModel,
    build_hgnn_inputs,
    save_hgnn_model,
    swap_hgnn_inputs,
)

setup_logging_config()
logger = logging.getLogger(__name__)

EPS = 1e-12
# Benchmarked on RTX 5070 Ti: 8192 hits the allocator cliff; 7424 maximizes
# samples/s while preserving a little memory headroom.
HGNN_TRAIN_BATCH = 7424


@dataclass(frozen=True)
class RawTensorSplit:
    win_rate: torch.Tensor
    p1_cnt: torch.Tensor
    blue_win: torch.Tensor
    champion_id: torch.Tensor | None = None
    build_id: torch.Tensor | None = None
    matchup_1v1: torch.Tensor | None = None
    synergy_2vx: torch.Tensor | None = None
    m1v1_cnt: torch.Tensor | None = None
    s2vx_cnt: torch.Tensor | None = None
    identity_semantic: torch.Tensor | None = None
    identity_profile: torch.Tensor | None = None
    identity_context: torch.Tensor | None = None
    identity_context_support: torch.Tensor | None = None
    identity_context_raw: torch.Tensor | None = None
    m1v1_detail: torch.Tensor | None = None


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


_LONG_TENSOR_FIELDS = frozenset({"champion_id", "build_id"})


def _map_split(split: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply `fn` to every present field, rebuilding the same split dataclass."""
    return type(split)(
        **{
            f.name: (None if (value := getattr(split, f.name)) is None else fn(value))
            for f in fields(split)
        }
    )


def _limit_split(split: SplitData, max_games: int | None) -> SplitData:
    if max_games is None or split.blue_win.size <= max_games:
        return split
    n = int(max_games)
    return _map_split(split, lambda array: array[:n])


def _drop_unused_model_arrays(split: SplitData, config: HGNNConfig) -> SplitData:
    """Null out per-player classification arrays the configured model never reads.

    Production uses the threshold-tuned identity-conditioned raw context head as
    the naive semantic-context implementation. Direct 1v1/2vX integrations are
    retained as explicit research/legacy capacity but disabled by default, so
    their tables stay loader-visible without entering the active tensor pipeline.
    The legacy semantic / profile / 1v1-detail node paths remain disabled, and
    the two context heads are mutually exclusive, so this keeps multiple GB of
    unused tensors off the GPU without changing active model output
    (build_hgnn_inputs only forwards present arrays)."""
    conditioned = (
        config.use_identity_conditioned_context
        and config.identity_context_conditioning_type in {"low_rank", "film"}
        and config.identity_context_raw_dim > 0
    )
    drop: dict[str, bool] = {
        "identity_semantic": config.identity_semantic_dim <= 0,
        "identity_profile": config.identity_profile_dim <= 0,
        "m1v1_detail": config.m1v1_detail_dim <= 0,
        # raw block: only the conditioned head consumes it.
        "identity_context_raw": not conditioned,
        # 24-dim descriptor: the shared head always needs it; the conditioned head
        # needs it only for the raw_plus_dense source's dense tail.
        "identity_context": config.identity_context_dim <= 0
        or (conditioned and config.identity_context_source != "raw_plus_dense"),
        "matchup_1v1": not config.use_relationship_integrations,
        "synergy_2vx": not config.use_relationship_integrations,
        "m1v1_cnt": not config.use_relationship_integrations,
        "s2vx_cnt": not config.use_relationship_integrations,
        "m1v1_eff_n": not config.use_relationship_integrations,
        "s2vx_eff_n": not config.use_relationship_integrations,
    }
    overrides = {name: None for name, unused in drop.items() if unused}
    if not overrides:
        return split
    return type(split)(
        **{f.name: overrides.get(f.name, getattr(split, f.name)) for f in fields(split)}
    )


def _binary_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    n_pos = int(targets.sum())
    n_neg = int(targets.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sum_pos_ranks = ranks[targets > 0.5].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _nll(scores: np.ndarray, targets: np.ndarray) -> float:
    if scores.size == 0:
        return float("nan")
    p = np.clip(scores, EPS, 1.0 - EPS)
    return float(-np.mean(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))


def _ece(scores: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width expected calibration error for binary probabilities."""
    if scores.size == 0:
        return float("nan")
    p = np.clip(scores.astype(np.float64), 0.0, 1.0)
    y = (targets > 0.5).astype(np.float64)
    bin_idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)
    populated = counts > 0
    conf = np.bincount(bin_idx, weights=p, minlength=n_bins)[populated] / counts[populated]
    acc = np.bincount(bin_idx, weights=y, minlength=n_bins)[populated] / counts[populated]
    return float(np.sum(counts[populated] / p.size * np.abs(conf - acc)))


def _sigmoid_np(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    scale = max(float(temperature), EPS)
    z = np.clip(logits.astype(np.float64, copy=False) / scale, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64, copy=False)


def _logit_nll(logits: np.ndarray, targets: np.ndarray, *, temperature: float = 1.0) -> float:
    if logits.size == 0:
        return float("nan")
    scale = max(float(temperature), EPS)
    z = logits.astype(np.float64, copy=False) / scale
    y = targets.astype(np.float64, copy=False)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit one scalar temperature on validation logits only.

    This is deliberately report-only: saved checkpoints and predictor outputs
    continue to use the raw logits/probabilities unless a future runtime plan
    explicitly opts into calibration.
    """
    if logits.size == 0:
        return 1.0
    x = logits.astype(np.float64, copy=False)
    y = targets.astype(np.float64, copy=False)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return 1.0

    def best_on(grid: np.ndarray) -> float:
        losses = np.array([_logit_nll(x, y, temperature=float(t)) for t in grid])
        return float(grid[int(np.nanargmin(losses))])

    coarse = np.exp(np.linspace(math.log(0.05), math.log(10.0), 161))
    best = best_on(coarse)
    half_step = (math.log(10.0) - math.log(0.05)) / 160.0
    fine = np.exp(np.linspace(math.log(best) - half_step, math.log(best) + half_step, 81))
    return best_on(fine)


def _seed_torch(seed: int, *, device: str) -> None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _batch_indices(
    n_rows: int,
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    indices = rng.permutation(n_rows) if shuffle else np.arange(n_rows)
    for start in range(0, n_rows, batch_size):
        yield indices[start : start + batch_size]


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return _project_relative(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _cache_raw_tensor_split(
    split_name: str,
    split: SplitData,
    *,
    device: str,
) -> RawTensorSplit:
    started = time.monotonic()

    def to_tensor(name: str, value: np.ndarray) -> torch.Tensor:
        dtype = torch.long if name in _LONG_TENSOR_FIELDS else torch.float32
        return torch.tensor(value, dtype=dtype, device=device)

    result = RawTensorSplit(
        **{
            f.name: (
                None
                if (value := getattr(split, f.name)) is None
                else to_tensor(f.name, value)
            )
            for f in fields(RawTensorSplit)
        }
    )
    if device == "cuda":
        torch.cuda.synchronize()
    logger.info(
        "Cached raw %s tensors n=%s device=%s seconds=%.2f",
        split_name,
        split.blue_win.size,
        device,
        time.monotonic() - started,
    )
    return result


def _raw_batch(raw: RawTensorSplit, rows: slice | torch.Tensor) -> RawTensorSplit:
    def take(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(rows, slice):
            return tensor[rows]
        return tensor.index_select(0, rows)

    return _map_split(raw, take)


def _evaluate_predictions(scores: np.ndarray, split: SplitData) -> dict[str, float | int]:
    targets = split.blue_win.astype(np.float64, copy=False)
    if targets.size == 0:
        return {"n": 0, "accuracy": float("nan"), "auc": float("nan"),
                "nll": float("nan"), "ece": float("nan"), "brier": float("nan")}
    return {
        "n": int(targets.size),
        "accuracy": float(np.mean((scores >= 0.5) == (targets > 0.5))),
        "auc": _binary_auc(scores, targets),
        "nll": _nll(scores, targets),
        "ece": _ece(scores, targets),
        "brier": float(np.mean((scores - targets) ** 2)),
    }


SUPPORT_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("zero", 0.0, 0.0),
    ("low_1_4", 1.0, 4.0),
    ("medium_5_49", 5.0, 49.0),
    ("high_50_plus", 50.0, math.inf),
)


CHECKPOINT_METRICS = frozenset(
    {
        "val_threshold_accuracy",
        "val_accuracy",
        "val_auc",
        "val_nll",
        "val_nll_ece",
    }
)


AUTO_HGNN_OVERRIDE_DIMS = frozenset(
    {
        "identity_semantic_dim",
        "identity_profile_dim",
        "m1v1_detail_dim",
        "identity_context_dim",
        "context_interpretable_dim",
        "identity_context_raw_dim",
    }
)


def _resolve_hgnn_overrides_from_meta(
    overrides: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in overrides.items():
        if value != "auto":
            resolved[key] = value
            continue
        if key not in AUTO_HGNN_OVERRIDE_DIMS:
            raise ValueError(f"{key} does not support auto HGNN override resolution")
        if key not in classification:
            raise ValueError(f"{key}=auto requires classification.{key} in cache metadata")
        resolved[key] = int(classification[key])
    return resolved


def _metric_values(scores: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    if targets.size == 0:
        return {"n": 0, "auc": float("nan"), "nll": float("nan"),
                "ece": float("nan"), "brier": float("nan"),
                "model_mean": float("nan"), "label_mean": float("nan"),
                "calibration_gap": float("nan")}
    model_mean = float(np.mean(scores))
    label_mean = float(np.mean(targets))
    return {
        "n": int(targets.size),
        "auc": _binary_auc(scores, targets),
        "nll": _nll(scores, targets),
        "ece": _ece(scores, targets),
        "brier": float(np.mean((scores - targets) ** 2)),
        "model_mean": model_mean,
        "label_mean": label_mean,
        "calibration_gap": model_mean - label_mean,
    }


def _min_non_missing_support(counts: np.ndarray) -> np.ndarray:
    positive = np.where(counts > 0.0, counts, np.inf)
    out = positive.min(axis=1)
    return np.where(np.isinf(out), 0.0, out)


def _bucket_rows(
    values: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for bucket, lo, hi in SUPPORT_BUCKETS:
        if math.isinf(hi):
            mask = values >= lo
        elif lo == hi:
            mask = values == lo
        else:
            mask = (values >= lo) & (values <= hi)
        bucket_scores = scores[mask]
        bucket_targets = targets[mask]
        row = _metric_values(bucket_scores, bucket_targets)
        row["mean_support"] = float(np.mean(values[mask])) if np.any(mask) else float("nan")
        rows[bucket] = row
    return rows


def _support_bucket_metrics(scores: np.ndarray, split: SplitData) -> dict[str, object]:
    targets = split.blue_win.astype(np.float64, copy=False)
    out: dict[str, object] = {
        "overall": _metric_values(scores, targets),
    }
    prior_support = _prior_1vx_support_metrics(scores, split)
    if prior_support is not None:
        out["prior_1vx_support"] = prior_support
    context_support = _identity_context_support_metrics(scores, split)
    if context_support is not None:
        out["identity_context_support"] = context_support
    return out


PRIOR_1VX_SUPPORT_RISK_BUCKETS: tuple[str, ...] = (
    "zero_player",
    "min_1_4",
    "min_5_49",
    "min_50_plus",
)


def _prior_1vx_support_arrays(
    split: SplitData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if split.p1_cnt is None:
        return None
    support = np.asarray(split.p1_cnt, dtype=np.float64)
    if support.ndim != 2 or support.shape[0] != split.blue_win.size:
        raise ValueError("p1_cnt must have shape [games, players] for diagnostics")
    mean_support = support.mean(axis=1)
    min_support = _min_non_missing_support(support)
    zero_players = (support <= 0.0).sum(axis=1).astype(np.float64, copy=False)
    return mean_support, min_support, zero_players


def _prior_1vx_support_bucket_ids(split: SplitData) -> np.ndarray | None:
    arrays = _prior_1vx_support_arrays(split)
    if arrays is None:
        return None
    _, min_support, zero_players = arrays
    bucket = np.full(min_support.shape, 3, dtype=np.int64)
    has_zero = zero_players > 0.0
    bucket[has_zero] = 0
    no_zero = ~has_zero
    bucket[no_zero & (min_support < 5.0)] = 1
    bucket[no_zero & (min_support >= 5.0) & (min_support < 50.0)] = 2
    return bucket


def _prior_support_risk_bucket_rows(
    bucket_ids: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    mean_support: np.ndarray,
    min_support: np.ndarray,
    zero_players: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for idx, label in enumerate(PRIOR_1VX_SUPPORT_RISK_BUCKETS):
        mask = bucket_ids == idx
        row = _metric_values(scores[mask], targets[mask])
        row["mean_1vx_support"] = (
            float(np.mean(mean_support[mask])) if np.any(mask) else float("nan")
        )
        row["min_1vx_support"] = (
            float(np.mean(min_support[mask])) if np.any(mask) else float("nan")
        )
        row["mean_zero_1vx_players"] = (
            float(np.mean(zero_players[mask])) if np.any(mask) else float("nan")
        )
        rows[label] = row
    return rows


def _prior_1vx_support_metrics(
    scores: np.ndarray,
    split: SplitData,
) -> dict[str, object] | None:
    arrays = _prior_1vx_support_arrays(split)
    if arrays is None:
        return None
    mean_support, min_support, zero_players = arrays
    targets = split.blue_win.astype(np.float64, copy=False)
    bucket_ids = _prior_1vx_support_bucket_ids(split)
    if bucket_ids is None:
        return None
    return {
        "mean_support_bucket": _bucket_rows(mean_support, scores, targets),
        "min_support_bucket": _bucket_rows(min_support, scores, targets),
        "risk_bucket": _prior_support_risk_bucket_rows(
            bucket_ids,
            scores,
            targets,
            mean_support=mean_support,
            min_support=min_support,
            zero_players=zero_players,
        ),
    }


CONTEXT_SUPPORT_RISK_BUCKETS: tuple[str, ...] = (
    "zero_player",
    "min_1_29",
    "mean_30_199",
    "mean_200_plus",
)


def _identity_context_support_arrays(
    split: SplitData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if split.identity_context_support is None:
        return None
    support = np.asarray(split.identity_context_support, dtype=np.float64)
    if support.ndim != 2 or support.shape[0] != split.blue_win.size:
        raise ValueError(
            "identity_context_support must have shape [games, players] for diagnostics"
        )
    mean_support = support.mean(axis=1)
    min_support = _min_non_missing_support(support)
    zero_players = (support <= 0.0).sum(axis=1).astype(np.float64, copy=False)
    return mean_support, min_support, zero_players


def _identity_context_support_bucket_ids(split: SplitData) -> np.ndarray | None:
    arrays = _identity_context_support_arrays(split)
    if arrays is None:
        return None
    mean_support, min_support, zero_players = arrays
    bucket = np.full(mean_support.shape, 3, dtype=np.int64)
    has_zero = zero_players > 0.0
    bucket[has_zero] = 0
    no_zero = ~has_zero
    bucket[no_zero & (min_support < 30.0)] = 1
    bucket[no_zero & (min_support >= 30.0) & (mean_support < 200.0)] = 2
    return bucket


def _context_support_risk_bucket_rows(
    bucket_ids: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    mean_support: np.ndarray,
    min_support: np.ndarray,
    zero_players: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for idx, label in enumerate(CONTEXT_SUPPORT_RISK_BUCKETS):
        mask = bucket_ids == idx
        row = _metric_values(scores[mask], targets[mask])
        row["mean_identity_context_support"] = (
            float(np.mean(mean_support[mask])) if np.any(mask) else float("nan")
        )
        row["min_identity_context_support"] = (
            float(np.mean(min_support[mask])) if np.any(mask) else float("nan")
        )
        row["mean_zero_context_players"] = (
            float(np.mean(zero_players[mask])) if np.any(mask) else float("nan")
        )
        rows[label] = row
    return rows


def _identity_context_support_metrics(
    scores: np.ndarray,
    split: SplitData,
) -> dict[str, object] | None:
    arrays = _identity_context_support_arrays(split)
    if arrays is None:
        return None
    mean_support, min_support, zero_players = arrays
    targets = split.blue_win.astype(np.float64, copy=False)
    bucket_ids = _identity_context_support_bucket_ids(split)
    if bucket_ids is None:
        return None
    return {
        "mean_support_bucket": _bucket_rows(mean_support, scores, targets),
        "min_support_bucket": _bucket_rows(min_support, scores, targets),
        "risk_bucket": _context_support_risk_bucket_rows(
            bucket_ids,
            scores,
            targets,
            mean_support=mean_support,
            min_support=min_support,
            zero_players=zero_players,
        ),
    }


def _fit_context_support_temperatures(
    logits: np.ndarray,
    targets: np.ndarray,
    bucket_ids: np.ndarray,
    *,
    min_bucket_size: int,
) -> tuple[list[dict[str, object]], dict[int, float], float]:
    global_temperature = _fit_temperature(logits, targets)
    rows: list[dict[str, object]] = []
    temperatures: dict[int, float] = {}
    y = targets.astype(np.float64, copy=False)
    for idx, label in enumerate(CONTEXT_SUPPORT_RISK_BUCKETS):
        mask = bucket_ids == idx
        n = int(mask.sum())
        has_two_classes = np.unique(y[mask] > 0.5).size == 2 if n else False
        if n >= min_bucket_size and has_two_classes:
            temperature = _fit_temperature(logits[mask], y[mask])
            source = "bucket_val"
        else:
            temperature = global_temperature
            source = "global_val_fallback"
        temperatures[idx] = temperature
        rows.append(
            {
                "bucket": label,
                "n_val": n,
                "temperature": temperature,
                "fit_source": source,
            }
        )
    return rows, temperatures, global_temperature


def _apply_context_support_temperatures(
    logits: np.ndarray,
    bucket_ids: np.ndarray,
    temperatures: dict[int, float],
) -> np.ndarray:
    scaled = np.empty_like(logits, dtype=np.float64)
    for idx in range(len(CONTEXT_SUPPORT_RISK_BUCKETS)):
        mask = bucket_ids == idx
        if np.any(mask):
            scaled[mask] = _sigmoid_np(logits[mask], temperature=temperatures[idx])
    return scaled


def _context_support_temperature_report(
    prediction_logits: dict[str, np.ndarray],
    splits: dict[str, SplitData],
    *,
    min_bucket_size: int,
) -> tuple[dict[str, object], dict[str, dict[str, float | int]]]:
    val_bucket_ids = _identity_context_support_bucket_ids(splits["val"])
    if val_bucket_ids is None:
        return {
            "available": False,
            "reason": "identity_context_support is unavailable",
            "fit_split": "val",
            "report_only": True,
        }, {}
    fit_rows, temperatures, global_temperature = _fit_context_support_temperatures(
        prediction_logits["val"],
        splits["val"].blue_win,
        val_bucket_ids,
        min_bucket_size=min_bucket_size,
    )
    report: dict[str, object] = {
        "available": True,
        "fit_split": "val",
        "report_only": True,
        "bucket_key": "identity_context_support.risk_bucket",
        "min_bucket_size": min_bucket_size,
        "global_temperature": global_temperature,
        "buckets": fit_rows,
    }
    split_reports: dict[str, dict[str, float | int]] = {}
    split_details: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        bucket_ids = _identity_context_support_bucket_ids(splits[split_name])
        if bucket_ids is None:
            continue
        scaled = _apply_context_support_temperatures(
            prediction_logits[split_name],
            bucket_ids,
            temperatures,
        )
        metrics = _evaluate_predictions(scaled, splits[split_name])
        support_metrics = _identity_context_support_metrics(scaled, splits[split_name])
        if support_metrics is not None:
            metrics["identity_context_support"] = support_metrics
        split_reports[split_name] = metrics
        split_details[split_name] = {
            "overall": metrics,
            "identity_context_support": support_metrics,
        }
    report["splits"] = split_details
    return report, split_reports


def _select_threshold(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    if scores.size == 0:
        return 0.5, float("nan")
    y = targets > 0.5
    grid = np.linspace(0.30, 0.70, 401)
    acc = ((scores[None, :] >= grid[:, None]) == y[None, :]).mean(axis=1)
    best = int(np.argmax(acc))
    return float(grid[best]), float(acc[best])


def _threshold_accuracy(scores: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    if scores.size == 0:
        return float("nan")
    return float(np.mean((scores >= threshold) == (targets > 0.5)))


def _checkpoint_score(
    metric: str,
    *,
    val_metrics: dict[str, float | int],
    val_threshold_accuracy: float,
) -> float:
    if metric == "val_threshold_accuracy":
        return val_threshold_accuracy
    if metric == "val_accuracy":
        return float(val_metrics["accuracy"])
    if metric == "val_auc":
        return float(val_metrics["auc"])
    if metric == "val_nll":
        return -float(val_metrics["nll"])
    if metric == "val_nll_ece":
        return -(float(val_metrics["nll"]) + float(val_metrics["ece"]))
    raise ValueError(
        f"checkpoint_metric must be one of: {', '.join(sorted(CHECKPOINT_METRICS))}"
    )


def _validate_train_config(train_cfg: TrainConfig) -> None:
    if train_cfg.checkpoint_metric not in CHECKPOINT_METRICS:
        raise ValueError(
            f"checkpoint_metric must be one of: {', '.join(sorted(CHECKPOINT_METRICS))}"
        )
    if train_cfg.context_support_calibration_min_bucket < 1:
        raise ValueError("context_support_calibration_min_bucket must be >= 1")
    if train_cfg.context_auxiliary_loss_weight < 0.0:
        raise ValueError("context_auxiliary_loss_weight must be >= 0")
    if train_cfg.auc_ranking_loss_weight < 0.0:
        raise ValueError("auc_ranking_loss_weight must be >= 0")
    if train_cfg.auc_ranking_loss_pairs < 1:
        raise ValueError("auc_ranking_loss_pairs must be >= 1")


def _hgnn_config_from_meta(
    meta: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> HGNNConfig:
    classification = meta.get("classification", {}) if isinstance(meta, dict) else {}
    # Production classification context is the threshold-tuned raw-atlas
    # identity-conditioned head. It is the first deliberately naive semantic
    # context implementation: give every (champion, role, build) a draft-safe raw
    # semantic descriptor, then let a low-rank bottleneck learn which ally/enemy
    # context interactions matter for that identity. The shared 24-dim atlas head
    # remains available as an explicit baseline, but production uses the raw head
    # because the win-rate prior marginalises over enemy/ally composition and the
    # shared head under-fits identity-specific tails (armor tank vs physical
    # enemy, MR tank vs magic enemy, low-damage team vs enemy heal/shield, etc.).
    # The head is antisymmetric, support-gated, and zero-initialised, so it is
    # opt-in on top of the win-rate model.
    base = dict(
        n_champions=int(meta["n_champions"]),
        n_builds=int(meta["n_builds"]),
        build_vocab=tuple(meta["build_vocab"]),
        identity_semantic_dim=0,
        identity_profile_dim=0,
        m1v1_detail_dim=0,
        use_relationship_integrations=False,
        identity_context_dim=int(classification.get("identity_context_dim", 0)),
        context_interpretable_dim=int(classification.get("context_interpretable_dim", 14)),
        context_head_hidden=(32,),
        context_support_strength=30.0,
        context_include_ally=True,
        context_include_relational=True,
        identity_context_raw_dim=int(classification.get("identity_context_raw_dim", 0)),
        use_identity_conditioned_context=True,
        identity_context_conditioning_type="low_rank",
        identity_context_source="raw",
        identity_context_rank=16,
        identity_context_hidden_dim=64,
        identity_context_emb_dim=16,
        identity_context_init_scale=0.01,
        identity_context_dropout=0.0,
        identity_context_use_residual_mlp=False,
        identity_context_include_products=False,
        identity_context_include_support_features=False,
    )
    if overrides:
        base.update(_resolve_hgnn_overrides_from_meta(overrides, classification))
    return HGNNConfig(**base)


def _hgnn_inputs_from_raw(
    raw: RawTensorSplit,
    *,
    strength: float,
    device: str,
) -> dict[str, torch.Tensor]:
    if raw.champion_id is None or raw.build_id is None:
        raise ValueError("HGNN inputs require champion_id/build_id; rebuild the cache (v17).")
    include_relationship_features = (
        raw.matchup_1v1 is not None
        and raw.synergy_2vx is not None
        and raw.m1v1_cnt is not None
        and raw.s2vx_cnt is not None
    )
    return build_hgnn_inputs(
        champion_id=raw.champion_id,
        build_id=raw.build_id,
        win_rate=raw.win_rate,
        p1_cnt=raw.p1_cnt,
        strength=strength,
        matchup_1v1=raw.matchup_1v1,
        synergy_2vx=raw.synergy_2vx,
        m1v1_cnt=raw.m1v1_cnt,
        s2vx_cnt=raw.s2vx_cnt,
        include_relationship_features=include_relationship_features,
        identity_semantic=raw.identity_semantic,
        identity_profile=raw.identity_profile,
        identity_context=raw.identity_context,
        identity_context_support=raw.identity_context_support,
        identity_context_raw=raw.identity_context_raw,
        m1v1_detail=raw.m1v1_detail,
        device=device,
    )


def _predict_hgnn_logits(
    model: HGNNWinModel,
    split: RawTensorSplit,
    *,
    batch_size: int,
    strength: float,
    device: str,
) -> np.ndarray:
    return _predict_hgnn_outputs(
        model,
        split,
        batch_size=batch_size,
        strength=strength,
        device=device,
    )["final_logit"]


def _predict_hgnn_outputs(
    model: HGNNWinModel,
    split: RawTensorSplit,
    *,
    batch_size: int,
    strength: float,
    device: str,
) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {
        "final_logit": [],
        "base_logit": [],
        "context_logit": [],
    }
    with torch.no_grad():
        n_rows = split.blue_win.numel()
        for start in range(0, n_rows, batch_size):
            raw_batch = _raw_batch(split, slice(start, start + batch_size))
            inputs = _hgnn_inputs_from_raw(raw_batch, strength=strength, device=device)
            outputs = model(**inputs)
            for key in out:
                value = outputs.get(key)
                if value is None:
                    value = outputs["final_logit"].new_zeros(outputs["final_logit"].shape)
                out[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(values).astype(np.float64) for key, values in out.items()}


def _context_residual_metrics(context_logits: np.ndarray) -> dict[str, float]:
    values = np.asarray(context_logits, dtype=np.float64)
    abs_values = np.abs(values)
    if values.size == 0:
        return {
            "mean_logit": float("nan"),
            "mean_abs_logit": float("nan"),
            "rms_logit": float("nan"),
            "p95_abs_logit": float("nan"),
            "max_abs_logit": float("nan"),
        }
    return {
        "mean_logit": float(values.mean()),
        "mean_abs_logit": float(abs_values.mean()),
        "rms_logit": float(np.sqrt(np.mean(np.square(values)))),
        "p95_abs_logit": float(np.quantile(abs_values, 0.95)),
        "max_abs_logit": float(abs_values.max()),
    }


def _context_auxiliary_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_fn: nn.Module,
    weight: float,
) -> torch.Tensor:
    if weight <= 0.0:
        return outputs["final_logit"].new_zeros(())
    context_logit = outputs.get("context_logit")
    base_logit = outputs.get("base_logit")
    if context_logit is None or base_logit is None:
        return outputs["final_logit"].new_zeros(())
    context_only_logit = base_logit.detach() + context_logit
    return float(weight) * loss_fn(context_only_logit, labels)


def _auc_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight: float,
    max_pairs: int,
) -> torch.Tensor:
    if weight <= 0.0:
        return logits.new_zeros(())
    positives = logits[labels > 0.5]
    negatives = logits[labels <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.new_zeros(())

    n_all_pairs = positives.numel() * negatives.numel()
    if n_all_pairs <= max_pairs:
        margins = positives[:, None] - negatives[None, :]
    else:
        pos_idx = torch.randint(positives.numel(), (int(max_pairs),), device=logits.device)
        neg_idx = torch.randint(negatives.numel(), (int(max_pairs),), device=logits.device)
        margins = positives[pos_idx] - negatives[neg_idx]
    return float(weight) * torch.nn.functional.softplus(-margins).mean()


def _predict_hgnn(
    model: HGNNWinModel,
    split: RawTensorSplit,
    *,
    batch_size: int,
    strength: float,
    device: str,
) -> np.ndarray:
    return _sigmoid_np(
        _predict_hgnn_logits(
            model,
            split,
            batch_size=batch_size,
            strength=strength,
            device=device,
        )
    )


def train(
    dataset_cfg: DatasetConfig | None = None,
    train_cfg: TrainConfig | None = None,
    *,
    model_overrides: dict[str, Any] | None = None,
) -> Path:
    dataset_cfg = dataset_cfg or DatasetConfig()
    train_cfg = train_cfg or TrainConfig()
    _validate_train_config(train_cfg)
    device = resolve_device(train_cfg.device)
    _seed_torch(train_cfg.seed, device=device)
    started = time.monotonic()
    # The Beta-posterior variance strength reused for the HGNN confidence gate.
    strength = dataset_cfg.confidence_gate_strength
    # Cap the training batch because each step also runs a team-swapped copy.
    train_batch_size = min(train_cfg.batch_size, HGNN_TRAIN_BATCH)

    meta = identity_meta(dataset_cfg)
    model_config = _hgnn_config_from_meta(meta, overrides=model_overrides)

    splits = {
        name: _drop_unused_model_arrays(_limit_split(split, dataset_cfg.max_games), model_config)
        for name, split in load_splits(dataset_cfg, require_counts=True).items()
    }
    if splits["train"].blue_win.size == 0:
        raise ValueError("Training split is empty; rebuild the cache with train games.")
    tensor_splits = {
        name: _cache_raw_tensor_split(name, splits[name], device=device)
        for name in ("train", "val")
    }

    model = HGNNWinModel(model_config).to(device)
    if train_cfg.context_auxiliary_loss_weight > 0.0 and not (
        model.identity_conditioned_context_enabled or model.context_enabled
    ):
        raise ValueError("context_auxiliary_loss_weight requires an enabled context head")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(train_cfg.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_val_nll = math.inf
    best_checkpoint_val_nll = math.inf
    best_checkpoint_val_ece = math.inf
    best_checkpoint_score = -math.inf
    best_epoch = 0
    best_threshold = 0.5
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    logger.info(
        "HGNN training device=%s batch_size=%s max_epochs=%s strength=%s checkpoint_metric=%s min_delta=%s",
        device,
        train_batch_size,
        train_cfg.max_epochs,
        strength,
        train_cfg.checkpoint_metric,
        train_cfg.checkpoint_min_delta,
    )
    if device == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_aux_loss_sum = 0.0
        train_rank_loss_sum = 0.0
        train_seen = 0
        for batch_idx in _batch_indices(
            splits["train"].blue_win.size,
            batch_size=train_batch_size,
            shuffle=True,
            rng=rng,
        ):
            raw_batch = _raw_batch(
                tensor_splits["train"],
                torch.as_tensor(batch_idx, dtype=torch.long, device=device),
            )
            inputs = _hgnn_inputs_from_raw(raw_batch, strength=strength, device=device)
            labels = raw_batch.blue_win
            # Team-swap augmentation (design §8/§9): train on the match and its
            # mirror with the flipped label, enforcing approximate antisymmetry.
            optimizer.zero_grad(set_to_none=True)
            direct_outputs = model(**inputs)
            direct_loss = loss_fn(direct_outputs["final_logit"], labels)
            direct_aux_loss = _context_auxiliary_loss(
                direct_outputs,
                labels,
                loss_fn,
                train_cfg.context_auxiliary_loss_weight,
            )
            direct_rank_loss = _auc_ranking_loss(
                direct_outputs["final_logit"],
                labels,
                weight=train_cfg.auc_ranking_loss_weight,
                max_pairs=train_cfg.auc_ranking_loss_pairs,
            )
            (0.5 * (direct_loss + direct_aux_loss + direct_rank_loss)).backward()
            swapped_outputs = model(**swap_hgnn_inputs(inputs))
            swapped_loss = loss_fn(swapped_outputs["final_logit"], 1.0 - labels)
            swapped_aux_loss = _context_auxiliary_loss(
                swapped_outputs,
                1.0 - labels,
                loss_fn,
                train_cfg.context_auxiliary_loss_weight,
            )
            swapped_rank_loss = _auc_ranking_loss(
                swapped_outputs["final_logit"],
                1.0 - labels,
                weight=train_cfg.auc_ranking_loss_weight,
                max_pairs=train_cfg.auc_ranking_loss_pairs,
            )
            (0.5 * (swapped_loss + swapped_aux_loss + swapped_rank_loss)).backward()
            reg_loss = model.context_regularization_loss()
            if reg_loss.requires_grad:
                reg_loss.backward()
            loss = 0.5 * (direct_loss.detach() + swapped_loss.detach())
            aux_loss = 0.5 * (direct_aux_loss.detach() + swapped_aux_loss.detach())
            rank_loss = 0.5 * (direct_rank_loss.detach() + swapped_rank_loss.detach())
            if train_cfg.max_grad_norm is not None and train_cfg.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            train_loss_sum += float(loss.cpu().item()) * labels.numel() * 2
            train_aux_loss_sum += float(aux_loss.cpu().item()) * labels.numel() * 2
            train_rank_loss_sum += float(rank_loss.cpu().item()) * labels.numel() * 2
            train_seen += int(labels.numel() * 2)

        val_logits = _predict_hgnn_logits(
            model,
            tensor_splits["val"],
            batch_size=train_cfg.batch_size,
            strength=strength,
            device=device,
        )
        val_predictions = _sigmoid_np(val_logits)
        train_nll = train_loss_sum / max(train_seen, 1)
        train_context_auxiliary_loss = train_aux_loss_sum / max(train_seen, 1)
        train_auc_ranking_loss = train_rank_loss_sum / max(train_seen, 1)
        val_metrics = _evaluate_predictions(val_predictions, splits["val"])
        val_nll = float(val_metrics["nll"])
        val_threshold, val_threshold_accuracy = _select_threshold(
            val_predictions,
            splits["val"].blue_win,
        )
        checkpoint_score = _checkpoint_score(
            train_cfg.checkpoint_metric,
            val_metrics=val_metrics,
            val_threshold_accuracy=val_threshold_accuracy,
        )
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_auc": float(val_metrics["auc"]),
                "val_ece": float(val_metrics["ece"]),
                "val_threshold": val_threshold,
                "val_threshold_accuracy": val_threshold_accuracy,
                "checkpoint_score": checkpoint_score,
                "train_context_auxiliary_loss": train_context_auxiliary_loss,
                "train_auc_ranking_loss": train_auc_ranking_loss,
            }
        )
        logger.info(
            "epoch=%s train_nll=%.5f aux=%.5f rank=%.5f val_nll=%.5f val_acc=%.4f val_thr=%.3f val_thr_acc=%.4f",
            epoch,
            train_nll,
            train_context_auxiliary_loss,
            train_auc_ranking_loss,
            val_nll,
            val_metrics["accuracy"],
            val_threshold,
            val_threshold_accuracy,
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
        if checkpoint_score > best_checkpoint_score + train_cfg.checkpoint_min_delta:
            best_checkpoint_score = checkpoint_score
            best_checkpoint_val_nll = val_nll
            best_checkpoint_val_ece = float(val_metrics["ece"])
            best_epoch = epoch
            best_threshold = val_threshold
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= train_cfg.patience:
                break

    model.load_state_dict(best_state)
    save_hgnn_model(train_cfg.model_path, model, confidence_strength=strength)

    tensor_splits["test"] = _cache_raw_tensor_split("test", splits["test"], device=device)
    prediction_outputs = {
        split_name: _predict_hgnn_outputs(
            model,
            tensor_split,
            batch_size=train_cfg.batch_size,
            strength=strength,
            device=device,
        )
        for split_name, tensor_split in tensor_splits.items()
    }
    prediction_logits = {
        split_name: outputs["final_logit"]
        for split_name, outputs in prediction_outputs.items()
    }
    temperature = _fit_temperature(prediction_logits["val"], splits["val"].blue_win)
    predictions = {
        split_name: _sigmoid_np(logits)
        for split_name, logits in prediction_logits.items()
    }
    calibrated_predictions = {
        split_name: _sigmoid_np(logits, temperature=temperature)
        for split_name, logits in prediction_logits.items()
    }
    split_metrics = {
        split_name: _evaluate_predictions(predictions[split_name], splits[split_name])
        for split_name in ("train", "val", "test")
    }
    for split_name in ("train", "val", "test"):
        split_metrics[split_name]["threshold_accuracy"] = _threshold_accuracy(
            predictions[split_name],
            splits[split_name].blue_win,
            best_threshold,
        )
        split_metrics[split_name]["support_buckets"] = _support_bucket_metrics(
            predictions[split_name],
            splits[split_name],
        )
        split_metrics[split_name]["context_residual"] = _context_residual_metrics(
            prediction_outputs[split_name]["context_logit"]
        )
        calibrated = _evaluate_predictions(calibrated_predictions[split_name], splits[split_name])
        calibrated["support_buckets"] = _support_bucket_metrics(
            calibrated_predictions[split_name],
            splits[split_name],
        )
        split_metrics[split_name]["temperature_scaled"] = calibrated
    context_support_temperature_scaling: dict[str, object] = {
        "available": False,
        "report_only": True,
        "reason": "disabled",
    }
    if train_cfg.report_context_support_calibration:
        (
            context_support_temperature_scaling,
            context_support_scaled_metrics,
        ) = _context_support_temperature_report(
            prediction_logits,
            splits,
            min_bucket_size=train_cfg.context_support_calibration_min_bucket,
        )
        for split_name, calibrated in context_support_scaled_metrics.items():
            split_metrics[split_name]["context_support_temperature_scaled"] = calibrated
    metrics = {
        "model_type": "hgnn",
        "dataset_config": asdict(dataset_cfg),
        "train_config": asdict(train_cfg),
        "model_config": asdict(model_config),
        "model_path": train_cfg.model_path,
        "metrics_path": train_cfg.metrics_path,
        "device": device,
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "best_checkpoint_val_nll": best_checkpoint_val_nll,
        "best_checkpoint_val_ece": best_checkpoint_val_ece,
        "best_checkpoint_score": best_checkpoint_score,
        "decision_threshold": best_threshold,
        "temperature_scaling": {
            "temperature": temperature,
            "fit_split": "val",
            "report_only": True,
        },
        "context_support_temperature_scaling": context_support_temperature_scaling,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "train": split_metrics["train"],
        "val": split_metrics["val"],
        "test": split_metrics["test"],
    }
    _write_metrics(train_cfg.metrics_path, metrics)

    logger.info("Saved HGNN model: %s", _project_relative(train_cfg.model_path))
    logger.info("Saved metrics: %s", _project_relative(train_cfg.metrics_path))
    for split_name in ("train", "val", "test"):
        m = metrics[split_name]
        if isinstance(m, dict):
            logger.info(
                "%s n=%s acc=%.4f thr_acc=%.4f auc=%.4f nll=%.4f ece=%.4f brier=%.4f",
                split_name,
                m["n"],
                m["accuracy"],
                m["threshold_accuracy"],
                m["auc"],
                m["nll"],
                m["ece"],
                m["brier"],
            )

    return train_cfg.model_path


def _parse_quantiles(value: str) -> tuple[float, ...]:
    if not value:
        return ()
    return tuple(float(part) for part in value.split(",") if part.strip())


def _model_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """HGNNConfig overrides for production semantic context or its baselines."""
    conditioning_type = "none" if args.shared_context else args.identity_context_conditioning_type
    return {
        "use_identity_conditioned_context": conditioning_type != "none",
        "identity_context_conditioning_type": conditioning_type,
        "identity_context_source": args.identity_context_source,
        "identity_context_rank": args.identity_context_rank,
        "identity_context_hidden_dim": args.identity_context_hidden_dim,
        "identity_context_emb_dim": args.identity_context_emb_dim,
        "identity_context_init_scale": args.identity_context_init_scale,
        "identity_context_dropout": args.identity_context_dropout,
        "identity_context_use_residual_mlp": args.identity_context_residual_mlp,
        "identity_context_include_products": args.identity_context_products,
        "identity_context_film_regularization": args.identity_context_film_regularization,
        "context_set_encoder_type": args.context_set_encoder,
        "context_set_encoder_dim": args.context_set_encoder_dim,
        "context_set_encoder_heads": args.context_set_encoder_heads,
        "context_summary_topk": args.context_summary_topk,
        "context_summary_quantiles": _parse_quantiles(args.context_summary_quantiles),
        "structural_antisymmetry": args.structural_antisymmetry,
        "structural_antisymmetry_scale": args.structural_antisymmetry_scale,
    }


def _parse_args() -> tuple[DatasetConfig, TrainConfig, dict[str, Any]]:
    dataset_defaults = DatasetConfig()
    train_defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=dataset_defaults.cache_dir)
    parser.add_argument("--max-games", type=int, default=dataset_defaults.max_games)
    parser.add_argument("--model-path", type=Path, default=train_defaults.model_path)
    parser.add_argument("--metrics-path", type=Path, default=train_defaults.metrics_path)
    parser.add_argument("--batch-size", type=int, default=train_defaults.batch_size)
    parser.add_argument("--max-epochs", type=int, default=train_defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=train_defaults.patience)
    parser.add_argument("--learning-rate", type=float, default=train_defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=train_defaults.weight_decay)
    parser.add_argument("--device", default=train_defaults.device)
    parser.add_argument("--seed", type=int, default=train_defaults.seed)
    parser.add_argument("--max-grad-norm", type=float, default=train_defaults.max_grad_norm)
    parser.add_argument("--checkpoint-metric", default=train_defaults.checkpoint_metric)
    parser.add_argument(
        "--checkpoint-min-delta",
        type=float,
        default=train_defaults.checkpoint_min_delta,
    )
    parser.add_argument(
        "--report-context-support-calibration",
        action="store_true",
        help=(
            "Fit validation-only temperature diagnostics by identity-context "
            "support bucket; report-only, does not affect served probabilities."
        ),
    )
    parser.add_argument(
        "--context-support-calibration-min-bucket",
        type=int,
        default=train_defaults.context_support_calibration_min_bucket,
    )
    parser.add_argument(
        "--context-auxiliary-loss-weight",
        type=float,
        default=train_defaults.context_auxiliary_loss_weight,
        help=(
            "Experimental training-only weight for BCE on detached base_logit + "
            "context_logit, so gradients from this term update only the context residual."
        ),
    )
    parser.add_argument(
        "--auc-ranking-loss-weight",
        type=float,
        default=train_defaults.auc_ranking_loss_weight,
        help=(
            "Experimental training-only weight for a sampled positive/negative "
            "pairwise ranking loss that directly targets validation AUC."
        ),
    )
    parser.add_argument(
        "--auc-ranking-loss-pairs",
        type=int,
        default=train_defaults.auc_ranking_loss_pairs,
        help="Maximum sampled positive/negative pairs per direct or swapped batch.",
    )
    parser.add_argument(
        "--identity-conditioned",
        action="store_true",
        help=(
            "Deprecated no-op: the low-rank identity-conditioned raw context "
            "head is now the production default."
        ),
    )
    parser.add_argument(
        "--shared-context",
        action="store_true",
        help="Use the legacy shared 24-dim context-atlas head instead of production semantic context.",
    )
    parser.add_argument(
        "--context-set-encoder",
        default="mean",
        choices=("mean", "deepsets", "set_transformer", "attention", "summary_stats"),
        help="Permutation-invariant encoder for unordered ally/enemy context sets.",
    )
    parser.add_argument("--context-set-encoder-dim", type=int, default=32)
    parser.add_argument("--context-set-encoder-heads", type=int, default=4)
    parser.add_argument("--context-summary-topk", type=int, default=2)
    parser.add_argument("--context-summary-quantiles", default="0.25,0.5,0.75")
    parser.add_argument(
        "--identity-context-conditioning-type",
        default="low_rank",
        choices=("none", "low_rank", "film"),
    )
    parser.add_argument(
        "--identity-context-source",
        default="raw",
        choices=("raw", "raw_plus_dense"),
    )
    parser.add_argument("--identity-context-rank", type=int, default=16)
    parser.add_argument("--identity-context-hidden-dim", type=int, default=64)
    parser.add_argument("--identity-context-emb-dim", type=int, default=16)
    parser.add_argument("--identity-context-init-scale", type=float, default=0.01)
    parser.add_argument("--identity-context-dropout", type=float, default=0.0)
    parser.add_argument("--identity-context-residual-mlp", action="store_true")
    parser.add_argument(
        "--identity-context-products",
        action="store_true",
        help=(
            "Append global interpretable context products to the "
            "identity-conditioned context projector."
        ),
    )
    parser.add_argument("--identity-context-film-regularization", type=float, default=1.0e-3)
    parser.add_argument("--structural-antisymmetry", action="store_true")
    parser.add_argument("--structural-antisymmetry-scale", type=float, default=0.5)
    args = parser.parse_args()
    return (
        DatasetConfig(cache_dir=args.cache_dir, max_games=args.max_games),
        TrainConfig(
            model_path=args.model_path,
            metrics_path=args.metrics_path,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
            max_grad_norm=args.max_grad_norm,
            checkpoint_metric=args.checkpoint_metric,
            checkpoint_min_delta=args.checkpoint_min_delta,
            report_context_support_calibration=args.report_context_support_calibration,
            context_support_calibration_min_bucket=args.context_support_calibration_min_bucket,
            context_auxiliary_loss_weight=args.context_auxiliary_loss_weight,
            auc_ranking_loss_weight=args.auc_ranking_loss_weight,
            auc_ranking_loss_pairs=args.auc_ranking_loss_pairs,
        ),
        _model_overrides_from_args(args),
    )


def main() -> None:
    dataset_cfg, train_cfg, model_overrides = _parse_args()
    train(dataset_cfg, train_cfg, model_overrides=model_overrides)


if __name__ == "__main__":
    main()
