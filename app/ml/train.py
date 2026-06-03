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
    identity_static_sidecar: torch.Tensor | None = None
    identity_full_game_sidecar: torch.Tensor | None = None
    identity_temporal_sidecar: torch.Tensor | None = None
    identity_encoder_support: torch.Tensor | None = None


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
    """Null out optional relationship arrays when the configured model ignores them."""
    sidecar_enabled = (
        config.use_identity_static_sidecar
        or config.use_identity_full_game_sidecar
        or config.use_identity_temporal_sidecar
        or config.use_identity_semantic_context_head
    )
    context_enabled = bool(config.use_identity_semantic_context_head)
    if context_enabled:
        missing = [
            name
            for name in (
                "identity_static_sidecar",
                "identity_full_game_sidecar",
                "identity_temporal_sidecar",
                "identity_encoder_support",
            )
            if getattr(split, name) is None
        ]
        if missing:
            raise ValueError(
                "semantic context head requires cache arrays: "
                + ", ".join(missing)
                + ". Rebuild the dataset cache with encoder_sidecar_path set "
                "to a valid three-latent sidecar artifact."
            )
        for name in (
            "identity_static_sidecar",
            "identity_full_game_sidecar",
            "identity_temporal_sidecar",
        ):
            value = getattr(split, name)
            if value.ndim != 3 or value.shape[1] != 10 or value.shape[2] <= 0:
                raise ValueError(f"semantic context head requires non-empty {name} [games, 10, dim]")
        support = split.identity_encoder_support
        if support.ndim != 2 or support.shape[1] != 10:
            raise ValueError("semantic context head requires identity_encoder_support [games, 10]")
    drop: dict[str, bool] = {
        "matchup_1v1": not config.use_relationship_integrations,
        "synergy_2vx": not config.use_relationship_integrations,
        "m1v1_cnt": not config.use_relationship_integrations,
        "s2vx_cnt": not config.use_relationship_integrations,
        "m1v1_eff_n": not config.use_relationship_integrations,
        "s2vx_eff_n": not config.use_relationship_integrations,
        "identity_static_sidecar": not (config.use_identity_static_sidecar or context_enabled),
        "identity_full_game_sidecar": not (config.use_identity_full_game_sidecar or context_enabled),
        "identity_temporal_sidecar": not (config.use_identity_temporal_sidecar or context_enabled),
        "identity_encoder_support": not sidecar_enabled,
    }
    overrides = {name: None for name, unused in drop.items() if unused}
    if not overrides:
        return split
    return type(split)(
        **{f.name: overrides.get(f.name, getattr(split, f.name)) for f in fields(split)}
    )


def _validate_split_targets(splits: dict[str, SplitData]) -> None:
    for split_name in ("train", "val", "test"):
        labels = np.asarray(splits[split_name].blue_win)
        if labels.ndim != 1:
            raise ValueError(
                f"{split_name} split blue_win labels must be one-dimensional; "
                "rebuild the dataset cache."
            )
        if labels.size == 0:
            continue
        unique = np.unique(labels)
        if not np.isin(unique, [0.0, 1.0]).all():
            raise ValueError(
                f"{split_name} split blue_win labels must be binary; "
                "rebuild the dataset cache."
            )
        positives = int(np.count_nonzero(labels > 0.5))
        negatives = int(labels.size - positives)
        if positives == 0 or negatives == 0:
            raise ValueError(
                f"{split_name} split has degenerate blue_win labels "
                f"(positives={positives}, negatives={negatives}, n={labels.size}); "
                "rebuild the dataset cache. This usually means the cache split "
                "metadata/ranges do not match the array contents."
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


def _resolve_hgnn_overrides_from_meta(
    overrides: dict[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in overrides.items():
        if value != "auto":
            resolved[key] = value
            continue
        raise ValueError(f"{key} does not support auto HGNN override resolution")
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
    if train_cfg.auc_ranking_loss_weight < 0.0:
        raise ValueError("auc_ranking_loss_weight must be >= 0")
    if train_cfg.auc_ranking_loss_pairs < 1:
        raise ValueError("auc_ranking_loss_pairs must be >= 1")


def _hgnn_config_from_meta(
    meta: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> HGNNConfig:
    base = dict(
        n_champions=int(meta["n_champions"]),
        n_builds=int(meta["n_builds"]),
        build_vocab=tuple(meta["build_vocab"]),
        use_relationship_integrations=False,
    )
    sidecar = meta.get("identity_encoder_sidecar")
    if isinstance(sidecar, dict):
        dims = sidecar.get("dims", {})
        if isinstance(dims, dict):
            base.update(
                {
                    "identity_static_sidecar_dim": int(dims.get("static", 0)),
                    "identity_full_game_sidecar_dim": int(dims.get("full_game", 0)),
                    "identity_temporal_sidecar_dim": int(dims.get("temporal", 0)),
                }
            )
    if overrides:
        base.update(_resolve_hgnn_overrides_from_meta(overrides))
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
        identity_static_sidecar=raw.identity_static_sidecar,
        identity_full_game_sidecar=raw.identity_full_game_sidecar,
        identity_temporal_sidecar=raw.identity_temporal_sidecar,
        identity_encoder_support=raw.identity_encoder_support,
        include_relationship_features=include_relationship_features,
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
        "base_logit": [],
        "context_logit": [],
        "final_logit": [],
    }
    with torch.no_grad():
        n_rows = split.blue_win.numel()
        for start in range(0, n_rows, batch_size):
            raw_batch = _raw_batch(split, slice(start, start + batch_size))
            inputs = _hgnn_inputs_from_raw(raw_batch, strength=strength, device=device)
            outputs = model(**inputs)
            for key in out:
                value = outputs[key]
                out[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(values).astype(np.float64) for key, values in out.items()}


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
    _validate_split_targets(splits)
    if splits["train"].blue_win.size == 0:
        raise ValueError("Training split is empty; rebuild the cache with train games.")
    tensor_splits = {
        name: _cache_raw_tensor_split(name, splits[name], device=device)
        for name in ("train", "val")
    }

    model = HGNNWinModel(model_config).to(device)
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
            direct_rank_loss = _auc_ranking_loss(
                direct_outputs["final_logit"],
                labels,
                weight=train_cfg.auc_ranking_loss_weight,
                max_pairs=train_cfg.auc_ranking_loss_pairs,
            )
            (0.5 * (direct_loss + direct_rank_loss)).backward()
            swapped_outputs = model(**swap_hgnn_inputs(inputs))
            swapped_loss = loss_fn(swapped_outputs["final_logit"], 1.0 - labels)
            swapped_rank_loss = _auc_ranking_loss(
                swapped_outputs["final_logit"],
                1.0 - labels,
                weight=train_cfg.auc_ranking_loss_weight,
                max_pairs=train_cfg.auc_ranking_loss_pairs,
            )
            (0.5 * (swapped_loss + swapped_rank_loss)).backward()
            loss = 0.5 * (direct_loss.detach() + swapped_loss.detach())
            rank_loss = 0.5 * (direct_rank_loss.detach() + swapped_rank_loss.detach())
            if train_cfg.max_grad_norm is not None and train_cfg.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            train_loss_sum += float(loss.cpu().item()) * labels.numel() * 2
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
                "train_auc_ranking_loss": train_auc_ranking_loss,
            }
        )
        logger.info(
            "epoch=%s train_nll=%.5f rank=%.5f val_nll=%.5f val_acc=%.4f val_thr=%.3f val_thr_acc=%.4f",
            epoch,
            train_nll,
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
        calibrated = _evaluate_predictions(calibrated_predictions[split_name], splits[split_name])
        calibrated["support_buckets"] = _support_bucket_metrics(
            calibrated_predictions[split_name],
            splits[split_name],
        )
        split_metrics[split_name]["temperature_scaled"] = calibrated
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


def _model_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """HGNNConfig overrides exposed by the training CLI."""
    use_all_sidecars = bool(args.use_all_identity_sidecars)
    return {
        "structural_antisymmetry": args.structural_antisymmetry,
        "structural_antisymmetry_scale": args.structural_antisymmetry_scale,
        "use_identity_static_sidecar": bool(args.use_identity_static_sidecar or use_all_sidecars),
        "use_identity_full_game_sidecar": bool(args.use_identity_full_game_sidecar or use_all_sidecars),
        "use_identity_temporal_sidecar": bool(args.use_identity_temporal_sidecar or use_all_sidecars),
        "identity_encoder_sidecar_support_strength": args.identity_encoder_sidecar_support_strength,
        "identity_encoder_sidecar_dropout": args.identity_encoder_sidecar_dropout,
        "use_identity_semantic_context_head": args.use_identity_semantic_context_head,
        "semantic_context_dim": args.semantic_context_dim,
        "semantic_context_dropout": args.semantic_context_dropout,
        "semantic_context_support_strength": args.semantic_context_support_strength,
    }


def _parse_args() -> tuple[DatasetConfig, TrainConfig, dict[str, Any]]:
    dataset_defaults = DatasetConfig()
    train_defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=dataset_defaults.cache_dir)
    parser.add_argument("--max-games", type=int, default=dataset_defaults.max_games)
    parser.add_argument("--encoder-sidecar-path", type=Path, default=dataset_defaults.encoder_sidecar_path)
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
    parser.add_argument("--structural-antisymmetry", action="store_true")
    parser.add_argument("--structural-antisymmetry-scale", type=float, default=0.5)
    parser.add_argument("--use-identity-static-sidecar", action="store_true")
    parser.add_argument("--use-identity-full-game-sidecar", action="store_true")
    parser.add_argument("--use-identity-temporal-sidecar", action="store_true")
    parser.add_argument(
        "--use-all-identity-sidecars",
        action="store_true",
        help="Enable static, full-game, and temporal frozen encoder sidecar blocks.",
    )
    parser.add_argument("--identity-encoder-sidecar-support-strength", type=float, default=30.0)
    parser.add_argument("--identity-encoder-sidecar-dropout", type=float, default=0.0)
    parser.add_argument(
        "--use-identity-semantic-context-head",
        action="store_true",
        help="Enable own/ally/enemy latent-context interaction over all three identity sidecars.",
    )
    parser.add_argument("--semantic-context-dim", type=int, default=96)
    parser.add_argument("--semantic-context-dropout", type=float, default=0.0)
    parser.add_argument("--semantic-context-support-strength", type=float, default=30.0)
    args = parser.parse_args()
    return (
        DatasetConfig(
            cache_dir=args.cache_dir,
            max_games=args.max_games,
            encoder_sidecar_path=args.encoder_sidecar_path,
        ),
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
