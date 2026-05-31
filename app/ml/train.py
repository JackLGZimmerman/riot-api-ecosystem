# pyright: reportPrivateImportUsage=false

"""Train the production HGNN win-rate model.

Run with:
    python -m app.ml.train
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

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
    matchup_1v1: torch.Tensor
    synergy_2vx: torch.Tensor
    p1_cnt: torch.Tensor
    m1v1_cnt: torch.Tensor
    s2vx_cnt: torch.Tensor
    m1v1_eff_n: torch.Tensor
    s2vx_eff_n: torch.Tensor
    blue_win: torch.Tensor
    champion_id: torch.Tensor | None = None
    build_id: torch.Tensor | None = None


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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
    result = RawTensorSplit(
        win_rate=torch.tensor(split.win_rate, dtype=torch.float32, device=device),
        matchup_1v1=torch.tensor(split.matchup_1v1, dtype=torch.float32, device=device),
        synergy_2vx=torch.tensor(split.synergy_2vx, dtype=torch.float32, device=device),
        p1_cnt=torch.tensor(split.p1_cnt, dtype=torch.float32, device=device),
        m1v1_cnt=torch.tensor(split.m1v1_cnt, dtype=torch.float32, device=device),
        s2vx_cnt=torch.tensor(split.s2vx_cnt, dtype=torch.float32, device=device),
        m1v1_eff_n=torch.tensor(
            split.m1v1_eff_n if split.m1v1_eff_n is not None else split.m1v1_cnt,
            dtype=torch.float32, device=device,
        ),
        s2vx_eff_n=torch.tensor(
            split.s2vx_eff_n if split.s2vx_eff_n is not None else split.s2vx_cnt,
            dtype=torch.float32, device=device,
        ),
        blue_win=torch.tensor(split.blue_win, dtype=torch.float32, device=device),
        champion_id=(
            torch.tensor(split.champion_id, dtype=torch.long, device=device)
            if split.champion_id is not None
            else None
        ),
        build_id=(
            torch.tensor(split.build_id, dtype=torch.long, device=device)
            if split.build_id is not None
            else None
        ),
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

    return RawTensorSplit(
        win_rate=take(raw.win_rate),
        matchup_1v1=take(raw.matchup_1v1),
        synergy_2vx=take(raw.synergy_2vx),
        p1_cnt=take(raw.p1_cnt),
        m1v1_cnt=take(raw.m1v1_cnt),
        s2vx_cnt=take(raw.s2vx_cnt),
        m1v1_eff_n=take(raw.m1v1_eff_n),
        s2vx_eff_n=take(raw.s2vx_eff_n),
        blue_win=take(raw.blue_win),
        champion_id=take(raw.champion_id) if raw.champion_id is not None else None,
        build_id=take(raw.build_id) if raw.build_id is not None else None,
    )


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
    raise ValueError(
        "checkpoint_metric must be one of: val_threshold_accuracy, val_accuracy, val_auc, val_nll"
    )


def _hgnn_inputs_from_raw(
    raw: RawTensorSplit,
    *,
    strength: float,
    device: str,
) -> dict[str, torch.Tensor]:
    if raw.champion_id is None or raw.build_id is None:
        raise ValueError("HGNN inputs require champion_id/build_id; rebuild the cache (v17).")
    return build_hgnn_inputs(
        champion_id=raw.champion_id,
        build_id=raw.build_id,
        win_rate=raw.win_rate,
        matchup_1v1=raw.matchup_1v1,
        synergy_2vx=raw.synergy_2vx,
        p1_cnt=raw.p1_cnt,
        m1v1_cnt=raw.m1v1_cnt,
        s2vx_cnt=raw.s2vx_cnt,
        m1v1_eff_n=raw.m1v1_eff_n,
        s2vx_eff_n=raw.s2vx_eff_n,
        strength=strength,
        device=device,
    )


def _predict_hgnn(
    model: HGNNWinModel,
    split: RawTensorSplit,
    *,
    batch_size: int,
    strength: float,
    device: str,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        n_rows = split.blue_win.numel()
        for start in range(0, n_rows, batch_size):
            raw_batch = _raw_batch(split, slice(start, start + batch_size))
            inputs = _hgnn_inputs_from_raw(raw_batch, strength=strength, device=device)
            logits = model(**inputs)["final_logit"]
            predictions.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def train(
    dataset_cfg: DatasetConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Path:
    dataset_cfg = dataset_cfg or DatasetConfig()
    train_cfg = train_cfg or TrainConfig()
    device = resolve_device(train_cfg.device)
    _seed_torch(train_cfg.seed, device=device)
    started = time.monotonic()
    # The Beta-posterior variance strength reused for the HGNN confidence gate.
    strength = dataset_cfg.confidence_gate_strength
    # Cap the training batch — message-passing intermediates are heavy and each step also
    # runs a team-swapped copy.
    train_batch_size = min(train_cfg.batch_size, HGNN_TRAIN_BATCH)

    splits = load_splits(dataset_cfg, require_counts=True)
    if splits["train"].blue_win.size == 0:
        raise ValueError("Training split is empty; rebuild the cache with train games.")
    tensor_splits = {
        name: _cache_raw_tensor_split(name, splits[name], device=device)
        for name in ("train", "val")
    }

    meta = identity_meta(dataset_cfg)
    model_config = HGNNConfig(
        n_champions=int(meta["n_champions"]),
        n_builds=int(meta["n_builds"]),
        build_vocab=tuple(meta["build_vocab"]),
    )
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
            direct_loss = loss_fn(model(**inputs)["final_logit"], labels)
            (0.5 * direct_loss).backward()
            swapped_loss = loss_fn(
                model(**swap_hgnn_inputs(inputs))["final_logit"],
                1.0 - labels,
            )
            (0.5 * swapped_loss).backward()
            loss = 0.5 * (direct_loss.detach() + swapped_loss.detach())
            if train_cfg.max_grad_norm is not None and train_cfg.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            train_loss_sum += float(loss.cpu().item()) * labels.numel() * 2
            train_seen += int(labels.numel() * 2)

        val_predictions = _predict_hgnn(
            model,
            tensor_splits["val"],
            batch_size=train_cfg.batch_size,
            strength=strength,
            device=device,
        )
        train_nll = train_loss_sum / max(train_seen, 1)
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
            }
        )
        logger.info(
            "epoch=%s train_nll=%.5f val_nll=%.5f val_acc=%.4f val_thr=%.3f val_thr_acc=%.4f",
            epoch,
            train_nll,
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
    predictions = {
        split_name: _predict_hgnn(
            model,
            tensor_split,
            batch_size=train_cfg.batch_size,
            strength=strength,
            device=device,
        )
        for split_name, tensor_split in tensor_splits.items()
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
        "best_checkpoint_score": best_checkpoint_score,
        "decision_threshold": best_threshold,
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


if __name__ == "__main__":
    train()
