"""Fit a linear model from the per-game prior arrays.

Run with:
    python -m app.ml.train
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.core.config.settings import PROJECT_ROOT
from app.core.logging.logger import setup_logging_config
from app.ml.config import DatasetConfig, TrainConfig
from app.ml.dataset import SplitData, load_splits
from app.ml.model import FEATURE_NAMES, WinRateLinearModel, fit_logistic_regression

setup_logging_config()
logger = logging.getLogger(__name__)

EPS = 1e-12


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


def _adaptive_ece(
    scores: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-mass binning (Nguyen & O'Connor 2015) — robust to non-uniform p."""
    n = scores.size
    if n == 0:
        return float("nan")
    order = np.argsort(scores)
    s = scores[order]
    t = targets[order]
    bins = np.array_split(np.arange(n), n_bins)
    total = 0.0
    for idx in bins:
        if idx.size == 0:
            continue
        total += idx.size * abs(s[idx].mean() - t[idx].mean())
    return float(total / n)


def _tail_ece(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    tail_quantile: float = 0.1,
    n_bins: int = 5,
) -> float:
    """Adaptive ECE restricted to the bottom and top tails of the score distribution."""
    n = scores.size
    if n == 0:
        return float("nan")
    k = max(int(round(n * tail_quantile)), 1)
    order = np.argsort(scores)
    s_sorted = scores[order]
    t_sorted = targets[order]
    low_s, low_t = s_sorted[:k], t_sorted[:k]
    high_s, high_t = s_sorted[-k:], t_sorted[-k:]
    return 0.5 * (
        _adaptive_ece(low_s, low_t, n_bins=n_bins)
        + _adaptive_ece(high_s, high_t, n_bins=n_bins)
    )


def _nll(scores: np.ndarray, targets: np.ndarray) -> float:
    if scores.size == 0:
        return float("nan")
    p = np.clip(scores, EPS, 1.0 - EPS)
    return float(-np.mean(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))


def _brier(scores: np.ndarray, targets: np.ndarray) -> float:
    if scores.size == 0:
        return float("nan")
    return float(np.mean(np.square(scores - targets)))


def _entropy(scores: np.ndarray) -> float:
    if scores.size == 0:
        return float("nan")
    p = np.clip(scores, EPS, 1.0 - EPS)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(np.mean(h))


def _evaluate(model: WinRateLinearModel, split: SplitData) -> dict[str, float | int]:
    targets = split.blue_win.astype(np.float64, copy=False)
    if targets.size == 0:
        keys = ("n", "accuracy", "auc", "nll", "brier", "entropy",
                "adaptive_ece", "tail_ece")
        return {k: 0 if k == "n" else float("nan") for k in keys}

    predictions = model.predict(
        split.win_rate.astype(np.float64, copy=False),
        split.matchup_1v1.astype(np.float64, copy=False),
        split.synergy_2vx.astype(np.float64, copy=False),
    )
    accuracy = float(np.mean((predictions >= 0.5) == (targets > 0.5)))
    return {
        "n": int(targets.size),
        "accuracy": accuracy,
        "auc": _binary_auc(predictions, targets),
        "nll": _nll(predictions, targets),
        "brier": _brier(predictions, targets),
        "entropy": _entropy(predictions),
        "adaptive_ece": _adaptive_ece(predictions, targets),
        "tail_ece": _tail_ece(predictions, targets),
    }


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return _project_relative(value)
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


def train(
    dataset_cfg: DatasetConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Path:
    dataset_cfg = dataset_cfg or DatasetConfig()
    train_cfg = train_cfg or TrainConfig()

    splits = load_splits(dataset_cfg)
    if splits["train"].blue_win.size == 0:
        raise ValueError("Training split is empty; rebuild the cache with train games.")

    model = fit_logistic_regression(
        splits["train"].win_rate,
        splits["train"].matchup_1v1,
        splits["train"].synergy_2vx,
        splits["train"].blue_win,
        l2=train_cfg.l2,
    )
    model.save(train_cfg.model_path)

    metrics = {
        "dataset_config": asdict(dataset_cfg),
        "l2": train_cfg.l2,
        "model_path": train_cfg.model_path,
        "metrics_path": train_cfg.metrics_path,
        "intercept": model.intercept,
        "feature_names": FEATURE_NAMES,
        "weights": model.weights,
        "train": _evaluate(model, splits["train"]),
        "val": _evaluate(model, splits["val"]),
        "test": _evaluate(model, splits["test"]),
    }
    _write_metrics(train_cfg.metrics_path, metrics)

    logger.info("Saved model: %s", _project_relative(train_cfg.model_path))
    logger.info("Saved metrics: %s", _project_relative(train_cfg.metrics_path))
    for split_name in ("train", "val", "test"):
        m = metrics[split_name]
        if isinstance(m, dict):
            logger.info(
                "%s n=%s acc=%.4f auc=%.4f nll=%.4f brier=%.4f "
                "entropy=%.4f adaptive_ece=%.4f tail_ece=%.4f",
                split_name, m["n"], m["accuracy"], m["auc"], m["nll"],
                m["brier"], m["entropy"], m["adaptive_ece"], m["tail_ece"],
            )

    return train_cfg.model_path


if __name__ == "__main__":
    train()
