"""Fit a linear model from the 10 player win-rate inputs.

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
from app.ml.model import FEATURE_NAMES, WinRateLinearModel, fit_linear_regression

setup_logging_config()
logger = logging.getLogger(__name__)


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


def _evaluate(model: WinRateLinearModel, split: SplitData) -> dict[str, float | int]:
    targets = split.blue_win.astype(np.float64, copy=False)
    if targets.size == 0:
        return {
            "n": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "accuracy": float("nan"),
            "auc": float("nan"),
        }

    predictions = model.predict(split.win_rate.astype(np.float64, copy=False))
    mse = float(np.mean(np.square(predictions - targets)))
    accuracy = float(np.mean((predictions >= 0.5) == (targets > 0.5)))
    return {
        "n": int(targets.size),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "accuracy": accuracy,
        "auc": _binary_auc(predictions, targets),
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

    model = fit_linear_regression(splits["train"].win_rate, splits["train"].blue_win)
    model.save(train_cfg.model_path)

    metrics = {
        "dataset_config": asdict(dataset_cfg),
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
        split_metrics = metrics[split_name]
        if isinstance(split_metrics, dict):
            logger.info(
                "%s n=%s mse=%.4f accuracy=%.4f auc=%.4f",
                split_name,
                split_metrics["n"],
                split_metrics["mse"],
                split_metrics["accuracy"],
                split_metrics["auc"],
            )

    return train_cfg.model_path


if __name__ == "__main__":
    train()
