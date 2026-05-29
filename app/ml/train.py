# pyright: reportPrivateImportUsage=false

"""Train the production structured win-rate model.

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
from app.ml.dataset import SplitData, load_splits
from app.ml.structured_model import (
    DeltaBaselineMode,
    LOGIT_EPS,
    MATCHUP_BLUE_INDEX,
    MATCHUP_RED_INDEX,
    StructuredModelConfig,
    StructuredWinModel,
    TEAM_PAIRS,
    resolve_device,
    role_pair_type_ids,
    save_structured_model,
    validate_delta_mode,
)
from app.ml.utils.calibration import expected_calibration_error

setup_logging_config()
logger = logging.getLogger(__name__)

EPS = 1e-12


@dataclass(frozen=True)
class RawTensorSplit:
    win_rate: torch.Tensor
    matchup_1v1: torch.Tensor
    synergy_2vx: torch.Tensor
    p1_cnt: torch.Tensor
    m1v1_cnt: torch.Tensor
    s2vx_cnt: torch.Tensor
    blue_win: torch.Tensor
    role_pair_ids: torch.Tensor
    pair_a_idx: torch.Tensor
    pair_b_idx: torch.Tensor
    pair_slot_idx: torch.Tensor
    matchup_blue_idx: torch.Tensor
    matchup_red_idx: torch.Tensor


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


def _adaptive_ece(scores: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
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
    n = scores.size
    if n == 0:
        return float("nan")
    k = max(int(round(n * tail_quantile)), 1)
    order = np.argsort(scores)
    s_sorted = scores[order]
    t_sorted = targets[order]
    return 0.5 * (
        _adaptive_ece(s_sorted[:k], t_sorted[:k], n_bins=n_bins)
        + _adaptive_ece(s_sorted[-k:], t_sorted[-k:], n_bins=n_bins)
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


def _seed_torch(seed: int, *, device: str) -> None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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
        blue_win=torch.tensor(split.blue_win, dtype=torch.float32, device=device),
        role_pair_ids=torch.as_tensor(role_pair_type_ids(), dtype=torch.long, device=device),
        pair_a_idx=torch.as_tensor(
            [pair[0] for pair in TEAM_PAIRS],
            dtype=torch.long,
            device=device,
        ),
        pair_b_idx=torch.as_tensor(
            [pair[1] for pair in TEAM_PAIRS],
            dtype=torch.long,
            device=device,
        ),
        pair_slot_idx=torch.arange(len(TEAM_PAIRS), dtype=torch.long, device=device),
        matchup_blue_idx=torch.as_tensor(MATCHUP_BLUE_INDEX, dtype=torch.long, device=device),
        matchup_red_idx=torch.as_tensor(MATCHUP_RED_INDEX, dtype=torch.long, device=device),
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


def _logit_prob_tensor(probabilities: torch.Tensor) -> torch.Tensor:
    p = probabilities.to(torch.float64).clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)
    return torch.log(p / (1.0 - p)).to(torch.float32)


def _confidence_from_counts_tensor(
    counts: torch.Tensor,
    *,
    prior_strength: float,
) -> torch.Tensor:
    count_tensor = counts.to(torch.float64).clamp_min(0.0)
    return (count_tensor / (count_tensor + float(prior_strength))).to(torch.float32)


def _structured_tensors_from_raw(
    raw: RawTensorSplit,
    *,
    prior_strength: float,
    delta_baseline_mode: DeltaBaselineMode,
) -> dict[str, torch.Tensor]:
    win_rate = raw.win_rate
    identity_logits = _logit_prob_tensor(win_rate)
    blue_logits = identity_logits[:, :5]
    red_logits = identity_logits[:, 5:]
    base_features = torch.cat([blue_logits, red_logits, blue_logits - red_logits], dim=1)

    synergy_sides: list[torch.Tensor] = []
    for player_offset, pair_offset in ((0, 0), (5, 10)):
        side_rates = win_rate[:, player_offset : player_offset + 5]
        side_logits = identity_logits[:, player_offset : player_offset + 5]
        joint_logit = _logit_prob_tensor(raw.synergy_2vx[:, pair_offset + raw.pair_slot_idx])
        a_logit = side_logits.index_select(1, raw.pair_a_idx)
        b_logit = side_logits.index_select(1, raw.pair_b_idx)
        if delta_baseline_mode == "logit":
            expected_logit = 0.5 * (a_logit + b_logit)
        else:
            a_rate = side_rates.index_select(1, raw.pair_a_idx)
            b_rate = side_rates.index_select(1, raw.pair_b_idx)
            expected_logit = _logit_prob_tensor((a_rate + b_rate) / 2.0)
        confidence = _confidence_from_counts_tensor(
            raw.s2vx_cnt[:, pair_offset + raw.pair_slot_idx],
            prior_strength=prior_strength,
        )
        synergy_sides.append(
            torch.stack(
                [
                    joint_logit,
                    a_logit,
                    b_logit,
                    expected_logit,
                    confidence,
                    joint_logit - expected_logit,
                ],
                dim=-1,
            )
        )
    synergy_objects = torch.stack(synergy_sides, dim=1)

    matchup_logit = _logit_prob_tensor(raw.matchup_1v1)
    matchup_blue_logit = blue_logits.index_select(1, raw.matchup_blue_idx)
    matchup_red_logit = red_logits.index_select(1, raw.matchup_red_idx)
    if delta_baseline_mode == "logit":
        expected_matchup_logit = matchup_blue_logit - matchup_red_logit
    else:
        blue_rates = win_rate[:, :5].index_select(1, raw.matchup_blue_idx)
        red_rates = win_rate[:, 5:].index_select(1, raw.matchup_red_idx)
        expected_matchup_logit = _logit_prob_tensor(0.5 + (blue_rates - red_rates) / 2.0)
    matchup_confidence = _confidence_from_counts_tensor(
        raw.m1v1_cnt,
        prior_strength=prior_strength,
    )
    matchup_objects = torch.stack(
        [
            matchup_logit,
            matchup_blue_logit,
            matchup_red_logit,
            expected_matchup_logit,
            matchup_confidence,
            matchup_logit - expected_matchup_logit,
        ],
        dim=-1,
    )

    p1_conf = _confidence_from_counts_tensor(raw.p1_cnt, prior_strength=prior_strength)
    m1_conf = matchup_confidence
    s2_conf = _confidence_from_counts_tensor(raw.s2vx_cnt, prior_strength=prior_strength)
    confidence_summaries = torch.stack(
        [
            p1_conf.mean(dim=1),
            m1_conf.mean(dim=1),
            m1_conf.max(dim=1).values,
            s2_conf.mean(dim=1),
            s2_conf.max(dim=1).values,
            s2_conf[:, :10].mean(dim=1),
            s2_conf[:, 10:].mean(dim=1),
        ],
        dim=1,
    )

    return {
        "base_features": base_features,
        "confidence_summaries": confidence_summaries,
        "synergy_objects": synergy_objects,
        "matchup_objects": matchup_objects,
        "role_pair_ids": raw.role_pair_ids,
    }


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
        blue_win=take(raw.blue_win),
        role_pair_ids=raw.role_pair_ids,
        pair_a_idx=raw.pair_a_idx,
        pair_b_idx=raw.pair_b_idx,
        pair_slot_idx=raw.pair_slot_idx,
        matchup_blue_idx=raw.matchup_blue_idx,
        matchup_red_idx=raw.matchup_red_idx,
    )


def _predict_raw_tensor_split(
    model: StructuredWinModel,
    split: RawTensorSplit,
    *,
    batch_size: int,
    prior_strength: float,
    delta_baseline_mode: DeltaBaselineMode,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        n_rows = split.blue_win.numel()
        for start in range(0, n_rows, batch_size):
            raw_batch = _raw_batch(split, slice(start, start + batch_size))
            logits = model(
                **_structured_tensors_from_raw(
                    raw_batch,
                    prior_strength=prior_strength,
                    delta_baseline_mode=delta_baseline_mode,
                )
            )["final_logit"]
            predictions.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def _evaluate_predictions(scores: np.ndarray, split: SplitData) -> dict[str, float | int]:
    targets = split.blue_win.astype(np.float64, copy=False)
    if targets.size == 0:
        keys = (
            "n",
            "accuracy",
            "auc",
            "nll",
            "brier",
            "entropy",
            "ece",
            "adaptive_ece",
            "tail_ece",
        )
        return {k: 0 if k == "n" else float("nan") for k in keys}
    return {
        "n": int(targets.size),
        "accuracy": float(np.mean((scores >= 0.5) == (targets > 0.5))),
        "auc": _binary_auc(scores, targets),
        "nll": _nll(scores, targets),
        "brier": _brier(scores, targets),
        "entropy": _entropy(scores),
        "ece": expected_calibration_error(scores, targets),
        "adaptive_ece": _adaptive_ece(scores, targets),
        "tail_ece": _tail_ece(scores, targets),
    }


def train(
    dataset_cfg: DatasetConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Path:
    dataset_cfg = dataset_cfg or DatasetConfig()
    train_cfg = train_cfg or TrainConfig()
    delta_mode = validate_delta_mode(train_cfg.delta_baseline_mode)
    device = resolve_device(train_cfg.device)
    _seed_torch(0, device=device)
    started = time.monotonic()

    splits = load_splits(dataset_cfg, require_counts=True)
    if splits["train"].blue_win.size == 0:
        raise ValueError("Training split is empty; rebuild the cache with train games.")
    tensor_splits = {
        name: _cache_raw_tensor_split(name, splits[name], device=device)
        for name in ("train", "val")
    }

    # Leakage-robust interaction config. Train interaction priors are in-sample
    # (a train game's own outcome is in its 1v1/2vx priors), so low-support
    # deltas leak the label. "raw" objects drop the leakage-isolating
    # expected/delta columns, confidence_gate zeroes low-support interactions,
    # and weighted pooling drops the max/min ops that select the leakiest pairs.
    # See app/ml/documentation/README.md.
    model_config = StructuredModelConfig(
        use_synergy=True,
        use_matchup=True,
        use_cross=True,
        delta_baseline_mode=delta_mode,
        object_feature_mode="raw",
        confidence_gate=True,
        pooling_ops=("weighted",),
    )
    model = StructuredWinModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    best_state = copy.deepcopy(model.state_dict())
    best_val_nll = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    logger.info(
        "Structured training device=%s batch_size=%s max_epochs=%s delta_mode=%s",
        device,
        train_cfg.batch_size,
        train_cfg.max_epochs,
        delta_mode,
    )
    if device == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        for batch_idx in _batch_indices(
            splits["train"].blue_win.size,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            rng=rng,
        ):
            raw_batch = _raw_batch(
                tensor_splits["train"],
                torch.as_tensor(batch_idx, dtype=torch.long, device=device),
            )
            batch = _structured_tensors_from_raw(
                raw_batch,
                prior_strength=dataset_cfg.smoothing_prior_strength,
                delta_baseline_mode=delta_mode,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch)["final_logit"]
            loss = loss_fn(logits, raw_batch.blue_win)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu().item()) * raw_batch.blue_win.numel()
            train_seen += int(raw_batch.blue_win.numel())

        val_predictions = _predict_raw_tensor_split(
            model,
            tensor_splits["val"],
            batch_size=train_cfg.batch_size,
            prior_strength=dataset_cfg.smoothing_prior_strength,
            delta_baseline_mode=delta_mode,
        )
        train_nll = train_loss_sum / max(train_seen, 1)
        val_nll = _nll(val_predictions, splits["val"].blue_win)
        history.append(
            {"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll}
        )
        logger.info(
            "epoch=%s train_nll=%.5f val_nll=%.5f",
            epoch,
            train_nll,
            val_nll,
        )
        if val_nll < best_val_nll - 1e-6:
            best_val_nll = val_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= train_cfg.patience:
                break

    model.load_state_dict(best_state)
    save_structured_model(
        train_cfg.model_path,
        model,
        prior_strength=dataset_cfg.smoothing_prior_strength,
    )

    tensor_splits["test"] = _cache_raw_tensor_split("test", splits["test"], device=device)
    predictions = {
        split_name: _predict_raw_tensor_split(
            model,
            tensor_split,
            batch_size=train_cfg.batch_size,
            prior_strength=dataset_cfg.smoothing_prior_strength,
            delta_baseline_mode=delta_mode,
        )
        for split_name, tensor_split in tensor_splits.items()
    }
    metrics = {
        "model_type": "structured_interaction_cross",
        "dataset_config": asdict(dataset_cfg),
        "train_config": asdict(train_cfg),
        "model_config": asdict(model_config),
        "model_path": train_cfg.model_path,
        "metrics_path": train_cfg.metrics_path,
        "device": device,
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "train": _evaluate_predictions(predictions["train"], splits["train"]),
        "val": _evaluate_predictions(predictions["val"], splits["val"]),
        "test": _evaluate_predictions(predictions["test"], splits["test"]),
    }
    _write_metrics(train_cfg.metrics_path, metrics)

    logger.info("Saved structured model: %s", _project_relative(train_cfg.model_path))
    logger.info("Saved metrics: %s", _project_relative(train_cfg.metrics_path))
    for split_name in ("train", "val", "test"):
        m = metrics[split_name]
        if isinstance(m, dict):
            logger.info(
                "%s n=%s acc=%.4f auc=%.4f nll=%.4f brier=%.4f "
                "ece=%.4f adaptive_ece=%.4f tail_ece=%.4f",
                split_name,
                m["n"],
                m["accuracy"],
                m["auc"],
                m["nll"],
                m["brier"],
                m["ece"],
                m["adaptive_ece"],
                m["tail_ece"],
            )

    return train_cfg.model_path


if __name__ == "__main__":
    train()
