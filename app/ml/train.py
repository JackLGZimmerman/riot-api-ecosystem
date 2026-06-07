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
from app.ml.config import DEFAULT_TRAIN_BATCH_CAP, DatasetConfig, TrainConfig
from app.ml.context_audit_lens import AuditLens
from app.ml.context_audit_specs import (
    AuditSpec,
    POSITIONS,
    audit_specs,
    eb_shrink_targets,
    group_audit_specs,
)
from app.ml.dataset import SplitData, identity_meta, load_splits
from app.ml.encoder_sidecar import EncoderSidecarLookup, SidecarGatherTables
from app.core.utils.common import resolve_device_str as resolve_device
from app.ml.hgnn_model import (
    HGNNConfig,
    HGNNWinModel,
    build_hgnn_inputs,
    save_hgnn_model,
    swap_hgnn_inputs,
)
from app.ml.loadout_patch_features import (
    LOADOUT_SIGNED_FEATURE_INDICES,
    PATCH_SIGNED_FEATURE_INDICES,
)
from app.ml.semantic_group_features import (
    CONTEXT_AXIS_INDEX,
    FOCUS_HP_LOW_THRESHOLD,
    HIGH_HP_THRESHOLD,
    LOW_OWN_DAMAGE_THRESHOLD,
    SEMANTIC_GROUP_FEATURE_INDEX,
    static_hp_range_lookups,
)

setup_logging_config()
logger = logging.getLogger(__name__)

EPS = 1e-12
# Reference single-run cap for the team-swapped training loop on the local
# RTX 5070 Ti. Use epoch timing telemetry and --train-batch-cap for sweeps.
HGNN_TRAIN_BATCH = DEFAULT_TRAIN_BATCH_CAP
PRODUCTION_SEMANTIC_MOE_ARCHITECTURE = "convex_encoder_mix"
PRODUCTION_SEMANTIC_MODEL_OVERRIDES: dict[str, Any] = {
    "use_identity_static_sidecar": False,
    "use_identity_full_game_sidecar": False,
    "use_identity_temporal_sidecar": False,
    "use_learned_semantic_moe": True,
    "use_semantic_group_features": True,
    "semantic_moe_architecture": PRODUCTION_SEMANTIC_MOE_ARCHITECTURE,
    "semantic_moe_num_experts": 128,
    "semantic_moe_top_k": 32,
}


def production_semantic_model_overrides() -> dict[str, Any]:
    """Return the promoted all-encoder semantic HGNN recipe."""

    return dict(PRODUCTION_SEMANTIC_MODEL_OVERRIDES)


@dataclass(frozen=True)
class RawTensorSplit:
    win_rate: torch.Tensor
    p1_cnt: torch.Tensor
    blue_win: torch.Tensor
    champion_id: torch.Tensor | None = None
    build_id: torch.Tensor | None = None
    identity_static_sidecar: torch.Tensor | None = None
    identity_full_game_sidecar: torch.Tensor | None = None
    identity_temporal_sidecar: torch.Tensor | None = None
    identity_encoder_support: torch.Tensor | None = None
    semantic_group_features: torch.Tensor | None = None
    loadout_features: torch.Tensor | None = None
    patch_features: torch.Tensor | None = None


class _SidecarGatherer:
    """Per-batch gather of frozen identity latents from the dedup'd artifact.

    Replaces the materialised per-game sidecar arrays: holds the small latent
    tables on-device and gathers ``(batch, 10, dim)`` blocks from
    ``champion_id`` / ``build_id`` so the cache no longer stores one latent copy
    per game-slot. The static block is champion-keyed and zeroed for identities
    whose ``(role, build)`` row is absent, matching the artifact lookup.
    """

    def __init__(self, tables: SidecarGatherTables, *, device: str) -> None:
        self.dense_index = torch.as_tensor(
            tables.dense_index, dtype=torch.long, device=device
        )
        self.static_by_champion = torch.as_tensor(
            tables.static_by_champion, dtype=torch.float32, device=device
        )
        self.full_game = torch.as_tensor(
            tables.full_game, dtype=torch.float32, device=device
        )
        self.temporal = torch.as_tensor(
            tables.temporal, dtype=torch.float32, device=device
        )
        self.support = torch.as_tensor(
            tables.support, dtype=torch.float32, device=device
        )
        self.slot_role = torch.as_tensor(
            tables.slot_role, dtype=torch.long, device=device
        )
        self.n_champions = int(tables.n_champions)
        self.n_builds = int(tables.n_builds)
        self.pad_row = int(tables.pad_row)

    def gather(
        self, champion_id: torch.Tensor, build_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        champ = champion_id.clamp(0, self.n_champions)
        build = build_id.clamp(0, self.n_builds)
        role = self.slot_role.view(1, -1).expand_as(champ)
        row = self.dense_index[champ, role, build]
        present = (row != self.pad_row).unsqueeze(-1).to(torch.float32)
        return {
            "identity_static_sidecar": self.static_by_champion[champ] * present,
            "identity_full_game_sidecar": self.full_game[row],
            "identity_temporal_sidecar": self.temporal[row],
            "identity_encoder_support": self.support[row],
        }


def _model_uses_sidecar(config: HGNNConfig) -> bool:
    return bool(
        config.use_identity_static_sidecar
        or config.use_identity_full_game_sidecar
        or config.use_identity_temporal_sidecar
        or config.use_identity_semantic_context_head
        or config.use_learned_semantic_moe
    )


def _build_sidecar_gatherer(
    dataset_cfg: DatasetConfig,
    meta: dict[str, Any],
    config: HGNNConfig,
    *,
    device: str,
) -> _SidecarGatherer:
    """Load the frozen sidecar artifact and precompute on-device gather tables."""
    path = dataset_cfg.encoder_sidecar_path
    if path is None:
        sidecar_meta = meta.get("identity_encoder_sidecar")
        recorded = sidecar_meta.get("path") if isinstance(sidecar_meta, dict) else None
        if not recorded:
            raise ValueError(
                "Model uses identity-encoder sidecars but the cache has no per-game "
                "sidecar arrays and no encoder_sidecar_path. Pass --encoder-sidecar-path "
                "or rebuild the cache (v28) with the artifact recorded in its meta."
            )
        path = Path(recorded)
    tables = EncoderSidecarLookup.load(path).gather_tables(
        build_vocab=list(config.build_vocab),
        n_champions=int(config.n_champions),
        n_builds=int(config.n_builds),
    )
    return _SidecarGatherer(tables, device=device)


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


def _drop_unused_model_arrays(
    split: SplitData,
    config: HGNNConfig,
    *,
    keep_semantic_group_features: bool = False,
) -> SplitData:
    """Null out optional sidecar arrays when the configured model ignores them."""
    sidecar_enabled = (
        config.use_identity_static_sidecar
        or config.use_identity_full_game_sidecar
        or config.use_identity_temporal_sidecar
        or config.use_identity_semantic_context_head
        or config.use_learned_semantic_moe
    )
    semantic_group_features_enabled = bool(
        keep_semantic_group_features
        or (config.use_learned_semantic_moe and config.use_semantic_group_features)
    )
    requires_all_sidecars = bool(
        config.use_identity_semantic_context_head or config.use_learned_semantic_moe
    )
    if requires_all_sidecars:
        sidecar_names = (
            "identity_static_sidecar",
            "identity_full_game_sidecar",
            "identity_temporal_sidecar",
            "identity_encoder_support",
        )
        present = [name for name in sidecar_names if getattr(split, name) is not None]
        # All absent => latents are gathered per batch from the frozen artifact
        # (v28 cache). Partial presence means a corrupt or legacy cache: fail early.
        if present and len(present) < len(sidecar_names):
            missing = [name for name in sidecar_names if getattr(split, name) is None]
            raise ValueError(
                "semantic context/MoE head requires cache arrays: "
                + ", ".join(missing)
                + ". Rebuild the dataset cache with encoder_sidecar_path set "
                "to a valid three-latent sidecar artifact."
            )
        if len(present) == len(sidecar_names):
            for name in (
                "identity_static_sidecar",
                "identity_full_game_sidecar",
                "identity_temporal_sidecar",
            ):
                value = getattr(split, name)
                if value.ndim != 3 or value.shape[1] != 10 or value.shape[2] <= 0:
                    raise ValueError(
                        f"semantic context/MoE heads require non-empty {name} [games, 10, dim]"
                    )
            support = split.identity_encoder_support
            if support.ndim != 2 or support.shape[1] != 10:
                raise ValueError(
                    "semantic context/MoE heads require identity_encoder_support [games, 10]"
                )
    drop: dict[str, bool] = {
        "identity_static_sidecar": not (
            config.use_identity_static_sidecar or requires_all_sidecars
        ),
        "identity_full_game_sidecar": not (
            config.use_identity_full_game_sidecar or requires_all_sidecars
        ),
        "identity_temporal_sidecar": not (
            config.use_identity_temporal_sidecar or requires_all_sidecars
        ),
        "identity_encoder_support": not sidecar_enabled,
        "semantic_group_features": not semantic_group_features_enabled,
        "loadout_features": int(config.loadout_feature_dim) <= 0,
        "patch_features": int(config.patch_feature_dim) <= 0,
    }
    for name, dim in (
        ("loadout_features", int(config.loadout_feature_dim)),
        ("patch_features", int(config.patch_feature_dim)),
    ):
        if dim <= 0:
            continue
        value = getattr(split, name)
        if value is None or value.ndim != 2 or value.shape[1] != dim:
            raise ValueError(
                f"HGNN config enables {name}, but the cache is missing "
                f"{name} [games, {dim}]; rebuild the dataset cache."
            )
    if semantic_group_features_enabled:
        value = split.semantic_group_features
        if (
            value is None
            or value.ndim != 3
            or value.shape[1] != 10
            or value.shape[2] != int(config.semantic_group_feature_dim)
        ):
            raise ValueError(
                "learned semantic MoE semantic group features require "
                f"semantic_group_features [games, 10, {config.semantic_group_feature_dim}]"
            )
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
    conf = (
        np.bincount(bin_idx, weights=p, minlength=n_bins)[populated] / counts[populated]
    )
    acc = (
        np.bincount(bin_idx, weights=y, minlength=n_bins)[populated] / counts[populated]
    )
    return float(np.sum(counts[populated] / p.size * np.abs(conf - acc)))


def _sigmoid_np(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    scale = max(float(temperature), EPS)
    z = np.clip(logits.astype(np.float64, copy=False) / scale, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64, copy=False)


def _logit_nll(
    logits: np.ndarray, targets: np.ndarray, *, temperature: float = 1.0
) -> float:
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
    fine = np.exp(
        np.linspace(math.log(best) - half_step, math.log(best) + half_step, 81)
    )
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
    max_rows: int | None = None,
) -> Iterator[np.ndarray]:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when set")
    indices = rng.permutation(n_rows) if shuffle else np.arange(n_rows)
    if max_rows is not None and max_rows < n_rows:
        indices = indices[:max_rows]
    for start in range(0, indices.size, batch_size):
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
        # CPU caches can share storage with the already loaded NumPy arrays when
        # dtype/layout allow it, avoiding a second large host-memory copy for
        # parallel ablation runs. CUDA caches still materialise on the GPU.
        if device == "cpu":
            return torch.as_tensor(value, dtype=dtype, device=device)
        return torch.tensor(value, dtype=dtype, device=device)

    result = RawTensorSplit(
        **{
            f.name: (
                None
                if (value := getattr(split, f.name, None)) is None
                else to_tensor(f.name, value)
            )
            for f in fields(RawTensorSplit)
        }
    )
    if str(device).startswith("cuda"):
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


def _raw_split_to_device(raw: RawTensorSplit, *, device: str) -> RawTensorSplit:
    """Move a raw tensor split or minibatch to the model device."""

    return _map_split(raw, lambda tensor: tensor.to(device, non_blocking=True))


def _raw_index_tensor(raw: RawTensorSplit, batch_idx: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(batch_idx, dtype=torch.long, device=raw.blue_win.device)


def _evaluate_predictions(
    scores: np.ndarray, split: SplitData
) -> dict[str, float | int]:
    targets = split.blue_win.astype(np.float64, copy=False)
    if targets.size == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "auc": float("nan"),
            "nll": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
        }
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
        "val_context_gap_mse",
        "val_group_eb_gap_mse",
        "val_group_systematic_gap_mse",
        "val_group_systematic_gap_mse_clipped",
        "val_group_floor_normalized_eb_gap",
        "val_group_clipped_nll_ece",
        "val_group_first_clipped_nll_ece",
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
        return {
            "n": 0,
            "auc": float("nan"),
            "nll": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "model_mean": float("nan"),
            "label_mean": float("nan"),
            "calibration_gap": float("nan"),
        }
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
        row["mean_support"] = (
            float(np.mean(values[mask])) if np.any(mask) else float("nan")
        )
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


def _threshold_accuracy(
    scores: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    if scores.size == 0:
        return float("nan")
    return float(np.mean((scores >= threshold) == (targets > 0.5)))


def _std_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values.astype(np.float64, copy=False), ddof=0))


def _logit_diagnostics(outputs: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "base_logit_std": _std_or_nan(outputs["base_logit"]),
        "context_logit_std": _std_or_nan(outputs["context_logit"]),
        "final_logit_std": _std_or_nan(outputs["final_logit"]),
    }


def _semantic_moe_diagnostics(
    outputs: dict[str, np.ndarray],
) -> dict[str, object] | None:
    if "semantic_moe_expert_usage" not in outputs:
        return None
    diagnostics: dict[str, object] = {
        "expert_usage": outputs["semantic_moe_expert_usage"],
        "expert_selected_fraction": outputs["semantic_moe_expert_selected_fraction"],
    }
    if "semantic_moe_view_usage" in outputs:
        diagnostics["view_usage"] = outputs["semantic_moe_view_usage"]
    if "semantic_moe_view_selected_fraction" in outputs:
        diagnostics["view_selected_fraction"] = outputs[
            "semantic_moe_view_selected_fraction"
        ]
    scalar_keys = (
        "semantic_moe_router_entropy",
        "semantic_moe_factor_norm",
        "semantic_moe_balance_loss",
        "semantic_moe_entropy_loss",
        "semantic_moe_factor_orthogonality_loss",
        "semantic_moe_factor_variance_loss",
        "semantic_moe_factor_std_mean",
        "semantic_moe_factor_std_min",
        "semantic_moe_context_token_keep_fraction",
        "semantic_moe_delta_l2_loss",
        "semantic_moe_slot_delta_max_abs",
        "semantic_moe_max_abs_slot_delta",
        "semantic_moe_group_relationship_l2_loss",
        "semantic_moe_group_relationship_coeff_norm",
        "semantic_moe_group_relationship_context_norm",
        "semantic_moe_group_relationship_enabled",
        "semantic_moe_regularization_loss",
        "semantic_moe_group_features_enabled",
        "semantic_moe_group_feature_dim",
        "semantic_moe_view_entropy",
        "semantic_moe_view_balance_loss",
        "semantic_moe_view_entropy_loss",
        "semantic_moe_view_top_k",
        "semantic_moe_convex_encoder_mix_enabled",
        "semantic_moe_full_game_slot_delta_mean_abs",
        "semantic_moe_temporal_slot_delta_mean_abs",
    )
    for key in scalar_keys:
        if key in outputs:
            diagnostics[key.removeprefix("semantic_moe_")] = float(outputs[key])
    if "router_entropy" in diagnostics and "expert_usage" in diagnostics:
        usage = np.asarray(diagnostics["expert_usage"], dtype=np.float64)
        selected_fraction = np.asarray(
            diagnostics["expert_selected_fraction"],
            dtype=np.float64,
        )
        selected_per_slot = float(np.sum(selected_fraction))
        if selected_per_slot > 1.0:
            diagnostics["router_entropy_fraction_of_topk_max"] = float(
                diagnostics["router_entropy"]
            ) / math.log(selected_per_slot)
        diagnostics["expert_usage_min"] = (
            float(np.min(usage)) if usage.size else float("nan")
        )
        diagnostics["expert_usage_max"] = (
            float(np.max(usage)) if usage.size else float("nan")
        )
    return diagnostics


def _attach_output_diagnostics(
    split_metrics: dict[str, dict[str, object]],
    prediction_outputs: dict[str, dict[str, np.ndarray]],
) -> None:
    for split_name, outputs in prediction_outputs.items():
        split_metrics[split_name]["logit_diagnostics"] = _logit_diagnostics(outputs)
        semantic_moe = _semantic_moe_diagnostics(outputs)
        if semantic_moe is not None:
            split_metrics[split_name]["semantic_moe_diagnostics"] = semantic_moe


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
    if metric == "val_context_gap_mse":
        return -float(val_metrics["context_gap_mse"])
    if metric == "val_group_eb_gap_mse":
        return -float(val_metrics["group_eb_gap_mse"])
    if metric == "val_group_systematic_gap_mse":
        return -float(val_metrics["group_systematic_gap_mse"])
    if metric == "val_group_systematic_gap_mse_clipped":
        return -float(val_metrics["group_systematic_gap_mse_clipped"])
    if metric == "val_group_floor_normalized_eb_gap":
        floor = max(float(val_metrics.get("group_eb_floor", 0.0)), EPS)
        return -(float(val_metrics["group_eb_gap_mse"]) / floor)
    if metric == "val_group_clipped_nll_ece":
        return -(
            float(val_metrics["group_systematic_gap_mse_clipped"])
            + 100.0 * float(val_metrics["nll"])
            + 100.0 * float(val_metrics["ece"])
        )
    if metric == "val_group_first_clipped_nll_ece":
        return -(
            100.0 * float(val_metrics["group_systematic_gap_mse_clipped"])
            + float(val_metrics["nll"])
            + float(val_metrics["ece"])
        )
    raise ValueError(
        f"checkpoint_metric must be one of: {', '.join(sorted(CHECKPOINT_METRICS))}"
    )


def _validate_train_config(train_cfg: TrainConfig) -> None:
    if train_cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if train_cfg.train_batch_cap is not None and train_cfg.train_batch_cap < 0:
        raise ValueError("train_batch_cap must be >= 0")
    if (
        train_cfg.train_epoch_max_games is not None
        and train_cfg.train_epoch_max_games < 0
    ):
        raise ValueError("train_epoch_max_games must be >= 0")
    if train_cfg.raw_tensor_cache_device not in {"model", "cpu"}:
        raise ValueError("raw_tensor_cache_device must be 'model' or 'cpu'")
    if train_cfg.checkpoint_metric not in CHECKPOINT_METRICS:
        raise ValueError(
            f"checkpoint_metric must be one of: {', '.join(sorted(CHECKPOINT_METRICS))}"
        )
    if train_cfg.auc_ranking_loss_weight < 0.0:
        raise ValueError("auc_ranking_loss_weight must be >= 0")
    if train_cfg.auc_ranking_loss_pairs < 1:
        raise ValueError("auc_ranking_loss_pairs must be >= 1")
    if (
        train_cfg.freeze_warm_start_loaded_parameters
        and train_cfg.warm_start_model_path is None
    ):
        raise ValueError(
            "freeze_warm_start_loaded_parameters requires warm_start_model_path"
        )
    if train_cfg.semantic_context_calibration_loss_weight < 0.0:
        raise ValueError("semantic_context_calibration_loss_weight must be >= 0")
    if train_cfg.semantic_context_calibration_min_count < 1:
        raise ValueError("semantic_context_calibration_min_count must be >= 1")
    if train_cfg.semantic_context_calibration_tail_weight <= 0.0:
        raise ValueError("semantic_context_calibration_tail_weight must be > 0")
    if train_cfg.semantic_context_calibration_objective not in {
        "absolute",
        "residual",
    }:
        raise ValueError(
            "semantic_context_calibration_objective must be 'absolute' or 'residual'"
        )
    for name in (
        "semantic_context_calibration_group_residual_shrink_strength",
        "semantic_context_calibration_context_residual_shrink_strength",
    ):
        if float(getattr(train_cfg, name)) < 0.0:
            raise ValueError(f"{name} must be >= 0")
    for name in (
        "semantic_context_calibration_group_residual_clip",
        "semantic_context_calibration_context_residual_clip",
    ):
        if float(getattr(train_cfg, name)) <= 0.0:
            raise ValueError(f"{name} must be > 0")
    for name in (
        "semantic_context_calibration_group_residual_scale",
        "semantic_context_calibration_context_residual_scale",
    ):
        if float(getattr(train_cfg, name)) < 0.0:
            raise ValueError(f"{name} must be >= 0")
    if train_cfg.semantic_context_calibration_target not in {
        "champion_raw",
        "context_eb",
        "group_eb",
        "group_context_eb",
    }:
        raise ValueError(
            "semantic_context_calibration_target must be 'champion_raw', "
            "'context_eb', 'group_eb', or 'group_context_eb'"
        )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    normalized = value.strip()
    if normalized.lower() in {"", "none", "empty"}:
        return ()
    try:
        parsed = tuple(int(part.strip()) for part in normalized.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        ) from exc
    if any(dim <= 0 for dim in parsed):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive integers")
    return parsed


def _hgnn_config_from_meta(
    meta: dict[str, Any],
    *,
    encoder_sidecar_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> HGNNConfig:
    base = dict(
        n_champions=int(meta["n_champions"]),
        n_builds=int(meta["n_builds"]),
        build_vocab=tuple(meta["build_vocab"]),
    )
    if int(meta.get("loadout_feature_dim", 0)) > 0:
        base["loadout_feature_dim"] = int(meta["loadout_feature_dim"])
    if int(meta.get("patch_feature_dim", 0)) > 0:
        base["patch_feature_dim"] = int(meta["patch_feature_dim"])
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
    elif encoder_sidecar_path is not None:
        dims = EncoderSidecarLookup.load(encoder_sidecar_path).dims
        base.update(
            {
                "identity_static_sidecar_dim": int(dims.static),
                "identity_full_game_sidecar_dim": int(dims.full_game),
                "identity_temporal_sidecar_dim": int(dims.temporal),
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
    gatherer: _SidecarGatherer | None = None,
) -> dict[str, torch.Tensor]:
    if raw.champion_id is None or raw.build_id is None:
        raise ValueError(
            "HGNN inputs require champion_id/build_id; rebuild the cache (v17)."
        )
    # v28 caches omit per-game sidecar arrays; gather them from the frozen table
    # using the batch's identity ids. Legacy caches that still carry per-game
    # arrays are used directly.
    if gatherer is not None and raw.identity_static_sidecar is None:
        sidecar = gatherer.gather(raw.champion_id, raw.build_id)
    else:
        sidecar = {
            "identity_static_sidecar": raw.identity_static_sidecar,
            "identity_full_game_sidecar": raw.identity_full_game_sidecar,
            "identity_temporal_sidecar": raw.identity_temporal_sidecar,
            "identity_encoder_support": raw.identity_encoder_support,
        }
    return build_hgnn_inputs(
        champion_id=raw.champion_id,
        build_id=raw.build_id,
        win_rate=raw.win_rate,
        p1_cnt=raw.p1_cnt,
        strength=strength,
        semantic_group_features=raw.semantic_group_features,
        loadout_features=raw.loadout_features,
        patch_features=raw.patch_features,
        device=device,
        **sidecar,
    )


def _predict_hgnn_logits(
    model: HGNNWinModel,
    split: RawTensorSplit,
    *,
    batch_size: int,
    strength: float,
    device: str,
    gatherer: _SidecarGatherer | None = None,
) -> np.ndarray:
    return _predict_hgnn_outputs(
        model,
        split,
        batch_size=batch_size,
        strength=strength,
        device=device,
        gatherer=gatherer,
    )["final_logit"]


def _predict_hgnn_outputs(
    model: HGNNWinModel,
    split: RawTensorSplit,
    *,
    batch_size: int,
    strength: float,
    device: str,
    gatherer: _SidecarGatherer | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {
        "base_logit": [],
        "context_logit": [],
        "loadout_logit": [],
        "patch_logit": [],
        "feature_logit": [],
        "final_logit": [],
        "focus_side_probability": [],
    }
    weighted_stats: dict[str, tuple[torch.Tensor, int]] = {}
    semantic_moe_stat_keys = {
        "semantic_moe_expert_usage",
        "semantic_moe_expert_selected_fraction",
        "semantic_moe_router_entropy",
        "semantic_moe_factor_norm",
        "semantic_moe_balance_loss",
        "semantic_moe_entropy_loss",
        "semantic_moe_factor_orthogonality_loss",
        "semantic_moe_factor_variance_loss",
        "semantic_moe_factor_std_mean",
        "semantic_moe_factor_std_min",
        "semantic_moe_context_token_keep_fraction",
        "semantic_moe_delta_l2_loss",
        "semantic_moe_slot_delta_max_abs",
        "semantic_moe_max_abs_slot_delta",
        "semantic_moe_group_relationship_l2_loss",
        "semantic_moe_group_relationship_coeff_norm",
        "semantic_moe_group_relationship_context_norm",
        "semantic_moe_group_relationship_enabled",
        "semantic_moe_regularization_loss",
        "semantic_moe_group_features_enabled",
        "semantic_moe_group_feature_dim",
        "semantic_moe_view_usage",
        "semantic_moe_view_selected_fraction",
        "semantic_moe_view_entropy",
        "semantic_moe_view_balance_loss",
        "semantic_moe_view_entropy_loss",
        "semantic_moe_view_top_k",
        "semantic_moe_convex_encoder_mix_enabled",
        "semantic_moe_full_game_slot_delta_mean_abs",
        "semantic_moe_temporal_slot_delta_mean_abs",
    }

    def add_weighted_stat(key: str, value: torch.Tensor, weight: int) -> None:
        detached = value.detach().to(device="cpu", dtype=torch.float64)
        current = weighted_stats.get(key)
        weighted = detached * float(weight)
        if current is None:
            weighted_stats[key] = (weighted, weight)
        else:
            total, seen = current
            weighted_stats[key] = (total + weighted, seen + weight)

    with torch.no_grad():
        n_rows = split.blue_win.numel()
        for start in range(0, n_rows, batch_size):
            raw_batch = _raw_split_to_device(
                _raw_batch(split, slice(start, start + batch_size)),
                device=device,
            )
            inputs = _hgnn_inputs_from_raw(
                raw_batch, strength=strength, device=device, gatherer=gatherer
            )
            outputs = model(**inputs)
            focus_side_probability = _focus_side_probabilities_from_outputs(outputs)
            for key in (
                "base_logit",
                "context_logit",
                "loadout_logit",
                "patch_logit",
                "feature_logit",
                "final_logit",
            ):
                value = outputs[key]
                out[key].append(value.detach().cpu().numpy())
            out["focus_side_probability"].append(
                focus_side_probability.detach().cpu().numpy()
            )
            batch_weight = int(raw_batch.blue_win.numel())
            for key in semantic_moe_stat_keys:
                value = outputs.get(key)
                if value is None:
                    continue
                add_weighted_stat(key, value, batch_weight)
    result = {
        key: np.concatenate(values).astype(np.float64) for key, values in out.items()
    }
    for key, (total, seen) in weighted_stats.items():
        result[key] = (total / max(seen, 1)).numpy().astype(np.float64)
    return result


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
        pos_idx = torch.randint(
            positives.numel(), (int(max_pairs),), device=logits.device
        )
        neg_idx = torch.randint(
            negatives.numel(), (int(max_pairs),), device=logits.device
        )
        margins = positives[pos_idx] - negatives[neg_idx]
    return float(weight) * torch.nn.functional.softplus(-margins).mean()


def _side_probabilities_torch(logits: torch.Tensor) -> torch.Tensor:
    blue = torch.sigmoid(logits).view(-1, 1)
    return torch.cat([blue.expand(-1, 5), (1.0 - blue).expand(-1, 5)], dim=1)


def _focus_side_logits_from_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    include_semantic_delta: bool = True,
) -> torch.Tensor:
    slot_delta = outputs.get("semantic_moe_slot_delta")
    if slot_delta is None:
        logits = outputs["final_logit"].view(-1, 1)
        return torch.cat([logits.expand(-1, 5), -logits.expand(-1, 5)], dim=1)

    base_logit = outputs["base_logit"]
    context_logit = outputs["context_logit"]
    semantic_moe_logit = outputs.get("semantic_moe_logit")
    if semantic_moe_logit is None:
        semantic_moe_logit = base_logit.new_zeros(base_logit.shape)
    feature_logit = outputs.get("feature_logit")
    if feature_logit is None:
        feature_logit = base_logit.new_zeros(base_logit.shape)
    shared_logit = base_logit + context_logit - semantic_moe_logit + feature_logit
    if not include_semantic_delta:
        return torch.cat(
            [shared_logit[:, None].expand(-1, 5), -shared_logit[:, None].expand(-1, 5)],
            dim=1,
        )
    blue_delta = slot_delta[:, :5]
    red_delta = slot_delta[:, 5:]
    blue_focus_logit = (
        shared_logit[:, None]
        + blue_delta
        - red_delta.mean(
            dim=1,
            keepdim=True,
        )
    )
    red_focus_logit = (
        -shared_logit[:, None]
        + red_delta
        - blue_delta.mean(
            dim=1,
            keepdim=True,
        )
    )
    return torch.cat([blue_focus_logit, red_focus_logit], dim=1)


def _focus_side_probabilities_from_outputs(
    outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.sigmoid(_focus_side_logits_from_outputs(outputs))


def _probability_logit_torch(values: torch.Tensor) -> torch.Tensor:
    return torch.logit(values.clamp(1.0e-5, 1.0 - 1.0e-5))


def _side_labels_torch(labels: torch.Tensor) -> torch.Tensor:
    blue = labels.to(dtype=torch.float32).view(-1, 1)
    return torch.cat([blue.expand(-1, 5), (1.0 - blue).expand(-1, 5)], dim=1)


def _side_probabilities_np(scores: np.ndarray) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[1] != 10:
            raise ValueError("side probability arrays must have shape [games, 10]")
        return array
    blue = array.reshape(-1, 1)
    return np.concatenate(
        [np.repeat(blue, 5, axis=1), np.repeat(1.0 - blue, 5, axis=1)],
        axis=1,
    )


def _side_labels_np(labels: np.ndarray) -> np.ndarray:
    blue = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
    return np.concatenate(
        [np.repeat(blue, 5, axis=1), np.repeat(1.0 - blue, 5, axis=1)],
        axis=1,
    )


def _team_feature_average_torch(
    features: torch.Tensor,
    feature_name: str,
    *,
    enemy: bool,
) -> torch.Tensor:
    values = features[:, :, SEMANTIC_GROUP_FEATURE_INDEX[feature_name]]
    blue = values[:, :5].mean(dim=1)
    red = values[:, 5:].mean(dim=1)
    blue_focus = red if enemy else blue
    red_focus = blue if enemy else red
    return torch.cat(
        [blue_focus[:, None].expand(-1, 5), red_focus[:, None].expand(-1, 5)],
        dim=1,
    )


def _side_feature_count_torch(
    features: torch.Tensor,
    feature_name: str,
    *,
    enemy: bool,
) -> torch.Tensor:
    values = features[:, :, SEMANTIC_GROUP_FEATURE_INDEX[feature_name]] > 0.5
    blue = values[:, :5].sum(dim=1).to(dtype=torch.float32)
    red = values[:, 5:].sum(dim=1).to(dtype=torch.float32)
    blue_focus = red if enemy else blue
    red_focus = blue if enemy else red
    return torch.cat(
        [blue_focus[:, None].expand(-1, 5), red_focus[:, None].expand(-1, 5)],
        dim=1,
    )


def _team_feature_average_np(
    features: np.ndarray,
    feature_name: str,
    *,
    enemy: bool,
) -> np.ndarray:
    values = features[:, :, SEMANTIC_GROUP_FEATURE_INDEX[feature_name]]
    blue = values[:, :5].mean(axis=1)
    red = values[:, 5:].mean(axis=1)
    blue_focus = red if enemy else blue
    red_focus = blue if enemy else red
    return np.concatenate(
        [
            np.repeat(blue_focus[:, None], 5, axis=1),
            np.repeat(red_focus[:, None], 5, axis=1),
        ],
        axis=1,
    )


def _side_feature_count_np(
    features: np.ndarray,
    feature_name: str,
    *,
    enemy: bool,
) -> np.ndarray:
    values = features[:, :, SEMANTIC_GROUP_FEATURE_INDEX[feature_name]] > 0.5
    blue = values[:, :5].sum(axis=1).astype(np.float64)
    red = values[:, 5:].sum(axis=1).astype(np.float64)
    blue_focus = red if enemy else blue
    red_focus = blue if enemy else red
    return np.concatenate(
        [
            np.repeat(blue_focus[:, None], 5, axis=1),
            np.repeat(red_focus[:, None], 5, axis=1),
        ],
        axis=1,
    )


def _slot_hp_torch(champion_id: torch.Tensor, hp_lookup: torch.Tensor) -> torch.Tensor:
    valid = (champion_id >= 0) & (champion_id < hp_lookup.numel())
    clamped = champion_id.clamp(0, max(int(hp_lookup.numel()) - 1, 0))
    return torch.where(
        valid, hp_lookup[clamped], hp_lookup.new_zeros(champion_id.shape)
    )


def _slot_hp_np(champion_id: np.ndarray, hp_lookup: np.ndarray) -> np.ndarray:
    champions = np.asarray(champion_id, dtype=np.int64)
    out = np.zeros(champions.shape, dtype=np.float64)
    valid = (champions >= 0) & (champions < hp_lookup.shape[0])
    out[valid] = hp_lookup[champions[valid]]
    return out


def _audit_axis_torch(features: torch.Tensor, spec: AuditSpec) -> torch.Tensor:
    axis = spec.axis
    if axis.startswith("enemy_") and axis.removeprefix("enemy_") in CONTEXT_AXIS_INDEX:
        return _team_feature_average_torch(
            features,
            axis.removeprefix("enemy_"),
            enemy=True,
        )
    if axis.startswith("ally_") and axis.removeprefix("ally_") in CONTEXT_AXIS_INDEX:
        return _team_feature_average_torch(
            features,
            axis.removeprefix("ally_"),
            enemy=False,
        )
    if axis == "enemy_burst_count":
        return _side_feature_count_torch(features, "burst", enemy=True)
    if axis == "enemy_hard_cc_count":
        return _side_feature_count_torch(features, "hard_cc", enemy=True)
    if axis == "enemy_frontline_count":
        return _side_feature_count_torch(features, "frontline", enemy=True)
    if axis == "enemy_heavy_taken_count":
        return _side_feature_count_torch(features, "heavy_taken", enemy=True)
    if axis == "enemy_high_hp_count":
        return _side_feature_count_torch(features, "high_hp", enemy=True)
    if axis == "enemy_ranged_count":
        return _side_feature_count_torch(features, "ranged", enemy=True)
    if axis == "same_role_range":
        ranged = features[:, :, SEMANTIC_GROUP_FEATURE_INDEX["same_role_range"]] > 0.5
        return torch.where(
            ranged,
            features.new_full(ranged.shape, 251.0),
            features.new_full(ranged.shape, 250.0),
        )
    if axis == "ally_skirmish_count":
        return _side_feature_count_torch(features, "skirmish", enemy=False)
    raise ValueError(f"unknown audit axis: {axis}")


def _audit_axis_np(features: np.ndarray, spec: AuditSpec) -> np.ndarray:
    axis = spec.axis
    if axis.startswith("enemy_") and axis.removeprefix("enemy_") in CONTEXT_AXIS_INDEX:
        return _team_feature_average_np(
            features, axis.removeprefix("enemy_"), enemy=True
        )
    if axis.startswith("ally_") and axis.removeprefix("ally_") in CONTEXT_AXIS_INDEX:
        return _team_feature_average_np(
            features, axis.removeprefix("ally_"), enemy=False
        )
    if axis == "enemy_burst_count":
        return _side_feature_count_np(features, "burst", enemy=True)
    if axis == "enemy_hard_cc_count":
        return _side_feature_count_np(features, "hard_cc", enemy=True)
    if axis == "enemy_frontline_count":
        return _side_feature_count_np(features, "frontline", enemy=True)
    if axis == "enemy_heavy_taken_count":
        return _side_feature_count_np(features, "heavy_taken", enemy=True)
    if axis == "enemy_high_hp_count":
        return _side_feature_count_np(features, "high_hp", enemy=True)
    if axis == "enemy_ranged_count":
        return _side_feature_count_np(features, "ranged", enemy=True)
    if axis == "same_role_range":
        ranged = features[:, :, SEMANTIC_GROUP_FEATURE_INDEX["same_role_range"]] > 0.5
        return np.where(ranged, 251.0, 250.0)
    if axis == "ally_skirmish_count":
        return _side_feature_count_np(features, "skirmish", enemy=False)
    raise ValueError(f"unknown audit axis: {axis}")


def _audit_focus_mask_torch(
    raw: RawTensorSplit,
    spec: AuditSpec,
    *,
    hp_lookup: torch.Tensor,
    build_label_ids: dict[str, int],
) -> torch.Tensor:
    if (
        raw.champion_id is None
        or raw.build_id is None
        or raw.semantic_group_features is None
    ):
        raise ValueError(
            "semantic context calibration requires champion/build/group tensors"
        )
    mask = torch.ones_like(raw.champion_id, dtype=torch.bool)
    if spec.champions:
        champion_mask = torch.zeros_like(mask)
        for champion in spec.champions:
            champion_mask |= raw.champion_id == int(champion)
        mask &= champion_mask
    if spec.positions:
        slot_mask = torch.zeros(10, dtype=torch.bool, device=raw.champion_id.device)
        for position in spec.positions:
            idx = POSITIONS.index(position)
            slot_mask[idx] = True
            slot_mask[idx + 5] = True
        mask &= slot_mask.view(1, 10)
    if spec.builds:
        build_mask = torch.zeros_like(mask)
        for label in spec.builds:
            build_idx = build_label_ids.get(label)
            if build_idx is not None:
                build_mask |= raw.build_id == int(build_idx)
        mask &= build_mask

    features = raw.semantic_group_features
    if spec.focus_condition == "low_own_damage":
        side_anchor = torch.zeros(10, dtype=torch.bool, device=raw.champion_id.device)
        side_anchor[0] = True
        side_anchor[5] = True
        mask &= side_anchor.view(1, 10)
        mask &= _team_feature_average_torch(features, "damage", enemy=False) <= (
            LOW_OWN_DAMAGE_THRESHOLD
        )
    elif spec.focus_condition == "focus_hp_low":
        mask &= _slot_hp_torch(raw.champion_id, hp_lookup) <= FOCUS_HP_LOW_THRESHOLD
    elif spec.focus_condition == "focus_hp_high":
        mask &= _slot_hp_torch(raw.champion_id, hp_lookup) >= HIGH_HP_THRESHOLD
    elif spec.focus_condition == "selected_enchanter":
        mask &= features[:, :, SEMANTIC_GROUP_FEATURE_INDEX["selected_enchanter"]] > 0.5
    elif spec.focus_condition is not None:
        raise ValueError(f"unknown focus condition: {spec.focus_condition}")
    return mask


def _audit_focus_mask_np(
    split: SplitData,
    spec: AuditSpec,
    *,
    hp_lookup: np.ndarray,
    build_label_ids: dict[str, int],
) -> np.ndarray:
    if (
        split.champion_id is None
        or split.build_id is None
        or split.semantic_group_features is None
    ):
        raise ValueError("semantic context metrics require champion/build/group arrays")
    mask = np.ones(split.champion_id.shape, dtype=bool)
    if spec.champions:
        mask &= np.isin(split.champion_id, list(spec.champions))
    if spec.positions:
        slot_mask = np.zeros(10, dtype=bool)
        for position in spec.positions:
            idx = POSITIONS.index(position)
            slot_mask[idx] = True
            slot_mask[idx + 5] = True
        mask &= slot_mask[None, :]
    if spec.builds:
        ids = [
            build_label_ids[label] for label in spec.builds if label in build_label_ids
        ]
        mask &= np.isin(split.build_id, ids)

    features = split.semantic_group_features
    if spec.focus_condition == "low_own_damage":
        side_anchor = np.zeros(10, dtype=bool)
        side_anchor[[0, 5]] = True
        mask &= side_anchor[None, :]
        mask &= _team_feature_average_np(features, "damage", enemy=False) <= (
            LOW_OWN_DAMAGE_THRESHOLD
        )
    elif spec.focus_condition == "focus_hp_low":
        mask &= _slot_hp_np(split.champion_id, hp_lookup) <= FOCUS_HP_LOW_THRESHOLD
    elif spec.focus_condition == "focus_hp_high":
        mask &= _slot_hp_np(split.champion_id, hp_lookup) >= HIGH_HP_THRESHOLD
    elif spec.focus_condition == "selected_enchanter":
        mask &= features[:, :, SEMANTIC_GROUP_FEATURE_INDEX["selected_enchanter"]] > 0.5
    elif spec.focus_condition is not None:
        raise ValueError(f"unknown focus condition: {spec.focus_condition}")
    return mask


class _SemanticContextCalibrationLoss:
    def __init__(
        self,
        *,
        build_vocab: tuple[str, ...],
        train_cfg: TrainConfig,
        device: str,
    ) -> None:
        hp_lookup, _ = static_hp_range_lookups()
        self.device = device
        self.hp_lookup = torch.as_tensor(hp_lookup, dtype=torch.float32, device=device)
        self.build_label_ids = {label: idx for idx, label in enumerate(build_vocab)}
        self.target_family = str(train_cfg.semantic_context_calibration_target)
        self.objective = str(train_cfg.semantic_context_calibration_objective)
        self.eb_shrink = self.target_family != "champion_raw"
        if self.target_family == "group_eb":
            self.specs = group_audit_specs()
            self.spec_components = ("group",) * len(self.specs)
        elif self.target_family == "group_context_eb":
            group_specs = group_audit_specs()
            context_specs = audit_specs()
            self.specs = (*group_specs, *context_specs)
            self.spec_components = (
                *(("group",) * len(group_specs)),
                *(("context",) * len(context_specs)),
            )
        else:
            self.specs = audit_specs()
            self.spec_components = ("context",) * len(self.specs)
        self.weight = float(train_cfg.semantic_context_calibration_loss_weight)
        self.min_count = int(train_cfg.semantic_context_calibration_min_count)
        self.tail_weight = float(train_cfg.semantic_context_calibration_tail_weight)
        self.group_residual_shrink_strength = float(
            train_cfg.semantic_context_calibration_group_residual_shrink_strength
        )
        self.group_residual_clip = float(
            train_cfg.semantic_context_calibration_group_residual_clip
        )
        self.group_residual_scale = float(
            train_cfg.semantic_context_calibration_group_residual_scale
        )
        self.context_residual_shrink_strength = float(
            train_cfg.semantic_context_calibration_context_residual_shrink_strength
        )
        self.context_residual_clip = float(
            train_cfg.semantic_context_calibration_context_residual_clip
        )
        self.context_residual_scale = float(
            train_cfg.semantic_context_calibration_context_residual_scale
        )
        self.reference_targets: dict[tuple[int, int], torch.Tensor] = {}
        self.reference_counts: dict[tuple[int, int], int] = {}
        self.reference_residual_targets: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0

    @property
    def residual_objective(self) -> bool:
        return self.objective == "residual"

    def _residual_params(self, component: str) -> tuple[float, float, float]:
        if component == "group":
            return (
                self.group_residual_shrink_strength,
                self.group_residual_clip,
                self.group_residual_scale,
            )
        return (
            self.context_residual_shrink_strength,
            self.context_residual_clip,
            self.context_residual_scale,
        )

    def fit_reference(
        self,
        raw: RawTensorSplit,
        *,
        reference_predictions: torch.Tensor | None = None,
        reference_semantic_deltas: torch.Tensor | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self.residual_objective and (
            reference_predictions is None or reference_semantic_deltas is None
        ):
            raise ValueError(
                "residual semantic context calibration requires warm-start "
                "reference_predictions and reference_semantic_deltas"
            )
        if raw.semantic_group_features is None:
            raise ValueError(
                "semantic context calibration requires semantic_group_features; "
                "enable --use-semantic-group-features or the calibration loss."
            )
        targets = _side_labels_torch(raw.blue_win)
        self.reference_targets.clear()
        self.reference_counts.clear()
        self.reference_residual_targets.clear()
        device = targets.device
        hp_lookup = self.hp_lookup.to(device=device, non_blocking=True)
        for spec_idx, spec in enumerate(self.specs):
            focus = _audit_focus_mask_torch(
                raw,
                spec,
                hp_lookup=hp_lookup,
                build_label_ids=self.build_label_ids,
            )
            axis = _audit_axis_torch(raw.semantic_group_features, spec)
            bin_idxs: list[int] = []
            means: list[float] = []
            counts: list[int] = []
            for bin_idx, bin_spec in enumerate(spec.bins):
                mask = focus & bin_spec.predicate(axis)
                count = int(mask.sum().detach().cpu().item())
                if count < self.min_count:
                    continue
                bin_idxs.append(bin_idx)
                means.append(float(targets[mask].mean().detach().cpu().item()))
                counts.append(count)
            if not bin_idxs:
                continue
            if self.eb_shrink:
                eb, _ = eb_shrink_targets(np.asarray(counts), np.asarray(means))
                target_values = eb.tolist()
            else:
                target_values = means
            for bin_idx, value, count in zip(bin_idxs, target_values, counts):
                key = (spec_idx, bin_idx)
                self.reference_targets[key] = torch.tensor(
                    float(value), dtype=torch.float32, device=self.device
                )
                self.reference_counts[key] = count
                if self.residual_objective:
                    assert reference_predictions is not None
                    assert reference_semantic_deltas is not None
                    focus = _audit_focus_mask_torch(
                        raw,
                        spec,
                        hp_lookup=hp_lookup,
                        build_label_ids=self.build_label_ids,
                    )
                    axis = _audit_axis_torch(raw.semantic_group_features, spec)
                    mask = focus & spec.bins[bin_idx].predicate(axis)
                    reference_mean = reference_predictions[mask].mean()
                    initial_delta = reference_semantic_deltas[mask].mean()
                    raw_delta = (
                        _probability_logit_torch(
                            torch.tensor(float(value), dtype=torch.float32, device=device)
                        )
                        - _probability_logit_torch(reference_mean)
                    )
                    shrink_strength, clip, scale = self._residual_params(
                        self.spec_components[spec_idx]
                    )
                    shrink = float(count) / (float(count) + float(shrink_strength))
                    correction = torch.clamp(
                        raw_delta * float(shrink),
                        min=-float(clip),
                        max=float(clip),
                    )
                    total_delta = initial_delta + float(scale) * correction
                    self.reference_residual_targets[key] = total_delta.detach().to(
                        self.device, non_blocking=True
                    )

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        raw: RawTensorSplit,
    ) -> torch.Tensor:
        logits = outputs["final_logit"]
        if not self.enabled:
            return logits.new_zeros(())
        if raw.semantic_group_features is None:
            raise ValueError(
                "semantic context calibration requires semantic_group_features; "
                "enable --use-semantic-group-features or the calibration loss."
            )
        predictions = _focus_side_probabilities_from_outputs(outputs)
        if self.residual_objective:
            focus_logits = _focus_side_logits_from_outputs(outputs)
            base_logits = _focus_side_logits_from_outputs(
                outputs,
                include_semantic_delta=False,
            )
            semantic_deltas = focus_logits - base_logits
        else:
            semantic_deltas = None
        targets = _side_labels_torch(labels)
        total = logits.new_zeros(())
        denom = 0.0
        for spec_idx, spec in enumerate(self.specs):
            focus = _audit_focus_mask_torch(
                raw,
                spec,
                hp_lookup=self.hp_lookup,
                build_label_ids=self.build_label_ids,
            )
            axis = _audit_axis_torch(raw.semantic_group_features, spec)
            last_bin_idx = len(spec.bins) - 1
            for bin_idx, bin_spec in enumerate(spec.bins):
                mask = focus & bin_spec.predicate(axis)
                count = int(mask.sum().detach().cpu().item())
                if count < self.min_count:
                    continue
                reference_target = self.reference_targets.get((spec_idx, bin_idx))
                target = (
                    targets[mask].mean()
                    if reference_target is None
                    else reference_target
                )
                if self.residual_objective:
                    reference_delta = self.reference_residual_targets.get(
                        (spec_idx, bin_idx)
                    )
                    if reference_delta is None:
                        continue
                    assert semantic_deltas is not None
                    gap = semantic_deltas[mask].mean() - reference_delta
                else:
                    gap = predictions[mask].mean() - target
                bin_weight = self.tail_weight if bin_idx in {0, last_bin_idx} else 1.0
                total = total + float(bin_weight) * gap.square()
                denom += float(bin_weight)
        if denom <= 0.0:
            return logits.new_zeros(())
        return self.weight * total / denom


def _semantic_reference_predictions(
    model: HGNNWinModel,
    raw: RawTensorSplit,
    *,
    strength: float,
    device: str,
    gatherer: EncoderSidecarLookup | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = model.training
    model.eval()
    probabilities: list[torch.Tensor] = []
    semantic_deltas: list[torch.Tensor] = []
    reference_device = raw.blue_win.device
    with torch.no_grad():
        for batch_idx in _batch_indices(
            raw.blue_win.shape[0],
            batch_size=batch_size,
            shuffle=False,
            rng=np.random.default_rng(0),
        ):
            rows = _raw_index_tensor(raw, batch_idx)
            raw_batch = _raw_split_to_device(_raw_batch(raw, rows), device=device)
            outputs = model(
                **_hgnn_inputs_from_raw(
                    raw_batch,
                    strength=strength,
                    device=device,
                    gatherer=gatherer,
                )
            )
            focus_logits = _focus_side_logits_from_outputs(outputs)
            base_logits = _focus_side_logits_from_outputs(
                outputs,
                include_semantic_delta=False,
            )
            probabilities.append(
                torch.sigmoid(focus_logits).detach().to(
                    reference_device, non_blocking=True
                )
            )
            semantic_deltas.append(
                (focus_logits - base_logits).detach().to(
                    reference_device, non_blocking=True
                )
            )
    if was_training:
        model.train()
    return torch.cat(probabilities, dim=0), torch.cat(semantic_deltas, dim=0)


def _swap_context_calibration_raw(raw: RawTensorSplit) -> RawTensorSplit:
    def swap_slots(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return torch.cat([value[:, 5:], value[:, :5]], dim=1)

    def flip_signed(
        value: torch.Tensor | None,
        signed_indices: tuple[int, ...],
    ) -> torch.Tensor | None:
        if value is None:
            return None
        out = value.clone()
        if signed_indices:
            idx = torch.as_tensor(signed_indices, dtype=torch.long, device=value.device)
            out.index_copy_(1, idx, -value.index_select(1, idx))
        return out

    return RawTensorSplit(
        win_rate=raw.win_rate,
        p1_cnt=raw.p1_cnt,
        blue_win=raw.blue_win,
        champion_id=swap_slots(raw.champion_id),
        build_id=swap_slots(raw.build_id),
        semantic_group_features=swap_slots(raw.semantic_group_features),
        loadout_features=flip_signed(
            raw.loadout_features,
            LOADOUT_SIGNED_FEATURE_INDICES,
        ),
        patch_features=flip_signed(
            raw.patch_features,
            PATCH_SIGNED_FEATURE_INDICES,
        ),
    )


def _semantic_context_gap_metrics(
    scores: np.ndarray,
    split: SplitData,
    *,
    build_vocab: tuple[str, ...],
) -> dict[str, float | int]:
    if (
        split.context_raw is None
        or split.champion_id is None
        or split.build_id is None
    ):
        return {
            "context_gap_mse": float("nan"),
            "context_mean_abs_gap": float("nan"),
            "context_max_abs_gap": float("nan"),
            "context_populated_bins": 0,
        }
    lens = AuditLens(
        champion_id=split.champion_id,
        build_id=split.build_id,
        context_raw=split.context_raw,
        build_vocab=build_vocab,
    )
    predictions = _side_probabilities_np(scores)
    targets = _side_labels_np(split.blue_win)
    gaps: list[float] = []
    for spec in audit_specs():
        focus = lens.focus_mask(spec)
        axis = lens.axis(spec.axis)
        for bin_spec in spec.bins:
            mask = focus & bin_spec.predicate(axis)
            if not np.any(mask):
                continue
            gaps.append(float(np.mean(predictions[mask]) - np.mean(targets[mask])))
    if not gaps:
        return {
            "context_gap_mse": float("nan"),
            "context_mean_abs_gap": float("nan"),
            "context_max_abs_gap": float("nan"),
            "context_populated_bins": 0,
        }
    gap_array = np.asarray(gaps, dtype=np.float64) * 100.0
    return {
        "context_gap_mse": float(np.mean(gap_array**2)),
        "context_mean_abs_gap": float(np.mean(np.abs(gap_array))),
        "context_max_abs_gap": float(np.max(np.abs(gap_array))),
        "context_populated_bins": int(gap_array.size),
    }


def _semantic_group_eb_gap_metrics(
    scores: np.ndarray,
    split: SplitData,
    *,
    build_vocab: tuple[str, ...],
) -> dict[str, float | int]:
    if (
        split.context_raw is None
        or split.champion_id is None
        or split.build_id is None
    ):
        return {
            "group_n_bins": 0,
            "group_median_n": float("nan"),
            "group_min_n": 0,
            "group_raw_gap_mse": float("nan"),
            "group_raw_floor": float("nan"),
            "group_eb_gap_mse": float("nan"),
            "group_eb_floor": float("nan"),
            "group_systematic_gap_mse": float("nan"),
            "group_systematic_gap_mse_clipped": float("nan"),
            "group_eb_mean_abs_gap": float("nan"),
            "group_eb_max_abs_gap": float("nan"),
        }
    lens = AuditLens(
        champion_id=split.champion_id,
        build_id=split.build_id,
        context_raw=split.context_raw,
        build_vocab=build_vocab,
    )
    predictions = _side_probabilities_np(scores)
    targets = _side_labels_np(split.blue_win)
    raw_gaps: list[float] = []
    eb_gaps: list[float] = []
    sampling_vars: list[float] = []
    eb_vars: list[float] = []
    counts_all: list[int] = []
    for spec in group_audit_specs():
        focus = lens.focus_mask(spec)
        axis = lens.axis(spec.axis)
        counts: list[int] = []
        empirical: list[float] = []
        hgnn: list[float] = []
        for bin_spec in spec.bins:
            mask = focus & bin_spec.predicate(axis)
            count = int(mask.sum())
            if count <= 0:
                continue
            counts.append(count)
            empirical.append(float(np.mean(targets[mask])))
            hgnn.append(float(np.mean(predictions[mask])))
        if not counts:
            continue
        count_array = np.asarray(counts, dtype=np.float64)
        empirical_array = np.asarray(empirical, dtype=np.float64)
        hgnn_array = np.asarray(hgnn, dtype=np.float64)
        eb_target, eb_var = eb_shrink_targets(count_array, empirical_array)
        raw_gap = (hgnn_array - empirical_array) * 100.0
        eb_gap = (hgnn_array - eb_target) * 100.0
        sampling_var = empirical_array * (1.0 - empirical_array) / count_array
        raw_gaps.extend(raw_gap.tolist())
        eb_gaps.extend(eb_gap.tolist())
        sampling_vars.extend((sampling_var * 1.0e4).tolist())
        eb_vars.extend((eb_var * 1.0e4).tolist())
        counts_all.extend(counts)
    if not eb_gaps:
        return {
            "group_n_bins": 0,
            "group_median_n": float("nan"),
            "group_min_n": 0,
            "group_raw_gap_mse": float("nan"),
            "group_raw_floor": float("nan"),
            "group_eb_gap_mse": float("nan"),
            "group_eb_floor": float("nan"),
            "group_systematic_gap_mse": float("nan"),
            "group_systematic_gap_mse_clipped": float("nan"),
            "group_eb_mean_abs_gap": float("nan"),
            "group_eb_max_abs_gap": float("nan"),
        }
    raw_gap_array = np.asarray(raw_gaps, dtype=np.float64)
    eb_gap_array = np.asarray(eb_gaps, dtype=np.float64)
    sampling_var_array = np.asarray(sampling_vars, dtype=np.float64)
    eb_var_array = np.asarray(eb_vars, dtype=np.float64)
    return {
        "group_n_bins": int(eb_gap_array.size),
        "group_median_n": float(np.median(counts_all)),
        "group_min_n": int(min(counts_all)),
        "group_raw_gap_mse": float(np.mean(raw_gap_array**2)),
        "group_raw_floor": float(np.mean(sampling_var_array)),
        "group_eb_gap_mse": float(np.mean(eb_gap_array**2)),
        "group_eb_floor": float(np.mean(eb_var_array)),
        "group_systematic_gap_mse": float(
            np.mean(eb_gap_array**2) - np.mean(eb_var_array)
        ),
        "group_systematic_gap_mse_clipped": float(
            np.mean(np.maximum(0.0, eb_gap_array**2 - eb_var_array))
        ),
        "group_eb_mean_abs_gap": float(np.mean(np.abs(eb_gap_array))),
        "group_eb_max_abs_gap": float(np.max(np.abs(eb_gap_array))),
    }


def _predict_hgnn(
    model: HGNNWinModel,
    split: RawTensorSplit,
    *,
    batch_size: int,
    strength: float,
    device: str,
    gatherer: _SidecarGatherer | None = None,
) -> np.ndarray:
    return _sigmoid_np(
        _predict_hgnn_logits(
            model,
            split,
            batch_size=batch_size,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
    )


def _warm_start_hgnn_model(
    model: HGNNWinModel,
    path: Path,
    *,
    device: str,
) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(f"warm-start HGNN checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"warm-start HGNN checkpoint is invalid: {path}")
    checkpoint_state = payload["state_dict"]
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"warm-start HGNN checkpoint state_dict is invalid: {path}")
    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped_shape_mismatches: list[str] = []
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is not None and value.shape != target.shape:
            skipped_shape_mismatches.append(
                f"{key}: checkpoint={tuple(value.shape)} model={tuple(target.shape)}"
            )
            continue
        compatible_state[key] = value
    if skipped_shape_mismatches:
        logger.warning(
            "Skipped %s warm-start tensors with incompatible shapes from %s: %s",
            len(skipped_shape_mismatches),
            _project_relative(path),
            "; ".join(skipped_shape_mismatches[:8]),
        )
    incompatible = model.load_state_dict(compatible_state, strict=False)
    logger.info(
        "Warm-started HGNN model from %s missing=%s unexpected=%s",
        _project_relative(path),
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )
    return tuple(str(key) for key in incompatible.missing_keys)


def _freeze_warm_start_loaded_parameters(
    model: HGNNWinModel,
    *,
    missing_keys: tuple[str, ...],
) -> None:
    missing = set(missing_keys)
    frozen = 0
    trainable = 0
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        is_new = name in missing
        parameter.requires_grad_(is_new)
        if is_new:
            trainable += parameter.numel()
            trainable_names.append(name)
        else:
            frozen += parameter.numel()
    if trainable <= 0:
        raise ValueError(
            "freeze_warm_start_loaded_parameters left no trainable parameters; "
            "the warm-start checkpoint appears to match the model shape."
        )
    logger.info(
        "Froze %s warm-start-loaded parameters; training %s new parameters (%s)",
        frozen,
        trainable,
        ", ".join(trainable_names[:12]) + ("..." if len(trainable_names) > 12 else ""),
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
    # Cap the training batch by default because each step also runs a
    # team-swapped copy. Explicit throughput sweeps can disable the cap.
    train_batch_cap = train_cfg.train_batch_cap
    if train_batch_cap is None or train_batch_cap == 0:
        train_batch_size = train_cfg.batch_size
    else:
        train_batch_size = min(train_cfg.batch_size, train_batch_cap)
    train_epoch_max_games = (
        None
        if train_cfg.train_epoch_max_games in (None, 0)
        else int(train_cfg.train_epoch_max_games)
    )
    if model_overrides is None:
        model_overrides = production_semantic_model_overrides()

    meta = identity_meta(dataset_cfg)
    model_config = _hgnn_config_from_meta(
        meta,
        encoder_sidecar_path=dataset_cfg.encoder_sidecar_path,
        overrides=model_overrides,
    )

    semantic_context_calibration_enabled = (
        train_cfg.semantic_context_calibration_loss_weight > 0.0
    )
    load_semantic_group_features = bool(
        semantic_context_calibration_enabled
        or (
            model_config.use_learned_semantic_moe
            and model_config.use_semantic_group_features
        )
    )
    loaded_splits = {
        name: _limit_split(split, dataset_cfg.max_games)
        for name, split in load_splits(
            dataset_cfg,
            require_counts=True,
            load_semantic_group_features=load_semantic_group_features,
            load_context_raw=load_semantic_group_features,
        ).items()
    }
    # v28 caches omit per-game sidecar arrays; build the on-device gather table
    # from the frozen artifact when the model consumes identity latents.
    gatherer = None
    if (
        _model_uses_sidecar(model_config)
        and loaded_splits["train"].identity_static_sidecar is None
    ):
        gatherer = _build_sidecar_gatherer(
            dataset_cfg, meta, model_config, device=device
        )
    splits = {
        name: _drop_unused_model_arrays(
            split,
            model_config,
            keep_semantic_group_features=semantic_context_calibration_enabled,
        )
        for name, split in loaded_splits.items()
    }
    _validate_split_targets(splits)
    if splits["train"].blue_win.size == 0:
        raise ValueError("Training split is empty; rebuild the cache with train games.")
    raw_cache_device = (
        "cpu" if train_cfg.raw_tensor_cache_device == "cpu" else device
    )
    tensor_splits = {
        name: _cache_raw_tensor_split(name, splits[name], device=raw_cache_device)
        for name in ("train", "val")
    }

    model = HGNNWinModel(model_config).to(device)
    warm_start_missing_keys: tuple[str, ...] = ()
    if train_cfg.warm_start_model_path is not None:
        warm_start_missing_keys = _warm_start_hgnn_model(
            model,
            train_cfg.warm_start_model_path,
            device=device,
        )
    if train_cfg.freeze_warm_start_loaded_parameters:
        _freeze_warm_start_loaded_parameters(
            model,
            missing_keys=warm_start_missing_keys,
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    context_calibration_loss = _SemanticContextCalibrationLoss(
        build_vocab=tuple(model_config.build_vocab),
        train_cfg=train_cfg,
        device=device,
    )
    reference_predictions: torch.Tensor | None = None
    reference_semantic_deltas: torch.Tensor | None = None
    if context_calibration_loss.enabled and context_calibration_loss.residual_objective:
        logger.info("Fitting residual semantic calibration teacher from warm-start outputs")
        reference_predictions, reference_semantic_deltas = _semantic_reference_predictions(
            model,
            tensor_splits["train"],
            strength=strength,
            device=device,
            gatherer=gatherer,
            batch_size=train_batch_size,
        )
    context_calibration_loss.fit_reference(
        tensor_splits["train"],
        reference_predictions=reference_predictions,
        reference_semantic_deltas=reference_semantic_deltas,
    )
    rng = np.random.default_rng(train_cfg.seed)
    semantic_moe_enabled = bool(model_config.use_learned_semantic_moe)
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
        "HGNN training device=%s raw_tensor_cache_device=%s batch_size=%s requested_batch_size=%s train_batch_cap=%s train_epoch_max_games=%s max_epochs=%s strength=%s checkpoint_metric=%s min_delta=%s",
        device,
        raw_cache_device,
        train_batch_size,
        train_cfg.batch_size,
        train_batch_cap,
        train_epoch_max_games,
        train_cfg.max_epochs,
        strength,
        train_cfg.checkpoint_metric,
        train_cfg.checkpoint_min_delta,
    )
    if device == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    def synchronize_training_device() -> None:
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    if train_cfg.warm_start_model_path is not None:
        val_outputs = _predict_hgnn_outputs(
            model,
            tensor_splits["val"],
            batch_size=train_cfg.batch_size,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
        val_logits = val_outputs["final_logit"]
        val_predictions = _sigmoid_np(val_logits)
        val_metrics = _evaluate_predictions(val_predictions, splits["val"])
        if load_semantic_group_features:
            val_metrics.update(
                _semantic_context_gap_metrics(
                    val_outputs["focus_side_probability"],
                    splits["val"],
                    build_vocab=tuple(model_config.build_vocab),
                )
            )
            val_metrics.update(
                _semantic_group_eb_gap_metrics(
                    val_outputs["focus_side_probability"],
                    splits["val"],
                    build_vocab=tuple(model_config.build_vocab),
                )
            )
        val_threshold, val_threshold_accuracy = _select_threshold(
            val_predictions,
            splits["val"].blue_win,
        )
        best_val_nll = float(val_metrics["nll"])
        best_checkpoint_val_nll = float(val_metrics["nll"])
        best_checkpoint_val_ece = float(val_metrics["ece"])
        best_checkpoint_score = _checkpoint_score(
            train_cfg.checkpoint_metric,
            val_metrics=val_metrics,
            val_threshold_accuracy=val_threshold_accuracy,
        )
        best_epoch = 0
        best_threshold = val_threshold
        best_state = copy.deepcopy(model.state_dict())
        if "context_gap_mse" in val_metrics:
            logger.info(
                "epoch=0 warm_start val_nll=%.5f val_context_gap_mse=%.4f mean_abs=%.3f max_abs=%.3f",
                val_metrics["nll"],
                val_metrics["context_gap_mse"],
                val_metrics["context_mean_abs_gap"],
                val_metrics["context_max_abs_gap"],
            )
        else:
            logger.info(
                "epoch=0 warm_start val_nll=%.5f val_acc=%.4f val_thr=%.3f val_thr_acc=%.4f",
                val_metrics["nll"],
                val_metrics["accuracy"],
                val_threshold,
                val_threshold_accuracy,
            )

    for epoch in range(1, train_cfg.max_epochs + 1):
        synchronize_training_device()
        epoch_started = time.perf_counter()
        train_started = epoch_started
        model.train()
        train_loss_sum = 0.0
        train_rank_loss_sum = 0.0
        train_semantic_moe_regularization_loss_sum = 0.0
        train_context_calibration_loss_sum = 0.0
        train_seen = 0
        for batch_idx in _batch_indices(
            splits["train"].blue_win.size,
            batch_size=train_batch_size,
            shuffle=True,
            rng=rng,
            max_rows=train_epoch_max_games,
        ):
            raw_batch = _raw_split_to_device(
                _raw_batch(
                    tensor_splits["train"],
                    _raw_index_tensor(tensor_splits["train"], batch_idx),
                ),
                device=device,
            )
            inputs = _hgnn_inputs_from_raw(
                raw_batch, strength=strength, device=device, gatherer=gatherer
            )
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
            direct_semantic_moe_regularization_loss = (
                model.semantic_moe_regularization_loss(direct_outputs)
            )
            direct_context_calibration_loss = context_calibration_loss(
                direct_outputs,
                labels,
                raw_batch,
            )
            (
                0.5
                * (
                    direct_loss
                    + direct_rank_loss
                    + direct_semantic_moe_regularization_loss
                    + direct_context_calibration_loss
                )
            ).backward()
            swapped_outputs = model(**swap_hgnn_inputs(inputs))
            swapped_loss = loss_fn(swapped_outputs["final_logit"], 1.0 - labels)
            swapped_rank_loss = _auc_ranking_loss(
                swapped_outputs["final_logit"],
                1.0 - labels,
                weight=train_cfg.auc_ranking_loss_weight,
                max_pairs=train_cfg.auc_ranking_loss_pairs,
            )
            swapped_semantic_moe_regularization_loss = (
                model.semantic_moe_regularization_loss(swapped_outputs)
            )
            swapped_context_calibration_loss = context_calibration_loss(
                swapped_outputs,
                1.0 - labels,
                _swap_context_calibration_raw(raw_batch),
            )
            (
                0.5
                * (
                    swapped_loss
                    + swapped_rank_loss
                    + swapped_semantic_moe_regularization_loss
                    + swapped_context_calibration_loss
                )
            ).backward()
            loss = 0.5 * (direct_loss.detach() + swapped_loss.detach())
            rank_loss = 0.5 * (direct_rank_loss.detach() + swapped_rank_loss.detach())
            semantic_moe_regularization_loss = 0.5 * (
                direct_semantic_moe_regularization_loss.detach()
                + swapped_semantic_moe_regularization_loss.detach()
            )
            context_calibration_loss_value = 0.5 * (
                direct_context_calibration_loss.detach()
                + swapped_context_calibration_loss.detach()
            )
            if train_cfg.max_grad_norm is not None and train_cfg.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            train_loss_sum += float(loss.cpu().item()) * labels.numel() * 2
            train_rank_loss_sum += float(rank_loss.cpu().item()) * labels.numel() * 2
            if semantic_moe_enabled:
                train_semantic_moe_regularization_loss_sum += (
                    float(semantic_moe_regularization_loss.cpu().item())
                    * labels.numel()
                    * 2
                )
            if context_calibration_loss.enabled:
                train_context_calibration_loss_sum += (
                    float(context_calibration_loss_value.cpu().item())
                    * labels.numel()
                    * 2
                )
            train_seen += int(labels.numel() * 2)

        synchronize_training_device()
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        val_outputs = _predict_hgnn_outputs(
            model,
            tensor_splits["val"],
            batch_size=train_cfg.batch_size,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
        synchronize_training_device()
        val_logits = val_outputs["final_logit"]
        val_predictions = _sigmoid_np(val_logits)
        train_nll = train_loss_sum / max(train_seen, 1)
        train_auc_ranking_loss = train_rank_loss_sum / max(train_seen, 1)
        train_semantic_moe_regularization_loss = (
            train_semantic_moe_regularization_loss_sum / max(train_seen, 1)
        )
        train_context_calibration_loss = train_context_calibration_loss_sum / max(
            train_seen, 1
        )
        val_metrics = _evaluate_predictions(val_predictions, splits["val"])
        if load_semantic_group_features:
            val_metrics.update(
                _semantic_context_gap_metrics(
                    val_outputs["focus_side_probability"],
                    splits["val"],
                    build_vocab=tuple(model_config.build_vocab),
                )
            )
            val_metrics.update(
                _semantic_group_eb_gap_metrics(
                    val_outputs["focus_side_probability"],
                    splits["val"],
                    build_vocab=tuple(model_config.build_vocab),
                )
            )
        val_seconds = time.perf_counter() - val_started
        epoch_seconds = time.perf_counter() - epoch_started
        train_rows_per_second = (train_seen / 2.0) / max(train_seconds, EPS)
        train_epoch_games_seen = int(train_seen // 2)
        train_augmented_samples_per_second = train_seen / max(train_seconds, EPS)
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
        history_row: dict[str, float | int] = {
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
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "epoch_seconds": epoch_seconds,
            "train_rows_per_second": train_rows_per_second,
            "train_epoch_games_seen": train_epoch_games_seen,
            "train_epoch_max_games": 0
            if train_epoch_max_games is None
            else int(train_epoch_max_games),
            "train_augmented_samples_per_second": train_augmented_samples_per_second,
            "train_batch_size": train_batch_size,
        }
        if "context_gap_mse" in val_metrics:
            history_row["val_context_gap_mse"] = float(val_metrics["context_gap_mse"])
            history_row["val_context_mean_abs_gap"] = float(
                val_metrics["context_mean_abs_gap"]
            )
            history_row["val_context_max_abs_gap"] = float(
                val_metrics["context_max_abs_gap"]
            )
        if "group_systematic_gap_mse" in val_metrics:
            history_row["val_group_eb_gap_mse"] = float(
                val_metrics["group_eb_gap_mse"]
            )
            history_row["val_group_eb_floor"] = float(val_metrics["group_eb_floor"])
            history_row["val_group_floor_normalized_eb_gap"] = float(
                val_metrics["group_eb_gap_mse"]
            ) / max(float(val_metrics["group_eb_floor"]), EPS)
            history_row["val_group_systematic_gap_mse"] = float(
                val_metrics["group_systematic_gap_mse"]
            )
            history_row["val_group_systematic_gap_mse_clipped"] = float(
                val_metrics["group_systematic_gap_mse_clipped"]
            )
            history_row["val_group_eb_mean_abs_gap"] = float(
                val_metrics["group_eb_mean_abs_gap"]
            )
        if semantic_moe_enabled:
            history_row["train_semantic_moe_regularization_loss"] = (
                train_semantic_moe_regularization_loss
            )
        if context_calibration_loss.enabled:
            history_row["train_context_calibration_loss"] = (
                train_context_calibration_loss
            )
        history.append(history_row)
        if semantic_moe_enabled:
            logger.info(
                "epoch=%s train_nll=%.5f rank=%.5f semantic_moe_reg=%.5f val_nll=%.5f val_acc=%.4f val_thr=%.3f val_thr_acc=%.4f",
                epoch,
                train_nll,
                train_auc_ranking_loss,
                train_semantic_moe_regularization_loss,
                val_nll,
                val_metrics["accuracy"],
                val_threshold,
                val_threshold_accuracy,
            )
        else:
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
        if "context_gap_mse" in val_metrics:
            logger.info(
                "epoch=%s context_cal=%.6f val_context_gap_mse=%.4f mean_abs=%.3f max_abs=%.3f",
                epoch,
                train_context_calibration_loss,
                val_metrics["context_gap_mse"],
                val_metrics["context_mean_abs_gap"],
                val_metrics["context_max_abs_gap"],
            )
        if "group_systematic_gap_mse" in val_metrics:
            logger.info(
                "epoch=%s val_group_systematic_gap_mse=%.4f eb_mse=%.4f mean_abs=%.3f max_abs=%.3f",
                epoch,
                val_metrics["group_systematic_gap_mse"],
                val_metrics["group_eb_gap_mse"],
                val_metrics["group_eb_mean_abs_gap"],
                val_metrics["group_eb_max_abs_gap"],
            )
        logger.info(
            "epoch=%s timing train_seconds=%.2f val_seconds=%.2f epoch_seconds=%.2f train_games=%s train_rows_per_s=%.1f train_augmented_samples_per_s=%.1f batch_size=%s",
            epoch,
            train_seconds,
            val_seconds,
            epoch_seconds,
            train_epoch_games_seen,
            train_rows_per_second,
            train_augmented_samples_per_second,
            train_batch_size,
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

    if train_cfg.skip_final_evaluation:
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
            "elapsed_seconds": time.monotonic() - started,
            "history": history,
            "final_evaluation_skipped": True,
        }
        _write_metrics(train_cfg.metrics_path, metrics)
        logger.info("Saved HGNN model: %s", _project_relative(train_cfg.model_path))
        logger.info(
            "Saved timing-only metrics: %s",
            _project_relative(train_cfg.metrics_path),
        )
        return train_cfg.model_path

    tensor_splits["test"] = _cache_raw_tensor_split(
        "test", splits["test"], device=raw_cache_device
    )
    prediction_outputs = {
        split_name: _predict_hgnn_outputs(
            model,
            tensor_split,
            batch_size=train_cfg.batch_size,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
        for split_name, tensor_split in tensor_splits.items()
    }
    if train_cfg.audit_prediction_cache_path is not None:
        audit_probabilities = np.concatenate(
            [
                prediction_outputs[split_name]["focus_side_probability"]
                for split_name in ("train", "val", "test")
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        train_cfg.audit_prediction_cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        np.save(train_cfg.audit_prediction_cache_path, audit_probabilities)
        logger.info(
            "Saved audit prediction cache: %s",
            _project_relative(train_cfg.audit_prediction_cache_path),
        )
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
    if load_semantic_group_features:
        for split_name in ("train", "val", "test"):
            split_metrics[split_name].update(
                _semantic_context_gap_metrics(
                    prediction_outputs[split_name]["focus_side_probability"],
                    splits[split_name],
                    build_vocab=tuple(model_config.build_vocab),
                )
            )
            split_metrics[split_name].update(
                _semantic_group_eb_gap_metrics(
                    prediction_outputs[split_name]["focus_side_probability"],
                    splits[split_name],
                    build_vocab=tuple(model_config.build_vocab),
                )
            )
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
        calibrated = _evaluate_predictions(
            calibrated_predictions[split_name], splits[split_name]
        )
        calibrated["support_buckets"] = _support_bucket_metrics(
            calibrated_predictions[split_name],
            splits[split_name],
        )
        split_metrics[split_name]["temperature_scaled"] = calibrated
    _attach_output_diagnostics(split_metrics, prediction_outputs)
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
            logit_diagnostics = m.get("logit_diagnostics", {})
            logger.info(
                "%s n=%s acc=%.4f thr_acc=%.4f auc=%.4f nll=%.4f ece=%.4f brier=%.4f base_logit_std=%.4f context_logit_std=%.4f final_logit_std=%.4f",
                split_name,
                m["n"],
                m["accuracy"],
                m["threshold_accuracy"],
                m["auc"],
                m["nll"],
                m["ece"],
                m["brier"],
                logit_diagnostics.get("base_logit_std", float("nan")),
                logit_diagnostics.get("context_logit_std", float("nan")),
                logit_diagnostics.get("final_logit_std", float("nan")),
            )
            if "context_gap_mse" in m:
                logger.info(
                    "%s context_gap_mse=%.4f mean_abs=%.3f max_abs=%.3f populated_bins=%s",
                    split_name,
                    m["context_gap_mse"],
                    m["context_mean_abs_gap"],
                    m["context_max_abs_gap"],
                    m["context_populated_bins"],
                )
            if "group_systematic_gap_mse" in m:
                logger.info(
                    "%s group_systematic_gap_mse=%.4f eb_mse=%.4f mean_abs=%.3f max_abs=%.3f bins=%s",
                    split_name,
                    m["group_systematic_gap_mse"],
                    m["group_eb_gap_mse"],
                    m["group_eb_mean_abs_gap"],
                    m["group_eb_max_abs_gap"],
                    m["group_n_bins"],
                )
            semantic_moe = m.get("semantic_moe_diagnostics")
            if isinstance(semantic_moe, dict):
                logger.info(
                    "%s semantic_moe usage=%s selected=%s router_entropy=%.4f factor_orthogonality=%.6f factor_variance=%.6f factor_std_mean=%.4f factor_std_min=%.4f token_keep=%.4f reg=%.6f",
                    split_name,
                    _json_value(semantic_moe.get("expert_usage", [])),
                    _json_value(semantic_moe.get("expert_selected_fraction", [])),
                    semantic_moe.get("router_entropy", float("nan")),
                    semantic_moe.get("factor_orthogonality_loss", float("nan")),
                    semantic_moe.get("factor_variance_loss", float("nan")),
                    semantic_moe.get("factor_std_mean", float("nan")),
                    semantic_moe.get("factor_std_min", float("nan")),
                    semantic_moe.get("context_token_keep_fraction", float("nan")),
                    semantic_moe.get("regularization_loss", float("nan")),
                )

    return train_cfg.model_path


def _model_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """HGNNConfig overrides exposed by the training CLI."""
    use_all_sidecars = bool(args.use_all_identity_sidecars)
    return {
        "structural_antisymmetry": args.structural_antisymmetry,
        "structural_antisymmetry_scale": args.structural_antisymmetry_scale,
        "use_identity_static_sidecar": bool(
            args.use_identity_static_sidecar or use_all_sidecars
        ),
        "use_identity_full_game_sidecar": bool(
            args.use_identity_full_game_sidecar or use_all_sidecars
        ),
        "use_identity_temporal_sidecar": bool(
            args.use_identity_temporal_sidecar or use_all_sidecars
        ),
        "identity_encoder_sidecar_support_strength": args.identity_encoder_sidecar_support_strength,
        "identity_encoder_sidecar_dropout": args.identity_encoder_sidecar_dropout,
        "use_identity_semantic_context_head": args.use_identity_semantic_context_head,
        "semantic_context_dim": args.semantic_context_dim,
        "semantic_context_dropout": args.semantic_context_dropout,
        "semantic_context_support_strength": args.semantic_context_support_strength,
        "use_learned_semantic_moe": args.use_learned_semantic_moe,
        "semantic_moe_num_experts": args.semantic_moe_num_experts,
        "semantic_moe_top_k": args.semantic_moe_top_k,
        "semantic_moe_factor_dim": args.semantic_moe_factor_dim,
        "semantic_moe_factor_hidden": args.semantic_moe_factor_hidden,
        "semantic_moe_router_hidden": args.semantic_moe_router_hidden,
        "semantic_moe_expert_hidden": args.semantic_moe_expert_hidden,
        "semantic_moe_dropout": args.semantic_moe_dropout,
        "semantic_moe_context_token_dropout": args.semantic_moe_context_token_dropout,
        "semantic_moe_architecture": args.semantic_moe_architecture,
        "semantic_moe_view_gate_hidden": args.semantic_moe_view_gate_hidden,
        "semantic_moe_view_top_k": args.semantic_moe_view_top_k,
        "semantic_moe_view_router_noise": args.semantic_moe_view_router_noise,
        "semantic_moe_view_balance_weight": args.semantic_moe_view_balance_weight,
        "semantic_moe_view_entropy_weight": args.semantic_moe_view_entropy_weight,
        "semantic_moe_temperature": args.semantic_moe_temperature,
        "semantic_moe_support_strength": args.semantic_moe_support_strength,
        "semantic_moe_balance_weight": args.semantic_moe_balance_weight,
        "semantic_moe_entropy_weight": args.semantic_moe_entropy_weight,
        "semantic_moe_factor_orthogonality_weight": (
            args.semantic_moe_factor_orthogonality_weight
        ),
        "semantic_moe_factor_variance_weight": args.semantic_moe_factor_variance_weight,
        "semantic_moe_factor_std_floor": args.semantic_moe_factor_std_floor,
        "semantic_moe_delta_l2_weight": args.semantic_moe_delta_l2_weight,
        "semantic_moe_max_abs_slot_delta": args.semantic_moe_max_abs_slot_delta,
        "use_semantic_group_features": args.use_semantic_group_features,
        "semantic_group_relationship_hidden": args.semantic_group_relationship_hidden,
        "semantic_group_relationship_dropout": args.semantic_group_relationship_dropout,
        "semantic_group_relationship_l2_weight": args.semantic_group_relationship_l2_weight,
    }


def _parse_args() -> tuple[DatasetConfig, TrainConfig, dict[str, Any]]:
    dataset_defaults = DatasetConfig()
    train_defaults = TrainConfig()
    model_defaults = HGNNConfig()
    production_model_defaults = production_semantic_model_overrides()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=dataset_defaults.cache_dir)
    parser.add_argument("--max-games", type=int, default=dataset_defaults.max_games)
    parser.add_argument(
        "--encoder-sidecar-path",
        type=Path,
        default=dataset_defaults.encoder_sidecar_path,
    )
    parser.add_argument("--model-path", type=Path, default=train_defaults.model_path)
    parser.add_argument(
        "--metrics-path", type=Path, default=train_defaults.metrics_path
    )
    parser.add_argument(
        "--audit-prediction-cache-path",
        type=Path,
        default=train_defaults.audit_prediction_cache_path,
        help=(
            "Optional .npy path for train+val+test focus-side probabilities. "
            "When set, context/group audits can reuse the final evaluation pass."
        ),
    )
    parser.add_argument(
        "--warm-start-model-path",
        type=Path,
        default=train_defaults.warm_start_model_path,
        help="Optional HGNN checkpoint to load before training/fine-tuning.",
    )
    parser.add_argument(
        "--freeze-warm-start-loaded-parameters",
        action="store_true",
        default=train_defaults.freeze_warm_start_loaded_parameters,
        help=(
            "After warm-starting, freeze parameters loaded from the checkpoint "
            "and train only parameters missing from that checkpoint."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=train_defaults.batch_size)
    parser.add_argument(
        "--train-batch-cap",
        type=int,
        default=train_defaults.train_batch_cap,
        help=(
            "Effective training batch safety cap for the team-swapped HGNN loop. "
            "Set 0 to disable for explicit throughput/allocator sweeps."
        ),
    )
    parser.add_argument(
        "--train-epoch-max-games",
        type=int,
        default=train_defaults.train_epoch_max_games,
        help=(
            "Optional maximum train games sampled per epoch. Set 0 or omit for "
            "full train epochs; intended for deterministic ablation screening."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=train_defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=train_defaults.patience)
    parser.add_argument(
        "--learning-rate", type=float, default=train_defaults.learning_rate
    )
    parser.add_argument(
        "--weight-decay", type=float, default=train_defaults.weight_decay
    )
    parser.add_argument("--device", default=train_defaults.device)
    parser.add_argument(
        "--raw-tensor-cache-device",
        choices=("model", "cpu"),
        default=train_defaults.raw_tensor_cache_device,
        help=(
            "Where raw split tensors are cached before minibatch indexing. "
            "'model' keeps historical behavior; 'cpu' moves only minibatches "
            "to the model device."
        ),
    )
    parser.add_argument("--seed", type=int, default=train_defaults.seed)
    parser.add_argument(
        "--max-grad-norm", type=float, default=train_defaults.max_grad_norm
    )
    parser.add_argument("--checkpoint-metric", default=train_defaults.checkpoint_metric)
    parser.add_argument(
        "--checkpoint-min-delta",
        type=float,
        default=train_defaults.checkpoint_min_delta,
    )
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        default=train_defaults.skip_final_evaluation,
        help=(
            "Write epoch history/timing metrics and exit before the final "
            "train/val/test prediction pass. Intended for throughput sweeps."
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
        "--semantic-context-calibration-loss-weight",
        type=float,
        default=train_defaults.semantic_context_calibration_loss_weight,
        help=(
            "Optional weight for matching predicted and empirical win rates inside "
            "the shared HGNN semantic context audit bins."
        ),
    )
    parser.add_argument(
        "--semantic-context-calibration-min-count",
        type=int,
        default=train_defaults.semantic_context_calibration_min_count,
        help="Minimum side rows from a context bin required inside a batch loss term.",
    )
    parser.add_argument(
        "--semantic-context-calibration-tail-weight",
        type=float,
        default=train_defaults.semantic_context_calibration_tail_weight,
        help="Multiplier for first/last audit bins in the semantic context loss.",
    )
    parser.add_argument(
        "--semantic-context-calibration-target",
        choices=("champion_raw", "context_eb", "group_eb", "group_context_eb"),
        default=train_defaults.semantic_context_calibration_target,
        help=(
            "Calibration target family. '*_eb' fits empirical-Bayes-shrunk "
            "train targets. 'group_context_eb' combines large-n build/role "
            "groups with champion/context audit bins."
        ),
    )
    parser.add_argument(
        "--semantic-context-calibration-objective",
        choices=("absolute", "residual"),
        default=train_defaults.semantic_context_calibration_objective,
        help=(
            "'absolute' matches bin predictions to train targets. 'residual' "
            "matches semantic MoE logit corrections to bounded train-only "
            "residual deltas from the warm-start model."
        ),
    )
    parser.add_argument(
        "--semantic-context-calibration-group-residual-shrink-strength",
        type=float,
        default=(
            train_defaults.semantic_context_calibration_group_residual_shrink_strength
        ),
        help="Support shrink strength for group residual-teacher deltas.",
    )
    parser.add_argument(
        "--semantic-context-calibration-group-residual-clip",
        type=float,
        default=train_defaults.semantic_context_calibration_group_residual_clip,
        help="Logit clip for group residual-teacher deltas.",
    )
    parser.add_argument(
        "--semantic-context-calibration-group-residual-scale",
        type=float,
        default=train_defaults.semantic_context_calibration_group_residual_scale,
        help="Scale applied after clipping group residual-teacher deltas.",
    )
    parser.add_argument(
        "--semantic-context-calibration-context-residual-shrink-strength",
        type=float,
        default=(
            train_defaults.semantic_context_calibration_context_residual_shrink_strength
        ),
        help="Support shrink strength for champion/context residual-teacher deltas.",
    )
    parser.add_argument(
        "--semantic-context-calibration-context-residual-clip",
        type=float,
        default=train_defaults.semantic_context_calibration_context_residual_clip,
        help="Logit clip for champion/context residual-teacher deltas.",
    )
    parser.add_argument(
        "--semantic-context-calibration-context-residual-scale",
        type=float,
        default=train_defaults.semantic_context_calibration_context_residual_scale,
        help="Scale applied after clipping champion/context residual-teacher deltas.",
    )
    parser.add_argument("--structural-antisymmetry", action="store_true")
    parser.add_argument("--structural-antisymmetry-scale", type=float, default=0.5)
    parser.add_argument(
        "--use-identity-static-sidecar",
        action=argparse.BooleanOptionalAction,
        default=bool(production_model_defaults["use_identity_static_sidecar"]),
    )
    parser.add_argument(
        "--use-identity-full-game-sidecar",
        action=argparse.BooleanOptionalAction,
        default=bool(production_model_defaults["use_identity_full_game_sidecar"]),
    )
    parser.add_argument(
        "--use-identity-temporal-sidecar",
        action=argparse.BooleanOptionalAction,
        default=bool(production_model_defaults["use_identity_temporal_sidecar"]),
    )
    parser.add_argument(
        "--use-all-identity-sidecars",
        action="store_true",
        help="Enable static, full-game, and temporal frozen encoder sidecar blocks.",
    )
    parser.add_argument(
        "--identity-encoder-sidecar-support-strength", type=float, default=30.0
    )
    parser.add_argument("--identity-encoder-sidecar-dropout", type=float, default=0.0)
    parser.add_argument(
        "--use-identity-semantic-context-head",
        action="store_true",
        help="Enable own/ally/enemy latent-context interaction over all three identity sidecars.",
    )
    parser.add_argument("--semantic-context-dim", type=int, default=96)
    parser.add_argument("--semantic-context-dropout", type=float, default=0.0)
    parser.add_argument("--semantic-context-support-strength", type=float, default=30.0)
    parser.add_argument(
        "--use-learned-semantic-moe",
        action=argparse.BooleanOptionalAction,
        default=bool(production_model_defaults["use_learned_semantic_moe"]),
        help="Enable learned semantic MoE interaction over all three identity sidecars.",
    )
    parser.add_argument(
        "--semantic-moe-num-experts",
        type=int,
        default=model_defaults.semantic_moe_num_experts,
    )
    parser.add_argument(
        "--semantic-moe-top-k",
        type=int,
        default=model_defaults.semantic_moe_top_k,
    )
    parser.add_argument(
        "--semantic-moe-factor-dim",
        type=int,
        default=model_defaults.semantic_moe_factor_dim,
    )
    parser.add_argument(
        "--semantic-moe-factor-hidden",
        type=_parse_int_tuple,
        default=model_defaults.semantic_moe_factor_hidden,
        help="Comma-separated hidden sizes for the learned semantic MoE factor MLP.",
    )
    parser.add_argument(
        "--semantic-moe-router-hidden",
        type=_parse_int_tuple,
        default=model_defaults.semantic_moe_router_hidden,
        help="Comma-separated hidden sizes for the learned semantic MoE router MLP.",
    )
    parser.add_argument(
        "--semantic-moe-expert-hidden",
        type=_parse_int_tuple,
        default=model_defaults.semantic_moe_expert_hidden,
        help="Comma-separated hidden sizes for each learned semantic MoE expert MLP.",
    )
    parser.add_argument(
        "--semantic-moe-dropout",
        type=float,
        default=model_defaults.semantic_moe_dropout,
    )
    parser.add_argument(
        "--semantic-moe-context-token-dropout",
        type=float,
        default=model_defaults.semantic_moe_context_token_dropout,
    )
    parser.add_argument(
        "--semantic-moe-architecture",
        choices=("convex_encoder_mix",),
        default=str(production_model_defaults["semantic_moe_architecture"]),
        help="Production semantic MoE architecture.",
    )
    parser.add_argument(
        "--semantic-moe-view-gate-hidden",
        type=_parse_int_tuple,
        default=model_defaults.semantic_moe_view_gate_hidden,
        help="Comma-separated hidden sizes for the encoder-view gate.",
    )
    parser.add_argument(
        "--semantic-moe-view-top-k",
        type=int,
        default=model_defaults.semantic_moe_view_top_k,
    )
    parser.add_argument(
        "--semantic-moe-view-router-noise",
        type=float,
        default=model_defaults.semantic_moe_view_router_noise,
    )
    parser.add_argument(
        "--semantic-moe-view-balance-weight",
        type=float,
        default=model_defaults.semantic_moe_view_balance_weight,
    )
    parser.add_argument(
        "--semantic-moe-view-entropy-weight",
        type=float,
        default=model_defaults.semantic_moe_view_entropy_weight,
    )
    parser.add_argument(
        "--semantic-moe-temperature",
        type=float,
        default=model_defaults.semantic_moe_temperature,
    )
    parser.add_argument(
        "--semantic-moe-support-strength",
        type=float,
        default=model_defaults.semantic_moe_support_strength,
    )
    parser.add_argument(
        "--semantic-moe-balance-weight",
        type=float,
        default=model_defaults.semantic_moe_balance_weight,
    )
    parser.add_argument(
        "--semantic-moe-entropy-weight",
        type=float,
        default=model_defaults.semantic_moe_entropy_weight,
    )
    parser.add_argument(
        "--semantic-moe-factor-orthogonality-weight",
        type=float,
        default=model_defaults.semantic_moe_factor_orthogonality_weight,
    )
    parser.add_argument(
        "--semantic-moe-factor-variance-weight",
        type=float,
        default=model_defaults.semantic_moe_factor_variance_weight,
    )
    parser.add_argument(
        "--semantic-moe-factor-std-floor",
        type=float,
        default=model_defaults.semantic_moe_factor_std_floor,
    )
    parser.add_argument(
        "--semantic-moe-delta-l2-weight",
        type=float,
        default=model_defaults.semantic_moe_delta_l2_weight,
    )
    parser.add_argument(
        "--semantic-moe-max-abs-slot-delta",
        type=float,
        default=model_defaults.semantic_moe_max_abs_slot_delta,
        help=(
            "Optional smooth tanh cap for combined semantic MoE per-slot deltas. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--use-semantic-group-features",
        action=argparse.BooleanOptionalAction,
        default=bool(production_model_defaults["use_semantic_group_features"]),
        help=(
            "Feed compact audit semantic group summaries into the learned semantic "
            "MoE. Requires --use-learned-semantic-moe."
        ),
    )
    parser.add_argument(
        "--semantic-group-relationship-hidden",
        type=_parse_int_tuple,
        default=model_defaults.semantic_group_relationship_hidden,
        help=(
            "Comma-separated hidden sizes for the identity-conditioned semantic "
            "group relationship head."
        ),
    )
    parser.add_argument(
        "--semantic-group-relationship-dropout",
        type=float,
        default=model_defaults.semantic_group_relationship_dropout,
    )
    parser.add_argument(
        "--semantic-group-relationship-l2-weight",
        type=float,
        default=model_defaults.semantic_group_relationship_l2_weight,
    )
    args = parser.parse_args()
    if args.use_semantic_group_features and not args.use_learned_semantic_moe:
        parser.error(
            "--use-semantic-group-features requires --use-learned-semantic-moe"
        )
    return (
        DatasetConfig(
            cache_dir=args.cache_dir,
            max_games=args.max_games,
            encoder_sidecar_path=args.encoder_sidecar_path,
        ),
        TrainConfig(
            model_path=args.model_path,
            metrics_path=args.metrics_path,
            audit_prediction_cache_path=args.audit_prediction_cache_path,
            warm_start_model_path=args.warm_start_model_path,
            freeze_warm_start_loaded_parameters=(
                args.freeze_warm_start_loaded_parameters
            ),
            batch_size=args.batch_size,
            train_batch_cap=args.train_batch_cap,
            train_epoch_max_games=args.train_epoch_max_games,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            raw_tensor_cache_device=args.raw_tensor_cache_device,
            seed=args.seed,
            max_grad_norm=args.max_grad_norm,
            checkpoint_metric=args.checkpoint_metric,
            checkpoint_min_delta=args.checkpoint_min_delta,
            skip_final_evaluation=args.skip_final_evaluation,
            auc_ranking_loss_weight=args.auc_ranking_loss_weight,
            auc_ranking_loss_pairs=args.auc_ranking_loss_pairs,
            semantic_context_calibration_loss_weight=(
                args.semantic_context_calibration_loss_weight
            ),
            semantic_context_calibration_min_count=(
                args.semantic_context_calibration_min_count
            ),
            semantic_context_calibration_tail_weight=(
                args.semantic_context_calibration_tail_weight
            ),
            semantic_context_calibration_target=(
                args.semantic_context_calibration_target
            ),
            semantic_context_calibration_objective=(
                args.semantic_context_calibration_objective
            ),
            semantic_context_calibration_group_residual_shrink_strength=(
                args.semantic_context_calibration_group_residual_shrink_strength
            ),
            semantic_context_calibration_group_residual_clip=(
                args.semantic_context_calibration_group_residual_clip
            ),
            semantic_context_calibration_group_residual_scale=(
                args.semantic_context_calibration_group_residual_scale
            ),
            semantic_context_calibration_context_residual_shrink_strength=(
                args.semantic_context_calibration_context_residual_shrink_strength
            ),
            semantic_context_calibration_context_residual_clip=(
                args.semantic_context_calibration_context_residual_clip
            ),
            semantic_context_calibration_context_residual_scale=(
                args.semantic_context_calibration_context_residual_scale
            ),
        ),
        _model_overrides_from_args(args),
    )


def main() -> None:
    dataset_cfg, train_cfg, model_overrides = _parse_args()
    train(dataset_cfg, train_cfg, model_overrides=model_overrides)


if __name__ == "__main__":
    main()
