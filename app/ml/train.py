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
from app.ml.config import (
    DEFAULT_PRODUCTION_METRICS_PATH,
    DEFAULT_PRODUCTION_MODEL_PATH,
    DatasetConfig,
    TrainConfig,
)
from app.ml.build_catalog import build_catalog
from app.ml.dataset import SPLIT_ORDER, SplitData, identity_meta, load_splits
from app.ml.pregame import apply_modal_build_split, build_hypothesis_tables
from app.ml.priors import load_priors
from app.ml.encoder_sidecar import EncoderSidecarLookup, SidecarGatherTables
from app.core.utils.common import resolve_device_str as resolve_device
from app.ml.hgnn_model import (
    HGNNConfig,
    HGNNEnsemble,
    HGNNWinModel,
    build_hgnn_inputs,
    expected_encoder_sidecar_dims,
    hgnn_config_payload,
    model_requires_semantic_group_features,
    model_uses_encoder_sidecar,
    save_hgnn_model,
    swap_hgnn_inputs,
)

setup_logging_config()
logger = logging.getLogger(__name__)

EPS = 1e-12
PRODUCTION_SEMANTIC_MODEL_OVERRIDES: dict[str, Any] = {
    "use_learned_semantic_moe": True,
    "use_semantic_group_features": True,
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
    return model_uses_encoder_sidecar(config)


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
                "or rebuild the compact cache with the artifact recorded in its meta."
            )
        path = Path(recorded)
    lookup = EncoderSidecarLookup.load(path)
    expected_dims = expected_encoder_sidecar_dims(config)
    actual_dims = lookup.dims.as_dict()
    mismatches = [
        f"{name}: checkpoint={expected} sidecar={actual_dims.get(name)}"
        for name, expected in expected_dims.items()
        if int(expected) != int(actual_dims.get(name, -1))
    ]
    if mismatches:
        raise ValueError(
            "Encoder sidecar artifact dimensions do not match the HGNN config: "
            + ", ".join(mismatches)
        )
    tables = lookup.gather_tables(
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
) -> SplitData:
    """Null out optional sidecar arrays when the configured model ignores them."""
    sidecar_enabled = model_uses_encoder_sidecar(config)
    semantic_group_features_enabled = model_requires_semantic_group_features(config)
    requires_all_sidecars = sidecar_enabled
    if requires_all_sidecars:
        sidecar_names = (
            "identity_static_sidecar",
            "identity_full_game_sidecar",
            "identity_temporal_sidecar",
            "identity_encoder_support",
        )
        present = [name for name in sidecar_names if getattr(split, name) is not None]
        # All absent => latents are gathered per batch from the frozen artifact.
        # Partial presence means a corrupt or legacy cache: fail early.
        if present and len(present) < len(sidecar_names):
            missing = [name for name in sidecar_names if getattr(split, name) is None]
            raise ValueError(
                "semantic MoE head requires cache arrays: "
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
                        f"semantic MoE head requires non-empty {name} [games, 10, dim]"
                    )
            support = split.identity_encoder_support
            if support.ndim != 2 or support.shape[1] != 10:
                raise ValueError(
                    "semantic MoE head requires identity_encoder_support [games, 10]"
                )
    drop: dict[str, bool] = {
        "identity_static_sidecar": not requires_all_sidecars,
        "identity_full_game_sidecar": not requires_all_sidecars,
        "identity_temporal_sidecar": not requires_all_sidecars,
        "identity_encoder_support": not sidecar_enabled,
        "semantic_group_features": not semantic_group_features_enabled,
        "patch_features": int(config.patch_feature_dim) <= 0,
    }
    if int(config.patch_feature_dim) > 0:
        value = split.patch_features
        if value is None or value.ndim != 2 or value.shape[1] != int(config.patch_feature_dim):
            raise ValueError(
                f"HGNN config enables patch_features, but the cache is missing "
                f"patch_features [games, {int(config.patch_feature_dim)}]; rebuild the dataset cache."
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
    for split_name, split in splits.items():
        labels = np.asarray(split.blue_win)
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


def _nll(scores: np.ndarray, targets: np.ndarray) -> float:
    if scores.size == 0:
        return float("nan")
    p = np.clip(scores, EPS, 1.0 - EPS)
    return float(-np.mean(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    z = np.clip(logits.astype(np.float64, copy=False), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64, copy=False)


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
        # parallel candidate runs. CUDA caches still materialise on the GPU.
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
        return {"n": 0, "accuracy": float("nan"), "nll": float("nan")}
    return {
        "n": int(targets.size),
        "accuracy": float(np.mean((scores >= 0.5) == (targets > 0.5))),
        "nll": _nll(scores, targets),
    }


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


def _same_output_path(left: Path, right: Path) -> bool:
    left_resolved = Path(left).expanduser().resolve(strict=False)
    right_resolved = Path(right).expanduser().resolve(strict=False)
    return left_resolved == right_resolved


def _validate_train_output_paths(train_cfg: TrainConfig) -> None:
    if train_cfg.allow_production_artifact_overwrite:
        return
    blocked: list[str] = []
    if _same_output_path(train_cfg.model_path, DEFAULT_PRODUCTION_MODEL_PATH):
        blocked.append(f"model_path={_project_relative(train_cfg.model_path)}")
    if _same_output_path(train_cfg.metrics_path, DEFAULT_PRODUCTION_METRICS_PATH):
        blocked.append(f"metrics_path={_project_relative(train_cfg.metrics_path)}")
    if blocked:
        raise ValueError(
            "Training refuses to overwrite production artifacts by default "
            f"({', '.join(blocked)}). Use experiment output paths or pass "
            "--allow-production-artifact-overwrite for an explicit promotion run."
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
    # Compact caches omit per-game sidecar arrays; gather from the frozen
    # artifact using the batch's identity ids. Legacy caches that still carry
    # per-game arrays are used directly.
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
        patch_features=raw.patch_features,
        device=device,
        **sidecar,
    )


def _predict_hgnn_logits(
    model: HGNNWinModel | HGNNEnsemble,
    split: RawTensorSplit,
    *,
    batch_size: int,
    strength: float,
    device: str,
    gatherer: _SidecarGatherer | None = None,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    logits: list[np.ndarray] = []
    try:
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
                logits.append(
                    outputs["final_logit"]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float64, copy=False)
                )
    finally:
        if was_training:
            model.train()
    if not logits:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(logits).astype(np.float64, copy=False)


def train(
    dataset_cfg: DatasetConfig | None = None,
    train_cfg: TrainConfig | None = None,
    *,
    model_overrides: dict[str, Any] | None = None,
) -> Path:
    dataset_cfg = dataset_cfg or DatasetConfig()
    train_cfg = train_cfg or TrainConfig()
    _validate_train_config(train_cfg)
    _validate_train_output_paths(train_cfg)
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

    load_semantic_group_features = model_requires_semantic_group_features(model_config)
    loaded_splits = {
        name: _limit_split(split, dataset_cfg.max_games)
        for name, split in load_splits(
            dataset_cfg,
            require_counts=True,
            load_semantic_group_features=load_semantic_group_features,
        ).items()
    }
    if train_cfg.pregame_modal_builds:
        # No-build comparator for the leakage-free pregame path: every
        # build-dependent input becomes the modal build per (champion, role)
        # from the train-only catalog, on both splits. Evaluate the result
        # with `marginal_eval --mode modal`, never against observed builds.
        priors = load_priors()
        build_vocab = tuple(model_config.build_vocab)
        catalog = build_catalog(priors.p1, build_vocab)
        tables = build_hypothesis_tables(
            dataset_cfg,
            priors,
            n_champions=int(model_config.n_champions),
            build_vocab=build_vocab,
        )
        loaded_splits = {
            name: apply_modal_build_split(
                split,
                catalog,
                tables,
                build_vocab=build_vocab,
                needs_semantic=load_semantic_group_features,
            )
            for name, split in loaded_splits.items()
        }
        logger.info("Applied pregame modal-build transform (catalog %s)", catalog.version)
    # Compact caches omit per-game sidecar arrays; build the on-device gather
    # table from the frozen artifact when the model consumes identity latents.
    gatherer = None
    if (
        _model_uses_sidecar(model_config)
        and loaded_splits["train"].identity_static_sidecar is None
    ):
        gatherer = _build_sidecar_gatherer(
            dataset_cfg, meta, model_config, device=device
        )
    splits = {
        name: _drop_unused_model_arrays(split, model_config)
        for name, split in loaded_splits.items()
    }
    _validate_split_targets(splits)
    if splits["train"].blue_win.size == 0:
        raise ValueError("Training split is empty; rebuild the cache with train games.")
    raw_cache_device = "cpu" if train_cfg.raw_tensor_cache_device == "cpu" else device
    tensor_splits = {
        name: _cache_raw_tensor_split(name, splits[name], device=raw_cache_device)
        for name in SPLIT_ORDER
    }

    model = HGNNWinModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(train_cfg.seed)
    semantic_moe_enabled = bool(model_config.use_learned_semantic_moe)
    best_state = copy.deepcopy(model.state_dict())
    best_test_nll = math.inf
    best_checkpoint_test_nll = math.inf
    best_checkpoint_accuracy = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    logger.info(
        "HGNN training device=%s raw_tensor_cache_device=%s batch_size=%s requested_batch_size=%s train_batch_cap=%s train_epoch_max_games=%s max_epochs=%s strength=%s checkpoint=test_accuracy",
        device,
        raw_cache_device,
        train_batch_size,
        train_cfg.batch_size,
        train_batch_cap,
        train_epoch_max_games,
        train_cfg.max_epochs,
        strength,
    )
    if device == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    def synchronize_training_device() -> None:
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    for epoch in range(1, train_cfg.max_epochs + 1):
        synchronize_training_device()
        epoch_started = time.perf_counter()
        train_started = epoch_started
        model.train()
        train_loss_sum = 0.0
        train_semantic_moe_regularization_loss_sum = 0.0
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
            direct_semantic_moe_regularization_loss = (
                model.semantic_moe_regularization_loss(direct_outputs)
            )
            (0.5 * (direct_loss + direct_semantic_moe_regularization_loss)).backward()
            swapped_outputs = model(**swap_hgnn_inputs(inputs))
            swapped_loss = loss_fn(swapped_outputs["final_logit"], 1.0 - labels)
            swapped_semantic_moe_regularization_loss = (
                model.semantic_moe_regularization_loss(swapped_outputs)
            )
            (0.5 * (swapped_loss + swapped_semantic_moe_regularization_loss)).backward()
            loss = 0.5 * (direct_loss.detach() + swapped_loss.detach())
            semantic_moe_regularization_loss = 0.5 * (
                direct_semantic_moe_regularization_loss.detach()
                + swapped_semantic_moe_regularization_loss.detach()
            )
            if train_cfg.max_grad_norm is not None and train_cfg.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            train_loss_sum += float(loss.cpu().item()) * labels.numel() * 2
            if semantic_moe_enabled:
                train_semantic_moe_regularization_loss_sum += (
                    float(semantic_moe_regularization_loss.cpu().item())
                    * labels.numel()
                    * 2
                )
            train_seen += int(labels.numel() * 2)

        synchronize_training_device()
        train_seconds = time.perf_counter() - train_started
        test_started = time.perf_counter()
        test_logits = _predict_hgnn_logits(
            model,
            tensor_splits["test"],
            batch_size=train_cfg.batch_size,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
        synchronize_training_device()
        test_predictions = _sigmoid_np(test_logits)
        train_nll = train_loss_sum / max(train_seen, 1)
        train_semantic_moe_regularization_loss = (
            train_semantic_moe_regularization_loss_sum / max(train_seen, 1)
        )
        test_metrics = _evaluate_predictions(test_predictions, splits["test"])
        test_seconds = time.perf_counter() - test_started
        epoch_seconds = time.perf_counter() - epoch_started
        train_rows_per_second = (train_seen / 2.0) / max(train_seconds, EPS)
        train_epoch_games_seen = int(train_seen // 2)
        train_augmented_samples_per_second = train_seen / max(train_seconds, EPS)
        test_nll = float(test_metrics["nll"])
        checkpoint_accuracy = float(test_metrics["accuracy"])
        history_row: dict[str, float | int] = {
            "epoch": epoch,
            "train_nll": train_nll,
            "test_nll": test_nll,
            "test_accuracy": float(test_metrics["accuracy"]),
            "checkpoint_accuracy": checkpoint_accuracy,
        }
        if semantic_moe_enabled:
            history_row["train_semantic_moe_regularization_loss"] = (
                train_semantic_moe_regularization_loss
            )
        history.append(history_row)
        if semantic_moe_enabled:
            logger.info(
                "epoch=%s train_nll=%.5f semantic_moe_reg=%.5f test_nll=%.5f test_acc=%.4f",
                epoch,
                train_nll,
                train_semantic_moe_regularization_loss,
                test_nll,
                test_metrics["accuracy"],
            )
        else:
            logger.info(
                "epoch=%s train_nll=%.5f test_nll=%.5f test_acc=%.4f",
                epoch,
                train_nll,
                test_nll,
                test_metrics["accuracy"],
            )
        logger.info(
            "epoch=%s timing train_seconds=%.2f test_seconds=%.2f epoch_seconds=%.2f train_games=%s train_rows_per_s=%.1f train_augmented_samples_per_s=%.1f batch_size=%s",
            epoch,
            train_seconds,
            test_seconds,
            epoch_seconds,
            train_epoch_games_seen,
            train_rows_per_second,
            train_augmented_samples_per_second,
            train_batch_size,
        )
        if test_nll < best_test_nll:
            best_test_nll = test_nll
        if checkpoint_accuracy > best_checkpoint_accuracy:
            best_checkpoint_accuracy = checkpoint_accuracy
            best_checkpoint_test_nll = test_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= train_cfg.patience:
                break

    model.load_state_dict(best_state)
    save_hgnn_model(train_cfg.model_path, model, confidence_strength=strength)

    prediction_split_names = SPLIT_ORDER
    prediction_logits = {
        split_name: _predict_hgnn_logits(
            model,
            tensor_split,
            batch_size=train_cfg.batch_size,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
        for split_name, tensor_split in tensor_splits.items()
    }
    predictions = {
        split_name: _sigmoid_np(logits)
        for split_name, logits in prediction_logits.items()
    }
    split_metrics = {
        split_name: _evaluate_predictions(predictions[split_name], splits[split_name])
        for split_name in prediction_split_names
    }
    metrics = {
        "model_type": "hgnn",
        "dataset_config": asdict(dataset_cfg),
        "train_config": asdict(train_cfg),
        "model_config": hgnn_config_payload(model_config),
        "model_path": train_cfg.model_path,
        "metrics_path": train_cfg.metrics_path,
        "device": device,
        "best_epoch": best_epoch,
        "selection_split": "test",
        "best_test_nll": best_test_nll,
        "best_checkpoint_test_nll": best_checkpoint_test_nll,
        "best_checkpoint_accuracy": best_checkpoint_accuracy,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "evaluated_splits": list(prediction_split_names),
        "train": split_metrics["train"],
        "test": split_metrics["test"],
    }
    _write_metrics(train_cfg.metrics_path, metrics)

    logger.info("Saved HGNN model: %s", _project_relative(train_cfg.model_path))
    logger.info("Saved metrics: %s", _project_relative(train_cfg.metrics_path))
    for split_name in prediction_split_names:
        m = metrics[split_name]
        if isinstance(m, dict):
            logger.info(
                "%s n=%s acc=%.4f nll=%.4f",
                split_name,
                m["n"],
                m["accuracy"],
                m["nll"],
            )

    return train_cfg.model_path


def _model_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """HGNNConfig overrides exposed by the training CLI."""
    return {
        "use_learned_semantic_moe": args.use_learned_semantic_moe,
        "semantic_moe_num_experts": args.semantic_moe_num_experts,
        "semantic_moe_top_k": args.semantic_moe_top_k,
        "semantic_moe_factor_dim": args.semantic_moe_factor_dim,
        "semantic_moe_factor_hidden": args.semantic_moe_factor_hidden,
        "semantic_moe_router_hidden": args.semantic_moe_router_hidden,
        "semantic_moe_expert_hidden": args.semantic_moe_expert_hidden,
        "semantic_moe_dropout": args.semantic_moe_dropout,
        "semantic_moe_context_token_dropout": args.semantic_moe_context_token_dropout,
        "semantic_moe_view_gate_hidden": args.semantic_moe_view_gate_hidden,
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
            "full train epochs; intended for deterministic candidate screening."
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
            "'cpu' moves only minibatches to the model device; 'model' keeps "
            "the full raw cache on the model device for explicit sweeps."
        ),
    )
    parser.add_argument(
        "--allow-production-artifact-overwrite",
        action="store_true",
        default=train_defaults.allow_production_artifact_overwrite,
        help=(
            "Allow training to overwrite the promoted production model or "
            "metrics paths. Intended only for explicit promotion runs."
        ),
    )
    parser.add_argument("--seed", type=int, default=train_defaults.seed)
    parser.add_argument(
        "--pregame-modal-builds",
        action=argparse.BooleanOptionalAction,
        default=train_defaults.pregame_modal_builds,
        help=(
            "Replace every build-dependent input with the modal build per "
            "(champion, role) from the train-only catalog (no-build "
            "comparator; evaluate with marginal_eval --mode modal)."
        ),
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=train_defaults.max_grad_norm
    )
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
        "--semantic-moe-view-gate-hidden",
        type=_parse_int_tuple,
        default=model_defaults.semantic_moe_view_gate_hidden,
        help="Comma-separated hidden sizes for the encoder-view gate.",
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
            pregame_modal_builds=args.pregame_modal_builds,
            allow_production_artifact_overwrite=(
                args.allow_production_artifact_overwrite
            ),
        ),
        _model_overrides_from_args(args),
    )


def main() -> None:
    dataset_cfg, train_cfg, model_overrides = _parse_args()
    train(dataset_cfg, train_cfg, model_overrides=model_overrides)


if __name__ == "__main__":
    main()
