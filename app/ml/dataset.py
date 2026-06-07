from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, fields

import numpy as np

from app.ml.cache_layout import (
    CACHE_FORMAT,
    CACHE_META_FILE,
    array_paths,
    sidecar_array_paths,
)
from app.ml.config import DatasetConfig
from app.ml.semantic_group_features import materialize_semantic_group_feature_cache

LEGACY_CACHE_FORMATS = frozenset(
    {
        "npy-memmap-v15",
        "npy-memmap-v17",
        "npy-memmap-v18",
        "npy-memmap-v19",
        "npy-memmap-v20",
        "npy-memmap-v21",
        "npy-memmap-v23",
        "npy-memmap-v24",
        "npy-memmap-v25",
        "npy-memmap-v26",
        "npy-memmap-v27",
        "npy-memmap-v28",
    }
)
COUNT_ARRAY_NAMES = ("p1_cnt",)
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class SplitData:
    win_rate: np.ndarray
    p1_cnt: np.ndarray
    blue_win: np.ndarray
    # Per-slot identity ids (HGNN identity embeddings); None for legacy caches.
    champion_id: np.ndarray | None = None
    build_id: np.ndarray | None = None
    # Production game-level residual features. Signed edge columns are already
    # blue-minus-red; coverage columns are unsigned.
    loadout_features: np.ndarray | None = None
    patch_features: np.ndarray | None = None
    # Optional frozen three-encoder sidecar blocks; None for caches built
    # without an encoder_sidecar_path.
    identity_static_sidecar: np.ndarray | None = None
    identity_full_game_sidecar: np.ndarray | None = None
    identity_temporal_sidecar: np.ndarray | None = None
    identity_encoder_support: np.ndarray | None = None
    # Optional compact semantic audit-group features [games, 10, G]. Loaded only
    # when the learned semantic MoE group-feature flag is enabled.
    semantic_group_features: np.ndarray | None = None
    # Raw identity/context audit axes [games, 10, C]. Loaded only when train-time
    # audit metrics need to match the markdown/group audit hard-bin lens.
    context_raw: np.ndarray | None = None


def _slice(arrays: dict[str, np.ndarray], lo: int, hi: int) -> SplitData:
    # Required fields (no default) KeyError if absent, as before; optional
    # fields default to None when their array was not loaded.
    return SplitData(
        **{
            f.name: (
                arrays[f.name][lo:hi]
                if f.name in arrays or f.default is MISSING
                else None
            )
            for f in fields(SplitData)
        }
    )


def identity_meta(cfg: DatasetConfig) -> dict:
    """Identity embedding metadata recorded by build_dataset.

    Keys: ``n_champions`` and ``n_builds`` (embedding-table sizes) and
    ``build_vocab`` (sorted build labels -> embedding index)."""
    meta = json.loads((cfg.cache_dir / CACHE_META_FILE).read_text())
    identity = dict(meta["identity"])
    if "identity_encoder_sidecar" in meta:
        identity["identity_encoder_sidecar"] = meta["identity_encoder_sidecar"]
    features = meta.get("production_features")
    if isinstance(features, dict):
        loadout_names = tuple(str(name) for name in features.get("loadout_feature_names", ()))
        patch_names = tuple(str(name) for name in features.get("patch_feature_names", ()))
        identity["loadout_feature_names"] = loadout_names
        identity["patch_feature_names"] = patch_names
        identity["loadout_feature_dim"] = len(loadout_names)
        identity["patch_feature_dim"] = len(patch_names)
    return identity


def _split_counts(meta: dict, n_games: int) -> dict[str, int]:
    raw = meta.get("splits")
    if not isinstance(raw, dict):
        raise ValueError("Cache metadata is missing split counts; rebuild the cache.")
    missing = [name for name in SPLIT_ORDER if name not in raw]
    if missing:
        raise ValueError(
            "Cache metadata is missing split counts for "
            + ", ".join(missing)
            + "; rebuild the cache."
        )
    counts = {name: int(raw[name]) for name in SPLIT_ORDER}
    if sum(counts.values()) != int(n_games):
        raise ValueError("Cache split counts do not match n_games; rebuild the cache.")
    return counts


def _split_order(meta: dict) -> tuple[str, str, str]:
    raw = meta.get("split_order", SPLIT_ORDER)
    order = tuple(str(name) for name in raw)
    if sorted(order) != sorted(SPLIT_ORDER):
        raise ValueError(
            "Cache split_order is invalid; expected train, val, and test. "
            "Rebuild the cache."
        )
    return (order[0], order[1], order[2])


def _range_pair(raw: object) -> tuple[int, int]:
    if isinstance(raw, dict):
        return int(raw["start"]), int(raw["stop"])
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    raise ValueError("Cache split range is invalid; rebuild the cache.")


def _split_ranges(meta: dict, n_games: int) -> dict[str, tuple[int, int]]:
    counts = _split_counts(meta, n_games)
    raw_ranges = meta.get("split_ranges")
    if raw_ranges is None:
        offset = 0
        ranges: dict[str, tuple[int, int]] = {}
        for split_name in _split_order(meta):
            count = counts[split_name]
            ranges[split_name] = (offset, offset + count)
            offset += count
        return {name: ranges[name] for name in SPLIT_ORDER}
    if not isinstance(raw_ranges, dict):
        raise ValueError("Cache split_ranges metadata is invalid; rebuild the cache.")

    ranges = {}
    covered: list[tuple[int, int, str]] = []
    for split_name in SPLIT_ORDER:
        if split_name not in raw_ranges:
            raise ValueError(
                f"Cache split_ranges is missing {split_name}; rebuild the cache."
            )
        lo, hi = _range_pair(raw_ranges[split_name])
        expected = counts[split_name]
        if lo < 0 or hi < lo or hi > n_games or hi - lo != expected:
            raise ValueError(
                f"Cache split range for {split_name} is inconsistent with "
                "split counts; rebuild the cache."
            )
        ranges[split_name] = (lo, hi)
        covered.append((lo, hi, split_name))

    offset = 0
    for lo, hi, _split_name in sorted(covered):
        if lo != offset:
            raise ValueError("Cache split ranges do not cover n_games; rebuild the cache.")
        offset = hi
    if offset != n_games:
        raise ValueError("Cache split ranges do not cover n_games; rebuild the cache.")
    return ranges


def _validate_blue_win(values: np.ndarray) -> None:
    unique = np.unique(values)
    if not np.isin(unique, [0.0, 1.0]).all():
        raise ValueError("Cache blue_win labels must be binary; rebuild the cache.")


def load_splits(
    cfg: DatasetConfig,
    *,
    require_counts: bool = False,
    load_semantic_group_features: bool = False,
    semantic_group_feature_dim: int | None = None,
    load_context_raw: bool = False,
) -> dict[str, SplitData]:
    meta = json.loads((cfg.cache_dir / CACHE_META_FILE).read_text())
    cache_format = meta.get("format")
    if cache_format != CACHE_FORMAT and cache_format not in LEGACY_CACHE_FORMATS:
        raise ValueError(
            f"Dataset cache format is stale (found {cache_format}, "
            f"expected {CACHE_FORMAT}); rebuild the cache."
        )
    if require_counts and cache_format == "npy-memmap-v15":
        raise ValueError(
            f"Dataset cache format {cache_format} does not include support counts; "
            "rebuild the cache."
        )

    n = int(meta["n_games"])
    split_ranges = _split_ranges(meta, n)

    paths = array_paths(cfg.cache_dir)
    arrays = {
        "win_rate": np.load(paths["win_rate"], mmap_mode="r")[:n],
        "blue_win": np.load(paths["blue_win"], mmap_mode="r")[:n].astype(np.float64),
    }
    for name in COUNT_ARRAY_NAMES:
        path = paths[name]
        if path.exists():
            arrays[name] = np.load(path, mmap_mode="r")[:n]
        elif require_counts:
            raise ValueError(
                f"Dataset cache is missing required support-count array {path.name}; "
                "rebuild the cache."
            )
        else:
            arrays[name] = np.zeros(arrays["win_rate"].shape, dtype=np.float32)
    for name in ("champion_id", "build_id"):
        if paths[name].exists():
            arrays[name] = np.load(paths[name], mmap_mode="r")[:n]
    feature_meta = meta.get("production_features")
    feature_dims = {}
    if isinstance(feature_meta, dict):
        feature_dims = {
            "loadout_features": len(feature_meta.get("loadout_feature_names", ())),
            "patch_features": len(feature_meta.get("patch_feature_names", ())),
        }
    for name in ("loadout_features", "patch_features"):
        path = paths[name]
        if path.exists():
            value = np.load(path, mmap_mode="r")[:n]
            expected_dim = feature_dims.get(name, 0)
            if expected_dim > 0 and (value.ndim != 2 or value.shape[1] != expected_dim):
                raise ValueError(
                    f"Dataset cache {name} must have shape [games, {expected_dim}]; "
                    "rebuild the cache."
                )
            arrays[name] = value
        elif feature_dims.get(name, 0) > 0:
            raise ValueError(
                f"Dataset cache metadata declares {name}, but {path.name} is missing; "
                "rebuild the cache."
            )
    for name, path in sidecar_array_paths(cfg.cache_dir).items():
        if path.exists():
            arrays[name] = np.load(path, mmap_mode="r")[:n]
    if load_semantic_group_features:
        identity = meta.get("identity")
        if not isinstance(identity, dict) or "build_vocab" not in identity:
            raise ValueError("Cache metadata is missing identity.build_vocab; rebuild the cache.")
        try:
            arrays["semantic_group_features"] = materialize_semantic_group_feature_cache(
                cache_dir=cfg.cache_dir,
                n_games=n,
                build_vocab=tuple(identity["build_vocab"]),
            )
        except ValueError as exc:
            if semantic_group_feature_dim is None:
                raise
            feature_path = cfg.cache_dir / "semantic_group_features.npy"
            if not feature_path.exists():
                raise
            features = np.load(feature_path, mmap_mode="r")[:n]
            expected_dim = int(semantic_group_feature_dim)
            if (
                features.ndim != 3
                or features.shape[0] != n
                or features.shape[1] != 10
                or features.shape[2] != expected_dim
            ):
                raise ValueError(
                    "semantic_group_features.npy is incompatible with the saved "
                    f"model; expected [games, 10, {expected_dim}]."
                ) from exc
            arrays["semantic_group_features"] = features
    if load_context_raw:
        context_path = cfg.cache_dir / "identity_context_raw.npy"
        if not context_path.exists():
            raise ValueError(
                "Dataset cache is missing identity_context_raw.npy; rebuild the cache."
            )
        context_raw = np.load(context_path, mmap_mode="r")[:n]
        if context_raw.ndim != 3 or context_raw.shape[1] != 10:
            raise ValueError(
                "identity_context_raw.npy must have shape [games, 10, context_dim]; "
                "rebuild the cache."
            )
        arrays["context_raw"] = context_raw
    _validate_blue_win(arrays["blue_win"])
    return {name: _slice(arrays, *split_ranges[name]) for name in SPLIT_ORDER}
