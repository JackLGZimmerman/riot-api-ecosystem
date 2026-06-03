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
    }
)
COUNT_ARRAY_NAMES = ("p1_cnt", "m1v1_cnt", "s2vx_cnt")
SPLIT_ORDER = ("train", "val", "test")
# Effective sample size per interaction edge after nested EB pooling (cache
# v18+); legacy caches fall back to the raw support counts (no backoff info).
EFF_N_FALLBACK = {"m1v1_eff_n": "m1v1_cnt", "s2vx_eff_n": "s2vx_cnt"}


@dataclass(frozen=True)
class SplitData:
    win_rate: np.ndarray
    matchup_1v1: np.ndarray
    synergy_2vx: np.ndarray
    p1_cnt: np.ndarray
    m1v1_cnt: np.ndarray
    s2vx_cnt: np.ndarray
    blue_win: np.ndarray
    # Effective sample size per interaction edge after nested pooling; loaded by
    # load_splits, defaulting to the raw counts for legacy caches.
    m1v1_eff_n: np.ndarray | None = None
    s2vx_eff_n: np.ndarray | None = None
    # Per-slot identity ids (HGNN identity embeddings); None for legacy caches.
    champion_id: np.ndarray | None = None
    build_id: np.ndarray | None = None
    # Optional frozen three-encoder sidecar blocks; None for caches built
    # without an encoder_sidecar_path.
    identity_static_sidecar: np.ndarray | None = None
    identity_full_game_sidecar: np.ndarray | None = None
    identity_temporal_sidecar: np.ndarray | None = None
    identity_encoder_support: np.ndarray | None = None


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
        "matchup_1v1": np.load(paths["matchup_1v1"], mmap_mode="r")[:n],
        "synergy_2vx": np.load(paths["synergy_2vx"], mmap_mode="r")[:n],
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
            if name == "p1_cnt":
                shape = arrays["win_rate"].shape
            elif name == "m1v1_cnt":
                shape = arrays["matchup_1v1"].shape
            else:
                shape = arrays["synergy_2vx"].shape
            arrays[name] = np.zeros(shape, dtype=np.float32)
    for name, fallback in EFF_N_FALLBACK.items():
        path = paths[name]
        if path.exists():
            arrays[name] = np.load(path, mmap_mode="r")[:n]
        else:
            # Legacy cache without nested pooling: effective N is the raw count.
            arrays[name] = arrays[fallback]
    for name in ("champion_id", "build_id"):
        if paths[name].exists():
            arrays[name] = np.load(paths[name], mmap_mode="r")[:n]
    for name, path in sidecar_array_paths(cfg.cache_dir).items():
        if path.exists():
            arrays[name] = np.load(path, mmap_mode="r")[:n]
    _validate_blue_win(arrays["blue_win"])
    return {name: _slice(arrays, *split_ranges[name]) for name in SPLIT_ORDER}
