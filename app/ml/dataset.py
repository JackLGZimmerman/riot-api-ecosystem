from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from app.ml.cache_layout import CACHE_FORMAT, CACHE_META_FILE, array_paths
from app.ml.config import DatasetConfig

LEGACY_CACHE_FORMATS = frozenset(
    {"npy-memmap-v15", "npy-memmap-v17", "npy-memmap-v18", "npy-memmap-v19"}
)
COUNT_ARRAY_NAMES = ("p1_cnt", "m1v1_cnt", "s2vx_cnt")
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
    identity_semantic: np.ndarray | None = None
    identity_profile: np.ndarray | None = None
    m1v1_detail: np.ndarray | None = None
    s2vx_detail: np.ndarray | None = None


def _slice(arrays: dict[str, np.ndarray], lo: int, hi: int) -> SplitData:
    return SplitData(
        win_rate=arrays["win_rate"][lo:hi],
        matchup_1v1=arrays["matchup_1v1"][lo:hi],
        synergy_2vx=arrays["synergy_2vx"][lo:hi],
        p1_cnt=arrays["p1_cnt"][lo:hi],
        m1v1_cnt=arrays["m1v1_cnt"][lo:hi],
        s2vx_cnt=arrays["s2vx_cnt"][lo:hi],
        m1v1_eff_n=arrays["m1v1_eff_n"][lo:hi] if "m1v1_eff_n" in arrays else None,
        s2vx_eff_n=arrays["s2vx_eff_n"][lo:hi] if "s2vx_eff_n" in arrays else None,
        blue_win=arrays["blue_win"][lo:hi],
        champion_id=arrays["champion_id"][lo:hi] if "champion_id" in arrays else None,
        build_id=arrays["build_id"][lo:hi] if "build_id" in arrays else None,
        identity_semantic=arrays["identity_semantic"][lo:hi] if "identity_semantic" in arrays else None,
        identity_profile=arrays["identity_profile"][lo:hi] if "identity_profile" in arrays else None,
        m1v1_detail=arrays["m1v1_detail"][lo:hi] if "m1v1_detail" in arrays else None,
        s2vx_detail=arrays["s2vx_detail"][lo:hi] if "s2vx_detail" in arrays else None,
    )


def identity_meta(cfg: DatasetConfig) -> dict:
    """Identity embedding metadata recorded by build_dataset.

    Keys: ``n_champions`` and ``n_builds`` (embedding-table sizes) and
    ``build_vocab`` (sorted build labels -> embedding index)."""
    meta = json.loads((cfg.cache_dir / CACHE_META_FILE).read_text())
    return meta["identity"]


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
    split_counts = meta["splits"]
    n_train = int(split_counts["train"])
    n_val = int(split_counts["val"])
    n_test = int(split_counts["test"])
    if n_train + n_val + n_test != n:
        raise ValueError("Cache split counts do not match n_games; rebuild the cache.")

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
    for name in ("identity_semantic", "identity_profile", "m1v1_detail", "s2vx_detail"):
        if paths[name].exists():
            arrays[name] = np.load(paths[name], mmap_mode="r")[:n]
    return {
        "train": _slice(arrays, 0, n_train),
        "val": _slice(arrays, n_train, n_train + n_val),
        "test": _slice(arrays, n_train + n_val, n),
    }
