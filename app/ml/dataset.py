from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from app.ml.cache_layout import CACHE_FORMAT, CACHE_META_FILE, array_paths
from app.ml.config import DatasetConfig


@dataclass(frozen=True)
class SplitData:
    win_rate: np.ndarray
    blue_win: np.ndarray


def load_splits(cfg: DatasetConfig) -> dict[str, SplitData]:
    meta = json.loads((cfg.cache_dir / CACHE_META_FILE).read_text())
    if meta.get("format") != CACHE_FORMAT:
        raise ValueError("Dataset cache format is stale; rebuild the cache.")

    n = int(meta["n_games"])
    split_counts = meta["splits"]
    if not isinstance(split_counts, dict):
        raise ValueError("Cache metadata is missing split counts; rebuild the cache.")

    n_train = int(split_counts["train"])
    n_val = int(split_counts["val"])
    n_test = int(split_counts["test"])
    if n_train + n_val + n_test != n:
        raise ValueError("Cache split counts do not match n_games; rebuild the cache.")

    paths = array_paths(cfg.cache_dir)
    win_rate = np.load(paths["win_rate"], mmap_mode="r")[:n]
    blue_win = np.load(paths["blue_win"], mmap_mode="r")[:n].astype(np.float64)
    return {
        "train": SplitData(win_rate[:n_train], blue_win[:n_train]),
        "val": SplitData(
            win_rate[n_train : n_train + n_val],
            blue_win[n_train : n_train + n_val],
        ),
        "test": SplitData(win_rate[n_train + n_val : n], blue_win[n_train + n_val : n]),
    }
