from __future__ import annotations

import json

import numpy as np
import pytest

from app.ml.cache_layout import (
    ARRAY_SHAPES,
    CACHE_FORMAT,
    CACHE_META_FILE,
    DISK_DTYPES,
    array_paths,
)
from app.ml.config import DatasetConfig
from app.ml.dataset import load_splits


def _write_minimal_cache(cache_dir, labels: list[int], meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_games = len(labels)
    for name, path in array_paths(cache_dir).items():
        shape = (n_games, *ARRAY_SHAPES[name])
        values = np.zeros(shape, dtype=DISK_DTYPES[name])
        if name == "win_rate":
            values.fill(0.5)
        elif name == "blue_win":
            values = np.asarray(labels, dtype=DISK_DTYPES[name])
        np.save(path, values)
    (cache_dir / CACHE_META_FILE).write_text(json.dumps(meta))


def test_load_splits_uses_explicit_metadata_ranges(tmp_path) -> None:
    _write_minimal_cache(
        tmp_path,
        [0, 1, 0, 1],
        {
            "format": CACHE_FORMAT,
            "n_games": 4,
            "splits": {"test": 1, "train": 3},
            "split_order": ["test", "train"],
            "split_ranges": {
                "test": {"start": 0, "stop": 1},
                "train": {"start": 1, "stop": 4},
            },
        },
    )

    splits = load_splits(DatasetConfig(cache_dir=tmp_path), require_counts=True)

    assert splits["train"].blue_win.tolist() == [1.0, 0.0, 1.0]
    assert splits["test"].blue_win.tolist() == [0.0]


def test_load_splits_rejects_v32_cache_without_explicit_ranges(tmp_path) -> None:
    _write_minimal_cache(
        tmp_path,
        [0, 1, 0, 1],
        {
            "format": CACHE_FORMAT,
            "n_games": 4,
            "splits": {"test": 1, "train": 3},
            "split_order": ["test", "train"],
        },
    )

    with pytest.raises(ValueError, match="split_ranges"):
        load_splits(DatasetConfig(cache_dir=tmp_path), require_counts=True)


def test_load_splits_rejects_stale_val_cache(tmp_path) -> None:
    _write_minimal_cache(
        tmp_path,
        [0, 1, 0, 1],
        {
            "format": "npy-memmap-v31",
            "n_games": 4,
            "splits": {"train": 2, "val": 1, "test": 1},
            "split_order": ["train", "val", "test"],
        },
    )

    with pytest.raises(ValueError, match="rebuild the cache"):
        load_splits(DatasetConfig(cache_dir=tmp_path), require_counts=True)
