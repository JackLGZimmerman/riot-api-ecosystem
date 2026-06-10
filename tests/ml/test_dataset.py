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
            "splits": {"test": 1, "train": 2, "val": 1},
            "split_order": ["test", "train", "val"],
            "split_ranges": {
                "test": {"start": 0, "stop": 1},
                "train": {"start": 1, "stop": 3},
                "val": {"start": 3, "stop": 4},
            },
        },
    )

    splits = load_splits(DatasetConfig(cache_dir=tmp_path), require_counts=True)

    assert splits["train"].blue_win.tolist() == [1.0, 0.0]
    assert splits["val"].blue_win.tolist() == [1.0]
    assert splits["test"].blue_win.tolist() == [0.0]


def test_load_splits_uses_legacy_split_order_when_ranges_are_absent(tmp_path) -> None:
    _write_minimal_cache(
        tmp_path,
        [0, 1, 0, 1],
        {
            "format": CACHE_FORMAT,
            "n_games": 4,
            "splits": {"test": 1, "train": 2, "val": 1},
            "split_order": ["test", "train", "val"],
        },
    )

    splits = load_splits(DatasetConfig(cache_dir=tmp_path), require_counts=True)

    assert splits["train"].blue_win.tolist() == [1.0, 0.0]
    assert splits["val"].blue_win.tolist() == [1.0]
    assert splits["test"].blue_win.tolist() == [0.0]


def test_load_splits_can_skip_test_label_validation(tmp_path) -> None:
    _write_minimal_cache(
        tmp_path,
        [0, 1, 0, 2],
        {
            "format": CACHE_FORMAT,
            "n_games": 4,
            "splits": {"train": 2, "val": 1, "test": 1},
            "split_order": ["train", "val", "test"],
        },
    )

    splits = load_splits(
        DatasetConfig(cache_dir=tmp_path),
        require_counts=True,
        split_names=("train", "val"),
    )

    assert set(splits) == {"train", "val"}
    with pytest.raises(ValueError, match="blue_win labels"):
        load_splits(DatasetConfig(cache_dir=tmp_path), require_counts=True)
