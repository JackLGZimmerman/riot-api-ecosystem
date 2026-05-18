from __future__ import annotations

from pathlib import Path

import numpy as np

VOCAB_FILE = "vocab.json"
CACHE_META_FILE = "cache_meta.json"
CACHE_FORMAT = "npy-memmap-v6"

PLAYER_CHAMPION_BUILD_IDX_FILE = "player_champion_build_idx.npy"
BLUE_WIN_FILE = "blue_win.npy"

DISK_ARRAY_DTYPES = {
    "player_champion_build_idx": np.uint16,
    "blue_win": np.uint8,
}

LOAD_ARRAY_DTYPES = {
    "champion_idx": np.int32,
    "role_idx": np.int32,
    "build_idx": np.int32,
    "blue_win": np.float32,
}

ARRAY_FILES = {
    "player_champion_build_idx": PLAYER_CHAMPION_BUILD_IDX_FILE,
    "blue_win": BLUE_WIN_FILE,
}

OBSOLETE_ARRAY_FILES = (
    "champion_idx.npy",
    "role_idx.npy",
    "build_idx.npy",
    "interaction_score.npy",
    "interaction_score_raw.tmp.npy",
    "normalization.npz",
)


def array_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / filename for name, filename in ARRAY_FILES.items()}


__all__ = [
    "ARRAY_FILES",
    "BLUE_WIN_FILE",
    "CACHE_FORMAT",
    "CACHE_META_FILE",
    "DISK_ARRAY_DTYPES",
    "LOAD_ARRAY_DTYPES",
    "OBSOLETE_ARRAY_FILES",
    "PLAYER_CHAMPION_BUILD_IDX_FILE",
    "VOCAB_FILE",
    "array_paths",
]
