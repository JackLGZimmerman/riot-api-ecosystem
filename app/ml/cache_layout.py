from __future__ import annotations

from pathlib import Path

import numpy as np

VOCAB_FILE = "vocab.json"
NORM_FILE = "normalization.npz"
CACHE_META_FILE = "cache_meta.json"
CACHE_FORMAT = "npy-memmap-v5"

INTERACTION_SCORE_FILE = "interaction_score.npy"
PLAYER_CHAMPION_BUILD_IDX_FILE = "player_champion_build_idx.npy"
CHAMPION_IDX_FILE = "champion_idx.npy"
ROLE_IDX_FILE = "role_idx.npy"
BUILD_IDX_FILE = "build_idx.npy"
BLUE_WIN_FILE = "blue_win.npy"

DISK_ARRAY_DTYPES = {
    "interaction_score": np.float16,
    "player_champion_build_idx": np.uint16,
    "blue_win": np.uint8,
}

LOAD_ARRAY_DTYPES = {
    "interaction_score": np.float32,
    "champion_idx": np.int32,
    "role_idx": np.int32,
    "build_idx": np.int32,
    "blue_win": np.float32,
}

ARRAY_FILES = {
    "interaction_score": INTERACTION_SCORE_FILE,
    "player_champion_build_idx": PLAYER_CHAMPION_BUILD_IDX_FILE,
    "blue_win": BLUE_WIN_FILE,
}

LEGACY_ARRAY_FILES = {
    "interaction_score": INTERACTION_SCORE_FILE,
    "champion_idx": CHAMPION_IDX_FILE,
    "role_idx": ROLE_IDX_FILE,
    "build_idx": BUILD_IDX_FILE,
    "blue_win": BLUE_WIN_FILE,
}

OBSOLETE_ARRAY_FILES = (
    CHAMPION_IDX_FILE,
    ROLE_IDX_FILE,
    BUILD_IDX_FILE,
    "interaction_score_raw.tmp.npy",
)


def array_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / filename for name, filename in ARRAY_FILES.items()}


__all__ = [
    "ARRAY_FILES",
    "BLUE_WIN_FILE",
    "BUILD_IDX_FILE",
    "CACHE_FORMAT",
    "CACHE_META_FILE",
    "CHAMPION_IDX_FILE",
    "DISK_ARRAY_DTYPES",
    "INTERACTION_SCORE_FILE",
    "LEGACY_ARRAY_FILES",
    "LOAD_ARRAY_DTYPES",
    "NORM_FILE",
    "OBSOLETE_ARRAY_FILES",
    "PLAYER_CHAMPION_BUILD_IDX_FILE",
    "ROLE_IDX_FILE",
    "VOCAB_FILE",
    "array_paths",
]
