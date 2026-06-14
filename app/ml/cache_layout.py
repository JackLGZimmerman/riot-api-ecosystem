from __future__ import annotations

from pathlib import Path

import numpy as np

# v32 replaces the global 80/10/10 train/val/test split with the per-patch
# chronological 80/20 train/test split (no validation range); caches built
# under the old protocol must be rebuilt. The identity-encoder latents
# gather per batch from the small frozen artifact.
CACHE_FORMAT = "npy-memmap-v32"
CACHE_META_FILE = "cache_meta.json"

WIN_RATE_FILE = "win_rate.npy"
P1_CNT_FILE = "p1_cnt.npy"
CHAMPION_ID_FILE = "champion_id.npy"
BUILD_ID_FILE = "build_id.npy"
BLUE_WIN_FILE = "blue_win.npy"
PATCH_FEATURES_FILE = "patch_features.npy"
IDENTITY_STATIC_SIDECAR_FILE = "identity_static_sidecar.npy"
IDENTITY_FULL_GAME_SIDECAR_FILE = "identity_full_game_sidecar.npy"
IDENTITY_TEMPORAL_SIDECAR_FILE = "identity_temporal_sidecar.npy"
IDENTITY_ENCODER_SUPPORT_FILE = "identity_encoder_support.npy"

N_PLAYERS_PER_GAME = 10
ARRAY_FILES = {
    "win_rate": WIN_RATE_FILE,
    "p1_cnt": P1_CNT_FILE,
    "champion_id": CHAMPION_ID_FILE,
    "build_id": BUILD_ID_FILE,
    "blue_win": BLUE_WIN_FILE,
    "patch_features": PATCH_FEATURES_FILE,
}

DISK_DTYPES = {
    "win_rate": np.float32,
    "p1_cnt": np.float32,
    "champion_id": np.int16,
    "build_id": np.int16,
    "blue_win": np.uint8,
    "patch_features": np.float32,
}

ARRAY_SHAPES = {
    "win_rate": (N_PLAYERS_PER_GAME,),
    "p1_cnt": (N_PLAYERS_PER_GAME,),
    "champion_id": (N_PLAYERS_PER_GAME,),
    "build_id": (N_PLAYERS_PER_GAME,),
    "blue_win": (),
    "patch_features": (2,),
}

SIDECAR_ARRAY_FILES = {
    "identity_static_sidecar": IDENTITY_STATIC_SIDECAR_FILE,
    "identity_full_game_sidecar": IDENTITY_FULL_GAME_SIDECAR_FILE,
    "identity_temporal_sidecar": IDENTITY_TEMPORAL_SIDECAR_FILE,
    "identity_encoder_support": IDENTITY_ENCODER_SUPPORT_FILE,
}


def array_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / filename for name, filename in ARRAY_FILES.items()}


def sidecar_array_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / filename for name, filename in SIDECAR_ARRAY_FILES.items()}
