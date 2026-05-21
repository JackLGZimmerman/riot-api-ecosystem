from __future__ import annotations

from pathlib import Path

import numpy as np

VOCAB_FILE = "vocab.json"
CACHE_META_FILE = "cache_meta.json"
CACHE_FORMAT = "npy-memmap-v9"

PLAYER_CHAMPION_BUILD_IDX_FILE = "player_champion_build_idx.npy"
PLAYER_PROFILE_FILE = "player_profile.npy"
BLUE_WIN_FILE = "blue_win.npy"

N_PROFILE_BINS = 4
PROFILE_FEATURE_COLUMNS = (
    "log_matchups",
    "win_rate",
    "avg_gold",
    "avg_xp",
    "avg_item_completions",
    "avg_total_cs",
    "avg_kills",
    "avg_kills_assists",
    "avg_total_damage_dealt",
    "physical_damage_share",
    "magic_damage_share",
    "true_damage_share",
    "avg_damage_taken",
    "avg_durability",
    "damage_to_taken_ratio",
    "avg_time_ccing_others",
    "avg_protection",
    "avg_epic_monster_takedowns",
    "avg_turret_takedowns",
    "avg_damage_to_objectives",
    "avg_vision_score",
    "avg_control_wards_bought",
)
N_PROFILE_FEATURES = len(PROFILE_FEATURE_COLUMNS)

DISK_ARRAY_DTYPES = {
    "player_champion_build_idx": np.uint16,
    "player_profile": np.float16,
    "blue_win": np.uint8,
}

LOAD_ARRAY_DTYPES = {
    "champion_idx": np.int32,
    "role_idx": np.int32,
    "build_idx": np.int32,
    "player_profile": np.float16,
    "blue_win": np.float32,
}

ARRAY_FILES = {
    "player_champion_build_idx": PLAYER_CHAMPION_BUILD_IDX_FILE,
    "player_profile": PLAYER_PROFILE_FILE,
    "blue_win": BLUE_WIN_FILE,
}


def array_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / filename for name, filename in ARRAY_FILES.items()}


__all__ = [
    "ARRAY_FILES",
    "BLUE_WIN_FILE",
    "CACHE_FORMAT",
    "CACHE_META_FILE",
    "DISK_ARRAY_DTYPES",
    "LOAD_ARRAY_DTYPES",
    "N_PROFILE_BINS",
    "N_PROFILE_FEATURES",
    "PLAYER_CHAMPION_BUILD_IDX_FILE",
    "PLAYER_PROFILE_FILE",
    "PROFILE_FEATURE_COLUMNS",
    "VOCAB_FILE",
    "array_paths",
]
