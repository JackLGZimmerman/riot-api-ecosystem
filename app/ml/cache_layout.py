from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_FORMAT = "npy-memmap-v14"
CACHE_META_FILE = "cache_meta.json"

WIN_RATE_FILE = "win_rate.npy"
MATCHUP_1V1_FILE = "matchup_1v1.npy"
SYNERGY_2VX_FILE = "synergy_2vx.npy"
BLUE_WIN_FILE = "blue_win.npy"

N_PLAYERS_PER_GAME = 10
N_MATCHUPS_1V1 = 25
N_SYNERGIES_2VX = 20

ARRAY_FILES = {
    "win_rate": WIN_RATE_FILE,
    "matchup_1v1": MATCHUP_1V1_FILE,
    "synergy_2vx": SYNERGY_2VX_FILE,
    "blue_win": BLUE_WIN_FILE,
}

DISK_DTYPES = {
    "win_rate": np.float32,
    "matchup_1v1": np.float32,
    "synergy_2vx": np.float32,
    "blue_win": np.uint8,
}

ARRAY_SHAPES = {
    "win_rate": (N_PLAYERS_PER_GAME,),
    "matchup_1v1": (N_MATCHUPS_1V1,),
    "synergy_2vx": (N_SYNERGIES_2VX,),
    "blue_win": (),
}


def array_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / filename for name, filename in ARRAY_FILES.items()}
