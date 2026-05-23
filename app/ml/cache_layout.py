from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_FORMAT = "npy-memmap-v13"
CACHE_META_FILE = "cache_meta.json"

WIN_RATE_FILE = "win_rate.npy"
BLUE_WIN_FILE = "blue_win.npy"

ARRAY_FILES = {
    "win_rate": WIN_RATE_FILE,
    "blue_win": BLUE_WIN_FILE,
}

DISK_DTYPES = {
    "win_rate": np.float32,
    "blue_win": np.uint8,
}


def array_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / filename for name, filename in ARRAY_FILES.items()}
