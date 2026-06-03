from __future__ import annotations

from pathlib import Path

import numpy as np

# v28 stops materialising per-game identity-encoder sidecar latents; they are
# gathered per batch from the small frozen artifact via champion_id/build_id.
CACHE_FORMAT = "npy-memmap-v28"
CACHE_META_FILE = "cache_meta.json"

WIN_RATE_FILE = "win_rate.npy"
MATCHUP_1V1_FILE = "matchup_1v1.npy"
SYNERGY_2VX_FILE = "synergy_2vx.npy"
P1_CNT_FILE = "p1_cnt.npy"
MATCHUP_1V1_CNT_FILE = "m1v1_cnt.npy"
SYNERGY_2VX_CNT_FILE = "s2vx_cnt.npy"
# Effective sample size after nested empirical-Bayes pooling (build_dataset):
# inherits a dense parent level's support, so the HGNN's posterior variance /
# φ-gate sees the true confidence of a backed-off interaction edge.
MATCHUP_1V1_EFF_N_FILE = "m1v1_eff_n.npy"
SYNERGY_2VX_EFF_N_FILE = "s2vx_eff_n.npy"
CHAMPION_ID_FILE = "champion_id.npy"
BUILD_ID_FILE = "build_id.npy"
BLUE_WIN_FILE = "blue_win.npy"
IDENTITY_STATIC_SIDECAR_FILE = "identity_static_sidecar.npy"
IDENTITY_FULL_GAME_SIDECAR_FILE = "identity_full_game_sidecar.npy"
IDENTITY_TEMPORAL_SIDECAR_FILE = "identity_temporal_sidecar.npy"
IDENTITY_ENCODER_SUPPORT_FILE = "identity_encoder_support.npy"

N_PLAYERS_PER_GAME = 10
N_MATCHUPS_1V1 = 25
N_SYNERGIES_2VX = 20
ARRAY_FILES = {
    "win_rate": WIN_RATE_FILE,
    "matchup_1v1": MATCHUP_1V1_FILE,
    "synergy_2vx": SYNERGY_2VX_FILE,
    "p1_cnt": P1_CNT_FILE,
    "m1v1_cnt": MATCHUP_1V1_CNT_FILE,
    "s2vx_cnt": SYNERGY_2VX_CNT_FILE,
    "m1v1_eff_n": MATCHUP_1V1_EFF_N_FILE,
    "s2vx_eff_n": SYNERGY_2VX_EFF_N_FILE,
    "champion_id": CHAMPION_ID_FILE,
    "build_id": BUILD_ID_FILE,
    "blue_win": BLUE_WIN_FILE,
}

DISK_DTYPES = {
    "win_rate": np.float32,
    "matchup_1v1": np.float32,
    "synergy_2vx": np.float32,
    "p1_cnt": np.float32,
    "m1v1_cnt": np.float32,
    "s2vx_cnt": np.float32,
    "m1v1_eff_n": np.float32,
    "s2vx_eff_n": np.float32,
    "champion_id": np.int16,
    "build_id": np.int16,
    "blue_win": np.uint8,
}

ARRAY_SHAPES = {
    "win_rate": (N_PLAYERS_PER_GAME,),
    "matchup_1v1": (N_MATCHUPS_1V1,),
    "synergy_2vx": (N_SYNERGIES_2VX,),
    "p1_cnt": (N_PLAYERS_PER_GAME,),
    "m1v1_cnt": (N_MATCHUPS_1V1,),
    "s2vx_cnt": (N_SYNERGIES_2VX,),
    "m1v1_eff_n": (N_MATCHUPS_1V1,),
    "s2vx_eff_n": (N_SYNERGIES_2VX,),
    "champion_id": (N_PLAYERS_PER_GAME,),
    "build_id": (N_PLAYERS_PER_GAME,),
    "blue_win": (),
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
