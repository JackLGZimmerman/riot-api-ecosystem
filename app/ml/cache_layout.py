from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_FORMAT = "npy-memmap-v20"
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
IDENTITY_SEMANTIC_FILE = "identity_semantic.npy"
IDENTITY_PROFILE_FILE = "identity_profile.npy"
M1V1_DETAIL_FILE = "m1v1_detail.npy"
S2VX_DETAIL_FILE = "s2vx_detail.npy"
BLUE_WIN_FILE = "blue_win.npy"

N_PLAYERS_PER_GAME = 10
N_MATCHUPS_1V1 = 25
N_SYNERGIES_2VX = 20
IDENTITY_SEMANTIC_DIM = 64
IDENTITY_PROFILE_DIM = 5
RELATIONSHIP_DETAIL_DIM = 16

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
    "identity_semantic": IDENTITY_SEMANTIC_FILE,
    "identity_profile": IDENTITY_PROFILE_FILE,
    "m1v1_detail": M1V1_DETAIL_FILE,
    "s2vx_detail": S2VX_DETAIL_FILE,
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
    "identity_semantic": np.float32,
    "identity_profile": np.float32,
    "m1v1_detail": np.float32,
    "s2vx_detail": np.float32,
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
    "identity_semantic": (N_PLAYERS_PER_GAME, IDENTITY_SEMANTIC_DIM),
    "identity_profile": (N_PLAYERS_PER_GAME, IDENTITY_PROFILE_DIM),
    "m1v1_detail": (N_MATCHUPS_1V1, RELATIONSHIP_DETAIL_DIM),
    "s2vx_detail": (N_SYNERGIES_2VX, RELATIONSHIP_DETAIL_DIM),
    "blue_win": (),
}


def array_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / filename for name, filename in ARRAY_FILES.items()}
