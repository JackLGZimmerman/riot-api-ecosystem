from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from app.core.config.settings import PROJECT_ROOT

ML_DATA_DIR = PROJECT_ROOT / "app" / "ml" / "data"
CACHE_DIR = ML_DATA_DIR / "cache"
CHECKPOINT_DIR = ML_DATA_DIR / "checkpoints"

PARTICIPANT_TABLE = "game_data_filtered.participant_stats"
ITEM_VALUE_TABLE = "game_data_filtered.participant_item_value_totals"
SPLIT_TABLE = "game_data_filtered.ml_game_split"
PLAYER_PIVOT_TABLE = "game_data_filtered.ml_game_player_pivot"
INTERACTION_COUNTS_TABLE = "game_data_filtered.ml_interaction_counts"

POSITIONS: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")

ROLE_PAIR_COMBOS: tuple[tuple[int, int], ...] = tuple(
    combinations(range(len(POSITIONS)), 2)
)
ROLE_TRIO_COMBOS: tuple[tuple[int, int, int], ...] = tuple(
    combinations(range(len(POSITIONS)), 3)
)

# Bayesian smoothing toward 0.5 win rate; reliability saturates at K matchups.
MATCHUP_PRIOR_N = 25.0
MATCHUP_RELIABILITY_K = 100.0

# Token type ids for the hybrid champion + interaction model. UNK=0 stays
# unused at runtime but reserves the embedding row for safety.
TOKEN_TYPE_UNK = 0
TOKEN_TYPE_PLAYER = 1
TOKEN_TYPE_SYNERGY_SINGLE = 2
TOKEN_TYPE_MATCHUP_1V1 = 3
TOKEN_TYPE_SYNERGY_PAIR = 4
TOKEN_TYPE_MATCHUP_2V1 = 5
TOKEN_TYPE_SYNERGY_TRIO = 6
N_TOKEN_TYPES = 7

# Side ids: blue/red for same-side tokens, cross for matchups that span teams.
SIDE_UNK = 0
SIDE_BLUE = 1
SIDE_RED = 2
SIDE_CROSS = 3
N_SIDES = 4

# Reserve index 0 for unknown / padding across all categorical embeddings.
UNK_INDEX = 0

# Role embedding ids used in interaction token metadata. POSITIONS index i maps
# to role_to_idx[POSITIONS[i]] = i + 1; UNK_INDEX (0) is the "no role" slot.
def _role_id(i: int) -> int:
    return i + 1


N_ROLE_SLOTS = 4


def _build_interaction_layout() -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, int, int, int], ...],
]:
    """Return (token_type, side, role_slots) for each interaction token.

    Token order:
      1) single synergies blue (5)
      2) single synergies red (5)
      3) 1v1 matchups (5 blue x 5 red)
      4) pair synergies blue (10)
      5) pair synergies red (10)
      6) 2v1 blue-pair vs red-single (10 x 5)
      7) 2v1 red-pair vs blue-single (10 x 5)
      8) trio synergies blue (10)
      9) trio synergies red (10)

    Each token carries N_ROLE_SLOTS role ids. Unused slots are UNK_INDEX.
    Side is the token's "primary" side (blue, red, or cross). build_dataset
    and the model rely on this exact order.
    """
    types: list[int] = []
    sides: list[int] = []
    roles: list[tuple[int, int, int, int]] = []

    for side in (SIDE_BLUE, SIDE_RED):
        for role in range(len(POSITIONS)):
            types.append(TOKEN_TYPE_SYNERGY_SINGLE)
            sides.append(side)
            roles.append((_role_id(role), UNK_INDEX, UNK_INDEX, UNK_INDEX))

    for blue_role in range(len(POSITIONS)):
        for red_role in range(len(POSITIONS)):
            types.append(TOKEN_TYPE_MATCHUP_1V1)
            sides.append(SIDE_CROSS)
            roles.append(
                (_role_id(blue_role), _role_id(red_role), UNK_INDEX, UNK_INDEX)
            )

    for side in (SIDE_BLUE, SIDE_RED):
        for a, b in ROLE_PAIR_COMBOS:
            types.append(TOKEN_TYPE_SYNERGY_PAIR)
            sides.append(side)
            roles.append((_role_id(a), _role_id(b), UNK_INDEX, UNK_INDEX))

    # 2v1 blue-pair vs red-single: pair = primary blue side
    for blue_a, blue_b in ROLE_PAIR_COMBOS:
        for red_role in range(len(POSITIONS)):
            types.append(TOKEN_TYPE_MATCHUP_2V1)
            sides.append(SIDE_BLUE)
            roles.append(
                (_role_id(blue_a), _role_id(blue_b), _role_id(red_role), UNK_INDEX)
            )

    # 2v1 red-pair vs blue-single: pair = primary red side
    for red_a, red_b in ROLE_PAIR_COMBOS:
        for blue_role in range(len(POSITIONS)):
            types.append(TOKEN_TYPE_MATCHUP_2V1)
            sides.append(SIDE_RED)
            roles.append(
                (_role_id(red_a), _role_id(red_b), _role_id(blue_role), UNK_INDEX)
            )

    for side in (SIDE_BLUE, SIDE_RED):
        for a, b, c in ROLE_TRIO_COMBOS:
            types.append(TOKEN_TYPE_SYNERGY_TRIO)
            sides.append(side)
            roles.append((_role_id(a), _role_id(b), _role_id(c), UNK_INDEX))

    return tuple(types), tuple(sides), tuple(roles)


INTERACTION_TYPES, INTERACTION_SIDES, INTERACTION_ROLES = _build_interaction_layout()
N_INTERACTION_TOKENS = len(INTERACTION_TYPES)

N_SINGLE_SYNERGY_PER_SIDE = len(POSITIONS)
N_MATCHUP_1V1 = len(POSITIONS) ** 2
N_MATCHUP_2V1_PER_SIDE = len(ROLE_PAIR_COMBOS) * len(POSITIONS)
N_PAIR_SYNERGY_PER_SIDE = len(ROLE_PAIR_COMBOS)
N_TRIO_SYNERGY_PER_SIDE = len(ROLE_TRIO_COMBOS)


def _slice(start: int, length: int) -> tuple[slice, int]:
    return slice(start, start + length), start + length


SLICE_BLUE_SINGLE_SYN, _next = _slice(0, N_SINGLE_SYNERGY_PER_SIDE)
SLICE_RED_SINGLE_SYN, _next = _slice(_next, N_SINGLE_SYNERGY_PER_SIDE)
SLICE_MATCHUP_1V1, _next = _slice(_next, N_MATCHUP_1V1)
SLICE_BLUE_PAIR_SYN, _next = _slice(_next, N_PAIR_SYNERGY_PER_SIDE)
SLICE_RED_PAIR_SYN, _next = _slice(_next, N_PAIR_SYNERGY_PER_SIDE)
SLICE_BLUE_2V1, _next = _slice(_next, N_MATCHUP_2V1_PER_SIDE)
SLICE_RED_2V1, _next = _slice(_next, N_MATCHUP_2V1_PER_SIDE)
SLICE_BLUE_TRIO_SYN, _next = _slice(_next, N_TRIO_SYNERGY_PER_SIDE)
SLICE_RED_TRIO_SYN, _next = _slice(_next, N_TRIO_SYNERGY_PER_SIDE)
assert _next == N_INTERACTION_TOKENS


@dataclass(frozen=True)
class DatasetConfig:
    cache_dir: Path = CACHE_DIR
    max_games: int | None = None
    build_chunk_games: int = 50_000
    min_matchup_count: int = 5
    val_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.15
    head_hidden: int = 256


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4096
    epochs: int = 5
    lr: float = 2e-3
    weight_decay: float = 5e-2
    warmup_steps: int = 100
    grad_clip: float = 1.0
    num_workers: int = 4
    log_interval: int = 500
    checkpoint_dir: Path = CHECKPOINT_DIR
    metrics_file: str = "metrics.jsonl"
    latest_metrics_file: str = "metrics_latest.json"
    device: str = "cuda"
    use_amp: bool = False
    early_stopping_patience: int = 7
    seed: int = 42
