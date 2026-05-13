from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

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

# Wilson score smoothing shrinks weak interaction win-rate signals toward 0
# unless the observed effect clears its binomial uncertainty band.
MATCHUP_CONFIDENCE_Z = 1.96

# Token type ids for the hybrid champion + interaction model. UNK=0 stays
# unused at runtime but reserves the embedding row for safety.
TOKEN_TYPE_UNK = 0
TOKEN_TYPE_PLAYER = 1
TOKEN_TYPE_SYNERGY_SINGLE = 2
TOKEN_TYPE_MATCHUP_1V1 = 3
TOKEN_TYPE_SYNERGY_PAIR = 4
N_TOKEN_TYPES = 5

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


N_ROLE_SLOTS = 2


def _build_interaction_layout() -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, int], ...],
]:
    """Return (token_type, side, role_slots) for each interaction token.

    Active token order:
      1) single synergies blue (5)
      2) single synergies red (5)

    Each token carries N_ROLE_SLOTS role ids. Unused slots are UNK_INDEX.
    Side is the token's "primary" side (blue, red, or cross). build_dataset
    and the model rely on this exact order.
    """
    types: list[int] = []
    sides: list[int] = []
    roles: list[tuple[int, int]] = []

    for side in (SIDE_BLUE, SIDE_RED):
        for role in range(len(POSITIONS)):
            types.append(TOKEN_TYPE_SYNERGY_SINGLE)
            sides.append(side)
            roles.append((_role_id(role), UNK_INDEX))

    # Disabled for the current 1vX-only training session:
    #
    # for blue_role in range(len(POSITIONS)):
    #     for red_role in range(len(POSITIONS)):
    #         types.append(TOKEN_TYPE_MATCHUP_1V1)
    #         sides.append(SIDE_CROSS)
    #         roles.append((_role_id(blue_role), _role_id(red_role)))
    #
    # for side in (SIDE_BLUE, SIDE_RED):
    #     for a, b in ROLE_PAIR_COMBOS:
    #         types.append(TOKEN_TYPE_SYNERGY_PAIR)
    #         sides.append(side)
    #         roles.append((_role_id(a), _role_id(b)))

    return tuple(types), tuple(sides), tuple(roles)


INTERACTION_TYPES, INTERACTION_SIDES, INTERACTION_ROLES = _build_interaction_layout()
N_INTERACTION_TOKENS = len(INTERACTION_TYPES)

N_SINGLE_SYNERGY_PER_SIDE = len(POSITIONS)
N_MATCHUP_1V1 = len(POSITIONS) ** 2
N_PAIR_SYNERGY_PER_SIDE = len(ROLE_PAIR_COMBOS)


def _slice(start: int, length: int) -> tuple[slice, int]:
    return slice(start, start + length), start + length


SLICE_BLUE_SINGLE_SYN, _next = _slice(0, N_SINGLE_SYNERGY_PER_SIDE)
SLICE_RED_SINGLE_SYN, _next = _slice(_next, N_SINGLE_SYNERGY_PER_SIDE)
SLICE_MATCHUP_1V1 = slice(_next, _next)
SLICE_BLUE_PAIR_SYN = slice(_next, _next)
SLICE_RED_PAIR_SYN = slice(_next, _next)
# Disabled for the current 1vX-only training session:
#
# SLICE_MATCHUP_1V1, _next = _slice(_next, N_MATCHUP_1V1)
# SLICE_BLUE_PAIR_SYN, _next = _slice(_next, N_PAIR_SYNERGY_PER_SIDE)
# SLICE_RED_PAIR_SYN, _next = _slice(_next, N_PAIR_SYNERGY_PER_SIDE)
assert _next == N_INTERACTION_TOKENS


@dataclass(frozen=True)
class DatasetConfig:
    cache_dir: Path = CACHE_DIR
    max_games: int | None = None
    build_chunk_games: int = 150_000
    min_matchup_count: int = 5
    smooth_interaction_scores: bool = False
    val_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 1536
    dropout: float = 0.15
    attention_dropout: float = 0.10
    head_dropout: float = 0.0
    pooling: Literal["cls", "mean", "attention", "concat_cls_mean", "gated"] = "gated"
    head_hidden: int = 256

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("ModelConfig.d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("ModelConfig.n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("ModelConfig.d_model must be divisible by n_heads")
        if self.n_layers <= 0:
            raise ValueError("ModelConfig.n_layers must be positive")
        if self.dim_feedforward < self.d_model:
            raise ValueError("ModelConfig.dim_feedforward must be >= d_model")
        if self.pooling not in {
            "cls",
            "mean",
            "attention",
            "concat_cls_mean",
            "gated",
        }:
            raise ValueError(f"Unsupported ModelConfig.pooling: {self.pooling}")
        for name, value in (
            ("dropout", self.dropout),
            ("attention_dropout", self.attention_dropout),
            ("head_dropout", self.head_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"ModelConfig.{name} must satisfy 0 <= value < 1")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 10_240
    gradient_accumulation_steps: int = 1
    epochs: int = 100
    optimizer: Literal["lion"] = "lion"
    lr: float = 1e-5
    weight_decay: float = 2.5e-2
    lion_betas: tuple[float, float] = (0.9, 0.99)
    target_min: float = 0.15
    target_max: float = 0.85
    warmup_steps: int = 125
    grad_clip: float = 1.0
    log_interval: int = 40
    attention_diagnostics_interval: int = 5
    attention_diagnostics_batch_size: int = 256
    attention_diagnostics_eval_samples: int = 1024
    attention_diagnostics_eval_batches: int = 0
    checkpoint_dir: Path = CHECKPOINT_DIR
    metrics_file: str = "metrics.jsonl"
    latest_metrics_file: str = "metrics_latest.json"
    tensorboard_dir: str | None = "tensorboard"
    device: str = "cuda"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    early_stopping_patience: int = 8
    seed: int = 42
