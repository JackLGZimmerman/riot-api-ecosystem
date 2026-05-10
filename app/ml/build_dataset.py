"""Build the training cache.

Streams one row per game from `game_data_filtered.ml_game_player_pivot`
plus the long-form `ml_interaction_counts` table (token-level matchups +
primary-side wins, materialised by the 6901 SQL build with matchups >= 5
applied at the source). Per-chunk we apply leave-one-out for train games
and Bayesian smoothing in a single vectorised numpy pass, then write
memory-mapped .npy arrays plus metadata.

Run with:
    python -m app.ml.build_dataset
"""

from __future__ import annotations

import json
import logging
import time
from hashlib import sha1
from pathlib import Path

import numpy as np
from clickhouse_connect.driver.external import ExternalData

from app.core.logging.logger import setup_logging_config
from app.ml.config import (
    INTERACTION_COUNTS_TABLE,
    INTERACTION_ROLES,
    INTERACTION_SIDES,
    INTERACTION_TYPES,
    ITEM_VALUE_TABLE,
    MATCHUP_PRIOR_N,
    MATCHUP_RELIABILITY_K,
    N_INTERACTION_TOKENS,
    N_ROLE_SLOTS,
    PARTICIPANT_TABLE,
    PLAYER_PIVOT_TABLE,
    POSITIONS,
    SIDE_RED,
    SPLIT_TABLE,
    UNK_INDEX,
    DatasetConfig,
)
from database.clickhouse.client import get_client

setup_logging_config()
logger = logging.getLogger(__name__)

VOCAB_FILE = "vocab.json"
NORM_FILE = "normalization.npz"
CACHE_META_FILE = "cache_meta.json"

INTERACTION_SCORE_FILE = "interaction_score.npy"
INTERACTION_RELIABILITY_FILE = "interaction_reliability.npy"
CHAMPION_IDX_FILE = "champion_idx.npy"
ROLE_IDX_FILE = "role_idx.npy"
BUILD_IDX_FILE = "build_idx.npy"
BLUE_WIN_FILE = "blue_win.npy"

ARRAY_FILES = {
    "interaction_score": INTERACTION_SCORE_FILE,
    "interaction_reliability": INTERACTION_RELIABILITY_FILE,
    "champion_idx": CHAMPION_IDX_FILE,
    "role_idx": ROLE_IDX_FILE,
    "build_idx": BUILD_IDX_FILE,
    "blue_win": BLUE_WIN_FILE,
}

# --------------------------------------------------------------------------
# Per-token LOO metadata (computed once at module import)
# --------------------------------------------------------------------------

# True for tokens whose primary side is red; LOO subtracts (1 - blue_win)
# from primary_wins for these; everything else (BLUE / CROSS) subtracts blue_win.
_TOKEN_IS_RED = np.array(
    [side == SIDE_RED for side in INTERACTION_SIDES], dtype=bool
)
# All tokens count each observed combo once.
_TOKEN_LOO_WEIGHT = np.ones(len(INTERACTION_TYPES), dtype=np.int64)


# --------------------------------------------------------------------------
# Match selection
# --------------------------------------------------------------------------


def _sql_split_matchids(cfg: DatasetConfig) -> tuple[list[str], list[str], list[str]]:
    rows = (
        get_client()
        .query(
            f"""
            SELECT split, matchid
            FROM {SPLIT_TABLE}
            WHERE split IN ('train', 'validation', 'test')
            ORDER BY split_index
            """
        )
        .result_rows
    )
    by_split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for split, matchid in rows:
        by_split[str(split)].append(str(matchid))

    if cfg.max_games is not None:
        n = int(cfg.max_games)
        n_test = int(round(n * cfg.test_fraction))
        n_val = int(round(n * cfg.val_fraction))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError("Training split is empty; increase max_games.")
        by_split["train"] = by_split["train"][:n_train]
        by_split["validation"] = by_split["validation"][:n_val]
        by_split["test"] = by_split["test"][:n_test]

    if not by_split["train"]:
        raise ValueError(
            f"{SPLIT_TABLE} has no train rows. Run the 5900 split schema/build SQL."
        )
    return by_split["train"], by_split["validation"], by_split["test"]


def _matchids_external_data(matchids: list[str]) -> ExternalData:
    data = "\n".join(matchids).encode()
    if data:
        data += b"\n"
    return ExternalData(
        data=data,
        file_name="selected_matchids.tsv",
        fmt="TabSeparated",
        structure="matchid String",
    )


def _iter_query_rows(query: str, matchids: list[str]):
    with get_client().query_column_block_stream(
        query, external_data=_matchids_external_data(matchids)
    ) as stream:
        for block in stream:
            if not block or len(block[0]) == 0:
                continue
            yield from zip(*block, strict=False)


def _matchids_hash(matchids: list[str]) -> str:
    digest = sha1()
    for matchid in matchids:
        digest.update(matchid.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


def _build_vocabularies() -> dict[str, object]:
    client = get_client()
    champions = client.query(
        f"""
        SELECT DISTINCT assumeNotNull(championid) AS championid
        FROM {PARTICIPANT_TABLE}
        WHERE championid IS NOT NULL AND teamposition != 'UNKNOWN'
        ORDER BY championid
        """
    ).result_rows
    builds = client.query(
        f"""
        SELECT DISTINCT highest_value_label AS build
        FROM {ITEM_VALUE_TABLE}
        ORDER BY build
        """
    ).result_rows

    champion_to_idx = {int(c[0]): i + 1 for i, c in enumerate(champions)}
    build_to_idx = {str(b[0]): i + 1 for i, b in enumerate(builds)}
    role_to_idx = {p: i + 1 for i, p in enumerate(POSITIONS)}

    return {
        "champion_to_idx": champion_to_idx,
        "build_to_idx": build_to_idx,
        "role_to_idx": role_to_idx,
        "n_champions": len(champion_to_idx) + 1,
        "n_builds": len(build_to_idx) + 1,
        "n_roles": len(role_to_idx) + 1,
        "n_sides": 4,
        "unk_index": UNK_INDEX,
        "n_interaction_tokens": N_INTERACTION_TOKENS,
        "interaction_types": list(INTERACTION_TYPES),
        "interaction_sides": list(INTERACTION_SIDES),
        "interaction_roles": [list(roles) for roles in INTERACTION_ROLES],
    }


# --------------------------------------------------------------------------
# Cache layout
# --------------------------------------------------------------------------


def _array_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / filename for name, filename in ARRAY_FILES.items()}


def _allocate_arrays(n_games: int, cache_dir: Path) -> dict[str, np.ndarray]:
    return {
        "interaction_score": np.lib.format.open_memmap(
            cache_dir / INTERACTION_SCORE_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(n_games, N_INTERACTION_TOKENS),
        ),
        "interaction_reliability": np.lib.format.open_memmap(
            cache_dir / INTERACTION_RELIABILITY_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(n_games, N_INTERACTION_TOKENS),
        ),
        "champion_idx": np.lib.format.open_memmap(
            cache_dir / CHAMPION_IDX_FILE,
            mode="w+",
            dtype=np.int64,
            shape=(n_games, 10),
        ),
        "role_idx": np.lib.format.open_memmap(
            cache_dir / ROLE_IDX_FILE,
            mode="w+",
            dtype=np.int64,
            shape=(n_games, 10),
        ),
        "build_idx": np.lib.format.open_memmap(
            cache_dir / BUILD_IDX_FILE,
            mode="w+",
            dtype=np.int64,
            shape=(n_games, 10),
        ),
        "blue_win": np.lib.format.open_memmap(
            cache_dir / BLUE_WIN_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(n_games,),
        ),
    }


# --------------------------------------------------------------------------
# Vectorised LOO + Bayesian smoothing
# --------------------------------------------------------------------------


def _score_and_reliability(
    matchups: np.ndarray,
    primary_wins: np.ndarray,
    blue_wins: np.ndarray,
    exclude_self: bool,
    min_matchups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply LOO (for train chunks) and Bayesian smoothing across all tokens.

    matchups, primary_wins: (n, N_INTERACTION_TOKENS) uint32 from the SQL
    aggregate table (rows for unobserved (game, token) pairs are zero).
    blue_wins: (n,) 0/1 outcomes.
    """
    has_row = matchups > 0
    m = matchups.astype(np.int64)
    w = primary_wins.astype(np.int64)

    if exclude_self:
        bw = blue_wins.astype(np.int64).reshape(-1, 1)
        primary_won = np.where(_TOKEN_IS_RED, 1 - bw, bw)
        delta_w = _TOKEN_LOO_WEIGHT * primary_won
        m = np.where(has_row, m - _TOKEN_LOO_WEIGHT, 0)
        w = np.where(has_row, w - delta_w, 0)
        m = np.maximum(m, 0)
        w = np.maximum(w, 0)

    valid = m >= max(1, int(min_matchups))
    support = m.astype(np.float32)
    score = np.where(
        valid,
        (w.astype(np.float32) + MATCHUP_PRIOR_N * 0.5) / (support + MATCHUP_PRIOR_N) - 0.5,
        np.float32(0.0),
    ).astype(np.float32)
    reliability = np.where(
        valid,
        support / (support + MATCHUP_RELIABILITY_K),
        np.float32(0.0),
    ).astype(np.float32)
    return score, reliability


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def _resolve_player_idx(
    cid: int,
    pos: str,
    build: str,
    champion_lut: np.ndarray,
    role_map: dict[str, int],
    build_map: dict[str, int],
) -> tuple[int, int, int]:
    champ_idx = int(champion_lut[cid]) if 0 <= cid < champion_lut.shape[0] else UNK_INDEX
    role_idx = role_map.get(pos, UNK_INDEX)
    build_idx = build_map.get(build, UNK_INDEX)
    return champ_idx, role_idx, build_idx


def _build_champion_lut(vocab: dict) -> np.ndarray:
    keys = vocab["champion_to_idx"]
    max_id = max(keys) if keys else 0
    lut = np.zeros(max_id + 1, dtype=np.int32)
    for k, v in keys.items():
        lut[k] = v
    return lut


def _player_pivot_query() -> str:
    return f"""
    SELECT
        g.matchid,
        g.blue_win,
        g.blue_players,
        g.red_players
    FROM {PLAYER_PIVOT_TABLE} AS g
    INNER JOIN selected_matchids AS sg ON g.matchid = sg.matchid
    """


def _interaction_counts_query() -> str:
    return f"""
    SELECT ic.matchid, ic.token_idx, ic.matchups, ic.primary_wins
    FROM {INTERACTION_COUNTS_TABLE} AS ic
    INNER JOIN selected_matchids AS sg ON ic.matchid = sg.matchid
    """


def _stream_into_arrays(
    cfg: DatasetConfig,
    vocab: dict,
    arrays: dict[str, np.ndarray],
    matchids: list[str],
    start_write_idx: int,
    exclude_self: bool,
) -> int:
    champion_lut = _build_champion_lut(vocab)
    role_map = vocab["role_to_idx"]
    build_map = vocab["build_to_idx"]
    min_matchups = max(1, int(cfg.min_matchup_count))

    t0 = time.perf_counter()
    pivot_rows = list(_iter_query_rows(_player_pivot_query(), matchids))
    if not pivot_rows:
        return 0
    n = len(pivot_rows)
    matchid_to_idx = {str(row[0]): i for i, row in enumerate(pivot_rows)}
    blue_wins = np.zeros(n, dtype=np.int64)

    for i, (_matchid, blue_win, blue_players, red_players) in enumerate(pivot_rows):
        for slot, (cid, pos, build) in enumerate(blue_players):
            c, r, b = _resolve_player_idx(
                int(cid), str(pos), str(build), champion_lut, role_map, build_map
            )
            arrays["champion_idx"][start_write_idx + i, slot] = c
            arrays["role_idx"][start_write_idx + i, slot] = r
            arrays["build_idx"][start_write_idx + i, slot] = b
        for slot, (cid, pos, build) in enumerate(red_players):
            c, r, b = _resolve_player_idx(
                int(cid), str(pos), str(build), champion_lut, role_map, build_map
            )
            arrays["champion_idx"][start_write_idx + i, 5 + slot] = c
            arrays["role_idx"][start_write_idx + i, 5 + slot] = r
            arrays["build_idx"][start_write_idx + i, 5 + slot] = b
        blue_wins[i] = int(blue_win)
        arrays["blue_win"][start_write_idx + i] = float(blue_win)

    matchups_chunk = np.zeros((n, N_INTERACTION_TOKENS), dtype=np.uint32)
    primary_wins_chunk = np.zeros((n, N_INTERACTION_TOKENS), dtype=np.uint32)
    n_token_rows = 0
    for matchid, token_idx, matchups, primary_wins in _iter_query_rows(
        _interaction_counts_query(), matchids
    ):
        i = matchid_to_idx.get(str(matchid))
        if i is None:
            continue
        matchups_chunk[i, int(token_idx)] = int(matchups)
        primary_wins_chunk[i, int(token_idx)] = int(primary_wins)
        n_token_rows += 1

    score, reliability = _score_and_reliability(
        matchups_chunk, primary_wins_chunk, blue_wins, exclude_self, min_matchups
    )
    arrays["interaction_score"][start_write_idx:start_write_idx + n] = score
    arrays["interaction_reliability"][start_write_idx:start_write_idx + n] = reliability

    logger.info(
        "Chunk filled in %.1fs (games=%d token_rows=%d exclude_self=%s)",
        time.perf_counter() - t0, n, n_token_rows, exclude_self,
    )
    return n


def _flush_arrays(arrays: dict[str, np.ndarray]) -> None:
    for arr in arrays.values():
        flush = getattr(arr, "flush", None)
        if flush is not None:
            flush()


def _compute_normalization(
    features: np.ndarray, chunk_games: int = 4_096
) -> tuple[np.ndarray, np.ndarray]:
    n_features = features.shape[-1]
    total = 0
    total_sum = np.zeros(n_features, dtype=np.float64)
    total_sum_sq = np.zeros(n_features, dtype=np.float64)
    for start in range(0, features.shape[0], chunk_games):
        chunk = np.asarray(
            features[start : start + chunk_games], dtype=np.float64
        ).reshape(-1, n_features)
        np.nan_to_num(chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        total += chunk.shape[0]
        total_sum += chunk.sum(axis=0)
        total_sum_sq += np.square(chunk).sum(axis=0)
    mean = total_sum / max(1, total)
    variance = np.maximum(total_sum_sq / max(1, total) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_in_place(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    chunk_games: int = 4_096,
) -> None:
    for start in range(0, features.shape[0], chunk_games):
        chunk = features[start : start + chunk_games]
        chunk -= mean
        chunk /= std
        np.nan_to_num(chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    flush = getattr(features, "flush", None)
    if flush is not None:
        flush()


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build(cfg: DatasetConfig | None = None) -> Path:
    cfg = cfg or DatasetConfig()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building vocabularies")
    vocab = _build_vocabularies()
    logger.info(
        "Vocab sizes: champions=%d builds=%d roles=%d",
        vocab["n_champions"],
        vocab["n_builds"],
        vocab["n_roles"],
    )

    logger.info("Selecting split matchids from %s", SPLIT_TABLE)
    train_matchids, val_matchids, test_matchids = _sql_split_matchids(cfg)
    n_games = len(train_matchids) + len(val_matchids) + len(test_matchids)
    logger.info(
        "Selected games: train=%d val=%d test=%d",
        len(train_matchids),
        len(val_matchids),
        len(test_matchids),
    )

    logger.info(
        "Allocating arrays for %d games x %d interaction tokens (%.1f MB scores+reliability)",
        n_games,
        N_INTERACTION_TOKENS,
        n_games * N_INTERACTION_TOKENS * 4 * 2 / 1e6,
    )
    arrays = _allocate_arrays(n_games, cfg.cache_dir)

    n_written = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_specs = (
        ("train", train_matchids, True),
        ("val", val_matchids, False),
        ("test", test_matchids, False),
    )
    for split_name, split_matchids, exclude_self in split_specs:
        for split_offset in range(0, len(split_matchids), cfg.build_chunk_games):
            chunk = split_matchids[
                split_offset : split_offset + cfg.build_chunk_games
            ]
            logger.info(
                "Streaming %s game chunk offset=%d limit=%d",
                split_name,
                split_offset,
                len(chunk),
            )
            chunk_written = _stream_into_arrays(
                cfg, vocab, arrays, chunk, n_written, exclude_self
            )
            if chunk_written == 0:
                logger.warning("Stopping %s split after an empty chunk", split_name)
                break
            n_written += chunk_written
            split_counts[split_name] += chunk_written
    logger.info("Wrote %d / %d games", n_written, n_games)

    if n_written < n_games:
        for k, arr in arrays.items():
            arrays[k] = arr[:n_written]

    train_games = split_counts["train"]
    if train_games <= 0:
        raise ValueError("Training split is empty; increase the dataset size.")

    logger.info("Computing train-only score normalization")
    score_mean, score_std = _compute_normalization(
        arrays["interaction_score"][:train_games]
    )
    logger.info("Normalizing interaction scores in place")
    _normalize_in_place(arrays["interaction_score"], score_mean, score_std)
    _flush_arrays(arrays)

    norm_path = cfg.cache_dir / NORM_FILE
    vocab_path = cfg.cache_dir / VOCAB_FILE
    meta_path = cfg.cache_dir / CACHE_META_FILE

    logger.info("Writing %s", norm_path)
    np.savez(norm_path, score_mean=score_mean, score_std=score_std)
    vocab_path.write_text(json.dumps(vocab, indent=2))
    meta_path.write_text(
        json.dumps(
            {
                "format": "npy-memmap-v3",
                "n_games": n_written,
                "n_interaction_tokens": N_INTERACTION_TOKENS,
                "n_role_slots": N_ROLE_SLOTS,
                "min_matchup_count": max(1, int(cfg.min_matchup_count)),
                "score_normalized": True,
                "splits": {
                    "strategy": "sql_ml_game_split",
                    "train": split_counts["train"],
                    "val": split_counts["val"],
                    "test": split_counts["test"],
                    "val_fraction": cfg.val_fraction,
                    "test_fraction": cfg.test_fraction,
                },
                "matchup_lookup": {
                    "scope": "sql_ml_interaction_counts",
                    "train_matchids_hash": _matchids_hash(train_matchids),
                    "train_self_leakage": "leave_one_out",
                    "source_table": INTERACTION_COUNTS_TABLE,
                    "player_pivot_table": PLAYER_PIVOT_TABLE,
                },
                "normalization_scope": "train_only",
                "arrays": {
                    name: str(path.name)
                    for name, path in _array_paths(cfg.cache_dir).items()
                },
            },
            indent=2,
        )
    )

    logger.info(
        "Done. n_games=%d interaction_tokens=%d", n_written, N_INTERACTION_TOKENS
    )
    return meta_path


if __name__ == "__main__":
    build()


__all__ = [
    "ARRAY_FILES",
    "CACHE_META_FILE",
    "NORM_FILE",
    "VOCAB_FILE",
    "_array_paths",
    "build",
]
