"""Build the training cache.

Streams one row per game from `game_data_filtered.ml_game_player_pivot`
plus the long-form `ml_interaction_counts` table (token-level matchups +
primary-side wins, materialised by the 6901 SQL build with matchups >= 5
applied at the source). Per-chunk we apply leave-one-out for train games,
compute interaction scores, then write memory-mapped .npy arrays plus metadata.

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
from app.ml.cache_layout import (
    ARRAY_FILES,
    BLUE_WIN_FILE,
    CACHE_FORMAT,
    CACHE_META_FILE,
    DISK_ARRAY_DTYPES,
    INTERACTION_SCORE_FILE,
    NORM_FILE,
    OBSOLETE_ARRAY_FILES,
    PLAYER_CHAMPION_BUILD_IDX_FILE,
    VOCAB_FILE,
    array_paths,
)
from app.ml.config import (
    INTERACTION_COUNTS_TABLE,
    INTERACTION_ROLES,
    INTERACTION_SIDES,
    INTERACTION_TYPES,
    ITEM_VALUE_TABLE,
    MATCHUP_CONFIDENCE_Z,
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
_N_PLAYER_TOKENS = 10
_PLAYER_TOKEN_LAYOUT = tuple(
    {"token_idx": i, "side": "blue", "role": role}
    for i, role in enumerate(POSITIONS)
) + tuple(
    {"token_idx": i + len(POSITIONS), "side": "red", "role": role}
    for i, role in enumerate(POSITIONS)
)
_PLAYER_SPARSITY_KEYS = (
    "champion_unknown_by_token",
    "role_unknown_by_token",
    "build_unknown_by_token",
    "any_unknown_by_token",
)
_INTERACTION_SPARSITY_KEYS = (
    "source_present_by_token",
    "valid_support_by_token",
    "zero_score_by_token",
)


def _empty_token_counts(
    token_count: int,
    keys: tuple[str, ...],
) -> dict[str, object]:
    return {
        "games": 0,
        **{key: np.zeros(token_count, dtype=np.int64) for key in keys},
    }


def _merge_token_counts(
    target: dict[str, object],
    source: dict[str, object],
    keys: tuple[str, ...],
) -> None:
    target["games"] = int(target["games"]) + int(source["games"])
    for key in keys:
        target[key] = (
            np.asarray(target[key], dtype=np.int64)
            + np.asarray(source[key], dtype=np.int64)
        )


# --------------------------------------------------------------------------
# Player token sparsity diagnostics
# --------------------------------------------------------------------------


def _empty_player_sparsity_counts() -> dict[str, object]:
    return _empty_token_counts(_N_PLAYER_TOKENS, _PLAYER_SPARSITY_KEYS)


def _collect_player_sparsity_counts(
    champion_idx: np.ndarray,
    role_idx: np.ndarray,
    build_idx: np.ndarray,
) -> dict[str, object]:
    counts = _empty_player_sparsity_counts()
    champion_unknown = champion_idx == UNK_INDEX
    role_unknown = role_idx == UNK_INDEX
    build_unknown = build_idx == UNK_INDEX
    any_unknown = champion_unknown | role_unknown | build_unknown

    counts["games"] = int(champion_idx.shape[0])
    counts["champion_unknown_by_token"] = champion_unknown.sum(axis=0).astype(np.int64)
    counts["role_unknown_by_token"] = role_unknown.sum(axis=0).astype(np.int64)
    counts["build_unknown_by_token"] = build_unknown.sum(axis=0).astype(np.int64)
    counts["any_unknown_by_token"] = any_unknown.sum(axis=0).astype(np.int64)
    return counts


def _merge_player_sparsity_counts(
    target: dict[str, object],
    source: dict[str, object],
) -> None:
    _merge_token_counts(target, source, _PLAYER_SPARSITY_KEYS)


def _summarize_player_sparsity_counts(
    counts: dict[str, object],
) -> dict[str, object]:
    games = int(counts["games"])
    slots = games * _N_PLAYER_TOKENS
    champion_unknown = np.asarray(
        counts["champion_unknown_by_token"], dtype=np.int64
    )
    role_unknown = np.asarray(counts["role_unknown_by_token"], dtype=np.int64)
    build_unknown = np.asarray(counts["build_unknown_by_token"], dtype=np.int64)
    any_unknown = np.asarray(counts["any_unknown_by_token"], dtype=np.int64)

    champion_unknown_total = int(champion_unknown.sum())
    role_unknown_total = int(role_unknown.sum())
    build_unknown_total = int(build_unknown.sum())
    any_unknown_total = int(any_unknown.sum())

    return {
        "games": games,
        "slots": slots,
        "champion_unknown_slots": champion_unknown_total,
        "champion_unknown_frac": _rate(champion_unknown_total, slots),
        "role_unknown_slots": role_unknown_total,
        "role_unknown_frac": _rate(role_unknown_total, slots),
        "build_unknown_slots": build_unknown_total,
        "build_unknown_frac": _rate(build_unknown_total, slots),
        "any_unknown_slots": any_unknown_total,
        "any_unknown_frac": _rate(any_unknown_total, slots),
        "all_known_slots": slots - any_unknown_total,
        "all_known_frac": _rate(slots - any_unknown_total, slots),
        "by_token": [
            {
                **_PLAYER_TOKEN_LAYOUT[i],
                "champion_unknown": int(champion_unknown[i]),
                "champion_unknown_frac": _rate(int(champion_unknown[i]), games),
                "role_unknown": int(role_unknown[i]),
                "role_unknown_frac": _rate(int(role_unknown[i]), games),
                "build_unknown": int(build_unknown[i]),
                "build_unknown_frac": _rate(int(build_unknown[i]), games),
                "any_unknown": int(any_unknown[i]),
                "any_unknown_frac": _rate(int(any_unknown[i]), games),
                "all_known": games - int(any_unknown[i]),
                "all_known_frac": _rate(games - int(any_unknown[i]), games),
            }
            for i in range(_N_PLAYER_TOKENS)
        ],
    }


def _log_player_sparsity_summary(split_name: str, counts: dict[str, object]) -> None:
    summary = _summarize_player_sparsity_counts(counts)
    logger.info(
        (
            "Player token sparsity %s: games=%d champion_unk=%d/%d %.2f%% "
            "role_unk=%d/%d %.2f%% build_unk=%d/%d %.2f%% any_unk=%d/%d %.2f%%"
        ),
        split_name,
        summary["games"],
        summary["champion_unknown_slots"],
        summary["slots"],
        summary["champion_unknown_frac"] * 100.0,
        summary["role_unknown_slots"],
        summary["slots"],
        summary["role_unknown_frac"] * 100.0,
        summary["build_unknown_slots"],
        summary["slots"],
        summary["build_unknown_frac"] * 100.0,
        summary["any_unknown_slots"],
        summary["slots"],
        summary["any_unknown_frac"] * 100.0,
    )


# --------------------------------------------------------------------------
# Interaction sparsity diagnostics
# --------------------------------------------------------------------------


def _empty_sparsity_counts() -> dict[str, object]:
    return _empty_token_counts(N_INTERACTION_TOKENS, _INTERACTION_SPARSITY_KEYS)


def _support_valid_mask(
    matchups: np.ndarray,
    exclude_self: bool,
    min_matchups: int,
) -> np.ndarray:
    has_row = matchups > 0
    support = matchups.astype(np.int64)
    if exclude_self:
        support = np.where(has_row, support - _TOKEN_LOO_WEIGHT, 0)
        support = np.maximum(support, 0)
    return support >= max(1, int(min_matchups))


def _collect_sparsity_counts(
    matchups: np.ndarray,
    valid_support: np.ndarray,
    score: np.ndarray,
) -> dict[str, object]:
    counts = _empty_sparsity_counts()
    source_present = matchups > 0
    zero_score = score == np.float32(0.0)
    counts["games"] = int(matchups.shape[0])
    counts["source_present_by_token"] = source_present.sum(axis=0).astype(np.int64)
    counts["valid_support_by_token"] = valid_support.sum(axis=0).astype(np.int64)
    counts["zero_score_by_token"] = zero_score.sum(axis=0).astype(np.int64)
    return counts


def _merge_sparsity_counts(
    target: dict[str, object],
    source: dict[str, object],
) -> None:
    _merge_token_counts(target, source, _INTERACTION_SPARSITY_KEYS)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _summarize_sparsity_counts(counts: dict[str, object]) -> dict[str, object]:
    games = int(counts["games"])
    slots = games * N_INTERACTION_TOKENS
    source_present = np.asarray(counts["source_present_by_token"], dtype=np.int64)
    valid_support = np.asarray(counts["valid_support_by_token"], dtype=np.int64)
    zero_score = np.asarray(counts["zero_score_by_token"], dtype=np.int64)

    present_total = int(source_present.sum())
    valid_total = int(valid_support.sum())
    zero_total = int(zero_score.sum())
    missing_total = slots - present_total
    support_filtered_total = present_total - valid_total

    return {
        "games": games,
        "slots": slots,
        "source_present_slots": present_total,
        "source_present_frac": _rate(present_total, slots),
        "source_missing_slots": missing_total,
        "source_missing_frac": _rate(missing_total, slots),
        "valid_support_slots": valid_total,
        "valid_support_frac": _rate(valid_total, slots),
        "support_filtered_slots": support_filtered_total,
        "support_filtered_frac": _rate(support_filtered_total, slots),
        "zero_score_slots": zero_total,
        "zero_score_frac": _rate(zero_total, slots),
        "nonzero_score_slots": slots - zero_total,
        "nonzero_score_frac": _rate(slots - zero_total, slots),
        "by_token": [
            {
                "token_idx": i,
                "token_type": int(INTERACTION_TYPES[i]),
                "side": int(INTERACTION_SIDES[i]),
                "role_slots": [int(role) for role in INTERACTION_ROLES[i]],
                "source_present": int(source_present[i]),
                "source_present_frac": _rate(int(source_present[i]), games),
                "source_missing": games - int(source_present[i]),
                "source_missing_frac": _rate(games - int(source_present[i]), games),
                "valid_support": int(valid_support[i]),
                "valid_support_frac": _rate(int(valid_support[i]), games),
                "support_filtered": int(source_present[i] - valid_support[i]),
                "support_filtered_frac": _rate(
                    int(source_present[i] - valid_support[i]), games
                ),
                "zero_score": int(zero_score[i]),
                "zero_score_frac": _rate(int(zero_score[i]), games),
                "nonzero_score": games - int(zero_score[i]),
                "nonzero_score_frac": _rate(games - int(zero_score[i]), games),
            }
            for i in range(N_INTERACTION_TOKENS)
        ],
    }


def _log_sparsity_summary(split_name: str, counts: dict[str, object]) -> None:
    summary = _summarize_sparsity_counts(counts)
    logger.info(
        (
            "Interaction sparsity %s: games=%d missing=%d/%d %.2f%% "
            "valid_support=%d/%d %.2f%% zero_score=%d/%d %.2f%%"
        ),
        split_name,
        summary["games"],
        summary["source_missing_slots"],
        summary["slots"],
        summary["source_missing_frac"] * 100.0,
        summary["valid_support_slots"],
        summary["slots"],
        summary["valid_support_frac"] * 100.0,
        summary["zero_score_slots"],
        summary["slots"],
        summary["zero_score_frac"] * 100.0,
    )


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
        by_split["train"] = by_split["train"][:n_train]
        by_split["validation"] = by_split["validation"][:n_val]
        by_split["test"] = by_split["test"][:n_test]

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


def _max_dtype_value(dtype: np.dtype | type[np.generic]) -> int:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return int(np.iinfo(dtype).max)
    raise TypeError(f"Expected integer dtype, got {dtype}")


def _validate_compact_dtype_bounds(vocab: dict[str, object]) -> None:
    n_builds = int(vocab["n_builds"])
    bounds = {
        "player_champion_build_idx": (int(vocab["n_champions"]) - 1) * n_builds
        + int(vocab["n_builds"])
        - 1,
    }
    for name, max_index in bounds.items():
        dtype = DISK_ARRAY_DTYPES[name]
        dtype_max = _max_dtype_value(dtype)
        if max_index > dtype_max:
            raise ValueError(
                f"{name} max index {max_index} exceeds cache dtype "
                f"{np.dtype(dtype).name} max {dtype_max}"
            )


# --------------------------------------------------------------------------
# Cache layout
# --------------------------------------------------------------------------


def _array_paths(cache_dir: Path) -> dict[str, Path]:
    return array_paths(cache_dir)


def _allocate_arrays(n_games: int, cache_dir: Path) -> dict[str, np.ndarray]:
    return {
        "interaction_score": np.lib.format.open_memmap(
            cache_dir / INTERACTION_SCORE_FILE,
            mode="w+",
            dtype=DISK_ARRAY_DTYPES["interaction_score"],
            shape=(n_games, N_INTERACTION_TOKENS),
        ),
        "player_champion_build_idx": np.lib.format.open_memmap(
            cache_dir / PLAYER_CHAMPION_BUILD_IDX_FILE,
            mode="w+",
            dtype=DISK_ARRAY_DTYPES["player_champion_build_idx"],
            shape=(n_games, 10),
        ),
        "blue_win": np.lib.format.open_memmap(
            cache_dir / BLUE_WIN_FILE,
            mode="w+",
            dtype=DISK_ARRAY_DTYPES["blue_win"],
            shape=(n_games,),
        ),
    }


# --------------------------------------------------------------------------
# Vectorised LOO + scoring
# --------------------------------------------------------------------------


def _centered_win_rate(
    wins: np.ndarray,
    support: np.ndarray,
    smooth: bool,
) -> np.ndarray:
    wins_f = wins.astype(np.float32)
    win_rate = np.zeros_like(support, dtype=np.float32)
    np.divide(wins_f, support, out=win_rate, where=support > 0)
    if not smooth:
        return (win_rate - 0.5).astype(np.float32)

    z = np.float32(MATCHUP_CONFIDENCE_Z)
    safe_support = np.maximum(support, np.float32(1.0))
    z2 = z * z
    denominator = np.float32(1.0) + z2 / safe_support
    center = (win_rate + z2 / (np.float32(2.0) * safe_support)) / denominator
    variance = (
        win_rate * (np.float32(1.0) - win_rate)
        + z2 / (np.float32(4.0) * safe_support)
    ) / safe_support
    margin = (
        z
        * np.sqrt(variance)
        / denominator
    )
    lower = center - margin
    upper = center + margin

    positive_score = np.maximum(lower - np.float32(0.5), np.float32(0.0))
    negative_score = np.minimum(upper - np.float32(0.5), np.float32(0.0))
    return np.where(win_rate >= np.float32(0.5), positive_score, negative_score).astype(
        np.float32
    )


def _score_interactions(
    matchups: np.ndarray,
    primary_wins: np.ndarray,
    blue_wins: np.ndarray,
    exclude_self: bool,
    min_matchups: int,
    smooth_scores: bool,
) -> np.ndarray:
    """Apply LOO for train chunks and compute per-token interaction scores.

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
    centered_win_rate = _centered_win_rate(w, support, smooth_scores)
    score = np.where(
        valid,
        centered_win_rate,
        np.float32(0.0),
    ).astype(np.float32)
    return score


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


def _pack_champion_build_idx(
    champion_idx: np.ndarray,
    build_idx: np.ndarray,
    n_builds: int,
) -> np.ndarray:
    packed = champion_idx.astype(np.uint32) * np.uint32(n_builds)
    packed += build_idx.astype(np.uint32)
    return packed.astype(DISK_ARRAY_DTYPES["player_champion_build_idx"])


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
) -> tuple[int, dict[str, object], dict[str, object]]:
    champion_lut = _build_champion_lut(vocab)
    role_map = vocab["role_to_idx"]
    build_map = vocab["build_to_idx"]
    n_builds = int(vocab["n_builds"])
    min_matchups = max(1, int(cfg.min_matchup_count))

    t0 = time.perf_counter()
    pivot_rows = list(_iter_query_rows(_player_pivot_query(), matchids))
    if not pivot_rows:
        return 0, _empty_sparsity_counts(), _empty_player_sparsity_counts()
    n = len(pivot_rows)
    matchid_to_idx = {str(row[0]): i for i, row in enumerate(pivot_rows)}
    blue_wins = np.zeros(n, dtype=DISK_ARRAY_DTYPES["blue_win"])
    champion_idx_chunk = np.zeros(
        (n, _N_PLAYER_TOKENS),
        dtype=DISK_ARRAY_DTYPES["player_champion_build_idx"],
    )
    role_idx_chunk = np.zeros(
        (n, _N_PLAYER_TOKENS), dtype=np.uint8,
    )
    build_idx_chunk = np.zeros(
        (n, _N_PLAYER_TOKENS), dtype=np.uint8,
    )

    for i, (_matchid, blue_win, blue_players, red_players) in enumerate(pivot_rows):
        for slot, (cid, pos, build) in enumerate(blue_players):
            c, r, b = _resolve_player_idx(
                int(cid), str(pos), str(build), champion_lut, role_map, build_map
            )
            champion_idx_chunk[i, slot] = c
            role_idx_chunk[i, slot] = r
            build_idx_chunk[i, slot] = b
        for slot, (cid, pos, build) in enumerate(red_players):
            c, r, b = _resolve_player_idx(
                int(cid), str(pos), str(build), champion_lut, role_map, build_map
            )
            chunk_slot = len(POSITIONS) + slot
            champion_idx_chunk[i, chunk_slot] = c
            role_idx_chunk[i, chunk_slot] = r
            build_idx_chunk[i, chunk_slot] = b
        blue_wins[i] = int(blue_win)
        arrays["blue_win"][start_write_idx + i] = int(blue_win)

    player_sparsity_counts = _collect_player_sparsity_counts(
        champion_idx_chunk,
        role_idx_chunk,
        build_idx_chunk,
    )
    arrays["player_champion_build_idx"][start_write_idx:start_write_idx + n] = (
        _pack_champion_build_idx(champion_idx_chunk, build_idx_chunk, n_builds)
    )

    matchups_chunk = np.zeros((n, N_INTERACTION_TOKENS), dtype=np.uint32)
    primary_wins_chunk = np.zeros((n, N_INTERACTION_TOKENS), dtype=np.uint32)
    n_token_rows = 0
    for matchid, token_idx, matchups, primary_wins in _iter_query_rows(
        _interaction_counts_query(), matchids
    ):
        i = matchid_to_idx.get(str(matchid))
        tidx = int(token_idx)
        if i is None or tidx >= N_INTERACTION_TOKENS:
            continue
        matchups_chunk[i, tidx] = int(matchups)
        primary_wins_chunk[i, tidx] = int(primary_wins)
        n_token_rows += 1

    score = _score_interactions(
        matchups_chunk,
        primary_wins_chunk,
        blue_wins,
        exclude_self,
        min_matchups,
        cfg.smooth_interaction_scores,
    )
    valid_support = _support_valid_mask(matchups_chunk, exclude_self, min_matchups)
    sparsity_counts = _collect_sparsity_counts(
        matchups_chunk, valid_support, score
    )
    arrays["interaction_score"][start_write_idx:start_write_idx + n] = score

    logger.info(
        "Chunk filled in %.1fs (games=%d token_rows=%d exclude_self=%s)",
        time.perf_counter() - t0, n, n_token_rows, exclude_self,
    )
    return n, sparsity_counts, player_sparsity_counts


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


def _remove_obsolete_cache_files(cache_dir: Path) -> None:
    for filename in OBSOLETE_ARRAY_FILES:
        path = cache_dir / filename
        if path.exists():
            path.unlink()


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
    _validate_compact_dtype_bounds(vocab)

    logger.info("Selecting split matchids from %s", SPLIT_TABLE)
    logger.info(
        "Interaction score smoothing: %s",
        "on" if cfg.smooth_interaction_scores else "off",
    )
    train_matchids, val_matchids, test_matchids = _sql_split_matchids(cfg)
    n_games = len(train_matchids) + len(val_matchids) + len(test_matchids)
    logger.info(
        "Selected games: train=%d val=%d test=%d",
        len(train_matchids),
        len(val_matchids),
        len(test_matchids),
    )

    estimated_cache_mb = (
        n_games
        * (
            N_INTERACTION_TOKENS
            * np.dtype(DISK_ARRAY_DTYPES["interaction_score"]).itemsize
            + _N_PLAYER_TOKENS
            * np.dtype(DISK_ARRAY_DTYPES["player_champion_build_idx"]).itemsize
            + np.dtype(DISK_ARRAY_DTYPES["blue_win"]).itemsize
        )
        / 1e6
    )
    logger.info(
        "Allocating compact arrays for %d games x %d interaction tokens (%.1f MB)",
        n_games,
        N_INTERACTION_TOKENS,
        estimated_cache_mb,
    )
    arrays = _allocate_arrays(n_games, cfg.cache_dir)

    n_written = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_sparsity_counts = {
        "train": _empty_sparsity_counts(),
        "val": _empty_sparsity_counts(),
        "test": _empty_sparsity_counts(),
    }
    overall_sparsity_counts = _empty_sparsity_counts()
    split_player_sparsity_counts = {
        "train": _empty_player_sparsity_counts(),
        "val": _empty_player_sparsity_counts(),
        "test": _empty_player_sparsity_counts(),
    }
    overall_player_sparsity_counts = _empty_player_sparsity_counts()
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
            (
                chunk_written,
                chunk_sparsity_counts,
                chunk_player_sparsity_counts,
            ) = _stream_into_arrays(cfg, vocab, arrays, chunk, n_written, exclude_self)
            if chunk_written == 0:
                logger.warning("Stopping %s split after an empty chunk", split_name)
                break
            n_written += chunk_written
            split_counts[split_name] += chunk_written
            _merge_sparsity_counts(
                split_sparsity_counts[split_name], chunk_sparsity_counts
            )
            _merge_sparsity_counts(overall_sparsity_counts, chunk_sparsity_counts)
            _merge_player_sparsity_counts(
                split_player_sparsity_counts[split_name],
                chunk_player_sparsity_counts,
            )
            _merge_player_sparsity_counts(
                overall_player_sparsity_counts,
                chunk_player_sparsity_counts,
            )
    logger.info("Wrote %d / %d games", n_written, n_games)

    for split_name in ("train", "val", "test"):
        _log_player_sparsity_summary(
            split_name, split_player_sparsity_counts[split_name]
        )
        _log_sparsity_summary(split_name, split_sparsity_counts[split_name])
    _log_player_sparsity_summary("overall", overall_player_sparsity_counts)
    _log_sparsity_summary("overall", overall_sparsity_counts)

    if n_written < n_games:
        for k, arr in arrays.items():
            arrays[k] = arr[:n_written]

    train_games = split_counts["train"]
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
                "format": CACHE_FORMAT,
                "n_games": n_written,
                "n_interaction_tokens": N_INTERACTION_TOKENS,
                "n_role_slots": N_ROLE_SLOTS,
                "min_matchup_count": max(1, int(cfg.min_matchup_count)),
                "smooth_interaction_scores": cfg.smooth_interaction_scores,
                "score_smoothing": {
                    "enabled": cfg.smooth_interaction_scores,
                    "method": "wilson_score_confidence_bound",
                    "confidence_z": MATCHUP_CONFIDENCE_Z,
                },
                "score_normalized": True,
                "player_encoding": {
                    "champion_build": "packed_uint16_champion_idx_times_n_builds_plus_build_idx",
                    "role_idx": "implied_by_player_slot",
                },
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
                "player_token_sparsity": {
                    "scope": "player_embedding_indices",
                    "notes": [
                        "champion_unknown, role_unknown, and build_unknown count "
                        "player-token embedding ids equal to UNK_INDEX / 0.",
                        "any_unknown counts player slots where at least one of "
                        "champion_idx, role_idx, or build_idx is 0.",
                    ],
                    "overall": _summarize_player_sparsity_counts(
                        overall_player_sparsity_counts
                    ),
                    "splits": {
                        split_name: _summarize_player_sparsity_counts(
                            split_player_sparsity_counts[split_name]
                        )
                        for split_name in ("train", "val", "test")
                    },
                },
                "interaction_sparsity": {
                    "scope": "pre_normalization_scores",
                    "notes": [
                        "source_missing counts token slots absent from "
                        "ml_interaction_counts; these are instantiated as 0.0 "
                        "before normalization.",
                        "support_filtered counts source rows that fail "
                        "min_matchup_count after train leave-one-out.",
                        "zero_score also includes valid neutral 50% scores and "
                        "Wilson-smoothed scores shrunk to 0.0.",
                    ],
                    "overall": _summarize_sparsity_counts(
                        overall_sparsity_counts
                    ),
                    "splits": {
                        split_name: _summarize_sparsity_counts(
                            split_sparsity_counts[split_name]
                        )
                        for split_name in ("train", "val", "test")
                    },
                },
                "normalization_scope": "train_only",
                "arrays": {
                    name: str(path.name)
                    for name, path in _array_paths(cfg.cache_dir).items()
                },
                "array_dtypes": {
                    name: np.dtype(dtype).name
                    for name, dtype in DISK_ARRAY_DTYPES.items()
                },
            },
            indent=2,
        )
    )

    logger.info(
        "Done. n_games=%d interaction_tokens=%d", n_written, N_INTERACTION_TOKENS
    )
    _remove_obsolete_cache_files(cfg.cache_dir)
    return meta_path


if __name__ == "__main__":
    build()


__all__ = [
    "ARRAY_FILES",
    "CACHE_FORMAT",
    "CACHE_META_FILE",
    "NORM_FILE",
    "VOCAB_FILE",
    "_array_paths",
    "build",
]
