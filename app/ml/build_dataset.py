"""Build the training cache.

Streams one row per game from `game_data_filtered.ml_game_player_pivot`,
joins 6002's `game_data_filtered.synergy_1vx` per-player profile features,
and writes memory-mapped .npy arrays plus metadata.

Run with:
    python -m app.ml.build_dataset
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from clickhouse_connect.driver.external import ExternalData

from app.core.logging.logger import setup_logging_config
from app.ml.cache_layout import (
    BLUE_WIN_FILE,
    CACHE_FORMAT,
    CACHE_META_FILE,
    DISK_ARRAY_DTYPES,
    N_PROFILE_BINS,
    N_PROFILE_FEATURES,
    PLAYER_CHAMPION_BUILD_IDX_FILE,
    PLAYER_PROFILE_FILE,
    PROFILE_FEATURE_COLUMNS,
    VOCAB_FILE,
    array_paths,
)
from app.ml.config import (
    CHAMPION_ID_NAME_DICT,
    ITEM_VALUE_MAP_DICT,
    PLAYER_PIVOT_TABLE,
    POSITIONS,
    SPLIT_TABLE,
    SYNERGY_1VX_TABLE,
    DatasetConfig,
)
from database.clickhouse.client import get_client

N_PLAYER_TOKENS = 10
_ITEM_VALUE_KEY_COLUMNS = frozenset({"championid", "teamposition", "itemid"})
_EMPTY_BUILD_LABEL = "none"

setup_logging_config()
logger = logging.getLogger(__name__)

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


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


def _build_vocabularies() -> dict[str, object]:
    client = get_client()
    champions = client.query(
        f"""
        SELECT _key AS championid
        FROM {CHAMPION_ID_NAME_DICT}
        WHERE _key > 0
        ORDER BY _key
        """
    ).result_rows
    item_value_db, item_value_table = ITEM_VALUE_MAP_DICT.split(".", maxsplit=1)
    key_column_sql = ", ".join(f"'{name}'" for name in sorted(_ITEM_VALUE_KEY_COLUMNS))
    builds = client.query(
        f"""
        SELECT name
        FROM system.columns
        WHERE
            database = '{item_value_db}'
            AND table = '{item_value_table}'
            AND startsWith(type, 'Float')
            AND name NOT IN ({key_column_sql})
        ORDER BY position
        """
    ).result_rows

    champion_to_idx = {int(c[0]): i for i, c in enumerate(champions)}
    build_labels = [_EMPTY_BUILD_LABEL, *(str(b[0]) for b in builds)]
    build_to_idx = {label: i for i, label in enumerate(build_labels)}
    role_to_idx = {p: i for i, p in enumerate(POSITIONS)}

    return {
        "champion_to_idx": champion_to_idx,
        "build_to_idx": build_to_idx,
        "role_to_idx": role_to_idx,
        "champion_vocab_source": CHAMPION_ID_NAME_DICT,
        "build_vocab_source": f"{ITEM_VALUE_MAP_DICT} value columns",
        "n_champions": len(champion_to_idx),
        "n_builds": len(build_to_idx),
        "n_roles": len(role_to_idx),
    }


def _max_dtype_value(dtype: np.dtype | type[np.generic]) -> int:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return int(np.iinfo(dtype).max)
    raise TypeError(f"Expected integer dtype, got {dtype}")


def _validate_compact_dtype_bounds(vocab: dict[str, object]) -> None:
    n_builds = int(cast(int, vocab["n_builds"]))
    n_champions = int(cast(int, vocab["n_champions"]))
    bounds = {
        "player_champion_build_idx": (n_champions - 1) * n_builds + n_builds - 1,
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


def _allocate_arrays(n_games: int, cache_dir: Path) -> dict[str, np.ndarray]:
    return {
        "player_champion_build_idx": np.lib.format.open_memmap(
            cache_dir / PLAYER_CHAMPION_BUILD_IDX_FILE,
            mode="w+",
            dtype=DISK_ARRAY_DTYPES["player_champion_build_idx"],
            shape=(n_games, N_PLAYER_TOKENS),
        ),
        "player_profile": np.lib.format.open_memmap(
            cache_dir / PLAYER_PROFILE_FILE,
            mode="w+",
            dtype=DISK_ARRAY_DTYPES["player_profile"],
            shape=(n_games, N_PLAYER_TOKENS, N_PROFILE_BINS, N_PROFILE_FEATURES),
        ),
        "blue_win": np.lib.format.open_memmap(
            cache_dir / BLUE_WIN_FILE,
            mode="w+",
            dtype=DISK_ARRAY_DTYPES["blue_win"],
            shape=(n_games,),
        ),
    }


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
) -> tuple[int, int]:
    if not (0 <= cid < champion_lut.shape[0] and champion_lut[cid] >= 0):
        raise ValueError(f"Unknown champion id {cid}")
    if pos not in role_map:
        raise ValueError(f"Unknown role {pos!r}")
    if build not in build_map:
        raise ValueError(f"Unknown build {build!r}")
    return int(champion_lut[cid]), build_map[build]


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
    lut = np.full(max_id + 1, -1, dtype=np.int32)
    for k, v in keys.items():
        lut[k] = v
    return lut


def _player_pivot_query() -> str:
    profile_columns = ",\n        ".join(PROFILE_FEATURE_COLUMNS)
    profile_tuple_fields = ", ".join(
        f"s.{column}" for column in PROFILE_FEATURE_COLUMNS
    )
    return f"""
    WITH
    expanded AS (
        SELECT
            p.matchid,
            p.blue_win,
            p.blue_players,
            p.red_players,
            toUInt16(tupleElement(token, 2)) AS token_idx,
            tupleElement(tupleElement(token, 1), 1) AS championid,
            tupleElement(tupleElement(token, 1), 2) AS teamposition,
            tupleElement(tupleElement(token, 1), 3) AS build
        FROM {PLAYER_PIVOT_TABLE} AS p
        INNER JOIN selected_matchids AS sg ON p.matchid = sg.matchid
        ARRAY JOIN [
            tuple(p.blue_players[1], toUInt16(0)),
            tuple(p.blue_players[2], toUInt16(1)),
            tuple(p.blue_players[3], toUInt16(2)),
            tuple(p.blue_players[4], toUInt16(3)),
            tuple(p.blue_players[5], toUInt16(4)),
            tuple(p.red_players[1], toUInt16(5)),
            tuple(p.red_players[2], toUInt16(6)),
            tuple(p.red_players[3], toUInt16(7)),
            tuple(p.red_players[4], toUInt16(8)),
            tuple(p.red_players[5], toUInt16(9))
        ] AS token
    )
    SELECT
        e.matchid,
        any(e.blue_win) AS blue_win,
        any(e.blue_players) AS blue_players,
        any(e.red_players) AS red_players,
        groupArrayIf(
            (
                e.token_idx,
                s.bin_idx,
                {profile_tuple_fields}
            ),
            s.bin_idx > 0
        ) AS profile_rows
    FROM expanded AS e
    ALL LEFT JOIN (
        SELECT
            championid,
            teamposition,
            build,
            bin_idx,
            {profile_columns}
        FROM {SYNERGY_1VX_TABLE}
        WHERE split = 'train'
    ) AS s
        ON
            s.championid = e.championid
            AND s.teamposition = e.teamposition
            AND s.build = e.build
    GROUP BY e.matchid
    SETTINGS join_algorithm = 'hash'
    """


def _profile_rows_to_array(profile_rows: object) -> np.ndarray:
    profile = np.zeros(
        (N_PLAYER_TOKENS, N_PROFILE_BINS, N_PROFILE_FEATURES),
        dtype=DISK_ARRAY_DTYPES["player_profile"],
    )
    if not isinstance(profile_rows, Sequence) or isinstance(profile_rows, (str, bytes)):
        return profile

    for row in profile_rows:
        if not isinstance(row, Sequence):
            raise TypeError(f"Unexpected 1vX profile row type: {type(row).__name__}")
        if len(row) != 2 + N_PROFILE_FEATURES:
            raise ValueError(
                "Unexpected 1vX profile row shape from "
                f"{SYNERGY_1VX_TABLE}: {len(row)}"
            )
        token_idx = int(row[0])
        bin_idx = int(row[1])
        if not (0 <= token_idx < N_PLAYER_TOKENS and 1 <= bin_idx <= N_PROFILE_BINS):
            continue
        profile[token_idx, bin_idx - 1] = np.asarray(
            row[2:],
            dtype=DISK_ARRAY_DTYPES["player_profile"],
        )
    return profile


def _stream_into_arrays(
    vocab: dict,
    arrays: dict[str, np.ndarray],
    matchids: list[str],
    start_write_idx: int,
) -> int:
    champion_lut = _build_champion_lut(vocab)
    role_map = vocab["role_to_idx"]
    build_map = vocab["build_to_idx"]
    n_builds = int(vocab["n_builds"])

    t0 = time.perf_counter()
    pivot_rows = list(_iter_query_rows(_player_pivot_query(), matchids))
    if not pivot_rows:
        return 0
    n = len(pivot_rows)
    champion_idx_chunk = np.zeros(
        (n, N_PLAYER_TOKENS), dtype=DISK_ARRAY_DTYPES["player_champion_build_idx"]
    )
    build_idx_chunk = np.zeros((n, N_PLAYER_TOKENS), dtype=np.uint8)

    for i, (_matchid, blue_win, blue_players, red_players, profile_rows) in enumerate(
        pivot_rows
    ):
        for slot, (cid, pos, build) in enumerate(blue_players):
            c, b = _resolve_player_idx(
                int(cid), str(pos), str(build), champion_lut, role_map, build_map
            )
            champion_idx_chunk[i, slot] = c
            build_idx_chunk[i, slot] = b
        for slot, (cid, pos, build) in enumerate(red_players):
            c, b = _resolve_player_idx(
                int(cid), str(pos), str(build), champion_lut, role_map, build_map
            )
            chunk_slot = len(POSITIONS) + slot
            champion_idx_chunk[i, chunk_slot] = c
            build_idx_chunk[i, chunk_slot] = b
        arrays["player_profile"][start_write_idx + i] = _profile_rows_to_array(
            profile_rows
        )
        arrays["blue_win"][start_write_idx + i] = int(blue_win)

    arrays["player_champion_build_idx"][start_write_idx : start_write_idx + n] = (
        _pack_champion_build_idx(champion_idx_chunk, build_idx_chunk, n_builds)
    )

    logger.info(
        "Chunk filled in %.1fs (games=%d)",
        time.perf_counter() - t0,
        n,
    )
    return n


def _profile_standardization(
    profile: np.ndarray,
    n_train: int,
    chunk: int = 100_000,
) -> dict[str, list[float]]:
    """Per-feature mean/std over present train profile rows.

    A bin row is present iff any feature is non-zero (matches the model's
    presence mask); zero-filled missing rows are excluded so sparsity does
    not pull the stats toward zero. Accumulated in chunks to bound memory.
    """
    s1 = np.zeros(N_PROFILE_FEATURES, dtype=np.float64)
    s2 = np.zeros(N_PROFILE_FEATURES, dtype=np.float64)
    count = 0
    for start in range(0, n_train, chunk):
        block = np.asarray(
            profile[start : min(start + chunk, n_train)], dtype=np.float64
        ).reshape(-1, N_PROFILE_FEATURES)
        rows = block[np.abs(block).sum(axis=1) > 0]
        count += rows.shape[0]
        s1 += rows.sum(axis=0)
        s2 += np.square(rows).sum(axis=0)
    if count == 0:
        raise ValueError("No present train profile rows for standardization")
    mean = s1 / count
    std = np.sqrt(np.maximum(s2 / count - np.square(mean), 0.0))
    std[std < 1e-6] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def _flush_arrays(arrays: dict[str, np.ndarray]) -> None:
    for arr in arrays.values():
        flush = getattr(arr, "flush", None)
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
    _validate_compact_dtype_bounds(vocab)

    logger.info("Selecting split matchids from %s", SPLIT_TABLE)
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
            N_PLAYER_TOKENS
            * np.dtype(DISK_ARRAY_DTYPES["player_champion_build_idx"]).itemsize
            + N_PLAYER_TOKENS
            * N_PROFILE_BINS
            * N_PROFILE_FEATURES
            * np.dtype(DISK_ARRAY_DTYPES["player_profile"]).itemsize
            + np.dtype(DISK_ARRAY_DTYPES["blue_win"]).itemsize
        )
        / 1e6
    )
    logger.info("Allocating arrays for %d games (%.1f MB)", n_games, estimated_cache_mb)
    arrays = _allocate_arrays(n_games, cfg.cache_dir)

    n_written = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    for split_name, split_matchids in (
        ("train", train_matchids),
        ("val", val_matchids),
        ("test", test_matchids),
    ):
        for split_offset in range(0, len(split_matchids), cfg.build_chunk_games):
            chunk = split_matchids[split_offset : split_offset + cfg.build_chunk_games]
            logger.info(
                "Streaming %s game chunk offset=%d limit=%d",
                split_name,
                split_offset,
                len(chunk),
            )
            chunk_written = _stream_into_arrays(vocab, arrays, chunk, n_written)
            if chunk_written == 0:
                logger.warning("Stopping %s split after an empty chunk", split_name)
                break
            if chunk_written != len(chunk):
                raise RuntimeError(
                    "Short cache chunk while streaming "
                    f"{split_name}: expected {len(chunk)} games, "
                    f"wrote {chunk_written}. Rebuild aborted so cache metadata "
                    "cannot silently drift from SQL split counts."
                )
            n_written += chunk_written
            split_counts[split_name] += chunk_written
    logger.info("Wrote %d / %d games", n_written, n_games)

    if n_written < n_games:
        for k, arr in arrays.items():
            arrays[k] = arr[:n_written]

    _flush_arrays(arrays)

    logger.info("Computing train profile standardization")
    profile_standardization = _profile_standardization(
        arrays["player_profile"], split_counts["train"]
    )

    vocab_path = cfg.cache_dir / VOCAB_FILE
    meta_path = cfg.cache_dir / CACHE_META_FILE

    vocab_path.write_text(json.dumps(vocab, indent=2))
    meta_path.write_text(
        json.dumps(
            {
                "format": CACHE_FORMAT,
                "n_games": n_written,
                "player_encoding": {
                    "champion_build": "packed_uint16_champion_idx_times_n_builds_plus_build_idx",
                    "role_idx": "implied_by_player_slot",
                },
                "player_profile": {
                    "source_table": SYNERGY_1VX_TABLE,
                    "source_schema": "database/clickhouse/schema/6002_1vx_aggregations_schema.sql",
                    "shape": [
                        "n_games",
                        N_PLAYER_TOKENS,
                        N_PROFILE_BINS,
                        N_PROFILE_FEATURES,
                    ],
                    "bin_idx": "1..4 in source, stored as axis indices 0..3",
                    "feature_columns": list(PROFILE_FEATURE_COLUMNS),
                    "missing_profile_rows": "zero_filled",
                    "grain": "per match, player token, and scaling bin; not pairwise interactions",
                },
                "profile_standardization": {
                    "scope": "per-feature mean/std over present train profile rows",
                    **profile_standardization,
                },
                "splits": {
                    "strategy": "sql_ml_game_split",
                    "train": split_counts["train"],
                    "val": split_counts["val"],
                    "test": split_counts["test"],
                    "val_fraction": cfg.val_fraction,
                    "test_fraction": cfg.test_fraction,
                },
                "arrays": {
                    name: str(path.name)
                    for name, path in array_paths(cfg.cache_dir).items()
                },
                "array_dtypes": {
                    name: np.dtype(dtype).name
                    for name, dtype in DISK_ARRAY_DTYPES.items()
                },
            },
            indent=2,
        )
    )

    logger.info("Done. n_games=%d", n_written)
    return meta_path


if __name__ == "__main__":
    build()
