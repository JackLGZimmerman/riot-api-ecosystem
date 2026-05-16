"""Build the training cache.

Streams one row per game from `game_data_filtered.ml_game_player_pivot`
and writes memory-mapped .npy arrays plus metadata.

Run with:
    python -m app.ml.build_dataset
"""

from __future__ import annotations

import json
import logging
import time
from hashlib import sha1
from pathlib import Path
from typing import cast

import numpy as np
from clickhouse_connect.driver.external import ExternalData

from app.core.logging.logger import setup_logging_config
from app.ml.cache_layout import (
    ARRAY_FILES,
    BLUE_WIN_FILE,
    CACHE_FORMAT,
    CACHE_META_FILE,
    DISK_ARRAY_DTYPES,
    OBSOLETE_ARRAY_FILES,
    PLAYER_CHAMPION_BUILD_IDX_FILE,
    VOCAB_FILE,
    array_paths,
)
from app.ml.config import (
    ITEM_VALUE_TABLE,
    PARTICIPANT_TABLE,
    PLAYER_PIVOT_TABLE,
    POSITIONS,
    SPLIT_TABLE,
    UNK_INDEX,
    DatasetConfig,
)
from app.ml.utils.cache_sparsity import (
    N_PLAYER_TOKENS,
    cache_sparsity_metadata,
    collect_player_sparsity_counts,
    empty_player_sparsity_counts,
    log_player_sparsity_summary,
    merge_player_sparsity_counts,
)
from database.clickhouse.client import get_client

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
        SELECT DISTINCT championid
        FROM {PARTICIPANT_TABLE}
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
    champ_idx = (
        int(champion_lut[cid]) if 0 <= cid < champion_lut.shape[0] else UNK_INDEX
    )
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


def _stream_into_arrays(
    cfg: DatasetConfig,
    vocab: dict,
    arrays: dict[str, np.ndarray],
    matchids: list[str],
    start_write_idx: int,
) -> tuple[int, dict[str, object]]:
    champion_lut = _build_champion_lut(vocab)
    role_map = vocab["role_to_idx"]
    build_map = vocab["build_to_idx"]
    n_builds = int(vocab["n_builds"])

    t0 = time.perf_counter()
    pivot_rows = list(_iter_query_rows(_player_pivot_query(), matchids))
    if not pivot_rows:
        return 0, empty_player_sparsity_counts()
    n = len(pivot_rows)
    champion_idx_chunk = np.zeros((n, N_PLAYER_TOKENS), dtype=DISK_ARRAY_DTYPES["player_champion_build_idx"])
    role_idx_chunk = np.zeros((n, N_PLAYER_TOKENS), dtype=np.uint8)
    build_idx_chunk = np.zeros((n, N_PLAYER_TOKENS), dtype=np.uint8)

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
        arrays["blue_win"][start_write_idx + i] = int(blue_win)

    player_sparsity_counts = collect_player_sparsity_counts(
        champion_idx_chunk,
        role_idx_chunk,
        build_idx_chunk,
    )
    arrays["player_champion_build_idx"][start_write_idx : start_write_idx + n] = (
        _pack_champion_build_idx(champion_idx_chunk, build_idx_chunk, n_builds)
    )

    logger.info(
        "Chunk filled in %.1fs (games=%d)",
        time.perf_counter() - t0,
        n,
    )
    return n, player_sparsity_counts


def _flush_arrays(arrays: dict[str, np.ndarray]) -> None:
    for arr in arrays.values():
        flush = getattr(arr, "flush", None)
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
            N_PLAYER_TOKENS * np.dtype(DISK_ARRAY_DTYPES["player_champion_build_idx"]).itemsize
            + np.dtype(DISK_ARRAY_DTYPES["blue_win"]).itemsize
        )
        / 1e6
    )
    logger.info("Allocating arrays for %d games (%.1f MB)", n_games, estimated_cache_mb)
    arrays = _allocate_arrays(n_games, cfg.cache_dir)

    n_written = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_player_sparsity_counts = {
        "train": empty_player_sparsity_counts(),
        "val": empty_player_sparsity_counts(),
        "test": empty_player_sparsity_counts(),
    }
    overall_player_sparsity_counts = empty_player_sparsity_counts()

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
            chunk_written, chunk_player_sparsity_counts = _stream_into_arrays(
                cfg, vocab, arrays, chunk, n_written
            )
            if chunk_written == 0:
                logger.warning("Stopping %s split after an empty chunk", split_name)
                break
            n_written += chunk_written
            split_counts[split_name] += chunk_written
            merge_player_sparsity_counts(
                split_player_sparsity_counts[split_name], chunk_player_sparsity_counts
            )
            merge_player_sparsity_counts(
                overall_player_sparsity_counts, chunk_player_sparsity_counts
            )
    logger.info("Wrote %d / %d games", n_written, n_games)

    for split_name in ("train", "val", "test"):
        log_player_sparsity_summary(
            logger, split_name, split_player_sparsity_counts[split_name]
        )
    log_player_sparsity_summary(logger, "overall", overall_player_sparsity_counts)

    if n_written < n_games:
        for k, arr in arrays.items():
            arrays[k] = arr[:n_written]

    _flush_arrays(arrays)

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
                "splits": {
                    "strategy": "sql_ml_game_split",
                    "train": split_counts["train"],
                    "val": split_counts["val"],
                    "test": split_counts["test"],
                    "val_fraction": cfg.val_fraction,
                    "test_fraction": cfg.test_fraction,
                },
                "train_matchids_hash": _matchids_hash(train_matchids),
                **cache_sparsity_metadata(
                    overall_player_sparsity_counts,
                    split_player_sparsity_counts,
                ),
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
    _remove_obsolete_cache_files(cfg.cache_dir)
    return meta_path


if __name__ == "__main__":
    build()


__all__ = [
    "ARRAY_FILES",
    "CACHE_FORMAT",
    "CACHE_META_FILE",
    "VOCAB_FILE",
    "build",
]
