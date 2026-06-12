"""Keystone-conditioned build prior artifacts for the pregame marginal eval path.

Provides train-only aggregation of (championid, teamposition, primary_perk_1,
build) counts and per-split per-slot keystone arrays that mirror the cache row
order. Used by marginal_eval --condition keystone to reweight build priors with
P(build | champ, role, keystone) instead of P(build | champ, role).

Leakage contract:
- All aggregations filter split = 'train' via game_data_filtered.ml_game_split.
- Outcome columns (win, blue_win) are never read in this module.
- puuid appears only as a join predicate for row alignment; it is never
  persisted in any artifact.

CLI:
    python -m app.ml.loadout_condition build --split train
    python -m app.ml.loadout_condition build --split test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from app.core.logging.logger import setup_logging_config
from app.ml.config import ML_DATA_DIR, PLAYER_PIVOT_TABLE, POSITIONS, DatasetConfig

setup_logging_config()
logger = logging.getLogger(__name__)

# Artifact paths
_CONDITION_DIR = ML_DATA_DIR / "marginal_condition"
KEYSTONE_COUNTS_PATH = _CONDITION_DIR / "keystone_counts.json"

_CHUNK_SIZE = 50_000
_CH_SETTINGS = {"max_memory_usage": 4_000_000_000}

# Slot order: blue team 100 (slots 0-4) TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY,
# then red team 200 (slots 5-9) same order.
_SLOT_POSITIONS = POSITIONS + POSITIONS  # 10 entries
_SLOT_TEAMIDS = (100,) * 5 + (200,) * 5


def slot_array_path(split: str) -> Path:
    return _CONDITION_DIR / f"{split}_keystone.npy"


def _slot_meta_path(split: str) -> Path:
    return _CONDITION_DIR / f"{split}_keystone_meta.json"


def _sha1_bytes(arr: np.ndarray) -> str:
    return hashlib.sha1(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Keystone counts artifact
# ---------------------------------------------------------------------------


# In-memory contract: counts bucketed per (championid, teamposition, keystone)
# cell, each cell mapping build label -> n. O(1) cell lookup during eval.
KeystoneCounts = dict[tuple[int, str, int], dict[str, int]]


def build_keystone_counts(client) -> KeystoneCounts:
    """Query train-only (champ, role, keystone, build) counts from ClickHouse."""
    query = """
SELECT
    toInt32(ifNull(ps.championid, 0)) AS championid,
    toString(ps.teamposition) AS teamposition,
    toInt32(pki.primary_perk_1) AS primary_perk_1,
    toString(ivt.highest_value_label) AS highest_value_label,
    count() AS n
FROM game_data_filtered.participant_stats AS ps
INNER JOIN game_data_filtered.ml_game_split AS sp
    ON ps.matchid = sp.matchid AND sp.split = 'train'
INNER JOIN game_data.participant_perk_ids AS pki
    ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
INNER JOIN game_data_filtered.participant_item_value_totals AS ivt
    ON ps.matchid = ivt.matchid AND ps.participantid = ivt.participantid
GROUP BY championid, teamposition, primary_perk_1, highest_value_label
"""
    rows = client.query(query, settings=_CH_SETTINGS).result_rows
    return _bucket_rows(rows)


def _bucket_rows(rows) -> KeystoneCounts:
    counts: KeystoneCounts = {}
    for champ, role, ks, label, n in rows:
        cell = counts.setdefault((int(champ), str(role), int(ks)), {})
        cell[str(label)] = int(n)
    return counts


def _save_keystone_counts(counts: KeystoneCounts) -> None:
    _CONDITION_DIR.mkdir(parents=True, exist_ok=True)
    payload = [
        [c, r, k, b, n]
        for (c, r, k), cell in counts.items()
        for b, n in cell.items()
    ]
    KEYSTONE_COUNTS_PATH.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    logger.info("wrote keystone counts: %d rows -> %s", len(payload), KEYSTONE_COUNTS_PATH)


def load_keystone_counts() -> KeystoneCounts:
    """Load counts from disk, building from ClickHouse if the file is missing."""
    if not KEYSTONE_COUNTS_PATH.exists():
        logger.info("keystone counts not found; building from ClickHouse")
        from database.clickhouse.client import get_client
        counts = build_keystone_counts(get_client())
        _save_keystone_counts(counts)
        return counts
    return _bucket_rows(json.loads(KEYSTONE_COUNTS_PATH.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Per-split keystone slot array
# ---------------------------------------------------------------------------


def _slot_query(
    split: str, last_matchid: str, chunk: int, table: str = PLAYER_PIVOT_TABLE
) -> str:
    """Build a keyset-paginated pivot query for one chunk of games.

    Returns one row per game with columns:
      matchid, champ_0..9, ks_0..9

    Slot order: blue TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY (slots 0-4),
    red TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY (slots 5-9).
    anyIf collapses the flat join back to one row per game.
    LEFT JOIN to participant_perk_ids so missing rune rows yield keystone 0.
    """
    champ_cols = ",\n    ".join(
        f"anyIf(toInt32(ifNull(ps.championid, 0)), "
        f"ps.teamid = {_SLOT_TEAMIDS[s]} AND ps.teamposition = '{_SLOT_POSITIONS[s]}')"
        f" AS champ_{s}"
        for s in range(10)
    )
    ks_cols = ",\n    ".join(
        f"anyIf(toInt32(ifNull(pki.primary_perk_1, 0)), "
        f"ps.teamid = {_SLOT_TEAMIDS[s]} AND ps.teamposition = '{_SLOT_POSITIONS[s]}')"
        f" AS ks_{s}"
        for s in range(10)
    )
    return f"""
SELECT
    p.matchid AS matchid,
    {champ_cols},
    {ks_cols}
FROM {table} AS p
INNER JOIN game_data_filtered.participant_stats AS ps
    ON p.matchid = ps.matchid
LEFT JOIN game_data.participant_perk_ids AS pki
    ON ps.matchid = pki.matchid AND ps.teamid = pki.teamid AND ps.puuid = pki.puuid
WHERE p.split = '{split}' AND p.matchid > '{last_matchid}'
GROUP BY p.matchid
ORDER BY p.matchid
LIMIT {chunk}
"""


def build_slot_keystones(split: str, cfg: DatasetConfig) -> None:
    """Build and save the [n_games, 10] keystone array for one split.

    Row order mirrors the cache: ORDER BY matchid, keyset-paginated.
    After assembly the champion columns are hard-validated against the cache.
    """
    from database.clickhouse.client import get_client

    client = get_client()

    # Count rows in this split
    count_rows = client.query(
        f"SELECT count() FROM {cfg.player_pivot_table} WHERE split = '{split}'",
        settings=_CH_SETTINGS,
    ).result_rows
    n_games = int(count_rows[0][0])
    logger.info("split %s: %d games", split, n_games)

    champ_arr = np.zeros((n_games, 10), dtype=np.int32)
    ks_arr = np.zeros((n_games, 10), dtype=np.int32)
    row_idx = 0
    last_matchid = ""

    while row_idx < n_games:
        chunk = min(_CHUNK_SIZE, n_games - row_idx)
        query = _slot_query(split, last_matchid, chunk, cfg.player_pivot_table)
        rows = client.query(query, settings=_CH_SETTINGS).result_rows
        if not rows:
            break
        n = len(rows)
        for i, row in enumerate(rows):
            # row: matchid, champ_0..9, ks_0..9
            for s in range(10):
                champ_arr[row_idx + i, s] = int(row[1 + s])
                ks_arr[row_idx + i, s] = int(row[11 + s])
        row_idx += n
        last_matchid = str(rows[-1][0])
        logger.info("assembled %d/%d rows (last matchid: %s)", row_idx, n_games, last_matchid)
        if n < chunk:
            break

    if row_idx != n_games:
        raise RuntimeError(
            f"slot array assembly got {row_idx} rows but expected {n_games} for split={split}"
        )

    # Validate champion columns against the cache
    cache_champ = _load_cache_champion_id(split, cfg)
    if not np.array_equal(champ_arr.astype(np.int32), cache_champ.astype(np.int32)):
        raise ValueError(
            f"Keystone slot array champion_id mismatch for split={split}; "
            "cache and pivot query disagree — rebuild the slot array after rebuilding the cache"
        )

    champion_sha1 = _sha1_bytes(cache_champ)
    _CONDITION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(slot_array_path(split)), ks_arr)
    _slot_meta_path(split).write_text(
        json.dumps({"split": split, "n_games": n_games, "champion_sha1": champion_sha1}),
        encoding="utf-8",
    )
    logger.info("saved %s keystone slot array: shape=%s", split, ks_arr.shape)


def _load_cache_champion_id(split: str, cfg: DatasetConfig) -> np.ndarray:
    """Load champion_id from the cache for the given split."""
    from app.ml.dataset import load_splits

    splits = load_splits(cfg)
    champ = splits[split].champion_id
    if champ is None:
        raise ValueError(f"Cache for split={split} has no champion_id array")
    return np.asarray(champ, dtype=np.int32)


def load_slot_keystones(split: str, cfg: DatasetConfig) -> np.ndarray:
    """Load the per-slot keystone array, raising on staleness."""
    path = slot_array_path(split)
    meta_path = _slot_meta_path(split)
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Keystone slot array for split={split} not found at {path}. "
            "Run: python -m app.ml.loadout_condition build --split {split}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cache_champ = _load_cache_champion_id(split, cfg)
    current_sha1 = _sha1_bytes(cache_champ)
    if meta["champion_sha1"] != current_sha1:
        raise ValueError(
            f"Keystone slot array for split={split} is stale "
            f"(expected sha1={meta['champion_sha1']!r}, got {current_sha1!r}). "
            "Rebuild with: python -m app.ml.loadout_condition build --split {split}"
        )
    return np.load(str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build keystone-condition artifacts for marginal_eval."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build", help="Build slot keystone array for a split.")
    build_p.add_argument(
        "--split", choices=("train", "test"), required=True,
        help="Which split to build the slot array for."
    )
    args = parser.parse_args()
    cfg = DatasetConfig()
    if args.command == "build":
        if not KEYSTONE_COUNTS_PATH.exists():
            from database.clickhouse.client import get_client
            counts = build_keystone_counts(get_client())
            _save_keystone_counts(counts)
        build_slot_keystones(args.split, cfg)


if __name__ == "__main__":
    main()


__all__ = [
    "KEYSTONE_COUNTS_PATH",
    "build_keystone_counts",
    "build_slot_keystones",
    "load_keystone_counts",
    "load_slot_keystones",
    "slot_array_path",
]
