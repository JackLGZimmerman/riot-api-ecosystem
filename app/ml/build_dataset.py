"""Build per-game arrays (priors + outcomes) used by the linear model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from app.core.logging.logger import setup_logging_config
from app.ml.cache_layout import (
    ARRAY_SHAPES,
    CACHE_FORMAT,
    CACHE_META_FILE,
    DISK_DTYPES,
    array_paths,
)
from app.ml.config import PLAYER_PIVOT_TABLE, DatasetConfig
from app.ml.utils.bayesian_smoothing import bayesian_smoothed_rate
from database.clickhouse.client import get_client

SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))

setup_logging_config()
logger = logging.getLogger(__name__)

# Per-team C(5,2) pair indices, 1-based for ClickHouse array element access.
_TEAM_PAIRS_SQL = (
    "[(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]"
)

# arrayMap lookup for a team's 10 synergy_2vx_dict probes, canonicalising the
# pair so the smaller (championid, teamposition, build) tuple is in slot 1.
def _team_2vx_sql(team_col: str, attr: str, default: str) -> str:
    t = team_col
    return f"""
        arrayMap(ix ->
            dictGetOrDefault(
                'game_data_filtered.synergy_2vx_dict', '{attr}',
                if({t}[ix.1] <= {t}[ix.2],
                    (
                        tupleElement({t}[ix.1], 1),
                        tupleElement({t}[ix.1], 2),
                        tupleElement({t}[ix.1], 3),
                        tupleElement({t}[ix.2], 1),
                        tupleElement({t}[ix.2], 2),
                        tupleElement({t}[ix.2], 3)
                    ),
                    (
                        tupleElement({t}[ix.2], 1),
                        tupleElement({t}[ix.2], 2),
                        tupleElement({t}[ix.2], 3),
                        tupleElement({t}[ix.1], 1),
                        tupleElement({t}[ix.1], 2),
                        tupleElement({t}[ix.1], 3)
                    )
                ),
                {default}
            ),
            {_TEAM_PAIRS_SQL}
        )
    """


_CHUNK_QUERY_TEMPLATE = f"""
SELECT
    blue_win,
    arrayMap(p -> dictGetOrDefault(
        'game_data_filtered.synergy_1vx_dict', 'win_rate',
        (tupleElement(p, 1), tupleElement(p, 2), tupleElement(p, 3)),
        toFloat32(0.5)
    ), arrayConcat(blue_players, red_players)) AS p1_raw,
    arrayMap(p -> dictGetOrDefault(
        'game_data_filtered.synergy_1vx_dict', 'matchups',
        (tupleElement(p, 1), tupleElement(p, 2), tupleElement(p, 3)),
        toUInt32(0)
    ), arrayConcat(blue_players, red_players)) AS p1_cnt,
    arrayFlatten(arrayMap(b -> arrayMap(r ->
        if(b <= r,
            dictGetOrDefault(
                'game_data_filtered.matchup_1v1_dict', 'left_win_rate',
                (
                    tupleElement(b, 1), tupleElement(b, 2), tupleElement(b, 3),
                    tupleElement(r, 1), tupleElement(r, 2), tupleElement(r, 3)
                ),
                toFloat32(0.5)
            ),
            toFloat32(1.0) - dictGetOrDefault(
                'game_data_filtered.matchup_1v1_dict', 'left_win_rate',
                (
                    tupleElement(r, 1), tupleElement(r, 2), tupleElement(r, 3),
                    tupleElement(b, 1), tupleElement(b, 2), tupleElement(b, 3)
                ),
                toFloat32(0.5)
            )
        ),
        red_players
    ), blue_players)) AS m1v1_raw,
    arrayFlatten(arrayMap(b -> arrayMap(r ->
        dictGetOrDefault(
            'game_data_filtered.matchup_1v1_dict', 'matchups',
            if(b <= r,
                (
                    tupleElement(b, 1), tupleElement(b, 2), tupleElement(b, 3),
                    tupleElement(r, 1), tupleElement(r, 2), tupleElement(r, 3)
                ),
                (
                    tupleElement(r, 1), tupleElement(r, 2), tupleElement(r, 3),
                    tupleElement(b, 1), tupleElement(b, 2), tupleElement(b, 3)
                )
            ),
            toUInt64(0)
        ),
        red_players
    ), blue_players)) AS m1v1_cnt,
    arrayConcat(
        {_team_2vx_sql("blue_players", "win_rate", "toFloat32(0.5)")},
        {_team_2vx_sql("red_players", "win_rate", "toFloat32(0.5)")}
    ) AS s2vx_raw,
    arrayConcat(
        {_team_2vx_sql("blue_players", "matchups", "toUInt64(0)")},
        {_team_2vx_sql("red_players", "matchups", "toUInt64(0)")}
    ) AS s2vx_cnt,
    matchid
FROM {{table}}
WHERE split = '{{split}}' AND matchid > '{{last_matchid}}'
ORDER BY matchid
LIMIT {{chunk}}
"""


def _split_counts(cfg: DatasetConfig) -> dict[str, int]:
    rows = get_client().query(
        f"""
        SELECT split, count()
        FROM {PLAYER_PIVOT_TABLE}
        WHERE split IN ('train', 'validation', 'test')
        GROUP BY split
        """
    )
    available = {str(split): int(count) for split, count in rows.result_rows}
    if cfg.max_games is None:
        return {
            "train": available.get("train", 0),
            "val": available.get("validation", 0),
            "test": available.get("test", 0),
        }

    n_test = round(cfg.max_games * cfg.test_fraction)
    n_val = round(cfg.max_games * cfg.val_fraction)
    n_train = cfg.max_games - n_val - n_test
    return {
        "train": min(n_train, available.get("train", 0)),
        "val": min(n_val, available.get("validation", 0)),
        "test": min(n_test, available.get("test", 0)),
    }


def _open_arrays(n_games: int, cache_dir: Path) -> dict[str, np.ndarray]:
    paths = array_paths(cache_dir)
    arrays: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        shape = (n_games, *ARRAY_SHAPES[name])
        arrays[name] = np.lib.format.open_memmap(
            path, mode="w+", dtype=DISK_DTYPES[name], shape=shape
        )
    return arrays


def _features_for_block(
    blue_win: np.ndarray,
    p1_raw: np.ndarray,
    p1_cnt: np.ndarray,
    m1v1_raw: np.ndarray,
    m1v1_cnt: np.ndarray,
    s2vx_raw: np.ndarray,
    s2vx_cnt: np.ndarray,
    *,
    prior_mean: float,
    prior_strength: float,
) -> dict[str, np.ndarray]:
    return {
        "blue_win": blue_win,
        "win_rate": bayesian_smoothed_rate(
            p1_raw, p1_cnt, prior_mean=prior_mean, prior_strength=prior_strength
        ).astype(DISK_DTYPES["win_rate"], copy=False),
        "matchup_1v1": bayesian_smoothed_rate(
            m1v1_raw, m1v1_cnt, prior_mean=prior_mean, prior_strength=prior_strength
        ).astype(DISK_DTYPES["matchup_1v1"], copy=False),
        "synergy_2vx": bayesian_smoothed_rate(
            s2vx_raw, s2vx_cnt, prior_mean=prior_mean, prior_strength=prior_strength
        ).astype(DISK_DTYPES["synergy_2vx"], copy=False),
    }


_CHUNK_SIZE = 200_000


def _stream_split(
    split: str, limit: int
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if limit <= 0:
        return
    remaining = int(limit)
    last_matchid = ""
    while remaining > 0:
        chunk = min(_CHUNK_SIZE, remaining)
        # Keyset pagination on matchid avoids OFFSET cost on the order key.
        query = _CHUNK_QUERY_TEMPLATE.format(
            table=PLAYER_PIVOT_TABLE,
            split=split,
            last_matchid=last_matchid,
            chunk=chunk,
        )
        rows = get_client().query(query).result_rows
        if not rows:
            break
        blue_win = np.fromiter(
            (int(r[0]) for r in rows),
            dtype=DISK_DTYPES["blue_win"],
            count=len(rows),
        )
        p1_raw = np.asarray([r[1] for r in rows], dtype=np.float64)
        p1_cnt = np.asarray([r[2] for r in rows], dtype=np.float64)
        m1v1_raw = np.asarray([r[3] for r in rows], dtype=np.float64)
        m1v1_cnt = np.asarray([r[4] for r in rows], dtype=np.float64)
        s2vx_raw = np.asarray([r[5] for r in rows], dtype=np.float64)
        s2vx_cnt = np.asarray([r[6] for r in rows], dtype=np.float64)
        last_matchid = str(rows[-1][7])
        yield blue_win, p1_raw, p1_cnt, m1v1_raw, m1v1_cnt, s2vx_raw, s2vx_cnt
        remaining -= len(rows)
        if len(rows) < chunk:
            break


def _write_split(
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    limit: int,
    offset: int,
    prior_mean: float,
    prior_strength: float,
) -> int:
    written = 0
    for chunk in _stream_split(split, limit):
        blue_win, p1_raw, p1_cnt, m1v1_raw, m1v1_cnt, s2vx_raw, s2vx_cnt = chunk
        block = _features_for_block(
            blue_win,
            p1_raw,
            p1_cnt,
            m1v1_raw,
            m1v1_cnt,
            s2vx_raw,
            s2vx_cnt,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
        )
        size = block["blue_win"].shape[0]
        start = offset + written
        end = start + size
        for name, data in block.items():
            arrays[name][start:end] = data
        written += size
    return written


def _write_meta(cfg: DatasetConfig, n_games: int, splits: dict[str, int]) -> Path:
    meta_path = cfg.cache_dir / CACHE_META_FILE
    meta_path.write_text(
        json.dumps(
            {
                "format": CACHE_FORMAT,
                "n_games": n_games,
                "splits": splits,
                "smoothing": {
                    "prior_mean": cfg.smoothing_prior_mean,
                    "prior_strength": cfg.smoothing_prior_strength,
                },
            },
            indent=2,
        )
    )
    return meta_path


def build(cfg: DatasetConfig | None = None) -> Path:
    cfg = cfg or DatasetConfig()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    counts = _split_counts(cfg)
    n_games = sum(counts.values())
    arrays = _open_arrays(n_games, cfg.cache_dir)

    logger.info("Building cache: games=%d splits=%s", n_games, counts)
    offset = 0
    for sql_split, meta_split in SPLITS:
        written = _write_split(
            arrays,
            split=sql_split,
            limit=counts[meta_split],
            offset=offset,
            prior_mean=cfg.smoothing_prior_mean,
            prior_strength=cfg.smoothing_prior_strength,
        )
        if written != counts[meta_split]:
            raise RuntimeError(
                f"{meta_split} wrote {written}, expected {counts[meta_split]}"
            )
        offset += written
        logger.info("Wrote split %s: %d games", meta_split, written)

    for array in arrays.values():
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()

    return _write_meta(cfg, n_games, counts)


if __name__ == "__main__":
    build()
