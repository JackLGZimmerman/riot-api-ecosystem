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
    N_MATCHUPS_1V1,
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
# Same pairs, 0-based, for building the per-side 2vx composite priors in Python.
# Order matches the synergy_2vx feature layout (blue pairs, then red pairs).
_TEAM_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4),
    (3, 4),
)


def _composite_interaction_priors(
    win_rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-side fallback priors for interaction features.

    Given the smoothed solo win rates (`win_rate`, shape ``(n, 10)``: blue 0-4,
    red 5-9), build the prior each interaction is shrunk toward when its own
    pair is under-sampled:

      * 1v1 (25 = 5 blue x 5 red, blue perspective): a blue identity is expected
        to win ``0.5 + (wr_blue - wr_red) / 2`` against a red identity, i.e. half
        the gap in their solo win rates. Symmetric and bounded in ``[0, 1]``.
      * 2vx (20 = 10 blue pairs + 10 red pairs): the team-win prior for a same
        team pair is the average of the two solo win rates.

    Feature order matches `_CHUNK_QUERY_TEMPLATE` (1v1 blue-outer/red-inner;
    2vx blue pairs then red pairs in `_TEAM_PAIRS` order).
    """
    blue, red = win_rate[:, 0:5], win_rate[:, 5:10]
    comp_1v1 = (0.5 + (blue[:, :, None] - red[:, None, :]) / 2.0).reshape(
        win_rate.shape[0], N_MATCHUPS_1V1
    )
    blue_pairs = [(blue[:, i] + blue[:, j]) / 2.0 for i, j in _TEAM_PAIRS]
    red_pairs = [(red[:, i] + red[:, j]) / 2.0 for i, j in _TEAM_PAIRS]
    comp_2vx = np.column_stack(blue_pairs + red_pairs)
    return comp_1v1, comp_2vx


_DICT = "game_data_filtered"


def _solo(attr: str, default: str) -> str:
    # The solo prior key is the player tuple (championid, teamposition, build) itself.
    return f"arrayMap(k -> dictGetOrDefault('{_DICT}.synergy_1vx_dict', '{attr}', k, {default}), solo_keys)"


def _pair_keys(team_col: str) -> str:
    # C(5,2) same-team keys, canonicalised smaller-tuple-first to match the dictionary.
    t = team_col
    return (
        f"arrayMap(ix -> if({t}[ix.1] <= {t}[ix.2], "
        f"tupleConcat({t}[ix.1], {t}[ix.2]), tupleConcat({t}[ix.2], {t}[ix.1])), "
        f"{_TEAM_PAIRS_SQL})"
    )


# Two-stage query: the subquery canonicalises every dictionary key once per game,
# then the outer SELECT resolves each key to its (win_rate, matchups) prior pair.
# 1v1 matchups are stored blue-perspective only when the key is unswapped, so the
# rate is inverted (1 - left_win_rate) when the canonical key swapped the sides.
_CHUNK_QUERY_TEMPLATE = f"""
SELECT
    blue_win,
    {_solo("win_rate", "toFloat32(0.5)")} AS p1_raw,
    {_solo("matchups", "toUInt32(0)")} AS p1_cnt,
    arrayMap((k, swapped) -> if(swapped,
        toFloat32(1.0) - dictGetOrDefault('{_DICT}.matchup_1v1_dict', 'left_win_rate', k, toFloat32(0.5)),
        dictGetOrDefault('{_DICT}.matchup_1v1_dict', 'left_win_rate', k, toFloat32(0.5))
    ), matchup_keys, matchup_swapped) AS m1v1_raw,
    arrayMap(k -> dictGetOrDefault('{_DICT}.matchup_1v1_dict', 'matchups', k, toUInt64(0)), matchup_keys) AS m1v1_cnt,
    arrayMap(k -> dictGetOrDefault('{_DICT}.synergy_2vx_dict', 'win_rate', k, toFloat32(0.5)), synergy_keys) AS s2vx_raw,
    arrayMap(k -> dictGetOrDefault('{_DICT}.synergy_2vx_dict', 'matchups', k, toUInt64(0)), synergy_keys) AS s2vx_cnt,
    matchid
FROM (
    SELECT
        matchid,
        blue_win,
        arrayConcat(blue_players, red_players) AS solo_keys,
        arrayFlatten(arrayMap(b -> arrayMap(r ->
            if(b <= r, tupleConcat(b, r), tupleConcat(r, b)), red_players), blue_players)) AS matchup_keys,
        arrayFlatten(arrayMap(b -> arrayMap(r -> b > r, red_players), blue_players)) AS matchup_swapped,
        arrayConcat({_pair_keys("blue_players")}, {_pair_keys("red_players")}) AS synergy_keys
    FROM {{table}}
    WHERE split = '{{split}}' AND matchid > '{{last_matchid}}'
    ORDER BY matchid
    LIMIT {{chunk}}
)
ORDER BY matchid
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


# Outer-SELECT columns of _CHUNK_QUERY_TEMPLATE, by position.
_RAW_COLUMNS = ("blue_win", "p1_raw", "p1_cnt", "m1v1_raw", "m1v1_cnt", "s2vx_raw", "s2vx_cnt")
_CHUNK_SIZE = 200_000


def _smoothed_features(
    raw: dict[str, np.ndarray],
    *,
    prior_mean: float,
    prior_strength: float,
    per_side_fallback: bool,
) -> dict[str, np.ndarray]:
    win_rate = bayesian_smoothed_rate(
        raw["p1_raw"], raw["p1_cnt"], prior_mean=prior_mean, prior_strength=prior_strength
    )
    prior_1v1, prior_2vx = (
        _composite_interaction_priors(win_rate)
        if per_side_fallback
        else (prior_mean, prior_mean)
    )

    def smooth(raw_key: str, cnt_key: str, prior: float | np.ndarray) -> np.ndarray:
        return bayesian_smoothed_rate(
            raw[raw_key], raw[cnt_key], prior_mean=prior, prior_strength=prior_strength
        )

    return {
        "blue_win": raw["blue_win"],
        "win_rate": win_rate.astype(DISK_DTYPES["win_rate"], copy=False),
        "matchup_1v1": smooth("m1v1_raw", "m1v1_cnt", prior_1v1).astype(
            DISK_DTYPES["matchup_1v1"], copy=False
        ),
        "synergy_2vx": smooth("s2vx_raw", "s2vx_cnt", prior_2vx).astype(
            DISK_DTYPES["synergy_2vx"], copy=False
        ),
    }


def _stream_split(split: str, limit: int) -> Iterable[dict[str, np.ndarray]]:
    """Yield raw prior columns in chunks, keyset-paginated on matchid (no OFFSET cost)."""
    remaining = int(limit)
    last_matchid = ""
    while remaining > 0:
        chunk = min(_CHUNK_SIZE, remaining)
        query = _CHUNK_QUERY_TEMPLATE.format(
            table=PLAYER_PIVOT_TABLE, split=split, last_matchid=last_matchid, chunk=chunk
        )
        rows = get_client().query(query).result_rows
        if not rows:
            return
        yield {
            name: np.asarray([r[i] for r in rows], dtype=np.float64)
            for i, name in enumerate(_RAW_COLUMNS)
        }
        remaining -= len(rows)
        last_matchid = str(rows[-1][len(_RAW_COLUMNS)])
        if len(rows) < chunk:
            return


def _write_split(
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    limit: int,
    offset: int,
    prior_mean: float,
    prior_strength: float,
    per_side_fallback: bool,
) -> int:
    written = 0
    for raw in _stream_split(split, limit):
        block = _smoothed_features(
            raw,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            per_side_fallback=per_side_fallback,
        )
        start = offset + written
        for name, data in block.items():
            arrays[name][start : start + len(data)] = data
        written += len(block["blue_win"])
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
                    "interaction_per_side_fallback": cfg.interaction_per_side_fallback,
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
            per_side_fallback=cfg.interaction_per_side_fallback,
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
