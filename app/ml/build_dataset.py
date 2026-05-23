"""Build the two-array cache used by the win-rate linear model."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.core.logging.logger import setup_logging_config
from app.ml.cache_layout import CACHE_FORMAT, CACHE_META_FILE, DISK_DTYPES, array_paths
from app.ml.config import (
    PLAYER_PIVOT_TABLE,
    SPLIT_TABLE,
    SYNERGY_1VX_TABLE,
    DatasetConfig,
)
from app.ml.model import N_PLAYER_FEATURES
from app.ml.utils.bayesian_smoothing import bayesian_smoothed_rate
from database.clickhouse.client import get_client

DEFAULT_WIN_RATE = 0.5
SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))

setup_logging_config()
logger = logging.getLogger(__name__)


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


def _arrays(n_games: int, cache_dir: Path) -> dict[str, np.ndarray]:
    paths = array_paths(cache_dir)
    return {
        "win_rate": np.lib.format.open_memmap(
            paths["win_rate"],
            mode="w+",
            dtype=DISK_DTYPES["win_rate"],
            shape=(n_games, N_PLAYER_FEATURES),
        ),
        "blue_win": np.lib.format.open_memmap(
            paths["blue_win"],
            mode="w+",
            dtype=DISK_DTYPES["blue_win"],
            shape=(n_games,),
        ),
    }


def _row_blocks(
    split: str,
    limit: int,
    *,
    smoothing_prior_mean: float,
    smoothing_prior_strength: float,
):
    if limit <= 0:
        return

    query = f"""
    SELECT
        any(blue_win) AS blue_win,
        arrayMap(
            x -> tupleElement(x, 2),
            arraySort(
                x -> tupleElement(x, 1),
                groupArray((slot, ifNull(prior.win_rate, toFloat32({DEFAULT_WIN_RATE}))))
            )
        ) AS win_rates,
        arrayMap(
            x -> tupleElement(x, 2),
            arraySort(
                x -> tupleElement(x, 1),
                groupArray((slot, ifNull(prior.matchups, toUInt32(0))))
            )
        ) AS matchup_counts
    FROM (
        SELECT
            s.split_index AS split_index,
            p.matchid AS matchid,
            p.blue_win AS blue_win,
            tupleElement(token, 1) AS slot,
            tupleElement(tupleElement(token, 2), 1) AS championid,
            tupleElement(tupleElement(token, 2), 2) AS teamposition,
            tupleElement(tupleElement(token, 2), 3) AS build
        FROM {SPLIT_TABLE} AS s
        INNER JOIN {PLAYER_PIVOT_TABLE} AS p
            ON s.matchid = p.matchid
        ARRAY JOIN arrayZip(range({N_PLAYER_FEATURES}), arrayConcat(p.blue_players, p.red_players)) AS token
        WHERE
            s.split = '{split}'
            AND p.split = '{split}'
    ) AS player
    ANY LEFT JOIN (
        SELECT championid, teamposition, build, matchups, win_rate
        FROM {SYNERGY_1VX_TABLE}
        WHERE split = 'train'
    ) AS prior
        ON prior.championid = player.championid
        AND prior.teamposition = player.teamposition
        AND prior.build = player.build
    GROUP BY split_index, matchid
    ORDER BY split_index
    LIMIT {int(limit)}
    SETTINGS join_use_nulls = 1
    """
    with get_client().query_column_block_stream(query) as stream:
        for block in stream:
            if not block or len(block[0]) == 0:
                continue
            blue_win = np.asarray(block[0], dtype=DISK_DTYPES["blue_win"])
            raw_win_rate = np.asarray(block[1], dtype=np.float64)
            matchup_counts = np.asarray(block[2], dtype=np.float64)
            if raw_win_rate.ndim != 2 or raw_win_rate.shape[1] != N_PLAYER_FEATURES:
                raise ValueError(
                    f"Expected win_rate block with shape [n, {N_PLAYER_FEATURES}], "
                    f"got {raw_win_rate.shape}"
                )
            if (
                matchup_counts.ndim != 2
                or matchup_counts.shape[1] != N_PLAYER_FEATURES
            ):
                raise ValueError(
                    f"Expected matchup count block with shape [n, {N_PLAYER_FEATURES}], "
                    f"got {matchup_counts.shape}"
                )
            win_rate = bayesian_smoothed_rate(
                raw_win_rate,
                matchup_counts,
                prior_mean=smoothing_prior_mean,
                prior_strength=smoothing_prior_strength,
            ).astype(DISK_DTYPES["win_rate"], copy=False)
            yield blue_win, win_rate


def _write_split(
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    limit: int,
    offset: int,
    smoothing_prior_mean: float,
    smoothing_prior_strength: float,
) -> int:
    written = 0
    for blue_win, win_rate in _row_blocks(
        split,
        limit,
        smoothing_prior_mean=smoothing_prior_mean,
        smoothing_prior_strength=smoothing_prior_strength,
    ):
        start = offset + written
        end = start + blue_win.shape[0]
        arrays["blue_win"][start:end] = blue_win
        arrays["win_rate"][start:end] = win_rate
        written += blue_win.shape[0]
    return written


def _write_meta(cfg: DatasetConfig, n_games: int, splits: dict[str, int]) -> Path:
    cache_dir = cfg.cache_dir
    meta_path = cache_dir / CACHE_META_FILE
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
    arrays = _arrays(n_games, cfg.cache_dir)
    logger.info("Building cache: games=%d splits=%s", n_games, counts)

    offset = 0
    for sql_split, meta_split in SPLITS:
        written = _write_split(
            arrays,
            split=sql_split,
            limit=counts[meta_split],
            offset=offset,
            smoothing_prior_mean=cfg.smoothing_prior_mean,
            smoothing_prior_strength=cfg.smoothing_prior_strength,
        )
        if written != counts[meta_split]:
            raise RuntimeError(
                f"{meta_split} wrote {written}, expected {counts[meta_split]}"
            )
        offset += written

    for array in arrays.values():
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()

    return _write_meta(cfg, n_games, counts)


if __name__ == "__main__":
    build()
