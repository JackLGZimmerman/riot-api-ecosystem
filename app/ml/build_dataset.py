"""Build per-game arrays used by the HGNN win-rate model."""

from __future__ import annotations

import argparse
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
    sidecar_array_paths,
)
from app.ml.config import DatasetConfig
from app.ml.encoder_sidecar import EncoderSidecarLookup
from app.ml.loadout_patch_features import (
    feature_metadata,
    write_loadout_patch_feature_arrays,
)
from app.core.utils.smoothing import smooth_rate_by_mode
from clickhouse_connect.driver.exceptions import StreamFailureError

from database.clickhouse.client import _local, get_client

SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))
SPLIT_ORDER = tuple(meta_split for _, meta_split in SPLITS)

setup_logging_config()
logger = logging.getLogger(__name__)

def _leave_one_out(raw: dict[str, np.ndarray]) -> None:
    """Subtract each train game's own outcome from its in-sample priors, in place.

    Own-outcome per slot is the focal side's win: blue 0-4 / red 5-9. Count-1
    cells collapse to count 0, so smoothing returns their composite prior.
    Applies to the solo (champion/role/build) prior and both per-player priors.
    """
    blue_win = raw["blue_win"]
    red_win = 1.0 - blue_win
    own = np.concatenate(
        [np.broadcast_to(blue_win[:, None], (blue_win.size, 5)),
         np.broadcast_to(red_win[:, None], (blue_win.size, 5))], axis=1
    )

    for rate_key, cnt_key in (("p1_raw", "p1_cnt"), ("pl_raw", "pl_cnt"), ("plc_raw", "plc_cnt")):
        count = raw[cnt_key]
        loo_count = count - 1.0
        loo_wins = np.rint(raw[rate_key] * count) - own
        safe = loo_count > 0.0
        raw[rate_key] = np.where(safe, loo_wins / np.where(safe, loo_count, 1.0), 0.5)
        raw[cnt_key] = np.maximum(loo_count, 0.0)


def _solo(attr: str, default: str) -> str:
    # The solo prior key is the player tuple (championid, teamposition, build) itself.
    return f"arrayMap(k -> dictGetOrDefault('{{solo_prior_dict}}', '{attr}', k, {default}), solo_keys)"


def _player(attr: str, default: str) -> str:
    # Per-player prior keyed by puuid alone.
    return f"arrayMap(k -> dictGetOrDefault('{{player_prior_dict}}', '{attr}', tuple(k), {default}), player_keys)"


def _player_champ(attr: str, default: str) -> str:
    # Per-(player, champion) prior; champion comes from the matching solo key.
    return (
        f"arrayMap((k, s) -> dictGetOrDefault('{{player_champ_prior_dict}}', '{attr}', "
        f"(k, toInt32(tupleElement(s, 1))), {default}), player_keys, solo_keys)"
    )


# Two-stage query: the subquery canonicalises the per-slot solo key once per game,
# then the outer SELECT resolves each key to its (win_rate, matchups) prior pair.
_CHUNK_QUERY_TEMPLATE = f"""
SELECT
    blue_win,
    {_solo("win_rate", "toFloat32(0.5)")} AS p1_raw,
    {_solo("matchups", "toUInt32(0)")} AS p1_cnt,
    arrayMap(k -> toInt16(k.1), solo_keys) AS champion_id,
    arrayMap(k -> toInt16(if(indexOf({{build_vocab}}, toString(k.3)) = 0, {{n_builds}}, indexOf({{build_vocab}}, toString(k.3)) - 1)), solo_keys) AS build_id,
    {_player("win_rate", "toFloat32(0.5)")} AS pl_raw,
    {_player("matchups", "toUInt32(0)")} AS pl_cnt,
    {_player_champ("win_rate", "toFloat32(0.5)")} AS plc_raw,
    {_player_champ("matchups", "toUInt32(0)")} AS plc_cnt,
    matchid
FROM (
    SELECT
        matchid,
        blue_win,
        arrayMap(p -> (tupleElement(p, 1), tupleElement(p, 2), {{key_build_expr}}), blue_players) AS blue_key_players,
        arrayMap(p -> (tupleElement(p, 1), tupleElement(p, 2), {{key_build_expr}}), red_players) AS red_key_players,
        arrayConcat(blue_key_players, red_key_players) AS solo_keys,
        arrayMap(p -> toString(tupleElement(p, 4)), arrayConcat(blue_players, red_players)) AS player_keys
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
        FROM {cfg.player_pivot_table}
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


def _remove_stale_sidecar_arrays(cache_dir: Path) -> None:
    """Drop any per-game sidecar arrays left by an older (<=v27) cache build.

    Current compact caches gather latents per batch from the frozen artifact, so
    stale per-game arrays must not linger or the loader would silently prefer them.
    """
    for path in sidecar_array_paths(cache_dir).values():
        path.unlink(missing_ok=True)


# Outer-SELECT columns of _CHUNK_QUERY_TEMPLATE, by position (matchid trails them).
_RAW_COLUMNS = (
    "blue_win", "p1_raw", "p1_cnt",
    "champion_id", "build_id",
    "pl_raw", "pl_cnt", "plc_raw", "plc_cnt",
)

_CHUNK_SIZE = 50_000


def _identity_meta(cfg: DatasetConfig) -> tuple[int, list[str]]:
    """Identity embedding metadata from train priors.

    Champion ids are used as raw embedding indices, so the table covers
    ``max(championid)+1`` rows. Builds are string labels mapped to a sorted vocab;
    the model reserves one extra row in each table for unknown ids at inference.
    """
    client = get_client()
    if cfg.use_final_build_labels:
        max_champ = client.query(
            f"SELECT toInt32(max(championid)) FROM {cfg.solo_prior_table} WHERE split = 'train'"
        ).result_rows[0][0]
    else:
        rows = client.query(
            f"""
            SELECT
                toInt32(max(championid)) AS max_championid,
                countIf(build = {{label:String}}) AS no_build_rows
            FROM {cfg.solo_prior_table}
            WHERE split = 'train'
            """,
            parameters={"label": cfg.draft_unknown_build_label},
        ).result_rows
        max_champ, no_build_rows = rows[0]
        if int(no_build_rows) <= 0:
            raise ValueError(
                "Draft-time-safe cache requested (use_final_build_labels=False), "
                "but no train priors use draft_unknown_build_label="
                f"{cfg.draft_unknown_build_label!r}. Rebuild the no-build aggregate "
                "priors before building this cache."
            )
        return int(max_champ) + 1, [cfg.draft_unknown_build_label]
    builds = client.query(
        f"SELECT DISTINCT build FROM {cfg.solo_prior_table} WHERE split = 'train' ORDER BY build"
    ).result_rows
    return int(max_champ) + 1, [str(b[0]) for b in builds]


def _smoothed_features(
    raw: dict[str, np.ndarray],
    cfg: DatasetConfig,
) -> dict[str, np.ndarray]:
    def smooth(rate: np.ndarray, count: np.ndarray, prior_mean) -> np.ndarray:
        return smooth_rate_by_mode(
            rate,
            count,
            prior_mean=prior_mean,
            prior_strength=cfg.smoothing_prior_strength,
            amplification_threshold=cfg.amplification_threshold,
            smoothing_mode=cfg.smoothing_mode,
            confidence_threshold=cfg.prior_confidence_matchups,
        )

    win_rate = smooth(raw["p1_raw"], raw["p1_cnt"], cfg.smoothing_prior_mean)
    player_rate = smooth(raw["pl_raw"], raw["pl_cnt"], cfg.smoothing_prior_mean)
    # Nested EB: the per-(player, champion) rate shrinks toward the player's
    # own smoothed overall rate, not the global mean.
    player_champ_rate = smooth(raw["plc_raw"], raw["plc_cnt"], player_rate)
    return {
        "blue_win": raw["blue_win"],
        "win_rate": win_rate.astype(DISK_DTYPES["win_rate"], copy=False),
        "p1_cnt": raw["p1_cnt"].astype(DISK_DTYPES["p1_cnt"], copy=False),
        "champion_id": raw["champion_id"].astype(DISK_DTYPES["champion_id"], copy=False),
        "build_id": raw["build_id"].astype(DISK_DTYPES["build_id"], copy=False),
        "player_rate": player_rate.astype(DISK_DTYPES["player_rate"], copy=False),
        "player_cnt": raw["pl_cnt"].astype(DISK_DTYPES["player_cnt"], copy=False),
        "player_champ_rate": player_champ_rate.astype(DISK_DTYPES["player_champ_rate"], copy=False),
        "player_champ_cnt": raw["plc_cnt"].astype(DISK_DTYPES["player_champ_cnt"], copy=False),
    }


def _fetch_chunk_rows(query: str, attempts: int = 4) -> list:
    """Run a chunk query, retrying on the intermittent ClickHouse StreamFailureError.
    A stream failure can leave the thread-local connection unusable, so drop it and
    reconnect before each retry."""
    for attempt in range(1, attempts + 1):
        try:
            return list(get_client().query(query).result_rows)
        except StreamFailureError:
            if attempt == attempts:
                raise
            logger.warning("StreamFailureError on chunk fetch (attempt %d), reconnecting", attempt)
            client = getattr(_local, "client", None)
            if client is not None:
                try:
                    client.close()
                finally:
                    _local.client = None
    return []


def _stream_split(
    cfg: DatasetConfig,
    split: str,
    limit: int,
    *,
    build_vocab_sql: str,
    n_builds: int,
    key_build_expr: str,
) -> Iterable[dict[str, np.ndarray]]:
    """Yield raw prior columns in chunks, keyset-paginated on matchid (no OFFSET cost)."""
    remaining = int(limit)
    last_matchid = ""
    while remaining > 0:
        chunk = min(_CHUNK_SIZE, remaining)
        query = _CHUNK_QUERY_TEMPLATE.format(
            table=cfg.player_pivot_table,
            solo_prior_dict=cfg.solo_prior_dict,
            player_prior_dict=cfg.player_prior_dict,
            player_champ_prior_dict=cfg.player_champ_prior_dict,
            split=split,
            last_matchid=last_matchid,
            chunk=chunk,
            build_vocab=build_vocab_sql,
            n_builds=n_builds,
            key_build_expr=key_build_expr,
        )
        rows = _fetch_chunk_rows(query)
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
    cfg: DatasetConfig,
    *,
    split: str,
    limit: int,
    offset: int,
    leave_one_out: bool,
    build_vocab_sql: str,
    n_builds: int,
    key_build_expr: str,
) -> int:
    written = 0
    for raw in _stream_split(
        cfg,
        split,
        limit,
        build_vocab_sql=build_vocab_sql,
        n_builds=n_builds,
        key_build_expr=key_build_expr,
    ):
        if leave_one_out:
            _leave_one_out(raw)
        block = _smoothed_features(raw, cfg)
        start = offset + written
        for name, data in block.items():
            arrays[name][start : start + len(data)] = data
        written += len(block["blue_win"])
    return written


def _write_meta(
    cfg: DatasetConfig,
    n_games: int,
    splits: dict[str, int],
    identity: dict,
    sidecar_lookup: EncoderSidecarLookup | None,
) -> Path:
    split_counts = {name: int(splits[name]) for name in SPLIT_ORDER}
    split_ranges: dict[str, dict[str, int]] = {}
    offset = 0
    for split_name in SPLIT_ORDER:
        count = split_counts[split_name]
        split_ranges[split_name] = {"start": offset, "stop": offset + count}
        offset += count
    if offset != int(n_games):
        raise ValueError("Cache split counts do not match n_games; rebuild aborted.")

    sidecar_meta = None
    if sidecar_lookup is not None:
        sidecar_meta = {
            "path": str(cfg.encoder_sidecar_path),
            "dims": sidecar_lookup.dims.as_dict(),
            "metadata": sidecar_lookup.metadata,
        }
    meta_path = cfg.cache_dir / CACHE_META_FILE
    meta_path.write_text(
        json.dumps(
            {
                "format": CACHE_FORMAT,
                "n_games": n_games,
                "splits": split_counts,
                "split_order": list(SPLIT_ORDER),
                "split_ranges": split_ranges,
                "identity": identity,
                "identity_encoder_sidecar": sidecar_meta,
                "production_features": feature_metadata(),
                "smoothing": {
                    "prior_mean": cfg.smoothing_prior_mean,
                    "prior_strength": cfg.smoothing_prior_strength,
                    "amplification_threshold": cfg.amplification_threshold,
                    "smoothing_mode": cfg.smoothing_mode,
                    "prior_confidence_matchups": cfg.prior_confidence_matchups,
                    "interaction_loo": cfg.interaction_loo,
                    "use_final_build_labels": cfg.use_final_build_labels,
                    "draft_unknown_build_label": cfg.draft_unknown_build_label,
                },
                "sources": {
                    "player_pivot_table": cfg.player_pivot_table,
                    "solo_prior_table": cfg.solo_prior_table,
                    "solo_prior_dict": cfg.solo_prior_dict,
                    "player_prior_dict": cfg.player_prior_dict,
                    "player_champ_prior_dict": cfg.player_champ_prior_dict,
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
    n_champions, build_vocab = _identity_meta(cfg)
    # The frozen sidecar artifact is loaded only to validate it and record its
    # path/dims in the cache meta; training gathers latents per batch from it
    # instead of materialising one copy per game-slot (≈3000x smaller on disk).
    sidecar_lookup = (
        EncoderSidecarLookup.load(cfg.encoder_sidecar_path)
        if cfg.encoder_sidecar_path is not None
        else None
    )
    _remove_stale_sidecar_arrays(cfg.cache_dir)
    arrays = _open_arrays(n_games, cfg.cache_dir)
    n_builds = len(build_vocab)
    build_vocab_sql = "[" + ",".join(f"'{b}'" for b in build_vocab) + "]"
    key_build_expr = (
        "toString(tupleElement(p, 3))"
        if cfg.use_final_build_labels
        else f"'{cfg.draft_unknown_build_label}'"
    )
    logger.info(
        "Building cache: games=%d splits=%s n_champions=%d n_builds=%d",
        n_games,
        counts,
        n_champions,
        n_builds,
    )
    offset = 0
    for sql_split, meta_split in SPLITS:
        written = _write_split(
            arrays,
            cfg,
            split=sql_split,
            limit=counts[meta_split],
            offset=offset,
            leave_one_out=cfg.interaction_loo and sql_split == "train",
            build_vocab_sql=build_vocab_sql,
            n_builds=n_builds,
            key_build_expr=key_build_expr,
        )
        if written != counts[meta_split]:
            raise RuntimeError(
                f"{meta_split} wrote {written}, expected {counts[meta_split]}"
            )
        offset += written
        logger.info("Wrote split %s: %d games", meta_split, written)

    write_loadout_patch_feature_arrays(
        cfg=cfg,
        arrays=arrays,
        split_counts=counts,
        split_order=SPLIT_ORDER,
    )

    for array in arrays.values():
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()

    return _write_meta(
        cfg,
        n_games,
        counts,
        {"n_champions": n_champions, "n_builds": n_builds, "build_vocab": build_vocab},
        sidecar_lookup,
    )


def _parse_args() -> DatasetConfig:
    defaults = DatasetConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--max-games", type=int, default=defaults.max_games)
    parser.add_argument(
        "--encoder-sidecar-path",
        type=Path,
        default=defaults.encoder_sidecar_path,
        help="Frozen three-encoder sidecar artifact to record in cache metadata.",
    )
    args = parser.parse_args()
    return DatasetConfig(
        cache_dir=args.cache_dir,
        max_games=args.max_games,
        encoder_sidecar_path=args.encoder_sidecar_path,
    )


def main() -> None:
    build(_parse_args())


if __name__ == "__main__":
    main()
