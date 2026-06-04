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
from app.ml.config import (
    DatasetConfig,
    MATCHUP_1V1_LEVEL_TABLES,
    SYNERGY_2VX_LEVEL_TABLES,
)
from app.ml.encoder_sidecar import EncoderSidecarLookup
from app.core.utils.common import TEAM_PAIRS
from app.core.utils.smoothing import (
    build_group_sql,
    eb_strength_from_moments,
    smooth_ml_prior_features,
)
from clickhouse_connect.driver.exceptions import StreamFailureError

from database.clickhouse.client import _local, get_client

SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))
SPLIT_ORDER = tuple(meta_split for _, meta_split in SPLITS)

setup_logging_config()
logger = logging.getLogger(__name__)

# Per-team C(5,2) pair indices, 1-based for ClickHouse array element access.
_TEAM_PAIRS_SQL = (
    "[(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]"
)
_KEY_BUILD_GROUP_EXPR = build_group_sql("{key_build_expr}", alias=None)


def _leave_one_out(raw: dict[str, np.ndarray]) -> None:
    """Subtract each train game's own outcome from its in-sample priors, in place.

    Own-outcome per slot is the focal side's win: solo blue 0-4 / red 5-9; 1v1 is
    blue-perspective for all 25; 2vx blue pairs 0-9 / red pairs 10-19. The own
    outcome is identical across every backoff level of an interaction (the build,
    no-build, and champion levels all counted this same game), so each level is
    LOO'd with the same own array. Count-1 cells collapse to count 0, so nested
    pooling returns their parent/composite prior.
    """
    blue_win = raw["blue_win"]
    red_win = 1.0 - blue_win
    solo_own = np.concatenate(
        [np.broadcast_to(blue_win[:, None], (blue_win.size, 5)),
         np.broadcast_to(red_win[:, None], (blue_win.size, 5))], axis=1
    )
    m1v1_own = np.broadcast_to(blue_win[:, None], (blue_win.size, 25))
    s2vx_own = np.concatenate(
        [np.broadcast_to(blue_win[:, None], (blue_win.size, 10)),
         np.broadcast_to(red_win[:, None], (blue_win.size, 10))], axis=1
    )

    levels: list[tuple[str, str, np.ndarray]] = [("p1_raw", "p1_cnt", solo_own)]
    levels += [(rk, ck, m1v1_own) for rk, ck in _M1V1_LEVELS]
    levels += [(rk, ck, s2vx_own) for rk, ck in _S2VX_LEVELS]
    for raw_key, cnt_key, own in levels:
        count = raw[cnt_key]
        loo_count = count - 1.0
        loo_wins = np.rint(raw[raw_key] * count) - own
        safe = loo_count > 0.0
        raw[raw_key] = np.where(safe, loo_wins / np.where(safe, loo_count, 1.0), 0.5)
        raw[cnt_key] = np.maximum(loo_count, 0.0)


def _solo(attr: str, default: str) -> str:
    # The solo prior key is the player tuple (championid, teamposition, build) itself.
    return f"arrayMap(k -> dictGetOrDefault('{{solo_prior_dict}}', '{attr}', k, {default}), solo_keys)"


def _pair_keys(team_col: str) -> str:
    # C(5,2) same-team keys, canonicalised smaller-tuple-first to match the dictionary.
    t = team_col
    return (
        f"arrayMap(ix -> if({t}[ix.1] <= {t}[ix.2], "
        f"tupleConcat({t}[ix.1], {t}[ix.2]), tupleConcat({t}[ix.2], {t}[ix.1])), "
        f"{_TEAM_PAIRS_SQL})"
    )


def _pair_keys_nobuild(team_col: str) -> str:
    # C(5,2) same-team keys with build dropped: (champ, role) per member,
    # canonicalised smaller-2-tuple-first to match synergy_2vx_nobuild_dict.
    t = team_col
    a = f"({t}[ix.1].1, {t}[ix.1].2)"
    b = f"({t}[ix.2].1, {t}[ix.2].2)"
    return (
        f"arrayMap(ix -> if({a} <= {b}, "
        f"({t}[ix.1].1, {t}[ix.1].2, {t}[ix.2].1, {t}[ix.2].2), "
        f"({t}[ix.2].1, {t}[ix.2].2, {t}[ix.1].1, {t}[ix.1].2)), "
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
        toFloat32(1.0) - dictGetOrDefault('{{matchup_1v1_dict}}', 'left_win_rate', k, toFloat32(0.5)),
        dictGetOrDefault('{{matchup_1v1_dict}}', 'left_win_rate', k, toFloat32(0.5))
    ), matchup_keys, matchup_swapped) AS m1v1_raw,
    arrayMap(k -> dictGetOrDefault('{{matchup_1v1_dict}}', 'matchups', k, toUInt64(0)), matchup_keys) AS m1v1_cnt,
    arrayMap(k -> dictGetOrDefault('{{matchup_1v1_nobuild_dict}}', 'blue_win_rate', k, toFloat32(0.5)), matchup_nb_keys) AS m1v1_nb_raw,
    arrayMap(k -> dictGetOrDefault('{{matchup_1v1_nobuild_dict}}', 'matchups', k, toUInt64(0)), matchup_nb_keys) AS m1v1_nb_cnt,
    arrayMap(k -> dictGetOrDefault('{{matchup_1v1_champ_dict}}', 'blue_win_rate', k, toFloat32(0.5)), matchup_champ_keys) AS m1v1_champ_raw,
    arrayMap(k -> dictGetOrDefault('{{matchup_1v1_champ_dict}}', 'matchups', k, toUInt64(0)), matchup_champ_keys) AS m1v1_champ_cnt,
    arrayMap(k -> dictGetOrDefault('{{synergy_2vx_dict}}', 'win_rate', k, toFloat32(0.5)), synergy_keys) AS s2vx_raw,
    arrayMap(k -> dictGetOrDefault('{{synergy_2vx_dict}}', 'matchups', k, toUInt64(0)), synergy_keys) AS s2vx_cnt,
    arrayMap(k -> dictGetOrDefault('{{synergy_2vx_build_group_dict}}', 'win_rate', k, toFloat32(0.5)), synergy_bg_keys) AS s2vx_bg_raw,
    arrayMap(k -> dictGetOrDefault('{{synergy_2vx_build_group_dict}}', 'matchups', k, toUInt64(0)), synergy_bg_keys) AS s2vx_bg_cnt,
    arrayMap(k -> dictGetOrDefault('{{synergy_2vx_nobuild_dict}}', 'win_rate', k, toFloat32(0.5)), synergy_nb_keys) AS s2vx_nb_raw,
    arrayMap(k -> dictGetOrDefault('{{synergy_2vx_nobuild_dict}}', 'matchups', k, toUInt64(0)), synergy_nb_keys) AS s2vx_nb_cnt,
    arrayMap(k -> toInt16(k.1), solo_keys) AS champion_id,
    arrayMap(k -> toInt16(if(indexOf({{build_vocab}}, toString(k.3)) = 0, {{n_builds}}, indexOf({{build_vocab}}, toString(k.3)) - 1)), solo_keys) AS build_id,
    matchid
FROM (
    SELECT
        matchid,
        blue_win,
        arrayMap(p -> (tupleElement(p, 1), tupleElement(p, 2), {{key_build_expr}}), blue_players) AS blue_key_players,
        arrayMap(p -> (tupleElement(p, 1), tupleElement(p, 2), {{key_build_expr}}), red_players) AS red_key_players,
        arrayMap(p -> (tupleElement(p, 1), tupleElement(p, 2), {_KEY_BUILD_GROUP_EXPR}), blue_players) AS blue_group_players,
        arrayMap(p -> (tupleElement(p, 1), tupleElement(p, 2), {_KEY_BUILD_GROUP_EXPR}), red_players) AS red_group_players,
        arrayConcat(blue_key_players, red_key_players) AS solo_keys,
        arrayFlatten(arrayMap(b -> arrayMap(r ->
            if(b <= r, tupleConcat(b, r), tupleConcat(r, b)), red_key_players), blue_key_players)) AS matchup_keys,
        arrayFlatten(arrayMap(b -> arrayMap(r -> b > r, red_key_players), blue_key_players)) AS matchup_swapped,
        arrayFlatten(arrayMap(b -> arrayMap(r -> (b.1, b.2, r.1, r.2), red_key_players), blue_key_players)) AS matchup_nb_keys,
        arrayFlatten(arrayMap(b -> arrayMap(r -> (b.1, r.1), red_key_players), blue_key_players)) AS matchup_champ_keys,
        arrayConcat({_pair_keys("blue_key_players")}, {_pair_keys("red_key_players")}) AS synergy_keys,
        arrayConcat({_pair_keys("blue_group_players")}, {_pair_keys("red_group_players")}) AS synergy_bg_keys,
        arrayConcat({_pair_keys_nobuild("blue_key_players")}, {_pair_keys_nobuild("red_key_players")}) AS synergy_nb_keys
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

    v28 gathers latents per batch from the frozen artifact, so stale per-game
    arrays must not linger or the loader would silently prefer them.
    """
    for path in sidecar_array_paths(cache_dir).values():
        path.unlink(missing_ok=True)


# Outer-SELECT columns of _CHUNK_QUERY_TEMPLATE, by position (matchid trails them).
_RAW_COLUMNS = (
    "blue_win", "p1_raw", "p1_cnt",
    "m1v1_raw", "m1v1_cnt", "m1v1_nb_raw", "m1v1_nb_cnt", "m1v1_champ_raw", "m1v1_champ_cnt",
    "s2vx_raw", "s2vx_cnt", "s2vx_bg_raw", "s2vx_bg_cnt", "s2vx_nb_raw", "s2vx_nb_cnt",
    "champion_id", "build_id",
)

# Nested-pooling level layout: finest -> coarsest raw/count column pairs per
# interaction, plus the cache key the smoothed rate and effective N are stored.
# 2vx deliberately stops at no-build and then falls to neutral 0.5; it does not
# use champion-pair or 1vx-average floors.
_M1V1_LEVELS = (("m1v1_raw", "m1v1_cnt"), ("m1v1_nb_raw", "m1v1_nb_cnt"), ("m1v1_champ_raw", "m1v1_champ_cnt"))
_S2VX_LEVELS = (("s2vx_raw", "s2vx_cnt"), ("s2vx_bg_raw", "s2vx_bg_cnt"), ("s2vx_nb_raw", "s2vx_nb_cnt"))
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


def _level_strengths(cfg: DatasetConfig) -> dict[str, list[float]]:
    """Empirical-Bayes Beta strength per interaction level (finest -> coarsest).

    Estimated once per build from each level table's support-weighted rate
    moments via `eb_strength_from_moments`. A level whose true effects are tiny
    relative to sampling noise gets a large pseudo-count (shrink hard toward its
    parent); a level with real spread gets a small one.
    """
    client = get_client()

    def strengths(tables: tuple[tuple[str, str], ...]) -> list[float]:
        out: list[float] = []
        for table, rate_col in tables:
            r = client.query(
                f"""
                SELECT
                    sum(toFloat64({rate_col}) * matchups) / sum(matchups) AS mu,
                    sum(toFloat64({rate_col}) * toFloat64({rate_col}) * matchups)
                        / sum(matchups) AS e_sq,
                    sum(toFloat64({rate_col}) * (1 - toFloat64({rate_col})))
                        / sum(matchups) AS within_var
                FROM {table}
                WHERE split = 'train' AND matchups > 0
                """
            ).result_rows[0]
            mu, e_sq, within_var = (float(r[0]), float(r[1]), float(r[2]))
            out.append(eb_strength_from_moments(mu, e_sq - mu * mu, within_var))
        return out

    return {
        "m1v1": strengths(MATCHUP_1V1_LEVEL_TABLES),
        "s2vx": strengths(SYNERGY_2VX_LEVEL_TABLES),
    }


def _smoothed_features(
    raw: dict[str, np.ndarray],
    cfg: DatasetConfig,
    *,
    level_strengths: dict[str, list[float]],
) -> dict[str, np.ndarray]:
    smoothed = smooth_ml_prior_features(
        raw,
        prior_mean=cfg.smoothing_prior_mean,
        prior_strength=cfg.smoothing_prior_strength,
        amplification_threshold=cfg.amplification_threshold,
        smoothing_mode=cfg.smoothing_mode,
        prior_confidence_matchups=cfg.prior_confidence_matchups,
        per_side_fallback=cfg.interaction_per_side_fallback,
        nested_pooling=cfg.interaction_nested_pooling,
        level_strengths=level_strengths,
        m1v1_levels=_M1V1_LEVELS,
        s2vx_levels=_S2VX_LEVELS,
        team_pairs=TEAM_PAIRS,
        s2vx_ladder=("build", "build_group", "nobuild"),
    )

    return {
        "blue_win": raw["blue_win"],
        "win_rate": smoothed["win_rate"].astype(DISK_DTYPES["win_rate"], copy=False),
        "matchup_1v1": smoothed["matchup_1v1"].astype(
            DISK_DTYPES["matchup_1v1"],
            copy=False,
        ),
        "synergy_2vx": smoothed["synergy_2vx"].astype(
            DISK_DTYPES["synergy_2vx"],
            copy=False,
        ),
        "p1_cnt": raw["p1_cnt"].astype(DISK_DTYPES["p1_cnt"], copy=False),
        "m1v1_cnt": raw["m1v1_cnt"].astype(DISK_DTYPES["m1v1_cnt"], copy=False),
        "s2vx_cnt": raw["s2vx_cnt"].astype(DISK_DTYPES["s2vx_cnt"], copy=False),
        "m1v1_eff_n": smoothed["m1v1_eff_n"].astype(
            DISK_DTYPES["m1v1_eff_n"],
            copy=False,
        ),
        "s2vx_eff_n": smoothed["s2vx_eff_n"].astype(
            DISK_DTYPES["s2vx_eff_n"],
            copy=False,
        ),
        "champion_id": raw["champion_id"].astype(DISK_DTYPES["champion_id"], copy=False),
        "build_id": raw["build_id"].astype(DISK_DTYPES["build_id"], copy=False),
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
            matchup_1v1_dict=cfg.matchup_1v1_dict,
            synergy_2vx_dict=cfg.synergy_2vx_dict,
            matchup_1v1_nobuild_dict=cfg.matchup_1v1_nobuild_dict,
            matchup_1v1_champ_dict=cfg.matchup_1v1_champ_dict,
            synergy_2vx_build_group_dict=cfg.synergy_2vx_build_group_dict,
            synergy_2vx_nobuild_dict=cfg.synergy_2vx_nobuild_dict,
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
    level_strengths: dict[str, list[float]],
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
        block = _smoothed_features(raw, cfg, level_strengths=level_strengths)
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
    level_strengths: dict[str, list[float]],
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
                "smoothing": {
                    "prior_mean": cfg.smoothing_prior_mean,
                    "prior_strength": cfg.smoothing_prior_strength,
                    "amplification_threshold": cfg.amplification_threshold,
                    "smoothing_mode": cfg.smoothing_mode,
                    "prior_confidence_matchups": cfg.prior_confidence_matchups,
                    "interaction_per_side_fallback": cfg.interaction_per_side_fallback,
                    "interaction_loo": cfg.interaction_loo,
                    "interaction_nested_pooling": cfg.interaction_nested_pooling,
                    "interaction_level_strengths": level_strengths,
                    "s2vx_ladder": ["build", "build_group", "nobuild"],
                    "s2vx_floor_prior": "neutral_0.5",
                    "use_final_build_labels": cfg.use_final_build_labels,
                    "draft_unknown_build_label": cfg.draft_unknown_build_label,
                },
                "sources": {
                    "player_pivot_table": cfg.player_pivot_table,
                    "solo_prior_table": cfg.solo_prior_table,
                    "solo_prior_dict": cfg.solo_prior_dict,
                    "matchup_1v1_dict": cfg.matchup_1v1_dict,
                    "synergy_2vx_dict": cfg.synergy_2vx_dict,
                    "synergy_2vx_build_group_dict": cfg.synergy_2vx_build_group_dict,
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
    # path/dims in the cache meta; v28 gathers latents per batch from it instead
    # of materialising one copy per game-slot (≈3000x smaller on disk).
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
    level_strengths = (
        _level_strengths(cfg)
        if cfg.interaction_nested_pooling
        else {"m1v1": [cfg.smoothing_prior_strength], "s2vx": [cfg.smoothing_prior_strength]}
    )
    logger.info(
        "Building cache: games=%d splits=%s n_champions=%d n_builds=%d eb_strengths=%s",
        n_games,
        counts,
        n_champions,
        n_builds,
        level_strengths,
    )
    offset = 0
    for sql_split, meta_split in SPLITS:
        written = _write_split(
            arrays,
            cfg,
            split=sql_split,
            limit=counts[meta_split],
            offset=offset,
            level_strengths=level_strengths,
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

    for array in arrays.values():
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()

    return _write_meta(
        cfg,
        n_games,
        counts,
        {"n_champions": n_champions, "n_builds": n_builds, "build_vocab": build_vocab},
        level_strengths,
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
        help="Frozen three-encoder sidecar artifact to record in the v28 cache meta.",
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
