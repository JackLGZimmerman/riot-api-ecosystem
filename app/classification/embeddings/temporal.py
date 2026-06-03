"""Temporal branch: per-identity (minute_bucket, metric) tensors.

The heavy per-(identity, bucket) aggregation is materialised once into the
ClickHouse table `game_data_filtered.temporal_identity_bins` (see
`build_temporal_table`). Each row is one (split, championid, teamposition, build,
bucket) cell carrying the frame count plus a SUM for every metric. The Python
loader then reads that compact table (a few hundred K rows) instead of scanning
352M frame rows on every build.

Two metric families share one frame-count denominator:
  * 45 `tl_participant_stats` gameplay stats -- bucket mean = SUM / frames.
  * 6 events -- champion kills/assists/deaths and turret plates per lane;
    per-minute rate = SUM / frames. The frame count per bucket is exactly the
    number of that identity's games that reached the minute (one frame per
    game-minute), i.e. the game-end-time normaliser.

Every cell is frame-count shrunk toward its champion-role then global bucket
mean (the per-minute prior), then median/MAD standardised. The resulting
`(n, 47, 51)` tensor feeds `app/classification/temporal_autoencoder.py`.

Identifiers (run_id/matchid/frame_timestamp/participantid) and side-mirrored
positions (position_x/position_y) are excluded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.classification.embeddings.config import EmbeddingConfig
from app.core.utils.common import median_mad_standardise, sql_literal
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)

BINS_TABLE = "game_data_filtered.temporal_identity_bins"
CHAMPION_KILL_TABLE = "game_data.tl_champion_kill"
TURRET_PLATE_TABLE = "game_data.tl_turret_plate_destroyed"

FRAME_MS = 60000
N_BUCKETS = 47  # minutes 0..45 plus one 46_plus overflow bucket
_OVERFLOW = N_BUCKETS - 1

# Per-frame gameplay stats; bucket value is their mean over the frames observed.
TEMPORAL_METRICS: tuple[str, ...] = (
    "abilityhaste",
    "abilitypower",
    "armor",
    "armorpen",
    "armorpenpercent",
    "attackdamage",
    "attackspeed",
    "bonusarmorpenpercent",
    "bonusmagicpenpercent",
    "ccreduction",
    "cooldownreduction",
    "health",
    "healthmax",
    "healthregen",
    "lifesteal",
    "magicpen",
    "magicpenpercent",
    "magicresist",
    "movementspeed",
    "omnivamp",
    "physicalvamp",
    "power",
    "powermax",
    "powerregen",
    "spellvamp",
    "currentgold",
    "magicdamagedone",
    "magicdamagedonetochampions",
    "magicdamagetaken",
    "physicaldamagedone",
    "physicaldamagedonetochampions",
    "physicaldamagetaken",
    "totaldamagedone",
    "totaldamagedonetochampions",
    "totaldamagetaken",
    "truedamagedone",
    "truedamagedonetochampions",
    "truedamagetaken",
    "goldpersecond",
    "jungleminionskilled",
    "level",
    "minionskilled",
    "timeenemyspentcontrolled",
    "totalgold",
    "xp",
)

# Event counts (per minute); same frame-count denominator as the stats above.
# Each maps an event row to a participant: kills/plates by killer, deaths by
# victim, assists by every assisting participant.
EVENT_METRICS: tuple[str, ...] = (
    "kills",
    "assists",
    "deaths",
    "plate_top",
    "plate_mid",
    "plate_bot",
)

METRIC_NAMES: tuple[str, ...] = (*TEMPORAL_METRICS, *EVENT_METRICS)

# Frame-count shrinkage strengths (in frames). A cell with this many frames is
# pulled halfway to its parent; well-sampled early buckets barely move while
# sparse late-game buckets lean on the parent.
PARENT_STRENGTH = 100.0  # identity -> champion-role bucket mean
GLOBAL_STRENGTH = 1000.0  # champion-role -> global bucket mean


def _bucket(ts_col: str) -> str:
    return f"least(intDiv({ts_col}, {FRAME_MS}), {_OVERFLOW})"


@dataclass(frozen=True)
class TemporalTensors:
    keys: list[tuple]  # (championid, teamposition, build) per axis-0 row
    values: np.ndarray  # (n, N_BUCKETS, n_metric) float32, smoothed + standardised
    mask: np.ndarray  # (n, N_BUCKETS) bool, True where the bucket was observed
    metric_names: tuple[str, ...]


def _load_bins(split: str) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Read the prepared table -> (keys, sums (n,B,M), frames (n,B))."""
    select_cols = ", ".join(
        ["championid", "teamposition", "build", "bucket", "frames"]
        + [f"sum_{m}" for m in TEMPORAL_METRICS]
        + [f"ev_{m}" for m in EVENT_METRICS]
    )
    sql = (
        f"SELECT {select_cols} FROM {BINS_TABLE}"
        f" WHERE split = {sql_literal(split)}"
    )
    rows = get_client().query(sql).result_rows
    n_metric = len(METRIC_NAMES)
    index: dict[tuple, int] = {}
    sum_rows: list[np.ndarray] = []
    cnt_rows: list[np.ndarray] = []
    for row in rows:
        key = (int(row[0]), str(row[1]), str(row[2]))
        idx = index.get(key)
        if idx is None:
            idx = len(sum_rows)
            index[key] = idx
            sum_rows.append(np.zeros((N_BUCKETS, n_metric), dtype=np.float64))
            cnt_rows.append(np.zeros(N_BUCKETS, dtype=np.float64))
        bucket = int(row[3])
        cnt_rows[idx][bucket] = float(row[4] or 0.0)
        sum_rows[idx][bucket, :] = [float(v or 0.0) for v in row[5:]]
    keys = list(index)
    if not keys:
        raise RuntimeError(
            f"{BINS_TABLE} has no rows for split={split!r}; run build_temporal_table()"
        )
    return keys, np.stack(sum_rows), np.stack(cnt_rows)


def _shrink(sums: np.ndarray, counts: np.ndarray, keys: list[tuple]) -> np.ndarray:
    """Frame-count shrink each (identity, bucket, metric) toward parents.

    identity -> champion-role bucket mean -> global bucket mean. Unobserved
    identity cells fall back to the champion-role (then global) bucket mean.
    """
    counts_b = counts[:, :, None]  # (n, B, 1)
    id_mean = np.divide(sums, counts_b, out=np.zeros_like(sums), where=counts_b > 0)

    role_of: dict[int, tuple] = {}
    role_ids: dict[tuple, list[int]] = {}
    for i, (cid, pos, _build) in enumerate(keys):
        role = (cid, pos)
        role_of[i] = role
        role_ids.setdefault(role, []).append(i)

    global_sum = sums.sum(axis=0)  # (B, M)
    global_cnt = counts.sum(axis=0)[:, None]  # (B, 1)
    global_mean = np.divide(
        global_sum, global_cnt, out=np.zeros_like(global_sum), where=global_cnt > 0
    )

    smoothed = np.zeros_like(sums)
    role_smoothed_cache: dict[tuple, np.ndarray] = {}
    for role, members in role_ids.items():
        r_sum = sums[members].sum(axis=0)  # (B, M)
        r_cnt = counts[members].sum(axis=0)[:, None]  # (B, 1)
        r_mean = np.divide(r_sum, r_cnt, out=np.zeros_like(r_sum), where=r_cnt > 0)
        role_smoothed_cache[role] = (r_cnt * r_mean + GLOBAL_STRENGTH * global_mean) / (
            r_cnt + GLOBAL_STRENGTH
        )

    for i in range(sums.shape[0]):
        parent = role_smoothed_cache[role_of[i]]  # (B, M)
        c = counts[i][:, None]  # (B, 1)
        smoothed[i] = (c * id_mean[i] + PARENT_STRENGTH * parent) / (c + PARENT_STRENGTH)
    return smoothed.astype(np.float64)


def _standardise(smoothed: np.ndarray, clip_value: float | None) -> np.ndarray:
    n, n_bucket, n_metric = smoothed.shape
    flat = smoothed.reshape(n, n_bucket * n_metric)
    std, _, _ = median_mad_standardise(flat)
    if clip_value is not None:
        std = np.clip(std, -float(clip_value), float(clip_value))
    return std.reshape(n, n_bucket, n_metric).astype(np.float32)


def _cache_path(cfg: EmbeddingConfig) -> Path:
    return cfg.cache_dir / "_raw" / f"{cfg.split}_temporal.npz"


def build_temporal_tensors(
    cfg: EmbeddingConfig | None = None, *, use_cache: bool = True
) -> TemporalTensors:
    cfg = cfg or EmbeddingConfig()
    path = _cache_path(cfg)
    if use_cache and path.exists():
        payload = np.load(path, allow_pickle=True)
        logger.info("Loaded cached temporal tensors: %s", path.name)
        return TemporalTensors(
            keys=[tuple(k) for k in payload["keys"].tolist()],
            values=payload["values"],
            mask=payload["mask"],
            metric_names=tuple(str(m) for m in payload["metric_names"].tolist()),
        )

    keys, sums, counts = _load_bins(cfg.split)
    smoothed = _shrink(sums, counts, keys)
    values = _standardise(smoothed, cfg.matrix_clip_value)
    mask = counts > 0
    tensors = TemporalTensors(keys, values, mask, METRIC_NAMES)

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            keys=np.asarray(keys, dtype=object),
            values=values,
            mask=mask,
            metric_names=np.asarray(METRIC_NAMES, dtype=object),
        )
    logger.info("Temporal tensors: %s, observed cells %d", values.shape, int(mask.sum()))
    return tensors
