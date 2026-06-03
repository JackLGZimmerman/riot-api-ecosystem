"""Static champion base-stats branch (Phase 2).

Per-champion base stats from `champion_basic_stats_flat.jsonl`, joined on the
numeric `championid`. These are deterministic champion constants, so they get no
Bayesian priors: they are signed-log1p compressed and median/MAD standardised
exactly like the full-game block. See METRIC_CATALOGUE_PLAN.md.
"""

from __future__ import annotations

import json

import numpy as np

from app.core.config.settings import PROJECT_ROOT
from app.core.utils.common import median_mad_standardise

STATIC_STATS_PATH = (
    PROJECT_ROOT
    / "database"
    / "clickhouse"
    / "support"
    / "champion_basic_stats_flat.jsonl"
)

# Stats whose in-game value scales linearly with level: value at level 18 is
# flat + 17 * perLevel. attackSpeed grows by a percentage formula and the
# radius/timing stats are ~constant, so they are excluded from the l18 derivation.
LEVEL18_STATS: tuple[str, ...] = (
    "health",
    "healthRegen",
    "mana",
    "manaRegen",
    "armor",
    "magicResistance",
    "attackDamage",
)

_EXCLUDED_KEYS = frozenset({"_key", "id"})
_RECORDS_CACHE: list[dict] | None = None


def _load_records() -> list[dict]:
    global _RECORDS_CACHE
    if _RECORDS_CACHE is None:
        with STATIC_STATS_PATH.open() as fh:
            _RECORDS_CACHE = [json.loads(line) for line in fh if line.strip()]
    return _RECORDS_CACHE


def static_stat_columns() -> tuple[str, ...]:
    """Raw stat columns in source order (every column except `_key`/`id`)."""
    first = _load_records()[0]
    return tuple(key for key in first if key not in _EXCLUDED_KEYS)


def static_feature_names() -> tuple[str, ...]:
    """Raw stat columns followed by the level-18 derived stats."""
    return (*static_stat_columns(), *(f"{stat}_l18" for stat in LEVEL18_STATS))


def load_static_by_id() -> dict[int, np.ndarray]:
    """Map numeric championid -> raw static feature vector (pre-standardise)."""
    columns = static_stat_columns()
    out: dict[int, np.ndarray] = {}
    for record in _load_records():
        base = [float(record[col]) for col in columns]
        derived = [
            float(record[f"{stat}_flat"]) + 17.0 * float(record[f"{stat}_perLevel"])
            for stat in LEVEL18_STATS
        ]
        out[int(record["id"])] = np.asarray(base + derived, dtype=np.float64)
    return out


def build_static_matrix(
    champion_ids: np.ndarray,
    *,
    clip_value: float | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Aligned, standardised static-stat matrix for the given champion ids.

    One row per `champion_ids` entry (champions repeat across identity rows).
    Unknown ids contribute a zero raw row. Returns (matrix, feature_names).
    """
    by_id = load_static_by_id()
    names = static_feature_names()
    raw = np.zeros((len(champion_ids), len(names)), dtype=np.float64)
    for i, cid in enumerate(champion_ids):
        vec = by_id.get(int(cid))
        if vec is not None:
            raw[i] = vec
    standardised, _, _ = median_mad_standardise(raw)
    if clip_value is not None:
        clip = float(clip_value)
        standardised = np.clip(standardised, -clip, clip)
    return standardised.astype(np.float32), names
