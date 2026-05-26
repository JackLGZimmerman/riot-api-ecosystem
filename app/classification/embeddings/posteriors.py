"""Smooth 6010 baseline rows with 9000-9040 Bayesian priors.

Each baseline row is a concrete (champion, role, build, phase) combination.
The prior tables are looked up against that row with decreasing contextual
relevance:

    9000 sibling build       same champion, same role, sibling build
    9010 champion_role       same champion, same role, all builds
    9020 role_build          same role, similar build group
    9030 champion_build      same champion, similar build group
    9040 build               similar build group

The smoothed baseline rows are the only rows passed into matrix construction.
"""

from __future__ import annotations

import logging

import numpy as np

from app.classification.embeddings.config import (
    LEVEL_KEY,
    PER_MINUTE_METRICS,
    PRIOR_LEVELS,
    RATE_LIKE_METRICS,
    ALL_METRICS,
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.load import LevelRows

logger = logging.getLogger(__name__)


def _row_keys(rows: LevelRows, key_cols: tuple[str, ...]) -> list[tuple]:
    cols = [rows.columns[c] for c in key_cols] + [rows.columns["phase"]]
    return [tuple(c[i] for c in cols) for i in range(rows.n)]


def _lookup_prior(
    target: LevelRows,
    prior: LevelRows,
    metrics: tuple[str, ...],
) -> dict[str, np.ndarray]:
    prior_keys_per_row = _row_keys(prior, prior.key_columns)
    prior_idx_by_key = {k: i for i, k in enumerate(prior_keys_per_row)}
    target_keys_per_row = _row_keys(target, prior.key_columns)
    target_prior_idx = np.array(
        [prior_idx_by_key.get(k, -1) for k in target_keys_per_row],
        dtype=np.int64,
    )
    valid = target_prior_idx >= 0

    def take(name: str) -> np.ndarray:
        values = np.full(target.n, np.nan, dtype=np.float64)
        if valid.any():
            values[valid] = prior.columns[name][target_prior_idx[valid]].astype(
                np.float64
            )
        return values

    return {"valid": valid, **{name: take(name) for name in ("matchups", *metrics)}}


def _prior_weight(
    level: IdentityType,
    lookup: dict[str, np.ndarray],
    cfg: EmbeddingConfig,
    *,
    per_minute: bool,
) -> np.ndarray:
    strengths = (
        cfg.prior_per_minute_strengths if per_minute else cfg.prior_rate_strengths
    )
    cap = float(strengths.get(level, 0.0))
    if cap <= 0.0:
        return np.zeros_like(lookup["matchups"], dtype=np.float64)
    valid = lookup["valid"] & np.isfinite(lookup["matchups"])
    if not per_minute:
        return np.where(valid, np.minimum(lookup["matchups"], cap), 0.0)
    rate_cap = float(cfg.prior_rate_strengths.get(level, 0.0))
    reliability = (
        np.ones_like(lookup["matchups"])
        if rate_cap <= 0.0
        else np.minimum(lookup["matchups"] / rate_cap, 1.0)
    )
    return np.where(valid, cap * reliability, 0.0)


def _smooth_metric(
    target: LevelRows,
    metric: str,
    obs_weight: np.ndarray,
    prior_lookups: dict[IdentityType, dict[str, np.ndarray]],
    prior_weights: dict[IdentityType, np.ndarray],
) -> np.ndarray:
    obs = target.columns[metric].astype(np.float64)
    numerator = obs_weight * obs
    denominator = obs_weight.copy()

    for level in PRIOR_LEVELS:
        lookup = prior_lookups.get(level)
        if lookup is None:
            continue
        weight = prior_weights[level]
        prior_value = lookup[metric]
        valid = weight > 0.0
        numerator[valid] += weight[valid] * prior_value[valid]
        denominator[valid] += weight[valid]

    return np.divide(
        numerator,
        denominator,
        out=obs.copy(),
        where=denominator > 0.0,
    ).astype(np.float32)


def _smooth_metrics(
    target: LevelRows,
    metrics: tuple[str, ...],
    obs_weight: np.ndarray,
    prior_lookups: dict[IdentityType, dict[str, np.ndarray]],
    prior_weights: dict[IdentityType, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        f"smoothed_{metric}": _smooth_metric(
            target, metric, obs_weight, prior_lookups, prior_weights
        )
        for metric in metrics
    }


def _ratio(numerator: np.ndarray, denominator_base: np.ndarray) -> np.ndarray:
    denominator = denominator_base + numerator
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    ).astype(np.float32)


def _add_weight_columns(
    new_cols: dict[str, np.ndarray],
    weights: dict[IdentityType, np.ndarray],
    prefix: str,
    n: int,
) -> np.ndarray:
    total = np.zeros(n, dtype=np.float64)
    for level, weight in weights.items():
        total += weight
        new_cols[f"{prefix}_{level.value}"] = weight.astype(np.float32)
    return total


def _posterior_std(
    target: LevelRows,
    matchups: np.ndarray,
    prior_lookups: dict[IdentityType, dict[str, np.ndarray]],
    rate_weights: dict[IdentityType, np.ndarray],
) -> np.ndarray:
    alpha = matchups * target.columns["win"].astype(np.float64)
    beta = matchups * (1.0 - target.columns["win"].astype(np.float64))
    for level, weight in rate_weights.items():
        valid = weight > 0.0
        alpha[valid] += weight[valid] * prior_lookups[level]["win"][valid]
        beta[valid] += weight[valid] * (1.0 - prior_lookups[level]["win"][valid])

    total = alpha + beta
    var = np.divide(
        alpha * beta,
        total * total * (total + 1.0),
        out=np.zeros_like(total),
        where=total > 0.0,
    )
    return np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)


def _smooth_baseline(
    target: LevelRows,
    priors: dict[IdentityType, LevelRows],
    cfg: EmbeddingConfig,
) -> LevelRows:
    prior_lookups = {
        level: _lookup_prior(target, priors[level], ALL_METRICS)
        for level in PRIOR_LEVELS
        if level in priors and priors[level].n > 0
    }
    matchups = target.columns["matchups"].astype(np.float64)
    threshold = float(cfg.extreme_low_sample_threshold)
    amplification = 1.0 + threshold / np.maximum(matchups, 1.0)
    rate_weights = {
        level: _prior_weight(level, lookup, cfg, per_minute=False) * amplification
        for level, lookup in prior_lookups.items()
    }
    per_minute_weights = {
        level: _prior_weight(level, lookup, cfg, per_minute=True) * amplification
        for level, lookup in prior_lookups.items()
    }

    for level in PRIOR_LEVELS:
        lookup = prior_lookups.get(level)
        if lookup is None:
            logger.warning("Prior %s unavailable for smoothing", level.value)
            continue
        coverage = float(lookup["valid"].mean()) if target.n else 0.0
        logger.info(
            "Prior %s coverage=%.3f mean_rate_weight=%.2f mean_pm_weight=%.2f",
            level.value,
            coverage,
            float(rate_weights[level].mean()) if target.n else 0.0,
            float(per_minute_weights[level].mean()) if target.n else 0.0,
        )

    sum_w_timeplayed = target.columns["sum_w_timeplayed"].astype(np.float64)
    new_cols = _smooth_metrics(
        target, RATE_LIKE_METRICS, matchups, prior_lookups, rate_weights
    )
    new_cols.update(
        _smooth_metrics(
            target,
            PER_MINUTE_METRICS,
            sum_w_timeplayed,
            prior_lookups,
            per_minute_weights,
        )
    )
    total_rate_prior = _add_weight_columns(
        new_cols, rate_weights, "prior_rate_weight", target.n
    )
    total_per_minute_prior = _add_weight_columns(
        new_cols, per_minute_weights, "prior_per_minute_weight", target.n
    )
    new_cols["prior_weight_ratio"] = _ratio(total_rate_prior, matchups)
    new_cols["prior_per_minute_weight_ratio"] = _ratio(
        total_per_minute_prior, sum_w_timeplayed
    )
    new_cols["posterior_std"] = _posterior_std(
        target, matchups, prior_lookups, rate_weights
    )

    smoothed = target.with_columns(new_cols)
    logger.info("Smoothed %s: %d rows", target.level.value, smoothed.n)
    return smoothed


def apply_hierarchical_shrinkage(
    levels: dict[IdentityType, LevelRows],
    cfg: EmbeddingConfig,
) -> dict[IdentityType, LevelRows]:
    """Return only the smoothed baseline rows that should be embedded."""
    target = levels.get(IdentityType.BASELINE)
    if target is None:
        return {
            IdentityType.BASELINE: LevelRows(
                IdentityType.BASELINE,
                LEVEL_KEY[IdentityType.BASELINE],
                {},
                0,
            )
        }
    return {
        IdentityType.BASELINE: _smooth_baseline(
            target,
            {level: rows for level, rows in levels.items() if level in PRIOR_LEVELS},
            cfg,
        )
    }
