"""Shared Bayesian shrinkage primitives for win-rate / metric smoothing.

Single source of truth for the smoothing math used by both the win-rate model
(`app/ml`) and the classification embeddings (`app/classification`). Changes to
the shrink equation or the dynamic low-sample weighting propagate to both.

Core primitives:

* build-group helpers — the shared build sibling policy used by the ML
  interaction backoff and classification prior tables.
* `bayesian_smoothed_rate` — the core shrink ``prior + (obs - prior)·n/(n + s)``.
  `prior_strength` may be a scalar or a per-element array (the array form is what
  makes dynamic, n-dependent strength possible).
* `amplification_factor` — ``sqrt(1 + threshold/max(n, 1))``. A multiplier that
  grows the prior's weight for low-sample identities (→ 1 as n grows).
* `dynamic_smoothed_rate` — `bayesian_smoothed_rate` with the strength amplified
  per element, so under-sampled identities are pulled harder toward the prior
  while well-sampled ones keep the base strength.
* `smooth_rate_by_mode` — the common additive/cascade dispatcher for callers that
  expose smoothing mode in config.
* `cascade_dynamic_smoothed_rate` — ML-facing cascade smoothing that stops
  applying a broad prior once the contextual sample count is confident.
* `cascade_selection` — hierarchy-facing cascade selection for choosing exactly
  one prior level per row.
* metric prior helpers — generic weighted blending used by classification
  embeddings after their prior lookups are materialised.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PRIOR_MEAN = 0.5
DEFAULT_PRIOR_STRENGTH = 20.0
S2VX_NEUTRAL_FLOOR_LADDER = ("build", "build_group", "nobuild")
LevelT = TypeVar("LevelT", bound=Hashable)

# Build siblings used by both classification priors and ML interaction backoff.
# Unlisted build labels are their own group.
BUILD_GROUPS: dict[str, tuple[str, ...]] = {
    "ap": ("ability_power", "ap_off_tank"),
    "ad": ("attack_damage", "ad_off_tank"),
    "tank": ("ar_tank", "mr_tank"),
    "utility": ("utility_enchanter", "utility_protection"),
}
BUILD_TO_GROUP: dict[str, str] = {
    build: group for group, builds in BUILD_GROUPS.items() for build in builds
}
SIBLING_BUILD_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (labels[0], labels[1]) for labels in BUILD_GROUPS.values()
)
SIBLING_BUILD_BY_LABEL: dict[str, str] = {}
for _left, _right in SIBLING_BUILD_PAIRS:
    SIBLING_BUILD_BY_LABEL.update({_left: _right, _right: _left})


def build_group_for(build: str) -> str:
    """Return the smoothing build group for `build`, defaulting to the label itself."""
    return BUILD_TO_GROUP.get(build, build)


def _sql_strings(values: Iterable[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def build_group_sql(column: str = "build", alias: str | None = "build_group") -> str:
    """ClickHouse expression that maps a build label to the shared build group."""
    clauses = ", ".join(
        f"{column} IN ({_sql_strings(labels)}), '{group}'"
        for group, labels in BUILD_GROUPS.items()
    )
    expr = f"multiIf({clauses}, {column})"
    return f"{expr} AS {alias}" if alias else expr


def sibling_build_sql(column: str = "build") -> str:
    """ClickHouse expression that maps a build label to its configured sibling."""
    clauses = ", ".join(
        f"{column} = '{source}', '{target}'"
        for source, target in SIBLING_BUILD_BY_LABEL.items()
    )
    return f"multiIf({clauses}, '')"


def bayesian_smoothed_rate(
    win_rate: np.ndarray,
    sample_count: np.ndarray,
    *,
    prior_mean: float | np.ndarray = DEFAULT_PRIOR_MEAN,
    prior_strength: float | np.ndarray = DEFAULT_PRIOR_STRENGTH,
) -> np.ndarray:
    """Shrink empirical rates toward a prior mean.

    `prior_mean` is a scalar prior (e.g. ``0.5``) or a per-element array of the
    same shape as `win_rate`. The array form is used for the per-side
    interaction fallback: each pair is shrunk toward a composite of its two
    sides' individual priors instead of a flat ``0.5``, so low-sample pairs get
    a real signal rather than the no-information default.

    `prior_strength` is measured in pseudo-games and may be a scalar or a
    per-element array matching `win_rate` (the array form supports dynamic,
    sample-dependent strength). With the scalar default, a ``0.5`` prior is
    equivalent to adding 10 wins and 10 losses to each observed aggregate.
    """
    prior = np.asarray(prior_mean, dtype=np.float64)
    strength = np.asarray(prior_strength, dtype=np.float64)
    if np.any(prior < 0.0) or np.any(prior > 1.0):
        raise ValueError("prior_mean must be between 0.0 and 1.0")
    if np.any(strength < 0.0):
        raise ValueError("prior_strength must be non-negative")

    rates = np.asarray(win_rate, dtype=np.float64)
    counts = np.asarray(sample_count, dtype=np.float64)
    if rates.shape != counts.shape:
        raise ValueError(
            f"win_rate and sample_count must have the same shape, got "
            f"{rates.shape} and {counts.shape}"
        )
    if prior.ndim != 0 and prior.shape != rates.shape:
        raise ValueError(
            f"prior_mean array must match win_rate shape {rates.shape}, "
            f"got {prior.shape}"
        )
    if strength.ndim != 0 and strength.shape != rates.shape:
        raise ValueError(
            f"prior_strength array must match win_rate shape {rates.shape}, "
            f"got {strength.shape}"
        )
    if np.any(counts < 0.0):
        raise ValueError("sample_count cannot contain negative values")

    denom = counts + strength
    # denom == 0 only when both n and strength are 0 (no information): return the
    # prior rather than dividing by zero.
    return np.where(
        denom > 0.0,
        prior + (rates - prior) * counts / np.where(denom > 0.0, denom, 1.0),
        prior,
    )


def amplification_factor(
    sample_count: np.ndarray, amplification_threshold: float
) -> np.ndarray:
    """Low-sample prior-weight multiplier ``sqrt(1 + amplification_threshold/max(n, 1))``.

    Equals 1 when `amplification_threshold` is 0; approaches
    ``sqrt(1 + amplification_threshold)`` as n → 0 and decays toward 1 as n grows
    past `amplification_threshold`. Multiplying a base prior strength by this makes
    under-sampled identities shrink harder toward the prior while well-sampled ones
    are left near the base strength.
    """
    if amplification_threshold < 0.0:
        raise ValueError("amplification_threshold must be non-negative")
    counts = np.asarray(sample_count, dtype=np.float64)
    return np.sqrt(1.0 + amplification_threshold / np.maximum(counts, 1.0))


def dynamic_smoothed_rate(
    win_rate: np.ndarray,
    sample_count: np.ndarray,
    *,
    prior_mean: float | np.ndarray = DEFAULT_PRIOR_MEAN,
    base_strength: float = DEFAULT_PRIOR_STRENGTH,
    amplification_threshold: float = 0.0,
) -> np.ndarray:
    """`bayesian_smoothed_rate` with sample-dependent (amplified) strength.

    Effective strength per element is
    ``base_strength * amplification_factor(n, amplification_threshold)``.
    With `amplification_threshold == 0` this is identical to
    `bayesian_smoothed_rate` at the scalar `base_strength`.
    """
    effective = base_strength * amplification_factor(
        sample_count, amplification_threshold
    )
    return bayesian_smoothed_rate(
        win_rate,
        sample_count,
        prior_mean=prior_mean,
        prior_strength=effective,
    )


def cascade_dynamic_smoothed_rate(
    win_rate: np.ndarray,
    sample_count: np.ndarray,
    *,
    prior_mean: float | np.ndarray = DEFAULT_PRIOR_MEAN,
    base_strength: float = DEFAULT_PRIOR_STRENGTH,
    amplification_threshold: float = 0.0,
    confidence_threshold: float = 50.0,
) -> np.ndarray:
    """Use the contextual rate directly once its own support is confident.

    This is the ML analogue of hierarchical cascade smoothing: the observed
    contextual prior wins when its sample count clears `confidence_threshold`;
    otherwise it shrinks toward the broader fallback prior with the normal
    dynamic low-sample weighting.
    """
    counts = np.asarray(sample_count, dtype=np.float64)
    effective = base_strength * amplification_factor(counts, amplification_threshold)
    effective = np.where(counts >= float(confidence_threshold), 0.0, effective)
    return bayesian_smoothed_rate(
        win_rate,
        counts,
        prior_mean=prior_mean,
        prior_strength=effective,
    )


def smooth_rate_by_mode(
    win_rate: np.ndarray,
    sample_count: np.ndarray,
    *,
    prior_mean: float | np.ndarray = DEFAULT_PRIOR_MEAN,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    amplification_threshold: float = 0.0,
    smoothing_mode: str = "cascade",
    confidence_threshold: float = 50.0,
) -> np.ndarray:
    """Smooth rates using the shared additive/cascade config vocabulary."""
    if smoothing_mode == "additive":
        return dynamic_smoothed_rate(
            win_rate,
            sample_count,
            prior_mean=prior_mean,
            base_strength=prior_strength,
            amplification_threshold=amplification_threshold,
        )
    if smoothing_mode == "cascade":
        return cascade_dynamic_smoothed_rate(
            win_rate,
            sample_count,
            prior_mean=prior_mean,
            base_strength=prior_strength,
            amplification_threshold=amplification_threshold,
            confidence_threshold=confidence_threshold,
        )
    raise ValueError(f"Unsupported smoothing_mode: {smoothing_mode!r}")


def composite_interaction_priors(
    win_rate: np.ndarray,
    team_pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-side fallback priors for 1v1 and same-team 2vx interactions.

    `win_rate` is ordered blue players then red players. The 1v1 prior is
    blue-perspective: ``0.5 + (blue_wr - red_wr) / 2``. The 2vx prior is each
    same-team pair's average solo posterior.
    """
    rates = np.asarray(win_rate, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[1] % 2 != 0:
        raise ValueError("win_rate must have shape (n, 2 * players_per_team)")

    players_per_team = rates.shape[1] // 2
    blue = rates[:, :players_per_team]
    red = rates[:, players_per_team:]
    comp_1v1 = (0.5 + (blue[:, :, None] - red[:, None, :]) / 2.0).reshape(
        rates.shape[0],
        players_per_team * players_per_team,
    )
    blue_pairs = [(blue[:, i] + blue[:, j]) / 2.0 for i, j in team_pairs]
    red_pairs = [(red[:, i] + red[:, j]) / 2.0 for i, j in team_pairs]
    return comp_1v1, np.column_stack(blue_pairs + red_pairs)


def smooth_interaction_levels(
    rates: Sequence[np.ndarray],
    counts: Sequence[np.ndarray],
    *,
    strengths: Sequence[float],
    floor_prior: float | np.ndarray,
    nested_pooling: bool,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    amplification_threshold: float = 0.0,
    smoothing_mode: str = "cascade",
    confidence_threshold: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth an interaction ladder with nested EB or legacy single-level fallback."""
    if nested_pooling:
        return nested_shrunk_rate(
            rates,
            counts,
            strengths=strengths,
            floor_prior=floor_prior,
            amplification_threshold=amplification_threshold,
        )
    if not rates or not counts:
        raise ValueError("smooth_interaction_levels requires at least one level")
    return (
        smooth_rate_by_mode(
            rates[0],
            counts[0],
            prior_mean=floor_prior,
            prior_strength=prior_strength,
            amplification_threshold=amplification_threshold,
            smoothing_mode=smoothing_mode,
            confidence_threshold=confidence_threshold,
        ),
        np.asarray(counts[0], dtype=np.float64),
    )


def s2vx_floor_prior_for_ladder(
    ladder: Sequence[str],
    *,
    neutral_prior: float | np.ndarray,
    per_side_prior: float | np.ndarray,
) -> float | np.ndarray:
    """Return the terminal 2vX floor prior for an interaction smoothing ladder."""
    if tuple(ladder) == S2VX_NEUTRAL_FLOOR_LADDER:
        return neutral_prior
    return per_side_prior


def smooth_ml_prior_features(
    raw: Mapping[str, np.ndarray],
    *,
    prior_mean: float,
    prior_strength: float,
    amplification_threshold: float,
    smoothing_mode: str,
    prior_confidence_matchups: float,
    per_side_fallback: bool,
    nested_pooling: bool,
    level_strengths: Mapping[str, Sequence[float]],
    m1v1_levels: Sequence[tuple[str, str]],
    s2vx_levels: Sequence[tuple[str, str]],
    team_pairs: Sequence[tuple[int, int]],
    s2vx_ladder: Sequence[str] | None = None,
    s2vx_floor_prior: float | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Smooth the ML cache/runtime prior feature arrays.

    The caller supplies raw lookup arrays and the active level ladders; this
    function owns the Bayesian choices for solo priors and interaction pooling.
    """
    win_rate = smooth_rate_by_mode(
        raw["p1_raw"],
        raw["p1_cnt"],
        prior_mean=prior_mean,
        prior_strength=prior_strength,
        amplification_threshold=amplification_threshold,
        smoothing_mode=smoothing_mode,
        confidence_threshold=prior_confidence_matchups,
    )
    prior_1v1, prior_2vx = (
        composite_interaction_priors(win_rate, team_pairs)
        if per_side_fallback
        else (prior_mean, prior_mean)
    )

    matchup_1v1, m1v1_eff_n = smooth_interaction_levels(
        [raw[rk] for rk, _ in m1v1_levels],
        [raw[ck] for _, ck in m1v1_levels],
        strengths=level_strengths["m1v1"],
        floor_prior=prior_1v1,
        nested_pooling=nested_pooling,
        prior_strength=prior_strength,
        amplification_threshold=amplification_threshold,
        smoothing_mode=smoothing_mode,
        confidence_threshold=prior_confidence_matchups,
    )
    synergy_2vx, s2vx_eff_n = smooth_interaction_levels(
        [raw[rk] for rk, _ in s2vx_levels],
        [raw[ck] for _, ck in s2vx_levels],
        strengths=level_strengths["s2vx"],
        floor_prior=(
            s2vx_floor_prior
            if s2vx_floor_prior is not None
            else s2vx_floor_prior_for_ladder(
                s2vx_ladder or (),
                neutral_prior=prior_mean,
                per_side_prior=prior_2vx,
            )
        ),
        nested_pooling=nested_pooling,
        prior_strength=prior_strength,
        amplification_threshold=amplification_threshold,
        smoothing_mode=smoothing_mode,
        confidence_threshold=prior_confidence_matchups,
    )
    return {
        "win_rate": win_rate,
        "matchup_1v1": matchup_1v1,
        "synergy_2vx": synergy_2vx,
        "m1v1_eff_n": m1v1_eff_n,
        "s2vx_eff_n": s2vx_eff_n,
    }


def eb_strength_from_moments(
    prior_mean: float,
    total_var: float,
    within_var: float,
    *,
    min_strength: float = 1.0,
    max_strength: float = 10_000.0,
    default: float = DEFAULT_PRIOR_STRENGTH,
) -> float:
    """Method-of-moments Beta-Binomial concentration (pseudo-count) ``kappa``.

    Given the population moments of a level's empirical cell rates:

    * ``prior_mean`` = support-weighted mean rate ``mu``,
    * ``total_var``  = variance of the observed cell rates,
    * ``within_var`` = mean sampling variance ``E[r(1-r)/n]``,

    the between-cell (true effect) variance is ``total_var - within_var`` and the
    Beta concentration that reproduces it is ``mu(1-mu)/between_var - 1``. Large
    when true effects are tiny (shrink hard), small when they spread out. Falls
    back to `default` for degenerate inputs (no spread, non-finite).
    """
    mu = float(prior_mean)
    spread = mu * (1.0 - mu)
    between = float(total_var) - float(within_var)
    if not np.isfinite(between) or between <= 0.0 or spread <= 0.0:
        return float(default)
    kappa = spread / between - 1.0
    if not np.isfinite(kappa):
        return float(default)
    return float(np.clip(kappa, min_strength, max_strength))


def eb_strength(
    win_rate: np.ndarray,
    sample_count: np.ndarray,
    **kwargs: float,
) -> float:
    """`eb_strength_from_moments` computed from a level's rate/count arrays.

    Cells with zero support carry no information and are dropped. The mean is
    support-weighted; ``within_var`` is the mean per-cell sampling variance.
    """
    rates = np.asarray(win_rate, dtype=np.float64).reshape(-1)
    counts = np.asarray(sample_count, dtype=np.float64).reshape(-1)
    valid = counts > 0.0
    if not np.any(valid):
        return float(kwargs.get("default", DEFAULT_PRIOR_STRENGTH))
    rates, counts = rates[valid], counts[valid]
    mu = float(np.sum(rates * counts) / np.sum(counts))
    total_var = float(np.average((rates - mu) ** 2, weights=counts))
    # Support-weighted mean sampling variance: sum(r(1-r)) / sum(n), matching the
    # ClickHouse moments the cache builder feeds to eb_strength_from_moments.
    within_var = float(np.sum(rates * (1.0 - rates)) / np.sum(counts))
    return eb_strength_from_moments(mu, total_var, within_var, **kwargs)


def nested_shrunk_rate(
    rates: Sequence[np.ndarray],
    counts: Sequence[np.ndarray],
    *,
    strengths: Sequence[float],
    floor_prior: float | np.ndarray = DEFAULT_PRIOR_MEAN,
    amplification_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nested empirical-Bayes partial pooling, finest level first.

    `rates`/`counts` are per-level arrays ordered most→least specific (L0..Lk);
    `strengths` is the matching per-level Beta pseudo-count. Each level is shrunk
    toward the posterior of its parent (the next-coarser level), the coarsest
    toward `floor_prior`. Reuses `bayesian_smoothed_rate` per level.

    Returns ``(posterior_mean, effective_n)``. The effective sample size inherits
    the parent's support in proportion to how much the estimate leaned on it
    (``n_eff = n + (1-w)·n_eff_parent`` with ``w = n/(n+kappa)``), so a sparse
    finest cell backed by a dense parent is treated as well-supported by the
    φ-gate rather than suppressed.
    """
    if not (len(rates) == len(counts) == len(strengths)):
        raise ValueError("rates, counts, and strengths must be the same length")
    if not rates:
        raise ValueError("nested_shrunk_rate requires at least one level")

    prior = np.broadcast_to(
        np.asarray(floor_prior, dtype=np.float64), np.asarray(rates[0]).shape
    ).astype(np.float64)
    prior_neff = np.zeros_like(prior)
    # Coarsest -> finest, so the finest posterior is produced last.
    for rate, count, base in zip(
        reversed(rates), reversed(counts), reversed(list(strengths))
    ):
        count_f = np.asarray(count, dtype=np.float64)
        kappa = float(base) * amplification_factor(count_f, amplification_threshold)
        mean = bayesian_smoothed_rate(
            np.asarray(rate, dtype=np.float64),
            count_f,
            prior_mean=prior,
            prior_strength=kappa,
        )
        denom = count_f + kappa
        own_weight = np.divide(
            count_f, denom, out=np.zeros_like(count_f), where=denom > 0.0
        )
        prior = mean
        prior_neff = count_f + (1.0 - own_weight) * prior_neff
    return prior, prior_neff


def cascade_selection(
    prior_lookups: Mapping[LevelT, Mapping[str, np.ndarray]],
    levels: Sequence[LevelT],
    threshold: float,
    *,
    sample_key: str = "matchups",
    valid_key: str = "valid",
    is_isolated: np.ndarray | None = None,
    isolated_excluded_levels: tuple[LevelT, ...] = (),
) -> dict[LevelT, np.ndarray]:
    """Pick one prior level per row from most to least contextual.

    The first valid level whose own sample size clears `threshold` wins. Rows
    with no confident level fall back to the broadest valid level available, but
    levels are never mixed for the same row.
    """
    present = [level for level in levels if level in prior_lookups]
    if not present:
        return {}

    n = prior_lookups[present[0]][sample_key].shape[0]
    valid = {
        level: prior_lookups[level][valid_key]
        & np.isfinite(prior_lookups[level][sample_key])
        for level in present
    }
    if is_isolated is not None and isolated_excluded_levels:
        for level in isolated_excluded_levels:
            if level in valid:
                valid[level] = valid[level] & ~is_isolated

    masks = {level: np.zeros(n, dtype=np.float64) for level in present}
    remaining = np.ones(n, dtype=bool)
    for level in present:
        take = (
            remaining
            & valid[level]
            & (prior_lookups[level][sample_key] >= float(threshold))
        )
        masks[level][take] = 1.0
        remaining &= ~take

    for level in reversed(present):
        take = remaining & valid[level]
        masks[level][take] = 1.0
        remaining &= ~take

    return masks


def capped_prior_weight(
    sample_count: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    cap: float,
    reliability_cap: float | None = None,
) -> np.ndarray:
    """Convert prior support into capped pseudo-count weight.

    Rate-like metrics use ``min(sample_count, cap)`` by leaving
    `reliability_cap` unset. Metrics whose observation denominator is not
    `sample_count` can pass `reliability_cap` to scale the cap by the prior
    sample-count reliability instead; a non-positive reliability cap means
    "treat every valid prior as fully reliable."
    """
    counts = np.asarray(sample_count, dtype=np.float64)
    cap_f = float(cap)
    if cap_f <= 0.0:
        return np.zeros_like(counts, dtype=np.float64)
    if np.any(counts < 0.0):
        raise ValueError("sample_count cannot contain negative values")

    valid_mask = np.isfinite(counts)
    if valid is not None:
        valid_arr = np.asarray(valid, dtype=bool)
        if valid_arr.shape != counts.shape:
            raise ValueError(
                f"valid mask must match sample_count shape {counts.shape}, "
                f"got {valid_arr.shape}"
            )
        valid_mask &= valid_arr

    if reliability_cap is None:
        weight = np.minimum(counts, cap_f)
    elif float(reliability_cap) <= 0.0:
        weight = np.full_like(counts, cap_f, dtype=np.float64)
    else:
        reliability = np.minimum(counts / float(reliability_cap), 1.0)
        weight = cap_f * reliability
    return np.where(valid_mask, weight, 0.0)


def apply_cascade_prior_weights(
    prior_weights: Mapping[LevelT, np.ndarray],
    selection: Mapping[LevelT, np.ndarray],
    *,
    match_total: bool,
) -> dict[LevelT, np.ndarray]:
    """Apply one-level cascade masks to prior weights.

    When `match_total` is true, the selected level receives the same total prior
    weight that the additive mixture would have assigned across all levels.
    """
    present = [level for level in prior_weights if level in selection]
    if not present:
        return {}
    if match_total:
        total = np.sum([prior_weights[level] for level in present], axis=0)
        return {level: total * selection[level] for level in present}
    return {level: prior_weights[level] * selection[level] for level in present}


def weighted_prior_blend(
    observed: np.ndarray,
    observed_weight: np.ndarray,
    prior_values: Mapping[LevelT, np.ndarray],
    prior_weights: Mapping[LevelT, np.ndarray],
    levels: Sequence[LevelT],
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Blend observed values with weighted prior values across ordered levels."""
    obs = np.asarray(observed, dtype=np.float64)
    obs_weight = np.asarray(observed_weight, dtype=np.float64)
    if obs.shape != obs_weight.shape:
        raise ValueError(
            f"observed and observed_weight must have the same shape, got "
            f"{obs.shape} and {obs_weight.shape}"
        )

    numerator = obs_weight * obs
    denominator = obs_weight.copy()

    for level in levels:
        if level not in prior_values or level not in prior_weights:
            continue
        value = np.asarray(prior_values[level], dtype=np.float64)
        weight = np.asarray(prior_weights[level], dtype=np.float64)
        if value.shape != obs.shape or weight.shape != obs.shape:
            raise ValueError(
                f"prior value/weight for {level!r} must match observed shape "
                f"{obs.shape}, got {value.shape} and {weight.shape}"
            )
        valid = (weight > 0.0) & np.isfinite(value)
        numerator[valid] += weight[valid] * value[valid]
        denominator[valid] += weight[valid]

    return np.divide(
        numerator, denominator, out=obs.copy(), where=denominator > 0.0
    ).astype(dtype)


def smooth_metrics_with_priors(
    observed_values: Mapping[str, np.ndarray],
    metrics: Sequence[str],
    observed_weight: np.ndarray,
    prior_lookups: Mapping[LevelT, Mapping[str, np.ndarray]],
    prior_weights: Mapping[LevelT, np.ndarray],
    levels: Sequence[LevelT],
    *,
    output_prefix: str = "smoothed_",
    dtype: np.dtype | type = np.float32,
) -> dict[str, np.ndarray]:
    """Smooth multiple metric arrays with the same hierarchy of prior weights."""
    return {
        f"{output_prefix}{metric}": weighted_prior_blend(
            observed_values[metric],
            observed_weight,
            {
                level: lookup[metric]
                for level, lookup in prior_lookups.items()
                if metric in lookup
            },
            prior_weights,
            levels,
            dtype=dtype,
        )
        for metric in metrics
    }


def _row_keys(rows: Any, key_cols: tuple[str, ...]) -> list[tuple]:
    columns = getattr(rows, "columns")
    n = int(getattr(rows, "n"))
    cols = [columns[c] for c in key_cols]
    return [tuple(c[i] for c in cols) for i in range(n)]


def _lookup_hierarchical_prior(
    target: Any,
    prior: Any,
    metrics: Sequence[str],
) -> dict[str, np.ndarray]:
    prior_key_columns = getattr(prior, "key_columns")
    prior_columns = getattr(prior, "columns")
    target_n = int(getattr(target, "n"))
    prior_idx_by_key = {
        k: i for i, k in enumerate(_row_keys(prior, prior_key_columns))
    }
    target_keys = _row_keys(target, prior_key_columns)
    target_prior_idx = np.array(
        [prior_idx_by_key.get(k, -1) for k in target_keys],
        dtype=np.int64,
    )
    valid = target_prior_idx >= 0

    def take(name: str) -> np.ndarray:
        values = np.full(target_n, np.nan, dtype=np.float64)
        if valid.any():
            values[valid] = prior_columns[name][target_prior_idx[valid]].astype(
                np.float64
            )
        return values

    return {"valid": valid, **{name: take(name) for name in ("matchups", *metrics)}}


def _hierarchical_prior_weight(
    level: LevelT,
    lookup: Mapping[str, np.ndarray],
    cfg: Any,
    *,
    per_minute: bool,
) -> np.ndarray:
    strengths = (
        cfg.prior_per_minute_strengths if per_minute else cfg.prior_rate_strengths
    )
    cap = float(strengths.get(level, 0.0))
    if not per_minute:
        return capped_prior_weight(lookup["matchups"], valid=lookup["valid"], cap=cap)
    rate_cap = float(cfg.prior_rate_strengths.get(level, 0.0))
    return capped_prior_weight(
        lookup["matchups"],
        valid=lookup["valid"],
        cap=cap,
        reliability_cap=rate_cap,
    )


def smooth_hierarchical_baseline(
    target: Any,
    priors: Mapping[LevelT, Any],
    cfg: Any,
    *,
    prior_levels: Sequence[LevelT],
    all_metrics: Sequence[str],
    rate_like_metrics: Sequence[str],
    per_minute_metrics: Sequence[str],
) -> Any:
    """Smooth classification baseline rows with a hierarchy of Bayesian priors."""
    prior_lookups = {
        level: _lookup_hierarchical_prior(target, priors[level], all_metrics)
        for level in prior_levels
        if level in priors and int(getattr(priors[level], "n")) > 0
    }
    columns = getattr(target, "columns")
    matchups = columns["matchups"].astype(np.float64)
    amplification = amplification_factor(matchups, float(cfg.amplification_threshold))
    rate_weights = {
        level: _hierarchical_prior_weight(level, lookup, cfg, per_minute=False)
        * amplification
        for level, lookup in prior_lookups.items()
    }
    per_minute_weights = {
        level: _hierarchical_prior_weight(level, lookup, cfg, per_minute=True)
        * amplification
        for level, lookup in prior_lookups.items()
    }

    is_isolated: np.ndarray | None = None
    if cfg.isolated_roles and cfg.isolated_role_excluded_levels:
        role_col = columns.get("teamposition")
        if role_col is not None:
            is_isolated = np.isin(role_col, list(cfg.isolated_roles))
            for level in cfg.isolated_role_excluded_levels:
                if level in rate_weights:
                    rate_weights[level] = np.where(is_isolated, 0.0, rate_weights[level])
                if level in per_minute_weights:
                    per_minute_weights[level] = np.where(
                        is_isolated,
                        0.0,
                        per_minute_weights[level],
                    )

    if cfg.smoothing_mode == "cascade":
        selection = cascade_selection(
            prior_lookups,
            prior_levels,
            float(cfg.prior_confidence_matchups),
            is_isolated=is_isolated,
            isolated_excluded_levels=cfg.isolated_role_excluded_levels,
        )
        rate_weights = apply_cascade_prior_weights(
            rate_weights,
            selection,
            match_total=cfg.cascade_match_weight,
        )
        per_minute_weights = apply_cascade_prior_weights(
            per_minute_weights,
            selection,
            match_total=cfg.cascade_match_weight,
        )
    elif cfg.smoothing_mode != "additive":
        raise ValueError(f"Unsupported smoothing_mode: {cfg.smoothing_mode!r}")

    for level in prior_levels:
        if level not in prior_lookups:
            label = getattr(level, "value", str(level))
            logger.warning("Prior %s unavailable for smoothing", label)

    sum_w_timeplayed = columns["sum_w_timeplayed"].astype(np.float64)
    new_cols = smooth_metrics_with_priors(
        columns,
        rate_like_metrics,
        matchups,
        prior_lookups,
        rate_weights,
        prior_levels,
    )
    new_cols.update(
        smooth_metrics_with_priors(
            columns,
            per_minute_metrics,
            sum_w_timeplayed,
            prior_lookups,
            per_minute_weights,
            prior_levels,
        )
    )
    smoothed = target.with_columns(new_cols)
    label = getattr(getattr(target, "level"), "value", str(getattr(target, "level")))
    logger.info("Smoothed %s: %d rows", label, int(getattr(smoothed, "n")))
    return smoothed


def apply_hierarchical_shrinkage(levels: Mapping[Any, Any], cfg: Any) -> dict[Any, Any]:
    """Apply classification embedding hierarchical Bayesian shrinkage.

    Imports classification row/config classes lazily because classification config
    imports this module for the shared build-group policy.
    """
    from app.classification.embeddings.config import (
        ALL_METRICS,
        LEVEL_KEY,
        PER_MINUTE_METRICS,
        PRIOR_LEVELS,
        RATE_LIKE_METRICS,
        IdentityType,
    )
    from app.classification.embeddings.load import LevelRows

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
        IdentityType.BASELINE: smooth_hierarchical_baseline(
            target,
            {level: rows for level, rows in levels.items() if level in PRIOR_LEVELS},
            cfg,
            prior_levels=PRIOR_LEVELS,
            all_metrics=ALL_METRICS,
            rate_like_metrics=RATE_LIKE_METRICS,
            per_minute_metrics=PER_MINUTE_METRICS,
        )
    }


__all__ = [
    "BUILD_GROUPS",
    "BUILD_TO_GROUP",
    "DEFAULT_PRIOR_MEAN",
    "DEFAULT_PRIOR_STRENGTH",
    "SIBLING_BUILD_BY_LABEL",
    "SIBLING_BUILD_PAIRS",
    "S2VX_NEUTRAL_FLOOR_LADDER",
    "amplification_factor",
    "apply_hierarchical_shrinkage",
    "apply_cascade_prior_weights",
    "bayesian_smoothed_rate",
    "build_group_for",
    "build_group_sql",
    "cascade_dynamic_smoothed_rate",
    "cascade_selection",
    "capped_prior_weight",
    "composite_interaction_priors",
    "dynamic_smoothed_rate",
    "eb_strength",
    "eb_strength_from_moments",
    "nested_shrunk_rate",
    "sibling_build_sql",
    "s2vx_floor_prior_for_ladder",
    "smooth_hierarchical_baseline",
    "smooth_interaction_levels",
    "smooth_ml_prior_features",
    "smooth_metrics_with_priors",
    "smooth_rate_by_mode",
    "weighted_prior_blend",
]
