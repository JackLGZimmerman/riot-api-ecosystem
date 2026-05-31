"""Shared Bayesian shrinkage primitives for win-rate / metric smoothing.

Single source of truth for the smoothing math used by both the win-rate model
(`app/ml`) and the classification embeddings (`app/classification`). Changes to
the shrink equation or the dynamic low-sample weighting propagate to both.

Core primitives:

* `bayesian_smoothed_rate` — the core shrink ``prior + (obs - prior)·n/(n + s)``.
  `prior_strength` may be a scalar or a per-element array (the array form is what
  makes dynamic, n-dependent strength possible).
* `amplification_factor` — ``sqrt(1 + threshold/max(n, 1))``. A multiplier that
  grows the prior's weight for low-sample identities (→ 1 as n grows).
* `dynamic_smoothed_rate` — `bayesian_smoothed_rate` with the strength amplified
  per element, so under-sampled identities are pulled harder toward the prior
  while well-sampled ones keep the base strength.
* `cascade_dynamic_smoothed_rate` — ML-facing cascade smoothing that stops
  applying a broad prior once the contextual sample count is confident.
* `cascade_selection` — hierarchy-facing cascade selection for choosing exactly
  one prior level per row.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

import numpy as np

DEFAULT_PRIOR_MEAN = 0.5
DEFAULT_PRIOR_STRENGTH = 20.0
LevelT = TypeVar("LevelT", bound=Hashable)


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


__all__ = [
    "DEFAULT_PRIOR_MEAN",
    "DEFAULT_PRIOR_STRENGTH",
    "amplification_factor",
    "bayesian_smoothed_rate",
    "cascade_dynamic_smoothed_rate",
    "cascade_selection",
    "dynamic_smoothed_rate",
    "eb_strength",
    "eb_strength_from_moments",
    "nested_shrunk_rate",
]
