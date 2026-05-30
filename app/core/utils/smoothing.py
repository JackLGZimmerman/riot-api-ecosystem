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
]
