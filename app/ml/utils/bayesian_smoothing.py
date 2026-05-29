from __future__ import annotations

import numpy as np

DEFAULT_PRIOR_MEAN = 0.5
DEFAULT_PRIOR_STRENGTH = 20.0


def bayesian_smoothed_rate(
    win_rate: np.ndarray,
    sample_count: np.ndarray,
    *,
    prior_mean: float | np.ndarray = DEFAULT_PRIOR_MEAN,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> np.ndarray:
    """Shrink empirical rates toward a prior mean.

    `prior_mean` is a scalar prior (e.g. ``0.5``) or a per-element array of the
    same shape as `win_rate`. The array form is used for the per-side
    interaction fallback: each pair is shrunk toward a composite of its two
    sides' individual priors instead of a flat ``0.5``, so low-sample pairs get
    a real signal rather than the no-information default.

    The prior strength is measured in pseudo-games. With the defaults, a scalar
    ``0.5`` prior is equivalent to adding 10 wins and 10 losses to each
    observed aggregate.
    """
    prior = np.asarray(prior_mean, dtype=np.float64)
    if np.any(prior < 0.0) or np.any(prior > 1.0):
        raise ValueError("prior_mean must be between 0.0 and 1.0")
    if prior_strength < 0.0:
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
    if np.any(counts < 0.0):
        raise ValueError("sample_count cannot contain negative values")

    if prior_strength == 0.0:
        return rates.copy()

    return prior + (rates - prior) * counts / (counts + prior_strength)


__all__ = [
    "DEFAULT_PRIOR_MEAN",
    "DEFAULT_PRIOR_STRENGTH",
    "bayesian_smoothed_rate",
]
