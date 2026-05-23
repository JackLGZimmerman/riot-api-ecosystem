from __future__ import annotations

import numpy as np

DEFAULT_PRIOR_MEAN = 0.5
DEFAULT_PRIOR_STRENGTH = 20.0


def bayesian_smoothed_rate(
    win_rate: np.ndarray,
    sample_count: np.ndarray,
    *,
    prior_mean: float = DEFAULT_PRIOR_MEAN,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> np.ndarray:
    """Shrink empirical rates toward a fixed prior mean.

    The prior strength is measured in pseudo-games. With the defaults, this is
    equivalent to adding 10 wins and 10 losses to each observed aggregate.
    """
    if not 0.0 <= prior_mean <= 1.0:
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
    if np.any(counts < 0.0):
        raise ValueError("sample_count cannot contain negative values")

    if prior_strength == 0.0:
        return rates.copy()

    return prior_mean + (rates - prior_mean) * counts / (counts + prior_strength)


__all__ = [
    "DEFAULT_PRIOR_MEAN",
    "DEFAULT_PRIOR_STRENGTH",
    "bayesian_smoothed_rate",
]
