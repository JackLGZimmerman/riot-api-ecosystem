from __future__ import annotations

import numpy as np


def expected_calibration_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Equal-width expected calibration error for binary probabilities."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    probabilities = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    if probabilities.shape != labels.shape:
        raise ValueError(
            f"predictions and targets must have the same shape, got "
            f"{probabilities.shape} and {labels.shape}"
        )
    if probabilities.size == 0:
        return float("nan")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    labels = labels > 0.5
    bin_idx = np.minimum((probabilities * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)
    if counts.sum() == 0:
        return float("nan")

    confidence = np.bincount(bin_idx, weights=probabilities, minlength=n_bins)
    accuracy = np.bincount(bin_idx, weights=labels.astype(np.float64), minlength=n_bins)
    populated = counts > 0
    confidence[populated] /= counts[populated]
    accuracy[populated] /= counts[populated]
    return float(
        np.sum(
            counts[populated]
            / probabilities.size
            * np.abs(confidence[populated] - accuracy[populated])
        )
    )


__all__ = ["expected_calibration_error"]
