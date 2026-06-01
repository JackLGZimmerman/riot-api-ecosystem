"""Low-level helpers shared by app/ml and app/classification.

Dependency-light (numpy only) so model and embedding code can both import these
without pulling in the heavier smoothing module. Holds the game-layout constants
and the small array/format primitives that previously had verbatim copies in
both sub-projects.
"""

from __future__ import annotations

import numpy as np

# Ordered team slots used everywhere a (champion, role, build) tuple is built.
POSITIONS: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")

# C(5, 2) same-team pair indices (0-based), in the canonical synergy_2vx order.
TEAM_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)


def sql_literal(value: str) -> str:
    """Single-quote a string for inline SQL, escaping embedded quotes."""
    return "'" + value.replace("'", "''") + "'"


def fit_last_dim(values: np.ndarray, dim: int) -> np.ndarray:
    """Pad/truncate the last axis of `values` to `dim` (float32)."""
    if values.shape[-1] == dim:
        return values.astype(np.float32, copy=False)
    if values.shape[-1] > dim:
        return values[..., :dim].astype(np.float32, copy=False)
    pad = np.zeros((*values.shape[:-1], dim - values.shape[-1]), dtype=np.float32)
    return np.concatenate([values.astype(np.float32, copy=False), pad], axis=-1)


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Sign-preserving log compression: sign(x) * log1p(|x|)."""
    return np.sign(values) * np.log1p(np.abs(values))


def median_mad_standardise(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed-log1p compress then standardise each column by median/MAD.

    Returns (standardised float32, median float32, MAD float32). The MAD is
    floored at 1.0 where it would be ~0, and that floored MAD is what both the
    returned values and `apply_median_mad` divide by.
    """
    flat = signed_log1p(values)
    med = np.median(flat, axis=0, keepdims=True)
    mad = np.median(np.abs(flat - med), axis=0, keepdims=True) * 1.4826
    mad = np.where(mad > 1e-8, mad, 1.0)
    standardised = ((flat - med) / mad).astype(np.float32)
    return standardised, med.astype(np.float32), mad.astype(np.float32)


def apply_median_mad(
    values: np.ndarray, med: np.ndarray, mad: np.ndarray
) -> np.ndarray:
    """Apply a previously fit median/MAD standardisation to new values."""
    flat = signed_log1p(values)
    return ((flat - med) / np.where(mad > 1e-8, mad, 1.0)).astype(np.float32)
