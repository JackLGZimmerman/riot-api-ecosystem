"""Cosine similarity and agglomerative threshold grouping.

Embeddings are L2-normalised, so the cosine sim of (a, b) is just `a @ b`.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, n). Assumes rows are L2-normalised."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    return embeddings @ embeddings.T


def median_pair_similarity(sim: np.ndarray, members: list[int]) -> float:
    """Median pairwise similarity for a group within a square similarity matrix."""
    if len(members) < 2:
        return 1.0
    arr = np.asarray(members, dtype=np.int64)
    iu, ju = np.triu_indices(arr.size, k=1)
    return float(np.median(sim[arr[iu], arr[ju]]))


def _sample_mask(
    n: int,
    sample_weight: np.ndarray | None,
    min_sample_weight: float,
) -> np.ndarray:
    if sample_weight is None or min_sample_weight <= 0:
        return np.ones(n, dtype=bool)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim > 1:
        weights = weights.mean(axis=1)
    if weights.shape != (n,):
        raise ValueError(
            f"sample_weight must have shape ({n},) or ({n}, phases), got {weights.shape}"
        )
    return weights >= min_sample_weight


def _sort_groups(groups: list[list[int]]) -> list[list[int]]:
    return sorted((sorted(g) for g in groups), key=lambda g: (-len(g), g[0]))


def group_by_threshold(
    embeddings: np.ndarray,
    threshold: float,
    *,
    sample_weight: np.ndarray | None = None,
    min_sample_weight: float = 0.0,
) -> list[list[int]]:
    """Group rows whose cosine similarity clears a threshold.

    Average-link agglomerative clustering uses `1 - threshold` as the cosine
    distance cutoff. Rows below `min_sample_weight` stay as singletons.
    """
    sim = cosine_similarity_matrix(embeddings)
    eligible = _sample_mask(sim.shape[0], sample_weight, min_sample_weight)
    eligible_idx = np.flatnonzero(eligible)
    if eligible_idx.size < 2 or threshold > 1.0:
        groups = [[int(i)] for i in eligible_idx]
        groups.extend([int(i)] for i in np.flatnonzero(~eligible))
        return _sort_groups(groups)

    distance = 1.0 - sim[np.ix_(eligible_idx, eligible_idx)]
    distance = np.clip(distance, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    clusters = fcluster(
        linkage(squareform(distance, checks=False), method="average"),
        t=1.0 - threshold,
        criterion="distance",
    )
    groups: list[list[int]] = [
        eligible_idx[clusters == cluster].astype(int).tolist()
        for cluster in np.unique(clusters)
    ]
    groups.extend([int(i)] for i in np.flatnonzero(~eligible))
    return _sort_groups(groups)
