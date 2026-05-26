"""Cosine similarity and agglomerative threshold grouping.

Embeddings are L2-normalised, so the cosine sim of (a, b) is just `a @ b`.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, n). Assumes rows are L2-normalised."""
    return embeddings @ embeddings.T


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


def _agglomerative_groups(
    sim: np.ndarray, threshold: float, eligible: np.ndarray
) -> list[list[int]]:
    eligible_idx = np.flatnonzero(eligible)
    if eligible_idx.size < 2 or threshold > 1.0:
        groups = [[int(i)] for i in eligible_idx]
        groups.extend([int(i)] for i in np.flatnonzero(~eligible))
        return _sort_groups(groups)

    distance_threshold = 1.0 - threshold
    distance = 1.0 - sim[np.ix_(eligible_idx, eligible_idx)]
    distance = np.clip(distance, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)

    condensed_distance = squareform(distance, checks=False)
    clusters = fcluster(
        linkage(condensed_distance, method="average"),
        t=distance_threshold,
        criterion="distance",
    )

    groups: list[list[int]] = []
    for cluster in np.unique(clusters):
        groups.append(eligible_idx[clusters == cluster].astype(int).tolist())

    groups.extend([int(i)] for i in np.flatnonzero(~eligible))
    return _sort_groups(groups)


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
    return _agglomerative_groups(sim, threshold, eligible)


def source_groups_by_threshold(
    embeddings: np.ndarray,
    threshold: float,
    *,
    sample_weight: np.ndarray | None = None,
    min_sample_weight: float = 0.0,
) -> list[list[int]]:
    """Return one threshold neighborhood per source row.

    Unlike agglomerative clustering, these groups can overlap: row `i` is the
    source identity and the members are all eligible identities similar enough
    to `i`, including `i` itself.
    """
    sim = cosine_similarity_matrix(embeddings)
    eligible = _sample_mask(sim.shape[0], sample_weight, min_sample_weight)
    groups: list[list[int]] = []
    for idx in range(sim.shape[0]):
        if not eligible[idx]:
            groups.append([idx])
            continue
        members = np.flatnonzero((sim[idx] >= threshold) & eligible)
        groups.append(members.astype(int).tolist())
    return groups
