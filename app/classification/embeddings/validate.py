"""Diagnostics for agglomerative embedding groups."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from app.classification.embeddings.config import IdentityType
from app.classification.embeddings.embed import LevelEmbeddings
from app.classification.embeddings.similarity import (
    cosine_similarity_matrix,
    group_by_threshold,
    source_groups_by_threshold,
)

logger = logging.getLogger(__name__)

GROUP_SIZE_TRIM_LARGEST = 10


@dataclass
class LevelDiagnostics:
    level: IdentityType
    n: int
    pairwise_sim_mean: float
    pairwise_sim_std: float
    pairwise_sim_p50: float
    pairwise_sim_p95: float
    group_count: int
    largest_group: int
    mean_group_size: float
    mean_non_singleton_group_size: float
    non_singleton_identity_share: float
    mean_source_group_size: float
    source_singleton_group_count: int
    singleton_group_count: int
    eligible_count: int
    non_singleton_group_count: int
    group_diversity_score: float
    group_quality_score: float
    min_group_pairwise_sim: float
    low_sample_dominance: float
    top_mid_mixed_group_count: int
    top_mid_mixed_identity_share: float
    top_mid_pairwise_sim_mean: float
    top_mid_pairwise_sim_p95: float


def _trimmed_group_size_mean(
    sizes: list[int],
    *,
    trim_largest: int = GROUP_SIZE_TRIM_LARGEST,
) -> float:
    """Mean group size after dropping the largest groups."""
    if not sizes:
        return 0.0
    trimmed = sorted(sizes, reverse=True)[trim_largest:]
    return float(np.mean(trimmed)) if trimmed else 0.0


def _min_group_similarity(sim: np.ndarray, groups: list[list[int]]) -> float:
    mins: list[float] = []
    for group in groups:
        if len(group) < 2:
            continue
        arr = np.array(group)
        iu, ju = np.triu_indices(len(arr), k=1)
        mins.append(float(sim[arr[iu], arr[ju]].min()))
    return min(mins) if mins else float("nan")


def _normalised_entropy(values: list[object], global_cardinality: int) -> float:
    if len(values) < 2 or global_cardinality < 2:
        return 0.0
    counts = np.array(list(Counter(values).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    max_entropy = float(np.log(min(global_cardinality, len(values))))
    return entropy / max_entropy if max_entropy > 0.0 else 0.0


def _group_diversity_score(
    embeddings: LevelEmbeddings,
    groups: list[list[int]],
) -> float:
    columns = [
        embeddings.key_columns.index(name)
        for name in ("teamposition", "build", "championid")
        if name in embeddings.key_columns
    ]
    if not columns:
        return float("nan")

    global_cardinality = {
        column: len({key[column] for key in embeddings.keys}) for column in columns
    }
    weighted_scores: list[float] = []
    weights: list[int] = []
    for group in groups:
        if len(group) < 2:
            continue
        scores = [
            _normalised_entropy(
                [embeddings.keys[idx][column] for idx in group],
                global_cardinality[column],
            )
            for column in columns
        ]
        weighted_scores.append(float(np.mean(scores)))
        weights.append(len(group))

    if not weighted_scores:
        return 0.0
    return float(np.average(weighted_scores, weights=weights))


def _top_mid_stats(
    embeddings: LevelEmbeddings,
    sim: np.ndarray,
    groups: list[list[int]],
) -> tuple[int, float, float, float]:
    if "teamposition" not in embeddings.key_columns:
        return 0, float("nan"), float("nan"), float("nan")

    role_idx = embeddings.key_columns.index("teamposition")
    roles = np.array([str(key[role_idx]) for key in embeddings.keys], dtype=object)
    top = roles == "TOP"
    middle = (roles == "MIDDLE") | (roles == "MID")
    top_mid_total = int((top | middle).sum())

    mixed_group_count = 0
    mixed_identity_count = 0
    for group in groups:
        arr = np.asarray(group, dtype=np.int64)
        if top[arr].any() and middle[arr].any():
            mixed_group_count += 1
            mixed_identity_count += int((top[arr] | middle[arr]).sum())

    top_idx = np.flatnonzero(top)
    middle_idx = np.flatnonzero(middle)
    if top_idx.size and middle_idx.size:
        cross = sim[np.ix_(top_idx, middle_idx)].reshape(-1)
        mean = float(cross.mean())
        p95 = float(np.percentile(cross, 95))
    else:
        mean = float("nan")
        p95 = float("nan")

    share = mixed_identity_count / top_mid_total if top_mid_total else float("nan")
    return mixed_group_count, share, mean, p95


def _summarise(
    embeddings: LevelEmbeddings,
    threshold: float,
    min_matchups: float,
) -> LevelDiagnostics:
    sim = cosine_similarity_matrix(embeddings.embeddings)
    iu, ju = np.triu_indices(sim.shape[0], k=1)
    upper = sim[iu, ju]
    mean_matchups = embeddings.matchups.mean(axis=1)
    groups = group_by_threshold(
        embeddings.embeddings,
        threshold,
        sample_weight=mean_matchups,
        min_sample_weight=min_matchups,
    )
    source_groups = source_groups_by_threshold(
        embeddings.embeddings,
        threshold,
        sample_weight=mean_matchups,
        min_sample_weight=min_matchups,
    )
    group_sizes = [len(g) for g in groups]
    source_group_sizes = [len(g) for g in source_groups]
    non_singleton_sizes = [size for size in group_sizes if size > 1]
    non_singleton_identity_count = sum(non_singleton_sizes)
    non_singleton_identity_share = (
        non_singleton_identity_count / sim.shape[0] if sim.shape[0] else 0.0
    )
    mean_group_size = _trimmed_group_size_mean(group_sizes)
    mean_non_singleton_group_size = _trimmed_group_size_mean(non_singleton_sizes)
    mean_source_group_size = _trimmed_group_size_mean(source_group_sizes)
    group_diversity_score = _group_diversity_score(embeddings, groups)
    group_quality_score = (
        mean_non_singleton_group_size
        * group_diversity_score
        * non_singleton_identity_share
    )
    (
        top_mid_mixed_group_count,
        top_mid_mixed_identity_share,
        top_mid_pairwise_sim_mean,
        top_mid_pairwise_sim_p95,
    ) = _top_mid_stats(embeddings, sim, groups)

    high = (
        (sim[iu, ju] >= threshold)
        & (mean_matchups[iu] >= min_matchups)
        & (mean_matchups[ju] >= min_matchups)
    )
    if high.any():
        log_mu = np.log1p(mean_matchups)
        diffs = np.abs(log_mu[iu[high]] - log_mu[ju[high]])
        low_sample_dominance = float((diffs > 2.0).mean())
    else:
        low_sample_dominance = 0.0

    return LevelDiagnostics(
        level=embeddings.level,
        n=sim.shape[0],
        pairwise_sim_mean=float(upper.mean()) if upper.size else float("nan"),
        pairwise_sim_std=float(upper.std()) if upper.size else float("nan"),
        pairwise_sim_p50=float(np.percentile(upper, 50))
        if upper.size
        else float("nan"),
        pairwise_sim_p95=float(np.percentile(upper, 95))
        if upper.size
        else float("nan"),
        group_count=len(groups),
        largest_group=max(group_sizes) if group_sizes else 0,
        mean_group_size=mean_group_size,
        mean_non_singleton_group_size=mean_non_singleton_group_size,
        non_singleton_identity_share=non_singleton_identity_share,
        mean_source_group_size=mean_source_group_size,
        source_singleton_group_count=sum(s == 1 for s in source_group_sizes),
        singleton_group_count=sum(s == 1 for s in group_sizes),
        eligible_count=int((mean_matchups >= min_matchups).sum()),
        non_singleton_group_count=sum(s > 1 for s in group_sizes),
        group_diversity_score=group_diversity_score,
        group_quality_score=group_quality_score,
        min_group_pairwise_sim=_min_group_similarity(sim, groups),
        low_sample_dominance=low_sample_dominance,
        top_mid_mixed_group_count=top_mid_mixed_group_count,
        top_mid_mixed_identity_share=top_mid_mixed_identity_share,
        top_mid_pairwise_sim_mean=top_mid_pairwise_sim_mean,
        top_mid_pairwise_sim_p95=top_mid_pairwise_sim_p95,
    )


def diagnose_all(
    embeddings: dict[IdentityType, LevelEmbeddings],
    threshold: float,
    min_matchups_by_level: Mapping[IdentityType, float] | None = None,
) -> dict[IdentityType, LevelDiagnostics]:
    min_matchups_by_level = min_matchups_by_level or {}
    out = {
        level: _summarise(e, threshold, min_matchups_by_level.get(level, 0.0))
        for level, e in embeddings.items()
    }
    for level, d in out.items():
        logger.info(
            "[%s] n=%d sim mean=%.3f p50=%.3f p95=%.3f | groups=%d largest=%d "
            "mean_ex_top10=%.2f non_singleton_mean_ex_top10=%.2f "
            "singletons=%d non_singleton=%d "
            "coverage=%.2f diversity=%.2f quality=%.2f "
            "source_mean_ex_top10=%.2f source_singletons=%d eligible=%d "
            "min_group_sim=%.3f low_sample_dominance=%.2f "
            "top_mid_mixed=%d top_mid_share=%.2f top_mid_pair_p95=%.3f",
            level.value,
            d.n,
            d.pairwise_sim_mean,
            d.pairwise_sim_p50,
            d.pairwise_sim_p95,
            d.group_count,
            d.largest_group,
            d.mean_group_size,
            d.mean_non_singleton_group_size,
            d.singleton_group_count,
            d.non_singleton_group_count,
            d.non_singleton_identity_share,
            d.group_diversity_score,
            d.group_quality_score,
            d.mean_source_group_size,
            d.source_singleton_group_count,
            d.eligible_count,
            d.min_group_pairwise_sim,
            d.low_sample_dominance,
            d.top_mid_mixed_group_count,
            d.top_mid_mixed_identity_share,
            d.top_mid_pairwise_sim_p95,
        )
    return out
