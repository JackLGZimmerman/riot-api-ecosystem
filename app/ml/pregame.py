# pyright: reportPrivateImportUsage=false

"""Pregame build-hypothesis surfaces shared by eval and training.

``HypothesisTables`` provides dense per-identity lookups for hypothesised
``(champion, role, build)`` keys; ``marginal_eval`` scores prior-weighted
worlds against them, and ``apply_modal_build_split`` rewrites a cache split so
every build-dependent input is the deterministic modal build per
``(champion, role)`` from the train-only catalog. A model trained on the
modal transform sees no observed build labels anywhere — its build inputs are
a function of champion identity and role alone — which makes it the
information-equivalent "no-build" comparator for the leakage-free pregame
path (see ``HGNN_BUILD_INTENT.md``, baseline 2).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from app.core.utils.smoothing import smooth_rate_by_mode
from app.ml.build_catalog import BuildCatalog
from app.ml.config import POSITIONS, DatasetConfig
from app.ml.dataset import SplitData
from app.ml.priors import PriorTables
from app.ml.semantic_context_lookup import load_semantic_context_raw_lookup
from app.ml.semantic_group_features import (
    SEMANTIC_CONTEXT_RAW_DIM,
    build_semantic_group_features,
    static_hp_range_lookups,
)

_SLOT_ROLES = np.arange(10) % 5


@dataclass(frozen=True)
class HypothesisTables:
    """Dense per-identity lookups for hypothesised (champ, role, build) keys.

    Indexed ``[champion_id, role_idx, build_id]`` with the reserve champion row
    at ``n_champions``. Priors carry the standard runtime smoothing (no LOO);
    context rows come from the same train-split identity-keyed lookup the
    runtime predictor serves from (it is also what filled the cache's
    ``identity_context_raw.npy``), so hypothesised worlds see exactly the
    surface observed train identities saw.
    """

    win_rate: np.ndarray  # [C+1, 5, B] float32, smoothed
    p1_cnt: np.ndarray  # [C+1, 5, B] float32
    context: np.ndarray  # [C+1, 5, B, SEMANTIC_CONTEXT_RAW_DIM] float32


def build_hypothesis_tables(
    cfg: DatasetConfig,
    priors: PriorTables,
    *,
    n_champions: int,
    build_vocab: tuple[str, ...],
) -> HypothesisTables:
    n_builds = len(build_vocab)
    build_index = {label: i for i, label in enumerate(build_vocab)}
    role_index = {role: i for i, role in enumerate(POSITIONS)}
    raw = np.full((n_champions + 1, 5, n_builds), 0.5, dtype=np.float64)
    cnt = np.zeros((n_champions + 1, 5, n_builds), dtype=np.float64)
    for (champ, role, label), (wr, matchups) in priors.p1.items():
        if 0 <= champ < n_champions:
            raw[champ, role_index[role], build_index[label]] = wr
            cnt[champ, role_index[role], build_index[label]] = matchups
    win_rate = smooth_rate_by_mode(
        raw,
        cnt,
        prior_mean=cfg.smoothing_prior_mean,
        prior_strength=cfg.smoothing_prior_strength,
        amplification_threshold=cfg.amplification_threshold,
        smoothing_mode=cfg.smoothing_mode,
        confidence_threshold=cfg.prior_confidence_matchups,
    ).astype(np.float32)

    context = np.zeros(
        (n_champions + 1, 5, n_builds, SEMANTIC_CONTEXT_RAW_DIM), dtype=np.float32
    )
    for (champ, role, label), row in load_semantic_context_raw_lookup().values.items():
        if 0 <= champ < n_champions and role in role_index and label in build_index:
            context[champ, role_index[role], build_index[label]] = row
    return HypothesisTables(
        win_rate=win_rate, p1_cnt=cnt.astype(np.float32), context=context
    )


def modal_build_table(catalog: BuildCatalog, n_champions: int) -> np.ndarray:
    """``[n_champions + 1, 5]`` modal vocab build id per (champion, role).

    The reserve row (and any champion without retained support) resolves
    through the catalog's role/global fallback to a real vocab id.
    """
    table = np.zeros((n_champions + 1, 5), dtype=np.int64)
    for champ in range(n_champions + 1):
        for role_idx, role in enumerate(POSITIONS):
            vector = catalog.prior_vector(champ, role)
            table[champ, role_idx] = int(
                vector.hgnn_build_ids[int(np.argmax(vector.probabilities))]
            )
    return table


def apply_modal_build_split(
    split: SplitData,
    catalog: BuildCatalog,
    tables: HypothesisTables,
    *,
    build_vocab: tuple[str, ...],
    needs_semantic: bool,
    chunk_rows: int = 65536,
) -> SplitData:
    """Rewrite every build-dependent array to the modal-build hypothesis.

    Observed ``build_id`` (and the prior/semantic arrays derived from it) is
    never read; the replacement is a deterministic function of
    ``(champion_id, slot role)``, so the transform is leakage-free on every
    split by construction.
    """
    if split.champion_id is None:
        raise ValueError("cache is missing champion_id")
    n_champions = tables.win_rate.shape[0] - 1
    champ = np.asarray(split.champion_id, dtype=np.int64)
    champ = np.where((champ < 0) | (champ >= n_champions), n_champions, champ)
    build_id = modal_build_table(catalog, n_champions)[champ, _SLOT_ROLES]
    win_rate = tables.win_rate[champ, _SLOT_ROLES, build_id]
    p1_cnt = tables.p1_cnt[champ, _SLOT_ROLES, build_id]

    semantic = split.semantic_group_features
    if needs_semantic:
        hp_lookup, range_lookup = static_hp_range_lookups()
        chunks = [
            build_semantic_group_features(
                context_raw=tables.context[
                    champ[lo : lo + chunk_rows], _SLOT_ROLES, build_id[lo : lo + chunk_rows]
                ],
                champion_id=champ[lo : lo + chunk_rows],
                build_id=build_id[lo : lo + chunk_rows],
                build_vocab=build_vocab,
                hp_lookup=hp_lookup,
                range_lookup=range_lookup,
            )
            for lo in range(0, champ.shape[0], chunk_rows)
        ]
        semantic = np.concatenate(chunks, axis=0) if chunks else semantic

    return replace(
        split,
        build_id=build_id,
        win_rate=win_rate,
        p1_cnt=p1_cnt,
        semantic_group_features=semantic,
    )
