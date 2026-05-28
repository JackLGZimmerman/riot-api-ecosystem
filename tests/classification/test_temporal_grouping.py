from __future__ import annotations

import numpy as np

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    ALL_METRICS,
    PHASES,
    EmbeddingConfig,
    IdentityType,
    SpecialistSpec,
)
from app.classification.embeddings.embed import LevelEmbeddings
from app.classification.embeddings.load import LevelRows
from app.classification.embeddings.matrices import LevelMatrix, build_level_matrix
from app.classification.embeddings.specialists import group_specialist_by_phase


def _baseline_rows(n_identities: int = 3) -> LevelRows:
    n_rows = n_identities * len(PHASES)
    championids = np.repeat(np.arange(1, n_identities + 1), len(PHASES))
    columns: dict[str, np.ndarray] = {
        "championid": championids.astype(np.int32),
        "teamposition": np.array(["TOP"] * n_rows, dtype=object),
        "build": np.array(["attack_damage"] * n_rows, dtype=object),
        "phase": np.tile(np.array(PHASES, dtype=object), n_identities),
        "matchups": np.full(n_rows, 100.0, dtype=np.float32),
    }
    for offset, metric in enumerate(ALL_METRICS, start=1):
        columns[f"smoothed_{metric}"] = (
            np.arange(n_rows, dtype=np.float32) + float(offset)
        )
    return LevelRows(
        level=IdentityType.BASELINE,
        key_columns=("championid", "teamposition", "build"),
        columns=columns,
        n=n_rows,
    )


def test_build_level_matrix_preserves_phase_axis() -> None:
    matrix = build_level_matrix(
        _baseline_rows(),
        EmbeddingConfig(feature_set=("kills", "deaths")),
    )

    assert matrix is not None
    assert matrix.matrix.shape == (3, len(PHASES), 2)
    assert matrix.feature_names == ("kills", "deaths")
    assert matrix.matchups.shape == (3, len(PHASES))


def test_embed_level_keeps_temporal_embeddings_in_shared_space() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(5, len(PHASES), 3)).astype(np.float32)
    matrix = LevelMatrix(
        level=IdentityType.BASELINE,
        keys=[(i, "TOP", "attack_damage") for i in range(5)],
        key_columns=("championid", "teamposition", "build"),
        matrix=raw,
        feature_names=("a", "b", "c"),
        matchups=np.ones((5, len(PHASES)), dtype=np.float32),
    )

    embeddings = embed.embed_level(
        matrix,
        EmbeddingConfig(feature_set=("a", "b", "c"), projection_keep_variance=1.0),
    )

    assert embeddings.embeddings.ndim == 3
    assert embeddings.embeddings.shape[:2] == (5, len(PHASES))
    assert np.allclose(
        np.linalg.norm(embeddings.embeddings, axis=-1),
        1.0,
        atol=1e-5,
    )


def test_group_specialist_by_phase_clusters_each_phase_independently() -> None:
    pos = np.array([1.0, 0.0], dtype=np.float32)
    near_pos = np.array([0.999, 0.001], dtype=np.float32)
    neg = np.array([-1.0, 0.0], dtype=np.float32)
    near_neg = np.array([-0.999, -0.001], dtype=np.float32)
    z = np.stack(
        [
            np.stack([pos, pos, pos, neg]),
            np.stack([near_pos, neg, neg, pos]),
            np.stack([neg, near_pos, near_pos, neg]),
            np.stack([near_neg, near_neg, near_neg, pos]),
        ],
        axis=0,
    )
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    embeddings = LevelEmbeddings(
        level=IdentityType.BASELINE,
        keys=[(i, "TOP", "attack_damage") for i in range(4)],
        key_columns=("championid", "teamposition", "build"),
        embeddings=z.astype(np.float32),
        feature_names=("axis",),
        matchups=np.ones((4, len(PHASES)), dtype=np.float32),
    )
    spec = SpecialistSpec(
        name="toy",
        feature_set=("axis",),
        similarity_threshold=0.99,
        projection_keep_variance=1.0,
        min_median_sim=0.99,
    )

    groupings = group_specialist_by_phase(embeddings, spec)

    assert {frozenset(group) for group in groupings[0].kept} == {
        frozenset({0, 1}),
        frozenset({2, 3}),
    }
    assert {frozenset(group) for group in groupings[1].kept} == {
        frozenset({0, 2}),
        frozenset({1, 3}),
    }
