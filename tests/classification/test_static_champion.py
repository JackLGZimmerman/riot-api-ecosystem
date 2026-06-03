"""Phase 2 static champion branch: level-18 math, shapes, opt-in integration."""

from __future__ import annotations

import numpy as np

from app.classification.embeddings import static_champion as S


def test_feature_names_layout() -> None:
    names = S.static_feature_names()
    assert len(names) == 40 + len(S.LEVEL18_STATS)  # 40 raw + 7 l18
    assert names[:40] == S.static_stat_columns()
    assert names[40:] == tuple(f"{s}_l18" for s in S.LEVEL18_STATS)
    assert "_key" not in names and "id" not in names


def test_level18_formula_for_known_champion() -> None:
    by_id = S.load_static_by_id()
    names = S.static_feature_names()
    aatrox = by_id[266]  # health_flat=650, health_perLevel=114
    health_l18 = aatrox[names.index("health_l18")]
    assert health_l18 == 650.0 + 17.0 * 114.0  # 2588.0
    armor_l18 = aatrox[names.index("armor_l18")]
    assert armor_l18 == 38.0 + 17.0 * 4.8


def test_build_static_matrix_shape_and_finiteness() -> None:
    ids = np.asarray([266, 266, 103, 999999], dtype=np.int64)  # repeat + unknown
    matrix, names = S.build_static_matrix(ids, clip_value=8.0)
    assert matrix.shape == (4, len(names))
    assert np.isfinite(matrix).all()
    assert matrix.dtype == np.float32
    # Same champion -> identical standardised row.
    assert np.array_equal(matrix[0], matrix[1])


def test_pipeline_includes_static_block_when_enabled() -> None:
    from app.classification.embeddings.config import EmbeddingConfig
    from app.classification.embeddings.pipeline import build_metric_matrices

    base = build_metric_matrices()
    aug = build_metric_matrices(EmbeddingConfig(include_static_champion=True))

    from app.classification.embeddings.config import IdentityType

    b = base[IdentityType.BASELINE]
    a = aug[IdentityType.BASELINE]
    assert a.matrix.shape[1] == b.matrix.shape[1] + len(S.static_feature_names())
    assert a.feature_names[: b.matrix.shape[1]] == b.feature_names
    assert a.feature_names[b.matrix.shape[1] :] == S.static_feature_names()
    # Full-game columns are unchanged by appending the static block.
    assert np.array_equal(a.matrix[:, : b.matrix.shape[1]], b.matrix)
