from __future__ import annotations

import numpy as np

from app.classification.embeddings.config import (
    RATE_LIKE_METRICS,
    TIMELINE_CHECKPOINT_MINUTES,
    identity_semantic_feature_set,
)
from app.classification.embeddings.load import _baseline_query, _timeline_checkpoint_query
from app.classification.embeddings.relationship_details import _m1v1_query, _s2vx_query
from app.classification.embeddings.runtime import (
    IdentitySemanticLookup,
    RelationshipDetailLookup,
)


def test_timeline_checkpoint_metric_catalogue_uses_requested_minutes() -> None:
    assert TIMELINE_CHECKPOINT_MINUTES == (3, 4, 5, 7, 10, 12, 15, 20, 22, 25)
    assert "tl_25_gold" in RATE_LIKE_METRICS
    assert "tl_25_missing" in RATE_LIKE_METRICS
    assert "tl_3_5_gold_delta" in identity_semantic_feature_set()
    assert "tl_20_25_champion_damage_per_minute" in identity_semantic_feature_set()


def test_baseline_query_marks_missing_checkpoints_without_zero_filling() -> None:
    baseline_query = _baseline_query("train")
    checkpoint_query = _timeline_checkpoint_query("train", 25)

    assert "toFloat32(0) AS tl_25_gold" in baseline_query
    assert "participant_challenges" not in baseline_query
    assert "avgIf(tupleElement(ts.stats" in checkpoint_query
    assert "tl_25_missing" in checkpoint_query
    assert "1.0 - (countIf(coalesce(ts.matchid, '') != '') / count())" in checkpoint_query


def test_relationship_detail_queries_do_not_touch_challenges() -> None:
    queries = (_m1v1_query("TOP", "TOP"), _s2vx_query("TOP", "JUNGLE"))
    forbidden = (
        "participant_challenges",
        "challenge_",
        "solokills",
        "maxcsadvantage",
        "maxlevellead",
        "turretplatestaken",
    )
    for query in queries:
        lower = query.lower()
        for term in forbidden:
            assert term not in lower


def test_identity_semantic_lookup_loads_vectors_and_falls_back(tmp_path) -> None:
    path = tmp_path / "identity_semantic_embedding.npz"
    np.savez(
        path,
        keys=np.array([(1, "TOP", "tank")], dtype=object),
        key_columns=np.array(("championid", "teamposition", "build"), dtype=object),
        embeddings=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        dim=np.array(3, dtype=np.int32),
    )

    lookup = IdentitySemanticLookup.load(path)
    values = lookup.lookup_players([(1, "TOP", "tank"), (2, "TOP", "tank")])

    assert values.shape == (2, 3)
    assert np.allclose(values[0], [1.0, 2.0, 3.0])
    assert np.allclose(values[1], [0.0, 0.0, 0.0])


def test_relationship_detail_lookup_orients_1v1_and_keeps_2vx_symmetric() -> None:
    vec = np.array([1.0, -2.0], dtype=np.float32)
    m1v1 = RelationshipDetailLookup(
        exact={(1, "TOP", "tank", 2, "TOP", "crit"): vec},
        exact_counts={(1, "TOP", "tank", 2, "TOP", "crit"): 100.0},
        build_group={},
        build_group_counts={},
        nobuild={},
        nobuild_counts={},
        champion={},
        champion_counts={},
        dim=2,
    )
    s2vx = RelationshipDetailLookup(
        exact={(1, "TOP", "tank", 2, "JUNGLE", "crit"): vec},
        exact_counts={(1, "TOP", "tank", 2, "JUNGLE", "crit"): 100.0},
        build_group={},
        build_group_counts={},
        nobuild={},
        nobuild_counts={},
        champion={},
        champion_counts={},
        dim=2,
    )

    one_v_one = m1v1.lookup_1v1_blue(
        [(2, "TOP", "crit")],
        [(1, "TOP", "tank")],
    )
    two_vx = s2vx.lookup_2vx_team(
        [
            (2, "JUNGLE", "crit"),
            (1, "TOP", "tank"),
            (3, "MIDDLE", "mage"),
            (4, "BOTTOM", "marksman"),
            (5, "UTILITY", "utility"),
        ]
    )

    assert np.allclose(one_v_one[0], -vec)
    assert np.allclose(two_vx[0], vec)
