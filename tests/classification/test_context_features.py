"""Phase 3 context features: math mirrors, SQL hygiene, MATCHUPS routing."""

from __future__ import annotations

import numpy as np

from app.classification.embeddings import config
from app.classification.embeddings import context_features as C
from app.classification.embeddings import load


def test_feature_name_layout() -> None:
    assert len(C.TEAM_SHARE_FEATURE_NAMES) == 21
    assert len(C.CONCENTRATION_FEATURE_NAMES) == 4  # HHI of the 4 share metrics
    assert len(C.TEAM_FEATURE_NAMES) == 25
    assert len(C.MATCHUP_FEATURE_NAMES) == 30  # 11 raw + 4 share, each diff+adv
    assert len(C.CONTEXT_FEATURE_NAMES) == 55
    assert len(set(C.CONTEXT_FEATURE_NAMES)) == 55
    assert "champion_damage_team_concentration" in C.CONTEXT_FEATURE_NAMES


def test_math_mirrors() -> None:
    assert C.team_share(np.array([2.0]), np.array([8.0]))[0] == 0.25
    assert C.team_share(np.array([2.0]), np.array([0.0]))[0] == 2.0  # denom floor 1
    assert C.matchup_diff(np.array([5.0]), np.array([3.0]))[0] == 2.0
    assert C.matchup_advantage(np.array([5.0]), np.array([3.0]))[0] == 2.0 / 8.0
    # |p|+|o| < 1 -> denominator floored to 1, so advantage == diff
    assert C.matchup_advantage(np.array([0.2]), np.array([0.1]))[0] == 0.1
    assert np.allclose(C.concentration(np.array([[0.5, 0.5], [1.0, 0.0]])), [0.5, 1.0])


def test_sql_hygiene() -> None:
    for fn in (C.team_share_query, C.matchup_query):
        sql, names = fn("train", 8, 3)
        assert "challenge" not in sql.lower()
        assert "participant_stats" in sql
        assert "count() AS cnt" in sql
        assert "cityHash64(ps.matchid) % 8 = 3" in sql  # shard predicate
        assert len(names) in (25, 30)
    # team query emits the Herfindahl concentration columns
    ts, ts_names = C.team_share_query("train")
    assert "champion_damage_team_concentration" in ts
    assert "champion_damage_team_concentration" in ts_names
    # opponent uses the pair-sum identity, not a participant self-join
    mu, _ = C.matchup_query("train")
    assert "HAVING count() = 2" in mu


def test_context_included_in_loader_queries_when_enabled() -> None:
    cfg = config.EmbeddingConfig(include_context_features=True)
    base_sql, base_cols = load._baseline_query(cfg)
    prior_sql, prior_cols = load._prior_query(config.IdentityType.CHAMPION_ROLE, cfg)

    # Team features divide by cnt_team, matchup features by cnt_matchup.
    assert "ifNull(c.sum_kills_team_share, 0) / greatest(ifNull(c.cnt_team, 0), 1)" in base_sql
    assert "greatest(ifNull(c.cnt_matchup, 0), 1)" in base_sql
    # Prior rollups pool the context sums over the coarser key.
    assert "sum(ifNull(c.sum_kills_team_share, 0))" in prior_sql
    for col in C.CONTEXT_FEATURE_NAMES:
        assert col in base_cols and col in prior_cols

    # Off by default: no context join leaks into the standard path.
    off_sql, off_cols = load._baseline_query(config.EmbeddingConfig())
    assert "cnt_team" not in off_sql
    assert not any(c in off_cols for c in C.CONTEXT_FEATURE_NAMES)
