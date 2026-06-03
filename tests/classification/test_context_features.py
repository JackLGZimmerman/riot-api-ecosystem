"""Phase 3 context features: math mirrors, SQL hygiene, MATCHUPS routing."""

from __future__ import annotations

import numpy as np

from app.classification.embeddings import config
from app.classification.embeddings import context_features as C
from app.classification.embeddings import load


def test_feature_name_layout() -> None:
    assert len(C.TEAM_SHARE_FEATURE_NAMES) == 22
    assert len(C.CONCENTRATION_FEATURE_NAMES) == 6  # HHI of the 6 concentration metrics
    assert len(C.TEAM_FEATURE_NAMES) == 28
    assert len(C.MATCHUP_FEATURE_NAMES) == 32  # 12 raw + 4 share, each diff+adv
    assert len(C.CONTEXT_FEATURE_NAMES) == 60
    assert len(set(C.CONTEXT_FEATURE_NAMES)) == 60
    assert "champion_damage_team_concentration" in C.CONTEXT_FEATURE_NAMES
    # Added families: cc participation, kills/damage-taken carry concentration,
    # and the lane-farm (CS) laning matchup.
    assert "cc_team_share" in C.TEAM_SHARE_FEATURE_NAMES
    assert "kills_team_concentration" in C.CONCENTRATION_FEATURE_NAMES
    assert "damage_taken_team_concentration" in C.CONCENTRATION_FEATURE_NAMES
    assert "lane_farm_vs_role_opponent_diff" in C.MATCHUP_FEATURE_NAMES
    assert "lane_farm_vs_role_opponent_advantage" in C.MATCHUP_FEATURE_NAMES
    # Concentration HHI normalises by the team total, only emitted for shares.
    assert set(C.CONCENTRATION_METRICS).issubset(C.TEAM_SHARE_METRICS)


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
        assert len(names) in (28, 32)
    # team query emits the Herfindahl concentration columns
    ts, ts_names = C.team_share_query("train")
    assert "champion_damage_team_concentration" in ts
    assert "champion_damage_team_concentration" in ts_names
    # Added participation + concentration features resolve to real raw columns.
    assert "cc_team_share" in ts and "cc_team_share" in ts_names
    assert "timeccingothers" in ts  # cc participation reads the raw CC column
    assert "kills_team_concentration" in ts_names
    assert "damage_taken_team_concentration" in ts_names
    # opponent uses the pair-sum identity, not a participant self-join
    mu, mu_names = C.matchup_query("train")
    assert "HAVING count() = 2" in mu
    # Added lane-farm (CS) laning matchup.
    assert "lane_farm_vs_role_opponent_diff" in mu_names


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

    # On by default: the full-game encoder surface uses all 215 metrics.
    default_sql, default_cols = load._baseline_query(config.EmbeddingConfig())
    assert "cnt_team" in default_sql
    assert all(c in default_cols for c in C.CONTEXT_FEATURE_NAMES)

    # Legacy profile-only builds can still opt out explicitly.
    off_sql, off_cols = load._baseline_query(
        config.EmbeddingConfig(include_context_features=False)
    )
    assert "cnt_team" not in off_sql
    assert not any(c in off_cols for c in C.CONTEXT_FEATURE_NAMES)
