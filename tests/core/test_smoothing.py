from __future__ import annotations

import numpy as np

from app.classification.embeddings import load as embedding_load
from app.core.utils.smoothing import (
    BUILD_GROUPS,
    SIBLING_BUILD_BY_LABEL,
    S2VX_NEUTRAL_FLOOR_LADDER,
    build_group_for,
    capped_prior_weight,
    s2vx_floor_prior_for_ladder,
    sibling_build_sql,
    smooth_ml_prior_features,
    smooth_metrics_with_priors,
    smooth_rate_by_mode,
)
from app.ml import config as ml_config


def test_build_group_policy_is_shared_across_ml_and_classification() -> None:
    assert ml_config.BUILD_GROUPS is BUILD_GROUPS
    assert ml_config.build_group_for("ap_off_tank") == "ap"
    assert build_group_for("crit") == "crit"
    assert SIBLING_BUILD_BY_LABEL["ability_power"] == "ap_off_tank"
    # Classification shares the same build-group/sibling policy via the
    # single smoothing module (load.py imports sibling_build_sql from it).
    assert embedding_load.sibling_build_sql is sibling_build_sql


def test_smooth_rate_by_mode_matches_additive_and_cascade_semantics() -> None:
    rates = np.array([0.8, 0.8], dtype=np.float64)
    counts = np.array([10.0, 60.0], dtype=np.float64)
    expected_additive = np.array([0.6, 0.725], dtype=np.float64)

    additive = smooth_rate_by_mode(
        rates,
        counts,
        prior_mean=0.5,
        prior_strength=20.0,
        amplification_threshold=0.0,
        smoothing_mode="additive",
    )
    cascade = smooth_rate_by_mode(
        rates,
        counts,
        prior_mean=0.5,
        prior_strength=20.0,
        amplification_threshold=0.0,
        smoothing_mode="cascade",
        confidence_threshold=50.0,
    )

    assert np.allclose(additive, expected_additive)
    assert np.allclose(cascade, [expected_additive[0], 0.8])


def test_capped_prior_weight_supports_rate_and_reliability_scaled_metrics() -> None:
    counts = np.array([4.0, 12.0, np.nan], dtype=np.float64)
    valid = np.array([True, True, True])

    rate_weight = capped_prior_weight(counts, valid=valid, cap=10.0)
    per_minute_weight = capped_prior_weight(
        counts[:2],
        valid=valid[:2],
        cap=100.0,
        reliability_cap=10.0,
    )
    full_reliability_weight = capped_prior_weight(
        counts[:2],
        valid=valid[:2],
        cap=100.0,
        reliability_cap=0.0,
    )

    assert np.allclose(rate_weight, [4.0, 10.0, 0.0])
    assert np.allclose(per_minute_weight, [40.0, 100.0])
    assert np.allclose(full_reliability_weight, [100.0, 100.0])


def test_smooth_metrics_with_priors_blends_observed_and_prior_weights() -> None:
    smoothed = smooth_metrics_with_priors(
        {"win": np.array([0.8, 0.2], dtype=np.float64)},
        ("win",),
        np.array([10.0, 10.0], dtype=np.float64),
        {"sibling": {"win": np.array([0.5, 0.9], dtype=np.float64)}},
        {"sibling": np.array([10.0, 0.0], dtype=np.float64)},
        ("sibling",),
    )

    assert smoothed["smoothed_win"].dtype == np.float32
    assert np.allclose(smoothed["smoothed_win"], [0.65, 0.2])


def test_smooth_ml_prior_features_centralises_solo_and_interaction_smoothing() -> None:
    raw = {
        "p1_raw": np.full((1, 10), 0.6, dtype=np.float64),
        "p1_cnt": np.full((1, 10), 20.0, dtype=np.float64),
        "m1v1_raw": np.full((1, 25), 0.7, dtype=np.float64),
        "m1v1_cnt": np.full((1, 25), 10.0, dtype=np.float64),
        "s2vx_raw": np.full((1, 20), 0.8, dtype=np.float64),
        "s2vx_cnt": np.full((1, 20), 10.0, dtype=np.float64),
    }

    smoothed = smooth_ml_prior_features(
        raw,
        prior_mean=0.5,
        prior_strength=10.0,
        amplification_threshold=0.0,
        smoothing_mode="additive",
        prior_confidence_matchups=50.0,
        per_side_fallback=True,
        nested_pooling=False,
        level_strengths={"m1v1": [10.0], "s2vx": [10.0]},
        m1v1_levels=(("m1v1_raw", "m1v1_cnt"),),
        s2vx_levels=(("s2vx_raw", "s2vx_cnt"),),
        team_pairs=(
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4),
            (3, 4),
        ),
    )

    assert np.allclose(smoothed["win_rate"], 0.5666666667)
    assert np.allclose(smoothed["matchup_1v1"], 0.6)
    assert np.allclose(smoothed["synergy_2vx"], 0.6833333333)
    assert np.allclose(smoothed["m1v1_eff_n"], 10.0)
    assert np.allclose(smoothed["s2vx_eff_n"], 10.0)


def test_s2vx_floor_prior_for_ladder_selects_neutral_build_group_floor() -> None:
    per_side = np.array([[0.6, 0.7]], dtype=np.float64)

    assert (
        s2vx_floor_prior_for_ladder(
            S2VX_NEUTRAL_FLOOR_LADDER,
            neutral_prior=0.5,
            per_side_prior=per_side,
        )
        == 0.5
    )
    assert s2vx_floor_prior_for_ladder(
        ("build", "nobuild", "champion"),
        neutral_prior=0.5,
        per_side_prior=per_side,
    ) is per_side
