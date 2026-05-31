from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.utils.smoothing import nested_shrunk_rate
from app.ml.hgnn_model import build_hgnn_inputs
from app.ml.predictor import _interaction_pooling_from_cache_meta
from app.ml.priors import PriorTables


def test_nested_pooling_backs_off_to_dense_parent_effective_support() -> None:
    build_rate = np.array([[0.50]], dtype=np.float64)
    build_count = np.array([[0.0]], dtype=np.float64)
    nobuild_rate = np.array([[0.50]], dtype=np.float64)
    nobuild_count = np.array([[0.0]], dtype=np.float64)
    champ_rate = np.array([[0.70]], dtype=np.float64)
    champ_count = np.array([[800.0]], dtype=np.float64)

    pooled, eff_n = nested_shrunk_rate(
        [build_rate, nobuild_rate, champ_rate],
        [build_count, nobuild_count, champ_count],
        strengths=[20.0, 20.0, 1.0],
        floor_prior=0.5,
        amplification_threshold=0.0,
    )

    assert pooled.shape == (1, 1)
    assert eff_n.shape == (1, 1)
    assert np.isclose(pooled[0, 0], (0.70 * 800.0 + 0.5) / 801.0)
    assert eff_n[0, 0] == 800.0


def test_hgnn_inputs_use_effective_support_for_variance_and_raw_support_features() -> None:
    raw_counts = np.zeros((1, 25), dtype=np.float32)
    effective_counts = np.full((1, 25), 800.0, dtype=np.float32)

    inputs = build_hgnn_inputs(
        champion_id=np.zeros((1, 10), dtype=np.int64),
        build_id=np.zeros((1, 10), dtype=np.int64),
        win_rate=np.full((1, 10), 0.5, dtype=np.float32),
        matchup_1v1=np.full((1, 25), 0.7, dtype=np.float32),
        synergy_2vx=np.full((1, 20), 0.6, dtype=np.float32),
        p1_cnt=np.zeros((1, 10), dtype=np.float32),
        m1v1_cnt=raw_counts,
        s2vx_cnt=np.zeros((1, 20), dtype=np.float32),
        m1v1_eff_n=effective_counts,
        s2vx_eff_n=np.full((1, 20), 400.0, dtype=np.float32),
        strength=30.0,
    )

    assert float(inputs["var_1v1"][0, 0]) < 0.001
    assert float(inputs["conf_1v1"][0, 0]) == 0.0
    assert float(inputs["log_count_1v1"][0, 0]) == 0.0
    assert float(inputs["missing_1v1"][0, 0]) == 1.0


def test_prior_table_backoff_lookups_match_training_orientation() -> None:
    priors = PriorTables(
        p1={},
        m1v1={(1, "TOP", "carry", 2, "JUNGLE", "tank"): (0.65, 10)},
        s2vx={(1, "TOP", "carry", 3, "MIDDLE", "mage"): (0.58, 12)},
        m1v1_nb={(1, "TOP", 2, "JUNGLE"): (0.62, 100)},
        m1v1_champ={(1, 2): (0.60, 800)},
        s2vx_nb={(1, "TOP", 3, "MIDDLE"): (0.57, 200)},
        s2vx_champ={(1, 3): (0.56, 900)},
    )

    blue = [(1, "TOP", "carry")]
    red = [(2, "JUNGLE", "tank")]
    assert priors.lookup_1v1_blue(blue, red)[0][0] == 0.65
    assert priors.lookup_1v1_blue_nobuild(blue, red)[0][0] == 0.62
    assert priors.lookup_1v1_blue_champ(blue, red)[0][0] == 0.60

    team = [
        (3, "MIDDLE", "mage"),
        (1, "TOP", "carry"),
        (9, "JUNGLE", "none"),
        (10, "BOTTOM", "none"),
        (11, "UTILITY", "none"),
    ]
    s2vx_wr, s2vx_cnt = priors.lookup_2vx_team(team)
    s2vx_nb_wr, _ = priors.lookup_2vx_team_nobuild(team)
    s2vx_ch_wr, _ = priors.lookup_2vx_team_champ(team)
    assert s2vx_wr[0] == 0.58
    assert s2vx_cnt[0] == 12
    assert s2vx_nb_wr[0] == 0.57
    assert s2vx_ch_wr[0] == 0.56


def test_predictor_reuses_cache_meta_interaction_strengths(tmp_path: Path) -> None:
    strengths = {
        "m1v1": [53.2, 149.1, 276.0],
        "s2vx": [54.1, 191.0, 292.9],
    }
    (tmp_path / "cache_meta.json").write_text(
        json.dumps(
            {
                "smoothing": {
                    "interaction_nested_pooling": True,
                    "interaction_level_strengths": strengths,
                }
            }
        )
    )

    nested_pooling, level_strengths = _interaction_pooling_from_cache_meta(
        tmp_path,
        fallback_strength=20.0,
    )

    assert nested_pooling is True
    assert level_strengths == strengths


def test_predictor_falls_back_when_cache_meta_lacks_complete_strengths(tmp_path: Path) -> None:
    (tmp_path / "cache_meta.json").write_text(
        json.dumps(
            {
                "smoothing": {
                    "interaction_nested_pooling": True,
                    "interaction_level_strengths": {"m1v1": [53.2], "s2vx": [54.1, 191.0, 292.9]},
                }
            }
        )
    )

    nested_pooling, level_strengths = _interaction_pooling_from_cache_meta(
        tmp_path,
        fallback_strength=20.0,
    )

    assert nested_pooling is False
    assert level_strengths == {"m1v1": [20.0], "s2vx": [20.0]}
