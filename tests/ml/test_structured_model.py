# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from app.ml.dataset import SplitData
from app.ml.structured_model import (
    MATCHUP_EMBEDDING_DIM,
    MATCHUP_OBJECT_DIM,
    PAIR_EMBEDDING_DIM,
    SYNERGY_OBJECT_DIM,
    MatchupEncoder,
    PairEncoder,
    StructuredModelConfig,
    build_matchup_objects,
    build_structured_input_arrays,
    build_synergy_objects,
    confidence_from_counts,
    logit_prob,
)
from app.ml.train import _cache_raw_tensor_split, _structured_tensors_from_raw


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    win_rate = np.array(
        [
            [0.55, 0.45, 0.60, 0.40, 0.52, 0.48, 0.57, 0.43, 0.51, 0.49],
            [0.35, 0.62, 0.58, 0.47, 0.53, 0.63, 0.38, 0.44, 0.56, 0.41],
        ],
        dtype=np.float32,
    )
    matchup_1v1 = np.linspace(0.31, 0.69, 50, dtype=np.float32).reshape(2, 25)
    synergy_2vx = np.linspace(0.36, 0.64, 40, dtype=np.float32).reshape(2, 20)
    p1_cnt = np.arange(1, 21, dtype=np.float32).reshape(2, 10)
    m1v1_cnt = np.arange(1, 51, dtype=np.float32).reshape(2, 25)
    s2vx_cnt = np.arange(1, 41, dtype=np.float32).reshape(2, 20)
    return win_rate, matchup_1v1, synergy_2vx, p1_cnt, m1v1_cnt, s2vx_cnt


def test_structured_objects_include_expected_baseline_shapes() -> None:
    win_rate, matchup_1v1, synergy_2vx, p1_cnt, m1v1_cnt, s2vx_cnt = _inputs()

    arrays = build_structured_input_arrays(
        win_rate=win_rate,
        matchup_1v1=matchup_1v1,
        synergy_2vx=synergy_2vx,
        p1_cnt=p1_cnt,
        m1v1_cnt=m1v1_cnt,
        s2vx_cnt=s2vx_cnt,
        confidence_strength=20.0,
    )

    assert arrays.synergy_objects.shape == (2, 2, 10, 6)
    assert arrays.matchup_objects.shape == (2, 25, 6)
    assert arrays.synergy_objects.shape[-1] == SYNERGY_OBJECT_DIM
    assert arrays.matchup_objects.shape[-1] == MATCHUP_OBJECT_DIM


def test_synergy_logit_delta_uses_expected_identity_baseline() -> None:
    win_rate, _, synergy_2vx, _, _, s2vx_cnt = _inputs()

    objects = build_synergy_objects(
        win_rate,
        synergy_2vx,
        s2vx_cnt,
        confidence_strength=20.0,
    )

    identity_logits = logit_prob(win_rate[:, :5])
    joint_logit = logit_prob(synergy_2vx[:, 0])
    expected_logit = 0.5 * (identity_logits[:, 0] + identity_logits[:, 1])

    np.testing.assert_allclose(objects[:, 0, 0, 0], joint_logit)
    np.testing.assert_allclose(objects[:, 0, 0, 1], identity_logits[:, 0])
    np.testing.assert_allclose(objects[:, 0, 0, 2], identity_logits[:, 1])
    np.testing.assert_allclose(objects[:, 0, 0, 3], expected_logit)
    np.testing.assert_allclose(
        objects[:, 0, 0, 4],
        confidence_from_counts(s2vx_cnt[:, 0], prior_strength=20.0),
    )
    np.testing.assert_allclose(objects[:, 0, 0, 5], joint_logit - expected_logit)


def test_matchup_logit_delta_uses_expected_identity_advantage() -> None:
    win_rate, matchup_1v1, _, _, m1v1_cnt, _ = _inputs()

    objects = build_matchup_objects(
        win_rate,
        matchup_1v1,
        m1v1_cnt,
        confidence_strength=20.0,
    )

    blue_idx = 2
    red_idx = 3
    feature_idx = blue_idx * 5 + red_idx
    blue_logits = logit_prob(win_rate[:, :5])
    red_logits = logit_prob(win_rate[:, 5:])
    matchup_logit = logit_prob(matchup_1v1[:, feature_idx])
    expected_logit = blue_logits[:, blue_idx] - red_logits[:, red_idx]

    np.testing.assert_allclose(objects[:, feature_idx, 0], matchup_logit)
    np.testing.assert_allclose(objects[:, feature_idx, 1], blue_logits[:, blue_idx])
    np.testing.assert_allclose(objects[:, feature_idx, 2], red_logits[:, red_idx])
    np.testing.assert_allclose(objects[:, feature_idx, 3], expected_logit)
    np.testing.assert_allclose(
        objects[:, feature_idx, 4],
        confidence_from_counts(m1v1_cnt[:, feature_idx], prior_strength=20.0),
    )
    np.testing.assert_allclose(objects[:, feature_idx, 5], matchup_logit - expected_logit)


def test_encoders_accept_updated_object_dimensions() -> None:
    config = StructuredModelConfig(
        role_embedding_dim=7,
        matchup_slot_embedding_dim=3,
        pair_hidden=(11,),
        matchup_hidden=(13,),
        dropout=0.0,
    )
    pair_encoder = PairEncoder(
        role_embedding_dim=config.role_embedding_dim,
        hidden=config.pair_hidden,
        dropout=config.dropout,
    )
    matchup_encoder = MatchupEncoder(config)

    pair_first_layer = pair_encoder.net[0]
    matchup_first_layer = matchup_encoder.net[0]
    assert isinstance(pair_first_layer, nn.Linear)
    assert isinstance(matchup_first_layer, nn.Linear)
    assert pair_first_layer.in_features == SYNERGY_OBJECT_DIM + config.role_embedding_dim
    assert matchup_first_layer.in_features == (
        MATCHUP_OBJECT_DIM + config.matchup_slot_embedding_dim
    )

    pair_features = torch.zeros(4, 10, SYNERGY_OBJECT_DIM)
    matchup_features = torch.zeros(4, 25, MATCHUP_OBJECT_DIM)
    role_pair_ids = torch.arange(10)

    assert pair_encoder(pair_features, role_pair_ids).shape == (4, 10, PAIR_EMBEDDING_DIM)
    assert matchup_encoder(matchup_features).shape == (4, 25, MATCHUP_EMBEDDING_DIM)


def test_raw_tensor_fast_path_matches_numpy_feature_builder() -> None:
    win_rate, matchup_1v1, synergy_2vx, p1_cnt, m1v1_cnt, s2vx_cnt = _inputs()
    arrays = build_structured_input_arrays(
        win_rate=win_rate,
        matchup_1v1=matchup_1v1,
        synergy_2vx=synergy_2vx,
        p1_cnt=p1_cnt,
        m1v1_cnt=m1v1_cnt,
        s2vx_cnt=s2vx_cnt,
        confidence_strength=20.0,
    )
    raw = _cache_raw_tensor_split(
        "test",
        SplitData(
            win_rate=win_rate,
            matchup_1v1=matchup_1v1,
            synergy_2vx=synergy_2vx,
            p1_cnt=p1_cnt,
            m1v1_cnt=m1v1_cnt,
            s2vx_cnt=s2vx_cnt,
            blue_win=np.array([1.0, 0.0], dtype=np.float32),
        ),
        device="cpu",
    )
    fast = _structured_tensors_from_raw(
        raw,
        confidence_strength=20.0,
        delta_baseline_mode="logit",
    )

    np.testing.assert_allclose(fast["base_features"].numpy(), arrays.base_features)
    np.testing.assert_allclose(fast["synergy_objects"].numpy(), arrays.synergy_objects)
    np.testing.assert_allclose(fast["matchup_objects"].numpy(), arrays.matchup_objects)
    np.testing.assert_allclose(
        fast["confidence_summaries"].numpy(),
        arrays.confidence_summaries,
    )
