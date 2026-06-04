from __future__ import annotations

import numpy as np
import torch

from app.ml.encoder_sidecar import save_encoder_sidecar
from app.ml.config import TrainConfig
from app.ml.semantic_group_features import (
    SEMANTIC_GROUP_FEATURE_DIM,
    SEMANTIC_GROUP_FEATURE_INDEX,
)
from app.ml.train import RawTensorSplit, _hgnn_config_from_meta, _SemanticContextCalibrationLoss


def _meta() -> dict:
    return {
        "n_champions": 10,
        "n_builds": 3,
        "build_vocab": ("ability_power", "ar_tank", "mr_tank"),
    }


def test_production_defaults_use_identity_prior_hgnn() -> None:
    cfg = _hgnn_config_from_meta(_meta())

    assert TrainConfig().checkpoint_metric == "val_threshold_accuracy"
    assert cfg.n_champions == 10
    assert cfg.n_builds == 3
    assert cfg.build_vocab == ("ability_power", "ar_tank", "mr_tank")
    assert cfg.use_relationship_integrations is False
    assert cfg.use_semantic_group_features is False
    assert cfg.semantic_group_feature_dim == SEMANTIC_GROUP_FEATURE_DIM


def test_relationship_override_can_be_enabled_explicitly() -> None:
    cfg = _hgnn_config_from_meta(
        _meta(),
        overrides={"use_relationship_integrations": True},
    )

    assert cfg.use_relationship_integrations is True


def test_sidecar_dims_are_loaded_from_cache_metadata() -> None:
    cfg = _hgnn_config_from_meta(
        {
            **_meta(),
            "identity_encoder_sidecar": {
                "dims": {
                    "static": 16,
                    "full_game": 64,
                    "temporal": 64,
                }
            },
        },
    )

    assert cfg.identity_static_sidecar_dim == 16
    assert cfg.identity_full_game_sidecar_dim == 64
    assert cfg.identity_temporal_sidecar_dim == 64


def test_sidecar_dims_fall_back_to_encoder_sidecar_path_when_cache_meta_is_empty(
    tmp_path,
) -> None:
    sidecar_path = save_encoder_sidecar(
        tmp_path / "sidecar.npz",
        champion_id=np.array([1], dtype=np.int32),
        teamposition=np.array(["TOP"]),
        build=np.array(["ability_power"]),
        static_latents=np.zeros((1, 2), dtype=np.float32),
        full_game_latents=np.zeros((1, 3), dtype=np.float32),
        temporal_latents=np.zeros((1, 4), dtype=np.float32),
        support=np.ones(1, dtype=np.float32),
    )

    cfg = _hgnn_config_from_meta(
        _meta(),
        encoder_sidecar_path=sidecar_path,
    )

    assert cfg.identity_static_sidecar_dim == 2
    assert cfg.identity_full_game_sidecar_dim == 3
    assert cfg.identity_temporal_sidecar_dim == 4


def test_auto_hgnn_override_rejects_removed_dimension_resolution() -> None:
    try:
        _hgnn_config_from_meta(_meta(), overrides={"dropout": "auto"})
    except ValueError as exc:
        assert "does not support auto" in str(exc)
    else:
        raise AssertionError("Expected unsupported auto override to fail")


def test_semantic_context_calibration_loss_rewards_matching_side_labels() -> None:
    build_vocab = ("ability_power", "ar_tank")
    features = torch.zeros((1, 10, SEMANTIC_GROUP_FEATURE_DIM), dtype=torch.float32)
    features[0, 5:8, SEMANTIC_GROUP_FEATURE_INDEX["burst"]] = 1.0
    features[0, 5:8, SEMANTIC_GROUP_FEATURE_INDEX["damage"]] = 1.0
    champion_id = torch.zeros((1, 10), dtype=torch.long)
    champion_id[0, 4] = 12
    champion_id[0, 9] = 12
    build_id = torch.zeros((1, 10), dtype=torch.long)
    build_id[0, 4] = 1
    build_id[0, 9] = 1
    raw = RawTensorSplit(
        win_rate=torch.zeros((1, 10), dtype=torch.float32),
        p1_cnt=torch.ones((1, 10), dtype=torch.float32),
        blue_win=torch.ones(1, dtype=torch.float32),
        champion_id=champion_id,
        build_id=build_id,
        semantic_group_features=features,
    )
    calibrator = _SemanticContextCalibrationLoss(
        build_vocab=build_vocab,
        train_cfg=TrainConfig(
            semantic_context_calibration_loss_weight=1.0,
            semantic_context_calibration_min_count=1,
            semantic_context_calibration_tail_weight=1.0,
        ),
        device="cpu",
    )

    wrong_loss = calibrator({"final_logit": torch.zeros(1)}, torch.ones(1), raw)
    aligned_loss = calibrator(
        {"final_logit": torch.full((1,), 20.0)},
        torch.ones(1),
        raw,
    )

    assert wrong_loss.item() > 0.05
    assert aligned_loss.item() < wrong_loss.item() * 0.01


def test_semantic_context_calibration_loss_uses_slot_delta_predictions() -> None:
    build_vocab = ("ability_power", "ar_tank")
    features = torch.zeros((1, 10, SEMANTIC_GROUP_FEATURE_DIM), dtype=torch.float32)
    features[0, 5:8, SEMANTIC_GROUP_FEATURE_INDEX["burst"]] = 1.0
    features[0, 5:8, SEMANTIC_GROUP_FEATURE_INDEX["damage"]] = 1.0
    champion_id = torch.zeros((1, 10), dtype=torch.long)
    champion_id[0, 4] = 12
    champion_id[0, 9] = 12
    build_id = torch.zeros((1, 10), dtype=torch.long)
    build_id[0, 4] = 1
    build_id[0, 9] = 1
    raw = RawTensorSplit(
        win_rate=torch.zeros((1, 10), dtype=torch.float32),
        p1_cnt=torch.ones((1, 10), dtype=torch.float32),
        blue_win=torch.ones(1, dtype=torch.float32),
        champion_id=champion_id,
        build_id=build_id,
        semantic_group_features=features,
    )
    calibrator = _SemanticContextCalibrationLoss(
        build_vocab=build_vocab,
        train_cfg=TrainConfig(
            semantic_context_calibration_loss_weight=1.0,
            semantic_context_calibration_min_count=1,
            semantic_context_calibration_tail_weight=1.0,
        ),
        device="cpu",
    )
    base_outputs = {
        "base_logit": torch.zeros(1),
        "context_logit": torch.zeros(1),
        "final_logit": torch.zeros(1),
        "semantic_moe_logit": torch.zeros(1),
        "semantic_moe_slot_delta": torch.zeros((1, 10)),
    }
    focused_outputs = {
        **base_outputs,
        "semantic_moe_slot_delta": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, -8.0]]
        ),
    }

    base_loss = calibrator(base_outputs, torch.ones(1), raw)
    focused_loss = calibrator(focused_outputs, torch.ones(1), raw)

    assert focused_loss.item() < base_loss.item() * 0.05
