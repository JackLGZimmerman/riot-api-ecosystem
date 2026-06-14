from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.ml.config import TrainConfig
from app.ml.dataset import SplitData
from app.ml.hgnn_model import HGNNConfig
from app.ml.train import (
    _drop_unused_model_arrays,
    _validate_split_targets,
    _validate_train_config,
)


def _split(labels: np.ndarray) -> SplitData:
    n = int(labels.size)
    return SplitData(
        win_rate=np.zeros((n, 10), dtype=np.float32),
        p1_cnt=np.zeros((n, 10), dtype=np.float32),
        blue_win=labels.astype(np.float64, copy=False),
        patch_features=np.zeros((n, 2), dtype=np.float32),
        identity_static_sidecar=np.zeros((n, 10, 2), dtype=np.float32),
        identity_full_game_sidecar=np.zeros((n, 10, 3), dtype=np.float32),
        identity_temporal_sidecar=np.zeros((n, 10, 4), dtype=np.float32),
        identity_encoder_support=np.zeros((n, 10), dtype=np.float32),
    )


def test_default_model_config_drops_optional_model_arrays_before_tensor_cache() -> None:
    split = _split(np.array([0, 1], dtype=np.float64))

    dropped = _drop_unused_model_arrays(split, HGNNConfig())
    assert dropped.patch_features is None
    assert dropped.identity_static_sidecar is None
    assert dropped.identity_full_game_sidecar is None
    assert dropped.identity_temporal_sidecar is None
    assert dropped.identity_encoder_support is None

    feature_kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(patch_feature_dim=2),
    )
    assert feature_kept.patch_features is split.patch_features

    moe_kept = _drop_unused_model_arrays(
        split,
        HGNNConfig(use_learned_semantic_moe=True),
    )
    assert moe_kept.identity_static_sidecar is split.identity_static_sidecar
    assert moe_kept.identity_full_game_sidecar is split.identity_full_game_sidecar
    assert moe_kept.identity_temporal_sidecar is split.identity_temporal_sidecar
    assert moe_kept.identity_encoder_support is split.identity_encoder_support


def test_train_config_rejects_unknown_raw_tensor_cache_device() -> None:
    with pytest.raises(ValueError, match="raw_tensor_cache_device"):
        _validate_train_config(
            replace(TrainConfig(), raw_tensor_cache_device="accelerator")
        )


def test_semantic_moe_head_fails_early_without_sidecar_cache_arrays() -> None:
    split = replace(
        _split(np.array([0, 1], dtype=np.float64)),
        identity_temporal_sidecar=None,
    )

    with pytest.raises(ValueError, match="semantic MoE head requires cache arrays"):
        _drop_unused_model_arrays(split, HGNNConfig(use_learned_semantic_moe=True))


def test_training_target_validation_catches_degenerate_cache_split() -> None:
    splits = {
        "train": _split(np.array([0, 1], dtype=np.float64)),
        "test": _split(np.zeros(3, dtype=np.float64)),
    }

    with pytest.raises(ValueError, match="test split has degenerate blue_win labels"):
        _validate_split_targets(splits)
