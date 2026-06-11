from __future__ import annotations

import numpy as np
import pytest
import torch

from app.ml.encoder_sidecar import save_encoder_sidecar
from app.ml.config import (
    DEFAULT_ENCODER_SIDECAR_PATH,
    DEFAULT_PRODUCTION_METRICS_PATH,
    DEFAULT_PRODUCTION_MODEL_PATH,
    DEFAULT_TRAIN_BATCH_CAP,
    DatasetConfig,
    TrainConfig,
)
from app.ml.semantic_group_features import (
    SEMANTIC_GROUP_FEATURE_DIM,
)
from app.ml.hgnn_model import HGNNConfig, hgnn_config_payload
from app.ml.train import (
    PRODUCTION_SEMANTIC_MOE_ARCHITECTURE,
    _freeze_warm_start_loaded_parameters,
    _hgnn_config_from_meta,
    _batch_indices,
    _validate_train_output_paths,
    _warm_start_hgnn_model,
    production_semantic_model_overrides,
)


def _meta() -> dict:
    return {
        "n_champions": 10,
        "n_builds": 3,
        "build_vocab": ("ability_power", "ar_tank", "mr_tank"),
    }


def test_production_defaults_use_all_identity_encoders() -> None:
    cfg = _hgnn_config_from_meta(
        {
            **_meta(),
            "identity_encoder_sidecar": {
                "dims": {"static": 16, "full_game": 64, "temporal": 64}
            },
        },
        overrides=production_semantic_model_overrides(),
    )

    dataset_cfg = DatasetConfig()
    train_cfg = TrainConfig()
    assert dataset_cfg.encoder_sidecar_path == DEFAULT_ENCODER_SIDECAR_PATH
    assert train_cfg.learning_rate == 3e-4
    assert train_cfg.weight_decay == 0.0
    assert train_cfg.patience == 5
    assert train_cfg.freeze_warm_start_loaded_parameters is False
    assert train_cfg.train_batch_cap == DEFAULT_TRAIN_BATCH_CAP
    assert train_cfg.raw_tensor_cache_device == "cpu"
    assert train_cfg.train_epoch_max_games is None
    assert not hasattr(train_cfg, "eval_test")
    assert dataset_cfg.test_fraction == 0.2
    assert not hasattr(dataset_cfg, "val_fraction")
    assert train_cfg.allow_production_artifact_overwrite is False
    assert train_cfg.model_path == DEFAULT_PRODUCTION_MODEL_PATH
    assert train_cfg.metrics_path == DEFAULT_PRODUCTION_METRICS_PATH
    assert cfg.n_champions == 10
    assert cfg.n_builds == 3
    assert cfg.build_vocab == ("ability_power", "ar_tank", "mr_tank")
    assert not hasattr(cfg, "use_relationship_integrations")
    assert cfg.identity_static_sidecar_dim == 16
    assert cfg.identity_full_game_sidecar_dim == 64
    assert cfg.identity_temporal_sidecar_dim == 64
    assert cfg.use_identity_static_sidecar is False
    assert cfg.use_identity_full_game_sidecar is False
    assert cfg.use_identity_temporal_sidecar is False
    assert cfg.use_learned_semantic_moe is True
    assert cfg.use_semantic_group_features is True
    assert cfg.semantic_moe_num_experts == 128
    assert cfg.semantic_moe_top_k == 32
    assert cfg.semantic_group_feature_dim == SEMANTIC_GROUP_FEATURE_DIM
    assert cfg.semantic_moe_architecture == PRODUCTION_SEMANTIC_MOE_ARCHITECTURE


def test_training_refuses_production_artifact_paths_without_promotion_flag() -> None:
    with pytest.raises(ValueError, match="overwrite production artifacts"):
        _validate_train_output_paths(TrainConfig())

    _validate_train_output_paths(
        TrainConfig(allow_production_artifact_overwrite=True)
    )


def test_config_payload_omits_deprecated_view_top_k() -> None:
    payload = hgnn_config_payload(HGNNConfig())

    assert "semantic_moe_view_top_k" not in payload


def test_batch_indices_can_cap_train_rows_per_epoch() -> None:
    batches = list(
        _batch_indices(
            10,
            batch_size=4,
            shuffle=False,
            rng=np.random.default_rng(0),
            max_rows=6,
        )
    )

    assert [batch.tolist() for batch in batches] == [[0, 1, 2, 3], [4, 5]]


def test_loadout_and_patch_dims_are_loaded_from_cache_metadata() -> None:
    cfg = _hgnn_config_from_meta(
        {
            **_meta(),
            "loadout_feature_dim": 10,
            "patch_feature_dim": 2,
        },
    )

    assert cfg.loadout_feature_dim == 10
    assert cfg.patch_feature_dim == 2


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


class _TinyWarmStartModel(torch.nn.Module):
    def __init__(self, *, expanded: bool = False) -> None:
        super().__init__()
        self.shared = torch.nn.Linear(2, 2)
        self.changed = torch.nn.Linear(2, 3 if expanded else 1)


def test_warm_start_skips_shape_mismatches_and_freezes_loaded_parameters(
    tmp_path,
) -> None:
    source = _TinyWarmStartModel()
    with torch.no_grad():
        source.shared.weight.fill_(0.25)
        source.shared.bias.fill_(0.5)
        source.changed.weight.fill_(0.75)
        source.changed.bias.fill_(1.0)
    checkpoint_path = tmp_path / "warm.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint_path)

    target = _TinyWarmStartModel(expanded=True)
    missing = _warm_start_hgnn_model(target, checkpoint_path, device="cpu")

    assert "changed.weight" in missing
    assert "changed.bias" in missing
    assert torch.allclose(target.shared.weight, source.shared.weight)
    assert torch.allclose(target.shared.bias, source.shared.bias)

    _freeze_warm_start_loaded_parameters(target, missing_keys=missing)

    assert target.shared.weight.requires_grad is False
    assert target.shared.bias.requires_grad is False
    assert target.changed.weight.requires_grad is True
    assert target.changed.bias.requires_grad is True


def test_auto_hgnn_override_rejects_removed_dimension_resolution() -> None:
    try:
        _hgnn_config_from_meta(_meta(), overrides={"dropout": "auto"})
    except ValueError as exc:
        assert "does not support auto" in str(exc)
    else:
        raise AssertionError("Expected unsupported auto override to fail")
