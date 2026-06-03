from __future__ import annotations

from app.ml.config import TrainConfig
from app.ml.train import _hgnn_config_from_meta


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


def test_auto_hgnn_override_rejects_removed_dimension_resolution() -> None:
    try:
        _hgnn_config_from_meta(_meta(), overrides={"dropout": "auto"})
    except ValueError as exc:
        assert "does not support auto" in str(exc)
    else:
        raise AssertionError("Expected unsupported auto override to fail")
