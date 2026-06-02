from __future__ import annotations

from app.ml.config import TrainConfig
from app.ml.train import _hgnn_config_from_meta


def _meta() -> dict:
    return {
        "n_champions": 10,
        "n_builds": 3,
        "build_vocab": ("ability_power", "ar_tank", "mr_tank"),
        "classification": {
            "identity_context_dim": 24,
            "context_interpretable_dim": 14,
            "identity_context_raw_dim": 62,
        },
    }


def test_production_defaults_use_threshold_tuned_raw_semantic_context() -> None:
    cfg = _hgnn_config_from_meta(_meta())

    assert TrainConfig().checkpoint_metric == "val_threshold_accuracy"
    assert cfg.use_identity_conditioned_context
    assert cfg.identity_context_conditioning_type == "low_rank"
    assert cfg.identity_context_source == "raw"
    assert cfg.identity_context_rank == 16
    assert cfg.identity_context_hidden_dim == 64
    assert cfg.use_relationship_integrations is False


def test_shared_context_override_disables_identity_conditioning() -> None:
    cfg = _hgnn_config_from_meta(
        _meta(),
        overrides={
            "use_identity_conditioned_context": False,
            "identity_context_conditioning_type": "none",
        },
    )

    assert not cfg.use_identity_conditioned_context
    assert cfg.identity_context_conditioning_type == "none"
    assert cfg.identity_context_dim == 24
    assert cfg.identity_context_raw_dim == 62


def test_auto_hgnn_override_dimensions_resolve_from_cache_metadata() -> None:
    cfg = _hgnn_config_from_meta(
        _meta()
        | {
            "classification": {
                **_meta()["classification"],
                "identity_profile_dim": 9,
                "m1v1_detail_dim": 16,
            }
        },
        overrides={
            "identity_profile_dim": "auto",
            "m1v1_detail_dim": "auto",
        },
    )

    assert cfg.identity_profile_dim == 9
    assert cfg.m1v1_detail_dim == 16


def test_auto_hgnn_override_rejects_unknown_or_missing_metadata() -> None:
    try:
        _hgnn_config_from_meta(_meta(), overrides={"dropout": "auto"})
    except ValueError as exc:
        assert "does not support auto" in str(exc)
    else:
        raise AssertionError("Expected unsupported auto override to fail")

    try:
        _hgnn_config_from_meta(_meta(), overrides={"m1v1_detail_dim": "auto"})
    except ValueError as exc:
        assert "classification.m1v1_detail_dim" in str(exc)
    else:
        raise AssertionError("Expected missing auto metadata to fail")
