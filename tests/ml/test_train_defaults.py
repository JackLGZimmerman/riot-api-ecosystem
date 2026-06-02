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
