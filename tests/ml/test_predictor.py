from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.ml import predictor as predictor_module
from app.ml.config import DatasetConfig, POSITIONS, TrainConfig
from app.ml.predictor import WinRatePredictor
from app.ml.priors import PriorTables
from app.ml.semantic_context_lookup import SemanticContextRawLookup
from app.ml.semantic_group_features import (
    CONTEXT_AXIS_INDEX,
    HIGH_HP_THRESHOLD,
    RANGED_ATTACK_RANGE_THRESHOLD,
    SEMANTIC_CONTEXT_RAW_DIM,
    SEMANTIC_GROUP_FEATURE_DIM,
    SEMANTIC_GROUP_FEATURE_INDEX,
)


class _SemanticModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            n_champions=1000,
            n_builds=1,
            build_vocab=("carry",),
            use_identity_static_sidecar=True,
            use_identity_full_game_sidecar=True,
            use_identity_temporal_sidecar=True,
            use_identity_semantic_context_head=False,
            use_learned_semantic_moe=True,
            use_semantic_group_features=True,
        )
        self.seen_features: torch.Tensor | None = None

    def to(self, _device: str) -> "_SemanticModel":
        return self

    def eval(self) -> "_SemanticModel":
        return self

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = inputs["semantic_group_features"]
        self.seen_features = features
        return {"final_logit": torch.zeros(1, dtype=torch.float32)}


def _priors() -> PriorTables:
    p1: dict[tuple[int, str, str], tuple[float, int]] = {}
    for champion, position in zip(range(1, 11), POSITIONS + POSITIONS):
        key = (champion, position, "carry")
        p1[key] = (0.5, 100)
    return PriorTables(p1=p1)


def _semantic_context_lookup() -> SemanticContextRawLookup:
    context: dict[tuple[int, str, str], np.ndarray] = {}
    for champion, position in zip(range(1, 11), POSITIONS + POSITIONS):
        key = (champion, position, "carry")
        context[key] = np.zeros(SEMANTIC_CONTEXT_RAW_DIM, dtype=np.float32)
    context[(1, "TOP", "carry")][CONTEXT_AXIS_INDEX["physical"]] = 0.42
    return SemanticContextRawLookup(context)


def _patch_static_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    hp = np.zeros(1001, dtype=np.float32)
    attack_range = np.zeros(1001, dtype=np.float32)
    hp[1] = HIGH_HP_THRESHOLD
    attack_range[1] = RANGED_ATTACK_RANGE_THRESHOLD + 1.0
    monkeypatch.setattr(
        predictor_module,
        "static_hp_range_lookups",
        lambda: (hp, attack_range),
    )


def test_predictor_supplies_semantic_group_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_lookups(monkeypatch)
    model = _SemanticModel()
    predictor = WinRatePredictor(
        model,
        _priors(),
        prior_strength=20.0,
        smoothing_prior_strength=20.0,
        amplification_threshold=0.0,
        smoothing_mode="cascade",
        prior_confidence_matchups=50.0,
        use_final_build_labels=True,
        draft_unknown_build_label="unknown",
        encoder_sidecar=None,
        semantic_context_lookup=_semantic_context_lookup(),
        device="cpu",
    )
    blue = list(range(1, 6))
    red = list(range(6, 11))

    probability = predictor(
        blue,
        red,
        {champion: position for champion, position in zip(blue, POSITIONS)},
        {champion: position for champion, position in zip(red, POSITIONS)},
        {champion: 0 for champion in blue},
        {champion: 0 for champion in red},
    )

    assert probability == pytest.approx(0.5)
    assert model.seen_features is not None
    assert tuple(model.seen_features.shape) == (1, 10, SEMANTIC_GROUP_FEATURE_DIM)
    idx = SEMANTIC_GROUP_FEATURE_INDEX
    assert model.seen_features[0, 0, idx["physical"]].item() == pytest.approx(0.42)
    assert model.seen_features[0, 0, idx["high_hp"]].item() == 1.0
    assert model.seen_features[0, 0, idx["ranged"]].item() == 1.0


def test_predictor_rejects_feature_heads_missing_from_runtime_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_lookups(monkeypatch)
    model = _SemanticModel()
    model.config.loadout_feature_dim = 10
    model.config.patch_feature_dim = 2
    model.config.use_player_priors = True

    with pytest.raises(
        ValueError,
        match="loadout_features, patch_features, player_prior_features",
    ):
        WinRatePredictor(
            model,
            _priors(),
            prior_strength=20.0,
            smoothing_prior_strength=20.0,
            amplification_threshold=0.0,
            smoothing_mode="cascade",
            prior_confidence_matchups=50.0,
            use_final_build_labels=True,
            draft_unknown_build_label="unknown",
            encoder_sidecar=None,
            semantic_context_lookup=_semantic_context_lookup(),
            device="cpu",
        )


def test_load_predictor_rejects_unsupported_heads_before_resource_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SemanticModel()
    model.config.loadout_feature_dim = 10

    monkeypatch.setattr(predictor_module, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        predictor_module,
        "load_hgnn_model",
        lambda _path, *, device: (model, None, 20.0),
    )
    monkeypatch.setattr(
        predictor_module,
        "load_priors",
        lambda: (_ for _ in ()).throw(AssertionError("load_priors called")),
    )

    with pytest.raises(ValueError, match="loadout_features"):
        predictor_module.load_predictor(cfg=TrainConfig(model_path=Path("unused.pt")))


def test_load_predictor_loads_semantic_context_for_grouped_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_lookups(monkeypatch)
    seen: dict[str, bool] = {}

    monkeypatch.setattr(predictor_module, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        predictor_module,
        "load_hgnn_model",
        lambda _path, *, device: (_SemanticModel(), None, 20.0),
    )
    monkeypatch.setattr(
        predictor_module.EncoderSidecarLookup,
        "load",
        lambda _path: None,
    )

    def fake_load_priors() -> PriorTables:
        seen["load_priors"] = True
        return _priors()

    def fake_load_semantic_context_raw_lookup() -> SemanticContextRawLookup:
        seen["load_semantic_context_raw_lookup"] = True
        return _semantic_context_lookup()

    monkeypatch.setattr(predictor_module, "load_priors", fake_load_priors)
    monkeypatch.setattr(
        predictor_module,
        "load_semantic_context_raw_lookup",
        fake_load_semantic_context_raw_lookup,
    )

    predictor_module.load_predictor(
        cfg=TrainConfig(model_path=Path("unused.pt")),
        dataset_cfg=DatasetConfig(encoder_sidecar_path=Path("sidecar.npz")),
    )

    assert seen["load_priors"] is True
    assert seen["load_semantic_context_raw_lookup"] is True
