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
            identity_static_sidecar_dim=2,
            identity_full_game_sidecar_dim=3,
            identity_temporal_sidecar_dim=4,
            use_learned_semantic_moe=True,
            use_semantic_group_features=True,
            patch_feature_dim=0,
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


def _fake_sidecar() -> SimpleNamespace:
    return SimpleNamespace(
        dims=SimpleNamespace(
            as_dict=lambda: {"static": 2, "full_game": 3, "temporal": 4, "total": 9}
        )
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


def test_predictor_rejects_mismatched_team_assignments(
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
        encoder_sidecar=None,
        semantic_context_lookup=_semantic_context_lookup(),
        device="cpu",
    )
    blue = list(range(1, 6))
    red = list(range(6, 11))

    with pytest.raises(ValueError, match="blue roles must match"):
        predictor(
            blue,
            red,
            {champion: position for champion, position in zip(blue[:-1], POSITIONS)},
            {champion: position for champion, position in zip(red, POSITIONS)},
            {champion: 0 for champion in blue},
            {champion: 0 for champion in red},
        )


class _BuildSensitiveModel:
    """Logit depends only on the assigned build ids; no semantic/sidecar path."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            n_champions=1000,
            n_builds=2,
            build_vocab=("carry", "tank"),
            identity_static_sidecar_dim=0,
            identity_full_game_sidecar_dim=0,
            identity_temporal_sidecar_dim=0,
            use_learned_semantic_moe=False,
            use_semantic_group_features=False,
            patch_feature_dim=0,
        )

    def to(self, _device: str) -> "_BuildSensitiveModel":
        return self

    def eval(self) -> "_BuildSensitiveModel":
        return self

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        build_id = inputs["build_id"]
        return {"final_logit": build_id.float().sum(dim=1) * 0.4 - 2.0}


class _PatchSensitiveModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            n_champions=1000,
            n_builds=1,
            build_vocab=("carry",),
            identity_static_sidecar_dim=0,
            identity_full_game_sidecar_dim=0,
            identity_temporal_sidecar_dim=0,
            use_learned_semantic_moe=False,
            use_semantic_group_features=False,
            patch_feature_dim=2,
        )
        self.seen_patch_features: torch.Tensor | None = None

    def to(self, _device: str) -> "_PatchSensitiveModel":
        return self

    def eval(self) -> "_PatchSensitiveModel":
        return self

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_features = inputs["patch_features"]
        self.seen_patch_features = patch_features.detach().cpu()
        return {"final_logit": patch_features[:, 0]}


class _PatchProvider:
    def __init__(self, features: tuple[float, float]) -> None:
        self.features = np.asarray(features, dtype=np.float32)

    def features_for_batch(self, n: int) -> np.ndarray:
        return np.tile(self.features.reshape(1, 2), (int(n), 1))


def test_predict_marginal_matches_brute_force_average() -> None:
    from itertools import product

    from app.ml.build_catalog import build_catalog

    p1: dict[tuple[int, str, str], tuple[float, int]] = {}
    for champion, position in zip(range(1, 11), POSITIONS + POSITIONS):
        p1[(champion, position, "carry")] = (0.5, 80)
        p1[(champion, position, "tank")] = (0.5, 20)
    catalog = build_catalog(p1, ("carry", "tank"))
    predictor = WinRatePredictor(
        _BuildSensitiveModel(),
        PriorTables(p1=p1),
        prior_strength=20.0,
        smoothing_prior_strength=20.0,
        amplification_threshold=0.0,
        smoothing_mode="cascade",
        prior_confidence_matchups=50.0,
        encoder_sidecar=None,
        semantic_context_lookup=None,
        device="cpu",
    )
    blue = list(range(1, 6))
    red = list(range(6, 11))

    result = predictor.predict_marginal(
        blue,
        red,
        {champion: position for champion, position in zip(blue, POSITIONS)},
        {champion: position for champion, position in zip(red, POSITIONS)},
        catalog=catalog,
        k_slot=2,
        max_worlds=2048,
        early_stop_mass=2.0,
    )

    # Identical cells -> one shared smoothed prior vector per slot; brute-force
    # the probability-space marginal over all 2^10 build worlds.
    probs = catalog.prior_vector(1, "TOP").probabilities
    numerator, mass = 0.0, 0.0
    for combo in product(range(2), repeat=10):
        weight = float(np.prod([probs[i] for i in combo]))
        numerator += weight / (1.0 + np.exp(-(sum(combo) * 0.4 - 2.0)))
        mass += weight
    assert result.probability == pytest.approx(numerator / mass)
    assert result.retained_joint_mass == pytest.approx(1.0)
    assert result.n_worlds == 1024
    assert result.low_confidence is False
    assert result.fallback_sources == ("champion_role",) * 10
    assert result.build_source == "pregame_marginal_build"


def test_predictor_rejects_patch_checkpoint_without_provider() -> None:
    with pytest.raises(ValueError, match="requires patch_features"):
        WinRatePredictor(
            _PatchSensitiveModel(),
            _priors(),
            prior_strength=20.0,
            smoothing_prior_strength=20.0,
            amplification_threshold=0.0,
            smoothing_mode="cascade",
            prior_confidence_matchups=50.0,
            encoder_sidecar=None,
            semantic_context_lookup=None,
            device="cpu",
        )


def test_predictor_supplies_runtime_patch_features() -> None:
    model = _PatchSensitiveModel()
    predictor = WinRatePredictor(
        model,
        _priors(),
        prior_strength=20.0,
        smoothing_prior_strength=20.0,
        amplification_threshold=0.0,
        smoothing_mode="cascade",
        prior_confidence_matchups=50.0,
        encoder_sidecar=None,
        semantic_context_lookup=None,
        device="cpu",
        patch_feature_provider=_PatchProvider((0.25, 1.0)),
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
    batch = predictor.predict_batch(
        [
            (
                blue,
                red,
                {champion: position for champion, position in zip(blue, POSITIONS)},
                {champion: position for champion, position in zip(red, POSITIONS)},
                {champion: 0 for champion in blue},
                {champion: 0 for champion in red},
            ),
            (
                blue,
                red,
                {champion: position for champion, position in zip(blue, POSITIONS)},
                {champion: position for champion, position in zip(red, POSITIONS)},
                {champion: 0 for champion in blue},
                {champion: 0 for champion in red},
            ),
        ]
    )

    assert probability == pytest.approx(1.0 / (1.0 + np.exp(-0.25)))
    assert batch.tolist() == pytest.approx([probability, probability])
    assert model.seen_patch_features is not None
    assert tuple(model.seen_patch_features.shape) == (2, 2)
    assert model.seen_patch_features[:, 0].tolist() == pytest.approx([0.25, 0.25])
    assert model.seen_patch_features[:, 1].tolist() == pytest.approx([1.0, 1.0])


def test_predictor_rejects_out_of_vocab_build_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_lookups(monkeypatch)
    predictor = WinRatePredictor(
        _SemanticModel(),
        _priors(),
        prior_strength=20.0,
        smoothing_prior_strength=20.0,
        amplification_threshold=0.0,
        smoothing_mode="cascade",
        prior_confidence_matchups=50.0,
        encoder_sidecar=None,
        semantic_context_lookup=_semantic_context_lookup(),
        device="cpu",
    )
    blue = list(range(1, 6))
    red = list(range(6, 11))

    with pytest.raises(ValueError, match="outside the model build vocab"):
        predictor(
            blue,
            red,
            {champion: position for champion, position in zip(blue, POSITIONS)},
            {champion: position for champion, position in zip(red, POSITIONS)},
            {champion: 1 for champion in blue},  # vocab size is 1: only id 0
            {champion: 0 for champion in red},
        )


def test_predictor_rejects_prior_vocab_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_lookups(monkeypatch)
    model = _SemanticModel()
    model.config.build_vocab = ("other",)

    with pytest.raises(ValueError, match="do not match the checkpoint build_vocab"):
        WinRatePredictor(
            model,
            _priors(),
            prior_strength=20.0,
            smoothing_prior_strength=20.0,
            amplification_threshold=0.0,
            smoothing_mode="cascade",
            prior_confidence_matchups=50.0,
            encoder_sidecar=None,
            semantic_context_lookup=_semantic_context_lookup(),
            device="cpu",
        )


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
        lambda _path: _fake_sidecar(),
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


def test_load_predictor_rejects_patch_checkpoint_without_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor_module, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        predictor_module,
        "load_hgnn_model",
        lambda _path, *, device: (_PatchSensitiveModel(), None, 20.0),
    )
    monkeypatch.setattr(predictor_module, "load_priors", _priors)

    with pytest.raises(ValueError, match="requires patch_features"):
        predictor_module.load_predictor(
            cfg=TrainConfig(model_path=Path("unused.pt")),
            dataset_cfg=DatasetConfig(encoder_sidecar_path=None),
        )


def test_load_predictor_builds_serving_patch_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, tuple[int, int]] = {}

    class FakeServingPatchFeatureProvider:
        @classmethod
        def from_train_aggregate(
            cls,
            *,
            cfg: DatasetConfig,
            season: int,
            patch: int,
        ) -> _PatchProvider:
            del cfg
            seen["serving_patch"] = (season, patch)
            return _PatchProvider((0.1, 1.0))

    monkeypatch.setattr(predictor_module, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        predictor_module,
        "load_hgnn_model",
        lambda _path, *, device: (_PatchSensitiveModel(), None, 20.0),
    )
    monkeypatch.setattr(predictor_module, "load_priors", _priors)
    monkeypatch.setattr(
        predictor_module,
        "ServingPatchFeatureProvider",
        FakeServingPatchFeatureProvider,
    )

    predictor_module.load_predictor(
        cfg=TrainConfig(model_path=Path("unused.pt")),
        dataset_cfg=DatasetConfig(encoder_sidecar_path=None),
        serving_patch=(16, 11),
    )

    assert seen["serving_patch"] == (16, 11)
