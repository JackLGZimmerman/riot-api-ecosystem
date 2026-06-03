from __future__ import annotations

import numpy as np
import pytest
import torch

from app.ml.encoder_sidecar import (
    EncoderSidecarLookup,
    build_encoder_sidecar_metadata,
    save_encoder_sidecar,
    validate_static_metadata,
    validate_train_only_metadata,
)
from app.ml.experiments.context_ablation import (
    SIDECAR_VARIANTS,
    sidecar_variant_overrides,
)
from app.ml.hgnn_model import HGNNConfig, HGNNWinModel, build_hgnn_inputs, swap_hgnn_inputs


def _save_tiny_sidecar(path):
    metadata = build_encoder_sidecar_metadata(
        static_features=("base_health", "armor_l18"),
        full_game_features=("damage_per_min",),
        temporal_features=("minute_0_damage",),
        split_metadata={"fit_split": "train"},
        encoder_configs={
            "static": {"latent_dim": 2},
            "full_game": {"latent_dim": 3},
            "temporal": {"latent_dim": 4},
        },
        extra={"static_encoder": {"source": "deterministic champion dictionary"}},
    )
    return save_encoder_sidecar(
        path,
        champion_id=np.array([1, 2], dtype=np.int32),
        teamposition=np.array(["TOP", "JUNGLE"]),
        build=np.array(["a", "b"]),
        static_latents=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        full_game_latents=np.ones((2, 3), dtype=np.float32),
        temporal_latents=np.full((2, 4), 2.0, dtype=np.float32),
        support=np.array([50.0, 0.0], dtype=np.float32),
        metadata=metadata,
    )


def _semantic_hgnn_inputs(batch_size: int = 2, seed: int = 9) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    champion_id = np.tile(np.arange(10), (batch_size, 1))
    build_id = np.tile(np.arange(10) % 3, (batch_size, 1))
    win_rate = rng.uniform(0.35, 0.65, size=(batch_size, 10)).astype(np.float32)
    p1_cnt = rng.uniform(0.0, 100.0, size=(batch_size, 10)).astype(np.float32)
    static = rng.normal(size=(batch_size, 10, 2)).astype(np.float32)
    full_game = rng.normal(size=(batch_size, 10, 3)).astype(np.float32)
    temporal = rng.normal(size=(batch_size, 10, 4)).astype(np.float32)
    support = rng.uniform(0.0, 80.0, size=(batch_size, 10)).astype(np.float32)
    return build_hgnn_inputs(
        champion_id=champion_id,
        build_id=build_id,
        win_rate=win_rate,
        p1_cnt=p1_cnt,
        strength=30.0,
        identity_static_sidecar=static,
        identity_full_game_sidecar=full_game,
        identity_temporal_sidecar=temporal,
        identity_encoder_support=support,
    )


def _semantic_context_config(**overrides) -> HGNNConfig:
    return HGNNConfig(
        n_champions=20,
        n_builds=3,
        dropout=0.0,
        identity_static_sidecar_dim=2,
        identity_full_game_sidecar_dim=3,
        identity_temporal_sidecar_dim=4,
        use_identity_semantic_context_head=True,
        **overrides,
    )


def _make_context_score_nonzero(model: HGNNWinModel) -> None:
    head = model.identity_semantic_context
    assert head is not None
    last = head.score[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.weight.fill_(0.03)
        if last.bias is not None:
            last.bias.zero_()


def test_sidecar_artifact_round_trips_all_three_blocks_and_missing_zero(tmp_path) -> None:
    path = _save_tiny_sidecar(tmp_path / "sidecar.npz")

    lookup = EncoderSidecarLookup.load(path)
    identities = [(1, "TOP", "a"), (2, "JUNGLE", "b"), (99, "TOP", "missing")]
    blocks, support = lookup.lookup_blocks(identities)

    assert lookup.dims.as_dict() == {
        "static": 2,
        "full_game": 3,
        "temporal": 4,
        "total": 9,
    }
    assert blocks["static"].shape == (3, 2)
    assert blocks["full_game"].shape == (3, 3)
    assert blocks["temporal"].shape == (3, 4)
    assert support.tolist() == [50.0, 0.0, 0.0]
    assert np.allclose(blocks["static"][2], 0.0)
    assert "feature_hashes" in lookup.metadata


def test_sidecar_artifact_missing_latents_fails_with_required_keys(tmp_path) -> None:
    path = tmp_path / "draft_identity_sidecar.npz"
    np.savez_compressed(
        path,
        champion_ids=np.array([1, 2], dtype=np.int32),
        build_ids=np.array([0, 1], dtype=np.int32),
        build_labels=np.array(["a", "b"]),
        split=np.array(["train", "train"]),
    )

    with pytest.raises(ValueError, match="missing required arrays: .*static_latents"):
        EncoderSidecarLookup.load(path)


def test_sidecar_game_lookup_requires_ten_identities(tmp_path) -> None:
    lookup = EncoderSidecarLookup.load(_save_tiny_sidecar(tmp_path / "sidecar.npz"))

    with pytest.raises(ValueError, match="expected 10 identities"):
        lookup.lookup_game_blocks([(1, "TOP", "a")])


def test_static_and_split_metadata_reject_leaky_inputs() -> None:
    with pytest.raises(ValueError, match="static encoder metadata"):
        validate_static_metadata({"source": "synergy_2vx_dict"})
    with pytest.raises(ValueError, match="train-split aggregates"):
        validate_train_only_metadata({"split_metadata": {"fit_split": "test"}})


def test_hgnn_sidecar_inputs_swap_and_preserve_structural_antisymmetry() -> None:
    rng = np.random.default_rng(9)
    champion_id = np.tile(np.arange(10), (2, 1))
    build_id = np.tile(np.arange(10) % 3, (2, 1))
    win_rate = rng.uniform(0.35, 0.65, size=(2, 10)).astype(np.float32)
    p1_cnt = rng.uniform(0.0, 100.0, size=(2, 10)).astype(np.float32)
    static = rng.normal(size=(2, 10, 2)).astype(np.float32)
    full_game = rng.normal(size=(2, 10, 3)).astype(np.float32)
    temporal = rng.normal(size=(2, 10, 4)).astype(np.float32)
    support = rng.uniform(0.0, 80.0, size=(2, 10)).astype(np.float32)
    inputs = build_hgnn_inputs(
        champion_id=champion_id,
        build_id=build_id,
        win_rate=win_rate,
        p1_cnt=p1_cnt,
        strength=30.0,
        identity_static_sidecar=static,
        identity_full_game_sidecar=full_game,
        identity_temporal_sidecar=temporal,
        identity_encoder_support=support,
    )

    swapped = swap_hgnn_inputs(inputs)
    assert torch.equal(swapped["identity_static_sidecar"][:, :5], inputs["identity_static_sidecar"][:, 5:])
    assert torch.equal(swapped["identity_encoder_support"][:, 5:], inputs["identity_encoder_support"][:, :5])

    model = HGNNWinModel(
        HGNNConfig(
            n_champions=20,
            n_builds=3,
            dropout=0.0,
            identity_static_sidecar_dim=2,
            identity_full_game_sidecar_dim=3,
            identity_temporal_sidecar_dim=4,
            use_identity_static_sidecar=True,
            use_identity_full_game_sidecar=True,
            use_identity_temporal_sidecar=True,
            structural_antisymmetry=True,
        )
    ).eval()
    with torch.no_grad():
        direct = model(**inputs)["final_logit"]
        mirrored = model(**swapped)["final_logit"]

    assert torch.isfinite(direct).all()
    assert torch.allclose(direct, -mirrored, atol=1.0e-5)


def test_hgnn_missing_sidecar_matches_explicit_zero_sidecar() -> None:
    champion_id = np.tile(np.arange(10), (1, 1))
    build_id = np.zeros((1, 10), dtype=np.int64)
    win_rate = np.full((1, 10), 0.5, dtype=np.float32)
    p1_cnt = np.full((1, 10), 10.0, dtype=np.float32)
    base = build_hgnn_inputs(
        champion_id=champion_id,
        build_id=build_id,
        win_rate=win_rate,
        p1_cnt=p1_cnt,
        strength=30.0,
    )
    zeros = build_hgnn_inputs(
        champion_id=champion_id,
        build_id=build_id,
        win_rate=win_rate,
        p1_cnt=p1_cnt,
        strength=30.0,
        identity_static_sidecar=np.zeros((1, 10, 2), dtype=np.float32),
        identity_encoder_support=np.zeros((1, 10), dtype=np.float32),
    )
    model = HGNNWinModel(
        HGNNConfig(
            n_champions=20,
            n_builds=1,
            dropout=0.0,
            identity_static_sidecar_dim=2,
            use_identity_static_sidecar=True,
        )
    ).eval()

    with torch.no_grad():
        assert torch.allclose(model(**base)["final_logit"], model(**zeros)["final_logit"])


def test_hgnn_semantic_context_head_returns_noop_decomposed_logits() -> None:
    inputs = _semantic_hgnn_inputs()
    model = HGNNWinModel(_semantic_context_config()).eval()

    with torch.no_grad():
        outputs = model(**inputs)

    assert set(outputs) == {"base_logit", "context_logit", "final_logit"}
    assert outputs["base_logit"].shape == (2,)
    assert outputs["context_logit"].shape == (2,)
    assert torch.allclose(outputs["context_logit"], torch.zeros_like(outputs["context_logit"]))
    assert torch.allclose(outputs["final_logit"], outputs["base_logit"])


def test_hgnn_semantic_context_head_requires_all_three_sidecars() -> None:
    inputs = _semantic_hgnn_inputs()
    model = HGNNWinModel(_semantic_context_config()).eval()
    missing_temporal = dict(inputs)
    missing_temporal.pop("identity_temporal_sidecar")

    with pytest.raises(ValueError, match="requires all identity sidecar inputs"):
        model(**missing_temporal)
    with pytest.raises(ValueError, match="static, full-game, and temporal sidecar dims"):
        HGNNWinModel(
            HGNNConfig(
                n_champions=20,
                n_builds=3,
                identity_static_sidecar_dim=2,
                identity_full_game_sidecar_dim=3,
                use_identity_semantic_context_head=True,
            )
        )


def test_hgnn_semantic_context_support_gate_can_zero_context_logit() -> None:
    inputs = _semantic_hgnn_inputs()
    inputs["identity_encoder_support"] = torch.zeros_like(inputs["identity_encoder_support"])
    model = HGNNWinModel(_semantic_context_config(semantic_context_hidden=())).eval()
    _make_context_score_nonzero(model)

    with torch.no_grad():
        outputs = model(**inputs)

    assert torch.allclose(outputs["context_logit"], torch.zeros_like(outputs["context_logit"]))
    assert torch.allclose(outputs["final_logit"], outputs["base_logit"])


def test_hgnn_semantic_context_team_swap_flips_context_sign() -> None:
    inputs = _semantic_hgnn_inputs(batch_size=3, seed=12)
    model = HGNNWinModel(_semantic_context_config(semantic_context_hidden=())).eval()
    _make_context_score_nonzero(model)

    with torch.no_grad():
        direct = model(**inputs)
        mirrored = model(**swap_hgnn_inputs(inputs))

    assert torch.isfinite(direct["context_logit"]).all()
    assert torch.allclose(direct["context_logit"], -mirrored["context_logit"], atol=1.0e-5)


def test_context_ablation_registry_exposes_three_encoder_variants() -> None:
    assert set(SIDECAR_VARIANTS) == {
        "static_only",
        "full_game_only",
        "temporal_only",
        "static_full_game",
        "static_temporal",
        "full_game_temporal",
        "all_three",
        "semantic_context_only",
        "all_three_plus_semantic_context",
        "all_three_plus_raw_context",
    }
    all_three = sidecar_variant_overrides("all_three")
    assert all_three["use_identity_static_sidecar"] is True
    assert all_three["use_identity_full_game_sidecar"] is True
    assert all_three["use_identity_temporal_sidecar"] is True
    semantic = sidecar_variant_overrides("all_three_plus_semantic_context")
    assert semantic["use_identity_static_sidecar"] is True
    assert semantic["use_identity_full_game_sidecar"] is True
    assert semantic["use_identity_temporal_sidecar"] is True
    assert semantic["use_identity_semantic_context_head"] is True
    semantic_only = sidecar_variant_overrides("semantic_context_only")
    assert semantic_only["use_identity_static_sidecar"] is False
    assert semantic_only["use_identity_semantic_context_head"] is True
    assert sidecar_variant_overrides("all_three_plus_raw_context")[
        "use_relationship_integrations"
    ] is True
