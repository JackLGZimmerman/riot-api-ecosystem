from __future__ import annotations

import numpy as np
import pytest
import torch

from app.core.utils.common import POSITIONS
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
from app.ml.semantic_group_features import SEMANTIC_GROUP_FEATURE_DIM
from app.ml.train import _SidecarGatherer


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


def _semantic_moe_config(**overrides) -> HGNNConfig:
    defaults = {
        "n_champions": 20,
        "n_builds": 3,
        "node_dim": 16,
        "edge_hidden": 8,
        "value_hidden": (),
        "gate_hidden": (),
        "node_init_hidden": (),
        "readout_hidden": (),
        "residual_head_hidden": (),
        "dropout": 0.0,
        "identity_static_sidecar_dim": 2,
        "identity_full_game_sidecar_dim": 3,
        "identity_temporal_sidecar_dim": 4,
        "use_learned_semantic_moe": True,
        "semantic_moe_num_experts": 4,
        "semantic_moe_top_k": 2,
        "semantic_moe_factor_dim": 8,
        "semantic_moe_factor_hidden": (),
        "semantic_moe_router_hidden": (),
        "semantic_moe_expert_hidden": (),
        "semantic_moe_dropout": 0.0,
        "semantic_moe_context_token_dropout": 0.0,
    }
    return HGNNConfig(**{**defaults, **overrides})


def _make_context_score_nonzero(model: HGNNWinModel) -> None:
    head = model.identity_semantic_context
    assert head is not None
    last = head.score[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.weight.fill_(0.03)
        if last.bias is not None:
            last.bias.zero_()


def _make_moe_experts_nonzero(model: HGNNWinModel) -> None:
    head = model.learned_semantic_moe
    assert head is not None
    with torch.no_grad():
        for expert in head.experts:
            last = expert[-1]
            assert isinstance(last, torch.nn.Linear)
            last.weight.fill_(0.04)
            if last.bias is not None:
                last.bias.zero_()


def _make_group_relationship_nonzero(model: HGNNWinModel) -> None:
    head = model.learned_semantic_moe
    assert head is not None
    assert head.group_relationship is not None
    inner = head.group_relationship[1]
    last = inner[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        assert last.bias is not None
        last.bias[:-1].fill_(0.03)
        last.bias[-1].zero_()


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
    group_features = rng.uniform(
        0.0,
        1.0,
        size=(2, 10, SEMANTIC_GROUP_FEATURE_DIM),
    ).astype(np.float32)
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
        semantic_group_features=group_features,
    )

    swapped = swap_hgnn_inputs(inputs)
    assert torch.equal(swapped["identity_static_sidecar"][:, :5], inputs["identity_static_sidecar"][:, 5:])
    assert torch.equal(swapped["identity_encoder_support"][:, 5:], inputs["identity_encoder_support"][:, :5])
    assert torch.equal(swapped["semantic_group_features"][:, :5], inputs["semantic_group_features"][:, 5:])

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


def test_learned_semantic_moe_flag_disabled_preserves_old_outputs() -> None:
    inputs = _semantic_hgnn_inputs(batch_size=2, seed=21)
    base_inputs = {
        key: value
        for key, value in inputs.items()
        if not key.startswith("identity_")
    }
    model = HGNNWinModel(
        _semantic_moe_config(use_learned_semantic_moe=False)
    ).eval()

    with torch.no_grad():
        without_sidecars = model(**base_inputs)
        with_ignored_sidecars = model(**inputs)

    assert set(without_sidecars) == {"base_logit", "context_logit", "final_logit"}
    assert torch.allclose(without_sidecars["context_logit"], torch.zeros(2))
    assert torch.allclose(without_sidecars["base_logit"], with_ignored_sidecars["base_logit"])
    assert torch.allclose(without_sidecars["final_logit"], with_ignored_sidecars["final_logit"])


def test_learned_semantic_moe_forward_shapes_topk_and_usage_stats() -> None:
    inputs = _semantic_hgnn_inputs(batch_size=3, seed=22)
    model = HGNNWinModel(_semantic_moe_config()).eval()

    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs["base_logit"].shape == (3,)
    assert outputs["context_logit"].shape == (3,)
    assert outputs["final_logit"].shape == (3,)
    assert outputs["semantic_moe_logit"].shape == (3,)
    assert outputs["semantic_moe_slot_delta"].shape == (3, 10)
    assert outputs["semantic_moe_router_probs"].shape == (3, 10, 4)
    assert outputs["semantic_moe_topk_indices"].shape == (3, 10, 2)
    assert outputs["semantic_moe_topk_weights"].shape == (3, 10, 2)
    assert outputs["semantic_moe_expert_usage"].shape == (4,)
    assert outputs["semantic_moe_expert_selected_fraction"].shape == (4,)
    assert outputs["semantic_moe_regularization_loss"].shape == ()

    router_probs = outputs["semantic_moe_router_probs"]
    assert torch.allclose(router_probs.sum(dim=-1), torch.ones(3, 10), atol=1.0e-6)
    assert torch.equal((router_probs > 0.0).sum(dim=-1), torch.full((3, 10), 2))
    assert torch.allclose(
        outputs["semantic_moe_expert_usage"].sum(),
        torch.ones(()),
        atol=1.0e-6,
    )
    assert float(outputs["semantic_moe_expert_selected_fraction"].sum()) == pytest.approx(2.0)

    stats = model.semantic_moe_stats(outputs)
    assert "semantic_moe_expert_usage" in stats
    assert "semantic_moe_router_entropy" in stats
    assert "semantic_moe_factor_std_min" in stats


def test_learned_semantic_moe_team_swap_flips_context_sign() -> None:
    inputs = _semantic_hgnn_inputs(batch_size=3, seed=23)
    model = HGNNWinModel(_semantic_moe_config()).eval()
    _make_moe_experts_nonzero(model)

    with torch.no_grad():
        direct = model(**inputs)
        mirrored = model(**swap_hgnn_inputs(inputs))

    assert float(direct["context_logit"].abs().sum()) > 0.0
    assert torch.allclose(direct["context_logit"], -mirrored["context_logit"], atol=1.0e-5)


def test_learned_semantic_moe_group_features_are_flagged_and_swap_antisymmetric() -> None:
    inputs = _semantic_hgnn_inputs(batch_size=3, seed=24)
    inputs["semantic_group_features"] = torch.rand(
        3,
        10,
        SEMANTIC_GROUP_FEATURE_DIM,
    )
    model = HGNNWinModel(
        _semantic_moe_config(use_semantic_group_features=True)
    ).eval()
    _make_moe_experts_nonzero(model)
    assert model.learned_semantic_moe is not None
    assert model.learned_semantic_moe.group_context is not None
    with torch.no_grad():
        inner = model.learned_semantic_moe.group_context[1]
        last = inner[-1]
        assert isinstance(last, torch.nn.Linear)
        last.weight.fill_(0.03)
        direct = model(**inputs)
        mirrored = model(**swap_hgnn_inputs(inputs))

    assert direct["semantic_moe_group_features_enabled"].item() == pytest.approx(1.0)
    assert direct["semantic_moe_group_feature_dim"].item() == pytest.approx(
        SEMANTIC_GROUP_FEATURE_DIM
    )
    assert direct["semantic_moe_group_relationship_enabled"].item() == pytest.approx(1.0)
    assert torch.allclose(direct["context_logit"], -mirrored["context_logit"], atol=1.0e-5)

    missing = dict(inputs)
    missing.pop("semantic_group_features")
    with pytest.raises(ValueError, match="semantic_group_features"):
        model(**missing)


def test_learned_semantic_group_relationship_head_is_noop_then_antisymmetric() -> None:
    inputs = _semantic_hgnn_inputs(batch_size=3, seed=25)
    inputs["semantic_group_features"] = torch.rand(
        3,
        10,
        SEMANTIC_GROUP_FEATURE_DIM,
    )
    model = HGNNWinModel(
        _semantic_moe_config(
            use_semantic_group_features=True,
            semantic_group_relationship_hidden=(),
        )
    ).eval()

    with torch.no_grad():
        noop = model(**inputs)

    assert torch.allclose(
        noop["semantic_moe_group_relationship_logit"],
        torch.zeros(3),
    )
    assert torch.allclose(noop["semantic_moe_logit"], torch.zeros(3))

    _make_group_relationship_nonzero(model)
    with torch.no_grad():
        direct = model(**inputs)
        mirrored = model(**swap_hgnn_inputs(inputs))

    assert float(direct["semantic_moe_group_relationship_logit"].abs().sum()) > 0.0
    assert torch.allclose(
        direct["semantic_moe_group_relationship_logit"],
        direct["semantic_moe_logit"],
        atol=1.0e-6,
    )
    assert torch.allclose(direct["context_logit"], -mirrored["context_logit"], atol=1.0e-5)


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


def test_sidecar_gather_matches_artifact_lookup(tmp_path) -> None:
    """The per-batch gather must reproduce the artifact's per-game lookup."""
    lookup = EncoderSidecarLookup.load(_save_tiny_sidecar(tmp_path / "sidecar.npz"))
    build_vocab = ["a", "b"]
    n_champions, n_builds = 5, 2
    gatherer = _SidecarGatherer(
        lookup.gather_tables(
            build_vocab=build_vocab, n_champions=n_champions, n_builds=n_builds
        ),
        device="cpu",
    )

    # slot 0: (champ 1, TOP, "a") and slot 1: (champ 2, JUNGLE, "b") are present;
    # the rest miss (wrong role/build or champ 0) and must come back zeroed.
    champ = np.zeros((1, 10), dtype=np.int64)
    build = np.zeros((1, 10), dtype=np.int64)
    champ[0, 0], build[0, 0] = 1, 0
    champ[0, 1], build[0, 1] = 2, 1
    champ[0, 2], build[0, 2] = 1, 0  # champ 1 at MIDDLE -> miss
    champ[0, 3], build[0, 3] = 2, 2  # unknown build -> miss

    tuples = [
        (
            int(champ[0, slot]),
            POSITIONS[slot % 5],
            build_vocab[int(build[0, slot])] if int(build[0, slot]) < n_builds else "",
        )
        for slot in range(10)
    ]
    ref_blocks, ref_support = lookup.lookup_game_blocks(tuples)

    gathered = gatherer.gather(torch.as_tensor(champ), torch.as_tensor(build))
    name_map = {
        "identity_static_sidecar": "static",
        "identity_full_game_sidecar": "full_game",
        "identity_temporal_sidecar": "temporal",
    }
    for tensor_name, block_name in name_map.items():
        assert np.allclose(gathered[tensor_name].numpy()[0], ref_blocks[block_name][0])
    assert np.allclose(gathered["identity_encoder_support"].numpy()[0], ref_support[0])
    # Spot-check the present/missing contract explicitly.
    assert np.allclose(gathered["identity_static_sidecar"].numpy()[0, 0], [1.0, 2.0])
    assert gathered["identity_encoder_support"].numpy()[0, 0] == 50.0
    assert np.allclose(gathered["identity_static_sidecar"].numpy()[0, 2], 0.0)
    assert gathered["identity_encoder_support"].numpy()[0, 2] == 0.0


def test_semantic_context_scale_amplifies_context_logit() -> None:
    inputs = _semantic_hgnn_inputs(batch_size=3, seed=5)
    model = HGNNWinModel(_semantic_context_config(semantic_context_hidden=())).eval()
    _make_context_score_nonzero(model)

    with torch.no_grad():
        base = model(**inputs)["context_logit"]
        model.identity_semantic_context.context_scale.mul_(2.0)
        scaled = model(**inputs)["context_logit"]

    assert float(base.abs().sum()) > 0.0
    assert torch.allclose(scaled, 2.0 * base, atol=1.0e-5)


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
        "learned_semantic_moe_only",
        "learned_semantic_moe_group_features_only",
        "all_three_plus_semantic_context",
        "all_three_plus_learned_semantic_moe",
        "all_three_plus_learned_semantic_moe_group_features",
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
    learned_moe = sidecar_variant_overrides("all_three_plus_learned_semantic_moe")
    assert learned_moe["use_identity_static_sidecar"] is True
    assert learned_moe["use_identity_full_game_sidecar"] is True
    assert learned_moe["use_identity_temporal_sidecar"] is True
    assert learned_moe["use_learned_semantic_moe"] is True
    learned_moe_only = sidecar_variant_overrides("learned_semantic_moe_only")
    assert learned_moe_only["use_identity_static_sidecar"] is False
    assert learned_moe_only["use_learned_semantic_moe"] is True
    learned_moe_grouped = sidecar_variant_overrides(
        "learned_semantic_moe_group_features_only"
    )
    assert learned_moe_grouped["use_learned_semantic_moe"] is True
    assert learned_moe_grouped["use_semantic_group_features"] is True
    all_three_grouped = sidecar_variant_overrides(
        "all_three_plus_learned_semantic_moe_group_features"
    )
    assert all_three_grouped["use_identity_static_sidecar"] is True
    assert all_three_grouped["use_semantic_group_features"] is True
    assert sidecar_variant_overrides("all_three_plus_raw_context")[
        "use_relationship_integrations"
    ] is True
