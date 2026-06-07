from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.build_encoder_sidecar import (
    _parse_args,
    align_multiview_latents,
    full_game_pca_latents,
    full_game_sample_weight,
    full_game_semantic_targets,
    full_game_sidecar_config,
    select_full_game_metric_columns,
    semantic_targets_as_latents,
    temporal_sidecar_config,
)


def _identity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "champion_id": [1, 2],
            "teamposition_id": [0, 1],
            "build_id": [0, 1],
            "damage": [0.1, 0.2],
            "durability": [0.3, 0.4],
        }
    )


def test_full_game_sidecar_width_profiles() -> None:
    frame = _identity_frame()
    compact = full_game_sidecar_config(
        frame,
        ("damage", "durability"),
        latent_dim=64,
        width_profile="compact",
    )
    standalone = full_game_sidecar_config(
        frame,
        ("damage", "durability"),
        latent_dim=640,
        width_profile="standalone",
    )

    assert compact.metrics_embedding_dim == 64
    assert compact.metrics_hidden_dims == (192, 96)
    assert compact.decoder_hidden_dims == (128, 96)
    assert compact.latent_dropout == pytest.approx(0.05)
    assert standalone.metrics_embedding_dim == 160
    assert standalone.metrics_hidden_dims == (320, 160)
    assert standalone.decoder_hidden_dims == (512, 384)
    assert standalone.latent_dropout == pytest.approx(0.10)


def test_sidecar_builder_defaults_match_production_epoch_budget(tmp_path) -> None:
    args = _parse_args(["--output", str(tmp_path / "sidecar.npz")])

    assert args.static_epochs == 500
    assert args.full_game_epochs == 200
    assert args.temporal_epochs == 200
    assert args.full_game_width_profile == "compact"
    assert args.temporal_width_profile == "compact"
    assert args.full_game_support_weighting == "none"
    assert args.full_game_semantic_target_mode == "none"
    assert args.full_game_semantic_target_weight == 0.0
    assert args.full_game_latent_export == "autoencoder"
    assert args.full_game_input_surface == "full"
    assert args.full_game_identity_mode == "normal"
    assert args.full_game_allow_outcome_metrics is False
    assert args.multiview_alignment_objective == "none"
    assert args.multiview_alignment_weight == 0.0
    assert args.multiview_alignment_dim == 16
    assert args.temporal_zero_unobserved_input is True


def test_temporal_sidecar_width_profiles() -> None:
    compact = temporal_sidecar_config(
        latent_dim=64,
        mask_as_input=True,
        architecture="flat",
        width_profile="compact",
    )
    standalone = temporal_sidecar_config(
        latent_dim=416,
        mask_as_input=False,
        architecture="flat",
        width_profile="standalone",
    )

    assert compact.metric_embed_dim == 48
    assert compact.hidden == 512
    assert compact.mask_as_input is True
    assert standalone.metric_embed_dim == 96
    assert standalone.hidden == 1536
    assert standalone.latent_dim == 416


def test_full_game_metric_surface_selection_preserves_matrix_order() -> None:
    columns = (
        "win",
        "kills",
        "physicaldamagedealttochampions_share",
        "kills_team_share",
        "gold_vs_role_opponent_diff",
        "deaths",
    )

    assert select_full_game_metric_columns(columns, surface="full") == columns[1:]
    assert select_full_game_metric_columns(columns, surface="raw_only") == (
        "kills",
        "deaths",
    )
    assert select_full_game_metric_columns(columns, surface="derived_only") == (
        "physicaldamagedealttochampions_share",
    )
    assert select_full_game_metric_columns(columns, surface="context_only") == (
        "kills_team_share",
        "gold_vs_role_opponent_diff",
    )
    assert select_full_game_metric_columns(columns, surface="raw_context") == (
        "kills",
        "kills_team_share",
        "gold_vs_role_opponent_diff",
        "deaths",
    )
    assert select_full_game_metric_columns(columns, surface="profile_only") == (
        "kills",
        "physicaldamagedealttochampions_share",
        "deaths",
    )
    assert select_full_game_metric_columns(
        columns,
        surface="raw_only",
        allow_outcome_metrics=True,
    ) == (
        "win",
        "kills",
        "deaths",
    )


def test_sidecar_width_profile_validation() -> None:
    with pytest.raises(ValueError, match="width_profile"):
        temporal_sidecar_config(
            latent_dim=64,
            mask_as_input=False,
            architecture="flat",
            width_profile="tiny",
        )


def test_full_game_sample_weight_log1p_normalizes_positive_support() -> None:
    support = pd.Series([0.0, 4.0, 99.0], dtype="float32").to_numpy()

    weights = full_game_sample_weight(support, mode="log1p")

    assert weights is not None
    assert weights[0] == pytest.approx(0.0)
    assert weights[1:].mean() == pytest.approx(1.0)
    assert weights[2] > weights[1]
    assert full_game_sample_weight(support, mode="none") is None


def test_full_game_semantic_targets_build_soft_v2_axes() -> None:
    frame = pd.DataFrame(
        {
            "champion_id": [43, 67],
            "teamposition_id": [0, 1],
            "build_id": [0, 1],
            "physicaldamagedealttochampions_share": [0.2, 0.7],
            "magicdamagedealttochampions_share": [0.7, 0.2],
            "truedamagedealttochampions_share": [0.1, 0.1],
            "totaldamagedealttochampions": [400.0, 950.0],
            "physicaldamagedealttochampions": [100.0, 700.0],
            "magicdamagedealttochampions": [280.0, 200.0],
            "truedamagedealttochampions": [20.0, 50.0],
            "totaldamagetaken": [600.0, 1500.0],
            "ally_support": [350.0, 20.0],
            "timeccingothers": [2.0, 0.1],
            "structure_damage": [200.0, 600.0],
            "goldearned": [450.0, 520.0],
        }
    )

    targets, names = full_game_semantic_targets(
        frame,
        build_vocab=("utility_enchanter", "ad_off_tank"),
        mode="soft_v2",
    )

    assert names == (
        "true_damage",
        "hard_cc_reliability",
        "frontline_intensity",
        "range_pressure",
        "burst_pressure",
        "scaling_pressure",
        "sustain_protection",
        "mixed_damage",
    )
    assert targets is not None
    assert targets.shape == (2, len(names))
    assert targets.min() >= 0.0
    assert targets.max() <= 1.0
    assert full_game_semantic_targets(
        frame,
        build_vocab=("utility_enchanter", "ad_off_tank"),
        mode="none",
    ) == (None, tuple())


def test_semantic_targets_as_latents_standardizes_and_pads() -> None:
    targets = pd.DataFrame(
        {
            "a": [0.0, 0.5, 1.0],
            "b": [1.0, 1.0, 1.0],
        }
    ).to_numpy(dtype="float32")

    latents = semantic_targets_as_latents(targets, latent_dim=4)

    assert latents.shape == (3, 4)
    assert latents[:, 0].mean() == pytest.approx(0.0, abs=1.0e-6)
    assert latents[:, 0].std() == pytest.approx(1.0)
    assert latents[:, 1].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert (latents[:, 2:] == 0.0).all()


def test_full_game_pca_latents_whitens_and_pads() -> None:
    frame = pd.DataFrame(
        {
            "metric_a": [0.0, 1.0, 2.0, 3.0],
            "metric_b": [1.0, 1.5, 1.0, 1.5],
            "metric_c": [2.0, 2.5, 4.0, 3.0],
        }
    )

    latents, summary = full_game_pca_latents(
        frame,
        ("metric_a", "metric_b", "metric_c"),
        latent_dim=5,
    )

    assert latents.shape == (4, 5)
    assert summary["pca_active_dims"] == pytest.approx(3.0)
    assert 0.0 <= summary["pca_explained_variance_ratio"] <= 1.0
    assert latents[:, :3].mean(axis=0).tolist() == pytest.approx(
        [0.0, 0.0, 0.0],
        abs=1.0e-6,
    )
    assert latents[:, :3].std(axis=0).tolist() == pytest.approx(
        [1.0, 1.0, 1.0],
        abs=1.0e-5,
    )
    assert (latents[:, 3:] == 0.0).all()


def test_align_multiview_latents_preserves_shapes_and_records_summary() -> None:
    static = np.asarray(
        [
            [0.0, 0.2, 0.4],
            [0.1, 0.3, 0.5],
            [0.2, 0.4, 0.6],
            [0.8, 0.4, 0.2],
            [0.9, 0.5, 0.3],
            [1.0, 0.6, 0.4],
        ],
        dtype=np.float32,
    )
    full_game = np.concatenate([static, static[:, :2] * 0.5], axis=1)
    temporal = np.concatenate([static[:, ::-1], static[:, :1] * 0.25], axis=1)
    support = np.asarray([4, 8, 16, 32, 64, 128], dtype=np.float32)

    aligned_static, aligned_full_game, aligned_temporal, summary = align_multiview_latents(
        static_latents=static,
        full_game_latents=full_game,
        temporal_latents=temporal,
        support=support,
        objective="vicreg",
        alignment_weight=0.05,
        alignment_dim=3,
        epochs=2,
        batch_size=3,
        lr=1.0e-2,
        seed=11,
        device="cpu",
    )

    assert aligned_static.shape == static.shape
    assert aligned_full_game.shape == full_game.shape
    assert aligned_temporal.shape == temporal.shape
    assert np.isfinite(aligned_static).all()
    assert np.isfinite(aligned_full_game).all()
    assert np.isfinite(aligned_temporal).all()
    assert summary["objective"] == "vicreg"
    assert summary["common_dim"] == pytest.approx(3.0)
    assert summary["history_last"]["loss"] >= 0.0
