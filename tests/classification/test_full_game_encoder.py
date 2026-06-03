from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from app.classification import full_game_encoder
from app.classification.full_game_encoder import (
    FullGameAutoencoder,
    FullGameProfileDataset,
    FullGameSemanticConfig,
    full_game_metric_columns,
    evaluate_autoencoder,
    extract_full_game_latents,
    find_max_train_batch_size,
    train_autoencoder,
    train_from_dataframe_or_csv,
)
from app.classification.embeddings.config import ALL_METRICS, DERIVED_METRIC_FUNCS
from app.classification.embeddings.context_features import CONTEXT_FEATURE_NAMES


METRICS = ("damage_per_min", "gold_per_min", "cc_per_min")


def _frame(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "champion_id": np.arange(n) % 4,
            "teamposition_id": np.arange(n) % 3,
            "build_id": np.arange(n) % 2,
            "damage_per_min": rng.normal(0.0, 1.0, size=n),
            "gold_per_min": rng.normal(0.0, 1.0, size=n),
            "cc_per_min": rng.normal(0.0, 1.0, size=n),
        }
    )


def _config() -> FullGameSemanticConfig:
    return FullGameSemanticConfig(
        n_champions=4,
        n_teampositions=3,
        n_builds=2,
        metrics_dim=len(METRICS),
        latent_dim=5,
        champion_embedding_dim=3,
        teamposition_embedding_dim=2,
        build_embedding_dim=2,
        metrics_embedding_dim=4,
        metrics_hidden_dims=(6,),
        fusion_hidden_dims=(7,),
        decoder_hidden_dims=(7,),
    )


def test_training_helper_defaults_match_locked_production_recipe() -> None:
    train_signature = inspect.signature(train_autoencoder)
    frame_signature = inspect.signature(train_from_dataframe_or_csv)

    assert train_signature.parameters["noise_std"].default == 0.003
    assert frame_signature.parameters["noise_std"].default == 0.003
    assert frame_signature.parameters["batch_size"].default == 1024
    assert (
        train_signature.parameters["latent_decorrelation_weight"].default
        == full_game_encoder.DEFAULT_LATENT_DECORRELATION_WEIGHT
    )
    config = FullGameSemanticConfig(
        n_champions=4,
        n_teampositions=3,
        n_builds=2,
        metrics_dim=len(METRICS),
    )
    assert config.latent_dim == 640
    assert config.metrics_embedding_dim == 160
    assert config.metrics_hidden_dims == (320, 160)
    assert config.decoder_hidden_dims == (512, 384)
    assert config.latent_dropout == 0.10


def test_dataset_from_dataframe_returns_expected_tensors() -> None:
    dataset = FullGameProfileDataset(_frame(5), METRICS)

    row = dataset[0]

    assert len(dataset) == 5
    assert row["champion_id"].dtype == torch.long
    assert row["teamposition_id"].dtype == torch.long
    assert row["build_id"].dtype == torch.long
    assert row["metrics"].dtype == torch.float32
    assert row["metrics"].shape == (len(METRICS),)


def test_default_metric_columns_use_all_raw_derived_and_context_metrics() -> None:
    columns = full_game_metric_columns()

    assert set(ALL_METRICS).issubset(columns)
    assert set(DERIVED_METRIC_FUNCS).issubset(columns)
    assert set(CONTEXT_FEATURE_NAMES).issubset(columns)
    assert len(columns) == len(ALL_METRICS) + len(DERIVED_METRIC_FUNCS) + len(CONTEXT_FEATURE_NAMES)
    assert "physicaldamagedealttochampions_share" in columns
    # Added intra-identity ratio + difference families.
    assert "physicaldamagetaken_share" in columns
    assert "net_kills" in columns
    assert "kills_team_share" in columns
    assert "gold_vs_role_opponent_diff" in columns
    assert "matchups" not in columns


def test_profile_only_metric_columns_exclude_context_features() -> None:
    profile_only = full_game_metric_columns(include_context=False)
    with_context = full_game_metric_columns()

    assert len(profile_only) == len(ALL_METRICS) + len(DERIVED_METRIC_FUNCS)
    assert with_context[: len(profile_only)] == profile_only
    assert with_context[len(profile_only):] == CONTEXT_FEATURE_NAMES
    assert not any(name in profile_only for name in CONTEXT_FEATURE_NAMES)


def test_encoder_trains_on_supplied_context_columns() -> None:
    columns = ("kills_team_share", "gold_vs_role_opponent_diff")
    frame = pd.DataFrame(
        {
            "champion_id": [0, 1, 2, 3],
            "teamposition_id": [0, 1, 2, 0],
            "build_id": [0, 1, 0, 1],
            "kills_team_share": [0.20, 0.30, 0.10, 0.25],
            "gold_vs_role_opponent_diff": [100.0, -50.0, 0.0, 20.0],
        }
    )

    model, history = train_from_dataframe_or_csv(
        frame,
        columns,
        batch_size=2,
        epochs=1,
        pin_memory=False,
        amp=False,
    )

    assert model.config.metrics_dim == 2
    assert np.isfinite(history[0]["loss"])


def test_dataset_rejects_matchups_as_profile_metric() -> None:
    frame = _frame(5)
    frame["matchups"] = np.arange(5)

    with pytest.raises(ValueError, match="matchups"):
        FullGameProfileDataset(frame, (*METRICS, "matchups"))


def test_dataset_computes_derived_metrics_from_source_columns() -> None:
    frame = pd.DataFrame(
        {
            "champion_id": [0, 1],
            "teamposition_id": [0, 1],
            "build_id": [0, 1],
            "physicaldamagedealttochampions": [3.0, 0.0],
            "totaldamagedealttochampions": [6.0, 0.0],
        }
    )
    dataset = FullGameProfileDataset(
        frame,
        ("physicaldamagedealttochampions_share",),
    )

    assert torch.allclose(dataset.metrics[:, 0], torch.tensor([0.5, 0.0]))


def test_autoencoder_outputs_reconstruction_and_latent_shapes() -> None:
    model = FullGameAutoencoder(_config())
    batch = next(iter(DataLoader(FullGameProfileDataset(_frame(6), METRICS), batch_size=6)))

    reconstruction, latent = model(
        batch["champion_id"],
        batch["teamposition_id"],
        batch["build_id"],
        batch["metrics"],
    )

    assert reconstruction.shape == (6, len(METRICS))
    assert latent.shape == (6, 5)
    assert torch.allclose(latent.mean(dim=0), torch.zeros(5), atol=1.0e-5)


def test_full_game_encoder_always_includes_champion_role_build_identity() -> None:
    config = FullGameSemanticConfig(
        n_champions=20,
        n_teampositions=5,
        n_builds=4,
        metrics_dim=len(METRICS),
        latent_dim=5,
        metrics_embedding_dim=4,
        metrics_hidden_dims=(6,),
        fusion_hidden_dims=(7,),
        decoder_hidden_dims=(7,),
        latent_norm="none",
    )
    model = FullGameAutoencoder(config).eval()
    metrics = torch.randn(2, len(METRICS))

    with torch.no_grad():
        latent_a = model.encoder(
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            metrics,
        )
        latent_b = model.encoder(
            torch.tensor([9, 8]),
            torch.tensor([4, 3]),
            torch.tensor([3, 2]),
            metrics,
        )

    # The encoder is fixed at the (champion, role, build) grain, so the same
    # metrics under different identities must produce different latents.
    assert not torch.allclose(latent_a, latent_b)


def test_batch_norm_latent_supports_single_row_training_batch() -> None:
    model = FullGameAutoencoder(_config())
    batch = next(iter(DataLoader(FullGameProfileDataset(_frame(1), METRICS), batch_size=1)))

    model.train()
    reconstruction, latent = model(
        batch["champion_id"],
        batch["teamposition_id"],
        batch["build_id"],
        batch["metrics"],
    )
    loss = reconstruction.square().mean() + latent.square().mean()
    loss.backward()

    assert reconstruction.shape == (1, len(METRICS))
    assert latent.shape == (1, 5)
    assert torch.isfinite(reconstruction).all()
    assert torch.isfinite(latent).all()


def test_layer_norm_option_normalizes_each_latent_row() -> None:
    config = FullGameSemanticConfig(
        n_champions=4,
        n_teampositions=3,
        n_builds=2,
        metrics_dim=len(METRICS),
        latent_dim=5,
        metrics_embedding_dim=4,
        metrics_hidden_dims=(6,),
        fusion_hidden_dims=(7,),
        decoder_hidden_dims=(7,),
        latent_norm="layer",
    )
    model = FullGameAutoencoder(config)
    batch = next(iter(DataLoader(FullGameProfileDataset(_frame(6), METRICS), batch_size=6)))

    _, latent = model(
        batch["champion_id"],
        batch["teamposition_id"],
        batch["build_id"],
        batch["metrics"],
    )

    assert torch.allclose(latent.mean(dim=-1), torch.zeros(6), atol=1.0e-5)


def test_latent_dropout_only_affects_training_decoder_input() -> None:
    config = FullGameSemanticConfig(
        n_champions=4,
        n_teampositions=3,
        n_builds=2,
        metrics_dim=len(METRICS),
        latent_dim=5,
        metrics_embedding_dim=4,
        metrics_hidden_dims=(6,),
        fusion_hidden_dims=(7,),
        decoder_hidden_dims=(7,),
        latent_dropout=1.0,
        latent_norm="none",
    )
    model = FullGameAutoencoder(config)
    batch = next(iter(DataLoader(FullGameProfileDataset(_frame(6), METRICS), batch_size=6)))

    model.train()
    expected_latent = model.encoder(
        batch["champion_id"],
        batch["teamposition_id"],
        batch["build_id"],
        batch["metrics"],
    )
    reconstruction, latent = model(
        batch["champion_id"],
        batch["teamposition_id"],
        batch["build_id"],
        batch["metrics"],
    )

    assert torch.allclose(latent, expected_latent)
    assert torch.allclose(reconstruction, model.decoder(torch.zeros_like(latent)))

    model.eval()
    eval_reconstruction, eval_latent = model(
        batch["champion_id"],
        batch["teamposition_id"],
        batch["build_id"],
        batch["metrics"],
    )

    assert torch.allclose(eval_latent, expected_latent)
    assert torch.allclose(eval_reconstruction, model.decoder(expected_latent))


def test_forward_validates_metric_width_and_batch_sizes() -> None:
    model = FullGameAutoencoder(_config())
    champion_id = torch.tensor([0, 1], dtype=torch.long)
    teamposition_id = torch.tensor([0, 1], dtype=torch.long)
    build_id = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="metrics width"):
        model(champion_id, teamposition_id, build_id, torch.zeros(2, 2))

    with pytest.raises(ValueError, match="batch size"):
        model(champion_id[:1], teamposition_id, build_id, torch.zeros(2, len(METRICS)))


def test_short_training_run_returns_finite_loss_history() -> None:
    model = FullGameAutoencoder(_config())
    dataloader = DataLoader(FullGameProfileDataset(_frame(16), METRICS), batch_size=4)

    history = train_autoencoder(
        model,
        dataloader,
        epochs=2,
        lr=1.0e-2,
        device="auto",
        amp=True,
    )

    assert len(history) == 2
    assert all(np.isfinite(row["loss"]) for row in history)


def test_gpu_option_helpers_are_cpu_safe() -> None:
    assert not full_game_encoder._resolve_amp(True, torch.device("cpu"))
    assert full_game_encoder._resolve_amp(True, torch.device("cuda"))
    assert not full_game_encoder._resolve_pin_memory(None, torch.device("cpu"))
    assert full_game_encoder._resolve_pin_memory(None, torch.device("cuda"))
    assert full_game_encoder._resolve_batch_size_request("auto") == "auto"
    assert full_game_encoder._resolve_batch_size_request("32") == 32


def test_evaluate_autoencoder_returns_clean_reconstruction_metrics() -> None:
    model = FullGameAutoencoder(_config())
    dataloader = DataLoader(FullGameProfileDataset(_frame(8), METRICS), batch_size=4)

    metrics = evaluate_autoencoder(model, dataloader, "auto")

    assert set(metrics) == {
        "mse",
        "mae",
        "rows",
        "latent_active_dims",
        "latent_mean_std",
        "latent_max_std",
        "latent_effective_rank",
        "latent_participation_rank",
        "latent_mean_abs_corr",
    }
    assert metrics["rows"] == 8.0
    assert metrics["latent_active_dims"] > 0.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_evaluate_autoencoder_can_score_metric_neighbor_preservation() -> None:
    model = FullGameAutoencoder(_config())
    dataloader = DataLoader(FullGameProfileDataset(_frame(8), METRICS), batch_size=4)

    metrics = evaluate_autoencoder(model, dataloader, "auto", neighbor_k=2)

    assert metrics["latent_metric_neighbor_k"] == 2.0
    assert 0.0 <= metrics["latent_metric_neighbor_recall"] <= 1.0
    assert np.isfinite(metrics["latent_metric_distance_corr"])


def test_evaluate_autoencoder_handles_single_row_latent_summary() -> None:
    model = FullGameAutoencoder(_config())
    dataloader = DataLoader(FullGameProfileDataset(_frame(1), METRICS), batch_size=1)

    metrics = evaluate_autoencoder(model, dataloader, "auto")

    assert metrics["rows"] == 1.0
    assert metrics["latent_effective_rank"] == 0.0
    assert metrics["latent_participation_rank"] == 0.0
    assert metrics["latent_mean_abs_corr"] == 0.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_metric_corruption_keeps_masked_values_zero_after_noise() -> None:
    metrics = torch.ones(4, len(METRICS))

    corrupted = full_game_encoder._corrupt_metrics(
        metrics,
        noise_std=1.0,
        mask_prob=1.0,
    )

    assert torch.equal(corrupted, torch.zeros_like(metrics))


def test_training_helper_supports_optional_denoising() -> None:
    model, history = train_from_dataframe_or_csv(
        _frame(16),
        METRICS,
        config=_config(),
        batch_size=4,
        epochs=1,
        noise_std=0.01,
        mask_prob=0.2,
        pin_memory=False,
        amp=False,
    )

    assert isinstance(model, FullGameAutoencoder)
    assert len(history) == 1
    assert np.isfinite(history[0]["loss"])
    assert np.isfinite(history[0]["reconstruction_loss"])
    assert np.isfinite(history[0]["latent_decorrelation_loss"])


def test_training_helper_supports_latent_decorrelation_regularizer() -> None:
    model, history = train_from_dataframe_or_csv(
        _frame(16),
        METRICS,
        config=_config(),
        batch_size=4,
        epochs=1,
        latent_decorrelation_weight=1.0e-3,
        pin_memory=False,
        amp=False,
    )

    assert isinstance(model, FullGameAutoencoder)
    assert history[0]["latent_decorrelation_loss"] >= 0.0
    assert history[0]["loss"] >= history[0]["reconstruction_loss"]


def test_training_helper_rejects_negative_latent_decorrelation_weight() -> None:
    model = FullGameAutoencoder(_config())
    dataloader = DataLoader(FullGameProfileDataset(_frame(8), METRICS), batch_size=4)

    with pytest.raises(ValueError, match="latent_decorrelation_weight"):
        train_autoencoder(
            model,
            dataloader,
            epochs=1,
            latent_decorrelation_weight=-1.0e-3,
        )


def test_training_helper_supports_auto_batch_size() -> None:
    model, history = train_from_dataframe_or_csv(
        _frame(16),
        METRICS,
        config=_config(),
        batch_size="auto",
        epochs=1,
        device="cpu",
        pin_memory=False,
        amp=True,
    )

    assert isinstance(model, FullGameAutoencoder)
    assert history[0]["batch_size"] == 16.0
    assert np.isfinite(history[0]["loss"])


def test_find_max_train_batch_size_respects_cpu_cap() -> None:
    dataset = FullGameProfileDataset(_frame(16), METRICS)
    model = FullGameAutoencoder(_config())

    batch_size = find_max_train_batch_size(
        model,
        dataset,
        "cpu",
        max_batch_size=7,
    )

    assert batch_size == 7


def test_find_max_train_batch_size_rejects_negative_decorrelation_weight() -> None:
    dataset = FullGameProfileDataset(_frame(16), METRICS)
    model = FullGameAutoencoder(_config())

    with pytest.raises(ValueError, match="latent_decorrelation_weight"):
        find_max_train_batch_size(
            model,
            dataset,
            "cpu",
            latent_decorrelation_weight=-1.0e-3,
        )


def test_training_helper_infers_vocab_sizes_when_config_omitted() -> None:
    model, _ = train_from_dataframe_or_csv(
        _frame(8),
        METRICS,
        batch_size=4,
        epochs=1,
    )

    assert model.config.n_champions == 4
    assert model.config.n_teampositions == 3
    assert model.config.n_builds == 2


def test_training_helper_validates_config_vocab_ranges_before_training() -> None:
    bad_config = FullGameSemanticConfig(
        n_champions=2,
        n_teampositions=3,
        n_builds=2,
        metrics_dim=len(METRICS),
    )

    with pytest.raises(ValueError, match=r"champion_id IDs must be in \[0, 2\)"):
        train_from_dataframe_or_csv(
            _frame(8),
            METRICS,
            config=bad_config,
            batch_size=4,
            epochs=1,
        )


def test_config_rejects_unknown_latent_norm() -> None:
    with pytest.raises(ValueError, match="latent_norm"):
        FullGameSemanticConfig(
            n_champions=4,
            n_teampositions=3,
            n_builds=2,
            metrics_dim=len(METRICS),
            latent_norm="whiten",  # type: ignore[arg-type]
        )


def test_config_rejects_invalid_latent_dropout() -> None:
    with pytest.raises(ValueError, match="latent_dropout"):
        FullGameSemanticConfig(
            n_champions=4,
            n_teampositions=3,
            n_builds=2,
            metrics_dim=len(METRICS),
            latent_dropout=1.5,
        )


def test_extract_full_game_latents_returns_identity_columns_and_latents() -> None:
    dataset = FullGameProfileDataset(_frame(7), METRICS)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    model = FullGameAutoencoder(_config())

    latents = extract_full_game_latents(model, dataloader, "auto")

    assert list(latents.columns) == [
        "champion_id",
        "teamposition_id",
        "build_id",
        "latent_0",
        "latent_1",
        "latent_2",
        "latent_3",
        "latent_4",
    ]
    assert len(latents) == 7
    assert np.isfinite(latents.filter(like="latent_").to_numpy()).all()
