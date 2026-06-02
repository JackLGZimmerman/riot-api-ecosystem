from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from app.classification import champion_semantics
from app.classification.champion_semantics import (
    ChampionAutoencoder,
    ChampionProfileDataset,
    ChampionSemanticConfig,
    champion_semantic_metric_columns,
    evaluate_autoencoder,
    extract_champion_latents,
    find_max_train_batch_size,
    train_autoencoder,
    train_from_dataframe_or_csv,
)
from app.classification.embeddings.config import ALL_METRICS, DERIVED_METRIC_FUNCS


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


def _config() -> ChampionSemanticConfig:
    return ChampionSemanticConfig(
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
    assert (
        train_signature.parameters["latent_decorrelation_weight"].default
        == champion_semantics.DEFAULT_LATENT_DECORRELATION_WEIGHT
    )
    assert ChampionSemanticConfig(
        n_champions=4,
        n_teampositions=3,
        n_builds=2,
        metrics_dim=len(METRICS),
    ).latent_dropout == 0.05


def test_dataset_from_dataframe_returns_expected_tensors() -> None:
    dataset = ChampionProfileDataset(_frame(5), METRICS)

    row = dataset[0]

    assert len(dataset) == 5
    assert row["champion_id"].dtype == torch.long
    assert row["teamposition_id"].dtype == torch.long
    assert row["build_id"].dtype == torch.long
    assert row["metrics"].dtype == torch.float32
    assert row["metrics"].shape == (len(METRICS),)


def test_default_metric_columns_use_all_raw_and_derived_metrics() -> None:
    columns = champion_semantic_metric_columns()

    assert set(ALL_METRICS).issubset(columns)
    assert set(DERIVED_METRIC_FUNCS).issubset(columns)
    assert "physicaldamagedealttochampions_share" in columns
    assert "matchups" not in columns


def test_dataset_rejects_matchups_as_profile_metric() -> None:
    frame = _frame(5)
    frame["matchups"] = np.arange(5)

    with pytest.raises(ValueError, match="matchups"):
        ChampionProfileDataset(frame, (*METRICS, "matchups"))


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
    dataset = ChampionProfileDataset(
        frame,
        ("physicaldamagedealttochampions_share",),
    )

    assert torch.allclose(dataset.metrics[:, 0], torch.tensor([0.5, 0.0]))


def test_autoencoder_outputs_reconstruction_and_latent_shapes() -> None:
    model = ChampionAutoencoder(_config())
    batch = next(iter(DataLoader(ChampionProfileDataset(_frame(6), METRICS), batch_size=6)))

    reconstruction, latent = model(
        batch["champion_id"],
        batch["teamposition_id"],
        batch["build_id"],
        batch["metrics"],
    )

    assert reconstruction.shape == (6, len(METRICS))
    assert latent.shape == (6, 5)
    assert torch.allclose(latent.mean(dim=0), torch.zeros(5), atol=1.0e-5)


def test_batch_norm_latent_supports_single_row_training_batch() -> None:
    model = ChampionAutoencoder(_config())
    batch = next(iter(DataLoader(ChampionProfileDataset(_frame(1), METRICS), batch_size=1)))

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
    config = ChampionSemanticConfig(
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
    model = ChampionAutoencoder(config)
    batch = next(iter(DataLoader(ChampionProfileDataset(_frame(6), METRICS), batch_size=6)))

    _, latent = model(
        batch["champion_id"],
        batch["teamposition_id"],
        batch["build_id"],
        batch["metrics"],
    )

    assert torch.allclose(latent.mean(dim=-1), torch.zeros(6), atol=1.0e-5)


def test_latent_dropout_only_affects_training_decoder_input() -> None:
    config = ChampionSemanticConfig(
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
    model = ChampionAutoencoder(config)
    batch = next(iter(DataLoader(ChampionProfileDataset(_frame(6), METRICS), batch_size=6)))

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
    model = ChampionAutoencoder(_config())
    champion_id = torch.tensor([0, 1], dtype=torch.long)
    teamposition_id = torch.tensor([0, 1], dtype=torch.long)
    build_id = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="metrics width"):
        model(champion_id, teamposition_id, build_id, torch.zeros(2, 2))

    with pytest.raises(ValueError, match="batch size"):
        model(champion_id[:1], teamposition_id, build_id, torch.zeros(2, len(METRICS)))


def test_short_training_run_returns_finite_loss_history() -> None:
    model = ChampionAutoencoder(_config())
    dataloader = DataLoader(ChampionProfileDataset(_frame(16), METRICS), batch_size=4)

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
    assert not champion_semantics._resolve_amp(True, torch.device("cpu"))
    assert champion_semantics._resolve_amp(True, torch.device("cuda"))
    assert not champion_semantics._resolve_pin_memory(None, torch.device("cpu"))
    assert champion_semantics._resolve_pin_memory(None, torch.device("cuda"))
    assert champion_semantics._resolve_batch_size_request("auto") == "auto"
    assert champion_semantics._resolve_batch_size_request("32") == 32


def test_evaluate_autoencoder_returns_clean_reconstruction_metrics() -> None:
    model = ChampionAutoencoder(_config())
    dataloader = DataLoader(ChampionProfileDataset(_frame(8), METRICS), batch_size=4)

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
    model = ChampionAutoencoder(_config())
    dataloader = DataLoader(ChampionProfileDataset(_frame(8), METRICS), batch_size=4)

    metrics = evaluate_autoencoder(model, dataloader, "auto", neighbor_k=2)

    assert metrics["latent_metric_neighbor_k"] == 2.0
    assert 0.0 <= metrics["latent_metric_neighbor_recall"] <= 1.0
    assert np.isfinite(metrics["latent_metric_distance_corr"])


def test_evaluate_autoencoder_handles_single_row_latent_summary() -> None:
    model = ChampionAutoencoder(_config())
    dataloader = DataLoader(ChampionProfileDataset(_frame(1), METRICS), batch_size=1)

    metrics = evaluate_autoencoder(model, dataloader, "auto")

    assert metrics["rows"] == 1.0
    assert metrics["latent_effective_rank"] == 0.0
    assert metrics["latent_participation_rank"] == 0.0
    assert metrics["latent_mean_abs_corr"] == 0.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_metric_corruption_keeps_masked_values_zero_after_noise() -> None:
    metrics = torch.ones(4, len(METRICS))

    corrupted = champion_semantics._corrupt_metrics(
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

    assert isinstance(model, ChampionAutoencoder)
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

    assert isinstance(model, ChampionAutoencoder)
    assert history[0]["latent_decorrelation_loss"] >= 0.0
    assert history[0]["loss"] >= history[0]["reconstruction_loss"]


def test_training_helper_rejects_negative_latent_decorrelation_weight() -> None:
    model = ChampionAutoencoder(_config())
    dataloader = DataLoader(ChampionProfileDataset(_frame(8), METRICS), batch_size=4)

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

    assert isinstance(model, ChampionAutoencoder)
    assert history[0]["batch_size"] == 16.0
    assert np.isfinite(history[0]["loss"])


def test_find_max_train_batch_size_respects_cpu_cap() -> None:
    dataset = ChampionProfileDataset(_frame(16), METRICS)
    model = ChampionAutoencoder(_config())

    batch_size = find_max_train_batch_size(
        model,
        dataset,
        "cpu",
        max_batch_size=7,
    )

    assert batch_size == 7


def test_find_max_train_batch_size_rejects_negative_decorrelation_weight() -> None:
    dataset = ChampionProfileDataset(_frame(16), METRICS)
    model = ChampionAutoencoder(_config())

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
    bad_config = ChampionSemanticConfig(
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
        ChampionSemanticConfig(
            n_champions=4,
            n_teampositions=3,
            n_builds=2,
            metrics_dim=len(METRICS),
            latent_norm="whiten",  # type: ignore[arg-type]
        )


def test_config_rejects_invalid_latent_dropout() -> None:
    with pytest.raises(ValueError, match="latent_dropout"):
        ChampionSemanticConfig(
            n_champions=4,
            n_teampositions=3,
            n_builds=2,
            metrics_dim=len(METRICS),
            latent_dropout=1.5,
        )


def test_extract_champion_latents_returns_identity_columns_and_latents() -> None:
    dataset = ChampionProfileDataset(_frame(7), METRICS)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    model = ChampionAutoencoder(_config())

    latents = extract_champion_latents(model, dataloader, "auto")

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
