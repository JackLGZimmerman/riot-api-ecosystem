from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from app.classification.static_identity_encoder import (
    StaticIdentityAutoencoder,
    StaticIdentityConfig,
    StaticIdentityDataset,
    _champion_recovery_accuracy,
    evaluate_static_autoencoder,
    extract_static_latents,
    static_identity_frame,
    train_static_autoencoder,
    validate_static_input_columns,
)


STATIC_COLUMNS = ("base_health", "armor_l18", "attack_damage")


def _frame(n: int = 16) -> pd.DataFrame:
    """Champion-level frame: one row per champion, no role/build columns."""
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "champion_id": np.arange(n),
            "base_health": rng.normal(size=n),
            "armor_l18": rng.normal(size=n),
            "attack_damage": rng.normal(size=n),
        }
    )


def _config() -> StaticIdentityConfig:
    return StaticIdentityConfig(
        continuous_dim=len(STATIC_COLUMNS),
        latent_dim=6,
        hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        latent_norm="layer",
    )


def test_static_defaults_match_deterministic_identity_recipe() -> None:
    train_signature = inspect.signature(train_static_autoencoder)
    config = StaticIdentityConfig(continuous_dim=len(STATIC_COLUMNS))

    assert config.latent_dim == 128
    assert config.hidden_dims == (192, 96)
    assert config.decoder_hidden_dims == (96, 192)
    assert config.dropout == 0.0
    assert config.latent_dropout == 0.0
    assert train_signature.parameters["noise_std"].default == 0.0


@pytest.mark.parametrize(
    "column",
    ["win_rate", "matchups", "synergy_2vx", "p1_cnt", "support", "raw_count_value"],
)
def test_static_encoder_rejects_empirical_prior_or_count_columns(column: str) -> None:
    with pytest.raises(ValueError, match="static encoder inputs"):
        validate_static_input_columns((*STATIC_COLUMNS, column))


def test_static_encoder_is_champion_level_with_no_role_or_build_parameters() -> None:
    """The static branch must not parametrise the encoding by role or build."""
    model = StaticIdentityAutoencoder(_config())
    param_names = [name for name, _ in model.named_parameters()]

    assert not any(
        "teamposition" in name or "build" in name for name in param_names
    )
    # Champion is defined by its static vector, not a learned champion embedding.
    assert not any("embedding" in name.lower() for name in param_names)


def test_static_dataset_and_autoencoder_smoke_train_evaluate_extract() -> None:
    dataset = StaticIdentityDataset(_frame(), STATIC_COLUMNS)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = StaticIdentityAutoencoder(_config())

    reconstruction, latent = model(dataset.continuous[:8])
    assert reconstruction.shape == (8, len(STATIC_COLUMNS))
    assert latent.shape == (8, 6)

    history = train_static_autoencoder(
        model,
        loader,
        epochs=1,
        lr=1.0e-2,
        device="cpu",
        noise_std=0.0,
        amp=False,
    )
    metrics = evaluate_static_autoencoder(model, loader, device="cpu")
    latents = extract_static_latents(model, loader, device="cpu")

    assert len(history) == 1
    assert np.isfinite(history[0]["loss"])
    assert np.isfinite(history[0]["continuous_loss"])
    assert history[0]["latent_decorrelation_loss"] >= 0.0
    assert np.isfinite(metrics["mse"])
    assert 0.0 <= metrics["champion_recovery_accuracy"] <= 1.0
    assert metrics["latent_active_dims"] > 0
    assert list(latents.columns[:1]) == ["champion_id"]
    assert latents.shape == (len(dataset), 1 + model.config.latent_dim)


def test_static_training_amp_flag_is_cpu_safe() -> None:
    """amp=True on CPU resolves to no-op autocast/scaler and still converges."""
    dataset = StaticIdentityDataset(_frame(), STATIC_COLUMNS)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = StaticIdentityAutoencoder(_config())

    history = train_static_autoencoder(
        model, loader, epochs=2, lr=1.0e-2, device="cpu", amp=True
    )

    assert len(history) == 2
    assert all(np.isfinite(row["loss"]) for row in history)


def test_static_single_row_training_batch_does_not_crash_batchnorm() -> None:
    model = StaticIdentityAutoencoder(
        StaticIdentityConfig(
            continuous_dim=len(STATIC_COLUMNS),
            latent_dim=6,
            hidden_dims=(8,),
            decoder_hidden_dims=(8,),
            latent_norm="batch",
        )
    )
    model.train()
    reconstruction, latent = model(torch.randn(1, len(STATIC_COLUMNS)))
    (reconstruction.square().mean() + latent.square().mean()).backward()

    assert reconstruction.shape == (1, len(STATIC_COLUMNS))
    assert latent.shape == (1, 6)
    assert torch.isfinite(latent).all()


def test_champion_recovery_accuracy_measures_recoverability() -> None:
    true = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, -5.0]])
    champ = np.array([10, 20, 30])

    # Perfect reconstruction -> every champion is recoverable.
    assert _champion_recovery_accuracy(true.copy(), true, champ) == 1.0

    # Collapsed reconstructions -> most champions are no longer recoverable.
    collapsed = np.zeros_like(true)
    accuracy = _champion_recovery_accuracy(collapsed, true, champ)
    assert 0.0 <= accuracy < 1.0


def test_static_identity_frame_is_champion_level_from_dictionary_stats() -> None:
    frame = static_identity_frame([2, 1, 2, 3])

    assert list(frame.columns[:1]) == ["champion_id"]
    # Unique, sorted champions only; role/build never appear.
    assert frame["champion_id"].tolist() == [1, 2, 3]
    assert "teamposition_id" not in frame.columns
    assert "build_id" not in frame.columns
    assert np.isfinite(frame.drop(columns=["champion_id"]).to_numpy()).all()
