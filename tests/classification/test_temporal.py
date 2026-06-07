"""Temporal branch: SQL hygiene, frame-count shrinkage, autoencoder smoke."""

from __future__ import annotations

import numpy as np
import pytest

from app.classification.embeddings import temporal as T


def test_metric_set_excludes_identifiers_and_position() -> None:
    assert len(T.TEMPORAL_METRICS) == 45
    assert len(T.EVENT_METRICS) == 6
    assert T.METRIC_NAMES == (*T.TEMPORAL_METRICS, *T.EVENT_METRICS)
    assert len(T.METRIC_NAMES) == 51
    for banned in ("run_id", "matchid", "frame_timestamp", "participantid",
                   "position_x", "position_y"):
        assert banned not in T.METRIC_NAMES
    assert T.N_BUCKETS == 47


def test_frame_count_shrinkage_toward_parents() -> None:
    # Two identities, same champion-role; one bucket sparse, one unobserved.
    keys = [(1, "TOP", "a"), (1, "TOP", "b")]
    sums = np.zeros((2, 2, 1))
    counts = np.zeros((2, 2))
    sums[0, 0, 0], counts[0, 0] = 1000.0, 1000.0  # id0 b0 mean 1.0 (well sampled)
    sums[1, 0, 0], counts[1, 0] = 3000.0, 1000.0  # id1 b0 mean 3.0
    sums[1, 1, 0], counts[1, 1] = 50.0, 10.0      # id1 b1 mean 5.0 (sparse)
    # id0 b1 unobserved (count 0)

    smoothed = T._shrink(sums, counts, keys)
    assert np.isfinite(smoothed).all()
    # well-sampled cell stays between its own mean (1.0) and the role mean (2.0)
    assert 1.0 < smoothed[0, 0, 0] < 2.0
    # unobserved cell falls back to the parent (role-smoothed) bucket mean (5.0)
    assert smoothed[0, 1, 0] == pytest.approx(5.0, rel=1e-6)


def test_standardise_shape_and_finite() -> None:
    rng = np.random.default_rng(0)
    smoothed = rng.normal(size=(20, T.N_BUCKETS, 4))
    std = T._standardise(smoothed, clip_value=8.0)
    assert std.shape == (20, T.N_BUCKETS, 4)
    assert np.isfinite(std).all()
    assert std.max() <= 8.0 and std.min() >= -8.0


# --- autoencoder ---


def _synthetic_tensors(n=64, n_metric=6):
    rng = np.random.default_rng(1)
    values = rng.normal(size=(n, T.N_BUCKETS, n_metric)).astype(np.float32)
    mask = rng.random((n, T.N_BUCKETS)) > 0.3
    keys = [(100 + (i % 5), ["TOP", "MIDDLE", "BOTTOM"][i % 3], ["a", "b"][i % 2])
            for i in range(n)]
    return T.TemporalTensors(keys, values, mask.astype(bool), tuple(f"m{i}" for i in range(n_metric)))


def test_temporal_defaults_match_optimized_recipe() -> None:
    from app.classification.temporal_autoencoder import TemporalAEConfig

    cfg = TemporalAEConfig()

    assert cfg.metric_embed_dim == 96
    assert cfg.latent_dim == 416
    assert cfg.hidden == 1536
    assert cfg.dropout == 0.02
    assert cfg.latent_dropout == 0.0
    assert cfg.zero_unobserved_input
    assert not cfg.mask_as_input
    assert cfg.architecture == "flat"


def test_masked_mse_ignores_unobserved_buckets() -> None:
    import torch

    from app.classification.temporal_autoencoder import masked_mse

    target = torch.zeros(2, T.N_BUCKETS, 3)
    recon = torch.ones(2, T.N_BUCKETS, 3)
    mask = torch.zeros(2, T.N_BUCKETS)
    mask[:, 0] = 1.0  # only bucket 0 observed
    loss = masked_mse(recon, target, mask)
    assert loss == pytest.approx(1.0)  # (1-0)^2 over observed only
    assert masked_mse(recon, target, torch.zeros(2, T.N_BUCKETS)) == pytest.approx(0.0)


def test_autoencoder_smoke_trains_and_extracts() -> None:
    from app.classification.temporal_autoencoder import (
        TemporalAEConfig,
        extract_temporal_latents,
        train_temporal,
    )

    tensors = _synthetic_tensors()
    cfg = TemporalAEConfig(latent_dim=16, hidden=32, metric_embed_dim=8)
    assert cfg.latent_dropout == 0.0
    assert cfg.zero_unobserved_input
    model, history = train_temporal(tensors, epochs=5, batch_size=32, cfg=cfg, seed=0)
    assert all(np.isfinite(h["loss"]) for h in history)
    assert all(np.isfinite(h["masked_mse"]) for h in history)
    assert all(h["latent_decorrelation_loss"] >= 0.0 for h in history)
    assert history[-1]["loss"] < history[0]["loss"]  # learns something
    latents = extract_temporal_latents(model, tensors)
    assert latents.shape == (len(tensors.keys), 16)
    assert np.isfinite(latents).all()


def test_extract_temporal_latents_restores_training_mode() -> None:
    from app.classification.temporal_autoencoder import (
        TemporalAEConfig,
        TemporalAutoencoder,
        extract_temporal_latents,
    )

    tensors = _synthetic_tensors(n=16)
    cfg = TemporalAEConfig(latent_dim=8, hidden=16, metric_embed_dim=4)
    model = TemporalAutoencoder(T.N_BUCKETS, 6, 105, 3, 2, cfg)
    model.train()

    extract_temporal_latents(model, tensors)

    assert model.training


def test_temporal_encoder_always_includes_champion_role_build_identity() -> None:
    import torch

    from app.classification.temporal_autoencoder import (
        TemporalAEConfig,
        TemporalAutoencoder,
    )

    cfg = TemporalAEConfig(latent_dim=8, hidden=16, metric_embed_dim=4, dropout=0.0)
    model = TemporalAutoencoder(T.N_BUCKETS, 4, 6, 3, 2, cfg).eval()
    x = torch.randn(2, T.N_BUCKETS, 4)

    with torch.no_grad():
        latent_a = model.encode(
            x, torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor([0, 1])
        )
        latent_b = model.encode(
            x, torch.tensor([4, 5]), torch.tensor([2, 2]), torch.tensor([1, 0])
        )

    # Same trajectory under different champion/role/build must differ by default.
    assert not torch.allclose(latent_a, latent_b)


def test_temporal_encoder_can_zero_unobserved_input_buckets() -> None:
    import torch

    from app.classification.temporal_autoencoder import (
        TemporalAEConfig,
        TemporalAutoencoder,
    )

    cfg = TemporalAEConfig(
        latent_dim=8,
        hidden=16,
        metric_embed_dim=4,
        dropout=0.0,
        zero_unobserved_input=True,
    )
    model = TemporalAutoencoder(T.N_BUCKETS, 4, 5, 3, 2, cfg).eval()
    x = torch.randn(6, T.N_BUCKETS, 4)
    mask = torch.ones(6, T.N_BUCKETS)
    mask[:, 3:] = 0.0
    champ = torch.zeros(6, dtype=torch.long)
    pos = torch.zeros(6, dtype=torch.long)
    build = torch.zeros(6, dtype=torch.long)

    with torch.no_grad():
        masked_latent = model.encode(x, champ, pos, build, mask)
        zeroed_latent = model.encode(x * mask.unsqueeze(-1), champ, pos, build)

    assert torch.allclose(masked_latent, zeroed_latent)


def test_temporal_encoder_can_append_mask_as_input_channel() -> None:
    import torch

    from app.classification.temporal_autoencoder import (
        TemporalAEConfig,
        TemporalAutoencoder,
    )

    torch.manual_seed(0)
    cfg = TemporalAEConfig(
        latent_dim=8,
        hidden=16,
        metric_embed_dim=4,
        dropout=0.0,
        zero_unobserved_input=False,
        mask_as_input=True,
    )
    model = TemporalAutoencoder(T.N_BUCKETS, 4, 5, 3, 2, cfg).eval()
    x = torch.randn(6, T.N_BUCKETS, 4)
    observed = torch.ones(6, T.N_BUCKETS)
    sparse = observed.clone()
    sparse[:, 3:] = 0.0
    champ = torch.zeros(6, dtype=torch.long)
    pos = torch.zeros(6, dtype=torch.long)
    build = torch.zeros(6, dtype=torch.long)

    with torch.no_grad():
        observed_latent = model.encode(x, champ, pos, build, observed)
        sparse_latent = model.encode(x, champ, pos, build, sparse)

    assert model.metric_input_dim == 5
    assert not torch.allclose(observed_latent, sparse_latent)


@pytest.mark.parametrize("architecture", ["flat", "tcn", "gru", "transformer"])
def test_temporal_encoder_architecture_variants_emit_same_latent_shape(architecture: str) -> None:
    import torch

    from app.classification.temporal_autoencoder import (
        TemporalAEConfig,
        TemporalAutoencoder,
    )

    cfg = TemporalAEConfig(
        latent_dim=8,
        hidden=16,
        metric_embed_dim=4,
        dropout=0.0,
        architecture=architecture,
    )
    model = TemporalAutoencoder(T.N_BUCKETS, 4, 5, 3, 2, cfg).eval()
    x = torch.randn(6, T.N_BUCKETS, 4)
    mask = torch.ones(6, T.N_BUCKETS)
    mask[:, 8:] = 0.0
    champ = torch.zeros(6, dtype=torch.long)
    pos = torch.zeros(6, dtype=torch.long)
    build = torch.zeros(6, dtype=torch.long)

    with torch.no_grad():
        latent = model.encode(x, champ, pos, build, mask)

    assert latent.shape == (6, 8)
    assert torch.isfinite(latent).all()


def test_temporal_config_rejects_unknown_architecture() -> None:
    from app.classification.temporal_autoencoder import TemporalAEConfig

    with pytest.raises(ValueError, match="architecture"):
        TemporalAEConfig(architecture="cnn")  # type: ignore[arg-type]


def test_single_row_training_batch_does_not_crash_batchnorm() -> None:
    """A trailing 1-row batch must use BatchNorm running stats, not crash."""
    from app.classification.temporal_autoencoder import TemporalAEConfig, train_temporal

    # 33 rows with batch_size 32 yields a trailing batch of exactly one row.
    tensors = _synthetic_tensors(n=33)
    cfg = TemporalAEConfig(latent_dim=8, hidden=16, metric_embed_dim=4)
    model, history = train_temporal(tensors, epochs=1, batch_size=32, cfg=cfg, seed=0)
    assert np.isfinite(history[-1]["loss"])


def test_latent_dropout_only_corrupts_decoder_input() -> None:
    import torch

    from app.classification.temporal_autoencoder import (
        TemporalAEConfig,
        TemporalAutoencoder,
    )

    cfg = TemporalAEConfig(
        latent_dim=8, hidden=16, metric_embed_dim=4, dropout=0.0, latent_dropout=1.0
    )
    model = TemporalAutoencoder(T.N_BUCKETS, 4, 5, 3, 2, cfg)
    x = torch.randn(6, T.N_BUCKETS, 4)
    champ = torch.zeros(6, dtype=torch.long)
    pos = torch.zeros(6, dtype=torch.long)
    build = torch.zeros(6, dtype=torch.long)

    model.train()
    clean = model.encode(x, champ, pos, build)
    recon, latent = model(x, champ, pos, build)

    # Extraction-time latent stays clean; decoder saw an all-zero latent.
    assert torch.allclose(latent, clean)
    zero_recon = model.decoder(torch.zeros_like(latent)).reshape(6, T.N_BUCKETS, 4)
    assert torch.allclose(recon, zero_recon)


def test_train_temporal_rejects_negative_decorrelation_weight() -> None:
    from app.classification.temporal_autoencoder import train_temporal

    with pytest.raises(ValueError, match="latent_decorrelation_weight"):
        train_temporal(
            _synthetic_tensors(n=16),
            epochs=1,
            batch_size=8,
            latent_decorrelation_weight=-1.0e-3,
        )


def test_evaluate_temporal_autoencoder_reports_recon_and_latent_summary() -> None:
    from app.classification.temporal_autoencoder import (
        TemporalAEConfig,
        evaluate_temporal_autoencoder,
        train_temporal,
    )

    tensors = _synthetic_tensors()
    cfg = TemporalAEConfig(latent_dim=16, hidden=32, metric_embed_dim=8)
    model, _ = train_temporal(tensors, epochs=3, batch_size=32, cfg=cfg, seed=0)

    metrics = evaluate_temporal_autoencoder(model, tensors)

    assert metrics["rows"] == float(len(tensors.keys))
    assert metrics["masked_mse"] >= 0.0
    assert metrics["latent_effective_rank"] > 0.0
    assert metrics["latent_active_dims"] > 0.0
    assert all(np.isfinite(v) for v in metrics.values())
