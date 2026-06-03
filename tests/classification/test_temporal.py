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
    model, history = train_temporal(tensors, epochs=5, batch_size=32, cfg=cfg, seed=0)
    assert all(np.isfinite(h["loss"]) for h in history)
    assert history[-1]["loss"] < history[0]["loss"]  # learns something
    latents = extract_temporal_latents(model, tensors)
    assert latents.shape == (len(tensors.keys), 16)
    assert np.isfinite(latents).all()
