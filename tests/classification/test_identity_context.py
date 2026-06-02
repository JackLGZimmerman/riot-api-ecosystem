from __future__ import annotations

import numpy as np
import pytest

from app.classification.embeddings.config import (
    IDENTITY_CONTEXT_CACHE_PATH,
    IDENTITY_CONTEXT_DIM,
    IDENTITY_CONTEXT_INTERP_DIM,
    IDENTITY_CONTEXT_INTERP_FEATURES,
    IDENTITY_CONTEXT_RAW_DIM,
    IDENTITY_CONTEXT_RAW_FEATURES,
    identity_context_feature_set,
    identity_context_raw_extra_feature_set,
    identity_semantic_feature_set,
)
from app.classification.embeddings.runtime import IdentityContextLookup


def test_context_feature_set_excludes_all_challenge_metrics() -> None:
    semantic_feature_set = set(identity_semantic_feature_set())
    context_feature_set = set(identity_context_feature_set())
    assert semantic_feature_set, "semantic feature set is empty"
    assert context_feature_set, "context feature set is empty"
    raw_extra_feature_set = set(identity_context_raw_extra_feature_set())
    assert raw_extra_feature_set, "raw extra feature set is empty"
    assert not any("challenge" in name for name in semantic_feature_set)
    assert not any("challenge" in name for name in context_feature_set)
    assert not any("challenge" in name for name in raw_extra_feature_set)
    assert not any("challenge" in name for name in IDENTITY_CONTEXT_RAW_FEATURES)
    assert not any("challenge" in name for name in IDENTITY_CONTEXT_INTERP_FEATURES)
    # The wide RAW atlas keeps the interpretable axes as its debuggable core.
    assert IDENTITY_CONTEXT_RAW_FEATURES[:IDENTITY_CONTEXT_INTERP_DIM] == IDENTITY_CONTEXT_INTERP_FEATURES
    assert IDENTITY_CONTEXT_RAW_DIM > IDENTITY_CONTEXT_INTERP_DIM
    # First nine interpretable axes are the matchup-profile axes (generalised).
    assert IDENTITY_CONTEXT_INTERP_FEATURES[:6] == (
        "phys_offense_share",
        "magic_offense_share",
        "true_offense_share",
        "armor_resist_frac",
        "mr_resist_frac",
        "champion_damage_pressure",
    )


def test_context_lookup_loads_vectors_support_and_falls_back(tmp_path) -> None:
    path = tmp_path / "identity_context_embedding.npz"
    np.savez(
        path,
        keys=np.array([(1, "TOP", "tank"), (2, "JUNGLE", "bruiser")], dtype=object),
        key_columns=np.array(("championid", "teamposition", "build"), dtype=object),
        embeddings=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        matchups=np.array([500.0, 0.0], dtype=np.float32),
        dim=np.array(3, dtype=np.int32),
        interpretable_dim=np.array(2, dtype=np.int32),
    )
    lookup = IdentityContextLookup.load(path)

    vecs = lookup.lookup_players([(1, "TOP", "tank"), (9, "TOP", "tank")])
    sup = lookup.lookup_support([(1, "TOP", "tank"), (9, "TOP", "tank")])
    assert vecs.shape == (2, 3)
    assert np.allclose(vecs[0], [1.0, 2.0, 3.0])
    assert np.allclose(vecs[1], [0.0, 0.0, 0.0])  # missing identity -> zero vector
    assert sup[0] == 500.0
    assert sup[1] == 0.0  # missing identity -> zero support (gate closed)


def test_context_lookup_loads_raw_block_and_falls_back(tmp_path) -> None:
    path = tmp_path / "identity_context_embedding.npz"
    np.savez(
        path,
        keys=np.array([(1, "TOP", "tank"), (2, "JUNGLE", "bruiser")], dtype=object),
        key_columns=np.array(("championid", "teamposition", "build"), dtype=object),
        embeddings=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        raw_embeddings=np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32
        ),
        matchups=np.array([500.0, 0.0], dtype=np.float32),
        dim=np.array(3, dtype=np.int32),
        interpretable_dim=np.array(2, dtype=np.int32),
        raw_dim=np.array(4, dtype=np.int32),
    )
    lookup = IdentityContextLookup.load(path)
    assert lookup.dim == 3 and lookup.raw_dim == 4

    raw = lookup.lookup_raw([(1, "TOP", "tank"), (2, "JUNGLE", "bruiser"), (9, "TOP", "tank")])
    assert raw.shape == (3, 4)
    # Correct raw vector per identity; distinct identities are not collapsed.
    assert np.allclose(raw[0], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(raw[1], [0.5, 0.6, 0.7, 0.8])
    assert not np.allclose(raw[0], raw[1])
    # Missing identity -> zero raw vector (the conditioned head's gate closes).
    assert np.allclose(raw[2], [0.0, 0.0, 0.0, 0.0])


def test_legacy_artifact_without_raw_block_loads_empty_raw(tmp_path) -> None:
    path = tmp_path / "identity_context_embedding.npz"
    np.savez(
        path,
        keys=np.array([(1, "TOP", "tank")], dtype=object),
        key_columns=np.array(("championid", "teamposition", "build"), dtype=object),
        embeddings=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        matchups=np.array([500.0], dtype=np.float32),
        dim=np.array(3, dtype=np.int32),
        interpretable_dim=np.array(2, dtype=np.int32),
    )
    lookup = IdentityContextLookup.load(path)
    assert lookup.raw == {}
    # A missing raw block still returns a (zero) vector of the configured width.
    raw = lookup.lookup_raw([(1, "TOP", "tank")])
    assert raw.shape == (1, IDENTITY_CONTEXT_RAW_DIM)
    assert np.allclose(raw, 0.0)


def test_context_lookup_is_identity_keyed_no_match_leakage() -> None:
    # The descriptor is a function of (championid, teamposition, build) only, so
    # the same identity in two different drafts gets the same context. This is the
    # structural guarantee that nothing about the current match leaks in.
    lookup = IdentityContextLookup(
        values={(1, "TOP", "tank"): np.array([0.5, 0.25], dtype=np.float32)},
        support={(1, "TOP", "tank"): 123.0},
        dim=2,
    )
    a = lookup.lookup_players([(1, "TOP", "tank")])
    b = lookup.lookup_players([(1, "TOP", "tank")])
    assert np.array_equal(a, b)


@pytest.mark.skipif(
    not IDENTITY_CONTEXT_CACHE_PATH.exists(),
    reason="identity_context_embedding.npz not built",
)
def test_built_context_artifact_is_well_formed() -> None:
    with np.load(IDENTITY_CONTEXT_CACHE_PATH, allow_pickle=True) as z:
        emb = z["embeddings"]
        names = [str(n) for n in z["feature_names"]]
        interp = int(z["interpretable_dim"])
        dim = int(z["dim"])
        raw = z["raw_embeddings"]
        raw_names = [str(n) for n in z["raw_feature_names"]]
        raw_dim = int(z["raw_dim"])
    assert emb.ndim == 2 and emb.shape[1] == dim == IDENTITY_CONTEXT_DIM
    assert interp == IDENTITY_CONTEXT_INTERP_DIM
    assert len(names) == dim
    assert np.isfinite(emb).all()
    assert not any("challenge" in n for n in names)
    # Interpretable axes are natural-unit fractions/pressures in [0, 1].
    assert emb[:, :interp].min() >= -1e-6 and emb[:, :interp].max() <= 1.0 + 1e-6
    # Wide RAW block: same rows, declared width, no challenge leakage, and its
    # debuggable core is the interpretable axes (natural-unit, in [0, 1]).
    assert raw.ndim == 2 and raw.shape[0] == emb.shape[0]
    assert raw.shape[1] == raw_dim == IDENTITY_CONTEXT_RAW_DIM
    assert len(raw_names) == raw_dim
    assert np.isfinite(raw).all()
    assert not any("challenge" in n for n in raw_names)
    assert raw[:, :interp].min() >= -1e-6 and raw[:, :interp].max() <= 1.0 + 1e-6
