from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from app.ml.build_catalog import (
    BUILD_SOURCE_ORACLE_OBSERVED,
    BUILD_SOURCE_PREGAME_MARGINAL,
    BUILD_SOURCE_RL_CANDIDATE,
    BUILD_SOURCE_TRAIN_OBSERVED,
    CatalogGates,
    build_catalog,
    enumerate_joint_worlds,
    profile_id,
    validate_accepted_build_source,
)

VOCAB = ("a", "b", "c")


def _p1() -> dict[tuple[int, str, str], tuple[float, int]]:
    return {
        (1, "TOP", "a"): (0.52, 80),
        (1, "TOP", "b"): (0.48, 20),
        (1, "TOP", "c"): (0.50, 1),  # below profile_min_count -> pruned
        (2, "TOP", "a"): (0.51, 900),
        (2, "TOP", "b"): (0.49, 100),
    }


def test_validate_accepted_build_source() -> None:
    assert validate_accepted_build_source(BUILD_SOURCE_PREGAME_MARGINAL)
    assert validate_accepted_build_source(BUILD_SOURCE_RL_CANDIDATE)
    for rejected in (BUILD_SOURCE_TRAIN_OBSERVED, BUILD_SOURCE_ORACLE_OBSERVED):
        with pytest.raises(ValueError, match="diagnostics/training-only"):
            validate_accepted_build_source(rejected)
    with pytest.raises(ValueError, match="unknown build source"):
        validate_accepted_build_source("vibes")


def test_profile_id_is_stable() -> None:
    assert profile_id(42, "JUNGLE", "a") == "42:JUNGLE:a"


def test_build_catalog_rejects_label_outside_vocab() -> None:
    with pytest.raises(ValueError, match="not in the model"):
        build_catalog({(1, "TOP", "z"): (0.5, 100)}, VOCAB)


def test_build_catalog_prunes_and_smooths_toward_role_fallback() -> None:
    catalog = build_catalog(_p1(), VOCAB)
    vector = catalog.prior_vector(1, "TOP")

    # "c" pruned by profile_min_count; retained cells keep vocab indices.
    assert vector.fallback_source == "champion_role"
    assert vector.hgnn_build_ids == (0, 1)
    assert vector.support_counts == (80, 20)
    assert vector.retained_mass == pytest.approx(100 / 101)
    assert vector.pruned_mass == pytest.approx(1 / 101)
    assert sum(vector.probabilities) == pytest.approx(1.0)

    # EB smoothing toward the pruned role distribution: p = (n + tau*q) / (N + tau)
    role = catalog.role_fallback["TOP"]
    q = dict(zip(role.labels, role.probabilities))
    tau = catalog.gates.tau
    expected_a = (80 + tau * q["a"]) / (100 + tau)
    expected_b = (20 + tau * q["b"]) / (100 + tau)
    assert vector.probabilities == pytest.approx((expected_a, expected_b))


def test_prior_vector_fallback_chain() -> None:
    catalog = build_catalog(_p1(), VOCAB)

    role_vector = catalog.prior_vector(999, "TOP")
    assert role_vector.fallback_source == "role"
    # Role fallback = pooled TOP counts with min_share pruning ("c" drops).
    assert role_vector.hgnn_build_ids == (0, 1)
    assert role_vector.probabilities == pytest.approx((980 / 1100, 120 / 1100))

    global_vector = catalog.prior_vector(999, "JUNGLE")
    assert global_vector.fallback_source == "global"
    assert sum(global_vector.probabilities) == pytest.approx(1.0)


def test_validate_model_vocab_rejects_mismatch() -> None:
    catalog = build_catalog(_p1(), VOCAB)
    catalog.validate_model_vocab(VOCAB)
    with pytest.raises(ValueError, match="does not match the model checkpoint"):
        catalog.validate_model_vocab(("a", "c", "b"))


def test_catalog_version_tracks_counts() -> None:
    base = build_catalog(_p1(), VOCAB)
    assert build_catalog(_p1(), VOCAB).version == base.version
    bumped = _p1()
    bumped[(1, "TOP", "a")] = (0.52, 81)
    assert build_catalog(bumped, VOCAB).version != base.version


def test_support_tiers_respect_rl_core_gates() -> None:
    catalog = build_catalog(_p1(), VOCAB, CatalogGates())
    tiers = {
        (p.champion_id, p.primary_label): p.support_tier for p in catalog.profiles()
    }
    assert tiers[(1, "a")] == "core"  # 80 >= 50 and share 0.8 >= 0.02
    assert tiers[(1, "b")] == "supported"  # 20 < rl_core_min_count


def _brute_force(slots: list[np.ndarray]) -> list[tuple[float, tuple[int, ...]]]:
    combos = []
    for idx in product(*(range(s.size) for s in slots)):
        w = float(np.prod([s[i] for s, i in zip(slots, idx)]))
        if w > 0.0:
            combos.append((w, idx))
    combos.sort(key=lambda x: -x[0])
    return combos


def test_enumeration_matches_brute_force_product() -> None:
    slots = [
        np.array([0.5, 0.3, 0.2]),
        np.array([0.6, 0.4]),
        np.array([0.7, 0.2, 0.1]),
    ]
    selections, weights, mass = enumerate_joint_worlds(
        slots, k_slot=3, max_worlds=100, early_stop_mass=2.0
    )
    expected = _brute_force(slots)
    assert len(weights) == len(expected)
    assert weights == pytest.approx([w for w, _ in expected])
    assert mass == pytest.approx(1.0)
    for sel, weight in zip(selections, weights):
        assert weight == pytest.approx(
            float(np.prod([s[i] for s, i in zip(slots, sel)]))
        )


def test_enumeration_modal_reduction() -> None:
    slots = [np.array([0.2, 0.5, 0.3]), np.array([0.4, 0.6])]
    selections, weights, mass = enumerate_joint_worlds(
        slots, k_slot=1, max_worlds=1, early_stop_mass=2.0
    )
    assert selections.tolist() == [[1, 1]]
    assert weights == pytest.approx([0.3])
    assert mass == pytest.approx(0.3)


def test_enumeration_truncation_keeps_top_worlds() -> None:
    slots = [np.array([0.5, 0.3, 0.2]), np.array([0.6, 0.4])]
    _, weights, mass = enumerate_joint_worlds(
        slots, k_slot=3, max_worlds=3, early_stop_mass=2.0
    )
    expected = _brute_force(slots)[:3]
    assert weights == pytest.approx([w for w, _ in expected])
    assert mass == pytest.approx(sum(w for w, _ in expected))


def test_enumeration_early_stop_mass() -> None:
    slots = [np.array([0.5, 0.3, 0.2]), np.array([0.6, 0.4])]
    _, weights, mass = enumerate_joint_worlds(
        slots, k_slot=3, max_worlds=100, early_stop_mass=0.3
    )
    assert mass >= 0.3
    assert len(weights) < len(_brute_force(slots))


def test_enumeration_filters_zero_mass_candidates() -> None:
    slots = [np.array([0.9, 0.0, 0.1]), np.array([1.0])]
    _, weights, _ = enumerate_joint_worlds(
        slots, k_slot=3, max_worlds=100, early_stop_mass=2.0
    )
    assert len(weights) == 2
    assert all(w > 0.0 for w in weights)
    with pytest.raises(ValueError, match="positive-mass"):
        enumerate_joint_worlds([np.array([0.0, 0.0])], k_slot=2)
