from __future__ import annotations

import numpy as np
import pytest

from app.ml.context_examples_audit import (
    Predicate,
    discover_slice_predicates,
    _effect_summary,
    _resolve_active_splits,
    rank_residual_slices,
)


def test_rank_residual_slices_prioritizes_high_support_and_ignores_low_support() -> None:
    n = 5000
    labels = np.zeros(n, dtype=np.float64)
    model_prob = np.full(n, 0.50, dtype=np.float64)
    high = np.zeros(n, dtype=bool)
    high[:2500] = True
    low = np.zeros(n, dtype=bool)
    low[-50:] = True
    model_prob[:1250] = 0.60
    model_prob[1250:2500] = 0.70
    model_prob[low] = 0.99
    rows = rank_residual_slices(
        [
            Predicate("high_support_bias", high, "context_axis"),
            Predicate("low_support_noise", low, "identity_champion"),
        ],
        model_prob,
        labels,
        min_support=1000,
        shrink_strength=100.0,
        top=10,
    )

    assert [row["label"] for row in rows] == ["high_support_bias"]
    row = rows[0]
    assert row["n"] == 2500
    assert row["ci_half_width"] > 0.0
    assert row["ci_low"] <= row["shrunk_gap"] <= row["ci_high"]
    assert row["score"] > 0.0


def test_sparse_slice_is_shrunk_toward_parent_residual_when_not_ignored() -> None:
    n = 2000
    labels = np.zeros(n, dtype=np.float64)
    model_prob = np.full(n, 0.50, dtype=np.float64)
    parent = np.zeros(n, dtype=bool)
    parent[:1000] = True
    child = np.zeros(n, dtype=bool)
    child[:100] = True
    model_prob[parent] = 0.55
    model_prob[child] = 0.95

    rows = rank_residual_slices(
        [Predicate("child", child, "pair_context_identity", parent)],
        model_prob,
        labels,
        min_support=50,
        shrink_strength=1000.0,
        top=1,
    )

    row = rows[0]
    assert row["raw_gap"] > row["parent_gap"]
    assert row["shrunk_gap"] < row["raw_gap"]
    assert row["shrunk_gap"] > row["parent_gap"]


def test_discovery_uses_generic_identity_labels_not_documented_examples() -> None:
    n = 120
    d = {
        "champ": np.full((n, 5), 54),
        "build": np.zeros((n, 5), dtype=np.int64),
        "enemy_phys": np.linspace(0.0, 1.0, n),
        "enemy_magic": np.linspace(1.0, 0.0, n),
        "enemy_dmg": np.linspace(0.2, 0.8, n),
        "enemy_heal": np.linspace(0.8, 0.2, n),
        "focus_dmg": np.linspace(0.1, 0.9, n),
        "focus_heal": np.linspace(0.9, 0.1, n),
        "ally_skirmish": np.arange(n) % 3,
        "mean_support": np.full(n, 200.0),
        "min_support": np.full(n, 100.0),
        "zero_support_players": np.zeros(n),
    }

    predicates = discover_slice_predicates(d, {"ar_tank": 0}, min_support=20)
    labels = [predicate.label for predicate in predicates]
    assert any("champ:54" in label for label in labels)
    assert not any("Malphite" in label for label in labels)


def test_semantic_effect_summary_preserves_direction_conventions() -> None:
    rows = [
        {"label": "low", "n": 100, "emp": 0.50, "model": 0.52, "base": 0.51},
        {"label": "high", "n": 120, "emp": 0.60, "model": 0.55, "base": 0.52},
    ]

    high_low = _effect_summary(
        label="axis",
        axis="context",
        rows=rows,
        low_label="low",
        high_label="high",
    )
    low_high = _effect_summary(
        label="axis",
        axis="context",
        rows=rows,
        low_label="low",
        high_label="high",
        direction="low-high",
    )

    assert high_low["emp_effect"] == pytest.approx(0.10)
    assert high_low["model_effect"] == pytest.approx(0.03)
    assert high_low["delta_gap"] == pytest.approx(-0.07)
    assert low_high["emp_effect"] == pytest.approx(-0.10)
    assert low_high["model_effect"] == pytest.approx(-0.03)
    assert low_high["delta_gap"] == pytest.approx(0.07)
    assert high_low["max_abs_endpoint_gap"] == pytest.approx(0.05)


def test_audit_split_selection_keeps_discovery_validation_only() -> None:
    assert _resolve_active_splits(
        discover_slices=False,
        split="val",
        splits=None,
    ) == ("train", "val", "test")
    assert _resolve_active_splits(
        discover_slices=False,
        split="val",
        splits=("val",),
    ) == ("val",)

    with pytest.raises(ValueError, match="validation-only"):
        _resolve_active_splits(
            discover_slices=True,
            split="val",
            splits=("test",),
        )
    with pytest.raises(ValueError, match="Duplicate"):
        _resolve_active_splits(
            discover_slices=False,
            split="val",
            splits=("val", "val"),
        )
