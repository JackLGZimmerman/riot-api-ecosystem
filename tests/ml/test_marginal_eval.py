from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.ml.build_catalog import build_catalog
from app.ml.config import POSITIONS
from app.ml.marginal_eval import HypothesisTables, score_split_marginal

VOCAB = ("carry", "tank")
N_CHAMPIONS = 20
BLUE = list(range(1, 6))
RED = list(range(6, 11))


class _BuildSensitiveModel:
    """Fake HGNN whose logit depends only on the hypothesised build ids."""

    config = SimpleNamespace(
        use_learned_semantic_moe=False,
        use_semantic_group_features=False,
    )

    def eval(self) -> "_BuildSensitiveModel":
        return self

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        build_id = inputs["build_id"]
        return {"final_logit": build_id.float().sum(dim=1) * 0.4 - 2.0}


class _PatchSensitiveModel:
    """Fake HGNN whose logit depends on cached game-level patch features."""

    config = SimpleNamespace(
        use_learned_semantic_moe=False,
        use_semantic_group_features=False,
    )

    def __init__(self) -> None:
        self.seen_patch_features: list[torch.Tensor] = []

    def eval(self) -> "_PatchSensitiveModel":
        return self

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_features = inputs["patch_features"]
        self.seen_patch_features.append(patch_features.detach().cpu())
        return {"final_logit": patch_features[:, 0]}


class _RequiresPatchModel(_PatchSensitiveModel):
    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if "patch_features" not in inputs:
            raise ValueError("patch_features required")
        return super().__call__(**inputs)


def _p1() -> dict[tuple[int, str, str], tuple[float, int]]:
    p1: dict[tuple[int, str, str], tuple[float, int]] = {}
    for champion, position in zip(BLUE + RED, POSITIONS + POSITIONS):
        p1[(champion, position, "carry")] = (0.5, 80)
        p1[(champion, position, "tank")] = (0.5, 20)
    return p1


def _tables() -> HypothesisTables:
    shape = (N_CHAMPIONS + 1, 5, len(VOCAB))
    return HypothesisTables(
        win_rate=np.full(shape, 0.5, dtype=np.float32),
        p1_cnt=np.zeros(shape, dtype=np.float32),
        context=np.zeros((*shape, 14), dtype=np.float32),
    )


def _expected_marginal(slot_probs: tuple[float, ...]) -> float:
    """Brute-force probability-space marginal over all 2^10 build worlds."""
    numerator = 0.0
    mass = 0.0
    for combo in product(range(len(VOCAB)), repeat=10):
        weight = float(np.prod([slot_probs[i] for i in combo]))
        logit = sum(combo) * 0.4 - 2.0
        numerator += weight / (1.0 + np.exp(-logit))
        mass += weight
    return numerator / mass


def test_score_split_marginal_matches_brute_force_average() -> None:
    catalog = build_catalog(_p1(), VOCAB)
    split = SimpleNamespace(
        champion_id=np.array([BLUE + RED, BLUE + RED], dtype=np.int64),
        blue_win=np.array([1.0, 0.0]),
        patch_features=None,
    )

    scores = score_split_marginal(
        _BuildSensitiveModel(),
        split,
        catalog,
        _tables(),
        strength=20.0,
        device="cpu",
        gatherer=None,
        k_slot=2,
        max_worlds=2048,
        early_stop_mass=2.0,
        batch_rows=7,  # force mid-game flushes
        log_every=0,
    )

    # Every cell is identical, so every slot shares one smoothed prior vector.
    expected = _expected_marginal(catalog.prior_vector(1, "TOP").probabilities)
    assert scores.probabilities == pytest.approx([expected, expected])
    assert scores.retained_mass == pytest.approx([1.0, 1.0])
    assert scores.n_worlds.tolist() == [1024, 1024]
    assert scores.labels.tolist() == [1.0, 0.0]
    assert scores.fallback_counts == {"champion_role": 20}


def test_score_split_marginal_modal_is_top_world_probability() -> None:
    catalog = build_catalog(_p1(), VOCAB)
    split = SimpleNamespace(
        champion_id=np.array([BLUE + RED], dtype=np.int64),
        blue_win=np.array([1.0]),
        patch_features=None,
    )

    scores = score_split_marginal(
        _BuildSensitiveModel(),
        split,
        catalog,
        _tables(),
        strength=20.0,
        device="cpu",
        gatherer=None,
        k_slot=1,
        max_worlds=1,
        early_stop_mass=2.0,
        log_every=0,
    )

    # Modal world = all-"carry" (build id 0 everywhere) -> sigmoid(-2.0).
    assert scores.probabilities == pytest.approx([1.0 / (1.0 + np.exp(2.0))])
    assert scores.n_worlds.tolist() == [1]


def test_score_split_marginal_repeats_patch_features_across_worlds() -> None:
    catalog = build_catalog(_p1(), VOCAB)
    split = SimpleNamespace(
        champion_id=np.array([BLUE + RED, BLUE + RED], dtype=np.int64),
        blue_win=np.array([1.0, 0.0]),
        patch_features=np.array([[0.25, 1.0], [-0.5, 1.0]], dtype=np.float32),
    )
    model = _PatchSensitiveModel()

    scores = score_split_marginal(
        model,
        split,
        catalog,
        _tables(),
        strength=20.0,
        device="cpu",
        gatherer=None,
        k_slot=2,
        max_worlds=2,
        early_stop_mass=2.0,
        batch_rows=3,
        log_every=0,
    )

    assert scores.probabilities == pytest.approx(
        [1.0 / (1.0 + np.exp(-0.25)), 1.0 / (1.0 + np.exp(0.5))]
    )
    assert scores.n_worlds.tolist() == [2, 2]
    seen = np.concatenate(model.seen_patch_features, axis=0)
    assert seen.shape == (4, 2)
    assert seen[:, 0].tolist() == pytest.approx([0.25, 0.25, -0.5, -0.5])


def test_score_split_marginal_propagates_missing_patch_feature_failure() -> None:
    catalog = build_catalog(_p1(), VOCAB)
    split = SimpleNamespace(
        champion_id=np.array([BLUE + RED], dtype=np.int64),
        blue_win=np.array([1.0]),
        patch_features=None,
    )

    with pytest.raises(ValueError, match="patch_features required"):
        score_split_marginal(
            _RequiresPatchModel(),
            split,
            catalog,
            _tables(),
            strength=20.0,
            device="cpu",
            gatherer=None,
            k_slot=1,
            max_worlds=1,
            early_stop_mass=2.0,
            log_every=0,
        )
