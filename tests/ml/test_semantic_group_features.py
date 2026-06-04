from __future__ import annotations

import json

import numpy as np
import pytest

from app.ml.context_examples_audit import audit_specs
from app.ml.semantic_group_features import (
    BURST_DAMAGE_THRESHOLD,
    CONTEXT_AXIS_INDEX,
    HARD_CC_THRESHOLD,
    HEAVY_TAKEN_THRESHOLD,
    HIGH_HP_THRESHOLD,
    RANGED_ATTACK_RANGE_THRESHOLD,
    SEMANTIC_GROUP_FEATURE_DIM,
    SEMANTIC_GROUP_FEATURE_INDEX,
    SEMANTIC_GROUP_FEATURE_NAMES,
    audit_axis_is_covered,
    audit_focus_condition_is_covered,
    build_semantic_group_features,
    materialize_semantic_group_feature_cache,
)


def test_semantic_group_schema_covers_context_examples_audit_axes() -> None:
    specs = audit_specs()

    assert SEMANTIC_GROUP_FEATURE_DIM == len(SEMANTIC_GROUP_FEATURE_NAMES)
    for axis in CONTEXT_AXIS_INDEX:
        assert axis in SEMANTIC_GROUP_FEATURE_NAMES
    for spec in specs:
        assert audit_axis_is_covered(spec.axis), spec.axis
        assert audit_focus_condition_is_covered(spec.focus_condition), spec.focus_condition


def test_build_semantic_group_features_matches_promoted_audit_definitions() -> None:
    build_vocab = ("carry", "ar_tank", "utility_enchanter", "utility_protection")
    context = np.zeros((1, 10, max(CONTEXT_AXIS_INDEX.values()) + 1), dtype=np.float32)
    champions = np.array([[1, 2, 887, 37, 43, 5, 117, 26, 99, 24]], dtype=np.int16)
    builds = np.array([[0, 1, 0, 2, 0, 0, 3, 0, 0, 0]], dtype=np.int16)
    hp_lookup = np.zeros(1000, dtype=np.float32)
    range_lookup = np.zeros(1000, dtype=np.float32)

    context[:, :, CONTEXT_AXIS_INDEX["physical"]] = np.arange(10, dtype=np.float32)
    context[0, 0, CONTEXT_AXIS_INDEX["damage"]] = BURST_DAMAGE_THRESHOLD
    context[0, 1, CONTEXT_AXIS_INDEX["damage"]] = BURST_DAMAGE_THRESHOLD
    context[0, 2, CONTEXT_AXIS_INDEX["cc"]] = HARD_CC_THRESHOLD
    context[0, 3, CONTEXT_AXIS_INDEX["damage_taken"]] = HEAVY_TAKEN_THRESHOLD
    hp_lookup[1] = HIGH_HP_THRESHOLD
    hp_lookup[2] = HIGH_HP_THRESHOLD - 1.0
    range_lookup[1] = RANGED_ATTACK_RANGE_THRESHOLD
    range_lookup[5] = RANGED_ATTACK_RANGE_THRESHOLD + 1.0
    range_lookup[887] = RANGED_ATTACK_RANGE_THRESHOLD + 1.0

    features = build_semantic_group_features(
        context_raw=context,
        champion_id=champions,
        build_id=builds,
        build_vocab=build_vocab,
        hp_lookup=hp_lookup,
        range_lookup=range_lookup,
    )

    idx = SEMANTIC_GROUP_FEATURE_INDEX
    assert features.shape == (1, 10, SEMANTIC_GROUP_FEATURE_DIM)
    assert features[0, :, idx["physical"]].tolist() == pytest.approx(list(range(10)))
    assert features[0, 0, idx["burst"]] == 1.0
    assert features[0, 1, idx["burst"]] == 0.0
    assert features[0, 2, idx["hard_cc"]] == 1.0
    assert features[0, 1, idx["frontline"]] == 1.0
    assert features[0, 3, idx["heavy_taken"]] == 1.0
    assert features[0, 0, idx["high_hp"]] == 1.0
    assert features[0, 2, idx["ranged"]] == 1.0
    assert features[0, 2, idx["skirmish"]] == 1.0
    assert features[0, 3, idx["selected_enchanter"]] == 1.0
    assert features[0, 4, idx["selected_enchanter"]] == 0.0
    assert features[0, 0, idx["same_role_range"]] == 1.0
    assert features[0, 5, idx["same_role_range"]] == 0.0


def test_materialize_semantic_group_feature_cache_validates_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    build_vocab = ("carry", "ar_tank")
    context = np.zeros((2, 10, max(CONTEXT_AXIS_INDEX.values()) + 1), dtype=np.float32)
    champions = np.ones((2, 10), dtype=np.int16)
    builds = np.zeros((2, 10), dtype=np.int16)
    np.save(cache_dir / "identity_context_raw.npy", context)
    np.save(cache_dir / "champion_id.npy", champions)
    np.save(cache_dir / "build_id.npy", builds)

    features = materialize_semantic_group_feature_cache(
        cache_dir=cache_dir,
        n_games=2,
        build_vocab=build_vocab,
        chunk_size=1,
        hp_lookup=np.zeros(2, dtype=np.float32),
        range_lookup=np.zeros(2, dtype=np.float32),
    )

    assert features.shape == (2, 10, SEMANTIC_GROUP_FEATURE_DIM)
    meta_path = cache_dir / "semantic_group_features_meta.json"
    metadata = json.loads(meta_path.read_text())
    metadata["schema_version"] = -1
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata is stale or invalid"):
        materialize_semantic_group_feature_cache(
            cache_dir=cache_dir,
            n_games=2,
            build_vocab=build_vocab,
        )
