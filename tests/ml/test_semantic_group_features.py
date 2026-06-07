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
    SEMANTIC_CONTEXT_PRESSURE_CAPS,
    SEMANTIC_CONTEXT_RAW_DIM,
    RANGED_ATTACK_RANGE_THRESHOLD,
    SEMANTIC_GROUP_FEATURE_DIM,
    SEMANTIC_GROUP_FEATURE_INDEX,
    SEMANTIC_GROUP_FEATURE_NAMES,
    audit_axis_is_covered,
    audit_focus_condition_is_covered,
    build_identity_context_raw_from_metrics,
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
        assert audit_focus_condition_is_covered(spec.focus_condition), (
            spec.focus_condition
        )


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
    assert features[0, :, idx["true_damage"]].tolist() == pytest.approx([0.0] * 10)
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
    assert features[0, 2, idx["hard_cc_reliability"]] == pytest.approx(0.5)
    assert features[0, 1, idx["frontline_intensity"]] == pytest.approx(1.0)
    assert features[0, 2, idx["range_pressure"]] > 0.0
    assert features[0, 0, idx["burst_pressure"]] >= 0.5
    assert features[0, :, idx["scaling_pressure"]].tolist() == pytest.approx([0.0] * 10)
    assert features[0, 3, idx["sustain_protection"]] == pytest.approx(1.0)
    assert np.all((features[:, :, idx["mixed_damage"]] >= 0.0) & (features[:, :, idx["mixed_damage"]] <= 1.0))


def test_build_identity_context_raw_from_metrics_matches_promoted_axes() -> None:
    caps = SEMANTIC_CONTEXT_PRESSURE_CAPS
    metrics = {
        "physicaldamagedealttochampions_share": np.array([0.7, 0.2], dtype=np.float32),
        "magicdamagedealttochampions_share": np.array([0.2, 0.6], dtype=np.float32),
        "truedamagedealttochampions_share": np.array([0.1, 0.2], dtype=np.float32),
        "totaldamagedealttochampions": np.array(
            [caps["damage"] * 0.5, caps["damage"] * 2.0],
            dtype=np.float32,
        ),
        "physicaldamagedealttochampions": np.array(
            [0.0, caps["damage"]], dtype=np.float32
        ),
        "magicdamagedealttochampions": np.array(
            [caps["damage"] * 0.25, 0.0], dtype=np.float32
        ),
        "truedamagedealttochampions": np.array(
            [0.0, caps["damage"] * 0.75], dtype=np.float32
        ),
        "totaldamagetaken": np.array([caps["damage_taken"], 0.0], dtype=np.float32),
        "ally_support": np.array(
            [caps["heal_shield"] * 0.5, caps["heal_shield"]], dtype=np.float32
        ),
        "timeccingothers": np.array([0.0, caps["cc"] * 2.0], dtype=np.float32),
        "structure_damage": np.array([caps["siege"] * 0.25, 0.0], dtype=np.float32),
        "goldearned": np.array(
            [caps["scaling"] * 0.75, caps["scaling"]], dtype=np.float32
        ),
    }

    raw = build_identity_context_raw_from_metrics(metrics)

    assert raw.shape == (2, SEMANTIC_CONTEXT_RAW_DIM)
    assert raw[:, CONTEXT_AXIS_INDEX["physical"]].tolist() == pytest.approx([0.7, 0.2])
    assert raw[:, CONTEXT_AXIS_INDEX["magic"]].tolist() == pytest.approx([0.2, 0.6])
    assert raw[:, CONTEXT_AXIS_INDEX["true_damage"]].tolist() == pytest.approx(
        [0.1, 0.2]
    )
    assert raw[:, CONTEXT_AXIS_INDEX["damage"]].tolist() == pytest.approx([0.5, 1.0])
    assert raw[:, CONTEXT_AXIS_INDEX["damage_taken"]].tolist() == pytest.approx(
        [1.0, 0.0]
    )
    assert raw[:, CONTEXT_AXIS_INDEX["heal_shield"]].tolist() == pytest.approx(
        [0.5, 1.0]
    )
    assert raw[:, CONTEXT_AXIS_INDEX["cc"]].tolist() == pytest.approx([0.0, 1.0])
    assert raw[:, CONTEXT_AXIS_INDEX["siege"]].tolist() == pytest.approx([0.25, 0.0])
    assert raw[:, CONTEXT_AXIS_INDEX["scaling"]].tolist() == pytest.approx([0.75, 1.0])
    assert raw[:, 3].tolist() == pytest.approx([0.0, 0.0])
    assert raw[:, 4].tolist() == pytest.approx([0.0, 0.0])


def test_build_identity_context_raw_from_metrics_validates_metric_shapes() -> None:
    metrics = {
        "physicaldamagedealttochampions_share": np.array([0.7], dtype=np.float32),
    }
    with pytest.raises(KeyError, match="magicdamagedealttochampions_share"):
        build_identity_context_raw_from_metrics(metrics)


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


def test_semantic_group_feature_cache_detects_stale_soft_axis_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    build_vocab = ("carry", "ar_tank")
    context = np.zeros((1, 10, max(CONTEXT_AXIS_INDEX.values()) + 1), dtype=np.float32)
    np.save(cache_dir / "identity_context_raw.npy", context)
    np.save(cache_dir / "champion_id.npy", np.ones((1, 10), dtype=np.int16))
    np.save(cache_dir / "build_id.npy", np.zeros((1, 10), dtype=np.int16))

    materialize_semantic_group_feature_cache(
        cache_dir=cache_dir,
        n_games=1,
        build_vocab=build_vocab,
        hp_lookup=np.zeros(2, dtype=np.float32),
        range_lookup=np.zeros(2, dtype=np.float32),
    )
    meta_path = cache_dir / "semantic_group_features_meta.json"
    metadata = json.loads(meta_path.read_text())
    metadata["context_bin_edges"]["physical"][0] = -999.0
    meta_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="context_bin_edges"):
        materialize_semantic_group_feature_cache(
            cache_dir=cache_dir,
            n_games=1,
            build_vocab=build_vocab,
        )
