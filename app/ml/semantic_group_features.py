"""Shared semantic group definitions and compact per-slot feature builder.

The constants in this module are promoted from the context examples audit. They
remain diagnostic semantic groupings, not supervised labels: the model only sees
the deterministic compact features when the explicit feature flag is enabled.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.classification.embeddings.static_champion import load_static_by_id

SEMANTIC_GROUP_FEATURE_SCHEMA_VERSION = 1

# `identity_context_raw.npy` axis indices from the audit cache. These indices are
# fixed by the side-row context cache that produced HGNN_CONTEXT_EXAMPLES_AUDIT.md.
CONTEXT_AXIS_INDEX: dict[str, int] = {
    "physical": 0,
    "magic": 1,
    "damage": 5,
    "damage_taken": 9,
    "heal_shield": 10,
    "cc": 11,
    "siege": 12,
    "scaling": 13,
}

# Continuous audit bin edges are global side-row team-average percentiles.
CONTEXT_BIN_EDGES: dict[str, tuple[float, float, float, float]] = {
    "physical": (0.387, 0.448, 0.508, 0.557),
    "magic": (0.373, 0.423, 0.486, 0.549),
    "damage": (0.739, 0.764, 0.785, 0.813),
    "damage_taken": (0.639, 0.667, 0.692, 0.721),
    "heal_shield": (0.028, 0.077, 0.200, 0.202),
    "cc": (0.374, 0.429, 0.479, 0.539),
    "siege": (0.441, 0.471, 0.499, 0.530),
    "scaling": (0.829, 0.841, 0.852, 0.863),
}

TANK_BUILD_LABELS = frozenset({"ar_tank", "mr_tank", "ad_off_tank", "ap_off_tank"})
SKIRMISH_CHAMPIONS = frozenset({887, 24, 39, 114, 77, 5})  # Gwen, Jax, Irelia, Fiora, Udyr, XinZhao.
SELECTED_ENCHANTERS = frozenset({37, 43, 117, 26})  # Sona, Karma, Lulu, Zilean.
SELECTED_ENCHANTER_BUILDS = ("utility_enchanter", "utility_protection")

# Thresholds promoted verbatim from context_examples_audit.py. Static stat
# source-order provenance: attackRange_flat is index 10; health_flat and
# health_perLevel are indices 24 and 25 in champion_basic_stats_flat.jsonl.
BURST_DAMAGE_THRESHOLD = 0.952
HARD_CC_THRESHOLD = 0.696
HEAVY_TAKEN_THRESHOLD = 0.822
FOCUS_HP_LOW_THRESHOLD = 2309.0
HIGH_HP_THRESHOLD = 2478.5
RANGED_ATTACK_RANGE_THRESHOLD = 250.0
LOW_OWN_DAMAGE_THRESHOLD = CONTEXT_BIN_EDGES["damage"][0]

SEMANTIC_GROUP_FEATURE_NAMES: tuple[str, ...] = (
    "physical",
    "magic",
    "damage",
    "damage_taken",
    "heal_shield",
    "cc",
    "siege",
    "scaling",
    "burst",
    "hard_cc",
    "frontline",
    "heavy_taken",
    "high_hp",
    "ranged",
    "skirmish",
    "selected_enchanter",
    "same_role_range",
)
SEMANTIC_GROUP_FEATURE_DIM = len(SEMANTIC_GROUP_FEATURE_NAMES)
SEMANTIC_GROUP_FEATURE_INDEX = {
    name: idx for idx, name in enumerate(SEMANTIC_GROUP_FEATURE_NAMES)
}

AUDIT_AXIS_FEATURES: dict[str, tuple[str, ...]] = {
    "enemy_burst_count": ("burst",),
    "enemy_hard_cc_count": ("hard_cc",),
    "enemy_frontline_count": ("frontline",),
    "enemy_heavy_taken_count": ("heavy_taken",),
    "enemy_high_hp_count": ("high_hp",),
    "enemy_ranged_count": ("ranged",),
    "same_role_range": ("same_role_range",),
    "ally_skirmish_count": ("skirmish",),
}

AUDIT_FOCUS_FEATURES: dict[str, tuple[str, ...]] = {
    "low_own_damage": ("damage",),
    "focus_hp_low": ("high_hp",),
    "focus_hp_high": ("high_hp",),
    "selected_enchanter": ("selected_enchanter",),
}


def audit_axis_is_covered(axis: str) -> bool:
    """Whether the compact feature schema includes the semantic source of an audit axis."""

    if axis in AUDIT_AXIS_FEATURES:
        return True
    if axis.startswith(("enemy_", "ally_")):
        return axis.split("_", 1)[1] in CONTEXT_AXIS_INDEX
    return axis in CONTEXT_AXIS_INDEX


def audit_focus_condition_is_covered(focus_condition: str | None) -> bool:
    if focus_condition is None:
        return True
    return focus_condition in AUDIT_FOCUS_FEATURES


def semantic_group_feature_metadata(build_vocab: Sequence[str]) -> dict[str, Any]:
    """Metadata that makes compact feature caches schema-validatable."""

    build_vocab_tuple = tuple(str(label) for label in build_vocab)
    return {
        "schema_version": SEMANTIC_GROUP_FEATURE_SCHEMA_VERSION,
        "feature_names": list(SEMANTIC_GROUP_FEATURE_NAMES),
        "context_axis_index": dict(CONTEXT_AXIS_INDEX),
        "context_bin_edges": {
            name: list(edges) for name, edges in CONTEXT_BIN_EDGES.items()
        },
        "thresholds": {
            "burst_damage": BURST_DAMAGE_THRESHOLD,
            "hard_cc": HARD_CC_THRESHOLD,
            "heavy_taken": HEAVY_TAKEN_THRESHOLD,
            "focus_hp_low": FOCUS_HP_LOW_THRESHOLD,
            "high_hp": HIGH_HP_THRESHOLD,
            "ranged_attack_range": RANGED_ATTACK_RANGE_THRESHOLD,
            "low_own_damage": LOW_OWN_DAMAGE_THRESHOLD,
        },
        "tank_build_labels": sorted(TANK_BUILD_LABELS),
        "skirmish_champions": sorted(SKIRMISH_CHAMPIONS),
        "selected_enchanters": sorted(SELECTED_ENCHANTERS),
        "selected_enchanter_builds": list(SELECTED_ENCHANTER_BUILDS),
        "build_vocab": list(build_vocab_tuple),
        "build_vocab_sha256": _hash_strings(build_vocab_tuple),
    }


def validate_semantic_group_feature_metadata(
    metadata: Mapping[str, Any],
    *,
    build_vocab: Sequence[str],
) -> None:
    expected = semantic_group_feature_metadata(build_vocab)
    checks = (
        "schema_version",
        "feature_names",
        "context_axis_index",
        "thresholds",
        "tank_build_labels",
        "skirmish_champions",
        "selected_enchanters",
        "selected_enchanter_builds",
        "build_vocab_sha256",
    )
    mismatched = [key for key in checks if metadata.get(key) != expected[key]]
    if mismatched:
        raise ValueError(
            "semantic group feature cache metadata is stale or invalid: "
            + ", ".join(mismatched)
            + ". Rebuild semantic_group_features.npy."
        )


def build_semantic_group_features(
    *,
    context_raw: np.ndarray,
    champion_id: np.ndarray,
    build_id: np.ndarray,
    build_vocab: Sequence[str],
    hp_lookup: np.ndarray | None = None,
    range_lookup: np.ndarray | None = None,
) -> np.ndarray:
    """Build compact per-slot semantic group features.

    Returns a float32 tensor with shape ``[games, 10, G]``. Audit labels remain
    diagnostics; no outcome labels or supervised targets are used here.
    """

    context = np.asarray(context_raw)
    champions = np.asarray(champion_id)
    builds = np.asarray(build_id)
    if context.ndim != 3 or context.shape[1] != 10:
        raise ValueError("context_raw must have shape [games, 10, context_dim]")
    if champions.shape != context.shape[:2]:
        raise ValueError("champion_id must have shape [games, 10]")
    if builds.shape != context.shape[:2]:
        raise ValueError("build_id must have shape [games, 10]")
    required_context_dim = max(CONTEXT_AXIS_INDEX.values()) + 1
    if context.shape[2] < required_context_dim:
        raise ValueError(
            f"context_raw must have at least {required_context_dim} axes for semantic groups"
        )

    hp, attack_range = (
        static_hp_range_lookups() if hp_lookup is None or range_lookup is None else (hp_lookup, range_lookup)
    )
    slot_hp = _lookup_static(hp, champions)
    slot_range = _lookup_static(attack_range, champions)
    tank_ids = _build_label_ids(build_vocab, TANK_BUILD_LABELS)
    selected_enchanter_build_ids = _build_label_ids(build_vocab, SELECTED_ENCHANTER_BUILDS)
    non_tank = ~np.isin(builds, tank_ids)

    out = np.empty(
        (context.shape[0], 10, SEMANTIC_GROUP_FEATURE_DIM),
        dtype=np.float32,
    )
    for name, axis_idx in CONTEXT_AXIS_INDEX.items():
        out[:, :, SEMANTIC_GROUP_FEATURE_INDEX[name]] = context[:, :, axis_idx]

    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["burst"]] = (
        (context[:, :, CONTEXT_AXIS_INDEX["damage"]] >= BURST_DAMAGE_THRESHOLD) & non_tank
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["hard_cc"]] = (
        context[:, :, CONTEXT_AXIS_INDEX["cc"]] >= HARD_CC_THRESHOLD
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["frontline"]] = np.isin(builds, tank_ids)
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["heavy_taken"]] = (
        context[:, :, CONTEXT_AXIS_INDEX["damage_taken"]] >= HEAVY_TAKEN_THRESHOLD
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["high_hp"]] = slot_hp >= HIGH_HP_THRESHOLD
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["ranged"]] = (
        slot_range > RANGED_ATTACK_RANGE_THRESHOLD
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["skirmish"]] = np.isin(
        champions,
        list(SKIRMISH_CHAMPIONS),
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["selected_enchanter"]] = (
        np.isin(champions, list(SELECTED_ENCHANTERS))
        & np.isin(builds, selected_enchanter_build_ids)
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["same_role_range"]] = (
        np.concatenate([slot_range[:, 5:], slot_range[:, :5]], axis=1)
        > RANGED_ATTACK_RANGE_THRESHOLD
    )
    return out


def materialize_semantic_group_feature_cache(
    *,
    cache_dir: Path,
    n_games: int,
    build_vocab: Sequence[str],
    chunk_size: int = 50_000,
    hp_lookup: np.ndarray | None = None,
    range_lookup: np.ndarray | None = None,
) -> np.ndarray:
    """Create or validate ``semantic_group_features.npy`` in ``cache_dir``."""

    feature_path = cache_dir / "semantic_group_features.npy"
    meta_path = cache_dir / "semantic_group_features_meta.json"
    if feature_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        validate_semantic_group_feature_metadata(metadata, build_vocab=build_vocab)
        features = np.load(feature_path, mmap_mode="r")
        _validate_feature_shape(features, n_games)
        return features
    if feature_path.exists() != meta_path.exists():
        raise ValueError(
            "semantic group feature cache is incomplete; remove both "
            "semantic_group_features.npy and semantic_group_features_meta.json, then rebuild."
        )

    context_path = cache_dir / "identity_context_raw.npy"
    champion_path = cache_dir / "champion_id.npy"
    build_path = cache_dir / "build_id.npy"
    missing = [path.name for path in (context_path, champion_path, build_path) if not path.exists()]
    if missing:
        raise ValueError(
            "semantic group features require cache arrays: "
            + ", ".join(missing)
            + ". Rebuild the cache with identity context side-row arrays or precompute "
            "semantic_group_features.npy."
        )

    context = np.load(context_path, mmap_mode="r")[:n_games]
    champion = np.load(champion_path, mmap_mode="r")[:n_games]
    build = np.load(build_path, mmap_mode="r")[:n_games]
    if context.ndim != 3 or context.shape[1] != 10:
        raise ValueError("identity_context_raw.npy must have shape [games, 10, context_dim]")
    out = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_games, 10, SEMANTIC_GROUP_FEATURE_DIM),
    )
    for start in range(0, n_games, int(chunk_size)):
        stop = min(start + int(chunk_size), n_games)
        out[start:stop] = build_semantic_group_features(
            context_raw=context[start:stop],
            champion_id=champion[start:stop],
            build_id=build[start:stop],
            build_vocab=build_vocab,
            hp_lookup=hp_lookup,
            range_lookup=range_lookup,
        )
    out.flush()
    meta_path.write_text(
        json.dumps(semantic_group_feature_metadata(build_vocab), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    features = np.load(feature_path, mmap_mode="r")
    _validate_feature_shape(features, n_games)
    return features


def static_hp_range_lookups() -> tuple[np.ndarray, np.ndarray]:
    by_id = load_static_by_id()
    max_id = max(by_id) if by_id else 0
    hp = np.zeros(max_id + 1, dtype=np.float32)
    attack_range = np.zeros(max_id + 1, dtype=np.float32)
    for champion_id, values in by_id.items():
        if champion_id >= hp.size:
            continue
        attack_range[champion_id] = float(values[10])
        hp[champion_id] = float(values[24] + 17.0 * values[25])
    return hp, attack_range


def _validate_feature_shape(features: np.ndarray, n_games: int) -> None:
    expected = (int(n_games), 10, SEMANTIC_GROUP_FEATURE_DIM)
    if features.shape != expected:
        raise ValueError(
            f"semantic_group_features.npy has shape {features.shape}; expected {expected}"
        )


def _lookup_static(lookup: np.ndarray, champion_id: np.ndarray) -> np.ndarray:
    champions = np.asarray(champion_id, dtype=np.int64)
    out = np.zeros(champions.shape, dtype=np.float32)
    valid = (champions >= 0) & (champions < lookup.shape[0])
    out[valid] = lookup[champions[valid]]
    return out


def _build_label_ids(
    build_vocab: Sequence[str],
    labels: Sequence[str] | frozenset[str],
) -> np.ndarray:
    build_to_idx = {str(label): idx for idx, label in enumerate(build_vocab)}
    return np.asarray(
        [build_to_idx[label] for label in labels if label in build_to_idx],
        dtype=np.int64,
    )


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "AUDIT_AXIS_FEATURES",
    "AUDIT_FOCUS_FEATURES",
    "BURST_DAMAGE_THRESHOLD",
    "CONTEXT_AXIS_INDEX",
    "CONTEXT_BIN_EDGES",
    "FOCUS_HP_LOW_THRESHOLD",
    "HARD_CC_THRESHOLD",
    "HEAVY_TAKEN_THRESHOLD",
    "HIGH_HP_THRESHOLD",
    "LOW_OWN_DAMAGE_THRESHOLD",
    "RANGED_ATTACK_RANGE_THRESHOLD",
    "SELECTED_ENCHANTER_BUILDS",
    "SELECTED_ENCHANTERS",
    "SEMANTIC_GROUP_FEATURE_DIM",
    "SEMANTIC_GROUP_FEATURE_INDEX",
    "SEMANTIC_GROUP_FEATURE_NAMES",
    "SEMANTIC_GROUP_FEATURE_SCHEMA_VERSION",
    "SKIRMISH_CHAMPIONS",
    "TANK_BUILD_LABELS",
    "audit_axis_is_covered",
    "audit_focus_condition_is_covered",
    "build_semantic_group_features",
    "materialize_semantic_group_feature_cache",
    "semantic_group_feature_metadata",
    "static_hp_range_lookups",
    "validate_semantic_group_feature_metadata",
]
