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

SEMANTIC_GROUP_FEATURE_SCHEMA_VERSION = 2

# `identity_context_raw.npy` axis indices from the audit cache. These indices are
# fixed by the side-row context cache that produced HGNN_CONTEXT_EXAMPLES_AUDIT.md.
CONTEXT_AXIS_INDEX: dict[str, int] = {
    "physical": 0,
    "magic": 1,
    "true_damage": 2,
    "damage": 5,
    "damage_taken": 9,
    "heal_shield": 10,
    "cc": 11,
    "siege": 12,
    "scaling": 13,
}
SEMANTIC_CONTEXT_RAW_DIM = max(CONTEXT_AXIS_INDEX.values()) + 1
IDENTITY_CONTEXT_RAW_FEATURE_NAMES: tuple[str, ...] = (
    "physicaldamagedealttochampions_share",
    "magicdamagedealttochampions_share",
    "truedamagedealttochampions_share",
    "reserved_context_axis_3",
    "reserved_context_axis_4",
    "totaldamagedealttochampions_pressure",
    "physicaldamagedealttochampions_pressure",
    "magicdamagedealttochampions_pressure",
    "truedamagedealttochampions_pressure",
    "totaldamagetaken_pressure",
    "ally_support_pressure",
    "timeccingothers_pressure",
    "structure_damage_pressure",
    "goldearned_pressure",
)

# Promotion-time caps recovered from the semantic audit cache. They are train
# identity distribution caps, not per-game labels, and keep serving aligned with
# the cache-built semantic_group_features.npy tensor.
SEMANTIC_CONTEXT_PRESSURE_CAPS: dict[str, float] = {
    "damage": 1005.0244969622133,
    "damage_taken": 1405.9442238710412,
    "heal_shield": 306.31772381539395,
    "cc": 1.8998332504215742,
    "siege": 779.2227412050119,
    "scaling": 504.62949220513315,
}
if (
    len(IDENTITY_CONTEXT_RAW_FEATURE_NAMES) != SEMANTIC_CONTEXT_RAW_DIM
):  # pragma: no cover
    raise AssertionError("identity context raw feature schema must match raw dim")

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
SKIRMISH_CHAMPIONS = frozenset(
    {887, 24, 39, 114, 77, 5}
)  # Gwen, Jax, Irelia, Fiora, Udyr, XinZhao.
SELECTED_ENCHANTERS = frozenset({37, 43, 117, 26})  # Sona, Karma, Lulu, Zilean.
SELECTED_ENCHANTER_BUILDS = ("utility_enchanter", "utility_protection")

# Thresholds promoted verbatim from the retired context-examples audit. Static stat
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
    "true_damage",
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
    "hard_cc_reliability",
    "frontline_intensity",
    "range_pressure",
    "burst_pressure",
    "scaling_pressure",
    "sustain_protection",
    "mixed_damage",
)
SEMANTIC_GROUP_FEATURE_DIM = len(SEMANTIC_GROUP_FEATURE_NAMES)
SEMANTIC_GROUP_FEATURE_INDEX = {
    name: idx for idx, name in enumerate(SEMANTIC_GROUP_FEATURE_NAMES)
}

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
        "semantic_context_pressure_caps": dict(SEMANTIC_CONTEXT_PRESSURE_CAPS),
        "identity_context_raw_feature_names": list(IDENTITY_CONTEXT_RAW_FEATURE_NAMES),
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
        "soft_axis_definitions": {
            "true_damage": "identity-context true damage share",
            "hard_cc_reliability": "soft ramp around the hard-CC audit threshold",
            "frontline_intensity": "max of tank build, soft heavy-taken, and soft high-HP signals",
            "range_pressure": "static attack range scaled from ranged threshold to 650",
            "burst_pressure": "soft damage-pressure ramp for non-tank identities",
            "scaling_pressure": "identity-context gold scaling pressure",
            "sustain_protection": "max of heal/shield pressure and selected enchanter identity",
            "mixed_damage": "damage-type balance from physical, magic, and true-damage shares",
        },
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
        "context_bin_edges",
        "semantic_context_pressure_caps",
        "identity_context_raw_feature_names",
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
        static_hp_range_lookups()
        if hp_lookup is None or range_lookup is None
        else (hp_lookup, range_lookup)
    )
    slot_hp = _lookup_static(hp, champions)
    slot_range = _lookup_static(attack_range, champions)
    tank_ids = _build_label_ids(build_vocab, TANK_BUILD_LABELS)
    selected_enchanter_build_ids = _build_label_ids(
        build_vocab, SELECTED_ENCHANTER_BUILDS
    )
    non_tank = ~np.isin(builds, tank_ids)

    out = np.empty(
        (context.shape[0], 10, SEMANTIC_GROUP_FEATURE_DIM),
        dtype=np.float32,
    )
    for name, axis_idx in CONTEXT_AXIS_INDEX.items():
        out[:, :, SEMANTIC_GROUP_FEATURE_INDEX[name]] = context[:, :, axis_idx]

    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["burst"]] = (
        context[:, :, CONTEXT_AXIS_INDEX["damage"]] >= BURST_DAMAGE_THRESHOLD
    ) & non_tank
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
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["selected_enchanter"]] = np.isin(
        champions, list(SELECTED_ENCHANTERS)
    ) & np.isin(builds, selected_enchanter_build_ids)
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["same_role_range"]] = (
        np.concatenate([slot_range[:, 5:], slot_range[:, :5]], axis=1)
        > RANGED_ATTACK_RANGE_THRESHOLD
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["hard_cc_reliability"]] = _soft_threshold(
        context[:, :, CONTEXT_AXIS_INDEX["cc"]],
        HARD_CC_THRESHOLD,
        width=0.15,
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["frontline_intensity"]] = np.maximum.reduce(
        (
            out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["frontline"]],
            _soft_threshold(
                context[:, :, CONTEXT_AXIS_INDEX["damage_taken"]],
                HEAVY_TAKEN_THRESHOLD,
                width=0.12,
            ),
            _soft_threshold(slot_hp, HIGH_HP_THRESHOLD, width=225.0),
        )
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["range_pressure"]] = np.clip(
        (slot_range - RANGED_ATTACK_RANGE_THRESHOLD)
        / max(650.0 - RANGED_ATTACK_RANGE_THRESHOLD, 1.0),
        0.0,
        1.0,
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["burst_pressure"]] = (
        _soft_threshold(
            context[:, :, CONTEXT_AXIS_INDEX["damage"]],
            BURST_DAMAGE_THRESHOLD,
            width=0.08,
        )
        * non_tank.astype(np.float32)
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["scaling_pressure"]] = context[
        :,
        :,
        CONTEXT_AXIS_INDEX["scaling"],
    ]
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["sustain_protection"]] = np.maximum(
        context[:, :, CONTEXT_AXIS_INDEX["heal_shield"]],
        out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["selected_enchanter"]],
    )
    out[:, :, SEMANTIC_GROUP_FEATURE_INDEX["mixed_damage"]] = _mixed_damage_balance(
        physical=context[:, :, CONTEXT_AXIS_INDEX["physical"]],
        magic=context[:, :, CONTEXT_AXIS_INDEX["magic"]],
        true_damage=context[:, :, CONTEXT_AXIS_INDEX["true_damage"]],
    )
    return out


def build_identity_context_raw_from_metrics(
    metric_values: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Build the compact identity context surface consumed by semantic groups.

    The inputs are smoothed train identity metrics keyed by metric name. The
    output is a deterministic per-identity analogue of the audit cache's
    ``identity_context_raw.npy`` first 14 axes.
    """

    physical_share = _metric_column(
        metric_values,
        "physicaldamagedealttochampions_share",
    )
    n_rows = physical_share.shape[0]
    out = np.zeros((n_rows, SEMANTIC_CONTEXT_RAW_DIM), dtype=np.float32)
    out[:, CONTEXT_AXIS_INDEX["physical"]] = physical_share
    out[:, CONTEXT_AXIS_INDEX["magic"]] = _metric_column(
        metric_values,
        "magicdamagedealttochampions_share",
        expected=n_rows,
    )
    out[:, 2] = _metric_column(
        metric_values,
        "truedamagedealttochampions_share",
        expected=n_rows,
    )
    out[:, CONTEXT_AXIS_INDEX["damage"]] = _clip_pressure(
        _metric_column(metric_values, "totaldamagedealttochampions", expected=n_rows),
        SEMANTIC_CONTEXT_PRESSURE_CAPS["damage"],
    )
    out[:, 6] = _clip_pressure(
        _metric_column(
            metric_values,
            "physicaldamagedealttochampions",
            expected=n_rows,
        ),
        SEMANTIC_CONTEXT_PRESSURE_CAPS["damage"],
    )
    out[:, 7] = _clip_pressure(
        _metric_column(
            metric_values,
            "magicdamagedealttochampions",
            expected=n_rows,
        ),
        SEMANTIC_CONTEXT_PRESSURE_CAPS["damage"],
    )
    out[:, 8] = _clip_pressure(
        _metric_column(
            metric_values,
            "truedamagedealttochampions",
            expected=n_rows,
        ),
        SEMANTIC_CONTEXT_PRESSURE_CAPS["damage"],
    )
    out[:, CONTEXT_AXIS_INDEX["damage_taken"]] = _clip_pressure(
        _metric_column(metric_values, "totaldamagetaken", expected=n_rows),
        SEMANTIC_CONTEXT_PRESSURE_CAPS["damage_taken"],
    )
    out[:, CONTEXT_AXIS_INDEX["heal_shield"]] = _clip_pressure(
        _metric_column(metric_values, "ally_support", expected=n_rows),
        SEMANTIC_CONTEXT_PRESSURE_CAPS["heal_shield"],
    )
    out[:, CONTEXT_AXIS_INDEX["cc"]] = _clip_pressure(
        _metric_column(metric_values, "timeccingothers", expected=n_rows),
        SEMANTIC_CONTEXT_PRESSURE_CAPS["cc"],
    )
    out[:, CONTEXT_AXIS_INDEX["siege"]] = _clip_pressure(
        _metric_column(metric_values, "structure_damage", expected=n_rows),
        SEMANTIC_CONTEXT_PRESSURE_CAPS["siege"],
    )
    out[:, CONTEXT_AXIS_INDEX["scaling"]] = _clip_pressure(
        _metric_column(metric_values, "goldearned", expected=n_rows),
        SEMANTIC_CONTEXT_PRESSURE_CAPS["scaling"],
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
    missing = [
        path.name
        for path in (context_path, champion_path, build_path)
        if not path.exists()
    ]
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
        raise ValueError(
            "identity_context_raw.npy must have shape [games, 10, context_dim]"
        )
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
        json.dumps(
            semantic_group_feature_metadata(build_vocab), indent=2, sort_keys=True
        ),
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


def _metric_column(
    metric_values: Mapping[str, np.ndarray],
    name: str,
    *,
    expected: int | None = None,
) -> np.ndarray:
    try:
        values = np.asarray(metric_values[name], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"semantic identity context metric {name!r} is missing") from exc
    if values.ndim != 1:
        raise ValueError(
            f"semantic identity context metric {name!r} must be 1-D, got {values.shape}"
        )
    if expected is not None and values.shape[0] != expected:
        raise ValueError(
            f"semantic identity context metric {name!r} has length {values.shape[0]}; "
            f"expected {expected}"
        )
    return values


def _clip_pressure(values: np.ndarray, cap: float) -> np.ndarray:
    return np.clip(values / float(cap), 0.0, 1.0).astype(np.float32)


def _soft_threshold(values: np.ndarray, threshold: float, *, width: float) -> np.ndarray:
    if width <= 0.0:
        raise ValueError("soft threshold width must be positive")
    lower = float(threshold) - float(width)
    upper = float(threshold) + float(width)
    return np.clip((np.asarray(values, dtype=np.float32) - lower) / (upper - lower), 0.0, 1.0)


def _mixed_damage_balance(
    *,
    physical: np.ndarray,
    magic: np.ndarray,
    true_damage: np.ndarray,
) -> np.ndarray:
    shares = np.stack(
        [
            np.asarray(physical, dtype=np.float32),
            np.asarray(magic, dtype=np.float32),
            np.asarray(true_damage, dtype=np.float32),
        ],
        axis=-1,
    )
    total = shares.sum(axis=-1, keepdims=True)
    normalized = np.divide(
        shares,
        np.maximum(total, 1.0e-6),
        out=np.zeros_like(shares),
        where=total > 0.0,
    )
    dominance = normalized.max(axis=-1)
    return np.clip((1.0 - dominance) / (1.0 - (1.0 / 3.0)), 0.0, 1.0)


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
    "BURST_DAMAGE_THRESHOLD",
    "CONTEXT_AXIS_INDEX",
    "CONTEXT_BIN_EDGES",
    "FOCUS_HP_LOW_THRESHOLD",
    "HARD_CC_THRESHOLD",
    "HEAVY_TAKEN_THRESHOLD",
    "HIGH_HP_THRESHOLD",
    "IDENTITY_CONTEXT_RAW_FEATURE_NAMES",
    "LOW_OWN_DAMAGE_THRESHOLD",
    "RANGED_ATTACK_RANGE_THRESHOLD",
    "SELECTED_ENCHANTER_BUILDS",
    "SELECTED_ENCHANTERS",
    "SEMANTIC_CONTEXT_PRESSURE_CAPS",
    "SEMANTIC_CONTEXT_RAW_DIM",
    "SEMANTIC_GROUP_FEATURE_DIM",
    "SEMANTIC_GROUP_FEATURE_INDEX",
    "SEMANTIC_GROUP_FEATURE_NAMES",
    "SEMANTIC_GROUP_FEATURE_SCHEMA_VERSION",
    "SKIRMISH_CHAMPIONS",
    "TANK_BUILD_LABELS",
    "build_identity_context_raw_from_metrics",
    "build_semantic_group_features",
    "materialize_semantic_group_feature_cache",
    "semantic_group_feature_metadata",
    "static_hp_range_lookups",
    "validate_semantic_group_feature_metadata",
]
