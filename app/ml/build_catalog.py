"""Train-only build catalog: contracts, smoothed pregame priors, world enumeration.

Implements Phase A of documentation/HGNN_BUILD_INTENT.md. The catalog is the
single source of pregame build distributions: per (champion, teamposition) a
pruned, empirical-Bayes-smoothed probability vector over train-supported
build profiles, plus the lazy best-first enumeration of joint build worlds
used by both the cache-side eval harness and the runtime predictor.

Everything here derives from `synergy_1vx` train rows only (`load_priors`
selects `split = 'train'`); no held-out build label can reach a catalog.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np

# Build-source labels carried by every payload that involves a build
# assignment. Accepted test/serving modes must reject observed held-out
# labels mechanically, not by convention.
BUILD_SOURCE_PREGAME_MARGINAL = "pregame_marginal_build"
BUILD_SOURCE_RL_CANDIDATE = "rl_candidate"
BUILD_SOURCE_TRAIN_OBSERVED = "train_observed_build"
BUILD_SOURCE_ORACLE_OBSERVED = "oracle_observed_build"
BUILD_SOURCES = frozenset(
    {
        BUILD_SOURCE_PREGAME_MARGINAL,
        BUILD_SOURCE_RL_CANDIDATE,
        BUILD_SOURCE_TRAIN_OBSERVED,
        BUILD_SOURCE_ORACLE_OBSERVED,
    }
)
ACCEPTED_BUILD_SOURCES = frozenset(
    {BUILD_SOURCE_PREGAME_MARGINAL, BUILD_SOURCE_RL_CANDIDATE}
)

# Asset-family taxonomy: every ML asset belongs to exactly one family.
ASSET_FAMILY_BUILD_CONDITIONAL = "build_conditional"
ASSET_FAMILY_PREGAME_NATIVE = "pregame_native"

# Key-space descriptors for the two families.
KEY_SPACE_CHAMP_ROLE = ("championid", "teamposition")
KEY_SPACE_CHAMP_ROLE_BUILD = ("championid", "teamposition", "build")


def assert_pregame_native_key_space(keys) -> None:
    """Guard: pregame-native assets may key only on (championid, teamposition).
    Raise ValueError if any key carries a third (build) component or is not a
    (int championid, str teamposition) 2-tuple — i.e. a build-conditional asset
    has leaked into a pregame-native (accepted) surface."""
    for key in keys:
        if len(key) != len(KEY_SPACE_CHAMP_ROLE):
            raise ValueError(
                f"pregame-native key space is {KEY_SPACE_CHAMP_ROLE} but got key "
                f"{key!r} with {len(key)} components; build-conditional state must "
                "not reach a pregame-native/accepted surface"
            )
        champ, role = key
        if not isinstance(champ, (int, np.integer)):
            raise ValueError(
                f"pregame-native key {key!r}: championid must be an int, got "
                f"{type(champ).__name__!r}; build-conditional state must not reach "
                "a pregame-native/accepted surface"
            )
        if not isinstance(role, str):
            raise ValueError(
                f"pregame-native key {key!r}: teamposition must be a str, got "
                f"{type(role).__name__!r}; build-conditional state must not reach "
                "a pregame-native/accepted surface"
            )


def validate_accepted_build_source(source: str) -> str:
    """Reject non-draft-safe build sources for accepted test/serving scoring."""
    if source not in BUILD_SOURCES:
        raise ValueError(f"unknown build source label: {source!r}")
    if source not in ACCEPTED_BUILD_SOURCES:
        raise ValueError(
            f"build source {source!r} is diagnostics/training-only and must not "
            "reach accepted test or serving scoring"
        )
    return source


def profile_id(champion_id: int, teamposition: str, primary_label: str) -> str:
    """Canonical serialised id for one train-supported build profile."""
    return f"{int(champion_id)}:{teamposition}:{primary_label}"


@dataclass(frozen=True)
class CatalogGates:
    profile_min_count: int = 20
    profile_min_share: float = 0.01
    rl_core_min_count: int = 50
    rl_core_min_share: float = 0.02
    tau: float = 20.0  # EB smoothing strength toward the fallback distribution


@dataclass(frozen=True)
class BuildProfile:
    champion_id: int
    teamposition: str
    primary_label: str
    hgnn_build_id: int  # index into model.config.build_vocab
    support_count: int
    support_share: float  # share of the pre-pruning empirical distribution
    support_tier: str  # "core" (clears rl_core gates) | "supported"
    catalog_version: str

    @property
    def profile_id(self) -> str:
        return profile_id(self.champion_id, self.teamposition, self.primary_label)


@dataclass(frozen=True)
class BuildPriorVector:
    champion_id: int
    teamposition: str
    profile_ids: tuple[str, ...]
    hgnn_build_ids: tuple[int, ...]
    probabilities: tuple[float, ...]  # smoothed, sums to 1 over retained profiles
    support_counts: tuple[int, ...]
    retained_mass: float  # of the pre-pruning empirical distribution
    pruned_mass: float
    fallback_source: str  # "champion_role" | "role" | "global"
    smoothing_strength: float
    catalog_version: str


@dataclass(frozen=True)
class _FallbackDist:
    """Pruned raw label distribution used as smoothing target and fallback."""

    labels: tuple[str, ...]
    probabilities: tuple[float, ...]
    counts: tuple[int, ...]
    retained_mass: float


@dataclass(frozen=True)
class BuildCatalog:
    asset_family: ClassVar[str] = ASSET_FAMILY_PREGAME_NATIVE
    key_space: ClassVar[tuple[str, ...]] = KEY_SPACE_CHAMP_ROLE

    build_vocab: tuple[str, ...]
    gates: CatalogGates
    version: str
    vectors: dict[tuple[int, str], BuildPriorVector]
    role_fallback: dict[str, _FallbackDist]
    global_fallback: _FallbackDist

    def prior_vector(self, champion_id: int, teamposition: str) -> BuildPriorVector:
        """Pregame prior for one slot, falling back role -> global if unseen."""
        vector = self.vectors.get((int(champion_id), teamposition))
        if vector is not None:
            return vector
        fallback = self.role_fallback.get(teamposition)
        source = "role"
        if fallback is None or not fallback.labels:
            fallback, source = self.global_fallback, "global"
        if not fallback.labels:
            raise ValueError(
                f"no catalog support for ({champion_id}, {teamposition}) and no "
                "fallback distribution; the catalog is empty"
            )
        return self._vector_from_fallback(
            int(champion_id), teamposition, fallback, source
        )

    def _vector_from_fallback(
        self,
        champion_id: int,
        teamposition: str,
        fallback: _FallbackDist,
        source: str,
    ) -> BuildPriorVector:
        build_index = {label: i for i, label in enumerate(self.build_vocab)}
        return BuildPriorVector(
            champion_id=champion_id,
            teamposition=teamposition,
            profile_ids=tuple(
                profile_id(champion_id, teamposition, label)
                for label in fallback.labels
            ),
            hgnn_build_ids=tuple(build_index[label] for label in fallback.labels),
            probabilities=fallback.probabilities,
            support_counts=fallback.counts,
            retained_mass=fallback.retained_mass,
            pruned_mass=1.0 - fallback.retained_mass,
            fallback_source=source,
            smoothing_strength=self.gates.tau,
            catalog_version=self.version,
        )

    def profiles(self) -> list[BuildProfile]:
        """All retained champion-role profiles (fallback vectors excluded)."""
        out: list[BuildProfile] = []
        for (champion_id, teamposition), vector in sorted(self.vectors.items()):
            # Pre-pruning empirical total: retained counts cover retained_mass.
            total = sum(vector.support_counts) / vector.retained_mass
            for label_idx, count in zip(vector.hgnn_build_ids, vector.support_counts):
                label = self.build_vocab[label_idx]
                share = count / total if total else 0.0
                out.append(
                    BuildProfile(
                        champion_id=champion_id,
                        teamposition=teamposition,
                        primary_label=label,
                        hgnn_build_id=label_idx,
                        support_count=count,
                        support_share=share,
                        support_tier=_support_tier(count, share, self.gates),
                        catalog_version=self.version,
                    )
                )
        return out

    def validate_model_vocab(self, model_build_vocab: tuple[str, ...]) -> None:
        """The checkpoint's build_vocab is the single canonical ordering."""
        if tuple(model_build_vocab) != self.build_vocab:
            raise ValueError(
                "build catalog vocab does not match the model checkpoint "
                f"build_vocab: catalog={self.build_vocab} "
                f"model={tuple(model_build_vocab)}"
            )

    def assert_pregame_native(self) -> None:
        """Assert this catalog is a valid pregame-native asset (champ-role keys only)."""
        assert_pregame_native_key_space(self.vectors.keys())


def _support_tier(count: int, share: float, gates: CatalogGates) -> str:
    if count >= gates.rl_core_min_count and share >= gates.rl_core_min_share:
        return "core"
    return "supported"


def _fallback_dist(
    counts: dict[str, int], min_share: float
) -> _FallbackDist:
    total = sum(counts.values())
    if total <= 0:
        return _FallbackDist((), (), (), 0.0)
    retained = sorted(
        (label, n) for label, n in counts.items() if n / total >= min_share
    )
    retained_total = sum(n for _, n in retained)
    return _FallbackDist(
        labels=tuple(label for label, _ in retained),
        probabilities=tuple(n / retained_total for _, n in retained),
        counts=tuple(n for _, n in retained),
        retained_mass=retained_total / total,
    )


def catalog_version(
    build_vocab: tuple[str, ...],
    gates: CatalogGates,
    counts: dict[tuple[int, str, str], int],
) -> str:
    payload = json.dumps(
        {
            "build_vocab": list(build_vocab),
            "gates": asdict(gates),
            "counts": [[c, r, b, n] for (c, r, b), n in sorted(counts.items())],
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_catalog(
    p1: dict[tuple[int, str, str], tuple[float, int]],
    build_vocab: tuple[str, ...],
    gates: CatalogGates | None = None,
) -> BuildCatalog:
    """Construct the catalog from train-only solo prior rows.

    `p1` maps (championid, teamposition, build) -> (win_rate, matchups), as
    loaded by `app.ml.priors.load_priors` (split='train' only). `build_vocab`
    must be the checkpoint's canonical ordering.
    """
    gates = gates or CatalogGates()
    vocab = set(build_vocab)
    build_index = {label: i for i, label in enumerate(build_vocab)}
    counts: dict[tuple[int, str, str], int] = {}
    for (champion_id, teamposition, label), (_wr, matchups) in p1.items():
        if label not in vocab:
            raise ValueError(
                f"train prior build label {label!r} is not in the model "
                f"build_vocab; priors and checkpoint disagree"
            )
        counts[(int(champion_id), teamposition, label)] = int(matchups)

    version = catalog_version(tuple(build_vocab), gates, counts)
    by_cell: dict[tuple[int, str], dict[str, int]] = {}
    role_counts: dict[str, dict[str, int]] = {}
    global_counts: dict[str, int] = {}
    for (champion_id, teamposition, label), n in counts.items():
        by_cell.setdefault((champion_id, teamposition), {})[label] = n
        role = role_counts.setdefault(teamposition, {})
        role[label] = role.get(label, 0) + n
        global_counts[label] = global_counts.get(label, 0) + n

    role_fallback = {
        role: _fallback_dist(role_label_counts, gates.profile_min_share)
        for role, role_label_counts in role_counts.items()
    }
    global_fallback = _fallback_dist(global_counts, gates.profile_min_share)

    vectors: dict[tuple[int, str], BuildPriorVector] = {}
    for (champion_id, teamposition), cell_counts in by_cell.items():
        total = sum(cell_counts.values())
        retained = sorted(
            (label, n)
            for label, n in cell_counts.items()
            if n >= gates.profile_min_count and n / total >= gates.profile_min_share
        )
        if not retained:
            # Unsupported champion-role: served by the fallback chain at query
            # time so the fallback_source label stays accurate.
            continue
        retained_total = sum(n for _, n in retained)
        fallback = role_fallback.get(teamposition, global_fallback)
        q_raw = np.array(
            [dict(zip(fallback.labels, fallback.probabilities)).get(label, 0.0)
             for label, _ in retained],
            dtype=np.float64,
        )
        q = q_raw / q_raw.sum() if q_raw.sum() > 0 else np.full(
            len(retained), 1.0 / len(retained)
        )
        n_arr = np.array([n for _, n in retained], dtype=np.float64)
        probabilities = (n_arr + gates.tau * q) / (retained_total + gates.tau)
        vectors[(champion_id, teamposition)] = BuildPriorVector(
            champion_id=champion_id,
            teamposition=teamposition,
            profile_ids=tuple(
                profile_id(champion_id, teamposition, label) for label, _ in retained
            ),
            hgnn_build_ids=tuple(build_index[label] for label, _ in retained),
            probabilities=tuple(float(p) for p in probabilities),
            support_counts=tuple(int(n) for _, n in retained),
            retained_mass=retained_total / total,
            pruned_mass=1.0 - retained_total / total,
            fallback_source="champion_role",
            smoothing_strength=gates.tau,
            catalog_version=version,
        )

    return BuildCatalog(
        build_vocab=tuple(build_vocab),
        gates=gates,
        version=version,
        vectors=vectors,
        role_fallback=role_fallback,
        global_fallback=global_fallback,
    )


def build_catalog_from_priors(
    build_vocab: tuple[str, ...],
    gates: CatalogGates | None = None,
) -> BuildCatalog:
    """Catalog from the live train-only `synergy_1vx` rows."""
    from app.ml.priors import load_priors

    return build_catalog(load_priors().p1, build_vocab, gates)


@dataclass(frozen=True)
class ConditionGates:
    child_min_count: int = 50
    tau: float = 50.0


def conditioned_prior_vector(
    catalog: BuildCatalog,
    champion_id: int,
    teamposition: str,
    keystone: int,
    cell_counts: dict[str, int] | None,
    gates: ConditionGates,
) -> BuildPriorVector:
    """Return a keystone-conditioned build prior, or the parent if gating fails.

    ``cell_counts`` maps build label -> train count for this exact
    (champion, teamposition, keystone) cell. Conditioning can only reweight
    the parent's retained labels; it never introduces a build outside the
    parent (champ, role) supported set.

    Note: the returned ``retained_mass`` is cell-local (parent-retained rows /
    all rows in this keystone cell), not the parent's corpus-level mass.

    Falls back to parent when:
    - keystone <= 0 (missing rune data)
    - parent was not a direct champion-role fit (fallback_source != "champion_role")
    - the cell has no train rows, or its count restricted to parent labels is
      below gates.child_min_count
    """
    parent = catalog.prior_vector(champion_id, teamposition)
    if keystone <= 0 or parent.fallback_source != "champion_role" or not cell_counts:
        return parent

    # Restrict child counts to the parent's retained labels only.
    retained_labels = [catalog.build_vocab[hid] for hid in parent.hgnn_build_ids]
    restricted: dict[str, int] = {
        label: cell_counts.get(label, 0) for label in retained_labels
    }
    # Total cell rows including labels pruned by the parent (pruned mass).
    all_child_for_cell = sum(cell_counts.values())

    child_total_restricted = sum(restricted.values())
    if child_total_restricted < gates.child_min_count:
        return parent

    # EB: p_i = (n_i + tau * p_parent_i) / (N_child_restricted + tau)
    n_arr = np.array(
        [restricted[catalog.build_vocab[hid]] for hid in parent.hgnn_build_ids],
        dtype=np.float64,
    )
    p_parent = np.array(parent.probabilities, dtype=np.float64)
    N = float(child_total_restricted)
    tau = gates.tau
    probs = (n_arr + tau * p_parent) / (N + tau)
    retained_mass = N / all_child_for_cell if all_child_for_cell > 0 else 1.0

    return BuildPriorVector(
        champion_id=int(champion_id),
        teamposition=teamposition,
        profile_ids=parent.profile_ids,
        hgnn_build_ids=parent.hgnn_build_ids,
        probabilities=tuple(float(p) for p in probs),
        support_counts=tuple(int(restricted[catalog.build_vocab[hid]]) for hid in parent.hgnn_build_ids),
        retained_mass=retained_mass,
        pruned_mass=1.0 - retained_mass,
        fallback_source="champion_role_keystone",
        smoothing_strength=tau,
        catalog_version=parent.catalog_version,
    )


def enumerate_joint_worlds(
    slot_probabilities: list[np.ndarray],
    *,
    k_slot: int = 3,
    max_worlds: int = 512,
    early_stop_mass: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact lazy top-W enumeration of joint build assignments.

    `slot_probabilities` holds one unnormalised retained distribution per slot.
    Returns `(selections, weights, retained_joint_mass)`: `selections[w, s]`
    indexes slot `s`'s candidate list, `weights` are raw joint products
    (never silently renormalised), and `retained_joint_mass` is their sum
    relative to the full joint distribution.
    """
    if k_slot < 1 or max_worlds < 1:
        raise ValueError("k_slot and max_worlds must be >= 1")
    n_slots = len(slot_probabilities)
    orders: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for slot in slot_probabilities:
        arr = np.asarray(slot, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0 or np.any(arr < 0.0):
            raise ValueError("each slot needs a non-empty non-negative distribution")
        order = np.argsort(-arr)[:k_slot]
        order = order[arr[order] > 0.0]
        if order.size == 0:
            raise ValueError("each slot needs at least one positive-mass candidate")
        orders.append(order)
        probs.append(arr[order])

    start = (0,) * n_slots
    start_weight = float(np.prod([p[0] for p in probs]))
    heap: list[tuple[float, tuple[int, ...]]] = [(-start_weight, start)]
    visited: set[tuple[int, ...]] = {start}
    selections: list[tuple[int, ...]] = []
    weights: list[float] = []
    mass = 0.0
    while heap and len(selections) < max_worlds:
        neg_weight, idx = heapq.heappop(heap)
        weight = -neg_weight
        selections.append(idx)
        weights.append(weight)
        mass += weight
        if mass >= early_stop_mass:
            break
        for s in range(n_slots):
            if idx[s] + 1 >= probs[s].size:
                continue
            succ = idx[:s] + (idx[s] + 1,) + idx[s + 1 :]
            if succ in visited:
                continue
            visited.add(succ)
            succ_weight = weight / probs[s][idx[s]] * probs[s][idx[s] + 1]
            heapq.heappush(heap, (-succ_weight, succ))

    selection_arr = np.array(
        [[int(orders[s][idx[s]]) for s in range(n_slots)] for idx in selections],
        dtype=np.int64,
    )
    return selection_arr, np.asarray(weights, dtype=np.float64), float(mass)


__all__ = [
    "ACCEPTED_BUILD_SOURCES",
    "ASSET_FAMILY_BUILD_CONDITIONAL",
    "ASSET_FAMILY_PREGAME_NATIVE",
    "BUILD_SOURCES",
    "BUILD_SOURCE_ORACLE_OBSERVED",
    "BUILD_SOURCE_PREGAME_MARGINAL",
    "BUILD_SOURCE_RL_CANDIDATE",
    "BUILD_SOURCE_TRAIN_OBSERVED",
    "BuildCatalog",
    "BuildPriorVector",
    "BuildProfile",
    "CatalogGates",
    "ConditionGates",
    "KEY_SPACE_CHAMP_ROLE",
    "KEY_SPACE_CHAMP_ROLE_BUILD",
    "assert_pregame_native_key_space",
    "build_catalog",
    "build_catalog_from_priors",
    "catalog_version",
    "conditioned_prior_vector",
    "enumerate_joint_worlds",
    "profile_id",
    "validate_accepted_build_source",
]
