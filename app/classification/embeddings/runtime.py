"""Runtime lookup helpers for generated classification embedding artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from app.classification.embeddings.config import (
    DENSE_IDENTITY_CACHE_PATH,
    DENSE_IDENTITY_DIM,
    IDENTITY_CONTEXT_CACHE_PATH,
    IDENTITY_CONTEXT_DIM,
    IDENTITY_CONTEXT_RAW_DIM,
    IDENTITY_PROFILE_CACHE_PATH,
    IDENTITY_PROFILE_DIM,
    RELATIONSHIP_DETAIL_CACHE_DIR,
)
from app.core.utils.common import TEAM_PAIRS
from app.core.utils.smoothing import build_group_for

DEFAULT_RELATIONSHIP_DETAIL_DIM = 16


def _empty(dim: int) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _rows_to_dict(keys: np.ndarray, values: np.ndarray) -> dict[tuple, np.ndarray]:
    return {
        tuple(key.tolist() if hasattr(key, "tolist") else key): values[i].astype(np.float32)
        for i, key in enumerate(keys)
    }


def _rows_to_count_dict(keys: np.ndarray, counts: np.ndarray) -> dict[tuple, float]:
    return {
        tuple(key.tolist() if hasattr(key, "tolist") else key): float(counts[i])
        for i, key in enumerate(keys)
    }


@dataclass(frozen=True)
class IdentitySemanticLookup:
    values: dict[tuple[int, str, str], np.ndarray]
    dim: int = DENSE_IDENTITY_DIM

    @classmethod
    def load(cls, path: Path = DENSE_IDENTITY_CACHE_PATH) -> "IdentitySemanticLookup":
        if not path.exists():
            return cls(values={}, dim=DENSE_IDENTITY_DIM)
        with np.load(path, allow_pickle=True) as payload:
            embeddings = payload["embeddings"].astype(np.float32)
            keys = payload["keys"]
            dim = int(payload["dim"].item()) if "dim" in payload.files else embeddings.shape[1]
        values = {
            (int(key[0]), str(key[1]), str(key[2])): embeddings[i]
            for i, key in enumerate(keys)
        }
        return cls(values=values, dim=dim)

    def lookup_players(self, tuples: Iterable[tuple[int, str, str]]) -> np.ndarray:
        default = _empty(self.dim)
        return np.stack(
            [self.values.get((int(c), str(p), str(b)), default) for c, p, b in tuples],
            axis=0,
        ).astype(np.float32)


@dataclass(frozen=True)
class IdentityProfileLookup:
    """Per-identity matchup profile (offense + resistance axes), keyed like the
    dense descriptor. Missing identities default to a zero profile, which the
    HGNN's antisymmetric cross-team term maps to no contribution."""

    values: dict[tuple[int, str, str], np.ndarray]
    dim: int = IDENTITY_PROFILE_DIM

    @classmethod
    def load(cls, path: Path = IDENTITY_PROFILE_CACHE_PATH) -> "IdentityProfileLookup":
        if not path.exists():
            return cls(values={}, dim=IDENTITY_PROFILE_DIM)
        with np.load(path, allow_pickle=True) as payload:
            embeddings = payload["embeddings"].astype(np.float32)
            keys = payload["keys"]
            dim = int(payload["dim"].item()) if "dim" in payload.files else embeddings.shape[1]
        values = {
            (int(key[0]), str(key[1]), str(key[2])): embeddings[i]
            for i, key in enumerate(keys)
        }
        return cls(values=values, dim=dim)

    def lookup_players(self, tuples: Iterable[tuple[int, str, str]]) -> np.ndarray:
        default = _empty(self.dim)
        return np.stack(
            [self.values.get((int(c), str(p), str(b)), default) for c, p, b in tuples],
            axis=0,
        ).astype(np.float32)


@dataclass(frozen=True)
class IdentityContextLookup:
    """Per-identity context-atlas descriptor, keyed like the matchup profile, plus
    per-identity support (`matchups`).

    Two descriptor views are exposed:

    * ``values`` — the compressed descriptor ``[interpretable axes || low-rank PCA
      tail]`` consumed by the shared context head (the "none" baseline).
    * ``raw`` — the wide RAW block ``[interpretable axes || standardised extra
      metrics]``, the primary interaction source for the identity-conditioned
      head. Empty for legacy artifacts written before the raw block existed.

    Missing identities default to a zero vector and zero support, which both heads
    map to no contribution."""

    values: dict[tuple[int, str, str], np.ndarray]
    support: dict[tuple[int, str, str], float]
    raw: dict[tuple[int, str, str], np.ndarray] = field(default_factory=dict)
    dim: int = IDENTITY_CONTEXT_DIM
    interpretable_dim: int = 0
    raw_dim: int = IDENTITY_CONTEXT_RAW_DIM

    @classmethod
    def load(cls, path: Path = IDENTITY_CONTEXT_CACHE_PATH) -> "IdentityContextLookup":
        if not path.exists():
            return cls(values={}, support={}, dim=IDENTITY_CONTEXT_DIM)
        with np.load(path, allow_pickle=True) as payload:
            embeddings = payload["embeddings"].astype(np.float32)
            keys = payload["keys"]
            dim = int(payload["dim"].item()) if "dim" in payload.files else embeddings.shape[1]
            interpretable_dim = (
                int(payload["interpretable_dim"].item())
                if "interpretable_dim" in payload.files
                else 0
            )
            matchups = (
                payload["matchups"].astype(np.float32)
                if "matchups" in payload.files
                else np.zeros(len(keys), dtype=np.float32)
            )
            raw_embeddings = (
                payload["raw_embeddings"].astype(np.float32)
                if "raw_embeddings" in payload.files
                else None
            )
            raw_dim = (
                int(payload["raw_dim"].item())
                if "raw_dim" in payload.files
                else (
                    raw_embeddings.shape[1]
                    if raw_embeddings is not None
                    else IDENTITY_CONTEXT_RAW_DIM
                )
            )
        values = {
            (int(key[0]), str(key[1]), str(key[2])): embeddings[i]
            for i, key in enumerate(keys)
        }
        support = {
            (int(key[0]), str(key[1]), str(key[2])): float(matchups[i])
            for i, key in enumerate(keys)
        }
        raw = (
            {
                (int(key[0]), str(key[1]), str(key[2])): raw_embeddings[i]
                for i, key in enumerate(keys)
            }
            if raw_embeddings is not None
            else {}
        )
        return cls(
            values=values,
            support=support,
            raw=raw,
            dim=dim,
            interpretable_dim=interpretable_dim,
            raw_dim=raw_dim,
        )

    def lookup_players(self, tuples: Iterable[tuple[int, str, str]]) -> np.ndarray:
        default = _empty(self.dim)
        return np.stack(
            [self.values.get((int(c), str(p), str(b)), default) for c, p, b in tuples],
            axis=0,
        ).astype(np.float32)

    def lookup_raw(self, tuples: Iterable[tuple[int, str, str]]) -> np.ndarray:
        default = _empty(self.raw_dim)
        return np.stack(
            [self.raw.get((int(c), str(p), str(b)), default) for c, p, b in tuples],
            axis=0,
        ).astype(np.float32)

    def lookup_support(self, tuples: Iterable[tuple[int, str, str]]) -> np.ndarray:
        return np.array(
            [self.support.get((int(c), str(p), str(b)), 0.0) for c, p, b in tuples],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class RelationshipDetailLookup:
    exact: dict[tuple, np.ndarray]
    exact_counts: dict[tuple, float]
    build_group: dict[tuple, np.ndarray]
    build_group_counts: dict[tuple, float]
    nobuild: dict[tuple, np.ndarray]
    nobuild_counts: dict[tuple, float]
    champion: dict[tuple, np.ndarray]
    champion_counts: dict[tuple, float]
    dim: int = DEFAULT_RELATIONSHIP_DETAIL_DIM
    threshold: float = 50.0

    @classmethod
    def empty(cls, dim: int = DEFAULT_RELATIONSHIP_DETAIL_DIM) -> "RelationshipDetailLookup":
        return cls({}, {}, {}, {}, {}, {}, {}, {}, dim=dim)

    @classmethod
    def load(
        cls,
        kind: str,
        *,
        directory: Path = RELATIONSHIP_DETAIL_CACHE_DIR,
        threshold: float = 50.0,
    ) -> "RelationshipDetailLookup":
        path = directory / f"{kind}.npz"
        if not path.exists():
            return cls.empty()
        with np.load(path, allow_pickle=True) as payload:
            dim = (
                int(payload["dim"].item())
                if "dim" in payload.files
                else DEFAULT_RELATIONSHIP_DETAIL_DIM
            )
            kwargs = {}
            for level in ("exact", "build_group", "nobuild", "champion"):
                kwargs[level] = _rows_to_dict(payload[f"{level}_keys"], payload[f"{level}_values"])
                kwargs[f"{level}_counts"] = _rows_to_count_dict(
                    payload[f"{level}_keys"],
                    payload[f"{level}_matchups"],
                )
        return cls(**kwargs, dim=dim, threshold=threshold)

    def _select(self, candidates: tuple[tuple[tuple, float], ...]) -> np.ndarray:
        default = _empty(self.dim)
        fallback: np.ndarray | None = None
        for key, sign in candidates:
            for table, counts in (
                (self.exact, self.exact_counts),
                (self.build_group, self.build_group_counts),
                (self.nobuild, self.nobuild_counts),
                (self.champion, self.champion_counts),
            ):
                value = table.get(key)
                if value is None:
                    continue
                signed = value * sign
                if fallback is None:
                    fallback = signed
                if counts.get(key, 0.0) >= self.threshold:
                    return signed.astype(np.float32)
                break
        return (fallback if fallback is not None else default).astype(np.float32)

    @staticmethod
    def _m1v1_candidates(
        blue: tuple[int, str, str],
        red: tuple[int, str, str],
    ) -> tuple[tuple[tuple, float], ...]:
        exact_swapped = blue > red
        left, right = (red, blue) if exact_swapped else (blue, red)

        def orient(a: tuple, b: tuple) -> tuple[tuple, float]:
            return ((*a, *b), 1.0) if a <= b else ((*b, *a), -1.0)

        bg_b = (blue[0], blue[1], build_group_for(blue[2]))
        bg_r = (red[0], red[1], build_group_for(red[2]))
        nb_b = (blue[0], blue[1])
        nb_r = (red[0], red[1])
        ch_b = (blue[0],)
        ch_r = (red[0],)
        return (
            ((*left, *right), -1.0 if exact_swapped else 1.0),
            orient(bg_b, bg_r),
            orient(nb_b, nb_r),
            orient(ch_b, ch_r),
        )

    @staticmethod
    def _s2vx_candidates(
        a: tuple[int, str, str],
        b: tuple[int, str, str],
    ) -> tuple[tuple[tuple, float], ...]:
        def key(left: tuple, right: tuple) -> tuple:
            return (*left, *right) if left <= right else (*right, *left)

        return (
            (key(a, b), 1.0),
            (key((a[0], a[1], build_group_for(a[2])), (b[0], b[1], build_group_for(b[2]))), 1.0),
            (key((a[0], a[1]), (b[0], b[1])), 1.0),
            (key((a[0],), (b[0],)), 1.0),
        )

    def lookup_1v1_blue(
        self,
        blue_tuples: list[tuple[int, str, str]],
        red_tuples: list[tuple[int, str, str]],
    ) -> np.ndarray:
        return np.stack(
            [
                self._select(self._m1v1_candidates(blue, red))
                for blue in blue_tuples
                for red in red_tuples
            ],
            axis=0,
        ).astype(np.float32)

    def lookup_2vx_team(self, team_tuples: list[tuple[int, str, str]]) -> np.ndarray:
        return np.stack(
            [
                self._select(self._s2vx_candidates(team_tuples[a], team_tuples[b]))
                for a, b in TEAM_PAIRS
            ],
            axis=0,
        ).astype(np.float32)
