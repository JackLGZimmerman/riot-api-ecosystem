"""Frozen three-encoder sidecar artifact lookup.

The artifact is one row per `(championid, teamposition, build)` identity and
stores three independent latent blocks. Static latents are champion-level and
may repeat across role/build rows; full-game and temporal latents are native
to the full identity grain.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

N_PLAYERS = 10
SIDE_CAR_BLOCKS = ("static", "full_game", "temporal")
LATENT_KEYS = {
    "static": "static_latents",
    "full_game": "full_game_latents",
    "temporal": "temporal_latents",
}
REQUIRED_SIDECAR_ARRAYS = frozenset(
    {
        "champion_id",
        "teamposition",
        "build",
        "static_latents",
        "full_game_latents",
        "temporal_latents",
    }
)
EMPIRICAL_STATIC_METADATA_TOKENS = (
    "win_rate",
    "winrate",
    "matchups",
    "matchup_",
    "synergy",
    "prior",
    "_cnt",
    "_count",
)
TRAIN_ONLY_SPLIT_KEYS = frozenset(
    {
        "fit_split",
        "source_split",
        "source_splits",
        "input_split",
        "input_splits",
        "training_split",
        "training_splits",
        "encoder_fit_split",
        "encoder_source_split",
        "encoder_source_splits",
        "aggregation_split",
        "aggregation_splits",
    }
)
TRAIN_SPLIT_VALUES = frozenset({"train", "training"})


@dataclass(frozen=True)
class EncoderSidecarDims:
    static: int
    full_game: int
    temporal: int

    @property
    def total(self) -> int:
        return int(self.static + self.full_game + self.temporal)

    def as_dict(self) -> dict[str, int]:
        return {
            "static": int(self.static),
            "full_game": int(self.full_game),
            "temporal": int(self.temporal),
            "total": int(self.total),
        }


class EncoderSidecarLookup:
    def __init__(
        self,
        *,
        champion_id: np.ndarray,
        teamposition: np.ndarray,
        build: np.ndarray,
        static_latents: np.ndarray,
        full_game_latents: np.ndarray,
        temporal_latents: np.ndarray,
        support: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.champion_id = _as_1d_int("champion_id", champion_id)
        self.teamposition = _as_1d_str("teamposition", teamposition)
        self.build = _as_1d_str("build", build)
        n_rows = int(self.champion_id.shape[0])
        if self.teamposition.shape[0] != n_rows or self.build.shape[0] != n_rows:
            raise ValueError("identity key arrays must have the same row count")
        self.static_latents = _as_latents("static_latents", static_latents, n_rows)
        self.full_game_latents = _as_latents("full_game_latents", full_game_latents, n_rows)
        self.temporal_latents = _as_latents("temporal_latents", temporal_latents, n_rows)
        if support is None:
            self.support = np.ones(n_rows, dtype=np.float32)
        else:
            self.support = _as_1d_float("support", support, n_rows)
        self.metadata = metadata or {}
        validate_static_metadata(self.metadata.get("static_encoder", {}))
        validate_train_only_metadata(self.metadata)
        self.dims = EncoderSidecarDims(
            static=int(self.static_latents.shape[1]),
            full_game=int(self.full_game_latents.shape[1]),
            temporal=int(self.temporal_latents.shape[1]),
        )
        self._index = {
            (int(champ), str(role), str(build_label)): i
            for i, (champ, role, build_label) in enumerate(
                zip(self.champion_id, self.teamposition, self.build, strict=True)
            )
        }

    @classmethod
    def load(cls, path: Path | str) -> "EncoderSidecarLookup":
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            missing = sorted(REQUIRED_SIDECAR_ARRAYS.difference(data.files))
            if missing:
                raise ValueError(
                    f"encoder sidecar artifact {source} is missing required arrays: "
                    + ", ".join(missing)
                )
            metadata_json = str(data["metadata_json"].item()) if "metadata_json" in data else "{}"
            metadata = json.loads(metadata_json)
            return cls(
                champion_id=data["champion_id"],
                teamposition=data["teamposition"],
                build=data["build"],
                static_latents=data["static_latents"],
                full_game_latents=data["full_game_latents"],
                temporal_latents=data["temporal_latents"],
                support=data["support"] if "support" in data else None,
                metadata=metadata,
            )

    def lookup_blocks(
        self,
        identities: Iterable[tuple[int, str, str]],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        rows = list(identities)
        block_arrays = {
            "static": np.zeros((len(rows), self.dims.static), dtype=np.float32),
            "full_game": np.zeros((len(rows), self.dims.full_game), dtype=np.float32),
            "temporal": np.zeros((len(rows), self.dims.temporal), dtype=np.float32),
        }
        support = np.zeros(len(rows), dtype=np.float32)
        sources = {
            "static": self.static_latents,
            "full_game": self.full_game_latents,
            "temporal": self.temporal_latents,
        }
        for out_idx, (champion, role, build_label) in enumerate(rows):
            source_idx = self._index.get((int(champion), str(role), str(build_label)))
            if source_idx is None:
                continue
            support[out_idx] = self.support[source_idx]
            for block, source in sources.items():
                if source.shape[1] > 0:
                    block_arrays[block][out_idx] = source[source_idx]
        return block_arrays, support

    def lookup_game_blocks(
        self,
        identities: Iterable[tuple[int, str, str]],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        blocks, support = self.lookup_blocks(identities)
        if support.shape[0] != N_PLAYERS:
            raise ValueError(f"expected {N_PLAYERS} identities for one game")
        return (
            {name: values.reshape(1, N_PLAYERS, values.shape[1]) for name, values in blocks.items()},
            support.reshape(1, N_PLAYERS),
        )


def save_encoder_sidecar(
    path: Path | str,
    *,
    champion_id: np.ndarray,
    teamposition: np.ndarray,
    build: np.ndarray,
    static_latents: np.ndarray,
    full_game_latents: np.ndarray,
    temporal_latents: np.ndarray,
    support: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    metadata = metadata or {}
    validate_static_metadata(metadata.get("static_encoder", {}))
    lookup = EncoderSidecarLookup(
        champion_id=champion_id,
        teamposition=teamposition,
        build=build,
        static_latents=static_latents,
        full_game_latents=full_game_latents,
        temporal_latents=temporal_latents,
        support=support,
        metadata=metadata,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        champion_id=lookup.champion_id.astype(np.int32, copy=False),
        teamposition=lookup.teamposition,
        build=lookup.build,
        static_latents=lookup.static_latents,
        full_game_latents=lookup.full_game_latents,
        temporal_latents=lookup.temporal_latents,
        support=lookup.support,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return out


def feature_hash(values: Iterable[str]) -> str:
    """Stable SHA-256 digest for ordered feature names/config keys."""
    payload = json.dumps([str(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_encoder_sidecar_metadata(
    *,
    static_features: Iterable[str],
    full_game_features: Iterable[str],
    temporal_features: Iterable[str],
    split_metadata: dict[str, Any],
    encoder_configs: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the required metadata block for a three-encoder artifact."""
    metadata = {
        "feature_hashes": {
            "static": feature_hash(static_features),
            "full_game": feature_hash(full_game_features),
            "temporal": feature_hash(temporal_features),
        },
        "split_metadata": split_metadata,
        "encoder_configs": encoder_configs,
    }
    if extra:
        metadata.update(extra)
    validate_static_metadata(metadata.get("static_encoder", {}))
    validate_train_only_metadata(metadata)
    return metadata


def validate_static_metadata(metadata: Any) -> None:
    """Reject static-branch metadata that names empirical prior sources."""
    if metadata in (None, {}):
        return
    text = json.dumps(metadata, sort_keys=True).lower()
    for token in EMPIRICAL_STATIC_METADATA_TOKENS:
        if token in text:
            raise ValueError(
                "static encoder metadata cannot reference empirical priors, "
                f"win rates, or support counts: {token}"
            )


def validate_train_only_metadata(metadata: Any) -> None:
    """Reject explicit encoder source/fit split metadata that is not train-only."""
    for key, value in _walk_mapping_items(metadata):
        if str(key).lower() not in TRAIN_ONLY_SPLIT_KEYS:
            continue
        splits = _string_values(value)
        leaking = sorted({split for split in splits if split not in TRAIN_SPLIT_VALUES})
        if leaking:
            raise ValueError(
                "encoder sidecar artifacts must be fit from train-split aggregates only; "
                f"{key} includes {leaking}"
            )


def _walk_mapping_items(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_mapping_items(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mapping_items(child)


def _string_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip().lower(),)
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(_string_values(child))
        return tuple(out)
    if isinstance(value, (list, tuple, set)):
        out = []
        for child in value:
            out.extend(_string_values(child))
        return tuple(out)
    return ()


def _as_1d_int(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr.astype(np.int32, copy=False)


def _as_1d_str(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr.astype(str, copy=False)


def _as_1d_float(name: str, values: np.ndarray, n_rows: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != n_rows:
        raise ValueError(f"{name} must have shape [{n_rows}]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_latents(name: str, values: np.ndarray, n_rows: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != n_rows:
        raise ValueError(f"{name} must have shape [{n_rows}, dim]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


__all__ = [
    "EncoderSidecarDims",
    "EncoderSidecarLookup",
    "LATENT_KEYS",
    "REQUIRED_SIDECAR_ARRAYS",
    "SIDE_CAR_BLOCKS",
    "build_encoder_sidecar_metadata",
    "feature_hash",
    "save_encoder_sidecar",
    "validate_static_metadata",
    "validate_train_only_metadata",
]
