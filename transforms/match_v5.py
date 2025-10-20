from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    Optional,
    Set,
)

PRIMITIVES = (str, int, float, bool, type(None))


class SchemaError(ValueError):
    pass


def _sorted_list(xs: Iterable[str]) -> List[str]:
    return sorted(set(xs))


def _get_in(obj: Any, path: Sequence[Tuple[str | int, str]]) -> Any:
    """
    Safe nested lookup. `path` is list of (key/index, label) pairs for better errors.
    Example: [("info","info"), ("participants","participants"), (0,"participants[0]")]
    """
    cur = obj
    for key, label in path:
        try:
            cur = cur[key]
        except Exception:
            raise SchemaError(
                f"Missing or invalid path: {'.'.join(l for _, l in path)} (failed at: {label})"
            )
    return cur


def _get_nested_keys(nested_obj: Any) -> List[str]:
    """Collect primitive-valued keys anywhere inside a nested structure."""
    keys: Set[str] = set()

    def visit(o: Any) -> None:
        if isinstance(o, Mapping):
            for k, v in o.items():
                if isinstance(v, PRIMITIVES):
                    keys.add(k)
                else:
                    visit(v)
        elif isinstance(o, Sequence) and not isinstance(o, (str, bytes, bytearray)):
            for item in o:
                visit(item)

    visit(nested_obj)
    return _sorted_list(keys)


# --- individual checkers (kept small & pure) ---


def check_game_info(game: Mapping[str, Any]) -> List[str]:
    info = _get_in(game, [("info", "info")])
    if not isinstance(info, Mapping):
        raise SchemaError("info must be an object")
    return _sorted_list(info.keys())


def check_metadata(game: Mapping[str, Any]) -> List[str]:
    metadata = _get_in(game, [("metadata", "metadata")])
    if not isinstance(metadata, Mapping):
        raise SchemaError("metadata must be an object")
    return _sorted_list(metadata.keys())


def check_objectives(game: Mapping[str, Any]) -> List[str]:
    objs = _get_in(
        game,
        [
            ("info", "info"),
            ("teams", "teams"),
            (0, "teams[0]"),
            ("objectives", "objectives"),
        ],
    )
    if not isinstance(objs, Mapping):
        raise SchemaError("objectives must be an object")
    return _sorted_list(objs.keys())


def check_participants(game: Mapping[str, Any]) -> List[str]:
    p0 = _get_in(
        game,
        [("info", "info"), ("participants", "participants"), (0, "participants[0]")],
    )
    if not isinstance(p0, Mapping):
        raise SchemaError("participants[0] must be an object")
    return _sorted_list(p0.keys())


def check_perks(game: Mapping[str, Any]) -> List[str]:
    perks = _get_in(
        game,
        [
            ("info", "info"),
            ("participants", "participants"),
            (0, "participants[0]"),
            ("perks", "perks"),
        ],
    )
    return _get_nested_keys(perks)


def check_challenges(game: Mapping[str, Any]) -> List[str]:
    participants = _get_in(game, [("info", "info"), ("participants", "participants")])
    if not isinstance(participants, Sequence):
        raise SchemaError("participants must be a list")
    seen: Set[str] = set()
    for p in participants:
        if isinstance(p, Mapping):
            ch = p.get("challenges", {})
            if isinstance(ch, Mapping):
                seen.update(ch.keys())
    return _sorted_list(seen)


# --- configurable collector ---

CheckFn = Callable[[Mapping[str, Any]], List[str]]


@dataclass(frozen=True)
class MatchSchemaCollector:
    """Collects key snapshots from match JSON with consistent errors & logging."""

    strict: bool = (
        True  # if False, missing sections produce empty lists instead of raising
    )

    # registries allow easy extension / toggling of checks
    per_game: Mapping[str, CheckFn] = field(
        default_factory=lambda: {
            "game_info_keys": check_game_info,
            "metadata_keys": check_metadata,
            "objectives_keys": check_objectives,
            "participants_keys": check_participants,
            "perks_keys": check_perks,
        }
    )
    per_player: Mapping[str, CheckFn] = field(
        default_factory=lambda: {
            "challenges_keys": check_challenges,
        }
    )

    def collect(self, game: Mapping[str, Any]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for name, fn in {**self.per_game, **self.per_player}.items():
            try:
                out[name] = fn(game)
            except SchemaError as e:
                if self.strict:
                    raise
                out[name] = []
        return out
