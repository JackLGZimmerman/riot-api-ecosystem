"""Per-champion eligibility pool of (role, build) assignments.

Each entry says: "champion C is plausibly played as role R with build B,
with this much prior weight." The pool drives `make_pool_sampler` in
`app.rl.reward`, which enumerates valid role assignments and the top-K
build combinations per terminal team — replacing the old pick-order
role assignment, which silently assumed pick order == role.

File format (JSON):

    {
        "champion_id": [["ROLE", build_id, weight], ...],
        ...
    }

Generate from priors:

    python -m app.rl.pool generate [--min-matchups N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.core.config.settings import PROJECT_ROOT

DEFAULT_POOL_PATH: Path = PROJECT_ROOT / "app" / "rl" / "data" / "champion_pool.json"


@dataclass(frozen=True)
class PoolEntry:
    role: str
    build_id: int
    weight: float  # un-normalised; relative likelihood within a champion


@dataclass(frozen=True)
class ChampionPool:
    """Lookup of champion_id -> tuple[PoolEntry, ...]."""

    entries: dict[int, tuple[PoolEntry, ...]]

    def roles_for(self, champion_id: int) -> frozenset[str]:
        es = self.entries.get(champion_id)
        if not es:
            return frozenset()
        return frozenset(e.role for e in es)

    def builds_for(
        self, champion_id: int, role: str
    ) -> tuple[tuple[int, float], ...]:
        es = self.entries.get(champion_id)
        if not es:
            return ()
        return tuple((e.build_id, e.weight) for e in es if e.role == role)

    def contains(self, champion_id: int, role: str) -> bool:
        es = self.entries.get(champion_id)
        return bool(es) and any(e.role == role for e in es)


def load_pool(path: Path | None = None) -> ChampionPool:
    path = path or DEFAULT_POOL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Champion pool file not found: {path}. "
            f"Generate it with: python -m app.rl.pool generate"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[int, tuple[PoolEntry, ...]] = {}
    for cid_str, rows in raw.items():
        entries[int(cid_str)] = tuple(
            PoolEntry(role=str(r), build_id=int(b), weight=float(w))
            for r, b, w in rows
        )
    return ChampionPool(entries=entries)


def save_pool(pool: ChampionPool, path: Path | None = None) -> None:
    path = path or DEFAULT_POOL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {
        str(cid): [[e.role, e.build_id, e.weight] for e in es]
        for cid, es in sorted(pool.entries.items())
    }
    path.write_text(json.dumps(serial, indent=2), encoding="utf-8")


def build_pool_from_priors(
    p1: Iterable[tuple[tuple[int, str, str], tuple[float, int]]],
    build_labels: list[str],
    min_matchups: int,
) -> ChampionPool:
    """Construct a pool from priors.p1 keys (champion, role, build) -> (wr, matchups).

    `build_id` is the index into `build_labels` (the predictor's canonical
    build-string ordering). Entries below `min_matchups` are dropped.
    Weight = matchup count, so the sampler will favour role/build
    combinations that we have the most evidence for.
    """
    label_to_id = {label: i for i, label in enumerate(build_labels)}
    grouped: dict[int, list[PoolEntry]] = defaultdict(list)
    for (champion_id, role, build_str), (_wr, matchups) in p1:
        if matchups < min_matchups:
            continue
        build_id = label_to_id.get(build_str)
        if build_id is None:
            continue
        grouped[champion_id].append(
            PoolEntry(role=role, build_id=build_id, weight=float(matchups))
        )
    entries = {cid: tuple(es) for cid, es in grouped.items()}
    return ChampionPool(entries=entries)


def _generate_cli(min_matchups: int, out: Path) -> None:
    from app.ml.predictor import load_predictor
    from app.ml.priors import load_priors

    priors = load_priors()
    predictor = load_predictor()
    pool = build_pool_from_priors(
        priors.p1.items(),
        build_labels=predictor.build_labels,
        min_matchups=min_matchups,
    )
    save_pool(pool, out)
    print(f"Wrote {len(pool.entries)} champion pools to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="Build pool file from priors.p1")
    gen.add_argument("--min-matchups", type=int, default=50)
    gen.add_argument("--out", type=Path, default=DEFAULT_POOL_PATH)
    args = parser.parse_args()
    if args.cmd == "generate":
        _generate_cli(args.min_matchups, args.out)
