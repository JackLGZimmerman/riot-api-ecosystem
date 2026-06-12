"""Per-champion eligibility pool of (role, build) assignments.

Each entry says: "champion C is plausibly played as role R with build B,
with this much prior weight." The pool drives `make_pool_sampler` in
`app.rl.reward`, which enumerates valid role assignments and the top-K
build combinations per terminal team — replacing the old pick-order
role assignment, which silently assumed pick order == role.

File format (JSON):

    {
        "__meta__": {"catalog_version": "abc123def456"},
        "champion_id": [["ROLE", build_id, weight], ...],
        ...
    }

Generate from the train-only build catalog:

    python -m app.rl.pool generate [--include-supported] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config.settings import PROJECT_ROOT
from app.ml.build_catalog import BuildCatalog

_META_KEY = "__meta__"

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
    # Version stamp of the build catalog the pool was generated from; empty
    # for hand-written or legacy pool files.
    catalog_version: str = field(default="", compare=False)

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
    meta = raw.pop(_META_KEY, {})
    entries: dict[int, tuple[PoolEntry, ...]] = {}
    for cid_str, rows in raw.items():
        entries[int(cid_str)] = tuple(
            PoolEntry(role=str(r), build_id=int(b), weight=float(w))
            for r, b, w in rows
        )
    return ChampionPool(
        entries=entries, catalog_version=str(meta.get("catalog_version", ""))
    )


def save_pool(pool: ChampionPool, path: Path | None = None) -> None:
    path = path or DEFAULT_POOL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    serial: dict[str, object] = {
        _META_KEY: {"catalog_version": pool.catalog_version}
    }
    serial.update(
        {
            str(cid): [[e.role, e.build_id, e.weight] for e in es]
            for cid, es in sorted(pool.entries.items())
        }
    )
    path.write_text(json.dumps(serial, indent=2), encoding="utf-8")


def build_pool_from_catalog(
    catalog: BuildCatalog, *, core_only: bool = True
) -> ChampionPool:
    """RL candidate pool from the train-only build catalog.

    Entry weights approximate `P(role, build | champion)`: the catalog's
    smoothed per-cell `P(build | champion, role)` scaled by the cell's share
    of the champion's total observed games, so the sampler's role-permutation
    ranking still sees role plausibility (build_id is the checkpoint vocab
    index, the same ordering the catalog validated against). With `core_only`,
    profiles that miss the rl_core support gates are dropped, so the RL
    surface never proposes a thinly supported (role, build).
    """
    gates = catalog.gates
    # Pre-pruning per-cell totals; retained counts cover retained_mass.
    cell_totals = {
        key: sum(vector.support_counts) / vector.retained_mass
        for key, vector in catalog.vectors.items()
    }
    champ_totals: dict[int, float] = defaultdict(float)
    for (champion_id, _role), total in cell_totals.items():
        champ_totals[champion_id] += total
    grouped: dict[int, list[PoolEntry]] = defaultdict(list)
    for (champion_id, role), vector in sorted(catalog.vectors.items()):
        total = cell_totals[(champion_id, role)]
        role_share = total / champ_totals[champion_id]
        for build_id, prob, count in zip(
            vector.hgnn_build_ids, vector.probabilities, vector.support_counts
        ):
            share = count / total if total else 0.0
            if core_only and (
                count < gates.rl_core_min_count or share < gates.rl_core_min_share
            ):
                continue
            grouped[champion_id].append(
                PoolEntry(
                    role=role, build_id=build_id, weight=float(prob * role_share)
                )
            )
    return ChampionPool(
        entries={cid: tuple(es) for cid, es in grouped.items()},
        catalog_version=catalog.version,
    )


def _generate_cli(core_only: bool, out: Path) -> None:
    from app.ml.build_catalog import build_catalog_from_priors
    from app.ml.config import DEFAULT_PRODUCTION_MODEL_PATH
    from app.ml.hgnn_model import load_hgnn_model

    _, config, _ = load_hgnn_model(DEFAULT_PRODUCTION_MODEL_PATH)
    catalog = build_catalog_from_priors(tuple(config.build_vocab))
    pool = build_pool_from_catalog(catalog, core_only=core_only)
    save_pool(pool, out)
    print(
        f"Wrote {len(pool.entries)} champion pools "
        f"(catalog {catalog.version}) to {out}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="Build pool file from the build catalog")
    gen.add_argument(
        "--include-supported",
        action="store_true",
        help="Also include non-core (thinly supported) profiles",
    )
    gen.add_argument("--out", type=Path, default=DEFAULT_POOL_PATH)
    args = parser.parse_args()
    if args.cmd == "generate":
        _generate_cli(not args.include_supported, args.out)
