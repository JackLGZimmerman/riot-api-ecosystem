from __future__ import annotations

from pathlib import Path

from app.ml.build_catalog import build_catalog
from app.rl.pool import build_pool_from_catalog, load_pool, save_pool

VOCAB = ("a", "b", "c")


def _catalog():
    return build_catalog(
        {
            (1, "TOP", "a"): (0.52, 80),  # core
            (1, "TOP", "b"): (0.48, 20),  # supported only (count < rl_core 50)
            (2, "JUNGLE", "a"): (0.51, 900),
        },
        VOCAB,
    )


def test_build_pool_from_catalog_core_gating() -> None:
    catalog = _catalog()

    core = build_pool_from_catalog(catalog, core_only=True)
    assert core.catalog_version == catalog.version
    assert core.builds_for(1, "TOP") == ((0, catalog.prior_vector(1, "TOP").probabilities[0]),)

    full = build_pool_from_catalog(catalog, core_only=False)
    assert [b for b, _ in full.builds_for(1, "TOP")] == [0, 1]


def test_pool_round_trip_preserves_entries_and_meta(tmp_path: Path) -> None:
    pool = build_pool_from_catalog(_catalog(), core_only=False)
    path = tmp_path / "pool.json"
    save_pool(pool, path)
    loaded = load_pool(path)
    assert loaded.entries == pool.entries
    assert loaded.catalog_version == pool.catalog_version
