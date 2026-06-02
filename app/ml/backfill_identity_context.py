"""Offline backfill of the identity_context cache arrays (no ClickHouse).

The production path writes ``identity_context`` / ``identity_context_support``
inside ``build_dataset.build`` from streamed games. This helper regenerates the
same two arrays for an *existing* cache directly from its already-built
``champion_id`` / ``build_id`` arrays + ``build_vocab`` and the
``identity_context_embedding.npz`` lookup, then patches ``cache_meta.json`` to
``npy-memmap-v25``. It exists so the context head can be trained/evaluated
without re-streaming the corpus from ClickHouse.

The reconstruction matches ``build_dataset._player_tuples`` exactly (slot order =
role order, raw champion ids, build-vocab indices), so the per-player context is
identical to what a full rebuild would produce.

Run with:
    python -m app.ml.backfill_identity_context
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from app.classification.embeddings.config import IDENTITY_CONTEXT_INTERP_DIM
from app.classification.embeddings.runtime import IdentityContextLookup
from app.core.logging.logger import setup_logging_config
from app.core.utils.common import POSITIONS
from app.ml.cache_layout import (
    CACHE_FORMAT,
    CACHE_META_FILE,
    IDENTITY_CONTEXT_DIM,
    array_paths,
)
from app.ml.config import DatasetConfig

logger = logging.getLogger(__name__)


def _decode(key: int, n_vocab: int) -> tuple[int, int]:
    base = n_vocab + 2
    return key // base, key % base


def backfill(cache_dir: Path) -> None:
    meta_path = cache_dir / CACHE_META_FILE
    meta = json.loads(meta_path.read_text())
    build_vocab: list[str] = list(meta["identity"]["build_vocab"])
    n_games = int(meta["n_games"])

    paths = array_paths(cache_dir)
    champ = np.load(paths["champion_id"], mmap_mode="r")[:n_games].astype(np.int64)
    build = np.load(paths["build_id"], mmap_mode="r")[:n_games].astype(np.int64)
    if champ.shape != (n_games, 10):
        raise ValueError(f"unexpected champion_id shape {champ.shape}")

    lookup = IdentityContextLookup.load()
    if not lookup.values:
        raise FileNotFoundError(
            "identity_context_embedding.npz is missing/empty; run "
            "`python -m app.classification.embeddings.context` first."
        )
    dim = lookup.dim
    raw_dim = lookup.raw_dim
    if dim != IDENTITY_CONTEXT_DIM:
        logger.warning("lookup dim %d != cache_layout IDENTITY_CONTEXT_DIM %d", dim, IDENTITY_CONTEXT_DIM)
    if not lookup.raw:
        raise FileNotFoundError(
            "identity_context_embedding.npz has no raw_embeddings block; rebuild it "
            "with `python -m app.classification.embeddings.context` first."
        )
    zero = np.zeros(dim, dtype=np.float32)
    zero_raw = np.zeros(raw_dim, dtype=np.float32)
    base = len(build_vocab) + 2

    context = np.lib.format.open_memmap(
        paths["identity_context"], mode="w+", dtype=np.float32, shape=(n_games, 10, dim)
    )
    support = np.lib.format.open_memmap(
        paths["identity_context_support"], mode="w+", dtype=np.float32, shape=(n_games, 10)
    )
    context_raw = np.lib.format.open_memmap(
        paths["identity_context_raw"], mode="w+", dtype=np.float32, shape=(n_games, 10, raw_dim)
    )

    for slot in range(10):
        role = POSITIONS[slot % 5]
        key = champ[:, slot] * base + build[:, slot]
        uniq, inv = np.unique(key, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)  # numpy 2.0 briefly returned 2-D here
        vecs = np.zeros((uniq.size, dim), dtype=np.float32)
        raws = np.zeros((uniq.size, raw_dim), dtype=np.float32)
        sups = np.zeros(uniq.size, dtype=np.float32)
        for i, k in enumerate(uniq.tolist()):
            c, b = _decode(int(k), len(build_vocab))
            label = build_vocab[b] if 0 <= b < len(build_vocab) else ""
            tup = (int(c), role, label)
            vecs[i] = lookup.values.get(tup, zero)
            raws[i] = lookup.raw.get(tup, zero_raw)
            sups[i] = lookup.support.get(tup, 0.0)
        context[:, slot, :] = vecs[inv]
        context_raw[:, slot, :] = raws[inv]
        support[:, slot] = sups[inv]
        logger.info("slot %d (%s): %d unique identities", slot, role, uniq.size)

    context.flush()
    context_raw.flush()
    support.flush()

    meta["format"] = CACHE_FORMAT
    classification = dict(meta["identity"].get("classification", {}))
    classification["identity_context_dim"] = dim
    classification["context_interpretable_dim"] = IDENTITY_CONTEXT_INTERP_DIM
    classification["identity_context_raw_dim"] = raw_dim
    meta["identity"]["classification"] = classification
    meta_path.write_text(json.dumps(meta, indent=2))

    covered = float((support > 0).mean())
    logger.info(
        "Backfilled identity_context: games=%d dim=%d raw_dim=%d coverage=%.4f format=%s",
        n_games,
        dim,
        raw_dim,
        covered,
        CACHE_FORMAT,
    )


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DatasetConfig().cache_dir)
    args = parser.parse_args()
    backfill(args.cache_dir)


if __name__ == "__main__":
    main()
