from __future__ import annotations

import numpy as np

from app.ml.build_catalog import build_catalog
from app.ml.config import POSITIONS
from app.ml.dataset import SplitData
from app.ml.pregame import (
    HypothesisTables,
    apply_modal_build_split,
    modal_build_table,
)

VOCAB = ("alpha", "beta", "gamma")
N_CHAMPIONS = 3


def _catalog():
    # Champion 0: beta dominates in every role; champion 1: alpha dominates.
    # Champion 2 has no rows and must resolve through the role fallback.
    p1 = {}
    for role in POSITIONS:
        p1[(0, role, "beta")] = (0.52, 500)
        p1[(0, role, "alpha")] = (0.5, 100)
        p1[(1, role, "alpha")] = (0.48, 300)
        p1[(1, role, "gamma")] = (0.51, 60)
    return build_catalog(p1, VOCAB)


def _tables():
    rng = np.random.default_rng(0)
    shape = (N_CHAMPIONS + 1, 5, len(VOCAB))
    return HypothesisTables(
        win_rate=rng.uniform(0.4, 0.6, size=shape).astype(np.float32),
        p1_cnt=rng.integers(0, 500, size=shape).astype(np.float32),
        context=np.zeros(shape + (4,), dtype=np.float32),
    )


def test_modal_build_table_picks_top_prior_and_falls_back() -> None:
    table = modal_build_table(_catalog(), N_CHAMPIONS)

    assert table.shape == (N_CHAMPIONS + 1, 5)
    assert (table[0] == VOCAB.index("beta")).all()
    assert (table[1] == VOCAB.index("alpha")).all()
    # Champion 2 and the reserve row fall back to the role-level prior, which
    # is dominated by champion 0's beta counts.
    assert (table[2] == VOCAB.index("beta")).all()
    assert (table[N_CHAMPIONS] == VOCAB.index("beta")).all()


def test_apply_modal_build_split_is_deterministic_per_champ_role() -> None:
    catalog, tables = _catalog(), _tables()
    rng = np.random.default_rng(1)
    n = 32
    champion_id = rng.integers(0, N_CHAMPIONS, size=(n, 10)).astype(np.int64)
    champion_id[0, 0] = N_CHAMPIONS + 7  # out of range -> reserve row
    split = SplitData(
        win_rate=rng.uniform(size=(n, 10)).astype(np.float32),
        p1_cnt=np.ones((n, 10), dtype=np.float32),
        blue_win=rng.integers(0, 2, size=n).astype(np.float32),
        champion_id=champion_id,
        build_id=rng.integers(0, len(VOCAB), size=(n, 10)).astype(np.int64),
    )

    out = apply_modal_build_split(
        split, catalog, tables, build_vocab=VOCAB, needs_semantic=False
    )

    table = modal_build_table(catalog, N_CHAMPIONS)
    slot_roles = np.arange(10) % 5
    champ = np.where(
        (champion_id < 0) | (champion_id >= N_CHAMPIONS), N_CHAMPIONS, champion_id
    )
    np.testing.assert_array_equal(out.build_id, table[champ, slot_roles])
    np.testing.assert_array_equal(
        out.win_rate, tables.win_rate[champ, slot_roles, out.build_id]
    )
    np.testing.assert_array_equal(
        out.p1_cnt, tables.p1_cnt[champ, slot_roles, out.build_id]
    )
    # Observed labels are gone: identical (champ, slot) keys share one build.
    assert (out.build_id[champion_id == 1] == VOCAB.index("alpha")).all()
    # Untouched arrays pass through.
    np.testing.assert_array_equal(out.blue_win, split.blue_win)
