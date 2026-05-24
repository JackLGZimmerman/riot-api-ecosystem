"""In-memory prior tables keyed on (championid, teamposition, build) tuples.

Runtime cache for the RL predictor only. The ML dataset build path
(build_dataset.py) reads these priors directly through ClickHouse
dictionaries (synergy_1vx_dict, matchup_1v1_dict, synergy_2vx_dict) so it
never has to materialise them in Python.

Loaded once at predictor start-up; provides vectorized lookups returning
(win_rate, matchups) for individual players, cross-team 1v1 matchups, and
same-team 2vx pair synergies. Lookups return the (left, right) canonical
entry; call sites translate to blue-perspective values when needed.

Why not call the ClickHouse dictionaries per inference? Each predictor call
would round-trip the HTTP interface (~ms), dwarfing the in-CH dictGet cost
(<μs). For the RL hot path, a local hash-table lookup (~100 ns) is the right
data structure even though the same priors live in CH as dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from database.clickhouse.client import get_client

DEFAULT_WIN_RATE = 0.5
DEFAULT_MATCHUPS = 0


@dataclass(frozen=True)
class PriorTables:
    p1: dict[tuple[int, str, str], tuple[float, int]]
    m1v1: dict[
        tuple[int, str, str, int, str, str],
        tuple[float, int],
    ]
    s2vx: dict[
        tuple[int, str, str, int, str, str],
        tuple[float, int],
    ]

    def lookup_player(
        self, tuples: Iterable[tuple[int, str, str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        items = list(tuples)
        wr = np.empty(len(items), dtype=np.float64)
        cnt = np.empty(len(items), dtype=np.float64)
        get = self.p1.get
        default = (DEFAULT_WIN_RATE, DEFAULT_MATCHUPS)
        for i, key in enumerate(items):
            wr[i], cnt[i] = get(key, default)
        return wr, cnt

    def lookup_1v1_blue(
        self,
        blue_tuples: list[tuple[int, str, str]],
        red_tuples: list[tuple[int, str, str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return blue-perspective win rate and matchup count for 25 (b, r) pairs."""
        n_pairs = len(blue_tuples) * len(red_tuples)
        wr = np.empty(n_pairs, dtype=np.float64)
        cnt = np.empty(n_pairs, dtype=np.float64)
        get = self.m1v1.get
        default = (DEFAULT_WIN_RATE, DEFAULT_MATCHUPS)
        idx = 0
        for b in blue_tuples:
            for r in red_tuples:
                if b <= r:
                    left_wr, mt = get((*b, *r), default)
                    wr[idx] = left_wr
                else:
                    left_wr, mt = get((*r, *b), default)
                    wr[idx] = 1.0 - left_wr
                cnt[idx] = mt
                idx += 1
        return wr, cnt

    def lookup_2vx_team(
        self, team_tuples: list[tuple[int, str, str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return synergy win rate and matchup count for C(5, 2) = 10 pairs."""
        pairs = (
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4),
            (3, 4),
        )
        wr = np.empty(len(pairs), dtype=np.float64)
        cnt = np.empty(len(pairs), dtype=np.float64)
        get = self.s2vx.get
        default = (DEFAULT_WIN_RATE, DEFAULT_MATCHUPS)
        for i, (a, b) in enumerate(pairs):
            ta, tb = team_tuples[a], team_tuples[b]
            key = (*ta, *tb) if ta <= tb else (*tb, *ta)
            wr[i], cnt[i] = get(key, default)
        return wr, cnt


def load_priors() -> PriorTables:
    client = get_client()

    p1_rows = client.query(
        """
        SELECT championid, teamposition, build, win_rate, matchups
        FROM game_data_filtered.synergy_1vx
        WHERE split = 'train'
        """
    ).result_rows
    p1 = {
        (int(c), str(p), str(b)): (float(w), int(m))
        for c, p, b, w, m in p1_rows
    }

    m1v1_rows = client.query(
        """
        SELECT
            left_championid, left_teamposition, left_build,
            right_championid, right_teamposition, right_build,
            left_win_rate, matchups
        FROM game_data_filtered.matchup_1v1
        WHERE split = 'train'
        """
    ).result_rows
    m1v1 = {
        (int(lc), str(lp), str(lb), int(rc), str(rp), str(rb)): (float(w), int(m))
        for lc, lp, lb, rc, rp, rb, w, m in m1v1_rows
    }

    s2vx_rows = client.query(
        """
        SELECT
            championid_1, teamposition_1, build_1,
            championid_2, teamposition_2, build_2,
            win_rate, matchups
        FROM game_data_filtered.synergy_2vx
        WHERE split = 'train'
        """
    ).result_rows
    s2vx = {
        (int(c1), str(p1k), str(b1), int(c2), str(p2k), str(b2)): (float(w), int(m))
        for c1, p1k, b1, c2, p2k, b2, w, m in s2vx_rows
    }

    return PriorTables(p1=p1, m1v1=m1v1, s2vx=s2vx)


__all__ = ["DEFAULT_MATCHUPS", "DEFAULT_WIN_RATE", "PriorTables", "load_priors"]
