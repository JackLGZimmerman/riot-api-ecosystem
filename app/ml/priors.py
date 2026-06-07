"""In-memory solo prior table keyed on (championid, teamposition, build) tuples.

Runtime cache for the RL predictor only. The ML dataset build path
(build_dataset.py) reads the solo prior directly through the ClickHouse
synergy_1vx_dict dictionary so it never has to materialise it in Python.

Loaded once at predictor start-up; provides a vectorized lookup returning
(win_rate, matchups) for individual players.

Why not call the ClickHouse dictionary per inference? Each predictor call
would round-trip the HTTP interface (~ms), dwarfing the in-CH dictGet cost
(<μs). For the RL hot path, a local hash-table lookup (~100 ns) is the right
data structure even though the same prior lives in CH as a dictionary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from database.clickhouse.client import get_client

DEFAULT_WIN_RATE = 0.5
DEFAULT_MATCHUPS = 0


@dataclass(frozen=True)
class PriorTables:
    p1: dict[tuple[int, str, str], tuple[float, int]]

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


def load_priors() -> PriorTables:
    client = get_client()

    p1_rows = client.query(
        """
        SELECT championid, teamposition, build, win_rate, matchups
        FROM game_data_filtered.synergy_1vx
        WHERE split = 'train'
        """
    ).result_rows
    p1 = {(int(c), str(p), str(b)): (float(w), int(m)) for c, p, b, w, m in p1_rows}

    return PriorTables(p1=p1)


__all__ = ["DEFAULT_MATCHUPS", "DEFAULT_WIN_RATE", "PriorTables", "load_priors"]
