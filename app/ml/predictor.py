"""Bridge between WinRateLinearModel and the DraftEnv Predictor protocol."""

from __future__ import annotations

import numpy as np

from app.ml.config import POSITIONS, TrainConfig
from app.ml.model import (
    N_MATCHUP_1V1,
    N_PLAYER_FEATURES,
    N_SYNERGY_2VX,
    WinRateLinearModel,
)
from app.ml.priors import DEFAULT_WIN_RATE, PriorTables, load_priors


def _team_tuples(
    roles: dict[int, str],
    builds: dict[int, int],
    build_labels: list[str],
) -> list[tuple[int, str, str]]:
    role_to_champ = {role: champ for champ, role in roles.items()}
    tuples: list[tuple[int, str, str]] = []
    for pos in POSITIONS:
        champ = role_to_champ.get(pos)
        if champ is None:
            tuples.append((-1, pos, ""))
            continue
        build_id = builds.get(champ, 0)
        build_str = build_labels[build_id] if build_id < len(build_labels) else ""
        tuples.append((int(champ), pos, build_str))
    return tuples


class WinRatePredictor:
    """Satisfies app.rl.reward.Predictor.

    Loads the three prior tables once at init and assembles the per-game
    feature arrays at call time. RL role names MUST match ML positions.
    """

    def __init__(self, model: WinRateLinearModel, priors: PriorTables) -> None:
        self._model = model
        self._priors = priors
        self.build_labels: list[str] = sorted({b for _, _, b in priors.p1})
        self.champion_ids: tuple[int, ...] = tuple(sorted({c for c, _, _ in priors.p1}))

    def __call__(
        self,
        blue_team: list[int],
        red_team: list[int],
        blue_roles: dict[int, str],
        red_roles: dict[int, str],
        blue_builds: dict[int, int],
        red_builds: dict[int, int],
    ) -> float:
        b = _team_tuples(blue_roles, blue_builds, self.build_labels)
        r = _team_tuples(red_roles, red_builds, self.build_labels)

        wr = np.full((1, N_PLAYER_FEATURES), DEFAULT_WIN_RATE, dtype=np.float64)
        player_wr, _ = self._priors.lookup_player(b + r)
        wr[0] = player_wr

        m1v1 = np.full((1, N_MATCHUP_1V1), DEFAULT_WIN_RATE, dtype=np.float64)
        m1v1_wr, _ = self._priors.lookup_1v1_blue(b, r)
        m1v1[0] = m1v1_wr

        s2vx = np.full((1, N_SYNERGY_2VX), DEFAULT_WIN_RATE, dtype=np.float64)
        s2vx_blue_wr, _ = self._priors.lookup_2vx_team(b)
        s2vx_red_wr, _ = self._priors.lookup_2vx_team(r)
        s2vx[0, :10] = s2vx_blue_wr
        s2vx[0, 10:] = s2vx_red_wr

        return float(self._model.predict(wr, m1v1, s2vx)[0])


def load_predictor(cfg: TrainConfig | None = None) -> WinRatePredictor:
    cfg = cfg or TrainConfig()
    return WinRatePredictor(WinRateLinearModel.load(cfg.model_path), load_priors())
