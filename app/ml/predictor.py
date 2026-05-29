# pyright: reportPrivateImportUsage=false

"""Bridge between the structured win-rate model and the DraftEnv Predictor protocol."""

from __future__ import annotations

import numpy as np
import torch

from app.ml.cache_layout import N_MATCHUPS_1V1, N_PLAYERS_PER_GAME, N_SYNERGIES_2VX
from app.ml.config import POSITIONS, TrainConfig
from app.ml.priors import DEFAULT_MATCHUPS, DEFAULT_WIN_RATE, PriorTables, load_priors
from app.ml.structured_model import (
    DeltaBaselineMode,
    StructuredWinModel,
    build_structured_input_arrays,
    load_structured_model,
    resolve_device,
    structured_tensors,
)


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
    """Satisfies app.rl.reward.Predictor using the production structured model."""

    def __init__(
        self,
        model: StructuredWinModel,
        priors: PriorTables,
        *,
        prior_strength: float,
        delta_baseline_mode: DeltaBaselineMode,
        device: str,
    ) -> None:
        self._model = model.to(device).eval()
        self._priors = priors
        self._prior_strength = prior_strength
        self._delta_baseline_mode: DeltaBaselineMode = delta_baseline_mode
        self._device = device
        self.build_labels: list[str] = sorted({b for _, _, b in priors.p1})
        self.champion_ids: tuple[int, ...] = tuple(sorted({c for c, _, _ in priors.p1}))

    def _arrays_for_game(
        self,
        blue_tuples: list[tuple[int, str, str]],
        red_tuples: list[tuple[int, str, str]],
    ) -> dict[str, np.ndarray]:
        win_rate = np.full(
            (1, N_PLAYERS_PER_GAME),
            DEFAULT_WIN_RATE,
            dtype=np.float64,
        )
        p1_wr, p1_cnt = self._priors.lookup_player(blue_tuples + red_tuples)
        win_rate[0] = p1_wr

        matchup_1v1 = np.full(
            (1, N_MATCHUPS_1V1),
            DEFAULT_WIN_RATE,
            dtype=np.float64,
        )
        m1v1_wr, m1v1_cnt = self._priors.lookup_1v1_blue(blue_tuples, red_tuples)
        matchup_1v1[0] = m1v1_wr

        synergy_2vx = np.full(
            (1, N_SYNERGIES_2VX),
            DEFAULT_WIN_RATE,
            dtype=np.float64,
        )
        s2vx_cnt = np.full(
            (1, N_SYNERGIES_2VX),
            DEFAULT_MATCHUPS,
            dtype=np.float64,
        )
        s2vx_blue_wr, s2vx_blue_cnt = self._priors.lookup_2vx_team(blue_tuples)
        s2vx_red_wr, s2vx_red_cnt = self._priors.lookup_2vx_team(red_tuples)
        synergy_2vx[0, :10] = s2vx_blue_wr
        synergy_2vx[0, 10:] = s2vx_red_wr
        s2vx_cnt[0, :10] = s2vx_blue_cnt
        s2vx_cnt[0, 10:] = s2vx_red_cnt

        return {
            "win_rate": win_rate,
            "matchup_1v1": matchup_1v1,
            "synergy_2vx": synergy_2vx,
            "p1_cnt": p1_cnt.reshape(1, -1),
            "m1v1_cnt": m1v1_cnt.reshape(1, -1),
            "s2vx_cnt": s2vx_cnt,
        }

    def __call__(
        self,
        blue_team: list[int],
        red_team: list[int],
        blue_roles: dict[int, str],
        red_roles: dict[int, str],
        blue_builds: dict[int, int],
        red_builds: dict[int, int],
    ) -> float:
        del blue_team, red_team
        blue_tuples = _team_tuples(blue_roles, blue_builds, self.build_labels)
        red_tuples = _team_tuples(red_roles, red_builds, self.build_labels)
        raw = self._arrays_for_game(blue_tuples, red_tuples)
        arrays = build_structured_input_arrays(
            win_rate=raw["win_rate"],
            matchup_1v1=raw["matchup_1v1"],
            synergy_2vx=raw["synergy_2vx"],
            p1_cnt=raw["p1_cnt"],
            m1v1_cnt=raw["m1v1_cnt"],
            s2vx_cnt=raw["s2vx_cnt"],
            prior_strength=self._prior_strength,
            delta_baseline_mode=self._delta_baseline_mode,
        )
        with torch.no_grad():
            logits = self._model(**structured_tensors(arrays, device=self._device))[
                "final_logit"
            ]
            return float(torch.sigmoid(logits)[0].detach().cpu().item())


def load_predictor(cfg: TrainConfig | None = None) -> WinRatePredictor:
    cfg = cfg or TrainConfig()
    device = resolve_device(cfg.device)
    model, model_config, prior_strength = load_structured_model(
        cfg.model_path,
        device=device,
    )
    return WinRatePredictor(
        model,
        load_priors(),
        prior_strength=prior_strength,
        delta_baseline_mode=model_config.delta_baseline_mode,
        device=device,
    )
