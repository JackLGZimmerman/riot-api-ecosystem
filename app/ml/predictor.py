# pyright: reportPrivateImportUsage=false

"""Bridge between the structured win-rate model and the DraftEnv Predictor protocol."""

from __future__ import annotations

import numpy as np
import torch

from app.ml.cache_layout import N_MATCHUPS_1V1, N_PLAYERS_PER_GAME, N_SYNERGIES_2VX
from app.ml.config import POSITIONS, DatasetConfig, TrainConfig
from app.ml.priors import DEFAULT_MATCHUPS, DEFAULT_WIN_RATE, PriorTables, load_priors
from app.ml.structured_model import (
    DeltaBaselineMode,
    TEAM_PAIRS,
    StructuredWinModel,
    build_structured_input_arrays,
    load_structured_model,
    resolve_device,
    structured_tensors,
)
from app.core.utils.smoothing import (
    cascade_dynamic_smoothed_rate,
    dynamic_smoothed_rate,
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
        smoothing_prior_strength: float,
        amplification_threshold: float,
        smoothing_mode: str,
        prior_confidence_matchups: float,
        delta_baseline_mode: DeltaBaselineMode,
        device: str,
    ) -> None:
        self._model = model.to(device).eval()
        self._priors = priors
        self._prior_strength = prior_strength
        self._smoothing_prior_strength = smoothing_prior_strength
        self._amplification_threshold = amplification_threshold
        self._smoothing_mode = smoothing_mode
        self._prior_confidence_matchups = prior_confidence_matchups
        self._delta_baseline_mode: DeltaBaselineMode = delta_baseline_mode
        self._device = device
        self.build_labels: list[str] = sorted({b for _, _, b in priors.p1})
        self.champion_ids: tuple[int, ...] = tuple(sorted({c for c, _, _ in priors.p1}))

    def _arrays_for_game(
        self,
        blue_tuples: list[tuple[int, str, str]],
        red_tuples: list[tuple[int, str, str]],
    ) -> dict[str, np.ndarray]:
        s = self._smoothing_prior_strength
        t = self._amplification_threshold

        def smooth_rate(
            win_rate: np.ndarray,
            sample_count: np.ndarray,
            prior: float | np.ndarray = DEFAULT_WIN_RATE,
        ) -> np.ndarray:
            if self._smoothing_mode == "additive":
                return dynamic_smoothed_rate(
                    win_rate,
                    sample_count,
                    prior_mean=prior,
                    base_strength=s,
                    amplification_threshold=t,
                )
            if self._smoothing_mode == "cascade":
                return cascade_dynamic_smoothed_rate(
                    win_rate,
                    sample_count,
                    prior_mean=prior,
                    base_strength=s,
                    amplification_threshold=t,
                    confidence_threshold=self._prior_confidence_matchups,
                )
            raise ValueError(f"Unsupported smoothing_mode: {self._smoothing_mode!r}")

        win_rate = np.full((1, N_PLAYERS_PER_GAME), DEFAULT_WIN_RATE, dtype=np.float64)
        p1_wr, p1_cnt = self._priors.lookup_player(blue_tuples + red_tuples)
        p1_wr = smooth_rate(p1_wr, p1_cnt)
        win_rate[0] = p1_wr

        # Composite priors: each interaction shrinks toward a blend of its two sides'
        # solo win rates instead of a flat 0.5 (mirrors build_dataset._composite_interaction_priors).
        blue_wr, red_wr = p1_wr[:5], p1_wr[5:]
        prior_1v1 = (0.5 + (blue_wr[:, None] - red_wr[None, :]) / 2.0).reshape(-1)
        blue_pair_priors = np.array([(blue_wr[i] + blue_wr[j]) / 2.0 for i, j in TEAM_PAIRS])
        red_pair_priors = np.array([(red_wr[i] + red_wr[j]) / 2.0 for i, j in TEAM_PAIRS])

        matchup_1v1 = np.full((1, N_MATCHUPS_1V1), DEFAULT_WIN_RATE, dtype=np.float64)
        m1v1_wr, m1v1_cnt = self._priors.lookup_1v1_blue(blue_tuples, red_tuples)
        m1v1_wr = smooth_rate(m1v1_wr, m1v1_cnt, prior_1v1)
        matchup_1v1[0] = m1v1_wr

        synergy_2vx = np.full((1, N_SYNERGIES_2VX), DEFAULT_WIN_RATE, dtype=np.float64)
        s2vx_cnt = np.full((1, N_SYNERGIES_2VX), DEFAULT_MATCHUPS, dtype=np.float64)
        s2vx_blue_wr, s2vx_blue_cnt = self._priors.lookup_2vx_team(blue_tuples)
        s2vx_red_wr, s2vx_red_cnt = self._priors.lookup_2vx_team(red_tuples)
        s2vx_blue_wr = smooth_rate(s2vx_blue_wr, s2vx_blue_cnt, blue_pair_priors)
        s2vx_red_wr = smooth_rate(s2vx_red_wr, s2vx_red_cnt, red_pair_priors)
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
            confidence_strength=self._prior_strength,
            delta_baseline_mode=self._delta_baseline_mode,
        )
        with torch.no_grad():
            logits = self._model(**structured_tensors(arrays, device=self._device))[
                "final_logit"
            ]
            return float(torch.sigmoid(logits)[0].detach().cpu().item())


def load_predictor(
    cfg: TrainConfig | None = None,
    dataset_cfg: DatasetConfig | None = None,
) -> WinRatePredictor:
    cfg = cfg or TrainConfig()
    dataset_cfg = dataset_cfg or DatasetConfig()
    device = resolve_device(cfg.device)
    model, model_config, prior_strength = load_structured_model(
        cfg.model_path,
        device=device,
    )
    return WinRatePredictor(
        model,
        load_priors(),
        prior_strength=prior_strength,
        smoothing_prior_strength=dataset_cfg.smoothing_prior_strength,
        amplification_threshold=dataset_cfg.amplification_threshold,
        smoothing_mode=dataset_cfg.smoothing_mode,
        prior_confidence_matchups=dataset_cfg.prior_confidence_matchups,
        delta_baseline_mode=model_config.delta_baseline_mode,
        device=device,
    )
