# pyright: reportPrivateImportUsage=false

"""Bridge between the HGNN win-rate model and the DraftEnv Predictor protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.ml.cache_layout import (
    CACHE_META_FILE,
    N_PLAYERS_PER_GAME,
    N_SYNERGIES_2VX,
)
from app.ml.config import POSITIONS, DatasetConfig, TrainConfig
from app.ml.priors import DEFAULT_MATCHUPS, DEFAULT_WIN_RATE, PriorTables, load_priors
from app.ml.hgnn_model import (
    TEAM_PAIRS,
    HGNNWinModel,
    build_hgnn_inputs,
    load_hgnn_model,
    resolve_device,
)
from app.core.utils.smoothing import (
    cascade_dynamic_smoothed_rate,
    dynamic_smoothed_rate,
    nested_shrunk_rate,
)


def _team_tuples(
    roles: dict[int, str],
    builds: dict[int, int],
    build_labels: list[str],
    *,
    force_build_label: str | None = None,
) -> list[tuple[int, str, str]]:
    role_to_champ = {role: champ for champ, role in roles.items()}
    tuples: list[tuple[int, str, str]] = []
    for pos in POSITIONS:
        champ = role_to_champ.get(pos)
        if champ is None:
            tuples.append((-1, pos, ""))
            continue
        if force_build_label is None:
            build_id = builds.get(champ, 0)
            build_str = build_labels[build_id] if build_id < len(build_labels) else ""
        else:
            build_str = force_build_label
        tuples.append((int(champ), pos, build_str))
    return tuples


def _interaction_pooling_from_cache_meta(
    cache_dir: Path,
    *,
    fallback_strength: float,
) -> tuple[bool, dict[str, list[float]]]:
    """Return runtime interaction-pooling mode from the training cache metadata.

    The model is trained from the cache, so runtime must reuse the exact
    per-level EB strengths recorded in `cache_meta.json`. Missing or incomplete
    metadata means legacy cache: use single-level smoothing.
    """
    fallback = {"m1v1": [fallback_strength], "s2vx": [fallback_strength]}
    meta_path = cache_dir / CACHE_META_FILE
    if not meta_path.exists():
        return False, fallback

    smoothing = json.loads(meta_path.read_text()).get("smoothing", {})
    stored = smoothing.get("interaction_level_strengths")
    if not bool(smoothing.get("interaction_nested_pooling", False)) or not isinstance(stored, dict):
        return False, fallback

    candidate = {
        "m1v1": list(stored.get("m1v1", [])),
        "s2vx": list(stored.get("s2vx", [])),
    }
    if len(candidate["m1v1"]) != 3 or len(candidate["s2vx"]) != 3:
        return False, fallback
    return True, candidate


class WinRatePredictor:
    """Satisfies app.rl.reward.Predictor using the production HGNN model."""

    def __init__(
        self,
        model: HGNNWinModel,
        priors: PriorTables,
        *,
        prior_strength: float,
        smoothing_prior_strength: float,
        amplification_threshold: float,
        smoothing_mode: str,
        prior_confidence_matchups: float,
        use_final_build_labels: bool,
        draft_unknown_build_label: str,
        nested_pooling: bool,
        level_strengths: dict[str, list[float]],
        device: str,
    ) -> None:
        self._model = model.to(device).eval()
        self._priors = priors
        self._prior_strength = prior_strength
        self._smoothing_prior_strength = smoothing_prior_strength
        self._amplification_threshold = amplification_threshold
        self._smoothing_mode = smoothing_mode
        self._prior_confidence_matchups = prior_confidence_matchups
        self._use_final_build_labels = use_final_build_labels
        self._draft_unknown_build_label = draft_unknown_build_label
        self._nested_pooling = nested_pooling
        self._level_strengths = level_strengths
        self._device = device
        # Identity-embedding mapping from the trained artifact's config.
        self._n_champions = model.config.n_champions
        self._n_builds = model.config.n_builds
        self._build_to_idx = {label: idx for idx, label in enumerate(model.config.build_vocab)}
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

        # Composite priors: the terminal floor each interaction shrinks toward — a
        # blend of its two sides' solo win rates instead of a flat 0.5 (mirrors
        # build_dataset._composite_interaction_priors).
        blue_wr, red_wr = p1_wr[:5], p1_wr[5:]
        prior_1v1 = (0.5 + (blue_wr[:, None] - red_wr[None, :]) / 2.0).reshape(-1)
        blue_pair_priors = np.array([(blue_wr[i] + blue_wr[j]) / 2.0 for i, j in TEAM_PAIRS])
        red_pair_priors = np.array([(red_wr[i] + red_wr[j]) / 2.0 for i, j in TEAM_PAIRS])

        # 1v1 levels (blue-perspective, 25): build -> no-build -> champion pair.
        m1v1_l0_wr, m1v1_l0_cnt = self._priors.lookup_1v1_blue(blue_tuples, red_tuples)
        m1v1_nb_wr, m1v1_nb_cnt = self._priors.lookup_1v1_blue_nobuild(blue_tuples, red_tuples)
        m1v1_ch_wr, m1v1_ch_cnt = self._priors.lookup_1v1_blue_champ(blue_tuples, red_tuples)
        m1v1_wr, _ = self._pool_interaction(
            [m1v1_l0_wr, m1v1_nb_wr, m1v1_ch_wr],
            [m1v1_l0_cnt, m1v1_nb_cnt, m1v1_ch_cnt],
            self._level_strengths["m1v1"],
            prior_1v1,
        )

        # 2vx levels per team (own-team, 10 each): build -> no-build -> champion pair.
        synergy_2vx = np.full((1, N_SYNERGIES_2VX), DEFAULT_WIN_RATE, dtype=np.float64)
        s2vx_cnt = np.full((1, N_SYNERGIES_2VX), DEFAULT_MATCHUPS, dtype=np.float64)
        for offset, team, floor in (
            (0, blue_tuples, blue_pair_priors),
            (10, red_tuples, red_pair_priors),
        ):
            l0 = self._priors.lookup_2vx_team(team)
            nb = self._priors.lookup_2vx_team_nobuild(team)
            ch = self._priors.lookup_2vx_team_champ(team)
            wr, _ = self._pool_interaction(
                [l0[0], nb[0], ch[0]], [l0[1], nb[1], ch[1]],
                self._level_strengths["s2vx"], floor,
            )
            synergy_2vx[0, offset : offset + 10] = wr
            s2vx_cnt[0, offset : offset + 10] = l0[1]

        return {
            "win_rate": win_rate,
            "matchup_1v1": m1v1_wr.reshape(1, -1),
            "synergy_2vx": synergy_2vx,
            "p1_cnt": p1_cnt.reshape(1, -1),
            "m1v1_cnt": m1v1_l0_cnt.reshape(1, -1),
            "s2vx_cnt": s2vx_cnt,
        }

    def _pool_interaction(
        self,
        rates: list[np.ndarray],
        counts: list[np.ndarray],
        strengths: list[float],
        floor_prior: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nested EB pooling (shared math with build_dataset) or legacy fallback."""
        if self._nested_pooling:
            return nested_shrunk_rate(
                rates, counts, strengths=strengths, floor_prior=floor_prior,
                amplification_threshold=self._amplification_threshold,
            )
        s = self._smoothing_prior_strength
        t = self._amplification_threshold
        if self._smoothing_mode == "additive":
            smoothed = dynamic_smoothed_rate(
                rates[0], counts[0], prior_mean=floor_prior, base_strength=s,
                amplification_threshold=t,
            )
        else:
            smoothed = cascade_dynamic_smoothed_rate(
                rates[0], counts[0], prior_mean=floor_prior, base_strength=s,
                amplification_threshold=t, confidence_threshold=self._prior_confidence_matchups,
            )
        return smoothed, counts[0]

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
        forced_build = None if self._use_final_build_labels else self._draft_unknown_build_label
        blue_tuples = _team_tuples(
            blue_roles,
            blue_builds,
            self.build_labels,
            force_build_label=forced_build,
        )
        red_tuples = _team_tuples(
            red_roles,
            red_builds,
            self.build_labels,
            force_build_label=forced_build,
        )
        raw = self._arrays_for_game(blue_tuples, red_tuples)
        tuples = blue_tuples + red_tuples
        champion_id = np.array(
            [[c if 0 <= c < self._n_champions else self._n_champions for c, _, _ in tuples]],
            dtype=np.int64,
        )
        build_id = np.array(
            [[self._build_to_idx.get(b, self._n_builds) for _, _, b in tuples]],
            dtype=np.int64,
        )
        inputs = build_hgnn_inputs(
            champion_id=champion_id,
            build_id=build_id,
            win_rate=raw["win_rate"],
            matchup_1v1=raw["matchup_1v1"],
            synergy_2vx=raw["synergy_2vx"],
            p1_cnt=raw["p1_cnt"],
            m1v1_cnt=raw["m1v1_cnt"],
            s2vx_cnt=raw["s2vx_cnt"],
            strength=self._prior_strength,
            device=self._device,
        )
        with torch.no_grad():
            logits = self._model(**inputs)["final_logit"]
            return float(torch.sigmoid(logits)[0].detach().cpu().item())


def load_predictor(
    cfg: TrainConfig | None = None,
    dataset_cfg: DatasetConfig | None = None,
) -> WinRatePredictor:
    cfg = cfg or TrainConfig()
    dataset_cfg = dataset_cfg or DatasetConfig()
    device = resolve_device(cfg.device)
    model, _, prior_strength = load_hgnn_model(cfg.model_path, device=device)

    nested_pooling, level_strengths = _interaction_pooling_from_cache_meta(
        dataset_cfg.cache_dir,
        fallback_strength=dataset_cfg.smoothing_prior_strength,
    )

    return WinRatePredictor(
        model,
        load_priors(),
        prior_strength=prior_strength,
        smoothing_prior_strength=dataset_cfg.smoothing_prior_strength,
        amplification_threshold=dataset_cfg.amplification_threshold,
        smoothing_mode=dataset_cfg.smoothing_mode,
        prior_confidence_matchups=dataset_cfg.prior_confidence_matchups,
        use_final_build_labels=dataset_cfg.use_final_build_labels,
        draft_unknown_build_label=dataset_cfg.draft_unknown_build_label,
        nested_pooling=nested_pooling,
        level_strengths=level_strengths,
        device=device,
    )
