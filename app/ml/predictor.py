# pyright: reportPrivateImportUsage=false

"""Bridge between the HGNN win-rate model and the DraftEnv Predictor protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.ml.cache_layout import (
    CACHE_META_FILE,
    N_SYNERGIES_2VX,
)
from app.ml.config import POSITIONS, DatasetConfig, TrainConfig
from app.ml.encoder_sidecar import EncoderSidecarLookup
from app.ml.priors import DEFAULT_MATCHUPS, DEFAULT_WIN_RATE, PriorTables, load_priors
from app.ml.hgnn_model import (
    TEAM_PAIRS,
    HGNNWinModel,
    build_hgnn_inputs,
    load_hgnn_model,
    resolve_device,
)
from app.core.utils.smoothing import (
    smooth_rate_by_mode,
    smooth_ml_prior_features,
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
) -> tuple[bool, dict[str, list[float]], tuple[str, ...]]:
    """Return runtime interaction-pooling mode from the training cache metadata.

    The model is trained from the cache, so runtime must reuse the exact
    per-level EB strengths recorded in `cache_meta.json`. Missing or incomplete
    metadata means legacy cache: use single-level smoothing.
    """
    fallback = {"m1v1": [fallback_strength], "s2vx": [fallback_strength]}
    old_s2vx_ladder = ("build", "nobuild", "champion")
    meta_path = cache_dir / CACHE_META_FILE
    if not meta_path.exists():
        return False, fallback, old_s2vx_ladder

    smoothing = json.loads(meta_path.read_text()).get("smoothing", {})
    stored = smoothing.get("interaction_level_strengths")
    if not bool(smoothing.get("interaction_nested_pooling", False)) or not isinstance(stored, dict):
        return False, fallback, old_s2vx_ladder

    candidate = {
        "m1v1": list(stored.get("m1v1", [])),
        "s2vx": list(stored.get("s2vx", [])),
    }
    if len(candidate["m1v1"]) != 3 or len(candidate["s2vx"]) != 3:
        return False, fallback, old_s2vx_ladder
    s2vx_ladder = tuple(smoothing.get("s2vx_ladder", old_s2vx_ladder))
    return True, candidate, s2vx_ladder


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
        s2vx_ladder: tuple[str, ...],
        encoder_sidecar: EncoderSidecarLookup | None,
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
        self._s2vx_ladder = s2vx_ladder
        self._encoder_sidecar = encoder_sidecar
        self._use_relationship_integrations = bool(model.config.use_relationship_integrations)
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
        p1_raw, p1_cnt = self._priors.lookup_player(blue_tuples + red_tuples)
        raw: dict[str, np.ndarray] = {
            "p1_raw": p1_raw.reshape(1, -1),
            "p1_cnt": p1_cnt.reshape(1, -1),
        }
        if not self._use_relationship_integrations:
            return {
                "win_rate": smooth_rate_by_mode(
                    raw["p1_raw"],
                    raw["p1_cnt"],
                    prior_mean=DEFAULT_WIN_RATE,
                    prior_strength=self._smoothing_prior_strength,
                    amplification_threshold=self._amplification_threshold,
                    smoothing_mode=self._smoothing_mode,
                    confidence_threshold=self._prior_confidence_matchups,
                ),
                "p1_cnt": raw["p1_cnt"],
            }

        # 1v1 levels (blue-perspective, 25): build -> no-build -> champion pair.
        m1v1_l0_wr, m1v1_l0_cnt = self._priors.lookup_1v1_blue(blue_tuples, red_tuples)
        m1v1_nb_wr, m1v1_nb_cnt = self._priors.lookup_1v1_blue_nobuild(blue_tuples, red_tuples)
        m1v1_ch_wr, m1v1_ch_cnt = self._priors.lookup_1v1_blue_champ(blue_tuples, red_tuples)
        raw.update(
            {
                "m1v1_raw": m1v1_l0_wr.reshape(1, -1),
                "m1v1_cnt": m1v1_l0_cnt.reshape(1, -1),
                "m1v1_nb_raw": m1v1_nb_wr.reshape(1, -1),
                "m1v1_nb_cnt": m1v1_nb_cnt.reshape(1, -1),
                "m1v1_champ_raw": m1v1_ch_wr.reshape(1, -1),
                "m1v1_champ_cnt": m1v1_ch_cnt.reshape(1, -1),
            }
        )

        # 2vx levels per team (own-team, 10 each). New caches use
        # build -> build-group sibling -> no-build -> neutral floor. Legacy cache
        # metadata keeps build -> no-build -> champion pair -> 1vx-average floor.
        s2vx_raw = np.full((1, N_SYNERGIES_2VX), DEFAULT_WIN_RATE, dtype=np.float64)
        s2vx_cnt = np.full((1, N_SYNERGIES_2VX), DEFAULT_MATCHUPS, dtype=np.float64)
        s2vx_bg_raw = s2vx_raw.copy()
        s2vx_bg_cnt = s2vx_cnt.copy()
        s2vx_nb_raw = s2vx_raw.copy()
        s2vx_nb_cnt = s2vx_cnt.copy()
        s2vx_champ_raw = s2vx_raw.copy()
        s2vx_champ_cnt = s2vx_cnt.copy()
        raw.update(
            {
                "s2vx_raw": s2vx_raw,
                "s2vx_cnt": s2vx_cnt,
                "s2vx_bg_raw": s2vx_bg_raw,
                "s2vx_bg_cnt": s2vx_bg_cnt,
                "s2vx_nb_raw": s2vx_nb_raw,
                "s2vx_nb_cnt": s2vx_nb_cnt,
                "s2vx_champ_raw": s2vx_champ_raw,
                "s2vx_champ_cnt": s2vx_champ_cnt,
            }
        )

        for offset, team in ((0, blue_tuples), (10, red_tuples)):
            for raw_key, cnt_key, lookup in (
                ("s2vx_raw", "s2vx_cnt", self._priors.lookup_2vx_team(team)),
                (
                    "s2vx_bg_raw",
                    "s2vx_bg_cnt",
                    self._priors.lookup_2vx_team_build_group(team),
                ),
                ("s2vx_nb_raw", "s2vx_nb_cnt", self._priors.lookup_2vx_team_nobuild(team)),
                (
                    "s2vx_champ_raw",
                    "s2vx_champ_cnt",
                    self._priors.lookup_2vx_team_champ(team),
                ),
            ):
                raw[raw_key][0, offset : offset + 10] = lookup[0]
                raw[cnt_key][0, offset : offset + 10] = lookup[1]

        s2vx_level_map = {
            "build": ("s2vx_raw", "s2vx_cnt"),
            "build_group": ("s2vx_bg_raw", "s2vx_bg_cnt"),
            "nobuild": ("s2vx_nb_raw", "s2vx_nb_cnt"),
            "champion": ("s2vx_champ_raw", "s2vx_champ_cnt"),
        }
        smoothed = smooth_ml_prior_features(
            raw,
            prior_mean=DEFAULT_WIN_RATE,
            prior_strength=self._smoothing_prior_strength,
            amplification_threshold=self._amplification_threshold,
            smoothing_mode=self._smoothing_mode,
            prior_confidence_matchups=self._prior_confidence_matchups,
            per_side_fallback=True,
            nested_pooling=self._nested_pooling,
            level_strengths=self._level_strengths,
            m1v1_levels=(
                ("m1v1_raw", "m1v1_cnt"),
                ("m1v1_nb_raw", "m1v1_nb_cnt"),
                ("m1v1_champ_raw", "m1v1_champ_cnt"),
            ),
            s2vx_levels=tuple(s2vx_level_map[name] for name in self._s2vx_ladder),
            team_pairs=TEAM_PAIRS,
            s2vx_ladder=self._s2vx_ladder,
        )

        return {
            "win_rate": smoothed["win_rate"],
            "matchup_1v1": smoothed["matchup_1v1"],
            "synergy_2vx": smoothed["synergy_2vx"],
            "p1_cnt": raw["p1_cnt"],
            "m1v1_cnt": raw["m1v1_cnt"],
            "s2vx_cnt": raw["s2vx_cnt"],
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
        sidecar_blocks: dict[str, np.ndarray] | None = None
        sidecar_support: np.ndarray | None = None
        if self._encoder_sidecar is not None:
            sidecar_blocks, sidecar_support = self._encoder_sidecar.lookup_game_blocks(tuples)
        inputs = build_hgnn_inputs(
            champion_id=champion_id,
            build_id=build_id,
            win_rate=raw["win_rate"],
            p1_cnt=raw["p1_cnt"],
            strength=self._prior_strength,
            matchup_1v1=raw.get("matchup_1v1"),
            synergy_2vx=raw.get("synergy_2vx"),
            m1v1_cnt=raw.get("m1v1_cnt"),
            s2vx_cnt=raw.get("s2vx_cnt"),
            identity_static_sidecar=(
                None if sidecar_blocks is None else sidecar_blocks["static"]
            ),
            identity_full_game_sidecar=(
                None if sidecar_blocks is None else sidecar_blocks["full_game"]
            ),
            identity_temporal_sidecar=(
                None if sidecar_blocks is None else sidecar_blocks["temporal"]
            ),
            identity_encoder_support=sidecar_support,
            include_relationship_features=self._use_relationship_integrations,
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

    nested_pooling, level_strengths, s2vx_ladder = _interaction_pooling_from_cache_meta(
        dataset_cfg.cache_dir,
        fallback_strength=dataset_cfg.smoothing_prior_strength,
    )
    encoder_sidecar = (
        EncoderSidecarLookup.load(dataset_cfg.encoder_sidecar_path)
        if dataset_cfg.encoder_sidecar_path is not None
        else None
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
        s2vx_ladder=s2vx_ladder,
        encoder_sidecar=encoder_sidecar,
        device=device,
    )
