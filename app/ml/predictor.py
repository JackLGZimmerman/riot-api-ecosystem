# pyright: reportPrivateImportUsage=false

"""Bridge between the HGNN win-rate model and the DraftEnv Predictor protocol."""

from __future__ import annotations

import numpy as np
import torch

from app.ml.config import POSITIONS, DatasetConfig, TrainConfig
from app.ml.encoder_sidecar import EncoderSidecarLookup
from app.ml.priors import DEFAULT_WIN_RATE, PriorTables, load_priors
from app.ml.semantic_context_lookup import (
    SemanticContextRawLookup,
    load_semantic_context_raw_lookup,
)
from app.ml.semantic_group_features import (
    build_semantic_group_features,
    static_hp_range_lookups,
)
from app.ml.hgnn_model import (
    HGNNWinModel,
    build_hgnn_inputs,
    load_hgnn_model,
    resolve_device,
)
from app.core.utils.smoothing import smooth_rate_by_mode


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


def _model_requires_encoder_sidecar(model: HGNNWinModel) -> bool:
    config = model.config
    return bool(
        config.use_identity_static_sidecar
        or config.use_identity_full_game_sidecar
        or config.use_identity_temporal_sidecar
        or config.use_identity_semantic_context_head
        or config.use_learned_semantic_moe
    )


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
        encoder_sidecar: EncoderSidecarLookup | None,
        semantic_context_lookup: SemanticContextRawLookup | None,
        device: str,
    ) -> None:
        self._model = model.to(device).eval()
        self._priors = priors
        self._semantic_context_lookup = semantic_context_lookup
        self._prior_strength = prior_strength
        self._smoothing_prior_strength = smoothing_prior_strength
        self._amplification_threshold = amplification_threshold
        self._smoothing_mode = smoothing_mode
        self._prior_confidence_matchups = prior_confidence_matchups
        self._use_final_build_labels = use_final_build_labels
        self._draft_unknown_build_label = draft_unknown_build_label
        self._encoder_sidecar = encoder_sidecar
        self._device = device
        # Identity-embedding mapping from the trained artifact's config.
        self._n_champions = model.config.n_champions
        self._n_builds = model.config.n_builds
        self._build_to_idx = {
            label: idx for idx, label in enumerate(model.config.build_vocab)
        }
        self.build_labels: list[str] = sorted({b for _, _, b in priors.p1})
        self.champion_ids: tuple[int, ...] = tuple(sorted({c for c, _, _ in priors.p1}))
        self._semantic_hp_lookup: np.ndarray | None
        self._semantic_range_lookup: np.ndarray | None
        if model.config.use_semantic_group_features:
            if self._semantic_context_lookup is None:
                raise ValueError(
                    "HGNN checkpoint requires semantic context lookup for "
                    "semantic_group_features"
                )
            self._semantic_hp_lookup, self._semantic_range_lookup = (
                static_hp_range_lookups()
            )
        else:
            self._semantic_hp_lookup = None
            self._semantic_range_lookup = None

    def _arrays_for_game(
        self,
        blue_tuples: list[tuple[int, str, str]],
        red_tuples: list[tuple[int, str, str]],
    ) -> dict[str, np.ndarray]:
        p1_raw, p1_cnt = self._priors.lookup_player(blue_tuples + red_tuples)
        p1_raw = p1_raw.reshape(1, -1)
        p1_cnt = p1_cnt.reshape(1, -1)
        return {
            "win_rate": smooth_rate_by_mode(
                p1_raw,
                p1_cnt,
                prior_mean=DEFAULT_WIN_RATE,
                prior_strength=self._smoothing_prior_strength,
                amplification_threshold=self._amplification_threshold,
                smoothing_mode=self._smoothing_mode,
                confidence_threshold=self._prior_confidence_matchups,
            ),
            "p1_cnt": p1_cnt,
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
        forced_build = (
            None if self._use_final_build_labels else self._draft_unknown_build_label
        )
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
            [
                [
                    c if 0 <= c < self._n_champions else self._n_champions
                    for c, _, _ in tuples
                ]
            ],
            dtype=np.int64,
        )
        build_id = np.array(
            [[self._build_to_idx.get(b, self._n_builds) for _, _, b in tuples]],
            dtype=np.int64,
        )
        semantic_group_features = None
        if self._model.config.use_semantic_group_features:
            if self._semantic_context_lookup is None:
                raise ValueError("semantic context lookup is required")
            context_raw = self._semantic_context_lookup.lookup(tuples).reshape(
                1, 10, -1
            )
            semantic_group_features = build_semantic_group_features(
                context_raw=context_raw,
                champion_id=champion_id,
                build_id=build_id,
                build_vocab=self._model.config.build_vocab,
                hp_lookup=self._semantic_hp_lookup,
                range_lookup=self._semantic_range_lookup,
            )
        sidecar_blocks: dict[str, np.ndarray] | None = None
        sidecar_support: np.ndarray | None = None
        if self._encoder_sidecar is not None:
            sidecar_blocks, sidecar_support = self._encoder_sidecar.lookup_game_blocks(
                tuples
            )
        inputs = build_hgnn_inputs(
            champion_id=champion_id,
            build_id=build_id,
            win_rate=raw["win_rate"],
            p1_cnt=raw["p1_cnt"],
            strength=self._prior_strength,
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
            semantic_group_features=semantic_group_features,
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
    requires_semantic_context = bool(model.config.use_semantic_group_features)
    requires_encoder_sidecar = _model_requires_encoder_sidecar(model)
    if requires_encoder_sidecar and dataset_cfg.encoder_sidecar_path is None:
        raise ValueError(
            "HGNN checkpoint requires identity encoder sidecars, but "
            "DatasetConfig.encoder_sidecar_path is not set"
        )

    encoder_sidecar = (
        EncoderSidecarLookup.load(dataset_cfg.encoder_sidecar_path)
        if requires_encoder_sidecar and dataset_cfg.encoder_sidecar_path is not None
        else None
    )
    semantic_context_lookup = (
        load_semantic_context_raw_lookup() if requires_semantic_context else None
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
        encoder_sidecar=encoder_sidecar,
        semantic_context_lookup=semantic_context_lookup,
        device=device,
    )
