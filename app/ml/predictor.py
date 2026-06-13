# pyright: reportPrivateImportUsage=false

"""Bridge between the HGNN win-rate model and the DraftEnv Predictor protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.ml.build_catalog import (
    BUILD_SOURCE_PREGAME_MARGINAL,
    BuildCatalog,
    enumerate_joint_worlds,
    validate_accepted_build_source,
)
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
    expected_encoder_sidecar_dims,
    load_hgnn_model,
    model_requires_semantic_group_features,
    model_uses_encoder_sidecar,
    resolve_device,
    runtime_unsupported_inputs,
)
from app.core.utils.smoothing import smooth_rate_by_mode


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
        build_id = builds[champ]
        if not 0 <= build_id < len(build_labels):
            raise ValueError(
                f"build id {build_id} for champion {champ} is outside the model "
                f"build vocab (size {len(build_labels)})"
            )
        tuples.append((int(champ), pos, build_labels[build_id]))
    return tuples


def _validate_runtime_feature_contract(model: HGNNWinModel) -> None:
    missing = runtime_unsupported_inputs(model.config)
    if not missing:
        return
    raise ValueError(
        "HGNN checkpoint requires runtime inputs that app.rl.reward.Predictor "
        "does not supply: "
        + ", ".join(missing)
        + ". Use a predictor protocol that passes those feature tensors, or "
        "serve a checkpoint trained without those heads."
    )


def _validate_sidecar_dims(model: HGNNWinModel, sidecar: EncoderSidecarLookup) -> None:
    expected = expected_encoder_sidecar_dims(model.config)
    actual = sidecar.dims.as_dict()
    mismatches = [
        f"{name}: checkpoint={expected_dim} sidecar={actual.get(name)}"
        for name, expected_dim in expected.items()
        if int(expected_dim) != int(actual.get(name, -1))
    ]
    if mismatches:
        raise ValueError(
            "Encoder sidecar artifact dimensions do not match the HGNN checkpoint: "
            + ", ".join(mismatches)
        )


def _validate_team_assignment(
    side: str,
    team: list[int],
    roles: dict[int, str],
    builds: dict[int, int] | None,
) -> None:
    champions = set(team)
    if len(champions) != len(team):
        raise ValueError(f"{side} team contains duplicate champion ids")
    mappings: list[tuple[str, set[int]]] = [("roles", set(roles))]
    if builds is not None:
        mappings.append(("builds", set(builds)))
    for label, keys in mappings:
        if keys != champions:
            missing = sorted(champions.difference(keys))
            extra = sorted(keys.difference(champions))
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(
                f"{side} {label} must match the supplied team champions"
                + (": " + ", ".join(details) if details else "")
            )


@dataclass(frozen=True)
class MarginalPrediction:
    """Pregame marginal win probability over train-supported build worlds."""

    probability: float
    retained_joint_mass: float
    n_worlds: int
    low_confidence: bool
    fallback_sources: tuple[str, ...]  # per slot, blue 0-4 / red 5-9
    build_source: str = BUILD_SOURCE_PREGAME_MARGINAL


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
        encoder_sidecar: EncoderSidecarLookup | None,
        semantic_context_lookup: SemanticContextRawLookup | None,
        device: str,
    ) -> None:
        _validate_runtime_feature_contract(model)
        self._model = model.to(device).eval()
        self._priors = priors
        self._semantic_context_lookup = semantic_context_lookup
        self._prior_strength = prior_strength
        self._smoothing_prior_strength = smoothing_prior_strength
        self._amplification_threshold = amplification_threshold
        self._smoothing_mode = smoothing_mode
        self._prior_confidence_matchups = prior_confidence_matchups
        self._encoder_sidecar = encoder_sidecar
        self._device = device
        # Identity-embedding mapping from the trained artifact's config.
        self._n_champions = model.config.n_champions
        self._n_builds = model.config.n_builds
        # The checkpoint's build_vocab is the single canonical ordering; the
        # prior table must agree exactly, never be re-sorted into its own order.
        vocab = [str(label) for label in model.config.build_vocab]
        if not vocab:
            raise ValueError("HGNN checkpoint config has an empty build_vocab")
        prior_labels = sorted({b for _, _, b in priors.p1})
        if prior_labels != sorted(vocab):
            raise ValueError(
                "train prior build labels do not match the checkpoint build_vocab: "
                f"priors={prior_labels} vocab={sorted(vocab)}"
            )
        self._build_to_idx = {label: idx for idx, label in enumerate(vocab)}
        self.build_labels: list[str] = vocab
        self.champion_ids: tuple[int, ...] = tuple(sorted({c for c, _, _ in priors.p1}))
        self._semantic_hp_lookup: np.ndarray | None
        self._semantic_range_lookup: np.ndarray | None
        if model_requires_semantic_group_features(model.config):
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

    def _batch_inputs(
        self,
        games: list[list[tuple[int, str, str]]],
    ) -> dict[str, torch.Tensor]:
        """Assemble model inputs for a batch of (champion, role, build) games.

        Shared by the single-game protocol call and the marginal world batch,
        so both paths score identical tensors for identical assignments.
        """
        flat = [identity for game in games for identity in game]
        n = len(games)
        p1_raw, p1_cnt = self._priors.lookup_player(flat)
        p1_raw = p1_raw.reshape(n, 10)
        p1_cnt = p1_cnt.reshape(n, 10)
        win_rate = smooth_rate_by_mode(
            p1_raw,
            p1_cnt,
            prior_mean=DEFAULT_WIN_RATE,
            prior_strength=self._smoothing_prior_strength,
            amplification_threshold=self._amplification_threshold,
            smoothing_mode=self._smoothing_mode,
            confidence_threshold=self._prior_confidence_matchups,
        )
        champion_id = np.array(
            [
                [c if 0 <= c < self._n_champions else self._n_champions for c, _, _ in game]
                for game in games
            ],
            dtype=np.int64,
        )
        build_id = np.array(
            [
                [self._build_to_idx.get(b, self._n_builds) for _, _, b in game]
                for game in games
            ],
            dtype=np.int64,
        )
        semantic_group_features = None
        if model_requires_semantic_group_features(self._model.config):
            if self._semantic_context_lookup is None:
                raise ValueError("semantic context lookup is required")
            context_raw = self._semantic_context_lookup.lookup(flat).reshape(n, 10, -1)
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
            blocks, support = self._encoder_sidecar.lookup_blocks(flat)
            sidecar_blocks = {
                name: values.reshape(n, 10, values.shape[1])
                for name, values in blocks.items()
            }
            sidecar_support = support.reshape(n, 10)
        return build_hgnn_inputs(
            champion_id=champion_id,
            build_id=build_id,
            win_rate=win_rate,
            p1_cnt=p1_cnt,
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

    def _forward_probabilities(self, games: list[list[tuple[int, str, str]]]) -> np.ndarray:
        inputs = self._batch_inputs(games)
        with torch.no_grad():
            logits = self._model(**inputs)["final_logit"]
            return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

    def __call__(
        self,
        blue_team: list[int],
        red_team: list[int],
        blue_roles: dict[int, str],
        red_roles: dict[int, str],
        blue_builds: dict[int, int],
        red_builds: dict[int, int],
    ) -> float:
        _validate_team_assignment("blue", blue_team, blue_roles, blue_builds)
        _validate_team_assignment("red", red_team, red_roles, red_builds)
        blue_tuples = _team_tuples(blue_roles, blue_builds, self.build_labels)
        red_tuples = _team_tuples(red_roles, red_builds, self.build_labels)
        return float(self._forward_probabilities([blue_tuples + red_tuples])[0])

    def predict_batch(
        self,
        games: list[
            tuple[
                list[int],
                list[int],
                dict[int, str],
                dict[int, str],
                dict[int, int],
                dict[int, int],
            ]
        ],
    ) -> np.ndarray:
        """Score many (blue, red, roles, builds) games in one forward pass.

        Same per-game contract as ``__call__``; lets ``resolve_rewards`` build
        the whole role/build config matrix and evaluate it in a single batch.
        """
        rows: list[list[tuple[int, str, str]]] = []
        for blue_team, red_team, blue_roles, red_roles, blue_builds, red_builds in games:
            _validate_team_assignment("blue", blue_team, blue_roles, blue_builds)
            _validate_team_assignment("red", red_team, red_roles, red_builds)
            rows.append(
                _team_tuples(blue_roles, blue_builds, self.build_labels)
                + _team_tuples(red_roles, red_builds, self.build_labels)
            )
        return self._forward_probabilities(rows)

    def predict_marginal(
        self,
        blue_team: list[int],
        red_team: list[int],
        blue_roles: dict[int, str],
        red_roles: dict[int, str],
        *,
        catalog: BuildCatalog,
        k_slot: int = 3,
        max_worlds: int = 512,
        early_stop_mass: float = 0.90,
        mass_floor: float = 0.35,
    ) -> MarginalPrediction:
        """Pregame win probability marginalised over the train-supported catalog.

        Scores every retained joint build world in one batched forward pass and
        averages output probabilities with the unnormalised joint prior weights
        divided by retained mass. Never reads an observed build label.
        """
        validate_accepted_build_source(BUILD_SOURCE_PREGAME_MARGINAL)
        _validate_team_assignment("blue", blue_team, blue_roles, builds=None)
        _validate_team_assignment("red", red_team, red_roles, builds=None)
        catalog.validate_model_vocab(tuple(self._model.config.build_vocab))
        slots: list[tuple[int, str]] = []
        for roles in (blue_roles, red_roles):
            role_to_champ = {role: champ for champ, role in roles.items()}
            for pos in POSITIONS:
                champ = role_to_champ.get(pos)
                if champ is None:
                    raise ValueError(f"missing role assignment for position {pos}")
                slots.append((int(champ), pos))
        vectors = [catalog.prior_vector(champ, pos) for champ, pos in slots]
        selections, weights, retained_mass = enumerate_joint_worlds(
            [np.asarray(vector.probabilities) for vector in vectors],
            k_slot=k_slot,
            max_worlds=max_worlds,
            early_stop_mass=early_stop_mass,
        )
        games = [
            [
                (
                    slots[s][0],
                    slots[s][1],
                    catalog.build_vocab[vectors[s].hgnn_build_ids[selections[w, s]]],
                )
                for s in range(10)
            ]
            for w in range(selections.shape[0])
        ]
        probabilities = self._forward_probabilities(games)
        marginal = float(np.dot(probabilities, weights) / weights.sum())
        return MarginalPrediction(
            probability=marginal,
            retained_joint_mass=retained_mass,
            n_worlds=int(selections.shape[0]),
            low_confidence=retained_mass < mass_floor,
            fallback_sources=tuple(vector.fallback_source for vector in vectors),
        )


def load_predictor(
    cfg: TrainConfig | None = None,
    dataset_cfg: DatasetConfig | None = None,
) -> WinRatePredictor:
    cfg = cfg or TrainConfig()
    dataset_cfg = dataset_cfg or DatasetConfig()
    device = resolve_device(cfg.device)
    model, _, prior_strength = load_hgnn_model(cfg.model_path, device=device)
    _validate_runtime_feature_contract(model)
    requires_semantic_context = model_requires_semantic_group_features(model.config)
    requires_encoder_sidecar = model_uses_encoder_sidecar(model.config)
    if requires_encoder_sidecar and dataset_cfg.encoder_sidecar_path is None:
        raise ValueError(
            "HGNN checkpoint requires identity encoder sidecars, but "
            "DatasetConfig.encoder_sidecar_path is not set"
        )

    encoder_sidecar = None
    if requires_encoder_sidecar and dataset_cfg.encoder_sidecar_path is not None:
        encoder_sidecar = EncoderSidecarLookup.load(dataset_cfg.encoder_sidecar_path)
        _validate_sidecar_dims(model, encoder_sidecar)
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
        encoder_sidecar=encoder_sidecar,
        semantic_context_lookup=semantic_context_lookup,
        device=device,
    )
