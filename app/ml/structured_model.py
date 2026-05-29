# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn

POSITIONS: tuple[str, ...] = ("top", "jungle", "middle", "bottom", "utility")
TEAM_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)
ROLE_PAIR_TYPES: tuple[str, ...] = tuple(
    f"{POSITIONS[i]}+{POSITIONS[j]}" for i, j in TEAM_PAIRS
)
ROLE_PAIR_TYPE_TO_ID = {name: idx for idx, name in enumerate(ROLE_PAIR_TYPES)}
MATCHUP_BLUE_INDEX: tuple[int, ...] = tuple(i for i in range(5) for _ in range(5))
MATCHUP_RED_INDEX: tuple[int, ...] = tuple(j for _ in range(5) for j in range(5))

DeltaBaselineMode = Literal["logit", "probability"]

BASE_FEATURE_DIM = 15
SYNERGY_OBJECT_DIM = 6
MATCHUP_OBJECT_DIM = 6
SYNERGY_CONFIDENCE_INDEX = 4
MATCHUP_CONFIDENCE_INDEX = 4
PAIR_EMBEDDING_DIM = 16
MATCHUP_EMBEDDING_DIM = 16
TEAM_POOL_MULTIPLIER = 4
TEAM_REPRESENTATION_DIM = PAIR_EMBEDDING_DIM * TEAM_POOL_MULTIPLIER
CONFIDENCE_SUMMARY_DIM = 7
LOGIT_EPS = 1e-6


ObjectFeatureMode = Literal["full", "raw"]
# Object layout: [joint, a, b, expected, confidence, delta]. "raw" drops the
# expected baseline (3) and the joint-minus-expected delta (5): those two
# isolate the low-support in-sample leakage that wrecks generalization.
RAW_OBJECT_INDICES: tuple[int, ...] = (0, 1, 2, 4)
POOL_OPS_DEFAULT: tuple[str, ...] = ("mean", "max", "min", "weighted")


@dataclass(frozen=True)
class StructuredModelConfig:
    use_synergy: bool = True
    use_matchup: bool = True
    use_cross: bool = True
    delta_baseline_mode: DeltaBaselineMode = "logit"
    base_hidden: tuple[int, ...] = (32, 16)
    pair_hidden: tuple[int, ...] = (32,)
    matchup_hidden: tuple[int, ...] = (32,)
    head_hidden: tuple[int, ...] = (32,)
    role_embedding_dim: int = 4
    matchup_slot_embedding_dim: int = 4
    dropout: float = 0.05
    # Leakage-robustness knobs (see app/ml/documentation/README.md).
    object_feature_mode: ObjectFeatureMode = "full"
    confidence_gate: bool = False
    pooling_ops: tuple[str, ...] = POOL_OPS_DEFAULT

    def object_dim(self) -> int:
        return len(RAW_OBJECT_INDICES) if self.object_feature_mode == "raw" else SYNERGY_OBJECT_DIM

    def pool_mult(self) -> int:
        return len(self.pooling_ops)


@dataclass(frozen=True)
class StructuredInputArrays:
    base_features: np.ndarray
    synergy_objects: np.ndarray
    matchup_objects: np.ndarray
    confidence_summaries: np.ndarray
    role_pair_type_ids: np.ndarray


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_delta_mode(mode: str) -> DeltaBaselineMode:
    if mode == "logit":
        return "logit"
    if mode == "probability":
        return "probability"
    raise ValueError(f"Unknown delta baseline mode {mode!r}")


def role_pair_type_ids() -> np.ndarray:
    return np.arange(len(TEAM_PAIRS), dtype=np.int64)


def logit_prob(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities.astype(np.float64, copy=False), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p)).astype(np.float32)


def confidence_from_counts(counts: np.ndarray, *, prior_strength: float) -> np.ndarray:
    count_arr = np.maximum(counts.astype(np.float64, copy=False), 0.0)
    return (count_arr / (count_arr + float(prior_strength))).astype(np.float32)


def build_base_identity_features(win_rate: np.ndarray) -> np.ndarray:
    if win_rate.ndim != 2 or win_rate.shape[1] != 10:
        raise ValueError(f"win_rate shape {win_rate.shape}")
    logits = logit_prob(win_rate)
    blue = logits[:, :5]
    red = logits[:, 5:]
    return np.column_stack([blue, red, blue - red]).astype(np.float32)


def build_synergy_objects(
    win_rate: np.ndarray,
    synergy_2vx: np.ndarray,
    s2vx_cnt: np.ndarray,
    *,
    prior_strength: float,
    delta_baseline_mode: DeltaBaselineMode = "logit",
) -> np.ndarray:
    if win_rate.ndim != 2 or win_rate.shape[1] != 10:
        raise ValueError(f"win_rate shape {win_rate.shape}")
    if synergy_2vx.ndim != 2 or synergy_2vx.shape[1] != 20:
        raise ValueError(f"synergy_2vx shape {synergy_2vx.shape}")
    if s2vx_cnt.shape != synergy_2vx.shape:
        raise ValueError(f"s2vx_cnt shape {s2vx_cnt.shape}")

    n_rows = win_rate.shape[0]
    objects = np.empty((n_rows, 2, len(TEAM_PAIRS), SYNERGY_OBJECT_DIM), dtype=np.float32)
    for side_idx, (player_offset, pair_offset) in enumerate(((0, 0), (5, 10))):
        rates = win_rate[:, player_offset : player_offset + 5]
        logits = logit_prob(rates)
        for pair_idx, (a_idx, b_idx) in enumerate(TEAM_PAIRS):
            feature_idx = pair_offset + pair_idx
            joint_logit = logit_prob(synergy_2vx[:, feature_idx])
            if delta_baseline_mode == "logit":
                expected_logit = 0.5 * (logits[:, a_idx] + logits[:, b_idx])
            else:
                expected_logit = logit_prob((rates[:, a_idx] + rates[:, b_idx]) / 2.0)
            objects[:, side_idx, pair_idx, 0] = joint_logit
            objects[:, side_idx, pair_idx, 1] = logits[:, a_idx]
            objects[:, side_idx, pair_idx, 2] = logits[:, b_idx]
            objects[:, side_idx, pair_idx, 3] = expected_logit
            objects[:, side_idx, pair_idx, 4] = confidence_from_counts(
                s2vx_cnt[:, feature_idx],
                prior_strength=prior_strength,
            )
            objects[:, side_idx, pair_idx, 5] = joint_logit - expected_logit
    return objects


def build_matchup_objects(
    win_rate: np.ndarray,
    matchup_1v1: np.ndarray,
    m1v1_cnt: np.ndarray,
    *,
    prior_strength: float,
    delta_baseline_mode: DeltaBaselineMode = "logit",
) -> np.ndarray:
    if win_rate.ndim != 2 or win_rate.shape[1] != 10:
        raise ValueError(f"win_rate shape {win_rate.shape}")
    if matchup_1v1.ndim != 2 or matchup_1v1.shape[1] != 25:
        raise ValueError(f"matchup_1v1 shape {matchup_1v1.shape}")
    if m1v1_cnt.shape != matchup_1v1.shape:
        raise ValueError(f"m1v1_cnt shape {m1v1_cnt.shape}")

    blue_rates = win_rate[:, :5]
    red_rates = win_rate[:, 5:]
    blue_logits = logit_prob(blue_rates)
    red_logits = logit_prob(red_rates)
    matchup_logits = logit_prob(matchup_1v1)
    objects = np.empty((win_rate.shape[0], 25, MATCHUP_OBJECT_DIM), dtype=np.float32)
    for blue_idx in range(5):
        for red_idx in range(5):
            feature_idx = blue_idx * 5 + red_idx
            if delta_baseline_mode == "logit":
                expected_logit = blue_logits[:, blue_idx] - red_logits[:, red_idx]
            else:
                expected_rate = 0.5 + (blue_rates[:, blue_idx] - red_rates[:, red_idx]) / 2.0
                expected_logit = logit_prob(expected_rate)
            objects[:, feature_idx, 0] = matchup_logits[:, feature_idx]
            objects[:, feature_idx, 1] = blue_logits[:, blue_idx]
            objects[:, feature_idx, 2] = red_logits[:, red_idx]
            objects[:, feature_idx, 3] = expected_logit
            objects[:, feature_idx, 4] = confidence_from_counts(
                m1v1_cnt[:, feature_idx],
                prior_strength=prior_strength,
            )
            objects[:, feature_idx, 5] = matchup_logits[:, feature_idx] - expected_logit
    return objects


def build_confidence_summaries(
    p1_cnt: np.ndarray,
    m1v1_cnt: np.ndarray,
    s2vx_cnt: np.ndarray,
    *,
    prior_strength: float,
) -> np.ndarray:
    p1_conf = confidence_from_counts(p1_cnt, prior_strength=prior_strength)
    m1_conf = confidence_from_counts(m1v1_cnt, prior_strength=prior_strength)
    s2_conf = confidence_from_counts(s2vx_cnt, prior_strength=prior_strength)
    return np.column_stack(
        [
            p1_conf.mean(axis=1),
            m1_conf.mean(axis=1),
            m1_conf.max(axis=1),
            s2_conf.mean(axis=1),
            s2_conf.max(axis=1),
            s2_conf[:, :10].mean(axis=1),
            s2_conf[:, 10:].mean(axis=1),
        ]
    ).astype(np.float32)


def build_structured_input_arrays(
    *,
    win_rate: np.ndarray,
    matchup_1v1: np.ndarray,
    synergy_2vx: np.ndarray,
    p1_cnt: np.ndarray,
    m1v1_cnt: np.ndarray,
    s2vx_cnt: np.ndarray,
    prior_strength: float,
    delta_baseline_mode: DeltaBaselineMode = "logit",
) -> StructuredInputArrays:
    return StructuredInputArrays(
        base_features=build_base_identity_features(win_rate),
        synergy_objects=build_synergy_objects(
            win_rate,
            synergy_2vx,
            s2vx_cnt,
            prior_strength=prior_strength,
            delta_baseline_mode=delta_baseline_mode,
        ),
        matchup_objects=build_matchup_objects(
            win_rate,
            matchup_1v1,
            m1v1_cnt,
            prior_strength=prior_strength,
            delta_baseline_mode=delta_baseline_mode,
        ),
        confidence_summaries=build_confidence_summaries(
            p1_cnt,
            m1v1_cnt,
            s2vx_cnt,
            prior_strength=prior_strength,
        ),
        role_pair_type_ids=role_pair_type_ids(),
    )


def structured_tensors(
    arrays: StructuredInputArrays,
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    return {
        "base_features": torch.as_tensor(arrays.base_features, dtype=torch.float32, device=device),
        "confidence_summaries": torch.as_tensor(
            arrays.confidence_summaries,
            dtype=torch.float32,
            device=device,
        ),
        "synergy_objects": torch.as_tensor(arrays.synergy_objects, dtype=torch.float32, device=device),
        "matchup_objects": torch.as_tensor(arrays.matchup_objects, dtype=torch.float32, device=device),
        "role_pair_ids": torch.as_tensor(arrays.role_pair_type_ids, dtype=torch.long, device=device),
    }


def _mlp(
    input_dim: int,
    hidden: tuple[int, ...],
    output_dim: int,
    *,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for hidden_dim in hidden:
        layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class BaseIdentityBranch(nn.Module):
    def __init__(
        self,
        *,
        hidden: tuple[int, ...] = (32, 16),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.net = _mlp(BASE_FEATURE_DIM, hidden, 1, dropout=dropout)

    def forward(self, base_features: torch.Tensor) -> torch.Tensor:
        return self.net(base_features).squeeze(-1)


def select_object_features(objects: torch.Tensor, mode: ObjectFeatureMode) -> torch.Tensor:
    if mode == "raw":
        idx = torch.as_tensor(RAW_OBJECT_INDICES, device=objects.device)
        return objects.index_select(-1, idx)
    return objects


class PairEncoder(nn.Module):
    def __init__(
        self,
        *,
        object_dim: int = SYNERGY_OBJECT_DIM,
        role_embedding_dim: int = 4,
        hidden: tuple[int, ...] = (32,),
        embedding_dim: int = PAIR_EMBEDDING_DIM,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(len(ROLE_PAIR_TYPES), role_embedding_dim)
        self.net = _mlp(object_dim + role_embedding_dim, hidden, embedding_dim, dropout=dropout)

    def forward(self, pair_features: torch.Tensor, role_pair_ids: torch.Tensor) -> torch.Tensor:
        role_embeddings = self.role_embedding(role_pair_ids.to(pair_features.device))
        while role_embeddings.ndim < pair_features.ndim:
            role_embeddings = role_embeddings.unsqueeze(0)
        role_embeddings = role_embeddings.expand(*pair_features.shape[:-1], -1)
        return self.net(torch.cat([pair_features, role_embeddings], dim=-1))


def pool_embeddings(
    embeddings: torch.Tensor,
    confidence: torch.Tensor,
    ops: tuple[str, ...] = POOL_OPS_DEFAULT,
) -> torch.Tensor:
    weights = confidence.unsqueeze(-1).clamp_min(0.0)
    parts: list[torch.Tensor] = []
    for op in ops:
        if op == "mean":
            parts.append(embeddings.mean(dim=1))
        elif op == "max":
            parts.append(embeddings.max(dim=1).values)
        elif op == "min":
            parts.append(embeddings.min(dim=1).values)
        elif op == "weighted":
            parts.append((embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6))
        else:
            raise ValueError(f"Unknown pooling op {op!r}")
    return torch.cat(parts, dim=-1)


class SynergyHead(nn.Module):
    def __init__(
        self,
        *,
        team_repr_dim: int,
        hidden: tuple[int, ...] = (32,),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.net = _mlp(team_repr_dim * 4, hidden, 1, dropout=dropout)

    def forward(self, t_blue: torch.Tensor, t_red: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t_blue, t_red, t_blue - t_red, t_blue * t_red], dim=-1)).squeeze(-1)


class SynergyBranch(nn.Module):
    def __init__(self, config: StructuredModelConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredModelConfig()
        self.config = config
        self.pair_encoder = PairEncoder(
            object_dim=config.object_dim(),
            role_embedding_dim=config.role_embedding_dim,
            hidden=config.pair_hidden,
            dropout=config.dropout,
        )
        self.synergy_head = SynergyHead(
            team_repr_dim=PAIR_EMBEDDING_DIM * config.pool_mult(),
            hidden=config.head_hidden,
            dropout=config.dropout,
        )

    def forward(
        self,
        synergy_objects: torch.Tensor,
        role_pair_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        blue_features = synergy_objects[:, 0]
        red_features = synergy_objects[:, 1]
        blue_conf = blue_features[:, :, SYNERGY_CONFIDENCE_INDEX]
        red_conf = red_features[:, :, SYNERGY_CONFIDENCE_INDEX]
        mode = self.config.object_feature_mode
        blue_embeddings = self.pair_encoder(select_object_features(blue_features, mode), role_pair_ids)
        red_embeddings = self.pair_encoder(select_object_features(red_features, mode), role_pair_ids)
        if self.config.confidence_gate:
            blue_embeddings = blue_embeddings * blue_conf.unsqueeze(-1)
            red_embeddings = red_embeddings * red_conf.unsqueeze(-1)
        ops = self.config.pooling_ops
        t_blue = pool_embeddings(blue_embeddings, blue_conf, ops)
        t_red = pool_embeddings(red_embeddings, red_conf, ops)
        return self.synergy_head(t_blue, t_red), blue_embeddings, red_embeddings, t_blue, t_red


class MatchupEncoder(nn.Module):
    def __init__(self, config: StructuredModelConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredModelConfig()
        self.config = config
        self.slot_embedding = nn.Embedding(25, config.matchup_slot_embedding_dim)
        self.net = _mlp(
            config.object_dim() + config.matchup_slot_embedding_dim,
            config.matchup_hidden,
            MATCHUP_EMBEDDING_DIM,
            dropout=config.dropout,
        )

    def forward(self, matchup_objects: torch.Tensor) -> torch.Tensor:
        features = select_object_features(matchup_objects, self.config.object_feature_mode)
        slot_ids = torch.arange(matchup_objects.shape[1], device=matchup_objects.device)
        slot_embeddings = self.slot_embedding(slot_ids).unsqueeze(0)
        slot_embeddings = slot_embeddings.expand(*features.shape[:-1], -1)
        return self.net(torch.cat([features, slot_embeddings], dim=-1))


def identity_synergy_context(
    pair_embeddings: torch.Tensor,
    pair_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    contexts: list[torch.Tensor] = []
    for identity_idx in range(5):
        pair_indices = [idx for idx, pair in enumerate(TEAM_PAIRS) if identity_idx in pair]
        selected = pair_embeddings[:, pair_indices]
        if pair_confidence is None:
            contexts.append(selected.mean(dim=1))
            continue
        weights = pair_confidence[:, pair_indices].unsqueeze(-1).clamp_min(0.0)
        contexts.append((selected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6))
    return torch.stack(contexts, dim=1)


class CrossInteractionFFN(nn.Module):
    def __init__(self, config: StructuredModelConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredModelConfig()
        team_repr_dim = PAIR_EMBEDDING_DIM * config.pool_mult()
        input_dim = MATCHUP_EMBEDDING_DIM + PAIR_EMBEDDING_DIM * 3 + team_repr_dim * 2
        self.net = _mlp(input_dim, config.head_hidden, MATCHUP_EMBEDDING_DIM, dropout=config.dropout)

    def forward(self, cross_features: torch.Tensor) -> torch.Tensor:
        return self.net(cross_features)


class MatchupHead(nn.Module):
    def __init__(self, config: StructuredModelConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredModelConfig()
        self.config = config
        self.net = _mlp(
            MATCHUP_EMBEDDING_DIM * config.pool_mult(), config.head_hidden, 1, dropout=config.dropout
        )

    def forward(self, matchup_embeddings: torch.Tensor, matchup_confidence: torch.Tensor) -> torch.Tensor:
        if self.config.confidence_gate:
            matchup_embeddings = matchup_embeddings * matchup_confidence.unsqueeze(-1)
        pooled = pool_embeddings(matchup_embeddings, matchup_confidence, self.config.pooling_ops)
        return self.net(pooled).squeeze(-1)


class MatchupBranch(nn.Module):
    def __init__(self, config: StructuredModelConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredModelConfig()
        self.matchup_encoder = MatchupEncoder(config)
        self.matchup_head = MatchupHead(config)

    def forward(self, matchup_objects: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.matchup_encoder(matchup_objects)
        return self.matchup_head(embeddings, matchup_objects[:, :, MATCHUP_CONFIDENCE_INDEX]), embeddings


class FinalHead(nn.Module):
    def __init__(self, config: StructuredModelConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredModelConfig()
        self.net = _mlp(3 + CONFIDENCE_SUMMARY_DIM, config.head_hidden, 1, dropout=config.dropout)

    def forward(
        self,
        base_logit: torch.Tensor,
        synergy_logit: torch.Tensor,
        matchup_logit: torch.Tensor,
        confidence_summaries: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            torch.cat(
                [
                    base_logit.unsqueeze(1),
                    synergy_logit.unsqueeze(1),
                    matchup_logit.unsqueeze(1),
                    confidence_summaries,
                ],
                dim=1,
            )
        ).squeeze(-1)


class StructuredWinModel(nn.Module):
    def __init__(self, config: StructuredModelConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        if config is not None and overrides:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config or StructuredModelConfig(**overrides)
        if self.config.use_cross and not (self.config.use_synergy and self.config.use_matchup):
            raise ValueError("use_cross requires both use_synergy and use_matchup")
        self.base_branch = BaseIdentityBranch(
            hidden=self.config.base_hidden,
            dropout=self.config.dropout,
        )
        self.synergy_branch = SynergyBranch(self.config) if self.config.use_synergy else None
        self.matchup_branch = MatchupBranch(self.config) if self.config.use_matchup else None
        self.cross_layer = CrossInteractionFFN(self.config) if self.config.use_cross else None
        self.final_head = FinalHead(self.config)
        self.register_buffer("matchup_blue_index", torch.as_tensor(MATCHUP_BLUE_INDEX), persistent=False)
        self.register_buffer("matchup_red_index", torch.as_tensor(MATCHUP_RED_INDEX), persistent=False)

    def _cross_update(
        self,
        matchup_embeddings: torch.Tensor,
        blue_pair_embeddings: torch.Tensor,
        red_pair_embeddings: torch.Tensor,
        synergy_objects: torch.Tensor,
        t_blue: torch.Tensor,
        t_red: torch.Tensor,
    ) -> torch.Tensor:
        if self.cross_layer is None:
            return matchup_embeddings
        blue_context = identity_synergy_context(
            blue_pair_embeddings,
            synergy_objects[:, 0, :, SYNERGY_CONFIDENCE_INDEX],
        )
        red_context = identity_synergy_context(
            red_pair_embeddings,
            synergy_objects[:, 1, :, SYNERGY_CONFIDENCE_INDEX],
        )
        blue_idx = cast(torch.Tensor, self.matchup_blue_index)
        red_idx = cast(torch.Tensor, self.matchup_red_index)
        blue_ctx = blue_context.index_select(1, blue_idx)
        red_ctx = red_context.index_select(1, red_idx)
        t_blue_expanded = t_blue.unsqueeze(1).expand(-1, matchup_embeddings.shape[1], -1)
        t_red_expanded = t_red.unsqueeze(1).expand(-1, matchup_embeddings.shape[1], -1)
        return matchup_embeddings + self.cross_layer(
            torch.cat(
                [
                    matchup_embeddings,
                    blue_ctx,
                    red_ctx,
                    blue_ctx - red_ctx,
                    t_blue_expanded,
                    t_red_expanded,
                ],
                dim=-1,
            )
        )

    def forward(
        self,
        *,
        base_features: torch.Tensor,
        confidence_summaries: torch.Tensor,
        synergy_objects: torch.Tensor | None = None,
        matchup_objects: torch.Tensor | None = None,
        role_pair_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base_logit = self.base_branch(base_features)
        synergy_logit = base_logit.new_zeros(base_logit.shape)
        matchup_logit = base_logit.new_zeros(base_logit.shape)
        blue_pair_embeddings = red_pair_embeddings = t_blue = t_red = None

        if self.config.use_synergy:
            if synergy_objects is None or self.synergy_branch is None:
                raise ValueError("synergy_objects are required when use_synergy=True")
            ids = role_pair_ids if role_pair_ids is not None else torch.arange(len(TEAM_PAIRS), device=base_features.device)
            synergy_logit, blue_pair_embeddings, red_pair_embeddings, t_blue, t_red = self.synergy_branch(synergy_objects, ids)

        updated_matchup_embeddings = None
        if self.config.use_matchup:
            if matchup_objects is None or self.matchup_branch is None:
                raise ValueError("matchup_objects are required when use_matchup=True")
            matchup_logit, matchup_embeddings = self.matchup_branch(matchup_objects)
            updated_matchup_embeddings = matchup_embeddings
            if self.config.use_cross:
                if (
                    synergy_objects is None
                    or blue_pair_embeddings is None
                    or red_pair_embeddings is None
                    or t_blue is None
                    or t_red is None
                ):
                    raise RuntimeError("cross layer requires synergy branch outputs")
                updated_matchup_embeddings = self._cross_update(
                    matchup_embeddings,
                    blue_pair_embeddings,
                    red_pair_embeddings,
                    synergy_objects,
                    t_blue,
                    t_red,
                )
                matchup_logit = self.matchup_branch.matchup_head(
                    updated_matchup_embeddings,
                    matchup_objects[:, :, MATCHUP_CONFIDENCE_INDEX],
                )

        result = {
            "base_logit": base_logit,
            "synergy_logit": synergy_logit,
            "matchup_logit": matchup_logit,
            "final_logit": self.final_head(base_logit, synergy_logit, matchup_logit, confidence_summaries),
        }
        if updated_matchup_embeddings is not None:
            result["matchup_embeddings"] = updated_matchup_embeddings
        return result


team_pool = pool_embeddings
TeamSynergyHead = SynergyHead


def _config_from_payload(payload: dict[str, Any]) -> StructuredModelConfig:
    config_dict = dict(payload.get("model_config", {}))
    for key in ("base_hidden", "pair_hidden", "matchup_hidden", "head_hidden", "pooling_ops"):
        if key in config_dict:
            config_dict[key] = tuple(config_dict[key])
    return StructuredModelConfig(**config_dict)


def save_structured_model(path: Path, model: StructuredWinModel, *, prior_strength: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.config),
            "prior_strength": float(prior_strength),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_structured_model(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[StructuredWinModel, StructuredModelConfig, float]:
    payload = cast(dict[str, Any], torch.load(path, map_location=device, weights_only=True))
    config = _config_from_payload(payload)
    model = StructuredWinModel(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, float(payload.get("prior_strength", 20.0))
