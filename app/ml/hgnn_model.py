# pyright: reportPrivateImportUsage=false

"""Match-Outcome HGNN win-rate model.

Production currently uses the relationship-direct path plus the
identity-conditioned raw semantic-context residual: identity embeddings and the
1vX posterior initialise the 10 player nodes, then direct 1v1/2vX residual
features feed the residual head and prior shortcut. Training and inference share
one model shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from app.core.utils.common import TEAM_PAIRS as TEAM_PAIRS

N_PLAYERS = 10
N_RELATIONSHIP_EDGES = 45
LOGIT_EPS = 1e-6


@dataclass(frozen=True)
class HGNNConfig:
    n_champions: int = 951  # embedding rows for champion ids (raw id; size from cache meta)
    n_builds: int = 11  # embedding rows for build ids
    build_vocab: tuple[str, ...] = ()  # build label -> embedding index (for the predictor)
    node_dim: int = 96
    edge_hidden: int = 64  # phi output width
    value_hidden: tuple[int, ...] = (64,)
    gate_hidden: tuple[int, ...] = (32,)
    node_init_hidden: tuple[int, ...] = (96,)
    readout_hidden: tuple[int, ...] = (256,)
    residual_head_hidden: tuple[int, ...] = (128,)
    dropout: float = 0.1
    logit_clip: float | None = 5.0
    identity_semantic_dim: int = 64
    identity_profile_dim: int = 5
    profile_context_dim: int = 0
    profile_context_rank: int = 8
    profile_include_ally_context: bool = False
    profile_include_weighted_enemy_context: bool = False
    profile_include_resistance_products: bool = False
    profile_offense_dims: int = 3
    profile_damage_pressure_index: int = 5
    profile_head_hidden: tuple[int, ...] = (16,)
    # Context atlas head (generalises the profile head to every identity). The
    # per-player identity_context descriptor is crossed against permutation-aware
    # ally/enemy set summaries plus relational 1v1 (lane-opponent) and 2vX (ally)
    # products. identity_context_dim == 0 disables the head (default off). Axis
    # indices match app.classification.embeddings.config.
    identity_context_dim: int = 0
    context_interpretable_dim: int = 14
    context_offense_dims: int = 3
    context_armor_index: int = 3
    context_mr_index: int = 4
    context_damage_pressure_index: int = 5
    context_taken_index: int = 9
    context_heal_shield_index: int = 10
    context_head_hidden: tuple[int, ...] = (32,)
    context_support_strength: float = 30.0
    context_include_ally: bool = True
    context_include_relational: bool = True
    # Identity-conditioned context module. Instead of one shared
    # context head for every identity, a low-rank bottleneck lets the model learn
    # which context interactions matter per (champion, role, build): an identity
    # conditioner maps [champion/role/build embedding || self raw context] to a
    # rank-r vector, a context projector maps the player's raw set/relational
    # context to another rank-r vector, and their dot product is the per-player
    # context score. Same module/params for both teams; aggregated as the
    # antisymmetric, support-gated blue-minus-red residual. The production
    # config builder enables this raw-atlas path by default; when enabled it
    # REPLACES the shared context head's contribution.
    use_identity_conditioned_context: bool = False
    identity_context_conditioning_type: str = "none"  # "none" | "low_rank"
    identity_context_source: str = "raw"  # "raw" | "raw_plus_dense"
    identity_context_raw_dim: int = 0  # width of the wide RAW atlas block
    identity_context_rank: int = 16
    identity_context_hidden_dim: int = 64
    identity_context_emb_dim: int = 16
    identity_context_init_scale: float = 0.01
    identity_context_dropout: float = 0.0
    identity_context_use_residual_mlp: bool = False
    m1v1_detail_dim: int = 0
    # Classification-feature integration keeps relationship detail disabled in
    # production. The 1v1 detail path remains as an explicit experiment hook.
    detail_prior_gated: bool = True


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _logit(prob: torch.Tensor, clip: float | None = None) -> torch.Tensor:
    p = prob.clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)
    out = torch.log(p / (1.0 - p))
    if clip is not None and clip > 0.0:
        return out.clamp(-float(clip), float(clip))
    return out


def _clip_logit(value: torch.Tensor, clip: float | None = None) -> torch.Tensor:
    if clip is not None and clip > 0.0:
        return value.clamp(-float(clip), float(clip))
    return value


def relationship_logit_features(
    *,
    mu_1vx: torch.Tensor,
    mu_2vx: torch.Tensor,
    mu_1v1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Explicit relationship residuals over the generic 1vX expectation.

    The structured model previously handed the network the important residual:
    joint relationship logit minus the logit of the same generic 1vX baseline
    used by cache smoothing (`smoothing.composite_interaction_priors`).
    Keep the same idea in the HGNN tensor contract.
    """
    blue = mu_1vx[:, :5]
    red = mu_1vx[:, 5:]

    joint_1v1 = _logit(mu_1v1)
    expected_prob_1v1 = (0.5 + (blue[:, :, None] - red[:, None, :]) / 2.0).reshape(-1, 25)
    expected_1v1 = _logit(expected_prob_1v1)

    expected_2vx_parts: list[torch.Tensor] = []
    for offset in (0, 5):
        side = mu_1vx[:, offset : offset + 5]
        for a_idx, b_idx in TEAM_PAIRS:
            expected_2vx_parts.append(_logit(0.5 * (side[:, a_idx] + side[:, b_idx])))
    expected_2vx = torch.stack(expected_2vx_parts, dim=1)
    joint_2vx = _logit(mu_2vx)

    return {
        "delta_logit_1v1": joint_1v1 - expected_1v1,
        "delta_logit_2vx": joint_2vx - expected_2vx,
    }


def posterior_mean_var(
    rate: torch.Tensor,
    count: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Beta-Binomial posterior mean and variance for a cached smoothed rate.

    `rate` is the shrunk posterior mean (already smoothed upstream). Treating
    ``count + strength`` as the effective sample size, the posterior variance is
    ``mu(1-mu)/(count+strength+1)`` — small for well-observed edges (gate opens),
    large for barely-observed ones (gate suppresses).
    """
    mu = rate.clamp(0.0, 1.0)
    n_eff = count.clamp_min(0.0) + float(strength)
    var = mu * (1.0 - mu) / (n_eff + 1.0)
    return mu, var


def support_features(
    count: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit support features for node priors and direct relationship heads.

    Posterior variance has a tiny numeric range for this dataset, so the model also
    sees a direct confidence score, log support, and a missing flag.
    """
    count_f = count.clamp_min(0.0)
    denom = count_f + float(strength)
    confidence = torch.where(denom > 0.0, count_f / denom.clamp_min(1.0e-12), torch.zeros_like(count_f))
    log_count = torch.log1p(count_f)
    missing = (count_f <= 0.0).to(count.dtype)
    return confidence, log_count, missing


def build_hgnn_inputs(
    *,
    champion_id: Any,
    build_id: Any,
    win_rate: Any,
    matchup_1v1: Any,
    synergy_2vx: Any,
    p1_cnt: Any,
    m1v1_cnt: Any,
    s2vx_cnt: Any,
    strength: float,
    identity_semantic: Any | None = None,
    identity_profile: Any | None = None,
    identity_context: Any | None = None,
    identity_context_support: Any | None = None,
    identity_context_raw: Any | None = None,
    m1v1_detail: Any | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Turn raw cache/prior arrays into the model's node/edge tensors.

    Single source of truth shared by training and the runtime predictor. Accepts
    numpy arrays or tensors. mu values stay in probability space (the model maps
    them to logits internally); champion/build ids become long embedding indices.

    Relationship means already include nested-pooling backoff. Direct
    confidence/log-count/missing features use raw build-level support so the
    model knows when an exact relationship cell was sparse or absent.
    """

    def to_tensor(arr: Any) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def to_long(arr: Any) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.long, device=device)

    count_1vx = to_tensor(p1_cnt)
    count_2vx = to_tensor(s2vx_cnt)
    count_1v1 = to_tensor(m1v1_cnt)
    mu_1vx, var_1vx = posterior_mean_var(to_tensor(win_rate), count_1vx, strength)
    mu_2vx = to_tensor(synergy_2vx).clamp(0.0, 1.0)
    mu_1v1 = to_tensor(matchup_1v1).clamp(0.0, 1.0)
    conf_1vx, log_count_1vx, missing_1vx = support_features(count_1vx, strength)
    conf_2vx, log_count_2vx, missing_2vx = support_features(count_2vx, strength)
    conf_1v1, log_count_1v1, missing_1v1 = support_features(count_1v1, strength)
    inputs = {
        "champion_id": to_long(champion_id),
        "build_id": to_long(build_id),
        "mu_1vx": mu_1vx,
        "var_1vx": var_1vx,
        "conf_1vx": conf_1vx,
        "log_count_1vx": log_count_1vx,
        "missing_1vx": missing_1vx,
        "mu_2vx": mu_2vx,
        "conf_2vx": conf_2vx,
        "log_count_2vx": log_count_2vx,
        "missing_2vx": missing_2vx,
        "mu_1v1": mu_1v1,
        "conf_1v1": conf_1v1,
        "log_count_1v1": log_count_1v1,
        "missing_1v1": missing_1v1,
    }
    if identity_semantic is not None:
        inputs["identity_semantic"] = to_tensor(identity_semantic)
    if identity_profile is not None:
        inputs["identity_profile"] = to_tensor(identity_profile)
    if identity_context is not None:
        inputs["identity_context"] = to_tensor(identity_context)
    if identity_context_support is not None:
        inputs["identity_context_support"] = to_tensor(identity_context_support)
    if identity_context_raw is not None:
        inputs["identity_context_raw"] = to_tensor(identity_context_raw)
    if m1v1_detail is not None:
        inputs["m1v1_detail"] = to_tensor(m1v1_detail)
    inputs.update(relationship_logit_features(mu_1vx=mu_1vx, mu_2vx=mu_2vx, mu_1v1=mu_1v1))
    return inputs


def swap_hgnn_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Team-swap augmentation (design §8/§9): swap blue<->red, return mirrored inputs.

    Used with a flipped label so the learned function is approximately antisymmetric
    (``P(A beats B) = 1 - P(B beats A)``) and the training data is effectively doubled.
    """

    def swap_halves(x: torch.Tensor, half: int) -> torch.Tensor:
        return torch.cat([x[:, half:], x[:, :half]], dim=1)

    def swap_1v1_mu(x: torch.Tensor) -> torch.Tensor:
        # new[newblue=j, newred=i] = P(oldred_j beats oldblue_i) = 1 - old[i,j]
        return (1.0 - x.reshape(-1, 5, 5)).transpose(1, 2).reshape(-1, 25)

    def swap_1v1_matrix(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, 5, 5).transpose(1, 2).reshape(-1, 25)

    def swap_1v1_signed(x: torch.Tensor) -> torch.Tensor:
        return -x.reshape(-1, 5, 5).transpose(1, 2).reshape(-1, 25)

    def swap_1v1_detail(x: torch.Tensor) -> torch.Tensor:
        return -x.reshape(x.shape[0], 5, 5, x.shape[-1]).transpose(1, 2).reshape(x.shape[0], 25, x.shape[-1])

    swapped = {
        "champion_id": swap_halves(inputs["champion_id"], 5),
        "build_id": swap_halves(inputs["build_id"], 5),
        "mu_1vx": swap_halves(inputs["mu_1vx"], 5),
        "var_1vx": swap_halves(inputs["var_1vx"], 5),
        "mu_2vx": swap_halves(inputs["mu_2vx"], 10),
        "mu_1v1": swap_1v1_mu(inputs["mu_1v1"]),
    }
    if "identity_semantic" in inputs:
        swapped["identity_semantic"] = swap_halves(inputs["identity_semantic"], 5)
    if "identity_profile" in inputs:
        swapped["identity_profile"] = swap_halves(inputs["identity_profile"], 5)
    if "identity_context" in inputs:
        swapped["identity_context"] = swap_halves(inputs["identity_context"], 5)
    if "identity_context_support" in inputs:
        swapped["identity_context_support"] = swap_halves(inputs["identity_context_support"], 5)
    if "identity_context_raw" in inputs:
        swapped["identity_context_raw"] = swap_halves(inputs["identity_context_raw"], 5)
    if "m1v1_detail" in inputs:
        swapped["m1v1_detail"] = swap_1v1_detail(inputs["m1v1_detail"])
    if "var_2vx" in inputs:
        swapped["var_2vx"] = swap_halves(inputs["var_2vx"], 10)
    if "var_1v1" in inputs:
        swapped["var_1v1"] = swap_1v1_matrix(inputs["var_1v1"])
    for suffix, half, prefixes in (
        ("1vx", 5, ("conf", "log_count", "missing")),
        ("2vx", 10, ("conf", "log_count", "missing")),
    ):
        for prefix in prefixes:
            key = f"{prefix}_{suffix}"
            if key in inputs:
                swapped[key] = swap_halves(inputs[key], half)
    for prefix in ("conf", "log_count", "missing"):
        key = f"{prefix}_1v1"
        if key in inputs:
            swapped[key] = swap_1v1_matrix(inputs[key])
    if "delta_logit_2vx" in inputs:
        swapped["delta_logit_2vx"] = swap_halves(inputs["delta_logit_2vx"], 10)
    if "delta_logit_1v1" in inputs:
        swapped["delta_logit_1v1"] = swap_1v1_signed(inputs["delta_logit_1v1"])
    return swapped


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int, *, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for hidden_dim in hidden:
        layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class PhiEncoder(nn.Module):
    """Uncertainty-gated encoder for the 1vX node posterior."""

    def __init__(self, config: HGNNConfig) -> None:
        super().__init__()
        self.value = _mlp(5, config.value_hidden, config.edge_hidden, dropout=config.dropout)
        self.gate = _mlp(4, config.gate_hidden, config.edge_hidden, dropout=config.dropout)

    def forward(
        self,
        mu_logit: torch.Tensor,
        var: torch.Tensor,
        confidence: torch.Tensor | None = None,
        log_count: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if confidence is None or log_count is None or missing is None:
            raise ValueError("PhiEncoder requires confidence/log_count/missing tensors")
        precision = 1.0 / (1.0 + var)
        value_input = torch.stack([mu_logit, var, confidence, log_count, missing], dim=-1)
        gate_input = torch.stack([precision, confidence, log_count, missing], dim=-1)
        value = self.value(value_input)
        gate = torch.sigmoid(self.gate(gate_input))
        return gate * value


class IdentityEncoder(nn.Module):
    """Multiplicative (champion, role, build) node initialisation (design §3).

    role and build modulate the champion via ``(1 + W·e)``, so an untrained/rare
    modifier defaults to a near-identity transform of the champion embedding — the
    architectural mirror of the Bayesian shrinkage ``θ(c,r,b) → θ(c,r) → θ(c)``.
    The last embedding row of champion/build is reserved for unknown ids.
    """

    def __init__(self, n_champions: int, n_builds: int, dim: int) -> None:
        super().__init__()
        self.champion = nn.Embedding(n_champions + 1, dim)
        self.role = nn.Embedding(5, dim)
        self.build = nn.Embedding(n_builds + 1, dim)
        self.w_role = nn.Linear(dim, dim, bias=False)
        self.w_build = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer(
            "role_idx",
            torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long),
            persistent=False,
        )

    def forward(self, champion_id: torch.Tensor, build_id: torch.Tensor) -> torch.Tensor:
        e_c = self.champion(champion_id)  # [B, 10, dim]
        e_r = self.role(cast(torch.Tensor, self.role_idx)).unsqueeze(0)  # [1, 10, dim]
        e_b = self.build(build_id)  # [B, 10, dim]
        return self.norm(e_c * (1.0 + self.w_role(e_r)) * (1.0 + self.w_build(e_b)))


class AttnPool(nn.Module):
    """Learned-query attention pool over a team's 5 nodes (design §8)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.scale = dim**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 5, d]
        scores = (x @ self.query) / self.scale
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class IdentityConditionedContext(nn.Module):
    """Low-rank identity-conditioned context interaction.

    Generalises the shared context head: instead of one global function of the
    context summaries, the per-player score is a dot product between

    * ``z_id``  = identity_conditioner([champion_emb, role_emb, build_emb, self_raw])
    * ``z_ctx`` = context_projector([self, enemy_mean, enemy_weighted, lane_opp, ally_mean])

    so different identities can express different context sensitivities through one
    small, regularised low-rank bottleneck — no per-champion parameters and no
    sparse matchup keys. The same module/params score both teams; the team-level
    contribution is the antisymmetric, support-gated ``sum_blue - sum_red`` so it
    flips sign under team swap for any weights. The context projector's last layer
    is zero-initialised, so the whole residual starts at exactly zero (opt-in).
    """

    def __init__(self, config: HGNNConfig) -> None:
        super().__init__()
        c = config
        self.source_mode = c.identity_context_source
        self.raw_dim = c.identity_context_raw_dim
        self.dense_dim = (
            max(c.identity_context_dim - c.context_interpretable_dim, 0)
            if c.identity_context_source == "raw_plus_dense"
            else 0
        )
        self.source_dim = self.raw_dim + self.dense_dim
        self.support_strength = float(c.context_support_strength)
        self.damage_idx = c.context_damage_pressure_index
        self.init_scale = float(c.identity_context_init_scale)
        self.use_residual_mlp = bool(c.identity_context_use_residual_mlp)

        emb = c.identity_context_emb_dim
        rank = c.identity_context_rank
        hidden = c.identity_context_hidden_dim
        dropout = c.identity_context_dropout
        self.champion = nn.Embedding(c.n_champions + 1, emb)
        self.role = nn.Embedding(5, emb)
        self.build = nn.Embedding(c.n_builds + 1, emb)
        for table in (self.champion, self.role, self.build):
            nn.init.normal_(table.weight, std=0.02)
        self.register_buffer(
            "role_idx", torch.tensor([0, 1, 2, 3, 4], dtype=torch.long), persistent=False
        )

        cond_in = 3 * emb + self.source_dim
        ctx_in = 5 * self.source_dim  # self, enemy_mean, enemy_weighted, lane_opp, ally_mean
        self.identity_conditioner = _mlp(cond_in, (hidden,), rank, dropout=dropout)
        self.context_projector = _mlp(ctx_in, (hidden,), rank, dropout=dropout)
        nn.init.zeros_(self.context_projector[-1].weight)
        nn.init.zeros_(self.context_projector[-1].bias)
        if self.use_residual_mlp:
            self.residual_mlp = _mlp(ctx_in, (hidden,), 1, dropout=dropout)
            nn.init.zeros_(self.residual_mlp[-1].weight)
            nn.init.zeros_(self.residual_mlp[-1].bias)

    def _source(self, raw: torch.Tensor, dense: torch.Tensor | None) -> torch.Tensor:
        raw = _fit_dim(raw, self.raw_dim)
        if self.dense_dim <= 0:
            return raw
        if dense is None:
            dense = raw.new_zeros(*raw.shape[:-1], self.dense_dim)
        return torch.cat([raw, _fit_dim(dense, self.dense_dim)], dim=-1)

    def _team_score(
        self,
        self_src: torch.Tensor,  # [B, 5, S]
        enemy_src: torch.Tensor,  # [B, 5, S]
        self_champ: torch.Tensor,  # [B, 5]
        self_build: torch.Tensor,  # [B, 5]
        conf: torch.Tensor,  # [B, 5]
    ) -> torch.Tensor:
        b, n = self_src.shape[0], self_src.shape[1]
        enemy_mean = enemy_src.mean(dim=1, keepdim=True)
        weight = enemy_src[..., self.damage_idx : self.damage_idx + 1].clamp_min(0.0)
        denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        enemy_weighted = (enemy_src * weight).sum(dim=1, keepdim=True) / denom
        ally_mean = self_src.mean(dim=1, keepdim=True)
        # lane_opp == enemy_src (slots are role-ordered, so slot i faces slot i).
        ctx_feat = torch.cat(
            [
                self_src,
                enemy_mean.expand_as(self_src),
                enemy_weighted.expand_as(self_src),
                enemy_src,
                ally_mean.expand_as(self_src),
            ],
            dim=-1,
        )
        role = self.role(cast(torch.Tensor, self.role_idx)).unsqueeze(0).expand(b, n, -1)
        cond = torch.cat(
            [self.champion(self_champ), role, self.build(self_build), self_src], dim=-1
        )
        z_id = self.identity_conditioner(cond)
        z_ctx = self.context_projector(ctx_feat)
        raw_context = self.init_scale * (z_id * z_ctx).sum(dim=-1)
        if self.use_residual_mlp:
            raw_context = raw_context + self.init_scale * self.residual_mlp(ctx_feat).squeeze(-1)
        return (conf * raw_context).sum(dim=1)

    def forward(
        self,
        identity_context_raw: torch.Tensor,
        identity_context_support: torch.Tensor | None,
        champion_id: torch.Tensor,
        build_id: torch.Tensor,
        identity_context_dense: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src = self._source(identity_context_raw, identity_context_dense)
        blue_src, red_src = src[:, :5], src[:, 5:]
        blue_c, red_c = champion_id[:, :5], champion_id[:, 5:]
        blue_b, red_b = build_id[:, :5], build_id[:, 5:]
        if identity_context_support is None:
            blue_conf = blue_src.new_ones(blue_src.shape[0], 5)
            red_conf = red_src.new_ones(red_src.shape[0], 5)
        else:
            sup = identity_context_support.clamp_min(0.0)
            conf = sup / (sup + self.support_strength)
            blue_conf, red_conf = conf[:, :5], conf[:, 5:]
        fwd = self._team_score(blue_src, red_src, blue_c, blue_b, blue_conf)
        rev = self._team_score(red_src, blue_src, red_c, red_b, red_conf)
        return fwd - rev


def _fit_dim(values: torch.Tensor, dim: int) -> torch.Tensor:
    if values.shape[-1] == dim:
        return values
    if values.shape[-1] > dim:
        return values[..., :dim]
    pad = values.new_zeros(*values.shape[:-1], dim - values.shape[-1])
    return torch.cat([values, pad], dim=-1)


class HGNNWinModel(nn.Module):
    def __init__(self, config: HGNNConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        if config is not None and overrides:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config or HGNNConfig(**overrides)
        c = self.config

        self.phi = nn.ModuleDict({"1vx": PhiEncoder(c)})
        self.identity = IdentityEncoder(c.n_champions, c.n_builds, c.node_dim)
        self.identity_semantic_proj = (
            nn.Linear(c.identity_semantic_dim, c.node_dim)
            if c.identity_semantic_dim > 0
            else None
        )
        # Zero-init scalar gate: the semantic descriptor enters as an opt-in
        # residual on the learned identity embedding, starting at exactly the
        # no-feature model and only opening if it lowers loss.
        self.identity_semantic_gate = (
            nn.Parameter(torch.zeros(1)) if c.identity_semantic_dim > 0 else None
        )
        # Node init fuses the multiplicative identity with the 1vX posterior (§3).
        self.node_init = _mlp(
            c.node_dim + c.edge_hidden, c.node_init_hidden, c.node_dim, dropout=c.dropout
        )
        self.node_norm = nn.LayerNorm(c.node_dim)

        self.attn_pool = AttnPool(c.node_dim)
        self.team_proj = nn.Linear(c.node_dim * 3, c.node_dim)
        self.residual_head = _mlp(
            N_RELATIONSHIP_EDGES * 4,
            c.residual_head_hidden,
            c.node_dim,
            dropout=c.dropout,
        )
        self.m1v1_detail_enabled = c.m1v1_detail_dim > 0
        self.detail_enabled = self.m1v1_detail_enabled
        if self.m1v1_detail_enabled:
            self.m1v1_detail_lin = nn.Linear(c.m1v1_detail_dim, 1, bias=False)
            self.detail_gate = nn.Parameter(torch.zeros(1))
        # Cross-team matchup-profile interaction: each identity carries an
        # interpretable profile (offense damage-type mix + resistance fractions).
        # The win-rate prior marginalises over enemy compositions, so it cannot
        # express "armor-stacking identity vs a physical-damage enemy team". The
        # profile head scores each player against the enemy-team profile and then
        # subtracts the mirrored red score, making the term exactly antisymmetric.
        #
        # Profile v2 can additionally condition that score on a low-rank view of
        # the dense identity descriptor. That descriptor is historical/cache
        # context keyed by champion/role/build; it may summarize durability and
        # damage-taken tendencies, but it never sees the current match's realised
        # postgame fields. Keeping it inside this tiny profile bottleneck avoids
        # reintroducing the wide semantic node path that overfit in ablations.
        #
        # Profile v3 can also append a deterministic, damage-weighted enemy
        # offense context. The identity profile carries each opponent's expected
        # champion-damage pressure, so the context approximates "what share of
        # the enemy team's expected damage is physical/magic/true" at draft time
        # instead of treating all five enemy identities as equal contributors.
        # Product features make the key specialization axis explicit (for
        # example, self armor fraction × weighted enemy physical share) while
        # retaining the same global/shared profile head.
        self.profile_enabled = c.identity_profile_dim > 0
        if self.profile_enabled:
            self.profile_context_enabled = c.profile_context_dim > 0 and c.profile_context_rank > 0
            self.profile_weighted_context_dim = (
                min(c.profile_offense_dims, c.identity_profile_dim)
                if c.profile_include_weighted_enemy_context
                else 0
            )
            self.profile_resistance_product_dim = (
                4
                if c.profile_include_resistance_products
                and c.identity_profile_dim > 4
                and min(c.profile_offense_dims, c.identity_profile_dim) >= 2
                else 0
            )
            self.profile_context_proj = (
                nn.Sequential(
                    nn.LayerNorm(c.profile_context_dim),
                    nn.Linear(c.profile_context_dim, c.profile_context_rank),
                    nn.Tanh(),
                )
                if self.profile_context_enabled
                else None
            )
            profile_input_dim = 2 * c.identity_profile_dim
            if c.profile_include_ally_context:
                profile_input_dim += c.identity_profile_dim
            profile_input_dim += self.profile_weighted_context_dim
            profile_input_dim += self.profile_resistance_product_dim
            if self.profile_context_enabled:
                profile_input_dim += c.profile_context_rank
            self.profile_head = _mlp(
                profile_input_dim,
                c.profile_head_hidden,
                1,
                dropout=0.0,
            )
            nn.init.zeros_(self.profile_head[-1].weight)
            nn.init.zeros_(self.profile_head[-1].bias)
        else:
            self.profile_context_enabled = False
            self.profile_weighted_context_dim = 0
            self.profile_resistance_product_dim = 0
            self.profile_context_proj = None
        # Context-atlas head: the production generalisation of the profile head.
        # Each player's full identity_context descriptor is crossed against
        # permutation-aware ally/enemy set summaries (mean + damage-pressure
        # weighted mean = DeepSets), the same-role lane opponent (1v1 edge), and
        # explicit interpretable cross products (resistance vs enemy offense,
        # damage-taken vs enemy damage, own damage vs enemy heal/shield, own
        # heal/shield vs ally damage = enchanter/carry 2vX). The contribution is
        # support-gated per player and antisymmetric (fwd - rev); the final layer
        # is zero-initialised so the whole path is opt-in.
        self.context_enabled = c.identity_context_dim > 0
        if self.context_enabled:
            self.context_product_dim = 7 if c.context_include_relational else 0
            n_set = 4 + (1 if c.context_include_ally else 0)
            context_input_dim = n_set * c.identity_context_dim + self.context_product_dim
            self.context_head = _mlp(
                context_input_dim,
                c.context_head_hidden,
                1,
                dropout=0.0,
            )
            nn.init.zeros_(self.context_head[-1].weight)
            nn.init.zeros_(self.context_head[-1].bias)
        else:
            self.context_product_dim = 0
        # Identity-conditioned context module. Production enables this raw-atlas
        # path, which supplies the context residual in place of the shared head.
        self.identity_conditioned_context_enabled = (
            c.use_identity_conditioned_context
            and c.identity_context_conditioning_type == "low_rank"
            and c.identity_context_raw_dim > 0
        )
        if self.identity_conditioned_context_enabled:
            self.identity_conditioned_context = IdentityConditionedContext(c)
        self.prior_shortcut = _mlp(
            15 + N_RELATIONSHIP_EDGES * 2,
            (),
            1,
            dropout=0.0,
        )
        self.head = _mlp(c.node_dim * 5, c.readout_hidden, 1, dropout=c.dropout)

    def _readout(self, team: torch.Tensor) -> torch.Tensor:  # team: [B, 5, d]
        pooled = torch.cat([team.mean(dim=1), team.max(dim=1).values, self.attn_pool(team)], dim=-1)
        return self.team_proj(pooled)

    def _signed_relationship_tensor(
        self,
        tensor_2vx: torch.Tensor,
        tensor_1v1: torch.Tensor,
    ) -> torch.Tensor:
        signed_2vx = torch.cat([tensor_2vx[:, :10], -tensor_2vx[:, 10:]], dim=1)
        return _clip_logit(torch.cat([tensor_1v1, signed_2vx], dim=1), self.config.logit_clip)

    def _relationship_support_tensor(
        self,
        tensor_2vx: torch.Tensor | None,
        tensor_1v1: torch.Tensor | None,
        *,
        default_like: torch.Tensor,
        default_value: float,
    ) -> torch.Tensor:
        if tensor_1v1 is None or tensor_2vx is None:
            return torch.full_like(default_like, default_value)
        return torch.cat([tensor_1v1, tensor_2vx], dim=1)

    def _shortcut_relationship_features(
        self,
        *,
        delta_logit_2vx: torch.Tensor,
        delta_logit_1v1: torch.Tensor,
        conf_2vx: torch.Tensor | None,
        conf_1v1: torch.Tensor | None,
        missing_2vx: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
        include_support: bool,
    ) -> torch.Tensor:
        delta = self._signed_relationship_tensor(delta_logit_2vx, delta_logit_1v1)
        confidence = self._relationship_support_tensor(
            conf_2vx,
            conf_1v1,
            default_like=delta,
            default_value=1.0,
        )
        parts = [delta]
        if include_support:
            parts.append(confidence)
        parts.append(delta * confidence)
        if include_support:
            missing = self._relationship_support_tensor(
                missing_2vx,
                missing_1v1,
                default_like=delta,
                default_value=0.0,
            )
            parts.append(missing)
        return torch.cat(parts, dim=1)

    def _residual_readout(
        self,
        *,
        delta_logit_2vx: torch.Tensor,
        delta_logit_1v1: torch.Tensor,
        conf_2vx: torch.Tensor | None,
        conf_1v1: torch.Tensor | None,
        missing_2vx: torch.Tensor | None,
        missing_1v1: torch.Tensor | None,
    ) -> torch.Tensor:
        residual_input = self._shortcut_relationship_features(
            delta_logit_2vx=delta_logit_2vx,
            delta_logit_1v1=delta_logit_1v1,
            conf_2vx=conf_2vx,
            conf_1v1=conf_1v1,
            missing_2vx=missing_2vx,
            missing_1v1=missing_1v1,
            include_support=True,
        )
        return self.residual_head(residual_input)

    def _detail_logit(
        self,
        m1v1_detail: torch.Tensor | None,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Experimental scalar logit from blue-perspective 1v1 detail."""
        logit = like.new_zeros(like.shape[0])
        if self.m1v1_detail_enabled and m1v1_detail is not None:
            logit = logit + self.m1v1_detail_lin(m1v1_detail.mean(dim=1)).squeeze(-1)
        return logit

    def _fit_feature_dim(self, values: torch.Tensor, dim: int) -> torch.Tensor:
        if values.shape[-1] == dim:
            return values
        if values.shape[-1] > dim:
            return values[..., :dim]
        pad = values.new_zeros(*values.shape[:-1], dim - values.shape[-1])
        return torch.cat([values, pad], dim=-1)

    def _weighted_offense_context(self, team_profile: torch.Tensor) -> torch.Tensor:
        dims = min(self.config.profile_offense_dims, team_profile.shape[-1])
        if dims <= 0:
            return team_profile.new_zeros(team_profile.shape[0], 1, 0)
        offense = team_profile[..., :dims]
        pressure_idx = self.config.profile_damage_pressure_index
        if team_profile.shape[-1] > pressure_idx:
            pressure = team_profile[..., pressure_idx : pressure_idx + 1].clamp_min(0.0)
            denom = pressure.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            return (offense * pressure).sum(dim=1, keepdim=True) / denom
        return offense.mean(dim=1, keepdim=True)

    def _resistance_products(
        self,
        players: torch.Tensor,
        enemy_weighted: torch.Tensor,
    ) -> torch.Tensor:
        if self.profile_resistance_product_dim <= 0:
            return players.new_zeros(players.shape[0], players.shape[1], 0)
        armor = players[..., 3:4]
        magic_resist = players[..., 4:5]
        phys = enemy_weighted[..., 0:1].expand(-1, players.shape[1], -1)
        magic = enemy_weighted[..., 1:2].expand(-1, players.shape[1], -1)
        return torch.cat(
            [
                armor * phys,
                magic_resist * magic,
                armor * (phys - magic),
                magic_resist * (magic - phys),
            ],
            dim=-1,
        )

    def _profile_logit(
        self,
        identity_profile: torch.Tensor,
        identity_semantic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Antisymmetric cross-team profile interaction.

        Per-player so a single extreme identity (e.g. one armor-stacking champion)
        is not washed out by a team mean. The contribution is
        ``sum_blue h(blue_player, red_context) - sum_red h(red_player, blue_context)``
        which flips sign under team swap for any h. Optional profile-v2 context
        uses a low-rank projection of the historical dense identity descriptor to
        let different identities learn different profile sensitivities without
        exposing current-match postgame fields.
        """
        identity_profile = self._fit_feature_dim(
            identity_profile,
            self.config.identity_profile_dim,
        )
        blue = identity_profile[:, :5]  # [B, 5, P]
        red = identity_profile[:, 5:]
        blue_mean = blue.mean(dim=1, keepdim=True)
        red_mean = red.mean(dim=1, keepdim=True)

        blue_parts = [blue]
        red_parts = [red]
        if self.config.profile_include_ally_context:
            blue_parts.append(blue_mean.expand_as(blue))
            red_parts.append(red_mean.expand_as(red))
        blue_parts.append(red_mean.expand_as(blue))
        red_parts.append(blue_mean.expand_as(red))
        weighted_needed = (
            self.profile_weighted_context_dim > 0 or self.profile_resistance_product_dim > 0
        )
        if weighted_needed:
            red_weighted = self._weighted_offense_context(red)
            blue_weighted = self._weighted_offense_context(blue)
        if self.profile_weighted_context_dim > 0:
            blue_parts.append(red_weighted.expand(-1, blue.shape[1], -1))
            red_parts.append(blue_weighted.expand(-1, red.shape[1], -1))
        if self.profile_resistance_product_dim > 0:
            blue_parts.append(self._resistance_products(blue, red_weighted))
            red_parts.append(self._resistance_products(red, blue_weighted))

        if self.profile_context_enabled:
            if identity_semantic is None or self.profile_context_proj is None:
                context = identity_profile.new_zeros(
                    identity_profile.shape[0],
                    identity_profile.shape[1],
                    self.config.profile_context_rank,
                )
            else:
                semantic = self._fit_feature_dim(identity_semantic, self.config.profile_context_dim)
                context = self.profile_context_proj(semantic)
            blue_parts.append(context[:, :5])
            red_parts.append(context[:, 5:])

        fwd = self.profile_head(torch.cat(blue_parts, dim=-1)).squeeze(-1).sum(dim=1)
        rev = self.profile_head(torch.cat(red_parts, dim=-1)).squeeze(-1).sum(dim=1)
        return fwd - rev

    def _ctx_weighted_mean(self, team: torch.Tensor) -> torch.Tensor:
        """Damage-pressure-weighted team-context mean (DeepSets summary), [B, 1, D]."""
        idx = self.config.context_damage_pressure_index
        if team.shape[-1] > idx:
            weight = team[..., idx : idx + 1].clamp_min(0.0)
            denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            return (team * weight).sum(dim=1, keepdim=True) / denom
        return team.mean(dim=1, keepdim=True)

    def _ctx_products(
        self,
        players: torch.Tensor,  # [B, 5, D]
        enemy_weighted: torch.Tensor,  # [B, 1, D]
        ally_mean: torch.Tensor,  # [B, 1, D]
    ) -> torch.Tensor:
        """Explicit interpretable cross products (the audit's specialization axes).

        Generalises ``_resistance_products`` to the wider context descriptor: the
        same armor/MR × enemy-offense terms, plus damage-taken × enemy damage,
        own-damage × enemy heal/shield, and own heal/shield × ally damage (the
        enchanter/carry 2vX synergy).
        """
        c = self.config
        n_players = players.shape[1]
        enemy = enemy_weighted.expand(-1, n_players, -1)
        ally = ally_mean.expand(-1, n_players, -1)

        def col(t: torch.Tensor, i: int) -> torch.Tensor:
            return t[..., i : i + 1]

        armor = col(players, c.context_armor_index)
        mr = col(players, c.context_mr_index)
        enemy_phys = col(enemy, 0)
        enemy_magic = col(enemy, 1)
        enemy_damage = col(enemy, c.context_damage_pressure_index)
        enemy_heal = col(enemy, c.context_heal_shield_index)
        self_taken = col(players, c.context_taken_index)
        self_damage = col(players, c.context_damage_pressure_index)
        self_heal = col(players, c.context_heal_shield_index)
        ally_damage = col(ally, c.context_damage_pressure_index)
        return torch.cat(
            [
                armor * enemy_phys,
                mr * enemy_magic,
                armor * (enemy_phys - enemy_magic),
                mr * (enemy_magic - enemy_phys),
                self_taken * enemy_damage,
                self_damage * enemy_heal,
                self_heal * ally_damage,
            ],
            dim=-1,
        )

    def _context_team_score(
        self,
        self_team: torch.Tensor,  # [B, 5, D] (also the ally team)
        enemy_team: torch.Tensor,  # [B, 5, D]
        conf: torch.Tensor,  # [B, 5] support gate
    ) -> torch.Tensor:
        enemy_mean = enemy_team.mean(dim=1, keepdim=True)
        enemy_weighted = self._ctx_weighted_mean(enemy_team)
        ally_mean = self_team.mean(dim=1, keepdim=True)
        # Same-role lane opponent: slots are role-ordered, so enemy_team[:, i] is
        # the 1v1 opposite of self_team[:, i].
        parts = [
            self_team,
            enemy_mean.expand_as(self_team),
            enemy_weighted.expand_as(self_team),
            enemy_team,
        ]
        if self.config.context_include_ally:
            parts.append(ally_mean.expand_as(self_team))
        if self.config.context_include_relational:
            parts.append(self._ctx_products(self_team, enemy_weighted, ally_mean))
        h = self.context_head(torch.cat(parts, dim=-1)).squeeze(-1)  # [B, 5]
        return (conf * h).sum(dim=1)

    def _context_logit(
        self,
        identity_context: torch.Tensor,
        identity_context_support: torch.Tensor | None,
    ) -> torch.Tensor:
        """Antisymmetric, support-gated context-atlas contribution.

        ``sum_blue conf*head(self, ally_set, enemy_set, lane_opp, products)
        - sum_red (mirror)`` flips sign under team swap for any head, because the
        per-player support gate, the lane-opponent pairing, and the set summaries
        all swap with the players. Zero-init head keeps it opt-in.
        """
        ctx = self._fit_feature_dim(identity_context, self.config.identity_context_dim)
        blue, red = ctx[:, :5], ctx[:, 5:]
        if identity_context_support is None:
            blue_conf = blue.new_ones(blue.shape[0], 5)
            red_conf = red.new_ones(red.shape[0], 5)
        else:
            s = float(self.config.context_support_strength)
            sup = identity_context_support.clamp_min(0.0)
            conf = sup / (sup + s)
            blue_conf, red_conf = conf[:, :5], conf[:, 5:]
        fwd = self._context_team_score(blue, red, blue_conf)
        rev = self._context_team_score(red, blue, red_conf)
        return fwd - rev

    def _prior_shortcut_logit(
        self,
        *,
        mu_1vx: torch.Tensor,
        delta_logit_2vx: torch.Tensor | None,
        delta_logit_1v1: torch.Tensor | None,
        conf_2vx: torch.Tensor | None,
        conf_1v1: torch.Tensor | None,
    ) -> torch.Tensor:
        identity = _logit(mu_1vx, self.config.logit_clip)
        blue = identity[:, :5]
        red = identity[:, 5:]
        base = torch.cat([blue, red, blue - red], dim=1)
        if delta_logit_2vx is None or delta_logit_1v1 is None:
            raise ValueError("Direct prior shortcut requires relationship delta logits")
        relationship_input = self._shortcut_relationship_features(
            delta_logit_2vx=delta_logit_2vx,
            delta_logit_1v1=delta_logit_1v1,
            conf_2vx=conf_2vx,
            conf_1v1=conf_1v1,
            include_support=False,
        )
        shortcut_input = torch.cat([base, relationship_input], dim=1)
        return self.prior_shortcut(shortcut_input).squeeze(-1)

    def _forward_impl(
        self,
        *,
        champion_id: torch.Tensor,
        build_id: torch.Tensor,
        mu_1vx: torch.Tensor,
        var_1vx: torch.Tensor,
        mu_2vx: torch.Tensor,
        mu_1v1: torch.Tensor,
        identity_semantic: torch.Tensor | None = None,
        identity_profile: torch.Tensor | None = None,
        identity_context: torch.Tensor | None = None,
        identity_context_support: torch.Tensor | None = None,
        identity_context_raw: torch.Tensor | None = None,
        m1v1_detail: torch.Tensor | None = None,
        delta_logit_2vx: torch.Tensor | None = None,
        delta_logit_1v1: torch.Tensor | None = None,
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        missing_1vx: torch.Tensor | None = None,
        conf_2vx: torch.Tensor | None = None,
        log_count_2vx: torch.Tensor | None = None,
        missing_2vx: torch.Tensor | None = None,
        conf_1v1: torch.Tensor | None = None,
        log_count_1v1: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if delta_logit_2vx is None or delta_logit_1v1 is None:
            features = relationship_logit_features(mu_1vx=mu_1vx, mu_2vx=mu_2vx, mu_1v1=mu_1v1)
            delta_logit_2vx = features["delta_logit_2vx"]
            delta_logit_1v1 = features["delta_logit_1v1"]

        # Node init: multiplicative identity (§3) fused with the 1vX posterior.
        h0 = self.identity(champion_id, build_id)
        if identity_semantic is not None and self.identity_semantic_proj is not None:
            h0 = h0 + self.identity_semantic_gate * self.identity_semantic_proj(identity_semantic)
        phi_node = self.phi["1vx"](
            _logit(mu_1vx, self.config.logit_clip),
            var_1vx,
            conf_1vx,
            log_count_1vx,
            missing_1vx,
        )
        h = self.node_norm(self.node_init(torch.cat([h0, phi_node], dim=-1)))

        a = self._readout(h[:, :5])
        b = self._readout(h[:, 5:])
        head_parts = [a, b, a - b, a * b]
        head_parts.append(
            self._residual_readout(
                delta_logit_2vx=delta_logit_2vx,
                delta_logit_1v1=delta_logit_1v1,
                conf_2vx=conf_2vx,
                conf_1v1=conf_1v1,
                missing_2vx=missing_2vx,
                missing_1v1=missing_1v1,
            )
        )
        logit = self.head(torch.cat(head_parts, dim=-1)).squeeze(-1)
        logit = logit + self._prior_shortcut_logit(
            mu_1vx=mu_1vx,
            delta_logit_2vx=delta_logit_2vx,
            delta_logit_1v1=delta_logit_1v1,
            conf_2vx=conf_2vx,
            conf_1v1=conf_1v1,
        )
        if self.detail_enabled:
            detail_logit = self._detail_logit(m1v1_detail, like=logit)
            if self.config.detail_prior_gated:
                prob = torch.sigmoid(logit).detach()
                detail_logit = (4.0 * prob * (1.0 - prob)) * detail_logit
            logit = logit + self.detail_gate * detail_logit
        # Cross-team matchup-profile correction (ungated): moves the prediction
        # along the enemy-composition axis the win-rate prior is blind to.
        if self.profile_enabled and identity_profile is not None:
            logit = logit + self._profile_logit(identity_profile, identity_semantic)
        # Context correction. The identity-conditioned module, when enabled,
        # supplies the residual in place of the shared context head; both are
        # antisymmetric, support-gated, and zero-init opt-in.
        if self.identity_conditioned_context_enabled and identity_context_raw is not None:
            dense = None
            if self.identity_conditioned_context.dense_dim > 0 and identity_context is not None:
                dense = identity_context[..., self.config.context_interpretable_dim :]
            logit = logit + self.identity_conditioned_context(
                identity_context_raw,
                identity_context_support,
                champion_id,
                build_id,
                dense,
            )
        elif self.context_enabled and identity_context is not None:
            logit = logit + self._context_logit(identity_context, identity_context_support)
        return {"final_logit": logit}

    def forward(
        self,
        *,
        champion_id: torch.Tensor,
        build_id: torch.Tensor,
        mu_1vx: torch.Tensor,
        var_1vx: torch.Tensor,
        mu_2vx: torch.Tensor,
        mu_1v1: torch.Tensor,
        identity_semantic: torch.Tensor | None = None,
        identity_profile: torch.Tensor | None = None,
        identity_context: torch.Tensor | None = None,
        identity_context_support: torch.Tensor | None = None,
        identity_context_raw: torch.Tensor | None = None,
        m1v1_detail: torch.Tensor | None = None,
        delta_logit_2vx: torch.Tensor | None = None,
        delta_logit_1v1: torch.Tensor | None = None,
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        missing_1vx: torch.Tensor | None = None,
        conf_2vx: torch.Tensor | None = None,
        log_count_2vx: torch.Tensor | None = None,
        missing_2vx: torch.Tensor | None = None,
        conf_1v1: torch.Tensor | None = None,
        log_count_1v1: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "champion_id": champion_id,
            "build_id": build_id,
            "identity_semantic": identity_semantic,
            "identity_profile": identity_profile,
            "identity_context": identity_context,
            "identity_context_support": identity_context_support,
            "identity_context_raw": identity_context_raw,
            "mu_1vx": mu_1vx,
            "var_1vx": var_1vx,
            "mu_2vx": mu_2vx,
            "mu_1v1": mu_1v1,
            "m1v1_detail": m1v1_detail,
        }
        optional = {
            "delta_logit_2vx": delta_logit_2vx,
            "delta_logit_1v1": delta_logit_1v1,
            "conf_1vx": conf_1vx,
            "log_count_1vx": log_count_1vx,
            "missing_1vx": missing_1vx,
            "conf_2vx": conf_2vx,
            "log_count_2vx": log_count_2vx,
            "missing_2vx": missing_2vx,
            "conf_1v1": conf_1v1,
            "log_count_1v1": log_count_1v1,
            "missing_1v1": missing_1v1,
        }
        inputs.update({key: value for key, value in optional.items() if value is not None})
        return self._forward_impl(**inputs)


def _config_from_payload(payload: dict[str, Any]) -> HGNNConfig:
    raw_config = dict(payload.get("model_config", {}))
    allowed = {field.name for field in fields(HGNNConfig)}
    config_dict = {
        key: value
        for key, value in raw_config.items()
        if key in allowed
    }
    for key in (
        "build_vocab",
        "value_hidden",
        "gate_hidden",
        "node_init_hidden",
        "readout_hidden",
        "residual_head_hidden",
        "profile_head_hidden",
    ):
        if key in config_dict:
            config_dict[key] = tuple(config_dict[key])
    return HGNNConfig(**config_dict)


def save_hgnn_model(path: Path, model: HGNNWinModel, *, confidence_strength: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "hgnn",
            "model_config": asdict(model.config),
            "confidence_strength": float(confidence_strength),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_hgnn_model(path: Path, *, device: str = "cpu") -> tuple[HGNNWinModel, HGNNConfig, float]:
    payload = cast(dict[str, Any], torch.load(path, map_location=device, weights_only=True))
    config = _config_from_payload(payload)
    model = HGNNWinModel(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    strength = float(payload.get("confidence_strength", 30.0))
    return model, config, strength


__all__ = [
    "HGNNConfig",
    "HGNNWinModel",
    "IdentityEncoder",
    "TEAM_PAIRS",
    "build_hgnn_inputs",
    "load_hgnn_model",
    "posterior_mean_var",
    "relationship_logit_features",
    "resolve_device",
    "save_hgnn_model",
    "support_features",
    "swap_hgnn_inputs",
]
