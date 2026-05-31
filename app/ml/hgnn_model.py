# pyright: reportPrivateImportUsage=false

"""Match-Outcome HGNN win-rate model.

Production currently uses the relationship-direct path: identity embeddings and
the 1vX posterior initialise the 10 player nodes, then direct 1v1/2vX residual
features feed the residual head and prior shortcut. Training and inference share
one model shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

N_RELATIONSHIP_EDGES = 45
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
    used by cache smoothing (`build_dataset._composite_interaction_priors`).
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
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Turn raw cache/prior arrays into the model's node/edge tensors.

    Single source of truth shared by training and the runtime predictor. Accepts
    numpy arrays or tensors. mu values stay in probability space (the model maps
    them to logits internally); champion/build ids become long embedding indices.

    The direct relationship path uses raw build-level confidence/missing flags.
    Nested pooling is already reflected in the cached relationship means.
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
    conf_2vx, _, missing_2vx = support_features(count_2vx, strength)
    conf_1v1, _, missing_1v1 = support_features(count_1v1, strength)
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
        "missing_2vx": missing_2vx,
        "mu_1v1": mu_1v1,
        "conf_1v1": conf_1v1,
        "missing_1v1": missing_1v1,
    }
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

    swapped = {
        "champion_id": swap_halves(inputs["champion_id"], 5),
        "build_id": swap_halves(inputs["build_id"], 5),
        "mu_1vx": swap_halves(inputs["mu_1vx"], 5),
        "var_1vx": swap_halves(inputs["var_1vx"], 5),
        "mu_2vx": swap_halves(inputs["mu_2vx"], 10),
        "mu_1v1": swap_1v1_mu(inputs["mu_1v1"]),
    }
    for suffix, half, prefixes in (
        ("1vx", 5, ("conf", "log_count", "missing")),
        ("2vx", 10, ("conf", "missing")),
    ):
        for prefix in prefixes:
            key = f"{prefix}_{suffix}"
            if key in inputs:
                swapped[key] = swap_halves(inputs[key], half)
    for prefix in ("conf", "missing"):
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


class HGNNWinModel(nn.Module):
    def __init__(self, config: HGNNConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        if config is not None and overrides:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config or HGNNConfig(**overrides)
        c = self.config

        self.phi = nn.ModuleDict({"1vx": PhiEncoder(c)})
        self.identity = IdentityEncoder(c.n_champions, c.n_builds, c.node_dim)
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

    def _prior_shortcut_logit(
        self,
        *,
        mu_1vx: torch.Tensor,
        delta_logit_2vx: torch.Tensor,
        delta_logit_1v1: torch.Tensor,
        conf_2vx: torch.Tensor | None,
        conf_1v1: torch.Tensor | None,
    ) -> torch.Tensor:
        identity = _logit(mu_1vx, self.config.logit_clip)
        blue = identity[:, :5]
        red = identity[:, 5:]
        base = torch.cat([blue, red, blue - red], dim=1)
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
        delta_logit_2vx: torch.Tensor | None = None,
        delta_logit_1v1: torch.Tensor | None = None,
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        missing_1vx: torch.Tensor | None = None,
        conf_2vx: torch.Tensor | None = None,
        missing_2vx: torch.Tensor | None = None,
        conf_1v1: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if delta_logit_2vx is None or delta_logit_1v1 is None:
            features = relationship_logit_features(mu_1vx=mu_1vx, mu_2vx=mu_2vx, mu_1v1=mu_1v1)
            delta_logit_2vx = features["delta_logit_2vx"]
            delta_logit_1v1 = features["delta_logit_1v1"]

        # Node init: multiplicative identity (§3) fused with the 1vX posterior.
        h0 = self.identity(champion_id, build_id)
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
        delta_logit_2vx: torch.Tensor | None = None,
        delta_logit_1v1: torch.Tensor | None = None,
        conf_1vx: torch.Tensor | None = None,
        log_count_1vx: torch.Tensor | None = None,
        missing_1vx: torch.Tensor | None = None,
        conf_2vx: torch.Tensor | None = None,
        missing_2vx: torch.Tensor | None = None,
        conf_1v1: torch.Tensor | None = None,
        missing_1v1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "champion_id": champion_id,
            "build_id": build_id,
            "mu_1vx": mu_1vx,
            "var_1vx": var_1vx,
            "mu_2vx": mu_2vx,
            "mu_1v1": mu_1v1,
        }
        optional = {
            "delta_logit_2vx": delta_logit_2vx,
            "delta_logit_1v1": delta_logit_1v1,
            "conf_1vx": conf_1vx,
            "log_count_1vx": log_count_1vx,
            "missing_1vx": missing_1vx,
            "conf_2vx": conf_2vx,
            "missing_2vx": missing_2vx,
            "conf_1v1": conf_1v1,
            "missing_1v1": missing_1v1,
        }
        inputs.update({key: value for key, value in optional.items() if value is not None})
        return self._forward_impl(**inputs)


def _config_from_payload(payload: dict[str, Any]) -> HGNNConfig:
    allowed = {field.name for field in fields(HGNNConfig)}
    config_dict = {
        key: value
        for key, value in dict(payload.get("model_config", {})).items()
        if key in allowed
    }
    for key in (
        "build_vocab",
        "value_hidden",
        "gate_hidden",
        "node_init_hidden",
        "readout_hidden",
        "residual_head_hidden",
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
