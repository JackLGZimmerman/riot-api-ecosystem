from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from app.ml.config import (
    N_SIDES,
    N_TOKEN_TYPES,
    SIDE_BLUE,
    SIDE_RED,
    TOKEN_TYPE_PLAYER,
    ModelConfig,
)
from app.ml.dataset import InteractionLayout, Vocab


def _float_item(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _attention_example_slice(
    value: torch.Tensor,
    max_examples: int | None,
) -> torch.Tensor:
    if max_examples is None or max_examples <= 0:
        return value
    return value[:max_examples]


def _flat_sample_indices(
    numel: int,
    max_items: int,
    device: torch.device,
) -> torch.Tensor | None:
    if numel <= max_items:
        return None
    return torch.linspace(0, numel - 1, steps=max_items, device=device).long()


def _take_flat_sample(
    value: torch.Tensor,
    indices: torch.Tensor | None,
) -> torch.Tensor:
    flat = value.flatten()
    if indices is None:
        return flat
    return flat.index_select(0, indices)


def _quantiles(
    value: torch.Tensor,
    quantiles: tuple[float, ...],
) -> torch.Tensor:
    flat = value.flatten()
    if flat.numel() == 0:
        return torch.full(
            (len(quantiles),),
            float("nan"),
            dtype=torch.float64,
            device=value.device,
        )
    q = torch.tensor(quantiles, dtype=torch.float64, device=value.device)
    return torch.quantile(flat.double(), q)


def _mass_entropy(mass: torch.Tensor) -> torch.Tensor:
    mass = mass.double()
    total = mass.sum()
    if total <= 0:
        return torch.tensor(float("nan"), dtype=torch.float64, device=mass.device)
    probs = mass / total
    safe_probs = probs.clamp_min(torch.finfo(torch.float64).tiny)
    return -(probs * safe_probs.log()).sum()


def _normalised_mass(
    value: torch.Tensor,
    dims: tuple[int, ...],
) -> torch.Tensor:
    mass = value.double().sum(dim=dims)
    total = mass.sum()
    if total > 0:
        mass = mass / total
    return mass.float().cpu()


def _gini(probs: torch.Tensor) -> torch.Tensor:
    sorted_probs = probs.sort(dim=-1).values
    n = probs.shape[-1]
    weights = torch.arange(1, n + 1, dtype=probs.dtype, device=probs.device)
    total = sorted_probs.sum(dim=-1).clamp_min(torch.finfo(probs.dtype).tiny)
    gini = (2.0 * (sorted_probs * weights).sum(dim=-1) / (n * total)) - ((n + 1.0) / n)
    return gini.clamp_min(0.0)


def _topk_mass(probs: torch.Tensor, k: int) -> torch.Tensor:
    return probs.topk(k=min(k, probs.shape[-1]), dim=-1).values.sum(dim=-1)


def _attention_profile_vector(
    probs: torch.Tensor,
    entropy: torch.Tensor,
    token_mass: torch.Tensor,
    max_prob: torch.Tensor,
) -> torch.Tensor:
    hist = torch.histc(probs.float().clamp(0.0, 1.0), bins=16, min=0.0, max=1.0)
    hist_total = hist.sum()
    if hist_total > 0:
        hist = hist / hist_total

    pieces = [
        hist.double(),
        _quantiles(entropy, (0.1, 0.5, 0.9)),
        _quantiles(token_mass, (0.05, 0.5, 0.95)),
        _quantiles(max_prob, (0.5, 0.9, 0.99)),
    ]
    return torch.cat(pieces).float().cpu()


def _attach_attention_gradient_record(
    stats: dict[str, object],
    raw_probs: torch.Tensor,
    retained_probs: torch.Tensor,
    max_examples: int | None,
    max_gradient_items: int,
    gradient_scale: float,
) -> None:
    if max_gradient_items <= 0 or not retained_probs.requires_grad:
        return

    raw_slice = _attention_example_slice(raw_probs, max_examples)
    retained_slice = _attention_example_slice(retained_probs, max_examples)
    max_items = max(1, max_gradient_items)
    sample_indices = _flat_sample_indices(
        retained_slice.numel(),
        max_items,
        retained_slice.device,
    )
    record: dict[str, torch.Tensor] = {
        "prob_sample": _take_flat_sample(
            retained_slice.detach().float(),
            sample_indices,
        ).cpu(),
        "focus_key_mass": _normalised_mass(raw_slice.detach(), dims=(0, 1, 2)),
    }
    scale = max(float(gradient_scale), torch.finfo(torch.float32).tiny)

    def _capture_attention_grad(grad: torch.Tensor) -> None:
        grad_slice = _attention_example_slice(grad.detach(), max_examples).float()
        grad_slice = grad_slice / scale
        retained = retained_slice.detach().float()
        salience = retained.abs() * grad_slice.abs()
        record["grad_sample"] = _take_flat_sample(grad_slice, sample_indices).cpu()
        record["salience_sample"] = _take_flat_sample(salience, sample_indices).cpu()
        record["key_salience"] = _normalised_mass(salience, dims=(0, 1, 2))
        record["query_salience"] = _normalised_mass(salience, dims=(0, 1, 3))
        record["head_salience"] = _normalised_mass(salience, dims=(0, 2, 3))

    retained_probs.register_hook(_capture_attention_grad)
    stats["_attention_gradient_record"] = record


def _attention_layer_stats(
    attn_probs: torch.Tensor,
    retained_attn_probs: torch.Tensor | None = None,
) -> dict[str, object]:
    """Summarise one layer's attention behaviour without perturbing the graph."""
    with torch.no_grad():
        probs = attn_probs.detach().double()
        seq_len = probs.shape[-1]
        tiny = torch.finfo(torch.float64).tiny
        uniform_mass = 1.0 / max(1, seq_len)
        low_mass_threshold = 0.25 * uniform_mass

        safe_probs = probs.clamp_min(tiny)
        entropy = -(probs * safe_probs.log()).sum(dim=-1)
        effective_tokens = entropy.exp()
        max_prob = probs.max(dim=-1).values
        top5_mass = _topk_mass(probs, 5)
        token_mass = probs.mean(dim=(0, 1, 2))
        head_profiles = probs.mean(dim=0).flatten(1)
        max_q95 = _quantiles(max_prob, (0.95,))[0]

        n_heads = head_profiles.shape[0]
        if n_heads > 1:
            normalised_heads = F.normalize(head_profiles, p=2, dim=1, eps=tiny)
            similarity = normalised_heads @ normalised_heads.T
            head_similarity = (similarity.sum() - similarity.diagonal().sum()) / (
                n_heads * (n_heads - 1)
            )
            head_diversity = 1.0 - head_similarity
        else:
            head_similarity = torch.tensor(float("nan"), device=probs.device)
            head_diversity = torch.tensor(float("nan"), device=probs.device)

        stats: dict[str, object] = {
            "attention_entropy_mean": _float_item(entropy.mean()),
            "attention_entropy_std": _float_item(entropy.std(unbiased=False)),
            "attention_effective_tokens_mean": _float_item(effective_tokens.mean()),
            "attention_head_similarity_mean": _float_item(head_similarity),
            "attention_head_diversity_mean": _float_item(head_diversity),
            "attention_max_prob_mean": _float_item(max_prob.mean()),
            "attention_max_prob_p95": _float_item(max_q95),
            "attention_top5_mass_mean": _float_item(top5_mass.mean()),
            "attention_token_utilization": _float_item(
                (token_mass >= low_mass_threshold).float().mean()
            ),
            "attention_ignored_token_frac": _float_item(
                (token_mass < low_mass_threshold).float().mean()
            ),
            "_profile": _attention_profile_vector(
                probs,
                entropy,
                token_mass,
                max_prob,
            ),
        }
        return stats


def _public_attention_keys(stats: dict[str, object]) -> list[str]:
    return [key for key in stats if not key.startswith("_")]


def _finite_stat_values(
    layer_stats: list[dict[str, object]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for stats in layer_stats:
        value = stats.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


_LAYER_SPREAD_KEYS = {
    "attention_entropy_mean",
    "attention_effective_tokens_mean",
    "attention_head_diversity_mean",
    "attention_head_similarity_mean",
    "attention_max_prob_mean",
}


def _summarise_attention_layers(
    layer_stats: list[dict[str, object]],
) -> dict[str, object]:
    if not layer_stats:
        return {"attention_layers_observed": 0}

    summary: dict[str, object] = {"attention_layers_observed": len(layer_stats)}
    public_keys = sorted(
        {key for stats in layer_stats for key in _public_attention_keys(stats)}
    )
    for key in public_keys:
        values = _finite_stat_values(layer_stats, key)
        if not values:
            continue
        if key.endswith("_max"):
            summary[key] = max(values)
        elif key.endswith("_min"):
            summary[key] = min(values)
        else:
            summary[key] = sum(values) / len(values)
        if key in _LAYER_SPREAD_KEYS and len(values) > 1:
            values_tensor = torch.tensor(values, dtype=torch.float64)
            suffix = key.removeprefix("attention_")
            summary[f"attention_layer_{suffix}_std"] = _float_item(
                values_tensor.std(unbiased=False)
            )
            summary[f"attention_layer_{suffix}_range"] = max(values) - min(values)

    first = layer_stats[0]
    last = layer_stats[-1]
    for label, stats in (("first", first), ("last", last)):
        for key in (
            "attention_entropy_mean",
            "attention_effective_tokens_mean",
            "attention_head_diversity_mean",
            "attention_head_similarity_mean",
            "attention_max_prob_mean",
            "attention_token_utilization",
        ):
            value = stats.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                summary[f"attention_{label}_layer_{key.removeprefix('attention_')}"] = (
                    float(value)
                )

    profiles = [
        profile
        for profile in (stats.get("_profile") for stats in layer_stats)
        if isinstance(profile, torch.Tensor)
    ]
    if len(profiles) == len(layer_stats):
        summary["_profile"] = torch.cat(
            [profile.flatten().float() for profile in profiles]
        )

    gradient_records = [
        record
        for record in (stats.get("_attention_gradient_record") for stats in layer_stats)
        if isinstance(record, dict)
    ]
    if gradient_records:
        summary["_attention_gradient_records"] = gradient_records

    return summary


class _DiagnosticEncoderLayer(nn.Module):
    """TransformerEncoderLayer-equivalent layer with optional attention stats.

    Normal training uses PyTorch's MultiheadAttention SDPA fast path; diagnostic
    collection keeps the manual attention path so attention maps remain
    inspectable.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        attention_dropout: float,
        head_dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.head_dropout = float(head_dropout)

    def _apply_head_dropout(self, attn_probs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.head_dropout <= 0.0:
            return attn_probs
        keep_prob = 1.0 - self.head_dropout
        keep = torch.rand(
            attn_probs.shape[0],
            attn_probs.shape[1],
            1,
            1,
            device=attn_probs.device,
        ) < keep_prob
        return attn_probs * keep.to(attn_probs.dtype) / keep_prob

    def _self_attention_manual(
        self,
        x: torch.Tensor,
        collect_attention_diagnostics: bool,
        retain_attention_grad: bool,
        attention_diagnostics_sample_size: int | None,
        attention_gradient_sample_size: int,
        attention_gradient_scale: float,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        batch, seq_len, d_model = x.shape
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = attn_logits.softmax(dim=-1)
        retained_attn_probs = F.dropout(
            attn_probs,
            p=float(self.self_attn.dropout),
            training=self.training,
        )
        retained_attn_probs = self._apply_head_dropout(retained_attn_probs)
        stats = None
        if collect_attention_diagnostics:
            stats = _attention_layer_stats(
                _attention_example_slice(attn_probs, attention_diagnostics_sample_size),
                _attention_example_slice(
                    retained_attn_probs,
                    attention_diagnostics_sample_size,
                ),
            )
            if retain_attention_grad:
                _attach_attention_gradient_record(
                    stats,
                    attn_probs,
                    retained_attn_probs,
                    attention_diagnostics_sample_size,
                    attention_gradient_sample_size,
                    attention_gradient_scale,
                )

        out = retained_attn_probs @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.self_attn.out_proj(out), stats

    def _sa_block(
        self,
        x: torch.Tensor,
        collect_attention_diagnostics: bool,
        retain_attention_grad: bool,
        attention_diagnostics_sample_size: int | None,
        attention_gradient_sample_size: int,
        attention_gradient_scale: float,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        use_manual_attention = collect_attention_diagnostics or (
            self.training and self.head_dropout > 0.0
        )
        if use_manual_attention:
            out, stats = self._self_attention_manual(
                x,
                collect_attention_diagnostics,
                retain_attention_grad,
                attention_diagnostics_sample_size,
                attention_gradient_sample_size,
                attention_gradient_scale,
            )
        else:
            out = self.self_attn(x, x, x, need_weights=False)[0]
            stats = None
        return self.dropout1(out), stats

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.gelu(self.linear1(x), approximate="tanh")))
        return self.dropout2(x)

    def forward(
        self,
        x: torch.Tensor,
        collect_attention_diagnostics: bool = False,
        retain_attention_grad: bool = False,
        attention_diagnostics_sample_size: int | None = None,
        attention_gradient_sample_size: int = 200_000,
        attention_gradient_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        sa_out, stats = self._sa_block(
            self.norm1(x),
            collect_attention_diagnostics,
            retain_attention_grad,
            attention_diagnostics_sample_size,
            attention_gradient_sample_size,
            attention_gradient_scale,
        )
        x = x + sa_out
        x = x + self._ff_block(self.norm2(x))
        return x, stats


class _DiagnosticEncoder(nn.Module):
    """TransformerEncoder-equivalent with optional per-layer diagnostics."""

    def __init__(
        self,
        layer: _DiagnosticEncoderLayer,
        n_layers: int,
    ):
        super().__init__()
        from copy import deepcopy

        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        collect_attention_diagnostics: bool = False,
        retain_attention_grad: bool = False,
        attention_diagnostics_sample_size: int | None = None,
        attention_gradient_sample_size: int = 200_000,
        attention_gradient_scale: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        layer_stats: list[dict[str, object]] = []
        for layer in self.layers:
            x, stats = layer(
                x,
                collect_attention_diagnostics=collect_attention_diagnostics,
                retain_attention_grad=retain_attention_grad,
                attention_diagnostics_sample_size=attention_diagnostics_sample_size,
                attention_gradient_sample_size=attention_gradient_sample_size,
                attention_gradient_scale=attention_gradient_scale,
            )
            if stats is not None:
                layer_stats.append(stats)
        if collect_attention_diagnostics:
            return x, _summarise_attention_layers(layer_stats)
        return x


class HybridTokenModel(nn.Module):
    """Hybrid champion + interaction transformer.

    The model is PyTorch-native so current CUDA wheels can select optimized
    scaled-dot-product attention kernels for NVIDIA Blackwell GPUs.
    """

    def __init__(
        self,
        vocab: Vocab,
        layout: InteractionLayout,
        cfg: ModelConfig,
    ):
        super().__init__()
        d = cfg.d_model

        self.champ_emb = nn.Embedding(vocab.n_champions, d)
        self.role_emb = nn.Embedding(vocab.n_roles, d)
        self.build_emb = nn.Embedding(vocab.n_builds, d)
        self.side_emb = nn.Embedding(N_SIDES, d)
        self.type_emb = nn.Embedding(N_TOKEN_TYPES, d)
        self.score_proj = nn.Linear(1, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.register_buffer(
            "_player_side",
            torch.tensor([[SIDE_BLUE] * 5 + [SIDE_RED] * 5], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_player_type",
            torch.full((1, 10), TOKEN_TYPE_PLAYER, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_interaction_types", layout.types.unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "_interaction_sides", layout.sides.unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "_interaction_roles", layout.roles.unsqueeze(0), persistent=False
        )

        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(cfg.dropout)

        encoder_layer = _DiagnosticEncoderLayer(
            d_model=d,
            n_heads=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            head_dropout=cfg.head_dropout,
        )
        self.encoder = _DiagnosticEncoder(
            encoder_layer,
            cfg.n_layers,
        )

        self.pooling = cfg.pooling
        self.attention_pool = (
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1, bias=False))
            if cfg.pooling == "attention"
            else None
        )
        self.pool_gate = (
            nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.Sigmoid())
            if cfg.pooling == "gated"
            else None
        )
        head_input_dim = d * 2 if cfg.pooling == "concat_cls_mean" else d
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, cfg.head_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def _player_tokens(
        self,
        champion_idx: torch.Tensor,
        role_idx: torch.Tensor,
        build_idx: torch.Tensor,
    ) -> torch.Tensor:
        b = champion_idx.shape[0]
        side_idx = self._player_side.expand(b, -1)
        type_idx = self._player_type.expand(b, -1)
        return (
            self.champ_emb(champion_idx)
            + self.role_emb(role_idx)
            + self.build_emb(build_idx)
            + self.side_emb(side_idx)
            + self.type_emb(type_idx)
        )

    def _interaction_tokens(
        self,
        score: torch.Tensor,
    ) -> torch.Tensor:
        b = score.shape[0]
        types = self._interaction_types.expand(b, -1)
        sides = self._interaction_sides.expand(b, -1)
        roles = self._interaction_roles.expand(b, -1, -1)

        type_e = self.type_emb(types)
        side_e = self.side_emb(sides)
        role_e = self.role_emb(roles).sum(dim=2)

        score_e = self.score_proj(score.unsqueeze(-1))
        return type_e + side_e + role_e + score_e

    def _pool_encoded(self, encoded: torch.Tensor) -> torch.Tensor:
        cls = encoded[:, 0]
        tokens = encoded[:, 1:]
        mean = tokens.mean(dim=1)
        if self.pooling == "cls":
            return cls
        if self.pooling == "mean":
            return mean
        if self.pooling == "concat_cls_mean":
            return torch.cat([cls, mean], dim=-1)
        if self.pooling == "attention":
            if self.attention_pool is None:
                raise RuntimeError("attention pooling module is not initialised")
            weights = self.attention_pool(tokens).squeeze(-1).softmax(dim=-1)
            return torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        if self.pooling == "gated":
            if self.pool_gate is None:
                raise RuntimeError("gated pooling module is not initialised")
            gate = self.pool_gate(torch.cat([cls, mean], dim=-1))
            return gate * cls + (1.0 - gate) * mean
        raise RuntimeError(f"Unsupported pooling mode: {self.pooling}")

    def forward(
        self,
        champion_idx: torch.Tensor,
        role_idx: torch.Tensor,
        build_idx: torch.Tensor,
        interaction_score: torch.Tensor,
        return_attention_diagnostics: bool = False,
        retain_attention_grad: bool = False,
        attention_diagnostics_sample_size: int | None = None,
        attention_gradient_sample_size: int = 200_000,
        attention_gradient_scale: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        player = self._player_tokens(champion_idx, role_idx, build_idx)
        interaction = self._interaction_tokens(interaction_score)
        b = player.shape[0]

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, player, interaction], dim=1)
        x = self.input_dropout(self.input_norm(x))
        encoded = self.encoder(
            x,
            collect_attention_diagnostics=return_attention_diagnostics,
            retain_attention_grad=retain_attention_grad,
            attention_diagnostics_sample_size=attention_diagnostics_sample_size,
            attention_gradient_sample_size=attention_gradient_sample_size,
            attention_gradient_scale=attention_gradient_scale,
        )
        if return_attention_diagnostics:
            z, attention_diagnostics = encoded
        else:
            z = encoded
            attention_diagnostics = {}
        logits = self.head(self._pool_encoded(z)).squeeze(-1)
        if return_attention_diagnostics:
            return logits, attention_diagnostics
        return logits


__all__ = ["HybridTokenModel"]
