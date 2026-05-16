from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from app.ml.utils.metrics import metric_scalar

LAYER_SPREAD_KEYS = {
    "attention_entropy_mean",
    "attention_effective_tokens_mean",
    "attention_head_diversity_mean",
    "attention_head_similarity_mean",
    "attention_max_prob_mean",
}


def _float_item(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def attention_example_slice(
    value: torch.Tensor,
    max_examples: int | None,
) -> torch.Tensor:
    if max_examples is None or max_examples <= 0:
        return value
    return value[:max_examples]


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


def attention_layer_stats(
    attn_probs: torch.Tensor,
    player_token_count: int = 0,
) -> dict[str, object]:
    """Summarise one layer's attention behaviour without perturbing the graph.

    When `player_token_count > 0`, the key sequence is treated as
    `[CLS, players..., (other tokens)...]` and attention mass is split per token
    family. `attention_interaction_mass` is only emitted when tokens beyond the
    players are present.
    """
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
        if 0 < player_token_count < seq_len:
            player_end = 1 + player_token_count
            stats["attention_cls_mass"] = _float_item(token_mass[0:1].sum())
            stats["attention_player_mass"] = _float_item(
                token_mass[1:player_end].sum()
            )
            # Only present when the key sequence carries non-player tokens
            # beyond [CLS, players...].
            if player_end < seq_len:
                stats["attention_interaction_mass"] = _float_item(
                    token_mass[player_end:].sum()
                )
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


def summarise_attention_layers(
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
        if key in LAYER_SPREAD_KEYS and len(values) > 1:
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

    return summary


def attention_summary_from_metrics(
    metrics: dict[str, object] | None,
) -> dict[str, float]:
    if not metrics:
        return {}
    summary: dict[str, float] = {}
    for key, value in metrics.items():
        scalar = metric_scalar(value)
        if key.startswith("attention_") and scalar is not None:
            summary[key] = float(scalar)
    return summary


class AttentionMetricTracker:
    """Aggregate sampled attention diagnostics and estimate drift over time."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = {}
        self._previous_profile: torch.Tensor | None = None
        self._updates = 0
        self._examples = 0

    def reset(self, keep_profile: bool = True) -> None:
        """Clear accumulated values for a new epoch.

        `keep_profile=True` retains the last profile so `attention_drift_*` is
        measured epoch-over-epoch rather than only within a single epoch.
        """
        self._values = {}
        self._updates = 0
        self._examples = 0
        if not keep_profile:
            self._previous_profile = None

    def update(
        self,
        diagnostics: dict[str, object] | None,
        examples: int | None = None,
    ) -> dict[str, float]:
        fields = attention_summary_from_metrics(diagnostics)
        if fields:
            self._updates += 1
            self._examples += max(0, int(examples or 0))

        profile = diagnostics.get("_profile") if diagnostics else None
        if isinstance(profile, torch.Tensor):
            fields.update(self._drift_fields(profile))

        for key, value in fields.items():
            if math.isfinite(value):
                self._values.setdefault(key, []).append(value)
        return fields

    def _drift_fields(self, profile: torch.Tensor) -> dict[str, float]:
        current = profile.detach().float().cpu().flatten()
        fields: dict[str, float] = {}
        if (
            self._previous_profile is not None
            and self._previous_profile.shape == current.shape
        ):
            delta = current - self._previous_profile
            prev_norm = torch.linalg.vector_norm(self._previous_profile)
            current_norm = torch.linalg.vector_norm(current)
            denom = prev_norm * current_norm
            fields["attention_drift_l2"] = float(torch.linalg.vector_norm(delta).item())
            if denom > 0:
                cosine = (
                    torch.dot(current, self._previous_profile)
                    .div(denom)
                    .clamp(-1.0, 1.0)
                    .item()
                )
                fields["attention_drift_cosine"] = float(1.0 - cosine)
        self._previous_profile = current
        return fields

    def summary(self) -> dict[str, float]:
        summary: dict[str, float] = {
            "attention_diagnostic_samples": float(self._examples),
        }
        for key, values in self._values.items():
            if not values:
                continue
            if key.endswith("_max"):
                summary[key] = max(values)
            elif key.endswith("_min"):
                summary[key] = min(values)
            elif key.endswith("_observed"):
                summary[key] = max(values)
            else:
                summary[key] = sum(values) / len(values)
            if key in LAYER_SPREAD_KEYS and len(values) > 1:
                values_tensor = torch.tensor(values, dtype=torch.float64)
                summary[f"{key}_temporal_std"] = float(
                    values_tensor.std(unbiased=False).item()
                )
        return summary
