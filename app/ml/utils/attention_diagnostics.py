from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from app.ml.utils.metrics import metric_scalar


def _float_item(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def attention_example_slice(
    value: torch.Tensor,
    max_examples: int | None,
) -> torch.Tensor:
    if max_examples is None or max_examples <= 0:
        return value
    return value[:max_examples]


def _profile_histogram(probs: torch.Tensor) -> torch.Tensor:
    hist = torch.histc(probs.float().clamp(0.0, 1.0), bins=16, min=0.0, max=1.0)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.double().cpu()


def attention_layer_stats(
    attn_probs: torch.Tensor,
    player_token_count: int = 0,
) -> dict[str, object]:
    """Summarise one layer's attention behaviour without perturbing the graph.

    Emits four headline scalars: entropy (attention collapse), head diversity
    (head degeneration), ignored-token fraction (wasted capacity), and player
    mass (right token family). `_profile` is a hidden histogram used for
    epoch-over-epoch drift only.
    """
    with torch.no_grad():
        probs = attn_probs.detach().double()
        seq_len = probs.shape[-1]
        tiny = torch.finfo(torch.float64).tiny
        uniform_mass = 1.0 / max(1, seq_len)
        low_mass_threshold = 0.25 * uniform_mass

        safe_probs = probs.clamp_min(tiny)
        entropy = -(probs * safe_probs.log()).sum(dim=-1)
        token_mass = probs.mean(dim=(0, 1, 2))
        head_profiles = probs.mean(dim=0).flatten(1)

        n_heads = head_profiles.shape[0]
        if n_heads > 1:
            normalised_heads = F.normalize(head_profiles, p=2, dim=1, eps=tiny)
            similarity = normalised_heads @ normalised_heads.T
            head_similarity = (similarity.sum() - similarity.diagonal().sum()) / (
                n_heads * (n_heads - 1)
            )
            head_diversity = 1.0 - head_similarity
        else:
            head_diversity = torch.tensor(float("nan"), device=probs.device)

        stats: dict[str, object] = {
            "attention_entropy_mean": _float_item(entropy.mean()),
            "attention_head_diversity_mean": _float_item(head_diversity),
            "attention_ignored_token_frac": _float_item(
                (token_mass < low_mass_threshold).float().mean()
            ),
            "_profile": _profile_histogram(probs),
        }
        if 0 < player_token_count < seq_len:
            player_end = 1 + player_token_count
            stats["attention_player_mass"] = _float_item(
                token_mass[1:player_end].sum()
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
    """Mean each scalar across layers; pass `_profile` through for drift."""
    if not layer_stats:
        return {}

    summary: dict[str, object] = {}
    public_keys = sorted(
        {key for stats in layer_stats for key in _public_attention_keys(stats)}
    )
    for key in public_keys:
        values = _finite_stat_values(layer_stats, key)
        if values:
            summary[key] = sum(values) / len(values)

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

    def reset(self, keep_profile: bool = True) -> None:
        """Clear accumulated values for a new epoch.

        `keep_profile=True` retains the last profile so `attention_drift_cosine`
        is measured epoch-over-epoch rather than only within a single epoch.
        """
        self._values = {}
        if not keep_profile:
            self._previous_profile = None

    def update(
        self,
        diagnostics: dict[str, object] | None,
        examples: int | None = None,
    ) -> dict[str, float]:
        del examples  # kept for call-site compatibility
        fields = attention_summary_from_metrics(diagnostics)

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
            prev_norm = torch.linalg.vector_norm(self._previous_profile)
            current_norm = torch.linalg.vector_norm(current)
            denom = prev_norm * current_norm
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
        summary: dict[str, float] = {}
        for key, values in self._values.items():
            if values:
                summary[key] = sum(values) / len(values)
        return summary
