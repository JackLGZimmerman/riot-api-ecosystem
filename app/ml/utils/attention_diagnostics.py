"""Attention diagnostics for HybridTokenModel.

Per-layer summaries (entropy, head diversity, drift, ignored-token fraction)
plus League-specific relation diagnostics: the fraction of attention each
query routes to keys in named token-pair relationships defined by the fixed
10-token draft layout.

Token layout (must match HybridTokenModel._player_side and the dataset role
ordering):
    0-4: blue side, role order TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
    5-9: red side, same role order

Each `attention_relation_<name>` is the mean per-query attention mass on
that relation, averaged across batch / heads / queries that have at least
one matching key. Because softmaxed attention sums to 1 per query, values
are fractions in [0, 1] and directly comparable across layers and runs.

Uniform-attention baselines (a relation is favoured over random when its
metric exceeds the baseline):
    self 0.10        same_team 0.40         enemy_team 0.50
    same_role_enemy 0.10         cross_role_enemy 0.40
    bot_duo_ally 0.10            jungle_lane_ally 0.40
    jungle_lane_enemy 0.40

Headline interpretive cues:
    self              high (>0.4) -> layer acting near-identity (bad).
                      0.1-0.3 healthy.
    same_team         team-level composition (synergy, scaling, win
                      conditions). Tends to grow with depth.
    enemy_team        matchup / counter-pick reasoning.
    same_role_enemy   direct 1v1 lane matchup. Often peaks early since
                      lane matchups are the most local League signal.
    cross_role_enemy  global enemy interactions (roams, teamfight matchups,
                      pick comps). Tends to grow with depth.
    bot_duo_ally      ADC <-> SUP synergy. A head dedicated here is a
                      strong League-specific signal; often shallow/middle.
    jungle_lane_ally  jungle reading own lanes (gank planning, macro).
    jungle_lane_enemy jungle reading enemy lanes (counter-jungle reads).

Per-head metrics (`attention_head_<h>_<relation>`, averaged across layers
in the summary) surface specialisation. Healthy: some heads concentrated
on lane-matchup relations, others fanning out across team-level relations.
Identical per-head profiles -> head diversity collapsed; cross-check
`attention_head_diversity_mean`.

Heuristic layer-depth pattern (rough, not a contract):
    layer 1: same_role_enemy + bot_duo_ally dominate (local lane structure)
    layer 2: jungle_lane_* + cross_role_enemy rise (structural reasoning)
    layer 3: same_team / enemy_team spread, self-mass usually drops as the
             residual stream is reorganised for the head

In training, drift matters more than absolute levels: a sudden collapse to
self-dominance or single-relation saturation is the cleanest degeneration
signal.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from app.ml.utils.metrics import metric_scalar

# Fixed 10-token draft layout. Must match HybridTokenModel._player_side and
# the dataset's role index ordering. Relation diagnostics short-circuit for
# other sequence lengths since the relation taxonomy is meaningless without
# this exact layout.
PLAYER_ROLE_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4)
PLAYER_SIDE_ORDER: tuple[int, ...] = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
ROLE_TOP, ROLE_JUNGLE, ROLE_MID, ROLE_BOT, ROLE_SUP = 0, 1, 2, 3, 4
EXPECTED_SEQ_LEN = 10

_RELATION_MASK_CACHE: dict[tuple[torch.device, int], dict[str, torch.Tensor]] = {}


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


def build_player_relation_masks(
    seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Boolean (query, key) masks for League-specific token-pair relationships.

    Layout is fixed: tokens 0-4 = blue side, 5-9 = red side, both in role
    order (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY). Each mask is [seq_len,
    seq_len] with `mask[q, k]` True iff the (query q, key k) pair belongs
    to the named relation. Masks are cached per (device, seq_len) - the
    diagnostic path is hot enough to merit not rebuilding them, and they
    are tiny so the cache is essentially free.

    Returns {} for `seq_len != EXPECTED_SEQ_LEN` so callers using other
    layouts degrade silently to no relation diagnostics.

    Asymmetry choices:
      - `bot_duo_ally` is bidirectional ({ADC, SUP} on same team) because
        the bot duo functions as a pair regardless of direction.
      - `jungle_lane_{ally,enemy}` treat the jungle as the query and lanes
        as keys. This isolates "what is the jungle reading from lanes",
        which is the most directly interpretable framing for League pathing.
    """
    cache_key = (device, seq_len)
    cached = _RELATION_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if seq_len != EXPECTED_SEQ_LEN:
        _RELATION_MASK_CACHE[cache_key] = {}
        return {}

    role = torch.tensor(PLAYER_ROLE_ORDER, device=device, dtype=torch.long)
    side = torch.tensor(PLAYER_SIDE_ORDER, device=device, dtype=torch.long)
    q_role = role.unsqueeze(1).expand(seq_len, seq_len)
    k_role = role.unsqueeze(0).expand(seq_len, seq_len)
    q_side = side.unsqueeze(1).expand(seq_len, seq_len)
    k_side = side.unsqueeze(0).expand(seq_len, seq_len)

    eye = torch.eye(seq_len, device=device, dtype=torch.bool)
    same_side = q_side == k_side
    same_role = q_role == k_role
    same_team = same_side & ~eye
    enemy_team = ~same_side

    bot_or_sup_q = (q_role == ROLE_BOT) | (q_role == ROLE_SUP)
    bot_or_sup_k = (k_role == ROLE_BOT) | (k_role == ROLE_SUP)
    is_jungle_q = q_role == ROLE_JUNGLE
    is_laner_k = (
        (k_role == ROLE_TOP)
        | (k_role == ROLE_MID)
        | (k_role == ROLE_BOT)
        | (k_role == ROLE_SUP)
    )

    masks = {
        "self": eye,
        "same_team": same_team,
        "enemy_team": enemy_team,
        "same_role_enemy": enemy_team & same_role,
        "cross_role_enemy": enemy_team & ~same_role,
        "bot_duo_ally": same_team & bot_or_sup_q & bot_or_sup_k & ~same_role,
        "jungle_lane_ally": same_team & is_jungle_q & is_laner_k,
        "jungle_lane_enemy": enemy_team & is_jungle_q & is_laner_k,
    }
    _RELATION_MASK_CACHE[cache_key] = masks
    return masks


def relation_attention_mass(
    attn_probs: torch.Tensor,
    masks: dict[str, torch.Tensor] | None = None,
    per_head: bool = True,
) -> dict[str, float]:
    """Mean attention mass that queries route to each relationship group.

    For every relation, for every query position that has at least one
    matching key, sum the softmaxed attention over those keys and average
    across (batch, heads, valid_queries). Because attention sums to 1 per
    query, results are interpretable as "fraction of attention valid
    queries spend on this relation" - comparable across layers and runs.

    `per_head=True` also emits per-head averages (still averaged over batch
    and valid queries) under keys `attention_head_<h>_<relation>`. These
    are the dominant signal for head specialisation: if one head's
    `attention_head_<h>_bot_duo_ally` sits far above the rest, that head
    has learned the bot-duo edge.

    Relations whose mask has no valid query position are skipped so the
    output never contains 0/0 NaNs. With the fixed 10-token layout this is
    defensive only.
    """
    if attn_probs.dim() != 4:
        return {}
    if masks is None:
        masks = build_player_relation_masks(attn_probs.shape[-1], attn_probs.device)
    if not masks:
        return {}

    probs = attn_probs.detach().double()
    n_heads = probs.shape[1]
    results: dict[str, float] = {}
    for name, mask in masks.items():
        per_query_mass = (probs * mask).sum(dim=-1)  # [B, H, Q]
        valid_query = mask.any(dim=-1)  # [Q]
        if not bool(valid_query.any()):
            continue
        valid_mass = per_query_mass[..., valid_query]  # [B, H, n_valid]
        results[f"attention_relation_{name}"] = float(valid_mass.mean().item())
        if per_head and n_heads > 0:
            head_mean = valid_mass.mean(dim=(0, 2))  # [H]
            for head_idx, value in enumerate(head_mean.tolist()):
                results[f"attention_head_{head_idx}_{name}"] = float(value)
    return results


def attention_layer_stats(
    attn_probs: torch.Tensor,
    player_token_count: int = 0,
) -> dict[str, object]:
    """Summarise one layer's attention behaviour without perturbing the graph.

    Emits four generic headline scalars: entropy (attention collapse), head
    diversity (head degeneration), ignored-token fraction (wasted capacity),
    and player mass (right token family). Also emits League-specific
    `attention_relation_<name>` and per-head `attention_head_<h>_<name>`
    fractions (see module docstring for the relation taxonomy and
    interpretation guide). `_profile` is a hidden histogram used for
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

        # League-specific relation mass + per-head breakdown; no-op for any
        # layout other than the fixed 10-token draft.
        stats.update(
            relation_attention_mass(
                probs,
                masks=build_player_relation_masks(seq_len, probs.device),
                per_head=True,
            )
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
    ) -> dict[str, float]:
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
