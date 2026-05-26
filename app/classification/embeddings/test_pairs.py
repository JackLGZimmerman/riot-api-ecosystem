"""Curated semantic pair test set.

Each entry is a pair of `(champion_name, teamposition, build)` identities that
domain knowledge says *should* land in the same similarity neighbourhood. The
diagnostic measures, for a fixed embedding:

  * mean / min cosine similarity over the pair set
  * same-group rate (both members in the same agglomerative cluster at the
    current threshold)
  * the resulting "semantic threshold" — highest threshold at which >= 75% of
    the pairs are still grouped together

These are *expected* clusters, not ground truth. Pairs reflect either shared
scaling identity (e.g. infinite-stack scalers Aurelion Sol + Smolder), shared
role/build archetypes (AP MID vs AP BOT), or shared engage/peel kits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.classification.embeddings.embed import LevelEmbeddings
from app.classification.embeddings.similarity import (
    cosine_similarity_matrix,
    group_by_threshold,
)
from app.core.config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

CHAMPION_NAMES_PATH = (
    PROJECT_ROOT / "database" / "clickhouse" / "support" / "championid_name_map.jsonl"
)

# (champion_name, teamposition, build)
IdentityKey = tuple[str, str, str]
PairList = tuple[tuple[IdentityKey, IdentityKey], ...]


# Pairs that should cluster — grouped by the semantic reason.
SEMANTIC_PAIRS: dict[str, PairList] = {
    "scaling_carries": (
        # Late-game infinite-scalers across role/build axes.
        (("AurelionSol", "MIDDLE", "ability_power"), ("Smolder", "BOTTOM", "crit")),
        (("Veigar", "MIDDLE", "ability_power"), ("Kassadin", "MIDDLE", "ability_power")),
        (("Veigar", "MIDDLE", "ability_power"), ("Nasus", "TOP", "ad_off_tank")),
        (("AurelionSol", "MIDDLE", "ability_power"), ("Kassadin", "MIDDLE", "ability_power")),
        (("Smolder", "BOTTOM", "crit"), ("Kayle", "TOP", "on_hit")),
        (("AurelionSol", "MIDDLE", "ability_power"), ("Kayle", "TOP", "on_hit")),
    ),
    "ap_mage_cross_role": (
        # AP carries in MID and BOT should cluster together when the build is AP.
        (("Ahri", "MIDDLE", "ability_power"), ("Lux", "MIDDLE", "ability_power")),
        (("Lux", "MIDDLE", "ability_power"), ("Lux", "UTILITY", "ability_power")),
        (("Ziggs", "MIDDLE", "ability_power"), ("Ziggs", "BOTTOM", "ability_power")),
        (("Seraphine", "MIDDLE", "ability_power"), ("Seraphine", "BOTTOM", "ability_power")),
        (("Velkoz", "MIDDLE", "ability_power"), ("Velkoz", "UTILITY", "ability_power")),
        (("AurelionSol", "MIDDLE", "ability_power"), ("Anivia", "MIDDLE", "ability_power")),
        (("Swain", "MIDDLE", "ability_power"), ("Swain", "BOTTOM", "ability_power")),
    ),
    "adc_carries": (
        # Crit / lethality AD carries on BOT.
        (("Jinx", "BOTTOM", "crit"), ("Caitlyn", "BOTTOM", "crit")),
        (("Jinx", "BOTTOM", "crit"), ("Sivir", "BOTTOM", "crit")),
        (("Caitlyn", "BOTTOM", "crit"), ("Ashe", "BOTTOM", "crit")),
        (("Draven", "BOTTOM", "crit"), ("Jhin", "BOTTOM", "lethality")),
        (("MissFortune", "BOTTOM", "crit"), ("Jinx", "BOTTOM", "crit")),
    ),
    "on_hit_carries": (
        # Attack-speed/on-hit ranged carries — different stat profile from crit.
        (("Vayne", "TOP", "on_hit"), ("Irelia", "TOP", "on_hit")),
        (("Vayne", "TOP", "on_hit"), ("Jax", "TOP", "on_hit")),
        (("KogMaw", "TOP", "on_hit"), ("Vayne", "TOP", "on_hit")),
    ),
    "ad_juggernauts": (
        # AD frontline bruisers/juggernauts in TOP.
        (("Aatrox", "TOP", "attack_damage"), ("Darius", "TOP", "attack_damage")),
        (("Darius", "TOP", "attack_damage"), ("Garen", "TOP", "attack_damage")),
        (("Garen", "TOP", "attack_damage"), ("Sett", "TOP", "attack_damage")),
        (("Aatrox", "TOP", "attack_damage"), ("Sett", "TOP", "attack_damage")),
        (("Mordekaiser", "TOP", "ap_off_tank"), ("Sion", "TOP", "ar_tank")),
    ),
    "front_line_tanks": (
        # Engage/peel tanks — high mitigation, low DPS.
        (("Sion", "TOP", "ar_tank"), ("Ornn", "TOP", "ar_tank")),
        (("Ornn", "TOP", "ar_tank"), ("Maokai", "TOP", "ar_tank")),
        (("Chogath", "TOP", "ar_tank"), ("Sion", "TOP", "ar_tank")),
        (("Malphite", "TOP", "ar_tank"), ("Ornn", "TOP", "ar_tank")),
        (("Nautilus", "UTILITY", "ar_tank"), ("Leona", "UTILITY", "ar_tank")),
    ),
    "enchanter_supports": (
        # Heal/shield UTILITY enchanters — should NOT mix with engage tanks.
        (("Lulu", "UTILITY", "utility_enchanter"), ("Janna", "UTILITY", "utility_enchanter")),
        (("Janna", "UTILITY", "utility_enchanter"), ("Soraka", "UTILITY", "utility_enchanter")),
        (("Lulu", "UTILITY", "utility_enchanter"), ("Nami", "UTILITY", "utility_enchanter")),
        (("Yuumi", "UTILITY", "utility_enchanter"), ("Soraka", "UTILITY", "utility_enchanter")),
        (("Karma", "UTILITY", "utility_enchanter"), ("Lulu", "UTILITY", "utility_enchanter")),
    ),
    "lethality_assassins": (
        # Lethality burst — should pull together regardless of role.
        (("Zed", "MIDDLE", "lethality"), ("Talon", "MIDDLE", "lethality")),
        (("Leblanc", "MIDDLE", "lethality"), ("Zed", "MIDDLE", "lethality")),
        (("Talon", "MIDDLE", "lethality"), ("Qiyana", "MIDDLE", "lethality")),
        (("Pantheon", "MIDDLE", "lethality"), ("Talon", "MIDDLE", "lethality")),
    ),
    "skirmisher_fighters": (
        # Sustained-DPS melee skirmishers.
        (("Yasuo", "MIDDLE", "crit"), ("Yone", "MIDDLE", "crit")),
        (("Yasuo", "MIDDLE", "crit"), ("Tryndamere", "TOP", "crit")),
        (("Irelia", "TOP", "on_hit"), ("Jax", "TOP", "on_hit")),
        (("Camille", "TOP", "attack_damage"), ("Jax", "TOP", "attack_damage")),
        (("Riven", "TOP", "attack_damage"), ("Camille", "TOP", "attack_damage")),
    ),
    "top_apoffenders": (
        # AP top-lane bruisers / AP off-tanks.
        (("Lissandra", "TOP", "ap_off_tank"), ("Viktor", "TOP", "ap_off_tank")),
        (("Teemo", "TOP", "ability_power"), ("Singed", "TOP", "ability_power")),
        (("Singed", "TOP", "ability_power"), ("Swain", "TOP", "ability_power")),
        (("Mordekaiser", "TOP", "ap_off_tank"), ("Swain", "TOP", "ap_off_tank")),
    ),
    "engage_supports": (
        # Hook / engage UTILITY tanks — pull each other, not enchanters.
        (("Blitzcrank", "UTILITY", "utility_protection"), ("Thresh", "UTILITY", "utility_protection")),
        (("Nautilus", "UTILITY", "ar_tank"), ("Leona", "UTILITY", "ar_tank")),
        (("Alistar", "UTILITY", "ar_tank"), ("Nautilus", "UTILITY", "ar_tank")),
        (("Braum", "UTILITY", "ar_tank"), ("Alistar", "UTILITY", "ar_tank")),
    ),
    "jungle_bruisers": (
        # AD/bruiser jungle clearers.
        (("LeeSin", "JUNGLE", "attack_damage"), ("Viego", "JUNGLE", "attack_damage")),
        (("Viego", "JUNGLE", "attack_damage"), ("Rengar", "JUNGLE", "attack_damage")),
        (("Nocturne", "JUNGLE", "attack_damage"), ("Viego", "JUNGLE", "attack_damage")),
        (("Udyr", "JUNGLE", "ad_off_tank"), ("Volibear", "JUNGLE", "ar_tank")),
    ),
}


# Pairs that should NOT cluster — sanity checks that the embedding isn't
# collapsing distinct archetypes onto each other. Each pair is across an
# archetype boundary that domain knowledge says is *not* an alias.
ANTI_PAIRS: PairList = (
    # Enchanter support vs engage tank support — different stat profile.
    (("Lulu", "UTILITY", "utility_enchanter"), ("Leona", "UTILITY", "ar_tank")),
    (("Janna", "UTILITY", "utility_enchanter"), ("Nautilus", "UTILITY", "ar_tank")),
    (("Soraka", "UTILITY", "utility_enchanter"), ("Alistar", "UTILITY", "ar_tank")),
    # Crit ADC vs AP mage on the same row position — distinct damage type.
    (("Jinx", "BOTTOM", "crit"), ("Ziggs", "BOTTOM", "ability_power")),
    (("Caitlyn", "BOTTOM", "crit"), ("Seraphine", "BOTTOM", "ability_power")),
    (("Draven", "BOTTOM", "crit"), ("Swain", "BOTTOM", "ability_power")),
    # AP mage MID vs lethality assassin MID — different damage delivery.
    (("Lux", "MIDDLE", "ability_power"), ("Zed", "MIDDLE", "lethality")),
    (("Veigar", "MIDDLE", "ability_power"), ("Talon", "MIDDLE", "lethality")),
    (("Anivia", "MIDDLE", "ability_power"), ("Qiyana", "MIDDLE", "lethality")),
    # Tank TOP vs crit ADC TOP — opposite stat profiles.
    (("Ornn", "TOP", "ar_tank"), ("Caitlyn", "TOP", "crit")),
    (("Sion", "TOP", "ar_tank"), ("Vayne", "TOP", "on_hit")),
    (("Maokai", "TOP", "ar_tank"), ("Tryndamere", "TOP", "crit")),
    # Jungle bruiser vs UTILITY enchanter — totally different role/stat profile.
    (("LeeSin", "JUNGLE", "attack_damage"), ("Lulu", "UTILITY", "utility_enchanter")),
    (("Viego", "JUNGLE", "attack_damage"), ("Janna", "UTILITY", "utility_enchanter")),
    # AD juggernaut vs ranged crit ADC — same lane but very different combat.
    (("Garen", "TOP", "attack_damage"), ("Caitlyn", "TOP", "crit")),
    (("Darius", "TOP", "attack_damage"), ("Vayne", "TOP", "crit")),
)


@dataclass(frozen=True)
class SemanticScore:
    n_pairs_resolved: int
    n_pairs_total: int
    mean_sim: float
    p50_sim: float
    min_sim: float
    same_group_rate: float  # fraction at cfg.similarity_threshold
    by_category_same_group: dict[str, tuple[int, int]]  # category -> (hit, total)
    anti_mean_sim: float
    anti_same_group_rate: float
    separation: float  # mean_sim - anti_mean_sim
    semantic_threshold_75: float  # highest sim threshold at which 75% group


def _load_champion_name_map() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in CHAMPION_NAMES_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[str(row["name"])] = int(row["_key"])
    return out


def _key_index(embeddings: LevelEmbeddings) -> dict[tuple[int, str, str], int]:
    cols = {name: i for i, name in enumerate(embeddings.key_columns)}
    cid = cols["championid"]
    role = cols["teamposition"]
    build = cols["build"]
    return {
        (int(k[cid]), str(k[role]), str(k[build])): i
        for i, k in enumerate(embeddings.keys)
    }


def _resolve_pairs(
    pairs: Iterable[tuple[IdentityKey, IdentityKey]],
    name_to_id: dict[str, int],
    key_to_row: dict[tuple[int, str, str], int],
) -> list[tuple[int, int]]:
    resolved: list[tuple[int, int]] = []
    for left, right in pairs:
        l_cid = name_to_id.get(left[0])
        r_cid = name_to_id.get(right[0])
        if l_cid is None or r_cid is None:
            continue
        l_idx = key_to_row.get((l_cid, left[1], left[2]))
        r_idx = key_to_row.get((r_cid, right[1], right[2]))
        if l_idx is None or r_idx is None:
            continue
        resolved.append((l_idx, r_idx))
    return resolved


def _semantic_threshold_75(
    embeddings: np.ndarray, pair_indices: list[tuple[int, int]]
) -> float:
    """Highest threshold at which >= 75% of resolved pairs are still grouped."""
    if not pair_indices:
        return float("nan")
    candidates = np.linspace(0.50, 0.95, 19)  # 0.50, 0.525, ..., 0.95
    best = float("nan")
    for t in candidates[::-1]:
        groups = group_by_threshold(embeddings, float(t))
        group_of = np.full(embeddings.shape[0], -1, dtype=np.int64)
        for gid, members in enumerate(groups):
            for m in members:
                group_of[m] = gid
        hit = sum(group_of[i] == group_of[j] for i, j in pair_indices)
        if hit / len(pair_indices) >= 0.75:
            best = float(t)
            break
    return best


def evaluate_semantic_pairs(
    embeddings: LevelEmbeddings,
    threshold: float,
) -> SemanticScore:
    name_to_id = _load_champion_name_map()
    key_to_row = _key_index(embeddings)

    sim = cosine_similarity_matrix(embeddings.embeddings)
    groups = group_by_threshold(embeddings.embeddings, threshold)
    group_of = np.full(embeddings.embeddings.shape[0], -1, dtype=np.int64)
    for gid, members in enumerate(groups):
        for m in members:
            group_of[m] = gid

    all_pairs: list[tuple[int, int]] = []
    n_total = 0
    by_cat: dict[str, tuple[int, int]] = {}
    sims: list[float] = []
    for category, pairs in SEMANTIC_PAIRS.items():
        n_total += len(pairs)
        resolved = _resolve_pairs(pairs, name_to_id, key_to_row)
        all_pairs.extend(resolved)
        hits = sum(group_of[i] == group_of[j] for i, j in resolved)
        by_cat[category] = (hits, len(resolved))
        sims.extend(float(sim[i, j]) for i, j in resolved)

    anti_resolved = _resolve_pairs(ANTI_PAIRS, name_to_id, key_to_row)
    anti_sims = [float(sim[i, j]) for i, j in anti_resolved]
    anti_hits = sum(group_of[i] == group_of[j] for i, j in anti_resolved)

    sims_arr = np.array(sims) if sims else np.array([float("nan")])
    anti_arr = np.array(anti_sims) if anti_sims else np.array([float("nan")])
    return SemanticScore(
        n_pairs_resolved=len(all_pairs),
        n_pairs_total=n_total,
        mean_sim=float(sims_arr.mean()),
        p50_sim=float(np.median(sims_arr)),
        min_sim=float(sims_arr.min()),
        same_group_rate=(
            sum(group_of[i] == group_of[j] for i, j in all_pairs) / len(all_pairs)
            if all_pairs
            else float("nan")
        ),
        by_category_same_group=by_cat,
        anti_mean_sim=float(anti_arr.mean()),
        anti_same_group_rate=(
            anti_hits / len(anti_resolved) if anti_resolved else float("nan")
        ),
        separation=float(sims_arr.mean() - anti_arr.mean()),
        semantic_threshold_75=_semantic_threshold_75(
            embeddings.embeddings, all_pairs
        ),
    )


def log_semantic_score(score: SemanticScore) -> None:
    logger.info(
        "semantic pairs %d/%d resolved | mean=%.3f p50=%.3f min=%.3f same_group=%.2f "
        "anti_mean=%.3f anti_same=%.2f separation=%.3f t75=%.2f",
        score.n_pairs_resolved,
        score.n_pairs_total,
        score.mean_sim,
        score.p50_sim,
        score.min_sim,
        score.same_group_rate,
        score.anti_mean_sim,
        score.anti_same_group_rate,
        score.separation,
        score.semantic_threshold_75,
    )
    for category, (hit, total) in score.by_category_same_group.items():
        if total:
            logger.info("  %s: %d/%d (%.0f%%)", category, hit, total, 100 * hit / total)
