"""Sweep experiment harness for embedding configuration.

Loads ClickHouse raw data once (caching to /tmp on first run), then iterates
through `ExperimentSpec` configurations. Each spec varies levers
(`feature_set`, `projection_keep_variance`, `extreme_low_sample_threshold`,
prior strength multipliers, similarity threshold). For each run we record:

  * group quality (non-singleton size mean excluding the top 10 groups,
    coverage, diversity, quality)
  * semantic same-group rate and separation vs anti-pairs
  * t75 — highest similarity threshold at which 75% of expected pairs still
    cluster together

The composite ranking score is `semantic_same_group * separation * quality`,
which simultaneously rewards (a) expected pairs clustering, (b) embedding
distinguishing semantic vs anti-semantic pairs, and (c) the underlying
clustering being non-trivial.

Run:
    uv run python -m app.classification.embeddings.experiments
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path


from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    DEFAULT_DERIVED_RATIO_FEATURE_SET,
    DEFAULT_EMBEDDING_FEATURE_SET,
    DEFAULT_PRIOR_PER_MINUTE_STRENGTHS,
    DEFAULT_PRIOR_RATE_STRENGTHS,
    DEFAULT_RAW_FEATURE_SET,
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.load import LevelRows, load_all
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.test_pairs import (
    SemanticScore,
    evaluate_semantic_pairs,
)
from app.classification.embeddings.validate import diagnose_all
from app.core.logging.logger import setup_logging_config

logger = logging.getLogger(__name__)

RAW_CACHE_PATH = Path("/tmp/embed_exp/raw_levels.pkl")


def _load_raw_cached() -> dict[IdentityType, LevelRows]:
    if RAW_CACHE_PATH.exists():
        with RAW_CACHE_PATH.open("rb") as f:
            return pickle.load(f)
    RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    levels = load_all(EmbeddingConfig())
    with RAW_CACHE_PATH.open("wb") as f:
        pickle.dump(levels, f)
    return levels


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    feature_set: tuple[str, ...] = DEFAULT_EMBEDDING_FEATURE_SET
    projection_keep_variance: float = 0.91
    extreme_low_sample_threshold: float = 50.0
    similarity_threshold: float = 0.82
    prior_rate_multiplier: float = 1.0
    prior_per_minute_multiplier: float = 1.0
    prior_rate_overrides: dict[IdentityType, float] = field(default_factory=dict)
    prior_per_minute_overrides: dict[IdentityType, float] = field(default_factory=dict)

    def to_config(self) -> EmbeddingConfig:
        rate = {
            level: strength * self.prior_rate_multiplier
            for level, strength in DEFAULT_PRIOR_RATE_STRENGTHS.items()
        }
        rate.update(self.prior_rate_overrides)
        per_min = {
            level: strength * self.prior_per_minute_multiplier
            for level, strength in DEFAULT_PRIOR_PER_MINUTE_STRENGTHS.items()
        }
        per_min.update(self.prior_per_minute_overrides)
        return EmbeddingConfig(
            feature_set=self.feature_set,
            projection_keep_variance=self.projection_keep_variance,
            extreme_low_sample_threshold=self.extreme_low_sample_threshold,
            similarity_threshold=self.similarity_threshold,
            prior_rate_strengths=rate,
            prior_per_minute_strengths=per_min,
        )


@dataclass(frozen=True)
class Scorecard:
    name: str
    n_identities: int
    n_groups: int
    n_non_singleton_groups: int
    n_singletons: int
    largest_group: int
    largest_group_share: float
    mean_non_singleton: float  # excludes the top 10 largest non-singleton groups
    coverage: float
    diversity: float
    quality: float
    min_group_pairwise_sim: float
    low_sample_dominance: float
    top_mid_mixed_share: float
    pca_dims: int
    semantic_same_group: float
    semantic_mean_sim: float
    semantic_separation: float
    semantic_t75: float
    anti_same_group: float
    composite: float
    # Net = semantic recall minus anti-pair leakage, scaled by semantic
    # separation. Broad groups are audited by composition instead of being
    # penalised by size alone.
    net_score: float


def _run_one(spec: ExperimentSpec, raw_levels: dict[IdentityType, LevelRows]) -> Scorecard:
    cfg = spec.to_config()
    smoothed = apply_hierarchical_shrinkage(raw_levels, cfg)
    matrices = build_all_matrices(smoothed, cfg)
    embeddings = embed.embed_all(matrices, cfg)
    diagnostics = diagnose_all(
        embeddings, cfg.similarity_threshold, cfg.group_min_matchups
    )
    baseline_diag = diagnostics[IdentityType.BASELINE]
    baseline_emb = embeddings[IdentityType.BASELINE]
    score: SemanticScore = evaluate_semantic_pairs(
        baseline_emb, cfg.similarity_threshold
    )
    largest_share = baseline_diag.largest_group / max(baseline_diag.n, 1)
    composite = (
        max(score.same_group_rate, 0.0)
        * max(score.separation, 0.0)
        * baseline_diag.group_quality_score
    )
    net = max(
        score.same_group_rate - score.anti_same_group_rate,
        0.0,
    ) * max(score.separation, 0.0)
    return Scorecard(
        name=spec.name,
        n_identities=baseline_diag.n,
        n_groups=baseline_diag.group_count,
        n_non_singleton_groups=baseline_diag.non_singleton_group_count,
        n_singletons=baseline_diag.singleton_group_count,
        largest_group=baseline_diag.largest_group,
        largest_group_share=largest_share,
        mean_non_singleton=baseline_diag.mean_non_singleton_group_size,
        coverage=baseline_diag.non_singleton_identity_share,
        diversity=baseline_diag.group_diversity_score,
        quality=baseline_diag.group_quality_score,
        min_group_pairwise_sim=baseline_diag.min_group_pairwise_sim,
        low_sample_dominance=baseline_diag.low_sample_dominance,
        top_mid_mixed_share=baseline_diag.top_mid_mixed_identity_share,
        pca_dims=int(baseline_emb.embeddings.shape[1]),
        semantic_same_group=score.same_group_rate,
        semantic_mean_sim=score.mean_sim,
        semantic_separation=score.separation,
        semantic_t75=score.semantic_threshold_75,
        anti_same_group=score.anti_same_group_rate,
        composite=float(composite),
        net_score=float(net),
    )


# Curated derived metric subsets used by the experiment sweep.
DERIVED_HIGH_VALUE: tuple[str, ...] = (
    "kda_ratio",
    "first_blood_participation",
    "first_tower_participation",
    "physical_damage_share",
    "magic_damage_share",
    "true_damage_share",
    "damage_mitigated_ratio",
    "ally_protection",
    "champion_damage_per_gold",
    "xp_per_gold",
    "total_farm",
    "epic_kills",
)

DERIVED_RATIOS_ONLY: tuple[str, ...] = DEFAULT_DERIVED_RATIO_FEATURE_SET

DERIVED_COMPACT: tuple[str, ...] = (
    "physical_damage_share",
    "magic_damage_share",
    "damage_mitigated_ratio",
    "ally_protection",
    "champion_damage_per_gold",
    "total_farm",
)

DERIVED_EVENT_PRESSURE: tuple[str, ...] = (
    "first_blood_participation",
    "first_blood_kill_share",
    "first_blood_assist_share",
    "first_tower_participation",
    "first_tower_kill_share",
    "first_tower_assist_share",
    "kill_share_proxy",
    "assist_share_proxy",
    "death_pressure_ratio",
    "multikill_rate_per_kill",
)

DERIVED_DAMAGE_NICHE: tuple[str, ...] = (
    "physical_damage_share",
    "magic_damage_share",
    "true_damage_share",
    "physical_champion_damage_share",
    "magic_champion_damage_share",
    "physical_champion_damage_focus",
    "magic_champion_damage_focus",
    "champion_damage_focus",
    "non_champion_damage_share",
    "physical_vs_magic_champion_damage",
    "damage_dealt_taken_ratio",
    "net_champion_damage_trade",
)

DERIVED_SUPPORT_CONTROL: tuple[str, ...] = (
    "assist_to_kill_ratio",
    "assist_share_proxy",
    "teammate_heal_share",
    "ally_protection",
    "shield_to_teammate_heal_ratio",
    "protection_to_damage_taken",
    "protection_to_champion_damage",
    "vision_per_gold",
    "vision_per_death",
    "cc_to_takedowns",
    "cc_to_champion_damage",
    "cc_taken_pressure_ratio",
)

DERIVED_ECON_OBJECTIVE: tuple[str, ...] = (
    "gold_per_xp",
    "xp_per_gold",
    "gold_per_takedown",
    "champion_damage_per_gold",
    "damage_taken_per_gold",
    "objective_damage_per_gold",
    "xp_per_takedown",
    "total_farm",
    "neutral_farm_share",
    "lane_farm_share",
    "gold_per_farm",
    "xp_per_farm",
    "epic_kills",
    "baron_to_dragon_ratio",
    "objective_damage_share",
    "objective_vs_champion_damage",
)

DERIVED_NICHE_CORE: tuple[str, ...] = (
    "first_blood_participation",
    "first_tower_participation",
    "kill_share_proxy",
    "assist_share_proxy",
    "physical_damage_share",
    "magic_damage_share",
    "champion_damage_focus",
    "damage_dealt_taken_ratio",
    "ally_protection",
    "protection_to_champion_damage",
    "champion_damage_per_gold",
    "neutral_farm_share",
    "objective_vs_champion_damage",
    "vision_per_gold",
    "cc_to_takedowns",
)

# A leaner raw set focused on archetype-defining columns.
LEAN_RAW: tuple[str, ...] = (
    "win",
    "largestcriticalstrike",
    "kills",
    "deaths",
    "assists",
    "goldearned",
    "totaldamagedealttochampions",
    "physicaldamagedealttochampions",
    "magicdamagedealttochampions",
    "totaldamagetaken",
    "damageselfmitigated",
    "totalhealsonteammates",
    "totaldamageshieldedonteammates",
    "totalminionskilled",
    "neutralminionskilled",
    "visionscore",
)


def _feature_union(*parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(feature for part in parts for feature in part))


def default_experiment_specs() -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []

    # --- Phase 0: base references. ---
    specs.append(ExperimentSpec(name="00_active_base"))
    specs.append(
        ExperimentSpec(
            name="00_raw_base002_kv0.91_low50",
            feature_set=DEFAULT_RAW_FEATURE_SET,
        )
    )
    specs.append(
        ExperimentSpec(
            name="00_raw_base001_kv0.92_low100",
            feature_set=DEFAULT_RAW_FEATURE_SET,
            projection_keep_variance=0.92,
            extreme_low_sample_threshold=100.0,
        )
    )

    # --- Phase 1: PCA keep-variance around the live signal band. ---
    # 0.94+ fragmented and 0.85 over-compressed in earlier runs, so those
    # clearly weak edges are removed from the default deep sweep.
    for kv in (0.88, 0.89, 0.90, 0.91, 0.92, 0.93):
        specs.append(
            ExperimentSpec(name=f"01_pca_kv_{kv:g}", projection_keep_variance=kv)
        )

    # --- Phase 2: rare-row shrinkage around the useful range. ---
    for t in (0.0, 25.0, 50.0, 75.0, 100.0, 150.0):
        specs.append(
            ExperimentSpec(name=f"02_low_sample_{t:g}", extreme_low_sample_threshold=t)
        )

    # --- Phase 3: threshold lens. ---
    for thresh in (0.80, 0.82, 0.84):
        specs.append(
            ExperimentSpec(
                name=f"03_threshold_{thresh:g}",
                similarity_threshold=thresh,
            )
        )

    # --- Phase 4: prior strength. ---
    for mult in (0.5, 1.25, 1.5, 2.0):
        specs.append(
            ExperimentSpec(
                name=f"04_prior_x{mult:g}",
                prior_rate_multiplier=mult,
                prior_per_minute_multiplier=mult,
            )
        )
    specs.append(
        ExperimentSpec(
            name="04_role_build_heavy",
            prior_rate_overrides={IdentityType.ROLE_BUILD: 60.0},
            prior_per_minute_overrides={IdentityType.ROLE_BUILD: 12_000.0},
        )
    )
    specs.append(
        ExperimentSpec(
            name="04_champion_role_heavy",
            prior_rate_overrides={IdentityType.CHAMPION_ROLE: 12.0},
            prior_per_minute_overrides={IdentityType.CHAMPION_ROLE: 2_400.0},
        )
    )
    specs.append(
        ExperimentSpec(
            name="04_champion_role_light",
            prior_rate_overrides={IdentityType.CHAMPION_ROLE: 1.0},
            prior_per_minute_overrides={IdentityType.CHAMPION_ROLE: 200.0},
        )
    )

    # --- Phase 5: feature-family probes. ---
    feature_families = {
        "compact": DERIVED_COMPACT,
        "high_value": DERIVED_HIGH_VALUE,
        "event_pressure": DERIVED_EVENT_PRESSURE,
        "damage_niche": DERIVED_DAMAGE_NICHE,
        "support_control": DERIVED_SUPPORT_CONTROL,
        "econ_objective": DERIVED_ECON_OBJECTIVE,
        "niche_core": DERIVED_NICHE_CORE,
    }
    for name, derived in feature_families.items():
        specs.append(
            ExperimentSpec(
                name=f"05_default_plus_{name}",
                feature_set=_feature_union(DEFAULT_RAW_FEATURE_SET, derived),
            )
        )
    specs.append(ExperimentSpec(name="05_lean_raw_only", feature_set=LEAN_RAW))
    specs.append(
        ExperimentSpec(
            name="05_lean_plus_compact",
            feature_set=_feature_union(LEAN_RAW, DERIVED_COMPACT),
        )
    )
    specs.append(
        ExperimentSpec(
            name="05_lean_plus_niche_core",
            feature_set=_feature_union(LEAN_RAW, DERIVED_NICHE_CORE),
        )
    )

    # --- Phase 6: cross-lever candidates from the best historical regions. ---
    for kv in (0.88, 0.90, 0.92):
        specs.append(
            ExperimentSpec(
                name=f"06_kv{kv:g}_ratios_low50",
                feature_set=_feature_union(DEFAULT_RAW_FEATURE_SET, DERIVED_RATIOS_ONLY),
                projection_keep_variance=kv,
                extreme_low_sample_threshold=50.0,
            )
        )
        specs.append(
            ExperimentSpec(
                name=f"06_kv{kv:g}_niche_core_low50",
                feature_set=_feature_union(DEFAULT_RAW_FEATURE_SET, DERIVED_NICHE_CORE),
                projection_keep_variance=kv,
                extreme_low_sample_threshold=50.0,
            )
        )
        specs.append(
            ExperimentSpec(
                name=f"06_kv{kv:g}_event_pressure_low50",
                feature_set=_feature_union(
                    DEFAULT_RAW_FEATURE_SET, DERIVED_EVENT_PRESSURE
                ),
                projection_keep_variance=kv,
                extreme_low_sample_threshold=50.0,
            )
        )
    specs.append(
        ExperimentSpec(
            name="06_kv0.92_ratios_low50_t80",
            feature_set=_feature_union(DEFAULT_RAW_FEATURE_SET, DERIVED_RATIOS_ONLY),
            projection_keep_variance=0.92,
            extreme_low_sample_threshold=50.0,
            similarity_threshold=0.80,
        )
    )
    specs.append(
        ExperimentSpec(
            name="06_kv0.92_niche_core_low50_t80",
            feature_set=_feature_union(DEFAULT_RAW_FEATURE_SET, DERIVED_NICHE_CORE),
            projection_keep_variance=0.92,
            extreme_low_sample_threshold=50.0,
            similarity_threshold=0.80,
        )
    )

    # --- Phase 7: explicit low-sample interaction checks for niche metrics. ---
    for low in (25.0, 50.0, 75.0):
        specs.append(
            ExperimentSpec(
                name=f"07_niche_core_low{low:g}",
                feature_set=_feature_union(DEFAULT_RAW_FEATURE_SET, DERIVED_NICHE_CORE),
                extreme_low_sample_threshold=low,
            )
        )
        specs.append(
            ExperimentSpec(
                name=f"07_support_control_low{low:g}",
                feature_set=_feature_union(
                    DEFAULT_RAW_FEATURE_SET, DERIVED_SUPPORT_CONTROL
                ),
                extreme_low_sample_threshold=low,
            )
        )

    return specs


def _print_scorecard_table(scorecards: list[Scorecard]) -> str:
    header = (
        f"{'name':<40} {'n':>4} {'grp':>4} {'sing':>4} {'lg':>4} {'lg%':>4} "
        f"{'mnsx':>5} {'cov':>4} {'div':>4} {'qual':>5} {'wmin':>5} {'low':>4} "
        f"{'tmm':>5} {'sg%':>4} {'asg':>4} {'sim':>5} {'sep':>5} "
        f"{'t75':>4} {'net':>5} {'comp':>5} {'pca':>3}"
    )
    lines = [header, "-" * len(header)]
    for s in scorecards:
        lines.append(
            f"{s.name:<40} {s.n_identities:>4d} {s.n_groups:>4d} "
            f"{s.n_singletons:>4d} {s.largest_group:>4d} "
            f"{s.largest_group_share:>4.2f} {s.mean_non_singleton:>5.1f} "
            f"{s.coverage:>4.2f} {s.diversity:>4.2f} {s.quality:>5.2f} "
            f"{s.min_group_pairwise_sim:>5.2f} {s.low_sample_dominance:>4.2f} "
            f"{s.top_mid_mixed_share:>5.2f} {s.semantic_same_group:>4.2f} "
            f"{s.anti_same_group:>4.2f} {s.semantic_mean_sim:>5.2f} "
            f"{s.semantic_separation:>5.2f} {s.semantic_t75:>4.2f} "
            f"{s.net_score:>5.2f} {s.composite:>5.2f} {s.pca_dims:>3d}"
        )
    return "\n".join(lines)


def run_sweep(
    specs: list[ExperimentSpec] | None = None,
    output_path: Path | None = None,
) -> list[Scorecard]:
    specs = specs or default_experiment_specs()
    raw_levels = _load_raw_cached()
    scorecards: list[Scorecard] = []
    for spec in specs:
        logger.info("--- experiment: %s ---", spec.name)
        scorecards.append(_run_one(spec, raw_levels))

    table = _print_scorecard_table(scorecards)
    print()
    print("Sorted by net_score (descending):")
    by_net = sorted(scorecards, key=lambda s: -s.net_score)
    print(_print_scorecard_table(by_net))
    print()
    print("In configuration order:")
    print(table)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write("--- sorted by net_score ---\n")
            f.write(_print_scorecard_table(by_net))
            f.write("\n\n--- in configuration order ---\n")
            f.write(table)
            f.write("\n")
    return scorecards


def main() -> None:
    setup_logging_config()
    logging.getLogger().setLevel(logging.WARNING)
    run_sweep(output_path=Path("/tmp/embed_exp/scorecard.txt"))


if __name__ == "__main__":
    main()
