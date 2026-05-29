"""Generate the specialist tuning audit markdown."""

from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np

from app.classification.embeddings import embed
from app.classification.embeddings.config import (
    PHASES,
    SINGULAR_METRICS,
    SPECIALISTS,
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.report import _load_champion_names
from app.classification.embeddings.similarity import median_pair_similarity
from app.classification.embeddings.singular_metrics import _normalised_ordering
from app.classification.embeddings.specialists import group_specialist_by_phase
from app.classification.embeddings.tune import load_raw_cached
from app.core.config.settings import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "app" / "classification" / "SPECIALISATIONS.md"

METRIC_LABELS = {
    "movementspeed": "movement speed",
    "first_blood_participation": "first blood participation",
    "first_tower_participation": "first tower participation",
    "early_snowball_participation": "early snowball participation",
    "kills_to_deaths_ratio": "kill safety",
    "assists_to_deaths_ratio": "assist safety",
    "durability_total_to_deaths_ratio": "durability per death",
    "durability_total_to_goldearned_ratio": "durability per gold",
    "healthmax_to_goldearned_ratio": "health per gold",
    "self_heal_to_durability_total_ratio": "self-heal share",
    "damageselfmitigated_to_durability_total_ratio": "mitigation share",
    "magicdamagetaken_to_durability_total_ratio": "magic damage taken share",
    "physicaldamagetaken_to_durability_total_ratio": "physical damage taken share",
    "vamp_sustain": "vamp sustain",
    "durability_total_to_healthmax_ratio": "realised durability per health",
    "physicaldamagedealttochampions_share": "physical champion-damage share",
    "magicdamagedealttochampions_share": "magic champion-damage share",
    "truedamagedealttochampions_share": "true champion-damage share",
    "physicaldamagedealt_share": "physical total-damage share",
    "magicdamagedealt_share": "magic total-damage share",
    "truedamagedealt_share": "true total-damage share",
    "champion_damage_to_total_damage_ratio": "champion-damage focus",
    "champion_damage_share_to_deaths_ratio": "champion-damage safety",
    "totaldamagedealttochampions_to_goldearned_ratio": "champion damage per gold",
    "totaldamagedealttochampions_to_deaths_ratio": "champion damage per death",
    "kills_to_assists_ratio": "kill share of takedowns",
    "takedowns_to_deaths_ratio": "takedown safety",
    "largestkillingspree": "killing-spree ceiling",
    "largestmultikill": "multikill ceiling",
    "damageselfmitigated_to_goldearned_ratio": "mitigation per gold",
    "visionscore_to_goldearned_ratio": "vision per gold",
    "visionscore_to_ward_actions_ratio": "vision per ward action",
    "wardskilled_to_wardsplaced_ratio": "ward clear/place balance",
    "cc_to_assists_ratio": "CC per assist",
    "timeccingothers": "direct CC time",
    "totaltimeccdealt": "total CC dealt",
    "cc_effectiveness_ratio": "CC effectiveness",
    "ally_support_to_assists_ratio": "ally support per assist",
    "totalminionskilled": "lane farm",
    "neutralminionskilled": "neutral farm",
    "total_farm_to_goldearned_ratio": "farm per gold",
    "total_farm_to_deaths_ratio": "farm safety",
    "goldearned": "gold income",
    "champexperience": "experience income",
    "champexperience_to_goldearned_ratio": "experience per gold",
    "jungle_minions": "jungle farm",
    "jungle_minion_share": "jungle farm share",
    "enemy_jungle_minion_share": "enemy jungle share",
    "enemy_to_ally_jungle_minions_ratio": "enemy-to-ally jungle farm",
    "epic_kills": "epic kills",
    "objective_neutral_minions": "objective neutral farm",
    "objective_damage_to_goldearned_ratio": "objective damage per gold",
    "objective_damage_to_total_damage_ratio": "objective damage share",
    "epic_monster_damage_to_objective_damage_ratio": "epic-monster objective share",
    "damagedealttoobjectives_per_epic_kill_per_gold": "objective damage per epic per gold",
    "structure_takedowns": "structure takedowns",
    "structure_losses": "structure losses",
    "structure_damage": "structure damage",
    "structure_takedowns_to_structure_damage_ratio": "structure conversion",
    "structure_net_control": "structure net control",
    "structure_damage_to_goldearned_ratio": "structure damage per gold",
    "structure_damage_to_deaths_ratio": "structure damage per death",
    "structure_takedowns_to_losses_ratio": "structure takedowns per loss",
    "structure_takedowns_to_goldearned_ratio": "structure takedowns per gold",
    "epic_kills_to_goldearned_ratio": "epic kills per gold",
    "totalhealsonteammates_to_goldearned_ratio": "ally healing per gold",
    "totaldamageshieldedonteammates_to_goldearned_ratio": "ally shielding per gold",
    "ally_support_to_goldearned_ratio": "ally support per gold",
    "armor_to_goldearned_ratio": "armor per gold",
    "magicresist_to_goldearned_ratio": "magic resist per gold",
    "damage_taken_to_goldearned_ratio": "damage taken per gold",
    "totaldamagetaken_to_deaths_ratio": "damage taken per death",
    "abilitypower": "ability power",
    "abilitypower_to_goldearned_ratio": "ability power per gold",
    "magicresist": "magic resist",
    "attackdamage": "attack damage",
    "attackdamage_to_goldearned_ratio": "attack damage per gold",
    "largestcriticalstrike": "critical strike ceiling",
    "attackspeed": "attack speed",
    "deaths": "deaths",
}


def _metric_label(name: str) -> str:
    return METRIC_LABELS.get(name, name.replace("_", " "))


def _compact_counter(counter: Counter[str], limit: int = 3) -> str:
    return ", ".join(f"{label}:{count}" for label, count in counter.most_common(limit))


def _z_summary(features: tuple[str, ...], z: np.ndarray, limit: int = 4) -> str:
    ranked = sorted(zip(features, z, strict=True), key=lambda pair: abs(pair[1]), reverse=True)
    chunks: list[str] = []
    for feature, value in ranked[:limit]:
        direction = "high" if value >= 0 else "low"
        chunks.append(f"{direction} {_metric_label(feature)} {value:+.2f}")
    return "; ".join(chunks)


def _read_name(features: tuple[str, ...], z: np.ndarray) -> str:
    positives = [
        (_metric_label(feature), value)
        for feature, value in sorted(
            zip(features, z, strict=True), key=lambda pair: pair[1], reverse=True
        )
        if value >= 0.35
    ]
    negatives = [
        (_metric_label(feature), value)
        for feature, value in sorted(zip(features, z, strict=True), key=lambda pair: pair[1])
        if value <= -0.35
    ]
    if positives and negatives:
        return f"High {positives[0][0]}, low {negatives[0][0]}"
    if positives:
        return "High " + " + ".join(label for label, _ in positives[:2])
    if negatives:
        return "Low " + " + ".join(label for label, _ in negatives[:2])
    feature, value = max(zip(features, z, strict=True), key=lambda pair: abs(pair[1]))
    return f"{'High' if value >= 0 else 'Low'} {_metric_label(feature)}"


def _context(
    *,
    features: tuple[str, ...],
    z: np.ndarray,
    roles: Counter[str],
    builds: Counter[str],
    champions: Counter[str],
    size: int,
) -> str:
    role, role_count = roles.most_common(1)[0]
    build, build_count = builds.most_common(1)[0]
    role_share = role_count / max(size, 1)
    build_share = build_count / max(size, 1)
    role_note = (
        f"role-skewed to {role}"
        if role_share >= 0.65
        else f"mixed roles led by {role}"
    )
    build_note = (
        f"build-skewed to {build}"
        if build_share >= 0.45
        else f"mixed builds led by {build}"
    )
    return (
        f"{role_note}; {build_note}; champions {_compact_counter(champions)}. "
        f"Metric link: {_z_summary(features, z)}."
    )


def _quality(
    *,
    phase_count: int,
    budget: int,
    size: int,
    n_identities: int,
    median: float,
    z: np.ndarray,
) -> str:
    max_abs_z = float(np.max(np.abs(z))) if z.size else 0.0
    share = size / max(n_identities, 1)
    if phase_count > budget:
        return f"Reject: phase has {phase_count} groups over budget {budget}."
    if max_abs_z < 0.35:
        return "Weak: metric separation is shallow; prefer merging if this reappears."
    if size < 40:
        return "Small but direct: retained because the metric link is explicit."
    if share >= 0.65:
        return "Background anchor with clear opposite-sign metric read; safe as a broad negative class."
    if median < 0.90:
        return "Moderate coherence; keep only because the z-score read is direct."
    return "Good: coherent and traceable to the selected metrics."


def _pca_summary(flat: np.ndarray, feature_names: tuple[str, ...], keep: float) -> str:
    _, _, eigenvectors, n_axes, ratios = embed.fit_pca_basis(flat, keep)
    parts: list[str] = []
    for axis in range(n_axes):
        weights = eigenvectors[:, axis]
        ranked = np.argsort(np.abs(weights))[::-1][:3]
        loadings = ", ".join(
            f"{_metric_label(feature_names[index])}={weights[index]:+.2f}"
            for index in ranked
        )
        parts.append(f"PC{axis + 1} {ratios[axis]:.3f} ({loadings})")
    return "; ".join(parts)


def _specialist_markdown(spec, smoothed, champion_names: dict[int, str]) -> tuple[list[str], tuple[int, ...]]:
    cfg = EmbeddingConfig(
        feature_set=spec.feature_set,
        projection_keep_variance=spec.projection_keep_variance,
    )
    matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
    baseline = embed.embed_level(matrix, cfg)
    groupings = group_specialist_by_phase(baseline, spec)
    phase_counts = tuple(len(grouping.kept) for grouping in groupings)
    budget = math.ceil(len(spec.feature_set) * 1.5)
    n_identities = baseline.embeddings.shape[0]
    columns = {column: index for index, column in enumerate(baseline.key_columns)}
    lines = [
        f"### {spec.name}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Metrics Used | {', '.join(_metric_label(feature) for feature in spec.feature_set)} |",
        f"| Budget | <= {budget} groups per phase |",
        f"| Config | kv={spec.projection_keep_variance:.2f}, t={spec.similarity_threshold:.2f}, min_median={spec.min_median_sim:.2f} |",
        f"| Phase Groups | {', '.join(f'{phase}:{count}' for phase, count in zip(PHASES, phase_counts, strict=True))} |",
        f"| PCA | {_pca_summary(matrix.matrix.reshape(-1, matrix.matrix.shape[-1]), matrix.feature_names, spec.projection_keep_variance)} |",
        "",
        "| Phase | Group | Size | Read | Context | Quality |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for grouping in groupings:
        x = matrix.matrix[:, grouping.phase_index, :]
        mu = x.mean(axis=0)
        sd = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
        for gid, group in enumerate(sorted(grouping.kept, key=len, reverse=True), start=1):
            arr = np.asarray(group, dtype=np.int64)
            z = (x[arr].mean(axis=0) - mu) / sd
            roles = Counter(str(baseline.keys[i][columns["teamposition"]]) for i in group)
            builds = Counter(str(baseline.keys[i][columns["build"]]) for i in group)
            champions = Counter(
                champion_names.get(
                    int(baseline.keys[i][columns["championid"]]),
                    str(baseline.keys[i][columns["championid"]]),
                )
                for i in group
            )
            median = median_pair_similarity(grouping.sim, group)
            lines.append(
                "| "
                + " | ".join(
                    (
                        grouping.phase,
                        f"G{gid:02d}",
                        str(len(group)),
                        _read_name(matrix.feature_names, z),
                        _context(
                            features=matrix.feature_names,
                            z=z,
                            roles=roles,
                            builds=builds,
                            champions=champions,
                            size=len(group),
                        ),
                        _quality(
                            phase_count=len(grouping.kept),
                            budget=budget,
                            size=len(group),
                            n_identities=n_identities,
                            median=median,
                            z=z,
                        ),
                    )
                )
                + " |"
            )
    lines.append("")
    return lines, phase_counts


def _tail_context(
    matrix,
    champion_names: dict[int, str],
    indices: np.ndarray,
    columns: dict[str, int],
) -> str:
    roles = Counter(str(matrix.keys[i][columns["teamposition"]]) for i in indices)
    builds = Counter(str(matrix.keys[i][columns["build"]]) for i in indices)
    champions = Counter(
        champion_names.get(
            int(matrix.keys[i][columns["championid"]]),
            str(matrix.keys[i][columns["championid"]]),
        )
        for i in indices
    )
    return (
        f"roles {_compact_counter(roles)}; builds {_compact_counter(builds)}; "
        f"champions {_compact_counter(champions, limit=5)}"
    )


def _singular_markdown(spec, smoothed, champion_names: dict[int, str]) -> tuple[list[str], tuple[int, ...]]:
    cfg = EmbeddingConfig(feature_set=(spec.feature,))
    matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
    values = matrix.matrix[:, :, 0]
    unique_counts = tuple(
        int(np.unique(values[:, phase_index]).size)
        for phase_index in range(values.shape[1])
    )
    columns = {column: index for index, column in enumerate(matrix.key_columns)}
    direction = "higher is stronger" if spec.higher_is_more else "lower is stronger"
    lines = [
        f"### {spec.name}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Metric | {_metric_label(spec.feature)} |",
        f"| Direction | {direction} |",
        f"| Description | {spec.description or 'Phase-relative ordering.'} |",
        f"| Unique Values | {', '.join(f'{phase}:{count}' for phase, count in zip(PHASES, unique_counts, strict=True))} |",
        "",
        "| Phase | Top Tail Context | Bottom Tail Context | Quality |",
        "| --- | --- | --- | --- |",
    ]
    for phase_index, phase in enumerate(PHASES):
        phase_values = values[:, phase_index]
        _, _, scores = _normalised_ordering(
            phase_values,
            higher_is_more=spec.higher_is_more,
        )
        order = np.argsort(-scores, kind="mergesort")
        top = order[:50]
        bottom = order[-50:][::-1]
        top_context = _tail_context(matrix, champion_names, top, columns)
        bottom_context = _tail_context(matrix, champion_names, bottom, columns)
        if unique_counts[phase_index] < 100:
            quality = "Watch: many ties; use as a coarse ordering only."
        else:
            quality = "Good: high-cardinality phase-relative ordering."
        lines.append(
            "| "
            + " | ".join((phase, top_context, bottom_context, quality))
            + " |"
        )
    lines.append("")
    return lines, unique_counts


def generate_markdown() -> str:
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    champion_names = _load_champion_names()
    sections: list[str] = []
    summary_rows: list[str] = []
    for spec in SPECIALISTS:
        section, phase_counts = _specialist_markdown(spec, smoothed, champion_names)
        budget = math.ceil(len(spec.feature_set) * 1.5)
        status = "OK" if all(count <= budget for count in phase_counts) else "OVER"
        summary_rows.append(
            "| "
            + " | ".join(
                (
                    f"`{spec.name}`",
                    str(len(spec.feature_set)),
                    str(budget),
                    f"kv={spec.projection_keep_variance:.2f}, t={spec.similarity_threshold:.2f}",
                    "[" + ", ".join(str(count) for count in phase_counts) + "]",
                    status,
                )
            )
            + " |"
        )
        sections.extend(section)

    singular_sections: list[str] = []
    singular_rows: list[str] = []
    for spec in SINGULAR_METRICS:
        section, unique_counts = _singular_markdown(spec, smoothed, champion_names)
        singular_rows.append(
            "| "
            + " | ".join(
                (
                    f"`{spec.name}`",
                    _metric_label(spec.feature),
                    "higher" if spec.higher_is_more else "lower",
                    "[" + ", ".join(str(count) for count in unique_counts) + "]",
                    "OK",
                )
            )
            + " |"
        )
        singular_sections.extend(section)

    header = [
        "# Specialisation Audit",
        "",
        "Generated from the active `SpecialistSpec` and `SingularMetricSpec` registries in `embeddings/config.py` using the training split cached at `/tmp/embed_exp/raw_levels.pkl`.",
        "The per-phase budget is `ceil(1.5 * metrics_used)`. Group context is written per retained group from z-scores, role/build composition, champion concentration, size, and within-group cosine.",
        "",
        "## Specialist Summary",
        "",
        "| Specialist | Metrics | Budget | Config | Phase Groups | Status |",
        "| --- | ---: | ---: | --- | --- | --- |",
        *summary_rows,
        "",
        "## Specialist Context",
        "",
        "Every active specialist is listed below. Quality comments are per group, not copied across a whole specialist.",
        "",
    ]
    singular_header = [
        "## Singular Metric Summary",
        "",
        "| Singular Metric | Feature | Strong Direction | Unique Values By Phase | Status |",
        "| --- | --- | --- | --- | --- |",
        *singular_rows,
        "",
        "## Singular Metric Context",
        "",
        "Singular metrics are not clustered. They keep a phase-relative ordering and are inspected through top and bottom identity tails.",
        "",
    ]
    return "\n".join(header + sections + singular_header + singular_sections).rstrip() + "\n"


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(generate_markdown(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
