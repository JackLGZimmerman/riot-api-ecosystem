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
    SINGULAR_METRICS,
    SPECIALISTS,
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.matrices import build_all_matrices
from app.classification.embeddings.report import _load_champion_names
from app.classification.embeddings.similarity import median_pair_similarity
from app.classification.embeddings.singular_metrics import _normalised_ordering
from app.classification.embeddings.specialists import group_specialist
from app.classification.embeddings.tune import load_raw_cached
from app.core.config.settings import PROJECT_ROOT
from app.core.utils.smoothing import apply_hierarchical_shrinkage

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
    "totaldamagedealttochampions": "champion damage volume",
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
    group_count: int,
    budget: int,
    size: int,
    n_identities: int,
    median: float,
    z: np.ndarray,
) -> str:
    abs_z = np.sort(np.abs(z))[::-1] if z.size else np.asarray([], dtype=np.float32)
    max_abs_z = float(abs_z[0]) if abs_z.size else 0.0
    second_abs_z = float(abs_z[1]) if abs_z.size > 1 else 0.0
    share = size / max(n_identities, 1)
    if group_count > budget:
        return (
            "Watch: over the crude budget, so this group needs a distinct "
            f"semantic reason to survive ({group_count}>{budget})."
        )
    if max_abs_z < 0.35:
        return "Weak: metric separation is shallow; prefer merging if this reappears."
    if size < 40:
        if max_abs_z >= 0.70 and median >= 0.90:
            return (
                "Excellent small-signal group: tiny support, but the metric "
                f"signature is sharp (max |z| {max_abs_z:.2f}, med {median:.2f})."
            )
        return (
            "Watch: small support and only modest metric separation; keep an "
            "eye on split stability."
        )
    if share >= 0.65:
        return (
            "Excellent baseline contrast: broad by design, with a clean "
            f"opposite-class metric read (max |z| {max_abs_z:.2f})."
        )
    if median < 0.90:
        if max_abs_z >= 0.85:
            return (
                "Excellent broad-spectrum read: cosine is looser, but the "
                f"shared metric signal is strong (max |z| {max_abs_z:.2f})."
            )
        return "Watch: median coherence is below the audit target for this signal."
    if max_abs_z >= 1.25:
        return (
            "Excellent: standout specialist signature with strong metric "
            f"separation (max |z| {max_abs_z:.2f})."
        )
    if second_abs_z >= 0.45:
        return (
            "Excellent: multi-metric read with coherent within-group geometry "
            f"(top |z| {max_abs_z:.2f}/{second_abs_z:.2f}, med {median:.2f})."
        )
    return (
        "Excellent: clean single-axis specialist read; useful as a focused "
        f"contrast class (max |z| {max_abs_z:.2f}, med {median:.2f})."
    )


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


def _specialist_markdown(
    spec, smoothed, champion_names: dict[int, str]
) -> tuple[list[str], int, float, int]:
    cfg = EmbeddingConfig(
        feature_set=spec.feature_set,
        projection_keep_variance=spec.projection_keep_variance,
    )
    matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
    baseline = embed.embed_level(matrix, cfg)
    grouping = group_specialist(baseline, spec)
    group_count = len(grouping.kept)
    coverage = sum(len(group) for group in grouping.kept) / max(baseline.embeddings.shape[0], 1)
    dropped_count = len(grouping.dropped)
    budget = math.ceil(len(spec.feature_set) * 1.5)
    n_identities = baseline.embeddings.shape[0]
    columns = {column: index for index, column in enumerate(baseline.key_columns)}
    lines = [
        f"### {spec.name}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Metrics Used | {', '.join(_metric_label(feature) for feature in spec.feature_set)} |",
        f"| Budget | <= {budget} groups |",
        f"| Config | kv={spec.projection_keep_variance:.2f}, t={spec.similarity_threshold:.2f}, min_median={spec.min_median_sim:.2f} |",
        f"| Groups | {group_count} |",
        f"| Coverage | {coverage:.2f} |",
        f"| Dropped Groups | {dropped_count} |",
        f"| PCA | {_pca_summary(matrix.matrix, matrix.feature_names, spec.projection_keep_variance)} |",
        "",
        "| Group | Size | Read | Context | Quality |",
        "| ---: | ---: | --- | --- | --- |",
    ]
    x = matrix.matrix
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
                        group_count=group_count,
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
    return lines, group_count, coverage, dropped_count


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


def _singular_markdown(spec, smoothed, champion_names: dict[int, str]) -> tuple[list[str], int]:
    cfg = EmbeddingConfig(feature_set=(spec.feature,))
    matrix = build_all_matrices(smoothed, cfg)[IdentityType.BASELINE]
    values = matrix.matrix[:, 0]
    unique_count = int(np.unique(values).size)
    columns = {column: index for index, column in enumerate(matrix.key_columns)}
    direction = "higher is stronger" if spec.higher_is_more else "lower is stronger"
    _, _, scores = _normalised_ordering(
        values,
        higher_is_more=spec.higher_is_more,
    )
    order = np.argsort(-scores, kind="mergesort")
    top = order[:50]
    bottom = order[-50:][::-1]
    top_context = _tail_context(matrix, champion_names, top, columns)
    bottom_context = _tail_context(matrix, champion_names, bottom, columns)
    if unique_count < 100:
        quality = (
            "Watch: many ties; this is a coarse ordering rather than a "
            "fine-grained scalar signal."
        )
    else:
        quality = (
            "Excellent: dense identity ordering with distinct "
            f"tails ({unique_count} unique values)."
        )
    lines = [
        f"### {spec.name}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Metric | {_metric_label(spec.feature)} |",
        f"| Direction | {direction} |",
        f"| Description | {spec.description or 'Identity ordering.'} |",
        f"| Unique Values | {unique_count} |",
        "",
        "| Top Tail Context | Bottom Tail Context | Quality |",
        "| --- | --- | --- |",
        "| " + " | ".join((top_context, bottom_context, quality)) + " |",
        "",
    ]
    return lines, unique_count


def generate_markdown() -> str:
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    champion_names = _load_champion_names()
    sections: list[str] = []
    summary_rows: list[str] = []
    for spec in SPECIALISTS:
        section, group_count, coverage, dropped_count = _specialist_markdown(
            spec, smoothed, champion_names
        )
        budget = math.ceil(len(spec.feature_set) * 1.5)
        status = "OK" if group_count <= budget else "OVER"
        summary_rows.append(
            "| "
            + " | ".join(
                (
                    f"`{spec.name}`",
                    str(len(spec.feature_set)),
                    str(budget),
                    f"kv={spec.projection_keep_variance:.2f}, t={spec.similarity_threshold:.2f}",
                    str(group_count),
                    f"{coverage:.2f}",
                    str(dropped_count),
                    status,
                )
            )
            + " |"
        )
        sections.extend(section)

    singular_sections: list[str] = []
    singular_rows: list[str] = []
    for spec in SINGULAR_METRICS:
        section, unique_count = _singular_markdown(spec, smoothed, champion_names)
        singular_rows.append(
            "| "
            + " | ".join(
                (
                    f"`{spec.name}`",
                    _metric_label(spec.feature),
                    "higher" if spec.higher_is_more else "lower",
                    str(unique_count),
                    "OK",
                )
            )
            + " |"
        )
        singular_sections.extend(section)

    header = [
        "# Specialisation Audit",
        "",
        "Generated from the active `SpecialistSpec` and `SingularMetricSpec` registries in `embeddings/config.py` using the training split cached at `/tmp/embed_exp/raw_levels_non_temporal.pkl`.",
        "The group budget is `ceil(1.5 * metrics_used)`. Group context is written per retained group from z-scores, role/build composition, champion concentration, size, and within-group cosine.",
        "",
        "## Specialist Summary",
        "",
        "| Specialist | Metrics | Budget | Config | Groups | Coverage | Dropped | Status |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
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
        "| Singular Metric | Feature | Strong Direction | Unique Values | Status |",
        "| --- | --- | --- | ---: | --- |",
        *singular_rows,
        "",
        "## Singular Metric Context",
        "",
        "Singular metrics are not clustered. They keep an identity ordering and are inspected through top and bottom tails.",
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
