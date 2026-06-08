"""Shared semantic context audit specifications.

The markdown audit and the optional train-time calibration objective both use
these exact rows, axes, and bins. Keeping the definitions here prevents the
report and the optimizer from drifting apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.ml.semantic_group_features import (
    CONTEXT_BIN_EDGES,
    FOCUS_HP_LOW_THRESHOLD,
    HIGH_HP_THRESHOLD,
    RANGED_ATTACK_RANGE_THRESHOLD,
)

POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


@dataclass(frozen=True)
class BinSpec:
    label: str
    predicate: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class AuditSpec:
    section: str
    title: str
    read: str
    axis: str
    bins: tuple[BinSpec, ...]
    champions: tuple[int, ...] = ()
    positions: tuple[str, ...] = ()
    builds: tuple[str, ...] = ()
    focus_condition: str | None = None


def le(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis <= float(value)


def lt(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis < float(value)


def eq(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis == float(value)


def ge(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis >= float(value)


def gt(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: axis > float(value)


def between(lower: float, upper: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda axis: (axis > float(lower)) & (axis < float(upper))


def count_bins() -> tuple[BinSpec, ...]:
    return (
        BinSpec("0", eq(0)),
        BinSpec("1", eq(1)),
        BinSpec("2", eq(2)),
        BinSpec(">= 3", ge(3)),
    )


def range_count_bins() -> tuple[BinSpec, ...]:
    return (
        BinSpec("<= 1", le(1)),
        BinSpec("2", eq(2)),
        BinSpec("3", eq(3)),
        BinSpec(">= 4", ge(4)),
    )


def continuous_bins(a: float, b: float, c: float, d: float) -> tuple[BinSpec, ...]:
    return (
        BinSpec(f"<= {a:.3f}", le(a)),
        BinSpec(f"{a:.3f}-{b:.3f}", between(a, b)),
        BinSpec(f"{b:.3f}-{c:.3f}", between(b, c)),
        BinSpec(f"{c:.3f}-{d:.3f}", between(c, d)),
        BinSpec(f">= {d:.3f}", ge(d)),
    )


PHYSICAL_BINS = continuous_bins(*CONTEXT_BIN_EDGES["physical"])
MAGIC_BINS = continuous_bins(*CONTEXT_BIN_EDGES["magic"])
DAMAGE_BINS = continuous_bins(*CONTEXT_BIN_EDGES["damage"])
TAKEN_BINS = continuous_bins(*CONTEXT_BIN_EDGES["damage_taken"])
HEAL_BINS = continuous_bins(*CONTEXT_BIN_EDGES["heal_shield"])
CC_BINS = continuous_bins(*CONTEXT_BIN_EDGES["cc"])
SIEGE_BINS = continuous_bins(*CONTEXT_BIN_EDGES["siege"])
SCALING_BINS = continuous_bins(*CONTEXT_BIN_EDGES["scaling"])


def audit_specs() -> tuple[AuditSpec, ...]:
    headline = "Headline Trajectory Audit Tables"
    richer = "Richer Composition Trajectory Tables"
    retained = "Retained Prior And User-Requested Trajectory Tables"
    lower = "Inspected Lower-Signal Trajectory Tables"
    synergy = "Top-20 Matchup And Synergy Audits"
    # Champion ids are riot numeric keys. The noisiest champion-specific rows were
    # re-cut onto top-20 most-played champions that fit the same archetype + axis so
    # per-bin n is large; a few well-sampled or no-top-20-fit rows (Galio MR-tank,
    # Nilah melee-ADC, Malphite armor, Darius range, MasterYi) are kept as-is.
    return (
        AuditSpec(headline, "Yasuo TOP `crit` vs enemy siege", "Melee crit carry punished by poke and siege.", "enemy_siege", SIEGE_BINS, champions=(157,), positions=("TOP",), builds=("crit",)),
        AuditSpec(headline, "Graves JUNGLE `lethality` vs enemy damage", "Burst jungler into high enemy damage.", "enemy_damage", DAMAGE_BINS, champions=(104,), positions=("JUNGLE",), builds=("lethality",)),
        AuditSpec(headline, "Yasuo MIDDLE `crit` vs enemy siege", "Same melee-carry-into-poke pattern across lane.", "enemy_siege", SIEGE_BINS, champions=(157,), positions=("MIDDLE",), builds=("crit",)),
        AuditSpec(headline, "Ahri MIDDLE `ability_power` vs enemy scaling", "AP mid into scaling enemy compositions.", "enemy_scaling", SCALING_BINS, champions=(103,), positions=("MIDDLE",), builds=("ability_power",)),
        AuditSpec(headline, "Nautilus UTILITY `mr_tank` with ally damage", "Engage support with damage behind it.", "ally_damage", DAMAGE_BINS, champions=(111,), positions=("UTILITY",), builds=("mr_tank",)),
        AuditSpec(headline, "Galio MIDDLE `mr_tank` vs enemy magic", "Anti-magic tank itemization (kept off-list MR-tank).", "enemy_magic", MAGIC_BINS, champions=(3,), positions=("MIDDLE",), builds=("mr_tank",)),
        AuditSpec(headline, "Malphite TOP `ar_tank` vs enemy physical", "Armor tank into AD-heavy enemies.", "enemy_physical", PHYSICAL_BINS, champions=(54,), positions=("TOP",), builds=("ar_tank",)),
        AuditSpec(headline, "Sylas MIDDLE `ability_power` vs enemy range", "Short-range AP battlemage into enemy range pressure.", "enemy_ranged_count", range_count_bins(), champions=(517,), positions=("MIDDLE",), builds=("ability_power",)),
        AuditSpec(headline, "Nilah BOTTOM any build vs enemy range", "Melee bot lane into range-heavy teams (kept off-list melee-ADC).", "enemy_ranged_count", range_count_bins(), champions=(895,), positions=("BOTTOM",)),
        AuditSpec(headline, "Kaisa BOTTOM any build vs enemy range", "High-sample marksman vs enemy range pressure; large n keeps bins low-noise.", "enemy_ranged_count", range_count_bins(), champions=(145,), positions=("BOTTOM",)),
        AuditSpec(richer, "Kaisa BOTTOM `on_hit` vs enemy frontline count", "On-hit marksman shreds added enemy frontline.", "enemy_frontline_count", count_bins(), champions=(145,), positions=("BOTTOM",), builds=("on_hit",)),
        AuditSpec(richer, "Ahri MIDDLE `ability_power` vs enemy frontline count", "AP mid improves as enemies stack durable targets.", "enemy_frontline_count", count_bins(), champions=(103,), positions=("MIDDLE",), builds=("ability_power",)),
        AuditSpec(richer, "Sylas JUNGLE `ability_power` vs enemy frontline count", "Sustained AP skirmisher into beefy teams.", "enemy_frontline_count", count_bins(), champions=(517,), positions=("JUNGLE",), builds=("ability_power",)),
        AuditSpec(richer, "Sylas MIDDLE `ability_power` vs enemy frontline count", "Same AP anti-frontline pattern from lane.", "enemy_frontline_count", count_bins(), champions=(517,), positions=("MIDDLE",), builds=("ability_power",)),
        AuditSpec(richer, "Karma UTILITY any build vs enemy frontline count", "Utility support gains value as enemies stack frontline to zone.", "enemy_frontline_count", count_bins(), champions=(43,), positions=("UTILITY",)),
        AuditSpec(richer, "Vayne BOTTOM `on_hit` vs enemy frontline count", "Classic anti-tank marksman pattern.", "enemy_frontline_count", count_bins(), champions=(67,), positions=("BOTTOM",), builds=("on_hit",)),
        AuditSpec(richer, "Thresh UTILITY `ar_tank` vs enemy burst count", "Durable engage support punished by multiple burst threats.", "enemy_burst_count", count_bins(), champions=(412,), positions=("UTILITY",), builds=("ar_tank",)),
        AuditSpec(richer, "Nautilus UTILITY `mr_tank` vs enemy burst count", "High-HP engage tank loses into concentrated burst.", "enemy_burst_count", count_bins(), champions=(111,), positions=("UTILITY",), builds=("mr_tank",)),
        AuditSpec(richer, "Zed MIDDLE `lethality` vs enemy burst count", "Assassin into enemy burst stacking.", "enemy_burst_count", count_bins(), champions=(238,), positions=("MIDDLE",), builds=("lethality",)),
        AuditSpec(richer, "Nami UTILITY `utility_protection` vs enemy burst count", "Protective enchanter punished by burst-heavy enemies.", "enemy_burst_count", count_bins(), champions=(267,), positions=("UTILITY",), builds=("utility_protection",)),
        AuditSpec(richer, "Jinx BOTTOM `crit` vs enemy burst count", "Fragile crit carry into burst-heavy enemies.", "enemy_burst_count", count_bins(), champions=(222,), positions=("BOTTOM",), builds=("crit",)),
        AuditSpec(richer, "Malphite TOP `ar_tank` vs heavy damage-taken count", "Armor tank loses into teams with multiple high-soak targets.", "enemy_heavy_taken_count", count_bins(), champions=(54,), positions=("TOP",), builds=("ar_tank",)),
        AuditSpec(richer, "Viego JUNGLE any build vs enemy high-HP count", "On-hit bruiser jungler into high-HP enemy teams.", "enemy_high_hp_count", count_bins(), champions=(234,), positions=("JUNGLE",)),
        AuditSpec(retained, "Malphite all roles `ar_tank` vs enemy physical", "Original armor-stack audit, retained beyond TOP-only.", "enemy_physical", PHYSICAL_BINS, champions=(54,), builds=("ar_tank",)),
        AuditSpec(retained, "Galio all roles `mr_tank` vs enemy magic", "Original anti-magic tank family, broader than MIDDLE-only.", "enemy_magic", MAGIC_BINS, champions=(3,), builds=("mr_tank",)),
        AuditSpec(retained, "Nautilus all roles `mr_tank` vs enemy magic", "Top-20 MR-tank anti-magic case alongside Galio.", "enemy_magic", MAGIC_BINS, champions=(111,), builds=("mr_tank",)),
        AuditSpec(retained, "Nautilus all roles `ar_tank` vs enemy physical", "Physical-heavy enemy teams remain a support-tank check.", "enemy_physical", PHYSICAL_BINS, champions=(111,), builds=("ar_tank",)),
        AuditSpec(retained, "Darius TOP any build vs enemy range count", "Static team range pressure, stronger than lane-only range.", "enemy_ranged_count", range_count_bins(), champions=(122,), positions=("TOP",)),
        AuditSpec(retained, "Darius TOP any build vs same-role range", "User-requested static melee/ranged lane audit.", "same_role_range", (BinSpec(f"<= {RANGED_ATTACK_RANGE_THRESHOLD:.0f}", le(RANGED_ATTACK_RANGE_THRESHOLD)), BinSpec(f"> {RANGED_ATTACK_RANGE_THRESHOLD:.0f}", gt(RANGED_ATTACK_RANGE_THRESHOLD))), champions=(122,), positions=("TOP",)),
        AuditSpec(retained, "MasterYi JUNGLE any build vs enemy hard CC", "User-requested low-CC audit; unique even though gap is modest.", "enemy_hard_cc_count", count_bins(), champions=(11,), positions=("JUNGLE",)),
        AuditSpec(retained, "Selected enchanters UTILITY with skirmish allies", "Original enchanter-with-skirmishers synergy probe.", "ally_skirmish_count", (BinSpec("0", eq(0)), BinSpec("1", eq(1)), BinSpec(">= 2", ge(2))), positions=("UTILITY",), focus_condition="selected_enchanter"),
        AuditSpec(retained, "Low own-damage teams vs enemy heal/shield", "Original low-damage into sustain audit.", "enemy_heal_shield", HEAL_BINS, focus_condition="low_own_damage"),
        AuditSpec(retained, "Ambessa TOP `attack_damage` vs enemy damage", "Durable bruiser into enemy damage pressure.", "enemy_damage", DAMAGE_BINS, champions=(799,), positions=("TOP",), builds=("attack_damage",)),
        AuditSpec(retained, "LeeSin JUNGLE `ad_off_tank` vs enemy magic", "Bruiser jungler resisting magic-heavy enemies.", "enemy_magic", MAGIC_BINS, champions=(64,), positions=("JUNGLE",), builds=("ad_off_tank",)),
        AuditSpec(retained, "Thresh UTILITY `mr_tank` vs enemy magic", "MR-tank support anti-magic case.", "enemy_magic", MAGIC_BINS, champions=(412,), positions=("UTILITY",), builds=("mr_tank",)),
        AuditSpec(lower, f"Focus HP `<= {FOCUS_HP_LOW_THRESHOLD:.0f}` vs enemy burst count", "Broad HP-vs-burst check; useful but lower signal than champion-specific rows.", "enemy_burst_count", count_bins(), focus_condition="focus_hp_low"),
        AuditSpec(lower, f"Focus HP `>= {HIGH_HP_THRESHOLD:.0f}` vs enemy burst count", "High-HP slots also drop into burst stacks, so champion/build specificity matters.", "enemy_burst_count", count_bins(), focus_condition="focus_hp_high"),
        AuditSpec(lower, "Ahri MIDDLE `ability_power` vs heavy damage-taken count", "AP mid vs multiple high-soak enemies; weaker axis than frontline count.", "enemy_heavy_taken_count", count_bins(), champions=(103,), positions=("MIDDLE",), builds=("ability_power",)),
        AuditSpec(lower, "Kaisa BOTTOM `on_hit` vs heavy damage-taken count", "On-hit marksman vs high-soak enemies; frontline count is the stronger cut.", "enemy_heavy_taken_count", count_bins(), champions=(145,), positions=("BOTTOM",), builds=("on_hit",)),
        # --- New top-20 audits: enemy groups they are weak against / ally archetypes they synergize with.
        AuditSpec(synergy, "Yasuo MIDDLE `crit` with ally CC", "Yasuo's ult chains off ally knock-ups; scales with team CC.", "ally_cc", CC_BINS, champions=(157,), positions=("MIDDLE",), builds=("crit",)),
        AuditSpec(synergy, "Jhin BOTTOM `crit` with ally CC", "Immobile crit marksman; measured synergy with team CC is near flat.", "ally_cc", CC_BINS, champions=(202,), positions=("BOTTOM",), builds=("crit",)),
        AuditSpec(synergy, "Lulu UTILITY `utility_protection` with ally damage", "Enchanter value rises with carry damage to amplify and peel for.", "ally_damage", DAMAGE_BINS, champions=(117,), positions=("UTILITY",), builds=("utility_protection",)),
        AuditSpec(synergy, "Ezreal BOTTOM `attack_damage` vs enemy hard CC", "Skillshot poke marksman punished as enemy hard CC stacks.", "enemy_hard_cc_count", count_bins(), champions=(81,), positions=("BOTTOM",), builds=("attack_damage",)),
        AuditSpec(synergy, "Jayce TOP `attack_damage` vs enemy frontline count", "Poke bruiser empirically holds up into frontline-heavy teams; model heavily shrinks the effect.", "enemy_frontline_count", count_bins(), champions=(126,), positions=("TOP",), builds=("attack_damage",)),
        AuditSpec(synergy, "LeeSin JUNGLE `attack_damage` vs enemy scaling", "Early-tempo bruiser jungler fades as enemy scaling rises.", "enemy_scaling", SCALING_BINS, champions=(64,), positions=("JUNGLE",), builds=("attack_damage",)),
        AuditSpec(synergy, "Caitlyn BOTTOM `crit` vs enemy burst count", "Immobile siege ADC punished by multiple burst and dive threats.", "enemy_burst_count", count_bins(), champions=(51,), positions=("BOTTOM",), builds=("crit",)),
    )


# --- Group-level (semantic build/role) audit specs --------------------------
# Deterministic build/role groups pool every focus slot of a build family so per-bin
# n is large (median ~47k vs ~500 champion-specific). This collapses the sampling
# noise floor of the gap metric (~10.5 -> ~0.18 pp^2), making it meaningful to drive
# toward 0. Groups reuse the exact build vocabulary the relationship head consumes.
GROUP_TANK_BUILDS = ("ar_tank", "mr_tank", "ad_off_tank", "ap_off_tank")
GROUP_ARMOR_TANK = ("ar_tank",)
GROUP_MR_TANK = ("mr_tank",)
GROUP_AP_CASTER = ("ability_power",)
GROUP_ENCHANTER = ("utility_enchanter", "utility_protection")
GROUP_UTILITY_PROTECTION = ("utility_protection",)
GROUP_CRIT = ("crit",)
GROUP_ON_HIT = ("on_hit",)
GROUP_MARKSMAN = ("crit", "on_hit", "attack_damage")
GROUP_ATTACK_DAMAGE = ("attack_damage",)
GROUP_AD_OFF_TANK = ("ad_off_tank",)
GROUP_AD_FIGHTER = ("attack_damage", "ad_off_tank", "on_hit")
GROUP_LETHALITY = ("lethality",)

_GROUP_SECTION = "Group Trajectory Audit"


def group_audit_specs() -> tuple[AuditSpec, ...]:
    g = _GROUP_SECTION
    phys = continuous_bins(*CONTEXT_BIN_EDGES["physical"])
    magic = continuous_bins(*CONTEXT_BIN_EDGES["magic"])
    skirmish_bins = (BinSpec("0", eq(0)), BinSpec("1", eq(1)), BinSpec(">= 2", ge(2)))
    return (
        # Broad semantic counterparts to the promoted context examples audit. These
        # keep the `group_eb` target trainable at large n while covering the same
        # composition stories as the champion-specific report.
        AuditSpec(g, "Crit carries TOP vs enemy siege", "Melee/crit top carries into poke and siege.", "enemy_siege", SIEGE_BINS, builds=GROUP_CRIT, positions=("TOP",)),
        AuditSpec(g, "Lethality junglers vs enemy damage", "Burst junglers into high enemy damage.", "enemy_damage", DAMAGE_BINS, builds=GROUP_LETHALITY, positions=("JUNGLE",)),
        AuditSpec(g, "Crit carries MIDDLE vs enemy siege", "Mid-lane crit carries into poke and siege.", "enemy_siege", SIEGE_BINS, builds=GROUP_CRIT, positions=("MIDDLE",)),
        AuditSpec(g, "AP casters MIDDLE vs enemy scaling", "AP mids into scaling enemy compositions.", "enemy_scaling", SCALING_BINS, builds=GROUP_AP_CASTER, positions=("MIDDLE",)),
        AuditSpec(g, "MR tanks UTILITY with ally damage", "Engage supports with damage behind them.", "ally_damage", DAMAGE_BINS, builds=GROUP_MR_TANK, positions=("UTILITY",)),
        AuditSpec(g, "MR tanks MIDDLE vs enemy magic", "Mid-lane MR tanks into magic-heavy enemies.", "enemy_magic", magic, builds=GROUP_MR_TANK, positions=("MIDDLE",)),
        AuditSpec(g, "Armor tanks TOP vs enemy physical", "Top-lane armor tanks into physical-heavy enemies.", "enemy_physical", phys, builds=GROUP_ARMOR_TANK, positions=("TOP",)),
        AuditSpec(g, "AP casters MIDDLE vs enemy range count", "Short-range AP mids into enemy range pressure.", "enemy_ranged_count", range_count_bins(), builds=GROUP_AP_CASTER, positions=("MIDDLE",)),
        AuditSpec(g, "On-hit marksmen BOTTOM vs enemy range count", "On-hit bot carries into range-heavy enemies.", "enemy_ranged_count", range_count_bins(), builds=GROUP_ON_HIT, positions=("BOTTOM",)),
        AuditSpec(g, "Frontline tanks vs enemy physical", "Durable frontline gains value into AD-heavy enemies.", "enemy_physical", phys, builds=GROUP_TANK_BUILDS),
        AuditSpec(g, "Armor tanks vs enemy physical", "Armor itemization into physical damage.", "enemy_physical", phys, builds=GROUP_ARMOR_TANK),
        AuditSpec(g, "MR tanks vs enemy magic", "Magic-resist itemization into magic damage.", "enemy_magic", magic, builds=GROUP_MR_TANK),
        AuditSpec(g, "AP casters MIDDLE vs enemy frontline count", "AP damage scales with enemy durability targets.", "enemy_frontline_count", count_bins(), builds=GROUP_AP_CASTER, positions=("MIDDLE",)),
        AuditSpec(g, "AP casters vs enemy magic", "AP value vs magic-heavy enemy teams.", "enemy_magic", magic, builds=GROUP_AP_CASTER),
        AuditSpec(g, "Marksmen BOTTOM vs enemy frontline count", "Sustained DPS carries shred frontline.", "enemy_frontline_count", count_bins(), builds=GROUP_MARKSMAN, positions=("BOTTOM",)),
        AuditSpec(g, "On-hit carries vs enemy frontline count", "On-hit shred into durable enemies.", "enemy_frontline_count", count_bins(), builds=GROUP_ON_HIT),
        AuditSpec(g, "On-hit marksmen BOTTOM vs enemy frontline count", "On-hit bot carries into durable enemies.", "enemy_frontline_count", count_bins(), builds=GROUP_ON_HIT, positions=("BOTTOM",)),
        AuditSpec(g, "UTILITY slots vs enemy frontline count", "Support-role slots into enemy frontline stacking.", "enemy_frontline_count", count_bins(), positions=("UTILITY",)),
        AuditSpec(g, "Enchanters UTILITY vs enemy burst count", "Protective supports punished by concentrated burst.", "enemy_burst_count", count_bins(), builds=GROUP_ENCHANTER, positions=("UTILITY",)),
        AuditSpec(g, "Frontline tanks vs enemy burst count", "Durable frontline punished by stacked burst.", "enemy_burst_count", count_bins(), builds=GROUP_TANK_BUILDS),
        AuditSpec(g, "Armor tanks UTILITY vs enemy burst count", "Armor engage supports into concentrated burst.", "enemy_burst_count", count_bins(), builds=GROUP_ARMOR_TANK, positions=("UTILITY",)),
        AuditSpec(g, "MR tanks UTILITY vs enemy burst count", "MR engage supports into concentrated burst.", "enemy_burst_count", count_bins(), builds=GROUP_MR_TANK, positions=("UTILITY",)),
        AuditSpec(g, "Lethality assassins vs enemy burst count", "Fragile assassins into burst stacks.", "enemy_burst_count", count_bins(), builds=GROUP_LETHALITY),
        AuditSpec(g, "Lethality assassins MIDDLE vs enemy burst count", "Mid-lane assassins into burst stacking.", "enemy_burst_count", count_bins(), builds=GROUP_LETHALITY, positions=("MIDDLE",)),
        AuditSpec(g, "Utility protection UTILITY vs enemy burst count", "Protective enchanter builds into burst-heavy enemies.", "enemy_burst_count", count_bins(), builds=GROUP_UTILITY_PROTECTION, positions=("UTILITY",)),
        AuditSpec(g, "Marksmen BOTTOM vs enemy range count", "Melee/short carries into range pressure.", "enemy_ranged_count", range_count_bins(), builds=GROUP_MARKSMAN, positions=("BOTTOM",)),
        AuditSpec(g, "Frontline tanks vs heavy damage-taken count", "Tanks vs multiple high-soak enemies.", "enemy_heavy_taken_count", count_bins(), builds=GROUP_TANK_BUILDS),
        AuditSpec(g, "Armor tanks TOP vs heavy damage-taken count", "Top-lane armor tanks vs multiple high-soak enemies.", "enemy_heavy_taken_count", count_bins(), builds=GROUP_ARMOR_TANK, positions=("TOP",)),
        AuditSpec(g, "Frontline tanks vs enemy high-HP count", "Tanks vs high-HP enemy teams.", "enemy_high_hp_count", count_bins(), builds=GROUP_TANK_BUILDS),
        AuditSpec(g, "AP casters vs enemy high-HP count", "AP value vs high-HP enemy teams.", "enemy_high_hp_count", count_bins(), builds=GROUP_AP_CASTER),
        AuditSpec(g, "AD fighters JUNGLE vs enemy high-HP count", "Physical skirmish junglers into high-HP enemy teams.", "enemy_high_hp_count", count_bins(), builds=GROUP_AD_FIGHTER, positions=("JUNGLE",)),
        AuditSpec(g, "AD fighters TOP vs enemy range count", "Top-lane physical fighters into enemy range pressure.", "enemy_ranged_count", range_count_bins(), builds=GROUP_AD_FIGHTER, positions=("TOP",)),
        AuditSpec(g, "AD fighters TOP vs same-role range", "Top-lane physical fighters into ranged lane opponents.", "same_role_range", (BinSpec(f"<= {RANGED_ATTACK_RANGE_THRESHOLD:.0f}", le(RANGED_ATTACK_RANGE_THRESHOLD)), BinSpec(f"> {RANGED_ATTACK_RANGE_THRESHOLD:.0f}", gt(RANGED_ATTACK_RANGE_THRESHOLD))), builds=GROUP_AD_FIGHTER, positions=("TOP",)),
        AuditSpec(g, "On-hit junglers vs enemy hard CC", "On-hit jungle carries into enemy hard CC.", "enemy_hard_cc_count", count_bins(), builds=GROUP_ON_HIT, positions=("JUNGLE",)),
        AuditSpec(g, "Enchanters UTILITY with skirmish allies", "Enchanter synergy with skirmisher allies.", "ally_skirmish_count", skirmish_bins, builds=GROUP_ENCHANTER, positions=("UTILITY",)),
        AuditSpec(g, "Selected enchanters UTILITY with skirmish allies", "Champion-selected enchanters with skirmisher allies.", "ally_skirmish_count", skirmish_bins, positions=("UTILITY",), focus_condition="selected_enchanter"),
        AuditSpec(g, "Low own-damage teams vs enemy heal/shield", "Low-damage teams into enemy sustain and shielding.", "enemy_heal_shield", HEAL_BINS, focus_condition="low_own_damage"),
        AuditSpec(g, "Attack-damage TOP vs enemy damage", "Top-lane attack-damage fighters into enemy damage pressure.", "enemy_damage", DAMAGE_BINS, builds=GROUP_ATTACK_DAMAGE, positions=("TOP",)),
        AuditSpec(g, "AD off-tanks JUNGLE vs enemy magic", "Bruiser junglers into magic-heavy enemies.", "enemy_magic", magic, builds=GROUP_AD_OFF_TANK, positions=("JUNGLE",)),
        AuditSpec(g, "MR tanks UTILITY vs enemy magic", "MR support tanks into magic-heavy enemies.", "enemy_magic", magic, builds=GROUP_MR_TANK, positions=("UTILITY",)),
        AuditSpec(g, f"Focus HP <= {FOCUS_HP_LOW_THRESHOLD:.0f} vs enemy burst count", "Low-HP slots into enemy burst stacking.", "enemy_burst_count", count_bins(), focus_condition="focus_hp_low"),
        AuditSpec(g, f"Focus HP >= {HIGH_HP_THRESHOLD:.0f} vs enemy burst count", "High-HP slots into enemy burst stacking.", "enemy_burst_count", count_bins(), focus_condition="focus_hp_high"),
        AuditSpec(g, "AP casters MIDDLE vs heavy damage-taken count", "AP mids vs multiple high-soak enemies.", "enemy_heavy_taken_count", count_bins(), builds=GROUP_AP_CASTER, positions=("MIDDLE",)),
        AuditSpec(g, "On-hit marksmen BOTTOM vs heavy damage-taken count", "On-hit bot carries vs multiple high-soak enemies.", "enemy_heavy_taken_count", count_bins(), builds=GROUP_ON_HIT, positions=("BOTTOM",)),
        AuditSpec(g, "Crit carries MIDDLE with ally CC", "Mid-lane crit carries with allied CC setup.", "ally_cc", CC_BINS, builds=GROUP_CRIT, positions=("MIDDLE",)),
        AuditSpec(g, "Crit marksmen BOTTOM with ally CC", "Crit bot carries with allied CC setup.", "ally_cc", CC_BINS, builds=GROUP_CRIT, positions=("BOTTOM",)),
        AuditSpec(g, "Utility protection UTILITY with ally damage", "Protective utility supports with carry damage behind them.", "ally_damage", DAMAGE_BINS, builds=GROUP_UTILITY_PROTECTION, positions=("UTILITY",)),
        AuditSpec(g, "Attack-damage marksmen BOTTOM vs enemy hard CC", "Attack-damage bot carries into enemy hard CC.", "enemy_hard_cc_count", count_bins(), builds=GROUP_ATTACK_DAMAGE, positions=("BOTTOM",)),
        AuditSpec(g, "Attack-damage TOP vs enemy frontline count", "Top-lane attack-damage fighters into enemy frontline stacking.", "enemy_frontline_count", count_bins(), builds=GROUP_ATTACK_DAMAGE, positions=("TOP",)),
        AuditSpec(g, "Attack-damage junglers vs enemy scaling", "Physical damage junglers into scaling enemy compositions.", "enemy_scaling", SCALING_BINS, builds=GROUP_ATTACK_DAMAGE, positions=("JUNGLE",)),
        AuditSpec(g, "Crit marksmen BOTTOM vs enemy burst count", "Crit carries into burst-heavy enemies.", "enemy_burst_count", count_bins(), builds=GROUP_CRIT, positions=("BOTTOM",)),
    )


_TRAIN_CORE_GROUP_REPORT_ONLY_TITLES = frozenset(
    {
        "MR tanks MIDDLE vs enemy magic",
        "Armor tanks TOP vs enemy physical",
        "On-hit marksmen BOTTOM vs enemy frontline count",
        "MR tanks UTILITY vs enemy burst count",
        "Lethality assassins MIDDLE vs enemy burst count",
        "Armor tanks TOP vs heavy damage-taken count",
        "On-hit junglers vs enemy hard CC",
        "Selected enchanters UTILITY with skirmish allies",
        "AD off-tanks JUNGLE vs enemy magic",
        "MR tanks UTILITY vs enemy magic",
        "AP casters MIDDLE vs heavy damage-taken count",
        "On-hit marksmen BOTTOM vs heavy damage-taken count",
    }
)


def training_group_audit_specs(surface: str = "train_core") -> tuple[AuditSpec, ...]:
    """Return the group specs used by train-time calibration.

    The full group audit remains the reporting surface. ``train_core`` removes
    narrow or duplicated rows that were useful for diagnosis but too noisy as
    direct loss targets.
    """

    specs = group_audit_specs()
    if surface == "full":
        return specs
    if surface != "train_core":
        raise ValueError("training group audit surface must be 'full' or 'train_core'")
    return tuple(
        spec for spec in specs if spec.title not in _TRAIN_CORE_GROUP_REPORT_ONLY_TITLES
    )


def eb_shrink_targets(
    counts: np.ndarray, means: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian empirical-Bayes shrinkage of bin win rates toward the row mean.

    Shrinks each bin's observed win rate toward the n-weighted row mean by its
    sampling variance; the between-bin variance tau^2 is estimated by method of
    moments. Returns (eb_target, eb_var) where eb_var is the residual variance of
    the EB target estimate (the debiasing term). Inputs/outputs are in [0, 1].
    """
    counts = np.asarray(counts, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0 or counts.size == 0:
        return means.copy(), np.zeros_like(means)
    mu = float(np.sum(counts * means) / total)
    s2 = means * (1.0 - means) / np.maximum(counts, 1.0)
    weight = counts / total
    total_spread = float(np.sum(weight * (means - mu) ** 2))
    tau2 = max(0.0, total_spread - float(np.sum(weight * s2)))
    denom = tau2 + s2
    shrink = np.zeros_like(means)
    np.divide(tau2, denom, out=shrink, where=denom > 0)
    eb = shrink * means + (1.0 - shrink) * mu
    eb_var = np.zeros_like(means)
    np.divide(tau2 * s2, denom, out=eb_var, where=denom > 0)
    return eb, eb_var


__all__ = [
    "AuditSpec",
    "BinSpec",
    "POSITIONS",
    "audit_specs",
    "between",
    "continuous_bins",
    "count_bins",
    "eb_shrink_targets",
    "eq",
    "ge",
    "group_audit_specs",
    "gt",
    "le",
    "lt",
    "range_count_bins",
    "training_group_audit_specs",
]
