# Identity Encoder Metric Surfaces

This file lists the input metrics available to each classification identity
encoder. Counts are generated from the current code surfaces.

## Summary

| Encoder | Grain | Input count | Source |
| --- | --- | ---: | --- |
| Full-game | `(champion_id, teamposition_id, build_id)` | 215 metrics | `full_game_metric_columns()` |
| Static identity | `champion_id` | 47 features | `static_feature_names()` |
| Temporal | `(champion_id, teamposition_id, build_id)` | 47 buckets x 51 metrics = 2397 possible scalar values per identity | `METRIC_NAMES` over the temporal bucket axis |

The full-game default includes all 215 metrics: 66 raw profile metrics, 89
derived profile metrics, and 60 context features. Use
`full_game_metric_columns(include_context=False)` or CLI `--profile-only` only
for legacy 155-column profile-only matrices.

The temporal metrics are listed once. Each metric is available across the 47
minute buckets (`0..45` plus `46_plus`); unobserved buckets are masked during
training and evaluation.

## Full-Game Encoder

### Raw Profile Metrics (66)

- `win`
- `firstbloodkill`
- `firstbloodassist`
- `firsttowerkill`
- `firsttowerassist`
- `largestkillingspree`
- `largestmultikill`
- `largestcriticalstrike`
- `healthmax`
- `lifesteal`
- `movementspeed`
- `omnivamp`
- `physicalvamp`
- `spellvamp`
- `armor`
- `magicresist`
- `abilitypower`
- `attackdamage`
- `attackspeed`
- `kills`
- `deaths`
- `assists`
- `doublekills`
- `triplekills`
- `killingsprees`
- `goldearned`
- `champexperience`
- `totaldamagedealt`
- `totaldamagedealttochampions`
- `physicaldamagedealt`
- `physicaldamagedealttochampions`
- `magicdamagedealt`
- `magicdamagedealttochampions`
- `truedamagedealt`
- `truedamagedealttochampions`
- `damagedealttobuildings`
- `damagedealttoturrets`
- `damagedealttoobjectives`
- `damagedealttoepicmonsters`
- `totaldamagetaken`
- `physicaldamagetaken`
- `magicdamagetaken`
- `truedamagetaken`
- `damageselfmitigated`
- `totalheal`
- `totalhealsonteammates`
- `totaldamageshieldedonteammates`
- `timeccingothers`
- `totaltimeccdealt`
- `totalminionskilled`
- `neutralminionskilled`
- `totalallyjungleminionskilled`
- `totalenemyjungleminionskilled`
- `baronkills`
- `dragonkills`
- `inhibitorkills`
- `inhibitortakedowns`
- `inhibitorslost`
- `turretkills`
- `turrettakedowns`
- `turretslost`
- `visionscore`
- `wardsplaced`
- `wardskilled`
- `detectorwardsplaced`
- `visionwardsboughtingame`

### Derived Profile Metrics (89)

- `first_blood_participation`
- `first_tower_participation`
- `early_snowball_participation`
- `durability_total`
- `durability_total_to_deaths_ratio`
- `self_heal`
- `self_heal_to_durability_total_ratio`
- `vamp_sustain`
- `healthmax_to_goldearned_ratio`
- `durability_total_to_healthmax_ratio`
- `magicdamagetaken_to_durability_total_ratio`
- `physicaldamagetaken_to_durability_total_ratio`
- `damageselfmitigated_to_durability_total_ratio`
- `damageselfmitigated_to_goldearned_ratio`
- `durability_total_to_goldearned_ratio`
- `damage_taken_to_goldearned_ratio`
- `totaldamagetaken_to_deaths_ratio`
- `self_heal_to_goldearned_ratio`
- `self_heal_to_deaths_ratio`
- `totalheal_to_goldearned_ratio`
- `armor_to_goldearned_ratio`
- `magicresist_to_goldearned_ratio`
- `physicaldamagedealttochampions_share`
- `magicdamagedealttochampions_share`
- `truedamagedealttochampions_share`
- `champion_damage_to_total_damage_ratio`
- `champion_damage_share_to_deaths_ratio`
- `totaldamagedealttochampions_to_goldearned_ratio`
- `totaldamagedealttochampions_to_deaths_ratio`
- `physicaldamagedealt_share`
- `magicdamagedealt_share`
- `truedamagedealt_share`
- `abilitypower_to_goldearned_ratio`
- `attackdamage_to_goldearned_ratio`
- `takedowns`
- `kills_to_deaths_ratio`
- `assists_to_deaths_ratio`
- `takedowns_to_deaths_ratio`
- `kills_to_assists_ratio`
- `kills_to_assists_ratio_to_goldearned_ratio`
- `visionscore_to_ward_actions_ratio`
- `visionscore_to_goldearned_ratio`
- `wardskilled_to_wardsplaced_ratio`
- `jungle_minions`
- `jungle_minion_share`
- `jungle_minions_to_lane_minions_ratio`
- `total_farm`
- `enemy_to_ally_jungle_minions_ratio`
- `enemy_jungle_minion_share`
- `total_farm_to_goldearned_ratio`
- `total_farm_to_deaths_ratio`
- `champexperience_to_goldearned_ratio`
- `structure_takedowns`
- `structure_losses`
- `structure_damage`
- `structure_takedowns_to_structure_damage_ratio`
- `structure_damage_to_goldearned_ratio`
- `structure_damage_to_deaths_ratio`
- `structure_takedowns_to_goldearned_ratio`
- `structure_takedowns_to_deaths_ratio`
- `structure_takedowns_to_losses_ratio`
- `structure_net_control`
- `cc_effectiveness_ratio`
- `cc_to_assists_ratio`
- `epic_kills`
- `objective_neutral_minions`
- `objective_damage`
- `epic_kills_to_damagedealttoobjectives_ratio`
- `objective_damage_to_goldearned_ratio`
- `objective_damage_to_total_damage_ratio`
- `epic_monster_damage_to_objective_damage_ratio`
- `epic_kills_to_goldearned_ratio`
- `damagedealttoobjectives_per_epic_kill_per_gold`
- `ally_support`
- `totalhealsonteammates_to_goldearned_ratio`
- `totaldamageshieldedonteammates_to_goldearned_ratio`
- `ally_support_to_goldearned_ratio`
- `ally_support_to_assists_ratio`
- `totalhealsonteammates_to_deaths_ratio`
- `totaldamageshieldedonteammates_to_deaths_ratio`
- `ally_support_to_deaths_ratio`
- `physicaldamagetaken_share`
- `magicdamagetaken_share`
- `truedamagetaken_share`
- `champion_damage_to_damage_taken_ratio`
- `net_combat_damage`
- `net_kills`
- `largestcriticalstrike_to_attackdamage_ratio`
- `goldearned_to_deaths_ratio`

### Context Metrics (60)

Team-share features:

- `kills_team_share`
- `deaths_team_share`
- `assists_team_share`
- `takedowns_team_share`
- `gold_team_share`
- `xp_team_share`
- `total_farm_team_share`
- `lane_farm_team_share`
- `jungle_farm_team_share`
- `champion_damage_team_share`
- `total_damage_team_share`
- `damage_taken_team_share`
- `self_mitigated_team_share`
- `durability_total_team_share`
- `ally_support_team_share`
- `cc_team_share`
- `vision_score_team_share`
- `ward_actions_team_share`
- `objective_damage_team_share`
- `structure_damage_team_share`
- `epic_kills_team_share`
- `structure_takedowns_team_share`

Team-concentration features:

- `gold_team_concentration`
- `xp_team_concentration`
- `total_farm_team_concentration`
- `champion_damage_team_concentration`
- `kills_team_concentration`
- `damage_taken_team_concentration`

Role-matchup features:

- `kills_vs_role_opponent_diff`
- `kills_vs_role_opponent_advantage`
- `deaths_vs_role_opponent_diff`
- `deaths_vs_role_opponent_advantage`
- `takedowns_vs_role_opponent_diff`
- `takedowns_vs_role_opponent_advantage`
- `gold_vs_role_opponent_diff`
- `gold_vs_role_opponent_advantage`
- `xp_vs_role_opponent_diff`
- `xp_vs_role_opponent_advantage`
- `total_farm_vs_role_opponent_diff`
- `total_farm_vs_role_opponent_advantage`
- `lane_farm_vs_role_opponent_diff`
- `lane_farm_vs_role_opponent_advantage`
- `champion_damage_vs_role_opponent_diff`
- `champion_damage_vs_role_opponent_advantage`
- `damage_taken_vs_role_opponent_diff`
- `damage_taken_vs_role_opponent_advantage`
- `vision_score_vs_role_opponent_diff`
- `vision_score_vs_role_opponent_advantage`
- `objective_damage_vs_role_opponent_diff`
- `objective_damage_vs_role_opponent_advantage`
- `structure_damage_vs_role_opponent_diff`
- `structure_damage_vs_role_opponent_advantage`
- `gold_share_vs_role_opponent_diff`
- `gold_share_vs_role_opponent_advantage`
- `xp_share_vs_role_opponent_diff`
- `xp_share_vs_role_opponent_advantage`
- `total_farm_share_vs_role_opponent_diff`
- `total_farm_share_vs_role_opponent_advantage`
- `champion_damage_share_vs_role_opponent_diff`
- `champion_damage_share_vs_role_opponent_advantage`

## Static Identity Encoder

The static encoder consumes deterministic champion dictionary features only.
Role, build, empirical priors, win rates, matchup rates, support counts, and
challenge-derived columns are rejected.

- `acquisitionRadius_flat`
- `acquisitionRadius_perLevel`
- `armor_flat`
- `armor_perLevel`
- `attackCastTime_flat`
- `attackCastTime_perLevel`
- `attackDamage_flat`
- `attackDamage_perLevel`
- `attackDelayOffset_flat`
- `attackDelayOffset_perLevel`
- `attackRange_flat`
- `attackRange_perLevel`
- `attackSpeed_flat`
- `attackSpeed_perLevel`
- `attackSpeedRatio_flat`
- `attackSpeedRatio_perLevel`
- `attackTotalTime_flat`
- `attackTotalTime_perLevel`
- `criticalStrikeDamage_flat`
- `criticalStrikeDamage_perLevel`
- `criticalStrikeDamageModifier_flat`
- `criticalStrikeDamageModifier_perLevel`
- `gameplayRadius_flat`
- `gameplayRadius_perLevel`
- `health_flat`
- `health_perLevel`
- `healthRegen_flat`
- `healthRegen_perLevel`
- `magicResistance_flat`
- `magicResistance_perLevel`
- `mana_flat`
- `mana_perLevel`
- `manaRegen_flat`
- `manaRegen_perLevel`
- `movespeed_flat`
- `movespeed_perLevel`
- `pathingRadius_flat`
- `pathingRadius_perLevel`
- `selectionRadius_flat`
- `selectionRadius_perLevel`
- `health_l18`
- `healthRegen_l18`
- `mana_l18`
- `manaRegen_l18`
- `armor_l18`
- `magicResistance_l18`
- `attackDamage_l18`

## Temporal Encoder

The temporal encoder consumes one standardized trajectory tensor per
`(champion_id, teamposition_id, build_id)` identity. Each metric below appears
once per temporal bucket, but it is listed only once here.

Per-frame stat metrics:

- `abilityhaste`
- `abilitypower`
- `armor`
- `armorpen`
- `armorpenpercent`
- `attackdamage`
- `attackspeed`
- `bonusarmorpenpercent`
- `bonusmagicpenpercent`
- `ccreduction`
- `cooldownreduction`
- `health`
- `healthmax`
- `healthregen`
- `lifesteal`
- `magicpen`
- `magicpenpercent`
- `magicresist`
- `movementspeed`
- `omnivamp`
- `physicalvamp`
- `power`
- `powermax`
- `powerregen`
- `spellvamp`
- `currentgold`
- `magicdamagedone`
- `magicdamagedonetochampions`
- `magicdamagetaken`
- `physicaldamagedone`
- `physicaldamagedonetochampions`
- `physicaldamagetaken`
- `totaldamagedone`
- `totaldamagedonetochampions`
- `totaldamagetaken`
- `truedamagedone`
- `truedamagedonetochampions`
- `truedamagetaken`
- `goldpersecond`
- `jungleminionskilled`
- `level`
- `minionskilled`
- `timeenemyspentcontrolled`
- `totalgold`
- `xp`

Event metrics:

- `kills`
- `assists`
- `deaths`
- `plate_top`
- `plate_mid`
- `plate_bot`
