# participant_build_minute_averages

Populated by `5135_participant_build_stat_averages_build.sql`.

Per-build averages for each champion × role × build combination. All rate-based fields are normalised to **per-minute** values (raw value × 60 / timeplayed). Ratio and composite fields are derived from those per-minute averages — because both numerator and denominator carry the same time factor, it cancels and the resulting ratio equals the true game-level ratio.

---

## Source join

`participant_stats` INNER JOIN `participant_item_value_totals` on `(matchid, participantid)`.  
Games with `timeplayed = 0` or a null `championid` are excluded.  
`build` comes from `ivt.highest_value_label`.

---

## Fields

**Total: 88 columns** — 5 identity/metadata, 66 per-minute averages, 17 derived.

### Identity (5)

| Column | Type | Notes |
|---|---|---|
| `championid` | Int32 | |
| `championname` | LowCardinality(String) | Resolved via `championid_name_map_dict` |
| `teamposition` | LowCardinality(String) | |
| `build` | LowCardinality(String) | Highest-value item label |
| `participant_count` | UInt64 | Number of games in this group |

### Per-minute averages (66)

Fields marked **raw avg** are not time-normalised — they are peak/state values where dividing by time would be semantically incorrect.

#### Combat events
| Column | Normalisation |
|---|---|
| `win` | raw avg (boolean) |
| `firstbloodkill` | raw avg (boolean) |
| `firstbloodassist` | raw avg (boolean) |
| `firsttowerkill` | raw avg (boolean) |
| `firsttowerassist` | raw avg (boolean) |
| `kills` | per minute |
| `deaths` | per minute |
| `assists` | per minute |
| `doublekills` | per minute |
| `triplekills` | per minute |
| `killingsprees` | per minute |
| `largestkillingspree` | raw avg (peak value) |
| `largestmultikill` | raw avg (peak value) |
| `champlevel` | raw avg (final state, capped at 18) |
| `champexperience` | per minute |

#### Economy
| Column | Normalisation |
|---|---|
| `goldearned` | per minute |
| `goldspent` | per minute |

#### Damage dealt
| Column | Normalisation |
|---|---|
| `totaldamagedealt` | per minute |
| `totaldamagedealttochampions` | per minute |
| `physicaldamagedealt` | per minute |
| `physicaldamagedealttochampions` | per minute |
| `magicdamagedealt` | per minute |
| `magicdamagedealttochampions` | per minute |
| `truedamagedealt` | per minute |
| `truedamagedealttochampions` | per minute |
| `damagedealttobuildings` | per minute |
| `damagedealttoturrets` | per minute |
| `damagedealttoobjectives` | per minute |
| `damagedealttoepicmonsters` | per minute |

#### Damage taken & mitigation
| Column | Normalisation |
|---|---|
| `totaldamagetaken` | per minute |
| `physicaldamagetaken` | per minute |
| `magicdamagetaken` | per minute |
| `truedamagetaken` | per minute |
| `damageselfmitigated` | per minute |

#### Healing & shielding
| Column | Normalisation |
|---|---|
| `totalheal` | per minute |
| `totalhealsonteammates` | per minute |
| `totaldamageshieldedonteammates` | per minute |

#### Crowd control
| Column | Normalisation |
|---|---|
| `timeccingothers` | per minute |
| `totaltimeccdealt` | per minute |

#### Farming
| Column | Normalisation |
|---|---|
| `totalminionskilled` | per minute |
| `neutralminionskilled` | per minute |
| `totalallyjungleminionskilled` | per minute |
| `totalenemyjungleminionskilled` | per minute |

#### Objectives
| Column | Normalisation |
|---|---|
| `baronkills` | per minute |
| `dragonkills` | per minute |
| `inhibitorkills` | raw avg (capped at 3) |
| `inhibitortakedowns` | raw avg (capped at 3) |
| `inhibitorslost` | raw avg (capped at 3) |
| `turretkills` | raw avg (capped at 11) |
| `turrettakedowns` | raw avg (capped at 11) |
| `turretslost` | raw avg (capped at 11) |
| `objectivesstolen` | per minute |
| `objectivesstolenassists` | per minute |

#### Vision
| Column | Normalisation |
|---|---|
| `visionscore` | per minute |
| `wardsplaced` | per minute |
| `wardskilled` | per minute |
| `detectorwardsplaced` | per minute |
| `visionwardsboughtingame` | per minute |

#### Time & misc
| Column | Normalisation |
|---|---|
| `totaltimespentdead` | per minute |
| `longesttimespentliving` | raw avg (peak value) |
| `pings` | per minute (sum of all ping types) |

---

### Derived columns (17)

All derivations are computed in a final `SELECT` over the `aggregated` CTE. Because per-minute values retain the true ratio between any two stats (the time factor cancels), these are equivalent to computing the ratio from raw game totals.

#### Combat
| Column | Formula | Interpretation |
|---|---|---|
| `kda` | `(kills + assists) / greatest(deaths, 0.001)` | Kill/death/assist ratio; floor prevents division by zero for deathless builds |
| `ka` | `kills + assists` | Kill participation proxy (kill + assist sum per minute) |
| `firstblood_participation` | `firstbloodkill + firstbloodassist` | Average first blood involvement (0–1 range; >1 not possible) |

#### Support / protection
| Column | Formula | Interpretation |
|---|---|---|
| `totalprotectiononteammates` | `totalhealsonteammates + totaldamageshieldedonteammates` | Combined teammate healing and shielding per minute |

#### Durability
| Column | Formula | Interpretation |
|---|---|---|
| `expected_frontline_index` | `(totaldamagetaken + damageselfmitigated) / greatest(totaldamagedealttochampions, 1)` | Ratio of damage absorbed to damage dealt to champions; high values indicate a tank/frontline role |
| `expected_effective_durability` | `totaldamagetaken + damageselfmitigated + totalheal` | Total effective health budget per minute (damage soaked + healed) |

#### Vision
| Column | Formula | Interpretation |
|---|---|---|
| `expected_vision_denial_ratio` | `wardskilled / greatest(wardsplaced + wardskilled, 0.001)` | Share of ward activity that is denial vs. placement; ranges 0–1 |
| `expected_vision_action_score` | `wardsplaced + 1.5×wardskilled + 2×detectorwardsplaced + 2×visionwardsboughtingame` | Weighted vision contribution; denial and control wards weighted more heavily |

#### Objectives
| Column | Formula | Interpretation |
|---|---|---|
| `expected_epic_objective_score` | `dragonkills + 2×baronkills + 2×objectivesstolen + objectivesstolenassists` | Weighted epic objective contribution; baron and steals weighted 2× |
| `expected_structure_score` | `turretkills + turrettakedowns + 2×inhibitorkills + 2×inhibitortakedowns` | Weighted structure pressure; inhibitors weighted 2× |

#### Snowball
| Column | Formula | Interpretation |
|---|---|---|
| `expected_snowball_score` | `doublekills + 2×triplekills + killingsprees + 0.5×largestkillingspree` | Composite multi-kill and streak score |

#### Efficiency
| Column | Formula | Interpretation |
|---|---|---|
| `expected_damage_per_gold` | `totaldamagedealttochampions / greatest(goldearned, 1)` | Champion damage output per gold earned |
| `expected_physical_damage_share` | `physicaldamagedealttochampions / greatest(totaldamagedealttochampions, 1)` | Fraction of champion damage that is physical (0–1) |
| `expected_magic_damage_share` | `magicdamagedealttochampions / greatest(totaldamagedealttochampions, 1)` | Fraction of champion damage that is magic (0–1) |
| `expected_true_damage_share` | `truedamagedealttochampions / greatest(totaldamagedealttochampions, 1)` | Fraction of champion damage that is true damage (0–1) |
| `damage_to_taken_ratio` | `totaldamagedealttochampions / greatest(totaldamagetaken, 1)` | Champion damage output relative to damage received |

#### Farming
| Column | Formula | Interpretation |
|---|---|---|
| `totalcs` | `totalminionskilled + neutralminionskilled` | Total creep score per minute |
