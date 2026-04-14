# timeline p_stats

`p_stats` is an abbreviation for `participant_stats` in this metrics catalog. The underlying source table remains `game_data.tl_participant_stats`.

| id | name | data_source | description | calculation | version | fields |
|---|---|---|---|---|---|---|
| TLPS_R_001 | game_data.tl_p_stats.matchid | game_data.tl_participant_stats (see note above) | Raw value for column `matchid` from `game_data.tl_participant_stats` (see note above). | identity(matchid) | 1.0.0 | matchid |
| TLPS_R_002 | game_data.tl_p_stats.frame_timestamp | game_data.tl_participant_stats (see note above) | Raw value for column `frame_timestamp` from `game_data.tl_participant_stats` (see note above). | identity(frame_timestamp) | 1.0.0 | frame_timestamp |
| TLPS_R_003 | game_data.tl_p_stats.participantid | game_data.tl_participant_stats (see note above) | Raw value for column `participantid` from `game_data.tl_participant_stats` (see note above). | identity(participantid) | 1.0.0 | participantid |
| TLPS_R_004 | game_data.tl_p_stats.abilityhaste | game_data.tl_participant_stats (see note above) | Raw value for column `abilityhaste` from `game_data.tl_participant_stats` (see note above). | identity(abilityhaste) | 1.0.0 | abilityhaste |
| TLPS_R_005 | game_data.tl_p_stats.abilitypower | game_data.tl_participant_stats (see note above) | Raw value for column `abilitypower` from `game_data.tl_participant_stats` (see note above). | identity(abilitypower) | 1.0.0 | abilitypower |
| TLPS_R_006 | game_data.tl_p_stats.armor | game_data.tl_participant_stats (see note above) | Raw value for column `armor` from `game_data.tl_participant_stats` (see note above). | identity(armor) | 1.0.0 | armor |
| TLPS_R_007 | game_data.tl_p_stats.attackdamage | game_data.tl_participant_stats (see note above) | Raw value for column `attackdamage` from `game_data.tl_participant_stats` (see note above). | identity(attackdamage) | 1.0.0 | attackdamage |
| TLPS_R_008 | game_data.tl_p_stats.attackspeed | game_data.tl_participant_stats (see note above) | Raw value for column `attackspeed` from `game_data.tl_participant_stats` (see note above). | identity(attackspeed) | 1.0.0 | attackspeed |
| TLPS_R_009 | game_data.tl_p_stats.ccreduction | game_data.tl_participant_stats (see note above) | Raw value for column `ccreduction` from `game_data.tl_participant_stats` (see note above). | identity(ccreduction) | 1.0.0 | ccreduction |
| TLPS_R_010 | game_data.tl_p_stats.cooldownreduction | game_data.tl_participant_stats (see note above) | Raw value for column `cooldownreduction` from `game_data.tl_participant_stats` (see note above). | identity(cooldownreduction) | 1.0.0 | cooldownreduction |
| TLPS_R_011 | game_data.tl_p_stats.health | game_data.tl_participant_stats (see note above) | Raw value for column `health` from `game_data.tl_participant_stats` (see note above). | identity(health) | 1.0.0 | health |
| TLPS_R_012 | game_data.tl_p_stats.healthmax | game_data.tl_participant_stats (see note above) | Raw value for column `healthmax` from `game_data.tl_participant_stats` (see note above). | identity(healthmax) | 1.0.0 | healthmax |
| TLPS_R_013 | game_data.tl_p_stats.healthregen | game_data.tl_participant_stats (see note above) | Raw value for column `healthregen` from `game_data.tl_participant_stats` (see note above). | identity(healthregen) | 1.0.0 | healthregen |
| TLPS_R_014 | game_data.tl_p_stats.magicresist | game_data.tl_participant_stats (see note above) | Raw value for column `magicresist` from `game_data.tl_participant_stats` (see note above). | identity(magicresist) | 1.0.0 | magicresist |
| TLPS_R_015 | game_data.tl_p_stats.movementspeed | game_data.tl_participant_stats (see note above) | Raw value for column `movementspeed` from `game_data.tl_participant_stats` (see note above). | identity(movementspeed) | 1.0.0 | movementspeed |
| TLPS_R_016 | game_data.tl_p_stats.power | game_data.tl_participant_stats (see note above) | Raw value for column `power` from `game_data.tl_participant_stats` (see note above). | identity(power) | 1.0.0 | power |
| TLPS_R_017 | game_data.tl_p_stats.powermax | game_data.tl_participant_stats (see note above) | Raw value for column `powermax` from `game_data.tl_participant_stats` (see note above). | identity(powermax) | 1.0.0 | powermax |
| TLPS_R_018 | game_data.tl_p_stats.powerregen | game_data.tl_participant_stats (see note above) | Raw value for column `powerregen` from `game_data.tl_participant_stats` (see note above). | identity(powerregen) | 1.0.0 | powerregen |
| TLPS_R_019 | game_data.tl_p_stats.payload | game_data.tl_participant_stats (see note above) | Raw value for column `payload` from `game_data.tl_participant_stats` (see note above). | identity(payload) | 1.0.0 | payload |
| TLPS_R_020 | game_data.tl_p_stats.currentgold | game_data.tl_participant_stats (see note above) | Raw value for column `currentgold` from `game_data.tl_participant_stats` (see note above). | identity(currentgold) | 1.0.0 | currentgold |
| TLPS_R_021 | game_data.tl_p_stats.magicdamagedone | game_data.tl_participant_stats (see note above) | Raw value for column `magicdamagedone` from `game_data.tl_participant_stats` (see note above). | identity(magicdamagedone) | 1.0.0 | magicdamagedone |
| TLPS_R_022 | game_data.tl_p_stats.magicdamagedonetochampions | game_data.tl_participant_stats (see note above) | Raw value for column `magicdamagedonetochampions` from `game_data.tl_participant_stats` (see note above). | identity(magicdamagedonetochampions) | 1.0.0 | magicdamagedonetochampions |
| TLPS_R_023 | game_data.tl_p_stats.magicdamagetaken | game_data.tl_participant_stats (see note above) | Raw value for column `magicdamagetaken` from `game_data.tl_participant_stats` (see note above). | identity(magicdamagetaken) | 1.0.0 | magicdamagetaken |
| TLPS_R_024 | game_data.tl_p_stats.physicaldamagedone | game_data.tl_participant_stats (see note above) | Raw value for column `physicaldamagedone` from `game_data.tl_participant_stats` (see note above). | identity(physicaldamagedone) | 1.0.0 | physicaldamagedone |
| TLPS_R_025 | game_data.tl_p_stats.physicaldamagedonetochampions | game_data.tl_participant_stats (see note above) | Raw value for column `physicaldamagedonetochampions` from `game_data.tl_participant_stats` (see note above). | identity(physicaldamagedonetochampions) | 1.0.0 | physicaldamagedonetochampions |
| TLPS_R_026 | game_data.tl_p_stats.physicaldamagetaken | game_data.tl_participant_stats (see note above) | Raw value for column `physicaldamagetaken` from `game_data.tl_participant_stats` (see note above). | identity(physicaldamagetaken) | 1.0.0 | physicaldamagetaken |
| TLPS_R_027 | game_data.tl_p_stats.totaldamagedone | game_data.tl_participant_stats (see note above) | Raw value for column `totaldamagedone` from `game_data.tl_participant_stats` (see note above). | identity(totaldamagedone) | 1.0.0 | totaldamagedone |
| TLPS_R_028 | game_data.tl_p_stats.totaldamagedonetochampions | game_data.tl_participant_stats (see note above) | Raw value for column `totaldamagedonetochampions` from `game_data.tl_participant_stats` (see note above). | identity(totaldamagedonetochampions) | 1.0.0 | totaldamagedonetochampions |
| TLPS_R_029 | game_data.tl_p_stats.totaldamagetaken | game_data.tl_participant_stats (see note above) | Raw value for column `totaldamagetaken` from `game_data.tl_participant_stats` (see note above). | identity(totaldamagetaken) | 1.0.0 | totaldamagetaken |
| TLPS_R_030 | game_data.tl_p_stats.truedamagedone | game_data.tl_participant_stats (see note above) | Raw value for column `truedamagedone` from `game_data.tl_participant_stats` (see note above). | identity(truedamagedone) | 1.0.0 | truedamagedone |
| TLPS_R_031 | game_data.tl_p_stats.truedamagedonetochampions | game_data.tl_participant_stats (see note above) | Raw value for column `truedamagedonetochampions` from `game_data.tl_participant_stats` (see note above). | identity(truedamagedonetochampions) | 1.0.0 | truedamagedonetochampions |
| TLPS_R_032 | game_data.tl_p_stats.truedamagetaken | game_data.tl_participant_stats (see note above) | Raw value for column `truedamagetaken` from `game_data.tl_participant_stats` (see note above). | identity(truedamagetaken) | 1.0.0 | truedamagetaken |
| TLPS_R_033 | game_data.tl_p_stats.goldpersecond | game_data.tl_participant_stats (see note above) | Raw value for column `goldpersecond` from `game_data.tl_participant_stats` (see note above). | identity(goldpersecond) | 1.0.0 | goldpersecond |
| TLPS_R_034 | game_data.tl_p_stats.jungleminionskilled | game_data.tl_participant_stats (see note above) | Raw value for column `jungleminionskilled` from `game_data.tl_participant_stats` (see note above). | identity(jungleminionskilled) | 1.0.0 | jungleminionskilled |
| TLPS_R_035 | game_data.tl_p_stats.level | game_data.tl_participant_stats (see note above) | Raw value for column `level` from `game_data.tl_participant_stats` (see note above). | identity(level) | 1.0.0 | level |
| TLPS_R_036 | game_data.tl_p_stats.minionskilled | game_data.tl_participant_stats (see note above) | Raw value for column `minionskilled` from `game_data.tl_participant_stats` (see note above). | identity(minionskilled) | 1.0.0 | minionskilled |
| TLPS_R_037 | game_data.tl_p_stats.position_x | game_data.tl_participant_stats (see note above) | Raw value for column `position_x` from `game_data.tl_participant_stats` (see note above). | identity(position_x) | 1.0.0 | position_x |
| TLPS_R_038 | game_data.tl_p_stats.position_y | game_data.tl_participant_stats (see note above) | Raw value for column `position_y` from `game_data.tl_participant_stats` (see note above). | identity(position_y) | 1.0.0 | position_y |
| TLPS_R_039 | game_data.tl_p_stats.timeenemyspentcontrolled | game_data.tl_participant_stats (see note above) | Raw value for column `timeenemyspentcontrolled` from `game_data.tl_participant_stats` (see note above). | identity(timeenemyspentcontrolled) | 1.0.0 | timeenemyspentcontrolled |
| TLPS_R_040 | game_data.tl_p_stats.totalgold | game_data.tl_participant_stats (see note above) | Raw value for column `totalgold` from `game_data.tl_participant_stats` (see note above). | identity(totalgold) | 1.0.0 | totalgold |
| TLPS_R_041 | game_data.tl_p_stats.xp | game_data.tl_participant_stats (see note above) | Raw value for column `xp` from `game_data.tl_participant_stats` (see note above). | identity(xp) | 1.0.0 | xp |
| TLPS_S_001 | synthetic.tl_p_stats.nonChampionDamageDone | game_data.tl_participant_stats (see note above) | Total damage done to non-champion targets at frame level. | totaldamagedone - totaldamagedonetochampions | 1.0.0 | totaldamagedone, totaldamagedonetochampions |
| TLPS_S_002 | synthetic.tl_p_stats.nonChampionPhysicalDamageDone | game_data.tl_participant_stats (see note above) | Physical damage done to non-champion targets at frame level. | physicaldamagedone - physicaldamagedonetochampions | 1.0.0 | physicaldamagedone, physicaldamagedonetochampions |
| TLPS_S_003 | synthetic.tl_p_stats.nonChampionMagicDamageDone | game_data.tl_participant_stats (see note above) | Magic damage done to non-champion targets at frame level. | magicdamagedone - magicdamagedonetochampions | 1.0.0 | magicdamagedone, magicdamagedonetochampions |
| TLPS_S_004 | synthetic.tl_p_stats.nonChampionTrueDamageDone | game_data.tl_participant_stats (see note above) | True damage done to non-champion targets at frame level. | truedamagedone - truedamagedonetochampions | 1.0.0 | truedamagedone, truedamagedonetochampions |
| TLPS_S_005 | synthetic.tl_p_stats.spentGold | game_data.tl_participant_stats (see note above) | Total gold already converted into purchases at frame level. | totalgold - currentgold | 1.0.0 | totalgold, currentgold |
| TLPS_S_008 | synthetic.tl_p_stats.totalFarm | game_data.tl_participant_stats (see note above) | Total lane and jungle farm secured at frame level. | minionskilled + jungleminionskilled | 1.0.0 | minionskilled, jungleminionskilled |
| TLPS_S_010 | synthetic.tl_p_stats.spentGoldRatio | game_data.tl_participant_stats (see note above) | Share of earned gold already converted into purchases at frame level. | (totalgold - currentgold) / greatest(totalgold, 1) | 1.0.0 | totalgold, currentgold |
| TLPS_S_011 | synthetic.tl_p_stats.championDamageShare | game_data.tl_participant_stats (see note above) | Share of total damage that was dealt to champions at frame level. | totaldamagedonetochampions / greatest(totaldamagedone, 1) | 1.0.0 | totaldamagedonetochampions, totaldamagedone |
| TLPS_S_012 | synthetic.tl_p_stats.nonChampionDamageShare | game_data.tl_participant_stats (see note above) | Share of total damage that was dealt to non-champion targets at frame level. | (totaldamagedone - totaldamagedonetochampions) / greatest(totaldamagedone, 1) | 1.0.0 | totaldamagedone, totaldamagedonetochampions |
| TLPS_S_013 | synthetic.tl_p_stats.physicalChampionDamageShare | game_data.tl_participant_stats (see note above) | Share of physical damage that was dealt to champions at frame level. | physicaldamagedonetochampions / greatest(physicaldamagedone, 1) | 1.0.0 | physicaldamagedonetochampions, physicaldamagedone |
| TLPS_S_014 | synthetic.tl_p_stats.physicalNonChampionDamageShare | game_data.tl_participant_stats (see note above) | Share of physical damage that was dealt to non-champion targets at frame level. | (physicaldamagedone - physicaldamagedonetochampions) / greatest(physicaldamagedone, 1) | 1.0.0 | physicaldamagedone, physicaldamagedonetochampions |
| TLPS_S_015 | synthetic.tl_p_stats.magicChampionDamageShare | game_data.tl_participant_stats (see note above) | Share of magic damage that was dealt to champions at frame level. | magicdamagedonetochampions / greatest(magicdamagedone, 1) | 1.0.0 | magicdamagedonetochampions, magicdamagedone |
| TLPS_S_016 | synthetic.tl_p_stats.magicNonChampionDamageShare | game_data.tl_participant_stats (see note above) | Share of magic damage that was dealt to non-champion targets at frame level. | (magicdamagedone - magicdamagedonetochampions) / greatest(magicdamagedone, 1) | 1.0.0 | magicdamagedone, magicdamagedonetochampions |
| TLPS_S_017 | synthetic.tl_p_stats.trueChampionDamageShare | game_data.tl_participant_stats (see note above) | Share of true damage that was dealt to champions at frame level. | truedamagedonetochampions / greatest(truedamagedone, 1) | 1.0.0 | truedamagedonetochampions, truedamagedone |
| TLPS_S_018 | synthetic.tl_p_stats.trueNonChampionDamageShare | game_data.tl_participant_stats (see note above) | Share of true damage that was dealt to non-champion targets at frame level. | (truedamagedone - truedamagedonetochampions) / greatest(truedamagedone, 1) | 1.0.0 | truedamagedone, truedamagedonetochampions |
| TLPS_S_023 | synthetic.tl_p_stats.championDamageToDamageTakenRatio | game_data.tl_participant_stats (see note above) | Ratio of champion damage dealt to total damage taken at frame level. | totaldamagedonetochampions / greatest(totaldamagetaken, 1) | 1.0.0 | totaldamagedonetochampions, totaldamagetaken |
| TLPS_S_024 | synthetic.tl_p_stats.championDamagePerGoldEarned | game_data.tl_participant_stats (see note above) | Champion damage dealt per gold earned at frame level. | totaldamagedonetochampions / greatest(totalgold, 1) | 1.0.0 | totaldamagedonetochampions, totalgold |
| TLPS_S_025 | synthetic.tl_p_stats.physicalDamageShare | game_data.tl_participant_stats (see note above) | Share of total damage dealt that was physical at frame level. | physicaldamagedone / greatest(totaldamagedone, 1) | 1.0.0 | physicaldamagedone, totaldamagedone |
| TLPS_S_026 | synthetic.tl_p_stats.magicDamageShare | game_data.tl_participant_stats (see note above) | Share of total damage dealt that was magic at frame level. | magicdamagedone / greatest(totaldamagedone, 1) | 1.0.0 | magicdamagedone, totaldamagedone |
| TLPS_S_027 | synthetic.tl_p_stats.trueDamageShare | game_data.tl_participant_stats (see note above) | Share of total damage dealt that was true damage at frame level. | truedamagedone / greatest(totaldamagedone, 1) | 1.0.0 | truedamagedone, totaldamagedone |
| TLPS_S_028 | synthetic.tl_p_stats.physicalChampionDamageTypeShare | game_data.tl_participant_stats (see note above) | Share of champion damage dealt that was physical at frame level. | physicaldamagedonetochampions / greatest(totaldamagedonetochampions, 1) | 1.0.0 | physicaldamagedonetochampions, totaldamagedonetochampions |
| TLPS_S_029 | synthetic.tl_p_stats.magicChampionDamageTypeShare | game_data.tl_participant_stats (see note above) | Share of champion damage dealt that was magic at frame level. | magicdamagedonetochampions / greatest(totaldamagedonetochampions, 1) | 1.0.0 | magicdamagedonetochampions, totaldamagedonetochampions |
| TLPS_S_030 | synthetic.tl_p_stats.trueChampionDamageTypeShare | game_data.tl_participant_stats (see note above) | Share of champion damage dealt that was true damage at frame level. | truedamagedonetochampions / greatest(totaldamagedonetochampions, 1) | 1.0.0 | truedamagedonetochampions, totaldamagedonetochampions |
| TLPS_S_031 | synthetic.tl_p_stats.physicalNonChampionDamageTypeShare | game_data.tl_participant_stats (see note above) | Share of non-champion damage dealt that was physical at frame level. | (physicaldamagedone - physicaldamagedonetochampions) / greatest(totaldamagedone - totaldamagedonetochampions, 1) | 1.0.0 | physicaldamagedone, physicaldamagedonetochampions, totaldamagedone, totaldamagedonetochampions |
| TLPS_S_032 | synthetic.tl_p_stats.magicNonChampionDamageTypeShare | game_data.tl_participant_stats (see note above) | Share of non-champion damage dealt that was magic at frame level. | (magicdamagedone - magicdamagedonetochampions) / greatest(totaldamagedone - totaldamagedonetochampions, 1) | 1.0.0 | magicdamagedone, magicdamagedonetochampions, totaldamagedone, totaldamagedonetochampions |
| TLPS_S_033 | synthetic.tl_p_stats.trueNonChampionDamageTypeShare | game_data.tl_participant_stats (see note above) | Share of non-champion damage dealt that was true damage at frame level. | (truedamagedone - truedamagedonetochampions) / greatest(totaldamagedone - totaldamagedonetochampions, 1) | 1.0.0 | truedamagedone, truedamagedonetochampions, totaldamagedone, totaldamagedonetochampions |
| TLPS_S_034 | synthetic.tl_p_stats.physicalDamageTakenShare | game_data.tl_participant_stats (see note above) | Share of total damage taken that was physical at frame level. | physicaldamagetaken / greatest(totaldamagetaken, 1) | 1.0.0 | physicaldamagetaken, totaldamagetaken |
| TLPS_S_035 | synthetic.tl_p_stats.magicDamageTakenShare | game_data.tl_participant_stats (see note above) | Share of total damage taken that was magic at frame level. | magicdamagetaken / greatest(totaldamagetaken, 1) | 1.0.0 | magicdamagetaken, totaldamagetaken |
| TLPS_S_036 | synthetic.tl_p_stats.trueDamageTakenShare | game_data.tl_participant_stats (see note above) | Share of total damage taken that was true damage at frame level. | truedamagetaken / greatest(totaldamagetaken, 1) | 1.0.0 | truedamagetaken, totaldamagetaken |
| TLPS_S_037 | synthetic.tl_p_stats.laneFarmShare | game_data.tl_participant_stats (see note above) | Share of total farm coming from lane minions at frame level. | minionskilled / greatest(minionskilled + jungleminionskilled, 1) | 1.0.0 | minionskilled, jungleminionskilled |
| TLPS_S_038 | synthetic.tl_p_stats.jungleFarmShare | game_data.tl_participant_stats (see note above) | Share of total farm coming from jungle minions at frame level. | jungleminionskilled / greatest(minionskilled + jungleminionskilled, 1) | 1.0.0 | minionskilled, jungleminionskilled |

## Metric List

These are the timeline p_stats metrics. Here, `p_stats` abbreviates `participant_stats`. Raw metrics are listed together; derived metrics are grouped by meaning.

### Raw

- `matchid`: `identity(matchid)`
- `frame_timestamp`: `identity(frame_timestamp)`
- `participantid`: `identity(participantid)`
- `abilityhaste`: `identity(abilityhaste)`
- `abilitypower`: `identity(abilitypower)`
- `armor`: `identity(armor)`
- `attackdamage`: `identity(attackdamage)`
- `attackspeed`: `identity(attackspeed)`
- `ccreduction`: `identity(ccreduction)`
- `cooldownreduction`: `identity(cooldownreduction)`
- `health`: `identity(health)`
- `healthmax`: `identity(healthmax)`
- `healthregen`: `identity(healthregen)`
- `magicresist`: `identity(magicresist)`
- `movementspeed`: `identity(movementspeed)`
- `power`: `identity(power)`
- `powermax`: `identity(powermax)`
- `powerregen`: `identity(powerregen)`
- `payload`: `identity(payload)`
- `currentgold`: `identity(currentgold)`
- `magicdamagedone`: `identity(magicdamagedone)`
- `magicdamagedonetochampions`: `identity(magicdamagedonetochampions)`
- `magicdamagetaken`: `identity(magicdamagetaken)`
- `physicaldamagedone`: `identity(physicaldamagedone)`
- `physicaldamagedonetochampions`: `identity(physicaldamagedonetochampions)`
- `physicaldamagetaken`: `identity(physicaldamagetaken)`
- `totaldamagedone`: `identity(totaldamagedone)`
- `totaldamagedonetochampions`: `identity(totaldamagedonetochampions)`
- `totaldamagetaken`: `identity(totaldamagetaken)`
- `truedamagedone`: `identity(truedamagedone)`
- `truedamagedonetochampions`: `identity(truedamagedonetochampions)`
- `truedamagetaken`: `identity(truedamagetaken)`
- `goldpersecond`: `identity(goldpersecond)`
- `jungleminionskilled`: `identity(jungleminionskilled)`
- `level`: `identity(level)`
- `minionskilled`: `identity(minionskilled)`
- `position_x`: `identity(position_x)`
- `position_y`: `identity(position_y)`
- `timeenemyspentcontrolled`: `identity(timeenemyspentcontrolled)`
- `totalgold`: `identity(totalgold)`
- `xp`: `identity(xp)`

### Derived Non-Champion Damage

- `nonChampionDamageDone`: `totaldamagedone - totaldamagedonetochampions`
- `nonChampionPhysicalDamageDone`: `physicaldamagedone - physicaldamagedonetochampions`
- `nonChampionMagicDamageDone`: `magicdamagedone - magicdamagedonetochampions`
- `nonChampionTrueDamageDone`: `truedamagedone - truedamagedonetochampions`

### Derived Resource State

- `spentGold` (new): `totalgold - currentgold`
- `spentGoldRatio` (new): `(totalgold - currentgold) / greatest(totalgold, 1)`

### Derived Damage Shares

- `championDamageShare` (new): `totaldamagedonetochampions / greatest(totaldamagedone, 1)`
- `nonChampionDamageShare` (new): `(totaldamagedone - totaldamagedonetochampions) / greatest(totaldamagedone, 1)`
- `physicalChampionDamageShare` (new): `physicaldamagedonetochampions / greatest(physicaldamagedone, 1)`
- `physicalNonChampionDamageShare` (new): `(physicaldamagedone - physicaldamagedonetochampions) / greatest(physicaldamagedone, 1)`
- `magicChampionDamageShare` (new): `magicdamagedonetochampions / greatest(magicdamagedone, 1)`
- `magicNonChampionDamageShare` (new): `(magicdamagedone - magicdamagedonetochampions) / greatest(magicdamagedone, 1)`
- `trueChampionDamageShare` (new): `truedamagedonetochampions / greatest(truedamagedone, 1)`
- `trueNonChampionDamageShare` (new): `(truedamagedone - truedamagedonetochampions) / greatest(truedamagedone, 1)`

### Derived Damage Type Shares

- `physicalDamageShare` (new): `physicaldamagedone / greatest(totaldamagedone, 1)`
- `magicDamageShare` (new): `magicdamagedone / greatest(totaldamagedone, 1)`
- `trueDamageShare` (new): `truedamagedone / greatest(totaldamagedone, 1)`

### Derived Champion Damage Type Shares

- `physicalChampionDamageTypeShare` (new): `physicaldamagedonetochampions / greatest(totaldamagedonetochampions, 1)`
- `magicChampionDamageTypeShare` (new): `magicdamagedonetochampions / greatest(totaldamagedonetochampions, 1)`
- `trueChampionDamageTypeShare` (new): `truedamagedonetochampions / greatest(totaldamagedonetochampions, 1)`

### Derived Non-Champion Damage Type Shares

- `physicalNonChampionDamageTypeShare` (new): `(physicaldamagedone - physicaldamagedonetochampions) / greatest(totaldamagedone - totaldamagedonetochampions, 1)`
- `magicNonChampionDamageTypeShare` (new): `(magicdamagedone - magicdamagedonetochampions) / greatest(totaldamagedone - totaldamagedonetochampions, 1)`
- `trueNonChampionDamageTypeShare` (new): `(truedamagedone - truedamagedonetochampions) / greatest(totaldamagedone - totaldamagedonetochampions, 1)`

### Derived Damage Taken Shares

- `physicalDamageTakenShare` (new): `physicaldamagetaken / greatest(totaldamagetaken, 1)`
- `magicDamageTakenShare` (new): `magicdamagetaken / greatest(totaldamagetaken, 1)`
- `trueDamageTakenShare` (new): `truedamagetaken / greatest(totaldamagetaken, 1)`

### Derived Farm

- `totalFarm` (new): `minionskilled + jungleminionskilled`

### Derived Farm Composition

- `laneFarmShare` (new): `minionskilled / greatest(minionskilled + jungleminionskilled, 1)`
- `jungleFarmShare` (new): `jungleminionskilled / greatest(minionskilled + jungleminionskilled, 1)`

### Derived Efficiency

- `championDamageToDamageTakenRatio` (new): `totaldamagedonetochampions / greatest(totaldamagetaken, 1)`
- `championDamagePerGoldEarned` (new): `totaldamagedonetochampions / greatest(totalgold, 1)`

Note: `nonChampion...` is the most precise label available from this timeline schema. It removes champion damage from total damage, but does not isolate monsters from structures or other non-champion targets.

## Metrics Under Consideration

These candidate metrics are intentionally not in the main catalog yet. They stay within same-frame arithmetic only, which matches the current p_stats grain, and they are limited to stats that move meaningfully within the minute-level timeline grain.
