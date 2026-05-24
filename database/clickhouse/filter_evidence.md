# Filter Evidence

Per-rule evidence for the filter pipeline (`schema/4000_filter_schema.sql` + `schema/4000_filter_build.sql`).

## Snapshot

Per-rule percentages use the materialized filter population in `filter_stg_game_flags`: long games (gameduration > 990 s, i.e. > 16.5 min) with corrected participant rows. The f14 pre-stage currently contains 3,015,424 matchids; 2,757,375 have corrected participant rows and enter the participant/game flag stages.

| Metric | Count | % of base |
|---|---:|---:|
| Long games with corrected participants | 2,757,375 | 100% |
| Stage 1 survivors | 2,003,586 | 72.66% |
| Valid games (`any_filter_triggered = 0`) | 1,965,030 | 71.26% |
| Flagged by any rule | 792,345 | 28.74% |
| Baseline participant win rate | 0.500 | — |

`game_data.participant_stats_corrected` was rebuilt before this snapshot and now contains one row per participant key in the materialized filter population: 27,573,750 rows for 27,573,750 distinct `(matchid, teamid, participantid)` keys (2,757,375 long games × 10 participants).

Latest `analytics_builds/8003_filter_statistics.sql` output:

| Filter | Games | Total games | % games |
|---|---:|---:|---:|
| any-filter-triggered | 792,345 | 2,757,375 | 28.74 |
| stage1-survivors | 2,003,586 | 2,757,375 | 72.66 |
| final-survivors | 1,965,030 | 2,757,375 | 71.26 |
| 01-kda-lt-0.20 | 172,145 | 2,757,375 | 6.24 |
| 02-spent-lt-50%-earned-on-loss | 9,186 | 2,757,375 | 0.33 |
| 03-suspect-player-suffix-wr-gte-85% | 839 | 2,757,375 | 0.03 |
| 04-team-kd-ratio-lt-0.50-vs-enemy | 676,952 | 2,757,375 | 24.55 |
| 05-winning-player-kills-gt-75%-team-kills | 2,173 | 2,757,375 | 0.08 |
| 06-non-utility-dmg-share-lt-2% | 1,840 | 2,757,375 | 0.07 |
| 07-non-utility-cs-per-min-lt-3.0 | 12,705 | 2,757,375 | 0.46 |
| 08-team-non-utility-avg-cs-per-min-gt-2.0-below-enemy | 57,353 | 2,757,375 | 2.08 |
| 09-team-non-utility-dmg-to-champs-ratio-lt-1/2-vs-enemy | 25,984 | 2,757,375 | 0.94 |
| 10-low-build-value-lt-1.0 | 38,556 | 2,757,375 | 1.40 |
| 11-unknown-teamposition | 461 | 2,757,375 | 0.02 |

## Methodology

Flags are recorded at participant grain in `filter_stg_participant_flags`; team flags are denormalised to every participant on the affected team. Win rate is computed by joining each flag to `game_data.participant_stats_corrected.win`. `low_build_value` (f10) is participant-grained. `unknown_teamposition` (f11) is recorded from participant rows, rolled up with `max(...)`, and emitted as a game-scope bit so every row from a match with any `UNKNOWN` role is invalidated. `player_high_winrate` (f03) is precomputed in `filter_stg_player_high_winrate_flags` via a suffix-WR trim over each suspect player's collected games (sorted by `gamecreation`) and joined back into participant flags. Short games (≤ 990 s) are excluded via an explicit pre-filter (`filter_stg_f14_long_games`) and never enter any staging table.

| Scope | Expected WR behaviour | Reason |
|---|---|---|
| Player | Far below 50% | One participant flagged; strong individual signal of losing performance. |
| Team | Far below 50% | Team rules typically fire only on the losing team. |
| Game | ~50% by construction | Both teams flagged; wins and losses cancel. Value is removing structurally suspect games, not detecting a losing side. |

---

## Category A — Loss-forcing

Target: WR ≤ 0.30 for player filters; ≤ 0.20 for team filters.

| # | Filter | Scope | Bit | Games flagged | % games | Participants | WR |
|---|---|---|---:|---:|---:|---:|---:|
| 01 | `player_low_kda` KDA < 0.20 | Player | 0 | 172,145 | 6.24 | 194,542 | **0.026** |
| 02 | `player_gold_spent` spent < 50% earned AND loss | Player | 1 | 9,186 | 0.33 | 9,307 | **0.000** |
| 04 | `team_kills_to_deaths` team K/D < 0.50 | Team | 3 | 676,952 | 24.55 | 3,384,760 | **0.000** |
| 06 | `too_little_damage` non-UTILITY dmg share < 2% | Player | 5 | 1,840 | 0.07 | 1,863 | **0.227** |
| 07 | `low_minions_killed` non-UTILITY CS/min < 3.0 | Player | 6 | 12,705 | 0.46 | 12,964 | **0.271** |
| 08 | `team_non_utility_avg_cs_per_min` gap > 2.0 | Team | 7 | 57,353 | 2.08 | 286,765 | **0.001** |
| 09 | `team_non_utility_damage_to_champions_ratio` < 1/2 | Team | 8 | 25,984 | 0.94 | 129,920 | **0.001** |
| 10 | `low_build_value` highest_value < 1.0 | Player | 9 | 38,556 | 1.40 | 40,096 | **0.262** |

All Category A filters meet the WR target.

---

## Category B — Positive-outlier / win-forcing

Identifies structurally imbalanced games. WR > 0.60 expected (inverse direction).

| # | Filter | Scope | Bit | Games flagged | % games | Participants | WR |
|---|---|---|---:|---:|---:|---:|---:|
| 03 | `player_high_winrate` suspect + suffix-WR trim ≥ 85% | Player | 2 | 839 | 0.03 | 839 | **0.931** |
| 05 | `solo_carried` win=1 AND kills > 75% of team kills | Player | 4 | 2,173 | 0.08 | 2,173 | **1.000** |

f03 fires aggressively (WR 0.931 — flagged games are deterministically winning, as intended). f05 is now active after rebuilding `participant_stats_corrected` without duplicate participant keys.

---

## Category C — Data quality / role integrity

Identifies games with incomplete role metadata. Game-level WR is expected to be ~50% by construction; the value is removing rows that cannot be safely used by role-aware build and matchup features.

| # | Filter | Scope | Bit | Games flagged | % games | Participants in flagged games | WR |
|---|---|---|---:|---:|---:|---:|---:|
| 11 | `unknown_teamposition` any participant has `teamposition = 'UNKNOWN'` | Game | 10 | 461 | 0.02 | 4,610 | **0.500** |

There are 475 `UNKNOWN` participant rows inside those 461 games (unknown-row WR 0.274). 105 games are unique to f11 after accounting for the other active filters; 356 overlap an existing rule.

---

## Binning analysis

### F01 — KDA bins (participant grain, long-game population)

Active threshold: **KDA < 0.20** (`(k+a) × 10 < d × 2`).

| KDA range | Participants | WR |
|---|---:|---:|
| 0.00–0.10 | 140,410 | 0.029 |
| 0.10–0.20 | 173,884 | 0.022 |
| 0.20–0.30 (no longer flagged) | 347,068 | 0.031 |
| 0.30–0.40 (no longer flagged) | 390,656 | 0.039 |
| 0.40–0.50 | 436,516 | 0.043 |
| 0.50–1.00 | 4,242,442 | 0.075 |
| ≥1.00 | 40,281,800 | 0.553 |
| deaths=0 | 816,584 | 0.962 |

The 0.20–0.40 band (738k participants, WR 0.031–0.039) is no longer flagged. Both bins are still loss-forcing in isolation but their participants are heavily covered by f04 (team K/D) — keeping them inflates filter aggressiveness without adding unique signal. The 0.00–0.20 flagged band has WR 0.029/0.022 (≈ 314k participants) — strong loss-forcing concentration.

### F02 — gold spent / earned bins (participant grain, losses only)

Active threshold: **spent < 50% earned AND `win = 0`**.

| Ratio range | Losing participants | WR |
|---|---:|---:|
| 0.00–0.10 | 198 | 0.000 |
| 0.10–0.20 | 1,148 | 0.000 |
| 0.20–0.30 | 1,040 | 0.000 |
| 0.30–0.40 | 3,030 | 0.000 |
| 0.40–0.50 | 7,462 | 0.000 |
| 0.50–0.60 (no longer flagged) | 57,256 | 0.000 |
| 0.60–0.70 (no longer flagged) | 148,112 | 0.000 |
| ≥0.70 | 23,196,434 | 0.000 |

WR is 0 in every bin because the subset is `win = 0`. The losses-only restriction collapses the high-ratio noise band that previously sat at WR ≈ 0.89 (winners who never returned to shop). The flagged band (< 0.50 ratio AND loss) is now a pure "loss + low spend" signal — disengagement, early surrender posture, or a death streak preventing shop returns.

### F03 — suspect-player suffix-WR trim (participant grain)

**Active definition.** Two-step rule:

1. *Suspect set* — players with lifetime games > 40 AND lifetime WR > 70%.
2. *Trim* — within each suspect's collected long-games, sort by `gamecreation` ascending, then flag games from the earliest while the **suffix WR** (WR of games from the current row onwards) is ≥ 85%. The first game whose suffix WR falls below 85% stops the trim; that game and all later games are not flagged.

Computed in one pass with window functions over `(puuid ORDER BY gamecreation DESC)`: `suffix_wins = SUM(win) OVER (... ROWS UNBOUNDED PRECEDING)`, `suffix_count = ROW_NUMBER() OVER (...)`, flag if `suffix_wins * 100 >= suffix_count * 85`. Output is staged to `filter_stg_player_high_winrate_flags` and joined into `filter_stg_participant_flags`.

**Result.** 839 games flagged with participant WR 0.931 — flags are concentrated on deterministically winning games, consistent with a smurf / boosted-account regime. Volume is small because the 85 % suffix threshold is strict: most suspect players normalise quickly (within their first few games) and never accumulate a sustained ≥ 85 % WR window.

### F04 — team K/D bins (team grain)

Active threshold: **team K/D < 0.50** (`kills × 2 < deaths`).

Detailed bin counts retained from the prior rebuild (the change set did not alter f04):

| K/D range | Teams | WR |
|---|---:|---:|
| < 0.50 (flagged) | 497,424 | 0.000 |
| 0.50–0.60 | 378,785 | 0.002 |
| 0.60–0.70 | 359,960 | 0.012 |
| 0.70–0.80 | 292,772 | 0.051 |
| 0.80–0.90 | 237,926 | 0.164 |
| 0.90–1.00 | 168,821 | 0.360 |
| 1.00–1.10 | 197,091 | 0.604 |
| ≥ 1.10 | 1,780,233 | 0.965 |

The 0.50–0.60 bin (379k teams, WR 0.002) remains a strong expansion candidate.

### F05 — solo-carried

Active definition: **`win = 1` AND player kills > 75 % of team kills**. After rebuilding `participant_stats_corrected` with unique participant keys, this rule flags 2,173 participants/games with WR 1.0 by construction.

### F06 — non-UTILITY damage share bins (participant grain, non-UTILITY only)

Active threshold: **share < 2 %**. Bin counts retained from the prior rebuild (threshold unchanged):

| Share range | Participants | WR |
|---|---:|---:|
| 0.00–0.01 | 578 | 0.227 |
| 0.01–0.02 | 614 | 0.223 |
| 0.02–0.03 | 1,003 | 0.276 |
| 0.03–0.04 | 1,961 | 0.336 |
| 0.04–0.05 | 4,020 | 0.420 |
| 0.05–0.08 | 62,498 | 0.498 |
| ≥0.08 | 15,289,132 | 0.500 |

### F07 — non-UTILITY CS/min bins (participant grain, non-UTILITY only)

Active threshold: **CS/min < 3.0** (was < 4.0).

| CS/min range | Participants | WR |
|---|---:|---:|
| 0.0–1.0 | 2,666 | 0.272 |
| 1.0–2.0 | 4,958 | 0.292 |
| 2.0–3.0 | 12,992 | 0.288 |
| 3.0–4.0 (no longer flagged) | 116,286 | 0.271 |
| 4.0–5.0 | 1,087,332 | 0.296 |
| ≥5.0 | 36,239,468 | 0.507 |

Tightening from < 4.0 to < 3.0 drops the 3.0–4.0 bin (116k participants, WR 0.271) from the flagged set. That bin still met the ≤ 0.30 target but represents the weakest individual CS-deprivation signal; the remaining < 3.0 band (~21k participants in the long-game pool) concentrates the strongest signal. The broader team-level CS-gap effect is captured at team grain by f08.

### F08 — team CS/min gap vs enemy bins (team grain)

Active threshold: **gap > 2.0** (was > 1.0).

| Gap range | Teams | WR |
|---|---:|---:|
| ≤ 0.0 | 2,345,788 | 0.803 |
| 0.0–0.5 | 1,029,896 | 0.344 |
| 0.5–1.0 | 750,954 | 0.121 |
| 1.0–1.5 (no longer flagged) | 376,277 | 0.031 |
| 1.5–2.0 (no longer flagged) | 133,534 | 0.007 |
| 2.0–2.5 | 36,132 | 0.001 |
| 2.5–3.0 | 8,141 | 0.000 |
| > 3.0 | 1,760 | 0.000 |

Tightening from > 1.0 to > 2.0 removes the 1.0–2.0 band (~510k teams, WR 0.031 → 0.007). Both excluded bins are deeply loss-forcing on their own — the change deliberately gives up volume to retain only the cleanest stomps and reduce overlap with f04 (team K/D).

### F09 — team non-utility damage ratio bins (team grain)

Active threshold: **team/enemy ratio < 1/2** (`team × 2 < enemy`). Bin counts retained from the prior rebuild (threshold unchanged):

| Ratio range | Teams | WR |
|---|---:|---:|
| < 0.50 (flagged) | 19,666 | 0.001 |
| 0.50–1.0 | 1,936,819 | 0.139 |
| 1.0–1.5 | 1,631,692 | 0.836 |
| 1.5–2.0 | 305,161 | 0.994 |
| 2.0–2.5 | 18,671 | 0.999 |
| 2.5–3.0 | 919 | 0.999 |
| 3.0–3.5 | 71 | 1.000 |
| ≥ 3.5 | 13 | 0.846 |

The 0.50–1.0 bin (1.9M teams, WR 0.139) is a large population well below the baseline; the sharp jump at 1.0 suggests the threshold at 0.50 may underfit the signal.

### F10 — highest build value bins (participant grain, stage-1-clean pool)

Active threshold: **`highest_value < 1.0`**. Bin counts retained from the prior rebuild (threshold unchanged):

| Value range | Participants | WR |
|---|---:|---:|
| 0.00–0.25 | 15,015 | 0.228 |
| 0.25–0.50 | 1,363 | 0.229 |
| 0.50–0.75 | 6,123 | 0.265 |
| 0.75–1.00 | 5,787 | 0.307 |
| 1.00–1.25 | 398,477 | 0.393 |
| 1.25–1.50 | 426,529 | 0.393 |
| ≥1.50 | 12,840,256 | 0.507 |

The sharp WR jump from 0.307 → 0.393 at the 1.0 boundary confirms the threshold is well-calibrated.

### F11 — UNKNOWN teamposition (game grain)

Active definition: **any participant has `teamposition = 'UNKNOWN'`**. The rule is computed during the existing stage-1 participant scan as `ps.teamposition = 'UNKNOWN'`, rolled up with `max(unknown_teamposition)`, and written to `game_rule_mask` bit 10 (`1024`) so every participant row in the affected game is invalidated.

| Measure | Count / WR |
|---|---:|
| Games flagged | 461 |
| Share of materialized long-game pool | 0.02% |
| Participants in flagged games | 4,610 |
| Full-game participant WR | 0.500 |
| UNKNOWN participant rows | 475 |
| UNKNOWN-row WR | 0.274 |
| Games unique to this rule | 105 |
| Games overlapping an existing rule | 356 |

This is a schema-quality filter rather than a loss- or win-forcing filter. The full-game WR stays balanced, but role-aware downstream features cannot place an `UNKNOWN` participant into the expected TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY layout.

---

## NN prediction model fitness

The filter pipeline retains 71.26% of materialized long games (1,965,030 valid matches). This section considers how the filtered dataset compares to the full population as a training signal for win-prediction models.

### Why the filtered set is the better training target

**Removes mechanically determined outcomes.** Games flagged by f04 (team K/D < 0.50) have WR 0.000; f08 (CS gap > 2.0) has WR 0.001. An NN trivially learns these cases from raw statistics and inflates reported accuracy without learning competitive patterns. These are not prediction problems — they are bookkeeping.

**Tighter feature distributions.** Stomp games push CS, damage, and kill statistics to extremes. These outliers dominate gradient descent in dense layers, suppressing the subtle covariance signals (itemisation type, role synergy, objective timing) that are actually predictive in competitive play. Filtered games span a narrower, more uniform region of feature space.

**Genuine 50/50 label balance.** The valid pool is balanced by construction — every match has one winner and one loser, and neither team is structurally predetermined to be either. This is the correct inductive bias for a competitive win-prediction task.

**Signal density improves.** On filtered data, features that have low marginal predictive power in stomp games (e.g. vision score, baron control, objective priority) become relatively more informative. The model is forced to rely on these features rather than gross kill differentials.

### Tradeoffs

**Dataset reduction.** Removing ~29% of games shrinks training volume. For model families that benefit from massive datasets (large transformers, contrastive pre-training), this cost is non-trivial. A practical approach is to pre-train on the full population and fine-tune on the filtered set.

**Filter leakage.** The filter operates on end-of-game statistics, which are unavailable at prediction time if the model is used mid-game. If the deployment target is a mid-game model, the filtered population should be further restricted to exclude games whose outcome was already visible from early-game snapshots.

**Including lower-WR games as a separate task.** The flagged population (WR far from 0.50) is a valid training set for a *degraded-game detection* classifier — distinct from win prediction. A two-stage pipeline (detect degraded → predict winner) would use both populations optimally.

### Summary

For a win-prediction model targeting competitive play, the filtered set is strictly superior: it removes noise, sharpens feature gradients, and avoids over-reporting accuracy on trivially classified stomps. With ~1.97M valid games the dataset is large enough for all standard architectures. The full population is better suited to a dedicated anomaly/quality classifier.

---

## Summary

### Filters ranked by games removed

| # | Filter | Games removed |
|---|---|---:|
| 04 | `team_kills_to_deaths` | 676,952 (24.55%) |
| 01 | `player_low_kda` | 172,145 (6.24%) |
| 08 | `team_non_utility_avg_cs_per_min` gap > 2.0 | 57,353 (2.08%) |
| 10 | `low_build_value` | 38,556 (1.40%) |
| 09 | `team_non_utility_damage_to_champions_ratio` < 1/2 | 25,984 (0.94%) |
| 07 | `low_minions_killed` | 12,705 (0.46%) |
| 02 | `player_gold_spent` (losses only) | 9,186 (0.33%) |
| 05 | `solo_carried` | 2,173 (0.08%) |
| 06 | `too_little_damage` | 1,840 (0.07%) |
| 03 | `player_high_winrate` (suffix-WR trim) | 839 (0.03%) |
| 11 | `unknown_teamposition` | 461 (0.02%) |

### Loss-forcing filters vs WR target

| Filter | WR | Meets target? |
|---|---:|---|
| 04 `team_kills_to_deaths` | 0.000 | ✓ (team ≤ 0.20) |
| 02 `player_gold_spent` (losses only) | 0.000 | ✓ by construction |
| 08 `team_non_utility_avg_cs_per_min` gap > 2.0 | 0.001 | ✓ (team ≤ 0.20) |
| 09 `team_non_utility_damage_to_champions_ratio` < 1/2 | 0.001 | ✓ (team ≤ 0.20) |
| 01 `player_low_kda` (< 0.20) | 0.026 | ✓ (player ≤ 0.30) |
| 06 `too_little_damage` | 0.227 | ✓ (player ≤ 0.30) |
| 10 `low_build_value` | 0.262 | ✓ (player ≤ 0.30) |
| 07 `low_minions_killed` (< 3.0) | 0.271 | ✓ (player ≤ 0.30) |

f11 is excluded from the loss-forcing target because it is a game-integrity filter; full-game WR is 0.500 by construction.

---

## Threshold history

| # | Rule | Current threshold | Previous threshold |
|---|---|---|---|
| 01 | `player_low_kda` | KDA < 0.20 (`(k+a)×10 < d×2`) | KDA < 0.30 (`(k+a)×10 < d×3`); before that < 0.20 |
| 02 | `player_gold_spent` | spent < 50% of earned AND `win = 0` | spent < 50% of earned (any outcome) |
| 03 | `player_high_winrate` | suspect set (games > 40 AND lifetime WR > 70%); flag games while suffix WR ≥ 85% trim by `gamecreation ASC` | flat: games > 40 AND lifetime WR > 70%; before that > 50 games AND > 65%; > 40 games AND > 70%; > 30 games AND > 65% |
| 04 | `team_kills_to_deaths` | team K/D < 0.50 (`kills×2 < deaths`) | K/D < 0.45 |
| 05 | `solo_carried` | `win = 1` AND kills > 75% of team kills | kills > 75% (no win gate) |
| 06 | `too_little_damage` | non-UTILITY dmg share < 2% (`dmg×50 < team_dmg`) | not in pipeline |
| 07 | `low_minions_killed` | non-UTILITY CS/min < 3.0 (`(cs+ncs)×60 < time×3`) | CS/min < 4.0; before that not in pipeline |
| 08 | `team_non_utility_avg_cs_per_min_gt_1_0_below_enemy` | gap > 2.0 | gap > 1.0; before that > 1.8; > 2.2; > 2.5 (column name retained from prior threshold) |
| 09 | `team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy` | team/enemy dmg < 1/2 (`team×2 < enemy`) | < 1/5 (3 games); before that < 1/3, < 1/4 |
| 10 | `low_build_value` | `highest_value < 1.0` | (unchanged) |
| 11 | `unknown_teamposition` | any participant has `teamposition = 'UNKNOWN'` | not in pipeline |
| Pre | `game_time_lte_16_5` | explicit pre-stage (`gameduration > 990 s`, i.e. > 16.5 min) | gameduration > 1080 s (18.0 min); before that > 990 s (16.5 min); before that inline WHERE in every stage |

---

## Bitmask reference

`rule_mask` is the match-level union used for `is_valid = (rule_mask = 0)`. `player_rule_mask`, `team_rule_mask`, and `game_rule_mask` retain row-local rule bits for the participant row; `game_rule_mask` is identical for every row in the same match. Bits are sequential and aligned to the f01..f11 numbering.

| Bit | Weight | Column | Scope | # |
|---:|---:|---|---|---|
| 0 | 1 | `player_low_kda` | Player | 01 |
| 1 | 2 | `player_gold_spent` | Player | 02 |
| 2 | 4 | `player_high_winrate` | Player | 03 |
| 3 | 8 | `team_kills_to_deaths` | Team | 04 |
| 4 | 16 | `solo_carried` | Player | 05 |
| 5 | 32 | `too_little_damage` | Player | 06 |
| 6 | 64 | `low_minions_killed` | Player | 07 |
| 7 | 128 | `team_non_utility_avg_cs_per_min_gt_1_0_below_enemy` | Team | 08 |
| 8 | 256 | `team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy` | Team | 09 |
| 9 | 512 | `low_build_value` | Player | 10 |
| 10 | 1024 | `unknown_teamposition` | Game | 11 |

---

## Reproduction queries

### Per-filter game counts

```sql
SELECT
  countIf(player_low_kda)                                                AS f01,
  countIf(player_gold_spent)                                             AS f02,
  countIf(player_high_winrate)                                           AS f03,
  countIf(team_kills_to_deaths)                                          AS f04,
  countIf(solo_carried)                                                  AS f05,
  countIf(too_little_damage)                                             AS f06,
  countIf(low_minions_killed)                                            AS f07,
  countIf(team_non_utility_avg_cs_per_min_gt_1_0_below_enemy)            AS f08,
  countIf(team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy)    AS f09,
  countIf(low_build_value)                                               AS f10,
  countIf(unknown_teamposition)                                          AS f11,
  countIf(any_filter_triggered)                                          AS total_flagged,
  count()                                                                AS total_games
FROM game_data.filter_stg_game_flags;
```

### Per-filter participant win rates

```sql
SELECT
  avgIf(ps.win, fpf.player_low_kda = 1)                                          AS wr_01,
  countIf(fpf.player_low_kda = 1)                                                AS n_01,
  avgIf(ps.win, fpf.player_gold_spent = 1)                                       AS wr_02,
  countIf(fpf.player_gold_spent = 1)                                             AS n_02,
  avgIf(ps.win, fpf.player_high_winrate = 1)                                     AS wr_03,
  countIf(fpf.player_high_winrate = 1)                                           AS n_03,
  avgIf(ps.win, fpf.team_kills_to_deaths = 1)                                    AS wr_04,
  countIf(fpf.team_kills_to_deaths = 1)                                          AS n_04,
  avgIf(ps.win, fpf.solo_carried = 1)                                            AS wr_05,
  countIf(fpf.solo_carried = 1)                                                  AS n_05,
  avgIf(ps.win, fpf.too_little_damage = 1)                                       AS wr_06,
  countIf(fpf.too_little_damage = 1)                                             AS n_06,
  avgIf(ps.win, fpf.low_minions_killed = 1)                                      AS wr_07,
  countIf(fpf.low_minions_killed = 1)                                            AS n_07,
  avgIf(ps.win, fpf.team_non_utility_avg_cs_per_min_gt_1_0_below_enemy = 1)      AS wr_08,
  countIf(fpf.team_non_utility_avg_cs_per_min_gt_1_0_below_enemy = 1)            AS n_08,
  avgIf(ps.win, fpf.team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy = 1) AS wr_09,
  countIf(fpf.team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy = 1)    AS n_09,
  avgIf(ps.win, fpf.unknown_teamposition = 1)                                    AS wr_11,
  countIf(fpf.unknown_teamposition = 1)                                          AS n_11
FROM game_data.filter_stg_participant_flags AS fpf
ANY INNER JOIN game_data.participant_stats_corrected AS ps
USING (matchid, teamid, participantid);
```

### F10 win rate (stage-1-clean pool)

```sql
SELECT count() AS n, avg(ps.win) AS wr
FROM game_data.filter_stg_participant_labels AS pl
ANY INNER JOIN game_data.participant_stats_corrected AS ps
USING (matchid, teamid, participantid)
WHERE pl.low_build_value = 1;
```

### Cumulative removal by category

```sql
SELECT
  countIf(
    player_low_kda OR player_gold_spent OR
    team_kills_to_deaths OR
    too_little_damage OR low_minions_killed OR
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy OR
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy OR
    low_build_value
  ) AS cat_a_loss_forcing,
  countIf(unknown_teamposition) AS data_quality_unknown_teamposition,
  countIf(
    player_low_kda OR player_gold_spent OR
    team_kills_to_deaths OR
    too_little_damage OR low_minions_killed OR
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy OR
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy OR
    low_build_value OR player_high_winrate OR solo_carried OR
    unknown_teamposition
  ) AS all_filters,
  countIf(any_filter_triggered) AS total_flagged,
  count() AS total
FROM game_data.filter_stg_game_flags;
```

---

## Open work

- **f04 further raise.** Team K/D 0.50–0.60 bin (379k teams, WR 0.002) is well-evidenced for raising threshold to 0.60.
- **f09 threshold review.** The damage-ratio jump from 0.001 (< 0.50) to 0.139 (0.50–1.0, 1.9M teams) suggests the < 0.50 cutoff may underfit. Consider raising toward 0.60–0.70 if a measured WR check confirms.
- **f03 calibration.** 839 flagged games is small relative to the suspect set (rebuild `SELECT count() FROM filter_stg_player_winrates WHERE wins+losses > 40 AND wins*100 > (wins+losses)*70`). If most suspects emit zero flagged games, the 85 % suffix threshold may be too strict — consider lowering to 80 %.
