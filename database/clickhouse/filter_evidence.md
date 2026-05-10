# Filter Evidence

Per-rule evidence for the filter pipeline (`schema/4000_filter_schema.sql` + `schema/4000_filter_build.sql`).

## Snapshot

Base population is `filter_stg_f14_long_games` (gameduration > 1080 s).

| Metric | Count | % of base |
|---|---:|---:|
| Long games (f14 base) | 1,974,712 | 100% |
| Stage 1 survivors | 1,047,726 | 53.07% |
| Valid games (`any_filter_triggered = 0`) | 1,022,838 | 51.81% |
| Flagged by any rule | 951,643 | 48.19% |
| Baseline participant win rate | 0.500 | — |

*Snapshot predates f04 threshold tightening, f14 threshold change to 1080 s, and f15/f16 removal; rebuild pending.*

## Methodology

Flags are recorded at participant grain in `filter_stg_participant_flags`; team flags are denormalised to every participant on the affected team. Win rate is computed by joining each flag to `game_data.participant_stats.win`. `low_build_value` is participant-grained. Short games (≤ 1080 s) are excluded via an explicit pre-filter (`filter_stg_f14_long_games`) and never enter any staging table.

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
| 01 | `player_low_kda` KDA < 0.30 | Player | 0 | 241,170 | 12.21 | 289,941 | **0.024** |
| 02 | `player_gold_spent` spent < 50% earned | Player | 1 | 2,702 | 0.14 | — | **0.227** |
| 03 | `kill_participation_low` non-UTILITY (k+a)/team_kills < 10% | Player | 2 | 143,259 | 7.25 | 152,603 | **0.172** |
| 05 | `team_kills_to_deaths` team K/D < 0.50 | Team | 5 | 509,550 | 25.81 | 2,547,750 | **0.000** |
| 07 | `too_little_damage` non-UTILITY dmg share < 2% | Player | 7 | 1,229 | 0.06 | 1,244 | **0.222** |
| 08 | `low_minions_killed` non-UTILITY CS/min < 4.0 | Player | 8 | 54,928 | 2.78 | 57,112 | **0.269** |
| 09 | `team_non_utility_avg_cs_per_min_gt_1_0_below_enemy` | Team | 9 | 475,350 | 24.09 | 2,376,750 | **0.022** |
| 10 | `team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy` | Team | 10 | 20,422 | 1.03 | 102,110 | **0.001** |
| 11 | `low_build_value` highest_value < 1.0 | Player | 11 | 17,234 | 0.87 | 17,809 | **0.267** |

All enabled loss-forcing filters meet the ≤ 0.30 target.

---

## Category B — Positive-outlier / win-forcing

Identifies structurally imbalanced games. WR > 0.60 expected (inverse direction).

| # | Filter | Scope | Bit | Games flagged | % games | Participants | WR |
|---|---|---|---:|---:|---:|---:|---:|
| 04 | `player_high_winrate` games > 40 AND WR > 70% | Player | 4 | ~29,100 | ~1.47 | ~31,132 | **0.656** |
| 06 | `solo_carried` win=1 AND kills > 75% of team kills | Player | 6 | 1,494 | 0.08 | 1,494 | **1.000** |

F06 WR is 1.0 by construction (flag requires `win = 1`). F04 stats approximate from cross-tab; rebuild pending.

---

## Binning analysis

### F01 — KDA bins (participant grain)

| KDA range | Participants | WR |
|---|---:|---:|
| 0.00–0.10 | 56,945 | 0.025 |
| 0.10–0.20 | 72,852 | 0.019 |
| 0.20–0.30 | 145,145 | 0.026 |
| 0.30–0.40 | 164,411 | 0.033 |
| 0.40–0.50 | 185,059 | 0.038 |
| 0.50–0.60 | 360,995 | 0.047 |
| 0.60–0.70 | 387,803 | 0.058 |
| 0.70–0.80 | 384,086 | 0.067 |
| 0.80–0.90 | 540,854 | 0.077 |
| 0.90–1.00 | 123,205 | 0.103 |
| ≥1.00 | 16,812,146 | 0.554 |
| deaths=0 | 333,829 | 0.969 |

WR stays below 0.04 for all bins up to KDA 0.50. The 0.30–0.40 bin (165k, WR 0.033) remains a candidate for a further raise.

### F02 — gold spent / earned bins (participant grain)

| Ratio range | Participants | WR |
|---|---:|---:|
| 0.00–0.10 | 125 | 0.416 |
| 0.10–0.20 | 519 | 0.139 |
| 0.20–0.30 | 394 | 0.228 |
| 0.30–0.40 | 649 | 0.176 |
| 0.40–0.50 | 987 | 0.282 |
| 0.50–0.60 | 4,067 | 0.681 |
| 0.60–0.70 | 51,120 | 0.890 |
| ≥0.70 | 19,141,809 | 0.499 |

Above 0.50 WR jumps sharply (winning players who never returned to shop). The 0.50 cutoff is well-placed.

### F03 — kill participation bins (participant grain, non-UTILITY only)

| KP range | Participants | WR |
|---|---:|---:|
| 0.00 (zero k+a, team has kills) | 56,914 | 0.032 |
| 0.01–0.04 (k+a > 0, < 5%) | 6,102 | 0.493 |
| 0.05–0.09 | 84,965 | 0.216 |
| 0.10–0.14 | 208,760 | 0.272 |
| 0.15–0.19 | 346,430 | 0.379 |
| ≥0.20 | 18,561,499 | 0.508 |
| team-no-kills | 360 | 0.028 |

**Why 0.01–0.04 shows WR 0.493:** Zero-KP players (0.032 WR) are almost certainly disconnected — the team lost a 4v5. Players with 1–2 k+a but < 5% share are different: they contributed *something* and exist equally on winning and losing sides. The effect is strongest in stomp games — when your team accumulates 30+ kills, even 1 assist leaves you below 5% KP, but your team wins the stomp. These players are minimally impactful but not absent, so their WR tracks the team result (~50%). The zero-KP bin is the AFK signal; the 1–4% bin is not. The filter threshold at 10% (`(k+a)*10 < team_kills`) includes this ambiguous bin, but its small volume (6k vs 148k total in the flagged band) does not materially distort the combined WR of 0.172.

### F04 — player win-rate cross-tab (participant grain, stage-1 population)

Active threshold: **games > 40 AND WR > 70%** (≈ 31,132 participants, WR 0.656).

| Min games | WR > 60% | WR > 65% | WR > 70% | WR > 75% | WR > 80% |
|---:|---:|---:|---:|---:|---:|
| > 20 | 437,678 / 0.593 | 114,991 / 0.627 | 31,132 / 0.656 | 7,483 / 0.687 | 2,841 / 0.689 |
| > 50 | 437,678 / 0.593 | 114,991 / 0.627 | 31,132 / 0.656 | 7,483 / 0.687 | 2,841 / 0.689 |
| > 60 | 436,732 / 0.593 | 114,322 / 0.627 | 30,463 / 0.656 | 7,033 / 0.691 | 2,566 / 0.695 |

Game-count threshold has virtually no effect on the stage-1 population. WR threshold is the only meaningful discriminator. Raising from 65% to 70% removes 73% of the flagged population with only a 3-point WR gain (0.627 → 0.656).

### F05 — team K/D bins (team grain, expansion zone)

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

The 0.50–0.60 bin (372k teams, WR 0.002) is a strong expansion candidate; raising the threshold to 0.60 would add this volume well within the ≤ 0.20 team target.

### F07 — non-UTILITY damage share bins (participant grain, non-UTILITY only)

| Share range | Participants | WR |
|---|---:|---:|
| 0.00–0.01 | 578 | 0.227 |
| 0.01–0.02 | 614 | 0.223 |
| 0.02–0.03 | 1,003 | 0.276 |
| 0.03–0.04 | 1,961 | 0.336 |
| 0.04–0.05 | 4,020 | 0.420 |
| 0.05–0.08 | 62,498 | 0.498 |
| ≥0.08 | 15,289,132 | 0.500 |

### F08 — non-UTILITY CS/min bins (participant grain, non-UTILITY only)

| CS/min range | Participants | WR |
|---|---:|---:|
| 0.0–0.5 | 450 | 0.244 |
| 0.5–1.0 | 645 | 0.284 |
| 1.0–1.5 | 854 | 0.264 |
| 1.5–2.0 | 1,187 | 0.290 |
| 2.0–2.5 | 1,777 | 0.283 |
| 2.5–3.0 | 3,703 | 0.277 |
| 3.0–4.0 | 46,873 | 0.267 |
| ≥4.0 | 15,304,333 | 0.501 |

### F09 — team CS/min gap vs enemy bins (team grain)

| Gap range | Teams | WR |
|---|---:|---:|
| 0.0–0.5 | 846,296 | 0.345 |
| 0.5–1.0 | 614,687 | 0.120 |
| 1.0–1.5 | 310,842 | 0.030 |
| 1.5–2.0 | 111,935 | 0.006 |
| 2.0–2.5 | 30,894 | 0.001 |
| 2.5–3.0 | 7,303 | 0.001 |
| ≥3.0 | 1,763 | 0.000 |

The 0.5–1.0 gap bin (615k teams, WR 0.120) is the largest remaining expansion opportunity.

### F10 — team non-utility damage ratio bins (team grain, expansion zone)

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

The 0.50–1.0 bin (1.9M teams, WR 0.139) represents a large population well below the baseline; the sharp jump at 1.0 suggests the threshold at 0.50 underfits the signal.

### F11 — highest build value bins (participant grain, stage-1-clean pool)

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

### F15 — rare role pick-rate (removed; reference only)

| Rate bin | Participants | WR |
|---|---:|---:|
| < 0.1% (flagged) | 20,740 | 0.450 |
| ≥ 0.1% | majority | ~0.500 |

Removed: WR 0.450 is insufficiently different from the 0.500 baseline; flagged volume is 0.21% of games. Weak signal, not loss-forcing.

### F16 — rare build label count (removed; reference only)

| Count bin | Participants | WR |
|---|---:|---:|
| < 6 (flagged) | 4,166 | 0.420 |
| ≥ 6 | majority | ~0.500 |

Removed: WR 0.420 provides weak signal relative to Category A filters; flagged volume is 0.21% of games.

---

## NN prediction model fitness

The filter pipeline retains ~52% of long games (1,022,838 valid matches). This section considers how the filtered dataset compares to the full population as a training signal for win-prediction models.

### Why the filtered set is the better training target

**Removes mechanically determined outcomes.** Games flagged by f05 (team K/D < 0.50) have WR 0.000; f09 (CS gap > 1.0) has WR 0.022. An NN trivially learns these cases from raw statistics and inflates reported accuracy without learning competitive patterns. These are not prediction problems — they are bookkeeping.

**Tighter feature distributions.** Stomp games push CS, damage, and kill statistics to extremes. These outliers dominate gradient descent in dense layers, suppressing the subtle covariance signals (itemisation type, role synergy, objective timing) that are actually predictive in competitive play. Filtered games span a narrower, more uniform region of feature space.

**Genuine 50/50 label balance.** The valid pool is balanced by construction — every match has one winner and one loser, and neither team is structurally predetermined to be either. This is the correct inductive bias for a competitive win-prediction task.

**Signal density improves.** On filtered data, features that have low marginal predictive power in stomp games (e.g. vision score, baron control, objective priority) become relatively more informative. The model is forced to rely on these features rather than gross kill differentials.

### Tradeoffs

**Dataset reduction.** Removing 48% of games shrinks training volume. For model families that benefit from massive datasets (large transformers, contrastive pre-training), this cost is non-trivial. A practical approach is to pre-train on the full population and fine-tune on the filtered set.

**Filter leakage.** The filter operates on end-of-game statistics, which are unavailable at prediction time if the model is used mid-game. If the deployment target is a mid-game model, the filtered population should be further restricted to exclude games whose outcome was already visible from early-game snapshots.

**Including lower-WR games as a separate task.** The flagged population (WR far from 0.50) is a valid training set for a *degraded-game detection* classifier — distinct from win prediction. A two-stage pipeline (detect degraded → predict winner) would use both populations optimally.

### Summary

For a win-prediction model targeting competitive play, the filtered set is strictly superior: it removes noise, sharpens feature gradients, and avoids over-reporting accuracy on trivially classified stomps. With ~1M valid games the dataset is large enough for all standard architectures. The full population is better suited to a dedicated anomaly/quality classifier.

---

## Summary

### Filters ranked by games removed

| # | Filter | Games removed |
|---|---|---:|
| 05 | `team_kills_to_deaths` | 509,550 (25.81%) |
| 09 | `team_non_utility_avg_cs_per_min_gt_1_0` | 475,350 (24.09%) |
| 01 | `player_low_kda` | 241,170 (12.21%) |
| 03 | `kill_participation_low` | 143,259 (7.25%) |
| 04 | `player_high_winrate` | ~29,100 (~1.47%) |
| 08 | `low_minions_killed` | 54,928 (2.78%) |
| 10 | `team_dmg_ratio` | 20,422 (1.03%) |

### Loss-forcing filters vs WR target

| Filter | WR | Meets ≤ 0.30? |
|---|---:|---|
| 05 `team_kills_to_deaths` | 0.000 | ✓ |
| 10 `team_dmg_ratio_lt_1/2` | 0.001 | ✓ |
| 09 `team_non_utility_avg_cs_per_min_gt_1_0` | 0.022 | ✓ |
| 01 `player_low_kda` | 0.024 | ✓ |
| 03 `kill_participation_low` | 0.172 | ✓ |
| 07 `too_little_damage` | 0.222 | ✓ |
| 02 `player_gold_spent` | 0.227 | ✓ borderline |
| 11 `low_build_value` | 0.267 | ✓ |
| 08 `low_minions_killed` | 0.269 | ✓ |

---

## Overlap analysis

### F05 / F09 (team K/D vs CS/min gap)

| Bucket | Games |
|---|---:|
| f05 only | 304,387 |
| f09 only | 270,187 |
| both f05 + f09 | 205,163 |
| either f05 or f09 | 779,737 |

40.3% of f05-flagged games also trigger f09; 43.2% of f09-flagged games also trigger f05. ~58% of each filter's flagged population is exclusive to that rule — both filters carry substantial unique coverage.

---

## Threshold history

| # | Rule | Current threshold | Previous threshold |
|---|---|---|---|
| 01 | `player_low_kda` | KDA < 0.30 (`(k+a)×10 < d×3`) | KDA < 0.20 |
| 02 | `player_gold_spent` | spent < 50% of earned | (unchanged) |
| 03 | `kill_participation_low` | non-UTILITY (k+a)/team_kills < 10% | < 5% all positions; before that removed |
| 04 | `player_high_winrate` | games > 40 AND WR > 70% | games > 50 AND WR > 65%; before that > 40 games AND > 70%; before that > 30 games AND > 65% |
| 05 | `team_kills_to_deaths` | team K/D < 0.50 (`kills×2 < deaths`) | K/D < 0.45 |
| 06 | `solo_carried` | `win = 1` AND kills > 75% of team kills | kills > 75% (no win gate) |
| 07 | `too_little_damage` | non-UTILITY dmg share < 2% (`dmg×50 < team_dmg`) | not in pipeline |
| 08 | `low_minions_killed` | non-UTILITY CS/min < 4.0 (`(cs+ncs)×60 < time×4`) | not in pipeline |
| 09 | `team_non_utility_avg_cs_per_min_gt_1_0_below_enemy` | gap > 1.0 | gap > 1.8; before that > 2.2; > 2.5 |
| 10 | `team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy` | team/enemy dmg < 1/2 (`team×2 < enemy`) | < 1/5 (3 games); before that < 1/3, < 1/4 |
| 11 | `low_build_value` | `highest_value < 1.0` | (unchanged) |
| Pre | `game_time_lte_18_0` | explicit pre-stage (`gameduration > 1080 s`) | gameduration > 990 s (16.5 min); before that inline WHERE in every stage |

---

## Bitmask reference

`player_rule_mask + team_rule_mask + game_rule_mask = rule_mask`. `is_valid = (rule_mask = 0)`.

| Bit | Weight | Column | Scope | # |
|---:|---:|---|---|---|
| 0 | 1 | `player_low_kda` | Player | 01 |
| 1 | 2 | `player_gold_spent` | Player | 02 |
| 2 | 4 | `kill_participation_low` | Player | 03 |
| 4 | 16 | `player_high_winrate` | Player | 04 |
| 5 | 32 | `team_kills_to_deaths` | Team | 05 |
| 6 | 64 | `solo_carried` | Player | 06 |
| 7 | 128 | `too_little_damage` | Player | 07 |
| 8 | 256 | `low_minions_killed` | Player | 08 |
| 9 | 512 | `team_non_utility_avg_cs_per_min_gt_1_0_below_enemy` | Team | 09 |
| 10 | 1024 | `team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy` | Team | 10 |
| 11 | 2048 | `low_build_value` | Player | 11 |
---

## Reproduction queries

### Per-filter game counts

```sql
SELECT
  countIf(player_low_kda)                                                        AS f01,
  countIf(player_gold_spent)                                                     AS f02,
  countIf(kill_participation_low)                                                AS f03,
  countIf(player_high_winrate)                                                   AS f04,
  countIf(team_kills_to_deaths)                                                  AS f05,
  countIf(solo_carried)                                                          AS f06,
  countIf(too_little_damage)                                                     AS f07,
  countIf(low_minions_killed)                                                    AS f08,
  countIf(team_non_utility_avg_cs_per_min_gt_1_0_below_enemy)                   AS f09,
  countIf(team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy)           AS f10,
  countIf(low_build_value)                                                       AS f11,
  countIf(any_filter_triggered)                                                  AS total_flagged,
  count()                                                                        AS total_games
FROM game_data.filter_stg_game_flags;
```

### Per-filter participant win rates (stage 1)

```sql
SELECT
  avgIf(ps.win, fpf.player_low_kda = 1)                                                            AS wr_01,
  countIf(fpf.player_low_kda = 1)                                                                  AS n_01,
  avgIf(ps.win, fpf.player_gold_spent = 1)                                                         AS wr_02,
  countIf(fpf.player_gold_spent = 1)                                                               AS n_02,
  avgIf(ps.win, fpf.kill_participation_low = 1)                                                    AS wr_03,
  countIf(fpf.kill_participation_low = 1)                                                          AS n_03,
  avgIf(ps.win, fpf.player_high_winrate = 1)                                                       AS wr_04,
  countIf(fpf.player_high_winrate = 1)                                                             AS n_04,
  avgIf(ps.win, fpf.team_kills_to_deaths = 1)                                                      AS wr_05,
  countIf(fpf.team_kills_to_deaths = 1)                                                            AS n_05,
  avgIf(ps.win, fpf.solo_carried = 1)                                                              AS wr_06,
  countIf(fpf.solo_carried = 1)                                                                    AS n_06,
  avgIf(ps.win, fpf.too_little_damage = 1)                                                         AS wr_07,
  countIf(fpf.too_little_damage = 1)                                                               AS n_07,
  avgIf(ps.win, fpf.low_minions_killed = 1)                                                        AS wr_08,
  countIf(fpf.low_minions_killed = 1)                                                              AS n_08,
  avgIf(ps.win, fpf.team_non_utility_avg_cs_per_min_gt_1_0_below_enemy = 1)                       AS wr_09,
  countIf(fpf.team_non_utility_avg_cs_per_min_gt_1_0_below_enemy = 1)                             AS n_09,
  avgIf(ps.win, fpf.team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy = 1)               AS wr_10,
  countIf(fpf.team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy = 1)                     AS n_10
FROM game_data.filter_stg_participant_flags AS fpf
ANY INNER JOIN game_data.participant_stats AS ps
USING (matchid, teamid, participantid);
```

### F11 win rate (stage-1-clean pool)

```sql
SELECT count() AS n, avg(ps.win) AS wr
FROM game_data.filter_stg_participant_labels AS pl
ANY INNER JOIN game_data.participant_stats AS ps
USING (matchid, teamid, participantid)
WHERE pl.low_build_value = 1;
```

### Cumulative removal by category

```sql
SELECT
  countIf(
    player_low_kda OR player_gold_spent OR kill_participation_low OR
    team_kills_to_deaths OR
    too_little_damage OR low_minions_killed OR
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy OR
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy OR
    low_build_value
  ) AS cat_a_loss_forcing,
  countIf(
    player_low_kda OR player_gold_spent OR kill_participation_low OR
    team_kills_to_deaths OR
    too_little_damage OR low_minions_killed OR
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy OR
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy OR
    low_build_value OR player_high_winrate OR solo_carried
  ) AS cat_ab_all,
  countIf(any_filter_triggered) AS total_flagged,
  count() AS total
FROM game_data.filter_stg_game_flags;
```

---

## Open work

- **f03 raise** — 0.10–0.14 bin (209k participants, WR 0.272) is a strong expansion candidate; raising to 15% KP adds significant volume within the ≤ 0.30 target.
- **f01 raise** — 0.30–0.40 bin (165k, WR 0.033) has equivalent signal quality to the current filtered range.
- **f05 further raise** — 0.50–0.60 bin (372k teams, WR 0.002) well-evidenced for raising threshold to 0.60.
- **f09 lower** — 0.5–1.0 gap bin (615k teams, WR 0.120) is the largest remaining expansion; lowering from 1.0 to 0.5 would add massive volume below the 0.20 team target.
