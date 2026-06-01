# Filter Evidence

Per-rule evidence for the filter pipeline (`schema/4000_filter_schema.sql` + `schema/4000_filter_build.sql`).

> Update note (2026-06-01): `schema/4000_filter_build.sql` now applies the
> threshold audit recommendations from `app/ml/experiments/filter_threshold_optimization.py`
> (`f04` team K/D tightened from `<0.50` to `<0.40`, `f10` build value
> tightened from `<1.0` to `<0.5`) and adds `f12 game_ruining_behavior`
> plus exact subfilters `f13 was_severe_transgressor`,
> `f14 caused_game_end_from_ignb_surrender`, and `f15 team_ignb_surrendered`
> from `app/ml/experiments/mentality_filter_review.py`. It also adds targeted integrity
> bits `f16`, `f17`, `f18`, and `f20` for premade transgressor metadata and
> exact loss-gated low-engagement signals. Broader candidates f19, f21, and
> f22 were evaluated and retired from the filtering path after the candidate
> HGNN missed the accuracy/AUC regression gates.

## Snapshot

Per-rule percentages use the materialized filter population in `filter_stg_game_flags`: the latest season in `game_data.info` first, then long games (gameduration > 990 s, i.e. > 16.5 min) with corrected participant rows. With current data this resolves to season 16 only; future rebuilds dynamically use `max(season)`.

Current rebuilt snapshot:

| Metric | Count |
|---|---:|
| Latest season | 16 |
| Latest-season long-game matchids in `filter_stg_f14_long_games` | 1,713,488 |
| Latest-season long games with corrected participants | 1,713,273 |

| Metric | Count | % of base |
|---|---:|---:|
| Latest-season long games with corrected participants | 1,713,273 | 100% |
| Stage 1 survivors | 1,443,210 | 84.24% |
| Valid games (`any_filter_triggered = 0`) | 1,431,313 | 83.54% |
| Flagged by any rule | 281,960 | 16.46% |
| Baseline participant win rate | 0.500 | — |

`game_data.participant_stats_corrected` is rebuilt for the latest-season long-game population before the filter snapshot and contains one row per participant key in that materialized population: 17,132,730 rows for 17,132,730 distinct `(matchid, teamid, participantid)` keys (1,713,273 long games × 10 participants).

Latest `analytics_builds/8003_filter_statistics.sql` output:

| Filter | Games | Total games | % games |
|---|---:|---:|---:|
| any-filter-triggered | 281,960 | 1,713,273 | 16.46 |
| stage1-survivors | 1,443,210 | 1,713,273 | 84.24 |
| final-survivors | 1,431,313 | 1,713,273 | 83.54 |
| 01-kda-lt-0.20 | 108,568 | 1,713,273 | 6.34 |
| 02-spent-lt-50%-earned-on-loss | 4,472 | 1,713,273 | 0.26 |
| 03-suspect-player-suffix-wr-gte-85% | 930 | 1,713,273 | 0.05 |
| 04-team-kd-ratio-lt-0.40-vs-enemy | 187,118 | 1,713,273 | 10.92 |
| 05-winning-player-kills-gt-75%-team-kills | 1,546 | 1,713,273 | 0.09 |
| 06-non-utility-dmg-share-lt-2% | 997 | 1,713,273 | 0.06 |
| 07-non-utility-cs-per-min-lt-3.0 | 5,529 | 1,713,273 | 0.32 |
| 08-team-non-utility-avg-cs-per-min-gt-2.0-below-enemy | 43,333 | 1,713,273 | 2.53 |
| 09-team-non-utility-dmg-to-champs-ratio-lt-1/2-vs-enemy | 16,403 | 1,713,273 | 0.96 |
| 10-low-build-value-lt-0.5 | 11,897 | 1,713,273 | 0.69 |
| 11-unknown-teamposition | 173 | 1,713,273 | 0.01 |
| 12-game-ruining-behavior | 22 | 1,713,273 | 0.00 |
| 13-was-severe-transgressor | 22 | 1,713,273 | 0.00 |
| 14-caused-game-end-from-ignb-surrender | 17 | 1,713,273 | 0.00 |
| 15-team-ignb-surrendered | 17 | 1,713,273 | 0.00 |
| 16-was-premade-with-ignb-game-end-causer | 1 | 1,713,273 | 0.00 |
| 17-was-premade-with-severe-transgressor | 1 | 1,713,273 | 0.00 |
| 18-zero-spell-casts-loss | 36 | 1,713,273 | 0.00 |
| 20-zero-item-purchases-loss | 17 | 1,713,273 | 0.00 |

## Methodology

Flags are recorded at participant grain in `filter_stg_participant_flags`; team flags are denormalised to every participant on the affected team. Win rate is computed by joining each flag to `game_data.participant_stats_corrected.win`. `low_build_value` (f10) is participant-grained. `unknown_teamposition` (f11) is recorded from participant rows, rolled up with `max(...)`, and emitted as a game-scope bit so every row from a match with any `UNKNOWN` role is invalidated. `player_high_winrate` (f03) is precomputed in `filter_stg_player_high_winrate_flags` via a suffix-WR trim over each suspect player's collected games (sorted by `gamecreation`) and joined back into participant flags. Older seasons are excluded first by selecting `max(season)` from `game_data.info`; short games (≤ 990 s) are then excluded via the explicit pre-filter (`filter_stg_f14_long_games`) and never enter any staging table.

| Scope | Expected WR behaviour | Reason |
|---|---|---|
| Player | Far below 50% | One participant flagged; strong individual signal of losing performance. |
| Team | Far below 50% | Team rules typically fire only on the losing team. |
| Game | ~50% by construction | Both teams flagged; wins and losses cancel. Value is removing structurally suspect games, not detecting a losing side. |

---

The detailed binning and model-fitness sections from earlier rebuilds were
removed from this document because their counts no longer match the active
latest-season snapshot. Re-run the reproduction queries below when fresh
per-filter WR or threshold calibration tables are needed.

## Threshold history

| # | Rule | Current threshold | Previous threshold |
|---|---|---|---|
| 01 | `player_low_kda` | KDA < 0.20 (`(k+a)×10 < d×2`) | KDA < 0.30 (`(k+a)×10 < d×3`); before that < 0.20 |
| 02 | `player_gold_spent` | spent < 50% of earned AND `win = 0` | spent < 50% of earned (any outcome) |
| 03 | `player_high_winrate` | suspect set (games > 40 AND lifetime WR > 70%); flag games while suffix WR ≥ 85% trim by `gamecreation ASC` | flat: games > 40 AND lifetime WR > 70%; before that > 50 games AND > 65%; > 40 games AND > 70%; > 30 games AND > 65% |
| 04 | `team_kills_to_deaths` | team K/D < 0.40 (`kills×5 < deaths×2`) | K/D < 0.50; before that < 0.45 |
| 05 | `solo_carried` | `win = 1` AND kills > 75% of team kills | kills > 75% (no win gate) |
| 06 | `too_little_damage` | non-UTILITY dmg share < 2% (`dmg×50 < team_dmg`) | not in pipeline |
| 07 | `low_minions_killed` | non-UTILITY CS/min < 3.0 (`(cs+ncs)×60 < time×3`) | CS/min < 4.0; before that not in pipeline |
| 08 | `team_non_utility_avg_cs_per_min_gt_1_0_below_enemy` | gap > 2.0 | gap > 1.0; before that > 1.8; > 2.2; > 2.5 (column name retained from prior threshold) |
| 09 | `team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy` | team/enemy dmg < 1/2 (`team×2 < enemy`) | < 1/5 (3 games); before that < 1/3, < 1/4 |
| 10 | `low_build_value` | `highest_value < 0.5` | `highest_value < 1.0` |
| 11 | `unknown_teamposition` | any participant has `teamposition = 'UNKNOWN'` | not in pipeline |
| 12 | `game_ruining_behavior` | any IGNB surrender / severe-transgressor participant metadata | not in pipeline |
| 13 | `was_severe_transgressor` | exact `wasseveretransgressor` participant metadata | not in pipeline |
| 14 | `caused_game_end_from_ignb_surrender` | exact `causedgameendfromignbsurrender` participant metadata | not in pipeline |
| 15 | `team_ignb_surrendered` | exact `teamignbsurrendered` team-side metadata | not in pipeline |
| 16 | `was_premade_with_ignb_game_end_causer` | exact `waspremadewithignbgameendcauser` participant metadata | not in pipeline |
| 17 | `was_premade_with_severe_transgressor` | exact `waspremadewithseveretransgressor` participant metadata | not in pipeline |
| 18 | `zero_spell_casts_loss` | losing participant with 0 champion spell casts | not in pipeline |
| 20 | `zero_item_purchases_loss` | losing participant with 0 item purchases | not in pipeline |
| Pre | `latest_season` | explicit pre-stage (`season = max(season)` from `game_data.info`; currently season 16) | not in pipeline |
| Pre | `game_time_lte_16_5` | explicit pre-stage after latest-season selection (`gameduration > 990 s`, i.e. > 16.5 min) | gameduration > 1080 s (18.0 min); before that > 990 s (16.5 min); before that inline WHERE in every stage |

---

## Bitmask reference

`rule_mask` is the match-level union used for `is_valid = (rule_mask = 0)`. `player_rule_mask`, `team_rule_mask`, and `game_rule_mask` retain row-local rule bits for the participant row; `game_rule_mask` is identical for every row in the same match. Bits align to the hard-filter numbering, with gaps reserved for retired candidate rules.

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
| 11 | 2048 | `game_ruining_behavior` | Game | 12 |
| 12 | 4096 | `was_severe_transgressor` | Game | 13 |
| 13 | 8192 | `caused_game_end_from_ignb_surrender` | Game | 14 |
| 14 | 16384 | `team_ignb_surrendered` | Game | 15 |
| 15 | 32768 | `was_premade_with_ignb_game_end_causer` | Game | 16 |
| 16 | 65536 | `was_premade_with_severe_transgressor` | Game | 17 |
| 17 | 131072 | `zero_spell_casts_loss` | Player | 18 |
| 19 | 524288 | `zero_item_purchases_loss` | Player | 20 |

Retired candidate bits are intentionally not materialized: f19 (`low_spell_casts_loss`), f21 (`low_item_purchases_loss`), and f22 (`no_vision_no_wards_loss`) were removed from staging tables and reports after model-gated review. f20 keeps its original bit weight (`524288`) so prior mask interpretation does not shift.

## Retired candidate evidence

Lean evidence retained from the 2026-06-01 candidate-filter audit:

| Retired rule | Prior candidate condition | Prior count | Conclusion |
|---|---|---:|---|
| f19 `low_spell_casts_loss` | losing participant with ≤20 champion spell casts | 527 (0.03%) | Too broad relative to exact-zero f18; more champion-style sensitive and did not improve model-gated behavior. |
| f21 `low_item_purchases_loss` | losing participant with ≤2 item purchases | 124 (0.01%) | Mostly a broader proxy around exact-zero f20; weak incremental value and game-state sensitive. |
| f22 `no_vision_no_wards_loss` | losing participant with 0 vision score, wards placed, and wards killed | 976 (0.06%) | Coverage-safe in isolation but role/champion sensitive; candidate HGNN still missed accuracy/AUC gates. |

Candidate HGNN comparison versus a fresh control held promotion: test accuracy `0.5720 → 0.5674` (Δ -0.0046, gate -0.0020) and test AUC `0.5980 → 0.5928` (Δ -0.0052, gate -0.0025). f16, f17, f18, and f20 remain the lean hard candidate set.

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
  countIf(game_ruining_behavior)                                         AS f12,
  countIf(was_severe_transgressor)                                       AS f13,
  countIf(caused_game_end_from_ignb_surrender)                           AS f14,
  countIf(team_ignb_surrendered)                                         AS f15,
  countIf(was_premade_with_ignb_game_end_causer)                         AS f16,
  countIf(was_premade_with_severe_transgressor)                           AS f17,
  countIf(zero_spell_casts_loss)                                         AS f18,
  countIf(zero_item_purchases_loss)                                      AS f20,
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
  countIf(fpf.unknown_teamposition = 1)                                          AS n_11,
  avgIf(ps.win, fpf.game_ruining_behavior = 1)                                   AS wr_12,
  countIf(fpf.game_ruining_behavior = 1)                                         AS n_12,
  avgIf(ps.win, fpf.was_severe_transgressor = 1)                                 AS wr_13,
  countIf(fpf.was_severe_transgressor = 1)                                       AS n_13,
  avgIf(ps.win, fpf.caused_game_end_from_ignb_surrender = 1)                     AS wr_14,
  countIf(fpf.caused_game_end_from_ignb_surrender = 1)                           AS n_14,
  avgIf(ps.win, fpf.team_ignb_surrendered = 1)                                   AS wr_15,
  countIf(fpf.team_ignb_surrendered = 1)                                         AS n_15,
  avgIf(ps.win, fpf.was_premade_with_ignb_game_end_causer = 1)                   AS wr_16,
  countIf(fpf.was_premade_with_ignb_game_end_causer = 1)                         AS n_16,
  avgIf(ps.win, fpf.was_premade_with_severe_transgressor = 1)                    AS wr_17,
  countIf(fpf.was_premade_with_severe_transgressor = 1)                          AS n_17,
  avgIf(ps.win, fpf.zero_spell_casts_loss = 1)                                   AS wr_18,
  countIf(fpf.zero_spell_casts_loss = 1)                                         AS n_18,
  avgIf(ps.win, fpf.zero_item_purchases_loss = 1)                                AS wr_20,
  countIf(fpf.zero_item_purchases_loss = 1)                                      AS n_20
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
  countIf(game_ruining_behavior) AS data_quality_game_ruining_behavior,
  countIf(was_severe_transgressor) AS data_quality_was_severe_transgressor,
  countIf(caused_game_end_from_ignb_surrender) AS data_quality_caused_ignb_surrender,
  countIf(team_ignb_surrendered) AS data_quality_team_ignb_surrendered,
  countIf(was_premade_with_ignb_game_end_causer) AS data_quality_premade_ignb_causer,
  countIf(was_premade_with_severe_transgressor) AS data_quality_premade_severe,
  countIf(zero_spell_casts_loss OR zero_item_purchases_loss) AS exact_low_engagement_loss,
  countIf(
    player_low_kda OR player_gold_spent OR
    team_kills_to_deaths OR
    too_little_damage OR low_minions_killed OR
    team_non_utility_avg_cs_per_min_gt_1_0_below_enemy OR
    team_non_utility_damage_to_champions_ratio_lt_1_2_vs_enemy OR
    low_build_value OR player_high_winrate OR solo_carried OR
    unknown_teamposition OR game_ruining_behavior OR
    was_severe_transgressor OR caused_game_end_from_ignb_surrender OR
    team_ignb_surrendered OR was_premade_with_ignb_game_end_causer OR
    was_premade_with_severe_transgressor OR zero_spell_casts_loss OR
    zero_item_purchases_loss
  ) AS all_filters,
  countIf(any_filter_triggered) AS total_flagged,
  count() AS total
FROM game_data.filter_stg_game_flags;
```

---

## Open work

- **Candidate model regression.** `app/ml/experiments/candidate_filter_model_review.py` held promotion: the f16/f17/f18/f20-hard candidate missed the test accuracy/AUC gates; broader f19/f21/f22 candidates are retired from filtering and retained only as lean evidence above.
- **Keep f08/f09 expansion as diagnostics only.** `app/ml/experiments/filter_threshold_optimization.py` found broader thresholds can pass the data gate, but `app/ml/experiments/filter_audit_model_review.py` did not support promoting those watch bands on held-out model behavior.
- **f03 calibration.** 930 flagged games is small relative to the suspect set (rebuild `SELECT count() FROM filter_stg_player_winrates WHERE wins+losses > 40 AND wins*100 > (wins+losses)*70`). If most suspects emit zero flagged games, the 85 % suffix threshold may be too strict — consider lowering to 80 % as a watch flag before any exclusion.
- **Monitor Riot game-ruining metadata and exact low-engagement losses.** `app/ml/experiments/mentality_filter_review.py` shows the current Riot-metadata population is tiny but high precision; if these fields become more populated, re-check false-positive and champion-coverage impact before expanding beyond the exact reason bits.
