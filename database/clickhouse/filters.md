# Filtering Architecture (Single MV + Bitmask)

## Objective

Use one materialized view to compute game validity directly from `participant_stats`, minimizing intermediate tables, storage, and cross-object complexity.

## Data Flow

```text
game_data.participant_stats
  -> game_data.mv_filter_game_validity
  -> game_data.filter_game_validity
```

## Schema Files

- `database/clickhouse/schema/4000_filter_game_validity_schema.sql` (table + MV)

## Filtered Dataset Pipeline

- Single SQL file:
  - `database/clickhouse/schema/5099_filtered_pipeline_all_schema.sql`
- Consolidation note:
  - Legacy per-table filtered schema files `5101`-`5124` were removed.
  - `5099_filtered_pipeline_all_schema.sql` is the canonical schema entrypoint for filtered base tables.
  - `5125_matchup_windows_3v3_schema.sql` is still separate and should be applied only when that derived dataset is needed.
- Apply in one step:
  - `docker exec -i clickhouse bash -lc "clickhouse-client --multiquery" < database/clickhouse/schema/5099_filtered_pipeline_all_schema.sql`
- Note:
  - This creates target `game_data_filtered.*` tables (not materialized views).
  - Populate them with explicit `INSERT ... SELECT` statements after table creation.

## Objects

- `game_data.filter_game_validity`
  - Final persisted result table.
  - Columns:
    - `gameid UInt64`
    - `teamid UInt8`
    - `participantid UInt8`
    - `player_rule_mask UInt32`
    - `rule_mask UInt32`
    - `is_valid UInt8`

- `game_data.mv_filter_game_validity`
  - Computes all rules directly from `game_data.participant_stats`.
  - Builds and persists a per-player mask.
  - Also rolls up to game-level with `groupBitOr` and writes game-level validity alongside each player row.

## Why Bitmask (`rule_mask`)

- Packs many rule outcomes into one field.
- Supports efficient rollups (`groupBitOr`) from player to game.
- Keeps `is_valid` simple: valid iff `rule_mask = 0`.
- Allows debugging by checking specific bits.

## Rule Semantics

`rule_mask` sets a bit when any relevant entity in the game fails that rule.

### Player-level bits (computed per player, then OR-rolled to game)

- Bit `0`: `((kills + assists) / deaths) < 0.2`
- Bit `1`: `(goldspent / goldearned) < 0.60`
- Bit `2`: `(kills + assists = 0) AND (deaths > 4)`
- Bit `3`: `(summoner1casts = 0) OR (summoner2casts = 0)`
- Bit `9`: `(kills / team_kills) > 0.65`
- Bit `10`: `(totaldamagedealttochampions / team_totaldamagedealttochampions) < 0.075 AND teamposition != 'UTILITY'`
- Bit `11`: `(totalminionskilled / (timeplayed / 60)) < 4.5 AND teamposition != 'UTILITY'`
- Bit `12`: `item0..item6` are all `0`
- Bit `13`: at least 5 of `item0..item6` have the same value

### Team-level bits

- Bit `8`: `((team_kills + team_assists) / team_deaths) < 0.25`

### Game-level bits

- Bit `16`: `timeplayed <= 15 * 60`

## Output Contract

- `player_rule_mask`: failure bitset for that player record.
- `rule_mask`: aggregate failure bitset for the game.
- `is_valid`: game-level validity (`1` when `rule_mask = 0`, else `0`) repeated per player row.

## Operational Notes

- Intermediate participant/team/game filter artifacts are removed in this model.
- Final table uses `MergeTree` ordered by `(gameid, teamid, participantid)`.

## Local Session Filters (Ad Hoc)

The file `Local ClickHouse.session.sql` contains an exploratory reporting query with a filter set that is separate from the MV bitmask rules above.

### Active Query Rule List

Current active rule names in `Local ClickHouse.session.sql`:

1. `01-kda-lt-0.2`
2. `02-spent-lt-60%-earned`
3. `03-kills+assists-is-0-and-deaths-gt-4`
4. `04-either-summoner-not-cast`
5. `05-player-games-gr-40-winrate-gt-70%`
6. `06-team-kd-ratio-lt-0.33-vs-enemy`
7. `07-player-kills-gt-65%-team-kills`
8. `08-non-utility-dmg-share-lt-7.5%`
9. `09-non-utility-cs-per-min-lt-4.5`
10. `10-team-non-utility-avg-cs-per-min-gt-2.5-below-enemy`
11. `11-team-non-utility-dmg-to-champs-ratio-lt-1/3-vs-enemy`
12. `12-all-items-0`
13. `13-all-items-same`
14. `14-game-time-lte-18`
15. `15-player-champion+position-lt-30-picks-and-position-lt-0.6%-of-champion-picks`

Rollup:

- `16-any-filter-triggered`

### Quick Filter Summary

- `01-kda-lt-0.2`: Player KDA proxy is very low (`(kills + assists) * 5 < deaths`).
- `02-spent-lt-60%-earned`: Player spent less than 60% of earned gold.
- `03-kills+assists-is-0-and-deaths-gt-4`: No kill participation with more than 4 deaths.
- `04-either-summoner-not-cast`: At least one summoner spell was never used.
- `05-player-games-gr-40-winrate-gt-70%`: Player has >40 games and >70% winrate in `game_data.players`.
- `06-team-kd-ratio-lt-0.33-vs-enemy`: Team kills are under one-third of team deaths.
- `07-player-kills-gt-65%-team-kills`: One player has over 65% of team kills.
- `08-non-utility-dmg-share-lt-7.5%`: Non-support player damage share is below 7.5%.
- `09-non-utility-cs-per-min-lt-4.5`: Non-support CS/min is below 4.5.
- `10-team-non-utility-avg-cs-per-min-gt-2.5-below-enemy`: Team non-support average CS/min trails enemy by more than 2.5.
- `11-team-non-utility-dmg-to-champs-ratio-lt-1/3-vs-enemy`: Team non-support total champion damage is below one-third of enemy.
- `12-all-items-0`: All tracked item slots are zero.
- `13-all-items-same`: All tracked item slots have the same value.
- `14-game-time-lte-18`: Match duration is 18 minutes or less.
- `15-player-champion+position-lt-30-picks-and-position-lt-0.6%-of-champion-picks`: Match contains a player on a champion-role combination that is under 0.6% of that champion's pick pool and that player has fewer than 30 picks on it.
- `16-any-filter-triggered`: Rollup flag for whether a match hit any of the 15 applied filters.

### Ad Hoc-Specific Additions

- Player historical win-rate check joins `game_data.players` by `puuid`.
  - Current logic is `wins / (wins + losses) > 0.70` with `(wins + losses) > 40`.
- Relative team filters compare each team against the opposing team within the same `matchid`.
- A commented-out alternate query variant is kept in the same session file.

### Sample Figures

Sample output from current active `Local ClickHouse.session.sql`:

| Filter | Number Of Games | Pct Of Total Games | Total Games |
|---|---:|---:|---:|
| `01-kda-lt-0.2` | 119696 | 11.38 | 1052177 |
| `02-spent-lt-60%-earned` | 22760 | 2.16 | 1052177 |
| `03-kills+assists-is-0-and-deaths-gt-4` | 27319 | 2.60 | 1052177 |
| `04-either-summoner-not-cast` | 59298 | 5.64 | 1052177 |
| `05-player-games-gr-40-winrate-gt-70%` | 16007 | 1.52 | 1052177 |
| `06-team-kd-ratio-lt-0.33-vs-enemy` | 93289 | 8.87 | 1052177 |
| `07-player-kills-gt-65%-team-kills` | 43998 | 4.18 | 1052177 |
| `08-non-utility-dmg-share-lt-7.5%` | 49603 | 4.71 | 1052177 |
| `09-non-utility-cs-per-min-lt-4.5` | 106058 | 10.08 | 1052177 |
| `10-team-non-utility-avg-cs-per-min-gt-2.5-below-enemy` | 10347 | 0.98 | 1052177 |
| `11-team-non-utility-dmg-to-champs-ratio-lt-1/3-vs-enemy` | 2410 | 0.23 | 1052177 |
| `12-all-items-0` | 17769 | 1.69 | 1052177 |
| `13-all-items-same` | 20782 | 1.98 | 1052177 |
| `14-game-time-lte-18` | 119252 | 11.33 | 1052177 |
| `15-player-champion+position-lt-30-picks-and-position-lt-0.6%-of-champion-picks` | 31701 | 3.01 | 1052177 |
| `16-any-filter-triggered` | 350742 | 33.33 | 1052177 |

## Future Considerations

- Add a default-runes filter:
- Goal: flag players likely using untouched/default rune pages.
- Candidate rule: use Riot-provided default rune page definitions directly, and flag players whose selected runes exactly match a Riot default page for their champion/role context.
- Data source: `game_data.participant_perk_ids` (and optionally `participant_perk_values` for stricter matching).
- Reference data: Riot default rune definitions should be versioned by patch and joined during evaluation.
- Suggested rollout: start in report-only mode (count and pct), then decide whether to include in `any-filter-triggered`.
