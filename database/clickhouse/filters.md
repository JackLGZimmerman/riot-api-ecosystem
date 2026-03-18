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
- `database/clickhouse/schema/4000_filter_game_validity.sql` (table + MV)

## Filtered Dataset Pipeline
- Single SQL file:
  - `database/clickhouse/schema/5099_filtered_pipeline_all.sql`
- Consolidation note:
  - Legacy per-table filtered schema files `5101`-`5124` were removed.
  - `5099_filtered_pipeline_all.sql` is the canonical schema entrypoint for filtered base tables.
  - `5125_matchup_windows_3v3.sql` is still separate and should be applied only when that derived dataset is needed.
- Apply in one step:
  - `docker exec -i clickhouse bash -lc "clickhouse-client --multiquery" < database/clickhouse/schema/5099_filtered_pipeline_all.sql`
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
13. `13-at-least-5-same-items`
14. `14-game-time-lte-18`
15. `15-any-filter-triggered`

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
- `13-at-least-5-same-items`: At least five item slots have the same value.
- `14-game-time-lte-18`: Match duration is 18 minutes or less.
- `15-any-filter-triggered`: Match hit at least one of the above filters.

### Ad Hoc-Specific Additions
- Player historical win-rate check joins `game_data.players` by `puuid`.
  - Current logic is `wins / (wins + losses) > 0.70` with `(wins + losses) > 40`.
- Relative team filters compare each team against the opposing team within the same `matchid`.
- A commented-out alternate query variant is kept in the same session file.

### Sample Figures
Sample output from current active `Local ClickHouse.session.sql` (run on 2026-03-02):

| Filter | Number Of Games | Pct Of Total Games | Total Games |
|---|---:|---:|---:|
| `01-kda-lt-0.2` | 8328 | 9.87 | 84381 |
| `02-spent-lt-60%-earned` | 2148 | 2.55 | 84381 |
| `03-kills+assists-is-0-and-deaths-gt-4` | 2176 | 2.58 | 84381 |
| `04-either-summoner-not-cast` | 5086 | 6.03 | 84381 |
| `05-player-games-gr-40-winrate-gt-70%` | 2347 | 2.78 | 84381 |
| `06-team-kd-ratio-lt-0.33-vs-enemy` | 7443 | 8.82 | 84381 |
| `07-player-kills-gt-65%-team-kills` | 3621 | 4.29 | 84381 |
| `08-non-utility-dmg-share-lt-7.5%` | 4416 | 5.23 | 84381 |
| `09-non-utility-cs-per-min-lt-4.5` | 11557 | 13.70 | 84381 |
| `10-team-non-utility-avg-cs-per-min-gt-2.5-below-enemy` | 723 | 0.86 | 84381 |
| `11-team-non-utility-dmg-to-champs-ratio-lt-1/3-vs-enemy` | 266 | 0.32 | 84381 |
| `12-all-items-0` | 1565 | 1.85 | 84381 |
| `13-at-least-5-same-items` | 1812 | 2.15 | 84381 |
| `14-game-time-lte-18` | 8266 | 9.80 | 84381 |
| `15-any-filter-triggered` | 28430 | 33.69 | 84381 |

## Future Considerations
- Add a default-runes filter:
  - Goal: flag players likely using untouched/default rune pages.
  - Candidate rule: use Riot-provided default rune page definitions directly, and flag players whose selected runes exactly match a Riot default page for their champion/role context.
  - Data source: `game_data.participant_perk_ids` (and optionally `participant_perk_values` for stricter matching).
  - Reference data: Riot default rune definitions should be versioned by patch and joined during evaluation.
  - Suggested rollout: start in report-only mode (count and pct), then decide whether to include in `any-filter-triggered`.
