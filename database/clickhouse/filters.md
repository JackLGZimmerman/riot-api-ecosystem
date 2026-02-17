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
- `database/clickhouse/schema/4000_filter_game_validity.sql`
- `database/clickhouse/schema/4001_mv_filter_game_validity.sql`

## Objects
- `game_data.filter_game_validity`
  - Final persisted result table.
  - Columns:
    - `gameid UInt64`
    - `rule_mask UInt32`
    - `is_valid UInt8`

- `game_data.mv_filter_game_validity`
  - Computes all rules directly from `game_data.participant_stats`.
  - Builds a per-player mask first, then rolls up to game-level with `groupBitOr`.

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
- Bit `16`: `gameendedinearlysurrender = 1`

## Output Contract
- `rule_mask`: aggregate failure bitset per game.
- `is_valid`: `1` when `rule_mask = 0`, else `0`.

## Operational Notes
- Intermediate participant/team/game filter artifacts are removed in this model.
- Final table uses `MergeTree` ordered by `gameid`.
