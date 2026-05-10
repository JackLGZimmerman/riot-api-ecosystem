# participant_build_minute_averages

Populated by `5134_participant_build_stat_averages_build.sql`.

This table contains one historical profile per `championid` × `teamposition` × `build`.
Each gameplay stat is stored as two conditional averages:

- `avg_in_wins_<stat>`
- `avg_in_losses_<stat>`

The build-level conditional averages are shrunk toward the less granular
`championid` × `teamposition` conditional averages. This keeps sparse builds in
the model while reducing win-rate inflation in kills, gold, objectives, damage,
vision, and similar outcome-sensitive aggregates.

The fallback conditional averages are also stored as
`champion_position_avg_in_wins_<stat>` and
`champion_position_avg_in_losses_<stat>` so the ML pipeline can feed
champion-position-relative residual features.

## Source Join

`participant_stats` joins `participant_item_value_totals` on `(matchid, participantid)`.
Rows with `timeplayed = 0`, null `championid`, or `UNKNOWN` team position are excluded.
`build` comes from `ivt.highest_value_label`.

## Smoothing

For each stat and outcome bucket:

```text
avg_in_wins_<stat> =
  (n_build_wins * build_avg_in_wins_<stat> + 20 * champion_position_avg_in_wins_<stat>)
  / (n_build_wins + 20)

avg_in_losses_<stat> =
  (n_build_losses * build_avg_in_losses_<stat> + 20 * champion_position_avg_in_losses_<stat>)
  / (n_build_losses + 20)
```

If the champion-position fallback has no examples for that outcome, the fallback
uses the champion-position unconditional average for that stat.

Win rate is smoothed separately:

```text
win_rate_smoothed =
  (n_wins + 50 * champion_position_win_rate)
  / (participant_count + 50)
```

## Threshold Flag

`passes_min_bucket` is true when:

```text
min(n_wins, n_losses) >= 10
```

or for extreme-win-rate profiles:

```text
max(win_rate_raw, 1 - win_rate_raw) >= 0.80
and min(n_wins, n_losses) >= 5
```

The ML dataset keeps sparse profiles by default and exposes this flag, plus
sample counts and support features, to the model. Set
`DatasetConfig(require_profile_bucket_threshold=True)` to hard-filter to this
threshold.

## Metadata Columns

| Column | Type | Notes |
|---|---|---|
| `championid` | Int32 | Champion id |
| `championname` | LowCardinality(String) | Resolved via `championid_name_map_dict` |
| `teamposition` | LowCardinality(String) | Riot role string |
| `build` | LowCardinality(String) | Highest-value item label |
| `participant_count` | UInt64 | Build-level sample count |
| `n_wins` | UInt64 | Build-level winning samples |
| `n_losses` | UInt64 | Build-level losing samples |
| `champion_position_participant_count` | UInt64 | Fallback sample count |
| `champion_position_n_wins` | UInt64 | Fallback winning samples |
| `champion_position_n_losses` | UInt64 | Fallback losing samples |
| `win_rate_raw` | Float32 | Build-level raw win rate |
| `champion_position_win_rate` | Float32 | Fallback raw win rate |
| `win_rate_smoothed` | Float32 | Build win rate shrunk toward fallback |
| `support_wins` | Float32 | `n_wins / (n_wins + 20)` |
| `support_losses` | Float32 | `n_losses / (n_losses + 20)` |
| `profile_confidence` | Float32 | `least(support_wins, support_losses)` |
| `passes_min_bucket` | UInt8 | Conditional-count threshold flag |

## Stats

The table stores 77 stats under four prefixes:

- `avg_in_wins_`
- `avg_in_losses_`
- `champion_position_avg_in_wins_`
- `champion_position_avg_in_losses_`

Most count-like fields are normalized to per-minute values before aggregation.
Peak/final-state fields such as `champlevel`, `largestkillingspree`,
`turretkills`, and `longesttimespentliving` are raw averages.

Derived stats such as `kda`, `expected_damage_per_gold`, damage shares, objective
scores, and `totalcs` are computed from the smoothed conditional base stats in
the final `SELECT`.
