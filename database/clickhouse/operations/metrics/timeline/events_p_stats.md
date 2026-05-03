# timeline events + p_stats

This catalog contains combined timeline metrics derived from `events` and `p_stats`.

`p_stats` is an abbreviation for `participant_stats` in this metrics catalog. The underlying source table remains `game_data.tl_participant_stats`.

This file is intentionally same-frame only and does not use `t - (t-1)` style delta formulas.

Unless noted otherwise:
- `frame_timestamp` is the canonical participant-minute bucket for both sources
- event bins are grouped on the raw event table's `frame_timestamp`, then joined to the `p_stats` snapshot with the same `(matchid, frame_timestamp, participantid)` key
- `totalFarm = minionskilled + jungleminionskilled`
- `spentGold = totalgold - currentgold`
- `nonChampionDamageDone = totaldamagedone - totaldamagedonetochampions`
- Where kill-side event bins are reused, `killerid` is aligned to `participantid`, following the convention already documented in `events.md`
- Ratio-style metrics use the documented zero-protection rule `greatest(denominator, 1)`

## Build relationship

1. `participant_stats.md` defines the raw and same-row snapshot-derived columns from `game_data.tl_participant_stats`.
2. `events.md` defines the event-side participant-minute rollups from the raw timeline event tables.
3. This file only covers metrics that need both sides after they have already been reduced to the shared participant-minute bucket.
4. In the current reduced participant-minute role table, this is the only
   cross-source metric that remains materialized.

| id | name | data_source | description | calculation | version | fields |
|---|---|---|---|---|---|---|
| TLX_S_001 | synthetic.tl_events_p_stats.championDamagePerDeathEvent | derived:TLE_S_025 + game_data.tl_participant_stats (see notes above) | Champion damage snapshot contextualized by death count in the aligned participant-minute bucket. | totaldamagedonetochampions / greatest(TLE_S_025, 1) grouped by (matchid, frame_timestamp, participantid) | 1.0.0 | totaldamagedonetochampions, TLE_S_025, frame_timestamp, participantid, matchid |

## Metric List

These are combined timeline metrics that join `events` bins with `p_stats` frame snapshots. All metrics in this file are derived and new.

### Derived Death-Normalized State

- [TLX_S_001] `championDamagePerDeathEvent` (new): `totaldamagedonetochampions / greatest(TLE_S_025, 1) grouped by (matchid, frame_timestamp, participantid)`
